# Note: This class is highly experimental
# It is intended to wrap the XRDMap class and convert every method to apply iteratively across a stack of XRDMaps
# This is instended for 3D RSM mapping where each method / parameter may change between maps

import os
import h5py
import numpy as np
import pandas as pd
import gc
import functools
import dask
import itertools
from matplotlib import patches, color_sequences
from matplotlib.collections import PatchCollection
from tqdm import tqdm
from tqdm.dask import TqdmCallback
import matplotlib.pyplot as plt

# Base classes must be relative imports
from . import XRDBaseScan
from . import XRDMap
from . import XRDRockingCurve
from xrdmaptools.utilities.utilities import (
    timed_iter,
    pathify,
    _check_dict_key,
    get_int_vector_map,
    get_max_vector_map,
    get_num_vector_map,
    copy_docstring
)
from xrdmaptools.io.hdf_io import (
    initialize_xrdmapstack_hdf,
    load_xrdmapstack_hdf,
    # _load_xrd_hdf_vectorized_map_data
    _load_xrd_hdf_vector_data
    )
from xrdmaptools.io.hdf_utils import (
    check_attr_overwrite,
    overwrite_attr
)
from xrdmaptools.geometry.geometry import (
    q_2_polar,
    QMask
)
from xrdmaptools.reflections.spot_blob_search_3D import (
    rsm_spot_search
)
# from xrdmaptools.reflections.spot_blob_indexing_3D_old import (
#     pair_casting_index_full_pattern,
#     _get_connection_indices
# )
from xrdmaptools.reflections.spot_blob_indexing_3D import(
    phase_index_all_grains
)
from xrdmaptools.crystal.crystal import (
    are_collinear,
    are_coplanar
)
from xrdmaptools.crystal.rsm import map_2_grid
from xrdmaptools.crystal.map_alignment import (
    relative_correlation_auto_alignment,
    com_auto_alignment,
    manual_alignment
)
from xrdmaptools.plot.general import return_plot_wrapper
from xrdmaptools.plot.image_stack import base_slider_plot
from xrdmaptools.plot.interactive import (
    interactive_3D_plot,
    interactive_3D_labeled_plot
    )


# Class for working with PROCESSED XRDMaps
# Maybe should be called XRDMapList
class XRDMapStack(list): 
    """
    Class for combining XRDMaps acquired along either an
    energy/wavelength or angle rocking axis for analyzing and
    processing the XRD data. Many XRDMap methods are included to work
    verbatim, iterably though the full stack, or modified with this
    class.

    Parameters
    ----------
    stack : iterable of XRDMaps, optional
        Iterable of xrdmap instances used to construct the XRDMapStack.
    shifts : iterable, optional
        Iterable of iterables indicating the
        [[y-shift, x-shifts], [...], ...] with a shape of (n, 2), where
        n is the number of XRDMaps in the stack.
    rocking_axis : {'energy', 'angle'}, optional
        String indicating which axis was used as the rocking axis to
        scan in reciprocal space. Will accept variations of 'energy',
        'wavelength', 'angle' and 'theta', but stored internally as
        'energy', or 'angle'. Defaults to 'energy'.
    xdms_wd : path str, optional
        Path str indicating the main working directory for the XRD
        data. Will be used as the default read/write location.
        Defaults to the current working directy.
    xdms_filename : str, optional
        Custom file name if not provided. Defaults to include scan IDs.
    xdms_hdf_filename : str, optional
        Custom file name of the HDF file. Defaults to include the
        filename parameter if not provided.
    xdms_hdf : 
        h5py File instance. Will be used to derive hdf_filename and hdf
        path location if provided.
    save_hdf : bool, optional
        If False, this flag disables all HDF file read/write functions.
        True by default.
    xdms_extra_metadata : dict, optional
        Dictionary of extra metadata to be stored with the XRD data.
        Extra metadata will be written to the HDF file if enabled, but
        is not intended to be interacted with during normal data
        processing.
    xdms_extra_attrs : dict, optional
        Dictionary of extra attributes to be given to XRDMapStack.
        These attributes are intended for those values generated during
        processing of the XRD data.
    """

    # Class variables
    _hdf_type = 'xrdmapstack'

    def __init__(self,
                 stack=None,
                 shifts=None,
                 rocking_axis=None,
                 wd=None,
                 xdms_filename=None,
                 xdms_hdf_filename=None,
                 xdms_hdf=None,
                 save_hdf=False,
                 xdms_extra_metadata=None,
                 xdms_extra_attrs=None
                 ):
        
        # Create list to build around
        if stack is None:
            stack = []
        list.__init__(self, stack)

        # Strict input requirements
        for i, xrdmap in enumerate(self):
            if not isinstance(xrdmap, XRDMap):
                raise ValueError(f'Stack index {i} is not an XRDMap!')

        # Set up metdata
        if wd is None:
            wd = self.wd[0] # Grab from first xrdmap
        self.xdms_wd = wd
        if xdms_filename is None:
            scan_str = f'{np.min(self.scan_id)}-{np.max(self.scan_id)}'
            xdms_filename = f'scan{scan_str}_{self._hdf_type}'
        self.xdms_filename = xdms_filename
        if xdms_extra_metadata is None:
            xdms_extra_metadata = {}
        self.xdms_extra_metadata = xdms_extra_metadata

        # Pre-define values
        self._shifts = [(0, 0),] * len(self)

        # Collect unique phases from individual XRDMaps
        self.phases = {}
        for xrdmap in self:
            for phase in xrdmap.phases.values():
                if phase.name not in self.phases.keys():
                     self.phases[phase.name] = phase
            # Redefine individual phases to reference stack phases
            xrdmap.phases = self.phases

        # Find rocking axis
        self._set_rocking_axis(rocking_axis=rocking_axis)
        
        # Enable features
        # TODO: Fix me!
        if (not self.use_stage_rotation
            and self.rocking_axis == 'angle'):
            self.use_stage_rotation = True

        # Instantiate dedicated xrdmapstack hdf
        # For metadata, coordinating individual xrdmaps,
        # reduced datasets, and analysis
        if not save_hdf:
            # No dask considerations at this higher level
            self.xdms_hdf_path = None
            self.xdms_hdf = None
        else:
            self.start_saving_xrdmapstack_hdf(
                        xdms_hdf=xdms_hdf,
                        xdms_hdf_filename=xdms_hdf_filename)

        # If none, map pixels may not correspond to each other
        if shifts is not None:
            self.shifts = [tuple(shift) for shift in shifts]
        
        # Catch-all of extra attributes.
        # Gets them into __init__ sooner
        if xdms_extra_attrs is not None:
            for key, value in xdms_extra_attrs.items():
                setattr(self, key, value)

        if self._swapped_xdms_axes:
            self._swap_xdms_axes()


    ################################################
    ### Re-Written XRDMap Methods and Properties ###
    ################################################
     
    
    def _list_property_constructor(property_name,
                                   include_set=False,
                                   include_del=False):
        """

        """

        # Define getter
        def get_property(self):
            prop_list = []
            for i, xrdmap in enumerate(self):
                if hasattr(xrdmap, property_name):
                    prop_list.append(getattr(xrdmap, property_name))
                else:
                    err_str = (f'XRDMap [{i}] does not have '
                               + f'attribute {property_name}.')
                    raise AttributeError(err_str)
            return prop_list
        
        set_property, del_property = None, None

        # Define generic docstring
        doc = (f'List property of {property_name}.\nGetting this '
               + f'value will return a list of {property_name} for '
               + 'each XRDMap in the stack.')

        # Define setter if called
        if include_set:
            def set_property(self, values):
                [setattr(xrdmap, property_name, val)
                 for xrdmap, val in zip(self, values)]
            doc += ('\nSetting this property must be done with an '
                    + 'iterable of values matching the stack length.')

        # Defin deleter if called
        if include_del:
            def del_property(self):
                [delattr(xrdmap, property_name) for xrdmap in self]
            doc += ('\nDeleting this property will delete the property'
                    + ' from every XRDMap in the stack.')

        return property(get_property,
                        set_property,
                        del_property,
                        doc=doc)
    
    
    def _universal_property_constructor(property_name,
                                        include_set=False,
                                        include_del=False):
        """

        """
        # Uses first xrdmap in list as container object
        # Sets and deletes from all list elements
        def get_property(self):
            if hasattr(self[0], property_name):
                return getattr(self[0], property_name)
            else:
                err_str = ('XRDMap [0] does not have universal '
                           + f'attribute {property_name}.')
                raise AttributeError(err_str)
        
        set_property, del_property = None, None

        # Define generic docstring
        doc = (f'Unverisal property of {property_name}.\nGetting this '
               + f'value will return the value of from the first XRDMap'
               + ' in the stack, which should match every XRDMap.')
        
        if include_set:
            def set_property(self, value): # This may break some of them...
                [setattr(xrdmap, property_name, value)
                 for xrdmap in self]
            doc += ('\nSetting this value will iteratively set every '
                    + 'property in the stack with the single value.')

        if include_del:
            def del_property(self):
                [delattr(xrdmap, property_name) for xrdmap in self]
            doc += ('\nDeleting this property will delete the property'
                    + ' from every XRDMap in the stack.')

        return property(get_property,
                        set_property,
                        del_property,
                        doc=doc)


    energy = _list_property_constructor(
                                'energy',
                                include_set=True)
    wavelength = _list_property_constructor(
                                'wavelength',
                                include_set=True)
    tth_arr = _universal_property_constructor(
                                'tth_arr',
                                include_del=True)
    chi_arr = _universal_property_constructor(
                                'chi_arr',
                                include_del=True)
    scattering_units = _universal_property_constructor(
                                'scattering_units',
                                include_set=True)
    polar_units = _universal_property_constructor(
                                'polar_units',
                                include_set=True)
    image_scale = _universal_property_constructor(
                                'image_scale',
                                include_set=True)
    integration_scale = _universal_property_constructor(
                                'integration_scale',
                                include_set=True)
    dtype = _universal_property_constructor(
                                'dtype',
                                include_set=True)
    use_stage_rotation = _universal_property_constructor(
                                'use_stage_rotation',
                                include_set=True)
    
    ### Turn other individual attributes in properties ###

    # List attributes
    scan_id = _list_property_constructor('scan_id')
    filename = _list_property_constructor('filename')
    wd = _list_property_constructor('wd') # this one may change
    time_stamp = _list_property_constructor('time_stamp')
    scan_input = _list_property_constructor('scan_input')
    extra_metadata = _list_property_constructor('extra_metadata')
    dwell = _list_property_constructor('dwell',
                                       include_set=True)
    theta = _list_property_constructor('theta',
                                       include_set=True)
    hdf = _list_property_constructor('hdf')
    hdf_path = _list_property_constructor('hdf_path')
    xrf_path = _list_property_constructor('xrf_path',
                                          include_set=True)
    tth = _list_property_constructor('tth')
    chi = _list_property_constructor('chi')
    ai = _list_property_constructor('ai')
    pos_dict = _list_property_constructor('pos_dict')
    sclr_dict = _list_property_constructor('sclr_dict')

    # Universal attributes
    beamline = _universal_property_constructor('beamline')
    facility = _universal_property_constructor('facility')
    detector = _universal_property_constructor('detector')
    shape = _universal_property_constructor('shape')
    image_shape = _universal_property_constructor('image_shape')
    map_shape = _universal_property_constructor('map_shape')
    num_images = _universal_property_constructor('num_images')

    # Should be universal, could lead to some unexpected behaviors...
    tth_resolution = _list_property_constructor('tth_resolution',
                                                include_set=True)
    chi_resolution = _list_property_constructor('chi_resolution',
                                                include_set=True)

    ##############################
    ### XRDMapStack Properties ###
    ##############################

    @property
    def q_arr(self):
        """

        """

        if hasattr(self, '_q_arr'):
            return self._q_arr
        else:

            rocking = getattr(self, self.rocking_axis)
            if (rocking != sorted(rocking)
                and rocking != sorted(rocking, reverse=True)):
                err_str = ('Rocking axis values but be sorted before '
                           + 'generating the combined q_arr attribute.')
                raise RuntimeError(err_str)

            q_arr_list = []
            for i, xrdmap in enumerate(self):
                if hasattr(xrdmap, 'q_arr'):
                    q_arr_list.append(getattr(xrdmap, 'q_arr'))
                else:
                    err_str = (f'XRDMap [{i}] does not have '
                                + f'attribute q_arr.')
                    raise AttributeError(err_str)
            self._q_arr = np.asarray(q_arr_list)
            return self._q_arr
    
    @q_arr.deleter
    def q_arr(self):
        del self._q_arr


    @property
    def xrf(self):
        """

        """

        if hasattr(self, '_xrf'):
            # Check that all xrdmaps are still there
            if len(list(self._xrf.values())[0]) == len(self):
                return self._xrf
            else:
                warn_str = (f'Length of XRDMapStack ({len(self)}) '
                            + 'does not match previous XRF data '
                            + f'({len(self._xrf.values()[0])})'
                            + '\nDetermining new xrf length.')
                print(warn_str)
                del self._xrf
                return self.xrf # Woo recursion!
        else:
            for i, xrdmap in enumerate(self):
                if not hasattr(xrdmap, 'xrf') or xrdmap is None:
                    err_str = (f'XRDMap {i} does not '
                               + 'have xrf attribute.')
                    raise AttributeError(err_str)
            
            # Find which data to add
            _xrf = {}
            all_keys = []
            [all_keys.append(key) for key in xrdmap.xrf.keys()
            for xrdmap in self if key not in all_keys]
            _empty_lists = [[] for _ in range(len(all_keys))]

            _xrf = dict(zip(all_keys, _empty_lists))

            # Add data to internal dictionary
            for xrdmap in self:
                for key in all_keys:
                    if key in xrdmap.xrf.keys():
                        _xrf[key].append(xrdmap.xrf[key])
                    else:
                        # This is to handle variable element fitting
                        _xrf[key].append(None) 

            # Clean up None inputs to zero arrays.
            # Not as straightfoward with previous loop
            for key in all_keys:
                none_indices = []
                for i, data in enumerate(_xrf[key]):
                    if data is None:
                        none_indices.append(i)
                    else:
                        clean_index = i
                for idx in none_indices:
                    _xrf[key][idx] = np.zeros_like(
                                        _xrf[key][clean_index])
            
            self._xrf = _xrf
            return self._xrf
        
        @xrf.deleter
        def xrf(self):
            del self._xrf
        

    @property
    def shifts(self):
        """

        """

        return self._shifts

    @shifts.setter
    def shifts(self, shifts):
        self._shifts = shifts

        # Re-write hdf values
        @XRDMapStack._protect_xdms_hdf()
        def save_attrs(self): # Not sure if this needs self...
            overwrite_attr(self.xdms_hdf[self._hdf_type].attrs,
                           'shifts',
                           self.shifts)
        save_attrs(self)
    
    @shifts.deleter
    def shifts(self):
        del self._shifts
    

    @property
    def _swapped_axes(self):
        """

        """

        return [xdm._swapped_axes for xdm in self]
    
    @_swapped_axes.setter
    def _swapped_axes(self, val):
        if not isinstance(val, bool):
            raise TypeError('Swapped axes flag must be boolean.')
        
        for xdm in self:
            if xdm._swapped_axes != val:
                xdm.swap_axes()

    
    @property
    def qmask(self):
        """

        """
        if hasattr(self, '_qmask'):
            return self._qmask
        else:
            self._qmask = QMask.from_XRDRockingScan(self)
            return self._qmask
    
    @qmask.deleter
    def qmask(self):
        del self._qmask


    #####################################
    ### Loading data into XRDMapStack ###
    #####################################

    @classmethod
    def from_XRDMap_hdfs(cls,
                         hdf_filenames,
                         wd=None,
                         xdms_wd=None,
                         xdms_filename=None,
                         xdms_hdf_filename=None,
                         xdms_hdf=None,
                         save_hdf=True,
                         dask_enabled=False,
                         image_data_key=None, # Load empty datasets
                         integration_data_key=None, # Load empty datasets
                         load_blob_masks=False, # Load empty datasets
                         load_vector_maps=False, # Load emtpy datasets
                         rocking_axis=None,
                         **kwargs):
        """

        """

        # Setup working directories
        if wd is None or xdms_wd is None:
            if wd is None and xdms_wd is None:
                wd = os.getcwd()
                xdms_wd = wd
            elif wd is None:
                wd = xdms_wd
            elif xdms_wd is None:
                xdms_wd = wd
        
        # Check working directories and their immediate subdirectories for files
        all_wd = [wd, xdms_wd] + [x.path for x in os.scandir(wd) if x.is_dir()]
        if wd != xdms_wd:
            all_wd += [x.path for x in os.scandir(xdms_wd) if x.is_dir()]

        for d in all_wd:
            for filename in hdf_filenames:
                path = pathify(d, filename, '.h5')
                if not os.path.exists(path):
                    warn_str = (f'File {path} cannot be found. Checking'
                                + ' another directory for files.')
                    print(warn_str)
                    break
            else:
                wd = d
                break
        else:
            err_str = ('One or more files is missing from each '
                       + 'attempted directory in:\t'
                       + '\t'.join(all_wd))
            raise FileNotFoundError(err_str)

        
        xrdmap_list = []
        for hdf_filename in timed_iter(hdf_filenames,
                                       total=len(hdf_filenames),
                                       iter_name='xrdmap'):
            xrdmap_list.append(
                XRDMap.from_hdf(
                    hdf_filename,
                    wd=wd,
                    dask_enabled=dask_enabled,
                    image_data_key=image_data_key,
                    integration_data_key=integration_data_key,
                    load_blob_masks=load_blob_masks,
                    load_vector_map=load_vector_maps,
                    save_hdf=save_hdf,
                    **kwargs
                )
            )

        return cls(stack=xrdmap_list,
                   wd=xdms_wd,
                   xdms_filename=xdms_filename,
                   xdms_hdf_filename=xdms_hdf_filename,
                   xdms_hdf=xdms_hdf,
                   save_hdf=save_hdf,
                   rocking_axis=rocking_axis)

    
    @classmethod
    def from_hdf(cls,
                 xdms_hdf_filename,
                 # converts to xdms_wd, but kept wd for consistency
                 wd=None, 
                 load_xdms_vector_map=True,
                 save_hdf=True,
                 dask_enabled=False,
                 image_data_key=None,
                 integration_data_key=None,
                 load_blob_masks=False,
                 load_vector_maps=False,
                 map_shape=None,
                 image_shape=None,
                 **kwargs
                 ):
        """

        """

        if wd is None:
            wd = os.getcwd()
        
        xdms_hdf_path = pathify(wd, xdms_hdf_filename, '.h5')
        if not os.path.exists(xdms_hdf_path):
            # Should be redundant...
            raise FileNotFoundError(f'No HDF file at {xdms_hdf_path}.')
        
        # File exists, attempt to load data!
        print('Loading data from the HDF file...')
        input_dict = load_xrdmapstack_hdf(
                            os.path.basename(xdms_hdf_path),
                            os.path.dirname(xdms_hdf_path),
                            load_xdms_vector_map=load_xdms_vector_map)
        
        hdf_path = input_dict['base_md'].pop('hdf_path')
        if 'shifts' in input_dict['base_md']:
            shifts = input_dict['base_md'].pop('shifts')
        else:
            shifts = None
        rocking_axis = input_dict['base_md'].pop('rocking_axis')
        xdms_extra_metadata = input_dict.pop('xdms_extra_metadata')

        xdms_extra_attrs = {}
        if input_dict['vector_dict'] is not None:
            (xdms_extra_attrs['xdms_vector_map'] # rename
             ) = input_dict['vector_dict'].pop('vector_map')
            xdms_extra_attrs.update(input_dict['vector_dict'])
        
        if input_dict['spots_3D'] is not None:
            xdms_extra_attrs['spots_3D'] = input_dict.pop('spots_3D')
        
        # Check working directories and their immediate subdirectories for files
        xdms_wd = wd
        hdf_wd = [os.path.dirname(path) for path in hdf_path]
        hdf_filenames = [os.path.basename(path) for path in hdf_path]

        all_wd = [wd] # Start with designated wd
        for d in hdf_wd:
            if d not in all_wd:
                # Then try each unique wd of individual files
                all_wd.append(d) 
        # Then try all possible subdirectories
        all_wd += [x.path for x in os.scandir(wd) if x.is_dir()]
        
        for d in all_wd:
            for filename in hdf_filenames:
                path = pathify(d, filename, '.h5')
                if not os.path.exists(path):
                    warn_str = (f'File {path} cannot be found. Checking'
                                + ' another directory for files.')
                    print(warn_str)
                    break
            else:
                wd = d
                break
        else:
            err_str = ('One or more files is missing from each '
                       + 'attempted directory in:\n'
                       + '\t'.join(all_wd))
            raise FileNotFoundError(err_str)

        xrdmap_list = []
        for filename in timed_iter(hdf_filenames, iter_name='xrdmap'):
            xrdmap_list.append(
                XRDMap.from_hdf(
                    filename,
                    wd=wd,
                    dask_enabled=dask_enabled,
                    image_data_key=image_data_key,
                    integration_data_key=integration_data_key,
                    load_blob_masks=load_blob_masks,
                    load_vector_map=load_vector_maps,
                    map_shape=map_shape,
                    image_shape=image_shape,
                    save_hdf=save_hdf,
                    **kwargs
                )
            )

        inst = cls(stack=xrdmap_list,
                   wd=xdms_wd,
                   rocking_axis=rocking_axis,
                   shifts=shifts,
                   xdms_filename=xdms_hdf_filename[:-3],
                   xdms_hdf_filename=xdms_hdf_filename,
                   # xdms_hdf=xdms_hdf,
                   save_hdf=save_hdf,
                   xdms_extra_metadata=xdms_extra_metadata,
                   xdms_extra_attrs=xdms_extra_attrs)
        
        print(f'{cls.__name__} loaded!')
        return inst


    #####################################
    ### Iteratively Wrapped Functions ###
    #####################################

    # This is currently called during __init__,
    # but may work as a decorator within the class
    def _get_iterated_method(method,
                             variable_inputs=False,
                             timed_iterator=False):
        """

        """

        # Select iterator
        if timed_iterator:
            iterator = timed_iter
        else:
            iterator = lambda iterable, **k : list(iterable)

        if variable_inputs:
            # Lists of inputs
            def iterated_method(self, *arglists, **kwarglists):

                # Check and fix arglists
                for i, arg in enumerate(arglists):
                    if (isinstance(arg, list)
                        and len(arg) == len(self)):
                        args.append(arg)
                    elif (isinstance(arg, list)
                          and len(arg) != len(self)):
                        err_str = ('Length of arguments do not match '
                                   + 'length of XRDMapStack.')
                        raise ValueError(err_str)
                    else:
                        # Redefine arg as repeated list
                        arglists[i] = [arg,] * len(self)

                # Check and fix kwarglists
                for key, kwarg in kwarglists.items():
                    if (isinstance(kwarg, list)
                        and len(kwarg) == len(self)):
                        pass # All is well
                    elif (isinstance(kwarg, list)
                          and len(kwarg) != len(self)):
                        err_str = ('Length of arguments do not match '
                                   + 'length of XRDMapStack.')
                        raise ValueError(err_str)
                    else:
                        # Redefine kwarg as repeated list
                        kwarglists[key] = [kwarg,] * len(self)

                # Call actual method
                for i, xrdmap in iterator(enumerate(self),
                                          total=len(self),
                                          iter_name='XRDMap'):
                    args = [arg[i] for arg in arglists]
                    kwargs = dict(
                                zip(kwarglists.keys(),
                                    [val[i]
                                     for val in kwarglists.values()]))
                    
                    getattr(xrdmap, method)(*args, **kwargs)

        else:
            # Constant inputs
            def iterated_method(self, *args, **kwargs):
                for i, xrdmap in iterator(enumerate(self),
                                          total=len(self),
                                          iter_name='XRDMap'):
                    getattr(xrdmap, method)(*args, **kwargs)
        
        # Generate decorated wrapper of original method docstring
        doc = f'Iterated wrapper of XRDMap.{method} method:\n'
        if getattr(XRDMap, method).__doc__:
            doc += getattr(XRDMap, method).__doc__
        else:
            print(f'{method} is missing a docstring!')
        iterated_method.__doc__ = doc
        
        return iterated_method

    
    # List of iterable methods
    iterated_methods = (
        # (XRDMapStack function, XRDMap function,
        #  variable_inputs, timed_iterator)
        # hdf functions
        ('start_saving_hdf', 'start_saving_hdf', False, False),
        ('save_current_hdf', 'save_current_hdf', False, True),
        ('stop_saving_hdf', 'stop_saving_hdf', False, False),
        # Light image manipulation
        ('load_images_from_hdf', 'load_images_from_hdf', True, True),
        ('dump_images', 'dump_images', False, False),
        # Working with calibration
        # Do not accept variable calibrations
        ('set_calibration', 'set_calibration', False, False), 
        ('save_calibration', 'save_calibration', False, True),
        ('integrate1D_map', 'integrate1D_map', False, True),
        ('integrate2D_map', 'integrate2D_map', False, True),
        ('save_reciprocal_positions', 'save_reciprocal_positions',
         False, True),
        # Working with positions only
        ('set_positions', 'set_positions', True, False), 
        ('save_sclr_pos', 'save_sclr_pos', False, False),
        ('swap_axes', 'swap_axes', False, False),
        ('map_extent', 'map_extent', False, False),
        # Working with phases
        # This one saves to individual hdfs
        ('save_phases', 'save_phases', False, False), 
        # Working with spots
        ('find_2D_blobs', 'find_blobs', False, True),
        ('find_2D_spots', 'find_spots', False, True),
        ('recharacterize_2D_spots', 'recharacterize_spots',
         False, True),
        ('fit_2D_spots', 'fit_spots', False, True),
        ('initial_2D_spot_analysis', 'initial_spot_analysis',
         False, True),
        ('trim_2D_spots', 'trim_spots', False, False),
        ('remove_2D_spot_guesses', 'remove_spot_guesses',
         False, False),
        ('remove_2D_spot_fits', 'remove_spot_fits', False, False),
        ('save_2D_spots', 'save_spots', False, True),
        ('vectorize_map_data', 'vectorize_map_data', False, True),
        # Working with xrfmap
        # Multiple inputs may be crucial here
        ('load_xrfmap', 'load_xrfmap', True, False) 
    )

    # Define iterated methods
    for vals in iterated_methods:
        # vars() directly adds to __dict__. This could overwrite values?
        vars()[vals[0]] = _get_iterated_method(*vals[1:])

    
    ########################################################
    ### Verbatim Functions, Pseudo-inherited from XRDMap ###
    ########################################################
    
    def _get_verbatim_method(method):
        """

        """

        def verbatim_method(self, *args, **kwargs):
            getattr(self[0], method)(*args, **kwargs)

        # Generate decorated wrapper of original method docstring
        doc = f'Verbatim wrapper of XRDMap.{method} method:\n'
        doc += getattr(XRDMap, method).__doc__
        verbatim_method.__doc__ = doc
        
        return verbatim_method

            
    # List of verbatim methods
    verbatim_methods = (
        'estimate_polar_coords',
        'estimate_image_coords',
        'integrate1D_image',
        'integrate2D_image',
        # This will modify first xrdmap.phases
        # This is just a reference to self.phases
        'add_phase',
        'remove_phase',
        'load_phase',
        'clear_phases'
    )
    
    # Define verbatim methods
    for method in verbatim_methods:
        # vars() directly adds to __dict__. This could overwrite values?
        vars()[method] = _get_verbatim_method(method)


    def _modify_pseudo_inherited_plot(func, verbatim=True):
        """

        """

        @copy_docstring(func)
        @return_plot_wrapper
        def modified_plot(self, *args, **kwargs):

            xdms_title = None
            if 'title' in kwargs:
                xdms_title = kwargs.pop('title')
            title_scan_id = True
            if 'title_scan_id' in kwargs:
                title_scan_id = kwargs.pop('title_scan_id')

            if verbatim:
                internal_self = self[0]
            else:
                internal_self = self
            
            fig, ax = func(internal_self,
                           *args,
                           **kwargs,
                           return_plot=True)

            title = self._title_with_scan_id(
                                    xdms_title,
                                    default_title='Custom Map',
                                    title_scan_id=title_scan_id)
            
            ax.set_title(title)

            return fig, ax
        
        return modified_plot

    # Define modified plots
    plot_map = _modify_pseudo_inherited_plot(XRDMap.plot_map)
    plot_detector_geometry = _modify_pseudo_inherited_plot(
                                    XRDMap.plot_detector_geometry)       

    ##################################################
    ### Pseudo-inherited XRDRockingCurve Functions ###
    ##################################################
    
    get_sampled_edges = XRDRockingCurve.get_sampled_edges
    _set_rocking_axis = XRDRockingCurve._set_rocking_axis
    _title_with_scan_id = XRDRockingCurve._title_with_scan_id
    plot_sampled_volume_outline = XRDRockingCurve.plot_sampled_volume_outline


    ################################
    ### XRDMapStack HDF Methods  ###
    ################################

    # Re-defined from XRDData class
    # New names and no dask concerns
    def _protect_xdms_hdf(pandas=False):
        """
        Decorator for safely opening and closing HDF files.

        Parameters
        ----------
        pandas : bool, optional
            Flag to alter behavior for the pandas.DataFrame.to_hdf
            function, which requires a closed h5py.File object.
        """
        def protect_hdf_inner(func):
            @functools.wraps(func)
            def protector(self, *args, **kwargs):
                # Check to see if read/write is enabled
                if self.xdms_hdf_path is not None:
                    # Is an hdf reference currently active?
                    active_hdf = self.xdms_hdf is not None

                    if pandas: # Fully close reference
                        self.close_xdms_hdf()
                    elif not active_hdf: # Make temp reference
                        self.xdms_hdf = h5py.File(self.xdms_hdf_path,
                                                  'a')

                    # Call function
                    try:
                        func(self, *args, **kwargs)
                        err = None
                    except Exception as e:
                        err = e

                    # Clean up hdf state
                    if pandas and active_hdf:
                        self.open_xdms_hdf()
                    elif not active_hdf:
                        self.close_xdms_hdf()
                    
                    # Re-raise any exceptions, after cleaning up hdf
                    if err is not None:
                        raise(err)
            return protector
        return protect_hdf_inner


    # Opens an active reference to xdms
    def open_xdms_hdf(self):
        """

        """
        if self.xdms_hdf is not None:
            # Should this raise errors or just ping warnings
            note_str = ('NOTE: the HDF file is already open. '
                        + 'Proceeding without changes.')
            print(note_str)
            return
        else:
            self.xdms_hdf = h5py.File(self.xdms_hdf_path, 'a')
    

    # Closes any active reference to xdms_hdf
    def close_xdms_hdf(self):
        """

        """
        if self.xdms_hdf is not None:
            self.xdms_hdf.close()
            self.xdms_hdf = None
        

    def start_saving_xrdmapstack_hdf(self,
                                     xdms_hdf=None,
                                     xdms_hdf_filename=None,
                                     xdms_hdf_path=None,
                                     save_current=False):
        """

        """
        
        # Check for previous iterations
        if ((hasattr(self, 'xdms_hdf')
             and self.xdms_hdf is not None)
            or (hasattr(self, 'xdms_hdf_path')
                and self.xdms_hdf_path is not None)):
            warn_str = ('WARNING: Trying to save to the HDF file, but '
                        'a file or location has already been '
                        'specified!\nSwitching save files or locations'
                        ' should use the "switch_xrdmapstack_hdf" '
                        'function.\nProceeding without changes.')
            print(warn_str)
            return

        # Specify hdf path and name
        if xdms_hdf is not None: # biases towards already open hdf
            # This might break if hdf is a close file and not None
            self.xdms_hdf_path = xdms_hdf.filename
        elif xdms_hdf_filename is None:
            if xdms_hdf_path is None:
                self.xdms_hdf_path = pathify(self.xdms_wd,
                                             self.xdms_filename,
                                             '.h5',
                                             check_exists=False)
            else:
                self.xdms_hdf_path = pathify(xdms_hdf_path,
                                             self.xdms_filename,
                                             '.h5',
                                             check_exists=False)
        else:
            if xdms_hdf_path is None:
                self.xdms_hdf_path = pathify(self.xdms_wd,
                                             xdms_hdf_filename,
                                             '.h5',
                                             check_exists=False)
            else:
                self.xdms_hdf_path = pathify(xdms_hdf_path,
                                             xdms_hdf_filename,
                                             '.h5',
                                             check_exists=False)

        # Check for hdf and initialize if new
               
        if not os.path.exists(self.xdms_hdf_path):
            # Initialize base structure
            initialize_xrdmapstack_hdf(self, self.xdms_hdf_path)
            # Clear hdf for protection
            self.xdms_hdf = None            
        else:
            self.xdms_hdf = None
            # Update hdf_path information
            @XRDMapStack._protect_xdms_hdf()
            def save_attrs(self):
                overwrite_attr(self.xdms_hdf[self._hdf_type].attrs,
                            'hdf_path',
                            self.hdf_path)
            save_attrs(self)

        if save_current:
            self.save_current_xrdmapstack_hdf()


    # Saves current major features
    # Calls several other save functions
    @_protect_xdms_hdf()
    def save_current_xrdmapstack_hdf(self):
        """

        """
        
        if self.xdms_hdf_path is None:
            print('WARNING: Changes cannot be written to the HDF file '
                  + 'without first indicating a file location.'
                  + '\nProceeding without changes.')
            return

        # Save stacked vector_map
        if ((hasattr(self, 'xdms_vector_map')
             and self.xdms_vector_map is not None)
            and (hasattr(self, 'edges') and self.edges is not None)):
            self.save_xdms_vector_map()
        
        # Save 3D spots
        if hasattr(self, 'spots_3D'):
            self.save_3D_spots()

    
    # Ability to toggle hdf saving and proceed without writing to disk.
    def stop_saving_xrdmapstack_hdf(self):
        """

        """

        self.close_xdms_hdf()
        self.xdms_hdf_path = None
    

    @_protect_xdms_hdf()
    def switch_xrdmapstack_hdf(self,
                             xdms_hdf=None,
                             xdms_hdf_path=None,
                             xdms_hdf_filename=None,
                             save_current=False):
        """

        """

        # Check to make sure the change is appropriate and correct.
        # Not sure if this should raise and error or just print a warning
        if xdms_hdf is None and xdms_hdf_path is None:
            ostr = ('Neither xdms_hdf nor xdms_hdf_path were provided. '
                     + '\nCannot switch the HDF file save locations '
                     + 'without providing alternative location.')
            print(ostr)
            return
        
        elif xdms_hdf == self.xdms_hdf:
            ostr = (f'WARNING: provided HDF file '
                    + f'({self.xdms_hdf.filename}) is already the '
                    + 'current save location. '
                    + '\nProceeding without changes')
            print(ostr)
            return
        
        elif xdms_hdf_path == self.xdms_hdf_path:
            ostr = (f'WARNING: provided hdf_path ({self.xdms_hdf_path})'
                    + ' is already the current save location. '
                    + '\nProceeding without changes')
            print(ostr)
            return
        
        else:
            # Success actually changes the write location
            old_base_attrs = dict(self.xdms_hdf[self._hdf_type].attrs)

            self.stop_saving_xrdmapstack_hdf()
            self.start_saving_xrdmapstack_hdf(
                            xdms_hdf=xdms_hdf,
                            xdms_hdf_path=xdms_hdf_path,
                            xdms_hdf_filename=xdms_hdf_filename,
                            save_current=save_current)
            self.open_xdms_hdf()

            # Overwrite from old values
            for key, value in old_base_attrs.items():
                self.xdms_hdf[self._hdf_type].attrs[key] = value
    

    def repackage_without_images(self):
        """

        """

        # Checks
        if not hasattr(self, 'xdms_vector_map') or self.xdms_vector_map is None:
            raise AttributeError()

        # Make folder for new information.
        # Odd formatting to allow for correct file/folder names
        new_xdms_wd = os.path.join(self.xdms_wd, f'{self.xdms_filename}_repackaged')
        os.mkdir(new_xdms_wd)

        # Move over each xrdmap first
        for xdm in self:
            xdm.wd = new_xdms_wd
            xdm.switch_hdf(hdf_path=new_xdms_wd,
                           save_current=True,
                           verbose=False)
            xdm.save_images(title='empty')

        # Move xrdmapstack and start new
        self.xdms_wd = new_xdms_wd
        self.switch_xrdmapstack_hdf(xdms_hdf_path=new_xdms_wd,
                                    save_current=True)


    ##########################
    ### Modified Functions ###
    ##########################

    @property
    def _swapped_xdms_axes(self):
        """

        """

        # Check if all axes are the same
        if (not (all(self._swapped_axes)
                 or all([not _ for _ in self._swapped_axes]))):
            warn_str = ('WARNING: Inconsistent swapped axes across '
                        + 'XRDMaps! Axes checks will be disabled until '
                        + 'fixed, which may cause further '
                        + 'inconsistencies!\nRe-instantiating '
                        + 'XRDMapStack from consistent XRDMaps is '
                        + 'suggested.')
            print(warn_str)
            return False

        # Determine swapped axes
        return self._swapped_axes[0]


    # Internal function for swapping XRDMapStack-specific mapped attributes
    # Does not interact with individual maps; use swap_axes instead
    def _swap_xdms_axes(self):
        """

        """

        # Update vector map and related data
        if (hasattr(self, 'xdms_vector_map')
            and self.xdms_vector_map is not None):
            self.xdms_vector_map = self.xdms_vector_map.swapaxes(0, 1)
        if (hasattr(self, 'xdms_spot_label_map')
            and self.xdms_spot_label_map is not None):
            self.xdms_spot_label_map = self.xdms_spot_label_map.swapaxes(0, 1)

        # Update spot map_indices
        if hasattr(self, 'spots'):
            map_x_ind = self.spots_3D['map_x'].values
            map_y_ind = self.spots_3D['map_y'].values
            self.spots_3D['map_x'] = map_y_ind
            self.spots_3D['map_y'] = map_x_ind

    
    # Wrapper for XRDMapStack swapped axes for consistency
    def _check_xdms_swapped_axes(func):
        """
        Decorator for transposing data with swapped fast and slow axes
        after read operations and before write operations to maintain
        consistent data shape.
        """
        @functools.wraps(func)
        def wrapped(self, *args, **kwargs):

            if self._swapped_xdms_axes:
                self._swap_xdms_axes()
            
            try:
                func(self, *args, **kwargs)
                err = None
            except Exception as e:
                err = e
            
            if self._swapped_xdms_axes:
                self._swap_xdms_axes()

            # Re-raise any exceptions
            if err is not None:
                raise(err)

        return wrapped


    @_check_xdms_swapped_axes
    @_protect_xdms_hdf()
    def save_xdms_vector_map(self,
                             xdms_vector_map=None,
                             edges=None,
                             rewrite_data=False):
        """

        """

        # Check input
        if xdms_vector_map is None:
            if (hasattr(self, 'xdms_vector_map')
                and self.xdms_vector_map is not None):
                xdms_vector_map = self.xdms_vector_map
            else:
                err_str = ('Must provide xdms_vector_map or '
                        + f'{self.__class__.__name__} must have '
                        + 'xdms_vector_map attribute.')
                raise AttributeError(err_str)
        if edges is None:
            if (hasattr(self, 'edges')
                and self.edges is not None):
                edges = self.edges
            else:
                err_str = ('Must provide edges or '
                        + f'{self.__class__.__name__} must have edges'
                        + ' attribute.')
                raise AttributeError(err_str)
    
        XRDMap._save_vector_map(self, # this might break
                                self.xdms_hdf,
                                vector_map=xdms_vector_map,
                                edges=edges,
                                rewrite_data=rewrite_data)


    @_check_xdms_swapped_axes
    @_protect_xdms_hdf()
    def save_xdms_vector_map_information(self,
                                         vector_map_info,
                                         vector_map_info_title,
                                         rewrite_data=True,
                                         extra_attrs=None):
        """

        """

        XRDMap._save_vector_map(self, # this might break
                                self.xdms_hdf,
                                vector_map=vector_map_info,
                                vector_map_title=vector_map_info_title,
                                edges=None,
                                rewrite_data=rewrite_data)

        # Add extra information
        if extra_attrs is not None:
            for key, value in extra_attrs.items():
                overwrite_attr(
                    self.hdf['vectorized_data'][vector_map_info_title].attrs,
                    key,
                    value)
    

    # Almost verbatim with XRDMap.load_vector_map
    @_check_xdms_swapped_axes
    @XRDBaseScan._protect_hdf()
    def load_vector_map(self):
        """

        """
        XRDBaseScan._load_vectors(self.xdms_hdf)


    # Need to modify to not look for a random image
    def plot_image(self, *args, **kwargs):
        raise NotImplementedError()    


    ############################################
    ### XRDMapStack Specific Utility Methods ###
    ############################################                           

    def sort_by_attr(self,
                     attr,
                     reverse=False):
        """

        """
        
        # Check for attr
        for i, xrdmap in enumerate(self):
            if not hasattr(xrdmap, attr):
                err_str = (f'XRDMap [{i}] does not have '
                           + f'attributre {attr}.')
                raise AttributeError(err_str)
        
        # Actually sort
        attr_list = [getattr(xrdmap, attr) for xrdmap in self]
        self.sort(key=dict(zip(self, attr_list)).get,
                  reverse=reverse)

        # Delete sorted attrs built from indvidual xrdmaps
        if hasattr(self, '_xrf'):
            del self._xrf
        if hasattr(self, '_q_arr'):
            del self._q_arr

        # Sort inherent XRDMapStack attrs
        # Not in-place since they may not be lists
        sorted_attrs = [
            'shifts'
        ]

        for sorted_attr in sorted_attrs:
            if (not hasattr(self, sorted_attr)
                or getattr(self, sorted_attr) is None):
                continue
            attr_type = type(getattr(self, sorted_attr))
            orig_attr_list = list(getattr(self, sorted_attr))
            sorted_list = sorted(
                        orig_attr_list,
                        key=dict(zip(orig_attr_list, attr_list)).get,
                        reverse=reverse)
            setattr(self, sorted_attr, attr_type(sorted_list))

        # Re-write sorted values in hdf for consistency
        @XRDMapStack._protect_xdms_hdf()
        def save_attrs(self):
            sorted_attrs = [
                'scan_id',
                'energy',
                'theta',
                'hdf_path',
                'dwell'
            ]
            for attr in sorted_attrs:
                overwrite_attr(self.xdms_hdf[self._hdf_type].attrs,
                               attr,
                               getattr(self, attr))
        save_attrs(self)

    
    # Convenience wrapper for batch processing scans
    def batched_processing(self,
                           # Input must be xrdmap and kwarglists
                           batched_functions, 
                           **kwarglists):
        """

        """

        # Check and fix karglists
        for key, kwarg in kwarglists.items():
            if isinstance(kwarg, list) and len(kwarg) == len(self):
                pass # All is well
            elif isinstance(kwarg, list) and len(kwarg) != len(self):
                err_str = ('Length of arguments do not match '
                           + 'length of XRDMapStack.')
                raise ValueError(err_str)
            else:
                # Redefine kwarg as repeated list
                kwarglists[key] = [kwarg,] * len(self)

        for i, xrdmap in timed_iter(enumerate(self),
                                    total=len(self),
                                    iter_name='XRDMap'):

            kwargs = dict(zip(kwarglists.keys(),
                    [val[i] for val in kwarglists.values()]))

            batched_functions(xrdmap, **kwargs)


    # Invalid with shifts??
    def stack_2D_spots(self):
        """

        """
    
        all_spots_list = []
        for xrdmap in self:
            # This copy might cause memory issues, but should 
            # protect the individual dataframes to some extent
            spots = xrdmap.spots.copy() 

            scan_id_list = [xrdmap.scan_id for _ in range(len(spots))]
            energy_list = [xrdmap.energy for _ in range(len(spots))]
            wavelength_list = [xrdmap.wavelength
                               for _ in range(len(spots))]
            theta_list = [xrdmap.theta for _ in range(len (spots))]

            spots.insert(
                loc=0,
                column='scan_id',
                value=scan_id_list
            )
            spots.insert(
                loc=1,
                column='energy',
                value=energy_list
            )
            spots.insert(
                loc=2,
                column='wavelength',
                value=wavelength_list
            )
            spots.insert(
                loc=3,
                column='theta',
                value=theta_list
            )
            all_spots_list.append(spots)
        
        self.spots = pd.concat(all_spots_list, ignore_index=True)

    
    def pixel_2D_spots(self, map_indices):
        """

        """
        pixel_spots = self.spots[
                    (self.spots['map_x'] == map_indices[1])
                     & (self.spots['map_y'] == map_indices[0])].copy()
        return pixel_spots
    

    ########################
    ### Vectorizing Data ###
    ########################

    # TODO: Add center of mass alignment method
    def align_maps(self,
                   map_stack,
                   method='correlation', # Default first map may not be the best
                   **kwargs):
        """

        """

        method = method.lower()

        if method in ['correlation', 'phase_correlation']:
            shifts = relative_correlation_auto_alignment(
                            map_stack,
                            **kwargs
                        )
        elif method in ['manual']:
            shifts = manual_alignment(
                            map_stack,
                            **kwargs
                        )
        else:
            err_str = f'Unknown method ({method}) indicated.'
            raise ValueError(err_str)

        self.shifts = list(shifts)

    
    def interpolate_map_positions(self,
                                  shifts=None,
                                  map_shape=None,
                                  plotme=False):
        """

        """

        # Check inputs
        if shifts is None:
            if (hasattr(self, 'shifts')
                and self.shifts is not None):
                shifts = self.shifts
            else:
                err_str = ('Must provide shifts between maps or have '
                           + 'internally saved these shifts.')
                raise AttributeError(err_str)
        if map_shape is None:
            if (hasattr(self, 'map_shape')
                and self.map_shape is not None):
                map_shape = self.map_shape
            else:
                err_str = ('Must provide map_shape or have internal '
                        + 'map_shape attribute.')
                raise AttributeError(err_str)

        # Create generic regular coordinates
        x = np.arange(0, map_shape[1])
        y = np.arange(0, map_shape[0])

        # Redefine y-shifts to match matplotlib axes...
        shifts = np.asarray(shifts)
        shifts[:, 0] *= -1

        # Shifts stats
        x_step = np.mean(np.diff(x))
        y_step = np.mean(np.diff(y))
        ymin, xmin = np.min(shifts, axis=0)
        ymax, xmax = np.max(shifts, axis=0)
        # matching matplotlib description
        xx, yy = np.meshgrid(x, y[::-1])  

        # Determine virtual grid centers based on mean positions
        mean_shifts = np.mean(shifts, axis=0)
        xx_virt = xx + mean_shifts[1]
        yy_virt = yy + mean_shifts[0]

        # Mask out incomplete virtual pixels
        mask = np.all([
            xx_virt > np.min(x) + xmax - (x_step / 2), # left edge
            xx_virt < np.max(x) + xmin + (x_step / 2), # right edge
            yy_virt > np.min(y) + ymax - (y_step / 2), # bottom edge
            yy_virt < np.max(y) + ymin + (y_step / 2)], # top edge
                axis=0)
        xx_virt = xx_virt[mask]
        yy_virt = yy_virt[mask]
        virt_shape = (len(np.unique(yy_virt)), len(np.unique(xx_virt)))

        if plotme:
            fig, ax = plt.subplots()
            grid_colors = plt.cm.jet(np.linspace(0, 1, len(self)))

        # Contruct virtual masks of full grids to fill virtual grid
        virtual_masks = []
        for i, shift in enumerate(shifts):
            xxi = (xx + shift[1])
            yyi = (yy + shift[0])

            xx_ind, yy_ind = xx_virt[0], yy_virt[0]
            vmask_x0 = np.argmin(np.abs(xxi[0] - xx_ind))
            vmask_y0 = np.argmin(np.abs(yyi[:, 0] - yy_ind))
            y_start = xx.shape[0] - (virt_shape[0] + vmask_y0)
            y_end = xx.shape[0] - vmask_y0
            x_start = vmask_x0
            x_end = vmask_x0 + virt_shape[1]

            # print(vmask_x0, vmask_y0)
            vmask = np.zeros_like(xx, dtype=np.bool_)
            vmask[y_start : y_end,
                  x_start : x_end] = True
            virtual_masks.append(vmask)

            if plotme:
                ax.scatter(xxi.flatten(),
                           yyi.flatten(),
                           s=5,
                           color=grid_colors[i])

        # Store parameter. Write to hdf?
        self.virtual_masks = virtual_masks

        if plotme:
            ax.scatter(xx_virt,
                       yy_virt,
                       s=20,
                       c='r',
                       marker='*')

            # This can probably be done with RegularPolyCollection
            # but that proved finicky
            rect_list = []
            for xi, yi in zip(xx_virt, yy_virt):
                # Create a Rectangle patch
                rect = patches.Rectangle((xi - (x_step / 2),
                                          yi - (y_step / 2)),
                                          x_step,
                                          y_step,
                                          linewidth=1,
                                          edgecolor='gray',
                                          facecolor='none')
                rect_list.append(rect)
            pc = PatchCollection(rect_list, match_original=True)
            ax.add_collection(pc)

            ax.set_aspect('equal')
            fig.show()

    
    def stack_vector_maps(self,
                          vector_maps=None,
                          virtual_masks=None,
                          int_cutoff=None,
                          rewrite_data=False):
        """

        """

        # Check inputs
        if virtual_masks is None:
            if (hasattr(self, 'virtual_masks')
                and self.virtual_masks is not None):
                virtual_masks = self.virtual_masks
            else:
                err_str = ('Must provide virtual_masks or have '
                           + 'internal virutal_masks attribute.')
                raise AttributeError(err_str)
        
        # Compare inputs if provided
        if vector_maps is not None: 
            if (len(vector_maps) != len(self)
                or len(virtual_masks) != len(self)):
                err_str = ('Provided vector_maps length of '
                        + f'{len(vector_maps)} or virtual_masks length '
                        + f'of {len(virtual_masks)} does not match '
                        + f'XRDMapStack length of {len(self)}.')
                raise ValueError(err_str)
        # Check everything is available before loading        
        else: 
            @XRDMap._protect_hdf()
            def check_xrdmap_for_vector_map(xrdmap, i):
                if ('vectorized_map'
                    not in xrdmap.hdf[xrdmap._hdf_type]):
                        err_str = ('Could not find vector_map for '
                                   + f'XRDMap[{i}] internally or in '
                                   + 'HDF file.')
                        raise RuntimeError(err_str)

            for i, xrdmap in enumerate(self):
                if (not hasattr(xrdmap, 'vector_map')
                    or xrdmap.vector_map is None):
                    check_xrdmap_for_vector_map(xrdmap, i)

        # Construct full vector array
        print('Combining all vectorized maps...')
        vmask_shape = (np.max(virtual_masks[0].sum(axis=0)),
                       np.max(virtual_masks[0].sum(axis=1)))

        full_vector_map = np.empty(vmask_shape, dtype=object)

        for i, xrdmap in enumerate(self):
            if vector_maps is None:
                if (hasattr(xrdmap, 'vector_map')
                    and xrdmap.vector_map is not None):
                    vector_map = xrdmap.vector_map
                    # Remove reference to save on memory later
                    del xrdmap.vector_map
                else:
                    xrdmap.load_vector_map()
                    vector_map = xrdmap.vector_map
                    # Remove reference to save on memory later
                    del xrdmap.vector_map 
            else:
                vector_map = vector_maps[i]
                # Remove reference to save on memory later
                vector_maps[i] = None

            virt_index = 0
            for indices in self[0].indices:
                # Skip if not in virtual_mask
                if not virtual_masks[i][indices]:
                    continue
                
                virt_indices = np.unravel_index(virt_index,
                                                vmask_shape)
                
                if int_cutoff is not None:
                    mask = vector_map[indices][:, -1] > int_cutoff
                else:
                    mask = np.ones(len(vector_map[indices]), dtype=np.bool_)
                
                if full_vector_map[virt_indices] is None:
                    full_vector_map[virt_indices] = vector_map[indices][mask]
                else:
                    full_vector_map[virt_indices] = np.vstack([
                            full_vector_map[virt_indices],
                            vector_map[indices][mask]
                        ])
                virt_index += 1
            
            # Release memory
            del vector_map
        
        # Store internally
        self.xdms_vector_map = full_vector_map
        print('done!')
        
        # Find edges too
        self.get_sampled_edges()

        # Write to hdf
        self.save_xdms_vector_map(rewrite_data=rewrite_data)

    ################################################
    ### Spot Search and Indexing Vectorized Data ###
    ################################################


    def find_3D_spots(self,
                      abs_int_cutoff=None,
                      max_dist=0.1,
                      significance=0.1,
                      subsample=1,
                      label_int_method='sum',
                      verbose=False,
                      save_to_hdf=True,
                      rewrite_data=True):
        """

        """

        if not hasattr(self, 'xdms_vector_map'):
            err_str = ("Vectors maps have not yet been combined for "
                       + "XRDMapStack. Call 'stack_vector_maps' to "
                       + "combine vector maps from each XRDMap with "
                       + "after determining appropriate shifts.")
            raise AttributeError(err_str)
        
        if hasattr(self, 'spots_3D'):
            warn_str = ('WARNING: 3D spots already determined. '
                        + 'Rewriting with new spots.')
            print(warn_str)

        map_shape = self.xdms_vector_map.shape

        # Creating a holder for dictionaries of each map pixel
        df_keys = ['map_x',
                   'map_y',
                   'height',
                   'intensity',
                   'qx',
                   'qy',
                   'qz',
                   'q_mag',
                   'tth',
                   'chi',
                   'wavelength',
                   'theta']

        # Setup search function. Breaking down individually might be fast
        @dask.delayed()
        def delayed_spot_search(indices):

            # Break down individual vectors
            q_vectors = self.xdms_vector_map[indices][:, :-1]
            intensity = self.xdms_vector_map[indices][:, -1]

            # Some level of assigning significance
            if abs_int_cutoff is not None:
                int_mask = intensity > abs_int_cutoff
                q_vectors = q_vectors[int_mask]
                intensity = intensity[int_mask]

            # Find spots if there are vectors to index
            if len(intensity) > 0:
                (spot_labels,
                 spots,
                 label_ints,
                 label_maxs) = rsm_spot_search(
                                    q_vectors,
                                    intensity,
                                    max_dist=max_dist,
                                    significance=significance,
                                    subsample=subsample,
                                    label_int_method=label_int_method,
                                    verbose=verbose)
                
                if len(spots) == 0:
                    spot_labels = []
                    temp_lists = [[] for _ in range(len(df_keys))]
                    return temp_lists, spot_labels
                
                # Convert reciprocal positions to polar units
                if self.rocking_axis == 'energy':
                    if self.use_stage_rotation:
                        stage_rotation = self.theta[0]
                    else:
                        stage_rotation = 0
                    tth, chi, wavelength = q_2_polar(spots,
                                        stage_rotation=stage_rotation,
                                        degrees=(
                                        self.polar_units == 'deg'))
                    theta = [self.theta[0],] * len(spots)
                else: # angle
                    tth, chi, theta = q_2_polar(spots,
                                        wavelength=self.wavelength[0],
                                        degrees=(
                                        self.polar_units == 'deg'))
                    wavelength = [self.wavelength[0],] * len(wavelength)
            
                # Create temporary dictionary to store information
                temp_lists = [[indices[1],]*len(spots), # map_x
                              [indices[0],]*len(spots),  # map_y
                              label_maxs, # height
                              label_ints, # intensity
                              *spots.T, # qx, qy, qz
                              np.linalg.norm(spots, axis=1), # q_mag
                              tth,
                              chi, 
                              wavelength,
                              theta
                              ]
            else:
                spot_labels = []
                temp_lists = [[] for _ in range(len(df_keys))]

            return temp_lists, spot_labels

        # Iterate through each spatial pixel of map
        delayed_list = []
        for index in range(np.prod(map_shape)):
            indices = np.unravel_index(index, map_shape)
            
            # Collect scheduled calls
            delayed_list.append(delayed_spot_search(indices))

        # Compute scheduled operations
        if not verbose:
            with TqdmCallback(tqdm_class=tqdm):
                proc_list = dask.compute(*delayed_list)
        else:
            proc_list = dask.compute(*delayed_list)

        # Unpack data into useable format
        df_data = []
        for i in range(len(df_keys)):
            df_data.append(list(itertools.chain(*[res[0][i] for res in proc_list])))

        # Compile spots
        full_dict = dict(zip(df_keys, df_data))
        self.spots_3D = pd.DataFrame.from_dict(full_dict)

        # Store spot labels
        self.xdms_spot_labels_map = np.empty(map_shape, dtype=object)
        for index in range(np.prod(map_shape)):
            indices = np.unravel_index(index, map_shape)
            self.xdms_spot_labels_map[indices] = np.asarray((proc_list[index][-1])).squeeze()

        # Write to hdf
        if save_to_hdf:
            self.save_3D_spots(
                    extra_attrs={
                            'abs_int_cutoff' : abs_int_cutoff,
                            'max_dist' : max_dist,
                            'significance' : significance,
                            'subsample' : subsample,
                            'label_int_method' : label_int_method})
            self.save_xdms_vector_map_information(
                    self.xdms_spot_labels_map,
                    'spot_labels_map',
                    rewrite_data=rewrite_data,
                    extra_attrs={
                            'abs_int_cutoff' : abs_int_cutoff,
                            'max_dist' : max_dist,
                            'significance' : significance,
                            'subsample' : subsample,
                            'label_int_method' : label_int_method})

    
    def index_all_3D_spots(self,
                           near_q,
                           near_angle,
                           degrees=True,
                           phase=None,
                           save_to_hdf=True,
                           verbose=False,
                           symmetrize='lattice',
                           **kwargs):
        """

        """

        if not hasattr(self, 'spots_3D') or self.spots_3D is None:
            err_str = 'Spots must be found before they can be indexed.'
            raise AttributeError(err_str)

        if phase is None:
            if len(self.phases) == 1:
                phase = list(self.phases.values())[0]
            else:
                err_str = 'Phase must be provided for indexing.'
                raise ValueError(err_str)
        
        # Effective map shape
        map_shape = (np.max(self.spots_3D['map_y']) + 1,
                     np.max(self.spots_3D['map_x']) + 1)

        # Get phase information
        max_q = np.max(self.spots_3D['q_mag'])
        phase.generate_reciprocal_lattice(1.15 * max_q)
        all_ref_qs = phase.all_qs.copy()
        all_ref_hkls = phase.all_hkls.copy()
        all_ref_fs = phase.all_fs.copy()

        ref_mags = np.linalg.norm(all_ref_qs, axis=1)
        
        # Update spots dataframe with new columns
        self.spots_3D['phase'] = ''
        self.spots_3D[['grain_id', 'h', 'k', 'l', 'qof']] = np.nan

        # Construct iterable
        if verbose:
            iterable = timed_iter(range(np.prod(map_shape)))
        else:
            iterable = tqdm(range(np.prod(map_shape)))

        # Iterate through each spatial pixel of map
        for index in iterable:
            indices = np.unravel_index(index, map_shape)
            if verbose:
                print(f'Indexing for map indices {indices}.')
            
            pixel_df = self.pixel_3D_spots(indices, copied=False)
            spots = pixel_df[['qx', 'qy', 'qz']].values
            
            if len(spots) > 1 and not are_collinear(spots):
                spot_mags = pixel_df['q_mag'].values
                spot_ints = pixel_df['intensity'].values
                ext = 0.15
                ref_mask = ((ref_mags > spot_mags.min() * (1 - ext))
                            & (ref_mags < spot_mags.max() * (1 + ext)))
                
                # Modify phase attributes for indexing
                phase.all_qs = all_ref_qs[ref_mask]
                phase.all_hkls = all_ref_hkls[ref_mask]
                phase.all_fs = all_ref_fs[ref_mask]
                
                (indexings,
                 qofs) = phase_index_all_grains(
                                    phase,
                                    spots,
                                    spot_ints,
                                    near_q,
                                    near_angle,
                                    self.qmask,
                                    degrees=degrees,
                                    verbose=verbose,
                                    symmetrize=symmetrize,
                                    **kwargs
                                    )

                for gid, (ind, qof) in enumerate(zip(indexings, qofs)):
                    if np.isnan(qof):
                        continue

                    spot_inds, ref_inds = ind.T
                    hkls = all_ref_hkls[ref_mask][ref_inds]

                    # Assign values
                    rel_ind = pixel_df.index[spot_inds]
                    self.spots_3D.loc[rel_ind, 'phase'] = phase.name
                    self.spots_3D.loc[rel_ind, 'grain_id'] = gid
                    self.spots_3D.loc[rel_ind, 'qof'] = qof
                    self.spots_3D.loc[rel_ind, ['h', 'k', 'l']] = hkls

        # Write to hdf
        if save_to_hdf:
            self.save_3D_spots(
                    extra_attrs={'near_q' : near_q,
                                 'near_angle' : near_angle,
                                 'degrees' : int(degrees)})
    

    def pixel_3D_spots(self,
                       map_indices,
                       spots_3D=None,
                       copied=True):
        """

        """

        if spots_3D is None:
            spots_3D = self.spots_3D

        return XRDMap._pixel_spots(spots_3D,
                                   map_indices,
                                   copied=copied)


    @_check_xdms_swapped_axes
    @_protect_xdms_hdf(pandas=True)
    def save_3D_spots(self, extra_attrs=None):
        """

        """

        print('Saving 3D spots to the HDF file...',
              end='', flush=True)
        hdf_str = f'{self._hdf_type}/reflections/'
        self.spots_3D.to_hdf(self.xdms_hdf_path,
                             key=f'{hdf_str}/spots_3D',
                             format='table')

        if extra_attrs is not None:
            if self.xdms_hdf is None:
                self.open_xdms_hdf()
            for key, value in extra_attrs.items():
                overwrite_attr(self.xdms_hdf[hdf_str].attrs, key, value)      
        print('done!')


    @_protect_xdms_hdf()
    def trim_3D_spots(self,
                      remove_less=0.01,
                      key='intensity',
                      save_spots=False,
                      save_metadata=True):
        """

        """
        
        XRDBaseScan._trim_spots(self.spots_3D,
                                remove_less=remove_less,
                                key=key)

        # Overwrite current spots. This will fully delete trimmed spots
        if save_spots:
            self.save_spots()
        
        # Track trimming metadata. Needed to fully replicate processing
        if save_metadata:
            self.open_xdms_hdf()
            hdf_str = f'{self._hdf_type}/reflections/spots_3D'
            overwrite_attr(self.xdms_hdf[hdf_str].attrs,
                           'trimmed_key', key)
            overwrite_attr(self.xdms_hdf[hdf_str].attrs,
                           'trimmed_value', remove_less)

    
    def _spots_3D_to_vectors(self,
                             spots_3D=None,
                             map_shape=None):
        """
        Construct vector map from 3D spots pandas dataframe.
        """

        if (spots_3D is None
            and (not hasattr(self, 'spots_3D')
                 or self.spots_3D is None)):
            err_str = ("XRDMapStack must have 'spots_3D' attribute or "
                       + "they must be provided.")
            raise AttributeError(err_str)

        if map_shape is None:
            if (hasattr(self, 'xdms_vector_map')
                and self.xdms_vector_map is not None):
                map_shape = self.xdms_vector_map.shape
            else:
                warn_str = ('WARNING: Ambiguous map shape when '
                            + 'converting from 3D spots to vectors. '
                            + 'Using shape estimated from spot map '
                            + 'extents.')
                print(warn_str)
                map_shape = (np.max(self.spots_3D['map_y']),
                            np.max(self.spots_3D['map_x']))
        
        vector_map = np.empty(map_shape, dtype=object)

        for index in range(np.prod(vector_map.shape)):
            indices = np.unravel_index(index, vector_map.shape)
            df = self.pixel_3D_spots(indices)
            vector_map[indices] = df[['qx', 'qy', 'qz', 'intensity']].values

        return vector_map


    ##########################
    ### Plotting Functions ###
    ##########################

    # Disable q-space plotting
    def plot_q_space(*args, **kwargs):
        """

        """
        err_str = ('Q-space plotting not supported for '
                   + 'XRDMapStack, since Ewald sphere and/or '
                   + 'crystal orientation changes during scanning.')
        raise NotImplementedError(err_str)


    @return_plot_wrapper
    def plot_map_stack(self,
                       map_stack,
                       slider_vals=None,
                       slider_label=None,
                       title=None,
                       shifts=None,
                       title_scan_id=True,
                       **kwargs,
                       ):
        """

        """

        if slider_vals is None:
            if self.rocking_axis == 'energy':
                slider_vals = self.energy
            if self.rocking_axis == 'angle':
                slider_vals = self.theta
        
        if slider_label is None:
            if self.rocking_axis == 'energy':
                slider_label = 'Energy [keV]'
            if self.rocking_axis == 'angle':
                slider_label = 'Angle [deg]'

        if (shifts is None
            and hasattr(self, 'shifts')):
            if np.any(np.isnan(self.shifts)):
                shifts = None
            else:
                shifts = self.shifts

        if title is None:
            title = 'Map Stack'
        if title_scan_id:
            title = (f'scan{self.scan_id[0]}-{self.scan_id[-1]}:'
                     + f' {title}')

        fig, ax, slider = base_slider_plot(
                                map_stack,
                                slider_vals=slider_vals,
                                slider_label=slider_label,
                                title=title,
                                shifts=shifts,
                                **kwargs
                                )
        
        # matplotlib likes to keep a reference
        self._map_stack_slider = slider 
        return fig, ax
    

    @return_plot_wrapper
    def plot_interactive_map(self,
                             spots_3D=False,
                             dyn_kw=None,
                             map_kw=None,
                             labeled=False,
                             use_grains=False,
                             title_scan_id=True,
                             **kwargs):
        """

        """
        
        # Python doesn't play well with mutable default kwargs
        if dyn_kw is None:
            dyn_kw = {}
        if map_kw is None:
            map_kw = {}
        default_map_title = None

        # Try for spots first
        if _check_dict_key(dyn_kw, 'data'):
            pass
        elif isinstance(spots_3D, pd.DataFrame):
            dyn_kw['data'] = self._spots_3D_to_vectors(
                                        spots_3D=spots_3D)
            default_map_title = 'Max Spot Intensity'
        elif spots_3D:
            if hasattr(self, 'spots_3D') and self.spots_3D is not None:
                spots_3D = self.spots_3D
                dyn_kw['data'] = self._spots_3D_to_vectors()
                default_map_title = 'Max Spot Intensity'
        
        # Parse labeled inputs
        if labeled:
            if (not isinstance(spots_3D, pd.DataFrame)
                or not all([index in spots_3D
                            for index in {'h', 'k', 'l'}])):
                err_str = ('Cannot plot labeled values without '
                        + 'labeled 3D spots.')
                raise ValueError(err_str)
        
        # Try vectors if still no data
        if _check_dict_key(dyn_kw, 'data'):
            pass
        elif not hasattr(self, 'xdms_vector_map'):
            err_str = ('Could not find stacked vector map to plot '
                       + 'data!')
            raise ValueError(err_str)
        else:
            dyn_kw['data'] = self.xdms_vector_map
            default_map_title = 'Max Vector Intensity'
        
        # Add edges
        if not _check_dict_key(dyn_kw, 'edges'):
            if hasattr(self, 'edges') and self.edges is not None:
                dyn_kw['edges'] = self.edges

        # Add default map_kw information if not already included
        if _check_dict_key(map_kw, 'map'):
            if map_kw['map'].shape != dyn_kw['data'].shape:
                warn_str = ("WARNING: Provided map of shape "
                            + f"{map_kw['map'].shape} does not match "
                            + f"data of shape {dyn_kw['data'].shape}. "
                            + "Using default map instead.")
                print(warn_str)
                del map_kw['map']

        # Construct default map
        if not _check_dict_key(map_kw, 'map'):
            map_kw['map'] = get_max_vector_map(dyn_kw['data'])
            if default_map_title is None:
                default_map_title = 'Max Vector Intensity'
            map_kw['title'] = default_map_title

        # Construct default x ticks
        if not _check_dict_key(map_kw, 'x_ticks'):
            x_step = np.max([np.mean(np.diff(self[0].pos_dict['interp_x'],
                                             axis=i))
                             for i in range(2)])
            x_ext = map_kw['map'].shape[1]
            map_kw['x_ticks'] = np.linspace(-x_ext / 2,
                                            x_ext / 2,
                                            int(np.round(x_ext
                                                         / x_step)))

            if not _check_dict_key(map_kw, 'x_label'):
                map_kw['x_label'] = ('relative x position '
                                     + f'[{self[0].position_units}]')
        
        # Construct default y ticks
        if not _check_dict_key(map_kw, 'y_ticks'):
            y_step = np.max([np.mean(np.diff(self[0].pos_dict['interp_y'],
                                             axis=i))
                             for i in range(2)])
            y_ext = map_kw['map'].shape[0]
            map_kw['y_ticks'] = np.linspace(-y_ext / 2,
                                            y_ext / 2,
                                            int(np.round(y_ext
                                                         / y_step)))

            if not _check_dict_key(map_kw, 'y_label'):
                map_kw['y_label'] = ('relative y position '
                                     + f'[{self[0].position_units}]')

        # Construct default labels 
        if not _check_dict_key(map_kw, 'x_label'):
            map_kw['x_label'] = ('x position '
                                    + f'[{self[0].position_units}]')
        if not _check_dict_key(map_kw, 'y_label'):
            map_kw['y_label'] = ('y position '
                                    + f'[{self[0].position_units}]')
        
        # Construct / append title
        if 'title' not in map_kw:
            map_kw['title'] = None
        map_kw['title'] = self._title_with_scan_id(
                            map_kw['title'],
                            default_title='Custom Map',
                            title_scan_id=title_scan_id)

        if not labeled:
            # Plot!
            return interactive_3D_plot(dyn_kw,
                                       map_kw,
                                       **kwargs)
        
        else: # Build labels
            map_shape = map_kw['map'].shape
            labels = np.empty(map_shape, dtype=object)
            label_spots = labels.copy()
            label_colors = labels.copy()

            color_sequence = color_sequences['tab10']
            max_inds = len(color_sequence)
            for index in range(np.prod(map_shape)):
                indices = np.unravel_index(index, map_shape)
                df = self.pixel_3D_spots(indices, spots_3D=spots_3D)
                grain_ids = df['grain_id'].values
                grain_ids = np.unique(grain_ids[~np.isnan(grain_ids)]).astype(int)
                if use_grains:
                    grain_ids = grain_ids[grain_ids < max_inds]
                elif len(grain_ids) > max_inds:
                    grain_ids = grain_ids[:max_inds]
                
                # Build labels
                hkl_list = []
                spot_list = []
                color_list = []
                for i, grain_id in enumerate(grain_ids):
                    grain_mask = df['grain_id'] == grain_id
                    spots = df[grain_mask][['qx', 'qy', 'qz']].values
                    hkls = df[grain_mask][['h', 'k', 'l']].values
                    hkl_list.extend([f'({int(h)} {int(k)} {int(l)})' for h, k, l in hkls])
                    spot_list.extend(spots)
                    if use_grains:
                        colors = [color_sequence[grain_id],] * sum(grain_mask)
                    else:
                        colors = [color_sequence[i],] * sum(grain_mask)
                    color_list.extend(colors)
                
                labels[indices] = hkl_list
                label_spots[indices] = spot_list
                label_colors[indices] = color_list
            
            dyn_kw['labels'] = labels
            dyn_kw['label_spots'] = label_spots
            dyn_kw['label_colors'] = label_colors

            # Plot!
            return interactive_3D_labeled_plot(
                                    dyn_kw,
                                    map_kw,
                                    **kwargs)


