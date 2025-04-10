# Note: This class is highly experimental
# It is intended to wrap the XRDMap class and convert every method to apply iteratively across a stack of XRDMaps
# This is instended for 3D RSM mapping where each method / parameter may change between maps

import os
import h5py
import numpy as np
import pandas as pd
import functools
from matplotlib import patches
from matplotlib.collections import PatchCollection

from xrdmaptools.XRDMap import XRDMap
from xrdmaptools.XRDRockingCurve import XRDRockingCurve
from xrdmaptools.utilities.utilities import (
    timed_iter,
    pathify,
    _check_dict_key,
    get_int_vector_map,
    get_max_vector_map,
    get_num_vector_map
)
from xrdmaptools.io.hdf_io import (
    initialize_xrdmapstack_hdf,
    load_xrdmapstack_hdf,
    _load_xrd_hdf_vectorized_map_data
    )
from xrdmaptools.io.hdf_utils import (
    check_attr_overwrite,
    overwrite_attr
)
from xrdmaptools.geometry.geometry import QMask
from xrdmaptools.crystal.rsm import map_2_grid
from xrdmaptools.crystal.map_alignment import (
    relative_correlation_auto_alignment,
    com_auto_alignment,
    manual_alignment
)
from xrdmaptools.plot.image_stack import base_slider_plot
from xrdmaptools.plot.interactive import (
    interactive_3D_plot
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
        Custom file name of hdf file. Defaults to include the filename
        parameter if not provided.
    xdms_hdf : 
        h5py File instance. Will be used to derive hdf_filename and hdf
        path location if provided.
    save_hdf : bool, optional
        If False, this flag disables all hdf read/write functions.
        True by default.
    xdms_extra_metadata : dict, optional
        Dictionary of extra metadata to be stored with the XRD data.
        Extra metadata will be written to the hdf file if enabled, but
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
        self._shifts = [np.nan,] * len(self)

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
            self.shifts = shifts
        
        # Catch-all of extra attributes.
        # Gets them into __init__ sooner
        if xdms_extra_attrs is not None:
            for key, value in xdms_extra_attrs.items():
                setattr(self, key, value)

        # Define several methods
        # Probably not the Pythonic way to do this...
        self._construct_iterable_methods()
        self._construct_verbatim_methods()


    ################################################
    ### Re-Written XRDMap Methods and Properties ###
    ################################################
     
    
    def _list_property_constructor(property_name,
                                   include_set=False,
                                   include_del=False):

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

        if include_set:
            def set_property(self, values):
                [setattr(xrdmap, property_name, val)
                 for xrdmap, val in zip(self, values)]

        if include_del:
            def del_property(self):
                [delattr(xrdmap, property_name) for xrdmap in self]

        return property(get_property,
                        set_property,
                        del_property)
    
    
    def _universal_property_constructor(property_name,
                                        include_set=False,
                                        include_del=False):
        # Uses first xrdmap in list as container object
        # Sets and deletes from all list elements though
        def get_property(self):
            if hasattr(self[0], property_name):
                return getattr(self[0], property_name)
            else:
                err_str = ('XRDMap [0] does not have universal '
                           + f'attribute {property_name}.')
                raise AttributeError(err_str)
        
        set_property, del_property = None, None
        
        if include_set:
            def set_property(self, value): # This may break some of them...
                [setattr(xrdmap, property_name, value)
                 for xrdmap in self]

        if include_del:
            def del_property(self):
                [delattr(xrdmap, property_name) for xrdmap in self]

        return property(get_property,
                        set_property,
                        del_property)


    energy = _list_property_constructor(
                                'energy',
                                include_set=True)
    wavelength = _list_property_constructor(
                                'wavelength',
                                include_set=True)
    # q_arr = _list_property_constructor(
    #                             'q_arr',
    #                             include_del=True)
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
    _swapped_axes = _list_property_constructor('_swapped_axes',
                                               include_set=True)

    # Universal attributes
    beamline = _universal_property_constructor('beamline')
    facility = _universal_property_constructor('facility')
    shape = _universal_property_constructor('shape')
    image_shape = _universal_property_constructor('image_shape')
    map_shape = _universal_property_constructor('map_shape')
    num_images = _universal_property_constructor('num_images')

    # Should be universal, could lead to some unexpected behaviors...
    tth_resolution = _list_property_constructor('tth_resolution',
                                                include_set=True)
    chi_resolution = _list_property_constructor('chi_resolution',
                                                include_set=True)

    # Not guaranteed attributes. May throw errors
    # blob_masks = _list_property_constructor('blob_masks')

    ##############################
    ### XRDMapStack Properties ###
    ##############################

    @property
    def q_arr(self):

        if hasattr(self, '_q_arr'):
            return self._q_arr
        else:
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
        return [xdm._swapped_axes for xdm in self]
    
    @_swapped_axes.setter
    def _swapped_axes(self, val):
        if not isinstance(val, bool):
            raise TypeError('Swapped axes flag must be boolean.')
        
        for xdm in self:
            if xdm._swapped_axes != val:
                xdm.swap_axes()


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
                         map_shape=None,
                         image_shape=None,
                         **kwargs):

        if wd is None:
            wd = [os.getcwd(),] * len(hdf_filenames)
        elif isinstance(wd, str):
            wd = [wd,] * len(hdf_filenames)
        
        if xdms_wd is None:
            xdms_wd = wd[0]

        # Check that each file exists
        for filename, wdi in zip(hdf_filenames, wd):
            path = pathify(wdi, filename, '.h5')
            if not os.path.exists(path):
                err_str = f'File {path} cannot be found.'
                raise FileNotFoundError(err_str)
        
        xrdmap_list = []
        for hdf_filename, wdi in timed_iter(zip(hdf_filenames,
                                                wd),
                                            total=len(hdf_filenames),
                                            iter_name='xrdmap'):
            xrdmap_list.append(
                XRDMap.from_hdf(
                    hdf_filename,
                    wd=wdi,
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

        return cls(stack=xrdmap_list,
                   wd=xdms_wd,
                   xdms_filename=xdms_filename,
                   xdms_hdf_filename=xdms_hdf_filename,
                   xdms_hdf=xdms_hdf,
                   save_hdf=save_hdf)

    
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

        if wd is None:
            wd = os.getcwd()
        
        xdms_hdf_path = pathify(wd, xdms_hdf_filename, '.h5')
        if not os.path.exists(xdms_hdf_path):
            # Should be redundant...
            raise FileNotFoundError(f'No hdf file at {xdms_hdf_path}.')
        
        # File exists, attempt to load data!
        print('Loading data from hdf file...')
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
        # xdms_hdf = input_dict.pop('xdms_hdf')

        xdms_extra_attrs = {}
        if input_dict['vector_dict'] is not None:
            (xdms_extra_attrs['xdms_vector_map'] # rename
             ) = input_dict['vector_dict'].pop('vector_map')
            xdms_extra_attrs.update(input_dict['vector_dict'])
        
        xrdmap_list = []
        for hdf_path_i in timed_iter(hdf_path, iter_name='xrdmap'):
            hdf_filename_i = os.path.basename(hdf_path_i)
            wd_i = os.path.dirname(hdf_path_i)
            xrdmap_list.append(
                XRDMap.from_hdf(
                    hdf_filename_i,
                    wd=wd_i,
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
                   wd=wd,
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

    # List wrapper to allow kwarg inputs
    def _blank_iterator(self, iterable, **kwargs):
        return list(iterable)


    # This is currently called during __init__,
    # but may work as a decorator within the class
    def _get_iterable_method(self,
                             method,
                             variable_inputs=False,
                             timed_iterator=False):

        flags = (variable_inputs, timed_iterator)
        
        # Check that method exists in all xrdmaps
        for i, xrdmap in enumerate(self):
            if not hasattr(xrdmap, method):
                err_str = (f'XRDMap [{i}] does not '
                           + f'have {method} method.')
                raise AttributeError(err_str)

        # Select iterator
        if timed_iterator:
            iterator = timed_iter
        else:
            iterator = self._blank_iterator

        if variable_inputs:
            # Lists of inputs
            def iterated_method(*arglists, **kwarglists):

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
            def iterated_method(*args, **kwargs):
                for i, xrdmap in iterator(enumerate(self),
                                          total=len(self),
                                          iter_name='XRDMap'):
                    getattr(xrdmap, method)(*args, **kwargs)
        
        return iterated_method
    

    def _construct_iterable_methods(self):
        for (method, var, timed) in self.iterable_methods:
            setattr(self, method,
                    self._get_iterable_method(
                        method,
                        variable_inputs=var,
                        timed_iterator=timed
                    ))


    # List of iterable methods
    iterable_methods = ( # function, variable_inputs, timed_iterator
        # hdf functions
        ('start_saving_hdf', False, False),
        ('save_current_hdf', False, True),
        ('stop_saving_hdf', False, False),
        # Light image manipulation
        ('load_images_from_hdf', True, True),
        ('dump_images', False, False),
        # Working with calibration
        # Do not accept variable calibrations
        ('set_calibration', False, False), 
        ('save_calibration', False, True),
        ('integrate1d_map', False, True),
        ('integrate2d_map', False, True),
        ('save_reciprocal_positions', False, True),
        # Working with positions only
        ('set_positions', True, False), 
        ('save_sclr_pos', False, False),
        ('swap_axes', False, False),
        ('map_extent', False, False),
        # Working with phases
        # This one saves to individual hdfs
        ('save_phases', False, False), 
        # Working with spots
        ('find_blobs', False, True),
        ('find_spots', False, True),
        ('recharacterize_spots', False, True),
        ('fit_spots', False, True),
        ('initial_spot_analysis', False, True),
        ('trim_spots', False, False),
        ('remove_spot_guesses', False, False),
        ('remove_spot_fits', False, False),
        ('save_spots', False, True),
        ('vectorize_map_data', False, True),
        # Working with xrfmap
        # Multiple inputs may be crucial here
        ('load_xrfmap', True, False) 
    )
        
    
    ##########################
    ### Verbatim Functions ###
    ##########################
    
    def _get_verbatim_method(self, method):

        def verbatim_method(*args, **kwargs):
            getattr(self[0], method)(*args, **kwargs)
        
        return verbatim_method


    # Convenience function to call within __init__
    def _construct_verbatim_methods(self):
        
        for method in self.verbatim_methods:
            setattr(self,
                    method,
                    self._get_verbatim_method(method))
            
    # List of verbatim methods
    verbatim_methods = (
        'estimate_polar_coords',
        'estimate_image_coords',
        'integrate1d_image',
        'integrate2d_image',
        # This will modify first xrdmap.phases
        # This is just a reference to self.phases
        'add_phase',
        'remove_phase',
        'load_phase',
        'clear_phases',
        # Plotting functions
        # '_title_with_scan_id',
        'plot_detector_geometry',
        'plot_map' # May break on extent
    )

    ################################
    ### XRDMapStack HDF Methods  ###
    ################################

    # Re-defined from XRDData class
    # New names and no dask concerns
    def _protect_xdms_hdf(pandas=False):
        def protect_hdf_inner(func):
            @functools.wraps(func)
            def protector(self, *args, **kwargs):
                # Check to see if read/write is enabled
                if self.xdms_hdf_path is not None:
                    # Is a hdf reference currently active?
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

    
    # Closes any active reference to xdms_hdf
    def close_xdms_hdf(self):
        if self.xdms_hdf is not None:
            self.xdms_hdf.close()
            self.xdms_hdf = None
        

    # Opens an active reference to xdms
    def open_xdms_hdf(self):
        if self.xdms_hdf is not None:
            # Should this raise errors or just ping warnings
            note_str = ('NOTE: hdf is already open. '
                        + 'Proceeding without changes.')
            print(note_str)
            return
        else:
            self.xdms_hdf = h5py.File(self.xdms_hdf_path, 'a')


    def start_saving_xrdmapstack_hdf(self,
                                     xdms_hdf=None,
                                     xdms_hdf_filename=None,
                                     xdms_hdf_path=None,
                                     save_current=False):
        
        # Check for previous iterations
        if ((hasattr(self, 'xdms_hdf')
             and self.xdms_hdf is not None)
            or (hasattr(self, 'xdms_hdf_path')
                and self.xdms_hdf_path is not None)):
            warn_str = ('WARNING: Trying to save to hdf, but a '
                        'file or location has already been specified!'
                        '\nSwitching save files or locations should '
                        'use the "switch_xrdmapstack_hdf" function.'
                        '\nProceeding without changes.')
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

        if save_current:
            self.save_current_xrdmapstack_hdf()


    # Saves current major features
    # Calls several other save functions
    @_protect_xdms_hdf()
    def save_current_xrdmapstack_hdf(self):
        
        if self.xdms_hdf_path is None:
            print('WARNING: Changes cannot be written to hdf without '
                  + 'first indicating a file location.\nProceeding '
                  + 'without changes.')
            return

        # Save stacked vector_map
        if ((hasattr(self, 'xdms_vector_map')
             and self.xdms_vector_map is not None)
            and (hasattr(self, 'edges') and self.edges is not None)):
            self.save_xdms_vector_map()

    
    # Ability to toggle hdf saving and proceed without writing to disk.
    def stop_saving_xrdmapstack_hdf(self):
        self.close_xdms_hdf()
        self.xdms_hdf_path = None
    

    @_protect_xdms_hdf()
    def switch_xrdmapstack_hdf(self,
                             xdms_hdf=None,
                             xdms_hdf_path=None,
                             xdms_hdf_filename=None,
                             save_current=False):

        # Check to make sure the change is appropriate and correct.
        # Not sure if this should raise and error or just print a warning
        if xdms_hdf is None and xdms_hdf_path is None:
            ostr = ('Neither xdms_hdf nor xdms_hdf_path were provided. '
                     + '\nCannot switch hdf save locations without '
                     + 'providing alternative.')
            print(ostr)
            return
        
        elif xdms_hdf == self.xdms_hdf:
            ostr = (f'WARNING: provided hdf ({self.xdms_hdf.filename})'
                    + ' is already the current save location. '
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

    @_protect_xdms_hdf()
    def save_xdms_vector_map(self,
                             xdms_vector_map=None,
                             edges=None,
                             rewrite_data=False):

        # Allows for more customizability with other functions
        hdf = getattr(self, 'xdms_hdf')

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
                                hdf,
                                vector_map=xdms_vector_map,
                                edges=edges,
                                rewrite_data=rewrite_data)
        # Remove secondary reference
        del hdf
    

    # Mostly verbatim
    @_protect_xdms_hdf()
    def load_xdms_vector_map(self):
        vector_dict = _load_xrd_hdf_vectorized_map_data(
                                        self.xdms_hdf[self._hdf_type])
        self.vector_map = vector_dict['vector_map']
        self.edges = vector_dict['edges']


    # Need to modify to not look for a random image
    def plot_image(self, *args, **kwargs):
        raise NotImplementedError()

    ##################################
    ### Pseudo Inherited Functions ###
    ##################################

    def get_sampled_edges(self,
                          q_arr=None):
        XRDRockingCurve.get_sampled_edges(self,
                                          q_arr=q_arr)


    def _set_rocking_axis(self,
                          rocking_axis=None):
        XRDRockingCurve._set_rocking_axis(self,
                                          rocking_axis=rocking_axis)

    
    def _title_with_scan_id(self,
                            *args,
                            **kwargs):
        return XRDRockingCurve._title_with_scan_id(self,
                                                   *args,
                                                   **kwargs)  


    def plot_sampled_volume_outline(self,
                                    *args,
                                    **kwargs):
        XRDRockingCurve.plot_sampled_volume_outline(self,
                                                    *args,
                                                    **kwargs)      


    #####################################
    ### XRDMapStack Specific Methods  ###
    #####################################                               

    def sort_by_attr(self,
                     attr,
                     reverse=False):
        
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
    def stack_spots(self):
    
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

    
    def pixel_spots(self, map_indices):
        pixel_spots = self.spots[
                    (self.spots['map_x'] == map_indices[1])
                     & (self.spots['map_y'] == map_indices[0])].copy()
        return pixel_spots
    

    ########################
    ### Vectorizing Data ###
    ########################

    def align_maps(self,
                   map_stack,
                   method='correlation', # Default first map may not be the best
                   **kwargs):

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

        self.shifts = np.asarray(shifts)

    
    def interpolate_map_positions(self,
                                  shifts=None,
                                  map_shape=None,
                                  plotme=False):

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
                          rewrite_data=False):

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
                        + f'of {len(virutal_masks)} does not match '
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
                    # retrieve_vector_map_from_hdf(xrdmap)
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
                
                if full_vector_map[virt_indices] is None:
                    full_vector_map[virt_indices] = vector_map[indices]
                else:
                    full_vector_map[virt_indices] = np.vstack([
                            full_vector_map[virt_indices],
                            vector_map[indices]
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

    ################################
    ### Indexing Vectorized Data ###
    ################################

    @property
    def qmask(self):
        if hasattr(self, '_qmask'):
            return self._qmask
        else:
            self._qmask = QMask.from_XRDRockingScan(self)
            return self._qmask
    
    @qmask.deleter
    def qmask(self):
        del self._qmask


    ##########################
    ### Plotting Functions ###
    ##########################

    # Disable q-space plotting
    def plot_q_space(*args, **kwargs):
        err_str = ('Q-space plotting not supported for '
                   + 'XRDMapStack, since Ewald sphere and/or '
                   + 'crystal orientation changes during scanning.')
        raise NotImplementedError(err_str)


    def plot_map_stack(self,
                       map_stack,
                       slider_vals=None,
                       slider_label=None,
                       title=None,
                       shifts=None,
                       title_scan_id=True,
                       return_plot=False,
                       **kwargs,
                       ):

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

        (fig,
         ax,
         slider) = base_slider_plot(
                                map_stack,
                                slider_vals=slider_vals,
                                slider_label=slider_label,
                                title=title,
                                shifts=shifts,
                                **kwargs
                                )
        
        if return_plot:
            # matplotlib likes to keep a reference
            self.__slider = slider 
            return fig, ax
        else:
            # matplotlib likes to keep a reference
            self.__slider = slider
            fig.show()
    

    def plot_interactive_map(self,
                             dyn_kw=None,
                             map_kw=None,
                             title_scan_id=True,
                             return_plot=False,
                             **kwargs):
        
        # Python doesn't play well with mutable default kwargs
        if dyn_kw is None:
            dyn_kw = {}
        if map_kw is None:
            map_kw = {}

        if _check_dict_key(dyn_kw, 'data'):
            pass
        elif not hasattr(self, 'xdms_vector_map'):
            err_str = ('Could not find stacked vector map to plot '
                       + 'data!')
            raise ValueError(err_str)
        else:
            dyn_kw['data'] = self.xdms_vector_map
        
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
            map_kw['title'] = 'Max Vector Intensity'

        # Construct default x ticks
        if not _check_dict_key(map_kw, 'x_ticks'):
            x_step = np.max([np.mean(np.diff(self[0].pos_dict['interp_x'], axis=i)) for i in range(2)])
            x_ext = map_kw['map'].shape[1]
            map_kw['x_ticks'] = np.linspace(-x_ext / 2, x_ext / 2, int(np.round(x_ext / x_step)))

            if not _check_dict_key(map_kw, 'x_label'):
                map_kw['x_label'] = ('relative x position '
                                     + f'[{self[0].position_units}]')
        
        # Construct default y ticks
        if not _check_dict_key(map_kw, 'y_ticks'):
            y_step = np.max([np.mean(np.diff(self[0].pos_dict['interp_y'], axis=i)) for i in range(2)])
            y_ext = map_kw['map'].shape[0]
            map_kw['y_ticks'] = np.linspace(-y_ext / 2, y_ext / 2, int(np.round(y_ext / y_step)))

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

        # Plot!
        fig, ax = interactive_3D_plot(dyn_kw,
                                      map_kw,
                                      **kwargs)

        if return_plot:
            return fig, ax
        else:
            fig.show()  
