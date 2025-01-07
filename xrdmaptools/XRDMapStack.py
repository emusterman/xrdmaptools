# Note: This class is highly experimental
# It is intended to wrap the XRDMap class and convert every method to apply iteratively across a stack of XRDMaps
# This is instended for 3D RSM mapping where each method / parameter may change between maps

import os
import h5py
import numpy as np
import pandas as pd

from xrdmaptools.XRDMap import XRDMap
from xrdmaptools.utilities.utilities import (
    timed_iter,
    pathify
)
from xrdmaptools.crystal.rsm import map_2_grid


# Class for working with PROCESSED XRDMaps
# Maybe should be called XRDMapList
class XRDMapStack(list): 
    # Does not inherit XRDMap class
    # But many methods have been rewritten to interact
    # with the XRDMap methods of the same name

    # Class variables
    _hdf_type = 'xrdmapstack'

    def __init__(self,
                 stack=None,
                 shifts=None,
                 filename=None,
                 wd=None,
                 hdf=None,
                 hdf_path=None,
                 save_hdf=False,
                 ):
        
        if stack is None:
            stack = []
        list.__init__(self, stack)

        # Strict input requirements
        for i, xrdmap in enumerate(self):
            if not isinstance(xrdmap, XRDMap):
                raise ValueError(f'Stack index {i} is not an XRDMap!')

        # Collect unique phases from individual XRDMaps
        self.phases = {}
        for xrdmap in self:
            for phase in xrdmap.phases.items():
                if phase.name not in self.phases.keys():
                     self.phases[phase.name] = phase
            # Redefine individual phases to reference stack phases
            xrdmap.phases = self.phases

        # If none, map pixels may not correspond to each other
        self.shifts = shifts

        # Define several methods
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
    q_arr = _list_property_constructor(
                                'q_arr',
                                include_del=True)

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
    blob_masks = _list_property_constructor('blob_masks')

    ### Special attributes ###

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


    #####################################
    ### Loading data into XRDMapStack ###
    #####################################

    @classmethod
    def from_XRDMap_hdfs(cls,
                         hdf_filenames,
                         wd=None,
                         save_hdf=True,
                         dask_enabled=True,
                         image_data_key='recent',
                         integration_data_key='recent',
                         map_shape=None,
                         image_shape=None,
                         **kwargs):

        if wd is None:
            wd = os.getcwd()

        # Check that each file exists
        for filename in hdf_filenames:
            path = pathify(wd, filename, '.h5')
            if not os.path.exists(path):
                err_str = f'File {path} cannot be found.'
                raise FileNotFoundError(err_str)
        
        xrdmap_list = []

        for hdf_filename in timed_iter(hdf_filenames,
                                iter_name='xrdmap'):
            xrdmap_list.append(
                XRDMap.from_hdf(
                    hdf_filename,
                    wd=wd,
                    dask_enabled=dask_enabled,
                    image_data_key=image_data_key,
                    integration_data_key=integration_data_key,
                    map_shape=map_shape,
                    image_shape=image_shape,
                    save_hdf=save_hdf,
                    **kwargs
                )
            )

        return cls(stack=xrdmap_list,
                   wd=wd,
                   save_hdf=save_hdf)

    
    @classmethod
    def from_hdf(cls,
                 hdf_filename,
                 wd=None,
                 save_hdf=True,
                 ):
        raise NotImplementedError()

        # # out = load_XRDMapStack_hdf()
        
        # inst = cls.from_XRDMap_hdfs(
        #     out['list of hdfs']
        # )

        # inst.attr = out[attr]

        # return inst

    
    def load_XRDMaps(self,
                     dask_enabled=True):
        raise NotImplementedError()

        print('WARNING:')


    #########################
    ### Utility Functions ###
    #########################



    #####################################
    ### Iteratively Wrapped Functions ###
    #####################################

    # List wrapper to allow kwarg inputs
    def _blank_iterator(self, iterable, **kwargs):
        return list(iterable)


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

                # Check and fix karglists
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
    def plot_image():
        return iterated_method
    

    def _construct_iterable_methods(self):
        for (method, var, timed) in self.iterable_methods:
            setattr(self, method,
                    self._get_iterable_method(
                        method,
                        variable_inputs=var,
                        timed_iterator=timed
                    ))


    # List of iterated methods
    iterable_methods = (
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
        # Unlikely, but should support interferometers in future
        ('set_positions', True, False), 
        ('save_sclr_pos', False, False),
        ('swap_axes', False, False),
        # Working with phases
        # This one saves to individual hdfs
        ('update_phases', False, False), 
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
        # Working with xrfmap
        # Multiple inputs may be crucial here
        ('load_xrfmap', True, False) 
    )
        
    
    ##########################
    ### Verbatim Functions ###
    ##########################
    
    def _get_verbatim_method(self, method):

        def verbatim_method(*args, **kwargs):
            setattr(self,
                    method,
                    getattr(self[0], method))(*args, **kwargs)
        
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
        # Positional
        # No swap_axes or interpolate_positions.
        # Should be called during processing.
        'map_extent',
        # Plotting functions
        'plot_detector_geometry',
        'plot_map'
    )

    #####################################
    ### XRDMapStack Specific Methods ###
    #####################################

    def start_saving_xrdmapstack_hdf(self):
        raise NotImplementedError()

    def stop_saving_xrdmapstack_hdf(self):
        raise NotImplementedError()


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
        self.sort(key = lambda xrdmap : getattr(xrdmap, attr),
                  reverse=reverse)
        
        # Delete attributes based on sorting order
        if hasattr(self, '_xrf'):
            del self._xrf

    
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
            all_spots_list.append(spots)
        
        self.spots = pd.concat(all_spots_list, ignore_index=True)

    
    def pixel_spots(self, map_indices):
        pixel_spots = self.spots[
                    (self.spots['map_x'] == map_indices[1])
                     & (self.spots['map_y'] == map_indices[0])].copy()
        return pixel_spots


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
            shifts = manually_align_maps(
                            map_stack,
                            **kwargs
                        )
        else:
            err_str = f'Unknown method ({method}) indicated.'
            raise ValueError(err_str)

        self.shifts = shifts


    def vectorize_images(self):
        raise NotImplementedError()

        for i, xrdmap in enumerate(self):
            if (not hasattr(xrdmap, 'blob_masks')
                or xrdmap.blob_masks is None):
                err_str = (f'XRDMap [{i}] does not have blob_masks. '
                           + 'These are needed to avoid unecessarily '
                           + 'large file sizes.')
                raise AttributeError(err_str)

        
        edges = ([[] for _ in range(12)])

        # Reserve memory, a little faster and throws errors sooner
        q_vectors = np.zeros((len(self.wavelength), 3),
                              dtype=self.dtype)

        print('Vectorizing images...')
        filled_indices = 0
        for i, wavelength in tqdm(enumerate(self.wavelength),
                                  total=self.num_images):
            q_arr = get_q_vect(self.tth_arr,
                               self.chi_arr,
                               wavelength=wavelength,
                               degrees=self.polar_units == 'deg'
                               ).astype(self.dtype)
            q_vectors[i] = q_arr

            # Find edges
            if i == 0:
                edges[4] = q_arr[0].T
                edges[5] = q_arr[-1].T
                edges[6] = q_arr[:, 0].T
                edges[7] = q_arr[:, -1].T
            elif i == len(self.wavelength) - 1:
                edges[8] = q_arr[0].T
                edges[9] = q_arr[-1].T
                edges[10] = q_arr[:, 0].T
                edges[11] = q_arr[:, -1].T
            else: # Corners
                edges[0].append(q_arr[0, 0])
                edges[1].append(q_arr[0, -1])
                edges[2].append(q_arr[-1, 0])
                edges[3].append(q_arr[-1, -1])
        
        for i in range(4):
            edges[i] = np.asarray(edges[i])
        
        # Assign useful variables
        self.edges = edges
        self.q_vectors = q_vectors
        
        # Get all q-coordinates

        # Assign vectorization to images within blob_masks




    def save_vectorized_data(self):
        raise NotImplementedError()
    
    
    
    # Def get 3D q-coordinates

    # Combined spots...Useful for not 3D RSM analysis
    
    # Segment 3D data

    # Center of mass

    # Indexing

    # Strain math

    # 3D plotting

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
        
        if slider_labels is None:
            if self.rocking_axis == 'energy':
                slider_label = 'Energy [keV]'
            if self.rocking_axis == 'angle':
                slider_label = 'Angle [deg]'

        if (shifts is None
            and hasattr(self.shifts)):
            shifts = self.shifts

        title = self._title_with_scan_id(
                    title,
                    default_title=('XRD '
                        + f'{self.rocking_axis.capitalize()} '
                        + 'Map Stack'),
                    title_scan_id=title_scan_id)

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
            self.__slider = slider
            return fig, ax
        else:
            self.__slider = slider
            fig.show()
