# Note: This class is highly experimental
# It is intended to wrap the XRDMap class and convert every method to apply iteratively across a stack of XRDMaps
# This is instended for 3D RSM mapping where each method / parameter may change between maps

import numpy as np
import os
import h5py



# from .XRDMap import XRDMap
# from .utilities.utilities import timed_iter

from xrdmaptools.XRDMap import XRDMap
from xrdmaptools.utilities.utilities import timed_iter



class XRDMapStack(list): # Maybe should be called XRDMapList
    # Does not inherit XRDMap class
    # But many methods have been rewritten to interact
    # with the XRDMap methods of the same name

    def __init__(self,
                 stack=None,
                 aligned_maps=None,
                 ):
        
        if stack is None:
            stack = []
        list.__init__(self, stack)

        # Strict input requirements
        for i, xrdmap in enumerate(self):
            if not isinstance(xrdmap, XRDMap):
                raise ValueError(f'Stack index {i} is not an XRDMap!')
            
        # Stack all phases if they exist
        phases = {}
        for i, xrdmap in enumerate(self):
              for phase in xrdmap.phases.items():
                    if phase.name not in phases.keys():
                          phases[phase.name] = phase
        self.phases = phases

        # Aligned maps flag
        if aligned_maps is None:
            aligned_maps = False
        self._aligned_maps = aligned_maps


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
            return [getattr(xrdmap, property_name) for xrdmap in self]
        
        set_property, del_property = None, None

        if include_set:
            def set_property(self, values):
                [setattr(xrdmap, property_name, val) for xrdmap, val in zip(self, values)]

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
            return getattr(self[0], property_name)
        
        set_property, del_property = None, None
        
        if include_set:
            def set_property(self, value): # This may break some of them...
                [setattr(xrdmap, property_name, value) for xrdmap in self]

        if include_del:
            def del_property(self):
                [delattr(xrdmap, property_name) for xrdmap in self]

        return property(get_property,
                        set_property,
                        del_property)


    energy = _list_property_constructor('energy', include_set=True)
    wavelength = _list_property_constructor('wavelength', include_set=True)
    q_arr = _list_property_constructor('q_arr', include_del=True)

    tth_arr = _universal_property_constructor('tth_arr',
                                              include_del=True)
    chi_arr = _universal_property_constructor('chi_arr',
                                              include_del=True)

    scattering_units = _universal_property_constructor('scattering_units',
                                                       include_set=True)
    polar_units = _universal_property_constructor('polar_units',
                                                  include_set=True)
    image_scale = _universal_property_constructor('image_scale',
                                                  include_set=True)
    integration_scale = _universal_property_constructor('integration_scale',
                                                        include_set=True)
    
    ### Turn other individual attributes in properties ###

    # List attributes
    scanid = _list_property_constructor('scanid')
    filename = _list_property_constructor('filename')
    wd = _list_property_constructor('wd') # this one may change
    time_stamp = _list_property_constructor('time_stamp')
    scan_input = _list_property_constructor('scan_input')
    extra_metadata = _list_property_constructor('extra_metadata')
    dwell = _list_property_constructor('dwell')
    theta = _list_property_constructor('theta')
    hdf = _list_property_constructor('hdf')
    hdf_path = _list_property_constructor('hdf_path')
    map = _list_property_constructor('map') # There will be special consideration for this one...
    xrf_path = _list_property_constructor('xrf_path')
    tth = _list_property_constructor('tth')
    tth_resolution = _list_property_constructor('tth_resolution') # Universal may be better
    chi = _list_property_constructor('chi')
    chi_resolution = _list_property_constructor('chi_resolution') # Universal may be better
    ai = _list_property_constructor('ai')
    pos_dict = _list_property_constructor('pos_dict')
    sclr_dict = _list_property_constructor('sclr_dict')

    # Universal attributes
    beamline = _universal_property_constructor('beamline')
    facility = _universal_property_constructor('facility')

    def spots():
        raise NotImplementedError()

    #####################################
    ### Loading data into XRDMapStack ###
    #####################################

    @classmethod
    def from_hdf(filenames,
                 wd=None,
                 dask_enabled=False,
                 **kwargs):
        raise NotImplementedError()
    # May require loading empty ImageMaps...

    #########################
    ### Utility Functions ###
    #########################

    def _check_attribute(self):
        raise NotImplementedError()
    
    #####################################
    ### Iteratively Wrapped Functions ###
    #####################################

    # List wrapper to allow kwarg inputs
    def _blank_iterator(iterable, **kwargs):
        return list(iterable)


    def _get_iterable_method(self,
                             method,
                             variable_inputs=False,
                             timed_iterator=False):

        flags = (variable_inputs, timed_iterator)
        
        # Check that method exists in all xrdmaps
        for i, xrdmap in enumerate(self):
            if not hasattr(xrdmap, method):
                raise AttributeError(f'XRDMap [{i}] does not have {method} method.')
        
        #print(f'{method=}\n{variable_inputs=}\n{timed_iterator=}\n\n')

        # Select iterator
        if timed_iterator:
            iterator = timed_iter
        else:
            iterator = _blank_iterator

        if variable_inputs:
            # Lists of inputs
            def iterated_method(*arglists, **kwarglists):

                # Check and fix arglists
                for i, arg in enumerate(arglists):
                    if isinstance(arg, list) and len(arg) == len(self):
                        args.append(arg)
                    elif isinstance(arg, list) and len(arg) != len(self):
                        raise ValueError(f'Length of arguments do not match length of XRDMapStack.')
                    else:
                        # Redefine arg as repeated list
                        arglists[i] = [arg,] * len(self)

                # Check and fix karglists
                for key, kwarg in kwarglists.items():
                    if isinstance(kwarg, list) and len(kwarg) == len(self):
                        pass # All is well
                    elif isinstance(kwarg, list) and len(kwarg) != len(self):
                        raise ValueError(f'Length of arguments do not match length of XRDMapStack.')
                    else:
                        # Redefine kwarg as repeated list
                        kwarglists[key] = [kwarg,] * len(self)

                # Call actual method
                for i, xrdmap in iterator(enumerate(self),
                                          total=len(self),
                                          iter_name='XRDMap'):
                    args = [arg[i] for arg in arglists]
                    kwargs = dict(zip(kwarglists.keys(),
                                      [val[i] for val in kwarglists.values()]))
                    
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


    # List of iterated methods
    iterable_methods = (
        # hdf functions
        ('save_hdf', False, False),
        ('save_current_xrdmap', False, True),
        ('stop_saving_hdf', False, False),
        # Light image manipulation
        ('load_images_from_hdf', True, True),
        ('dump_images', False, False),
        # Working with calibration
        ('set_calibration', False, False), # Do not accept variable calibrations
        ('save_calibration', False, True),
        ('integrate1d_map', False, True),
        ('integrate2d_map', False, True),
        ('save_reciprocal_positions', False, True),
        # Working with positions only
        ('set_positions', True, False), # Unlikely, but should support interferometers in future
        ('save_sclr_pos', False, False),
        ('swap_axes', False, False),
        # Working with phases
        ('add_phase', False, False),
        ('remove_phase', False, False),
        ('load_phase', False, False),
        ('clear_phases', False, False),
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
        ('load_xrfmap', True, False) # Multiple inputs may be crucial here
    )
    

    ##########################
    ### Verbatim Functions ###
    ##########################
    
    def _get_verbatim_method(self, method):

        def verbatim_method(*args, **kwargs):
            setattr(self[0],
                    method,
                    getattr(self, method))(*args, **kwargs)
        
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
        'integrate_2d_image'
    )

    #####################################
    ### XRDMap Stack-specific Methods ###
    #####################################


    def sort_by_attr(self,
                     attr,
                     reverse=False):
        
        # Check for attr
        for i, xrdmap in enumerate(self):
            if not hasattr(xrdmap, attr):
                raise AttributeError(f'XRDMap [{i}] does not have attributre {attr}.')
        
        self.sort(key = lambda xrdmap : getattr(self, attr), reverse=reverse)




    # Create specific functions

    def align_maps(self,):
        raise NotImplementedError()
    
    def save_vectorized_data(self, ):
        raise NotImplementedError()
    
    def get_q_coordinates(self, ):
        raise NotImplementedError()
    
    # Def get 3D q-coordinates

    # Combined spots...Useful for not 3D RSM analysis
    
    # Segment 3D data

    # Center of mass

    # Indexing

    # Strain math

    # 3D plotting
