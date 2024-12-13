import numpy as np
import os
import h5py
import pyFAI
import pandas as pd
from pyFAI.io import ponifile
from enum import IntEnum # Only for ponifile orientation
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
import time as ttime
import matplotlib.pyplot as plt
from collections import OrderedDict
import dask.array as da
import skimage.io as io
from dask_image import imread as dask_io
from tqdm import tqdm

# Local imports
from xrdmaptools.XRDData import XRDData
from xrdmaptools.utilities.math import *
from xrdmaptools.utilities.utilities import (
    delta_array,
    pathify,
    _check_dict_key,
    generate_intensity_mask
)
from xrdmaptools.io.hdf_io import (
    initialize_xrdbase_hdf,
    load_xrdbase_hdf
    )
from xrdmaptools.io.hdf_utils import (
    check_hdf_current_images
)
from xrdmaptools.plot.general import (
    _plot_parse_xrdmap,
    _xrdmap_image,
    _xrdmap_integration,
    plot_image,
    plot_integration
    )
from xrdmaptools.plot.geometry import (
    plot_q_space,
    plot_detector_geometry
)
from xrdmaptools.geometry.geometry import *
from xrdmaptools.crystal.Phase import Phase, phase_selector


class XRDBaseScan(XRDData):
    '''
    Base class for general functions working with XRD data.
    Built more for analysis and interpretable data
    '''

    # Class variables
    _hdf_type = 'xrdbase'

    def __init__(self,
                 scan_id=None,
                 wd=None,
                 filename=None,
                 hdf_filename=None,
                 hdf=None,
                 energy=None,
                 wavelength=None,
                 dwell=None,
                 theta=None,
                 use_stage_rotation=False,
                 poni_file=None,
                 sclr_dict=None,
                 tth_resolution=None,
                 chi_resolution=None,
                 tth=None, # Used when loading from hdf
                 chi=None, # Used when loading from hdf
                 beamline='5-ID (SRX)',
                 facility='NSLS-II',
                 scan_input=None,
                 time_stamp=None,
                 extra_metadata=None,
                 save_hdf=True,
                 dask_enabled=False,
                 extra_attrs=None,
                 **xrddatakwargs
                 ):
        
        # Adding some metadata
        self.scan_id = scan_id
        if filename is None:
            filename = f'scan{scan_id}_{self._hdf_type}'
        self.filename = filename

        if wd is None:
            wd = os.getcwd()
        self.wd = wd

        self.beamline = beamline
        self.facility = facility
        self.time_stamp = time_stamp
        self.scan_input = scan_input
        if extra_metadata is None:
            extra_metadata = {}
        self.extra_metadata = extra_metadata

        # Pre-define some values
        self._energy = np.nan
        self._wavelength = np.nan
        self._theta = np.nan
        self._use_stage_rotation = False # overwrite later

        if not save_hdf:
            if dask_enabled:
                err_str = ('Enabling dask requires an '
                           + 'hdf file for storage.')
                raise ValueError(err_str)
            else:
                self.hdf_path = None
                self.hdf = None
        else:
            self.start_saving_hdf(hdf=hdf,
                                  hdf_filename=hdf_filename,
                                  dask_enabled=dask_enabled)
        
        XRDData.__init__(
            self,
            hdf_path=self.hdf_path,
            hdf=self.hdf,
            dask_enabled=dask_enabled,
            hdf_type=self._hdf_type, # Gets redefined as same value...
            **xrddatakwargs
        )    
            
        # Store energy, dwell, and theta
        if energy is not None:
            self.energy = energy
        elif wavelength is not None: # Favors energy definition
            self.wavelength = wavelength
        else:
            print('WARNING: No energy or wavelength provided.')
            self.energy = np.nan
        
        self.dwell = dwell
        if theta is None:
            print('WARNING: No theta provided. Assuming 0 degrees.')
            theta = 0
        self.theta = theta
        self.use_stage_rotation = bool(use_stage_rotation)
        
        self.phases = {} # Place holder for potential phases
        if poni_file is not None:
            self.set_calibration(poni_file)
        else:
            self.ai = None # Place holder for calibration

        self.sclr_dict = None
        if sclr_dict is not None:
            self.set_scalers(sclr_dict)
        
        # Default units and flags
        # Not fully implemented
        # 'rad' or 'deg'
        self._polar_units = 'deg' 
        # 'rad', 'deg', 'nm^-1', 'A^-1'
        self._scattering_units = 'deg'
        # 'linear' or 'log'
        self._image_scale = 'linear' 

        if tth is not None and len(tth) == 0:
            tth = None
        self.tth = tth
        if tth_resolution is None:
            tth_resolution = 0.01 # in degrees...
        self.tth_resolution = tth_resolution

        if chi is not None and len(chi) == 0:
            chi = None
        self.chi = chi
        if chi_resolution is None:
            chi_resolution = 0.05 # in degrees...
        self.chi_resolution = chi_resolution
        
        # Catch-all of extra attributes.
        # Gets them into __init__ sooner
        if extra_attrs is not None:
            for key, value in extra_attrs.items():
                setattr(self, key, value)

    
    # Overwrite parent functions
    def __str__(self):
        ostr = (f'{self._hdf_type}:  scan_id={self.scan_id}, '
                + f'energy={self.energy}, '
                + f'shape={self.images.shape}')
        return ostr


    # Overwrite parent function
    def __repr__(self):
        # Native info
        ostr = f'{self._hdf_type}:'
        ostr += f'\n\tFacility:\t{self.facility}'
        ostr += f'\n\tBeamline:\t{self.beamline}'
        if self.scan_id is not None:
            ostr += f'\n\tscan_id:\t\t{self.scan_id}'
        ostr += f'\n\tEnergy:\t\t{self.energy} keV'
        if self.hdf_path is not None:
            ostr += f'\n\tHDF Path:\t{self.hdf_path}\n'

        # Data info
        ostr += ('\t'
                 + '\t'.join(XRDData.__repr__(self).splitlines(True)))

        # Other info
        if hasattr(self, 'ai'): # pull geometry info
            ostr += '\n\tGeometry:  \n'
            ostr += ('\t\t'
                    + '\t\t'.join(self.ai.__repr__().splitlines(True)))
        if len(self.phases) > 0: # pull phase info
            ostr += '\n\tPhases:'
            for key in self.phases.keys():
                ostr += '\n\t\t' + self.phases[key].__repr__()
        if hasattr(self, 'spots'): # pull limited spot info
            ostr += '\n\tSpots:'
            ostr += '\n\t\tNumber:  ' + str(len(self.spots))
            if hasattr(self, 'spot_model'):
                ostr += ('\n\t\tModel Fitting:  '
                         + self.spot_model.name
                         + ' Spot Model')
        return ostr

    #####################################
    ### Loading data into XRDBaseScan ###
    #####################################

    @classmethod
    def from_hdf(cls,
                 hdf_filename,
                 wd=None,
                 dask_enabled=False,
                 image_data_key='recent',
                 integration_data_key='recent',
                 map_shape=None,
                 image_shape=None,
                 **kwargs):
        
        if wd is None:
            wd = os.getcwd()
        
        # Load from previously saved data, including all processed data...
        hdf_path = pathify(wd, hdf_filename, '.h5')
        if os.path.exists(hdf_path):
            print('Loading data from hdf file...')
            input_dict = load_xrdbase_hdf(
                            hdf_filename,
                            cls._hdf_type,
                            wd,
                            image_data_key=image_data_key,
                            integration_data_key=integration_data_key,
                            map_shape=map_shape,
                            image_shape=image_shape,
                            dask_enabled=dask_enabled)

            # Remove several pieces to allow for unpacking
            base_md = input_dict.pop('base_md')
            image_attrs = input_dict.pop('image_attrs')
            image_corrections = input_dict.pop('image_corrections')
            image_data_key = input_dict.pop('image_data_key')
            integration_attrs = input_dict.pop('integration_attrs')
            integration_corrections = input_dict.pop(
                                        'integration_corrections')
            integration_data_key = input_dict.pop(
                                        'integration_data_key')
            recip_pos = input_dict.pop('recip_pos')
            phases = input_dict.pop('phases')
            spots = input_dict.pop('spots')
            spot_model = input_dict.pop('spot_model')
            spots_3D = input_dict.pop('spots_3D')
            vect_dict = input_dict.pop('vect_dict')

            # Scrub data keys. For backward compatibility
            if image_data_key is not None:
                image_title = '_'.join([x for x in 
                                        image_data_key.split('_')
                                        if x not in ['images',
                                                     'integrations']])
            else:
                image_title = None
            if integration_data_key is not None:
                integration_title = '_'.join([x for x in
                                        integration_data_key.split('_')
                                        if x not in ['images',
                                                     'integrations']])
            else:
                integration_title = None

            # Compare image and integration data
            title = image_title
            corrections = image_corrections
            if (input_dict['image_data'] is not None
                and input_dict['integration_data'] is not None):

                if image_title != integration_title:
                    warn_str = (f'WARNING: Image data from '
                                + f'({image_data_key}) does '
                                + 'not match integration data '
                                + f'from ({integration_data_key}).')
                    print(warn_str)
                
                # Check corrections if the titles match
                elif not np.all([val1 == val2
                            for val1, val2 in zip(
                                image_corrections.values(),
                                integration_corrections.values())]):
                    warn_str = ('WARNING: Different corrections '
                                + 'applied to images and integrations.'
                                + ' Using image corrections.')
                    print(warn_str)
            elif (input_dict['image_data'] is None
                  and input_dict['integration_data'] is not None):
                title = integration_title # truncate _integrations
                corrections = integration_corrections        
            
            # Remove unused values
            # (keeps pos_dict out of rocking curve...)
            for key, value in list(input_dict.items()):
                if value is None:
                    del input_dict[key]

            # Set up extra attributes
            extra_attrs = {}
            extra_attrs.update(image_attrs)
            extra_attrs.update(integration_attrs)
            # Add phases
            extra_attrs['phases'] = phases
            # Add spots
            if spots is not None:
                extra_attrs['spots'] = spots
            if spot_model is not None:
                extra_attrs['spot_model'] = spot_model
            if spots_3D is not None:
                extra_attrs['spots_3D'] = spots_3D

             # Add vector information. Not super elegant
            if vect_dict is not None:
                for key, value in vect_dict.items():
                    # Check for int cutoffs
                    # Wait to make sure intensity is processed
                    if key == 'blob_int_cutoff':
                        (extra_attrs['blob_int_mask']
                        ) = generate_intensity_mask(
                            vect_dict['intensity'],
                            intensity_cutoff=vect_dict['blob_int_cutoff'])
                    elif key == 'spot_int_cutoff':
                        (extra_attrs['spot_int_mask']
                        ) = generate_intensity_mask(
                            vect_dict['intensity'],
                            intensity_cutoff=vect_dict['spot_int_cutoff'])
                    else:
                        extra_attrs[key] = value

            # Instantiate XRDBaseScan
            inst = cls(**input_dict,
                       **base_md,
                       **recip_pos,
                    #    map_shape=map_shape, # Now in input_dict
                    #    image_shape=image_shape, # Now in input_dict
                       title=title,
                       corrections=corrections,
                       wd=wd,
                       filename=hdf_filename[:-3], # remove the .h5 extention
                       hdf_filename=hdf_filename,
                       dask_enabled=dask_enabled,
                       extra_attrs=extra_attrs,
                       **kwargs)
            
            print(f'{cls.__name__} loaded!')
            return inst
        
        else:
            # Should be redundant...
            raise FileNotFoundError(f'No hdf file at {hdf_path}.')
        

    @classmethod 
    def from_image_stack(cls,
                         filename,
                         wd=None,
                         title='raw_images',
                         dask_enabled=False,
                         **kwargs):
        
        # Load from image stack
        if wd is None:
            wd = os.getcwd()
        
        image_path = pathify(wd,
                             filename,
                             ['.tif', '.tiff', '.jpeg', '.png'])

        print('Loading images...', end='', flush=True)
        if dask_enabled:
            image_data = dask_io.imread(image_path)
        else:
            image_data = io.imread(image_path)
        print('done!')

        return cls(image_data=image_data,
                   wd=wd,
                   title=title,
                   **kwargs)
    
    
    ##################
    ### Properties ###
    ##################

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, energy):
        if (np.all(energy is not None)
            and np.all(~np.isnan(energy))):
            self._energy = energy
            self._wavelength = energy_2_wavelength(energy)
        else:
            self._energy = energy
        # Propogate changes...
        if hasattr(self, 'ai') and self.ai is not None:
            # ai energy is not used by any methods called here
            self.ai.energy = self._energy 
            if hasattr(self, '_q_arr'):
                delattr(self, '_q_arr')
        if hasattr(self, 'phases'):
            for key in self.phases.keys():
                self.phases[key].energy = self._energy
        
        # Re-write hdf values
        @XRDBaseScan.protect_hdf()
        def save_attrs(self): # Not sure if this needs self...
            attrs = self.hdf[self._hdf_type].attrs
            attrs['energy'] = self.energy
            attrs['wavelength'] = self.wavelength
        save_attrs(self)


    @property
    def wavelength(self):
            return self._wavelength

    @wavelength.setter
    def wavelength(self, wavelength):
        if (np.all(wavelength is not None)
            and np.all(~np.isnan(wavelength))):
            self._wavelength = wavelength
            self._energy = wavelength_2_energy(wavelength)
        else:
            self._wavelength = wavelength
        # Propogate changes...
        if hasattr(self, 'ai') and self.ai is not None:
            self.ai.energy = self._energy
            if hasattr(self, '_q_arr'):
                delattr(self, '_q_arr')
        if hasattr(self, 'phases'):
            for key in self.phases.keys():
                self.phases[key].energy = self._energy

        # Re-write hdf values
        @XRDData.protect_hdf()
        def save_attrs(self): # Not sure if this needs self...
            attrs = self.hdf[self._hdf_type].attrs
            attrs['energy'] = self.energy
            attrs['wavelength'] = self.wavelength
        save_attrs(self)
        
    
    # y-axis stage rotation
    @property
    def theta(self):
        return self._theta
    
    @theta.setter
    def theta(self, theta):
        self._theta = theta
        # Propagate changes
        if hasattr(self, 'ai') and self.ai is not None:
            if hasattr(self, '_q_arr'):
                delattr(self, '_q_arr')

        # Re-write hdf values
        @XRDBaseScan.protect_hdf()
        def save_attrs(self): # Not sure if this needs self...
            attrs = self.hdf[self._hdf_type].attrs
            attrs['theta'] = self.theta
        save_attrs(self)


    @property
    def use_stage_rotation(self):
        return self._use_stage_rotation

    @use_stage_rotation.setter
    def use_stage_rotation(self, use_stage_rotation):
        self._use_stage_rotation = use_stage_rotation
        # Propagate changes
        if hasattr(self, 'ai') and self.ai is not None:
            if hasattr(self, '_q_arr'):
                delattr(self, '_q_arr')

        # Re-write hdf values
        @XRDBaseScan.protect_hdf()
        def save_attrs(self): # Not sure if this needs self...
            attrs = self.hdf[self._hdf_type].attrs
            attrs['use_stage_rotation'] = int(self.use_stage_rotation)
        save_attrs(self)


    # Flags for units and scales

    def angle_units_factory(property_name, options):
        def get_angle_units(self):
            return getattr(self, f'_{property_name}')
        
        def set_angle_units(self, val):
            if val in options:
                setattr(self, f'_{property_name}', val)
                # Delete old arrays in case units changed
                self._del_arr() # Deletes everything else too...
            else:
                err_str = 'Only '
                for option in options:
                    err_str += f'{option}, '
                err_str = (err_str[:-2]
                           + f' are supported for {property_name}')
                raise ValueError(err_str)
        
        return property(get_angle_units, set_angle_units)
    

    scattering_units = angle_units_factory('scattering_units',
                                    ['rad', 'deg', '1/nm', '1/A'])
    polar_units = angle_units_factory('polar_units',
                                    ['rad', 'deg'])
    

    def scale_property_factory(property_name):
        def get_scale(self):
            return getattr(self, f'_{property_name}')

        def set_scale(self, val):
            if val in ['linear', 'log']:
                setattr(self, f'_{property_name}', val)
            else:
                raise ValueError(f"{property_name} can only have "
                                 "'linear' or 'log' scales.")
            
        return property(get_scale, set_scale)
    

    image_scale = scale_property_factory('image_scale')
    integration_scale = scale_property_factory('integration_scale')

    # Convenience properties for working with the detector arrays
    # These are mostly wrappers for pyFAI functions

    def detector_angle_array_factory(arr_name, ai_arr_name, units):
        def get_angle_array(self):
            if hasattr(self, f'_{arr_name}'):
                return getattr(self, f'_{arr_name}')
            elif (self.corrections['polar_calibration']):
                # I should rename...
                if hasattr(self, 'tth') and hasattr(self, 'chi'):
                    # Set both tth and chi!
                    tth_arr, chi_arr = np.meshgrid(self.tth,
                                                   self.chi[::-1])
                    self._tth_arr = tth_arr
                    self._chi_arr = chi_arr
                    return getattr(self, f'_{arr_name}')
            elif hasattr(self, 'ai') and self.ai is not None:
                # default is radians
                ai_arr = getattr(self.ai, ai_arr_name)()
                if arr_name == 'chi_arr':
                    # Negative to match SRX coordinates
                    ai_arr = -ai_arr 

                if getattr(self, units) == 'rad':
                    pass
                elif getattr(self, units) == 'deg':
                    ai_arr = np.degrees(ai_arr)
                elif getattr(self, units) == '1/nm':
                    err_str = '1/nm units not yet fully supported.'
                    raise NotImplementedError(err_str)
                elif getattr(self, units) == '1/A':
                    err_str = '1/A units not yet fully supported.'
                    raise NotImplementedError(err_str)
                else:
                    raise ValueError('Unknown units specified.')

                setattr(self, f'_{arr_name}', ai_arr)
                return getattr(self, f'_{arr_name}')
            
            err_str = 'AzimuthalIntegrator (ai) not specified.'
            raise AttributeError(err_str)

        def del_angle_array(self):
            delattr(self, f'_{arr_name}')

        # Delta arrays are not cached
        def get_delta_array(self):
            arr = getattr(self, arr_name) # should call the property

            max_arr = np.max(np.abs(arr))
            delta_arr = delta_array(arr)

            # Modular shift values if there is a discontinuity
            # Should only be for chi_arr between 180 and -180 degrees
            if np.max(delta_arr) > max_arr:
                # Degrees
                if max_arr > np.pi: shift_value = 2 * 180
                # Radians
                else: shift_value = 2 * np.pi
                # Shift and recalculate
                arr[arr < 0] += shift_value
                delta_arr = delta_array(arr)

            return delta_arr

        return (property(get_angle_array, None, del_angle_array),
                property(get_delta_array))
    

    tth_arr, delta_tth = detector_angle_array_factory('tth_arr',
                                           'twoThetaArray',
                                           'polar_units')
    chi_arr, delta_chi = detector_angle_array_factory('chi_arr',
                                           'chiArray',
                                           'polar_units')


    # Full q-vector, not just magnitude
    @property
    def q_arr(self):
        if hasattr(self, '_q_arr'):
            return self._q_arr
        elif not hasattr(self, 'ai'):
            err_str = 'Cannot calculate q-space without calibration.'
            raise RuntimeError(err_str)
        else:
            if self.use_stage_rotation:
                theta = self.theta
            else:
                theta = None

            q_arr = get_q_vect(self.tth_arr,
                               self.chi_arr,
                               wavelength=self.wavelength,
                               stage_rotation=theta,
                               degrees=self.polar_units == 'deg')
            self._q_arr = q_arr
            return self._q_arr

    @q_arr.deleter
    def q_arr(self):
        self._del_arr()
    
    # Convenience function
    def _del_arr(self):
        if hasattr(self, '_tth_arr'):
            delattr(self, '_tth_arr')
        if hasattr(self, '_chi_arr'):
            delattr(self, '_chi_arr')
        if hasattr(self, '_q_arr'):
            delattr(self, '_q_arr')
        if hasattr(self, 'ai'):
            self.ai._cached_array = {}

    
    ############################
    ### Dask / HDF Functions ###
    ############################

    def start_saving_hdf(self,
                 hdf=None,
                 hdf_filename=None,
                 hdf_path=None,
                 dask_enabled=False,
                 save_current=False):
        
        # Check for previous iterations
        if ((hasattr(self, 'hdf')
             and self.hdf is not None)
            or (hasattr(self, 'hdf_path')
                and self.hdf_path is not None)):
            os_str = ('WARNING: Trying to save to hdf, but a '
                      'file or location has already been specified!'
                      '\nSwitching save files or locations should '
                      'use the "switch_hdf" function.'
                      '\nProceeding without changes.')
            print(os_str)
            return

        # Specify hdf path and name
        if hdf is not None: # biases towards already open hdf
            # This might break if hdf is a close file and not None
            self.hdf_path = hdf.filename
        elif hdf_filename is None:
            if hdf_path is None:
                #self.hdf_path = f'{self.wd}{self.filename}.h5'
                self.hdf_path = pathify(self.wd,
                                        self.filename,
                                        '.h5',
                                        check_exists=False)
            else:
                #self.hdf_path = f'{hdf_path}{self.filename}.h5'
                self.hdf_path = pathify(hdf_path,
                                        self.filename,
                                        '.h5',
                                        check_exists=False)
        else:
            if hdf_path is None:
                #self.hdf_path = f'{self.wd}{hdf_filename}'
                self.hdf_path = pathify(self.wd,
                                        hdf_filename,
                                        '.h5',
                                        check_exists=False)
            else:
                #self.hdf_path = f'{hdf_path}{hdf_filename}.h5'
                self.hdf_path = pathify(hdf_path,
                                        hdf_filename,
                                        '.h5',
                                        check_exists=False)

        # Check for hdf and initialize if new            
        if not os.path.exists(self.hdf_path):
            # Initialize base structure
            initialize_xrdbase_hdf(self, self.hdf_path) 

        # Open hdf if required
        if dask_enabled:
            self.hdf = h5py.File(self.hdf_path, 'a')
        else:
            self.hdf = None

        if save_current:
            self.save_current_hdf()


    # Saves current major features
    # Calls several other save functions
    def save_current_hdf(self):
        
        if self.hdf_path is None:
            print('WARNING: Changes cannot be written to hdf without '
                  + 'first indicating a file location.\nProceeding '
                  + 'without changes.')
            return # Hard-coded even though all should pass

        if hasattr(self, 'images'):
            if self._dask_enabled:
                self.dask_2_hdf()
            else:
                self.save_images()
        # Save integrations
        if (hasattr(self, 'integrations')
            and self.integrations is not None):
            self.save_integrations()
        
        if hasattr(self, 'poni') and self.poni is not None:
            self.save_calibration()

        # Save positions
        if hasattr(self, 'pos_dict') and self.pos_dict is not None:
            # Write to hdf file
            self.save_sclr_pos('positions',
                                self.pos_dict,
                                self.position_units)

        # Save scalers
        if hasattr(self, 'sclr_dict') and self.sclr_dict is not None:
            self.save_sclr_pos('scalers',
                                self.sclr_dict,
                                self.scaler_units)
        
        # Save phases
        if hasattr(self, 'phases') and self.phases is not None:
            self.update_phases()

        # Save spots
        if hasattr(self, 'spots'):
            self.save_spots()

    
    # Ability to toggle hdf saving and proceed without writing to disk.
    def stop_saving_hdf(self):

        if self._dask_enabled:
            err_str = ('WARNING: Image data is lazy loaded. Stopping '
                       + 'or switching hdf is likely to cause '
                       + 'problems.\nSave progress and close the hdf '
                       + 'with "close_hdf" function before changing '
                       + 'save location.')
            raise RuntimeError(err_str)
        
        self.close_hdf()
        self.hdf_path = None
    

    def switch_hdf(self,
                   hdf=None,
                   hdf_path=None,
                   hdf_filename=None,
                   dask_enabled=False):

        # Check to make sure the change is appropriate and correct.
        # Not sure if this should raise and error or just print a warning
        if hdf is None and hdf_path is None:
            ostr = ('Neither hdf nor hdf_path were provided. '
                     + '\nCannot switch hdf save locations without '
                     + 'providing alternative.')
            print(ostr)
            return
        
        elif hdf == self.hdf:
            ostr = (f'WARNING: provided hdf ({self.hdf.filename}) is '
                    + 'already the current save location. '
                    + '\nProceeding without changes')
            print(ostr)
            return
        
        elif hdf_path == self.hdf_path:
            ostr = (f'WARNING: provided hdf_path ({self.hdf_path}) is '
                    + 'already the current save location. '
                    + '\nProceeding without changes')
            print(ostr)
            return
        
        else:
            # Success actually changes the write location
            # And likely initializes a new hdf
            self.stop_saving_hdf()
            self.start_saving_hdf(hdf=hdf,
                                  hdf_path=hdf_path,
                                  hdf_filename=hdf_filename,
                                  dask_enabled=dask_enabled)
            

    ##############################
    ### Calibrating Map Images ###
    ##############################
    
    def set_calibration(self,
                        poni_file,
                        energy=None,
                        wd=None):
        if wd is None:
            wd = self.wd

        if isinstance(poni_file, str):
            if not os.path.exists(f'{wd}{poni_file}'):
                err_str = f'{wd}{poni_file} does not exist.'
                raise FileNotFoundError(err_str)

            if poni_file[-4:] != 'poni':
                raise RuntimeError('Please provide a .poni file.')

            print('Setting detector calibration...')
            self.ai = pyFAI.load(f'{wd}{poni_file}')
        
        elif isinstance(poni_file, OrderedDict):
            print('Setting detector calibration...')
            self.ai = AzimuthalIntegrator().set_config(poni_file)

        elif isinstance(poni_file, ponifile.PoniFile):
            print('Setting detector calibration...')
            self.ai = AzimuthalIntegrator().set_config(
                                            poni_file.as_dict())
        
        else:
            err_str = (f'{type(poni_file)} is unknown '
                       + 'and not supported!')
            raise TypeError(err_str)

        # Update energy if different from poni file
        if energy is None:
            if self.energy is not None:
                # Allows calibrations acquired at any energy
                self.ai.energy = self.energy 
            else:
                ostr = ('Energy has not been defined. '
                        + 'Defaulting to .poni file value.')
                print()
                self.energy = self.ai.energy
        else:
            self.ai.energy = energy # Do not update energy...

        # Update detector shape and pixel
        # size if different from poni file
        try:
            image_shape = list(self.image_shape)
        except AttributeError:
            image_shape = list(self.images.shape[:-2])

        if self.ai.detector.shape != image_shape:
            ostr = ('Calibration performed under different settings. '
                    + 'Adjusting calibration.')
            print(ostr)

            # Exctract old values
            poni_shape = self.ai.detector.shape
            poni_pixel1 = self.ai.detector.pixel1
            poni_pixel2 = self.ai.detector.pixel2

            bin_est = np.array(image_shape) / np.array(poni_shape)
            # Forces whole number binning, either direction
            # This would prevent custom resizing for preprocessing
            if all([any(bin_est != np.round(bin_est, 0)),
                    any(1 / bin_est != np.round(1 / bin_est, 0))]):
                err_str = ("Calibration file was performed with an "
                            + "image that is not an integral "
                            + "multiple of the current map's images."
                            + "\n\t\tEnsure the calibration is for "
                            + "the correct detector with the "
                            + "appropriate binning.")
                raise ValueError(err_str)

            # Overwrite values
            self.ai.detector.shape = image_shape
            # Not exactly correct, but more convenient
            self.ai.detector.max_shape = image_shape 
            self.ai.detector.pixel1 = poni_pixel1 / bin_est[0]
            self.ai.detector.pixel2 = poni_pixel2 / bin_est[1]

        # Extract calibration parameters to save
        self.poni = self.ai.get_config()

        # reset any previous calibrated arrays
        self._del_arr()
        
        # Save poni files as dictionary 
        # Updates poni information to update detector settings
        self.save_calibration()
    

    @XRDData.protect_hdf()
    def save_calibration(self):

        # Write data to hdf
        curr_grp = self.hdf[self._hdf_type].require_group('reciprocal_positions')
        new_grp = curr_grp.require_group('poni_file')
        # I don't really like saving values as attributes
        # They are well setup for this type of thing, though
        for key, value in self.poni.items():
            # For detector which is a nested ordered dictionary...
            if isinstance(value, (dict, OrderedDict)):
                new_new_grp = new_grp.require_group(key)
                for key_i, value_i in value.items():
                    # Check for orientation value added in poni_file version 2.1
                    if isinstance(value_i, IntEnum):
                        value_i = value_i.value
                    new_new_grp.attrs[key_i] = value_i
            else:
                new_grp.attrs[key] = value


    # One off 1D integration
    def integrate1d_image(self,
                          image=None,
                          tth_resolution=None,
                          tth_num=None,
                          unit='2th_deg',
                          **kwargs):

        if image is None:
            if self.corrections['polar_calibration']:
                err_str = ('You are trying to calibrate '
                           + 'already calibrated images!')
                raise RuntimeError(err_str)
            else:
                image = self.composite_image
        
        if tth_resolution is None:
            tth_resolution = self.tth_resolution

        tth_min = np.min(self.tth_arr)
        tth_max = np.max(self.tth_arr)
        if tth_num is None:
            tth_num = int(np.round((tth_max - tth_min)
                                   / tth_resolution))
        elif tth_num is None and tth_resolution is None:
            err_str = 'Must define either tth_num or tth_resolution.'
            raise ValueError(err_str)
        
        return self.ai.integrate1d_ng(image,
                                      tth_num,
                                      unit=unit,
                                      correctSolidAngle=False,
                                      polarization_factor=None,
                                      **kwargs)
    

    # One off 2D integration
    def integrate2d_image(self,
                          image,
                          tth_num=None,
                          tth_resolution=None,
                          chi_num=None,
                          chi_resolution=None,
                          unit='2th_deg',
                          **kwargs):

        if image is None:
            if self.corrections['polar_calibration']:
                err_str = ('You are trying to clibrate '
                           + 'already calibrated images!')
                raise RuntimeError(err_str)
            else:
                image = self.composite_image

        if tth_resolution is None:
            tth_resolution = self.tth_resolution
        if chi_resolution is None:
            chi_resolution = self.chi_resolution

        # Get tth numbers
        tth_min = np.min(self.tth_arr)
        tth_max = np.max(self.tth_arr)
        if tth_num is None:
            tth_num = int(np.round((tth_max - tth_min)
                                   / tth_resolution))
        elif tth_num is not None:
            tth_resolution = (tth_max - tth_min) / tth_num
        elif tth_num is None and tth_resolution is None:
            err_str = 'Must define either tth_num or tth_resolution.'
            raise ValueError(err_str)
        
        # Get chi numbers
        chi_min = np.min(self.chi_arr)
        chi_max = np.max(self.chi_arr)
        if chi_num is None:
            chi_num = int(np.round((chi_max - chi_min)
                                   / chi_resolution))
        elif chi_num is not None:
            chi_resolution = (chi_max - chi_min) / chi_num
        elif chi_num is None and chi_resolution is None:
            err_str = 'Must define either chi_num or chi_resolution.'
            raise ValueError(err_str)
       
        return self.ai.integrate2d_ng(image, tth_num, chi_num,
                                      unit=unit,
                                      correctSolidAngle=False,
                                      polarization_factor=None,
                                      **kwargs)


    # Convenience function for image to polar coordinate transformation (estimate!)
    def estimate_polar_coords(self,
                              coords,
                              method='linear'):
        return estimate_polar_coords(coords,
                                     self.tth_arr,
                                     self.chi_arr,
                                     method=method)
    

    # Convenience function for polar to image coordinate transformation (estimate!)
    def estimate_image_coords(self,
                              coords,
                              method='nearest'):
        return estimate_image_coords(coords,
                                     self.tth_arr,
                                     self.chi_arr,
                                     method=method)

    
    @XRDData.protect_hdf()
    def save_reciprocal_positions(self):
                 
        if self.tth is None:
            tth = []
        else:
            tth = self.tth
        if self.chi is None:
            chi = []
        else:
            chi = self.chi

        print('Writing reciprocal positions to disk...', end='', flush=True)
        # This group may already exist if poni file was already initialized
        curr_grp = self.hdf[self._hdf_type].require_group('reciprocal_positions')
        if hasattr(self, 'extent'):
            curr_grp.attrs['extent'] = self.extent

        labels = ['tth_pos', 'chi_pos']
        comments = ["'tth', is the two theta scattering angle",
                    "'chi' is the azimuthal angle"]
        keys = ['tth', 'chi']
        data = [tth, chi]
        resolution = [self.tth_resolution, self.chi_resolution]

        for i, key in enumerate(keys):
            # Skip values that are None
            if data[i] is None:
                data[i] = np.array([])
            else:
                data[i] = np.asarray(data[i])

            if key in curr_grp.keys():
                del curr_grp[key]
            dset = curr_grp.require_dataset(key,
                                            data=data[i],
                                            dtype=data[i].dtype,
                                            shape=data[i].shape)
            dset.attrs['labels'] = labels[i]
            dset.attrs['comments'] = comments[i]
            #dset.attrs['units'] = self.calib_unit #'° [deg.]'
            dset.attrs['dtype'] = str(data[i].dtype)
            dset.attrs['time_stamp'] = ttime.ctime()
            dset.attrs[f'{key}_resolution'] = resolution[i]

        print('done!')
    
    ##################################
    ### Scaler and Position Arrays ###
    ##################################

    def set_scalers(self,
                    sclr_dict,
                    scaler_units='counts'):

        # Store sclr_dict as attribute
        for key, value in list(sclr_dict.items()):
            if value.ndim  != 2: # Only intended for rocking curves
                sclr_dict[key] = value.reshape(self.map_shape)

        self.sclr_dict = sclr_dict
        self.scaler_units = scaler_units

        # Write to hdf file
        self.save_sclr_pos('scalers',
                           self.sclr_dict,
                           self.scaler_units)

    @XRDData.protect_hdf()
    def save_sclr_pos(self,
                      group_name,
                      map_dict,
                      unit_name):

        # Write data to hdf
        curr_grp = self.hdf[self._hdf_type].require_group(group_name)
        curr_grp.attrs['time_stamp'] = ttime.ctime()

        for key, value in map_dict.items():
            value = np.asarray(value)
            
            if key in curr_grp.keys():
                dset = curr_grp[key]
                if (value.shape == dset.shape
                    and value.dtype == dset.dtype):
                    dset[...] = value
                else:
                    del curr_grp[key] # deletes flag, but not the data...
                    dset = curr_grp.require_dataset(
                                                key,
                                                data=value,
                                                shape=value.shape,
                                                dtype=value.dtype
                                                )
            else:
                dset = curr_grp.require_dataset(
                                    key,
                                    data=value,
                                    shape=value.shape,
                                    dtype=value.dtype
                                    )
            
            # Update attrs everytime
            dset.attrs['labels'] = ['map_x', 'map_y']
            dset.attrs['units'] = unit_name
            dset.attrs['dtype'] = str(value.dtype)
        
    
    #########################################
    ### Manipulating and Selecting Phases ###
    #########################################

    # Updating potential phase list
    def add_phase(self, phase):
        if hasattr(phase, 'name'):
            phase_name = phase.name
        elif isinstance(phase, str):
            phase_name = phase
        else:
            raise TypeError(f'Unsure how to handle {phase} type.')

        if phase_name not in self.phases.keys():
            self.phases[phase_name] = phase
        else:
            ostr = (f'Did not add {phase_name} since it is '
                    + 'already a possible phase.')
            print()


    def remove_phase(self, phase):
        # Allow for phase object or name to work
        if hasattr(phase, 'name'):
            phase_name = phase.name
        elif isinstance(phase, str):
            phase_name = phase
        else:
            raise TypeError(f'Unsure how to handle {phase} type.')

        if phase_name in self.phases.keys():
            del self.phases[phase_name]
        else:
            ostr = (f'Cannot remove {phase_name} since it is '
                    + 'not in possible phases.')
            print(ostr)
        

    def load_phase(self, filename, wd=None, phase_name=None):
        if wd is None:
            wd = self.wd
        
        phase_path = pathify(wd, filename,
                             ['.cif', '.txt', '.D', '.h5'])

        if not os.path.exists(f'{phase_path}'):
            err_str = f'Specified path does not exist:\n{phase_path}'
            raise FileNotFoundError(err_str)
        
        if phase_path[-4:] == '.cif':
            phase = Phase.fromCIF(f'{phase_path}')
        elif phase_path[-4:] == '.txt':
            raise NotImplementedError()
        elif phase_path[-2:] in ['.D']:
            raise NotImplementedError()
        elif phase_path[-2:] == '.h5':
            raise NotImplementedError()
        else:
            err_str = (f'Unsure how to read {phase_path}. Either '
                       + 'specifiy file type or this file type '
                       + 'is not supported.')
            raise TypeError(err_str)
        
        if phase_name is not None:
            phase.name = phase_name
        
        self.add_phase(phase)
    

    def clear_phases(self):
        self.phases = {}

    @XRDData.protect_hdf()
    def update_phases(self):
        
        if len(self.phases) > 0:
            phase_grp = self.hdf[self._hdf_type].require_group(
                                                    'phase_list')

            # Delete any no longer included phases
            for phase in phase_grp.keys():
                if phase not in self.phases.keys():
                    del(phase_grp[phase])

            # Save any new phases
            for phase in self.phases.values():
                phase.save_to_hdf(phase_grp)

            print('Updated phases saved in hdf.')


    def select_phases(self,
                      remove_less_than=-1,
                      image=None,
                      energy=None,
                      tth_num=4096,
                      unit='2th_deg',
                      ignore_less=1,
                      save_to_hdf=True):
        
        if not hasattr(self, 'ai') or self.ai is None:
            err_str = ('Must first set calibration '
                       + 'before selecting phases.')
            raise AttributeError(err_str)
        
        if image is None:
            if self.corrections['polar_calibration']:
                image = self._processed_images_composite
            else:
                image = self.composite_image
        
        if energy is None:
            if isinstance(self.energy, list):
                err_str = ('A specific incident X-ray energy must be '
                        + 'provided to use the phase selector tool.')
                raise RuntimeError(err_str)
            else:
                energy = self.energy
        
        tth, xrd = self.integrate1d_image(image=image,
                                          tth_num=tth_num,
                                          unit=unit)

        # Plot phase_selector
        phase_vals = phase_selector(xrd,
                                    list(self.phases.values()),
                                    tth,
                                    ignore_less=ignore_less)

        old_phases = list(self.phases.keys())
        for phase in old_phases:
            if phase_vals[phase] <= remove_less_than:
                self.remove_phase(phase)
        
        # Write phases to disk
        if save_to_hdf:
            self.update_phases()

    
    # This might be replaced with generate reciprocal lattice
    def _get_all_reflections(self, ignore_less=1):
        for phase in self.phases:
            self.phases[phase].get_hkl_reflections(
                tth_range=(0, # Limited to zero for large d-spacing
                              # Used for indexing later
                           np.max(self.tth)),
                ignore_less=ignore_less)
         

    ##########################
    ### Plotting Functions ###
    ##########################

    def plot_image(self,
                   image=None,
                   indices=None,
                   title=None,
                   mask=False,
                   spots=False,
                   contours=False,
                   fig=None,
                   ax=None,
                   aspect='auto',
                   return_plot=False,
                   **kwargs):
        
        image, indices = _xrdmap_image(self,
                                       image=image,
                                       indices=indices)
        
        out = _plot_parse_xrdmap(self,
                                 indices,
                                 mask=mask,
                                 spots=spots,
                                 contours=contours)

        fig, ax = plot_image(image,
                             indices=indices,
                             title=title,
                             mask=out[0],
                             spots=out[1],
                             contours=out[2],
                             fig=fig,
                             ax=ax,
                             aspect=aspect,
                             **kwargs)
        
        if return_plot:
            return fig, ax
        else:
            fig.show()


    def plot_integration(self,
                         integration=None,
                         indices=None,
                         tth=None,
                         units=None,
                         title=None,
                         fig=None,
                         ax=None,
                         y_min=None,
                         y_max=None,
                         return_plot=False,
                         **kwargs):
        
        if tth is None:
            tth = self.tth
        
        if units is None:
            units = self.scattering_units
        
        integration, indices = _xrdmap_integration(self,
                                       integration=integration,
                                       indices=indices)

        fig, ax = plot_integration(integration,
                                   indices=indices,
                                   tth=tth,
                                   units=units,
                                   title=title,
                                   fig=fig,
                                   ax=ax,
                                   y_min=y_min,
                                   y_max=y_max,
                                   **kwargs)
        
        if return_plot:
            return fig, ax
        else:
            fig.show()
    



    ##################################
    ### Plot Experimental Geometry ###
    ##################################

    def plot_q_space(self,
                     indices=None,
                     skip=500,
                     detector=True,
                     Ewald_sphere=True,
                     beam_path=True,
                     fig=None,
                     ax=None,
                     return_plot=False):
 
        fig, ax = plot_q_space(self,
                               indices=indices,
                               skip=skip,
                               detector=detector,
                               Ewald_sphere=Ewald_sphere,
                               beam_path=beam_path,
                               fig=fig,
                               ax=ax)
        
        if return_plot:
            return fig, ax
        else:
            fig.show()


    def plot_detector_geometry(self,
                               skip=300,
                               fig=None,
                               ax=None,
                               return_plot=False):
        
        fig, ax = plot_detector_geometry(self,
                                         skip=skip,
                                         fig=fig,
                                         ax=ax)
        
        if return_plot:
            return fig, ax
        else:
            fig.show()
