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
    _check_dict_key
)
from xrdmaptools.io.hdf_io_rev import (
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
from xrdmaptools.plot.geometry import plot_q_space, plot_detector_geometry
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
                 scanid=None,
                 wd=None,
                 filename=None,
                 hdf_filename=None,
                 hdf=None,
                 image_data=None,
                 integration_data=None,
                 map_shape=None,
                 image_shape=None,
                 title=None,
                 map_labels=None,
                 null_map=None,
                 chunks=None,
                 energy=None,
                 wavelength=None,
                 dwell=None,
                 theta=None,
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
                 ):
        
        # Adding some metadata
        self.scanid = scanid
        if filename is None:
            filename = f'scan{scanid}_{self._hdf_type}' # This may break some things...
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

        # Add conditional to check approximate image_data size and force dask?

        if not save_hdf:
            if dask_enabled:
                raise ValueError('Enabling dask requires an hdf file for storage.')
            else:
                self.hdf_path = None
                self.hdf = None
        else:
            self.start_saving_hdf(hdf=hdf,
                                  hdf_filename=hdf_filename,
                                  dask_enabled=dask_enabled)
        
        XRDData.__init__(
            self,
            image_data=image_data,
            integration_data=integration_data,
            map_shape=map_shape,
            image_shape=image_shape,
            title=title,
            map_labels=map_labels,
            hdf_path=self.hdf_path,
            hdf=self.hdf,
            chunks=chunks,
            null_map=null_map,
            dask_enabled=dask_enabled,
            hdf_type=self._hdf_type # Gets redefined as same value...
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
        self._polar_units = 'deg' # 'rad' or 'deg'
        self._scattering_units = 'deg' # 'rad', 'deg', 'nm^-1', 'A^-1'
        self._image_scale = 'linear' # 'linear' or 'log'

        if tth is not None and len(tth) == 0:
            tth = None
        self.tth = tth
        if tth_resolution is None:
            tth_resolution = 0.01
        self.tth_resolution = tth_resolution

        if chi is not None and len(chi) == 0:
            chi = None
        self.chi = chi
        if chi_resolution is None:
            chi_resolution = 0.05
        self.chi_resolution = chi_resolution

    
    # Overwrite parent functions
    def __str__(self):
        ostr = (f'{self._hdf_type}:  scanid={self.scanid}, '
                + f'energy={self.energy}, '
                + f'shape={self.images.shape}')
        return ostr


    # Overwrite parent function
    def __repr__(self):
        # Native info
        ostr = f'{self._hdf_type}:'
        ostr += f'\n\tFacility:\t{self.facility}'
        ostr += f'\n\tBeamline:\t{self.beamline}'
        if self.scanid is not None:
            ostr += f'\n\tScanid:\t\t{self.scanid}'
        ostr += f'\n\tEnergy:\t\t{self.energy} keV'
        if self.hdf_path is not None:
            ostr += f'\n\tHDF Path:\t{self.hdf_path}\n'

        # Data info
        ostr += '\t' + '\t'.join(XRDData.__repr__(self).splitlines(True))

        # Other info
        if hasattr(self, 'ai'): # pull geometry info
            ostr += '\n\tGeometry:  \n'
            ostr += '\t\t' + '\t\t'.join(self.ai.__repr__().splitlines(True))
        if len(self.phases) > 0: # pull phase info
            ostr += '\n\tPhases:'
            for key in self.phases.keys():
                ostr += '\n\t\t' + self.phases[key].__repr__()
        if hasattr(self, 'spots'): # pull limited spot info
            ostr += '\n\tSpots:'
            ostr += '\n\t\tNumber:  ' + str(len(self.spots))
            if hasattr(self, 'spot_model'):
                ostr += '\n\t\tModel Fitting:  ' + self.spot_model.name + ' Spot Model'
        return ostr

    #####################################
    ### Loading data into XRDBaseScan ###
    ################################

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
            input_dict = load_xrdbase_hdf(hdf_filename,
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
            integration_attrs = input_dict.pop('integration_attrs')
            integration_corrections = input_dict.pop('integration_corrections')
            recip_pos = input_dict.pop('recip_pos')
            phases = input_dict.pop('phases')
            spots = input_dict.pop('spots')
            spot_model = input_dict.pop('spot_model')
            spots_3D = input_dict.pop('spots_3D')
            vect_dict = input_dict.pop('vect_dict')

            # Compare image and integration data
            if (input_dict['image_data'] is not None
                and input_dict['integration_data'] is not None):
                if not np.all([val1 == val2
                               for val1, val2 in zip(image_corrections.values(),
                                                     integration_corrections.values())]):
                    warn_str = ('WARNING: Different corrections applied to images and integrations.'
                                + ' Using image corrections.')
                    print(warn_str)
            
            # Remove unused values (keeps pos_dict out of rocking curve...)
            for key, value in list(input_dict.items()):
                if value is None:
                    del input_dict[key]      

            # Instantiate XRDBaseScan
            inst = cls(**input_dict,
                       **base_md,
                       **recip_pos,
                       map_shape=map_shape,
                       image_shape=image_shape,
                       wd=wd,
                       filename=hdf_filename[:-3], # remove the .h5 extention
                       hdf_filename=hdf_filename,
                       dask_enabled=dask_enabled,
                       **kwargs)
            
            # Add corrections. Bias towards images
            if image_corrections is not None:
                inst.corrections = image_corrections
            
            # Add extra attributes
            for key, value in image_attrs.items():
                setattr(inst, key, value)
            for key, value in integration_attrs.items():
                setattr(inst, key, value)
            
            # Add phases
            inst.phases = phases
            # Add spots
            if spots is not None:
                inst.spots = spots
            if spot_model is not None:
                inst.spot_model = spot_model
            if spots_3D is not None:
                inst.spots_3D = spots_3D
            
            # Add vector information. Not super elegant
            if vect_dict is not None:
                has_blob_int, has_spot_int = False, False
                for key, value in vect_dict.items():
                    # Check for int cutoffs
                    # Wait to make sure intensity is processed
                    if key == 'blob_int_cutoff':
                        has_blob_int = True
                    elif key == 'spot_int_cutoff':
                        has_spot_int = True
                    else:
                        setattr(inst, key, value)
                if has_blob_int:
                    inst.blob_int_mask = inst.get_vector_int_mask(
                            intensity_cutoff=vect_dict['blob_int_cutoff'])
                if has_spot_int:
                    inst.spot_int_mask = inst.get_vector_int_mask(
                            intensity_cutoff=vect_dict['spot_int_cutoff'])
            
            print(f'{cls.__name__} loaded!')
            return inst
        
        else:
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
        
        image_path = pathify(wd, filename, ['.tif', '.tiff', '.jpeg', '.png'])

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
        # if (hasattr(self, '_energy')
        #     and np.all(self._energy is not None)
        #     and np.all(~np.isnan(self._energy))):
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
            self.ai.energy = self._energy # ai energy is not used by any methods called here
            if hasattr(self, '_q_arr'):
                delattr(self, '_q_arr')
        if hasattr(self, 'phases'):
            for key in self.phases.keys():
                self.phases[key].energy = self._energy
        
        # Re-write hdf values
        if hasattr(self, 'hdf_path') and self.hdf_path is not None:
            if self._dask_enabled:
                self.hdf[self._hdf_type].attrs['energy'] = self.energy
                self.hdf[self._hdf_type].attrs['wavelength'] = self.wavelength
            else:
                with h5py.File(self.hdf_path, 'a') as f:
                    f[self._hdf_type].attrs['energy'] = self.energy
                    f[self._hdf_type].attrs['wavelength'] = self.wavelength


    @property
    def wavelength(self):
        # if (hasattr(self, '_wavelength')
        #     and np.all(self._wavelength is not None)
        #     and np.all(~np.isnan(self._wavelength))):
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
        if hasattr(self, 'hdf_path') and self.hdf_path is not None:
            if self._dask_enabled:
                self.hdf[self._hdf_type].attrs['energy'] = self.energy
                self.hdf[self._hdf_type].attrs['wavelength'] = self.wavelength
            else:
                with h5py.File(self.hdf_path, 'a') as f:
                    f[self._hdf_type].attrs['energy'] = self.energy
                    f[self._hdf_type].attrs['wavelength'] = self.wavelength
    

    @property
    def theta(self):
        # if (hasattr(self, '_theta')
        #     and np.all(self._theta is not None)
        #     and np.all(~np.isnan(self._theta))):
        return self._theta
    
    @theta.setter
    def theta(self, theta):
        self._theta = theta
        # Propagate changes
        if hasattr(self, 'ai') and self.ai is not None:
            if hasattr(self, '_q_arr'):
                delattr(self, '_q_arr')

        # Re-write hdf values
        if hasattr(self, 'hdf_path') and self.hdf_path is not None:
            if self._dask_enabled:
                self.hdf[self._hdf_type].attrs['theta'] = self.theta
            else:
                with h5py.File(self.hdf_path, 'a') as f:
                    f[self._hdf_type].attrs['theta'] = self.theta


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
                err_str = err_str[:-2] + f' are supported for {property_name}' 
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
            elif (self.corrections['polar_calibration']): # I should rename...
                if hasattr(self, 'tth') and hasattr(self, 'chi'):
                    # Set both tth and chi!
                    tth_arr, chi_arr = np.meshgrid(self.tth, self.chi[::-1])
                    self._tth_arr = tth_arr
                    self._chi_arr = chi_arr
                    return getattr(self, f'_{arr_name}')
            elif hasattr(self, 'ai') and self.ai is not None:
                ai_arr = getattr(self.ai, ai_arr_name)() # default is radians
                if arr_name == 'chi_arr':
                    ai_arr = -ai_arr # Negative to match SRX coordinates

                if getattr(self, units) == 'rad':
                    pass
                elif getattr(self, units) == 'deg':
                    ai_arr = np.degrees(ai_arr)
                elif getattr(self, units) == '1/nm':
                    raise NotImplementedError('1/nm units not yet fully supported.')
                elif getattr(self, units) == '1/A':
                    raise NotImplementedError('1/A units not yet fully supported.')
                else:
                    raise ValueError('Unknown units specified.')

                setattr(self, f'_{arr_name}', ai_arr)
                return getattr(self, f'_{arr_name}')
            raise AttributeError('AzimuthalIntegrator (ai) not specified.')

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
            raise RuntimeError('Cannot calculate q-space without calibration.')
        else:
            q_arr = get_q_vect(self.tth_arr,
                               self.chi_arr,
                               wavelength=self.wavelength,
                               degrees=self.polar_units == 'deg')
            self._q_arr = q_arr
            return self._q_arr

    @q_arr.deleter
    def q_arr(self):
        self._del_arr()
    
    
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
        if ((hasattr(self, 'hdf') and self.hdf is not None)
            or (hasattr(self, 'hdf_path') and self.hdf_path is not None)):
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
                self.hdf_path = f'{self.wd}{self.filename}.h5'
            else:
                self.hdf_path = f'{hdf_path}{self.filename}.h5'
        else:
            if hdf_path is None:
                self.hdf_path = f'{self.wd}{hdf_filename}' # TODO: Add check for .h5
            else:
                self.hdf_path = f'{hdf_path}{hdf_filename}.h5'

        # Check for hdf and initialize if new            
        if not os.path.exists(self.hdf_path):
            initialize_xrdbase_hdf(self, self.hdf_path) # Initialize base structure

        # Open hdf if required
        if dask_enabled:
            self.hdf = h5py.File(self.hdf_path, 'a')
        else:
            self.hdf = None

        if save_current:
            self.save_current_hdf()


    # Saves current major features
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
            err_str = ('WARNING: Image data is lazy loaded. '
                       'Stopping or switching hdf is likely to cause problems.'
                       '\nSave progress and close the hdf with "close_hdf" '
                       'function before changing save location.')
            raise RuntimeError(err_str)
        
        self.close_hdf()
        self.hdf_path = None
    

    def switch_hdf(self, hdf=None, hdf_path=None, hdf_filename=None, dask_enabled=False):

        # Check to make sure the change is appropriate and correct.
        # Not sure if this should raise and error or just print a warning
        if hdf is None and hdf_path is None:
            os_str = ('Neither hdf nor hdf_path were provided. '
                       'Cannot switch hdf save locations without providing alternative.')
            print(os_str)
            return
        
        elif hdf == self.hdf:
            os_str = (f'WARNING: provided hdf ({self.hdf.filename}) is already the current save location. '
                      'Proceeding without changes')
            print(os_str)
            return
        
        elif hdf_path == self.hdf_path:
            os_str = (f'WARNING: provided hdf_path ({self.hdf_path}) is already the current save location. '
                      'Proceeding without changes')
            print(os_str)
            return
        
        else:
            # Success actually changes the write location
            # And likely initializes a new hdf
            self.stop_saving_hdf()
            self.start_saving_hdf(hdf=hdf,
                                  hdf_path=hdf_path,
                                  hdf_filename=hdf_filename,
                                  dask_enabled=dask_enabled)
    
    ### Helper functions for lazy loading with Dask ###

    # This function does NOT stop saving to hdf
    # It only closes open hdf locations and stops lazy loading images
    def close_hdf(self):
        if self.hdf is not None:
            self._dask_2_hdf()
            self.hdf.close()
            self.hdf = None
        

    def open_hdf(self, dask_enabled=False):
        if self.hdf is not None:
            # Should this raise errors or just ping warnings
            print('WARNING: hdf is already open. Proceeding without changes.')
            return
        else:
            self.hdf = h5py.File(self.hdf_path, 'a')

        if dask_enabled or self._dask_enabled: # This flag persists even when the dataset is closed!
            img_grp = self.hdf[f'{self._hdf_type}/image_data']
            if self.title == 'final_images':
                if check_hdf_current_images(self.title, hdf=self.hdf):
                    dset = img_grp[self.title]
            elif check_hdf_current_images('_temp_images', hdf=self.hdf):
                dset = img_grp['_temp_images']
            self.images = da.asarray(dset) # I had .persist(), but it broke things...
            self._hdf_store = dset
            

    ##############################
    ### Calibrating Map Images ###
    ##############################
    
    def set_calibration(self,
                        poni_file,
                        energy=None,
                        filedir=None):
        if filedir is None:
            filedir = self.wd

        if isinstance(poni_file, str):
            if not os.path.exists(f'{filedir}{poni_file}'):
                raise FileNotFoundError(f"{filedir}{poni_file} does not exist")

            if poni_file[-4:] != 'poni':
                raise RuntimeError("Please provide a .poni file.")

            print('Setting detector calibration...')
            self.ai = pyFAI.load(f'{filedir}{poni_file}')
        
        elif isinstance(poni_file, OrderedDict):
            print('Setting detector calibration...')
            self.ai = AzimuthalIntegrator().set_config(poni_file)

        elif isinstance(poni_file, ponifile.PoniFile):
            print('Setting detector calibration...')
            self.ai = AzimuthalIntegrator().set_config(poni_file.as_dict())
        
        else:
            raise TypeError(f"{type(poni_file)} is unknown and not supported!")

        # Update energy if different from poni file
        if energy is None:
            if self.energy is not None:
                self.ai.energy = self.energy # Allows calibrations acquired at any energy
            else:
                print('Energy has not been defined. Defaulting to .poni file value.')
                self.energy = self.ai.energy
        else:
            self.ai.energy = energy # Do not update energy...

        # Update detector shape and pixel size if different from poni file
        try:
            image_shape = list(self.image_shape)
        except AttributeError:
            image_shape = list(self.images.shape[:-2])

        if self.ai.detector.shape != image_shape:
            print('Calibration performed under different settings. Adjusting calibration.')

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
                            + "image that is not an integral multiple "
                            + "of the current map's images."
                            + "\n\t\tEnsure the calibration is for the "
                            + "correct detector with the appropriate binning.")
                raise ValueError(err_str)

            # Overwrite values
            self.ai.detector.shape = image_shape
            self.ai.detector.max_shape = image_shape # Not exactly correct, but more convenient
            self.ai.detector.pixel1 = poni_pixel1 / bin_est[0]
            self.ai.detector.pixel2 = poni_pixel2 / bin_est[1]

        # Extract calibration parameters to save
        self.poni = self.ai.get_config()

        # reset any previous calibrated arrays
        self._del_arr()
        
        # Save poni files as dictionary 
        # Updates poni information to update detector settings
        self.save_calibration()
    

    def save_calibration(self):
        if self.hdf_path is not None:
            # Open hdf flag
            keep_hdf = True
            if self.hdf is None:
                self.hdf = h5py.File(self.hdf_path, 'a')
                keep_hdf = False

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
                    #print(value)
                    new_grp.attrs[key] = value

            # Close hdf and reset attribute
            if not keep_hdf:
                self.hdf.close()
                self.hdf = None


    # One off 1D integration
    def integrate1d_image(self,
                          image=None,
                          tth_resolution=None,
                          tth_num=None,
                          unit='2th_deg',
                          **kwargs):
        # Intended for one-off temporary results

        if image is None:
            if self.corrections['polar_calibration']:
                raise RuntimeError("You are trying to calibrate already calibrated images!")
            else:
                image = self.composite_image
        
        if tth_resolution is None:
            tth_resolution = self.tth_resolution

        tth_min = np.min(self.tth_arr)
        tth_max = np.max(self.tth_arr)
        if tth_num is None:
            tth_num = int(np.round((tth_max - tth_min) / tth_resolution))
        elif tth_num is None and tth_resolution is None:
            raise ValueError('Must define either tth_num or tth_resolution.')
        
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
        # Intended for one-off temporary results

        if image is None:
            if self.corrections['polar_calibration']:
                raise RuntimeError("You are trying to clibrate already calibrated images!")
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
            tth_num = int(np.round((tth_max - tth_min) / tth_resolution))
        elif tth_num is not None:
            tth_resolution = (tth_max - tth_min) / tth_num
        elif tth_num is None and tth_resolution is None:
            raise ValueError('Must define either tth_num or tth_resolution.')
        
        # Get chi numbers
        chi_min = np.min(self.chi_arr)
        chi_max = np.max(self.chi_arr)
        if chi_num is None:
            chi_num = int(np.round((chi_max - chi_min) / chi_resolution))
        elif chi_num is not None:
            chi_resolution = (chi_max - chi_min) / chi_num
        elif chi_num is None and chi_resolution is None:
            raise ValueError('Must define either chi_num or chi_resolution.')
       
        return self.ai.integrate2d_ng(image, tth_num, chi_num,
                                      unit=unit,
                                      correctSolidAngle=False,
                                      polarization_factor=None,
                                      **kwargs)

    # Convenience function for image to polar coordinate transformation (estimate!)
    def estimate_polar_coords(self, coords, method='linear'):
        return estimate_polar_coords(coords, self.tth_arr, self.chi_arr, method=method)
    

    # Convenience function for polar to image coordinate transformation (estimate!)
    def estimate_image_coords(self, coords, method='nearest'):
        return estimate_image_coords(coords, self.tth_arr, self.chi_arr, method=method)
    
    ##################################
    ### Scaler and Position Arrays ###
    ##################################

    def set_scalers(self,
                    sclr_dict,
                    scaler_units='counts'):

        # Store sclr_dict as attribute
        for key, value in list(sclr_dict.items()):
            if value.shape != self.map_shape:
                sclr_dict[key] = value.reshape(self.map_shape)

        self.sclr_dict = sclr_dict
        self.scaler_units = scaler_units

        # Write to hdf file
        self.save_sclr_pos('scalers',
                            self.sclr_dict,
                            self.scaler_units)

    
    def save_sclr_pos(self, group_name, map_dict, unit_name):
        # Write to hdf file
        if self.hdf_path is not None:

            # Open hdf flag
            keep_hdf = True
            if self.hdf is None:
                self.hdf = h5py.File(self.hdf_path, 'a')
                keep_hdf = False

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
                        del dset # deletes flag, but not the data...
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

            # Close hdf and reset attribute
            if not keep_hdf:
                self.hdf.close()
                self.hdf = None
        
    
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
            raise TypeError(f"Unsure how to handle {phase} type.")

        if phase_name not in self.phases.keys():
            self.phases[phase_name] = phase
        else:
            print(f"Did not add {phase_name} since it is already a possible phase.")


    def remove_phase(self, phase):
        # Allow for phase object or name to work
        if hasattr(phase, 'name'):
            phase_name = phase.name
        elif isinstance(phase, str):
            phase_name = phase
        else:
            raise TypeError(f"Unsure how to handle {phase} type.")

        if phase_name in self.phases.keys():
            del self.phases[phase_name]
        else:
            print(f"Cannot remove {phase_name} since it is not in possible phases.")
        

    def load_phase(self, filename, filedir=None, phase_name=None):
        if filedir is None:
            filedir = self.wd
        
        phase_path = pathify(filedir, filename,
                             ['.cif', '.txt', '.D', '.h5'])

        if not os.path.exists(f'{phase_path}'):
            raise FileNotFoundError(f"Specified path does not exist:\n{phase_path}")
        
        if filename[-4:] == '.cif':
            phase = Phase.fromCIF(f'{phase_path}')
        elif filename[-4:] == '.txt':
            raise NotImplementedError()
        elif filename[-2:] in ['.D']:
            raise NotImplementedError()
        elif filename[-2:] == '.h5':
            raise NotImplementedError()
        else:
            raise TypeError(f'''Unsure how to read {filename}. 
                            Either specifiy file type or this file type is not supported.''')
        
        if phase_name is not None:
            phase.name = phase_name
        
        self.add_phase(phase)
    

    def clear_phases(self):
        self.phases = {}

    
    def update_phases(self):
        if (self.hdf_path is not None) and (len(self.phases) > 0):

            # Open hdf flag
            keep_hdf = True
            if self.hdf is None:
                self.hdf = h5py.File(self.hdf_path, 'a')
                keep_hdf = False

            phase_grp = self.hdf[self._hdf_type].require_group('phase_list')

            # Delete any no longer included phases
            for phase in phase_grp.keys():
                if phase not in self.phases.keys():
                    del(phase_grp[phase])

            # Save any new phases
            for phase in self.phases.values():
                phase.save_to_hdf(phase_grp)

            print('Updated phases saved in hdf.')
            
            # Close hdf and reset attribute
            if not keep_hdf:
                self.hdf.close()
                self.hdf = None


    def select_phases(self,
                      remove_less_than=-1,
                      image=None,
                      energy=None,
                      tth_num=4096,
                      unit='2th_deg',
                      ignore_less=1,
                      save_to_hdf=True):
        
        if not hasattr(self, 'ai') or self.ai is None:
            raise AttributeError('Must first set calibration before selecting phases.')
        
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
        
        tth, xrd = self.integrate1d_image(image=image, tth_num=tth_num, unit=unit)

        # Plot phase_selector
        phase_vals = phase_selector(xrd, list(self.phases.values()), tth, ignore_less=ignore_less)

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
            self.phases[phase].get_hkl_reflections(tth_range=(0, # Limited to zero for large d-spacing
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
