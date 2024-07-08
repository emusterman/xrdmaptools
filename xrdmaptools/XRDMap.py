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
from skimage.measure import label
from dask_image import imread as dask_io
from tqdm import tqdm

# Local imports
from .ImageMap import ImageMap
#from .utilities.hdf_io import initialize_xrdmap_hdf, load_xrdmap_hdf
#from .utilities.hdf_utils import check_hdf_current_images
#from .utilities.db_io import load_data
from .utilities.math import *
from .utilities.utilities import (
    delta_array,
    check_ext,
    pathify
)

from .io.hdf_io import initialize_xrdmap_hdf, load_xrdmap_hdf
from .io.hdf_utils import check_hdf_current_images
from .io.db_io import load_data

from .reflections.spot_blob_indexing import get_q_vect, _initial_spot_analysis
from .reflections.SpotModels import GaussianFunctions
from .reflections.spot_blob_search import (
    find_spots,
    find_spot_stats,
    make_stat_df,
    prepare_fit_spots,
    fit_spots,
    find_blob_contours
    )

from .plot.interactive_plotting import (interactive_dynamic_2d_plot,
                                        interactive_dynamic_1d_plot)
from .plot.general import (
    _plot_parse_xrdmap,
    _xrdmap_image,
    _xrdmap_integration,
    plot_image,
    plot_integration,
    plot_reconstruction,
    plot_map,
    plot_cake
)
from .plot.geometry import plot_q_space, plot_detector_geometry

from .geometry.geometry import *

from .crystal.Phase import Phase, phase_selector


class XRDMap():
    '''
    Main class object for sXRD map.
    Inherits nothing!
    Multiple iteratations of image processing across full map cannot be saved in memory...
    '''

    def __init__(self,
                 scanid=None,
                 wd=None,
                 filename=None,
                 hdf_filename=None,
                 hdf=None,
                 image_data=None,
                 integration_data=None,
                 map_title=None,
                 dataset_shape=None,
                 energy=None,
                 wavelength=None,
                 dwell=None,
                 theta=None,
                 poni_file=None,
                 sclr_dict=None,
                 pos_dict=None,
                 tth_resolution=0.01,
                 chi_resolution=0.05,
                 tth=None,
                 chi=None,
                 beamline='5-ID (SRX)',
                 facility='NSLS-II',
                 time_stamp=None,
                 extra_metadata=None,
                 save_hdf=True,
                 dask_enabled=False,
                 xrf_path=None):
        
        # Adding some metadata
        self.scanid = scanid
        if filename is None:
            filename = f'scan{scanid}_xrd'
        self.filename = filename

        if wd is None:
            wd = os.getcwd()
        self.wd = wd

        self.beamline = beamline
        self.facility = facility
        self.time_stamp = time_stamp
        self.dwell = dwell
        self.extra_metadata = extra_metadata # not sure about this...
        # Scan.start.uid: 7469f8f8-8076-47d5-85a1-ee147fe89d3c
        # Scan.start.ctime: Mon Feb 27 12:57:12 2023
        # uid: 7469f8f8-8076-47d5-85a1-ee147fe89d3c
        # sample.name: 

        # Store energy, dwell, and theta
        self._energy = None
        self._wavelength = None
        if energy is not None:
            self.energy = energy
        elif wavelength is not None: # Favors energy definition
            self.wavelength = wavelength
        
        if dwell is not None:
            self.dwell = dwell
        if theta is not None:
            self.theta = theta

        # Add conditional to check approximate image_data size and force dask?

        if not save_hdf:
            if dask_enabled:
                raise ValueError('Enabling dask requires an hdf file for storage.')
            else:
                self.hdf_path = None
                self.hdf = None
        else:
            self.save_hdf(hdf=hdf,
                          hdf_filename=hdf_filename,
                          dask_enabled=dask_enabled,
                          save_current=False)

        if (image_data is not None
            or integration_data is not None):
            if isinstance(image_data, ImageMap):
                self.map = image_data
                # Used to specify read-only when loading from hdf
                if not save_hdf:
                    self.map.hdf_path = None
            else:
                self.map = ImageMap(image_data=image_data,
                                    integration_data=integration_data,
                                    title=map_title,
                                    wd=self.wd,
                                    hdf_path=self.hdf_path,
                                    hdf=hdf,
                                    dataset_shape=dataset_shape,
                                    dask_enabled=dask_enabled)
        else:
            raise NotImplementedError('XRDMaps without image or integration data is not currently supported.')
        
        self.phases = {} # Place holder for potential phases
        if poni_file is not None:
            self.set_calibration(poni_file)
        else:
            self.ai = None # Place holder for calibration
            self.map.ai = None # Redundant

        self.sclr_dict = None
        if sclr_dict is not None:
            self.set_scalers(sclr_dict)

        self.pos_dict = None
        if pos_dict is not None:
            self.set_positions(pos_dict)

        # Save xrf_path location. Do not load unless explicitly called
        self.xrf_path = xrf_path
        
        # Default units and flags
        # Not fully implemented
        self._polar_units = 'deg' # 'rad' or 'deg'
        self._scattering_units = 'deg' # 'rad', 'deg', 'nm^-1', 'A^-1'
        self._integration_scale = 'linear' # 'linear' or 'log'
        self._image_scale = 'linear' # 'linear' or 'log'

        if tth is not None and len(tth) == 0:
            tth = None
        self.tth = tth
        self.tth_resolution = tth_resolution
        if chi is not None and len(chi) == 0:
            chi = None
        self.chi = chi
        self.chi_resolution = chi_resolution


    def __str__(self):
        ostr = f'XRDMap:  scanid= {self.scanid}, energy= {self.energy}, shape= {self.map.images.shape}'
        return ostr


    def __repr__(self):
        # Native info
        ostr = f'XRDMap:'
        ostr += f'\n\tFacility:\t{self.facility}'
        ostr += f'\n\tBeamline:\t{self.beamline}'
        if self.scanid is not None:
            ostr += f'\n\tScanid:\t\t{self.scanid}'
        ostr += f'\n\tEnergy:\t\t{self.energy} keV'
        if self.hdf_path is not None:
            ostr += f'\n\tHDF Path:\t{self.hdf_path}\n'

        # Other info
        if hasattr(self, 'map'): # pull ImageMap info
            #ostr += '\t' + self.map.__repr__() 
            ostr += '\n\t' + '\t'.join(self.map.__repr__() .splitlines(True))
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


    ################################
    ### Loading data into XRDMap ###
    ################################

    @classmethod # Allows me to define and initiatie the class simultaneously
    def from_image_stack(cls, filename, wd=None,
                         dataset_shape=None,
                         map_title='raw_images',
                         **kwargs):
        
        # Load from image stack
        if wd is None:
            wd = os.getcwd()
        
        image_path = pathify(wd, filename, ['.tif', '.tiff', '.jpeg', '.png'])
        
        dask_enabled=False
        if 'dask_enabled' in kwargs:
            dask_enabled = kwargs['dask_enabled']

        print('Loading images...', end='', flush=True)
        if dask_enabled:
            image_data = dask_io.imread(image_path)
        else:
            image_data = io.imread(image_path)
        print('done!')
        return cls(image_data=image_data, wd=wd,
                   map_title=map_title, dataset_shape=dataset_shape,
                   **kwargs)


    @classmethod # Allows me to define and initiatie the class simultaneously
    def from_hdf(cls, hdf_filename, wd=None, dask_enabled=False, **kwargs):
        if wd is None:
            wd = os.getcwd()
        # Load from previously saved data, including all processed data...
        hdf_path = pathify(wd, hdf_filename, '.h5')
        if os.path.exists(hdf_path):
            print('Loading data from hdf file...')
            input_dict = load_xrdmap_hdf(hdf_filename, wd=wd, dask_enabled=dask_enabled)

            #for key in kwargs.keys():
            #    if key in input_dict.keys():
            #        print((f"Warning: '{key}' found in hdf file and in keyword argument. "
            #              + "Defaulting to user-specification."))
            #    input_dict['base_md'][key] = kwargs[key]

            inst = cls(image_data=input_dict['image_data'],
                    wd=wd,
                    filename=hdf_filename[:-3], # remove the .h5 extention
                    hdf_filename=hdf_filename,
                    hdf=input_dict['image_data'].hdf, # Will always be None
                    dask_enabled=dask_enabled,
                    poni_file=input_dict['poni_od'],
                    sclr_dict=input_dict['sclr_dict'],
                    pos_dict=input_dict['pos_dict'],
                    **input_dict['base_md'],
                    **input_dict['recip_pos'],
                    #tth_resolution=input_dict['recip_pos']['tth_resolution'],
                    #chi_resolution=input_dict['recip_pos']['chi_resolution'],
                    #tth=input_dict['recip_pos']['tth'],
                    #chi=input_dict['recip_pos']['chi'],
                    **kwargs)
            
            # Add a few more attributes if they exist
            if hasattr(inst.map, 'extent'):
                inst.extent = inst.map.extent
            if hasattr(inst.map, 'calibrated_shape'):
                inst.calibrated_shape = inst.map.calibrated_shape
            
            # Load phases
            if 'phase_dict' in input_dict.keys():
                inst.phases = input_dict['phase_dict']

            # Load spots
            if input_dict['spots'] is not None:
                inst.spots = input_dict['spots']

            if input_dict['spot_model'] is not None:
                inst.spot_model = input_dict['spot_model']

            print('XRD Map loaded!')
            return inst
        
        else:
            raise FileNotFoundError('No such hdf file.')
        

    @ classmethod
    def from_db(cls,
                scanid=-1,
                broker='manual',
                filedir=None,
                filename=None,
                poni_file=None,
                data_keys=None,
                save_hdf=True,
                repair_method='replace'):
    
        # No fluorescence key
        pos_keys = ['enc1', 'enc2']
        sclr_keys = ['i0', 'i0_time', 'im', 'it']
        
        if data_keys is None:
            data_keys = pos_keys + sclr_keys

        data_dict, scan_md, data_keys, xrd_dets = load_data(
                                                    scanid=scanid,
                                                    broker=broker,
                                                    detectors=None,
                                                    data_keys=data_keys,
                                                    returns=['data_keys',
                                                                'xrd_dets'],
                                                    repair_method=repair_method)

        xrd_data = [data_dict[f'{xrd_det}_image'] for xrd_det in xrd_dets]

        # Make position dictionary
        pos_dict = {key:value for key, value in data_dict.items() if key in pos_keys}

        # Make scaler dictionary
        sclr_dict = {key:value for key, value in data_dict.items() if key in sclr_keys}

        if len(xrd_data) > 1:
            pass
            # Add more to filename to prevent overwriting...

        extra_md = {}
        for key in scan_md.keys():
            if key not in ['scan_id',
                           'beamline',
                           'energy',
                           'dwell',
                           'theta',
                           'start_time']:
                extra_md[key] = scan_md[key]
        
        xrdmaps = []
        for xrd_data_i in xrd_data:
            xrdmap = cls(scanid=scan_md['scan_id'],
                         wd=filedir,
                         filename=filename,
                         #hdf_filename=None, # ???
                         #hdf=None,
                         image_data=xrd_data_i,
                         #map_title=None,
                         #dataset_shape=None,
                         energy=scan_md['energy'],
                         #wavelength=None,
                         dwell=scan_md['dwell'],
                         theta=scan_md['theta'],
                         poni_file=poni_file,
                         sclr_dict=sclr_dict,
                         pos_dict=pos_dict,
                         #tth_resolution=None,
                         #chi_resolution=None,
                         #tth=None,
                         #chi=None,
                         beamline=scan_md['beamline_id'],
                         facility='NSLS-II',
                         time_stamp=scan_md['time_str'],
                         extra_metadata=extra_md,
                         save_hdf=save_hdf
                         )
            
            xrdmaps.append(xrdmap)

        if len(xrdmaps) > 1:
            return tuple(xrdmaps)
        else:
            # Don't bother returning a tuple or list of xrdmaps
            return xrdmaps[0]
    
    
    ##################
    ### Properties ###
    ##################

    @property
    def energy(self):
        return self._energy
    
    @energy.setter
    def energy(self, energy):
        if energy is not None:
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


    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, wavelength):
        if wavelength is not None:
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
            elif ((hasattr(self, 'map'))
                  and (self.map is not None)
                  and (self.map.corrections['polar_calibration'])): # I should rename...
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
            raise AttributeError('AzimuthalIntegrator (ai) not specified for XRDMap.')

        def del_angle_array(self):
            delattr(self, f'_{arr_name}')

        # Delta arrays are not cached
        def get_delta_array(self):
            arr = getattr(self, arr_name) # should call the property

            max_arr = np.max(np.abs(arr))
            delta_arr = delta_array(arr)

            # Modular shift values if there is a discontinuity
            # Should only be for chi_arr between 180 and -180 degrees
            # I could just shift everything...
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
        

    ##########################
    ### Image manipulation ###
    ##########################


    # TODO: test if this actually works, move to ImageMap
    def load_images_from_hdf(self, image_dataset):
        # Only most processed images will be loaded from hdf
        # Deletes current image map and loads new values from hdf
        print(f'Loading {image_dataset}')

        # Open hdf flag
        keep_hdf = True
        if self.hdf is None:
            self.hdf = h5py.File(self.hdf_path, 'r')
            keep_hdf = False
        
        # Working with dask flag
        dask_enabled = self.map._dask_enabled

        # Actually load the data
        image_grp = self.hdf['xrdmap/image_data']
        if check_hdf_current_images(image_dataset, hdf=self.hdf):
            del(self.map.images) # Delete previous images from ImageMap to save memory
            img_dset = image_grp[image_dataset]
            
            if dask_enabled:
                self.map.images = da.asarray(img_dset)
            else:
                self.map.images = np.asarray(img_dset)

            # Rebuild correction dictionary
            corrections = {}
            for key in image_grp[image_dataset].attrs.keys():
                # _{key}_correction
                corrections[key[1:-11]] = image_grp[image_dataset].attrs[key]
            self.map.corrections = corrections

        # Close hdf and reset attribute
        if not keep_hdf:
            self.hdf.close()
            self.hdf = None

        self.map.update_map_title()
        self.map._dask_2_hdf()


    def dump_images(self):
        del(self.map)
        # Intended to clear up memory when only working with indentified spots
        # May not be useful

    
    ############################
    ### Dask / HDF Functions ###
    ############################

    def save_hdf(self,
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
        
        # If nothing is provided, it should be able to build a default location if properly saved
        #if hdf is None and hdf_path is None and hdf_filename is None:
        #    raise ValueError('Cannot write to hdf without specifying a file or location.')

        # Specify hdf path and name
        if hdf is not None:# biases towards already open hdf
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
        if os.path.exists(self.hdf_path):
            pass
        else:
            initialize_xrdmap_hdf(self, self.hdf_path) # Initialize base structure

        # Open hdf if required
        if dask_enabled:
            self.hdf = h5py.File(self.hdf_path, 'a')
        else:
            self.hdf = None
        
        # Saves current images, integrations, phases, calibration, etc.
        if save_current:
            self.save_current_xrdmap()


    # Saves current major features of the xrdmap
    def save_current_xrdmap(self):
        if hasattr(self, 'map') and self.map is not None:
            # Save images
            if hasattr(self.map, 'images'):
                if self.map._dask_enabled:
                    self.map.dask_2_hdf()
                else:
                    self.map.save_images()
            # Save integrations
            if (hasattr(self.map, 'integrations')
                and self.map.integrations is not None):
                self.map.save_integrations()
        
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

        if hasattr(self, 'map') and self.map._dask_enabled:
            err_str = ('WARNING: Image data is lazy loaded. '
                       'Stopping or switching hdf is likely to cause problems.'
                       '\nSave progress and close the hdf with "close_hdf" '
                       'function before changing save location.')
            raise RuntimeError(err_str)
        
        self.close_hdf()
        self.hdf_path = None
        self.map.hdf_path
    

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
            self.initial_save_hdf(hdf=hdf,
                          hdf_path=hdf_path,
                          hdf_filename=hdf_filename,
                          dask_enabled=dask_enabled)
    
    ### Helper functions for lazy loading with Dask ###

    # This function does NOT stop saving to hdf
    # It only closes open hdf locations and stops lazy loading images
    def close_hdf(self):
        if self.hdf is not None:
            self.hdf.close()
            self.hdf = None
        
        if self.map.hdf is not None:
            self.map._dask_2_hdf()
            # If using dask, probably shouldn't try to load data
            # self.map._dask_2_numpy()
            self.map.hdf.close()
            self.map.hdf = None


    def open_hdf(self, dask_enabled=False):
        if self.hdf is not None:
            # Should this raise errors or just pring warnings
            raise ValueError('XRDMap HDF file is already open.')
        else:
            self.hdf = h5py.File(self.hdf_path, 'a')

        if self.map.hdf is not None:
            raise ValueError('ImageMap HDF file is already open.')
        else:
            self.map.hdf = self.hdf

        if dask_enabled or self.map._dask_enabled: # This flag persists even when the dataset is closed!
            img_grp = self.map.hdf['xrdmap/image_data']
            if self.map.title == 'final_images':
                if check_hdf_current_images(self.map.title, hdf=self.hdf):
                    dset = img_grp[self.map.title]
            elif check_hdf_current_images('_temp_images', hdf=self.hdf):
                dset = img_grp['_temp_images']
            self.map.images = da.asarray(dset).persist()
            self.map._hdf_store = dset
            

    ##############################
    ### Calibrating Map Images ###
    ##############################
    
    def set_calibration(self, poni_file, filedir=None):
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
        if self.energy is not None:
            self.ai.energy = self.energy # Allows calibrations acquired at any energy
        else:
            print('Energy has not been defined. Defaulting to .poni file value.')
            self.energy = self.ai.energy

        # Update detector shape and pixel size if different from poni file
        if hasattr(self, 'map'):
            try:
                image_shape = list(self.map.image_shape)
            except AttributeError:
                image_shape = list(self.map.images.shape[:-2])

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

        else:
            print('Warning: Could not find any images to compare calibration!')
            print('Defaulting to detectors settings used for calibration.')

        # Extract calibration parameters to save
        self.poni = self.ai.get_config()

        # Share ai with ImageMap
        if hasattr(self, 'map') and self.map is not None:
            self.map.ai = self.ai

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
            curr_grp = self.hdf[f'/xrdmap'].require_group('reciprocal_positions')
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
        

    def integrate1d_map(self,
                        tth_num=None,
                        tth_resolution=None,
                        unit='2th_deg',
                        **kwargs):
        
        if not hasattr(self, 'ai'):
            raise RuntimeError("Images cannot be calibrated without any calibration files!")
        
        if tth_resolution is None:
            tth_resolution = self.tth_resolution
        
        tth_min = np.min(self.tth_arr)
        tth_max = np.max(self.tth_arr)
        if tth_num is None:
            tth_num = int(np.round((tth_max - tth_min) / tth_resolution))
        elif tth_num is not None:
            tth_resolution = (tth_max - tth_min) / tth_num
        elif tth_num is None and tth_resolution is None:
            raise ValueError('Must define either tth_num or tth_resolution.')

        # Set up empty array to fill
        integrated_map1d = np.empty((self.map.num_pixels, 
                                     tth_num), 
                                     dtype=(self.map.dtype))
        
        # Fill array!
        print('Integrate images to 1D...')
        # TODO: Parallelize this
        for i, pixel in tqdm(enumerate(self.map.images.reshape(
                                       self.map.num_pixels,
                                       *self.map.image_shape)),
                                       total=self.map.num_pixels):
        
            tth, I, = self.integrate1d_image(image=pixel,
                                             tth_num=tth_num,
                                             unit=unit,
                                             **kwargs)            

            integrated_map1d[i] = I

        # Reshape into (map_x, map_y, tth)
        # Does not explicitly match the same shape as 2d integration
        integrated_map1d = integrated_map1d.reshape(
                                *self.map.map_shape, tth_num)
        #self.map.images = integrated_map1d
        self.map.integrations = integrated_map1d
        
        # Save a few potentially useful parameters
        self.tth = tth
        self.extent = [np.min(self.tth), np.max(self.tth)]
        self.tth_resolution = tth_resolution

        # Save integrations to hdf
        if self.hdf_path is not None:
            print('Compressing and writing integrations to disk...')
            self.map.save_integrations()
            print('done!')
            self.save_reciprocal_positions()
        

    # Briefly doubles memory. No Dask support
    def integrated2d_map(self,
                         tth_num=None,
                         tth_resolution=None,
                         chi_num=None,
                         chi_resolution=None,
                         unit='2th_deg',
                         **kwargs):
        
        if not hasattr(self, 'ai'):
            raise RuntimeError("Images cannot be calibrated without any calibration files!")
        
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

        # Set up empty array to fill
        integrated_map2d = np.empty((self.map.num_pixels, 
                                     chi_num, tth_num), 
                                     dtype=(self.map.dtype))
        
        # Fill array!
        print('Integrated images to 2D...')
        # TODO: Parallelize this
        for i, pixel in tqdm(enumerate(self.map.images.reshape(
                                       self.map.num_pixels,
                                       *self.map.image_shape)),
                                       total=self.map.num_pixels):
        
            I, tth, chi = self.integrate2d_image(image=pixel,
                                                 tth_num=tth_num,
                                                 unit=unit,
                                                 **kwargs)            

            integrated_map2d[i] = I

        # Reshape into (map_x, map_y, chi, tth)
        integrated_map2d = integrated_map2d.reshape(
                                *self.map.map_shape, chi_num, tth_num)
        self.map.images = integrated_map2d
        
        # Save a few potentially useful parameters
        self.tth = tth
        self.chi = chi
        self.extent = [np.min(self.tth), np.max(self.tth),
                       np.min(self.chi), np.max(self.chi)]
        self.tth_resolution = tth_resolution
        self.chi_resolution = chi_resolution

        # Save recirpocal space positions
        self.save_reciprocal_positions()
        

    # One off 1D integration
    def integrate1d_image(self,
                          image=None,
                          tth_resolution=None,
                          tth_num=None,
                          unit='2th_deg',
                          **kwargs):
        # Intended for one-off temporary results

        if image is None:
            if self.map.corrections['polar_calibration']:
                raise RuntimeError("You are trying to calibrate already calibrated images!")
            else:
                image = self.map.composite_image
        
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
            if self.map.corrections['polar_calibration']:
                raise RuntimeError("You are trying to clibrate already calibrated images!")
            else:
                image = self.map.composite_image

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
    

    def save_reciprocal_positions(self):
        
        if self.hdf_path is not None:
            # Check for values to save           
            if self.tth is None:
                tth = []
            else:
                tth = self.tth
            if self.chi is None:
                chi = []
            else:
                chi = self.chi

            print('Writing reciprocal positions to disk...', end='', flush=True)
            # Open hdf flag
            keep_hdf = True
            if self.hdf is None:
                self.hdf = h5py.File(self.hdf_path, 'a')
                keep_hdf = False

            # This group may already exist if poni file was already initialized
            curr_grp = self.hdf[f'/xrdmap'].require_group('reciprocal_positions')
            if hasattr(self, 'extent'):
                curr_grp.attrs['extent'] = self.extent

            labels = ['tth_pos', 'chi_pos']
            comments = [''''tth', is the two theta scattering angle''',
                        ''''chi' is the azimuthal angle''']
            keys = ['tth', 'chi']
            data = [tth, chi]
            resolution = [self.tth_resolution, self.chi_resolution]

            for i, key in enumerate(keys):
                # Skip values that are None
                if data[i] is None:
                    data[i] = np.array([])

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

            # Close hdf and reset attribute
            if not keep_hdf:
                self.hdf.close()
                self.hdf = None
            print('done!')


    # Convenience function for image to polar coordinate transformation (estimate!)
    def estimate_polar_coords(self, coords, method='linear'):
        return estimate_polar_coords(coords, self.tth_arr, self.chi_arr, method=method)
    

    # Convenience function for polar to image coordinate transformation (estimate!)
    def estimate_image_coords(self, coords, method='nearest'):
        return estimate_image_coords(coords, self.tth_arr, self.chi_arr, method=method)
    
    ##################################
    ### Scaler and Position Arrays ###
    ##################################

    def set_scalers(self, sclr_dict, scaler_units='counts'):

        # Store sclr_dict as attribute
        self.sclr_dict = sclr_dict

        # Share scalers with ImageMap
        if hasattr(self, 'map') and self.map is not None:
            self.map.sclr_dict = self.sclr_dict

        self.scaler_units = scaler_units

        # Write to hdf file
        self.save_sclr_pos('scalers',
                            self.sclr_dict,
                            self.scaler_units)
    

    def set_positions(self, pos_dict, position_units=None):

        # Re-work dictionary keys into stable format
        temp_dict = {}
        for key in list(pos_dict.keys()):
            if key in ['enc1', '1', 'x', 'X', 'map_x', 'map_X']:
                temp_dict['map_x'] = pos_dict[key]
            elif key in ['enc2', '2' 'y', 'Y', 'map_y', 'map_Y']:
                temp_dict['map_y'] = pos_dict[key]
        del pos_dict
        pos_dict = temp_dict

        # Store pos_dict as attribute
        self.pos_dict = pos_dict
        # Positions are not shared with ImageMap...

        # Set position units
        if position_units is None:
            position_units = 'μm' # default to microns, not that reliable...
        self.position_units = position_units

        # Determine fast scanning direction for map extent
        if (np.mean(np.diff(self.pos_dict['map_x'], axis=1))
            > np.mean(np.diff(self.pos_dict['map_x'], axis=0))):
            # Fast x-axis. Standard orientation.
            map_extent = [
                np.mean(self.pos_dict['map_x'][:, 0]),
                np.mean(self.pos_dict['map_x'][:, -1]),
                np.mean(self.pos_dict['map_y'][0]),
                np.mean(self.pos_dict['map_y'][-1])
            ]
        else: # Fast y-axis. Consider swapping axes???
            map_extent = [
                np.mean(self.pos_dict['map_y'][:, 0]),
                np.mean(self.pos_dict['map_y'][:, -1]),
                np.mean(self.pos_dict['map_x'][0]),
                np.mean(self.pos_dict['map_x'][-1])
            ]
        self.map_extent = map_extent

        # Write to hdf file
        self.save_sclr_pos('positions',
                            self.pos_dict,
                            self.position_units)

    
    def save_sclr_pos(self, group_name, map_dict, unit_name):
        # Write to hdf file
        if self.hdf_path is not None:

            # Open hdf flag
            keep_hdf = True
            if self.hdf is None:
                self.hdf = h5py.File(self.hdf_path, 'a')
                keep_hdf = False

            # Write data to hdf
            curr_grp = self.hdf[f'/xrdmap'].require_group(group_name)
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


    # Convenience function for loading scalers and positions from standard map_parameters text file
    def load_map_parameters(self, filename, filedir=None, position_units=None):  
        
        if filedir is None:
            filedir = self.wd

        path = pathify(filedir, filename, '.txt')
        arr = np.genfromtxt(path)

        pos_dict, sclr_dict = {}, {}

        pos_dict['enc1'] = arr[0].reshape(self.map.map_shape)
        pos_dict['enc2'] = arr[1].reshape(self.map.map_shape)

        sclr_dict['i0'] = arr[2].reshape(self.map.map_shape)
        sclr_dict['i0_time'] = arr[3].reshape(self.map.map_shape)
        sclr_dict['im'] = arr[4].reshape(self.map.map_shape)
        sclr_dict['it'] = arr[5].reshape(self.map.map_shape)

        self.set_positions(pos_dict, position_units)
        self.set_scalers(sclr_dict)


    # Method to swap axes, specifically swapping the default format of fast and slow axes
    def swap_axes(self, exclude_imagemap=False, save_updates=False):
        # This will break if images are loaded with dask.
        # _temp_images will be of the wrong shape...
        # could be called before _temp_images dataset is instantiated??
        # exclude_imagemap included to swap axes upon instantiation
        # Never save imagemap. Leave that for specific situations...

        if self.map.title == 'final_images' and not exclude_imagemap:
            warn_str = ('WARNING: ImageMap has been finalized.'
                        + '\nSaving other attributes with swapped '
                        + 'axes may create inconsistencies.')
            print(warn_str)

        self.map.images = self.map.images.swapaxes(0, 1)

        if hasattr(self, 'pos_dict'):
            for key in list(self.pos_dict.keys()):
                self.pos_dict[key] = self.pos_dict[key].swapaxes(0, 1)
        if hasattr(self, 'sclr_dict'):
            for key in list(self.sclr_dict.keys()):
                self.sclr_dict[key] = self.sclr_dict[key].swapaxes(0, 1)
        if hasattr(self.map, 'spot_masks'):
            self.map.spot_masks = self.map.spot_masks.swapaxes(0, 1)

        # Update imagemap attrs if included
        if not exclude_imagemap:
            # Update shape values
            self.map.shape = self.map.images.shape
            #self.map.num_pixels = np.multiply(*self.map.images.shape[:2])
            self.map.map_shape = self.map.shape[:2]
            #self.map.image_shape = self.map.shape[2:]

            # Delete any cached ImaegeMap maps
            old_attr = list(self.map.__dict__.keys())       
            for attr in old_attr:
                if attr in ['_min_map',
                            '_max_map',
                            '_med_map',
                            '_sum_map',
                            '_mean_map',]:
                    delattr(self.map, attr)

        # Exchange map_extent
        if hasattr(self, 'map_extent'):
            map_extent = self.map_extent.copy()
            self.map_extent = [map_extent[2],
                               map_extent[3],
                               map_extent[0],
                               map_extent[1]]
            
        # Update spot map_indices
        if hasattr(self, 'spots'):
            map_x_ind = self.spots['map_x'].values
            map_y_ind = self.spots['map_y'].values
            self.spots['map_x'] = map_y_ind
            self.spots['map_y'] = map_x_ind

        if save_updates:
            if hasattr(self, 'pos_dict'):
                # Write to hdf file
                self.save_sclr_pos('positions',
                                    self.pos_dict,
                                    self.position_units)
                
            if hasattr(self, 'sclr_dict'):
                self.save_sclr_pos('scalers',
                                    self.sclr_dict,
                                    self.scaler_units)
                
            if hasattr(self, 'spots'):
                self.save_spots()

    
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
            print(f"Did not add {phase_name} since it is already a possible phases.")


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
        
        if self.energy is not None:
            phase.energy = self.energy
        
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

            phase_grp = self.hdf['xrdmap'].require_group('phase_list')

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


    def select_phases(self, remove_less_than=-1,
                      image=None, tth_num=4096,
                      unit='2th_deg', ignore_less=1,
                      save_to_hdf=True):
        
        if image is None:
            if self.map.corrections['polar_calibration']:
                image = self.map._processed_images_composite
            else:
                image = self.map.composite_image
            

        tth, xrd = self.integrate1d_image(image=image, tth_num=tth_num, unit=unit)
        # Add background subraction??

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
    
    #################################
    ### Spot Search and Selection ###
    #################################

    def find_spots(self, threshold_method='gaussian',
                   multiplier=5, size=3,
                   radius=5, expansion=None):
        
        # Cleanup images as necessary
        self.map._dask_2_numpy()
        if np.max(self.map.images) != 100:
            print('Rescaling images to max of 100 and min around 0.')
            self.map.rescale_images(arr_min=0, upper=100)

        # Estimate remaining map noise to determine peak significance
        #self.map_noise = estimate_map_noise(self.map, sample_number=200)

        # Search each image for significant spots
        spot_list, mask_list = find_spots(self.map,
                                          mask=self.map.mask,
                                          threshold_method=threshold_method,
                                          multiplier=multiplier,
                                          size=size,
                                          expansion=expansion)

        # Initial characterization of each spot
        stat_list = find_spot_stats(self.map,
                                    spot_list,
                                    self.tth_arr,
                                    self.chi_arr,
                                    radius=radius)

        # Convert spot stats into dict, then pandas dataframe
        stat_df = make_stat_df(stat_list, self.map.map_shape)

        # .spots attribute will be the basis will be treated similarly to .map
        # Most subsequent analysis will be built here
        # Consider wrapping it in a class like ImageMap or Phase
        self.spots = stat_df
        self.map.spot_masks = np.asarray(mask_list).reshape(*self.map.map_shape,
                                                       *self.map.images.shape[-2:])

        # Save spots to hdf
        self.save_spots()

        # Save masks to hdf
        self.map.save_images(images=self.map.spot_masks,
                                title='_spot_masks',
                                units='bool',
                                extra_attrs={'threshold_method' : threshold_method,
                                            'size' : size,
                                            'multiplier' : multiplier,
                                            'window_radius' : radius})


    def fit_spots(self, SpotModel, max_dist=0.5, sigma=1):

        # Find spots in self or from hdf
        if not hasattr(self, 'spots'):
            print('No reflection spots found...')
            if self.hdf_path is not None:
                # Open hdf flag
                keep_hdf = True
                if self.hdf is None:
                    self.hdf = h5py.File(self.hdf_path, 'r')
                    keep_hdf = False

                if 'reflections' in self.hdf['xrdmap'].keys():
                    print('Loading reflection spots from hdf...', end='', flush=True)
                    self.close_hdf()
                    spots = pd.read_hdf(self.hdf_path, key='xrdmap/reflections/spots')
                    self.spots = spots
                    self.open_hdf()

                    # Close hdf and reset attribute
                    if not keep_hdf:
                        self.hdf.close()
                        self.hdf = None
                    print('done!')

                else:
                    raise AttributeError('XRDMap does not have any reflection spots! Please find spots first.')
            else:
                raise AttributeError('XRDMap does not have any reflection spots! Please find spots first.')

        # Generate list of x, y, I, and spot indices for each blob/spots fits
        spot_fit_info_list = prepare_fit_spots(self, max_dist=max_dist, sigma=sigma)
        
        # Fits spots and adds the fit results to the spots dataframe
        fit_spots(self, spot_fit_info_list, SpotModel)
        self.spot_model = SpotModel

        # Save spots to hdf
        self.save_spots(extra_attrs={'spot_model' : self.spot_model.name})


    def initial_spot_analysis(self, SpotModel=None):

        if SpotModel is None and hasattr(self, 'spot_model'):
            SpotModel = self.spot_model

        # Initial spot analysis...
        _initial_spot_analysis(self, SpotModel=SpotModel)

        # Save spots to hdf
        self.save_spots()


    def trim_spots(self, remove_less=0.01, metric='height', save_spots=False):
        if not hasattr(self, 'spots') or self.spots is None:
            raise ValueError('Cannot trim spots if XRDMap has not no spots.')

        metric = str(metric).lower()
        if any([x[:3] == 'fit' for x in self.spots.iloc[0].keys()]):
            if metric in ['height', 'amp']:
                significance = self.spots['fit_amp'] - self.spots['fit_offset']
            elif metric in ['intensity', 'int', 'breadth', 'integrated', 'volume']:
                significance = self.spots['fit_integrated'] # this should account for offset too
            else:
                raise ValueError('Unknown metric specification.')
        else:
            if metric in ['height', 'amp']:
                significance = self.spots['guess_height']
            elif metric in ['intensity', 'int', 'breadth', 'integrated', 'volume']:
                significance = self.spots['guess_int']
            else:
                raise ValueError('Unknown metric specification.')

        # Find relative indices where conditional is true
        mask = np.where(significance.values < remove_less)[0]

        # Convert relative indices into dataframe index
        drop_indices = self.spots.iloc[mask].index.values # awful call

        # Drop indices
        self.spots.drop(index=drop_indices, inplace=True)
        print(f'Trimmed {len(drop_indices)} spots less than {remove_less} significance.')

        if save_spots:
            self.save_spots()


    def remove_spot_fits(self):
        drop_keys = []
        for key in list(self.spots.keys()):
            if 'fit' in key:
                print(key)
                drop_keys.append(key)

        self.spots.drop(drop_keys, axis=1, inplace=True)
    

    def pixel_spots(self, map_indices):
        # TODO: These values may need to be reversed. Check with mapping values...
        pixel_spots = self.spots[(self.spots['map_x'] == map_indices[0])
                               & (self.spots['map_y'] == map_indices[1])].copy()
        return pixel_spots
    

    def save_spots(self, extra_attrs=None):
        # Save spots to hdf
        if self.hdf_path is not None:
            print('Saving spots to hdf...', end='', flush=True)

            # Open hdf flag
            keep_hdf = True
            if self.hdf is None:
                self.hdf = h5py.File(self.hdf_path, 'a')
                keep_hdf = False

            # Save to hdf
            self.close_hdf()
            self.spots.to_hdf(self.hdf_path, key='xrdmap/reflections/spots', format='table')

            if extra_attrs is not None:
                self.open_hdf()
                for key, value in extra_attrs.items():
                    self.hdf['xrdmap/reflections'].attrs[key] = value

            if keep_hdf:
                self.open_hdf()
            else:
                self.close_hdf()
            
            print('done!')

    #################################
    ### Analysis of Selected Data ###
    #################################
        


    #############################################################
    ### Interfacing with Fluorescence Data from pyxrf Results ###
    #############################################################
    
    def load_xrfmap(self, xrf_dir=None, xrf_name=None, full_data=True):

        # Look for path if no information is provided
        if (xrf_dir is None
            and xrf_name is None
            and self.xrf_path is not None):
            xrf_path = self.xrf_path
        else:
            if xrf_dir is None:
                xrf_dir = self.wd

            # Try default name for SRX
            if xrf_name is None:
                xrf_name =  f'scan2D_{self.scanid}_xs_sum8ch'
            
            xrf_path = pathify(xrf_dir, xrf_name, '.h5')        

        if not os.path.exists(xrf_path):
            raise FileNotFoundError(f"{xrf_path} does not exist.")
        else:
            self.xrf_path = xrf_path

        # Load the data
        xrf = {}
        with h5py.File(self.xrf_path, 'r') as f:
            
            if full_data:
                xrf['data'] = f['xrfmap/detsum/counts'][:]
                xrf['energy'] = np.arange(xrf['data'].shape[-1]) / 100

            xrf_fit_names = [d.decode('utf-8') for d in f['xrfmap/detsum/xrf_fit_name'][:]]
            xrf_fit = f['xrfmap/detsum/xrf_fit'][:]

            i0 = f['xrfmap/scalers/val'][..., 0]
            xrf_fit = np.concatenate((xrf_fit, np.expand_dims(i0, axis=0)), axis=0)
            xrf_fit = np.transpose(xrf_fit, axes=(0, 2, 1))
            xrf_fit_names.append('i0')

            for key, value in zip(xrf_fit_names, xrf_fit):
                xrf[key] = value

            xrf['E0'] = f['xrfmap/scan_metadata'].attrs['instrument_mono_incident_energy']
        
        # Track as attribute
        self.xrf = xrf

        # Save xrf_path to hdf
        if self.hdf_path is not None:
            # Open hdf flag
            keep_hdf = True
            if self.hdf is None:
                self.hdf = h5py.File(self.hdf_path, 'a')
                keep_hdf = False

            # Only save the path to connect two files
            self.hdf['xrdmap'].attrs['xrf_path'] = self.xrf_path

            if not keep_hdf:
                self.hdf.close()
                self.hdf = None


        

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

    # Interactive plots do not currently accept fig, ax inputs

    def plot_interactive_map(self,
                             image_data=None,
                             xticks=None,
                             yticks=None,
                             return_plot=False,
                             **kwargs):

        if image_data is not None:
            image_data = np.asarray(image_data)
        elif not hasattr(self, 'map'):
            raise ValueError('Could not find ImageMap to plot data!')
        elif self.map.images.ndim != 4:
            raise ValueError(f'ImageMap data shape is not 4D, but {self.map.images.ndim}')
        else:
            image_data = self.map.images

        if xticks is None and self.map.corrections['polar_calibration']:
            if hasattr(self, 'tth') and self.tth is not None:
                xticks = self.tth
            if hasattr(self, 'chi') and self.chi is not None:
                yticks = self.chi

        # Check for, extract, or determine displaymap
        if 'display_map' not in kwargs.keys():
            display_map = self.map.sum_map
        else:
            display_map = kwargs['display_map']
            del kwargs['display_map']

        fig, ax = interactive_dynamic_2d_plot(image_data,
                                              xticks=xticks,
                                              yticks=yticks,
                                              display_map=display_map,
                                              **kwargs)
        if return_plot:
            return fig, ax
        else:
            fig.show()
    

    def plot_interactive_integration_map(self,
                                         integrated_data=None,
                                         xticks=None,
                                         return_plot=False,
                                         **kwargs):
        # Map integrated patterns for dynamic exploration of dataset
        # May throw an error if data has not yet been integrated
        if integrated_data is not None:
            integrated_data = np.asarray(integrated_data)
        elif not hasattr(self, 'map'):
            raise ValueError('Could not find ImageMap to plot data!')
        elif not hasattr(self.map, 'integrations'):
            raise ValueError('Could not find integrations in ImageMap to plot data!')
        elif self.map.integrations.ndim != 3:
            raise ValueError(f'ImageMap data shape is not 4D, but {self.map.integrations.ndim}')
        else:
            integrated_data = self.map.integrations

        if xticks is not None:
            if hasattr(self, 'tth') and self.tth is not None:
                xticks = self.tth
    
        # Check for, extract, or determine displaymap
        if 'display_map' not in kwargs.keys():
            display_map = self.map.sum_map
        else:
            display_map = kwargs['display_map']
            del kwargs['display_map']
    
        fig, ax = interactive_dynamic_1d_plot(integrated_data,
                                              xticks=xticks,
                                              display_map=display_map,
                                              **kwargs)
        if return_plot:
            return fig, ax
        else:
            fig.show()
    

    def plot_map(self,
                 value,
                 map_extent=None,
                 position_units=None,
                 fig=None,
                 ax=None,
                 return_plot=False,
                 **kwargs):
        
        if map_extent is None:
            map_extent = self.map_extent
        if position_units is None:
            position_units = self.position_units
        
        fig, ax = plot_map(value,
                           map_extent=map_extent,
                           position_units=position_units,
                           fig=fig,
                           ax=ax,
                           **kwargs)
        
        if return_plot:
            return fig, ax
        else:
            fig.show()
    

    # Dual plotting a map with a representation of the full data would be very interesting
    # Something like the full strain tensor which updates over each pixel
    # Or similar for dynamically updating pole figures


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
