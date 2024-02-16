import numpy as np
import os
import h5py
import pyFAI
from pyFAI.io import ponifile
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from tqdm import tqdm
import time as ttime
import matplotlib.pyplot as plt
from collections import OrderedDict
import dask.array as da

import scipy.io as io
from dask_image import imread as dask_io


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
                 image_map=None,
                 map_title=None,
                 map_shape=None,
                 energy=None,
                 wavelength=None,
                 dwell=None,
                 poni_file=None,
                 tth_resolution=None,
                 chi_resolution=None,
                 tth=None,
                 chi=None,
                 beamline='5-ID (SRX)',
                 facility='NSLS-II',
                 time_stamp=None,
                 extra_metadata=None,
                 save_hdf=True,
                 dask_enabled=False):
        
        # Adding some metadata
        self.scanid = scanid
        if filename is None:
            filename = f'scan{scanid}_xrd'
        self.filename = filename

        if wd is None:
            wd = '/home/xf05id1/current_user_data/'
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

        self._energy = None
        self._wavelength = None
        if energy is not None:
            self.energy = energy
        elif wavelength is not None: # Favors energy definition
            self.wavelength = wavelength

        # Only take values from hdf when opening from the class method
        if save_hdf:
            if hdf is not None:
                self.hdf_path = hdf.filename
            elif hdf_filename is None:
                self.hdf_path = f'{wd}{filename}.h5'
            else:
                self.hdf_path = f'{wd}{hdf_filename}' # TODO: Add check for .h5   
            
            if os.path.exists(self.hdf_path):
                pass
            else:
                initialize_xrdmap_hdf(self, self.hdf_path) # Initialize base structure

            # Open hdf if required
            if dask_enabled:
                self.hdf = h5py.File(self.hdf_path, 'a')
        
        elif not save_hdf and dask_enabled:
            raise ValueError('Enabling dask requires an hdf file for storage.')
        
        else: # save_hdf is False and dask_enabled is False
            self.hdf_path = None
            self.hdf = None

        # Load image map
        if isinstance(image_map, (np.ndarray,
                                  da.core.Array,
                                  h5py._hl.dataset.Dataset)):
            self.map = ImageMap(image_map,
                                title=map_title,
                                wd=self.wd,
                                hdf_path=self.hdf_path,
                                hdf=hdf,
                                map_shape=map_shape,
                                dask_enabled=dask_enabled)
        elif isinstance(image_map, ImageMap):
            self.map = image_map
        else:
            raise TypeError(f"Unknown image_map input type: {type(image_map)}")
            
        self.phases = {} # Place holder for potential phases
        if poni_file is not None:
            self.set_calibration(poni_file)
        else:
            self.ai = None # Place holder for calibration
        
        if tth_resolution is not None:
            self.tth_resolution = tth_resolution
        if chi_resolution is not None:
            self.chi_resolution = chi_resolution
        if tth is not None:
            self.tth = tth
        if chi is not None:
            self.chi = chi


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
            ostr += '\t' + '\t'.join(self.map.__repr__() .splitlines(True))
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
                         map_shape=None,
                         map_title='raw_images',
                         **kwargs):
        
        # Load from image stack
        if wd is None:
            wd = '/home/xf05id1/current_user_data/'
        
        dask_enabled=False
        if 'dask_enabled' in kwargs:
            dask_enabled = kwargs['dask_enabled']

        print('Loading images...', end='', flush=True)
        if dask_enabled:
            dask_io.imread(f'{wd}{filename}')
        else:
            image_map = io.imread(f'{wd}{filename}')
        print('done!')
        return cls(image_map=image_map, wd=wd,
                   map_title=map_title, map_shape=map_shape,
                   **kwargs)


    @classmethod # Allows me to define and initiatie the class simultaneously
    def from_hdf(cls, hdf_filename, wd=None, dask_enabled=False, **kwargs):
        if wd is None:
            wd = '/home/xf05id1/current_user_data/'
        # Load from previously saved data, including all processed data...
        if os.path.exists(f'{wd}{hdf_filename}'):
            print('Loading data from hdf file...')
            input_dict = load_XRD_hdf(hdf_filename, wd=wd, dask_enabled=dask_enabled)

            #for key in kwargs.keys():
            #    if key in input_dict.keys():
            #        print((f"Warning: '{key}' found in hdf file and in keyword argument. "
            #              + "Defaulting to user-specification."))
            #    input_dict['base_md'][key] = kwargs[key]

            inst = cls(image_map=input_dict['image_data'],
                    wd=wd, filename=hdf_filename[:-3], # remove the .h5 extention
                    hdf_filename=hdf_filename,
                    hdf=input_dict['image_data'].hdf,
                    dask_enabled=dask_enabled,
                    **input_dict['base_md'],
                    poni_file=input_dict['poni_od'],
                    tth_resolution=input_dict['image_data'].tth_resolution,
                    chi_resolution=input_dict['image_data'].chi_resolution,
                    tth=input_dict['recip_pos']['tth'],
                    chi=input_dict['recip_pos']['chi'],
                    **kwargs)
            
            # Add a few more attributes if they exist
            if input_dict['recip_pos']['calib_units'] is not None:
                inst.calib_units = input_dict['recip_pos']['calib_units']
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
    

    @classmethod # Allows me to define and initiatie the class simultaneously
    def from_db(cls, scanid, wd=None, **kwargs):
        # Load from databroker
        raise NotImplementedError()
        if wd is None:
            wd = '/home/xf05id1/current_user_data/'

        h = db.start[scanid]
        # Get metadata from start documents
        # I'll need to call load_xrd_tiffs or something like that from IO
        return cls(scanid, energy=energy, image_map=working_map, wd=wd,
                   map_title='raw_images', **kwargs)
    

    @staticmethod
    def make_hdf(cls, scanid, wd=None):
        # make an hdf file for the xrdmap class from the databroker
        # without returning the instance
        raise NotImplementedError()
    
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
        if hasattr(self, 'ai'):
            self.ai.energy = self._energy
            self._del_arr()
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
        if hasattr(self, 'ai'):
            self.ai.energy = self._energy
            self._del_arr()
        if hasattr(self, 'phases'):
            for key in self.phases.keys():
                self.phases[key].energy = self._energy

    
    @property
    def tth_arr(self):
        if hasattr(self, '_tth_arr'):
            return self._tth_arr

        elif ((hasattr(self, 'map'))
            and (self.map is not None)
            and (self.map.corrections['polar_calibration'])):

            if hasattr(self, 'tth') and hasattr(self, 'chi'):
                tth_arr, _ = np.meshgrid(self.tth, self.chi[::-1])
                self._tth_arr = tth_arr
                return self._tth_arr

        elif hasattr(self, 'ai') and self.ai is not None:
            self._tth_arr = np.degrees(self.ai.twoThetaArray())
            return self._tth_arr
    
    @tth_arr.deleter
    def tth_arr(self):
        self._del_arr()
    
    @property
    def delta_tth(self):
        return delta_array(self.tth_arr)

    
    @property
    def chi_arr(self):
        if hasattr(self, '_chi_arr'):
            return self._chi_arr

        elif ((hasattr(self, 'map'))
            and (self.map is not None)
            and (self.map.corrections['polar_calibration'])):

            if hasattr(self, 'tth') and hasattr(self, 'chi'):
                _, chi_arr = np.meshgrid(self.tth, self.chi[::-1])
                self._chi_arr = chi_arr
                return self._chi_arr

        elif hasattr(self, 'ai') and self.ai is not None:
            self._chi_arr = np.degrees(self.ai.chiArray())
            return self._chi_arr
    
    @chi_arr.deleter
    def chi_arr(self):
        self._del_arr()
    
    @property
    def delta_chi(self):
        pos_chi = self.chi_arr.copy()
        # Check for discontinuities
        if np.max(np.gradient(pos_chi)) > 30: # Semi-arbitrary cut off
            pos_chi[pos_chi < 0] += 360

        return delta_array(pos_chi)

    
    @property
    def q_arr(self):
        if hasattr(self, '_q_arr'):
            return self._q_arr
        # This value will not be cached. It could be, but that is a lot of logic
        q_arr = get_q_vect(self.tth_arr, self.chi_arr, wavelength=self.wavelength)
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
        
    # This is to help when working with lazy loaded images
    def _close_hdf(self):
        if self.hdf is not None:
            self.hdf.close()
            self.hdf = None
        
        if self.map.hdf is not None:
            self.map.hdf.close()
            self.map.hdf = None

    def _open_hdf(self, dask_enabled=False):
        if self.hdf is not None:
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
        
        # If dask is not enabled, the images should not be lazy loaded
            

    #############################
    ### Correcting Map Images ###
    #############################

    # Not sure if I should wrap this here or keep it inside of the imagemap class

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

            self.ai = pyFAI.load(f'{filedir}{poni_file}')
        
        elif isinstance(poni_file, OrderedDict):
            self.ai = AzimuthalIntegrator().set_config(poni_file)

        elif isinstance(poni_file, ponifile.PoniFile):
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
                self.ai.detector.pixel1 = poni_pixel1 / bin_est[0]
                self.ai.detector.pixel2 = poni_pixel2 / bin_est[1]

        else:
            print('Warning: Could not find any images to compare calibration!')
            print('Defaulting to detectors settings used for calibration.')

        # Extract calibration parameters to save
        self.poni = self.ai.get_config()
        
        # Save poni files as dictionary 
        # Updates poni information to update detector settings
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
                if isinstance(value, OrderedDict):
                    new_new_grp = new_grp.require_group(key)
                    for key_i, value_i in value.items():
                        new_new_grp.attrs[key_i] = value_i
                else:
                    new_grp.attrs[key] = value

            # Close hdf and reset attribute
            if not keep_hdf:
                self.hdf.close()
                self.hdf = None


    # Move this to a function in geometry??
    def calibrate_images(self, poni_file=None, filedir=None,
                         title='calibrated_images', unit='2th_deg',
                         tth_resolution = 0.02, chi_resolution = 0.05,
                         polarization_factor=None,
                         Lorentz_correction=None,
                         **kwargs):
        if poni_file is not None:
            self.set_calibration(poni_file, filedir=filedir)
        if not hasattr(self, 'ai'):
            raise RuntimeError("Images cannot be calibrated without any calibration files!")

        
        if hasattr(self, 'tth_resolution'):
            tth_resolution = self.tth_resolution
        if hasattr(self, 'chi_resolution'):
            chi_resolution = self.chi_resolution
        
        out = self.map.calibrate_images(self.ai, title=title,
                                        tth_resolution=tth_resolution,
                                        chi_resolution=chi_resolution,
                                        unit=unit,
                                        polarization_factor=polarization_factor,
                                        Lorentz_correction=Lorentz_correction,
                                        **kwargs)
        
        self.tth = out[0]
        self.chi = out[1]
        self.extent = out[2]
        self.calibrated_shape = out[3]
        self.tth_resolution = out[4]
        self.chi_resolution = out[5]

        # Clean up a few cached properties. Will reset when next called
        del self.tth_arr
        del self.chi_arr
        

    def integrate_1d(self, image=None, tth_num=4096,
                     unit='2th_deg',
                     polarization_factor=None, correctSolidAngle=False, **kwargs):
        # Intended for one-off temporary results

        if image is None:
            if self.map.corrections['polar_calibration']:
                raise RuntimeError("You are trying to calibrate already calibrated images!")
            else:
                image = self.map.composite_image  
        
        return self.ai.integrate1d_ng(image, tth_num,
                                      unit=unit,
                                      polarization_factor=polarization_factor,
                                      correctSolidAngle=correctSolidAngle,
                                      **kwargs)
    

    def integrate_2d(self, image, tth_num, chi_num, unit='2th_deg',
                     polarization_factor=None, correctSolidAngle=False, **kwargs):
        # Intented for one-off temporary results
        if image is None:
            if self.map.corrections['polar_calibration']:
                raise RuntimeError("You are trying to clibrate already calibrated images!")
            else:
                image = self.map.composite_image
       
        return self.ai.integrate2d_ng(image, tth_num, chi_num,
                                        unit=unit,
                                        polarization_factor=polarization_factor,
                                        correctSolidAngle=correctSolidAngle,
                                        **kwargs)
    

    # Convenience function for image to polar coordinate transformation (estimate!)
    def estimate_polar_coords(self, coords, method='linear'):
        return estimate_polar_coords(coords, self.tth_arr, self.chi_arr, method=method)
    

    # Convenience function for polar to image coordinate transformation (estimate!)
    def estimate_image_coords(self, coords, method='nearest'):
        return estimate_image_coords(coords, self.tth_arr, self.chi_arr, method=method)
    
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
            print(f"Did not add {phase_name} since it is already in possible phases.")


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

        if not os.path.exists(f'{filedir}{filename}'):
            raise FileNotFoundError(f"Specified path does not exist:\n{filedir}{filename}")
        
        if filename[-4:] == '.cif':
            phase = Phase.fromCIF(f'{filedir}{filename}')
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
            

        tth, xrd = self.integrate_1d(image=image, tth_num=tth_num, unit=unit)
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
    
    ######################################################
    ### Blob, Ring, Peak and Spot Search and Selection ###
    ######################################################

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
        # Not sure about this one...
        #self.map.blurred_images = np.asarray(thresh_list).reshape(*self.map.map_shape,
        #                                                          *self.map.calibrated_shape)

        # Save spots to hdf
        if self.hdf_path is not None:
            
            # Open hdf flag
            keep_hdf = True
            if self.hdf is None:
                self.hdf = h5py.File(self.hdf_path, 'a')
                keep_hdf = False

            print('Saving spots to hdf...', end='', flush=True)
            # Save spots to hdf
            self._close_hdf() # pandas cannot work with already open files...
            self.spots.to_hdf(self.hdf_path, 'xrdmap/reflections/spots', format='table')
            self._open_hdf()

            # Save masks to hdf
            self.map.save_images(images=self.map.spot_masks,
                                 title='_spot_masks',
                                 units='bool',
                                 extra_attrs={'threshold_method' : threshold_method,
                                              'size' : size,
                                              'multiplier' : multiplier,
                                              'window_radius' : radius})
            
            # Close hdf and reset attribute
            if not keep_hdf:
                self.hdf.close()
                self.hdf = None

            print('done!')


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
                    self._close_hdf()
                    spots = pd.read_hdf(self.hdf_path, key='xrdmap/reflections/spots')
                    self.spots = spots
                    self._open_hdf()

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

        # TEMP: trying to see the relative size of this variable in memory
        self.spot_fit_info_list = spot_fit_info_list
        
        # Fits spots and adds the fit results to the spots dataframe
        fit_spots(self, spot_fit_info_list, SpotModel)
        self.spot_model = SpotModel

        # Save spots to hdf
        if self.hdf_path is not None:
            print('Saving spots to hdf...', end='', flush=True)
            #with h5py.File(self.hdf_path, 'a') as f:
            #    # Check for and remove existing spots...
            #    if 'spots' in f['xrdmap/reflections']:
            #        del f['xrdmap/reflections/spots']

            # Open hdf flag
            keep_hdf = True
            if self.hdf is None:
                self.hdf = h5py.File(self.hdf_path, 'a')
                keep_hdf = False

            # Save to hdf
            self._close_hdf()
            self.spots.to_hdf(self.hdf_path, 'xrdmap/reflections/spots', format='table')
            self._open_hdf()
            self.hdf['xrdmap/reflections'].attrs['spot_model'] = SpotModel.name

            # Close hdf and reset attribute
            if not keep_hdf:
                self.hdf.close()
                self.hdf = None
            
            print('done!')


    def initial_spot_analysis(self, SpotModel=None):

        if SpotModel is None and hasattr(self, 'spot_model'):
            SpotModel = self.spot_model

        # Initial spot analysis...
        _initial_spot_analysis(self, SpotModel=SpotModel)

        # Save spots to hdf
        if self.hdf_path is not None:
            self._close_hdf()
            self.spots.to_hdf(self.hdf_path, 'xrdmap/reflections/spots', format='table')
            self._open_hdf()


    def trim_spots(spots, remove_less=0.1):
        raise NotImplementedError()

    #################################
    ### Analysis of Selected Data ###
    #################################
        


    ############################################################
    ### Interfacing with Fluorescence Data and pyxrf Results ###
    ############################################################
    
    def incorporate_xrf_map():
        raise NotImplementedError()
        # Allow for plotting of xrf results from pyXRF

    ##########################
    ### Plotting Functions ###
    ##########################

    def plot_image(self, image=None, indices=None, title=None,
                mask=None, spots=False, contours=False,
                aspect='auto', vmin=None, 
                return_plot=False,
                **kwargs):
        
        # Check image type
        if image is not None:
            image = np.asarray(image)
            if len(image.shape) == 1 and len(image) == 2:
                indices = tuple(iter(image))
                image = self.map.images[indices]
            elif len(image.shape) == 2:
                if indices is not None:
                    indices = tuple(indices)
            else:
                raise ValueError(f"Incorrect image shape of {image.shape}. Should be two-dimensional.")
        else:
            # Evaluate images
            self.map._dask_2_dask()

            if indices is not None:
                indices = tuple(indices)
                image = self.map.images[indices]
                image = np.asarray(image)
            else:
                i = np.random.randint(self.map.map_shape[0])
                j = np.random.randint(self.map.map_shape[1])
                indices = (i, j)
                image = self.map.images[indices]
                image = np.asarray(image)

        # Check for mask
        if mask is not None:
            if mask is True:
                image = image * self.map.mask
            elif np.asarray(mask).shape == image.shape:
                image = image * mask
            else:
                raise RuntimeError("Error handling mask input.")
            
        # Plot image
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=200)
        # Allow some flexibility for kwarg inputs
        plot_kwargs = {'c' : 'r',
                    'lw' : 0.5,
                    's' : 1}
        for key in plot_kwargs.keys():
            if key in kwargs.keys():
                plot_kwargs[key] = kwargs[key]
                del kwargs[key]

        if hasattr(self.map, 'extent'):
            if vmin == None:
                vmin = 0
            im = ax.imshow(image, extent=self.extent, vmin=vmin, aspect=aspect, **kwargs)
            ax.set_xlabel('Scattering Angle, 2θ [°]') # Assumes degrees. Need to change...
            ax.set_ylabel('Azimuthal Angle, χ [°]')
        else:
            im = ax.imshow(image, vmin=vmin, aspect=aspect, **kwargs)
            ax.set_xlabel('X index')
            ax.set_ylabel('Y index')
        fig.colorbar(im, ax=ax) 

        if title is not None:
            ax.set_title(title)
        elif indices is not None:
            ax.set_title(f'Row = {indices[0]}, Col = {indices[1]}')
        elif self.map.title is not None:
            ax.set_title(self.map.title)
        else:
            ax.set_title('Input Image')

        if indices is not None:
            # Set some default values
            
            # Plot spots
            if spots and hasattr(self, 'spots'):
                pixel_df = self.spots[(self.spots['map_x'] == indices[0]) & (self.spots['map_y'] == indices[1])].copy()
                if any([x[:3] == 'fit' for x in pixel_df.keys()]):
                    pixel_df.dropna(axis=0, inplace=True)
                    spots = pixel_df[['fit_chi0', 'fit_tth0']].values
                else:
                    spots = pixel_df[['guess_cen_chi', 'guess_cen_tth']].values

                if not self.map.corrections['polar_calibration']:
                    spots = estimate_image_coords(spots[:, ::-1], self.tth_arr, self.chi_arr)[:, ::-1]
                ax.scatter(spots[:, 1], spots[:, 0], s=plot_kwargs['s'], c=plot_kwargs['c'])
            
            elif spots and not hasattr(self, 'spots'):
                print('Warning: Plotting spots requested, but xrdmap does not have any spots!')

            # Plot contours
            if contours and hasattr(self.map, 'spot_masks'):
                blob_img = label(self.map.spot_masks[indices])
                blob_contours = find_blob_contours(blob_img)
                for contour in blob_contours:
                    if self.map.corrections['polar_calibration']:
                        contour = estimate_polar_coords(contour.T, self.tth_arr, self.chi_arr).T
                    ax.plot(*contour, c=plot_kwargs['c'], lw=plot_kwargs['lw'])
                
            elif contours and not hasattr(self, 'spot_masks'):
                print('Warning: Plotting spots requested, but xrdmap does not have any spots!')
        
        elif spots or contours:
            print('Warning: Cannot request spots or contours without providing map indices!')

        if return_plot:
            return fig, ax
        
        plt.show()


    def plot_reconstruction(self, indices=None, plot_residual=False, **kwargs):
        if not hasattr(self, 'spots'):
            raise RuntimeError('xrdmap does not have any spots!')

        if indices is None:
            i = np.random.randint(self.map.map_shape[0])
            j = np.random.randint(self.map.map_shape[1])
            indices = (i, j)
        else:
            indices = tuple(indices)
            if (indices[0] < 0 or indices[0] > self.map.map_shape[0]):
                raise IndexError(f'Indices ({indices}) is out of bounds along axis 0 for map shape ({self.map.map_shape})')
            elif (indices[1] < 0 or indices[1] > self.map.map_shape[1]):
                raise IndexError(f'Indices ({indices}) is out of bounds along axis 1 for map shape ({self.map.map_shape})')
        
        if hasattr(self, 'spot_model'):
            spot_model = self.spot_model
        else:
            print('Warning: No spot model saved. Defaulting to Gaussian.')
            spot_model = GaussianFunctions
        
        pixel_df = self.spots[(self.spots['map_x'] == indices[0]) & (self.spots['map_y'] == indices[1])].copy()

        if any([x[:3] == 'fit' for x in pixel_df.keys()]):
            prefix = 'fit'
            pixel_df.dropna(axis=0, inplace=True)
            param_labels = [x for x in self.spots.loc[0].keys() if x[:3] == 'fit'][:6]
        else:
            prefix = 'guess'
            param_labels = ['height', 'cen_tth', 'cen_chi', 'fwhm_tth', 'fwhm_chi']
            param_labels = [f'guess_{param_label}' for param_label in param_labels]
            spot_model = GaussianFunctions

        fit_args = []
        for index in pixel_df.index:
            fit_args.extend(pixel_df.loc[index, param_labels].values)
            if prefix == 'guess':
                fit_args.append(0) # Filling in theta value

        if len(fit_args) > 0:
            #return fit_args
            recon_image = spot_model.multi_2d([self.tth_arr.ravel(), self.chi_arr.ravel()], 0, *fit_args)
            recon_image = recon_image.reshape(self.map.images.shape[-2:])
        else:
            recon_image = np.zeros(self.map.images.shape[-2:])

        if not plot_residual:
            fig, ax = self.plot_image(recon_image,
                                return_plot=True, indices=indices,
                                **kwargs)
            plt.show()

        else:
            image = self.map.images[indices]
            residual = recon_image - image
            ext = np.max(np.abs(residual[self.map.mask]))
            fig, ax = self.plot_image(residual,
                                title=f'Residual of (Row = {indices[0]}, Col = {indices[1]})',
                                return_plot=True, indices=indices,
                                vmin=-ext, vmax=ext, cmap='bwr', # c='k',
                                **kwargs)
            plt.show()


    def plot_interactive_map(self, tth=None, chi=None, **kwargs):
        # I should probably rebuild these to not need tth and chi
        if hasattr(self, 'tth') and tth is None:
            tth = self.tth
        if hasattr(self, 'chi') and chi is None:
            chi = self.chi

        interactive_dynamic_2d_plot(np.asarray(self.map.images), tth=tth, chi=chi, **kwargs)
    

    def plot_interactive_integration_map(self, display_map=None, display_title=None):
        # Map integrated patterns for dynamic exploration of dataset
        # May throw an error if data has not yet been integrated
        raise NotImplementedError()
    

    def plot_map_value(self, data, cmap='viridis'):
        # Simple base mapping function for analyzed values.
        # Will expand to map phase assignment, phase integrated intensity, etc.
        raise NotImplementedError()
    
    # Dual plotting a map with a representation of the full data would be very interesting
    # Something like the full strain tensor which updates over each pixel
    # Or similar for dynamically updating pole figures


    ##################################
    ### Plot Experimental Geometry ###
    ##################################

    def plot_q_space(self, pixel_indices=None, skip=500, return_plot=False):
 
        q = get_q_vect(self.tth_arr, self.chi_arr, wavelength=self.wavelength)

        if pixel_indices is not None:
            pixel_df = self.spots[(self.spots['map_x'] == pixel_indices[0])
                                    & (self.spots['map_y'] == pixel_indices[1])].copy()

        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=200, subplot_kw={'projection':'3d'})

        # Plot sampled Ewald sphere
        q_mask = q[:, self.map.mask]
        ax.plot_trisurf(q_mask[0].ravel()[::skip],
                        q_mask[1].ravel()[::skip],
                        q_mask[2].ravel()[::skip],
                        alpha=0.5, label='detector')

        # Plot full Ewald sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        radius = 2 * np.pi / self.wavelength
        x =  radius * np.outer(np.cos(u), np.sin(v))
        y = radius * np.outer(np.sin(u), np.sin(v))
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z - radius, alpha=0.2, color='k', label='Ewald sphere')

        if pixel_indices is not None:
            ax.scatter(*pixel_df[['qx', 'qy', 'qz']].values.T, s=1, c='r', label='spots')

        # Sample geometry
        ax.quiver([0, 0], [0, 0], [-2 * radius, -radius], [0, 0], [0, 0], [radius, radius], colors='k')
        ax.scatter(0, 0, 0, marker='o', s=10, facecolors='none', edgecolors='k', label='transmission')
        ax.scatter(0, 0, -radius, marker='h', s=10, c='b', label='sample')

        ax.set_xlabel('qx [Å⁻¹]')
        ax.set_ylabel('qy [Å⁻¹]')
        ax.set_zlabel('qz [Å⁻¹]')
        ax.set_aspect('equal')

        # Initial view
        ax.view_init(elev=-45, azim=90, roll=0)
        plt.show()

        if return_plot:
            return fig, ax


    def plot_detector_geometry(self, skip=300, return_plot=False):

        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=200, subplot_kw={'projection':'3d'})

        # Plot detector position
        xyz = self.ai.position_array()

        xyz[:, :, 0] *= -1 # Transform to synchrotron standard. Not sure if correct

        x = xyz[:, :, 0].ravel()[::skip]
        y = xyz[:, :, 1].ravel()[::skip]
        z = xyz[:, :, 2].ravel()[::skip]

        ax.plot_trisurf(x, y, z,
                        alpha=0.5, label='detector')

        # X-ray beam
        radius = self.ai.dist
        ax.quiver([0], [0], [-radius], [0], [0], [radius], colors='k')
        ax.scatter(0, 0, 0, marker='h', s=10, c='b', label='sample')

        # Detector
        corner_indices = np.array([[0, 0], [-1, 0], [0, -1], [-1, -1]]).T
        corn = xyz[*corner_indices].T
        ax.quiver([0,] * 4,
                [0,] * 4,
                [0,] * 4,
                corn[0],
                corn[1],
                corn[2], colors='gray', lw=0.5)

        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        ax.set_aspect('equal')

        # Initial view
        ax.view_init(elev=-60, azim=90, roll=0)
        plt.show()

        if return_plot:
            return fig, ax