import numpy as np
import os
import h5py
import pyFAI
from pyFAI.io import ponifile
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
import skimage.io as io
from skimage.restoration import rolling_ball
from tqdm import tqdm
import time as ttime
import matplotlib.pyplot as plt
from collections import OrderedDict


class XRDMap():
    '''
    Main class object for sXRD map.
    Inherits nothing!
    Multiple iteratations of image processing across full map cannot be saved in memory...
    '''

    def __init__(self, scanid=None, wd=None, filename=None, h5_filename=None,
                 image_map=None, map_title=None, map_shape=None,
                 energy=None, wavelength=None, poni_file=None,
                 tth_resolution=None, chi_resolution=None,
                 tth=None, chi=None,
                 beamline='5-ID (SRX)', facility='NSLS-II',
                 time_stamp=None,
                 extra_metadata=None,
                 save_h5=True):
        
        # Adding some metadata
        self.scanid = scanid
        if filename is None:
            filename = f'scan{scanid}_xrd'
        self.filename = filename
        self.wd = wd
        self.beamline = beamline
        self.facility = facility
        self.time_stamp = time_stamp
        self.extra_metadata = extra_metadata # not sure about this...
        # Beamline.name: SRX
        # Facility.name: NSLS-II
        # Facility.ring_current:400.1146824791909
        # Scan.start.uid: 7469f8f8-8076-47d5-85a1-ee147fe89d3c
        # Scan.start.time: 1677520632.6182282
        # Scan.start.ctime: Mon Feb 27 12:57:12 2023
        # Mono.name: Si 111
        # uid: 7469f8f8-8076-47d5-85a1-ee147fe89d3c
        # sample.name: 

        # Probably a better way to do this..
        self._energy = energy
        self._wavelength = wavelength
        if self._energy is not None:
            self.energy = energy
        elif self._wavelength is not None: # Favors energy definition
            self.wavelength = wavelength

        # Only take values from h5 when opening from the class method
        if save_h5:
            if h5_filename is None:
                self.h5 = f'{wd}{filename}.h5'
                if os.path.exists(f'{wd}{filename}.h5'):
                    pass
                else:
                    initialize_xrdmap_h5(self, self.h5) # Initialize base structure
            else: # specify specific h5 file
                self.h5 = f'{wd}{h5_filename}'
        else:
            self.h5 = None


        # Load image map
        if isinstance(image_map, np.ndarray):
            self.map = ImageMap(image_map, title=map_title,
                                h5=self.h5, map_shape=map_shape)
        elif isinstance(image_map, ImageMap):
            self.map = image_map
        else:
            raise IOError(f"Unknown image_map input type: {type(image_map)}")

        # Immediately save to h5 if not already done so
        # Should only ever be called upon first loading of a raw dataset
        #print(f'imagemap_title is {self.map.title}')
        if self.h5 is not None:
            if not check_h5_current_images(self, self.h5):
                print('Writing images to h5...', end='', flush=True)
                self.map.save_images(units='counts',
                                     labels=['x_ind',
                                             'y_ind',
                                             'img_y',
                                             'img_x'])
                print('done!')
            
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
        return
    

    def __repr__(self):
        ostr = f'XRDMap'
        return ostr

    ################################
    ### Loading data into XRDMap ###
    ################################

    @classmethod # Allows me to define and initiatie the class simultaneously
    def from_db(cls, scanid, wd=None):
        # Load from databroker
        raise NotImplementedError()
        if wd is None:
            wd = '/home/xf05id1/current_user_data/'

        h = db.start[scanid]
        # Get metadata from start documents
        # I'll need to call load_xrd_tiffs or something like that from IO
        return cls(scanid, energy=energy, image_map=working_map, wd=wd,
                   map_title='raw_images')


    @classmethod # Allows me to define and initiatie the class simultaneously
    def from_image_stack(cls, filename, wd=None,
                         map_shape=None, **kwargs):
        # Load from image stack
        if wd is None:
            wd = '/home/xf05id1/current_user_data/'
        
        print('Loading images...', end='', flush=True)
        image_map = io.imread(f'{wd}{filename}')
        print('done!')
        return cls(image_map=image_map, wd=wd, map_title='raw_images', **kwargs)


    @classmethod # Allows me to define and initiatie the class simultaneously
    def from_h5(cls, h5_filename, wd=None):
        # Load from previously saved data, including all processed data...
        print('Loading data from h5 file...')
        input_dict = load_XRD_h5(h5_filename, wd=wd)

        inst = cls(image_map=input_dict['image_data'],
                   wd=wd, filename=h5_filename[:-3], # remove the .h5 extention
                   h5_filename=h5_filename,
                   **input_dict['base_md'],
                   poni_file=input_dict['poni_od'],
                   tth_resolution=input_dict['image_data'].tth_resolution,
                   chi_resolution=input_dict['image_data'].chi_resolution,
                   tth=input_dict['recip_pos']['tth'],
                   chi=input_dict['recip_pos']['chi'])
        
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

        print('XRD Map loaded!')
        return inst
    
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
    
    ###########################################
    ### Loading and Saving Data of Instance ###
    ###########################################

    def load_images_from_h5(self, image_dataset):
        # Only most processed images will be loaded from h5
        # Deletes current image map and loads new values from h5
        print(f'Loading {image_dataset}')
        with h5py.File(self.h5, 'r') as f:
            image_grp = f['xrdmap/image_data']
            if image_dataset in image_grp.keys():
                del(self.map.images) # Delete previous images from ImageMap to save memory
                img_dset = image_grp[image_dataset]
                self.map.images = img_dset[:] # Load image array into ImageMap

                    # Rebuild correction dictionary
                corrections = {}
                for key in image_grp[image_dataset].attrs.keys():
                    # _{key}_correction
                    corrections[key[1:-11]] = image_grp[image_dataset].attrs[key]
                self.map.corrections = corrections

        self.map.update_map_title()


    def dump_images(self):
        del(self.map)
        # Intented to clear up memory when only working with indentified spots
        # May not be useful

    #############################
    ### Correcting Map Images ###
    #############################

    def correct_images(self):
        raise NotImplementedError()

    ##############################
    ### Calibrating Map Images ###
    ##############################
    
    def set_calibration(self, poni_file, filedir=None):
        if filedir is None:
            filedir = self.wd

        if isinstance(poni_file, str):
            if not os.path.exists(f'{filedir}{poni_file}'):
                raise IOError(f"{filedir}{poni_file} does not exist")

            if poni_file[-4:] != 'poni':
                raise RuntimeError("Please provide a .poni file.")

            self.ai = pyFAI.load(f'{filedir}{poni_file}')
        
        elif isinstance(poni_file, OrderedDict):
            self.ai = AzimuthalIntegrator().set_config(poni_file)

        elif isinstance(poni_file, ponifile.PoniFile):
            self.ai = AzimuthalIntegrator().set_config(poni_file.as_dict())
        
        else:
            raise TypeError(f"{type(poni_file)} is unknown and not supported!")

        # Update energy
        if self.energy is not None:
            self.ai.energy = self.energy # Allows calibrations acquired at any energy
        else:
            print('Energy has not been defined. Defualting to .poni file value.')
            self.energy = self.ai.energy

        # Extract calibration parameters to save
        self.poni = self.ai.get_config()
        
        # Save poni files as dictionary 
        if self.h5 is not None:
            with h5py.File(self.h5, 'a') as f:
                curr_grp = f[f'/xrdmap'].require_group('reciprocal_positions')
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


    # Move this to a function in geometry??
    def calibrate_images(self, poni_file=None, filedir=None,
                         title='calibrated_images', unit='2th_deg',
                         tth_resolution = 0.02, chi_resolution = 0.05,
                         polarization_factor=0.9,
                         Lorentz_correction=True,
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
        


    def integrate_1d(self, image=None, tth_num=4096,
                     unit='2th_deg',
                     polarization_factor=0.9, **kwargs):
        # Intended for one-off temporary results

        if (self.map.title == 'calibrated_images'
            and image is None):
            # Assumes the image input should be correct...
            raise RuntimeError("You are trying to clibrate already clibrated images!")
        
        elif image is None:
            image = self.map.composite_image
            
        
        return self.ai.integrate1d_ng(image, tth_num,
                                      unit=unit,
                                      polarization_factor=polarization_factor,
                                      **kwargs)
    

    def integrate_2d(self, image, tth_num, chi_num, unit='2th_deg',
                     polarization_factor=0.9, **kwargs):
        # Intented for one-off temporary results
        if image is None:
            image = self.map.composite_image
        
        if (self.map.title == 'calibrated_images'
            and image is None):
            # Assumes the image input should be correct...
            raise RuntimeError("You are trying to clibrate already clibrated images!")
       
        return self.ai.integrate2d_ng(image, tth_num, chi_num,
                                        unit=unit,
                                        polarization_factor=polarization_factor,
                                        **kwargs)
    

    def estimate_img_coords(self, coords):
        return estimate_img_coords(coords, self.map.image_shape, tth=self.tth, chi=self.chi)


    def estimate_reciprocal_coords(self, coords):
        return estimate_reciprocal_coords(coords, self.map.image_shape, tth=self.tth, chi=self.chi)
    


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
            raise OSError(f"Specified path does not exist:\n{filedir}{filename}")
        
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
                            Either specifiy file type or this filetype is not supported.''')
        
        if phase_name is not None:
            phase.name = phase_name
        
        if self.energy is not None:
            phase.energy = self.energy
        
        self.add_phase(phase)
    

    def clear_phases(self):
        self.phases = {}

    
    def update_phases(self):
        if (self.h5 is not None) and (len(self.phases) > 0):
            with h5py.File(self.h5, 'a') as f:
                phase_grp = f['xrdmap'].require_group('phase_list')

                # Delete any no longer included phases
                for phase in phase_grp.keys():
                    if phase not in self.phases.keys():
                        del(phase_grp[phase])

                # Save any new phases
                for phase in self.phases.values():
                    phase.save_to_h5(phase_grp)

                #dt = h5py.string_dtype(encoding='utf-8')
                #phase_names = np.char.encode(np.array(list(self.phases.keys())),
                #                             encoding='utf-8', errors=None)
                #f.require_dataset('phase_names2', data=phase_names,
                #                  dtype=dt, shape=(len(self.phases),))
                # phase_names = np.char.decode([string for string in phase_dataset[:]])


    def select_phases(self, remove_less_than=-1,
                      image=None, tth_num=4096,
                      unit='2th_deg', ignore_less=1,
                      save_to_h5=True):
        # Plot phase_selector

        if image is None:
            if self.map.title == 'calibrated_images':
                image = self.map._processed_images_composite
            else:
                image = self.map.composite_image
            

        tth, xrd = self.integrate_1d(image=image, tth_num=tth_num, unit=unit)
        # Add background subraction??

        phase_vals = phase_selector(xrd, list(self.phases.values()), tth, ignore_less=ignore_less)

        old_phases = list(self.phases.keys())
        for phase in old_phases:
            if phase_vals[phase] <= remove_less_than:
                self.remove_phase(phase)
        
        # Write phases to disk
        if save_to_h5:
            self.update_phases()
    
    ######################################################
    ### Blob, Ring, Peak and Spot Search and Selection ###
    ######################################################

    def find_spots(self, threshold=None, multiplier=10, sigma=3, radius=5):

        # Estimate remaining map noise to determine peak significance
        self.map_noise = estimate_map_noise(self.map, sample_number=200)

        # Search each image for significant spots
        spot_list, mask_list = find_spots(self.map,
                                          bkg_noise=self.map_noise,
                                          threshold=threshold,
                                          multiplier=multiplier,
                                          sigma=sigma)

        # Initial characterization of each spot
        stat_list = find_spot_stats(self.map,
                                    spot_list,
                                    tth=self.tth,
                                    chi=self.chi,
                                    radius=radius)

        # Convert spot stats into dict, then pandas dataframe
        stat_df = make_stat_df(stat_list, self.map.map_shape)

        # .spots attribute will be the basis will be treated similarly to .map
        # Most subsequent analysis will be built here
        # Consider wrapping it in a class like ImageMap or Phase
        self.spots = stat_df
        self.map.masks = np.asarray(mask_list).reshape(*self.map.map_shape,
                                                       *self.map.calibrated_shape)
        # Not sure about this one...
        #self.map.blurred_images = np.asarray(thresh_list).reshape(*self.map.map_shape,
        #                                                          *self.map.calibrated_shape)

        # Save spots to h5
        if self.h5 is not None:
            # Save spots to h5
            self.spots.to_hdf(self.h5, 'xrdmap/reflections/spots', format='table')

            # Save masks to h5
            self.map.save_images(images=self.map.masks,
                                 title='_masks',
                                 units='bool', 
                                 labels=['x_ind',
                                         'y_ind',
                                         'chi_ind',
                                         'tth_ind'],
                                 extra_attrs={'map_noise' : self.map_noise,
                                              'sigma' : sigma,
                                              'multiplier' : multiplier,
                                              'window_radius' : radius})


    def fit_spots(self, PeakModel, max_dist=0.75):

        # Find spots in self or from h5
        if not hasattr(self, 'spots'):
            print('No reflection spots found...')
            if self.h5 is not None:
                with h5py.File(self.h5, 'r') as f:
                    if 'reflections' in f['xrdmap'].keys():
                        print('Loading reflection spots from h5...', end='', flush=True)
                        spots = pd.read_hdf(h5_path, key='xrdmap/reflections/spots')
                        self.spots = spots
                        print('done!')
                    else:
                        raise IOError('XRDMap does not have any reflection spots! Please find spots first.')
            else:
                raise IOError('XRDMap does not have any reflection spots! Please find spots first.')

        # Generate the base list of spots from the refined guess parameters
        #spot_list = remake_spot_list(self.spots, self.map.map_shape)

        # Generate list of x, y, I, and spot indices for each blob/spots fits
        spot_fit_info_list = prepare_fit_spots(self, max_dist=max_dist)
        
        # Fits spots and adds the fit results to the spots dataframe
        fit_spots(self, spot_fit_info_list, PeakModel)

        # Save spots to h5
        if self.h5 is not None:
            #with h5py.File(self.h5, 'a') as f:
            #    # Check for and remove existing spots...
            #    if 'spots' in f['xrdmap/reflections']:
            #        del f['xrdmap/reflections/spots']
            self.spots.to_hdf(self.h5, 'xrdmap/reflections/spots', format='table')



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

    def interactive_image_map(self, display_map=None, display_title=None):
        # Map current images for dynamic exploration of dataset
        raise NotImplementedError()
    
    def interactive_integration_map(self, display_map=None, display_title=None):
        # Map integrated patterns for dynamic exploration of dataset
        # May throw an error if data has not yet been integrated
        raise NotImplementedError()
    
    def map_value(self, data, cmap='Viridis'):
        # Simple base mapping function for analyzed values.
        # Will expand to map phase assignment, phase integrated intensity, etc.
        raise NotImplementedError()
    
    # Dual plotting a map with a representation of the full data would be very interesting
    # Something like the full strain tensor which updates over each pixel
    # Or similar for dynamically updating pole figures