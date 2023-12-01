import numpy as np
import os
import h5py
import pyFAI
from pyFAI.io import ponifile
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
import skimage.io as io
from skimage.restoration import rolling_ball
from tqdm.auto import tqdm
import time as ttime
import matplotlib.pyplot as plt
from collections import OrderedDict


class XRDMap():
    '''
    Main class object for sXRD map.
    Inherits nothing!
    Multiple iteratations of image processing across full map cannot be saved in memory...
    '''

    def __init__(self, scanid=None, wd=None, filename=None, h5_file=None,
                 image_map=None, map_title=None,
                 energy=None, wavelength=None, poni_file=None,
                 tth_resolution=None, chi_resolution=None,
                 tth=None, chi=None,
                 beamline='5-ID (SRX)', facility='NSLS-II',
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
            if h5_file is None:
                self.h5 = f'{wd}{filename}.h5'
                if os.path.exists(f'{wd}{filename}.h5'):
                    pass
                else:
                    initialize_xrdmap_h5(self, self.h5) # Initialize base structure
            else: # specify specific h5 file
                self.h5 = f'{wd}{h5_file}'
        else:
            self.h5 = None


        # Load image map
        self.map = ImageMap(image_map, title=map_title, h5=self.h5)
        # Save to h5 if not already done so
        if self.h5 is not None:
            if not check_h5_current_images(self, self.h5):
                self.map.save_current_images(units='counts',
                                             labels=['x_ind',
                                                     'y_ind',
                                                     'img_y',
                                                     'img_x'])
                # Add method to avoid overwriting raw image data
            
        self.phases = {} # Place holder for potential phases
        if poni_file is not None:
            self.set_calibration(self, poni_file)
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
        return

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
    def from_image_stack(cls, filename, wd=None, **kwargs):
        # Load from image stack
        if wd is None:
            wd = '/home/xf05id1/current_user_data/'
        
        image_map = io.imread(f'{wd}{filename}')
        return cls(image_map=image_map, wd=wd, map_title='raw_images', **kwargs)


    @classmethod # Allows me to define and initiatie the class simultaneously
    def from_h5(cls, filename, wd=None):
        # Load from previously saved data, including all processed data...
        # Will need to figure out the save function first...
        raise NotImplementedError()
        # Make sure to load calibration information if saved
        h5 = h5py.File(f'{wd}{filename}', 'r') # Only read
        # extract some values
        # check for most current image_map
        h5.close()
        return cls()
    
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
        images = self.h5[f'/xrdmap/image_data/{image_dataset}'][:]
        self.map.images = images


    def save_h5(self):
        raise NotImplementedError()
        # Make sure to save calibration (tth, chi, etc..)
        # With image compression!!!!
        save_xrd_h5(self)


    def save_current_images(self):
        raise NotImplementedError()
        # Save current images to an h5 file

        dset = self.h5['/xrdmap/image_data'].create_dataset(
                                            self.map.title,
                                            data=self.map.images)
        dset.attrs['labels'] = ['x_ind', 'y_ind', 'img_y', 'img_x']
        dset.attrs['units'] = 'counts'
        dset.attrs['dtype'] = np.uint16
        dset.attrs['time_stamp'] = ttime.ctime()


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
                curr_grp = f[f'/xrdmap'].require_group('recipriocal_positions')
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
                         title='calibrated_images', unit='2th_deg'):
        if poni_file is not None:
            self.set_calibration(poni_file, filedir=filedir)
        if not hasattr(self, 'ai'):
            raise RuntimeError("Images cannot be calibrated without any calibration files!")

        tth_resolution = 0.02
        chi_resolution = 0.05
        if hasattr(self, 'tth_resolution'):
            tth_resolution = self.tth_resolution
        if hasattr(self, 'chi_resolution'):
            chi_resolution = self.chi_resolution
        
        out = self.map.calibrate_images(self.ai, title=title,
                                        tth_resolution=tth_resolution,
                                        chi_resolution=chi_resolution,
                                        unit=unit)
        
        self.tth = out[0]
        self.chi = out[1]
        self.extent = out[2]
        self.calibrated_shape = out[3]
        self.tth_resolution = out[4]
        self.chi_resolution = out[5]
        


    def integrate_1d(self, image=None, tth_num=4096,
                     unit='2th_deg'):
        # Intented for one-off temporary results
        if image is None:
            image = self.map.composite_image
        
        if (self.map.title == 'calibrated_images'
            and image is None):
            # Assumes the image input should be correct...
            raise RuntimeError("You are trying to clibrate already clibrated images!")
        
        return self.ai.integrate1d_ng(image, tth_num,
                                      unit=unit)
    

    def integrate_2d(self, image, tth_num, chi_num, unit='2th_deg'):
        # Intented for one-off temporary results
        if image is None:
            image = self.map.composite_image
        
        if (self.map.title == 'calibrated_images'
            and image is None):
            # Assumes the image input should be correct...
            raise RuntimeError("You are trying to clibrate already clibrated images!")
       
        return self.ai.integrate2d_ng(image, tth_num, chi_num,
                                        unit=unit)
    

    def estimate_img_coords(self, coords):
        return estimate_img_coords(coords, self.map.image_shape, tth=self.tth, chi=self.chi)


    def estimate_recipricol_coords(self, coords):
        return estimate_recipricol_coords(coords, self.map.image_shape, tth=self.tth, chi=self.chi)
    


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
        elif filename[-2] in ['.D']:
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

    
    # Only saves phase names
    # TODO: Replace with values to reconstruct phase objects...
    def save_phases(self):
        if (self.h5 is not None) and (len(self.phases) > 0):
            with h5py.File(self.h5, 'a') as f:
                phase_grp = f['/xrdmap'].require_group('phase_list')
                dt = h5py.string_dtype(encoding='utf-8')
                phase_names = np.char.encode(np.array(list(self.phases.keys())),
                                             encoding='utf-8', errors=None)
                f.require_dataset('phase_names2', data=phase_names,
                                  dtype=dt, shape=(len(self.phases),))
                # phase_names = np.char.decode([string for string in phase_dataset[:]])


    def select_phases(self, remove_less_than=-1,
                      image=None, tth_num=4096,
                      unit='2th_deg', ignore_less=1):
        # Plot phase_selector

        tth, xrd = self.integrate_1d(image=image, tth_num=tth_num, unit=unit)
        # Add background subraction??

        phase_vals = phase_selector(xrd, list(self.phases.values()), tth, ignore_less=ignore_less)

        old_phases = list(self.phases.keys())
        for phase in old_phases:
            if phase_vals[phase] <= remove_less_than:
                self.remove_phase(phase)
        
        # Write phases to disk
        self.save_phases()
    
    ######################################################
    ### Blob, Ring, Peak and Spot Search and Selection ###
    ######################################################

    def find_significant_regions():
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



class ImageMap:
    # This could be useful for plotting
    # This may be frowned upon because I am making too many classes...
    # This class is only intended for direct image manipulation
    # Analysis and interpretation of the meaning are reserved for the XRDMap class

    #def __new__(cls, input_array, dtype=None, title=None):
    #    if dtype is not None:
    #        obj = np.asarray(input_array, dtype=dtype).view(cls)
    #    else:
    #        obj = np.asarray(input_array).view(cls)
    #    #obj = np.asarray(input_array)
    #    if dtype is not None:
    #        obj.dtype = dtype
    #    obj.title = title
    #    return obj


    def __init__(self, input_array, dtype=None, title=None, h5=None):
        self.images = np.asarray(input_array)
        if dtype is None:
            dtype = self.images.dtype
        self.title = title
        self._dtype = dtype
        
        # Share h5, so either can write processed data to file. Will default to XRDMap
        # This is probably not very pythonic...
        self.h5 = h5

        #print('Defining new attributes')
        # Some redundant attributes, but maybe useful
        self.shape = self.images.shape
        self.num_images = np.multiply(*self.images.shape[:2])
        self.map_shape = self.shape[:2]
        self.image_shape = self.shape[2:]


    def __str__(self):
        return
    
    def __repr__(self):
        return


    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        # Used to update the main dtype. Should add a memory usage estimate as well
        self.images = self.images.astype(dtype) # This may have to copy the array, which could be memory intensive
        self._dtype = dtype

    
    def projection_factory(property_name, function, axes):
        property_name = f'_{property_name}' # This line is crucial! 
        def get_projection(self):
            if hasattr(self, property_name):
                return getattr(self, property_name)
            else:
                # This will have some precision issues depending on the specified dtype
                setattr(self, property_name, function(self.images, axis=axes).astype(self.dtype))
                return getattr(self, property_name)
        
        return property(get_projection)


    min_map = projection_factory('min_map', np.min, (2, 3))
    min_image = projection_factory('min_image', np.min, (0, 1))

    max_map = projection_factory('max_map', np.max, (2, 3))
    max_image = projection_factory('max_image', np.max, (0, 1))

    med_map = projection_factory('med_map', np.median, (2, 3))
    med_image = projection_factory('med_image', np.median, (0, 1))

    # Will not be accurate at default dtype of np.uint16. Consider upconversion?
    mean_map = projection_factory('mean_map', np.mean, (2, 3))
    mean_image = projection_factory('mean_image', np.mean, (0, 1))

    # Will probably cause overflow errors
    sum_map = projection_factory('sum_map', np.sum, (2, 3))
    sum_image = projection_factory('sum_image', np.sum, (0, 1))


    # Adaptively saves for whatever the current processing stage
    # Add other methods, or rename to something more intuitive
    # I should save this image to the h5...
    @property
    def composite_image(self):
        if hasattr(self, f'_{self.title}_composite'):
            return getattr(self, f'_{self.title}_composite')
        else:
            setattr(self, f'_{self.title}_composite',
                    self.max_image - self.min_image)
            
            # Set the generic value to this as well
            self._composite_image = getattr(self, f'_{self.title}_composite')

            # Save image to h5
            self.save_single_image(self._composite_image, f'_{self.title}_composite')

            # Finally return the requested attribute
            return getattr(self, f'_{self.title}_composite')
    

    # Function to dump accummulated processed images and maps
    # Not sure if will be needed between processing the full map
    def reset_attributes(self):
        old_attr = list(self.__dict__.keys())
        for attr in old_attr:
            if (attr not in ['images',
                             '_dtype',
                             'title',
                            'shape',
                            'num_images',
                            'map_shape',
                            'image_shape', 
                            'h5',
                            'dark_field_method',
                            'flat_field_method']# Preserve certain attributes between processing
                or attr[-10:] == '_composite'): 
                delattr(self, attr)
    

    # Change background to dark-field.
    # Will need to change the estimate
    def estimate_dark_field(self, method='min'):
        method = str(method).lower()
        if method in ['min', 'minimum']:
            print('Estimating dark-field with minimum method.')
            self.dark_field = self.min_image # does not account for changing electronic noise
            self.dark_field_method = 'minimum'
        else:
            raise NotImplementedError("Method input not implemented!")
    

    def correct_dark_field(self, dark_field=None, title=None):
        # TODO: Save correction to h5, with overwrite
        if dark_field is not None:
            self.dark_field = dark_field
        elif hasattr(self, 'dark_field'):
            dark_field = self.dark_field
        else:
            raise RuntimeError("Please specify background to subtract.")

        if self.dark_field.dtype > self.dtype:
            self.dtype = self.dark_field.dtype
        
        if check_precision(self.dtype)[0].min >= 0:
            # Not sure how to decide which int precision
            # Both Merlin and Dexela output uint16, so this should be sufficient for now
            # Trying to save memory if possible
            if np.max(self.dark_field) > np.min(self.min_image):
                self.dtype = np.int32
        
        print('Correcting dark-field...', end='', flush=True)
        self.images -= self.dark_field
        print('done!')


    def estimate_flat_field(self, method='med', **kwargs):
        method = str(method).lower()
        # Add method to use the scalar information too.
        if method in ['med', 'median']:
            print('Estimating flat-field from median values.')
            self.flat_field = np.multiply.outer(self.med_map, self.med_image)
            self.flat_field_method = 'median'
            
        elif method in ['ball', 'rolling ball', 'rolling_ball']:
            print('Estimating flat-field with rolling ball method.')
            self.flat_field = rolling_ball(self.images, **kwargs)
            self.flat_field_method = 'rolling ball'

        elif method in ['spline', 'spline fit', 'spline_fit']:
            raise NotImplementedError()
            self.flat_field_method = 'spline'
        else:
            raise NotImplementedError("Method input not implemented!")
    

    def correct_flat_field(self, flat_field=None, title=None):
        # TODO: Save correction to h5, with overwrite
        if flat_field is not None:
            self.flat_field = flat_field
        elif hasattr(self, 'flat_field'):
            flat_field = self.flat_field
        else:
            raise RuntimeError("Please specify background to subtract.")

        #if self.flat_field.dtype > self.dtype:
        #    self.dtype = self.flat_field.dtype

        self.dtype = np.float32
        print('Correcting flat-field...', end='', flush=True)
        self.images /= self.flat_field
        print('done!')
    

    def correct_images(self, dark_method='min', flat_method='med', **kwargs):
        # Convenience functions that will help keep track of all corrections
        self.estimate_dark_field(dark_method, **kwargs)
        self.correct_dark_field()

        self.estimate_flat_field(flat_method, **kwargs)
        self.correct_flat_field()

        print('Image corrections complete!')
        self.title = 'processed_images'

        print('Compressing and writing processed images to disk...', end='', flush=True)
        self.save_current_images(units='a.u.',
                    labels=['x_ind',
                            'y_ind',
                            'img_y',
                            'img_x'],
                    extra_attrs={'dark_field_method':self.dark_field_method,
                                'flat_field_method':self.flat_field_method})
        print('done!')

        self.reset_attributes()

        # Save correction methods
        if self.h5 is not None:
            with h5py.File(self.h5, 'a') as f:
                dset = f[f'xrdmap/image_data/{self.title}']
                dset.attrs['dark_field_method'] = self.dark_field_method
                #dset.attrs['dark_field'] = self.dark_field
                dset.attrs['flat_field_method'] = self.flat_field_method
    

    # Should move to geometry.py
    def calibrate_images(self, ai, title='calibrated_images',
                         unit='2th_deg',
                         tth_resolution=0.02, chi_resolution=0.05):
        
        # Check for background subtraction
        if self.title != 'processed_images':
            print('Warning: Calibrating unprocessed images. Proceeding without image corrections.')
            # Maybe I should add a list of processing step performed to keep track of everything...

        # Ensure the processed composite image exists - useful for phase selection
        if self.title == 'processed_images' and not hasattr(self, 'processed_images_composite'):
            print('Composite of processed images not saved. Creating composite.')
            self.composite_image;

        # Clear old values
        self.reset_attributes()

        # Set units for metadata
        self.calib_unit = unit

        # These should be properties...
        self.tth_resolution = tth_resolution # Degrees
        self.chi_resolution = chi_resolution # Degrees

        # Surely there is better way to find the extent without a full calibration
        # It's fast some maybe doesn't matter
        _, tth, chi = ai.integrate2d_ng(self.images.reshape(
                                self.num_images, 
                                *self.image_shape)[0],
                                100, 100, unit=self.calib_unit)
        
        # Interpolation bounds should be limited by the intrument resolution AND the original image size
        self.tth_num = int(np.abs(np.max(tth) - np.min(tth))
                           // self.tth_resolution)
        self.chi_num = int(np.abs(np.max(chi) - np.min(chi))
                           // self.chi_resolution)

        calibrated_map = np.zeros((self.num_images,
                                   self.chi_num, 
                                   self.tth_num), 
                                   dtype=(self.dtype))
        
        print('Calibrating images...', end='', flush=True)
        # TODO: Parallelize this
        for i, pixel in tqdm(enumerate(self.images.reshape(
                                       self.num_images,
                                       *self.image_shape)),
                                       total=self.num_images):
            
            res, tth, chi = ai.integrate2d_ng(pixel,
                                          self.tth_num,
                                          self.chi_num,
                                          unit=self.calib_unit)
            calibrated_map[i] = res

        calibrated_map = calibrated_map.reshape(*self.map_shape,
                                                self.chi_num, 
                                                self.tth_num)
        # Consider rescaling and downgrading data type to save memory...
        self.images = calibrated_map
        self.tth = tth
        self.chi = chi
        self.extent = [self.tth[0], self.tth[-1],
                       self.chi[0], self.chi[-1]]

        print('done!')
        
        if title is not None:
            self.title = title

        print('''Compressing and writing calibrated images to disk.\nThis may take awhile...''', end='', flush=True)
        self.save_current_images(units=self.calib_unit,
                                 labels=['x_ind',
                                         'y_ind',
                                         'chi_ind',
                                         'tth_ind'],
                                extra_attrs={'tth_resolution' : self.tth_resolution,
                                             'chi_resolution' : self.chi_resolution,
                                             'extent' : self.extent})
        
        # Add calibration positions dataset
        if self.h5 is not None:
            with h5py.File(self.h5, 'a') as f:
                # This group may already exist if poni file was already initialized
                curr_grp = f[f'/xrdmap'].require_group('recipriocal_positions')
                #curr_grp.create_dataset('name', data=['tth_pos', 'chi_pos'])
                dset = curr_grp.require_dataset('pos',
                            data=np.stack(np.meshgrid(self.tth, self.chi)),
                            dtype=self.dtype, shape=(2, self.tth_num, self.chi_num))
                dset.attrs['labels'] = ['tth', 'chi']
                dset.attrs['comments'] = """'tth' is the two_theta scattering angle\n
                                        'chi' is the azimuthal angle"""
                dset.attrs['units'] = self.calib_unit #'° [deg.]'
                dset.attrs['dtype'] = str(self.dtype)
                dset.attrs['time_stamp'] = ttime.ctime()
        print('done!')

        
        # Direct set to avoid resetting the map images again
        self._dtype = self.images.dtype

        # Pass these values up the line
        return self.tth, self.chi, self.extent, (self.chi_num, self.tth_num), self.tth_resolution, self.chi_resolution, 

        
    def interactive_map(self):
        raise NotImplementedError()
    

    def plot_image(self, image=None, vmin=None, aspect='auto', **kwargs):
        rand_img = False
        if image is None:
            i = np.random.randint(self.map_shape[0])
            j = np.random.randint(self.map_shape[1])
            image = self.images[i, j]
            rand_img = True

        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=200)

        if hasattr(self, 'extent'):
            if vmin == None:
                vmin = 0
            im = ax.imshow(image, extent=self.extent, vmin=vmin, aspect=aspect, **kwargs)
            ax.set_xlabel('Scattering Angle, 2θ [°]') # Assumes degrees. Need to change...
            ax.set_ylabel('Azimuthal Angle, χ [°]')
        else:
            im = ax.imshow(image, aspect=aspect)
            ax.set_xlabel('X index')
            ax.set_ylabel('Y index')
        
        if rand_img:
            ax.set_title(f'Row = {i}, Col = {j}')
        else:
            ax.set_title('Input Image')
        fig.colorbar(im, ax=ax)
        plt.show()

        
        
    
    def disk_size(self):
        # Return current map size which should be most of the memory usage
        # Helps to estimate file size too
        disk_size = self.images.size
        units = 'B'
        if disk_size > 2**10:
            disk_size = disk_size / 2**10
            units = 'KB'
            if disk_size > 2**10:
                disk_size = disk_size / 2**10
                units = 'MB'
                if disk_size > 2**10:
                    disk_size = disk_size / 2**10
                    units = 'GB'
        
        print(f'Diffraction map size is {disk_size:.3f} {units}.')
        #return self.images.size / 2**10 / 2**10 / 2**10 # in GB
    

    def save_current_images(self, units='', labels='',
                            compression='gzip', compression_opts=8,
                            mode='a', extra_attrs=None):
        
        if self.h5 is not None: # Should disable working with h5 if no information is provided
            with h5py.File(self.h5, mode) as f:
                dset = f['/xrdmap/image_data'].require_dataset(
                                self.title,
                                data=self.images,
                                shape=self.images.shape,
                                dtype=self.dtype,
                                compression=compression,
                                compression_opts=compression_opts)
                dset.attrs['labels'] = labels
                dset.attrs['units'] = units
                dset.attrs['dtype'] = str(self.dtype)
                dset.attrs['time_stamp'] = ttime.ctime()

                # Add non-standard extra metadata attributes
                if extra_attrs is not None:
                    for key, value in extra_attrs.items():
                        dset.attrs[key] = value


    def save_single_image(self, image, title, units='', labels='', mode='a', extra_attrs=None):
        if self.h5 is not None: # Should disable working with h5 if no information is provided
            with h5py.File(self.h5, mode) as f:
                dset = f['/xrdmap/image_data'].require_dataset(
                                title,
                                data=image,
                                shape=image.shape,
                                dtype=image.dtype)
                dset.attrs['labels'] = labels
                dset.attrs['units'] = units
                dset.attrs['dtype'] = str(image.dtype)
                dset.attrs['time_stamp'] = ttime.ctime()

                # Add non-standard extra metadata attributes
                if extra_attrs is not None:
                    for key, value in extra_attrs.items():
                        dset.attrs[key] = value
    