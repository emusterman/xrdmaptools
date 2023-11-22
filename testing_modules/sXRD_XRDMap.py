import numpy as np
import os
import h5py
import pyFAI
import skimage.io as io
from skimage.restoration import rolling_ball
from tqdm.auto import tqdm
import time as ttime


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
                 beamline=None, facility=None,
                 extra_metadata=None):
        
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
        if h5_file is None:
            if os.path.exists(f'{wd}{filename}.h5'):
                self._h5 = h5py.File(f'{wd}{filename}.h5', 'a')
            else:
                self._h5 = h5py.File(f'{wd}{filename}.h5', 'w-')
                # Should not fail, but better to be careful
                # If not h5 file, create one
                initialize_xrdmap_h5(self) # Initialize base structure
        else: # specify specific h5 file
            self._h5 = h5py.File(f'{wd}{h5_file}', 'a')
        #self._h5.close()

        # Load image map
        self.map = ImageMap(image_map, title=map_title, h5=self._h5)
        # Save to h5 if not already done so
        if self._h5 is not None:
            if self.map.title not in self._h5['/xrdmap/image_data']:
                self.map.save_current_images(units='counts',
                                             labels=['x_ind',
                                                     'y_ind',
                                                     'img_x',
                                                     'img_y'])
            
        self.phases = {} # Place holder for potential phases
        if poni_file is not None:
            self.set_calibration(self, poni_file)
        else:
            self.ai = None # Place holder for calibration
        
        if tth_resolution is not None:
            self.tth_resolution = tth_resolution
        if chi_resolution is not None:
            self.chi_resolution = chi_resolution
    

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
        images = self._h5[f'/xrdmap/image_data/{image_dataset}'][:]
        self.map.images = images


    def save_h5(self):
        raise NotImplementedError()
        # Make sure to save calibration (tth, chi, etc..)
        # With image compression!!!!
        save_xrd_h5(self)


    def save_current_images(self):
        # Save current images to an h5 file

        dset = self._h5['/xrdmap/image_data'].create_dataset(
                                            self.map.title,
                                            data=self.map.images)
        dset.attrs['labels'] = ['x_ind', 'y_ind', 'img_x', 'img_y']
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
        
        if not os.path.exists(f'{filedir}{poni_file}'):
            raise IOError(f"{filedir}{poni_file} does not exist")

        if poni_file[-4:] != 'poni':
            raise RuntimeError("Please provide a .poni file.")

        self.ai = pyFAI.load(f'{filedir}{poni_file}')
        if self.energy is not None:
            self.ai.energy = self.energy # Allows calibrations acquired at any energy
        else:
            print('Energy has not been defined. Defualting to .poni file value.')
            self.energy = self.ai.energy


    # Move this to a function in geometry??
    def calibrate_images(self, poni_file=None, filedir=None,
                         title='calibrated_images'):
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
                                        chi_resolution=chi_resolution)
        
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
        
        if filename[-3:] == 'cif':
            phase = Phase.fromCIF(f'{filedir}{filename}')
        elif filename[-3:] == 'txt':
            raise NotImplementedError()
        elif filename[-1] in ['D']:
            raise NotImplementedError()
        else:
            raise TypeError(f'''Unsure how to read {filename}. 
                            Either specifiy file type or this filetype is not supported.''')
        
        if phase_name is not None:
            phase.name = phase_name
        
        self.add_phase(phase)
    

    def select_phases(self, remove_less_than=-1,
                      image=None, tth_num=4096,
                      unit='2th_deg', ignore_less=1):
        # Plot phase_selector
        # Is there a way to automatically deselect phases from this??

        tth, xrd = self.integrate_1d(image=image, tth_num=tth_num, unit=unit)
        # Add background subraction??

        phase_vals = phase_selector(xrd, list(self.phases.values()), tth, ignore_less=ignore_less)

        old_phases = list(self.phases.keys())
        for phase in old_phases:
            if phase_vals[phase] <= remove_less_than:
                self.remove_phase(phase)


    def clear_phases(self):
        self.phases = {}

    
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
        if h5 is not None:
            self._h5 = h5

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
            self._composite_image = self.max_image - self.min_image
            return getattr(self, f'_{self.title}_composite')
    
    #@property
    #def composite_image(self):
    #    if hasattr(self, '_composite_image'):
    #        return self._composite_image
    #    else:
    #        self._composite_image = self.max_image - self.min_image
    #        return self._composite_image

    # Function to dump accummulated processed images and maps
    # Not sure if will be needed between processing the full map
    def reset_attributes(self):
        old_attr = list(self.__dict__.keys())
        for attr in old_attr:
            if (attr not in ['images', '_dtype', 'title',
                            'shape', 'num_images',
                            'map_shape', 'image_shape']# Preserve certain attributes between processing
                or attr[-10:] == '_composite'): 
                delattr(self, attr)
    
    # Change background to dark-field.
    # Will need to change the estimate
    def estimate_dark_field(self, method):
        method = str(method).lower()
        if method in ['min', 'minimum']:
            self.dark_field = self.min_image # does not account for changing electronic noise
        else:
            raise NotImplementedError("Method input not implemented!")
    
    def correct_dark_field(self, dark_field=None, title=None):
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
            
        self.images -= self.dark_field
        self.reset_attributes()


    def estimate_flat_field(self, method=None, **kwargs):
        method = str(method).lower()
        # Add method to use the scalar information too.
        if method in ['med', 'median']:
            self.flat_field = self.med_image # simple, but not very accurate
        elif method in ['ball', 'rolling ball', 'rolling_ball']:
            self.flat_field = rolling_ball(self.images, **kwargs)
        elif method in ['spline', 'spline fit', 'spline_fit']:
            raise NotImplementedError()
        else:
            raise NotImplementedError("Method input not implemented!")
    
    def correct_flat_field(self, flat_field=None, title=None):
        if flat_field is not None:
            self.flat_field = flat_field
        elif hasattr(self, 'flat_field'):
            flat_field = self.flat_field
        else:
            raise RuntimeError("Please specify background to subtract.")

        if self.flat_field.dtype > self.dtype:
            self.dtype = self.flat_field.dtype
        
        if check_precision(self.dtype)[0].min >= 0:
            # Not sure how to decide which int precision
            # Both Merlin and Dexela output uint16, so this should be sufficient for now
            # Trying to save memory if possible
            if np.max(self.flat_field) > np.min(self.min_image):
                self.dtype = np.float32
                # This will ballon memory
                # Consider rescaling into a new data type range
            
        self.images /= self.dark_field
        self.reset_attributes()
    
    def correct_images(self, dark_method=None, flat_method=None, **kwargs):
        raise NotImplementedError()
        # Convenience functions that will help keep track of all corrections
        self.estimate_dark_field(dark_method, **kwargs)
        self.correct_dark_field()

        self.estimate_flat_field(flat_method, **kwargs)
        self.correct_flat_field()

        self.title = 'processed_images'
        # Add label and units!
        self.save_current_images()
        # Save correction details. NOT the full correction, unless small



    '''# Removing background functions in favor of dark- and flat-field corrections
    def estimate_background(self, method, **kwargs):
        method = str(method).lower()

        if method in ['med', 'median']:
            self.background = self.med_image # simple, but not very accurate
        elif method in ['ball', 'rolling ball', 'rolling_ball']:
            self.background = rolling_ball(self.images, **kwargs)
        elif method in ['spline', 'spline fit', 'spline_fit']:
            raise NotImplementedError()
        else:
            raise NotImplementedError("Method input not implemented!")
    
    def subtract_background(self, background=None, title=None):
        # Simple subtraction

        if background is not None:
            self.background = background

        elif hasattr(self, 'background'):
            background = self.background
        else:
            raise RuntimeError("Please specify background to subtract.")
        
        if self._h5 is not None:
            if self.title not in self._h5['/xrdmap/image_data']:
                self.save_current_images(units='counts',
                                         labels=['x_ind', 'y_ind', 'img_x', 'img_y'])

        if title is None:
            title = 'processed_images'
        
        if self.background.dtype > self.dtype:
            self.dtype = self.background.dtype
        
        if check_precision(self.dtype)[0].min >= 0:
            # Not sure how to decide which int precision
            # Both Merlin and Dexela output uint16, so this should be sufficient for now
            # Trying to save memory if possible
            if np.max(self.background) > np.min(self.min_image):
                self.dtype = np.int32
            
        self.images -= background
        # print('Bacground subtracted!')
        self.reset_attributes()

        self.background = background # Keep background for back-conversion if necessary
        self.title = title # Update title to new iteration'''

    

    # Should move to geometry.py
    def calibrate_images(self, ai, title='calibrated_images',
                         tth_resolution=0.02, chi_resolution=0.05):
        
        # Check for background subtraction
        if self.title != 'processed_images':
            print('Warning: Calibrating unprocessed images. Proceeding without image corrections.')
            # Maybe I should add a list of processing step performed to keep track of everything...

        # Ensure the processed composite image exists - useful for phase selection
        if self.title == 'processed_images' and not hasattr(self, 'processed_images_composite'):
            self.composite_image;

        # Clear old values
        self.reset_attributes()

        # These should be properties...
        self.tth_resolution = tth_resolution # Degrees
        self.chi_resolution = chi_resolution # Degrees

        # Surely there is better way to find the extent without a full calibration
        # It's fast some maybe doesn't matter
        res = ai.integrate2d_ng(self.images.reshape(
                                self.num_images, 
                                *self.image_shape)[0],
                                100, 100, unit='2th_deg')
        self.tth = res[1]
        self.chi = res[2]
        self.tth_num = int(np.abs(np.max(self.tth) - np.min(self.tth))
                           // self.tth_resolution)
        self.chi_num = int(np.abs(np.max(self.chi) - np.min(self.chi))
                           // self.chi_resolution)
        # Interpolation bounds should be limited by the intrument resolution AND the original image size

        # There is a better way I could do this...
        res = ai.integrate2d_ng(self.images.reshape(
                                self.num_images, 
                                *self.image_shape)[0],
                                self.tth_num, self.chi_num, unit='2th_deg') 
        self.tth = res[1]
        self.chi = res[2]
        self.extent = [self.tth[0], self.tth[-1],
                       self.chi[0], self.chi[-1]]

        # Calibrate the full map. Can I parallelize this??
        calibrated_map = np.zeros((self.num_images,
                                   self.chi_num, 
                                   self.tth_num), 
                                   dtype=(self.dtype))
        
        # Consider the tqdm package??
        for i, pixel in tqdm(enumerate(self.images.reshape(
                                       self.num_images,
                                       *self.image_shape)),
                                       total=self.num_images):
            
            res, _, _ = ai.integrate2d_ng(pixel,
                                          self.tth_num,
                                          self.chi_num,
                                          unit='2th_deg')
            calibrated_map[i] = res

        calibrated_map = calibrated_map.reshape(*self.map_shape,
                                                self.chi_num, 
                                                self.tth_num)
        # Consider rescaling and downgrading data type to save memory...
        self.images = calibrated_map
        
        if title is not None:
            self.title = title
        
        self.save_current_images() # Add units and labels
        
        # Direct set to avoid resetting the map images again
        self._dtype = self.images.dtype

        # Pass these values up the line
        return self.tth, self.chi, self.extent, (self.tth_num, self.chi_num), self.tth_resolution, self.chi_resolution, 

        
    def interactive_map():
        raise NotImplementedError()
    
    def map_disk_size(self):
        # Return current map size which should be most of the memory usage
        # Helps to estimate file size too
        disk_size = self.images.size / 2**10 / 2**10 / 2**10
        print(f'Diffraction map size is {disk_size:.3f} GB.')
        return disk_size
    

    def save_current_images(self, units='', labels=''):
        dset = self._h5['/xrdmap/image_data'].require_dataset(
                        self.title,
                        data=self.images,
                        shape=self.images.shape,
                        dtype=self.dtype)
        dset.attrs['labels'] = labels
        dset.attrs['units'] = units
        dset.attrs['dtype'] = str(self.dtype)
        dset.attrs['time_stamp'] = ttime.ctime()