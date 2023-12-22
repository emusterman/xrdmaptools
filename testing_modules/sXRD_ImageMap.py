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


    def __init__(self, image_array, map_shape=None,
                 dtype=None, title=None, h5=None,
                 corrections=None):
        
        image_array = np.asarray(image_array)

        if map_shape is not None:
            self.images = image_array.reshape(tuple(map_shape))
        elif len(image_array.shape) == 3:
            print('WARNING: Input array given as 3D object. Assuming square map...')
            input_shape = image_array.shape
            map_side = np.sqrt(input_shape[0])
            if map_side % 1 != 0:
                raise RuntimeError("Input array not 4D, no shape provided, and not a square map.")
            new_shape = (int(map_side), int(map_side), *input_shape[1:])
            print(f'Assumed map shape is {new_shape[:2]} with images of {new_shape[-2:]}')
            self.images = image_array.reshape(new_shape)
        elif len(image_array.shape) == 4:
            self.images = image_array
        else:
            raise RuntimeError("Input data incorrect shape or unknown type. 4D array is preferred.")

        if dtype is None:
            dtype = self.images.dtype
        self._dtype = dtype
        
        if isinstance(corrections, dict):
            self.correction = corrections
        else:
            self.corrections = {
                'dark_field' : False,
                'flat_field' : False,
                'outliers' : False,
                'pixel_defects' : False,
                'pixel_distortions' : False,
                'polar_calibration' : False,
                'solid_angle' : False,
                'polarization' : False,
                'lorentz' : False,
                'scaler_intensity' : False,
                'background' : False
            }
        
        self.update_map_title(title=title)        
        
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
    
    ######################################
    ### Class Properties and Functions ###
    ######################################


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
                #return getattr(self, property_name)
            
                if hasattr(self, 'calibration_mask') and axes == (2, 3):
                    blank_map = np.zeros(self.map_shape)

                    for i, image in enumerate(self.images.reshape(self.num_images, *self.calibrated_shape)):
                        val = function(image[self.calibration_mask])
                        indices = np.unravel_index(i, self.map_shape)
                        blank_map[indices] = val
                    
                    setattr(self, property_name, blank_map)
                
                return getattr(self, property_name)

        def set_projection(self, value):
            setattr(self, property_name, value)

        def del_projection(self):
            delattr(self, property_name)
        
        return property(get_projection, set_projection, del_projection)


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
            self.save_images(self._composite_image,
                             f'_{self.title}_composite',
                             units=f'same as {self.title}',
                             labels=f'same as {self.title}')

            # Finally return the requested attribute
            return getattr(self, f'_{self.title}_composite')
        
    @composite_image.deleter
    def composite_image(self):
        delattr(self, '_composite_image')
    

    # Function to dump accummulated processed images and maps
    # Not sure if will be needed between processing the full map
    def reset_attributes(self):
        old_attr = list(self.__dict__.keys())
        #for attr in old_attr:
        #    if (attr not in ['images',
        #                     '_dtype',
        #                     'title',
        #                    'shape',
        #                    'num_images',
        #                    'map_shape',
        #                    'image_shape', 
        #                    'h5',
        #                    'dark_field_method',
        #                    'flat_field_method',
        #                    'calibration_mask']# Preserve certain attributes between processing
        #        and attr[-10:] != '_composite'): 
        #        delattr(self, attr)
       
        for attr in old_attr:
            if attr in ['_composite_image',
                        '_min_map', '_min_image',
                        '_max_map', '_max_image',
                        '_med_map', '_med_image',
                        '_sum_map', '_sum_image',
                        '_mean_map', '_mean_image',]:
                delattr(self, attr)


    def update_map_title(self, title=None):
        # The title will never be able to capture everything
        # This is intended to only capture major changes
        # Will be used to create new groups when saving to h5
        if hasattr(self, 'title'):
            old_title = self.title
        else:
            old_title = None

        if title is not None:
            self.title = title
    
        elif np.all(~np.array(list(self.corrections.values()))):
            self.title = 'raw_images' # This should not be needed...

        elif (self.corrections['dark_field']):
            self.title = 'processed_images'

            if (self.corrections['polar_calibration']):
                self.title = 'calibrated_images'

                if (self.corrections['background']):
                    self.title = 'final_images'
        else:
            # No update
            pass

        if old_title != self.title:
            self.reset_attributes()


    ########################################
    ### Image Corrections and Transforms ###
    ########################################

    ### Initial image corrections ###

    def correct_dark_field(self, dark_field=None):

        if dark_field is None:
            print('No dark-field correction.')
        else:
            self.dark_field = dark_field

            if self.dark_field.dtype > self.dtype:
                self.dtype = self.dark_field.dtype
            
            if check_precision(self.dtype)[0].min >= 0:
                # Not sure how to decide which int precision
                # Trying to save memory if possible
                if np.max(self.dark_field) > np.min(self.min_image):
                    self.dtype = np.int32
            
            print('Correcting dark-field...', end='', flush=True)
            self.images -= self.dark_field
            self.save_images(self.dark_field,
                            'dark_field',
                            units=str(self.dark_field.dtype), 
                            labels=['img_y','img_x'])
            self.corrections['dark_field'] = True
            self.update_map_title()
            print('done!')


    def correct_flat_field(self, flat_field=None):

        if flat_field is None:
            print('No flat-field correction.')
        else:
            self.flat_field = flat_field
            self.dtype = np.float32
            print('Correcting flat-field...', end='', flush=True)
            self.images /= self.flat_field
            self.save_images(self.flat_field,
                            'flat_field',
                            units=str(self.flat_field.dtype), 
                            labels=['img_y','img_x'])
            self.corrections['flat_field'] = True
            self.update_map_title()
            print('done!')


    def correct_outliers(self, size=2, tolerance=3, significance=None):
        
        print('Finding and correcting image outliers...', end='', flush=True)
        for image_ind in tqdm(range(self.num_images)):
            indices = np.unravel_index(image_ind, self.map_shape)
            fixed_image = find_outlier_pixels(self.images[indices],
                                              size=size,
                                              tolerance=tolerance,
                                              significance=significance)
            self.images[indices] = fixed_image
        self.corrections['outliers'] = True
        self.update_map_title()
        print('done!')


    def apply_defect_mask(self, mask):
        raise NotImplementedError()
        self.update_map_title()
        self.corrections['pixel_defects'] = True


    # Change background to dark-field.
    # Will need to change the estimate
    @deprecated
    def estimate_dark_field(self, method='min', dark_field=None):
        method = str(method).lower()
        if dark_field is None:
            if method in ['min', 'minimum']:
                print('Estimating dark-field with minimum method.')
                self.dark_field = self.min_image # does not account for changing electronic noise
                self.dark_field_method = 'minimum'
            
            elif method in ['none']:
                print('No dark field correction will be used.')
                self.dark_field = None
                self.dark_field_method = 'none'
            
            else:
                raise NotImplementedError("Method input not implemented!")
        else:
            print('User-specified dark-field.')
            self.dark_field = dark_field
            self.dark_field_method = 'custom'
    

    # This is not correct...more of a background subtraction
    @deprecated
    def estimate_flat_field(self, method='med', flat_field=None, **kwargs):
        method = str(method).lower()

        if flat_field is None:
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

            elif method in ['none']:
                print('No flat field correction will be used.')
                self.flat_field = None
                self.flat_field_method = 'none'
            else:
                raise NotImplementedError("Method input not implemented!")
        else:
            print('User-specified flat-field.')
            self.flat_field = flat_field
            self.flat_field_method = 'custom'
    
    @deprecated
    def correct_images(self, dark_method='min', flat_method='med', **kwargs):
        # Convenience functions that will help keep track of all corrections
        self.estimate_dark_field(method=dark_method, **kwargs)
        self.correct_dark_field()

        self.estimate_flat_field(method=flat_method, **kwargs)
        self.correct_flat_field()

        print('Image corrections complete!')
        self.title = 'processed_images'

        print('Compressing and writing processed images to disk...', end='', flush=True)
        self.save_images(units='a.u.',
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
    

    ### Pixel to angle transformation ###

    # Should move to geometry.py
    # Geometric calibration, polarization and Lorentz corrections
    def calibrate_images(self, ai, title=None,
                         unit='2th_deg',
                         tth_resolution=0.02, chi_resolution=0.05,
                         polarization_factor=0.9,
                         Lorentz_correction=True,
                         **kwargs):
        
        # Check the current state of the map
        self.update_map_title()
        
        if self.title == 'raw_images':
            print('Warning: Calibrating unprocessed images. Proceeding without image corrections.')
            _ = self.composite_image
            # Maybe I should add a list of processing step performed to keep track of everything...

        elif self.corrections['polar_calibration']:
            raise RuntimeError("""Cannot calibrate already calibrated images! 
                            \nRevert to processed images in order to recalibrate.""")
        
        elif self.title == 'processed_images':
            if not hasattr(self, '_processed_images_composite'):
                print('Composite of processed images not saved. Creating composite.')
                _ = self.composite_image
            else:
                pass
                
        else:
            print('Warning: Unknown image state. Proceeding, but be cautious of results.')

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
                                          unit=self.calib_unit,
                                          polarization_factor=polarization_factor, 
                                          **kwargs)
            
            if Lorentz_correction: # Yong was disappointed I did not have this already
                rad = np.radians(tth / 2)
                res /=  1 / (np.sin(rad) * np.sin(2 * rad))

            calibrated_map[i] = res

        calibrated_map = calibrated_map.reshape(*self.map_shape,
                                                self.chi_num, 
                                                self.tth_num)
        # Consider rescaling and downgrading data type to save memory...
        self.images = calibrated_map
        self.tth = tth
        self.chi = chi
        self.calibrated_shape = (self.chi_num, self.tth_num) # V x H
        self.extent = [self.tth[0], self.tth[-1],
                       self.chi[0], self.chi[-1]]

        print('done!')

        #print('''Compressing and writing calibrated images to disk.\nThis may take awhile...''')
        #self.save_images(units=self.calib_unit,
        #                         labels=['x_ind',
        #                                 'y_ind',
        #                                 'chi_ind',
        #                                 'tth_ind'])
        
        # Add calibration positions dataset
        print('Writing reciprocal positions...')
        if self.h5 is not None:
            with h5py.File(self.h5, 'a') as f:
                # This group may already exist if poni file was already initialized
                curr_grp = f[f'/xrdmap'].require_group('reciprocal_positions')
                curr_grp.attrs['extent'] = self.extent

                labels = ['tth_pos', 'chi_pos']
                comments = [''''tth', is the two theta scattering angle''',
                            ''''chi' is the azimuthal angle''']
                keys = ['tth', 'chi']
                data = [self.tth, self.chi]
                resolution = [self.tth_resolution, self.chi_resolution]

                for i, key in enumerate(keys):
                    if key in curr_grp.keys():
                        del curr_grp[key]
                    dset = curr_grp.require_dataset(key,
                                                    data=data[i],
                                                    dtype=data[i].dtype,
                                                    shape=data[i].shape)
                    dset.attrs['labels'] = labels[i]
                    dset.attrs['comments'] = comments[i]
                    dset.attrs['units'] = self.calib_unit #'° [deg.]'
                    dset.attrs['dtype'] = str(data[i].dtype)
                    dset.attrs['time_stamp'] = ttime.ctime()
                    dset.attrs[f'{key}_resolution'] = resolution[i]
        print('done!')

        # Acquire mask for useless pixels for subsequent analysis
        print('Acquring and writing calibration mask...')
        self.calibration_mask = self.get_calibration_mask(ai)
        self.save_images(self.calibration_mask,
                         'calibration_mask',
                         units=str(self.calibration_mask.dtype), 
                         labels=['chi_ind','tth_ind'])
        
        # internal record keeping
        self.corrections['polar_calibration'] = True
        if Lorentz_correction:
            self.corrections['lorentz'] = True
        if polarization_factor is not None:
            self.corrections['polarization'] = True
        if 'correctSolidAngle' in kwargs.keys():
            self.corrections['solid_angle'] = kwargs['correctSolidAngle']
        else:
            self.corrections['solid_angle'] = True
        print('done!')
        
        # Direct set to avoid resetting the map images again
        self._dtype = self.images.dtype

        # Update title
        self.update_map_title(title=title)

        # Pass these values up the line to the xrdmap
        return self.tth, self.chi, self.extent, self.calibrated_shape, self.tth_resolution, self.chi_resolution
    

    def get_calibration_mask(self, ai, tth_num=None, chi_num=None, units='2th_deg'):

        if tth_num is None:
            tth_num = self.tth_num
        if chi_num is None:
            chi_num = self.chi_num
        if units is None:
            units = self.calib_unit

        dummy_image = 100 * np.ones(self.image_shape)

        image, _, _ = ai.integrate2d_ng(dummy_image, tth_num, chi_num, unit=units)

        calibration_mask = (image != 0)

        return calibration_mask
    

    ### Final image corrections ###

    def normalize_scaler(self, scaler_arr=None):
        
        if scaler_arr is None:
            print('No scaler array given. Approximating with image medians...')
            scaler_arr = self.med_map

        else:
            scaler_arr = np.asarray(scaler_arr)
            if scaler_arr.shape != self.map_shape:
                raise TypeError(f'''Scaler array of shape {scaler_arr.shape} does not 
                                match the map shape of {self.map_shape}!''')

        print('Normalize image scalers...', end='', flush=True)
        self.images /= scaler_arr.reshape(*self.map_shape, 1, 1)
        self.scaler_map = scaler_arr # Do not save to h5, since scalers should be recorded...
        self.corrections['scaler_intensity'] = True
        self.update_map_title()
        print('done!')


    def estimate_background(self, method='med', background=None, **kwargs):
        method = str(method).lower()

        if flat_field is None:
            # Add method to use the scalar information too.
            if method in ['med', 'median']:
                print('Estimating background from median values.')
                self.background = self.med_image
                self.background_method = 'median'

            elif method in ['min', 'minimum']:
                print('Estimating dark-field with minimum method.')
                self.background = self.min_image # does not account for changing electronic noise
                self.background_method = 'minimum'
                
            elif method in ['ball', 'rolling ball', 'rolling_ball']:
                print('Estimating background with rolling ball method.')
                self.background = rolling_ball(self.images, **kwargs)
                self.background_method = 'rolling ball'

            elif method in ['spline', 'spline fit', 'spline_fit']:
                raise NotImplementedError()
                self.background_method = 'spline'

            elif method in ['poly', 'poly fit', 'poly_fit']:
                raise NotImplementedError()
                self.background_method = 'polynomial'

            elif method in ['gauss', 'gauss fit', 'gauss_fit']:
                raise NotImplementedError()
                self.background_method = 'gaussian'

            elif method in ['none']:
                print('No background correction will be used.')
                self.background = None
                self.background_method = 'none'
            
            else:
                raise NotImplementedError("Method input not implemented!")
        
        else:
            print('User-specified background.')
            self.background = background
            self.background_method = 'custom'
    

    def remove_background(self, background=None):
        if background is None:
            if hasattr(self, 'background'):
                background = getattr(self, 'background')
            else:
                print('No background removal.')
                return
        else:
            self.background = background
            
        print('Removing background...', end='', flush=True)
        self.images -= self.background
        
        # Save background if it is only one image
        # No need to waste storage space otherwise
        if np.squeeze(self.background).shape == self.calibrated_shape:
            self.save_images(self.background,
                             'static_background',
                             units=str(self.background.dtype), 
                             labels=['chi_ind','tth_ind'])
        self.corrections['background'] = True
        self.update_map_title()
        print('done!')


    ##########################
    ### Plotting Functions ###
    ##########################
        
    def interactive_map(self, tth=None, chi=None, **kwargs):
        # I should probably rebuild these to not need tth and chi
        if hasattr(self, 'tth') and tth is None:
            tth = self.tth
        if hasattr(self, 'chi') and chi is None:
            chi = self.chi

        interactive_dynamic_2d_plot(self.images, tth=tth, chi=chi, **kwargs)
    

    def plot_image(self, image=None, map_title=None, vmin=None, aspect='auto', **kwargs):
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
        elif map_title is None:
            ax.set_title('Input Image')
        else:
            ax.set_title(map_title)
        fig.colorbar(im, ax=ax)
        plt.show()

    ####################
    ### IO Functions ###
    ####################

    def disk_size(self):
        # Return current map size which should be most of the memory usage
        # Helps to estimate file size too
        disk_size = self.images.nbytes
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


    def save_images(self, images=None, title=None, units='', labels='',
                    compression=None, compression_opts=None,
                    mode='a', extra_attrs=None):
        
        if self.h5 is None:
            return # Should disable working with h5 if no information is provided
        
        if images is None:
            images = self.images
        else:
            images = np.asarray(images)
        
        if title is None:
            title = self.title
        
        if len(images.shape) == 2:
            if title[0] != '_':
                title = f'_{title}'
        elif len(images.shape) != 4:
            raise IOError(f'Images input has {len(images.shape)} dimensions instead of 2 (image) or 4 (ImageMap).')
        elif len(images.shape) == 4 and compression is None:
            compression = 'gzip'
            compression_opts = 8
        else:
            raise RuntimeError('Unknown image type detected!')
        
        with h5py.File(self.h5, mode) as f:
            img_grp = f['/xrdmap/image_data']
            #print(f'imagemap_title is f{title}')
            if title not in img_grp.keys():
                dset = img_grp.require_dataset(
                                title,
                                data=images,
                                shape=images.shape,
                                dtype=images.dtype,
                                compression=compression,
                                compression_opts=compression_opts)
            else: # Overwrite data. No checks are performed
                dset = img_grp[title]

                if dset.shape == images.shape and dset.dtype == images.dtype:
                    dset[...] = images # Replace data if the size and shape match
                    
                else: # Delete and create new dataset if new size on disk
                    del img_grp[title]
                    dset = img_grp.create_dataset(
                                title,
                                data=images,
                                shape=images.shape,
                                dtype=images.dtype,
                                compression=compression,
                                compression_opts=compression_opts)
            
            dset.attrs['labels'] = labels
            dset.attrs['units'] = units
            dset.attrs['dtype'] = str(images.dtype)
            dset.attrs['time_stamp'] = ttime.ctime()

            # Add non-standard extra metadata attributes
            if extra_attrs is not None:
                for key, value in extra_attrs.items():
                    dset.attrs[key] = value

            # Add correction information to each dataset
            if title[0] != '_':
                for key, value in self.corrections.items():
                    dset.attrs[f'_{key}_correction'] = value

    

    # These two functions can probably be combined...
    @deprecated
    def save_current_images(self, units='', labels='',
                            compression='gzip', compression_opts=8,
                            mode='a', extra_attrs=None):
        
        if self.h5 is not None: # Should disable working with h5 if no information is provided
            with h5py.File(self.h5, mode) as f:
                img_grp = f['/xrdmap/image_data']
                if self.title not in img_grp.keys():
                    dset = img_grp.require_dataset(
                                    self.title,
                                    data=self.images,
                                    shape=self.images.shape,
                                    dtype=self.dtype,
                                    compression=compression,
                                    compression_opts=compression_opts)
                else: # Overwrite data. No checks are performed
                    dset = img_grp[self.title]
                    if dset.shape == self.shape and dset.dtype == self.dtype:
                        dset[...] = self.images # Replace data if the size and shape match
                    else: # Delete and create new dataset if new size on disk
                        del img_grp[self.title]
                        dset = img_grp.create_dataset(
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

    @deprecated
    def save_single_image(self, image, title, units='', labels='', mode='a', extra_attrs=None):

        image = np.asarray(image)
        # Prevents loading these as full image data. Maybe best to separate into groups...
        if title[0] != '_':
            title = f'_{title}'
        
        if self.h5 is not None: # Should disable working with h5 if no information is provided
            with h5py.File(self.h5, mode) as f:
                img_grp = f['/xrdmap/image_data']

                if title not in img_grp.keys():
                    dset = img_grp.require_dataset(
                                    title,
                                    data=image,
                                    shape=image.shape,
                                    dtype=image.dtype)
                else:
                    dset = img_grp[title]
                    if dset.shape == image.shape and dset.dtype == image.dtype:
                        dset[...] = image # Replace data if the size and shape match
                    else: # Delete and create new dataset if new size on disk
                        del img_grp[title]
                        dset = img_grp.create_dataset(
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
    