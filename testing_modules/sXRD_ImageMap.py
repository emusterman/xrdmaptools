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
import dask
import dask.array as da


class ImageMap:
    # This class is only intended for direct image manipulation
    # Analysis and interpretation of diffraction are reserved for the XRDMap class

    def __init__(self, image_array, map_shape=None,
                 dtype=None, title=None, h5=None,
                 corrections=None):
        
        # Non-paralleized image processing
        #image_array = np.asarray(image_array)

        # Parallized image processing
        # WIP
        image_array = da.from_array(image_array)

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
            self.corrections = corrections
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
            
                if np.any(self.mask != 1) and axes == (2, 3):
                    blank_map = np.zeros(self.map_shape)

                    for i, image in enumerate(self.images.reshape(self.num_images, *self.images.shape[-2:])):
                        val = function(image[self.mask])
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


    # Adaptively saves for whatever the current processing state
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
                             f'_{self.title}_composite')
                             #units=f'same as {self.title}',
                             #labels=f'same as {self.title}')

            # Finally return the requested attribute
            return getattr(self, f'_{self.title}_composite')
        
    @composite_image.deleter
    def composite_image(self):
        delattr(self, '_composite_image')

    
    @property
    def mask(self):
        # Generic mask with everything
        mask = np.ones(self.images.shape[2:], dtype=np.bool_)

        # Remove unused calibration pixels
        if hasattr(self, 'calibration_mask'):
            if self.calibration_mask.shape == mask.shape:
                mask *= self.calibration_mask
            else:
                print('Warning: Calibration mask found, but shape does not match images.')

        # Remove image defects
        if hasattr(self, 'defect_mask'):
            if self.defect_mask.shape == mask.shape:
                mask *= self.defect_mask
            else:
                print('Warning: Defect mask found, but shape does not match images.')

        if hasattr(self, 'custom_mask'):
            if self.custom_mask.shape == mask.shape:
                mask *= self.custom_mask
            else:
                print('Warning: Custom mask found, but shape does not match images.')

        return mask


    # Function to dump accummulated processed images and maps
    # Not sure if will be needed between processing the full map
    def reset_attributes(self):
        old_attr = list(self.__dict__.keys())       
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

        elif any([self.corrections[key]
                    for key in ['dark_field',
                                'flat_field']]):
                self.title = 'detector_corrected'

                if any([self.corrections[key]
                        for key in ['lorentz',
                                    'polarization',
                                    'solid_angle']]):
                    self.title = 'geometry_corrected'

                    if (self.corrections['background']):
                        self.title = 'background_corrected'

                        if (self.corrections['polar_calibration']):
                            self.title = 'calibrated_images'
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

        if self.corrections['dark_field']:
            print('''Warning: Dark-field correction already applied! 
                  Proceeding without any changes''')
        elif dark_field is None:
            print('No dark-field correction.')
        else:
            self.dark_field = dark_field
            
            # convert from integer to float if necessary
            if (np.issubdtype(self.dtype, np.integer)
                and np.issubdtype(self.dark_field.dtype, np.floating)):
                self.dtype = self.dark_field.dtype

            # Switch to greater precision if necessary
            elif self.dark_field.dtype > self.dtype:
                self.dtype = self.dark_field.dtype
            
            # Switch to int if uint and values will go negative
            elif check_precision(self.dtype)[0].min >= 0:
                # Not sure how to decide which int precision
                # Trying to save memory if possible
                if np.max(self.dark_field) > np.min(self.min_image):
                    self.dtype = np.int32
            
            print('Correcting dark-field...', end='', flush=True)
            self.images -= self.dark_field

            self.save_images(self.dark_field,
                            'dark_field',
                            units='counts')
            
            self.corrections['dark_field'] = True
            self.update_map_title()
            print('done!')


    def correct_flat_field(self, flat_field=None):

        if self.corrections['flat_field']:
            print('''Warning: Flat-field correction already applied! 
                  Proceeding without any changes''')
        elif flat_field is None:
            print('No flat-field correction.')
        else:
            self.flat_field = flat_field
            self.dtype = np.float32
            print('Correcting flat-field...', end='', flush=True)
            self.images /= self.flat_field

            self.save_images(self.flat_field,
                            'flat_field')
            
            self.corrections['flat_field'] = True
            self.update_map_title()
            print('done!')


    def correct_outliers(self, size=2, tolerance=3, significance=None):

        if self.corrections['outliers']:
            print('''Warning: Outlier correction already applied!
                  Proceeding to find and correct new outliers:''')
        
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


    # No correction for defect mask, since is used whenever mask is called
    def apply_defect_mask(self, min_bounds=(-np.inf, 0),
                          max_bounds=(0, np.inf), mask=None):
        if mask is not None:
            self.defect_mask = np.asarray(mask).astype(np.bool_)
        else:
            mask = np.ones_like(self.min_image, dtype=np.bool_)
            mask *= (self.min_image >= min_bounds[0]) & (self.min_image <= min_bounds[1])
            mask *= (self.max_image >= max_bounds[0]) & (self.max_image <= max_bounds[1])
            #mask *= self.min_image <= lower # Removes hot pixels
            #mask *= self.max_image >= upper # Remove dead pixels
            self.defect_mask = mask
        
        self.update_map_title()
        self.corrections['pixel_defects'] = True
        # Write mask to disk
        self.save_images(self.defect_mask,
                         'defect_mask')


    # No correction for custom mask, since is used whenever mask is called
    def apply_custom_mask(self, mask=None):
        if mask is not None:
            self.custom_mask = np.asarray(mask).astype(np.bool_)
            # Write mask to disk
            self.save_images(self.custom_mask,
                             'custom_mask')
        else:
            print('No custom mask provided!')


    ### Geometric corrections ###
    # TODO: Add conditionals to allow corrections to be applied to calibrated images

    def apply_lorentz_correction(self, ai, apply=True):

        if self.corrections['lorentz']:
            print('''Warning: Lorentz correction already applied! 
                  Proceeding without any changes''')
            return

        # In radians
        tth_arr = ai.twoThetaArray()

        # Old Lorentz correction. TODO: Add conditional for calibrated images
        #if Lorentz_correction: # Yong was disappointed I did not have this already
        #    rad = np.radians(tth / 2)
        #    res /=  1 / (np.sin(rad) * np.sin(2 * rad))

        lorentz_correction = 1 / (np.sin(tth_arr / 2) * np.sin(tth_arr))
        self.lorentz_correction = lorentz_correction
        self.save_images(self.lorentz_correction,
                            'lorentz_correction')
        
        if apply:
            print('Applying Lorentz correction...', end='', flush=True)
            self.images /= self.lorentz_correction
            self.corrections['lorentz'] = True
            self.update_map_title()
            print('done!')

    
    def apply_polarization_correction(self, ai, polarization=0.9, apply=True):

        if self.corrections['polarization']:
            print('''Warning: polarization correction already applied! 
                  Proceeding without any changes''')
            return
        
        #p = -polarization

        #tth_arr = ai.twoThetaArray()
        #chi_arr = ai.chiArray()

        # From GISAS-II supposedly
        #polar = ([(1 - p) * np.cos(chi_arr)**2 + p * np.sin(chi_arr)**2] * np.cos(tth_arr)**2
        #         + (1 - p) * np.sin(chi_arr)**2 + p * np.cos(chi_arr)**2)
        #polar = polar.squeeze()

        # From pyFAI
        #cos2_tth = np.cos(tth_arr) ** 2
        #polar = 0.5 * (1.0 + cos2_tth -
        #                polarization * np.cos(2.0 * (chi_arr)) * (1.0 - cos2_tth))

        # From pyFAI
        polar = ai.polarization(factor=polarization)
        self.polarization_correction = polar
        self.save_images(self.polarization_correction,
                            'polarization_correction')
        
        if apply:
            print('Applying X-ray polarization correction...', end='', flush=True)
            self.images /= self.polarization_correction
            self.corrections['polarization'] = True
            self.update_map_title()
            print('done!')

    
    def apply_solidangle_correction(self, ai, apply=True):

        if self.corrections['solid_angle']:
            print('''Warning: Solid angle correction already applied! 
                  Proceeding without any changes''')
            return

        #tth_arr = ai.twoThetaArray()
        #chi_arr = ai.chiArray()

        # pyFAI
        # 'SA = pixel1 * pixel2 / dist^2 * cos(incidence)^3'

        # From pyFAI
        solidangle_correction = ai.solidAngleArray()
        self.solidangle_correction = solidangle_correction
        self.save_images(self.solidangle_correction,
                            'solidangle_correction')
        
        if apply:
            print('Applying solid angle correction...', end='', flush=True)
            self.images /= self.solidangle_correction
            self.corrections['solid_angle'] = True
            self.update_map_title()
            print('done!')

    ### Final image corrections ###

    def normalize_scaler(self, scaler_arr=None):
        
        if scaler_arr is None:
            print('No scaler array given. Approximating with image medians...')
            scaler_arr = self.med_map

        else:
            scaler_arr = np.asarray(scaler_arr)
            if scaler_arr.shape != self.map_shape:
                raise ValueError(f'''Scaler array of shape {scaler_arr.shape} does not 
                                match the map shape of {self.map_shape}!''')

        print('Normalize image scalers...', end='', flush=True)
        self.images /= scaler_arr.reshape(*self.map_shape, 1, 1)
        self.scaler_map = scaler_arr # Do not save to h5, since scalers should be recorded...
        self.corrections['scaler_intensity'] = True
        self.update_map_title()
        print('done!')


    def estimate_background(self, method=None, background=None, **kwargs):
        method = str(method).lower()

        if background is None:
            # Many different background methods have been implemented
            if method in ['med', 'median']:
                print('Estimating background from median values.')
                self.background = self.med_image
                self.background_method = 'median'

            elif method in ['min', 'minimum']:
                print('Estimating background with minimum method.')
                self.background = self.min_image
                self.background_method = 'minimum'
                
            elif method in ['ball', 'rolling ball', 'rolling_ball']:
                raise NotImplementedError('Cannot not yet exclude contribution from masked regions.')
                print('Estimating background with rolling ball method.')
                self.background = rolling_ball(self.images, **kwargs)
                self.background_method = 'rolling ball'

            elif method in ['spline', 'spline fit', 'spline_fit']:
                print('Estimating background with spline fit.')
                self.background = fit_spline_bkg(self, **kwargs)
                self.background_method = 'spline'

            elif method in ['poly', 'poly fit', 'poly_fit']:
                print('Estimating background with polynomial fit.')
                print('Warning: This method is slow and not very accurate.')
                self.background = fit_poly_bkg(self, **kwargs)
                self.background_method = 'polynomial'

            elif method in ['Gaussian', 'gaussian', 'gauss']:
                print('Estimating background with gaussian convolution.')
                print('Note: Progress bar is unavailable for this method.')
                self.background = masked_gaussian_background(self, **kwargs)
                self.background_method = 'gaussian'

            elif method in ['Bruckner', 'bruckner']:
                print('Estimating background with Bruckner algorithm.')
                self.background = masked_bruckner_background(self, **kwargs)
                self.background_method = 'bruckner'

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
    

    def remove_background(self, background=None, save_images=False):
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
        # Unlikely to be used
        if np.squeeze(self.background).shape == self.images.shape[-2:]:
            self.save_images(self.background,
                             'static_background')
        self.corrections['background'] = True
        self.update_map_title()
        print('done!')

        if save_images:
            print('''Compressing and writing images to disk.\nThis may take awhile...''')
            self.save_images(extra_attrs={'background_method'
                                          : self.background_method})
            print('done!')


    ### Polar correction ###

    # Should move to geometry.py
    # Geometric calibration
    def calibrate_images(self, ai, title=None,
                         unit='2th_deg',
                         tth_resolution=0.02,
                         chi_resolution=0.05,
                         polarization_factor=None,
                         correctSolidAngle=None,
                         Lorentz_correction=None,
                         **kwargs):
        
        # Check to see if calibration even makes sense
        if self.corrections['polar_calibration']:
            raise RuntimeError("""Cannot calibrate already calibrated images! 
                            \nRevert to uncalibrated images in order to recalibrate.""")
        
        # Check the current state of the map
        self.update_map_title()
        
        # Check other states of images
        if self.title == 'raw_images':
            print('Warning: Calibrating unprocessed images. Proceeding without any image corrections.')
            _ = self.composite_image
            # Maybe I should add a list of processing step performed to keep track of everything...
        
        elif not hasattr(self, f'_{self.title}_images_composite'):
                print('Composite of current images is not saved. Creating composite.')
                _ = self.composite_image
        
        # Check a few other corrections which can be rolled into calibration
        # Recommendation is to perform each correction individually
                
        # Polarization correction
        if polarization_factor is not None and self.corrections['polarization']:
            print(('Warning: Polarization factor specified, '
                  'but images arleady corrected for polarization!'))
            print('No polarization correction will be applied.')
            polarization_factor = None

        elif polarization_factor is None and not self.corrections['polarization']:
            print(('Warning: No polarization correction applied or specified. '
                  'Images will not be polarization corrected.'))

        # Solid angle correction 
        if correctSolidAngle and self.corrections['solid_angle']:
            print(('Warning: correctSolidAngle specified, '
                  'but images arleady corrected for solid angle!'))
            print('No solid angle correction will be applied.')
            correctSolidAngle = False

        elif correctSolidAngle is None and not self.corrections['solid_angle']:
            print(('Warning: No solid angle correction applied or specified. '
                  'Images will not be corrected for solid angle.'))
            correctSolidAngle = False

        # Lorentz correction
        if Lorentz_correction and self.corrections['lorentz']:
            print(('Warning: Lorentz correction specified, '
                  'but Lorentz correction already applied!'))
            print('No Lorentz correction will be applied.')
            Lorentz_correction = False

        elif Lorentz_correction is None and not self.corrections['lorentz']:
            print(('Warning: No Lorentz correction applied or specified. '
                  'Images will not be Lorentz corrected.'))
            print('Warning: The independent Lorentz correction, currently does not work on calibrated images.')
            Lorentz_correction = False

        elif Lorentz_correction:
            self.apply_lorentz_correction(ai=ai)

        
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
                                          correctSolidAngle=correctSolidAngle,
                                          **kwargs)
            
            # Lorentz_correction is deprecated in favor of an independent version
            #if Lorentz_correction: # Yong was disappointed I did not have this already
            #    rad = np.radians(tth / 2)
            #    res /=  1 / (np.sin(rad) * np.sin(2 * rad))

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
        self.corrections['polar_calibration'] = True

        print('done!')

        if correctSolidAngle:
            self.corrections['solid_angle'] = True
        
        if polarization_factor is not None:
            self.corrections['polarization'] = True

        #print('''Compressing and writing calibrated images to disk.\nThis may take awhile...''')
        #self.save_images(units=self.calib_unit,
        #                         labels=['x_ind',
        #                                 'y_ind',
        #                                 'chi_ind',
        #                                 'tth_ind'])
        
        # Add calibration positions dataset
        print('Writing reciprocal positions...', end='', flush=True)
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
        print('Acquring and writing calibration mask...', end='', flush=True)
        self.calibration_mask = self.get_calibration_mask(ai)
        self.save_images(self.calibration_mask,
                         'calibration_mask')
        
        # Update defect mask
        if hasattr(self, 'defect_mask'):
            if self.defect_mask.shape == self.image_shape:
                new_mask, _, _ = ai.integrate2d_ng(self.defect_mask,
                                                self.tth_num,
                                                self.chi_num,
                                                unit=self.calib_unit)
                self.apply_defect_mask(mask=new_mask)
        
        # Update custom mask
        if hasattr(self, 'custom_mask'):
            if self.custom_mask.shape == self.image_shape:
                new_mask, _, _ = ai.integrate2d_ng(self.custom_mask,
                                                self.tth_num,
                                                self.chi_num,
                                                unit=self.calib_unit)
                self.apply_custom_mask(mask=new_mask)
        
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
    

    def rescale_images(self, lower=0, upper=100,
                       arr_min=None, arr_max=None,
                       mask=None):

        if mask is None and np.any(self.mask != 1):
            mask = np.empty_like(self.images, dtype=np.bool_)
            mask[:, :] = self.mask
        
        self.images = rescale_array(self.images,
                                    lower=lower,
                                    upper=upper,
                                    arr_min=arr_min,
                                    arr_max=arr_max,
                                    mask=mask)


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


    def save_images(self, images=None, title=None, units=None, labels=None,
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

        _units, _labels = self._get_save_labels(images.shape)
        if units is None:
            units = _units
        if labels is None:
            labels = _labels
        
        if len(images.shape) == 2:
            if title[0] != '_':
                title = f'_{title}'
        elif len(images.shape) != 4:
            raise ValueError(f'Images input has {len(images.shape)} dimensions instead of 2 (image) or 4 (ImageMap).')
        elif len(images.shape) == 4 and compression is None:
            compression = 'gzip'
            compression_opts = 8
        else:
            raise TypeError('Unknown image type detected!')
        
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


    def _get_save_labels(self, arr_shape):
        units = 'a.u.'
        labels = []

        if len(arr_shape) == 4:
            labels = ['x_ind',
                      'y_ind']
        
        if self.corrections['polar_calibration']:
            labels.extend(['chi_ind', 'tth_ind'])
        else:
            labels.extend(['img_y', 'img_x'])

        return units, labels

    

    # These two functions can probably be combined...
    # @deprecated
    """def save_current_images(self, units='', labels='',
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
                        dset.attrs[key] = value"""

    # @deprecated
    """def save_single_image(self, image, title, units='', labels='', mode='a', extra_attrs=None):

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
                        dset.attrs[key] = value"""
    