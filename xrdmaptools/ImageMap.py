import numpy as np
import h5py
import os
from skimage.restoration import rolling_ball
from tqdm import tqdm
import time as ttime
import dask.array as da

# Local imports
#from .utilities.hdf_utils import check_hdf_current_images, get_optimal_chunks
from .io.hdf_utils import check_hdf_current_images, get_optimal_chunks
from .utilities.math import check_precision
from .utilities.utilities import delta_array
from .utilities.image_corrections import find_outlier_pixels, rescale_array
from .utilities.background_estimators import (
    fit_spline_bkg,
    fit_poly_bkg,
    masked_gaussian_background,
    masked_bruckner_background
)


class ImageMap:
    # This class is only intended for direct image manipulation
    # Analysis and interpretation of diffraction are reserved for the XRDMap class

    def __init__(self,
                 image_data=None,
                 integration_data=None,
                 map_shape=None,
                 image_shape=None,
                 dtype=None,
                 title=None,
                 hdf_path=None,
                 hdf=None,
                 wd=None,
                 ai=None,
                 sclr_dict=None,
                 corrections=None,
                 dask_enabled=False):
        
        if (image_data is None
            and integration_data is None
            and (map_shape is None
                 or image_shape is None)):
            raise ValueError('Must specify image_data, integration_data, or dataset_shape.')
        
        # Load image data
        if image_data is not None:
            if not isinstance(image_data,
                (list,
                np.ndarray,
                da.core.Array,
                h5py._hl.dataset.Dataset)
                ):
                raise TypeError(f'Incorrect image_data type ({type(image_data)}).')

            # Parallized image processing
            if dask_enabled:
                # Unusual chunk shapes are fixed later
                # Not the best though....
                image_data = da.asarray(image_data)
            else:
                # Non-paralleized image processing
                if isinstance(image_data, da.core.Array):
                    image_data = image_data.compute()
                # list input is handled lazily, but should only be from databroker
                elif isinstance(image_data, list):
                    print('Converting rows of images into lazily loaded 4D array...')
                    image_data = da.stack(image_data) # bad chunking will be fixed later
                # Otherwise as a numpy array
                else:
                    image_data = np.asarray(image_data)

            # Working wtih image_data shape
            # Check inputs first. If these are given, assumed they will be used.
            for shape, name in zip([map_shape, image_shape], ['map_shape', 'image_shape']):
                if shape is not None and len(shape) != 2:
                    raise ValueError(f'Given {name} ({shape}) is not 2D!')
            # Fully defined inputs
            if map_shape is not None and image_shape is not None:
                dataset_shape = (*map_shape, *image_shape)
                self.images = image_data.reshape(dataset_shape)
            # 3D input.
            elif image_data.ndim == 3:
                input_shape = image_data.shape
                print('WARNING: image_data given as 3D object.')
                if map_shape is not None:
                    self.images = image_data.reshape((*map_shape, *input_shape[-2:]))
                    image_shape = self.images.shape[-2:]
                    print(f'Reshaping image_data into given map_shape as {self.images.shape}.')
                elif image_shape is not None:  # This may break wtih a theta or energy channel
                    self.images = image_data.reshape((*input_shape[:-2], *image_shape))
                    map_shape = self.images.shape[:-2]
                    print(f'Reshaping data into given image_shape as {self.images.shape}.')
                    print('WARNING: This is not a recommended way to store or load image data.')
                else:
                    print('Not map_shape or image_shape given. Assuming square map as list of images...')
                    input_shape = image_data.shape
                    map_side = np.sqrt(input_shape[0])
                    if map_side % 1 != 0:
                        raise RuntimeError('Assummed square map could not be constructed...')
                    new_shape = (int(map_side), int(map_side), *input_shape[1:])
                    print(f'Assumed map shape is {new_shape[:2]} with images of {new_shape[-2:]}')
                    self.images = image_data.reshape(new_shape)
                    map_shape = self.images.shape[:-2]
                    image_shape = self.images.shape[-2:]
            elif image_data.ndim == 4:
                self.images = image_data
                map_shape = self.images.shape[:-2]
                image_shape = self.images.shape[-2:]
            else:
                err_str = (f'Insufficient data provided to resolve '
                           + f'image_data of shape ({image_data.shape}).'
                           + ' 4D array is preferred input.')
                raise ValueError(err_str)

            # Some useful parameters
            self.shape = self.images.shape
            self.map_shape = map_shape
            self.image_shape = image_shape
            self.num_pixels = np.prod(*self.image_shape)
        else:
            if dask_enabled:
                print('WARNING: Cannot enable dask without image_data. Proceeding without dask.')
                dask_enabled = False
            self.images = None      
        
        # Working with integration_data shape
        if integration_data is not None:
            # No lazy loading of integration_data. Might be worth it, but requires a lot of support
            integration_data = np.asarray(integration_data)

            # Explicit map_shape provided
            if map_shape is not None:
                if len(map_shape) != 2:
                    raise ValueError(f'Given map_shape ({map_shape}) is not 2D!')
                self.integrations = integration_data.reshape((self.map_shape, -1))
            
            # 2D integration_data
            elif integration_data.ndim == 2:
                print('WARNING: integration_data given as 2D object without providing map_shape.')
                print('Assuming 2D map shape...')
                input_shape = integration_data.shape
                map_side = np.sqrt(input_shape[0])
                if map_side % 1 != 0:
                    raise RuntimeError('Assummed square map could not be constructed...')
                new_shape = (int(map_side), int(map_side), input_shape[-1])
                print(f'Assumed map_shape is {new_shape[:2]} with integrations of {new_shape[-1]}')
                self.integrations = integration_data.reshape(new_shape)
                map_shape = self.integrations.shape[:2]

            # 3D integration_data
            elif integration_data.ndim == 3:
                self.integrations = integration_data
                map_shape = self.integrations.shape[:2]
            
            else:
                err_str = (f'Insufficient data provided to resolve '
                           + f'image_data of shape ({integration_data.shape}).'
                           + ' 3D array is preferred input.')
                raise ValueError(err_str)
        else:
            self.integrations = None
        
        # Some useful parameters
        self.map_shape = map_shape
        self.image_shape = image_shape
        if self.map_shape is not None:
            self.num_pixels = np.prod(*self.image_shape)
        else:
            self.num_pixels = 0
        if hasattr(self, 'images') and self.images is not None:
            self.shape = self.images.shape
        
        # If dask_enabled:
        if isinstance(self.images, da.core.Array):
            # Redo chunking along image dimensions if not already
            if self.images.chunksize[-2:] != self.images.shape[-2:]:
                self._get_optimal_chunks()
                self.images = self.images.rechunk(chunks=self._chunks)
            else:
                self._chunks = self.images.chunksize
        
        # Working with the many iteraction of hdf
        # Too much information to pass back and forth
        if isinstance(hdf, h5py._hl.files.File) and hdf_path is None:
            try:
                hdf_path = hdf.filename
            except ValueError:
                raise ValueError('ImageMap cannot be instantiated with closed hdf file!')
        self.hdf = hdf
        self.hdf_path = hdf_path

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
                'lorentz' : False,
                'polarization' : False,
                'solid_angle' : False,
                'absorption' : False,              
                'scaler_intensity' : False,
                'background' : False
            }

        self.update_map_title(title=title)

        # Should only trigger on first call to save images to hdf
        # Or if the title has been changed for some reason
        if self.hdf_path is not None:
            if not check_hdf_current_images(self.title, self.hdf_path, self.hdf):
                print('Writing images to hdf...', end='', flush=True)
                self.save_images(units='counts',
                                 labels=['x_ind',
                                         'y_ind',
                                         'img_y',
                                         'img_x'])
                print('done!')
        
        if dask_enabled:
            if self.hdf_path is None and self.hdf is None:
                raise RuntimeError("Cannot have dask enabled processing without specifying hdf file!")
            elif self.hdf is None:
                # Open and leave open hdf file object
                self.hdf = h5py.File(self.hdf_path, 'a')

            # Check for finalized images
            if self.title == 'final_images':
                self._hdf_store = self.hdf['xrdmap/image_data/final_images']

            # Otherwise set a temporary storage location in the hdf file
            else:
                # Check for previously worked on data
                if check_hdf_current_images('_temp_images', self.hdf_path, self.hdf):
                    self._hdf_store = self.hdf['xrdmap/image_data/_temp_images']
                    # Change datatype and chunking to match previous _temp_images
                    if self.images.dtype != self._hdf_store.dtype:
                        self.images = self.images.astype(self._hdf_store.dtype)
                    if self.images.chunksize != self._hdf_store.chunks:
                        self.images = self.images.rechunk(self._hdf_store.chunks)
                else:
                    # Upcast before writing to hdf
                    self.images = self.images.astype(np.float32)
                    self._hdf_store = self.hdf.require_dataset('xrdmap/image_data/_temp_images',
                                                shape=self.images.shape,
                                                dtype=np.float32,
                                                chunks=self._chunks,
                                                compression_opts=4) # This may slow it down

                # Might be best NOT to call this to preserve previous data
                self.images = da.store(self.images, self._hdf_store,
                                       compute=True, return_stored=True)[0]       
        
        if wd is None:
            wd = os.getcwd()
        self.wd = wd
        self.ai = ai
        self.sclr_dict = sclr_dict
    

    def __str__(self):
        ostr = f'ImageMap: ({self.images.shape}), dtype={self.images.dtype}'
        return ostr

    
    def __repr__(self):
        ostr = 'ImageMap:'
        ostr += f'\n\tShape:  {self.images.shape}'
        ostr += f'\n\tDtype:  {self.images.dtype}'
        ostr += f'\n\tState:  {self.title}'
        return ostr
    
    ######################################
    ### Class Properties and Functions ###
    ######################################

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        # Update datatype if different
        if dtype != self._dtype:
            # Unfortunately, this has to copy the dataset
            self.images = self.images.astype(dtype)
            self._dtype = dtype

    
    def projection_factory(property_name, function, axes):
        property_name = f'_{property_name}' # This line is crucial! 
        def get_projection(self):
            if hasattr(self, property_name):
                return getattr(self, property_name)
            else:
                # Compute any scheduled operations
                self._dask_2_dask()

                # More complicated to account for masked values
                if np.any(self.mask != 1) and axes == (2, 3):

                    mask_map = np.empty_like(self.images, dtype=np.bool_)
                    mask_map[:, :] = self.mask

                    # Contribution
                    zero_image = self.images.copy() # Expensive...
                    zero_image[~mask_map] = 0 # should be redundant
                    gauss_zero = function(zero_image, axis=axes)

                    # This block is redundant when calling this serveral several times...
                    div_image = np.ones_like(self.images)
                    div_image[~mask_map] = 0
                    gauss_div = function(div_image, axis=axes)

                    val = gauss_zero / gauss_div
                    if self._dask_enabled:
                        val = val.compute()

                    setattr(self, property_name, val)
                else:
                    # This will have some precision issues depending on the specified dtype
                    # Ignores mask values since they should not vary with projection
                    val =  function(self.images, axis=axes).astype(self.dtype)
                    if self._dask_enabled:
                        val = val.compute()
                    setattr(self, property_name, val)
                
                return getattr(self, property_name)

        def del_projection(self):
            delattr(self, property_name)
        
        return property(get_projection, None, del_projection)


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
    @property
    def composite_image(self):
        if hasattr(self, f'_composite_image'):
            return getattr(self, f'_composite_image')
        else:
            setattr(self, f'_{self.title}_composite',
                    self.max_image - self.min_image)
            
            # Set the generic value to this as well
            self._composite_image = getattr(self, f'_{self.title}_composite')

            # Save image to hdf. Should update if changed
            self.save_images(self._composite_image,
                             f'_{self.title}_composite')

            # Finally return the requested attribute
            return getattr(self, f'_composite_image')
        
    @composite_image.deleter
    def composite_image(self):
        delattr(self, '_composite_image')

    
    @property
    def mask(self):
        # a bit redundant considering only 4D shapes are allowed
        img_slc = (0,) * (self.images.ndim - 2) 
        #mask = np.ones_like(self.images[img_slc], dtype=np.bool_)
        mask = np.ones(self.images[img_slc].shape, dtype=np.bool_)

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
        # Will be used to create new groups when saving to hdf
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

        # Clear old values everytime this is checked
        self.reset_attributes()
    

    ######################
    ### Dask Functions ###
    ######################
        
    # Flag to check if images are dask or numpy array
    @property
    def _dask_enabled(self):
        return isinstance(self.images, da.core.Array)

    # Return standardized chunk size around images for hdf and dask
    def _get_optimal_chunks(self, approx_chunk_size=None):
        self._chunks =  get_optimal_chunks(self.images,
                                  approx_chunk_size=approx_chunk_size)
        return self._chunks
    
    def _dask_2_numpy(self):
        # Computes dask to numpy array. Intended for final images
        if self._dask_enabled:
            self.images = self.images.compute()

    def _numpy_2_dask(self):
        # Computes numpy array into dask. Unlikely to be used
        if not self._dask_enabled:
            self._dask_2_hdf()
            self.images = da.from_array(self.images)
            self.hdf.close()
            self.hdf = None

    def _dask_2_dask(self):
        # Computes and updates dask array to avoid too many lazy computations
        # Will have faster compute times than _dask_2_hdf()
        if self._dask_enabled:
            self.images = self.images.persist()
            
    def _dask_2_hdf(self):
        # Computes and stores current iteration of lazy computation to hdf file
        # Probably the most useful
        if self.title == 'final_images':
            err_str = ('You are trying to update images that have already been finalized!'
                       + '\nConsider reloading a previous image state, or reprocessing the raw images.')
            raise ValueError(err_str)

        if self.hdf is not None and self._dask_enabled:
            self.images = da.store(self.images, self._hdf_store,
                                   compute=True, return_stored=True)[0]


    ########################################
    ### Image Corrections and Transforms ###
    ########################################

    ### Initial image corrections ###

    def correct_dark_field(self, dark_field=None):

        if self.corrections['dark_field']:
            print('''Warning: Dark-field correction already applied! 
                  Proceeding without any changes''')
        elif dark_field is None:
            print('No dark-field given for correction.')
        elif dark_field.shape != self.image_shape:
            err_str = (f'Dark-field shape of {dark_field.shape} does '
                      + f'not match image shape of {self.image_shape}.')
            raise ValueError(err_str)
        else:
            self.dark_field = dark_field
            
            # Convert from integer to float if necessary
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
                if np.max(self.dark_field) > np.min(self.images):
                    self.dtype = np.float32 # Go ahead and make it the final size...
            
            print('Correcting dark-field...', end='', flush=True)
            self.images -= self.dark_field

            self.save_images(self.dark_field,
                            'dark_field',
                            units='counts')
            
            self.corrections['dark_field'] = True
            self.update_map_title()
            self._dask_2_hdf()
            print('done!')


    # TODO: This fails if image medians are too close to zero
    #       flat_field values near zero blow_up pixel values,
    #       but the means/medians of the flat field and images should be similar
    #       Consider adding a way to rescale the image array along with the flat_field
    #       As of now, this should be done immediately after the dark field correction       
    def correct_flat_field(self, flat_field=None):

        if self.corrections['flat_field']:
            print('''Warning: Flat-field correction already applied! 
                  Proceeding without any changes''')
        elif flat_field is None:
            print('No flat-field correction.')
        elif flat_field.shape != self.image_shape:
            err_str = (f'Flat-field shape of {flat_field.shape} does '
                      + f'not match image shape of {self.image_shape}.')
            raise ValueError(err_str)
        else:
            self.dtype = np.float32
            print('Correcting flat-field...', end='', flush=True)
            
            # Shift flat_field correction to center image map center
            if self._dask_enabled:
                correction = np.mean(self.images) - np.mean(flat_field)
            else:  # Dask does not handle medians very well...
                correction = np.median(self.images) - np.mean(flat_field)
            
            self.flat_field = flat_field + correction
            self.images /= self.flat_field

            self.save_images(self.flat_field,
                            'flat_field')
            
            self.corrections['flat_field'] = True
            self.update_map_title()
            self._dask_2_hdf()
            print('done!')


    def correct_outliers(self, size=2, tolerance=3):

        if self.corrections['outliers']:
            print('''Warning: Outlier correction already applied!
                  Proceeding to find and correct new outliers:''')
        
        print('Finding and correcting image outliers...', end='', flush=True)
        self.images = find_outlier_pixels(self.images,
                                          size=size,
                                          tolerance=tolerance)
        
        self.corrections['outliers'] = True
        self.update_map_title()
        self._dask_2_hdf()
        print('done!')


    # No correction for defect mask, since it is used whenever mask is called
    def apply_defect_mask(self, min_bounds=(-np.inf, 0),
                          max_bounds=(0, np.inf), mask=None):
        if mask is not None:
            self.defect_mask = np.asarray(mask).astype(np.bool_)
        else:
            mask = np.ones_like(self.min_image, dtype=np.bool_)
            mask *= (self.min_image >= min_bounds[0]) & (self.min_image <= min_bounds[1])
            mask *= (self.max_image >= max_bounds[0]) & (self.max_image <= max_bounds[1])
            self.defect_mask = mask

        # Write mask to disk
        self.save_images(self.defect_mask,
                         'defect_mask') 
        
        self.corrections['pixel_defects'] = True
        self.update_map_title()
        self._dask_2_hdf()



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
    def apply_lorentz_correction(self, powder=False, apply=True):

        if self.corrections['lorentz']:
            print('''Warning: Lorentz correction already applied! 
                  Proceeding without any changes''')
            return

        # In radians
        tth_arr = self.ai.twoThetaArray().astype(self.dtype)

        # Static area detector. Seems to work well
        lorentz_correction = 1 / np.sin(tth_arr / 2)

        if powder:
            lorentz_correction *= np.cos(tth_arr / 2)
        
        # Save and apply Lorentz corrections
        self.lorentz_correction = lorentz_correction
        self.save_images(self.lorentz_correction,
                            'lorentz_correction')
    
        if apply:
            print('Applying Lorentz correction...', end='', flush=True)
            self.images /= self.lorentz_correction
            self.corrections['lorentz'] = True
            self.update_map_title()
            self._dask_2_hdf()
            print('done!')

    
    def apply_polarization_correction(self, polarization=0.9, apply=True):

        if self.corrections['polarization']:
            print('''Warning: polarization correction already applied! 
                  Proceeding without any changes''')
            return
        
        # TODO: Does not work with already calibrated images
        
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
        polar = self.ai.polarization(factor=polarization).astype(self.dtype)
        self.polarization_correction = polar
        self.save_images(self.polarization_correction,
                            'polarization_correction')
        
        if apply:
            print('Applying X-ray polarization correction...', end='', flush=True)
            self.images /= self.polarization_correction
            self.corrections['polarization'] = True
            self.update_map_title()
            self._dask_2_hdf()
            print('done!')

    
    def apply_solidangle_correction(self, apply=True):

        if self.corrections['solid_angle']:
            print('''Warning: Solid angle correction already applied! 
                  Proceeding without any changes''')
            return

        # TODO: Does not work with already calibrated images
        #tth_arr = ai.twoThetaArray()
        #chi_arr = ai.chiArray()

        # From pyFAI
        # 'SA = pixel1 * pixel2 / dist^2 * cos(incidence)^3'

        # From pyFAI
        solidangle_correction = self.ai.solidAngleArray().astype(self.dtype)
        self.solidangle_correction = solidangle_correction
        self.save_images(self.solidangle_correction,
                            'solidangle_correction')
        
        if apply:
            print('Applying solid angle correction...', end='', flush=True)
            self.images /= self.solidangle_correction
            self.corrections['solid_angle'] = True
            self.update_map_title()
            self._dask_2_hdf()
            print('done!')


    def apply_absorption_correction(self, exp_dict, apply=True):

        #exp_dict = {
        #    'attenuation_length' : value,
        #    'mode' : 'transmission',
        #    'thickness' : value,
        #    'theta' : value # in degrees!
        #}

        if self.corrections['absorption']:
            print('''Warning: Absorption correction already applied! 
                  Proceeding without any changes''')
            return
        
        # In radians
        tth_arr = self.ai.twoThetaArray()
        chi_arr = self.ai.chiArray()

        if not all(x in list(exp_dict.keys()) for x in ['attenuation_length', 'mode', 'thickness', 'theta']):
            raise ValueError("""Experimental dictionary does not have all the necessary keys: 
                             'attenuation_length', 'mode', and 'thickness'.""")

        # Semi-infinite plate
        if exp_dict['mode'] == 'transmission':
            t = exp_dict['thickness']
            a = exp_dict['attenuation_length']
            theta = np.radians(exp_dict['theta'])
            if theta == 0:
                x = t / np.cos(tth_arr) # path length
            else:
                # OPTIMIZE ME!
                # Intersection coordinates of lines. One for the far surface and another for the diffracted beam
                # Origin at intersection of initial surface and transmitted beam
                # y1 = x1 / np.tan(tth) # diffracted beam
                # y2 = np.cos(chi) * np.tan(theta) + t / np.cos(theta) # far surface
                xi = (t / (np.cos(theta))) / ((1 / np.tan(tth_arr)) - (np.cos(chi_arr) * np.tan(theta)))
                yi = xi / (np.tan(tth_arr))
                # Distance of intersection point
                x = np.sqrt(xi**2 + yi**2)

            abs_arr = np.exp(-x / a)
            self.absorption_correction = abs_arr

        elif exp_dict['mode'] == 'reflection':
            raise NotImplementedError()
        else:
            raise ValueError(f"{exp_dict['mode']} is unknown.")
        
        if apply:
            print('Applying absorption correction...', end='', flush=True)
            self.images /= self.absorption_correction
            self.corrections['absorption'] = True
            self.update_map_title()
            self._dask_2_hdf()
            print('done!')


    def normalize_scaler(self, scaler_arr=None):

        if self.corrections['scaler_intensity']:
            print('''Warning: Images have already been normalized by the scaler! 
                  Proceeding without any changes''')
            return
        
        elif scaler_arr is None:
            if hasattr(self, 'sclr_dict') and self.sclr_dict is not None:
                if 'i0' in self.sclr_dict.keys():
                    scaler_arr = self.sclr_dict['i0']
                    sclr_key = 'i0'
                elif 'im' in self.sclr_dict.keys():
                    print('"i0" could not be found. Switching to "im".')
                    scaler_arr = self.sclr_dict['im']
                    sclr_key = 'im'
                else:
                    first_key = list(self.sclr_dict.keys())[0]
                    scaler_arr = self.sclr_dict[first_key]
                    print(f'Unrecognized scaler keys. Using "{first_key}" instead.')
                    sclr_key = first_key
            else:
                print('No scaler array given or found. Approximating with image medians.')
                scaler_arr = self.med_map
                sclr_key = 'med'

        elif scaler_arr.shape != self.map_shape:
            err_str = (f'Scaler array shape of {scaler_arr.shape} does '
                      + f'not match map shape of {self.map_shape}.')
            raise ValueError(err_str)
        else:
            sclr_key = 'input'

        # Check shape everytime
        scaler_arr = np.asarray(scaler_arr)
        if scaler_arr.shape != self.map_shape:
            raise ValueError(f'''Scaler array of shape {scaler_arr.shape} does not 
                            match the map shape of {self.map_shape}.''')
   
        print(f'Normalize image by {sclr_key} scaler...', end='', flush=True)
        self.images /= scaler_arr.reshape(*self.map_shape, 1, 1)
        self.scaler_map = scaler_arr # Do not save to hdf, since scalers should be recorded...
        self.corrections['scaler_intensity'] = True
        self.update_map_title()
        self._dask_2_hdf()
        print('done!')


    # TODO: Not all of these have been enabled with dask arrays
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
                raise NotImplementedError('Cannot yet exclude contribution from masked regions.')
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
                raise NotImplementedError(f'Method "{method}" not implemented!')
    
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
        else:
            # Delete full map background to save on memory
            del self.background

        self.corrections['background'] = True
        self.update_map_title()
        self._dask_2_hdf()
        print('done!')

        if save_images:
            print('''Compressing and writing images to disk.\nThis may take awhile...''')
            self.save_images(extra_attrs={'background_method'
                                          : self.background_method})
            print('done!')


    def get_polar_mask(self, tth_num=None, chi_num=None, units='2th_deg'):

        if tth_num is None:
            tth_num = self.tth_num
        if chi_num is None:
            chi_num = self.chi_num
        if units is None:
            units = self.calib_unit

        dummy_image = 100 * np.ones(self.image_shape)

        image, _, _ = self.ai.integrate2d_ng(dummy_image, tth_num, chi_num, unit=units)

        calibration_mask = (image != 0)

        return calibration_mask
    

    def rescale_images(self, lower=0, upper=100,
                       arr_min=0, arr_max=None,
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
        
        # Update _temp images
        self._dask_2_hdf()
        

    def finalize_images(self, save_images=True):

        if np.any(self.corrections.values()):
            print('Caution: Images not corrected for:')
        # Check corrections:
        for key in self.corrections:
            if not self.corrections[key]:
                print(f'\t{key}')

        # Some cleanup
        print('Cleaning and updating image information...')
        self._dask_2_hdf()
        self.update_map_title(title='final_images')
        self.disk_size()

        if save_images:
            if self.hdf_path is None and self.hdf is None:
                print('No hdf file specified. Images will not be saved.')
            else:
                print('Compressing and writing images to disk.\nThis may take awhile...')

                if check_hdf_current_images('_temp_images', self.hdf_path, self.hdf):
                    # Save images to current store location. Should be _temp_images
                    self.save_images(title='_temp_images')
                    temp_dset = self.hdf['xrdmap/image_data/_temp_images']
                    
                    # Must be done explicitly outside of self.save_images()
                    for key, value in self.corrections.items():
                        temp_dset.attrs[f'_{key}_correction'] = value

                    # Relabel to final images and delt temp dataset
                    self.hdf['xrdmap/image_data/final_images'] = temp_dset
                    del temp_dset

                    # Remap store location. Should only reference data from now on
                    self._hdf_store = self.hdf['xrdmap/image_data/final_images']

                else:
                    self.save_images()
                print('done!')


    #######################
    ### Integrated Data ###
    #######################
    # Dask is not implemented for integrations...

    ### 1D integration projections ###

    def integration_projection_factory(property_abbreviation, function, axes):
        property_name = f'_{property_abbreviation}' # This line is crucial! 
        def get_projection(self):
            if hasattr(self, property_name):
                return getattr(self, property_name)
            else:
                projection = function(self.integrations, axis=axes)
                setattr(self, property_name, projection)

        def del_projection(self):
            delattr(self, property_name)
        
        return property(get_projection, None, del_projection)
    
    min_integration = integration_projection_factory('min_integration', np.min, (0, 1))
    min_integration_map = integration_projection_factory('min_integration_map', np.min, (2))

    max_integration = integration_projection_factory('max_integration', np.max, (0, 1))
    max_integration_map = integration_projection_factory('max_integration_map', np.max, (2))

    sum_integration = integration_projection_factory('max_integration', np.sum, (0, 1))
    sum_integration_map = integration_projection_factory('max_integration_map', np.sum, (2))

    med_integration = integration_projection_factory('med_integration', np.median, (0, 1))
    med_integration_map = integration_projection_factory('med_integration_map', np.median, (2))

    mean_integration = integration_projection_factory('mean_integration', np.mean, (0, 1))
    mean_integration_map = integration_projection_factory('mean_integration_map', np.mean, (2))


    @property
    def composite_integration(self):
        if hasattr(self, f'_composite_integration'):
            return getattr(self, f'_composite_integration')
        else:
            setattr(self, f'_{self.title}_composite_integration',
                    self.max_integration - self.min_integration)
            
            # Set the generic value to this as well
            self._composite_integration = getattr(self, f'_{self.title}_composite_integration')

            # Save image to hdf. Should update if changed
            self.save_images(self._composite_integration,
                             f'_{self.title}_composite_integration')

            # Finally return the requested attribute
            return getattr(self, f'_composite_integration')
        
    @composite_integration.deleter
    def composite_integration(self):
        delattr(self, '_composite_integration')

    ### 1D integration corrections
    
    def estimate_integration_background(self, method=None, background=None, **kwargs):
        method = str(method).lower()

        if background is None:
            # Many different background methods have been implemented
            if method in ['med', 'median']:
                print('Estimating background from median values.')
                self.integration_background = self.med_integration
                self.integration_background_method = 'median'

            elif method in ['min', 'minimum']:
                print('Estimating background with minimum method.')
                self.integration_background = self.min_integration
                self.integration_background_method = 'minimum'
                
            elif method in ['ball', 'rolling ball', 'rolling_ball']:
                print('Estimating background with rolling ball method.')
                self.integration_background = rolling_ball(self.integrations, **kwargs)
                self.integration_background_method = 'rolling ball'

            elif method in ['spline', 'spline fit', 'spline_fit']:
                raise NotImplementedError('Still need to write function for integrations.')
                print('Estimating background with spline fit.')
                self.integration_background = fit_spline_bkg(self, **kwargs)
                self.integration_background_method = 'spline'

            elif method in ['poly', 'poly fit', 'poly_fit']:
                raise NotImplementedError('Still need to write function for integrations.')
                print('Estimating background with polynomial fit.')
                print('Warning: This method is slow and not very accurate.')
                self.integration_background = fit_poly_bkg(self, **kwargs)
                self.integration_background_method = 'polynomial'

            elif method in ['Gaussian', 'gaussian', 'gauss']:
                raise NotImplementedError('Still need to write function for integrations.')
                print('Estimating background with gaussian convolution.')
                print('Note: Progress bar is unavailable for this method.')
                self.integration_background = masked_gaussian_background(self, **kwargs)
                self.integration_background_method = 'gaussian'

            elif method in ['Bruckner', 'bruckner']:
                raise NotImplementedError('Still need to write function for integrations.')
                print('Estimating background with Bruckner algorithm.')
                self.integration_background = masked_bruckner_background(self, **kwargs)
                self.integration_background_method = 'bruckner'

            elif method in ['none']:
                print('No background correction will be used.')
                self.integration_background = None
                self.integration_background_method = 'none'
            
            else:
                raise NotImplementedError(f'Method "{method}" not implemented!')
    
        else:
            print('User-specified background.')
            self.integration_background = background
            self.integration_background_method = 'custom'
    

    def remove_integration_background(background=None):
        raise NotImplementedError()
        if background is None:
            if hasattr(self, 'integration_background'):
                background = getattr(self, 'integration_background')
            else:
                print('No background removal.')
                return
        else:
            self.integration_background = background
            
        print('Removing background...', end='', flush=True)
        self.integrations -= self.integration_background

        # Save updated integrations
        self.save_integrations()

        # Save backgrounds
        # Not as costly as image backgrounds

        # Remove no longer needed backgrounds
        del self.background
    

    def rescale_integrations():
        raise NotImplementedError()


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

    # WIP apparently???
    # This should probably not be in ImageMap anyway
    @staticmethod
    def estimate_disk_size(size):
        # External reference function to estimate map size before acquisition

        if not isinstance(size, (tuple, list, np.ndarray)):
            raise TypeError('Size argument must be iterable of map dimensions.')

        disk_size = np.prod([*size, 2])


    def _get_save_labels(self, arr_shape):
        units = 'a.u.'
        labels = []

        if len(arr_shape) == 4: # Image map
            labels = ['x_ind',
                      'y_ind']
            if self.corrections['polar_calibration']:
                labels.extend(['chi_ind', 'tth_ind'])
            else:
                labels.extend(['img_y', 'img_x'])

        elif len(arr_shape) == 2: # Image
            if self.corrections['polar_calibration']:
                labels.extend(['chi_ind', 'tth_ind'])
            else:
                labels.extend(['img_y', 'img_x'])

        elif len(arr_shape) == 3: # Integrations
            labels = ['x_ind',
                      'y_ind',
                      'tth_ind']

        return units, labels


    def save_images(self, images=None, title=None, units=None, labels=None,
                    compression=None, compression_opts=None,
                    mode='a', extra_attrs=None):
        
        if self.hdf_path is None:
            return # Should disable working with hdf if no information is provided
        
        if images is None:
            images = self.images
        else:
            # This conditional is for single images mostly (e.g., dark-field)
            images = np.asarray(images)
        
        if title is None:
            title = self.title

        # Get labels
        _units, _labels = self._get_save_labels(images.shape)
        if units is None:
            units = _units
        if labels is None:
            labels = _labels
        
        # Massage title based on shape
        if images.ndim == 2:
            if title[0] != '_':
                title = f'_{title}'
        elif images.ndim != 4:
            raise ValueError(f'Images input has {images.ndim} dimensions instead of 2 (image) or 4 (ImageMap).')
        elif images.ndim == 4 and compression is None:
            compression = 'gzip'
            compression_opts = 4 # changed default from 8 to h5py default
        else:
            raise TypeError('Unknown image type detected!')
        
        # Get chunks
        # Maybe just grab current chunk size to be consistent??
        if not hasattr(self, '_chunks') or self._chunks is None: 
            chunks = self._get_optimal_chunks()
        else:
            chunks = self._chunks
        chunks = chunks[-images.ndim:]
        
        # Flag of state hdf
        close_flag = False
        if self.hdf is None:
            close_flag = True
            self.hdf = h5py.File(self.hdf_path, mode)

        # Grab some metadata
        image_shape = images.shape
        image_dtype = images.dtype

        dask_flag=False
        if isinstance(images, da.core.Array):
            dask_images = images.copy()
            images = None
            dask_flag = True

        img_grp = self.hdf['/xrdmap/image_data']
        
        if title not in img_grp.keys():
            dset = img_grp.require_dataset(
                            title,
                            data=images,
                            shape=image_shape,
                            dtype=image_dtype,
                            compression=compression,
                            compression_opts=compression_opts,
                            chunks=chunks)
        else: # Overwrite data
            dset = img_grp[title]
            
            # Check array compatibility
            if (dset.shape == image_shape
                and dset.dtype == image_dtype
                and dset.chunks == chunks):
                if not dask_flag:
                    dset[...] = images # Replace data if the size, shape, and chunks match
                else:
                    pass # Leave the dataset for now
            else:
                # This is not the best, because this only deletes the flag. The data stays
                del img_grp[title]
                dset = img_grp.create_dataset(
                            title,
                            data=images,
                            shape=image_shape,
                            dtype=image_dtype,
                            compression=compression,
                            compression_opts=compression_opts,
                            chunks=chunks)
        
        # Fill in data from dask after setting up the datasets
        if dask_flag:
            # This should ensure lazy operations
            da.store(dask_images, dset, compute=True)
        
        dset.attrs['labels'] = labels
        dset.attrs['units'] = units
        dset.attrs['dtype'] = str(image_dtype)
        dset.attrs['time_stamp'] = ttime.ctime()

        # Add non-standard extra metadata attributes
        if extra_attrs is not None:
            for key, value in extra_attrs.items():
                dset.attrs[key] = value

        # Add correction information to each dataset
        if title[0] != '_':
            for key, value in self.corrections.items():
                dset.attrs[f'_{key}_correction'] = value
        
        if close_flag:
            self.hdf.close()
            self.hdf = None


    # TODO: update _get_save_labels to handle integrations
    def save_integrations(self, integrations=None, title=None,
                          units=None, labels=None,
                          mode='a', extra_attrs=None):
        # No dask support

        if self.hdf_path is None:
            return # Should disable working with hdf if no information is provided
        
        if integrations is None:
            if hasattr(self, 'integrations') and self.integrations is not None:
                integrations = self.integrations
            else:
                err_str = 'ImageMap does not have integrations and none have been given.'
                raise ValueError(err_str)
        else:
            # This conditional is for single images mostly (e.g., dark-field)
            integrations = np.asarray(integrations)
        
        # Set title
        if title is None:
            title = f'{self.title}_integrations'
        
        # Check shape and set title
        if integrations.ndim != 3:
            raise ValueError(f'Integrations must have 3 dimensions, not {len(integrations.shape)}.')
        
        # Get labels
        _units, _labels = self._get_save_labels(integrations.shape)
        if units is None:
            units = _units
        if labels is None:
            labels = _labels

        # Flag of state hdf
        close_flag = False
        if self.hdf is None:
            close_flag = True
            self.hdf = h5py.File(self.hdf_path, mode)

        # Grab some metadata
        integrations_shape = integrations.shape
        integrations_dtype = integrations.dtype

        int_grp = self.hdf['xrdmap'].require_group('integration_data')
        
        if title not in int_grp.keys():
            dset = int_grp.require_dataset(
                            title,
                            data=integrations,
                            shape=integrations_shape,
                            dtype=integrations_dtype,
                            compression=None, # default
                            compression_opts=None, # default
                            chunks=None) # default
        else: # Overwrite data. No checks are performed
            dset = int_grp[title]

            if (dset.shape == integrations_shape
                and dset.dtype == integrations_dtype):
                dset[...] = integrations
            else:
                # This is not the best, because this only deletes the flag. The data stays
                del int_grp[title]
                dset = int_grp.create_dataset(
                            title,
                            data=integrations,
                            shape=integrations_shape,
                            dtype=integrations_dtype,
                            compression=None, # default
                            compression_opts=None, # default
                            chunks=None) # defualt
        
        dset.attrs['labels'] = labels
        dset.attrs['units'] = units
        dset.attrs['dtype'] = str(integrations_dtype)
        dset.attrs['time_stamp'] = ttime.ctime()

        # Add non-standard extra metadata attributes
        if extra_attrs is not None:
            for key, value in extra_attrs.items():
                dset.attrs[key] = value

        # Add correction information to each dataset
        if title[0] != '_':
            for key, value in self.corrections.items():
                dset.attrs[f'_{key}_correction'] = value
        
        if close_flag:
            self.hdf.close()
            self.hdf = None