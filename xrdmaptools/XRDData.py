import numpy as np
import h5py
import os
from skimage.restoration import rolling_ball
from tqdm import tqdm
import time as ttime
import dask.array as da
import functools

# Local imports
from xrdmaptools.io.hdf_utils import (
    check_hdf_current_images,
    get_optimal_chunks
)
from xrdmaptools.utilities.math import check_precision
from xrdmaptools.utilities.utilities import rescale_array
from xrdmaptools.utilities.image_corrections import (
    find_outlier_pixels,
    iterative_outlier_correction,
    )
from xrdmaptools.utilities.background_estimators import (
    fit_spline_bkg,
    fit_poly_bkg,
    masked_gaussian_background,
    masked_bruckner_background
)


class XRDData:
    # This class is intended to hold and process raw data with I/O
    # Analysis and interpretations will be reserved for child classes

    def __init__(self,
                 image_data=None,
                 integration_data=None,
                 map_shape=None,
                 image_shape=None,
                 map_labels=None,
                 dtype=None,
                 title=None,
                 hdf_path=None,
                 hdf=None,
                 ai=None,
                 sclr_dict=None,
                 null_map=None,
                 chunks=None,
                 corrections=None,
                 dask_enabled=False,
                 hdf_type=None):
        
        if (image_data is None
            and integration_data is None
            and (map_shape is None
                 or image_shape is None)):
            err_str = ('Must specify image_data, integration_data, '
                       + 'or image and map shapes.')
            raise ValueError(err_str)
        
        if hdf_path is None and hdf_type is None:
            raise ValueError('Must specify hdf_type to use hdf.')
        else:
            self._hdf_type = hdf_type
        
        # Load image data
        if image_data is not None:
            if not isinstance(image_data,
                (list,
                np.ndarray,
                da.core.Array,
                h5py._hl.dataset.Dataset)
                ):
                err_str = ('Incorrect image_data type '
                           + f'({type(image_data)}).')
                raise TypeError(err_str)

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
                    if dask_enabled:
                        # print('Converting rows of images into lazily loaded 4D array...')
                        image_data = da.stack(image_data) # bad chunking will be fixed later
                    else:
                        image_data = np.stack(image_data)
                # Otherwise as a numpy array
                else:
                    image_data = np.asarray(image_data)

            # Working wtih image_data shape
            # Check inputs first. If these are given, assumed they will be used.
            for shape, name in zip([map_shape, image_shape],
                                   ['map_shape', 'image_shape']):
                if shape is not None and len(shape) != 2:
                    err_str = f'Given {name} ({shape}) is not 2D!'
                    raise ValueError(err_str)
            # Fully defined inputs
            if map_shape is not None and image_shape is not None:
                dataset_shape = (*map_shape, *image_shape)
                self.images = image_data.reshape(dataset_shape)
            # 3D input.
            elif image_data.ndim == 3:
                input_shape = image_data.shape
                print('WARNING: image_data given as 3D object.')
                if map_shape is not None:
                    self.images = image_data.reshape((*map_shape,
                                                      *input_shape[-2:]))
                    image_shape = self.images.shape[-2:]
                    ostr = ('Reshaping image_data into given '
                            + f'map_shape as {self.images.shape}.')
                    print(ostr)
                elif image_shape is not None:  # This may break wtih a theta or energy channel
                    self.images = image_data.reshape((*input_shape[:-2],
                                                      *image_shape))
                    map_shape = self.images.shape[:-2]
                    warn_str = ('Reshaping data into given image_shape'
                                + f' as {self.images.shape}.\nWARNING:'
                                + ' This is not a recommended way to '
                                + 'store or load image data.')
                    print(warn_str)
                else:
                    ostr = ('Not map_shape or image_shape given. '
                            + 'Assuming square map as '
                            + 'list of images...')
                    print(ostr)
                    input_shape = image_data.shape
                    map_side = np.sqrt(input_shape[0])
                    if map_side % 1 != 0:
                        err_str = ('Assummed square map could not '
                                   + 'be constructed...')
                        raise RuntimeError(err_str)
                    new_shape = (int(map_side),
                                 int(map_side),
                                 *input_shape[1:])
                    ostr = (f'Assumed map shape is {new_shape[:2]} '
                            + f'with images of {new_shape[-2:]}')
                    print(ostr)
                    self.images = image_data.reshape(new_shape)
                    map_shape = self.images.shape[:-2]
                    image_shape = self.images.shape[-2:]
            elif image_data.ndim == 4:
                self.images = image_data
                map_shape = self.images.shape[:-2]
                image_shape = self.images.shape[-2:]
            else:
                err_str = (f'Insufficient data provided to resolve '
                           + 'image_data of shape '
                           + f'({image_data.shape}).'
                           + ' 4D array is preferred input.')
                raise ValueError(err_str)

            # Some useful parameters
            self.map_shape = map_shape
            self.image_shape = image_shape
        else:
            if dask_enabled:
                warn_str = ('WARNING: Cannot enable dask without '
                            + 'image_data. Proceeding without dask.')
                print(warn_str)
                dask_enabled = False
            self.images = None

            if image_shape is None:
                err_str = ('image_shape must be provided if '
                           + 'image_data is not.')
                raise ValueError(err_str)     
        
        # Working with integration_data shape
        if integration_data is not None:
            # No lazy loading of integration_data.
            # Might be worth it, but requires a lot of support
            integration_data = np.asarray(integration_data)

            # Explicit map_shape provided
            if map_shape is not None:
                if len(map_shape) != 2:
                    err_str = (f'Given map_shape ({map_shape}) '
                               + 'is not 2D!')
                    raise ValueError(err_str)
                self.integrations = integration_data.reshape(
                                        (*self.map_shape, -1))
            
            # 2D integration_data
            elif integration_data.ndim == 2:
                warn_str = ('WARNING: integration_data given as 2D '
                            + 'object without providing map_shape.'
                            + '\nAssuming 2D map shape...')
                print(warn_str)
                input_shape = integration_data.shape
                map_side = np.sqrt(input_shape[0])
                if map_side % 1 != 0:
                    err_str = ('Assummed square map could not '
                               + 'be constructed...')
                    raise RuntimeError(err_str)
                new_shape = (int(map_side),
                             int(map_side),
                             input_shape[-1])
                ostr = (f'Assumed map_shape is {new_shape[:2]} '
                        + f'with integrations of {new_shape[-1]}')
                print(ostr)
                self.integrations = integration_data.reshape(new_shape)
                map_shape = self.integrations.shape[:2]

            # 3D integration_data
            elif integration_data.ndim == 3:
                self.integrations = integration_data
                map_shape = self.integrations.shape[:2]
            
            else:
                err_str = (f'Insufficient data provided to resolve '
                           + 'integration_data of shape '
                           + f'({integration_data.shape}).'
                           + ' 3D array is preferred input.')
                raise ValueError(err_str)
        else:
            self.integrations = None
            self.integrations_corrections = None
        
        # Some useful parameters
        if (not hasattr(self, 'map_shape')
            or self.map_shape is not None):
            self.map_shape = map_shape
        if (not hasattr(self, 'image_shape')
            or self.image_shape is not None):
            self.image_shape = image_shape
        
        if self.map_shape is not None:
            self.num_images = np.prod(self.map_shape)
        else:
            self.num_images = 0

        if (self.map_shape is not None
            and self.image_shape is not None):
            self.shape = (*self.map_shape, *self.image_shape)
        
        # Define/determine chunks and rechunk if necessary
        if isinstance(self.images, da.core.Array):
            # Redo chunking along image dimensions if not already
            if self.images.chunksize[-2:] != self.images.shape[-2:]:
                self._get_optimal_chunks()
                self.images = self.images.rechunk(chunks=self._chunks)
            else:
                self._chunks = self.images.chunksize
        else:
            if chunks is not None:
                self._chunks = chunks
            else:
                self._get_optimal_chunks()
        
        # Working with the many iteraction of hdf
        if isinstance(hdf, h5py._hl.files.File) and hdf_path is None:
            try:
                hdf_path = hdf.filename
            except ValueError:
                err_str = ('XRDData cannot be instantiated with '
                           + 'closed hdf file!')
                raise ValueError(err_str)
        self.hdf = hdf
        self.hdf_path = hdf_path

        if dtype is None:
            if (hasattr(self, 'images')
                and self.images is not None):
                dtype = self.images.dtype
            elif (hasattr(self, 'integrations')
                  and self.integrations is not None):
                dtype = self.integrations.dtype
            else:
                dtype = None
        self._dtype = dtype

        if isinstance(corrections, dict):
            self.corrections = corrections
        else:
            self.corrections = {
                'dark_field' : False,
                'flat_field' : False, 
                # Can be approximated with background
                'air_scatter' : False, 
                'outliers' : False,
                'pixel_defects' : False, # Just a mask
                'pixel_distortions' : False, # Uncommon
                'polar_calibration' : False, # Bulky
                'lorentz' : False,
                'polarization' : False,
                'solid_angle' : False,
                'absorption' : False, # Tricky      
                'scaler_intensity' : False,
                'background' : False
            }

        self.update_map_title(title=title)
        if isinstance(null_map, list):
            null_map = np.zeros(self.map_shape, dtype=np.bool_)
        self.null_map = null_map

        if map_labels is None:
            self.map_labels = ['null_ind',
                               'null_ind']
        elif len(map_labels) != 2:
            raise ValueError('Length of map_labels must be 2.')
        else:
            self.map_labels = map_labels

        # Should only trigger on first call to save images to hdf
        # Or if the title has been changed for some reason
        if (self.hdf_path is not None
            and self.images is not None):
            if not check_hdf_current_images(f'{self.title}_images',
                                            self.hdf_path,
                                            self.hdf):
                print('Writing images to hdf...', end='', flush=True)
                self.save_images(units='counts')
                print('done!')
                
            if self.null_map is not None:
                self.save_images(images='null_map',
                                 units='bool',
                                 labels=self.map_labels)
        
        if dask_enabled:
            if self.hdf_path is None and self.hdf is None:
                err_str = ('Cannot have dask enabled processing '
                           + 'without specifying hdf file!')
                raise RuntimeError(err_str)
            elif self.hdf is None:
                # Open and leave open hdf file object
                self.hdf = h5py.File(self.hdf_path, 'a')

            # Check for finalized images
            if self.title == 'final':
                hdf_str = f'{self._hdf_type}/image_data/final_images'
                self._hdf_store = self.hdf[hdf_str]

            # Otherwise set a temporary storage location in the hdf file
            else:
                # Check for previously worked on data
                if check_hdf_current_images('_temp_images',
                                            self.hdf_path,
                                            self.hdf):
                    hdf_str = (f'{self._hdf_type}/image_data'
                               + '/_temp_images')
                    self._hdf_store = self.hdf[hdf_str]
                    # Change datatype and chunking to match previous _temp_images
                    if self.images.dtype != self._hdf_store.dtype:
                        self.images = self.images.astype(
                                            self._hdf_store.dtype)
                    if self.images.chunksize != self._hdf_store.chunks:
                        self.images = self.images.rechunk(
                                            self._hdf_store.chunks)
                        self._chunks = self._hdf_store.chunks

                    # Might be best NOT to call this to preserve previous data
                    self.images = da.store(self.images,
                                           self._hdf_store,
                                           compute=True,
                                           return_stored=True)[0]
                else:
                    self._hdf_store = None
                    print(('WARNING: Dask Enabled \n'
                          + 'A temporary hdf storage dataset will be '
                          + 'generated when applying the first '
                          + 'correction: dark_field.'))
    
        self.ai = ai
        self.sclr_dict = sclr_dict


    def __str__(self):
        ostr = f'XRDData: ({self.shape}), dtype={self.dtype}'
        return ostr

    
    def __repr__(self):
        ostr = f'XRDData:'
        ostr += f'\n\tshape:  {self.shape}'
        ostr += f'\n\tdtype:  {self.dtype}'
        ostr += f'\n\tstate:  {self.title}'
        return ostr
    
    #################################################
    ### Properties and Internal Utility Functions ###
    #################################################

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        # Update datatype if different
        if dtype != self._dtype:
            if (hasattr(self, 'images')
                and self.images is not None):
                # Unfortunately, this has to copy the dataset
                self.images = self.images.astype(dtype)
                self._dtype = dtype
            if (hasattr(self, 'integrations')
                and self.integrations is not None):
                self.integrations = self.integrations.astype(dtype)
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
                if np.any(self.mask != 1) and axes != (0, 1):

                    # Mask away discounted pixels
                    val = function(self.images[..., self.mask],
                                   axis=-1).astype(self.dtype)

                    if self._dask_enabled:
                        val = val.compute()

                    setattr(self, property_name, val)
                else:
                    # dtype may cause overflow errors
                    val = function(self.images,
                                   axis=axes).astype(self.dtype)
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

    # Will not be accurate at default dtype of np.uint16
    mean_map = projection_factory('mean_map', np.mean, (2, 3))
    mean_image = projection_factory('mean_image', np.mean, (0, 1))

    # May cause overflow errors
    sum_map = projection_factory('sum_map', np.sum, (2, 3))
    sum_image = projection_factory('sum_image', np.sum, (0, 1))


    # Adaptively saves for whatever the current processing state
    # Add other methods, or rename to something more intuitive
    @property
    def composite_image(self):
        if hasattr(self, f'_composite_image'):
            return getattr(self, f'_composite_image')
        else:
            setattr(self, f'_{self.title}_composite_image',
                    self.max_image - self.min_image)
            
            # Set the generic value to this as well
            self._composite_image = getattr(self,
                                        f'_{self.title}_composite_image')

            # Save image to hdf. Should update if changed
            self.save_images(images='_composite_image',
                             title=f'_{self.title}_composite_image')

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
                warn_str = ('WARNING: Calibration mask found, but '
                            + 'shape does not match images.')
                print(warn_str)

        # Remove image defects
        if hasattr(self, 'defect_mask'):
            if self.defect_mask.shape == mask.shape:
                mask *= self.defect_mask
            else:
                warn_str = ('WARNING: Defect mask found, but shape '
                            + 'does not match images.')
                print(warn_str)

        if hasattr(self, 'custom_mask'):
            if self.custom_mask.shape == mask.shape:
                mask *= self.custom_mask
            else:
                warn_str = ('WARNING: Custom mask found, but shape '
                            + 'does not match images.')
                print(warn_str)

        return mask


    # Function to dump accummulated processed images and maps
    # Not sure if will be needed between processing the full map
    def reset_projections(self):
        old_attr = list(self.__dict__.keys())       
        for attr in old_attr:
            if attr in ['_composite_image',
                        '_min_map', '_min_image',
                        '_max_map', '_max_image',
                        '_med_map', '_med_image',
                        '_sum_map', '_sum_image',
                        '_mean_map', '_mean_image',
                        '_composite_integration',
                        '_min_integration_map',
                        '_min_integration',
                        '_max_integration_map',
                        '_max_integration',
                        '_med_integration_map',
                        '_med_integration',
                        '_sum_integration_map',
                        '_sum_integration',
                        '_mean_integration_map',
                        '_mean_integration']:
                delattr(self, attr)


    def update_map_title(self, title=None):
        # The title will never be able to capture everything
        # This is intended to only capture major changes
        # Will be used to create new groups when saving to hdf
        if title is not None:
            self.title = title
    
        elif np.all(~np.array(list(self.corrections.values()))):
            self.title = 'raw' # This should not be needed...

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
                            self.title = 'calibrated'
        else:
            # No update
            pass
        # Clear old values everytime this is checked
        self.reset_projections()


    # # Decorator to protect self.hdf during read/write functions
    # def protect_hdf(func):
    #     @functools.wraps(func)
    #     def protector(self, *args, **kwargs):
    #         # Check to see if read/write is enabled
    #         if self.hdf_path is not None:
                
    #             # Log if a reference in maintined to hdf
    #             make_temp_hdf = self.hdf is None
    #             if make_temp_hdf:
    #                 self.hdf = h5py.File(self.hdf_path, 'a')

    #             # Call function
    #             func(self, *args, **kwargs)

    #             # Remove temporary reference
    #             if make_temp_hdf:
    #                 self.hdf.close()
    #                 self.hdf = None

    #     return protector


    # Opens and closes hdf to esnure self.hdf can be used safely
    def protect_hdf(pandas=False):
        def protect_hdf_inner(func):
            @functools.wraps(func)
            def protector(self, *args, **kwargs):
                # Check to see if read/write is enabled
                if self.hdf_path is not None:
                    # Is a hdf reference currently active?
                    active_hdf = self.hdf is not None

                    if pandas: # Fully close reference
                        self.close_hdf()
                    elif not active_hdf: # Make temp reference
                        self.hdf = h5py.File(self.hdf_path, 'a')

                    # Call function
                    func(self, *args, **kwargs)
                    
                    # Clean up hdf state
                    if pandas and active_hdf:
                        self.open_hdf()
                    elif not active_hdf:
                        self.close_hdf()
            return protector
        return protect_hdf_inner

    
    @protect_hdf()
    def load_images_from_hdf(self,
                             image_data_key,
                             chunks=None):
        # Deletes current image map and loads new values from hdf
        print(f'Loading {image_data_key}')
        
        # Working with dask flag
        dask_enabled = self._dask_enabled

        # Actually load the data
        image_grp = self.hdf[f'{self._hdf_type}/image_data']
        if check_hdf_current_images(image_data_key, hdf=self.hdf):
            # Delete previous images from XRDData to save memory
            del(self.images) 
            img_dset = image_grp[image_data_key]
            
            if dask_enabled:
                self.images = da.asarray(img_dset)
            else:
                self.images = np.asarray(img_dset)

            # Rebuild correction dictionary
            corrections = {}
            for key in image_grp[image_data_key].attrs.keys():
                corrections[key[1:-11]] = image_grp[
                                            image_data_key].attrs[key]
            self.corrections = corrections

            if close_hdf_on_finish:
                self.hdf.close()
                self.hdf = None
            # Define/determine chunks and rechunk if necessary
            if isinstance(self.images, da.core.Array):
                # Redo chunking along image dimensions if not already
                if (self.images.chunksize[-2:]
                    != self.images.shape[-2:]):
                    self._get_optimal_chunks()
                    self.images = self.images.rechunk(
                                        chunks=self._chunks)
                else:
                    self._chunks = self.images.chunksize
            else:
                if chunks is not None:
                    self._chunks = chunks
                else:
                    self._get_optimal_chunks()

        # Force title; truncate '_images'
        self.update_map_title(title=image_data_key[:-7]) 
        self._dask_2_hdf()

        # Update useful data
        self._dtype = self.images.dtype
        self.shape = self.images.shape
        self.map_shape = self.shape[:2]
        self.image_shape = self.shape[-2:]
        self.num_images = np.prod(self.map_shape)
    

    def dump_images(self):
        del self.images


    ######################
    ### Dask Functions ###
    ######################
        
    # Flag to check if images are dask or numpy array
    @property
    def _dask_enabled(self):
        return isinstance(self.images, da.core.Array)


    # Return standardized chunk size around images for hdf and dask
    def _get_optimal_chunks(self, approx_chunk_size=None):
        if hasattr(self, 'images') and self.images is not None:
            self._chunks =  get_optimal_chunks(
                                self.images,
                                approx_chunk_size=approx_chunk_size)
        else:
            self._chunks = (*self.map_shape, *self.image_shape)
        return self._chunks
    

    def _dask_2_numpy(self):
        # Computes dask to numpy array. Intended for final images
        if self._dask_enabled:
            self.images = self.images.compute()


    def _numpy_2_dask(self):
        # Computes numpy array into dask.
        if not self._dask_enabled:
            if self.hdf is None:
                if self.hdf_path is None:
                    err_str = ('Cannot convert images from numpy to '
                               + 'dask without specifying hdf file!')
                    raise RuntimeError(err_str)
                else:
                    err_str = ('Trying to convert images to dask, '
                               + 'but hdf file has not been '
                               + 'opened with hdf_path.')
                    raise RuntimeError(err_str)
            else:
                self._dask_2_hdf()
                self.images = da.from_array(self.images)


    def _dask_2_dask(self):
        # Computes and updates dask array to
        # avoid too many lazy computations
        # Will have faster compute times than _dask_2_hdf()
        if self._dask_enabled:
            self.images = self.images.persist()
            

    def _dask_2_hdf(self):
        # Computes and stores current iteration of 
        # lazy computation to hdf file
        # Probably the most useful
        if self.hdf is not None and self._dask_enabled:
            if self.title == 'final':
                warn_str = ('WARNING: Images cannot be updated when '
                            + 'they have already been finalized.'
                            + '\nProceeding without updating images.')
                print(warn_str) # I am not sure about this. What is the
                                # point of linking final_images for 
                                # storage if not used???
            else:
                self.images = da.store(self.images,
                                       self._hdf_store,
                                       compute=True,
                                       return_stored=True)[0]


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
            warn_str = ('WARNING: hdf is already open. '
                        + 'Proceeding without changes.')
            print(warn_str)
            return
        else:
            self.hdf = h5py.File(self.hdf_path, 'a')

        # This flag persists even when the dataset is closed!
        if dask_enabled or self._dask_enabled: 
            img_grp = self.hdf[f'{self._hdf_type}/image_data']
            if self.title == 'final':
                if check_hdf_current_images(f'{self.title}_images',
                                            hdf=self.hdf):
                    dset = img_grp[f'{self.title}_images']
            elif check_hdf_current_images('_temp_images',
                                          hdf=self.hdf):
                dset = img_grp['_temp_images']
            self.images = da.asarray(dset) # I had .persist(), but it broke things...
            self._hdf_store = dset


    ########################################
    ### Image Corrections and Transforms ###
    ########################################

    def _check_correction(self, correction, override=False):
        apply_correction = True
        if self.title == 'final':
            warn_str = f'WARNING: XRDData has been finalized!'
            apply_correction = False
        elif self.corrections[correction]:
            warn_str = (f'WARNING: {correction} correction '
                        + 'already applied!')
            apply_correction = False
        
        if apply_correction:
            return False
        elif override:
            warn_str += ('\nOverriding warning and correcting '
                         + f'{correction} anyway.')
            print(warn_str)
            return False
        else:
            warn_str += f'\nProceeding without changes.'
            print(warn_str)
            return True


    ### Initial image corrections ###

    def correct_dark_field(self,
                           dark_field=None,
                           override=False):

        if self._check_correction('dark_field', override=override):
            return
        elif dark_field is None:
            print('No dark-field given for correction.')
        elif dark_field.shape != self.image_shape:
            err_str = (f'Dark-field shape of {dark_field.shape} does '
                       + 'not match image shape of 
                       + f'{self.image_shape}.')
            raise ValueError(err_str)
        
        self.dark_field = dark_field

        # Check for the start of corrections with dask.
        # Otherwise keep lazily loaded...
        if self._dask_enabled and self._hdf_store is None:
            print(('Dask enabled. Upcasting data and generating a '
                   + 'temporary dataset for performing corrections.\n'
                   + 'This may take a while...'))

            # Upcast before writing to hdf
            self.images = self.images.astype(np.float32)
            self._hdf_store = self.hdf.require_dataset(
                        f'{self._hdf_type}/image_data/_temp_images',
                        shape=self.images.shape,
                        dtype=np.float32,
                        chunks=self._chunks,
                        compression_opts=4,
                        compression='gzip') # This may slow it down

            # Might be best NOT to call this to preserve previous data
            self.images = da.store(self.images,
                                   self._hdf_store,
                                   compute=True,
                                   return_stored=True)[0]
        
        else:
            # Check for upcasting. Will probably upcast data

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

        self.save_images(images='dark_field',
                         units='counts')
        
        self.corrections['dark_field'] = True
        self.update_map_title()
        self._dask_2_hdf()
        print('done!')


    # TODO: This fails if image medians are too close to zero 
    def correct_flat_field(self,
                           flat_field=None,
                           override=False):

        if self._check_correction('flat_field', override=override):
            return
        elif flat_field is None:
            print('No flat-field correction.')
        elif flat_field.shape != self.image_shape:
            err_str = (f'Flat-field shape of {flat_field.shape} does '
                       + 'not match image shape of '
                       + f'{self.image_shape}.')
            raise ValueError(err_str)
        else:
            self.dtype = np.float32
            print('Correcting flat-field...', end='', flush=True)
            
            # Shift flat_field correction to center image map center
            if self._dask_enabled:
                correction = (np.mean(self.images)
                              - np.mean(flat_field))
            else:  # Dask does not handle medians very well...
                correction = (np.median(self.images)
                              - np.mean(flat_field))
            
            self.flat_field = flat_field + correction
            self.images /= self.flat_field

            self.save_images(images='flat_field')
            
            self.corrections['flat_field'] = True
            self.update_map_title()
            self._dask_2_hdf()
            print('done!')


    def correct_air_scatter(self,
                            air_scatter=None,
                            applied_corrections=None,
                            override=False):

        if self._check_correction('air_scatter', override=override):
            return
        elif air_scatter is None:
            print('No air_scatter given for correction.')
        elif air_scatter.shape != self.image_shape:
            err_str = (f'air_scatter shape of {air_scatter.shape} does'
                       + ' not match image shape of '
                       + f'{self.image_shape}.')
            raise ValueError(err_str)
        
        if applied_corrections is None:
            applied_corrections = {key:False
                                   for key in self.corrections.keys()}
        else:
            for key in applied_corrections.keys():
                if key not in self.corrections.keys():
                    err_str = ('Unknown correction in '
                               + f'applied_corrections: {key}')
                    raise RuntimeError(err_str)

        # Check for applied corrections 
        # which cannot be applied to air_scatter
        disallowed_corrections = [
            'background',
            'absorption'
        ]
        for key in disallowed_corrections:
            if self.corrections[key]:
                warn_str = (f'WARNING: air_scatter cannot be '
                            + f'corrected after {key} correction '
                            + 'has been applied.'
                            + '\nProceeding without changes.')
                print(warn_str)
                return
        
        # Check for corrections which can be applied to air_scatter
        allowed_corrections = [
            'dark_field',
            'flat_field',
            'lorentz',
            'polarization',
            'solid_angle'
        ]
        
        print('Correcting air scatter...', end='', flush=True)
        if (self.corrections['dark_field']
            and not applied_corrections['dark_field']):
            air_scatter -= self.dark_field

        if (self.corrections['flat_field']
            and not applied_corrections['flat_field']):
            air_scatter /= self.flat_field
        
        if (self.corrections['lorentz']
            and not applied_corrections['lorentz']):
            air_scatter /= self.lorentz_correction
        
        if (self.corrections['polarization']
            and not applied_corrections['polarization']):
            air_scatter /= self.polarization_correction
        
        if (self.corrections['solid_angle']
            and not applied_corrections['solid_angle']):
            air_scatter /= self.solidangle_correction

        self.air_scatter = air_scatter
        self.images -= self.air_scatter

        self.save_images(images='air_scatter')
        
        self.corrections['air_scatter'] = True
        self.update_map_title()
        self._dask_2_hdf()
        print('done!')

        
    def correct_outliers(self,
                         size=2,
                         tolerance=2,
                         override=False):

        if self._check_correction('outliers', override=override):
            return
        print('Finding and correcting image outliers...')
        # self.images = find_outlier_pixels(self.images,
        #                                   size=size,
        #                                   tolerance=tolerance)
        num_pixels_replaced = iterative_outlier_correction(
                                    self.images,
                                    size=size, 
                                    tolerance=tolerance)
        
        self.corrections['outliers'] = True
        self.update_map_title()
        self._dask_2_hdf()
        perc_corr = (num_pixels_replaced / self.images.size) * 100
        ostr = (f'Done! Replaced {num_pixels_replaced} '
                + f'({perc_corr:.2f} %) outlier pixels.')
        print(ostr)


    # No correction for defect mask,
    # since it is used whenever mask is called
    def apply_defect_mask(self,
                          min_bounds=(-np.inf, 0),
                          max_bounds=(0, np.inf),
                          mask=None,
                          override=False):
        
        if self._check_correction('pixel_defects', override=override):
            return

        if mask is not None:
            self.defect_mask = np.asarray(mask).astype(np.bool_)
        else:
            mask = np.ones_like(self.min_image, dtype=np.bool_)
            mask *= ((self.min_image >= min_bounds[0])
                     & (self.min_image <= min_bounds[1]))
            mask *= ((self.max_image >= max_bounds[0])
                     & (self.max_image <= max_bounds[1]))
            self.defect_mask = mask

        # Write mask to disk
        self.save_images(images='defect_mask') 
        
        self.corrections['pixel_defects'] = True
        self.update_map_title()
        self._dask_2_hdf()


    # No correction for custom mask,
    # since it is used whenever mask is called
    def apply_custom_mask(self, mask=None):
        if mask is not None:
            self.custom_mask = np.asarray(mask).astype(np.bool_)
            # Write mask to disk
            self.save_images(images='custom_mask')
        else:
            print('No custom mask provided!')


    ### Geometric corrections ###
    # TODO: Add conditionals to allow corrections
    # to be applied to calibrated images
    def apply_lorentz_correction(self,
                                 powder=False,
                                 apply=True,
                                 override=False):

        if self._check_correction('lorentz', override=override):
            return

        # In radians
        tth_arr = self.ai.twoThetaArray().astype(self.dtype)

        # Static area detector. Seems to work well
        lorentz_correction = 1 / np.sin(tth_arr / 2)

        if powder:
            lorentz_correction *= np.cos(tth_arr / 2)
        
        # Save and apply Lorentz corrections
        self.lorentz_correction = lorentz_correction
        self.save_images(images='lorentz_correction')
    
        if apply:
            print('Applying Lorentz correction...', end='', flush=True)
            self.images /= self.lorentz_correction
            self.corrections['lorentz'] = True
            self.update_map_title()
            self._dask_2_hdf()
            print('done!')

    
    def apply_polarization_correction(self,
                                      polarization=0.9,
                                      apply=True,
                                      override=False):

        if self._check_correction('polarization', override=override):
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
        polar = self.ai.polarization(
                    factor=polarization).astype(self.dtype)
        self.polarization_correction = polar
        self.save_images(images='polarization_correction')
        
        if apply:
            print('Applying X-ray polarization correction...',
                  end='', flush=True)
            self.images /= self.polarization_correction
            self.corrections['polarization'] = True
            self.update_map_title()
            self._dask_2_hdf()
            print('done!')

    
    def apply_solidangle_correction(self,
                                    apply=True,
                                    override=False):

        if self._check_correction('solid_angle', override=override):
            return

        # TODO: Does not work with already calibrated images
        #tth_arr = ai.twoThetaArray()
        #chi_arr = ai.chiArray()

        # From pyFAI
        # 'SA = pixel1 * pixel2 / dist^2 * cos(incidence)^3'

        # From pyFAI
        solidangle_correction = self.ai.solidAngleArray().astype(
                                                    self.dtype)
        self.solidangle_correction = solidangle_correction
        self.save_images(images='solidangle_correction')
        
        if apply:
            print('Applying solid angle correction...',
                   end='', flush=True)
            self.images /= self.solidangle_correction
            self.corrections['solid_angle'] = True
            self.update_map_title()
            self._dask_2_hdf()
            print('done!')


    # WIP
    def apply_absorption_correction(self,
                                    exp_dict,
                                    apply=True,
                                    override=False):

        #exp_dict = {
        #    'attenuation_length' : value,
        #    'mode' : 'transmission',
        #    'thickness' : value,
        #    'theta' : value # in degrees!
        #}

        if self._check_correction('absorption', override=override):
            return
        
        # In radians
        tth_arr = self.ai.twoThetaArray()
        chi_arr = self.ai.chiArray()

        if not all(x in list(exp_dict.keys()) for x in [
            'attenuation_length',
            'mode',
            'thickness',
            'theta'
            ]):
            
            err_str = ("Experimental dictionary does not have all the "
                       + "necessary keys: 'attenuation_length', 'mode'"
                       + ", and 'thickness'.")
            raise ValueError(err_str)

        # Semi-infinite plate
        if exp_dict['mode'] == 'transmission':
            t = exp_dict['thickness']
            a = exp_dict['attenuation_length']
            theta = np.radians(exp_dict['theta'])
            if theta == 0:
                x = t / np.cos(tth_arr) # path length
            else:
                # OPTIMIZE ME!
                # Intersection coordinates of lines. 
                # One for the far surface and another 
                # for the diffracted beam
                # Origin at intersection of initial surface 
                # and transmitted beam
                # y1 = x1 / np.tan(tth) # diffracted beam
                # y2 = np.cos(chi) * np.tan(theta) + t / np.cos(theta) # far surface
                xi = (t / (np.cos(theta))) / ((1 / np.tan(tth_arr)) - (np.cos(chi_arr) * np.tan(theta)))
                yi = xi / (np.tan(tth_arr))
                # Distance of intersection point
                x = np.sqrt(xi**2 + yi**2)

            abs_arr = np.exp(-x / a)
            self.absorption_correction = abs_arr
            # Not sure about saving this...
            self.save_images(images='absorption_correction')

        elif exp_dict['mode'] == 'reflection':
            raise NotImplementedError()
        else:
            raise ValueError(f"{exp_dict['mode']} is unknown.")
        
        if apply:
            print('Applying absorption correction...',
                  end='', flush=True)
            self.images /= self.absorption_correction
            self.corrections['absorption'] = True
            self.update_map_title()
            self._dask_2_hdf()
            print('done!')


    def normalize_scaler(self,
                         scaler_arr=None,
                         override=False):

        if self._check_correction('scaler_intensity', override=override):
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
                    warn_str = ("WARNING: Unrecognized scaler keys. "
                                + f"Using '{first_key}' instead.")
                    print(err_str)
                    sclr_key = first_key
            else:
                ostr = ('No scaler array given or found. '
                        + 'Approximating with image medians.')
                print(ostr)
                scaler_arr = self.med_map
                sclr_key = 'med'

        elif scaler_arr.shape != self.map_shape:
            err_str = (f'Scaler array shape of {scaler_arr.shape} does'
                      + f' not match map shape of {self.map_shape}.')
            raise ValueError(err_str)
        else:
            sclr_key = 'input'

        # Check shape everytime
        scaler_arr = np.asarray(scaler_arr)
        if scaler_arr.shape != self.map_shape:
            if 1 not in self.map_shape:
                err_str = (f'Scaler array of shape {scaler_arr.shape}'
                           + f' does not match the map shape of '
                           + f'{self.map_shape}.')
                raise ValueError(err_str)
   
        print(f'Normalizing images by {sclr_key} scaler...',
              end='', flush=True)
        self.images /= scaler_arr.reshape(*self.map_shape, 1, 1)
        self.scaler_map = scaler_arr
        # Trying to catch non-saved values
        if not hasattr(self, 'sclr_dict'): 
            self.save_images(images='scaler_map',
                             units='counts',
                             labels=self.map_labels) 
        self.corrections['scaler_intensity'] = True
        self.update_map_title()
        self._dask_2_hdf()
        print('done!')


    # TODO: Not all of these have been enabled with dask arrays
    def estimate_background(self,
                            method=None,
                            background=None,
                            **kwargs):
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
                err_str = ('Cannot yet exclude contribution from '
                           + 'masked regions.')
                raise NotImplementedError(err_str)
                ostr = ('Estimating background with '
                        + 'rolling ball method.')
                print(ostr)
                self.background = rolling_ball(self.images, **kwargs)
                self.background_method = 'rolling ball'

            elif method in ['spline', 'spline fit', 'spline_fit']:
                print('Estimating background with spline fit.')
                self.background = fit_spline_bkg(self, **kwargs)
                self.background_method = 'spline'

            elif method in ['poly', 'poly fit', 'poly_fit']:
                print('Estimating background with polynomial fit.')
                warn_str = ('WARNING: This method is slow and '
                            + 'not very accurate.')
                print(warn_str)
                self.background = fit_poly_bkg(self, **kwargs)
                self.background_method = 'polynomial'

            elif method in ['Gaussian', 'gaussian', 'gauss']:
                ostr = ('Estimating background with gaussian '
                        + 'convolution.\nNote: Progress bar is '
                        + 'unavailable for this method.')
                print(ostr)
                self.background = masked_gaussian_background(self,
                                                             **kwargs)
                self.background_method = 'gaussian'

            elif method in ['Bruckner', 'bruckner']:
                print('Estimating background with Bruckner algorithm.')
                self.background = masked_bruckner_background(self,
                                                             **kwargs)
                self.background_method = 'bruckner'

            elif method in ['none']:
                print('No background correction will be used.')
                self.background = None
                self.background_method = 'none'
            
            else:
                err_str = f"Method '{method}' not implemented!"
                raise NotImplementedError(err_str)
    
        else:
            print('User-specified background.')
            self.background = background
            self.background_method = 'custom'
    

    def remove_background(self,
                          background=None,
                          save_images=False,
                          override=False):

        if self._check_correction('background', override=override):
            if background is None and hasattr(self, 'background'):
                warn_str = ('WARNING: background attribute still '
                            + 'saved in memory.\nOverride background '
                            + 'remove or delete attribute to '
                            + 'release memory.')
            print(warn_str)
            return
        
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
            self.save_images(images='background',
                             title='static_background')
        else:
            # Delete full map background to save on memory
            del self.background

        self.corrections['background'] = True
        self.update_map_title()
        self._dask_2_hdf()
        print('done!')

        if save_images:
            ostr = ('Compressing and writing images to disk.'
                    + '\nThis may take a while...')
            print(ostr)
            self.save_images(extra_attrs={'background_method'
                                          : self.background_method})
            print('done!')


    def get_polar_mask(self,
                       tth_num=None,
                       chi_num=None,
                       units='2th_deg'):

        if tth_num is None:
            tth_num = self.tth_num
        if chi_num is None:
            chi_num = self.chi_num

        dummy_image = 100 * np.ones(self.image_shape)

        image, _, _ = self.ai.integrate2d_ng(dummy_image,
                                             tth_num,
                                             chi_num,
                                             unit=units)

        calibration_mask = (image != 0)

        return calibration_mask
    

    def rescale_images(self,
                       lower=0,
                       upper=100,
                       arr_min=0,
                       arr_max=None,
                       mask=None):

        if mask is None and np.any(self.mask != 1):
            mask = np.empty_like(self.images, dtype=np.bool_)
            mask[:, :] = self.mask
        
        rescale_array(
            self.images,
            lower=lower,
            upper=upper,
            arr_min=arr_min,
            arr_max=arr_max,
            mask=mask)
        
        # Update _temp images
        self._dask_2_hdf()
        self.reset_projections()


    # For estimating maximum saturated pixel
    # for comparison with other datasets
    def estimate_saturated_pixel(self,
                                 # Saturated value from detector 
                                 raw_max_val=(2**14 - 1), 
                                 method='median'):

        # Usually better statistics 
        if method.lower() in ['median', 'med']: 
            var_func = np.median
        # Better guesses true possible maximum,
        # even if unlikely in dataset
        if method.lower() in ['minimum', 'min']: 
            var_func = np.min
        
        if self.corrections['dark_field']:
            raw_max_val -= np.median(self.dark_field)
        if self.corrections['flat_field']:
            raw_max_val /= np.median(self.flat_field)
        # if self.corrections['air_scatter']:
        #     raw_max_val -= np.min(self.air_scatter)
        
        if self.corrections['scaler_intensity']:
            if hasattr(self, 'scaler_map') and self.scaler_map is not None:
                scaler_map = self.scaler_map
            elif hasattr(self, 'sclr_dict'):
                for key in ['i0', 'im']:
                    if key in self.sclr_dict.keys():
                        scaler_map = self.sclr_dict[key]
                        break
                    else:
                        # Hope for the best
                        scaler_map = list(self.sclr_dict.values())[0] 
            else:
                err_str = ('Not enough information to estimate '
                           + 'scaler contribution!')
                raise ValueError(err_str)
            raw_max_val /= np.median(scaler_map)

        if self.corrections['lorentz']:
            raw_max_val /= var_func(self.lorentz_correction)
        if self.corrections['polarization']:
            raw_max_val /= var_func(self.polarization_correction)
        if self.corrections['solid_angle']:
            raw_max_val /= var_func(self.solidangle_correction)
        if self.corrections['absorption']:
            raw_max_val /- var_func(self.absorption_correction)
        # Assuming the minimum background will be very close to zero
        if self.corrections['background']:
            (if hasattr(self, 'background')
             and self.background is not None):
                raw_max_val -= var_func(self.background)
        # All other corrections are isolated within the image

        return raw_max_val


    def construct_null_map(self, override=False):
        if (not override
            and hasattr(self, 'null_map')
            and self.null_map is not None):
            ostr = ('Null map already exists. '
                    + 'Proceeding without changes.')
            print(ostr)
            return
        else:
            if self.title == 'raw':
                if not self._dask_enabled or override:
                    raw_images = self.images # Should not be a copy
                else:
                    warn_str = ('WARNING: Dask is enabled. Set '
                                + 'override to True in order to '
                                + 'determine null_map.'
                                + '\nProceeding without changes.')
                    print(warn_str)
                    return

            else:
                hdf_str = f'{self._hdf_type}/image_data/raw_images'
                if self.hdf is not None:
                    raw_images = self.hdf[hdf_str]
                elif self.hdf_path is not None:
                    f = h5py.File(self.hdf_path, 'r')
                    hdf_str = 
                    raw_images = f[hdf_str]
                else:
                    err_str = ('Cannot determine null_map without '
                               + 'raw_images or access to hdf.')
                    raise RuntimeError(err_str)
            
            # Try to do this efficiently
            null_map = np.ones(self.map_shape, dtype=np.bool_)
            for index in range(self.num_images):
                indices = np.unravel_index(index, self.map_shape)
                if np.any(raw_images[indices] != 0):
                    null_map[indices] = False
            
            self.null_map = np.asarray(null_map)
            self.save_images(images='null_map',
                             units='bool',
                             labels=self.map_labels)

    
    def nullify_images(self):
        if not hasattr(self, 'null_map'):
            err_str = 'XRDData does not have null_map attribute.'
            raise AttributeError(err_str)
        elif not np.any(self.null_map):
            note_str = ('Null map is empty, there are no missing '
                        + 'pixels. Proceeding without changes')
            print(note_str)
            return

        self.images[self.null_map] = 0

        for attr in ['images', 'blob_masks', 'integrations']:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                # This should all be done in place
                getattr(self, attr)[self.null_map] = 0
        

    def finalize_images(self, save_images=True):

        if np.any(self.corrections.values()):
            print('Caution: Images not corrected for:')
        # Check corrections:
        for key in sorted(list(self.corrections.keys())):
            if not self.corrections[key]:
                print(f'\t{key}')

        # Some cleanup
        print('Cleaning and updating image information...')
        self._dask_2_hdf()
        self.update_map_title(title='final')
        self.disk_size()

        if save_images:
            if self.hdf_path is None and self.hdf is None:
                ostr = ('No hdf file specified. '
                        + 'Images will not be saved.')
                print(ostr)
            else:
                ostr = ('Compressing and writing images to disk.'
                        + '\nThis may take a while...')
                print(ostr)

                if check_hdf_current_images('_temp_images',
                                            self.hdf_path,
                                            self.hdf):
                    # Save images to current store location. 
                    # Should be _temp_images
                    self.save_images(title='_temp_images')
                    hdf_str = (f'{self._hdf_type}/image_data'
                               + '/_temp_images')
                    temp_dset = self.hdf[hdf_str]
                    
                    # Must be done explicitly outside of self.save_images()
                    for key, value in self.corrections.items():
                        temp_dset.attrs[f'_{key}_correction'] = value

                    # Relabel to final images and del temp dataset
                    hdf_str = (f'{self._hdf_type}/image_data'
                               + '/final_images')
                    self.hdf[hdf_str] = temp_dset
                    del temp_dset

                    # Remap store location. Should only reference data from now on
                    self._hdf_store = self.hdf[hdf_str]

                else:
                    self.save_images()
                print('done!')


    #######################
    ### Integrated Data ###
    #######################
    # Dask is not implemented for integrations...

    ### 1D integration projections ###

    # Simpler from image_projection_factor
    # because no mask is considered
    def integration_projection_factory(property_abbreviation,
                                       function, axes):
        # This line is crucial!
        property_name = f'_{property_abbreviation}'  
        def get_projection(self):
            if hasattr(self, property_name):
                return getattr(self, property_name)
            else:
                if (not hasattr(self, 'integrations')
                    or self.integrations is None):
                    err_str = ('Cannot determine integration '
                               + 'projection without integrations.')
                    raise AttributeError(err_str)
                projection = function(self.integrations, axis=axes)
                setattr(self, property_name, projection)
                return getattr(self, property_name)

        def del_projection(self):
            delattr(self, property_name)
        
        return property(get_projection, None, del_projection)
    
    min_integration = integration_projection_factory(
                            'min_integration', np.min, (0, 1))
    min_integration_map = integration_projection_factory(
                            'min_integration_map', np.min, (2))

    max_integration = integration_projection_factory(
                            'max_integration', np.max, (0, 1))
    max_integration_map = integration_projection_factory(
                            'max_integration_map', np.max, (2))

    sum_integration = integration_projection_factory(
                            'sum_integration', np.sum, (0, 1))
    sum_integration_map = integration_projection_factory(
                            'sum_integration_map', np.sum, (2))

    med_integration = integration_projection_factory(
                            'med_integration', np.median, (0, 1))
    med_integration_map = integration_projection_factory(
                            'med_integration_map', np.median, (2))

    mean_integration = integration_projection_factory(
                            'mean_integration', np.mean, (0, 1))
    mean_integration_map = integration_projection_factory(
                            'mean_integration_map', np.mean, (2))


    @property
    def composite_integration(self):
        if hasattr(self, f'_composite_integration'):
            return getattr(self, f'_composite_integration')
        else:
            setattr(self, f'_{self.title}_composite_integration',
                    self.max_integration - self.min_integration)
            
            # Set the generic value to this as well
            self._composite_integration = getattr(self,
                                f'_{self.title}_composite_integration')

            # Save image to hdf. Should update if changed
            self.save_integrations(
                    integrations='_composite_integration',
                    title=f'_{self.title}_composite_integration')

            # Finally return the requested attribute
            return getattr(self, f'_composite_integration')
        
    @composite_integration.deleter
    def composite_integration(self):
        delattr(self, '_composite_integration')

    ### 1D integration corrections
    
    def estimate_integration_background(self,
                                        method=None,
                                        background=None,
                                        **kwargs):
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
                raise NotImplementedError('Shape issues...')
                ostr = ('Estimating background with '
                        + 'rolling ball method.')
                print(ostr)
                self.integration_background = rolling_ball(
                                                self.integrations,
                                                **kwargs)
                self.integration_background_method = 'rolling ball'

            elif method in ['spline', 'spline fit', 'spline_fit']:
                err_str = ('Still need to write function '
                           + 'for integrations.')
                raise NotImplementedError(err_str)
                print('Estimating background with spline fit.')
                self.integration_background = fit_spline_bkg(self,
                                                             **kwargs)
                self.integration_background_method = 'spline'

            elif method in ['poly', 'poly fit', 'poly_fit']:
                err_str = ('Still need to write '
                           + 'function for integrations.')
                raise NotImplementedError(err_str)
                warn_str = ('Estimating background with polynomial '
                            + 'fit.\nWARNING: This method is slow and '
                            + 'not very accurate.')
                print(warn_str)
                self.integration_background = fit_poly_bkg(self,
                                                           **kwargs)
                self.integration_background_method = 'polynomial'

            elif method in ['Gaussian', 'gaussian', 'gauss']:
                err_str = ('Still need to write '
                           + 'function for integrations.')
                raise NotImplementedError(err_str)
                ostr = ('Estimating background with gaussian '
                        + 'convolution.\nNote: Progress bar is '
                        + 'unavailable for this method.')
                print(ostr)
                (self.integration_background
                ) = masked_gaussian_background(self, **kwargs)
                self.integration_background_method = 'gaussian'

            elif method in ['Bruckner', 'bruckner']:
                err_str = ('Still need to write '
                           + 'function for integrations.')
                raise NotImplementedError(err_str)
                print('Estimating background with Bruckner algorithm.')
                (self.integration_background
                ) = masked_bruckner_background(self, **kwargs)
                self.integration_background_method = 'bruckner'

            elif method in ['none']:
                print('No background correction will be used.')
                self.integration_background = None
                self.integration_background_method = 'none'
            
            else:
                err_str = f"Method '{method}' not implemented!"
                raise NotImplementedError(err_str)
    
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
    

    def rescale_integrations(self,
                             lower=0,
                             upper=100,
                             arr_min=0,
                             arr_max=None):
        
        rescale_array(
            self.integrations,
            lower=lower,
            upper=upper,
            arr_min=arr_min,
            arr_max=arr_max)


    ####################
    ### IO Functions ###
    ####################

    def disk_size(self, return_val=False, dtype=None):
        # Return current map size which
        # should be most of the memory usage
        # Helps to estimate file size too
        if dtype is None:
            byte_size = self.images.itemsize
        else:
            try:
                byte_size = dtype().itemsize
            except TypeError as e:
                err_str = (f'dtype input of {dtype} is not a '
                           + 'numpy datatype.')
                raise e(err_str)

        disk_size = self.images.size * byte_size
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
        
        if return_val:
            return disk_size
        print(f'Diffraction map size is {disk_size:.3f} {units}.')


    # WIP apparently???
    @staticmethod
    def estimate_disk_size(size):
        # External reference function to estimate map size before acquisition
        raise NotImplementedError()

        if not isinstance(size, (tuple, list, np.ndarray)):
            err_str = ('Size argument must be iterable of '
                       + 'map dimensions.')
            raise TypeError(err_str)

        disk_size = np.prod([*size, 2])


    def _get_save_labels(self,
                         arr_shape,
                         base_labels=None,
                         units='a.u.'):
        
        if base_labels is None:
            base_labels = self.map_labels

        if len(arr_shape) == 4: # Image map
            labels = base_labels
            if self.corrections['polar_calibration']:
                labels.extend(['chi_ind', 'tth_ind'])
            else:
                labels.extend(['img_y', 'img_x'])

        elif len(arr_shape) == 2: # Image
            labels = []
            if self.corrections['polar_calibration']:
                labels.extend(['chi_ind', 'tth_ind'])
            else:
                labels.extend(['img_y', 'img_x'])

        elif len(arr_shape) == 3: # Integrations
            labels = base_labels + ['tth_ind']

        return units, labels


    @protect_hdf()
    def save_images(self,
                    images=None,
                    title=None,
                    units=None,
                    labels=None,
                    compression=None,
                    compression_opts=None,
                    mode='a',
                    extra_attrs=None):
        
        # Save all images
        if images is None:
            if not hasattr(self, 'images') or self.images is None:
                err_str = 'Must provide images to write to hdf.'
                raise RuntimeError(err_str)
            images = self.images
            if title is None:
                title = f'{self.title}_images'
        
        # Save particular attribute of XRDData
        elif isinstance(images, str):
            # Can directly grap attributes.
            # This is for overwriting this function
            if (hasattr(self, images)
                and getattr(self, images) is not None):
                if title is None:
                    title = images
                images = getattr(self, images)
            else:
                err_str = (f"{self.__class__.__name__} does not "
                           + f"have attribute '{images}'.")
                raise AttributeError(err_str)

        # Save custom images
        else:
            # This conditional is for single images mostly
            # (e.g., dark-field)
            images = np.asarray(images)
            if title is None:
                err_str = 'Must define title to save custom images.'
                raise ValueError(err_str)

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
            err_str = (f'Images input has {images.ndim} dimensions '
                       + 'instead of 2 (image) or 4 (XRDData).')
            raise ValueError()
        elif images.ndim == 4 and compression is None:
            compression = 'gzip'
            compression_opts = 4 # changed default from 8 to h5py default
        else:
            raise TypeError('Unknown image type detected!')
        
        # Get chunks
        # For images. Saving map
        if not hasattr(self, '_chunks') or self._chunks is None: 
            chunks = self._get_optimal_chunks()
        else:
            chunks = self._chunks
        
        # Maps and images will not be chunked
        if images.ndim != 4:
            chunks = None

        # Grab some metadata
        image_shape = images.shape
        image_dtype = images.dtype

        dask_flag=False
        if isinstance(images, da.core.Array):
            dask_images = images.copy()
            images = None
            dask_flag = True
        
        # print(f'Image to save have shape {image_shape}')
        img_grp = self.hdf[f'{self._hdf_type}/image_data']
        
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
                    # Replace data if the size, shape, and chunks match
                    dset[...] = images 
                else:
                    pass # Leave the dataset for now
            else:
                # This is not the best, because this only
                # deletes the flag. The data stays
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


    @protect_hdf()
    def save_integrations(self,
                          integrations=None,
                          title=None,
                          units=None,
                          labels=None,
                          mode='a',
                          extra_attrs=None):
        # # No dask support

        # Save all integrations
        if integrations is None:
            if (not hasattr(self, 'integrations')
                or self.integrations is None):
                err_str = 'Must provide integrations to write to hdf.'
                raise RuntimeError(err_str)
            integrations = self.integrations
            if title is None:
                title = f'{self.title}_integrations'
        
        # Save particular attribute of XRDData
        elif isinstance(integrations, str):
            # Can directly grap attributes.
            # This is for overwriting this function
            if (hasattr(self, integrations)
                and getattr(self, integrations) is not None):
                if title is None:
                    title = integrations
                integrations = getattr(self, integrations)
            else:
                err_str = (f"{self.__class__.__name__} does not have "
                           + f"attribute '{integrations}'.")
                raise AttributeError(err_str)

        # Save custom images
        else:
            # This conditional is for single images mostly (e.g., dark-field)
            integrations = np.asarray(integrations)
            if title is None:
                err_str = 'Must define title to save custom images.'
                raise ValueError(err_str)
        
        # Check shape and set title
        if integrations.ndim != 3:
            err_str = ('Integrations must have 3 dimensions, '
                       + f'not {len(integrations.shape)}.')
            raise ValueError()
        
        # Get labels
        _units, _labels = self._get_save_labels(integrations.shape)
        if units is None:
            units = _units
        if labels is None:
            labels = _labels

        # Grab some metadata
        integrations_shape = integrations.shape
        integrations_dtype = integrations.dtype

        int_grp = self.hdf[self._hdf_type].require_group(
                                            'integration_data')
        
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
                # This is not the best, because this only
                # deletes the flag. The data stays
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
    