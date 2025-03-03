import numpy as np
import h5py
import os
from skimage.restoration import rolling_ball
from tqdm import tqdm
import time as ttime
from collections import OrderedDict
import dask.array as da
import functools

# Local imports
from xrdmaptools.io.hdf_utils import (
    check_hdf_current_images,
    overwrite_attr,
    get_optimal_chunks
)
from xrdmaptools.utilities.math import check_precision
from xrdmaptools.utilities.utilities import (
    rescale_array,
    Iterable2D
)
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

# This class is intended to hold and process raw data with I/O
# Analysis and interpretations are reserved for child classes
class XRDData:
    """
    A class for holding and processing XRD images with some I/O 
    functionality. Analysis and interpretation are reserved for child 
    classes, but there is some cross-communication.

    Parameters
    ----------
    image_data : 4D Numpy or Dask array, list, h5py dataset, optional
        Image data that can be fully loaded as a 4D array with XRDMap
        axes (map_y, map_x, image_y, image_x), or XRDRockingCurve axes
        (rocking_axis, 1, image_y, image_x)
    integration_data : 3D Numpy or Dask array, list, h5py dataset, optional
        Integration data as a 3D array with XRDMap axes (map_y, map_x,
        intensity), or XRDRockingCurve axes (rocking_axis, 1,
        intensity)
    map_shape : iterable length 2, optional
        Shape of first two axes in image_data and integration_data as
        (map_y, map_x) or (rocking_axis, 1).
    image_shape : iterable length 2, optional
        Shape of last two axes in image_data (image_y, image_x).
    map_labels : iterable length 2, optional
        Labels passed to HDF for map axes. Defaults to 'map_y_ind' and
        'map_x_ind' for XRDMap, and 'rocking_ind' and 'null_ind' for 
        XRDRockingCurve.
    dtype : dtype, optional
        Images and integrations will be converted to this data type if
        given.
    title : str, optional
        Custom title used to title HDF datasets when saving. Will be
        updated to default value after processing.
    hdf_path : path str, optional
        Path of HDF file. If None and hdf is None, then all save
        functions are disabled.
    hdf : h5py File, optional
        h5py File instance. If None and hdf is None, then all save
        functions are disabled.
    ai : AzimuthalIntegrator, optional
        pyFAI AzimuthalIntegrator instance of the calibrated detector
        geometry. This is instance and its methods are used for
        geometric corrections (e.g., polarization).
    sclr_dict : dict, optional
        Dictionary of 2D numpy arrays matching the map shape with
        scaler intensities used for intensity normalization.
    null_map : 2D array, optional
        Numpy array matching the map shape used to nullify bad
        images/map pixels (e.g., dropped frames).
    chunks : iterable, optional
        Iterable of length 4 with the chunk sizes for image data. The
        last two dimensions should match the image shape to avoid
        chunking data through images.
    corrections : OrderedDict, optional
        Dictionary of correction names and their boolean value of
        whether they have been applied to the image data.
    dask_enabled : bool, optional
        Flag to indicate whether the image data should be lazily loaded
        as a Dask array. Default is False.
    hdf_type : str, optional
        String passed from child classes indicating their type.

    Raises
    ------
    ValueError if insufficient information is provided to construct
    instance.
    ValueError if HDF functionality is intended, but hdf_type if not
    given.
    TypeError if image_data is not given as supported type.
    ValueError if image_data cannot be constructed into 4D data.
    ValueError if integration_data cannot be constructed into 3D data.
    ValueError if HDF file information cannot be properly resolved.
    ValueError if map_labels is not length two.
    RuntimeError if Dask is enabled without providing an active HDF.

    Methods
    -------
    __len__
    __str__
    __repr__

    Properties
    ----------


    """

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
                # list input is handled lazily,
                # but should only be from databroker
                elif isinstance(image_data, list):
                    if dask_enabled:
                        # bad chunking will be fixed later
                        image_data = da.stack(image_data) 
                    else:
                        image_data = np.stack(image_data)
                # Otherwise as a numpy array
                else:
                    image_data = np.asarray(image_data)

            # Working wtih image_data shape
            # Check inputs first.
            # If these are given, assumed they will be used.
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
                    self.images = image_data.reshape((
                                        *map_shape,
                                        *input_shape[-2:]))
                    image_shape = self.images.shape[-2:]
                    ostr = ('Reshaping image_data into given '
                            + f'map_shape as {self.images.shape}.')
                    print(ostr)
                # This may break wtih a theta or energy channel   
                elif image_shape is not None:  
                    self.images = image_data.reshape(
                                                (*input_shape[:-2],
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
                        raise ValueError(err_str)
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
                self.map_shape = map_shape
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
                    raise ValueError(err_str)
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
        
        # Collect other useful parameters
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
        if self._dask_enabled:
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

        self.build_correction_dictionary(corrections)
        # ordered_correction_keys = [
        #         'dark_field',
        #         'flat_field', 
        #         # Can be approximated with background
        #         'air_scatter', 
        #         'outliers',
        #         'pixel_defects', # Just a mask
        #         'pixel_distortions', # Unknown
        #         'scaler_intensity',
        #         'lorentz',
        #         'polarization',
        #         'solid_angle',
        #         'absorption', # Tricky      
        #         'background',
        #         'polar_calibration', # Bulky
        #     ]

        # if isinstance(corrections, (dict, OrderedDict)):
        #     self.corrections = OrderedDict([
        #         (key, corrections[key])
        #         for key in ordered_correction_keys
        #         if key in corrections
        #         ])
        # else:
        #     self.corrections = OrderedDict([
        #         (key, False) for key in ordered_correction_keys
        #         ])

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
        
        # Shared with child classes
        self.ai = ai
        self.sclr_dict = sclr_dict

    
    # Just for convenience
    def __len__(self):
        """
        Number of frames/images/XRD patterns in this object.

        Returns
        -------
        length : int
            Number of frames/images/XRD patterns in this object.
        """
        return self.num_images


    def __str__(self):
        """
        A simple represenation of the class.

        Returns
        -------
        outstring : str
            A simple representation of the class.
        """
        ostr = f'XRDData: ({self.shape}), dtype={self.dtype}'
        return ostr

    
    def __repr__(self):
        """
        A nice representation of the class with relevant information.

        Returns
        -------
        outstring : str
            A nice representation of the class with relevant
            information.
        """
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
        """
        Get the current data type of images and integrations arrays.
        Setting this to a new value will change the images and
        integrations data type to the new value if they exist.
        """
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

    
    @property
    def indices(self):
        """
        Get an Iterable2D class for easily iterating through all
        pixels/images/XRD patterns.
        """
        return Iterable2D(self.map_shape)

    
    def _projection_factory(property_name, function, axes):
        """
        Internal function for constructing projection properties from
        images.
        """
        property_name = f'_{property_name}' # This line is crucial! 
        def get_projection(self):
            if hasattr(self, property_name):
                return getattr(self, property_name)
            else:
                if (not hasattr(self, 'images')
                    or self.images is None):
                    err_str = ('Cannot determine image '
                               + 'projection without images.')
                    raise AttributeError(err_str)

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


    min_map = _projection_factory('min_map', np.min, (2, 3))
    min_image = _projection_factory('min_image', np.min, (0, 1))

    max_map = _projection_factory('max_map', np.max, (2, 3))
    max_image = _projection_factory('max_image', np.max, (0, 1))

    med_map = _projection_factory('med_map', np.median, (2, 3))
    med_image = _projection_factory('med_image', np.median, (0, 1))

    # Will not be accurate at default dtype of np.uint16
    mean_map = _projection_factory('mean_map', np.mean, (2, 3))
    mean_image = _projection_factory('mean_image', np.mean, (0, 1))

    # May cause overflow errors
    sum_map = _projection_factory('sum_map', np.sum, (2, 3))
    sum_image = _projection_factory('sum_image', np.sum, (0, 1))


    # Adaptively saves for whatever the current processing state
    # Add other methods, or rename to something more intuitive
    @property
    def composite_image(self):
        """
        Image generated from the minimum image pixel values subtracted
        from the maximum pixel values. This is a convenience method for
        quickly emphasizing signal across a dataset.
        """
        if hasattr(self, f'_composite_image'):
            return getattr(self, f'_composite_image')
        else:
            setattr(self, f'_{self.title}_composite_image',
                    self.max_image - self.min_image)
            
            # Set the generic value to this as well
            self._composite_image = getattr(
                                    self,
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
        """
        Get a combined mask matching the image shape from calibration,
        defect, or custom masks of the class. Truthy values will be
        considered for further analysis.
        """
        # a bit redundant considering only 4D shapes are allowed
        img_slc = (0,) * (self.images.ndim - 2) 
        #mask = np.ones_like(self.images[img_slc], dtype=np.bool_)
        mask = np.ones(self.images[img_slc].shape, dtype=np.bool_)

        # Remove unused calibration pixels
        if hasattr(self, 'polar_mask'):
            if self.polar_mask.shape == mask.shape:
                mask *= self.polar_mask
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
        """
        Delete cached projected images, integrations, and maps.
        """
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
        """
        Update internal title parameter.
        
        Title is updated according to given title or the current
        corrections applied to the dataset. This title is used for
        default naming in the HDF file.

        Parameters
        ----------
        title : str, optional
            New title of dataset.

        Notes
        -----
        Default Updated Titles
        - raw : no corrections
        - detector_corrected : dark_field or flat_field
        - geometry_corrected : lorentz, polarization, or solid_angle
        - background_corrected : background
        - calibrated : polar_calibration
        """
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


    def build_correction_dictionary(self, corrections=None):
        """
        Build an OrderedDict of booleans for tracking corrections.

        Build an OrderedDict of booleans from dictionary or nothing to
        keep track of applied corrections. Only certain correction keys
        considered; all others are ignored. By default this function
        creates a correction OrderedDict with all values set to False.

        The correction dictionary only considers keys in:
        - 'dark_field'
        - 'flat_field'
        - 'air_scatter'
        - 'outliers'
        - 'pixel_defects'
        - 'pixel_distortions'
        - 'scaler_intensity'
        - 'lorentz'
        - 'polarization'
        - 'solid_angle'
        - 'absorption'      
        - 'background'
        - 'polar_calibration'

        Parameters
        ----------
        corrections : dict or OrderedDict of bools, optional
            Dictionary of corrections to use to build internal
            correction dictionary. Default value is None and new
            correction dictionary will be create with all values set to
            False.
        """

        ordered_correction_keys = [
            'dark_field',
            'flat_field', 
            # Can be approximated with background
            'air_scatter', 
            'outliers',
            'pixel_defects', # Just a mask
            'pixel_distortions', # Unknown
            'scaler_intensity',
            'lorentz',
            'polarization',
            'solid_angle',
            'absorption', # Tricky      
            'background',
            'polar_calibration', # Bulky
        ]

        # Rebuild dictionary
        if isinstance(corrections, (dict, OrderedDict)):
            self.corrections = OrderedDict([
                (key, corrections[key])
                for key in ordered_correction_keys
                if key in corrections
                ])
        # Build new dictionary
        else:
            self.corrections = OrderedDict([
                (key, False)
                for key in ordered_correction_keys
                ])


    # Opens and closes hdf to ensure self.hdf can be used safely
    def _protect_hdf(pandas=False):
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
                    try:
                        func(self, *args, **kwargs)
                        err = None
                    except Exception as e:
                        err = e
                    
                    # Clean up hdf state
                    if pandas and active_hdf:
                        self.open_hdf()
                    elif not active_hdf:
                        self.close_hdf()

                    # Re-raise any exceptions, after cleaning up hdf
                    if err is not None:
                        raise(err)
            return protector
        return protect_hdf_inner

    
    @_protect_hdf()
    def load_images_from_hdf(self,
                             image_data_key='recent',
                             dask_enabled=None,
                             chunks=None):
        """
        Load images from HDF file.

        Load images from active HDF file according. Defaults to most
        recent set of images.

        Parameters
        ----------
        image_data_key : str, optional
            Dataset title in image_data group within HDF file used to
            load image data. Defaults to 'recent', which will load the
            most recently written image data.
        dask_enabled : bool, optional
            Flag to indicate whether the image data should be lazily
            loaded as a Dask array. Default is to check current images
            or False if there are not any images.
        chunks : iterable, optional
            Iterable of length 4 with the chunk sizes for image data.
            The last two dimensions should match the image shape to
            avoid chunking data through images.

        Raises
        ------
        RuntimeError if 'recent' image data is requested without any
        image data in the HDF file.
        """

        # Actually load the data. code is modified from hdf_io.py
        img_grp = self.hdf[f'{self._hdf_type}/image_data']

        # Check valid image_data_key
        if (str(image_data_key).lower() != 'recent'
            and image_data_key not in img_grp.keys()):
            warn_str = (f'WARNING: Requested image_data_key'
                        + f'({image_data_key}) not found in hdf. '
                        + 'Proceding without changes...')
            return

        # Set recent image data key
        if str(image_data_key).lower() == 'recent':
            time_stamps, img_keys = [], []
            for key in img_grp.keys():
                if key[0] != '_':
                    time_stamps.append(
                        img_grp[key].attrs['time_stamp'])
                    img_keys.append(key)
            if len(img_keys) < 1:
                raise RuntimeError('Could not find recent image data'
                                   + 'from hdf.')
            time_stamps = [ttime.mktime(ttime.strptime(x))
                           for x in time_stamps]
            image_data_key = img_keys[np.argmax(time_stamps)]

        # Setting up
        img_dset = img_grp[image_data_key]
        print(f'Loading images from ({image_data_key})...',
              end='',
              flush=True)

        # Loading full image dataset
        if image_data_key[0] != '_':
            # Preserve dask flag from previous images unless specified
            if dask_enabled is None:
                dask_enabled = self._dask_enabled

            # Delete previous images from XRDData to save memory
            self.images = None

            if dask_enabled:
                # Lazy load data
                image_data = da.from_array(img_dset,
                                           chunks=img_dset.chunks)
            else:
                # Fully load data
                image_data = img_dset[:]
            self.images = image_data

            # Rebuild correction dictionary
            image_corrections = {}
            for key, value in img_dset.attrs.items():
                if key[0] == '_' and key[-11:] == '_correction':
                    image_corrections[key[1:-11]] = value
            # self.corrections = image_corrections
            self.build_correction_dictionary(
                                corrections=image_corrections)

            # Define/determine chunks and rechunk if necessary
            # This code is from __init__
            if self._dask_enabled:
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
        
        # Loading only an image attribute
        else:
            # Convert dataset title to attribute. Some special cases
            if image_data_key == '_statice_background':
                attr_name = 'background'
            elif image_data_key == '_spot_masks':
                attr_name = 'blob_masks'
            else:
                attr_name = image_data_key[1:]
            
            # Load and set attribute
            setattr(self, attr_name, img_dset[:])

        print('done!')


    def dump_images(self):
        """
        Release current images from memory.
        """
        if self._dask_enabled:
            warn_str = ('WARNING: Dask will no longer be enabled '
                        + 'without images!')
            print(warn_str)
        del self.images # This may help with gc
        self.images = None

    
    @_protect_hdf()
    def load_integrations_from_hdf(self,
                                   integration_data_key='recent'):
        """
        Load integrations from HDF file.

        Load integrations from active HDF file according. Defaults to
        most recent set of integrations. Loaded data is not checked
        against loaded images to ensure the two datasets are
        compatible. This will be added in the future.

        Parameters
        ----------
        integration_data_key : str, optional
            Dataset title in integration_data group within HDF file
            used to load integration data. Defaults to 'recent',
            which will load the most recently written integration data.

        Raises
        ------
        RuntimeError if 'recent' integration data is requested without
        any integration data in the HDF file.
        """

        # Actually load the data. code is modified from hdf_io.py
        int_grp = self.hdf[f'{self._hdf_type}/integration_data']

        # Check valid integration_data_key
        if (str(integration_data_key).lower() != 'recent'
            and integration_data_key not in int_grp.keys()):
            warn_str = (f'WARNING: Requested integration_data_key '
                        + f'({integration_data_key}) not found in hdf.'
                        + ' Proceding without changes...')
            return

        # Set recent integration data key
        if str(integration_data_key).lower() == 'recent':
            time_stamps, int_keys = [], []
            for key in int_grp.keys():
                if key[0] != '_':
                    time_stamps.append(
                            int_grp[key].attrs['time_stamp'])
                    int_keys.append(key)
            if len(int_keys) < 1:
                err_str = ('Could not find recent '
                           + 'integration data from hdf.')
                raise RuntimeError(err_str)
            time_stamps = [ttime.mktime(ttime.strptime(x))
                           for x in time_stamps]
            integration_data_key = int_keys[np.argmax(time_stamps)]

        # Setting up
        int_dset = int_grp[integration_data_key]
        print(f'Loading integrations from ({integration_data_key})...',
              end='',
              flush=True)

        # Loading full image dataset
        if integration_data_key[0] != '_':
            # Delete and redefine integrations to save memory
            self.integration = None
            self.integrations = int_dset[:]

            # Rebuild correction dictionary
            integration_corrections = {}
            for key, value in (
                        int_grp[integration_data_key].attrs.items()):
                if key[0] == '_' and key[-11:] == '_correction':
                    integration_corrections[key[1:-11]] = value

            # Check state of integrations to images
            integration_title = '_'.join([x for x in
                                        integration_data_key.split('_')
                                        if x not in ['images',
                                                     'integrations']])
            if self.images is None:
                # self.corrections = integration_corrections
                self.build_correction_dictionary(
                                corrections=integration_corrections)
            else:
                warn_str = ('WARNING: Checks for matching loaded '
                            + 'integrations with currently loaded '
                            + 'images are not yet implemented.')
                print(warn_str)

        # Loading only an image attribute
        else:
            # Convert dataset title to attribute. 
            attr_name = integration_data_key[1:]
            
            # Load and set attribute
            setattr(self, attr_name, int_dset[:])

        print('done!')

    
    def dump_integrations(self):
        """
        Release current integrations from memory.
        """
        del self.integrations # This may help with gc
        self.integrations = None


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
        """
        Closes currently active HDF file.
        """
        if self.hdf is not None:
            self._dask_2_hdf()
            self.hdf.close()
            self.hdf = None
        

    def open_hdf(self, dask_enabled=None):
        """
        Opens HDF file.

        Opens HDF file from hdf path information if available. If dask
        is enabled, then a temporary image data storage location in the
        HDF file will be setup or accessed if it already exist.

        Parameters
        ----------
        dask_enabled : bool, optional
            Flag to determine if a reference to a temporary storage
            location needs to be setup in order to store the lazily
            loaded image data. By default this flag is set to whether
            the images are already lazily loaded.
        """
        if self.hdf is not None:
            # Should this raise errors or just ping warnings
            note_str = ('NOTE: HDF is already open. '
                        + 'Proceeding without changes.')
            print(note_str)
            return
        elif self.hdf_path is None:
            note_str = ('NOTE: HDF path is not being saved in this'
                        + 'instance. Run start_saving_hdf method to '
                        + 'start saving to HDF file.')
            print(note_str)
            return
        else:
            self.hdf = h5py.File(self.hdf_path, 'a')

        # This flag persists even when the dataset is closed!
        if dask_enabled is None:
            dask_enabled = self._dask_enabled

        if dask_enabled: 
            img_grp = self.hdf[f'{self._hdf_type}/image_data']
            if self.title == 'final':
                if check_hdf_current_images(f'{self.title}_images',
                                            hdf=self.hdf):
                    dset = img_grp[f'{self.title}_images']
            elif check_hdf_current_images('_temp_images',
                                          hdf=self.hdf):
                dset = img_grp['_temp_images']
            self.images = da.asarray(dset) # persist(), breaks things...
            self._hdf_store = dset


    ########################################
    ### Image Corrections and Transforms ###
    ########################################

    def _check_correction(self, correction, override=False):
        """
        Internal convenience function for checking if correction has
        already been applied and overriding if specified.
        """

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
        """
        Apply dark-field correction.

        Subtract dark-field from images if correction is not already
        applied or override requested. Internal dark_field attribute is
        used unless specified. Successful execution will then store the
        dark-field internally and in the HDF file.

        Parameters
        ----------
        dark_field : 2D array, optional
            2D array matching the image shape to be subtracted from
            images. Internal dark_field attribute is used unless
            specified.
        override : bool, optional
            Override already applied corrections if True. Default is
            False.

        Raises
        ------
        ValueError if dark-field shape does not match image shape.
        
        Notes
        -----
        Since this should be the first correction applied, this
        function will upcast the data to 32-bit floats to prevent
        future errors. This function will also generate a temporary
        storage dataset in the HDF file if dask is enabled and one
        does not already exist.
        """


        if self._check_correction('dark_field', override=override):
            return
        elif dark_field is None:
            if (hasattr(self, 'dark_field')
                and self.dark_field is not None):
                dark_field = self.dark_field
            else:
                print('No dark-field given for correction.')
        elif dark_field.shape != self.image_shape:
            err_str = (f'Dark-field shape of {dark_field.shape} does '
                       + 'not match image shape of '
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
                    # Upcast to final size
                    self.dtype = np.float32
        
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
        """
        Apply flat-field correction.

        Divide images by flat-field if correction is not already
        applied or override requested. Internal flat_field attribute is
        used unless specified. Successful execution will then store the
        flat-field internally and in the HDF File.

        Parameters
        ----------
        flat_field : 2D array, optional
            2D array matching the image shape to be divided from
            images. Internal flat_field attribute is used unless
            specified.
        override : bool, optional
            Override already applied corrections if True. Default is
            False.

        Raises
        ------
        ValueError if flat-field shape does not match image shape.
        
        Notes
        -----
        Flat-field corrections create erroneuous results for any pixel
        values too close to zero. Currently, there are no precautions
        against this.
        """

        if self._check_correction('flat_field', override=override):
            return
        elif flat_field is None:
            if (hasattr(self, 'flat_field')
                and self.flat_field is not None):
                flat_field = self.flat_field
            else:
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
        """
        Correct for air scatter.

        Subtract air scatter contribution from images if correction not
        already applied or override requested. Successful execution
        will then store the air scatter internally and in the HDF file.

        Parameters
        ----------
        air_scatter : 2D array, optional
            2D array matching the image shape to be subtracted from
            images. Internal air_scatter attribute is used unless
            specified.
        applied_corrections : dict, optional
            A dictionary with keys included in the internal correction
            dictionary of any corrections already applied to the
            provided air scatter array. Any corrections applied to the
            current images, but not listed in applied corrections will
            then be applied to the air scatter array before
            corrections. By default the applied_corrections are those
            already applied to the current images.
        override : bool, optional
            Override already applied corrections if True. Default is
            False.

        Raises
        ------
        ValueError if air scatter shape does not match image shape.
        
        Notes
        -----
        Only certain corrections can be reasonably applied to the air
        scatter reference. These include 'dark-field', 'flat-field',
        'Lorentz', 'polarization', and 'solid-angle'. Air scatter
        cannot and will not be corrected after either 'background' or
        'absorption' corrections.
        """

        if self._check_correction('air_scatter', override=override):
            return
        elif air_scatter is None:
            if (hasattr(self, 'air_scatter')
            and self.air_scatter is not None):
                air_scatter = self.air_scatter
            else:
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
            'scaler_intensity',
            'lorentz',
            'polarization',
            'solid_angle'
        ]
        
        print('Correcting air scatter...', end='', flush=True)
        # Copy and change type of air scatter as needed
        air_scatter = air_scatter.astype(self.dtype)

        if (self.corrections['dark_field']
            and not applied_corrections['dark_field']):
            air_scatter -= self.dark_field

        if (self.corrections['flat_field']
            and not applied_corrections['flat_field']):
            air_scatter /= self.flat_field
        
        # Attempt to normalize air_scatter based on known scalers
        # This is not a strictly valid fix...
        if (self.corrections['scaler_intensity']
            and not applied_corrections['scaler_intensity']):
            warn_str = ('\nWARNING: Attempting to normalize '
                        + 'air_scatter values by already applied '
                        + 'scaler normalization.')
            print(warn_str)
            air_scatter /= np.median(self.scaler_map)
        
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

        """
        Correct for outlier image pixels.

        Find and replace for outlier pixels with median intensity of
        surrounding pixels. These include hot pixels, dead pixels, and
        zingers. Pixels are found and replaced by defining a tolerance
        beyond a median of neighboring pixels.

        Parameters
        ----------
        size : int, optional
            Nearest neighbor distance used to apply median filter
            across images. Should be greater than or equal to 2.
        tolerance : float, optional
            Normalized difference of allowable fluctuations between
            original image and median-filtered image. For example, an
            original pixel value of 2 and median pixel value of 1 would
            have a 0.5 value to compare against the tolerance.
        override : bool, optional
            Override already applied corrections if True. Default is
            False.

        Raises
        ------
        ValueError if size is greater than or equal to 2 or tolerance
        is less than or equal to zero.
        """

        if self._check_correction('outliers', override=override):
            return

        if size < 2:
            err_str = ('Size for outlier correction must be greater'
                       + 'than or equal to 2.')
            raise ValueError(err_str)
        if tolerance < 0:
            err_str = ('Tolerance for outlier correction must be '
                       + 'greater than 0.')
            raise ValueError(err_str)

        print('Finding and correcting image outliers...')
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

    
    def normalize_scaler(self,
                         scaler_arr=None,
                         override=False):
        """
        Normalize images by their scaler intensity.

        Divide full images by their incident scaler X-ray intensity.
        Successful execution will then store the scaler intensity map
        internally and in the HDF file.

        Parameters
        ----------
        scaler_arr : 2D array, optional
            Scaler intensity map used for image normalization. If not
            provided, a map from the internal scaler dictionary will be
            used instead. If this fails, the image medians will be used
            instead. Default is to use internal scaler dictionary.
        override : bool, optional
            Override already applied corrections if True. Default is
            False.
        
        Raises
        ------
        ValueError if scaler array does not match map shape.
        """

        if self._check_correction('scaler_intensity',
                                  override=override):
            return
        
        elif scaler_arr is None:
            sclr_key = None
            if (hasattr(self, 'sclr_dict')
                and self.sclr_dict is not None
                and self.sclr_dict is not {}):
                for sclr_key in ['flux_i0',
                                 'flux_im',
                                 'energy_corrected_i0',
                                 'energy_corrected_im',
                                 'i0',
                                 'im']:
                    if sclr_key in self.sclr_dict.keys():
                        scaler_arr = self.sclr_dict[sclr_key]
                        break
                    else:
                        sclr_key = None
                if sclr_key is None:
                    # Hope for the best
                    sclr_key = list(self.sclr_dict.keys())[0]
                    scaler_arr = self.sclr_dict[sclr_key]
                    warn_str = ("WARNING: Unrecognized scaler keys. "
                                + f"Using '{sclr_key}' instead.")
                    print(err_str)
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


    # No correction for defect mask,
    # since it is used whenever mask is called
    def apply_defect_mask(self,
                          min_bounds=(-np.inf, 0),
                          max_bounds=(0, np.inf),
                          mask=None,
                          override=False):
        """
        Method for approximating defective pixels.

        Approximate defective pixels by defining bounds on the
        projected maximum and minimum images. This can be used to help
        mask overlly high pixels or pixel behind beam blocks.
        Successful execution will write the defect mask in the HDF
        file.

        Parameters
        ----------
        min_bounds : tuple, optional
            Tuple defining the (lower bound, upper bound) range of
            acceptable pixel values for the minimum projected image. By
            defualt these values are (-infinity, zero).
        max_bounds : tuple, optional
            Tuple defining the (lower bound, upper bound) range of
            acceptable pixel values for the maximum projected image. By
            default these values are (0, infinity).
        mask : 2D array, optional
            Starting mask to use when comparing bounds. Must match
            image shape and truthy values are pixels that will be
            considered.
        override : bool, optional
            Override already applied corrections if True. Default is
            False.
        """
        
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
    def apply_custom_mask(self,
                          mask=None):
        """
        Method for applying a custom mask.

        Apply custom mask to images. This can be used to ignore any
        image pixels across the dataset. Successful excution will then
        write the custom mask in the HDF file.

        Parameters
        ----------
        mask : 2D array, optional
            Custom mask matching image shape. Truthy pixels are counted
            towards analysis. Default is None and no custom mask will
            be considered.

        Raises
        ------
        ValueError if mask shape does not match image shape.
        """
        

        if mask is not None:
            mask = np.asarray(mask).astype(np.bool_)
            if mask.shape != self.image_shape:
                err_str = ('Mask shape should match image shape of '
                           + f'{self.image_shape} not {mask.shape}.')
                raise ValueError(err_str)

            self.custom_mask = mask
            # Write mask to disk
            self.save_images(images='custom_mask')
        else:
            print('No custom mask provided!')


    ### Geometric corrections ###

    def apply_lorentz_correction(self,
                                 powder=False,
                                 apply=True,
                                 override=False):
        """
        Apply the Lorentz correction.

        Divide images by Lorentz correction to correct for per pixel
        reciprocal space sampling allowing comparison of signals
        acquired at different scattering angles. This correction should
        not be applied if pixel signals will be converted into 3D
        reciprocal space.
        
        Like the other geometric corrections, this correction is only
        applicable to actual signals, even though it is applied across
        full images.  Detector calibration must already be set.
        Successful execution will then store the Lorentz correction
        internally and in the HDF file.

        Parameters
        ----------
        powder : bool, optional
            Flag to include powder contribution to Lorentz correction.
        apply : bool, optional
            Flag to apply correction to images. Default is True.
        override : bool, optional
            Override already applied corrections if True. Default is
            False.
        
        Raises
        ------
        AttributeError if calibration (ai attribute) has not been
        applied.
        """

        if self._check_correction('lorentz', override=override):
            return
        
        # Check for calibraitons
        if (not hasattr(self, 'ai') or self.ai is None):
            err_str = ('Lorentz correction cannot be applied without '
                       + 'calibration!')
            raise AttributeError(err_str)

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
        """
        Apply the polarization correction.

        Divide images by a polarization correction to account for
        differing scattering intensities from the X-ray polarization.
        
        Like the other geometric corrections, this correction is only
        applicable to actual signals, even though it is applied across
        full images. Detector calibration must already be set.
        Successful execution will then store the polarization
        correction internally and in the HDF file.

        Parameters
        ----------
        polarization : float, optional
            Float between -1 and 1 indicating the degree of X-ray
            polarization. -1 is vertically polarized, 0 is not
            polarization, and 1 is entirely horizontally polarized. The
            default value is 0.9, where synchrotron X-rays are
            typically horizontally polarized between 0.9 and 0.99.
        apply : bool, optional
            Flag to apply correction to images. Default is True.
        override : bool, optional
            Override already applied corrections if True. Default is
            False.

        Raises
        ------
        AttributeError if calibration (ai attribute) has not been
        applied.
        """

        if self._check_correction('polarization', override=override):
            return

        # Check for calibraitons
        if (not hasattr(self, 'ai') or self.ai is None):
            err_str = ('Polarization correction cannot be applied '
                       + 'without calibration!')
            raise AttributeError(err_str)
        
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
        """
        Apply solid-angle correction.

        Divide images by a solid-angle correction to account for the
        volume of scintillating material intersected by the diffracted
        beam. 
        
        Like the other geometric corrections, this correction is
        only applicable to actual signals, even though it is applied
        across full images. Detector calibration must already be set.
        Successful execution will then store the solid-angle correction
        internally and in the HDF file.

        Parameters
        ----------
        apply : bool, optional
            Flag to apply correction to images. Default is True.
        override : bool, optional
            Override already applied corrections if True. Default is
            False.

        Raises
        ------
        AttributeError if calibration (ai attribute) has not been
        applied.
        """

        if self._check_correction('solid_angle', override=override):
            return

        # Check for calibraitons
        if (not hasattr(self, 'ai') or self.ai is None):
            err_str = ('Solid-angle correction cannot be applied '
                       + 'without calibration!')
            raise AttributeError(err_str)

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


    # TODO: Add user-provided absorption correction!
    def apply_absorption_correction(self,
                                    exp_dict,
                                    apply=True,
                                    override=False):
        """
        Apply an absorption correction.

        Divide images by an absorption correction to account for
        the intensity lost by the diffracted beam traveling through the
        sample. This method depends on many factors and assumptions
        about the sample composition and geometry that are not often
        known; thus, is still a work in progress.
        
        Like the other geometric corrections, this correction is only
        applicable to actual signals, even though it is applied across
        full images. Detector calibration must already be set.
        Successful execution will then store the absorption correction
        internally and in the HDF file.

        Parameters
        ----------
        exp_dict : dict, optional
            Dictionary of values for 'attenuation_length', 'mode',
            'thickness', 'theta'. Currently for only a plane of
            material of homogenous composition.
        apply : bool, optional
            Flag to apply correction to images. Default is True.
        override : bool, optional
            Override already applied corrections if True. Default is
            False.

        Raises
        ------
        AttributeError if calibration (ai attribute) has not been
        applied.
        """

        #exp_dict = {
        #    'attenuation_length' : value,
        #    'mode' : 'transmission',
        #    'thickness' : value,
        #    'theta' : value # in degrees!
        #}

        if self._check_correction('absorption', override=override):
            return
    
        # Check for calibraitons
        if (not hasattr(self, 'ai') or self.ai is None):
            err_str = ('Absorption correction cannot be applied '
                       + 'without calibration!')
            raise AttributeError(err_str)
        
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
                xi = ((t / (np.cos(theta))) / ((1 / np.tan(tth_arr))
                      - (np.cos(chi_arr) * np.tan(theta))))
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


    # TODO: Not all of these have been enabled with dask arrays
    def estimate_background(self,
                            method=None,
                            background=None,
                            inplace=True,
                            override=False,
                            **kwargs):
        """
        Estimate image backgrounds.

        Determine image backgrounds by a variety of methods. These
        backgrounds can account for scatter from air, amorphous
        regions, or thermal vibrations and fluorescence from the sample
        or diffracted beams. Successful execution may temporarily store
        the backgrounds internally and will save static single images 
        to the HDF file.

        Parameters
        ----------
        method : str, optional
            String indicating the method for estimating background.
            Many different variants are accepted, but currently
            supported methods for estimating backgrounds are 'median',
            'minimum', 'gaussian', 'bruckner', 'none', and 'custom'.
            The method parameter is only considered if background is
            None and the default value is to not estimate any
            background or 'none'.
        background : 2D or 4D array, optional
            Array with dimensions matching either the image shape or
            full dataset shape that will be used as a custom
            background. None by default and will be estimated with the
            method perscribed.
        inplace : bool, optional
            Flag to indicate inplace background subtraction where
            possible. If true, the background is never stored
            internally and can save on memory usage.
        override : bool, optional
            Override already applied corrections if True. Default is
            False.
        kwargs : dict, optional
            Dictionary of kwargs passed to the individual method for
            estimating backgrounds.

        Raises
        ------
        NotImplementedError for not fully implemented methods for
        estimating background: 'rolling_ball', 'spline', and
        'polynomial'.
        """

        method = str(method).lower()

        # Check to see if background correction
        # has already been applied.
        if inplace and self._check_correction('background',
                                              override=override):
            if (hasattr(self, 'background')
                and self.background is not None):
                warn_str = ('WARNING: background attribute still '
                            + 'saved in memory.\nOverride background '
                            + 'remove or delete attribute to '
                            + 'release memory.')
                print(warn_str)
            return

        if background is None:
            # Many different background methods have been implemented
            if method in ['med', 'median']:
                print('Estimating background from median values.')
                self.background = self.med_image
                self.background_method = 'median'
                if inplace:
                    self.images -= self.background

            elif method in ['min', 'minimum']:
                print('Estimating background with minimum method.')
                self.background = self.min_image
                self.background_method = 'minimum'
                if inplace:
                    self.images -= self.background
                
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
                raise NotImplementedError(f'{method} not full supported.')
                print('Estimating background with spline fit.')
                self.background = fit_spline_bkg(self, **kwargs)
                self.background_method = 'spline'

            elif method in ['poly', 'poly fit', 'poly_fit']:
                raise NotImplementedError(f'{method} not full supported.')
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
                self.background = masked_gaussian_background(
                                                    self.images,
                                                    **kwargs)
                self.background_method = 'gaussian'

            elif method in ['Bruckner', 'bruckner']:
                print('Estimating background with Bruckner algorithm.')
                self.background = masked_bruckner_background(
                                                    self.images,
                                                    mask=self.mask,
                                                    inplace=inplace,
                                                    **kwargs)
                self.background_method = 'bruckner'

            elif method in ['none']:
                print('No background correction will be used.')
                self.background = None
                self.background_method = 'none'
                return # No inplace checks
            
            else:
                err_str = f"Method '{method}' not implemented!"
                raise NotImplementedError(err_str)
    
        else:
            print('User-specified background.')
            self.background = background
            self.background_method = 'custom'

        # Check to see if background was removed
        # This will fail for backgrounds without inplace subtraction.
        if inplace:
            self.corrections['background'] = True
            self.update_map_title()
            self._dask_2_hdf()
            print('Background removed!')

            if (self.background is not None
                and np.squeeze(self.background).shape
                == self.images.shape[-2:]):
                self.save_images(images='background',
                                 title='static_background')
            else:
                del self.background


    # TODO: Does this break with 2D backgrounds???
    def remove_background(self,
                          background=None,
                          save_images=False,
                          override=False):
        """
        Remove image backgrounds.

        Subtract backgrounds from images if provided or stored
        internally. Successful execution will save static single images
        to the HDF file.

        Parameters
        ----------
        background : 2D or 4D array, optional
            Array with dimensions matching either the image shape or
            full dataset shape that will be used a custom background.
            None by default to use internall stored backgrounds
        save_images : bool, optional
            Flag to indicate if images should be saved after subtacting
            backgrounds. False by default.
        override : bool, optional
            Override already applied corrections if True. Default is
            False.
        """

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
                print('No background to remove.')
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


    # Not often used.
    # TODO: Check for polar_calibration correction, maybe create a
    # weaker function that is called in XRDBaseScan
    def get_polar_mask(self,
                       tth_num=None,
                       chi_num=None,
                       units='2th_deg'):
        """
        Get polar mask.

        Create a mask for 2D image integrations. This discounts image
        pixels outside of the integration area.

        Parameters
        ----------
        tth_num : int, optional
            Number of steps to divide two theta scattering range, and
            shape of resulting integrations x-axis. Default is to look
            for internal tth_num.
        chi_num : int, optional
            Number of step sto divide chi azimuthal range, and shape of
            resulting integrations y-axis. Default is to look for
            internal chi_num.
        units : str, optional
            Units given to pyFAI AzimuthalIntegrator. Default is
            '2th_deg'.
        
        Raises
        ------
        AttributeError if calibration (ai attribute) has not been
        applied.
        """

        # Check for calibraitons
        if (not hasattr(self, 'ai') or self.ai is None):
            err_str = ('Absorption correction cannot be applied '
                       + 'without calibration!')
            raise AttributeError(err_str)

        if tth_num is None:
            tth_num = self.tth_num
        if chi_num is None:
            chi_num = self.chi_num

        dummy_image = 100 * np.ones(self.image_shape)

        image, _, _ = self.ai.integrate2d_ng(dummy_image,
                                             tth_num,
                                             chi_num,
                                             unit=units)

        polar_mask = (image != 0)

        self.polar_mask = polar_mask
    

    def rescale_images(self,
                       lower=0,
                       upper=100,
                       arr_min=0,
                       arr_max=None,
                       mask=None):
        """
        Rescale images inplace.

        Rescale images inplace into more useful intensity range. This
        method cannot be easily reversed.

        Parameters
        ----------
        lower : float, optional
            New lower bound of rescaled images. 0 by default.
        upper : float, optional
            New upper bound of rescaled images. 100 by default.
        arr_min : float, optional
            Minimum value of array used for rescaling. If None, the
            minimum value of images is used. If this number matches the
            'lower' parameter, then the images lower value will not be
            changed. 0 by default.
        arr_max : float, optional
            Maximum value of array used for rescaling. If None, the
            maximum value of images is used. If this number matches the
            'upper' parameter, then the images upper value will not be
            changed. None by default.
        mask : 2D array, optional
            Mask matching the image shape used to discount values from
            rescaling and from determining arr_min and arr_max if not
            provided.
        """

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
        """
        Estimate the maximum value of current images.

        Estimate the maximum value of the current images given the
        currently applied corrections. If this value is used as the
        'arr_max' parameter in the 'rescale_images' method, then
        rescaled images acquired from different scans can be more
        reasonably compared.

        Parameters
        ----------
        raw_max_val : float, optional
            Maximum possible raw value from detector. By default this
            value is the maximum of a 14 bit color depth detector.
        method : str, optional
            String describing either 'median' or 'minimum' function
            used to estimate the value of applied corrections. 'median'
            provides are more stastical and often more accurate
            estimate.

        Raises
        ------
        ValueError if not enough information is provided to determine
        image normalization.
        """

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
            if (hasattr(self, 'scaler_map')
                and self.scaler_map is not None):
                scaler_map = self.scaler_map
            elif hasattr(self, 'sclr_dict'):
                for key in ['flux_i0',
                            'flux_im',
                            'energy_corrected_i0',
                            'energy_corrected_im',
                            'i0',
                            'im']:
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
            if (hasattr(self, 'background')
                and self.background is not None):
                raw_max_val -= var_func(self.background)
        # All other corrections are isolated within the image

        return raw_max_val


    @_protect_hdf()
    def construct_null_map(self,
                           override=False):
        """
        Determine null map if not already provided.

        Determine which map pixels should not considered in subsequent
        analysis if a null map is not already stored intenally.
        
        This method requires data from the 'raw_images' dataset and
        will access the HDF file if necessary and allowed. This can
        take some time and cannot be performed while Dask is enabled or
        override is True.

        Parameters
        ----------
        override : bool, optional
            Flag to determine if null map should be constructed
            regardless if null map already exists or if Dask is
            enabled.
        """
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
                print('Loading (raw_images) from hdf...')
                hdf_str = f'{self._hdf_type}/image_data/raw_images'
                raw_images = self.hdf[hdf_str]
            
            # Try to do this efficiently
            print('Constructing null_map...')
            null_map = np.ones(self.map_shape, dtype=np.bool_)
            for index in range(self.num_images):
                indices = np.unravel_index(index, self.map_shape)
                if np.any(raw_images[indices] != 0):
                    null_map[indices] = False
            
            self.null_map = np.asarray(null_map)
            self.save_images(images='null_map',
                             units='bool',
                             labels=self.map_labels)
            print('done!')

    
    def nullify_images(self):
        """
        Nullify images based on null map attribute.

        Nullify images by setting all pixel values to zero based on the 
        internal 'null_map' attribute. This allows entire images to be
        discounted from later analysis.
        """
        if not hasattr(self, 'null_map'):
            err_str = 'XRDData does not have null_map attribute.'
            raise AttributeError(err_str)
        elif not np.any(self.null_map):
            note_str = ('Null map is empty, there are no missing '
                        + 'pixels. Proceeding without changes')
            print(note_str)
            return

        # self.images[self.null_map] = 0

        for attr in ['images', 'blob_masks', 'integrations']:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                # This should all be done in place
                getattr(self, attr)[self.null_map] = 0
        

    # Formalized function to declare no further image processing.
    def finalize_images(self,
                        save_images=True):
        """
        Formally declare image corrections finished.

        This method formally declares all image corrections to be
        finished. The internal title is changed to 'final' and all
        image corrections are disabled unless override is called. 
        
        If dask is not enabled, this method will write the final images
        to the HDF if 'save images' is True. If dask is enabled, the
        temporary dataset in the HDF file is renamed to the final
        images while still acting as the temporary storage location.
        Finally, any not applied corrections are called and the dataset
        size is called.

        Parameters
        ----------
        save_images : bool, optional
            Flag to determine if the final images will be written to
            the HDF file. True by default.
        """

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
                        overwrite_attr(temp_dset.attrs,
                                       f'_{key}_correction',
                                       value)
                        # temp_dset.attrs[f'_{key}_correction'] = value

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
    def _integration_projection_factory(property_abbreviation,
                                       function, axes):
        """
        Internal function for constructing projection properties from
        integrations.
        """
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
    

    min_integration = _integration_projection_factory(
                            'min_integration', np.min, (0, 1))
    min_integration_map = _integration_projection_factory(
                            'min_integration_map', np.min, (2))

    max_integration = _integration_projection_factory(
                            'max_integration', np.max, (0, 1))
    max_integration_map = _integration_projection_factory(
                            'max_integration_map', np.max, (2))

    sum_integration = _integration_projection_factory(
                            'sum_integration', np.sum, (0, 1))
    sum_integration_map = _integration_projection_factory(
                            'sum_integration_map', np.sum, (2))

    med_integration = _integration_projection_factory(
                            'med_integration', np.median, (0, 1))
    med_integration_map = _integration_projection_factory(
                            'med_integration_map', np.median, (2))

    mean_integration = _integration_projection_factory(
                            'mean_integration', np.mean, (0, 1))
    mean_integration_map = _integration_projection_factory(
                            'mean_integration_map', np.mean, (2))


    @property
    def composite_integration(self):
        """
        Integration generated from the minimum integration pixel values
        subtracted from the maximum pixel values. This is a convenience
        method for quickly emphasizing signal across a dataset.
        """
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
        """
        Estimate integration backgrounds.

        This method is not fully implemented.

        Determine integration backgrounds by a variety of methods.
        These backgrounds can account for scatter from air, amorphous
        regions, or thermal vibrations and fluorescence from the sample
        or diffracted beams. Successful execution may temporarily store
        the backgrounds internally.

        Parameters
        ----------
        method : str, optional
            String indicating the method for estimating background.
            Many different variants are accepted, but currently
            supported methods for estimating backgrounds are 'median',
            'minimum', 'none', and 'custom'.
            The method parameter is only considered if background is
            None and the default value is to not estimate any
            background or 'none'.
        background : 1D or 3D array, optional
            Array with dimensions matching either the integration shape
            or full dataset shape that will be used as a custom
            background. None by default and will be estimated with the
            method perscribed.
        kwargs : dict, optional
            Dictionary of kwargs passed to the individual method for
            estimating backgrounds.

        Raises
        ------
        NotImplementedError for not fully implemented methods for
        estimating background: 'rolling_ball', 'spline',
        'polynomial', 'gaussian', and 'bruckner'.
        """
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
        """
        Remove integration backgrounds.

        This method is not yet implemented.

        Subtract backgrounds from integrations if provided or stored
        internally.

        Parameters
        ----------
        background : 1D or 3D array, optional
            Array with dimensions matching either the integration shape
            or full dataset shape that will be used as a custom
            background. None by default and will use internal
            integration backgrounds.
        """
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
        """
        Rescale integrations inplace.

        Rescale integrations inplace into more useful intensity range.
        This method cannot be easily reversed.

        Parameters
        ----------
        lower : float, optional
            New lower bound of rescaled integrations. 0 by default.
        upper : float, optional
            New upper bound of rescaled integrations. 100 by default.
        arr_min : float, optional
            Minimum value of array used for rescaling. If None, the
            minimum value of integrations is used. If this number
            matches the 'lower' parameter, then the integrations lower
            value will not be changed. 0 by default.
        arr_max : float, optional
            Maximum value of array used for rescaling. If None, the
            maximum value of integrations is used. If this number
            matches the 'upper' parameter, then the integrations upper
            value will not be changed. None by default.
        """
        
        rescale_array(
            self.integrations,
            lower=lower,
            upper=upper,
            arr_min=arr_min,
            arr_max=arr_max)


    ####################
    ### IO Functions ###
    ####################

    def disk_size(self,
                  return_val=False,
                  dtype=None):
        """
        Get current images disk size.

        Print current images disk size is human readable format.

        Parameters
        ----------
        return_val : bool, optional
            Flag to indicate if the disk size and units should be
            returned. False by default.
        dtype : dtype, optional
            Datatype used to determine dataset size. Default is current
            dataset dtype, but can be used to estimate new dtype memory
            usage.
        
        Returns
        -------
        disk_size : float, optional
            Disk size of images if return_val is True. The units of
            this value is described by the 'units' returned value.
        units : str, optional
            Units of 'disk_size' if return_val is True. Possible values
            are 'B', 'KB', 'MB', and 'GB'.
        """
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
            return disk_size, units
        print(f'Disk size of images is {disk_size:.3f} {units}.')


    # WIP apparently???
    # @staticmethod
    # def estimate_disk_size(size):
    #     # External reference function to estimate map size before acquisition
    #     raise NotImplementedError()

    #     if not isinstance(size, (tuple, list, np.ndarray)):
    #         err_str = ('Size argument must be iterable of '
    #                    + 'map dimensions.')
    #         raise TypeError(err_str)

    #     disk_size = np.prod([*size, 2])


    def _get_save_labels(self,
                         arr_shape,
                         base_labels=None,
                         units='a.u.'):
        """
        Internal parameter to get save labels for HDF file.
        """
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


    @_protect_hdf()
    def save_images(self,
                    images=None,
                    title=None,
                    units=None,
                    labels=None,
                    compression=None,
                    compression_opts=None,
                    extra_attrs=None):
        """
        Save images to HDF file.

        Save images to 'image_data' in HDF File. This function is valid
        for full 4D images dataset and individual images. Several other
        parameters and details are written along with the image data.

        Parameters
        ----------
        images : 2D or 4D array or str, optional
            2D array as an image, 4D array as full image dataset, or
            string indicating an attribute to be written to HDF file.
            If this parameter is not provided, the current images will
            be used.
        title : dtype, optional
            Title to give the new dataset in the HDF file. If the saved
            array is 2D, the the title will be prepended with '_'. If
            title is None, then the current image title or attribute
            being saved will be used.
        units : str, optional
            Description of the units each image pixel represent. By
            default this is None and 'a.u.' will be used.
        labels : iterable of str, optional
            List of labels used for each axis of array. The length
            should match the number of dimensions in saved array. By
            default this uses the default dataset labels.
        compression : str, optional
            String describing the compression used in the h5py dataset.
            Defaults to 'gzip' for 4D images.
        compression_opts : int, optional
            Options corresponding to the compression string used in the
            h5py dataset. Defaults to 4 for 4D images if compression is
            None.
        extra_attrs : dict, optional
            Extra metadata used to describe the saved images.
        
        Raises
        ------
        AttributeError if images are not explicitly provided and cannot
        be found internally.
        ValueError if title is not provided for custom images.
        ValueError if images is not 2D or 4D.
        TypeError if images cannot be parsed.
        """
        
        # Save all images
        if images is None:
            if not hasattr(self, 'images') or self.images is None:
                err_str = 'Must provide images to write to hdf.'
                raise AttributeError(err_str)
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
            raise ValueError(err_str)
        elif images.ndim == 4 and compression is None:
            compression = 'gzip'
            compression_opts = 4
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
        
        overwrite_attr(dset.attrs, 'labels', labels)
        overwrite_attr(dset.attrs, 'units', units)
        overwrite_attr(dset.attrs, 'dtype', str(image_dtype))
        dset.attrs['time_stamp'] = ttime.ctime()

        # Add non-standard extra metadata attributes
        if extra_attrs is not None:
            for key, value in extra_attrs.items():
                overwrite_attr(dset.attrs, key, value)

        # Add correction information to each dataset
        if title[0] != '_':
            for key, value in self.corrections.items():
                overwrite_attr(dset.attrs, f'_{key}_correction', value)


    @_protect_hdf()
    def save_integrations(self,
                          integrations=None,
                          title=None,
                          units=None,
                          labels=None,
                          extra_attrs=None):
        """
        Save integrations to HDF file.

        Save integrations to 'integration_data' in HDF File. This
        function is valid for full 3D integration datasets and
        individual integrations. Several other parameters and details
        are written along with the integration data.

        Parameters
        ----------
        integrations : 1D or 3D array or str, optional
            1D array as an integration, 3D array as full integration 
            dataset, or string indicating an attribute to be written to
            HDF file. If this parameter is not provided, the current
            integrations will be used.
        title : dtype, optional
            Title to give the new dataset in the HDF file. If the saved
            array is 1D, the the title will be prepended with '_'. If
            title is None, then the current integration title or
            attribute being saved will be used.
        units : str, optional
            Description of the units each integration value represent.
            By default this is None and 'a.u.' will be used.
        labels : iterable of str, optional
            List of labels used for each axis of array. The length
            should match the number of dimensions in saved array. By
            default this uses the dataset labels.
        extra_attrs : dict, optional
            Extra metadata used to describe the saved integrations.
        
        Raises
        ------
        AttributeError if integrations are not explicitly provided and
        cannot be found internally.
        ValueError if title is not provided for custom integrations.
        ValueError if integrations is not 1D or 3D.
        TypeError if integrations cannot be parsed.
        """

        # Save all integrations
        if integrations is None:
            if (not hasattr(self, 'integrations')
                or self.integrations is None):
                err_str = 'Must provide integrations to write to hdf.'
                raise AttributeError(err_str)
            integrations = self.integrations
            if title is None:
                title = f'{self.title}_integrations'
        
        # Save particular attribute of XRDData
        elif isinstance(integrations, str):
            # Can directly grab attributes.
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
            # This conditional is for single integrations mostly
            integrations = np.asarray(integrations)
            if title is None:
                err_str = 'Must define title to save custom images.'
                raise ValueError(err_str)
        
        # Check shape and set title
        if integrations.ndim == 1:
            if title[0] != '_':
                title = f'_{title}'
        elif integrations.ndim != 3:
            err_str = ('Integrations must have 1 or 3 dimensions, '
                       + f'not {len(integrations.shape)}.')
            raise ValueError(err_str)
        
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
        
        overwrite_attr(dset.attrs, 'labels', labels)
        overwrite_attr(dset.attrs, 'units', units)
        overwrite_attr(dset.attrs, 'dtype', str(integrations_dtype))
        dset.attrs['time_stamp'] = ttime.ctime()

        # Add non-standard extra metadata attributes
        if extra_attrs is not None:
            for key, value in extra_attrs.items():
                overwrite_attr(dset.attrs, key, value)
                # dset.attrs[key] = value

        # Add correction information to each dataset
        if title[0] != '_':
            for key, value in self.corrections.items():
                overwrite_attr(dset.attrs, f'_{key}_correction', value)
                # dset.attrs[f'_{key}_correction'] = value
    