import numpy as np
import os
import h5py
import pyFAI
import scipy
import pandas as pd
from pyFAI.io import ponifile
from enum import IntEnum # Only for ponifile orientation
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
import time as ttime
import matplotlib.pyplot as plt
from collections import OrderedDict
import dask.array as da
import skimage.io as io
from dask_image import imread as dask_io
from tqdm import tqdm

# Local imports
from xrdmaptools.XRDData import XRDData
from xrdmaptools.utilities.math import *
from xrdmaptools.utilities import reference_data
from xrdmaptools.utilities.utilities import (
    delta_array,
    pathify,
    _check_dict_key,
    generate_intensity_mask
)
from xrdmaptools.io.hdf_io import (
    initialize_xrdbase_hdf,
    load_xrdbase_hdf,
    _load_xrd_hdf_vector_data
    )
from xrdmaptools.io.hdf_utils import (
    check_hdf_current_images,
    check_attr_overwrite,
    overwrite_attr,
)
from xrdmaptools.plot.general import (
    _plot_parse_xrdbasescan,
    _xrdbasescan_image,
    _xrdbasescan_integration,
    plot_image,
    plot_integration
    )
from xrdmaptools.plot.geometry import (
    plot_q_space,
    plot_detector_geometry
)
from xrdmaptools.geometry.geometry import *
from xrdmaptools.crystal.Phase import Phase, phase_selector


class XRDBaseScan(XRDData):
    """
    Base class for working with XRD scans, built from XRDData. This 
    class adds general utility analyzing and interpreting XRD data 
    without any assumptions or special considerations for the 
    individual experiment type. This class also lays the groundwork for
    hdf file read/write capabilities.

    Parameters
    ----------
    scan_id : int or str, optional
        Unique identifier for the scan used to acquire the XRD data.
        May be string if connecting series of scan IDs.
    wd : path str, optional
        Path str indicating the main working directory for the XRD
        data. Will be used as the default read/write location.
        Defaults to the current working directy.
    filename : str, optional
        Custom file name if not provided. Defaults to include scan ID
        and child class type.
    hdf_filename : str, optional
        Custom file name of hdf file. Defaults to include the filename
        parameter if not provided.
    hdf : h5py File, optional
        h5py File instance. Will be used to derive hdf_filename and hdf
        path location if provided.
    energy : float, optional
        Energy of incident X-rays. Will be used to determine wavelength
        if provided.
    wavelength : float, optional
        Wavelength of incident X-rays. Will be used to determin energy
        if energy is not provided.
    dwell : float, optional
        Dwell time of pixel/frame/image.
    theta : float, optional
        Angle of stage rotation about the vertical (y-axis). Defaults
        to 0, and currently only degree units are supported.
    use_stage_rotation : bool, optional
        Flag to indicate if the stage rotation should be included in
        determining recprocal space values. This should only be true
        when using the stage rotation to find specific reflections or
        to rock through reciprocal space. Default is false.
    poni_file : path str, OrderedDict, PoniFile, optional
        Calibration parameters for instantiating an AzimuthalIntegrator
        from pyFAI. Can be provided as a path string to a .poni file,
        and OrderedDict of parameters, or a PoniFile instance.
    sclr_dict : dict, optional
        Dictionary of 2D numpy arrays matching the map shape with
        scaler intensities used for intensity normalization.
    check_init_sets : bool, optional
        Flag to disable overwriting of AzimuthalIntegrator calibration
        parameters and scaler and position dictionaries. Only intended
        to be True when loading for hdf file, default is False.
    tth_resolution : float, optional
        Resolution of scattering angles for 1D image integrations and
        the x-axis of 2D image integrations. 0.01 degrees by default
        and currently only supports degree units.
    chi_resolution : float, optional 
        Resolution of azimuthal angles for the y-axis of 2D image
        integrations. 0.05 deg by default and currently only supports
        degree units.
    tth : iterable, optional
        Iterable of scattering angles used to interpret 1D image
        integrations or the x-axis of 2D image integrations. None by
        default, and length should match integrations of XRDData if
        provided and currently only supports degree units.
    chi : iterable, optional
        Iterable of azimuthal angles used to interpret the y-axis of
        cake or 2D image integrations. None by default and currently
        only supports degree units.
    beamline : str, optional
        String to record the beamline where the XRD data was acquired.
    facility : str, optional
        String to record the facility where the XRD data was acquired.
    scan_input : iterable, optional
        List of input parameters for the scan generating the XRD data.
        Should be given as [xstart, xend, xnum, ystart, yend, ynum, *].
        These values are used to determine the map extent if given.
    time_stamp : str, optional
        String intended to indicate the time which the XRD data was
        first acquired. Defaults to current time if not given.
    extra_metadata : dict, optional
        Dictionary of extra metadata to be stored with the XRD data.
        Extra metadata will be written to the hdf file if enabled, but
        is not intended to be interacted with during normal data
        processing.
    save_hdf : bool, optional
        If False, this flag disables all hdf read/write functions.
        True by default.
    dask_enabled : bool, optional
        Flag to indicate whether the image data should be lazily loaded
        as a Dask array. Default is False.
    extra_attrs : dict, optional
        Dictionary of extra attributes to be given to XRDBaseScan.
        These attributes are intended for those values generated during
        processing of the XRD data.
    xrddatakwargs : dict, optional 
        Dictionary of all other kwargs for parent XRDData class.
    
    Raises
    ------
    ValueError 
    """

    # Class variables
    _hdf_type = 'xrdbase'

    def __init__(self,
                 scan_id=None,
                 wd=None,
                 filename=None,
                 hdf_filename=None,
                 hdf=None,
                 energy=None,
                 wavelength=None,
                 dwell=None,
                 theta=None,
                 use_stage_rotation=False,
                 poni_file=None,
                 sclr_dict=None,
                 check_init_sets=False,
                 tth_resolution=None,
                 chi_resolution=None,
                 tth=None, # Used when loading from hdf
                 chi=None, # Used when loading from hdf
                 beamline='5-ID (SRX)',
                 facility='NSLS-II',
                 scan_input=None,
                 time_stamp=None,
                 extra_metadata=None,
                 save_hdf=True,
                 dask_enabled=False,
                 extra_attrs=None,
                 **xrddatakwargs
                 ):
        
        # Adding some metadata
        self.scan_id = scan_id
        if filename is None:
            filename = f'scan{scan_id}_{self._hdf_type}'
        self.filename = filename

        if wd is None:
            wd = os.getcwd()
        self.wd = wd

        self.beamline = beamline
        self.facility = facility
        if time_stamp is None:
            time_stamp = ttime.ctime()
        self.time_stamp = time_stamp
        self.scan_input = scan_input
        self.dwell = dwell
        if extra_metadata is None:
            extra_metadata = {}
        self.extra_metadata = extra_metadata

        # Pre-define some values
        self._energy = np.nan
        self._wavelength = np.nan
        self._theta = np.nan
        self._use_stage_rotation = False # overwrite later

        if not save_hdf:
            if dask_enabled:
                err_str = ('Enabling dask requires an '
                           + 'hdf file for storage.')
                raise ValueError(err_str)
            else:
                self.hdf_path = None
                self.hdf = None
        else:
            self.start_saving_hdf(hdf=hdf,
                                  hdf_filename=hdf_filename,
                                  dask_enabled=dask_enabled)
        
        XRDData.__init__(
            self,
            hdf_path=self.hdf_path,
            hdf=self.hdf,
            dask_enabled=dask_enabled,
            hdf_type=self._hdf_type, # Gets redefined as same value...
            **xrddatakwargs
        )
            
        # Store energy, dwell, and theta
        if energy is not None:
            self.energy = energy
        elif wavelength is not None: # Favors energy definition
            self.wavelength = wavelength
        else:
            print('WARNING: No energy or wavelength provided.')
            self.energy = np.nan
        
        # self.dwell = dwell
        if theta is None:
            print('WARNING: No theta provided. Assuming 0 degrees.')
            theta = 0
        self.theta = theta
        self.use_stage_rotation = bool(use_stage_rotation)
        
        self.phases = {} # Place holder for potential phases
        if poni_file is not None:
            self.set_calibration(poni_file,
                                 check_init_sets=check_init_sets)
        else:
            self.ai = None # Place holder for calibration

        self.sclr_dict = None
        if sclr_dict is not None:
            self.set_scalers(sclr_dict,
                             check_init_sets=check_init_sets)
        
        # Default units and flags
        # Not fully implemented
        # 'rad' or 'deg'
        self._polar_units = 'deg' 
        # 'rad', 'deg', 'nm^-1', 'A^-1'
        self._scattering_units = 'deg'
        # 'linear' or 'log'
        self._image_scale = 'linear' 

        if tth is not None and len(tth) == 0:
            tth = None
        self.tth = tth
        if tth_resolution is None:
            tth_resolution = 0.01 # in degrees...
        self.tth_resolution = tth_resolution

        if chi is not None and len(chi) == 0:
            chi = None
        self.chi = chi
        if chi_resolution is None:
            chi_resolution = 0.05 # in degrees...
        self.chi_resolution = chi_resolution
        
        # Catch-all of extra attributes.
        # Gets them into __init__ sooner
        if extra_attrs is not None:
            for key, value in extra_attrs.items():
                setattr(self, key, value)

    
    # Overwrite parent functions
    def __str__(self):
        """
        A simple represenation of the class.

        Returns
        -------
        outstring : str
            A simple representation of the class.
        """
        ostr = (f'{self._hdf_type}:  scan_id={self.scan_id}, '
                + f'energy={self.energy}, '
                + f'shape={self.images.shape}')
        return ostr


    # Overwrite parent function
    def __repr__(self):
        """
        A nice representation of the class with relevant information.

        Returns
        -------
        outstring : str
            A nice representation of the class with relevant
            information.
        """

        # Native info
        ostr = f'{self._hdf_type}:'
        ostr += f'\n\tFacility:\t{self.facility}'
        ostr += f'\n\tBeamline:\t{self.beamline}'
        if self.scan_id is not None:
            ostr += f'\n\tscan_id:\t\t{self.scan_id}'
        ostr += f'\n\tEnergy:\t\t{self.energy} keV'
        if self.hdf_path is not None:
            ostr += f'\n\tHDF Path:\t{self.hdf_path}\n'

        # Data info
        ostr += ('\t'
                 + '\t'.join(XRDData.__repr__(self).splitlines(True)))

        # Other info
        if hasattr(self, 'ai'): # pull geometry info
            ostr += '\n\tGeometry:  \n'
            ostr += ('\t\t'
                    + '\t\t'.join(self.ai.__repr__().splitlines(True)))
        if len(self.phases) > 0: # pull phase info
            ostr += '\n\tPhases:'
            for key in self.phases.keys():
                ostr += ('\n\t\t' + '\t\t'.join(self.phases[key].__repr__().splitlines(True)))
        if hasattr(self, 'spots'): # pull limited spot info
            ostr += '\n\tSpots:'
            ostr += '\n\t\tNumber:  ' + str(len(self.spots))
            if hasattr(self, 'spot_model'):
                ostr += ('\n\t\tModel Fitting:  '
                         + self.spot_model.name
                         + ' Spot Model')
        return ostr

    #####################################
    ### Loading data into XRDBaseScan ###
    #####################################

    @classmethod
    def from_hdf(cls,
                 hdf_filename,
                 wd=None,
                 dask_enabled=False,
                 image_data_key='recent',
                 integration_data_key='recent',
                 load_blob_masks=True,
                 load_vector_map=False, # Redundant information
                 map_shape=None,
                 image_shape=None,
                 **kwargs):
        """
        Instantiate class from data in HDF file.

        Loads data from HDF file of correct HDF type to instantiate
        class. This method will work with different child classes.

        Parameters
        ----------
        hdf_filename : str
            Name of HDF file.
        wd : path str, optional
            Path to HDF file. Default is current working directory.
        dask_enabled : bool, optional
            Flag to indicate whether the image data should be lazily
            loaded as a Dask array. Default is False.
        image_data_key : str, optional
            Dataset title in image_data group within HDF file used to
            load image data. Defaults to 'recent', which will load the
            most recently written image data.
        integration_data_key : str, optional
            Dataset title in integration_data group within HDF file
            used to load integration data. Defaults to 'recent',
            which will load the most recently written integration data.
        load_blob_masks : bool, optional
            Flag to load blob_masks from HDF file. Default is True.
        load_vector_map : bool, optional
            Flag to load vector_map and edges from HDF file. Default is
            True.
        map_shape : iterable length 2, optional
            Shape of first two axes in image_data and integration_data
            as (map_y, map_x) or (rocking_axis, 1).
        image_shape : iterable length 2, optional
            Shape of last two axes in image_data (image_y, image_x).
        kwargs : dict, optional
            Dictionary of keyword arguments passed to instantiate the
            class. These should not overlap with other keyword
            arguments read from the HDF file.
        
        Returns
        -------
        inst : instance of XRDBaseScan, XRDMap, or XRDRockingCurve
            Returns instance of XRDBaseScan or child classes (XRDMap or
            XRDRockingCurve) loaded from the HDF file.
        """
        
        if wd is None:
            wd = os.getcwd()
        
        # Load from previously saved data, including all processed data...
        hdf_path = pathify(wd, hdf_filename, '.h5')
        
        # File exists, attempt to load data
        print('Loading data from hdf file...')
        input_dict = load_xrdbase_hdf(
                        os.path.basename(hdf_path),
                        cls._hdf_type,
                        os.path.dirname(hdf_path),
                        image_data_key=image_data_key,
                        integration_data_key=integration_data_key,
                        load_blob_masks=load_blob_masks,
                        load_vector_map=load_vector_map,
                        map_shape=map_shape,
                        image_shape=image_shape,
                        dask_enabled=dask_enabled)

        # Remove several kwargs to allow for unpacking
        base_md = input_dict.pop('base_md')
        image_attrs = input_dict.pop('image_attrs')
        image_corrections = input_dict.pop('image_corrections')
        image_data_key = input_dict.pop('image_data_key')
        integration_attrs = input_dict.pop('integration_attrs')
        integration_corrections = input_dict.pop(
                                    'integration_corrections')
        integration_data_key = input_dict.pop(
                                    'integration_data_key')
        recip_pos = input_dict.pop('recip_pos')
        # phases = input_dict.pop('phases')
        # spots = input_dict.pop('spots')
        # spot_model = input_dict.pop('spot_model')
        # spots_3D = input_dict.pop('spots_3D')
        vector_dict = input_dict.pop('vector_dict')
        
        # Other kwargs needing special treatment
        # Bias swapped axes towards user input.
        # Might break things switching back and forth...it does.
        if 'swapped_axes' in kwargs:
            if 'swapped_axes' in base_md:
                del base_md['swapped_axes']

        # Scrub data keys. For backward compatibility
        if image_data_key is not None:
            image_title = '_'.join([x for x in 
                                    image_data_key.split('_')
                                    if x not in ['images',
                                                    'integrations']])
        else:
            image_title = 'empty'
        if integration_data_key is not None:
            integration_title = '_'.join([x for x in
                                    integration_data_key.split('_')
                                    if x not in ['images',
                                                    'integrations']])
        else:
            integration_title = 'empty'

        # Compare image and integration data
        title = image_title
        corrections = image_corrections
        if (input_dict['image_data'] is not None
            and input_dict['integration_data'] is not None):

            if image_title != integration_title:
                warn_str = (f'WARNING: Image data from '
                            + f'({image_data_key}) does '
                            + 'not match integration data '
                            + f'from ({integration_data_key}).')
                print(warn_str)
            
            # Check corrections if the titles match
            elif not np.all([val1 == val2
                        for val1, val2 in zip(
                            image_corrections.values(),
                            integration_corrections.values())]):
                warn_str = ('WARNING: Different corrections '
                            + 'applied to images and integrations.'
                            + ' Using image corrections.')
                print(warn_str)
        elif (input_dict['image_data'] is None
                and input_dict['integration_data'] is not None):
            title = integration_title # truncate _integrations
            corrections = integration_corrections  

        # Set up extra attributes that do not go into __init__
        extra_attrs = {}
        extra_attrs.update(image_attrs)
        extra_attrs.update(integration_attrs)
        for attr_name in ['phases',
                          'spots',
                          'spot_model',
                          'spots_3D']:
            if input_dict[attr_name] is not None:
                extra_attrs[attr_name] = input_dict.pop(attr_name)
        
        # Remove unused values
        # (keeps pos_dict out of rocking curve...)
        for key, value in list(input_dict.items()):
            if value is None:
                del input_dict[key]

        # # Set up extra attributes
        # extra_attrs = {}
        # extra_attrs.update(image_attrs)
        # extra_attrs.update(integration_attrs)
        # # Add phases
        # extra_attrs['phases'] = phases
        # # Add spots
        # if spots is not None:
        #     extra_attrs['spots'] = spots
        # if spot_model is not None:
        #     extra_attrs['spot_model'] = spot_model
        # if spots_3D is not None:
        #     extra_attrs['spots_3D'] = spots_3D

        # Add vector information. Not super elegant
        if vector_dict is not None:
            extra_attrs.update(cls._parse_vector_dict(vector_dict))
            # for key, value in vector_dict.items():
            #     # Check for int cutoffs
            #     # Wait to make sure intensity is processed
            #     if key == 'blob_int_cutoff':
            #         # Backwards compatibility
            #         if 'relative_cutoff' in vector_dict:
            #             relative_cutoff = bool(vector_dict['relative_cutoff'])
            #         else:
            #             relative_cutoff = True
            #         intensity = vector_dict['vectors'][:, -1]
            #         (extra_attrs['blob_int_mask']
            #         ) = generate_intensity_mask(
            #             intensity,
            #             int_cutoff=vector_dict['blob_int_cutoff'],
            #             relative_cutoff=relative_cutoff)
            #     elif key == 'spot_int_cutoff':
            #         # Backwards compatibilty
            #         if 'relative_cutoff' in vector_dict:
            #             relative_cutoff = bool(vector_dict['relative_cutoff'])
            #         else:
            #             relative_cutoff = True
            #         intensity = vector_dict['vectors'][:, -1]
            #         (extra_attrs['spot_int_mask']
            #         ) = generate_intensity_mask(
            #             intensity,
            #             int_cutoff=vector_dict['spot_int_cutoff'],
            #             relative_cutoff=relative_cutoff)
            #     else:
            #         extra_attrs[key] = value

        # Instantiate XRDBaseScan
        inst = cls(**input_dict,
                    **base_md,
                    **recip_pos,
                    title=title,
                    corrections=corrections,
                    wd=wd,
                    filename=hdf_filename[:-3], # remove the .h5 extention
                    hdf_filename=hdf_filename,
                    dask_enabled=dask_enabled,
                    extra_attrs=extra_attrs,
                    check_init_sets=True, # Don't overwrite hdf values
                    **kwargs)
        
        print(f'{cls.__name__} loaded!')
        return inst
        
    
    @classmethod 
    def from_image_stack(cls,
                         filename,
                         wd=None,
                         title='raw_images',
                         dask_enabled=False,
                         **kwargs):
        """
        Instantiate class from stack of images.

        Load images from stack of images. Only .tif, .tiff, .jpeg, and
        .png files are accepted.

        Parameters
        ----------
        filename : str
            Name of images file.
        wd : path str, optional
            Path to images file. Default is current working directory.
        title : str, optional
            Custom title used to title HDF datasets when saving. Will
            be updated to default value after processing.
        dask_enabled : bool, optional
            Flag to indicate whether the image data should be lazily
            loaded as a Dask array. Default is False.
        kwargs : dict, optional
            Dictionary of keyword arguments passed to instantiate the
            class.

        Returns
        -------
        inst : instance of XRDBaseScan, XRDMap, or XRDRockingCurve
            Returns instance of XRDBaseScan or child classes (XRDMap or
            XRDRockingCurve) loaded from the HDF file.
        """
        
        # Load from image stack
        if wd is None:
            wd = os.getcwd()
        
        image_path = pathify(wd,
                             filename,
                             ['.tif', '.tiff', '.jpeg', '.png'])

        print('Loading images...', end='', flush=True)
        if dask_enabled:
            image_data = dask_io.imread(image_path)
        else:
            image_data = io.imread(image_path)
        print('done!')

        return cls(image_data=image_data,
                   wd=wd,
                   title=title,
                   **kwargs)
    
    
    ##################
    ### Properties ###
    ##################

    @property
    def energy(self):
        """
        Get energy in keV. Setting this value will also change
        wavelength and write changes to the HDF file if available.
        """
        return self._energy

    @energy.setter
    def energy(self, energy):
        if (np.all(energy is not None)
            and np.all(~np.isnan(energy))):
            self._energy = energy
            self._wavelength = energy_2_wavelength(energy)
        else:
            self._energy = energy
        # Propogate changes...
        if hasattr(self, 'ai') and self.ai is not None:
            # ai energy is not used by any methods called here
            self.ai.energy = self._energy 
            if hasattr(self, '_q_arr'):
                delattr(self, '_q_arr')
        
        # Re-write hdf values
        @XRDData._protect_hdf()
        def save_attrs(self): # Not sure if this needs self...
            attrs = self.hdf[self._hdf_type].attrs
            overwrite_attr(attrs, 'energy', self.energy)
            overwrite_attr(attrs, 'wavelength', self.wavelength)
        save_attrs(self)


    @property
    def wavelength(self):
        """
        Get wavelength in Angstroms. Setting this value will also
        change energy and write changes to the HDF file if available.
        """
        return self._wavelength

    @wavelength.setter
    def wavelength(self, wavelength):
        if (np.all(wavelength is not None)
            and np.all(~np.isnan(wavelength))):
            self._wavelength = wavelength
            self._energy = wavelength_2_energy(wavelength)
        else:
            self._wavelength = wavelength
        # Propogate changes...
        if hasattr(self, 'ai') and self.ai is not None:
            self.ai.energy = self._energy
            if hasattr(self, '_q_arr'):
                delattr(self, '_q_arr')

        # Re-write hdf values
        @XRDData._protect_hdf()
        def save_attrs(self): # Not sure if this needs self...
            attrs = self.hdf[self._hdf_type].attrs
            overwrite_attr(attrs, 'energy', self.energy)
            overwrite_attr(attrs, 'wavelength', self.wavelength)
        save_attrs(self)
        
    
    # y-axis stage rotation
    @property
    def theta(self):
        """
        Get stage rotation, theta, in degrees. Setting this value will
        write changes to the HDF file if available.
        """
        return self._theta
    
    @theta.setter
    def theta(self, theta):
        self._theta = theta
        # Propagate changes
        if hasattr(self, 'ai') and self.ai is not None:
            if hasattr(self, '_q_arr'):
                delattr(self, '_q_arr')

        # Re-write hdf values
        @XRDData._protect_hdf()
        def save_attrs(self): # Not sure if this needs self...
            overwrite_attr(self.hdf[self._hdf_type].attrs,
                           'theta',
                           self.theta)
        save_attrs(self)


    @property
    def use_stage_rotation(self):
        """
        Get flag to indicate whether stage rotation should be used.
        Setting this value will write changes to the HDF file if
        available.
        """
        return self._use_stage_rotation

    @use_stage_rotation.setter
    def use_stage_rotation(self, use_stage_rotation):
        self._use_stage_rotation = use_stage_rotation
        # Propagate changes
        if hasattr(self, 'ai') and self.ai is not None:
            if hasattr(self, '_q_arr'):
                delattr(self, '_q_arr')

        # Re-write hdf values
        @XRDData._protect_hdf()
        def save_attrs(self): # Not sure if this needs self...
            overwrite_attr(self.hdf[self._hdf_type].attrs,
                           'use_stage_rotation',
                           int(self.use_stage_rotation))
        save_attrs(self)


    # Flags for units and scales

    def _angle_units_factory(property_name, options):
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
                err_str = (err_str[:-2]
                           + f' are supported for {property_name}')
                raise ValueError(err_str)
        
        return property(get_angle_units, set_angle_units)
    

    scattering_units = _angle_units_factory('scattering_units',
                                    ['rad', 'deg', '1/nm', '1/A'])
    polar_units = _angle_units_factory('polar_units',
                                    ['rad', 'deg'])
    

    def _scale_property_factory(property_name):
        def get_scale(self):
            return getattr(self, f'_{property_name}')

        def set_scale(self, val):
            if val in ['linear', 'log']:
                setattr(self, f'_{property_name}', val)
            else:
                raise ValueError(f"{property_name} can only have "
                                 "'linear' or 'log' scales.")
            
        return property(get_scale, set_scale)
    

    image_scale = _scale_property_factory('image_scale')
    integration_scale = _scale_property_factory('integration_scale')

    # Convenience properties for working with the detector arrays
    # These are mostly wrappers for pyFAI functions

    def _detector_angle_array_factory(arr_name, ai_arr_name, units):
        def get_angle_array(self):
            if hasattr(self, f'_{arr_name}'):
                return getattr(self, f'_{arr_name}')
            elif (self.corrections['polar_calibration']):
                # I should rename...
                if hasattr(self, 'tth') and hasattr(self, 'chi'):
                    # Set both tth and chi!
                    tth_arr, chi_arr = np.meshgrid(self.tth,
                                                   self.chi[::-1])
                    self._tth_arr = tth_arr
                    self._chi_arr = chi_arr
                    return getattr(self, f'_{arr_name}')
            elif hasattr(self, 'ai') and self.ai is not None:
                # default is radians
                ai_arr = getattr(self.ai, ai_arr_name)()
                if arr_name == 'chi_arr':
                    # Negative to match SRX coordinates
                    ai_arr = -ai_arr 

                if getattr(self, units) == 'rad':
                    pass
                elif getattr(self, units) == 'deg':
                    ai_arr = np.degrees(ai_arr)
                elif getattr(self, units) == '1/nm':
                    err_str = '1/nm units not yet fully supported.'
                    raise NotImplementedError(err_str)
                elif getattr(self, units) == '1/A':
                    err_str = '1/A units not yet fully supported.'
                    raise NotImplementedError(err_str)
                else:
                    raise ValueError('Unknown units specified.')

                setattr(self, f'_{arr_name}', ai_arr)
                return getattr(self, f'_{arr_name}')
            
            err_str = 'AzimuthalIntegrator (ai) not specified.'
            raise AttributeError(err_str)

        def del_angle_array(self):
            delattr(self, f'_{arr_name}')

        # Delta arrays are not cached
        def get_delta_array(self):
            arr = getattr(self, arr_name) # should call the property

            max_arr = np.max(np.abs(arr))
            delta_arr = delta_array(arr)

            # Modular shift values if there is a discontinuity
            # Should only be for chi_arr between 180 and -180 degrees
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
    

    tth_arr, delta_tth = _detector_angle_array_factory('tth_arr',
                                           'twoThetaArray',
                                           'polar_units')
    chi_arr, delta_chi = _detector_angle_array_factory('chi_arr',
                                           'chiArray',
                                           'polar_units')


    # Full q-vector, not just magnitude
    @property
    def q_arr(self):
        """
        Get 3D vector coordinates is reciprocal space (q-space) for 
        every pixel in image with shape (image_shape, 3). These values
        are detemined from tth_arr, chi_arr, and wavelength parameters.
        This array is cached and can be cleared by deleting.
        """

        if hasattr(self, '_q_arr'):
            return self._q_arr
        elif not hasattr(self, 'ai'):
            err_str = 'Cannot calculate q-space without calibration.'
            raise RuntimeError(err_str)
        else:
            if self.use_stage_rotation:
                theta = self.theta
            else:
                theta = None

            q_arr = get_q_vect(self.tth_arr,
                               self.chi_arr,
                               wavelength=self.wavelength,
                               stage_rotation=theta,
                               degrees=self.polar_units == 'deg')
            self._q_arr = q_arr.astype(self.dtype)
            # self._q_arr = q_arr

            return self._q_arr

    @q_arr.deleter
    def q_arr(self):
        del self._q_arr
    
    # Convenience function
    def _del_arr(self):
        """
        Internal function for deleting chached arrays related to
        detector calibration. Useful when changing incident X-ray
        energy/wavelength, sample rotation (theta) or detector
        calibration parameters.
        """

        if hasattr(self, '_tth_arr'):
            delattr(self, '_tth_arr')
        if hasattr(self, '_chi_arr'):
            delattr(self, '_chi_arr')
        if hasattr(self, '_q_arr'):
            delattr(self, '_q_arr')
        if hasattr(self, 'ai'):
            self.ai._cached_array = {}

    
    ############################
    ### Dask / HDF Functions ###
    ############################

    def start_saving_hdf(self,
                         hdf=None,
                         hdf_filename=None,
                         hdf_path=None,
                         dask_enabled=False,
                         save_current=False,
                         verbose=True):
        """
        Start saving data to HDF.

        If HDF does not already exist, a new version will be 
        initialized from current scan parameters. Defaulting naming and
        path will be used when not provided.

        Parameters
        ----------
        hdf : HDF File Object, optional
            HDF File Object that will be used for writing additional
            information.
        hdf_filename : str, optional
            Name used for HDF file or name of previous HDF file.
        hdf_path : path str, optional
            Path used for writing HDF file or path of previous HDF
            file.
        dask_enabled : bool, optional
        save_current : bool, optional 
        verbose : bool, optional

        """
        
        # Check for previous iterations
        if ((hasattr(self, 'hdf')
             and self.hdf is not None)
            or (hasattr(self, 'hdf_path')
                and self.hdf_path is not None)):
            os_str = ('WARNING: Trying to save to hdf, but a '
                      'file or location has already been specified!'
                      '\nSwitching save files or locations should '
                      'use the "switch_hdf" function.'
                      '\nProceeding without changes.')
            print(os_str)
            return

        # Specify hdf path and name
        if hdf is not None: # biases towards already open hdf
            # This might break if hdf is a close file and not None
            self.hdf_path = hdf.filename
        elif hdf_filename is None:
            if hdf_path is None:
                self.hdf_path = pathify(self.wd,
                                        self.filename,
                                        '.h5',
                                        check_exists=False)
            else:
                self.hdf_path = pathify(hdf_path,
                                        self.filename,
                                        '.h5',
                                        check_exists=False)
        else:
            if hdf_path is None:
                self.hdf_path = pathify(self.wd,
                                        hdf_filename,
                                        '.h5',
                                        check_exists=False)
            else:
                self.hdf_path = pathify(hdf_path,
                                        hdf_filename,
                                        '.h5',
                                        check_exists=False)

        # Check for hdf and initialize if new            
        if not os.path.exists(self.hdf_path):
            # Initialize base structure
            initialize_xrdbase_hdf(self, self.hdf_path) 

        # Open hdf if required
        if dask_enabled:
            self.hdf = h5py.File(self.hdf_path, 'a')
        else:
            self.hdf = None

        if save_current:
            self.save_current_hdf(verbose=verbose)


    # Saves current major features
    # Calls several other save functions
    def save_current_hdf(self, verbose=False):
        """

        """
        
        if self.hdf_path is None:
            print('WARNING: Changes cannot be written to hdf without '
                  + 'first indicating a file location.\nProceeding '
                  + 'without changes.')
            return # Hard-coded even though all should pass

        if hasattr(self, 'images') and self.images is not None:
            if self._dask_enabled:
                self.dask_2_hdf()
            else:
                self.save_images()

        # Save integrations
        if (hasattr(self, 'integrations')
            and self.integrations is not None):
            self.save_integrations()
        
        if hasattr(self, 'poni') and self.poni is not None:
            self.save_calibration()

        # Save scalers
        if hasattr(self, 'sclr_dict') and self.sclr_dict is not None:
            self.save_sclr_pos('scalers',
                                self.sclr_dict,
                                self.scaler_units)
        
        # Save phases
        if hasattr(self, 'phases') and self.phases is not None:
            self.save_phases(verbose=verbose)

    
    # Ability to toggle hdf saving and proceed without writing to disk.
    def stop_saving_hdf(self):
        """

        """

        if self._dask_enabled:
            err_str = ('WARNING: Image data is lazy loaded. Stopping '
                       + 'or switching hdf is likely to cause '
                       + 'problems.\nSave progress and close the hdf '
                       + 'with "close_hdf" function before changing '
                       + 'save location.')
            raise RuntimeError(err_str)
        
        self.close_hdf()
        self.hdf_path = None
    

    @XRDData._protect_hdf()
    def switch_hdf(self,
                   hdf=None,
                   hdf_path=None,
                   hdf_filename=None,
                   dask_enabled=False,
                   save_current=False,
                   verbose=True):
        """

        """

        # Check to make sure the change is appropriate and correct.
        # Not sure if this should raise and error or just print a warning
        if hdf is None and hdf_path is None:
            ostr = ('Neither hdf nor hdf_path were provided. '
                     + '\nCannot switch hdf save locations without '
                     + 'providing alternative.')
            print(ostr)
            return
        
        elif hdf == self.hdf:
            ostr = (f'WARNING: provided hdf ({self.hdf.filename}) is '
                    + 'already the current save location. '
                    + '\nProceeding without changes')
            print(ostr)
            return
        
        elif hdf_path == self.hdf_path:
            ostr = (f'WARNING: provided hdf_path ({self.hdf_path}) is '
                    + 'already the current save location. '
                    + '\nProceeding without changes')
            print(ostr)
            return
        
        else:
            # Success actually changes the write location
            # And likely initializes a new hdf
            old_base_attrs = dict(self.hdf[self._hdf_type].attrs)

            self.stop_saving_hdf()
            self.start_saving_hdf(hdf=hdf,
                                  hdf_path=hdf_path,
                                  hdf_filename=hdf_filename,
                                  dask_enabled=dask_enabled,
                                  save_current=save_current,
                                  verbose=verbose)
            self.open_hdf()
            
            # Overwrite from old values
            for key, value in old_base_attrs.items():
                self.hdf[self._hdf_type].attrs[key] = value


    ##############################
    ### Calibrating Map Images ###
    ##############################
    
    def set_calibration(self,
                        poni_file,
                        energy=None,
                        wd=None,
                        check_init_sets=False):
        """

        """

        if wd is None:
            wd = self.wd

        if isinstance(poni_file, str):
            poni_path = pathify(wd, poni_file, '.poni')

            print('Setting detector calibration...')
            self.ai = pyFAI.load(poni_path)
        
        elif isinstance(poni_file, OrderedDict):
            print('Setting detector calibration...')
            self.ai = AzimuthalIntegrator().set_config(poni_file)

        elif isinstance(poni_file, ponifile.PoniFile):
            print('Setting detector calibration...')
            self.ai = AzimuthalIntegrator().set_config(
                                            poni_file.as_dict())
        
        else:
            err_str = (f'{type(poni_file)} is unknown '
                       + 'and not supported!')
            raise TypeError(err_str)

        # Update energy if different from poni file
        if energy is None:
            if self.energy is not None:
                # Allows calibrations acquired at any energy
                self.ai.energy = self.energy 
            else:
                ostr = ('Energy has not been defined. '
                        + 'Defaulting to .poni file value.')
                print()
                self.energy = self.ai.energy
        else:
            self.ai.energy = energy # Do not update energy...

        # Update detector shape and pixel
        # size if different from poni file
        try:
            image_shape = list(self.image_shape)
        except AttributeError:
            image_shape = list(self.images.shape[:-2])

        if self.ai.detector.shape != image_shape:
            ostr = ('Calibration performed under different settings. '
                    + 'Adjusting calibration.')
            print(ostr)

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
                            + "image that is not an integral "
                            + "multiple of the current map's images."
                            + "\n\t\tEnsure the calibration is for "
                            + "the correct detector with the "
                            + "appropriate binning.")
                raise ValueError(err_str)

            # Overwrite values
            self.ai.detector.shape = image_shape
            # Not exactly correct, but more convenient
            self.ai.detector.max_shape = image_shape 
            self.ai.detector.pixel1 = poni_pixel1 / bin_est[0]
            self.ai.detector.pixel2 = poni_pixel2 / bin_est[1]

        # Extract calibration parameters to save
        self.poni = self.ai.get_config()

        # Reset any previous calibrated arrays
        self._del_arr()
        
        # Save poni files as dictionary 
        # Updates poni information to update detector settings
        self.save_calibration(check_init_sets=check_init_sets)
    

    @XRDData._protect_hdf()
    def save_calibration(self, 
                         check_init_sets=False):
        """

        """
        
        if check_init_sets:
            if 'reciprocal_positions' in self.hdf[self._hdf_type]:
                if ('poni_file' in 
                    self.hdf[self._hdf_type]['reciprocal_positions']):
                    return

        # Write data to hdf
        curr_grp = self.hdf[self._hdf_type].require_group('reciprocal_positions')
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
                    overwrite_attr(new_new_grp.attrs, key_i, value_i)
            else:
                overwrite_attr(new_grp.attrs, key, value)


    # One off 1D integration
    def integrate1D_image(self,
                          image=None,
                          tth_resolution=None,
                          tth_num=None,
                          unit='2th_deg',
                          **kwargs):
        """

        """

        if image is None:
            if self.corrections['polar_calibration']:
                err_str = ('You are trying to calibrate '
                           + 'already calibrated images!')
                raise RuntimeError(err_str)
            else:
                image = self.composite_image
        
        if tth_resolution is None:
            tth_resolution = self.tth_resolution

        tth_min = np.min(self.tth_arr)
        tth_max = np.max(self.tth_arr)
        if tth_num is None:
            tth_num = int(np.round((tth_max - tth_min)
                                   / tth_resolution))
        elif tth_num is None and tth_resolution is None:
            err_str = 'Must define either tth_num or tth_resolution.'
            raise ValueError(err_str)
        
        return self.ai.integrate1d_ng(image,
                                      tth_num,
                                      unit=unit,
                                      correctSolidAngle=False,
                                      polarization_factor=None,
                                      **kwargs)
    

    # One off 2D integration
    def integrate2D_image(self,
                          image,
                          tth_num=None,
                          tth_resolution=None,
                          chi_num=None,
                          chi_resolution=None,
                          unit='2th_deg',
                          **kwargs):
        """

        """

        if image is None:
            if self.corrections['polar_calibration']:
                err_str = ('You are trying to clibrate '
                           + 'already calibrated images!')
                raise RuntimeError(err_str)
            else:
                image = self.composite_image

        if tth_resolution is None:
            tth_resolution = self.tth_resolution
        if chi_resolution is None:
            chi_resolution = self.chi_resolution

        # Get tth numbers
        tth_min = np.min(self.tth_arr)
        tth_max = np.max(self.tth_arr)
        if tth_num is None:
            tth_num = int(np.round((tth_max - tth_min)
                                   / tth_resolution))
        elif tth_num is not None:
            tth_resolution = (tth_max - tth_min) / tth_num
        elif tth_num is None and tth_resolution is None:
            err_str = 'Must define either tth_num or tth_resolution.'
            raise ValueError(err_str)
        
        # Get chi numbers
        chi_min = np.min(self.chi_arr)
        chi_max = np.max(self.chi_arr)
        if chi_num is None:
            chi_num = int(np.round((chi_max - chi_min)
                                   / chi_resolution))
        elif chi_num is not None:
            chi_resolution = (chi_max - chi_min) / chi_num
        elif chi_num is None and chi_resolution is None:
            err_str = 'Must define either chi_num or chi_resolution.'
            raise ValueError(err_str)
       
        return self.ai.integrate2d_ng(image, tth_num, chi_num,
                                      unit=unit,
                                      correctSolidAngle=False,
                                      polarization_factor=None,
                                      **kwargs)

    # Backwards compatibility
    integrate1d_image = integrate1D_image
    integrate2d_image = integrate2D_image


    # Convenience function for image to polar coordinate transformation (estimate!)
    def estimate_polar_coords(self,
                              coords,
                              method='linear'):
        """

        """
        return estimate_polar_coords(coords,
                                     self.tth_arr,
                                     self.chi_arr,
                                     method=method)
    

    # Convenience function for polar to image coordinate transformation (estimate!)
    def estimate_image_coords(self,
                              coords,
                              method='nearest'):
        """

        """
        return estimate_image_coords(coords,
                                     self.tth_arr,
                                     self.chi_arr,
                                     method=method)

    
    @XRDData._protect_hdf()
    def save_reciprocal_positions(self):
        """

        """
                 
        if self.tth is None:
            tth = []
        else:
            tth = self.tth
        if self.chi is None:
            chi = []
        else:
            chi = self.chi

        print('Writing reciprocal positions to disk...', end='', flush=True)
        # This group may already exist if poni file was already initialized
        curr_grp = self.hdf[self._hdf_type].require_group('reciprocal_positions')
        if hasattr(self, 'extent'):
            overwrite_attr(curr_grp.attrs, 'extent', self.extent)

        labels = ['tth_pos', 'chi_pos']
        comments = ["'tth', is the two theta scattering angle",
                    "'chi' is the azimuthal angle"]
        keys = ['tth', 'chi']
        data = [tth, chi]
        resolution = [self.tth_resolution, self.chi_resolution]

        for i, key in enumerate(keys):
            # Skip values that are None
            if data[i] is None:
                data[i] = np.array([])
            else:
                data[i] = np.asarray(data[i])

            if key in curr_grp.keys():
                del curr_grp[key]
            dset = curr_grp.require_dataset(key,
                                            data=data[i],
                                            dtype=data[i].dtype,
                                            shape=data[i].shape)
            
            overwrite_attr(dset.attrs, 'labels', labels[i])
            overwrite_attr(dset.attrs, 'comments', comments[i])
            overwrite_attr(dset.attrs, 'dtype', str(data[i].dtype))
            overwrite_attr(dset.attrs,
                           f'{key}_resolution',
                           resolution[i])
            dset.attrs['time_stamp'] = ttime.ctime() # always new

        print('done!')
    
    ##################################
    ### Scaler and Position Arrays ###
    ##################################

    def set_scalers(self,
                    sclr_dict,
                    scaler_units='counts',
                    check_init_sets=False):
        """

        """

        # Store sclr_dict as attribute
        for key, value in list(sclr_dict.items()):
            if value.ndim  != 2: # Only intended for rocking curves
                sclr_dict[key] = value.reshape(self.map_shape)

        self.sclr_dict = sclr_dict
        self.scaler_units = scaler_units

        # Write to hdf file
        self.save_sclr_pos('scalers',
                            self.sclr_dict,
                            self.scaler_units,
                            check_init_sets=check_init_sets)
    

    def _get_scaler_absorption(self,
                               scaler_key='i0',
                               chamber_length=None,
                               gas_name=None):
        """

        """
        
        if (not hasattr(self, 'energy')
            or self.energy is None):
            err_str = f'Must define energy in {self._hdf_type}.'
            raise AttributeError(err_str)
        if (not hasattr(self, 'sclr_dict')
            or self.energy is None
            or self.sclr_dict == {}):
            err_str = f'Must define scalers in {self._hdf_type}.'
            raise AttributeError(err_str)
        elif scaler_key not in self.sclr_dict.keys():
            err_str = (f'Scaler {scaler_key} not in scaler dictionary.'
                       + f'\nChose from {self.sclr_dict.keys()}.')

        # Add default values for SRX
        if chamber_length is None:
            if scaler_key == 'i0':
                chamber_length = 1
                note_str = ('NOTE: Using default chamber length of 1 '
                            + f'cm for {scaler_key} scaler.')
                print(note_str)
            elif scaler_key == 'im':
                chamber_length = 10
                note_str = ('NOTE: Using default chamber length of 10 '
                            + f'cm for {scaler_key} scaler.')
                print(note_str)
            else:
                err_str = ('Must provide chamber length for '
                           + f'{scaler_key} scaler.')
                raise ValueError(err_str)
        if gas_name is None:
            if scaler_key == 'i0':
                gas_name = 'N2'
                note_str = ('NOTE: Using default chamber gas of N2 for'
                            + f' {scaler_key} scaler.')
                print(note_str)
            elif scaler_key == 'im':
                gas_name = 'air'
                note_str = ('NOTE: Using default chamber gas of air '
                            + f'for {scaler_key} scaler.')
                print(note_str)
            else:
                err_str = (f'Must provide gas for {scaler_key} scaler '
                           + 'key.')
                raise ValueError(err_str)
        elif gas_name not in reference_data.gases[gas_name]:
            err_str = (f'Unknown gas of {gas}. Only gases in '
                       + f'{reference_data.gases.keys()} are '
                       + 'supported.')
            raise ValueError(err_str)

        gas = reference_data.gases[gas_name]
        absorption = gas.get_absorption(self.energy, chamber_length)

        return absorption, scaler_key, chamber_length, gas_name
    

    # Scalers have an energy dependence
    # Remove this proportionality when comparing different energies
    def correct_scaler_energies(self,
                                scaler_key='i0',
                                chamber_length=None,
                                gas_name=None,
                                check_init_sets=False):
        """

        """

        # Check for energy
        if self.energy is None:
            err_str = ('Internal energy attribute is None. Cannot '
                       + 'correct scalers for X-ray energy without '
                       + 'energy.')
            raise AttributeError(err_str)

        # Get absorption and parse inputs
        absorption, _, _, _ = self._get_scaler_absorption(
                                    scaler_key=scaler_key,
                                    chamber_length=chamber_length,
                                    gas_name=gas_name)

        # Determine energy independent scaler values
        new_scaler_key = f'energy_corrected_{scaler_key}'

        # Considerations for singular and list of energies
        energy = np.asarray([self.energy]).squeeze()
        if energy.ndim > 0:
            energy = energy.reshape(self.map_shape)
            absorption = absorption.reshape(self.map_shape)
        
        new_sclr_arr = (self.sclr_dict[scaler_key]
                        / (absorption * energy))

        # Set values and write to hdf
        self.sclr_dict[new_scaler_key] = new_sclr_arr
        self.save_sclr_pos('scalers',
                           self.sclr_dict,
                           self.scaler_units,
                           check_init_sets=check_init_sets)


    # Post-conversion of scaler to real flux values
    def convert_scalers_to_flux(self,
                                preamp_sensitivity=None,
                                scaler_key='i0',
                                chamber_length=None,
                                gas_name=None,                                
                                f_range=10e6,
                                V_range=5,
                                check_init_sets=False):
        """

        """

        # Check for energy
        if self.energy is None:
            err_str = ('Internal energy attribute is None. Cannot '
                       + 'correct scalers for X-ray energy without '
                       + 'energy.')
            raise AttributeError(err_str)

        # Check for preamp sensitivity
        if preamp_sensitivity is None:
            if f'{scaler_key}_sensitivity' in self.extra_metadata:
                preamp_sensitivity = self.extra_metadata[f'{scaler_key}_sensitivity']
            else:
                err_str = ('Must provide preamp sensitivity for '
                           + f'{scaler_key} scaler, or this data must '
                           + 'be stored in extra_metadata.')
                raise AttributeError(err_str)
        
        # Get absorption and parse inputs
        (absorption,
         scaler_key,
         chamber_length,
         gas_name)= self._get_scaler_absorption(
                                    scaler_key=scaler_key,
                                    chamber_length=chamber_length,
                                    gas_name=gas_name)

        # Considerations for singular and list of energies
        energy = np.asarray([self.energy]).squeeze()
        if energy.ndim > 0:
            energy = energy.reshape(self.map_shape)
            absorption = absorption.reshape(self.map_shape)

            # Untested reshaping of preamp_sensitivity for if it changes over an extended energy rocking curve
            if np.asarray([preamp_sensitivity]).squeeze().ndim > 0:
                preamp_sensitivity = np.asarray(preamp_sensitivity).reshape(self.map_shape)
        
        (ionization_energy
         ) = reference_data.average_gas_ionization_energies[gas_name]

        charge_term = (ionization_energy
                       / (scipy.constants.e * energy * 1e3))
        
        v_to_f_term = ((V_range * preamp_sensitivity) / (f_range * self.dwell))
        
        # Determine energy independent scaler values
        new_scaler_key = f'flux_{scaler_key}'
        new_sclr_arr = (self.sclr_dict[scaler_key]
                        * charge_term
                        * v_to_f_term
                        / absorption)

        # Set values and write to hdf
        self.sclr_dict[new_scaler_key] = new_sclr_arr
        self.save_sclr_pos('scalers',
                           self.sclr_dict,
                           self.scaler_units,
                           check_init_sets=check_init_sets)


    @XRDData._protect_hdf()
    def save_sclr_pos(self,
                      group_name,
                      map_dict,
                      unit_name,
                      check_init_sets=False):
        """

        """

        if check_init_sets:
            if group_name in self.hdf[self._hdf_type]:
                # If all the keys are present, the do not save
                if all([key in self.hdf[self._hdf_type][group_name].keys()
                        for key in map_dict.keys()]):
                    return

        # Write data to hdf
        curr_grp = self.hdf[self._hdf_type].require_group(group_name)
        curr_grp.attrs['time_stamp'] = ttime.ctime()

        for key, value in map_dict.items():
            value = np.asarray(value)
            
            if key in curr_grp.keys():
                dset = curr_grp[key]
                if (value.shape == dset.shape
                    and value.dtype == dset.dtype):
                    dset[...] = value
                else:
                    del curr_grp[key] # deletes flag, but not the data...
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
            
            # Update attrs
            overwrite_attr(dset.attrs, 'labels', ['map_x', 'map_y'])
            overwrite_attr(dset.attrs, 'units', unit_name)
            overwrite_attr(dset.attrs, 'dtype', str(value.dtype))
        
    
    #########################################
    ### Manipulating and Selecting Phases ###
    #########################################

    # Updating potential phase list
    def add_phase(self, phase):
        """

        """

        if hasattr(phase, 'name'):
            phase_name = phase.name
        elif isinstance(phase, str):
            phase_name = phase
        else:
            raise TypeError(f'Unsure how to handle {phase} type.')

        if phase_name not in self.phases.keys():
            self.phases[phase_name] = phase
        else:
            ostr = (f'Did not add {phase_name} since it is '
                    + 'already a possible phase.')
            print()


    def remove_phase(self, phase):
        """

        """
        # Allow for phase object or name to work
        if hasattr(phase, 'name'):
            phase_name = phase.name
        elif isinstance(phase, str):
            phase_name = phase
        else:
            raise TypeError(f'Unsure how to handle {phase} type.')

        if phase_name in self.phases.keys():
            del self.phases[phase_name]
        else:
            ostr = (f'Cannot remove {phase_name} since it is '
                    + 'not in possible phases.')
            print(ostr)
        

    def load_phase(self, filename, wd=None, phase_name=None):
        """

        """

        if wd is None:
            wd = self.wd
        
        phase_path = pathify(wd, filename,
                             ['.cif', '.txt', '.D', '.h5'])

        if not os.path.exists(f'{phase_path}'):
            err_str = f'Specified path does not exist:\n{phase_path}'
            raise FileNotFoundError(err_str)
        
        if phase_path[-4:] == '.cif':
            phase = Phase.fromCIF(f'{phase_path}')
        elif phase_path[-4:] == '.txt':
            raise NotImplementedError()
        elif phase_path[-2:] in ['.D']:
            raise NotImplementedError()
        elif phase_path[-2:] == '.h5':
            raise NotImplementedError()
        else:
            err_str = (f'Unsure how to read {phase_path}. Either '
                       + 'specifiy file type or this file type '
                       + 'is not supported.')
            raise TypeError(err_str)
        
        if phase_name is not None:
            phase.name = phase_name
        
        self.add_phase(phase)
    

    def clear_phases(self):
        """

        """

        self.phases = {}


    @XRDData._protect_hdf()
    def save_phases(self, verbose=True):
        """

        """
        
        if len(self.phases) > 0:
            phase_grp = self.hdf[self._hdf_type].require_group(
                                                    'phase_list')

            # Delete any no longer included phases
            for phase in phase_grp.keys():
                if phase not in self.phases.keys():
                    del(phase_grp[phase])

            # Save any new phases
            for phase in self.phases.values():
                phase.save_to_hdf(phase_grp)

            if verbose:
                print('Phases saved in hdf.')


    def select_phases(self,
                      xrd=None,
                      energy=None,
                      tth_resolution=None,
                      tth_num=None,
                      unit='2th_deg',
                      ignore_less=0.5,
                      title=None,
                      title_scan_id=True,
                      save_to_hdf=False):
        """

        """
        
        if not hasattr(self, 'ai') or self.ai is None:
            err_str = ('Must first set calibration '
                       + 'before selecting phases.')
            raise AttributeError(err_str)

        if tth_resolution is None:
            tth_resolution = self.tth_resolution

        tth_min = np.min(self.tth_arr)
        tth_max = np.max(self.tth_arr)
        if tth_num is None:
            tth_num = int(np.round((tth_max - tth_min)
                                   / tth_resolution))
        elif tth_num is None and tth_resolution is None:
            err_str = 'Must define either tth_num or tth_resolution.'
            raise ValueError(err_str)
        
        if xrd is None:
            if self.corrections['polar_calibration']:
                xrd = self._processed_images_composite
            else:
                xrd = self.composite_image
        
        if energy is None:
            if isinstance(self.energy, list):
                err_str = ('A specific incident X-ray energy must be '
                        + 'provided to use the phase selector tool.')
                raise RuntimeError(err_str)
            else:
                energy = self.energy
        
        if xrd.ndim == 2:
            tth, xrd = self.integrate1D_image(image=xrd,
                                              tth_num=tth_num,
                                              unit=unit)
        elif xrd.ndim == 1:
            if not hasattr(self, 'tth') or self.tth is None:
                tth, _ = self.integrate1D_image(
                                image=np.zeros(xdm.image_shape),
                                tth_num=len(xrd.ndim),
                                unit=unit)
            else:
                tth = self.tth
        else:
            err_str = (f'Given XRD has {xrd.ndim} dimensions, but '
                       + 'phase selector tool can only handle 2D or '
                       + '1D XRD patterns.')
            raise ValueError(err_str)

        title = self._title_with_scan_id(
                            title,
                            default_title='Phase Selector',
                            title_scan_id=title_scan_id)

        # Plot phase_selector
        phase_vals = phase_selector(xrd,
                                    list(self.phases.values()),
                                    energy,
                                    tth,
                                    ignore_less=ignore_less,
                                    title=title,
                                    update_reflections=save_to_hdf)

        # Update reflections and write to hdf
        if save_to_hdf:
            old_phases = list(self.phases.keys())
            for phase in old_phases:
                if phase_vals[phase] <= 0:
                    self.remove_phase(phase)
            self.save_phases()


    ###########################
    ### Vectorized Data I/O ###
    ###########################

    # Called in other functions. Not protected.
    def _save_vector_map(self,
                         hdf,
                         vector_map,
                         edges=None,
                         rewrite_data=False,
                         verbose=False):
        """

        """

        # Write data to hdf
        print('Saving vectorized map data...')
        vector_grp = hdf[self._hdf_type].require_group(
                                                'vectorized_map')
        map_grp = vector_grp.require_group('vector_map')
        vector_grp.attrs['time_stamp'] = ttime.ctime()
        map_grp.attrs['vectorized_map_shape'] = vector_map.shape

        all_used_indices = [] # For potential vmask shape changes
        for index in range(np.prod(vector_map.shape)):
            indices = np.unravel_index(index, vector_map.shape)
            # e.g., '1,2'
            title = ','.join([str(ind) for ind in indices]) 
            all_used_indices.append(title)

            XRDBaseScan._save_vectors(map_grp,
                                      vector_map[indices],
                                      title=title,
                                      rewrite_data=rewrite_data,
                                      verbose=verbose)
    
        # In case virtual shape changed; remove extra datasets
        for dset_key in map_grp.keys():
            if dset_key not in all_used_indices:
                del map_grp[dset_key]
        # Backwards compatibility; scrub old data into new format
        for dset_key in vector_grp.keys():
            if dset_key not in all_used_indices + ['vector_map']:
                del vector_grp[dset_key]
        
        # Save edges if available. Only useful for XRDMapStack
        XRDBaseScan._save_edges(vector_grp,
                                edges=edges,
                                rewrite_data=rewrite_data,
                                verbose=verbose)
        print('done!')
    

    # Called in other functions. Not protected.
    def _save_rocking_vectorization(self,
                                    hdf,
                                    vectors,
                                    edges=None,
                                    rewrite_data=False,
                                    verbose=False):
        """

        """
        
        print('Saving vectorized image data...')
        # Write data to hdf
        vector_grp = hdf[self._hdf_type].require_group(
                                                'vectorized_data')
        vector_grp.attrs['time_stamp'] = ttime.ctime()

        # Save vectors
        XRDBaseScan._save_vectors(vector_grp,
                                  vectors,
                                  rewrite_data=rewrite_data,
                                  verbose=verbose)

        # Save edges which should be available
        XRDBaseScan._save_edges(vector_grp,
                                edges=edges,
                                rewrite_data=rewrite_data,
                                verbose=verbose)
        
        # Scrub old tags for backwards compatibility
        for dset_key in vector_grp.keys():
            if dset_key not in ['vectors', 'edges']:
                del vector_grp[dset_key]

        print('done!')
    

    # Called in other functions. Not protected.
    @staticmethod # For XRDMapStack
    def _save_vectors(vector_grp,
                      vectors,
                      title=None,
                      rewrite_data=False,
                      verbose=False): # required vectors
        """

        """
        
        if title is None:
            title = 'vectors' # Generic
        
        if title not in vector_grp:
            dset = vector_grp.require_dataset(
                        title,
                        data=vectors,
                        shape=vectors.shape,
                        dtype=vectors.dtype)
        else:
            dset = vector_grp[title]

            if (dset.shape == vectors.shape
                and dset.dtype == vectors.dtype):
                dset[...] = vectors
            else:
                warn_str = 'WARNING: '
                if dset.shape != vectors.shape:
                    warn_str += (f'Vectors shape of {vectors.shape} '
                                + 'does not match dataset shape '
                                + f'{dset.shape}. ')
                if dset.dtype != vectors.dtype:
                    warn_str += (f'Vectors dtype of {vectors.dtype} '
                                + 'does not match dataset dtype '
                                + f'{dset.dtype}. ')
                if rewrite_data:
                        warn_str += (f'\nOvewriting {title}. This '
                                    + 'may bloat the total file size.')
                        if verbose:
                            print(warn_str)
                        del vector_grp[title]
                        dset = vector_grp.require_dataset(
                            title,
                            data=vectors,
                            shape=vectors.shape,
                            dtype=vectors.dtype)
                else:
                    warn_str += '\nProceeding without changes.'
                    if verbose:
                        print(warn_str)
    

    # Called in other functions. Not protected.
    @staticmethod # For XRDMapStack
    def _save_edges(vector_grp,
                    edges=None,
                    rewrite_data=False,
                    verbose=False):
        """

        """

        # Only save edge information if given. Quiet if not.
        if edges is not None:
            edge_grp = vector_grp.require_group('edges')
            edge_grp.attrs['time_stamp'] = ttime.ctime()

            # Check for existenc and compatibility
            for i, edge in enumerate(edges):
                edge = np.asarray(edge)
                edge_title = f'edge_{i}'
                if edge_title not in edge_grp.keys():
                    edge_grp.require_dataset(
                        edge_title,
                        data=edge,
                        shape=edge.shape,
                        dtype=edge.dtype)
                else:
                    dset = edge_grp[edge_title]

                    if (dset.shape == edge.shape
                        and dset.dtype == edge.dtype):
                        dset[...] = edge
                    else:
                        warn_str = 'WARNING: '
                        if dset.shape != edge.shape:
                            warn_str += (f'Edge shape for {edge_title}'
                                        + f' {edge.shape} does not '
                                        + 'match dataset shape '
                                        + f'{dset.shape}. ')
                        if dset.dtype != edge.dtype:
                            warn_str += (f'Edge dtype for {edge_title}'
                                        + f' {edge.dtype} does not '
                                        + 'match dataset dtype '
                                        + f'{dset.dtype}. ')
                        if rewrite_data:
                            warn_str += ('\nOvewriting data. This may '
                                        + 'bloat the total file size.')
                            # Shape changes should not happen
                            # except from q_arr changes
                            if verbose:
                                print(warn_str)
                            del edge_grp[edge_title]
                            edge_grp.require_dataset(
                                    edge_title,
                                    data=edge,
                                    shape=edge.shape,
                                    dtype=edge.dtype)
                        else:
                            warn_str += '\nProceeding without changes.'
                            if verbose:
                                print(warn_str)
    
    # Called in other functions. Not protected.
    @staticmethod
    def _parse_vector_dict(vector_dict):
        vector_attrs = {}

        # Parse vector_dict
        if vector_dict is not None:
            for key, value in vector_dict.items():
                # Check for intensity masks
                if key == 'spot_int_cutoff':
                    # Backwards compatibility
                    relative_cutoff = True
                    if 'relative_cutoff' in vector_dict:
                        relative_cutoff = bool(vector_dict['relative_cutoff'])
                    intensity = vector_dict['vectors'][:, -1]
                    (vector_attrs['spot_int_mask']
                    ) = generate_intensity_mask(
                        intensity,
                        int_cutoff=vector_dict['spot_int_cutoff'],
                        relative_cutoff=relative_cutoff)
                elif key == 'blob_int_cutoff':
                    # Backwards compatibility
                    relative_cutoff = True
                    if 'relative_cutoff' in value.attrs:
                        relative_cutoff = bool(value.attrs['relative_cutoff'])
                    intensity = vector_dict['vectors'][:, -1]
                    (vector_attrs['blob_int_mask']
                    ) = generate_intensity_mask(
                        intensity,
                        int_cutoff=vector_dict['blob_int_cutoff'],
                        relative_cutoff=relative_cutoff)
                else:
                    vector_attrs[key] = value
        
        return vector_attrs
    

    @XRDData._protect_hdf()
    def load_vectors(self,
                     hdf):
        # Load data from hdf
        vector_dict = _load_xrd_hdf_vector_data(hdf[self._hdf_type])

        # Universal parsing from XRDBaseScan class
        vector_attrs = self._parse_vector_dict(vector_dict)

        # Attach to current instance
        for key, value in vector_attrs.items():
            setattr(self, key, value)

    #################################
    ### Generalized Spot Analysis ###
    #################################
    
    # Wrapped for child classes
    @staticmethod
    def _trim_spots(spots,
                    remove_less=0.01,
                    key='height'):
        
        if key not in spots.keys():
            err_str = (f'{key} not in spots. Choose from\n'
                       + f'{spots.keys()}')
            raise KeyError(err_str)
        
         # Find relative indices where conditional is true
        mask = np.nonzero(spots[key].values < remove_less)[0]
        drop_indices = spots.iloc[mask].index.values
        
        # Drop indices
        full_num = len(spots)
        drop_num = len(drop_indices)
        spots.drop(index=drop_indices, inplace=True)
        spots.index = range(len(spots))
        ostr = (f'Trimmed {drop_num} spots '
                + f'({drop_num / full_num * 100:.2f} %) below {key} of'
                + f' {remove_less}.')
        print(ostr)    
    
         
    ##########################
    ### Plotting Functions ###
    ##########################

    def _title_with_scan_id(self,
                            title,
                            default_title='',
                            title_scan_id=True):
        """

        """
        
        if title is None:
            title = default_title
        if title_scan_id:
            if title == '':
                return f'scan{self.scan_id}'
            else:
                return f'scan{self.scan_id}: {title}'
        else:
            return title


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
                   title_scan_id=True,
                   **kwargs):
        """

        """
        
        image, indices = _xrdbasescan_image(self,
                                       image=image,
                                       indices=indices)
        
        out = _plot_parse_xrdbasescan(self,
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
        
        # Reset title
        title = self._title_with_scan_id(
                            ax.title._text,
                            title_scan_id=title_scan_id)
        ax.set_title(title)
        
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
                         title_scan_id=True,
                         return_plot=False,
                         **kwargs):
        """

        """
        
        if tth is None:
            tth = self.tth
        
        if units is None:
            units = self.scattering_units
        
        integration, indices = _xrdbasescan_integration(self,
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

        # Reset title
        title = self._title_with_scan_id(
                            ax.title._text,
                            title_scan_id=title_scan_id)
        ax.set_title(title)
        
        if return_plot:
            return fig, ax
        else:
            fig.show()
    



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
                     title_scan_id=True,
                     return_plot=False):
        """

        """
 
        fig, ax = plot_q_space(self,
                               indices=indices,
                               skip=skip,
                               detector=detector,
                               Ewald_sphere=Ewald_sphere,
                               beam_path=beam_path,
                               fig=fig,
                               ax=ax)

        title = self._title_with_scan_id(
                            'Q-Space',
                            title_scan_id=title_scan_id)
        ax.set_title(title)
        
        if return_plot:
            return fig, ax
        else:
            fig.show()


    def plot_detector_geometry(self,
                               skip=300,
                               fig=None,
                               ax=None,
                               title_scan_id=True,
                               return_plot=False):
        """

        """
        
        fig, ax = plot_detector_geometry(self,
                                         skip=skip,
                                         fig=fig,
                                         ax=ax)

        title = self._title_with_scan_id(
                            'Detector Geometry',
                            title_scan_id=title_scan_id)
        ax.set_title(title)
        
        if return_plot:
            return fig, ax
        else:
            fig.show()
