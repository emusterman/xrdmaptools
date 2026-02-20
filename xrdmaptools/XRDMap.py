import numpy as np
import os
import h5py
import pyFAI
import pandas as pd
import time as ttime
import matplotlib.pyplot as plt
import dask.array as da
import skimage.io as io
from dask_image import imread as dask_io
from tqdm import tqdm
import functools

# Local imports
from xrdmaptools.XRDBaseScan import XRDBaseScan
from xrdmaptools.utilities.utilities import (
    pathify,
    _check_dict_key,
    copy_docstring
    )
# from xrdmaptools.io.hdf_io import _load_xrd_hdf_vectorized_map_data
from xrdmaptools.io.hdf_utils import (
    check_attr_overwrite,
    overwrite_attr,
    get_large_map_slices
    )
from xrdmaptools.io.db_io import (
    load_data,
    get_scantype
    )
from xrdmaptools.reflections.spot_blob_indexing import _initial_spot_analysis
from xrdmaptools.reflections.SpotModels import GaussianFunctions
from xrdmaptools.reflections.spot_blob_search import (
    find_blobs_spots,
    find_spot_stats,
    make_stat_df,
    remake_spot_list,
    fit_spots
    )
from xrdmaptools.plot.interactive import (
    interactive_2D_plot,
    interactive_1D_plot,
    interactive_1D_window_sum_plot,
    interactive_1D_window_com_plot,
    interactive_2D_window_sum_plot,
    interactive_2D_window_tth_com_plot,
    interactive_2D_window_chi_com_plot
    )
from xrdmaptools.plot.general import (
    return_plot_wrapper,
    plot_map,
    )


class XRDMap(XRDBaseScan):
    """
    Class for analyzing and processing XRD data acquired every
    pixel along a 2D spatial map.

    Parameters
    ----------
    pos_dict : dict, optional
        Dictionary of 2D numpy arrays matching the map shape with
        position values used to inform and scale map plots.
    swapped_axes: bool, optional
        Flag to indicate if the fast and slow scanning axes are
        swapped. Default is False assuming a fast x-axis and slow y-axis.
    xrf_path : str, optional
        Path string of associated xrfmap HDF file generate from pyXRF.
    check_init_sets : bool, optional
        Flag whether to overwrite data in the HDF file if available. By
        default this is set to False and should only be True when
        instantiating from the HDF file.
    xrdbasekwargs : dict, optional
        Dictionary of all other kwargs for parent XRDBaseScan class.
    """

    # Class variables
    _hdf_type = 'xrdmap'

    def __init__(self,
                 pos_dict=None,
                 swapped_axes=False,
                 xrf_path=None,
                 check_init_sets=False,
                 **xrdbasekwargs
                 ):

        # Force data into native shape
        # This prevents inconsistencies later
        # But will cause pos_dict and sclr_dict to be re-written
        self._swapped_axes = False
        
        XRDBaseScan.__init__(
            self,
            map_labels=['map_y_ind',
                        'map_x_ind'],
            check_init_sets=check_init_sets,
            **xrdbasekwargs,
            )

        # Set position dictionary
        save_init_sets = False
        if 'check_init_sets' in xrdbasekwargs:
            check_init_sets = xrdbasekwargs['check_init_sets']
        self.pos_dict = None
        if pos_dict is not None:
            self.set_positions(pos_dict,
                               check_init_sets=check_init_sets)

        # Interpolate positions
        # Can happen before or after swapped_axes
        # Only if interpolated positions are not already determined
        if (self.scan_input is not None
            and len(self.scan_input) != 0
            and ('interp_x' not in self.pos_dict 
                 or 'interp_y' not in self.pos_dict)):
            self.interpolate_positions(check_init_sets=check_init_sets)
        
        # Swap axes if called. Tranposes major data components
        # This flag is used to avoid changing the original saved data
        self._swapped_axes = bool(swapped_axes)
        if self._swapped_axes:
            # self.swap_axes(only_images=True)
            self.swap_axes(update_flag=False)
        
        # Save xrf_path location. Do not load unless explicitly called
        self.xrf_path = xrf_path


    ################################
    ### Loading data into XRDMap ###
    ################################
        

    @classmethod
    def from_db(cls,
                scan_id=-1,
                broker='manual',
                wd=None,
                filename=None,
                save_hdf=True,
                data_keys=None,
                xrd_dets=None,
                repair_method='fill',
                **kwargs):

        """
        Instantiate from data in database.

        Load data and metadata from database to create a new instance.
        This will copy, compile, and write the raw data into a single
        HDF by default.

        Parameters
        ----------
        scan_id : int, optional
            Scan ID of scan to load data from database. -1 by default
            which loads from the most recent scan.
        broker : {"manual", "tiled", "databroker"}, optional
            Broker/method used to load the data. "manual" will load
            data directly from raw HDF files, while "tiled" and
            "databroker" use their respective libraries. "manual" by
            default.
        wd : path str, optional
            Working directly used to write HDF file if saved. Uses
            current working directory if none provided.
        filename : str, optional
            Filename used used to write HDF file if saved. Uses
            "scan<scan_id>_xrdmap.h5" by default.
        save_hdf : bool, optional
            Flag to enable writing data to HDF file. True by default.
        data_keys : list, optional
            List of data keys to be loaded into the instance. Not all
            data keys are supported or will be used. ['enc1', 'enc2',
            'i0', 'i0_time', 'im', 'it'] be default along with any
            either the merlin or dexela area detectors if used.
        repair_method : {"fill", "flatten", "replace"}, optional
            Repair method used for missing data if data broker is
            "manual". "fill" auto-pads missing pixels with zero and is
            the default option. This also matches the behavior when
            broker is "tiled". "flatten" will flatten the entire
            dataset into a single line and is not recommended.
            "replace" will replace rows with missing data by the
            previously full row, matching the behavior of pyXRF.
        kwargs : dict, optional
            Other keyward arguments passed to __init__.
        
        Returns
        -------
        xrdmap : XRDMap
            Instance of XRDMap class with all data and metadata from
            the designaed scan ID.

        Raises
        ------
        OSError if working directory does not exist.
        RuntimeError if scan type is incorrect for XRDMap.
        """
        
        if wd is None:
            wd = os.getcwd()
        else:
            if not os.path.exists(wd):
                err_str = f'Cannot find directory {wd}'
                raise OSError(err_str)

        # Get scantype information from check...
        if broker == 'manual':
            temp_broker = 'tiled'
        else:
            temp_broker = broker
        scantype = get_scantype(scan_id,
                                broker=temp_broker)
        if scantype not in ['XRF_FLY', 'XRF_STEP']:
            err_str = (f"Scan of type {scantype} not currently "
                       + "supported. Scan type must be 'XRF_FLY' or "
                       + "'XRF_STEP'.")
            raise RuntimeError(err_str)
    
        # No fluorescence key
        pos_keys = ['enc1', 'enc2']
        sclr_keys = ['i0', 'i0_time', 'im', 'it']
        
        if data_keys is None:
            data_keys = pos_keys + sclr_keys

        (data_dict,
         scan_md,
         data_keys,
         xrd_dets) = load_data(
                            scan_id=scan_id,
                            broker=broker,
                            detectors=None,
                            data_keys=data_keys,
                            xrd_dets=xrd_dets,
                            returns=['data_keys',
                                     'xrd_dets'],
                            repair_method=repair_method)

        # Extract main image_data
        xrd_data = [data_dict[f'{xrd_det}_image']
                    for xrd_det in xrd_dets]

        # Make position dictionary
        pos_dict = {key:value for key, value in data_dict.items()
                    if key in pos_keys}

        # Make scaler dictionary
        sclr_dict = {key:value for key, value in data_dict.items()
                     if key in sclr_keys}

        # Determine null_map
        null_maps = []
        for det in xrd_dets:
            null_key = f'{det}_null_map'
            if null_key in data_dict:
                null_maps.append(np.asarray(data_dict[null_key]))
            else:
                null_maps.append(None)

        if len(xrd_data) > 1:
            filenames = [f'scan{scan_md["scan_id"]}_{det}_xrdmap.h5'
                         for det in xrd_dets]
        else:
            filenames = [filename]

        extra_md = {}
        for key in scan_md.keys():
            if key not in ['scan_id',
                           'beamline_id',
                           'energy',
                           'dwell',
                           'theta',
                           'start_time',
                           'scan_input']:
                extra_md[key] = scan_md[key]
        
        xrdmaps = []
        for i, (image_data, det) in enumerate(zip(xrd_data, xrd_dets)):
            xdm = cls(scan_id=scan_md['scan_id'],
                      wd=wd,
                      filename=filenames[i],
                      save_hdf=save_hdf,
                      image_data=image_data,
                      null_map=null_maps[i],
                      energy=scan_md['energy'],
                      dwell=scan_md['dwell'],
                      theta=scan_md['theta'],
                      sclr_dict=sclr_dict,
                      pos_dict=pos_dict,
                      beamline=scan_md['beamline_id'],
                      facility='NSLS-II',
                      detector=det,
                      scan_input=scan_md['scan_input'],
                      time_stamp=scan_md['time_str'],
                      extra_metadata=extra_md,
                      **kwargs
                      )
            
            # Check for dark-field. Save but do not apply correction.
            if f'{det}_dark' in data_dict:
                note_str = 'Automatic dark-field found'
                if save_hdf:
                    note_str += ' and written to disk'
                note_str += f' for {det}.'
                xdm.dark_field = data_dict[f'{det}_dark'] # Could be passed as extra_attrs
                xdm.save_images(images='dark_field', units='counts')
                print(note_str)
            
            xrdmaps.append(xdm)

        print(f'{cls.__name__} loaded!')
        if len(xrdmaps) > 1:
            return tuple(xrdmaps)
        else:
            # Don't bother returning a tuple or list of xrdmaps
            return xrdmaps[0]
    

    # Only accessible from a dask_enabled instance 
    # in order to spin up smaller XRDMap instances
    def fracture_large_map(self,
                           approx_new_map_sizes=10, # in GB
                           final_dtype=np.float32,
                           new_directory=True):
        """
        Fracture single large map into smaller maps.

        Fracture a large map by slicing along spatial dimensions to
        create a mosaic of smaller, more manageable datasets. Maps
        must be unprocessed and lazily loaded (dask enabled) to perform
        this function.

        Parameters
        ----------
        approx_new_map_size : float, optional
            Approximate new map size in GB targeted by the slicing
            routine. Real values can be larger, so smaller values
            can be safer. 10 GB by default.
        final_dtype : float, optional
            Final datatype of anticipated maps. Numpy.float32 by
            default. 
        new_directory : bool, optional
            Flag to write new subdirectory. True by default.
        
        Raises
        ------
        ValueError if maps have any level of processing or are not
        lazily loaded.
        RuntimeError if the approximate map size requested is about
        the same size as the current map.
        """

        if not self._dask_enabled:
            err_str = ('Images must be lazily loaded with Dask. '
                       + 'Please enable this.')
            raise ValueError(err_str)
        elif np.any(list(self.corrections.values())):
            err_str = ('XRDMap images have some corrections already '
                       + ' applied.\nFracturing datasets is only '
                       + 'supported for raw datasets.')
            raise ValueError(err_str)
        
        # Get slicing information
        slicings, _, _ = get_large_map_slices(
                        self.images,
                        approx_new_map_sizes=approx_new_map_sizes,
                        final_dtype=final_dtype
                        )
        
        if len(slicings) <= 1:
            err_str = ('Estimated fractured map size is equivalent '
                       + f'to full map size {self.images.shape}.'
                       + '\nEither designate a smaller new map size '
                       + 'or proceed with full map.')
            raise RuntimeError(err_str)
        
        if new_directory:
            new_dir = f'{self.wd}scan{self.scan_id}_fractured_maps/'
            os.makedirs(new_dir, exist_ok=True)
        else:
            new_dir = self.wd

        if not hasattr(self, '_energy'):
            energy = None
        else:
            energy = self.energy
        if not hasattr(self, 'poni'):
            poni = None
        else:
            poni = self.poni

        # Slicing of numpy arrays create veiws, not new copys
        sliced_images = []
        sliced_pos_dicts = []
        sliced_sclr_dicts = []
        sliced_scan_inputs = []
        full_x_vals = np.linspace(*self.scan_input[:2],
                                  int(self.scan_input[2]))
        full_y_vals = np.linspace(*self.scan_input[3:5],
                                  int(self.scan_input[5]))

        for slicing in slicings:
            i_st, i_end = slicing[0]
            j_st, j_end = slicing[1]
            
            # images
            sliced_images.append(self.images[i_st:i_end,
                                             j_st:j_end])

            # pos_dict
            new_pos_dict = {}
            for key in self.pos_dict.keys():
                new_pos_dict[key] = self.pos_dict[key][i_st:i_end,
                                                       j_st:j_end]
            sliced_pos_dicts.append(new_pos_dict)

            # sclr_dict
            new_sclr_dict = {}
            for key in self.sclr_dict.keys():
                new_sclr_dict[key] = self.sclr_dict[key][i_st:i_end,
                                                         j_st:j_end]
            sliced_sclr_dicts.append(new_sclr_dict)

            # scan_input
            sliced_scan_inputs.append([full_x_vals[j_st],
                                       full_x_vals[j_end - 1],
                                       j_end - j_st,
                                       full_y_vals[i_st],
                                       full_y_vals[i_end - 1],
                                       i_end - i_st])

        ostr = ('Fracturing large map into '
                + f'{len(sliced_images)} smaller maps.')
        print(ostr)
        for i in range(len(sliced_images)):
            print((f'Writing new XRDMap for scan'
                   + str(self.scan_id) + f'-{i + 1}\n'
                   + f'New shape: {sliced_images[i].shape}'))

            # Seems like a weird way to access the class from within...
            new_xrdmap = self.__class__(
                scan_id=str(self.scan_id) + f'-{i + 1}',
                wd=new_dir,
                # This will force a default to scan_id with iteration
                filename=None, 
                image_data=sliced_images[i],
                energy=energy, # To allow for no energy attribute
                dwell=self.dwell,
                theta=self.theta,
                poni_file=poni, # Often None, but may be loaded...
                sclr_dict=sliced_sclr_dicts[i],
                pos_dict=sliced_pos_dicts[i],
                tth_resolution=self.tth_resolution,
                chi_resolution=self.chi_resolution,
                tth=self.tth,
                chi=self.chi,
                beamline=self.beamline,
                facility=self.facility,
                detector=self.detector,
                scan_input=sliced_scan_inputs[i],
                time_stamp=self.time_stamp,
                extra_metadata=self.extra_metadata,
                save_hdf=True,
                # Keeping everything lazy causes some inconsistencies
                dask_enabled=False
            )
        
        print('Finished fracturing maps.')

    #################################
    ### Modified Parent Functions ###
    #################################

    # Re-writing save functions which 
    # will be affected by swapped axes
    def _check_swapped_axes(func):
        """
        Decorator for transposing data with swapped fast and slow axes
        after read operations and before write operations to maintain
        consistent data shape.
        """
        @functools.wraps(func)
        def wrapped(self, *args, **kwargs):

            if self._swapped_axes:
                self.swap_axes(update_flag=False)
            
            try:
                func(self, *args, **kwargs)
                err = None
            except Exception as e:
                err = e
            
            if self._swapped_axes:
                self.swap_axes(update_flag=False)

            # Re-raise any exceptions
            if err is not None:
                raise(err)

        return wrapped

    # There is probably a better way of
    # wrapping the parent class methods?
    load_images_from_hdf = _check_swapped_axes(
                            XRDBaseScan.load_images_from_hdf)
    save_images = _check_swapped_axes(
                            XRDBaseScan.save_images)
    save_integrations = _check_swapped_axes(
                            XRDBaseScan.save_integrations)
    _dask_2_hdf = _check_swapped_axes(
                            XRDBaseScan._dask_2_hdf)
    save_sclr_pos = _check_swapped_axes(
                            XRDBaseScan.save_sclr_pos)

    
    @copy_docstring(XRDBaseScan.save_current_hdf)
    def save_current_hdf(self, verbose=False):

        super().save_current_hdf(verbose=verbose)

        # Save positions
        if (hasattr(self, 'pos_dict')
            and self.pos_dict is not None):
            # Write to hdf file
            self.save_sclr_pos('positions',
                                self.pos_dict,
                                self.position_units)
        
        # Save spots
        if (hasattr(self, 'spots')
            and self.spots is not None):
            self.save_spots()
        
        # Save vector_map
        if ((hasattr(self, 'vector_map')
             and self.vectors is not None)
            and (hasattr(self, 'edges')
             and self.edges is not None)):
            self.save_vector_map(rewrite_data=True)
    
    
    ##################
    ### Properties ###
    ##################

    # Only inherited properties thus far...        

    ##############################
    ### Calibrating Map Images ###
    ##############################
        

    def integrate1D_map(self,
                        tth_resolution=None,
                        tth_num=None,
                        unit='2th_deg',
                        mask=True,
                        return_values=False,
                        save_to_hdf=True,
                        **kwargs):
        """
        Integrate every 2D pattern in the map into 1D patterns.
        
        Iterate through every 2D pattern in the map calling the
        integrate1D_image function to integrate each pattern into 1D.
        These patterns are returned, stored internally as the
        "integrations" attribute, and/or written to the HDF file
        depending on the keyword arguments. Writing to the HDF file
        only occurs if the file is available as the "integration_data"
        group, which is create if it does not already exist.

        Parameters
        ----------
        tth_resolution : float, optional
            Scattering angle, two theta, resolution of the integrated
            1D pattern in keyword 'unit' units. This number is used
            with the measured scattering angle extent to determine
            the tth_num number of bins given to the integration. By
            default this number uses the instance tth_resolution which
            defaults to 0.01 degrees.
        tth_num : int, optional
            Direct number of bins of scattering angle to transform the
            2D image into a 1D pattern. This value is None by default
            and determined by the tth_resolution parameter. If
            provided, this number will be used over tth_resolution.
        unit : str, optional
            Units of the 1D integration. This string is used by the
            internal pyFAI integration function and by default is
            '2th_deg' for degrees.
        mask : bool or Numpy.ndarray, optional
            Mask passed over images to ignore specified pixeld from
            integration. If True, the internal mask attribute will be
            used. None by default which does nothing.
        return_values : bool, optional
            Flag controlling if the data will be returned or stored. If
            True, then data will be returned. If False, data will be
            stored internally as the "integrations" attribute and
            written to the HDF file. False by default.
        save_to_hdf : bool, optional
            Flag to control if data is written to the HDF file. Only
            used when return_values is False. True by default.
        **kwargs : optional,
            Other keyword arguments passed to the pyFAI integration
            function. These should not include the correctSolidAngle
            or polarization_factor as these corrections are handled
            elsewhere.

        Returns
        -------
        integrations : Numpy.ndarray with shape (map_y, map_x, tth_num)
            3D array matching the map shape and tth_num of integration
            intensities.
        tth : Numpy.ndarray
            1D array of two theta, scattering angle, values with length
            matching the tth_num either given or determined by
            tth_resolution.
        tth_range : tuple of length 2
            Tuple with first index as the minimum of tth and the second
            as the maximum of tth.
        tth_resolution : float
            Resolution of tth.
        
        Raises
        ------
        AttributeError if the calibration information has not been
        loaded.
        ValueError if there is insufficient information to determine
        the scattering angle binning.
        ValueError if mask does not properly match images.
        """
        
        if not hasattr(self, 'ai') or self.ai is None:
            err_str = ('Cannot integrate images without first loading '
                       + 'the calibration information.')
            raise AttributeError(err_str)
        
        if tth_resolution is None:
            tth_resolution = self.tth_resolution
        
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

        if return_values:
            note_str = ('Values will be returned and not saved nor '
                        + 'stored in XRDMap.')
            print(note_str)

        # Setup mask
        if (mask is True and hasattr(self, 'mask')):
            mask = self.mask
        if mask is not None:
            if not isinstance(self.mask, np.ndarray):
                err_str = f'Mask must be numpy array not {type(mask)}.'
                raise ValueError(err_str)
            elif not (mask.shape == self.images.shape
                      or mask.shape == self.image_shape):
                err_str = (f'Mask shape {mask.shape} does not match '
                           + f'images {self.images.shape}')
                raise ValueError(err_str)

        # Set up empty array to fill
        integrated_map1d = np.empty((*self.map_shape, 
                                     tth_num), 
                                     dtype=(self.dtype))

        # Fill array!
        print('Integrating images to 1D...')
        # TODO: Parallelize this       
        for indices in tqdm(self.indices):
            
            image = self.images[indices].copy()

            if mask is not None:
                if mask.shape == self.image_shape:
                    image *= mask
                else:
                    image *= mask[indices]
        
            tth, I = self.integrate1D_image(image=image,
                                            tth_num=tth_num,
                                            unit=unit,
                                            **kwargs)            

            integrated_map1d[indices] = I

        if return_values:
            return (integrated_map1d,
                    tth,
                    (np.min(self.tth), np.max(self.tth)),
                    tth_resolution)

        self.integrations = integrated_map1d
        
        # Save a few potentially useful parameters
        self.tth = tth
        self.extent = [np.min(self.tth), np.max(self.tth)]
        self.tth_resolution = tth_resolution

        # Save integrations to hdf
        if save_to_hdf:
            print('Writing integrations to disk...')
            self.save_integrations()
            print('done!')
            self.save_reciprocal_positions()
        

    # Briefly doubles memory. No Dask support
    # TODO: change corrections, reset projections, update map_title, etc. from XRDData
    def integrate2D_map(self,
                        tth_resolution=None,
                        tth_num=None,
                        chi_resolution=None,
                        chi_num=None,
                        unit='2th_deg',
                        mask=True,
                        **kwargs):
        """
        Integrate every 2D pattern in the map into 2D cake plots.

        Iterate through every 2D pattern in the map calling the
        integrate2D_image function to integrate each pattern into 2D
        cake plots. These plots are stored internally by deleting the
        old images and replacing the "images" attribute with the new
        2D cake plots. The two-theta and azimuthal angles corresponding
        to the cake plots axes are written to the HDF if available.

        Parameters
        ----------
        tth_resolution : float, optional
            Scattering angle, two theta, resolution of the integrated
            2D cake plot in keyword 'unit' units. This number is used
            with the measured scattering angle extent to determine
            the tth_num number of bins given to the integration. By
            default this number uses the instance tth_resolution which
            defaults to 0.01 degrees.
        tth_num : int, optional
            Direct number of bins of scattering angle to transform the
            2D image into a 2D cake plot. This value is None by default
            and determined by the tth_resolution parameter. If
            provided, this number will be used over tth_resolution.
        chi_resolution : float, optional
            Azimuthal angle, chi, resolution of the integrated
            2D cake plot in keyword 'unit' units. This number is used
            with the measured azimuthal angle extent to determine
            the chi_num number of bins given to the integration. By
            default this number uses the instance chi_resolution which
            defaults to 0.05 degrees.
        chi_num : int, optional
            Direct number of bins of azimuthal angle to transform the
            2D image into a 2D cake plot. This value is None by default
            and determined by the chi_resolution parameter. If
            provided, this number will be used over chi_resolution.
        unit : str, optional
            Units of the 2D radial and azimuthal directions. This
            string is used by the internal pyFAI integration function
            and by default is '2th_deg' for degrees.
        mask : bool or Numpy.ndarray, optional
            Mask passed over images to ignore specified pixeld from
            integration. If True, the internal mask attribute will be
            used. None by default which does nothing.
        **kwargs : optional,
            Other keyword arguments passed to the pyFAI integration
            function. These should not include the correctSolidAngle
            or polarization_factor as these corrections are handled
            elsewhere.

        Raises
        ------
        AttributeError if the calibration information has not been
        loaded.
        ValueError if there is insufficient information to determine
        the scattering or azimuthal angles binning.
        ValueError if mask does not properly match images.
        """
        
        if not hasattr(self, 'ai') or self.ai is None:
            err_str = ('Cannot integrate images without first loading '
                       + 'the calibration information.')
            raise AttributeError(err_str)
        
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

        # Setup mask
        if (mask is True and hasattr(self, 'mask')):
            mask = self.mask
        if mask is not None:
            if not isinstance(self.mask, np.ndarray):
                err_str = f'Mask must be numpy array not {type(mask)}.'
                raise ValueError(err_str)
            elif not (mask.shape == self.images.shape
                      or mask.shape == self.image_shape):
                err_str = (f'Mask shape {mask.shape} does not match '
                           + f'images {self.images.shape}')
                raise ValueError(err_str)

        # Set up empty array to fill
        integrated_map2d = np.empty((*self.map_shape, 
                                     chi_num, tth_num), 
                                     dtype=(self.dtype))
        
        # Fill array!
        print('Integrating images to 2D...')
        # TODO: Parallelize this
        for indices in tqdm(self.indices):
            
            image = self.images[indices].copy()

            if mask is not None:
                if mask.shape == self.image_shape:
                    image *= mask
                else:
                    image *= mask[indices]
        
            I, tth, chi = self.integrate2D_image(image=image,
                                                 tth_num=tth_num,
                                                 unit=unit,
                                                 **kwargs)            

            integrated_map2d[indices] = I

        # Overwrite values to save on memory. Still briefly doubled...
        self.images = integrated_map2d
        
        # Save a few potentially useful parameters
        self.tth = tth
        self.chi = chi
        self.extent = [np.min(self.tth), np.max(self.tth),
                       np.min(self.chi), np.max(self.chi)]
        self.tth_resolution = tth_resolution
        self.chi_resolution = chi_resolution

        # Save recirpocal space positions
        self.save_reciprocal_positions()


    # Backwards compatibility
    integrate1d_map = integrate1D_map
    integrate2d_map = integrate2D_map
        
    
    #######################
    ### Position Arrays ###
    #######################

    def set_positions(self,
                      pos_dict,
                      position_units=None,
                      check_init_sets=False):
        """
        Set the position dictionary attribute.

        Set the internal position dictionary attribute along with the
        position units. If an HDF file is specified, these values will
        be written to the file.

        Parameters
        ----------
        pos_dict : dict of Numpy arrays
            Dictionary of Numpy arrays matching the shape of the first
            two dimensions of images (map shape). Each array is
            associated with the position values along the map by the
            keys.
        position_units : str, optional
            Units of the position measurements. 'μm' by default.
        check_init_sets : bool, optional
            Conditional flag to disable rewriting the position
            information to the HDF file when loading from the HDF file.
            False by default and typical usage.
        """

        # Re-work dictionary keys into stable format
        temp_dict = {}
        for key in list(pos_dict.keys()):
            # Reserved keys...
            if key == 'interp_x':
                temp_dict[key] = pos_dict[key]
            elif key == 'interp_y':
                temp_dict[key] = pos_dict[key]
            # More versatile to include data from multiple sources...
            elif key in ['enc1', '1', 'x', 'X', 'map_x', 'map_X']:
                temp_dict['map_x'] = pos_dict[key]
            elif key in ['enc2', '2' 'y', 'Y', 'map_y', 'map_Y']:
                temp_dict['map_y'] = pos_dict[key]
        del pos_dict
        pos_dict = temp_dict

        # Store pos_dict as attribute
        self.pos_dict = pos_dict

        # Set position units
        if position_units is None:
            # default to microns, not that reliable...
            position_units = 'μm' 
        self.position_units = position_units

        # Write to hdf file
        self.save_sclr_pos('positions',
                            self.pos_dict,
                            self.position_units,
                            check_init_sets=check_init_sets)

    
    def map_extent(self,
                   map_x=None,
                   map_y=None,
                   with_step=False):
        """
        Get the map extent for plotting.

        This function will determine the map extent passed to the
        extent keyward argument for plotting images with Matplotlib.
        These values are derived from the internal position dictionary
        if not provided explicitly and are the extents of the positions
        with a half step size appended to each end if requested.

        Parameters
        ----------
        map_x : Numpy.ndarray matching the map shape, optional
            2D array of x-positions with the units of the internally
            stored position units. By default this value will be
            derived from the internal position dictionary.
        map_y : Numpy.ndarray matching the map shape, optional
            2D array of y-positions with the units of the internally
            stored position units. By default this value will be
            derived from the internal position dictionary.
        with_step : bool, optional
            Flag to append the map extends by a half step in both
            directions. This allows the return to be passed directly
            to Matplotlib extent keyward argument for plotting images.

        Returns
        -------
        map_extent : tuple
            Tuple of (min_x, max_x, max_y, min_y) for maps without
            swapped axes, or (min_y, max_y, max_x, min_x) with swapped
            axes. If with_step is True, and half step will be appended
            in both directions for both axes.

        Raises
        ------
        AttributeError if XRDMap is missing the position dictionary or
        'map_x' or 'map_y' is not provided.
        ValueError if map_x or map_y is not provided and cannot be
        interpretted from the position dictionary.
        """

        if ((map_x is None or map_y is None)
             and not hasattr(self, 'pos_dict')):
            err_str = ('XRDMap has no loaded pos_dict.'
                       + '\nPlease load positions or specify '
                       + 'map_x and map_y.')
            raise AttributeError(err_str)

        if map_x is None and hasattr(self, 'pos_dict'):
            # Biased towards interpolated values. More regular
            if _check_dict_key(self.pos_dict, 'interp_x'): 
                map_x = self.pos_dict['interp_x']
            elif _check_dict_key(self.pos_dict, 'map_x'):
                map_x = self.pos_dict['map_x']
            else:
                err_str = ('Cannot find known key '
                           + 'for map_x coordinates.')
                raise ValueError(err_str)
        
        if map_y is None and hasattr(self, 'pos_dict'):
            # Biased towards interpolated values. More regular
            if _check_dict_key(self.pos_dict, 'interp_y'): 
                map_y = self.pos_dict['interp_y']
            elif _check_dict_key(self.pos_dict, 'map_y'):
                map_y = self.pos_dict['map_y']
            else:
                err_str = ('Cannot find known key '
                           + 'for map_y coordinates.')
                raise ValueError(err_str)
        
        x_step, y_step = 0, 0
        # Determine fast scanning direction for map extent
        if (np.mean(np.diff(map_x, axis=1))
            > np.mean(np.diff(map_x, axis=0))):
            # Fast x-axis. Standard orientation.
            #print('Fast x-axis!')
            if with_step:
                x_step = np.mean(np.diff(map_x, axis=1))
                y_step = np.mean(np.diff(map_y, axis=0))
            map_extent = [
                np.mean(map_x[:, 0]) - (x_step / 2),
                np.mean(map_x[:, -1]) + (x_step / 2),
                np.mean(map_y[-1]) + (y_step / 2), # reversed
                np.mean(map_y[0]) - (y_step / 2)# reversed
            ] # [min_x, max_x, max_y, min_y] reversed y for matplotlib
        else: # Fast y-axis. Consider swapping axes???
            if with_step:
                x_step = np.mean(np.diff(map_x, axis=0))
                y_step = np.mean(np.diff(map_y, axis=1))
            map_extent = [
                np.mean(map_y[:, 0]) - (y_step / 2), # reversed
                np.mean(map_y[:, -1]) + (y_step / 2), # reversed
                np.mean(map_x[-1]) + (x_step / 2),
                np.mean(map_x[0]) - (x_step / 2)
            ] # [min_y, max_y, max_x, min_x] reversed x for matplotlib
        
        return tuple(map_extent)


    # Convenience function for loading scalers and positions
    # from standard map_parameters text file
    def load_map_parameters_from_txt(self,
                                     filename,
                                     wd=None,
                                     position_units=None):
        """
        Load positions and scaler dictionaries from a text file.

        This function loads the map parameters from the output of the
        io.db_io.save_map_parameters function. This is intended to
        support loading map parameters after loading images from a 4D
        image stack. If the data is loaded using the 'from_db' method,
        this function is not needed.

        Parameters
        ----------
        filename : str
            Name of text file with map parameters.
        wd : path string, optional
            Path where the file can be found. Will use the internal
            working directory if not provided.
        position_units : string, optional
            The posiiton units assigned to the values loaded in the
            parameter text file. The default value used by the
            'set_positions' function will be used if not given.
        
        Notes
        -----
        This function is not commonly used. Loading the data with the
        'from_db' method will load map parameters by default.
        """ 
        
        if wd is None:
            wd = self.wd

        path = pathify(wd, filename, '.txt')
        arr = np.genfromtxt(path)

        pos_dict, sclr_dict = {}, {}

        pos_dict['enc1'] = arr[0].reshape(self.map_shape)
        pos_dict['enc2'] = arr[1].reshape(self.map_shape)

        sclr_dict['i0'] = arr[2].reshape(self.map_shape)
        sclr_dict['i0_time'] = arr[3].reshape(self.map_shape)
        sclr_dict['im'] = arr[4].reshape(self.map_shape)
        sclr_dict['it'] = arr[5].reshape(self.map_shape)

        self.set_positions(pos_dict, position_units)
        self.set_scalers(sclr_dict)


    # Method to swap axes, specifically swapping the 
    # default format of fast and slow axes
    def swap_axes(self,
                  update_flag=True,
                  ):
        """
        Swap the mapped axes.

        This method transposes all internal data to swap the mapped
        axes. Data stored in the HDF file will only be stored in the
        original form it was acquired. If the internal "_swapped_axes"
        flag is True, the this data will be swapped when reading from
        or writing to the HDF file.

        Parameters
        ----------
        update_flag : bool, optional
            Flag whether to toggle the internal "_swapped_axes" flag.
            This flag controls whether the swapped_axes function is
            called when reading and writing attributes affected by the
            swapped spatial axes.
        """     

        if self._dask_enabled and self.title != 'final':
            warn_str = ('WARNING: Dask is enabled and saving to a '
                        + 'temporary dataset! Swapping axes may '
                        + 'create issues when updating this dataset '
                        + 'or saving images.')
            print(warn_str)

        # Swap map axes
        if (hasattr(self, 'images')
            and self.images is not None):
            self.images = self.images.swapaxes(0, 1)
        if (hasattr(self, 'integrations')
            and self.integrations is not None):
            self.integrations = self.integrations.swapaxes(0, 1)

        # Update shape values
        # self.shape = self.images.shape # breaks if no images
        self.shape = (self.shape[1], self.shape[0],
                      self.shape[2], self.shape[3])
        self.map_shape = self.shape[:2]

        # Delete any cached maps
        old_attr = list(self.__dict__.keys())       
        for attr in old_attr:
            if attr in ['_min_map',
                        '_max_map',
                        '_med_map',
                        '_sum_map',
                        '_mean_map',
                        '_min_integration_map',
                        '_max_integration_map',
                        '_med_integration_map',
                        '_sum_integration_map',
                        '_mean_integration_map',]:
                delattr(self, attr)

        # Depending on the order of when the axes are
        # swapped any of these could break...
        for attr in ['null_map',
                     'scaler_map',
                     'background',
                     'blob_masks',
                     'vector_map']:
            if (hasattr(self, attr)
                and getattr(self, attr) is not None):
                setattr(self, attr, getattr(self, attr).swapaxes(0, 1))

        # Modify dictionaries
        if hasattr(self, 'pos_dict') and self.pos_dict is not None:
            for key in list(self.pos_dict.keys()):
                self.pos_dict[key] = self.pos_dict[key].swapaxes(0, 1)  
        if hasattr(self, 'sclr_dict') and self.sclr_dict is not None:
            for key in list(self.sclr_dict.keys()):
                self.sclr_dict[key] = self.sclr_dict[key].swapaxes(0, 1)

        # Modify saturated pixel tracking
        if (hasattr(self, 'saturated_pixels')
            and self.saturated_pixels is not None
            and len(self.saturated_pixels) > 0):
            self.saturated_pixels[:, [0, 1]] = self.saturated_pixels[:, [1, 0]]

        # Update spot map_indices
        if hasattr(self, 'spots'):
            map_x_ind = self.spots['map_x'].values
            map_y_ind = self.spots['map_y'].values
            self.spots['map_x'] = map_y_ind
            self.spots['map_y'] = map_x_ind

        if update_flag: 
            self._swapped_axes = not self._swapped_axes

        @XRDBaseScan._protect_hdf()
        def save_swapped_axes(self):
            overwrite_attr(self.hdf[self._hdf_type].attrs,
                            'swapped_axes',
                            int(self._swapped_axes))
        save_swapped_axes(self)


    # TODO: Add interpolation from position dictionary without scan
    # input
    def interpolate_positions(self,
                              scan_input=None,
                              check_init_sets=False):
        """
        Interpolate mapped positions onto a regular grid.

        This method generates new entries in the position dicitionary
        (as "interp_x" and "interp_y") for the nominal mapped positions
        on a regular grid. This method uses the "scan_input" attribute
        to interpolate the nominal mapped positions assuming a
        [xstart, xend, xnum, ystart, yend, ynum] format. New position
        dictionary values are stored internall and written to the HDF
        file if available. 

        Parameters
        ----------
        scan_input : iterable, optional
            Iterable of form [xstart, xend, xnum, ystart, yend, ynum].
            If this parameter is not provided, the internal
            "scan_input" attribute will ber used instead, if available.
        check_init_sets : bool, optional
            Flag whether to overwrite data in the HDF file if available.
            By default this is set to False and should only be True
            when instantiating from the HDF file.

        Raises
        ------
        ValueError if scan_input is not provided and internal attribute
        is not available.
        """

        if scan_input is None:
            if (hasattr(self, 'scan_input')
                and self.scan_input is not None):
                scan_input = self.scan_input
            else:
                err_str = ('Cannot interpolate positiions '
                           + 'without scan input.')
                raise ValueError(err_str)
        
        xstart, xend, xnum = scan_input[0:3]
        interp_x_val = np.linspace(xstart, xend, int(xnum))
        
        ystart, yend, ynum = scan_input[3:6]
        interp_y_val = np.linspace(ystart, yend, int(ynum))

        interp_y, interp_x = np.meshgrid(interp_y_val,
                                         interp_x_val,
                                         indexing='ij')

        if self._swapped_axes:
            interp_x = interp_x.swapaxes(0, 1)
            interp_y = interp_y.swapaxes(0, 1)

        if hasattr(self, 'pos_dict') and self.pos_dict is not None:
            self.pos_dict['interp_x'] = interp_x
            self.pos_dict['interp_y'] = interp_y
        
            # Write to hdf file
            self.save_sclr_pos('positions',
                               self.pos_dict,
                               self.position_units,
                               check_init_sets=check_init_sets)
        else:
            warn_str = ('WARNING: No pos_dict found. '
                        + 'Generating from interpolated positions.')
            print(warn_str)
            pos_dict = {
                'interp_x' : interp_x,
                'interp_y' : interp_y
            }
            # Store internally and save    
            self.set_positions(pos_dict,
                               check_init_sets=check_init_sets)
    
    
    #######################
    ### Blobs and Spots ###
    #######################

    @copy_docstring(XRDBaseScan._find_blobs)
    def find_blobs(self, *args, **kwargs):
        super()._find_blobs(*args, **kwargs)
        

    def find_spots(self,
                   filter_method='minimum',
                   multiplier=5,
                   size=3,
                   expansion=10,
                   min_distance=3,
                   radius=10,
                   override_rescale=False):
        """
        Find spots and their surrounding blobs in the images.

        This function first finds blobs like the "find_blobs" function.
        Within the blobs, maxima will be marked as spots, and 
        surrounding pixels within the radius will be used to
        approximate the spot characteristics (e.g., height, intensity,
        center of mass, etc.).

        Found blobs will be stored internally and written the HDF
        file if available as "blob_masks" in the "image_data" group.
        Found spots and their approximated characteristics will be 
        stored internally as "spots" and written to the HDF file in the
        "reflections" group.

        Parameters
        ----------
        filter_method : {"minimum", "gaussian", "median"}, optional
            Determines which filter method will be used from the
            scipy.ndimage module. The "minimum" filter will also call a
            small gaussian filter in order to avoid outliers. The
            minimum filter is used by default.
        multiplier
            The value multiplied by the standard deviation of the
            thresholded image noise used as the cutoff threshold of the
            smoothed images. This is the main tuning parameter for blob
            selection. Higher values lead to smaller and fewer blobs.
            By default this is 5.
        size : float or int, optional
            The size argument passed to the smoothing filter in pixel
            units. For "minimum" and "median" filters, this number
            should be an integer. For the "gaussian" filter, this
            number is the sigma value and can be a float. This value
            has a more complex relationship with the size and number
            of found blobs. By defaul this number is 3.
        expansion : int, optional
            How many pixels to expand beyond the thresholded pixels. By
            default this number is 10.
        min_distance : int, optional
            Minimum separation distance in pixels between maxima used
            as spots.
        radius : int, optional
            Radius in pixels around each found spots used to
            approximate the spot characteristics. By default the radius
            is 10.
        override_rescale : bool, optional
            Flag to override the internal check to ensure images have
            been rescaled before searching for blobs.

        Notes
        -----
        Spots should only be found for XRDMaps that will not be
        compared to other XRDMaps in a rotation or energy series.

        This function can only be performed on images after the
        "rescale_images" correction has been applied under the
        assumption that the 100 and 0 are the maximum and minimum
        measureable pixel intensities. This means the 0.01 value
        for determining the background noise is about 0.01% of the
        measurable intensity on the detector.
        """
        
        if (hasattr(self, 'blob_masks')
            and self.blob_masks is not None):
            warn_str = ('WARNING: XRDMap already has blob_masks '
                        + 'attribute. This will be overwritten with '
                        + 'new parameters.')
            print(warn_str)
            
        # Cleanup images as necessary
        self._dask_2_numpy()
        if not self.corrections['rescaled'] and not override_rescale:
            warn_str = ("Finding spots assumes images scaled between 0"
                        + " and around 100. Current images have not "
                        + "been rescaled. Apply this correction or "
                        + "set 'override_rescale' to True in order to"
                        + " continue.\nProceeding without changes.")
            print(warn_str)
            return
        
        # Search each image for significant blobs and spots
        spot_list, blob_mask_list = find_blobs_spots(
                    self.images,
                    mask=self.mask,
                    filter_method=filter_method,
                    multiplier=multiplier,
                    size=size,
                    expansion=expansion,
                    min_distance=min_distance)
        
        # Initial characterization of each spot
        stat_list = find_spot_stats(self,
                                    spot_list,
                                    self.tth_arr,
                                    self.chi_arr,
                                    radius=radius)
        
        # Convert spot stats into dict, then pandas dataframe
        self.spots = make_stat_df(stat_list, self.map_shape)
        
        # Reformat blobs
        self.blob_masks = np.asarray(
                                blob_mask_list).reshape(self.shape)

        # Save spots to hdf
        self.save_spots(extra_attrs={'radius' : radius})

        # Save blob_masks to hdf
        self.save_images(images='blob_masks',
                         title='_blob_masks',
                         units='bool',
                         extra_attrs={
                            'filter_method' : filter_method,
                            'size' : size,
                            'multiplier' : multiplier,
                            'expansion' : expansion})
        
    
    def recharacterize_spots(self,
                             radius=10):
        """
        Recharacterize spots.

        Re-approximate the spot characteristics. This method will
        remove old approximate spot characteristics.

        Parameters
        ----------
        radius : int, optional
            Radius in pixels around each found spots used to
            approximate the spot characteristics. By default the
            radius is 10.
        
        Raises
        ------
        AttributeError if XRDMap does not already have spots.
        """

        # Check for spots
        if (not hasattr(self, "spots")
            or not isinstance(self.spots, pd.DataFrame)):
            err_str = "XRDMap is missing 'spots' attribute."
            raise AttributeError(err_str)

        # Remove spot guesses
        print('Removing spot guess characteristics...')
        self.remove_spot_guesses()
    
        spot_list = remake_spot_list(self.spots, self.map_shape)

        # Initial characterization of each spot
        stat_list = find_spot_stats(self,
                                    spot_list,
                                    self.tth_arr,
                                    self.chi_arr,
                                    radius=radius)
        
        # Convert spot stats into dict, then pandas dataframe
        self.pixel_spots = make_stat_df(stat_list, self.map_shape)

        # Save spots to hdf
        self.save_spots()

    
    def fit_spots(self,
                  SpotModel,
                  max_dist=0.5,
                  sigma=1):
        """
        Fit found spots with a SpotModel.

        Use a SpotModel to fit found spots and better determine their
        characteristics. Blobs are segmented and spots are combined to
        fit several small clusters of spots within the image instead of
        every spot at once. Approximate spot parameters are used as the 
        intitial values for fitting.
        
        The internally stored spots attributed and HDF dataset, are
        expanded with the fit characteristics.

        Parameters
        ----------
        SpotModel : SpotModel
            SpotModel object from the reflections.SpotModels module
            used to fit spots. Available models are GaussianFunctions,
            LorentzFunctions, and PseudoVoigtFunctions.
        max_dist : float, optional
            Distance in pixels used to associate nearby spots for
            combined spot fitting. Larger value will attempt to fit
            more spots simultaneously yielding better fits, but will
            take more time. By default this value is 0.5.
        sigma : float, optional
            Sigma in pixel units of a guassian filter used to segment
            blobs. Larger values will result in larger segments and
            attempts to fit more spots simultaneously yielding better
            fits, but will take more time. By defualt this value is 1.

        Raises
        ------
        AttributeError if spots have not already been determined.

        Notes
        -----
        Monochromatic XRD which produces individual spots instead of
        powder rings lacks the availability of orientations to fully
        describe a crystal lattice. Detailed individual spot fitting
        of this partial dataset will not provide more information.
        Consider acquiring single XRD patterns (XRDRockingCurve) or
        full maps (XRDMapStack) though a series of sample rotations and
        incident X-ray energies instead.
        """

        # Find spots in self or from hdf
        if not hasattr(self, 'spots'):
            print('No reflection spots found...')
            if self.hdf_path is not None:
                keep_hdf = self.hdf is not None

                if 'reflections' in self.hdf[self._hdf_type].keys():
                    print('Loading reflection spots from the HDF file...',
                          end='', flush=True)
                    self.close_hdf()
                    spots = pd.read_hdf(
                            self.hdf_path,
                            key=f'{self._hdf_type}/reflections/spots')
                    self.spots = spots
                    self.open_hdf()

                    # Close hdf and reset attribute
                    if not keep_hdf:
                        self.hdf.close()
                        self.hdf = None
                    print('done!')

                else:
                    err_str = ('XRDMap does not have any reflection '
                               + 'spots! Please find spots first.')
                    raise AttributeError(err_str)
            else:
                err_str = ('XRDMap does not have any reflection '
                           + 'spots! Please find spots first.')
                raise AttributeError(srr_str)

        # Fit spots
        fit_spots(self, SpotModel, max_dist=max_dist, sigma=sigma)
        self.spot_model = SpotModel

        # Save spots to hdf
        self.save_spots(
            extra_attrs={
                'spot_model' : self.spot_model.name})


    def initial_spot_analysis(self,
                              SpotModel=None):
        """
        Expanded spot analsysis.

        Expanded spot analysis based on a SpotModel. Approximate spot
        significance and reciprocal space positions are determined and
        appended to the spots attribute.

        Parameters
        ----------
        SpotModel : SpotModel, optional
            SpotModel object from the reflections.SpotModels module
            used to fit spots. Available models are GaussianFunctions,
            LorentzFunctions, and PseudoVoigtFunctions.
            GaussianFunctions are used by default.
        """

        if SpotModel is None and hasattr(self, 'spot_model'):
            SpotModel = self.spot_model

        # Initial spot analysis...
        _initial_spot_analysis(self, SpotModel=SpotModel)

        # Save spots to hdf
        self.save_spots()
        

    def trim_spots(self,
                   remove_less=0.01,
                   key='guess_int',
                   save_spots=False):
        """
        Trim all spots below some value.

        Trim the spots dataframe with any value in a certain key
        below some cutoff.

        Paramters
        ---------
        remove_less : float, optional
            Cutoff value used for deciding which spots to trim. Default
            is 0.01.
        key : str, optional
            Key in dataframe to compare with the remove_less value.
            "guess_int" by default.
        save_spots : bool, optional
            Flag to save trimmed spots to HDF file if available.
            Saving new spots may distort the meaning of any metadata
            saved along with the original spots.

        Raises
        ------
        KeyError if the key parameter is not within the spots.
        """
        
        self._trim_spots(self.spots,
                         remove_less=remove_less,
                         key=key)

        if save_spots:
            self.save_spots()

    
    def _remove_spot_vals(self, drop_keys=[], drop_tags=[]):
        """
        Internal function for removing spot characterisitics.

        Parameters
        ----------
        drop_keys : list of str, optional
            List of specific keys in the spots DataFrame to remove.
            Empty by default.
        drop_tags : list of str, optional
            List of tags where any key in the spots DataFrame
            containing one of these tags will be removed.
        """


        for key in list(self.spots.keys()):
            for tag in drop_tags:
                if tag in key:
                    drop_keys.append(key)
        print(f'Removing spot values for {drop_keys}')
        self.spots.drop(drop_keys, axis=1, inplace=True)


    def remove_spot_guesses(self):
        """
        Remove all guess characteristics from spots.
        """

        self._remove_spot_vals(drop_tags=['guess'])
    

    def remove_spot_fits(self):
        """
        Remove all fit characteristics from spots.
        """

        self._remove_spot_vals(drop_tags=['fit'])


    @staticmethod
    def _pixel_spots(spots,
                     map_indices,
                     copied=True):
        """
        Internal function for creating a spots DataFrame from a single
        mapped pixel.

        Parameters
        ----------
        spots : Pandas DataFrame
            Spots DataFrame which will be parsed to a single mapped
            pixel. The spots DataFrame should "map_x" and "map_y" keys.
        map_indices : iterable ("map_y", "map_x")
            Indices of mapped pixel to return spots.
        copied : bool, optional
            Flag to signal whether the returned pixel spots is a subset
            of the original spots DataFrame, or a copy of this view.
            True by default.

        Returns
        -------
        pixel_spots : Pandas DataFrame
            A subset of the spots DataFrame matching the mapped
            indices. If there are no spots within the pixel, a
            DataFrame with the columns of spots, but without any rows
            will be returned.

        Raises
        ------
        KeyError if mapped indices are not in spots DataFrame.
        """

        if 'map_x' not in spots or 'map_y' not in spots:
            err_str = f'Mapped indices are not in spots.'
            raise KeyError(err_str)

        pixel_spots = spots[(spots['map_x'] == map_indices[1])
                            & (spots['map_y'] == map_indices[0])]
        
        # Copies to protect orginal spots from changes
        if copied:
            pixel_spots = pixel_spots.copy()
        
        return pixel_spots
    

    def pixel_spots(self, map_indices, copied=True):
        """
        Get spots from a single map pixel.
    
        Parameters
        ----------
        map_indices : iterable ("map_y", "map_x")
            Indices of mapped pixel to return spots.
        copied : bool, optional
            Flag to signal whether the returned pixel spots is a subset
            of the original spots DataFrame, or a copy of this view.
            True by default.

        Returns
        -------
        pixel_spots : Pandas DataFrame
            A subset of the spots DataFrame matching the mapped
            indices. If there are no spots within the pixel, a
            DataFrame with the columns of spots, but without any rows
            will be returned.

        Raises
        ------
        KeyError if mapped indices are not in spots DataFrame.
        """

        return self._pixel_spots(self.spots,
                                 map_indices,
                                 copied=copied)


    @_check_swapped_axes
    @XRDBaseScan._protect_hdf(pandas=True)
    def save_spots(self, extra_attrs=None):
        """
        Save spots to the HDF file if available.

        Parameters
        ----------
        extra_attrs : dict, optional
            Dictionary of extra metadata that will be written into the
            attributes of the reflections group in the HDF file. None
            by default.
        """

        print('Saving spots to the HDF file...', end='', flush=True)
        hdf_str = f'{self._hdf_type}/reflections'
        self.spots.to_hdf(
                        self.hdf_path,
                        key=f'{hdf_str}/spots',
                        format='table')

        if extra_attrs is not None:
            if self.hdf is None:
                self.open_hdf()
            for key, value in extra_attrs.items():
                overwrite_attr(self.hdf[hdf_str].attrs, key, value)
        print('done!')

    ############################
    ### Vectorizing Map Data ###
    ############################

    @XRDBaseScan._protect_hdf()
    def vectorize_map_data(self,
                           image_data_key='recent',
                           keep_images=False,
                           rewrite_data=True,
                           verbose=False):
        """
        Convert blobs into 3D reciprocal space vectors.

        Convert each pixel inside of found blobs into a list of 3D
        reciprocal space vectors. Each pixel is assigned it's q-space
        coordinates and intensity. The map of vectors is then stored
        internally and written to the HDF file if available as 
        "vector_map".

        Parameters
        ----------
        image_data_key : str, optional
            Image dataset to load from the HDF file if images are not
            already loaded. Loads the most recently saved images by
            default.
        keep_images : bool, optional
            Flag to keep or remove images after vectorization. Images
            are removed by default.
        rewrite_data : bool, optional,
            Flag to determine the behavior of overwriting previously
            written vector maps. False by default, preserving previously
            written data.
        verbose : bool, optional 
            Flag to determine the function's verbosity. False
            by default.

        Raises
        ------
        AttributeError if "blob_masks" attribute does not exist and
        cannot be loaded from the HDF File.

        Notes
        -----
        XRDMaps should only be vectorized when compared to other maps
        in a rotation or energy/wavelength series.
        """

        # Check required data
        remove_blob_masks_after = False
        if not hasattr(self, 'blob_masks') or self.blob_masks is None:
            if '_blob_masks' in self.hdf[
                                f'{self._hdf_type}/image_data'].keys():
                self.load_images_from_hdf('_blob_masks')
            # For backwards compatibility
            elif '_spot_masks' in self.hdf[
                                f'{self._hdf_type}/image_data'].keys():
                self.load_images_from_hdf('_spot_masks')
            else:
                err_str = (f'{self._hdf_type} does not have blob_masks'
                        + 'attribute.')
                raise AttributeError(err_str)
            remove_blob_masks_after = True

        remove_images_after = False
        if not hasattr(self, 'images') or self.images is None:
            remove_images_after = True
            self.load_images_from_hdf(image_data_key=image_data_key)

        print('Vectorizing map data...')
        vector_map = np.empty(self.map_shape, dtype=object)

        for indices in tqdm(self.indices):
            blob_mask = self.blob_masks[indices]
            intensity = self.images[indices][blob_mask]
            q_vectors = self.q_arr[blob_mask]
            
            vector_map[indices] = np.hstack([q_vectors,
                                             intensity.reshape(-1, 1)])

        # Record values
        self.vector_map = vector_map
        # Write to hdf
        self.save_vector_map(rewrite_data=rewrite_data,
                             verbose=verbose)
        
        # Cleaning up XRDMap state
        if not keep_images and remove_images_after:
            self.dump_images()
        if not keep_images and remove_blob_masks_after:
            del self.blob_masks
            self.blob_masks = None       
    

    @_check_swapped_axes
    @XRDBaseScan._protect_hdf()
    def save_vector_map(self,
                        vector_map=None,
                        edges=None,
                        rewrite_data=False,
                        verbose=False):
        """
        Save the vector map to the HDF file.

        Parameters
        ----------
        vector_map : Numpy.ndarray of objects
            Numpy array sharing the map shape. Each index
            contains a list of vectors or empty list.
        edges : list, optional
            List of lists of vectors defining the edges of the 
            sampled reciprocal space volume. These will be written
            into their own 'edges' group. Previous data will be
            rewritten according to the rewrite_data flag. By default
            no edges will be passed and nothing will be written. Only
            used by XRDMapStack.
        rewrite_data : bool, optional,
            Flag to determine the behavior of overwriting previously
            written vector maps. False by default, preserving previously
            written data.
        verbose : bool, optional 
            Flag to determine the function's verbosity. False
            by default.

        Raises
        ------
        AttributeError if "vector_map" is not provided and not and
        attribute of XRDMap.
        """

        # Check input
        if vector_map is None:
            if (hasattr(self, 'vector_map')
                and self.vector_map is not None):
                vector_map = self.vector_map
            else:
                err_str = ('Must provide vector_map or '
                        + f'{self.__class__.__name__} must have '
                        + 'vector_map attribute.')
                raise AttributeError(err_str)

        # Use internal edges if available. Only for XRDMapStack
        if edges is None:
            if hasattr(self, 'edges'):
                edges = self.edges
    
        self._save_vector_map(self.hdf,
                              vector_map=vector_map,
                              edges=edges,
                              rewrite_data=rewrite_data,
                              verbose=verbose)
        

    @_check_swapped_axes
    @XRDBaseScan._protect_hdf()
    def load_vector_map(self):
        """
        Load the vector map from the HDF file.
        """

        self._load_vectors(self.hdf)


    #################################
    ### Analysis of Selected Data ###
    #################################
        


    #############################################################
    ### Interfacing with Fluorescence Data from pyxrf Results ###
    #############################################################
    
    def load_xrfmap(self,
                    wd=None,
                    xrf_name=None,
                    include_scalers=False,
                    include_positions=False,
                    include_full_data=False):
        """
        Load XRF data from a PyXRF file.

        Load the XRF data from an associated PyXRF HDF file into the
        local instance. Data is stored in an internal "xrf" attribute
        as a dictionary of elemental fitting maps, scalers, positions,
        and full spectra. The path of the PyXRF HDF file is written
        to the XRDMap HDF file for easier loading for future
        instantiations.

        Parameters
        ----------
        wd : path str, optional
            Working directory of PyXRF HDF file. The function
            will use the same working directory as the XRDMap.
        xrf_name : str, optional
            Name of the PyXRF HDF file. The default filename assigned
            by PyXRF using the XRDMap scan ID will be used by default.
        include_scalers : bool, optional
            Flag to indicate if the scaler information will be loaded.
            This information should be duplicates of the internal
            scaler dictionary. False by default.
        include_positions : bool, optional
            Flag to indicate if the position information will be
            loaded. this information should be duplicates of the
            internal position dictionary. False by default.
        include_full_data : bool, optional
            Flag to indicate if the full XRF spectra should be loaded.
            False by default.
        
        Raises
        ------
        FileNotFoundError if the indicated file cannot be found at the
        specified working directory.
        """

        # Look for path if no information is provided
        if (wd is None
            and xrf_name is None
            and self.xrf_path is not None):
            xrf_path = self.xrf_path

            if not os.path.exists(xrf_path):
                raise FileNotFoundError(f"{xrf_path} does not exist.")
        else:
            if wd is None:
                wd = self.wd
            
            # Try various filenames
            for fname in [xrf_name,
                          f'scan2D_{self.scan_id}_xs_sum8ch',
                          f'autorun_scan2D_{self.scan_id}_xs_sum8ch']:
                if fname is None:
                    continue

                xrf_path = pathify(wd, fname, '.h5',
                                   check_exists=False)

                if os.path.exists(xrf_path):
                    self.xrf_path = xrf_path
                    break
            else:
                err_str = (f"Could not find a suitable file in {wd} "
                           + f" for scan {self.scan_id} without more "
                           + "information.")
                raise FileNotFoundError(err_str)

        # Load the data
        xrf = {}
        with h5py.File(self.xrf_path, 'r') as f:
            # Get fit maps  
            if 'xrf_fit_name' in f['xrfmap/detsum']:
                xrf_fit_names = [d.decode('utf-8')
                                 for d
                                 in f['xrfmap/detsum/xrf_fit_name'][:]]
                xrf_fit = f['xrfmap/detsum/xrf_fit'][:]
                
                for key, value in zip(xrf_fit_names, xrf_fit):
                    xrf[key] = value
            
            elif not include_full_data:
                warn_str = ('WARNING: XRF fitting not found and '
                            + 'include full_data flag not indicated. '
                            + 'No data loaded.')
                print(warn_str)
                return
            
            # Get scalers
            if include_scalers and 'scalers' in f['xrfmap']:
                scaler_names = [d.decode('utf-8')
                                for d
                                in f['xrfmap/scalers/name'][:]]
                scalers = np.moveaxis(f['xrfmap/scalers/val'][:], -1, 0)
                
                for key, value in zip(scaler_names, scalers):
                    xrf[key] = value

            # Get positions
            if include_positions and 'positions' in f['xrfmap']:
                position_names = [d.decode('utf-8')
                                for d
                                in f['xrfmap/positions/name'][:]]
                positions = f['xrfmap/positions/pos'][:]
                
                for key, value in zip(position_names, positions):
                    xrf[key] = value

            # Get full data
            if include_full_data:
                xrf['data'] = f['xrfmap/detsum/counts'][:]
                xrf['energy'] = (np.arange(xrf['data'].shape[-1])
                                 / 100)

            md_key = 'xrfmap/scan_metadata'
            E0_key = 'instrument_mono_incident_energy'
            xrf['E0'] = f[md_key].attrs[E0_key]
        
        # Track as attribute
        self.xrf = xrf
            
        @XRDBaseScan._protect_hdf()
        def save_xrf_path(self):
            overwrite_attr(self.hdf[self._hdf_type].attrs,
                           'xrf_path',
                           self.xrf_path)
        save_xrf_path(self)
        

    ##########################
    ### Plotting Functions ###
    ##########################
    
    @return_plot_wrapper
    def plot_map(self,
                 map_values,
                 map_extent=None,
                 position_units=None,
                 title=None,
                 fig=None,
                 ax=None,
                 title_scan_id=True,
                 **kwargs):
        """
        Plot a map.

        Map plotting function for any map matching the XRDMap map
        shape.

        Parameters
        ----------
        map_values : 2D Numpy.ndarray matching the map shape
            Values to be plotted as a map.
        map_extent : iterable, optional
            Mapped extent as an iterable with (min_x, max_x, max_y,
            min_y) values. By default the result of the internal
            "map_extent" method.
        position_units : str, optional
            Position units of the map_exent. Internal position units
            are used by default.
        title : str, optional
            Title of the map. By default "Custom Map" will be used.
        fig : Matplotlib Figure instance, optional
            Figure to be used for plotting. Must be given
            ax. None by default, generating an new plot.
        ax : Matplotlib Axes instance, optional
            Axes to be used for plotting. Must be given with
            fig. None by default, generating a new plot.
        title_scan_id : bool, optional
            Flag dictating if the plot title will be 
            prepended with the data scan ID. True by 
            default.
        kwargs : dict, optional
            Dictionary of keyword arguments passed to
            the internal Matplotlib.pyplot.imshow function.

        Returns
        -------
        No returns by default. Only returned if return_plot
        keyword argument is True.

        fig : Matplotlib Figure instance
            Figure of plot.
        ax : Matplotlib Axes instance, optional
            Individual axes of plot.

        Raises
        ------
        ValueError if the map_values array does not match the XRDMap
        map shape.
        """
        
        map_values = np.asarray(map_values)
        
        if (hasattr(self, 'pos_dict')
            and (map_values.shape
                   != list(self.pos_dict.values())[0].shape)):
            err_str = (f'Map input shape {map_values.shape} does '
                       + f'not match instance shape '
                       + f'of {list(self.pos_dict.values())[0].shape}')
            raise ValueError(err_str)
        
        if map_extent is None:
            map_extent = self.map_extent(with_step=True)
        if position_units is None:
            position_units = self.position_units

        title = self._title_with_scan_id(
                        title,
                        default_title='Custom Map',
                        title_scan_id=title_scan_id)
        
        return plot_map(map_values,
                        map_extent=map_extent,
                        position_units=position_units,
                        title=title,
                        fig=fig,
                        ax=ax,
                        **kwargs)


    # Interactive plots do not currently accept fig, ax inputs
    @return_plot_wrapper
    def plot_interactive_map(self,
                             dyn_kw=None,
                             map_kw=None,
                             title_scan_id=True,
                             **kwargs):
        """
        Plot an interactive map of images.

        Plot an interactive map, where each mapped pixel shows its
        associated 2D XRD pattern.

        Parameters
        ----------
        dyn_kw : dict, optional
            Dictionary of keyword arguments passed for dynamic
            plotting. Empty by default, and the values will be filled
            with the relevant instance parameters.

            Useful Keywords:
                vmin : float, optional
                    Starting maximum value for dynamic colormap.            
                vmax : float, optional
                    Starting minimum value for dynamic colormap.

        map_kw : dict, optional
            Dictionary of keyword arguments passed for the map
            plotting. Empty by default, and the values will be filled
            with the relevant instance parameters.

            Useful Keywords:
                map : 2D Numpy.ndarray matching the map shape, optional
                    Values to be plotted as a map.
                title : str, optional
                    Title of map plot.
                vmin : float, optional
                    Starting minimum value for map colormap.
                vmax : float, optional
                    Starting maximum value for map colormap.
        
        title_scan_id : bool, optional
            Flag dictating if the plot title will be 
            prepended with the data scan ID. True by 
            default.
        kwargs : dict, optional
            Dictionary of keyword arguments passed to interactive
            plotting function.
        
        Returns
        -------
        No returns by default. Only returned if return_plot
        keyword argument is True.

        fig : Matplotlib Figure instance
            Figure of plot.
        ax : Matplotlib Axes instance, optional
            Individual axes of plot.
        
        Raises
        ------
        ValueError if images for plotting dynamic XRD patterns cannot
        be found or are of the wrong shape.
        """
        
        # Python doesn't play well with mutable default kwargs
        if dyn_kw is None:
            dyn_kw = {}
        if map_kw is None:
            map_kw = {}

        if _check_dict_key(dyn_kw, 'data'):
            pass
        elif not hasattr(self, 'images'):
            raise ValueError('Could not find images to plot data!')
        elif self.images.ndim != 4:
            err_str = ('XRDData data shape is not 4D, '
                       + f'but {self.images.ndim}.')
            raise ValueError(err_str)
        else:
            dyn_kw['data'] = self.images

        if (self.corrections['polar_calibration']
            and not _check_dict_key(dyn_kw, 'x_ticks')):
            if hasattr(self, 'tth') and self.tth is not None:
                dyn_kw['x_ticks'] = self.tth
                dyn_kw['x_label'] = ('Scattering Angle, 2θ '
                                     + f'[{self.scattering_units}]')
            if hasattr(self, 'chi') and self.chi is not None:
                dyn_kw['y_ticks'] = self.chi
                dyn_kw['x_label'] = ('Azimuthal Angle, χ '
                                     + f'[{self.polar_units}]')

        # Add default map_kw information if not already included
        if not _check_dict_key(map_kw, 'map'):
            map_kw['map'] = self.max_map
            map_kw['title'] = 'Max Detector Intensity'
        if not _check_dict_key(map_kw, 'x_ticks'):
            map_kw['x_ticks'] = np.round(np.linspace(
                *self.map_extent()[:2], # without step
                self.map_shape[1]), 2)
        if not _check_dict_key(map_kw, 'y_ticks'):
            map_kw['y_ticks'] = np.round(np.linspace(
                *self.map_extent()[2:], # without step
                self.map_shape[0]), 2)
        if hasattr(self, 'position_units'):
            if not _check_dict_key(map_kw, 'x_label'):
                map_kw['x_label'] = ('x position '
                                     + f'[{self.position_units}]')
            if not _check_dict_key(map_kw, 'y_label'):
                map_kw['y_label'] = ('y position '
                                     + f'[{self.position_units}]')
        
        if 'title' not in map_kw:
            map_kw['title'] = None
        map_kw['title'] = self._title_with_scan_id(
                            map_kw['title'],
                            default_title='Custom Map',
                            title_scan_id=title_scan_id)

        return interactive_2D_plot(dyn_kw,
                                   map_kw,
                                   **kwargs)
                    


    @return_plot_wrapper
    def plot_interactive_integration_map(self,
                                         dyn_kw=None,
                                         map_kw=None,
                                         title_scan_id=True,
                                         **kwargs):
        """
        Plot an interactive map of integrations.

        Plot an interactive map, where each mapped pixel shows its
        associated 1D XRD pattern integration.

        Parameters
        ----------
        dyn_kw : dict, optional
            Dictionary of keyword arguments passed for dynamic
            plotting. Empty by default, and the values will be filled
            with the relevant instance parameters.

            Useful Keywords:
                y_min : float, optional
                    Starting maximum y-value (intensity) for dynamic
                    plot.            
                y_max : float, optional
                    Starting minimum y-value (intensity) for dynamic
                    plot.
                x_min : float, optional
                    Starting maximum x-value (scattering angle) for
                    dynamic plot.            
                x_max : float, optional
                    Starting minimum x-value (scattering angle) for
                    dynamic plot.

        map_kw : dict, optional
            Dictionary of keyword arguments passed for the map
            plotting. Empty by default, and the values will be filled
            with the relevant instance parameters.

            Useful Keywords:
                map : 2D Numpy.ndarray matching the map shape, optional
                    Values to be plotted as a map.
                title : str, optional
                    Title of map plot.
                vmin : float, optional
                    Starting minimum value for map colormap.
                vmax : float, optional
                    Starting maximum value for map colormap.

        title_scan_id : bool, optional
            Flag dictating if the plot title will be 
            prepended with the data scan ID. True by 
            default.
        kwargs : dict, optional
            Dictionary of keyword arguments passed to the interactive
            plotting function.
        
        Returns
        -------
        No returns by default. Only returned if return_plot
        keyword argument is True.

        fig : Matplotlib Figure instance
            Figure of plot.
        ax : Matplotlib Axes instance, optional
            Individual axes of plot.
        
        Raises
        ------
        ValueError if integrations for plotting dynamic XRD patterns
        cannot be found or are of the wrong shape.
        """
        
        # Python doesn't play well with mutable default kwargs
        if dyn_kw is None:
            dyn_kw = {}
        if map_kw is None:
            map_kw = {}

        if _check_dict_key(dyn_kw, 'data'):
            dyn_kw['data'] = np.asarray(dyn_kw['data'])
        elif not hasattr(self, 'integrations'):
            err_str = 'Could not find integrations to plot data!'
            raise ValueError(err_str)
        elif self.integrations.ndim != 3:
            err_str = ('Integration data shape is not 3D, '
                       + f'but {self.integrations.ndim}.')
            raise ValueError(err_str)
        else:
            dyn_kw['data'] = self.integrations

        if not _check_dict_key(dyn_kw, 'x_ticks'):
            if hasattr(self, 'tth') and self.tth is not None:
                dyn_kw['x_ticks'] = self.tth
                dyn_kw['x_label'] = ('Scattering Angle, 2θ '
                                     + f'[{self.scattering_units}]')
    
        # Add default map_kw information if not already included
        if not _check_dict_key(map_kw, 'map'):
            map_kw['map'] = self.max_integration_map
            map_kw['title'] = 'Max Detector Intensity'
        if not _check_dict_key(map_kw, 'x_ticks'):
            map_kw['x_ticks'] = np.round(np.linspace(
                *self.map_extent()[:2], # without step
                self.map_shape[1]), 2)
        if not _check_dict_key(map_kw, 'y_ticks'):
            map_kw['y_ticks'] = np.round(np.linspace(
                *self.map_extent()[2:], # without step
                self.map_shape[0]), 2)
        if hasattr(self, 'position_units'):
            if not _check_dict_key(map_kw, 'x_label'):
                map_kw['x_label'] = ('x position '
                                     + f'[{self.position_units}]')
            if not _check_dict_key(map_kw, 'y_label'):
                map_kw['y_label'] = ('y position '
                                     + f'[{self.position_units}]')
        
        if 'title' not in map_kw:
            map_kw['title'] = None
        map_kw['title'] = self._title_with_scan_id(
                            map_kw['title'],
                            default_title='Custom Map',
                            title_scan_id=title_scan_id)
    
        return interactive_1D_plot(dyn_kw,
                                   map_kw,
                                   **kwargs)


    @return_plot_wrapper
    def plot_1D_window_sum_map(self,
                               dyn_kw=None,
                               map_kw=None,
                               title_scan_id=True,
                               **kwargs):
        """
        Plot an interactive map of a 1D window sum.

        Draw a window across the maximum 1D integration and plot a map
        of the sum within this window.

        Parameters
        ----------
        dyn_kw : dict, optional
            Dictionary of keyword arguments passed for dynamic
            plotting. Empty by default, and the values will be filled
            with the relevant instance parameters.

            Useful Keywords:
                y_min : float, optional
                    Starting maximum y-value (intensity) for dynamic
                    plot.            
                y_max : float, optional
                    Starting minimum y-value (intensity) for dynamic
                    plot.
                x_min : float, optional
                    Starting maximum x-value (scattering angle) for
                    dynamic plot.            
                x_max : float, optional
                    Starting minimum x-value (scattering angle) for
                    dynamic plot.

        map_kw : dict, optional
            Dictionary of keyword arguments passed for the map
            plotting. Empty by default, and the values will be filled
            with the relevant instance parameters.

            Useful Keywords:
                title : str, optional
                    Title of map plot.
                vmin : float, optional
                    Starting minimum value for map colormap.
                vmax : float, optional
                    Starting maximum value for map colormap.

        title_scan_id : bool, optional
            Flag dictating if the plot title will be 
            prepended with the data scan ID. True by 
            default.
        kwargs : dict, optional
            Dictionary of keyword arguments passed to the interactive
            plotting function.
        
        Returns
        -------
        No returns by default. Only returned if return_plot
        keyword argument is True.

        fig : Matplotlib Figure instance
            Figure of plot.
        ax : Matplotlib Axes instance, optional
            Individual axes of plot.
        
        Raises
        ------
        ValueError if integrations for plotting dynamic XRD patterns
        cannot be found or are of the wrong shape.
        """
        
        dyn_kw, map_kw = self._prepare_1D_window_plot(
                                        dyn_kw=dyn_kw,
                                        map_kw=map_kw,
                                        title_scan_id=title_scan_id
                                        )
        
        fig, ax, span = interactive_1D_window_sum_plot(dyn_kw=dyn_kw,
                                                       map_kw=map_kw,
                                                       **kwargs)
        # Save internally for reference.
        self._1D_window_sum_span = span

        return fig, ax
        

    @return_plot_wrapper
    def plot_1D_window_com_map(self,
                               dyn_kw=None,
                               map_kw=None,
                               title_scan_id=True,
                               **kwargs):
        """
        Plot an interactive map of a 1D window center of mass.

        Draw a window across the maximum 1D integration and plot a map
        of the center of mass within this window.

        Parameters
        ----------
        dyn_kw : dict, optional
            Dictionary of keyword arguments passed for dynamic
            plotting. Empty by default, and the values will be filled
            with the relevant instance parameters.

            Useful Keywords:
                y_min : float, optional
                    Starting maximum y-value (intensity) for dynamic
                    plot.            
                y_max : float, optional
                    Starting minimum y-value (intensity) for dynamic
                    plot.
                x_min : float, optional
                    Starting maximum x-value (scattering angle) for
                    dynamic plot.            
                x_max : float, optional
                    Starting minimum x-value (scattering angle) for
                    dynamic plot.

        map_kw : dict, optional
            Dictionary of keyword arguments passed for the map
            plotting. Empty by default, and the values will be filled
            with the relevant instance parameters.

            Useful Keywords:
                title : str, optional
                    Title of map plot.
                vmin : float, optional
                    Starting minimum value for map colormap.
                vmax : float, optional
                    Starting maximum value for map colormap.

        title_scan_id : bool, optional
            Flag dictating if the plot title will be 
            prepended with the data scan ID. True by 
            default.
        kwargs : dict, optional
            Dictionary of keyword arguments passed to the interactive
            plotting function.
        
        Returns
        -------
        No returns by default. Only returned if return_plot
        keyword argument is True.

        fig : Matplotlib Figure instance
            Figure of plot.
        ax : Matplotlib Axes instance, optional
            Individual axes of plot.
        
        Raises
        ------
        ValueError if integrations for plotting dynamic XRD patterns
        cannot be found or are of the wrong shape.
        """

        dyn_kw, map_kw = self._prepare_1D_window_plot(
                                        dyn_kw=dyn_kw,
                                        map_kw=map_kw,
                                        title_scan_id=title_scan_id
                                        )
        
        fig, ax, span = interactive_1D_window_com_plot(dyn_kw=dyn_kw,
                                                       map_kw=map_kw,
                                                       **kwargs)
        
        # Save internally for reference.
        self._1D_window_com_span = span

        return fig, ax


    def _prepare_1D_window_plot(self,
                                dyn_kw=None,
                                map_kw=None,
                                title_scan_id=True):
        """
        Internal function for preparing 1D windowed plots.

        Parameters
        ----------
        dyn_kw : dict, optional
            Dictionary of keyword arguments passed for dynamic
            plotting. Empty by default, and the values will be filled
            with the relevant instance parameters.
        map_kw : dict, optional
            Dictionary of keyword arguments passed for the map
            plotting. Empty by default, and the values will be filled
            with the relevant instance parameters.
        title_scan_id : bool, optional
            Flag dictating if the plot title will be 
            prepended with the data scan ID. True by 
            default.

        Raises
        ------
        ValueError if integrations for plotting dynamic XRD patterns
        cannot be found or are of the wrong shape.
        """
        
        # Python doesn't play well with mutable default kwargs
        if dyn_kw is None:
            dyn_kw = {}
        if map_kw is None:
            map_kw = {}

        if _check_dict_key(dyn_kw, 'data'):
            dyn_kw['data'] = np.asarray(dyn_kw['data'])
        elif not hasattr(self, 'integrations'):
            err_str = 'Could not find integrations to plot data!'
            raise ValueError(err_str)
        elif self.integrations.ndim != 3:
            err_str = ('Integration data shape is not 3D, '
                       + f'but {self.integrations.ndim}.')
            raise ValueError(err_str)
        else:
            dyn_kw['data'] = self.integrations

        if not _check_dict_key(dyn_kw, 'x_ticks'):
            if hasattr(self, 'tth') and self.tth is not None:
                dyn_kw['x_ticks'] = self.tth
                dyn_kw['x_label'] = ('Scattering Angle, 2θ '
                                     + f'[{self.scattering_units}]')
    
        # Add default map_kw information if not already included
        if not _check_dict_key(map_kw, 'map'):
            map_kw['map'] = self.max_integration_map
            map_kw['title'] = 'Max Integration Intensity'
        if not _check_dict_key(map_kw, 'x_ticks'):
            map_kw['x_ticks'] = np.round(np.linspace(
                *self.map_extent()[:2], # without step
                self.map_shape[1]), 2)
        if not _check_dict_key(map_kw, 'y_ticks'):
            map_kw['y_ticks'] = np.round(np.linspace(
                *self.map_extent()[2:], # without step
                self.map_shape[0]), 2)
        if hasattr(self, 'position_units'):
            if not _check_dict_key(map_kw, 'x_label'):
                map_kw['x_label'] = ('x position '
                                     + f'[{self.position_units}]')
            if not _check_dict_key(map_kw, 'y_label'):
                map_kw['y_label'] = ('y position '
                                     + f'[{self.position_units}]')
        
        if 'title' not in map_kw:
            map_kw['title'] = None
        map_kw['title'] = self._title_with_scan_id(
                            map_kw['title'],
                            default_title='Custom Map',
                            title_scan_id=title_scan_id)
        if 'title' not in dyn_kw:
            dyn_kw['title'] = None
        dyn_kw['title'] = self._title_with_scan_id(
                            dyn_kw['title'],
                            default_title='',
                            title_scan_id=title_scan_id)    
        
        return dyn_kw, map_kw


    @return_plot_wrapper
    def plot_2D_window_sum_map(self,
                               dyn_kw=None,
                               map_kw=None,
                               title_scan_id=True,
                               **kwargs):
        """
        Plot an interactive map of a 2D window sum.

        Draw a window across the maximum 2D image and plot a map of the
        sum within this window.

        Parameters
        ----------
        dyn_kw : dict, optional
            Dictionary of keyword arguments passed for dynamic
            plotting. Empty by default, and the values will be filled
            with the relevant instance parameters.

            Useful Keywords:
                vmin : float, optional
                    Starting maximum value for dynamic colormap.            
                vmax : float, optional
                    Starting minimum value for dynamic colormap.

        map_kw : dict, optional
            Dictionary of keyword arguments passed for the map
            plotting. Empty by default, and the values will be filled
            with the relevant instance parameters.

            Useful Keywords:
                title : str, optional
                    Title of map plot.
                vmin : float, optional
                    Starting minimum value for map colormap.
                vmax : float, optional
                    Starting maximum value for map colormap.

        title_scan_id : bool, optional
            Flag dictating if the plot title will be 
            prepended with the data scan ID. True by 
            default.
        kwargs : dict, optional
            Dictionary of keyword arguments passed to the interactive
            plotting function.
        
        Returns
        -------
        No returns by default. Only returned if return_plot
        keyword argument is True.

        fig : Matplotlib Figure instance
            Figure of plot.
        ax : Matplotlib Axes instance, optional
            Individual axes of plot.
        
        Raises
        ------
        ValueError if images for plotting dynamic XRD patterns
        cannot be found or are of the wrong shape.
        """
        
        dyn_kw, map_kw = self._prepare_2D_window_plot(
                                        dyn_kw=dyn_kw,
                                        map_kw=map_kw,
                                        title_scan_id=title_scan_id
                                        )
        
        fig, ax, rect = interactive_2D_window_sum_plot(dyn_kw=dyn_kw,
                                                       map_kw=map_kw,
                                                       **kwargs)
        # Save internally for reference.
        self._2D_window_sum_rect = rect

        return fig, ax


    @return_plot_wrapper
    def plot_2D_window_tth_com_map(self,
                                   dyn_kw=None,
                                   map_kw=None,
                                   title_scan_id=True,
                                   **kwargs):
        """
        Plot an interactive map of a 2D window scattering angle center
        of mass.

        Draw a window across the maximum 2D image and plot a map of the
        scattering angle center of mass within this window.

        Parameters
        ----------
        dyn_kw : dict, optional
            Dictionary of keyword arguments passed for dynamic
            plotting. Empty by default, and the values will be filled
            with the relevant instance parameters.

            Useful Keywords:
                vmin : float, optional
                    Starting maximum value for dynamic colormap.            
                vmax : float, optional
                    Starting minimum value for dynamic colormap.

        map_kw : dict, optional
            Dictionary of keyword arguments passed for the map
            plotting. Empty by default, and the values will be filled
            with the relevant instance parameters.

            Useful Keywords:
                title : str, optional
                    Title of map plot.
                vmin : float, optional
                    Starting minimum value for map colormap.
                vmax : float, optional
                    Starting maximum value for map colormap.

        title_scan_id : bool, optional
            Flag dictating if the plot title will be 
            prepended with the data scan ID. True by 
            default.
        kwargs : dict, optional
            Dictionary of keyword arguments passed to the interactive
            plotting function.
        
        Returns
        -------
        No returns by default. Only returned if return_plot
        keyword argument is True.

        fig : Matplotlib Figure instance
            Figure of plot.
        ax : Matplotlib Axes instance, optional
            Individual axes of plot.
        
        Raises
        ------
        ValueError if images for plotting dynamic XRD patterns
        cannot be found or are of the wrong shape.
        """
        
        dyn_kw, map_kw = self._prepare_2D_window_plot(
                                        dyn_kw=dyn_kw,
                                        map_kw=map_kw,
                                        title_scan_id=title_scan_id
                                        )
        
        fig, ax, rect = interactive_2D_window_tth_com_plot(
                                                dyn_kw=dyn_kw,
                                                map_kw=map_kw,
                                                **kwargs)
        # Save internally for reference.
        self._2D_window_tth_com_rect = rect

        return fig, ax


    @return_plot_wrapper
    def plot_2D_window_chi_com_map(self,
                                   dyn_kw=None,
                                   map_kw=None,
                                   title_scan_id=True,
                                   **kwargs):
        """
        Plot an interactive map of a 2D window azimuthal angle center
        of mass.

        Draw a window across the maximum 2D image and plot a map of the
        azimuthal angle center of mass within this window.
    

            Parameters
        ----------
        dyn_kw : dict, optional
            Dictionary of keyword arguments passed for dynamic
            plotting. Empty by default, and the values will be filled
            with the relevant instance parameters.

            Useful Keywords:
                vmin : float, optional
                    Starting maximum value for dynamic colormap.            
                vmax : float, optional
                    Starting minimum value for dynamic colormap.

        map_kw : dict, optional
            Dictionary of keyword arguments passed for the map
            plotting. Empty by default, and the values will be filled
            with the relevant instance parameters.

            Useful Keywords:
                title : str, optional
                    Title of map plot.
                vmin : float, optional
                    Starting minimum value for map colormap.
                vmax : float, optional
                    Starting maximum value for map colormap.

        title_scan_id : bool, optional
            Flag dictating if the plot title will be 
            prepended with the data scan ID. True by 
            default.
        kwargs : dict, optional
            Dictionary of keyword arguments passed to the interactive
            plotting function.
        
        Returns
        -------
        No returns by default. Only returned if return_plot
        keyword argument is True.

        fig : Matplotlib Figure instance
            Figure of plot.
        ax : Matplotlib Axes instance, optional
            Individual axes of plot.
        
        Raises
        ------
        ValueError if images for plotting dynamic XRD patterns
        cannot be found or are of the wrong shape.
        """
        
        dyn_kw, map_kw = self._prepare_2D_window_plot(
                                        dyn_kw=dyn_kw,
                                        map_kw=map_kw,
                                        title_scan_id=title_scan_id
                                        )
        
        fig, ax, rect = interactive_2D_window_chi_com_plot(
                                                dyn_kw=dyn_kw,
                                                map_kw=map_kw,
                                                **kwargs)
        # Save internally for reference.
        self._2D_window_chi_com_rect = rect

        return fig, ax


    def _prepare_2D_window_plot(self,
                                dyn_kw=None,
                                map_kw=None,
                                title_scan_id=True):
        """
        Internal function for preparing 2D windowed plots.

        Parameters
        ----------
        dyn_kw : dict, optional
            Dictionary of keyword arguments passed for dynamic
            plotting. Empty by default, and the values will be filled
            with the relevant instance parameters.
        map_kw : dict, optional
            Dictionary of keyword arguments passed for the map
            plotting. Empty by default, and the values will be filled
            with the relevant instance parameters.
        title_scan_id : bool, optional
            Flag dictating if the plot title will be 
            prepended with the data scan ID. True by 
            default.

        Raises
        ------
        ValueError if images for plotting dynamic XRD patterns
        cannot be found or are of the wrong shape.
        """
        
        # Python doesn't play well with mutable default kwargs
        if dyn_kw is None:
            dyn_kw = {}
        if map_kw is None:
            map_kw = {}

        if _check_dict_key(dyn_kw, 'data'):
            dyn_kw['data'] = np.asarray(dyn_kw['data'])
        elif not hasattr(self, 'images'):
            err_str = 'Could not find images to plot data!'
            raise ValueError(err_str)
        elif self.images.ndim != 4:
            err_str = ('Image data shape is not 4D, '
                       + f'but {self.images.ndim}.')
            raise ValueError(err_str)
        else:
            dyn_kw['data'] = self.images

        # Add calibration information
        if not _check_dict_key(dyn_kw, 'tth_arr'):
            dyn_kw['tth_arr'] = self.tth_arr
        if not _check_dict_key(dyn_kw, 'chi_arr'):
            dyn_kw['chi_arr'] = self.chi_arr
    
        # Add default map_kw information if not already included
        if not _check_dict_key(map_kw, 'map'):
            map_kw['map'] = self.max_map
            map_kw['title'] = 'Max Detector Intensity'
        if not _check_dict_key(map_kw, 'x_ticks'):
            map_kw['x_ticks'] = np.round(np.linspace(
                *self.map_extent()[:2], # without step
                self.map_shape[1]), 2)
        if not _check_dict_key(map_kw, 'y_ticks'):
            map_kw['y_ticks'] = np.round(np.linspace(
                *self.map_extent()[2:], # without step
                self.map_shape[0]), 2)
        if hasattr(self, 'position_units'):
            if not _check_dict_key(map_kw, 'x_label'):
                map_kw['x_label'] = ('x position '
                                     + f'[{self.position_units}]')
            if not _check_dict_key(map_kw, 'y_label'):
                map_kw['y_label'] = ('y position '
                                     + f'[{self.position_units}]')
        
        # Update title
        if 'title' not in map_kw:
            map_kw['title'] = None
        map_kw['title'] = self._title_with_scan_id(
                            map_kw['title'],
                            default_title='Custom Map',
                            title_scan_id=title_scan_id)
        if 'title' not in dyn_kw:
            dyn_kw['title'] = None
        dyn_kw['title'] = self._title_with_scan_id(
                            dyn_kw['title'],
                            default_title='',
                            title_scan_id=title_scan_id)        
        
        return dyn_kw, map_kw
    

    @return_plot_wrapper
    def plot_waterfall(self, **kwargs):
        
        """
        Generate waterfall plot integrated along a map axis.

        Sum all integrations along a mapped axis and plot the results
        as a waterfall plot.

        Parameters
        ----------
        axis : int, optional
            Axis for integrating the data. Only 0 and 1 are
            allowed. 0 by default.
        integration_method : str, optional
            Method for integrating data. Accepts 'sum' or 'max'
            along the axis of integration. 'max' by default.
        v_offset : float, optional
            Relative vertical offset value of each integration.
            0 will have no offset, 1 will offset each integration
            by the maximum integration range preventing any overlap.
            0.25 by default.
        tth : iterable, optional
            Two theta scattering angle of the integration data.
            Must have the same length as the integration data.
            None by default and will look for an internal tth attribute,
            or use a simple range if unavaible.
        units : str, optional
            Units of two theta scattering angle. None by default
            and will use an internal scattering_angle attribute if 
            available.
        cmap : str, optional
            Colormap for distinguishing integrations. None by default
            and all integrations will be black.
        fig : Matplotlib Figure instance, optional
            Figure to be used for plotting. Must be given
            ax. None by default, generating an new plot.
        ax : Matplotlib Axes instance, optional
            Axes to be used for plotting. Must be given with
            fig. None by default, generating a new plot.
        title_scan_id : bool, optional
            Flag dictating if the plot title will be 
            prepended with the data scan ID. True by 
            default.
        kwargs : dict, optional
            Dictionary of keyword arguments passed to
            the internal Matplotlib.pyplot.plot function.

        Returns
        -------
        fig : Matplotlib Figure instance
            Figure of plot.
        ax : Matplotlib Axes instance, optional
            Individual axes of plot.

        Raises
        ------
        AttributeError for data without integrations.
        ValueError if the integration method is not
            acceptable.
        """

        if 'axis' in kwargs:
            axis = kwargs.pop('axis')
        else:
            axis = 0 # default
        
        if axis == 0:
            axis_text = 'Horizontal'
        elif axis == 1:
            axis_text = 'Vertical'

        return self._plot_waterfall(axis=axis,
                                    axis_text=axis_text,
                                    **kwargs)