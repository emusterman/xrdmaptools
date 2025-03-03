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
    _check_dict_key
)
from xrdmaptools.io.hdf_utils import (
    check_attr_overwrite,
    overwrite_attr,
    get_large_map_slices
)
from xrdmaptools.io.db_io import load_data
from xrdmaptools.reflections.spot_blob_indexing import _initial_spot_analysis
from xrdmaptools.reflections.SpotModels import GaussianFunctions
from xrdmaptools.reflections.spot_blob_search import (
    find_blobs,
    find_blobs_spots,
    find_spot_stats,
    make_stat_df,
    remake_spot_list,
    fit_spots
    )
from xrdmaptools.plot.interactive import (
    interactive_2D_plot,
    interactive_1D_plot
    )
from xrdmaptools.plot.general import (
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
        if self.scan_input is not None:
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
        

    @ classmethod
    def from_db(cls,
                scan_id=-1,
                broker='manual',
                wd=None,
                filename=None,
                poni_file=None,
                data_keys=None,
                save_hdf=True,
                dask_enabled=False,
                repair_method='fill'):
        
        if wd is None:
            wd = os.getcwd()
        else:
            if not os.path.exists(wd):
                err_str = f'Cannot find directory {wd}'
                raise OSError(err_str)
    
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
                            returns=['data_keys',
                                        'xrd_dets'],
                            repair_method=repair_method)

        xrd_data = [data_dict[f'{xrd_det}_image']
                    for xrd_det in xrd_dets]

        # Make position dictionary
        pos_dict = {key:value for key, value in data_dict.items()
                    if key in pos_keys}

        # Make scaler dictionary
        sclr_dict = {key:value for key, value in data_dict.items()
                     if key in sclr_keys}

        if 'null_map' in data_dict.keys():
            null_map = data_dict['null_map']
        else:
            null_map = None

        if len(xrd_data) > 1:
            filenames = [f'scan{scan_md["scan_id"]}_{det}_xrd.h5'
                         for det in xrd_dets]
        else:
            filenames = [filename]

        extra_md = {}
        for key in scan_md.keys():
            if key not in ['scan_id',
                           'beamline',
                           'energy',
                           'dwell',
                           'theta',
                           'start_time',
                           'scan_input']:
                extra_md[key] = scan_md[key]
        
        xrdmaps = []
        for i, xrd_data_i in enumerate(xrd_data):
            xrdmap = cls(scan_id=scan_md['scan_id'],
                         wd=wd,
                         filename=filenames[i],
                         image_data=xrd_data_i,
                         null_map=null_map,
                         energy=scan_md['energy'],
                         dwell=scan_md['dwell'],
                         theta=scan_md['theta'],
                         poni_file=poni_file,
                         sclr_dict=sclr_dict,
                         pos_dict=pos_dict,
                         beamline=scan_md['beamline_id'],
                         facility='NSLS-II',
                         scan_input=scan_md['scan_input'],
                         time_stamp=scan_md['time_str'],
                         extra_metadata=extra_md,
                         save_hdf=save_hdf,
                         dask_enabled=dask_enabled,
                         )
            
            xrdmaps.append(xrdmap)

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
        shapes = []

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
                # map_shape=None,         no energy attribute
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
                scan_input=self.scan_input,
                time_stamp=self.time_stamp,
                extra_metadata=self.extra_metadata,
                save_hdf=True,
                # Keeping everything lazy causes some inconsistencies
                #dask_enabled=True 
            )
        
        print('Finished fracturing maps.')

    #################################
    ### Modified Parent Functions ###
    #################################

    # Re-writing save functions which 
    # will be affected by swapped axes
    def _check_swapped_axes(func):
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

    
    def save_current_hdf(self):
        super().save_current_hdf() # no inputs!

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

    
    
    ##################
    ### Properties ###
    ##################

    # Only inherited properties thus far...        

    ##############################
    ### Calibrating Map Images ###
    ##############################
        

    def integrate1d_map(self,
                        tth_num=None,
                        tth_resolution=None,
                        unit='2th_deg',
                        mask=None,
                        return_values=False,
                        **kwargs):
        
        if not hasattr(self, 'ai'):
            err_str = ('Images cannot be calibrated without '
                       + 'any calibration files!')
            raise RuntimeError(err_str)
        
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

        # Set up empty array to fill
        integrated_map1d = np.empty((*self.map_shape, 
                                     tth_num), 
                                     dtype=(self.dtype))

        if mask is not None:
            mask = np.asarray(mask)
            if (mask.shape == self.images.shape
                or mask.shape == self.image_shape):
                pass
            else:
                err_str = (f'mask shape {mask.shape} does not match '
                           + f'images {self.image.shape}')
                raise ValueError(err_str)
        
        # Fill array!
        print('Integrating images to 1D...')
        # TODO: Parallelize this
        # for i, pixel in tqdm(enumerate(self.images.reshape(
        #                                self.num_images,
        #                                *self.image_shape)),
        #                                total=self.num_images):
        
        for indices in tqdm(self.indices):
            
            image = self.images[indices].copy()

            if mask is not None:
                if mask.shape == self.image_shape:
                    image *= mask
                else:
                    image *= mask[indices]
        
            tth, I = self.integrate1d_image(image=image,
                                            tth_num=tth_num,
                                            unit=unit,
                                            **kwargs)            

            integrated_map1d[indices] = I

        if return_values:
            return (integrated_map1d,
                    tth,
                    [np.min(self.tth), np.max(self.tth)],
                    tth_resolution)

        # Reshape into (map_x, map_y, tth)
        # Does not explicitly match the same shape as 2d integration
        # integrated_map1d = integrated_map1d.reshape(
        #                         *self.map_shape, tth_num)
        self.integrations = integrated_map1d
        
        # Save a few potentially useful parameters
        self.tth = tth
        self.extent = [np.min(self.tth), np.max(self.tth)]
        self.tth_resolution = tth_resolution

        # Save integrations to hdf
        if self.hdf_path is not None:
            print('Compressing and writing integrations to disk...')
            self.save_integrations()
            print('done!')
            self.save_reciprocal_positions()
        

    # Briefly doubles memory. No Dask support
    # TODO: change corrections, reset projections, update map_title, etc. from XRDData
    def integrate2d_map(self,
                        tth_num=None,
                        tth_resolution=None,
                        chi_num=None,
                        chi_resolution=None,
                        unit='2th_deg',
                        **kwargs):
        
        if (not hasattr(self, 'ai') or self.ai is None):
            err_str = ('Images cannot be calibrated without '
                       + 'any calibration files!')
            raise RuntimeError(err_str)
        
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

        # Set up empty array to fill
        integrated_map2d = np.empty((self.num_images,
                                     chi_num, tth_num), 
                                     dtype=(self.dtype))
        
        # Fill array!
        print('Integrating images to 2D...')
        # TODO: Parallelize this
        for i, pixel in tqdm(enumerate(self.images.reshape(
                                       self.num_images,
                                       *self.image_shape)),
                                       total=self.num_images):
        
            I, tth, chi = self.integrate2d_image(image=pixel,
                                                 tth_num=tth_num,
                                                 unit=unit,
                                                 **kwargs)            

            integrated_map2d[i] = I

        # Reshape into (map_x, map_y, chi, tth)
        integrated_map2d = integrated_map2d.reshape(
                                *self.map_shape, chi_num, tth_num)
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
        
    
    #######################
    ### Position Arrays ###
    #######################

    def set_positions(self,
                      pos_dict,
                      position_units=None,
                      check_init_sets=False):

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

    
    def map_extent(self, map_x=None, map_y=None):

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

        # Determine fast scanning direction for map extent
        if (np.mean(np.diff(map_x, axis=1))
            > np.mean(np.diff(map_x, axis=0))):
            # Fast x-axis. Standard orientation.
            #print('Fast x-axis!')
            map_extent = [
                np.mean(map_x[:, 0]),
                np.mean(map_x[:, -1]),
                np.mean(map_y[-1]), # reversed
                np.mean(map_y[0]) # reversed
            ] # [min_x, max_x, max_y, min_y] reversed y for matplotlib
        else: # Fast y-axis. Consider swapping axes???
            #print('Fast y-axis!')
            map_extent = [
                np.mean(map_y[:, 0]), # reversed
                np.mean(map_y[:, -1]), # reversed
                np.mean(map_x[-1]),
                np.mean(map_x[0])
            ] # [min_y, max_y, max_x, min_x] reversed x for matplotlib
        
        return map_extent


    # Convenience function for loading scalers and positions
    # from standard map_parameters text file
    def load_map_parameters(self,
                            filename,
                            wd=None,
                            position_units=None):  
        
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
                  # only_images=False,
                  update_flag=True,
                  ):
        # This will break if images are loaded with dask.
        # _temp_images will be of the wrong shape...
        # could be called before _temp_images dataset is instantiated??
        # exclude_images included to swap axes upon instantiation
        # Only save flag
        

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
        if (hasattr(self, 'blob_masks')
            and self.blob_masks is not None):
            self.blob_masks = self.blob_masks.swapaxes(0, 1)
        if (hasattr(self, 'null_map')
            and self.null_map is not None):
            self.null_map = self.null_map.swapaxes(0, 1)
        if (hasattr(self, 'scaler_map')
            and self.scaler_map is not None):
            self.scaler_map = self.scaler_map.swapaxes(0, 1)

        # Modify other attributes as needed
        if hasattr(self, 'pos_dict'):
            for key in list(self.pos_dict.keys()):
                self.pos_dict[key] = self.pos_dict[key].swapaxes(0, 1)  
        if hasattr(self, 'sclr_dict'):
            for key in list(self.sclr_dict.keys()):
                self.sclr_dict[key] = self.sclr_dict[key].swapaxes(0, 1)
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


    def interpolate_positions(self,
                              scan_input=None,
                              check_init_sets=False):

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
        else:
            warn_str = ('WARNING: No pos_dict found. '
                        + 'Generating from interpolated positions.')
            print(warn_str)
            self.pos_dict = {
                'interp_x' : interp_x,
                'interp_y' : interp_y
            }
        
        # Write to hdf file
        self.save_sclr_pos('positions',
                            self.pos_dict,
                            self.position_units,
                            check_init_sets=check_init_sets)
    
    
    #######################
    ### Blobs and Spots ###
    #######################

    def find_blobs(self,
                   threshold_method='minimum',
                   multiplier=5,
                   size=3,
                   expansion=10,
                   override_rescale=True):
    
        # Cleanup images as necessary
        self._dask_2_numpy()
        if not override_rescale and np.max(self.images) != 100:
            print('Rescaling images to max of 100 and min around 0.')
            self.rescale_images(arr_min=0, upper=100, lower=0)

            #         sclr_dict[key] = value.reshape(self.map_shape)
        # Search each image for significant spots
        blob_mask_list = find_blobs(
                            self.images,
                            mask=self.mask,
                            threshold_method=threshold_method,
                            multiplier=multiplier,
                            size=size,
                            expansion=expansion)
        
        self.blob_masks = np.asarray(
                                blob_mask_list).reshape(self.shape)

        # Save blob_masks to hdf
        self.save_images(images='blob_masks',
                         title='_blob_masks',
                         units='bool',
                         extra_attrs={
                            'threshold_method' : threshold_method,
                            'size' : size,
                            'multiplier' : multiplier,
                            'expansion' : expansion})
        

    def find_spots(self,
                   threshold_method='minimum',
                   multiplier=5,
                   size=3,
                   expansion=10,
                   min_distance=3,
                   radius=10,
                   override_rescale=True):
        
        if (hasattr(self, 'blob_masks')
            and self.blob_masks is not None):
            warn_str = ('WARNING: XRDMap already has blob_masks '
                        + 'attribute. This will be overwritten with '
                        + 'new parameters.')
            print(warn_str)
            
        # Cleanup images as necessary
        self._dask_2_numpy()
        if not override_rescale and np.max(self.images) != 100:
            print('Rescaling images to max of 100 and min around 0.')
            self.rescale_images(arr_min=0, upper=100, lower=0)
        
        # Search each image for significant blobs and spots
        spot_list, blob_mask_list = find_blobs_spots(
                    self.images,
                    mask=self.mask,
                    threshold_method=threshold_method,
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
                            'threshold_method' : threshold_method,
                            'size' : size,
                            'multiplier' : multiplier,
                            'expansion' : expansion})
        
    
    def recharacterize_spots(self,
                             radius=10):

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

    
    def fit_spots(self, SpotModel, max_dist=0.5, sigma=1):

        # Find spots in self or from hdf
        if not hasattr(self, 'spots'):
            print('No reflection spots found...')
            if self.hdf_path is not None:
                keep_hdf = self.hdf is not None

                if 'reflections' in self.hdf[self._hdf_type].keys():
                    print('Loading reflection spots from hdf...',
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


    def initial_spot_analysis(self, SpotModel=None):

        if SpotModel is None and hasattr(self, 'spot_model'):
            SpotModel = self.spot_model

        # Initial spot analysis...
        _initial_spot_analysis(self, SpotModel=SpotModel)

        # Save spots to hdf
        self.save_spots()


    def trim_spots(self,
                   remove_less=0.01,
                   metric='height',
                   save_spots=False):
        if not hasattr(self, 'spots') or self.spots is None:
            err_str = 'Cannot trim spots if XRDMap has not no spots.'
            raise ValueError(err_str)

        metric = str(metric).lower()
        if any([x[:3] == 'fit' for x in self.spots.iloc[0].keys()]):
            if metric in ['height', 'amp']:
                significance = (self.spots['fit_amp']
                                - self.spots['fit_offset'])
            elif metric in ['intensity',
                            'int',
                            'breadth',
                            'integrated',
                            'volume']:
                # this should account for offset too            
                significance = self.spots['fit_integrated'] 
            else:
                raise ValueError('Unknown metric specification.')
        else:
            if metric in ['height', 'amp']:
                significance = self.spots['guess_height']
            elif metric in ['intensity',
                            'int',
                            'breadth',
                            'integrated',
                            'volume']:
                significance = self.spots['guess_int']
            else:
                raise ValueError('Unknown metric specification.')

        # Find relative indices where conditional is true
        mask = np.where(significance.values < remove_less)[0]

        # Convert relative indices into dataframe index
        drop_indices = self.spots.iloc[mask].index.values # awful call

        # Drop indices
        self.spots.drop(index=drop_indices, inplace=True)
        ostr = (f'Trimmed {len(drop_indices)} spots less '
                + f'than {remove_less} significance.')
        print(ostr)

        if save_spots:
            self.save_spots()

    
    def _remove_spot_vals(self, drop_keys=[], drop_tags=[]):
        for key in list(self.spots.keys()):
            for tag in drop_tags:
                if tag in key:
                    drop_keys.append(key)
        print(f'Removing spot values for {drop_keys}')
        self.spots.drop(drop_keys, axis=1, inplace=True)

    def remove_spot_guesses(self):
        self._remove_spot_vals(drop_tags=['guess'])
    
    def remove_spot_fits(self):
        self._remove_spot_vals(drop_tags=['fit'])
    

    def pixel_spots(self, map_indices, copied=True):

        pixel_spots = self.spots[
                        (self.spots['map_x'] == map_indices[1])
                        & (self.spots['map_y'] == map_indices[0])]
        
        # Copies to protect orginal spots from changes
        if copied:
            pixel_spots = pixel_spots.copy()
        
        return pixel_spots

    @_check_swapped_axes
    @XRDBaseScan._protect_hdf(pandas=True)
    def save_spots(self, extra_attrs=None):
        print('Saving spots to hdf...', end='', flush=True)
        hdf_str = f'{self._hdf_type}/reflections/spots'
        self.spots.to_hdf(
                        self.hdf_path,
                        key=hdf_str,
                        format='table')

        if extra_attrs is not None:
            self.open_hdf()
            for key, value in extra_attrs.items():
                overwrite_attr(self.hdf[hdf_str].attrs, key, value)
                #self.hdf[hdf_str].attrs[key] = value        
        print('done!')

    ############################
    ### Vectorizing Map Data ###
    ############################

    @XRDBaseScan._protect_hdf()
    def vectorize_map_data(self,
                           image_data_key='recent',
                           keep_images=False,
                           rewrite_data=False):

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
        self.save_map_vectorization(rewrite_data=rewrite_data)
        
        # Cleaning up XRDMap state
        if not keep_images and remove_images_after:
            self.dump_images()
        if not keep_images and remove_blob_masks_after:
            del self.blob_masks
            self.blob_masks = None       
    
    
    @XRDBaseScan._protect_hdf()
    def save_map_vectorization(self,
                               vector_map=None,
                               edges=None,
                               rewrite_data=False):

        # Allows for more customizability with other functions
        hdf = getattr(self, 'hdf')

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
    
        self._save_map_vectorization(hdf,
                                     vector_map=vector_map,
                                     edges=edges,
                                     rewrite_data=rewrite_data)
        
        # Remove secondary reference
        del hdf

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

        # Look for path if no information is provided
        if (wd is None
            and xrf_name is None
            and self.xrf_path is not None):
            xrf_path = self.xrf_path
        else:
            if wd is None:
                wd = self.wd

            # Try default name for SRX
            if xrf_name is None:
                xrf_name =  f'scan2D_{self.scan_id}_xs_sum8ch'
            
            xrf_path = pathify(wd, xrf_name, '.h5')        

        if not os.path.exists(xrf_path): # redundant!
            raise FileNotFoundError(f"{xrf_path} does not exist.")
        else:
            self.xrf_path = xrf_path

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
    

    def plot_map(self,
                 map_values,
                 map_extent=None,
                 position_units=None,
                 title=None,
                 fig=None,
                 ax=None,
                 title_scan_id=True,
                 return_plot=False,
                 **kwargs):
        
        map_values = np.asarray(map_values)
        
        if (hasattr(self, 'pos_dict')
            and (map_values.shape
                   != list(self.pos_dict.values())[0].shape)):
            err_str = (f'Map input shape {map_values.shape} does '
                       + f'not match instance shape '
                       + f'of {list(self.pos_dict.values())[0].shape}')
            raise ValueError(err_str)
        
        if map_extent is None:
            map_extent = self.map_extent()
        if position_units is None:
            position_units = self.position_units

        title = self._title_with_scan_id(
                        title,
                        default_title='Custom Map',
                        title_scan_id=title_scan_id)
        
        fig, ax = plot_map(map_values,
                           map_extent=map_extent,
                           position_units=position_units,
                           title=title,
                           fig=fig,
                           ax=ax,
                           **kwargs)
        
        if return_plot:
            return fig, ax
        else:
            fig.show()


    # Interactive plots do not currently accept fig, ax inputs
    def plot_interactive_map(self,
                             dyn_kw=None,
                             map_kw=None,
                             title_scan_id=True,
                             return_plot=False,
                             **kwargs):
        
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
            map_kw['map'] = self.sum_map
            map_kw['title'] = 'Summed Intensity'
        if (hasattr(self, 'map')
            and hasattr(self, 'map_extent')):
            if not _check_dict_key(map_kw, 'x_ticks'):
                map_kw['x_ticks'] = np.round(np.linspace(
                    *self.map_extent()[:2],
                    self.map_shape[1]), 2)
            if not _check_dict_key(map_kw, 'y_ticks'):
                map_kw['y_ticks'] = np.round(np.linspace(
                    *self.map_extent()[2:],
                    self.map_shape[0]), 2)
        if hasattr(self, 'positions_units'):
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

        fig, ax = interactive_2D_plot(dyn_kw,
                                      map_kw,
                                      **kwargs)

        if return_plot:
            return fig, ax
        else:
            fig.show()


    def plot_interactive_integration_map(self,
                                         dyn_kw=None,
                                         map_kw=None,
                                         title_scan_id=True,
                                         return_plot=False,
                                         **kwargs):
        
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
            err_str = ('Integration data shape is not 4D, '
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
            map_kw['map'] = self.sum_map
            map_kw['title'] = 'Summed Intensity'
        if (hasattr(self, 'map')
            and hasattr(self, 'map_extent')):
            if not _check_dict_key(map_kw, 'x_ticks'):
                map_kw['x_ticks'] = np.round(np.linspace(
                    *self.map_extent()[:2],
                    self.map_shape[1]), 2)
            if not _check_dict_key(map_kw, 'y_ticks'):
                map_kw['y_ticks'] = np.round(np.linspace(
                    *self.map_extent()[2:],
                    self.map_shape[0]), 2)
        if hasattr(self, 'positions_units'):
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
    
        fig, ax = interactive_1D_plot(dyn_kw,
                                      map_kw,
                                      **kwargs)
        if return_plot:
            return fig, ax
        else:
            fig.show()
