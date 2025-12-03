
import os
import h5py
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
import json
import time as ttime
import matplotlib.pyplot as plt
from plotly import graph_objects as go
import dask.array as da

from xrdmaptools.XRDBaseScan import XRDBaseScan
from xrdmaptools.utilities.math import (
    energy_2_wavelength,
    wavelength_2_energy
)
from xrdmaptools.utilities.utilities import (
    generate_intensity_mask,
    copy_docstring
)
from xrdmaptools.crystal.rsm import map_2_grid
from xrdmaptools.geometry.geometry import (
    get_q_vect,
    q_2_polar,
    QMask
)
from xrdmaptools.io.hdf_utils import (
    check_attr_overwrite,
    overwrite_attr
)
from xrdmaptools.io.db_io import (
    get_scantype,
    load_step_rc_data,
    load_extended_energy_rc_data,
    load_flying_angle_rc_data
)
from xrdmaptools.reflections.spot_blob_search import (
    find_blobs
)
from xrdmaptools.reflections.spot_blob_search_3D import (
    rsm_blob_search,
    rsm_spot_search
)
from xrdmaptools.reflections.spot_blob_indexing_3D import (
    phase_index_best_grain,
    phase_index_all_grains
)
from xrdmaptools.crystal.crystal import LatticeParameters
from xrdmaptools.crystal.strain import get_strain_orientation
from xrdmaptools.plot.general import return_plot_wrapper
from xrdmaptools.plot.image_stack import base_slider_plot
from xrdmaptools.plot.volume import (
    plot_3D_scatter,
    plot_3D_isosurfaces
)
from xrdmaptools.plot.orientation import plot_3D_indexing


class XRDRockingCurve(XRDBaseScan):
    """
    Class for analyzing and processing XRD data acquired along
    either and energy/wavelength or angle rocking axis.

    Parameters
    ----------
    image_data : 3D or 4D Numpy array, Dask array, list, h5py dataset, optional
        Image data that can be fully loaded as a 4D array
        XRDRockingCurve axes (rocking_axis, 1, image_y, image_x). The
        extra axis will be added if data is provided as 3D array.
    map_shape : iterable, optional
        Shape of first two axes in image_data as (rocking_axis, 1).
    image_shape : iterable, optional
        Shape of last two axes in image_data (image_y, image_x).
    sclr_dict : dict, optional
        Dictionary of 2D numpy arrays or 1D iterables that matching the
        rocking_axis shape with scaler intensities used for intensity
        normalization.
    rocking_axis : {'energy', 'angle'}, optional
        String indicating which axis was used as the rocking axis to
        scan in reciprocal space. Will accept variations of 'energy',
        'wavelength', 'angle' and 'theta', but stored internally as
        'energy', or 'angle'. Defaults to 'energy'.
    xrdbasekwargs : dict, optional
        Dictionary of all other kwargs for parent XRDBaseScan class.
    """

    # Class variables
    _hdf_type = 'rsm'

    def __init__(self,
                 image_data=None,
                 map_shape=None,
                 image_shape=None,
                 sclr_dict=None,
                 rocking_axis=None,
                 dask_enabled=False,
                 **xrdbasekwargs
                 ):

        if (image_data is None
            and (map_shape is None
                 or image_shape is None)):
            err_str = ('Must specify image_data, '
                       + 'or image and map shapes.')
            raise ValueError(err_str)

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
        
        # Check shape
        if image_data is not None:
            data_ndim = image_data.ndim
            data_shape = image_data.shape 

            if data_ndim == 3: # Most likely
                if np.any(data_shape == 1):
                    err_str = (f'image_data shape is {data_shape}. '
                               + 'For 3D data, no axis should equal 1.')
                    raise ValueError(err_str)
            # This will work with standard hdf formatting
            elif data_ndim == 4: 
                if np.all(data_shape[:2] == 1):
                    err_str = (f'image_data shape is {data_shape}. '
                               + 'For 4D data, first two axes cannot '
                               + 'both equal 1.')
                    raise ValueError(err_str)
                elif (data_shape[0] == 1 or data_shape[1] != 1):
                    image_data = image_data.swapaxes(0, 1)
                elif (data_shape[0] != 1 or data_shape[1] == 1):
                    pass
                else:
                    err_str = (f'image_data shape is {data_shape}. '
                               + 'For 4D data, one of the first two '
                               + 'axes must equal 1.')
                    raise ValueError(err_str)
            else:
                err_str = (f'Unable to parse image_data with '
                           + f'{data_ndim} dimensions with '
                           + f'shape {data_shape}.')
                raise ValueError(err_str)

            # Define num_images early for energy and wavelength setters
            self.map_shape = image_data.shape[:2]
        else:
            self.map_shape = map_shape
        
        # Parse sclr_dict into useble format
        if sclr_dict is not None and isinstance(sclr_dict, dict):
            for key, value in sclr_dict.items():
                sclr_dict[key] = np.asarray(value).reshape(
                                                    self.map_shape)

        XRDBaseScan.__init__(
            self,
            image_data=image_data,
            image_shape=image_shape,
            map_shape=map_shape,
            sclr_dict=sclr_dict,
            dask_enabled=dask_enabled,
            map_labels=['rocking_ind',
                           'null_ind'],
            **xrdbasekwargs
            )
        
        # Null map may need to be reshaped
        if hasattr(self, 'null_map') and self.null_map is not None:
            if self.null_map.shape != self.map_shape:
                self.null_map = self.null_map.reshape(self.map_shape)        

        # Check for changes. Wavelength will depend on energy
        if (np.all(~np.isnan(self.energy))
            and np.all(~np.isnan(self.theta))):
            # If all the same raise error
            if all([
                all(np.round(i, 4) == np.round(self.energy[0], 4)
                    for i in self.energy),
                all(np.round(i, 4) == np.round(self.theta[0], 4)
                    for i in self.theta)
            ]):
                err_str = ('Rocking curves must be constructed '
                           + 'by varying energy/wavelength or '
                           + 'theta. Given values are constant.')
                raise ValueError(err_str)
        
        # Find rocking axis
        self._set_rocking_axis(rocking_axis=rocking_axis)
        
        # Enable features
        # This enables theta usage, but there are no offset
        # options to re-zero stage rotation
        if (not self.use_stage_rotation
            and self.rocking_axis == 'angle'):
            self.use_stage_rotation = True

        @XRDBaseScan._protect_hdf()
        def save_extra_attrs(self): # Not sure if this needs self...
            attrs = self.hdf[self._hdf_type].attrs
            overwrite_attr(attrs, 'energy', self.energy)
            overwrite_attr(attrs, 'wavelength', self.wavelength)
            overwrite_attr(attrs, 'theta', self.theta)
            overwrite_attr(attrs, 'rocking_axis', self.rocking_axis)
        save_extra_attrs(self)


    # Overwrite parent function
    @copy_docstring(XRDBaseScan.__str__)
    def __str__(self):
        ostr = (f'{self._hdf_type}:  scan_id={self.scan_id}, '
                + f'energy_range={min(self.energy):.3f}'
                + f'-{max(self.energy):.3f}, '
                + f'shape={self.images.shape}')
        return ostr
    

    # Modify parent function
    @copy_docstring(XRDBaseScan.__repr__)
    def __repr__(self):
        lines = XRDBaseScan.__repr__(self).splitlines(True)
        lines[4] = (f'\tEnergy Range:\t\t{min(self.energy):.3f}'
                    + f'-{max(self.energy):.3f} keV\n')
        ostr = ''.join(lines)
        return ostr

    #########################
    ### Utility Functions ###
    #########################

    def _parse_subscriptable_value(self, value, name):
        """

        """

        # Fill placeholders if value is nothing
        if (np.any(np.array(value) is None)
            or np.any(np.isnan(value))):
            setattr(self, f'_{name}', np.array([np.nan,]
                                               * self.num_images))
        # Proceed if object is subscriptable and not str or dict
        elif (hasattr(value, '__len__')
              and hasattr(value, '__getitem__')
              and not isinstance(value, (str, dict))):
            if len(value) == self.num_images:
                setattr(self, f'_{name}', np.asarray(value))
            # Assume single value is constant
            elif len(value) == 1:
                setattr(self, f'_{name}', np.array([value[0],]
                                                   * self.num_images))
            else:
                err_str = (f'{name} must have length '
                           + 'equal to number of images.')
                raise ValueError(err_str)
        # Assume single value is constant
        elif isinstance(value, (int, float)):
            setattr(self, f'_{name}', np.array([value,]
                                               * self.num_images))
        else:
            err_str = (f'Unable to handle {name} input. '
                       + 'Provide subscriptable or value.')
            raise TypeError(err_str)

        
    def _set_rocking_axis(self, rocking_axis=None):
        """

        """

        # Find rocking axis
        if rocking_axis is not None:
            if rocking_axis.lower() in ['energy', 'wavelength']:
                self.rocking_axis = 'energy'
            elif rocking_axis.lower() in ['angle', 'theta']:
                self.rocking_axis = 'angle'
            else:
                warn_str = (f'Rocking axis ({rocking_axis}) is not '
                            + 'supported. Attempting to find '
                            + 'automatically.')
                print(warn_str)
                # kick it back out and find automatically
                rocking_axis = None 
            
        if rocking_axis is None:
            min_en = np.min(self.energy)
            max_en = np.max(self.energy)
            min_ang = np.min(self.theta)
            max_ang = np.max(self.theta)

            # Convert to eV
            if max_en < 1000:
                max_en *= 1000
                min_en *= 1000

            mov_en = max_en - min_en > 5
            mov_ang = max_ang - min_ang > 0.05

            if mov_en and not mov_ang:
                self.rocking_axis = 'energy'
            elif mov_ang and not mov_en:
                self.rocking_axis = 'angle'
            elif mov_en and mov_ang:
                err_str = ('Ambiguous rocking axis. '
                            + 'Energy varies by more than 5 eV and '
                            + 'theta varies by more than 50 mdeg.')
                raise RuntimeError(err_str)
            else:
                err_str = ('Ambiguous rocking axis. '
                            + 'Energy varies by less than 5 eV and '
                            + 'theta varies by less than 50 mdeg.')
                raise RuntimeError(err_str)

        # Sanity check; should not be triggered
        if (not hasattr(self, 'rocking_axis')
            or self.rocking_axis is None):
            err_str = ('Something seriously failed when setting the '
                       + 'rocking axis.')
            raise RuntimeError(err_str) 


    #############################
    ### Re-Written Properties ###
    #############################

    # A lot of parsing inputs
    @XRDBaseScan.energy.setter
    def energy(self, energy):
        self._parse_subscriptable_value(energy, 'energy')
        self._wavelength = energy_2_wavelength(self._energy)

        @XRDBaseScan._protect_hdf()
        def save_attrs(self): # Not sure if this needs self...
            attrs = self.hdf[self._hdf_type].attrs
            overwrite_attr(attrs, 'energy', self.energy)
            overwrite_attr(attrs, 'wavelength', self.wavelength)
        save_attrs(self)


    # A lot of parsing inputs
    @XRDBaseScan.wavelength.setter
    def wavelength(self, wavelength):
        self._parse_subscriptable_value(wavelength, 'wavelength')
        self._energy = wavelength_2_energy(self._wavelength)

        @XRDBaseScan._protect_hdf()
        def save_attrs(self): # Not sure if this needs self...
            attrs = self.hdf[self._hdf_type].attrs
            overwrite_attr(attrs, 'energy', self.energy)
            overwrite_attr(attrs, 'wavelength', self.wavelength)
        save_attrs(self)

    
    @XRDBaseScan.theta.setter
    def theta(self, theta):
        # No theta input will be assumed as zero...
        if (np.any(np.array(theta) is None)
            or np.any(np.isnan(theta))):
            warn_str = ('WARNING: No theta value provided. '
                        + 'Assuming 0 deg.')
            print(warn_str)
            theta = 0
        self._parse_subscriptable_value(theta, 'theta')
            
        @XRDBaseScan._protect_hdf()
        def save_attrs(self): # Not sure if this needs self...
            overwrite_attr(self.hdf[self._hdf_type].attrs,
                           'theta',
                           self.theta)
        save_attrs(self)

    # Re-written as generator to save on memory for large rocking curves
    # Full q-vector, not just magnitude
    @XRDBaseScan.q_arr.getter
    def q_arr(self):
        """

        """
        if not hasattr(self, 'ai'):
            err_str = ('Cannot calculate q-space without calibration.')
            raise RuntimeError(err_str)
        else:
            for i in range(self.num_images):
                wavelength = self.wavelength[i]
                if self.use_stage_rotation:
                    theta = self.theta[i]
                else:
                    theta = None # no rotation!
                
                yield get_q_vect(
                            self.tth_arr,
                            self.chi_arr,
                            wavelength=wavelength,
                            stage_rotation=theta,
                            degrees=self.polar_units == 'deg',
                            rotation_axis='y') # hard-coded for srx
    

    ##########################
    ### Re-Written Methods ###
    ##########################

    @copy_docstring(XRDBaseScan.set_calibration)
    def set_calibration(self, *args, **kwargs):
        super().set_calibration(*args,
                                energy=self.energy[0],
                                **kwargs)    
    

    @copy_docstring(XRDBaseScan.save_current_hdf)
    def save_current_hdf(self, verbose=False):
        super().save_current_hdf(verbose=verbose) # no inputs!

        # Vectorized_data
        if ((hasattr(self, 'vectors')
             and self.vectors is not None)
            and (hasattr(self, 'edges')
             and self.edges is not None)):
            self.save_vectors(rewrite_data=True)

        # Save spots
        if (hasattr(self, 'spots_3D')
            and self.spots_3D is not None):
            self.save_3D_spots()

    
    # Override of XRDData absorption correction
    # def apply_absorption_correction():
    #     raise NotImplementedError()
     

    #####################
    ### I/O Functions ###
    #####################    

    @classmethod
    def from_db(cls,
                scan_id=-1,
                broker='manual',
                wd=None,
                filename=None,
                save_hdf=True,
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
        """
        
        if wd is None:
            wd = os.getcwd()
        else:
            if not os.path.exists(wd):
                err_str = f'Cannot find directory {wd}'
                raise OSError(err_str)
        
        if isinstance(scan_id, str):
            scantype = 'EXTENDED_ENERGY_RC'
            rocking_axis = 'energy'

            (data_dict,
             scan_md,
             xrd_dets) = load_extended_energy_rc_data(
                            start_id=scan_id[:6],
                            end_id=scan_id[-6:],
                            returns=['xrd_dets']
                            )

            filename_id = (f"{scan_md['scan_id'][0]}"
                           + f"-{scan_md['scan_id'][-1]}")

            for key in data_dict.keys():
                if key in [f'{xrd_det}_image'
                           for xrd_det in xrd_dets]:
                    data_dict[key] = np.vstack(data_dict[key])
                    data_shape = data_dict[key].shape
                    data_dict[key] = data_dict[key].reshape(
                                                    data_shape[0],
                                                    1,
                                                    *data_shape[-2:])
            
        else:
            # Get scantype information from check...
            if broker == 'manual':
                temp_broker = 'tiled'
            else:
                temp_broker = broker
            scantype = get_scantype(scan_id,
                                    broker=temp_broker)

            if scantype == 'ENERGY_RC':
                load_func = load_step_rc_data
                rocking_axis = 'energy'
            elif scantype == 'ANGLE_RC':
                load_func = load_step_rc_data
                rocking_axis = 'angle'
            elif scantype == 'XRF_FLY':
                load_func = load_flying_angle_rc_data
                rocking_axis = 'angle'    
            else:
                err_str = f'Unable to handle scan type of {scantype}.'
                raise RuntimeError(err_str)
        
            (data_dict,
             scan_md,
             xrd_dets) = load_func(
                            scan_id=scan_id,
                            returns=['xrd_dets'],
                                )
            
            filename_id = scan_md['scan_id']
        
        xrd_data = [data_dict[f'{xrd_det}_image']
                    for xrd_det in xrd_dets]

        # Make scaler dictionary
        sclr_keys = ['i0', 'i0_time', 'im', 'it']
        sclr_dict = {key:value for key, value in data_dict.items()
                     if key in sclr_keys}

        # Determine null_map
        null_maps = []
        for det in xrd_dets:
            null_key = f'{det}_null_map'
            if null_key in data_dict:
                null_maps.append(data_dict[null_key])
            else:
                null_maps.append(None)
        
        if filename is None:
            if len(xrd_data) < 1:
                # Iterate through detectors
                filenames = [f'scan{filename_id}_{det}_rsm'
                            for det in xrd_dets]
            else:
                filenames = [f'scan{filename_id}_rsm']
        else:
            if isinstance(filename, list):
                if len(filename) != len(xrd_data):
                    warn_str = ('WARNING: length of specified '
                                + 'filenames does not match the '
                                + 'number of detectors. Naming may '
                                + 'be unexpected.')
                    print(warn_str)
                    # Iterate through detectors
                    filenames = [f'{filename[0]}_{det}_rsm' ]
                else:
                    filenames = [filename]
            else:
                filenames = [f'{filename}_{det}_rsm'
                             for det in xrd_dets]

        extra_md = {}
        for key in scan_md.keys():
            if key not in ['scan_id',
                           'beamline_id',
                           'energy',
                           'dwell',
                           'theta',
                           'start_time',
                           'scan_input']:
                
                if isinstance(scan_md[key], (int, float, str)):
                    extra_md[key] = scan_md[key]
                elif isinstance(scan_md[key], (list, np.ndarray)):
                    if len(scan_md[key]) == 1:
                        if isinstance(scan_md[key][0], str):
                            extra_md[key] = str(scan_md[key][0])
                        else:
                            extra_md[key] = scan_md[key][0]
                    else:
                        if isinstance(scan_md[key][0], str):
                            extra_md[key] = [str(v) for v in scan_md[key]]
                        else:
                            extra_md[key] = list(scan_md[key])
                else:
                    warn_str = (f'WARNING: extra metadata of {key} '
                                + 'could not be properly handled and '
                                + 'will be ignored..')
                    print(warn_str)

        rocking_curves = []
        for i, (image_data, det) in enumerate(zip(xrd_data, xrd_dets)):
            rsm = cls(scan_id=scan_md['scan_id'],
                      wd=wd,
                      filename=filenames[i],
                      image_data=image_data,
                      # Not nominal values - those would be fine.
                      energy=data_dict['energy'], 
                      dwell=scan_md['dwell'],
                      # Not nominal values - those would be fine.
                      theta=data_dict['theta'],
                      sclr_dict=sclr_dict,
                      beamline=scan_md['beamline_id'],
                      facility='NSLS-II',
                      detector=det,
                      # time_stamp=scan_md['time_str'],
                      extra_metadata=extra_md,
                      save_hdf=save_hdf,
                      null_map=null_maps[i],
                      rocking_axis=rocking_axis,
                      **kwargs
                      )
            
            # Check for dark-field. Save but do not apply correction.
            if f'{det}_dark' in data_dict:
                note_str = f'Automatic dark-field found for {det}.'
                rsm.dark_field = data_dict[f'{det}_dark'] # Could be passed as extra_attrs
                rsm.save_images(images='dark_field', units='counts')
                print(note_str)
            
            rocking_curves.append(rsm)

        print(f'{cls.__name__} loaded!')
        if len(rocking_curves) > 1:
            return tuple(rocking_curves)
        else:
            return rocking_curves[0]
        

    def load_parameters_from_txt(self,
                                 filename=None,
                                 wd=None):
        """
        Load scaler dictionaries from a text file.

        This function loads the scaler parameters from the output of
        the io.db_io.save_map_parameters function without the encoder
        values. This is intended to support loading parameters after
        loading images from a 4D image stack. If the data is loaded
        using the 'from_db' method, this function is not needed.

        Parameters
        ----------
        filename : str
            Name of text file with parameters.
        wd : path string, optional
            Path where the file can be found. Will use the internal
            working directory if not provided.
        
        Notes
        -----
        This function is not commonly used. Loading the data with the
        'from_db' method will load parameters by default.
        """
        
        if wd is None:
            wd = self.wd

        if filename is None:
            mask = [(str(self.scan_id) in file
                    and 'parameters' in file)
                    for file in os.listdir(wd)]
            filename = np.asarray(os.listdir(wd))[mask][0]

        out = np.genfromtxt(f'{wd}{filename}')
        energy = out[0]
        sclrs = out[1:]

        base_sclr_names = ['i0', 'im', 'it']
        if len(sclrs) <= 3:
            sclr_names = base_sclr_names[:len(sclrs)]
        else:
            sclr_names = base_sclr_names + [f'i{i}' for i in
                                            range(3, len(sclrs) - 3)]
        
        sclr_dict = dict(zip(sclr_names, sclrs))

        self.energy = energy
        self.set_scalers(sclr_dict)
        

    ################################################
    ### Rocking Curve to 3D Reciprocal Space Map ###
    ################################################

    def get_sampled_edges(self,
                          q_arr=None):
        
        """
        Find the edges of the measured 3D reciprocal space volume.

        Determine the edge image pixels of the measured 3D reciprocal
        space volume and store them internally as lists of vectors
        for each of the 12 (4 for first image, 4 for last image, and 4
        for image corners) edges.

        Parameters
        ----------
        q_arr : Numpy.ndarray of shape (N, 3)
            Array of N rotations or energies/wavlengths matching the
            number of images. None by default and the internal q_arr
            attribute is used.

        Raises
        ------
        AttributeError if q_arr is not provided and does not exist in
        the current instance.
        """
        
        if q_arr is None:
            if (hasattr(self, 'q_arr')
                and self.q_arr is not None):
                q_arr = self.q_arr
            else:
                err_str = ('Must provide q_arr or have q_arr stored '
                           + 'internally.')
                raise AttributeError(err_str)

        edges = ([[] for _ in range(12)])
        for i, qi in tqdm(enumerate(q_arr), total=len(self.energy)):
            # Find edges
            if i == 0:
                edges[4] = qi[0]
                edges[5] = qi[-1]
                edges[6] = qi[:, 0]
                edges[7] = qi[:, -1]
            elif i == len(self.energy)- 1:
                edges[8] = qi[0]
                edges[9] = qi[-1]
                edges[10] = qi[:, 0]
                edges[11] = qi[:, -1]
            # Corners
            edges[0].append(qi[0, 0])
            edges[1].append(qi[0, -1])
            edges[2].append(qi[-1, 0])
            edges[3].append(qi[-1, -1])
        
        for i in range(4):
            edges[i] = np.asarray(edges[i])

        self.edges = edges


    def vectorize_images(self,
                         rewrite_data=False,
                         verbose=False):
        """
        Convert 2D blobs into 3D reciprocal space vectors.

        Convert each pixel inside of 2D blobs into a list of 3D
        reciprocal space vectors. Each pixel is assigned it's q-space
        coordinates and intensity. The vectors are then stored
        internally and written to the HDF file if available as 
        "vectors".

        Parameters
        ----------
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

        if not hasattr(self, 'blob_masks'):
            err_str = ('Must find 2D blobs first to avoid '
                       + 'overly large datasets.')
            raise ValueError(err_str)
        
        if not hasattr(self, 'edges') or self.edges is None:
            edges = ([[] for _ in range(12)])

        # Reserve memory, a little faster and throws errors sooner
        q_vectors = np.zeros((np.sum(self.blob_masks), 3),
                              dtype=self.dtype)

        print('Vectorizing images...')
        filled_indices = 0
        for i, qi in tqdm(enumerate(self.q_arr), total=self.num_images):
            qi = qi.astype(self.dtype)
            next_indices = np.sum(self.blob_masks[i])
            # Fill q_vectors from q_arr
            for idx in range(3):
                (q_vectors[filled_indices : (filled_indices
                                            + next_indices)]
                ) = qi[self.blob_masks[i].squeeze()]
            filled_indices += next_indices

            if not hasattr(self, 'edges') or self.edges is None:
                # Find edges
                if i == 0:
                    edges[4] = qi[0]
                    edges[5] = qi[-1]
                    edges[6] = qi[:, 0]
                    edges[7] = qi[:, -1]
                elif i == self.num_images - 1:
                    edges[8] = qi[0]
                    edges[9] = qi[-1]
                    edges[10] = qi[:, 0]
                    edges[11] = qi[:, -1]
                # Corners
                edges[0].append(qi[0, 0])
                edges[1].append(qi[0, -1])
                edges[2].append(qi[-1, 0])
                edges[3].append(qi[-1, -1])
        
        if not hasattr(self, 'edges') or self.edges is None:
            for i in range(4):
                edges[i] = np.asarray(edges[i])
            self.edges = edges
        
        # Assign useful variables
        self.vectors = np.hstack([
                        q_vectors,
                        self.images[self.blob_masks].reshape(-1, 1)
                        ])
        # Write to hdf
        self.save_vectors(rewrite_data=rewrite_data,
                          verbose=verbose)


    @XRDBaseScan._protect_hdf()
    def save_vectors(self,
                     vectors=None,
                     edges=None,
                     rewrite_data=False,
                     verbose=False):
        """
        Save the vectors to the HDF file.

        Parameters
        ----------
        vectors : list or Numpy.ndarray
            List or Numpy.ndarray of vectors to be written.
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
        AttributeError if "vectors" is not provided and not and
        attribute of XRDMap.
        """

        # Check input
        if vectors is None:
            if (hasattr(self, 'vectors')
                and self.vectors is not None):
                vectors = self.vectors
            else:
                err_str = ('Must provide vectors or '
                        + f'{self.__class__.__name__} must have '
                        + 'vectors attribute.')
                raise AttributeError(err_str)
        if edges is None:
            if (hasattr(self, 'edges')
                and self.edges is not None):
                edges = self.edges
            else:
                err_str = ('Must provide edges or '
                           + f'{self.__class__.__name__} must have '
                           + 'edges attribute.')
                raise AttributeError(err_str)
    
        self._save_rocking_vectorization(self.hdf,
                                         vectors,
                                         edges=edges,
                                         rewrite_data=rewrite_data,
                                         verbose=verbose)
    

    @XRDBaseScan._protect_hdf()
    def load_vectors(self):
        """
        Load the vectors from the HDF file.
        """

        self._load_vectors(self.hdf)
                            

    #######################
    ### Blobs and Spots ###
    #######################


    def get_vector_int_mask(self,
                            intensity=None,
                            int_cutoff=0,
                            relative_cutoff=True):
        """

        """

        if intensity is None:
            if (hasattr(self, 'vectors')
                and self.vectors is not None):
                intensity = self.vectors[:, -1]
        
        int_mask = generate_intensity_mask(
                        intensity,
                        int_cutoff=int_cutoff,
                        relative_cutoff=relative_cutoff)

        return int_mask


    @copy_docstring(XRDBaseScan._find_blobs)
    def find_2D_blobs(self, *args, **kwargs):
        super()._find_blobs(*args, **kwargs)

        
    # Uncommon
    def find_3D_blobs(self,
                      max_dist=0.05,
                      max_neighbors=5,
                      subsample=1,
                      int_cutoff=0,
                      relative_cutoff=True,
                      save_to_hdf=True):
        """

        """

        if (not hasattr(self, 'vectors')
            or self.vectors is None):
            err_str = ('Cannot perform 3D spot search without '
                       + 'first vectorizing images.')
            raise AttributeError(err_str)

        int_mask = self.get_vector_int_mask(
                        int_cutoff=int_cutoff,
                        relative_cutoff=relative_cutoff)

        labels = rsm_blob_search(self.vectors[:, :3][int_mask],
                                 max_dist=max_dist,
                                 max_neighbors=max_neighbors,
                                 subsample=subsample)
        
        self.blob_labels = labels
        self.blob_int_mask = int_mask

        if save_to_hdf:
            self.save_vector_information(
                self.blob_labels,
                'blob_labels',
                extra_attrs={'blob_int_cutoff' : int_cutoff,
                             'relative_cutoff' : int(relative_cutoff)})
        
    
    def find_3D_spots(self,
                      nn_dist=0.005,
                      significance=0.1,
                      subsample=1,
                      int_cutoff=0,
                      relative_cutoff=True,
                      label_int_method='mean',
                      save_to_hdf=True,
                      verbose=True):
        """

        """

        if (not hasattr(self, 'vectors')
            or self.vectors is None):
            err_str = ('Cannot perform 3D spot search without '
                       + 'first vectorizing images.')
            raise AttributeError(err_str)

        int_mask = self.get_vector_int_mask(
                        int_cutoff=int_cutoff,
                        relative_cutoff=relative_cutoff)
        
        (spot_labels,
         spots,
         label_ints,
         label_maxs) = rsm_spot_search(self.vectors[:, :3][int_mask],
                                       self.vectors[:, -1][int_mask],
                                       nn_dist=nn_dist,
                                       significance=significance,
                                       subsample=subsample,
                                       verbose=verbose)

        # Convert reciprocal positions to polar units
        if self.rocking_axis == 'energy':
            if self.use_stage_rotation:
                stage_rotation = self.theta[0]
            else:
                stage_rotation = 0
            tth, chi, wavelength = q_2_polar(spots,
                                stage_rotation=stage_rotation,
                                degrees=(
                                self.polar_units == 'deg'))
            theta = [self.theta[0],] * len(spots)
        else: # angle
            tth, chi, theta = q_2_polar(spots,
                                wavelength=self.wavelength[0],
                                degrees=(
                                self.polar_units == 'deg'))
            wavelength = [self.wavelength[0],] * len(spots)

        temp_dict = {
            'height' : label_maxs,
            'intensity' : label_ints,
            'qx' : spots[:, 0],
            'qy' : spots[:, 1],
            'qz' : spots[:, 2],
            'tth' : tth,
            'chi' : chi,
            'wavelength': wavelength,
            'theta' : theta
            }

        # Save 3D spots similar to 3D spots
        spots_3D = pd.DataFrame.from_dict(temp_dict)
        self.spots_3D = spots_3D
        del temp_dict, spots, label_ints
        # Information for rebuilding spots
        # from vectorized images
        self.spot_labels = spot_labels
        self.spot_int_mask = int_mask

        # Write to hdf
        if save_to_hdf:
            self.save_3D_spots()
            self.save_vector_information(
                self.spot_labels,
                'spot_labels',
                extra_attrs={'spot_int_cutoff' : int_cutoff,
                             'relative_cutoff' : int(relative_cutoff)})
        

    def trim_spots(self,
                   remove_less=0.01,
                   key='intensity',
                   save_spots=False):
        """

        """
        
        self._trim_spots(self.spots_3D,
                         remove_less=remove_less,
                         key=key)

        if save_spots:
            self.save_3D_spots()    


    # Analog of 2D spots from xrdmap
    @XRDBaseScan._protect_hdf(pandas=True)
    def save_3D_spots(self, extra_attrs=None):
        """

        """

        print('Saving 3D spots to the HDF file...',
              end='', flush=True)
        hdf_str = f'{self._hdf_type}/reflections'
        self.spots_3D.to_hdf(self.hdf_path,
                             key=f'{hdf_str}/spots_3D',
                             format='table')

        if extra_attrs is not None:
            if self.hdf is None:
                self.open_hdf()
            for key, value in extra_attrs.items():
                overwrite_attr(self.hdf[hdf_str].attrs, key, value)      
        print('done!')
    

    @XRDBaseScan._protect_hdf()
    def save_vector_information(self,
                                vector_info,
                                vector_info_title,
                                rewrite_data=True,
                                extra_attrs=None,
                                verbose=False):
        """

        """

        self._save_rocking_vectorization(self.hdf,
                                         vector_info,
                                         vector_title=vector_info_title,
                                         edges=None,
                                         rewrite_data=rewrite_data,
                                         verbose=verbose)

        # Add extra information
        if extra_attrs is not None:
            for key, value in extra_attrs.items():
                overwrite_attr(
                    self.hdf[self._hdf_type]['vectorized_data'][vector_info_title].attrs,
                    key,
                    value)


    ###########################################
    ### Indexing, Corrections, and Analysis ###
    ###########################################

    @property
    def qmask(self):
        """

        """
        if hasattr(self, '_qmask'):
            return self._qmask
        else:
            self._qmask = QMask.from_XRDRockingScan(self)
            return self._qmask


    def _parse_indexing_inputs(self,
                               spots=None,
                               intensity=None,
                               int_cutoff=0,
                               relative_cutoff=True
                               ):
        """

        """
        
        if spots is None and hasattr(self, 'spots_3D'):
            spots = self.spots_3D[['qx', 'qy', 'qz']].values
        else:
            err_str = ('Must specify 3D spots or rocking curve must '
                       + 'already have spots.')
            raise ValueError(err_str)
        
        if intensity is None and hasattr(self, 'spots_3D'):
            intensity = self.spots_3D['intensity'].values
        else:
            err_str = ('Must specify 3D spot intensities or rocking '
                       + 'curve must already have spot intensities.')
            raise ValueError(err_str)
        
        if len(spots) != len(intensity):
            err_str = (f'Length of spots ({len(intensity)}) does '
                       + 'not match length of intensity '
                       + f'({len(intensity)}).')
            raise RuntimeError(err_str)
        
        int_mask = self.get_vector_int_mask(
                    intensity=intensity,
                    int_cutoff=int_cutoff,
                    relative_cutoff=relative_cutoff)
        
        return spots[int_mask], intensity[int_mask], int_mask


    def index_best_grain(self,
                         near_q,
                         near_angle,
                         spots=None,
                         intensity=None,
                         int_cutoff=0,
                         relative_cutoff=True,
                         phase=None,
                         half_mask=True,
                         method='seed_casting',
                         save_to_hdf=True,
                         **kwargs
                         ):
        """

        """
        
        (spots,
         intensity,
         int_mask) = self._parse_indexing_inputs(
                                    spots,
                                    intensity,
                                    int_cutoff,
                                    relative_cutoff=relative_cutoff)

        if phase is None and len(self.phases) == 1:
            phase = list(self.phases.values())[0]
        else:
            if len(self.phases) == 0:
                err_str = 'No phase specified and no phases loaded.'
            else:
                err_str = ('No phase specified and ambiguous choice '
                           + 'from loaded phases.')
            raise ValueError(err_str)

        # Only pair casting indexing is currently supported
        if method.lower() in ['seed_casting']:
            (best_indexing,
             best_qof) = phase_index_best_grain(
                            phase,
                            spots,
                            intensity,
                            near_q,
                            near_angle,
                            self.qmask,
                            degrees=self.polar_units == 'deg',
                            half_mask=half_mask,
                            **kwargs)
        else:
            err_str = (f"Unknown method ({method}) specified. Only "
                       + "'seed_casting' is currently supported.")
            raise ValueError(err_str)

        if hasattr(self, 'spots_3D'):
            grains = np.asarray([np.nan,] * len(self.spots_3D))
            h, k, l = grains.copy(), grains.copy(), grains.copy()
            phases = ['',] * len(self.spots_3D)
            qofs = grains.copy()

            spot_inds, ref_inds = best_indexing.T
            spot_inds = int_mask.nonzero()[0][spot_inds]

            if half_mask:
                half_mask = phase.all_hkls[:, -1] <= 0
            else:
                half_mask = np.ones(len(phase.all_hkls), dtype=np.bool_)
            
            grains[spot_inds] = 0
            hkls = phase.all_hkls[half_mask][ref_inds.astype(int)]
            h[spot_inds] = hkls[:, 0]
            k[spot_inds] = hkls[:, 1]
            l[spot_inds] = hkls[:, 2]
            qofs[spot_inds] = best_qof
            for ind in spot_inds:
                phases[ind] = phase.name

            for key, values in zip(['phase', 'grain_id', 'h', 'k', 'l', 'qof'],
                                   [phases, grains, h, k, l, qofs]):
                self.spots_3D[key] = values
            
            # Write to hdf
            if save_to_hdf:
                self.save_3D_spots()

        return best_indexing, best_qof


    def index_all_grains(self,
                         near_q,
                         near_angle,
                         spots=None,
                         intensity=None,
                         int_cutoff=0,
                         relative_cutoff=True,
                         phase=None,
                         half_mask=True,
                         method='seed_casting',
                         save_to_hdf=True,
                         **kwargs
                         ):
        """

        """
        
        (spots,
         intensity,
         int_mask) = self._parse_indexing_inputs(
                                    spots,
                                    intensity,
                                    int_cutoff,
                                    relative_cutoff=relative_cutoff)

        if phase is None and len(self.phases) == 1:
            phase = list(self.phases.values())[0]
        else:
            if len(self.phases) == 0:
                err_str = 'No phase specified and no phases loaded.'
            else:
                err_str = ('No phase specified and ambiguous choice '
                           + 'from loaded phases.')
            raise ValueError(err_str)

        # Only pair casting indexing is currently supported
        if method.lower() in ['seed_casting']:
            (best_indexings,
             best_qofs) = phase_index_all_grains(
                            phase,
                            spots,
                            intensity,
                            near_q,
                            near_angle,
                            self.qmask,
                            degrees=self.polar_units == 'deg',
                            half_mask=half_mask,
                            **kwargs)
        else:
            err_str = (f"Unknown method ({method}) specified. Only "
                       + "'seed_casting' is currently supported.")
            raise ValueError(err_str)
        
        if hasattr(self, 'spots_3D'):
            grains = np.asarray([np.nan,] * len(self.spots_3D))
            h, k, l = grains.copy(), grains.copy(), grains.copy()
            phases = ['',] * len(self.spots_3D)
            qofs = grains.copy()

            if half_mask:
                half_mask = phase.all_hkls[:, -1] <= 0
            else:
                half_mask = np.ones(len(phase.all_hkls), dtype=np.bool_)

            for i, indexing in enumerate(best_indexings):

                spot_inds, ref_inds = indexing.T
                spot_inds = int_mask.nonzero()[0][spot_inds]
            
                grains[spot_inds] = i
                hkls = phase.all_hkls[half_mask][ref_inds.astype(int)]
                h[spot_inds] = hkls[:, 0]
                k[spot_inds] = hkls[:, 1]
                l[spot_inds] = hkls[:, 2]
                qofs[spot_inds] = best_qofs[i]
                for ind in spot_inds:
                    phases[ind] = phase.name

            for key, values in zip(['phase', 'grain_id', 'h', 'k', 'l', 'qof'],
                                   [phases, grains, h, k, l, qofs]):
                self.spots_3D[key] = values
            
            # Write to hdf
            if save_to_hdf:
                self.save_3D_spots()
        
        return best_indexings, best_qofs


    # Strain math
    # TODO: Gives strain in crystal coordinates. Convert to lab
    def get_strain_orientation(self,
                               q_vectors=None,
                               hkls=None,
                               phase=None,
                               grain_id=None):
        """

        """

        # Parse phase input if specified
        if phase is not None:
            if isinstance(phase, Phase):
                pass
            elif isinstance(phase, str):
                if not hasattr(self, 'phases'):
                    err_str = 'No phases found for comparison and phase not specified.'
                    raise AttributeError(err_str)
                elif phase not in self.phases:
                    err_str = (f'Phase of {phase} not found in phases of {list(self.phases.keys())}.')
                    raise RuntimeError(err_str)
                else:
                    phase = self.phases[phase]                       

        # Parse grain_id input if specified
        if grain_id is not None:
            if not hasattr(self, 'spots_3D'):
                err_str = ('Cannot use indexed grain without finding '
                           + 'and indexing spots!')
                raise ValueError(err_str)
            elif 'grain_id' not in self.spots_3D:
                err_str = ('Cannot used indexed grain if spots have '
                           + 'not been indexed!')
                raise ValueError(err_str)
            elif grain_id not in self.spots_3D['grain_id']:
                err_str = (f'Indexed grain {int(grain_id)} not found '
                           + 'in indexed spots!')
                raise ValueError(err_str)
            else:
                grain_mask = self.spots_3D['grain_id'] == grain_id
                q_vectors = self.spots_3D[['qx', 'qy', 'qz']][grain_mask].values
                hkls = self.spots_3D[['h', 'k', 'l']][grain_mask].values.astype(int)
                phase_name = self.spots_3D['phase'][grain_mask].to_list()[0]

                if phase is None: # Figure out phase if not specified already
                    if (not hasattr(self, 'phases') or len(self.phases) < 1):
                        err_str = 'No phases found for comparison and phase not specified.'
                        raise AttributeError(err_str)
                    elif phase_name not in self.phases:
                        err_str = ('Indexed phase {phase_name} must be in phases or explicitly given.')
                        raise RuntimeError(err_str)
                    else:
                        phase = self.phases[phase_name]
        # Final check for explicit values
        else:
            if q_vectors is None or hkls is None or phase is None:
                err_str = ('Must define either q_vectors and hkls and '
                           + 'phase or use grain_id from indexed '
                           + 'spots.')
                raise ValueError(err_str)

        # Do strain and orientation math
        (eij,
         U,
         strained) = get_strain_orientation(
                                q_vectors,
                                hkls,
                                LatticeParameters.from_Phase(phase))
        
        return eij, U

    
    # Not sure if this is feasible
    def get_zero_point_correction():
        raise NotImplementedError()


    ##########################
    ### Plotting Functions ###
    ##########################

    # Rewritten from XRDBaseScan
    def _title_with_scan_id(self,
                            title,
                            default_title='',
                            title_scan_id=True):
        """

        """

        # Should be list if iterable, but just in case
        if isinstance(self.scan_id, (list, np.ndarray)): 
            sorted_scans = sorted(self.scan_id)
            scan_id_str = f'{sorted_scans[0]}-{sorted_scans[-1]}'
        else:
            scan_id_str = self.scan_id
        
        if title is None:
            title = default_title
        if title_scan_id:
            if title == '':
                return f'scan{scan_id_str}'
            else:
                return f'scan{scan_id_str}: {title}'
        else:
            return title


    # Disable q-space plotting
    def plot_q_space(self, *args, **kwargs):
        """

        """

        err_str = ('Q-space plotting not supported for '
                   + 'XRDRockingCurves; Ewald sphere/or '
                   + 'crystal orientation changes during scanning.')
        raise NotImplementedError(err_str)


    @return_plot_wrapper
    def plot_image_stack(self,
                         images=None,
                         slider_vals=None,
                         slider_label='Index',
                         title=None,
                         vmin=None,
                         vmax=None,
                         title_scan_id=True,
                         **kwargs):
        """

        """

        if images is None:
            images = self.images.squeeze()

        if slider_vals is None:
            if self.rocking_axis == 'energy':
                slider_vals = self.energy
            if self.rocking_axis == 'angle':
                slider_vals = self.theta
        
        if slider_label is None:
            if self.rocking_axis == 'energy':
                slider_label = 'Energy [keV]'
            if self.rocking_axis == 'angle':
                slider_label = 'Angle [deg]'

        title = self._title_with_scan_id(
                            title,
                            default_title=('XRD '
                                + f'{self.rocking_axis.capitalize()} '
                                + 'Rocking Curve'),
                            title_scan_id=title_scan_id)
        
        fig, ax, slider = base_slider_plot(
                                images,
                                slider_vals=slider_vals,
                                slider_label=slider_label,
                                title=title,
                                vmin=vmin,
                                vmax=vmax,
                                **kwargs)
        
         # Need a slider reference
        self._image_stack_slider = slider
        return fig, ax
    

    @return_plot_wrapper
    def plot_waterfall(self, **kwargs):
        """

        """
        return self._plot_waterfall(
                            axis=0,
                            axis_text=self.rocking_axis.capitalize(),
                            **kwargs)
            
    @return_plot_wrapper
    def plot_3D_scatter(self,
                        q_vectors=None,
                        intensity=None,
                        spots_3D=None,
                        title=None,
                        edges=None,
                        skip=None,
                        title_scan_id=True,
                        **kwargs):
        """

        """

        # Set defaults
        default_title = '3D Scatter'

        # Parse inputs
        if spots_3D is not None:
            if isinstance(spots_3D, pd.DataFrame):
                q_vectors = spots_3D[['qx', 'qy', 'qz']].values
                intensity = spots_3D['intensity'].values
                default_title = '3D Spots'
                skip = 1
                kwargs['alpha'] = 1
            elif (hasattr(self, 'spots_3D')
                  and self.spots_3D is not None):
                q_vectors = self.spots_3D[['qx', 'qy', 'qz']].values
                intensity = self.spots_3D['intensity'].values
                default_title = '3D Spots'
                skip = 1
                kwargs['alpha'] = 1
            else:
                warn_str = ("WARNING: 'spots_3D' could not be properly"
                            + " parsed. Must be Pandas DataFrame or "
                            + "bool indicating an internal DataFrame. "
                            + "\nAttempting to plot vectors instead.")
                print(warn_str)

        # Plot direct vectors
        if q_vectors is None:
            if hasattr(self, 'vectors') and self.vectors is not None:
                q_vectors = self.vectors[:, :3]
            else:
                err_str = 'Must provide or already have q_vectors.'
                raise ValueError(err_str)
        if intensity is None:
            if (hasattr(self, 'vectors')
                and self.vectors is not None):
                intensity = self.vectors[:, -1]
            else:
                intensity = np.zeros(len(q_vectors),
                                     dtype=q_vectors.dtype)

        if edges is None and hasattr(self, 'edges'):
            edges = self.edges

        fig, ax = plot_3D_scatter(q_vectors,
                                  colors=intensity,
                                  skip=skip,
                                  edges=edges,
                                  **kwargs)

        title = self._title_with_scan_id(
                            title,
                            default_title=default_title,
                            title_scan_id=title_scan_id)
        ax.set_title(title)

        return fig, ax


    # TODO: Combine with plot_3D_scatter?
    @return_plot_wrapper
    def plot_3D_indexing(self,
                         spots_3D=None,
                         grain_ids=[0],
                         title=None,
                         edges=None,
                         title_scan_id=True,
                         **kwargs):
        """

        """

        # Parse inputs
        if spots_3D is not None and not isinstance(spots_3D, pd.DataFrame):
            warn_str = ("WARNING: 'spots_3D' must be Pandas DataFrame not "
                        + f"{type(spots_3D)}. Searching for internal "
                        + "spots_3D instead.")
            print(warn_str)
            spots_3D = None

        if spots_3D is None:
            if hasattr(self, 'spots_3D') and self.spots_3D is not None:
                spots_3D = self.spots_3D
            else:
                err_str = ("Must provide 'spots_3D' or must have internal "
                        + "'spots_3D' attribute.")
                raise AttributeError(err_str)
        
        for key in ['qx', 'qy', 'qz', 'grain_id', 'h', 'k', 'l']:
            if key not in spots_3D:
                err_str = (f"{key} not in spots_3D DataFrame. The data may"
                        + " not be indexed yet.")
                raise KeyError(err_str)
        
        # Get universal values
        all_spot_qs = spots_3D[['qx', 'qy', 'qz']].values
        all_spot_ints = spots_3D['intensity'].values

        # Parse phases
        phases = [spots_3D[spots_3D['grain_id'] == gi]['phase'].values[0] for gi in grain_ids]
        phase_mask = [True,] * len(phases)
        if np.any([phase != phases[0] for phase in phases]):
            warn_str = ('WARNING: Grain IDs selected for multiple '
                        + 'phases. Indexing plotting only for a single'
                        + ' phase which will default to the first '
                        + f'({phases[0]}).')
            print(warn_str)
            phase_mask = phases == phase[0]
        grain_ids = np.asarray(grain_ids)[phase_mask]

        # Get reference values
        phase = self.phases[phases[0]]
        phase.generate_reciprocal_lattice(qmax=np.linalg.norm(all_spot_qs, axis=1).max() * 1.15)
        all_ref_qs = phase.all_qs
        all_ref_hkls = phase.all_hkls
        all_ref_fs = phase.all_fs

        # Build indexing
        indexings = []
        for grain_id in grain_ids:
            df = spots_3D[spots_3D['grain_id'] == grain_id]
            spot_inds = df.index.values
            hkls = df[['h', 'k', 'l']].values.astype(int)
            ref_inds = [np.all(hkl == all_ref_hkls, axis=1).nonzero()[0][0] for hkl in hkls]

            indexings.append(np.asarray([spot_inds, ref_inds]).T)

        title = self._title_with_scan_id(
                            title,
                            default_title='3D RSM Indexing',
                            title_scan_id=title_scan_id)
        
        fig, ax = plot_3D_indexing(indexings,
                                   all_spot_qs,
                                   all_spot_ints,
                                   all_ref_qs,
                                   all_ref_hkls,
                                   all_ref_fs,
                                   self.qmask,
                                   edges=self.edges,
                                   title=title,
                                   **kwargs)
        
        return fig, ax
        

    # Broken for angle rocking curves
    def plot_3D_isosurfaces(self,
                            q_vectors=None,
                            intensity=None,
                            gridstep=0.01,
                            **kwargs):
        """

        """

        if q_vectors is None:
            if hasattr(self, 'vectors') and self.vectors is not None:
                q_vectors = self.vectors[:, :3]
            else:
                err_str = 'Must provide or already have q_vectors.'
                raise ValueError(err_str)
        
        if intensity is None:
            if (hasattr(self, 'vectors')
                and self.vectors is not None):
                intensity = self.vectors[:, -1]
            else:
                intensity = np.zeros(len(q_vectors),
                                     dtype=q_vectors.dtype)

        # Copy values; they will be modified.
        plot_qs = np.asarray(q_vectors).copy()
        plot_ints = np.asarray(intensity).copy()
        min_int = np.min(plot_ints)

        # Check given q_ext
        for axis in range(3):
            q_ext = (np.max(q_vectors[:, axis])
                    - np.min(q_vectors[:, axis]))
            if  q_ext < gridstep:
                err_str = (f'Gridstep ({gridstep}) is smaller than '
                        + f'q-vectors range along axis {axis} '
                        + f'({q_ext:.4f}).')
                raise ValueError(err_str)

        tth, chi, wavelength = q_2_polar(
                            plot_qs,
                            stage_rotation=0,
                            degrees=self.polar_units == 'deg')
        energy = wavelength_2_energy(wavelength)

        # Assumes energy is rocking axis...
        energy_step = np.abs(np.mean(np.diff(self.energy)))
        min_energy = np.min(self.energy)
        max_energy = np.max(self.energy)

        low_mask = energy <= min_energy + energy_step
        high_mask = energy >= max_energy - energy_step

        # If there are bounded pixels, padd with zeros
        if np.sum([low_mask, high_mask]) > 0:
            low_qs = get_q_vect(
                        tth[low_mask],
                        chi[low_mask],
                        wavelength=energy_2_wavelength(min_energy
                                                    - energy_step),
                        degrees=self.polar_units == 'deg'
                        ).astype(self.dtype)
            
            high_qs = get_q_vect(
                        tth[high_mask],
                        chi[high_mask],
                        wavelength=energy_2_wavelength(max_energy
                                                    + energy_step),
                        degrees=self.polar_units == 'deg'
                        ).astype(self.dtype)

            # Extend plot qs and ints
            plot_qs = np.vstack([plot_qs, low_qs, high_qs])
            plot_ints = np.hstack([
                            plot_ints,
                            np.ones(low_mask.sum()) * min_int,
                            np.ones(high_mask.sum()) * min_int])
        
        plot_3D_isosurfaces(plot_qs,
                            plot_ints,
                            gridstep=gridstep,
                            **kwargs)
    
    # TODO: Move this to plot volume module
    @return_plot_wrapper
    def plot_sampled_volume_outline(self,
                                    edges=None,
                                    title=None,
                                    title_scan_id=True):
        """

        """

        if edges is None:
            if hasattr(self, 'edges'):
                edges = self.edges
            else:
                err_str= ('Cannot plot sampled volume '
                        + 'without given or known edges!')
                raise ValueError(err_str)
        
        fig, ax = plt.subplots(1, 1, 
                               figsize=(5, 5),
                               dpi=200,
                               subplot_kw={'projection':'3d'})

        for edge in edges:
            ax.plot(*edge.T, c='gray', lw=1)

        title = self._title_with_scan_id(
                            title,
                            default_title='Sampled Volume Outline',
                            title_scan_id=title_scan_id)
        ax.set_title(title)

        ax.set_xlabel('qx [Å⁻¹]')
        ax.set_ylabel('qy [Å⁻¹]')
        ax.set_zlabel('qz [Å⁻¹]')
        ax.set_aspect('equal')

        return fig, ax

