# Note: This class is highly experimental
# It is intended to wrap the XRDMap class and convert every method to apply iteratively across a stack of XRDMaps
# This is instended for 3D RSM mapping where each method / parameter may change between maps

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

# from .XRDMap import XRDMap
# from .utilities.utilities import timed_iter

from xrdmaptools.XRDBaseScan import XRDBaseScan
from xrdmaptools.utilities.math import (
    energy_2_wavelength,
    wavelength_2_energy
)
from xrdmaptools.utilities.utilities import (
    generate_intensity_mask
)
from xrdmaptools.crystal.rsm import map_2_grid
from xrdmaptools.geometry.geometry import (
    get_q_vect,
    q_2_polar
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
    pair_casting_index_best_grain,
    pair_casting_index_full_pattern,
    QMask
)
from xrdmaptools.plot.image_stack import base_slider_plot
from xrdmaptools.plot.volume import (
    plot_3D_scatter,
    plot_3D_isosurfaces
)


class XRDRockingCurve(XRDBaseScan):

    # Class variables
    _hdf_type = 'rsm'

    def __init__(self,
                 image_data=None,
                 image_shape=None,
                 map_shape=None,
                 sclr_dict=None,
                 rocking_axis=None,
                 **xrdbasekwargs
                 ):

        if (image_data is None
            and (map_shape is None
                 or image_shape is None)):
            err_str = ('Must specify image_data, '
                       + 'or image and map shapes.')
            raise ValueError(err_str)
        
        if image_data is not None:
            # Parse image_data into useble format
            image_data = np.asarray(image_data)
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
                sclr_dict[key] = np.asarray(value).reshape(self.map_shape)

        XRDBaseScan.__init__(
            self,
            image_data=image_data,
            image_shape=image_shape,
            map_shape=map_shape,
            sclr_dict=sclr_dict,
            map_labels=['rocking_ind',
                           'null_ind'],
            **xrdbasekwargs
            )

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
                err_str = ('Ambiguous rocking direction. '
                            + 'Energy varies by more than 5 eV and '
                            + 'theta varies by more than 50 mdeg.')
                raise RuntimeError(err_str)
            else:
                err_str = ('Ambiguous rocking direction. '
                            + 'Energy varies by less than 5 eV and '
                            + 'theta varies by less than 50 mdeg.')
                raise RuntimeError(err_str)
        
        # Enable features
        if (not self.use_stage_rotation
            and self.rocking_axis == 'angle'):
            self.use_stage_rotation = True

        @XRDBaseScan.protect_hdf()
        def save_extra_attrs(self): # Not sure if this needs self...
            attrs = self.hdf[self._hdf_type].attrs
            attrs['energy'] = self.energy
            attrs['wavelength'] = self.wavelength
            attrs['theta'] = self.theta
            attrs['rocking_axis'] = self.rocking_axis
        save_extra_attrs(self)

    # Overwrite parent function
    def __str__(self):
        ostr = (f'{self._hdf_type}:  scan_id={self.scan_id}, '
                + f'energy_range={min(self.energy):.3f}'
                + f'-{max(self.energy):.3f}, '
                + f'shape={self.images.shape}')
        return ostr
    
    # Modify parent function
    def __repr__(self):
        lines = XRDBaseScan.__repr__(self).splitlines(True)
        lines[4] = (f'\tEnergy Range:\t\t{min(self.energy):.3f}'
                    + f'-{max(self.energy):.3f} keV\n')
        ostr = ''.join(lines)
        return ostr


    #############################
    ### Re-Written Properties ###
    #############################

    def _parse_subscriptable_value(self, value, name):
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


    # A lot of parsing inputs
    @XRDBaseScan.energy.setter
    def energy(self, energy):
        self._parse_subscriptable_value(energy, 'energy')
        self._wavelength = energy_2_wavelength(self._energy)

        # Propogate changes...
        if hasattr(self, 'ai') and self.ai is not None:
            self.ai.energy = self._energy[0]
            if hasattr(self, '_q_arr'):
                delattr(self, '_q_arr')

        @XRDBaseScan.protect_hdf()
        def save_attrs(self): # Not sure if this needs self...
            attrs = self.hdf[self._hdf_type].attrs
            attrs['energy'] = self.energy
            attrs['wavelength'] = self.wavelength
        save_attrs(self)


    # A lot of parsing inputs
    @XRDBaseScan.wavelength.setter
    def wavelength(self, wavelength):
        self._parse_subscriptable_value(wavelength, 'wavelength')
        self._energy = wavelength_2_energy(self._wavelength)

        # Propogate changes...
        if hasattr(self, 'ai') and self.ai is not None:
            self.ai.energy = self._energy[0]
            if hasattr(self, '_q_arr'):
                delattr(self, '_q_arr')

        @XRDBaseScan.protect_hdf()
        def save_attrs(self): # Not sure if this needs self...
            attrs = self.hdf[self._hdf_type].attrs
            attrs['energy'] = self.energy
            attrs['wavelength'] = self.wavelength
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

        # Propogate changes...
        if hasattr(self, 'ai'):
            if hasattr(self, '_q_arr'):
                delattr(self, '_q_arr')
            
        @XRDBaseScan.protect_hdf()
        def save_attrs(self): # Not sure if this needs self...
            attrs = self.hdf[self._hdf_type].attrs
            attrs['theta'] = self.theta
        save_attrs(self)


    # Full q-vector, not just magnitude
    @XRDBaseScan.q_arr.getter
    def q_arr(self):
        if hasattr(self, '_q_arr'):
            return self._q_arr
        elif not hasattr(self, 'ai'):
            raise RuntimeError('Cannot calculate q-space without calibration.')
        else:
            self._q_arr = np.empty((self.num_images,
                                    *self.image_shape,
                                    3),
                                    dtype=self.dtype)
            for i in tqdm(range(self.num_images)):
                wavelength = self.wavelength[i]
                if self.use_stage_rotation:
                    theta = self.theta[i]
                else:
                    theta = None # no rotation!
                
                q_arr = get_q_vect(self.tth_arr,
                                   self.chi_arr,
                                   wavelength=wavelength,
                                   stage_rotation=theta,
                                   degrees=self.polar_units == 'deg',
                                   rotation_axis='y') # hard-coded for srx

            # for i, wavelength in tqdm(enumerate(self.wavelength),
            #                           total=self.num_images):
            #     q_arr = get_q_vect(self.tth_arr,
            #                        self.chi_arr,
            #                        wavelength=wavelength,
            #                        degrees=self.polar_units == 'deg')
                self._q_arr[i] = q_arr
            return self._q_arr

    @q_arr.deleter
    def q_arr(self):
        self._del_arr()
    

    ##########################
    ### Re-Written Methods ###
    ##########################


    def set_calibration(self, *args, **kwargs):
        super().set_calibration(*args,
                                energy=self.energy[0],
                                **kwargs)    
    
    
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
                poni_file=None,
                save_hdf=True,
                repair_method='fill'):
        
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

        if 'null_map' in data_dict.keys():
            null_map = data_dict['null_map']
        else:
            null_map = None

        # filenames = [f'scan{filename_id}_{det}_rsm'
        #                 for det in xrd_dets]
        
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
                           'beamline',
                           'energy',
                           'dwell',
                           'theta',
                           'start_time',
                           'scan_input']:
                extra_md[key] = scan_md[key]

        rocking_curves = []
        for i, xrd_data_i in enumerate(xrd_data):
            rc = cls(
                    scan_id=scan_md['scan_id'],
                    wd=wd,
                    filename=filenames[i],
                    image_data=xrd_data_i,
                    # Not nominal values - those would be fine.
                    energy=data_dict['energy'], 
                    dwell=scan_md['dwell'],
                    # Not nominal values - those would be fine.
                    theta=data_dict['theta'],
                    poni_file=poni_file,
                    sclr_dict=sclr_dict,
                    beamline='5-ID (SRX)',
                    facility='NSLS-II',
                    # time_stamp=scan_md['time_str'],
                    extra_metadata=None,
                    save_hdf=save_hdf,
                    null_map=null_map,
                    rocking_axis=rocking_axis
                    )
            
            # # Add xrf if it exists
            # if 'xs_fluor' in data_dict.keys():
            #     rc.xrf = {'data':data_dict['xs_fluor']}
            
            rocking_curves.append(rc)

        print(f'{cls.__name__} loaded!')
        if len(rocking_curves) > 1:
            return tuple(rocking_curves)
        else:
            return rocking_curves[0]
        

    def load_parameters_from_txt(self,
                                 filename=None,
                                 wd=None):
        
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
    

    def load_metadata_from_txt(self,
                               filename=None,
                               wd=None):
        if wd is None:
            wd = self.wd

        if filename is None:
            mask = [(str(self.scan_id) in file
                    and 'metadata' in file)
                    for file in os.listdir(wd)]
            filename = np.asarray(os.listdir(wd))[mask][0]

        with open(f'{wd}{filename}', 'r') as f:
            json_str = f.read()
            md = json.loads(json_str)
        
        base_md = {key:value for key, value in md.items()
                   if key in ['scan_id', 'theta', 'dwell']}
        extra_md = {key:value for key, value in md.items()
                    if key not in base_md.keys()}
        if 'scan_id' in base_md.keys():
            base_md['scan_id'] = base_md['scan_id']
            del base_md['scan_id']
        
        base_md['scan_id'] = (f"{np.min(base_md['scan_id'])}"
                             + f"-{np.max(base_md['scan_id'])}")
        
        for key, value in base_md.items():
            setattr(self, key, value)
        setattr(self, 'extra_metadata', extra_md)
        

    ################################################
    ### Rocking Curve to 3D Reciprocal Space Map ###
    ################################################

    def vectorize_images(self,
                         override_blob_search=False,
                         rewrite_data=False):

        if (not override_blob_search
            and not hasattr(self, 'blob_masks')):
            err_str = ('Must find 2D blobs first to avoid '
                       + 'overly large datasets.')
            raise ValueError(err_str)
        
        edges = ([[] for _ in range(12)])

        # Reserve memory, a little faster and throws errors sooner
        q_vectors = np.zeros((np.sum(self.blob_masks), 3),
                             dtype=self.dtype)

        print('Vectorizing images...')
        filled_indices = 0
        for i in tqdm(range(self.num_images)):
            q_arr = self.q_arr[i].astype(self.dtype)
            next_indices = np.sum(self.blob_masks[i])
            # Fill q_vectors from q_arr
            for idx in range(3):
                (q_vectors[filled_indices : (filled_indices
                                            + next_indices)]
                ) = q_arr[self.blob_masks[i].squeeze()]
            filled_indices += next_indices

            # Find edges
            if i == 0:
                edges[4] = q_arr[0]
                edges[5] = q_arr[-1]
                edges[6] = q_arr[:, 0]
                edges[7] = q_arr[:, -1]
            elif i == len(self.wavelength) - 1:
                edges[8] = q_arr[0]
                edges[9] = q_arr[-1]
                edges[10] = q_arr[:, 0]
                edges[11] = q_arr[:, -1]
            else: # Corners
                edges[0].append(q_arr[0, 0])
                edges[1].append(q_arr[0, -1])
                edges[2].append(q_arr[-1, 0])
                edges[3].append(q_arr[-1, -1])
        
        for i in range(4):
            edges[i] = np.asarray(edges[i])
        
        # Assign useful variables
        self.edges = edges
        self.q_vectors = q_vectors
        # A bit redundant; copies data
        self.intensity = self.images[self.blob_masks] 
        # Write to hdf
        self.save_vectorization(q_vectors=self.q_vectors,
                                intensity=self.intensity,
                                edges=self.edges,
                                rewrite_data=rewrite_data)

    @XRDBaseScan.protect_hdf()
    def save_vectorization(self,
                           q_vectors=None,
                           intensity=None,
                           edges=None,
                           rewrite_data=False):

        print('Saving vectorized image data...')

        # Write data to hdf
        vect_grp = self.hdf[self._hdf_type].require_group(
                                                'vectorized_data')
        vect_grp.attrs['time_stamp'] = ttime.ctime()

        # Save q_vectors and intensity
        for attr, attr_name in zip([q_vectors, intensity],
                                    ['q_vectors', 'intensity']):

            # Check for values/attributes.
            # Must have both q_vectors and intensity
            if attr is None:
                if (hasattr(self, attr_name)
                    and getttr(self, attr_name) is not None):
                    attr = getattr(self, attr_name)
                else:
                    self.hdf.close()
                    self.hdf = None
                    err_str = (f'Cannot save {attr_name} if not '
                                + 'given or already an attribute.')
                    raise AttributeError(err_str)

            # Check for dataset and compatibility
            attr = np.asarray(attr)
            if attr_name not in vect_grp.keys():
                dset = vect_grp.require_dataset(
                        attr_name,
                        data=attr,
                        shape=attr.shape,
                        dtype=attr.dtype)
            else:
                dset = vect_grp[attr_name]

                if (dset.shape == attr.shape
                    and dset.dtype == attr.dtype):
                    dset[...] = attr

                else:
                    warn_str = 'WARNING:'
                    if dset.shape != attr.shape:
                        warn_str += (f'{attr_name} shape of'
                                    + f' {attr.shape} does not '
                                    + 'match dataset shape '
                                    + f'{dset.shape}. ')
                    if dset.dtype != attr.dtype:
                        warn_str += (f'{attr_name} dtype of'
                                    + f' {attr.dtype} does not '
                                    + 'match dataset dtype '
                                    + f'{dset.dtype}. ')
                    if rewrite_data:
                        warn_str += (f'\nOvewriting {attr_name}. This '
                                    + 'may bloat the total file size.')
                        # Shape changes should not happen
                        # except from q_arr changes
                        print(warn_str)
                        del vect_grp[attr_name]
                        dset = vect_grp.require_dataset(
                            attr_name,
                            data=attr,
                            shape=attr.shape,
                            dtype=attr.dtype)
                    else:
                        warn_str += '\nProceeding without changes.'
                        print(warn_str)
                
        
        # Check for edge information
        if edges is None:
            if hasattr(self, 'edges') and self.edges is not None:
                edges = self.edges
            else:
                warn_str = ('WARNING: No edges given or found. '
                            + 'Edges will not be saved.')
                print(warn_str)

        # Only save edge information if given
        if edges is not None:
            edge_grp = vect_grp.require_group('edges')
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
                        warn_str = 'WARNING:'
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
                            print(warn_str)
                            del edge_grp[edge_title]
                            edge_grp.require_dataset(
                                    edge_title,
                                    data=edge,
                                    shape=edge.shape,
                                    dtype=edge.dtype)
                        else:
                            warn_str += '\nProceeding without changes.'
                            print(warn_str)
                            

    #######################
    ### Blobs and Spots ###
    #######################


    def get_vector_int_mask(self,
                            intensity=None,
                            intensity_cutoff=0):

        if intensity is None:
            if (hasattr(self, 'intensity')
                and self.intensity is not None):
                intensity = self.intensity
        
        int_mask = generate_intensity_mask(
                        intensity,
                        intensity_cutoff)

        return int_mask


    # Same as XRDMap.find_blobs()
    def find_2D_blobs(self,
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
        

    def find_3D_blobs(self,
                      max_dist=0.05,
                      max_neighbors=5,
                      subsample=1,
                      intensity_cutoff=0,
                      save_to_hdf=True):

        if (not hasattr(self, 'q_vectors')
            or not hasattr(self, 'intensity')):
            err_str = ('Cannot perform 3D spot search without '
                       + 'first vectorizing images.')
            raise AttributeError(err_str)

        int_mask = self.get_vector_int_mask(
                        intensity_cutoff=intensity_cutoff)
                    
        labels = rsm_blob_search(self.q_vectors[int_mask],
                                 max_dist=max_dist,
                                 max_neighbors=max_neighbors,
                                 subsample=subsample)
        
        self.blob_labels = labels
        self.blob_int_mask = int_mask

        if save_to_hdf:
            self.save_vector_information(
                self.blob_labels,
                'blob_labels',
                extra_attrs={'blob_int_cutoff' : intensity_cutoff})
        
    
    def find_3D_spots(self,
                      nn_dist=0.005,
                      significance=0.1,
                      subsample=1,
                      intensity_cutoff=0,
                      label_int_method='mean',
                      save_2_hdf=True):

        if (not hasattr(self, 'q_vectors')
            or not hasattr(self, 'intensity')):
            err_str = ('Cannot perform 3D spot search without '
                       + 'first vectorizing images.')
            raise AttributeError(err_str)

        int_mask = self.get_vector_int_mask(
                        intensity_cutoff=intensity_cutoff)

        (spot_labels,
         spots,
         label_ints) = rsm_spot_search(self.q_vectors[int_mask],
                                       self.intensity[int_mask],
                                       nn_dist=nn_dist,
                                       significance=significance,
                                       subsample=subsample)

        tth, chi, wavelength = q_2_polar(spots,
                                         stage_rotation=0,
                                         degrees=(
                                            self.polar_units == 'deg'))

        temp_dict = {
            'intensity' : label_ints,
            'qx' : spots[:, 0],
            'qy' : spots[:, 1],
            'qz' : spots[:, 2],
            'tth' : tth,
            'chi' : chi,
            'wavelength': wavelength,
            # 'theta' : theta
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
        if save_2_hdf:
            self.save_3D_spots()
            self.save_vector_information(
                self.spot_labels,
                'spot_labels',
                extra_attrs={'spot_int_cutoff' : intensity_cutoff})
        
    
    # Analog of 2D spots from xrdmap
    @XRDBaseScan.protect_hdf(pandas=True)
    def save_3D_spots(self, extra_attrs=None):
        print('Saving 3D spots to hdf...', end='', flush=True)
        hdf_str = f'{self._hdf_type}/reflections/spots_3D'
        self.spots_3D.to_hdf(self.hdf_path,
                             key=hdf_str,
                             format='table')

        if extra_attrs is not None:
            self.open_hdf()
            for key, value in extra_attrs.items():
                self.hdf[hdf_str].attrs[key] = value        
        print('done!')


    # # Analog of 2D spots from xrdmap
    # def save_3D_spots(self, extra_attrs=None):
    #     # Save spots to hdf
    #     if self.hdf_path is not None:
    #         print('Saving 3D spots to hdf...')

    #         # Open hdf flag
    #         keep_hdf = True
    #         if self.hdf is None:
    #             self.hdf = h5py.File(self.hdf_path, 'a')
    #             keep_hdf = False

    #         # Save to hdf
    #         self.close_hdf()
    #         hdf_str = f'{self._hdf_type}/reflections/spots_3D'
    #         self.spots_3D.to_hdf(self.hdf_path,
    #                         key=hdf_str,
    #                         format='table')

    #         if extra_attrs is not None:
    #             self.open_hdf()
    #             for key, value in extra_attrs.items():
    #                 self.hdf[hdf_str].attrs[key] = value

    #         if keep_hdf:
    #             self.open_hdf()
    #         else:
    #             self.close_hdf()
            
    #         print('done!')


    @XRDBaseScan.protect_hdf()
    def save_vector_information(self,
                                data,
                                title,
                                extra_attrs=None):
            
        # Get vector group
        vect_grp = self.hdf[self._hdf_type].require_group(
                                                'vectorized_data')

        if title not in vect_grp.keys():
            dset = vect_grp.require_dataset(
                title,
                data=data,
                shape=data.shape,
                dtype=data.dtype)
        else:
            dset = vect_grp[title]

            if (dset.shape == data.shape
                and dset.dtype == data.dtype):
                dset[...] = data
            elif (dset.shape == data.shape
                    and dset.dtype != data.dtype):
                    dset[...] = data.astype(dset.dtype)
            else:
                warn_str = (f'WARNING: {title} dataset shape does '
                            + 'not match given data; rewriting '
                            + 'with new shape. This may bloat '
                            + 'the file size.')
                print(warn_str)
                del vect_grp[title]
                dset = vect_grp.require_dataset(
                    title,
                    data=data,
                    shape=data.shape,
                    dtype=data.dtype)
        
        # Add extra information
        if extra_attrs is not None:
            for key, value in extra_attrs.items():
                dset.attrs[key] = value


    ###########################################
    ### Indexing, Corrections, and Analysis ###
    ###########################################

    def _parse_indexing_inputs(self,
                               spots=None,
                               spot_intensity=None,
                               spot_intensity_cutoff=0,
                               ):
        
        if spots is None and hasattr(self, 'spots_3D'):
            spots = self.spots_3D[['qx', 'qy', 'qz']].values
        else:
            err_str = ('Must specify 3D spots or rocking curve must '
                       + 'already have spots.')
            raise ValueError(err_str)
        
        if spot_intensity is None and hasattr(self, 'spots_3D'):
            spot_intensity = self.spots_3D['intensity'].values
        else:
            err_str = ('Must specify 3D spot intensities or rocking '
                       + 'curve must already have spot intensities.')
            raise ValueError(err_str)
        
        if len(spots) != len(spot_intensity):
            err_str = (f'Length of spots ({len(spot_intensity)}) does '
                       + 'not match lenght of spot_intensity '
                       + f'({len(spot_intensity)}).')
            raise RuntimeError(err_str)
        
        int_mask = self.get_vector_int_mask(
                    intensity=spot_intensity,
                    intensity_cutoff=spot_intensity_cutoff)
        
        return spots[int_mask], spot_intensity[int_mask]


    def index_best_grain(self,
                         near_q,
                         near_angle,
                         spots=None,
                         spot_intensity=None,
                         spot_intensity_cutoff=0,
                         phase=None,
                         method='pair_casting',
                         **kwargs
                         ):
        
        spots, spot_intensity = self._parse_indexing_inputs(
                                    spots,
                                    spot_intensity)

        qmask = QMask.from_XRDRockingCurve(self)

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
        if method.lower() in ['pair_casting']:
            (best_connection,
             best_qof) = pair_casting_index_best_grain(
                    spots,
                    phase,
                    near_q,
                    near_angle,
                    qmask,
                    degrees=self.polar_units == 'deg',
                    **kwargs)
        else:
            err_str = (f"Unknown method ({method}) specified. Only "
                       + "'pair_casting' is currently supported.")
            raise ValueError(err_str)
        
        # TODO: Update spots_3D dataframe to get hkl and grain number...
        return best_connection, best_qof


    def index_all_spots(self,
                        near_q,
                        near_angle,
                        spots=None,
                        spot_intensity=None,
                        spot_intensity_cutoff=0,
                        phase=None,
                        method='pair_casting',
                        **kwargs
                        ):
        
        spots, spot_intensity = self._parse_indexing_inputs(
                                    spots,
                                    spot_intensity)

        qmask = QMask.from_XRDRockingCurve(self)

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
        if method.lower() in ['pair_casting']:
            (best_connections,
             best_qofs) = pair_casting_index_full_pattern(
                    spots,
                    phase,
                    near_q,
                    near_angle,
                    qmask,
                    degrees=self.polar_units == 'deg',
                    **kwargs)
        else:
            err_str = (f"Unknown method ({method}) specified. Only "
                       + "'pair_casting' is currently supported.")
            raise ValueError(err_str)
        
        # TODO: Update spots_3D dataframe to get hkl and grain number...
        return best_connections, best_qofs


    # Strain math
    def get_strain_orientation(self):
        raise NotImplementedError()

    
    def get_zero_point_correction():
        raise NotImplementedError()

    ##########################
    ### Plotting Functions ###
    ##########################

    def plot_image_stack(self,
                         images=None,
                         slider_vals=None,
                         slider_label='Index',
                         vmin=None,
                         vmax=None,
                         return_plot=False,
                         **kwargs):

        if images is None:
            images = self.images.squeeze()

        if slider_vals is None:
            slider_vals = self.energy
            slider_label = 'Energy [keV]'
        
        (fig,
        ax,
        slider) = base_slider_plot(
            images,
            slider_vals=slider_vals,
            slider_label=slider_label,
            vmin=vmin,
            vmax=vmax,
            **kwargs)
        

        if return_plot:
            # Need a slider reference
            self.__slider = slider
            return fig, ax
        else:
            # Need a slider reference
            self.__slider = slider
            fig.show()
            

    def plot_3D_scatter(self,
                        q_vectors=None,
                        intensity=None,
                        edges=None,
                        skip=None,
                        return_plot=False,
                        **kwargs):

        if q_vectors is None:
            if hasattr(self, 'q_vectors'):
                q_vectors = self.q_vectors
            else:
                err_str = 'Must provide or already have q_vectors.'
                raise ValueError(err_str)
        
        if intensity is None:
            if (hasattr(self, 'intensity')
                and len(self.intensity) == len(q_vectors)):
                intensity = self.intensity
            else:
                intensity = np.zeros(len(q_vectors),
                                     dtype=q_vectors.dtype)
        
        if edges is None and hasattr(self, 'edges'):
            edges = self.edges

        fig, ax = plot_3D_scatter(q_vectors,
                                  intensity,
                                  skip=skip,
                                  edges=edges,
                                  **kwargs)

        if return_plot:
            return fig, ax
        else:
            fig.show()
    

    # Convenience wrapper of plot_3D_scatter to plot found 3D spots
    def plot_3D_spots(self,
                      spots_3D=None,
                      edges=None,
                      skip=None,
                      return_plot=False,
                      **kwargs):
        
        if spots_3D is None:
            if not hasattr(self, 'spots_3D'):
                err_str = 'Cannot plot 3D spots without spots.'
                raise AttributeError(err_str)
            else:
                spots_3D = self.spots_3D
        elif not isinstance(spots_3D, pd.core.frame.DataFrame):
            err_str = ('Spots must be given as Pandas DataFrame '
                       + f'not {type(spots_3D)}.')
            raise TypeError(err_str)
        
        q_vectors = spots_3D[['qx', 'qy', 'qz']]
        intensity = spots_3D['intensity']

        fig, ax = self.plot_3D_scatter(
                        q_vectors=q_vectors,
                        intensity=intensity,
                        edges=None,
                        skip=None,
                        return_plot=True,
                        alpha=1,
                        **kwargs
                        )

        if return_plot:
            return fig, ax
        else:
            fig.show()      


    # Broken for angle rocking curves
    def plot_3D_isosurfaces(self,
                            q_vectors=None,
                            intensity=None,
                            gridstep=0.01,
                            **kwargs):

        if q_vectors is None:
            if hasattr(self, 'q_vectors'):
                q_vectors = self.q_vectors
            else:
                err_str = 'Must provide or already have q_vectors.'
                raise ValueError(err_str)

        if intensity is None:
            if (hasattr(self, 'intensity')
                and len(self.intensity) == len(q_vectors)):
                intensity = self.intensity
            else:
                err_str = 'Must provide or already have intensity.'
                raise ValueError(err_str)

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
        energy_step = np.abs(np.mean(np.gradient(self.energy)))
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
        
    
    def plot_sampled_outline(self,
                             edges=None,
                             return_plot=False):

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

        ax.set_xlabel('qx [Å⁻¹]')
        ax.set_ylabel('qy [Å⁻¹]')
        ax.set_zlabel('qz [Å⁻¹]')
        ax.set_aspect('equal')

        if return_plot:
            return fig, ax
        else:
            fig.show()
