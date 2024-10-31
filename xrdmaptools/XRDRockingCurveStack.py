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
from xrdmaptools.io.hdf_io_rev import (
    load_xrdbase_hdf
)
from xrdmaptools.reflections.spot_blob_search import (
    find_blobs
)
from xrdmaptools.reflections.spot_blob_search_3D import (
    rsm_blob_search,
    rsm_spot_search
)
from xrdmaptools.plot.image_stack import base_slider_plot
from xrdmaptools.plot.volume import (
    plot_3D_scatter,
    plot_3D_isosurfaces
)


class XRDRockingCurveStack(XRDBaseScan):

    # Class variables
    _hdf_type = 'rsm'

    def __init__(self,
                 image_data=None,
                 image_shape=None,
                 map_shape=None,
                 sclr_dict=None,
                 **xrdbasekwargs
                 ):

        if (image_data is None
            and (map_shape is None
                 or image_shape is None)):
            raise ValueError('Must specify image_data, or image and map shapes.')
        
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
            elif data_ndim == 4: # This will work with standard hdf formatting
                if np.all(data_shape[:2] == 1):
                    err_str = (f'image_data shape is {data_shape}. '
                            + 'For 4D data, first two axes cannot both equal 1.')
                    raise ValueError(err_str)
                elif (data_shape[0] == 1 or data_shape[1] != 1):
                    image_data = image_data.swapaxes(0, 1)
                elif (data_shape[0] != 1 or data_shape[1] == 1):
                    pass
                else:
                    err_str = (f'image_data shape is {data_shape}. '
                            + 'For 4D data, one of the first two axes must equal 1.')
                    raise ValueError(err_str)
            else:
                err_str = (f'Unable to parse image_data with {data_ndim} '
                        + f'dimensions with shape {data_shape}.')
                raise ValueError(err_str)

            # Define num_images early for energy and wavelength setters
            #self.num_images = np.prod(image_data.shape[:-2])
            self.map_shape = image_data.shape[:2]
            # self.num_images = np.prod(self._image_shape)
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
            if all([
                all(np.round(i, 4) == np.round(self.energy[0], 4) for i in self.energy),
                all(np.round(i, 4) == np.round(self.theta[0], 4) for i in self.theta)
            ]):
                err_str = ('Rocking curves must be constructed by varying '
                        + 'energy/wavelength or theta. Given values are constant.')
                raise ValueError(err_str)
        
        # Re-write certain values:
        if self.hdf_path is not None:
            if self._dask_enabled:
                self.hdf[self._hdf_type].attrs['energy'] = self.energy
                self.hdf[self._hdf_type].attrs['wavelength'] = self.wavelength
                self.hdf[self._hdf_type].attrs['theta'] = self.theta
            else:
                with h5py.File(self.hdf_path, 'a') as f:
                    f[self._hdf_type].attrs['energy'] = self.energy
                    f[self._hdf_type].attrs['wavelength'] = self.wavelength
                    f[self._hdf_type].attrs['theta'] = self.theta

    # Overwrite parent function
    def __str__(self):
        ostr = (f'{self._hdf_type}:  scanid={self.scanid}, '
                + f'energy_range={min(self.energy):.3f}-{max(self.energy):.3f}, '
                + f'shape={self.images.shape}')
        return ostr
    
    # Modify parent function
    def __repr__(self):
        lines = XRDBaseScan.__repr__(self).splitlines(True)
        lines[4] = f'\tEnergy Range:\t\t{min(self.energy):.3f}-{max(self.energy):.3f} keV\n'
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
        # # Fill placeholders if energy is nothing
        # if (np.any(np.array(energy) is None)
        #     or np.any(np.isnan(energy))):
        #     self._energy = np.array([np.nan,]
        #                             * self.num_images)
        # # Proceed if object is subscriptable and not str or dict
        # elif (hasattr(energy, '__len__')
        #       and hasattr(energy, '__getitem__')
        #       and not isinstance(energy, (str, dict))):
        #     if len(energy) == self.num_images:
        #         self._energy = np.asarray(energy)
        #     # Assume single value is constant
        #     elif len(energy) == 1:
        #         self._energy = np.array([energy[0],]
        #                                 * self.num_images)
        #     else:
        #         err_str = ('Energy must have length '
        #                    + 'equal to number of images.')
        #         raise ValueError(err_str)
        # # Assume single value is constant
        # elif isinstance(energy, (int, float)):
        #     self._energy = np.array([energy,]
        #                             * self.num_images)
        # else:
        #     err_str = ('Unable to handle energy input. '
        #                + 'Provide subscriptable or value.')
        #     raise TypeError(err_str)
        
        self._wavelength = energy_2_wavelength(self._energy)

        # Propogate changes...
        if hasattr(self, 'ai') and self.ai is not None:
            self.ai.energy = self._energy[0]
            if hasattr(self, '_q_arr'):
                delattr(self, '_q_arr')
        
        # Re-write hdf values
        if hasattr(self, 'hdf_path') and self.hdf_path is not None:
            if self._dask_enabled:
                self.hdf[self._hdf_type].attrs['energy'] = self.energy
                self.hdf[self._hdf_type].attrs['wavelength'] = self.wavelength
            else:
                with h5py.File(self.hdf_path, 'a') as f:
                    f[self._hdf_type].attrs['energy'] = self.energy
                    f[self._hdf_type].attrs['wavelength'] = self.wavelength


    # A lot of parsing inputs
    @XRDBaseScan.wavelength.setter
    def wavelength(self, wavelength):
        self._parse_subscriptable_value(wavelength, 'wavelength')
        # # Fill placeholders if wavelength is nothing
        # if (np.any(np.array(wavelength) is None)        # # Fill placeholders if energy is nothing
        # if (np.any(np.array(energy) is None)
        #     or np.any(np.isnan(energy))):
        #     self._energy = np.array([np.nan,]
        #                             * self.num_images)
        # # Proceed if object is subscriptable and not str or dict
        # elif (hasattr(energy, '__len__')
        #       and hasattr(energy, '__getitem__')
        #       and not isinstance(energy, (str, dict))):
        #     if len(energy) == self.num_images:
        #         self._energy = np.asarray(energy)
        #     # Assume single value is constant
        #     elif len(energy) == 1:
        #         self._energy = np.array([energy[0],]
        #                                 * self.num_images)
        #     else:
        #         err_str = ('Energy must have length '
        #                    + 'equal to number of images.')
        #         raise ValueError(err_str)
        # # Assume single value is constant
        # elif isinstance(energy, (int, float)):
        #     self._energy = np.array([energy,]
        #                             * self.num_images)
        # else:
        #     err_str = ('Unable to handle energy input. '
        #                + 'Provide subscriptable or value.')
        #     raise TypeError(err_str)
        # # Proceed if object is subscriptable and not str or dict
        # elif (hasattr(wavelength, '__len__')
        #       and hasattr(wavelength, '__getitem__')
        #       and not isinstance(wavelength, (str, dict))):
        #     if len(wavelength) == self.num_images:
        #         self._wavelength = np.asarray(wavelength)
        #     # Assume single value is constant
        #     elif len(wavelength) == 1:
        #         self._wavelength = np.array([wavelength[0],]
        #                                     * self.num_images)
        #     else:
        #         err_str = ('Wavelength must have length '
        #                    + 'equal to number of images.')
        #         raise ValueError(err_str)
        # # Assume single value is constant
        # elif isinstance(wavelength, (int, float)):
        #     self._wavelength = np.array([wavelength,]
        #                                 * self.num_images)
        # else:
        #     err_str = ('Unable to handle wavelength input. '
        #                + 'Provide subscriptable or value.')
        #     raise TypeError(err_str)
        
        self._energy = wavelength_2_energy(self._wavelength)

        # Propogate changes...
        if hasattr(self, 'ai') and self.ai is not None:
            self.ai.energy = self._energy[0]
            if hasattr(self, '_q_arr'):
                delattr(self, '_q_arr')

        # Re-write hdf values
        if hasattr(self, 'hdf_path') and self.hdf_path is not None:
            if self._dask_enabled:
                self.hdf[self._hdf_type].attrs['energy'] = self.energy
                self.hdf[self._hdf_type].attrs['wavelength'] = self.wavelength
            else:
                with h5py.File(self.hdf_path, 'a') as f:
                    f[self._hdf_type].attrs['energy'] = self.energy
                    f[self._hdf_type].attrs['wavelength'] = self.wavelength

    
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

        # Re-write hdf values
        if hasattr(self, 'hdf_path') and self.hdf_path is not None:
            if self._dask_enabled:
                self.hdf[self._hdf_type].attrs['theta'] = self.theta
            else:
                with h5py.File(self.hdf_path, 'a') as f:
                    f[self._hdf_type].attrs['theta'] = self.theta


    # Full q-vector, not just magnitude
    @XRDBaseScan.q_arr.getter
    def q_arr(self):
        if hasattr(self, '_q_arr'):
            return self._q_arr
        elif not hasattr(self, 'ai'):
            raise RuntimeError('Cannot calculate q-space without calibration.')
        else:
            self._q_arr = np.empty((self.num_images, 3, *self.image_shape),
                                    dtype=self.dtype)
            for i, wavelength in tqdm(enumerate(self.wavelength),
                                    total=self.num_images):
                q_arr = get_q_vect(self.tth_arr,
                                   self.chi_arr,
                                   wavelength=wavelength,
                                   degrees=self.polar_units == 'deg')
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
                scanid=-1,
                broker='manual',
                filedir=None,
                filename=None,
                poni_file=None,
                save_hdf=True,
                repair_method='fill'):
        
        if isinstance(scanid, str):
            scantype = 'EXTENDED_ENERGY_RC'

            (data_dict,
             scan_md,
             xrd_dets) = load_extended_energy_rc_data(
                            start_id=scanid[:6],
                            end_id=scanid[-6:],
                            returns=['xrd_dets']
                            )
        else:
            # Get scantype information from check...
            if broker == 'manual':
                temp_broker = 'tiled'
            else:
                temp_broker = broker
            scantype = get_scantype(scanid,
                                    broker=temp_broker)

            if scantype in ['ENERGY_RC', 'ANGLE_RC']:
                load_func = load_step_rc_data
            elif scantype == 'XRF_FLY':
                load_func = load_flying_angle_rc_data    
            else:
                err_str = f'Unable to handle scan type of {scantype}.'
                raise RuntimeError(err_str)
        
            (data_dict,
             scan_md,
             xrd_dets) = load_func(
                            scanid=scanid,
                            returns=['xrd_dets'],
                                )
        
        xrd_data = [data_dict[f'{xrd_det}_image'] for xrd_det in xrd_dets]

        # Make scaler dictionary
        sclr_keys = ['i0', 'i0_time', 'im', 'it']
        sclr_dict = {key:value for key, value in data_dict.items() if key in sclr_keys}

        if 'null_map' in data_dict.keys():
            null_map = data_dict['null_map']
        else:
            null_map = None

        if len(xrd_data) > 1:
            filenames = [f'scan{scan_md["scan_id"]}_{det}_xrd.h5' for det in xrd_dets]
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

        rocking_curves = []
        for i, xrd_data_i in enumerate(xrd_data):
            rc = cls(
                    scanid=scan_md['scan_id'],
                    wd=filedir,
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
                    null_map=null_map
                    )
            
            rocking_curves.append(rc)

        print(f'{cls.__name__} loaded!')
        if len(rocking_curves) > 1:
            return tuple(rocking_curves)
        else:
            return rocking_curves[0]
        

    def load_parameters_from_txt(self,
                                 filename=None,
                                 filedir=None):
        
        if filedir is None:
            filedir = self.wd

        if filename is None:
            mask = [(str(self.scanid) in file
                    and 'parameters' in file)
                    for file in os.listdir(filedir)]
            filename = np.asarray(os.listdir(filedir))[mask][0]

        out = np.genfromtxt(f'{filedir}{filename}')
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
                               filedir=None):
        if filedir is None:
            filedir = self.wd

        if filename is None:
            mask = [(str(self.scanid) in file
                    and 'metadata' in file)
                    for file in os.listdir(filedir)]
            filename = filename = np.asarray(os.listdir(filedir))[mask][0]

        with open(f'{filedir}{filename}', 'r') as f:
            json_str = f.read()
            md = json.loads(json_str)
        
        base_md = {key:value for key, value in md.items() if key in ['scan_id', 'theta', 'dwell']}
        extra_md = {key:value for key, value in md.items() if key not in base_md.keys()}
        if 'scan_id' in base_md.keys():
            base_md['scanid'] = base_md['scan_id']
            del base_md['scan_id']
        
        base_md['scanid'] = f"{np.min(base_md['scanid'])}-{np.max(base_md['scanid'])}"
        
        for key, value in base_md.items():
            setattr(self, key, value)
        setattr(self, 'extra_metadata', extra_md)
        

    ################################################
    ### Rocking Curve to 3D Reciprocal Space Map ###
    ################################################

    def vectorize_images(self,
                         override_blob_search=False,
                         rewrite_shape=False):

        if not override_blob_search and not hasattr(self, 'blob_masks'):
            raise ValueError('Must find 2D blobs first to avoid overly large datasets.')
        
        edges = ([[] for _ in range(12)])

        # Reserve memory, a little faster and throws errors sooner
        q_vectors = np.zeros((np.sum(self.blob_masks), 3), dtype=self.dtype)

        print('Vectorizing images...')
        filled_indices = 0
        for i in tqdm(range(self.num_images)):
        # for i, wavelength in tqdm(enumerate(self.wavelength), total=self.num_images):
            wavelength = self.wavelength[i]
            angle = self.theta[i]

            q_arr = get_q_vect(self.tth_arr,
                               self.chi_arr,
                               wavelength=wavelength,
                               degrees=self.polar_units == 'deg'
                               ).astype(self.dtype)
            
            next_indices = np.sum(self.blob_masks[i])
            # Fill q_vectors from q_arr
            for idx in range(3):
                q_vectors[filled_indices : filled_indices + next_indices,
                        idx] = q_arr[idx][self.blob_masks[i].squeeze()]
            filled_indices += next_indices

            # Find edges
            if i == 0:
                edges[4] = q_arr[:, 0].T
                edges[5] = q_arr[:, -1].T
                edges[6] = q_arr[:, :, 0].T
                edges[7] = q_arr[:, :, -1].T
            elif i == len(self.wavelength) - 1:
                edges[8] = q_arr[:, 0].T
                edges[9] = q_arr[:, -1].T
                edges[10] = q_arr[:, :, 0].T
                edges[11] = q_arr[:, :, -1].T
            else: # Corners
                edges[0].append(q_arr[:, 0, 0])
                edges[1].append(q_arr[:, 0, -1])
                edges[2].append(q_arr[:, -1, 0])
                edges[3].append(q_arr[:, -1, -1])
        
        for i in range(4):
            edges[i] = np.asarray(edges[i])
        
        # Assign useful variables
        self.edges = edges
        self.q_vectors = q_vectors
        self.intensity = self.images[self.blob_masks] # A bit redundant; copies data
        # Write to hdf
        self.save_vectorization(q_vectors=self.q_vectors,
                                intensity=self.intensity,
                                edges=self.edges,
                                rewrite_shape=rewrite_shape)


    def save_vectorization(self,
                           q_vectors=None,
                           intensity=None,
                           edges=None,
                           rewrite_shape=False):

        if self.hdf_path is not None:
            print('Saving vectorized image data...')
            # Open hdf flag
            keep_hdf = True
            if self.hdf is None:
                self.hdf = h5py.File(self.hdf_path, 'a')
                keep_hdf = False

            # Write data to hdf
            vect_grp = self.hdf[self._hdf_type].require_group('vectorized_data')
            vect_grp.attrs['time_stamp'] = ttime.ctime()

            # Save q_vectors and intensity
            for attr, attr_name in zip([q_vectors, intensity],
                                       ['q_vectors', 'intensity']):

                # Check for values/attributes. Must have both q_vectors and intensity
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
                    
                    # Avoid creating new data, unless specified explicitly
                    elif rewrite_shape:
                        warn_str = (f'WARNING: Rewriting {attr_name} '
                                   + 'dataset shape. This could bloat '
                                   + 'overall file size.')
                        print(warn_str)
                        del vect_grp['attr_name']
                        dset = vect_grp.require_dataset(
                            attr_name,
                            data=attr,
                            shape=attr.shape,
                            dtype=attr.dtype)
            
            # Check for edge information
            if edges is None:
                if hasattr(self, 'edges') and self.edges is not None:
                    edges = self.edges
                else:
                    print('WARNING: No edges given or found. Edges will not be saved.')

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
                        elif (dset.shape != edge.shape
                              and dset.dtype == edge.dtype):
                              dset[...] = edge.astype(dset.dtype)
                        else:
                            self.hdf.close()
                            self.hdf = None
                            err_str = (f'Edge shape for {edge_title} '
                                       + f'({edge.shape}) does not '
                                       + 'match datset shape '
                                       + f'({dset.shape}). '
                                       + 'This should not '
                                       + 'have happened.')
                            # Shape changes should not happen. Throw error
                            raise RuntimeError(err_str)

            # Close hdf and reset attribute
            if not keep_hdf:
                self.hdf.close()
                self.hdf = None


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
        
        self.blob_masks = np.asarray(blob_mask_list).reshape(self.shape)

        # Save blob_masks to hdf
        self.save_images(images='blob_masks',
                         title='_blob_masks',
                         units='bool',
                         extra_attrs={'threshold_method' : threshold_method,
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
        # print(self.q_vectors.shape)
        # print(int_mask.shape)
        # print(self.q_vectors[int_mask].shape)
                    
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
                      save_to_hdf=False):

        if not hasattr(self, 'q_vectors') or not hasattr(self, 'intensity'):
            raise AttributeError('Cannot perform 3D spot search without first vectorizing images.')

        int_mask = self.get_vector_int_mask(
                        intensity_cutoff=intensity_cutoff)

        (spot_labels,
         spots,
         label_ints) = rsm_spot_search(self.q_vectors[int_mask],
                                       self.intensity[int_mask],
                                       nn_dist=nn_dist,
                                       significance=significance,
                                       subsample=subsample)

        tth, chi, wavelength = q_2_polar(spots, degrees=(self.polar_units == 'deg'))

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

        # Write to hdffind_blobs(
        if save_to_hdf:
            self.save_3D_spots()
            self.save_vector_information(
                self.spot_labels,
                'spot_labels',
                extra_attrs={'spot_int_cutoff' : intensity_cutoff})


    # Analog of 2D spots from xrdmap
    def save_3D_spots(self, extra_attrs=None):
        # Save spots to hdf
        if self.hdf_path is not None:
            print('Saving 3D spots to hdf...')

            # Open hdf flag
            keep_hdf = True
            if self.hdf is None:
                self.hdf = h5py.File(self.hdf_path, 'a')
                keep_hdf = False

            # Save to hdf
            self.close_hdf()
            self.spots_3D.to_hdf(self.hdf_path,
                            key=f'{self._hdf_type}/reflections/spots_3D',
                            format='table')

            if extra_attrs is not None:
                self.open_hdf()
                for key, value in extra_attrs.items():
                    self.hdf[f'{self._hdf_type}/reflections/spots_3D'].attrs[key] = value

            if keep_hdf:
                self.open_hdf()
            else:
                self.close_hdf()
            
            print('done!')


    def save_vector_information(self,
                                data,
                                title,
                                extra_attrs=None):

        if self.hdf_path is not None:
            # Open hdf flag
            keep_hdf = True
            if self.hdf is None:
                self.hdf = h5py.File(self.hdf_path, 'a')
                keep_hdf = False
            
            # Get vector group
            vect_grp = self.hdf[self._hdf_type].require_group('vectorized_data')

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

            # Close hdf and reset attribute
            if not keep_hdf:
                self.hdf.close()
                self.hdf = None

    ###########################################
    ### Indexing, Corrections, and Analysis ###
    ###########################################

    def index_spots(self,
                    spots=None,
                    spot_intensity_cut_off=0,
                    method='pair_casting',
                    **kwargs
                    ):
        raise NotImplementedError()

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
            return fig, ax, slider # Not returning the slider may break it...
        else:
            fig.show()
            self.__slider = slider # Need a reference...


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
                raise AttributeError('Cannot plot 3D spots without spots.')
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
                        return_plot=False,
                        **kwargs
                        )

        if return_plot:
            return fig, ax
        else:
            fig.show()

    
    # def plot_3D_scatter(self,
    #                     skip=None,
    #                     q_vectors=None,
    #                     intensity=None,
    #                     edges=None,
    #                     return_plot=False,
    #                     **kwargs
    #                     ):

    #     fig, ax = plt.subplots(1, 1, 
    #                            figsize=(5, 5),
    #                            dpi=200,
    #                            subplot_kw={'projection':'3d'})

    #     if q_vectors is None:
    #         if hasattr(self, 'q_vectors'):
    #             q_vectors = self.q_vectors
    #         else:
    #             err_str = 'Must provide or already have q_vectors.'
    #             raise ValueError(err_str)
    #     else:
    #         q_vectors = np.asarray(q_vectors)
        
    #     if intensity is None:
    #         if (hasattr(self, 'intensity')
    #             and len(self.intensity) == len(q_vectors)):
    #             intensity = self.intensity
    #         else:
    #             intensity = np.zeros(len(q_vectors),
    #                                  dtype=q_vectors.dtype)
    #     else:
    #         intensity = np.asarray(intensity)
        
    #     if edges is None and hasattr(self, 'edges'):
    #         edges = self.edges
    #     else:
    #         edges = []
        
    #     if skip is None:
    #         skip = np.round(len(q_vectors) / 5000, 0).astype(int) # skips to about 5000 points
    #         if skip == 0:
    #             skip = 1
        
    #     kwargs.setdefault('s', 1)
    #     kwargs.setdefault('cmap', 'viridis')
    #     kwargs.setdefault('alpha', 0.1)

    #     if 'title' in kwargs:
    #         title = kwargs.pop('title')
    #         ax.set_title(title)

    #     ax.scatter(*q_vectors[::skip].T, c=intensity[::skip], **kwargs)

    #     for edge in edges:
    #         ax.plot(*edge.T, c='gray', lw=1)

    #     ax.set_xlabel('qx [Å⁻¹]')
    #     ax.set_ylabel('qy [Å⁻¹]')
    #     ax.set_zlabel('qz [Å⁻¹]')
    #     ax.set_aspect('equal')

    #     if return_plot:
    #         return fig, ax
    #     else:
    #         fig.show()        


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
                            degrees=self.polar_units == 'deg')
        energy = wavelength_2_energy(wavelength)

        # Assumes energy is rocking axis...
        energy_step = np.abs(np.mean(np.gradient(self.energy)))
        min_energy = np.min(self.energy)
        max_energy = np.max(self.energy)

        low_mask = energy <= min_energy + energy_step
        high_mask = energy >= max_energy - energy_step

        # If there are bounded pixels, padd with zeros
        # print(np.sum([low_mask, high_mask]))
        if np.sum([low_mask, high_mask]) > 0:
            low_qs = get_q_vect(
                        tth[low_mask],
                        chi[low_mask],
                        wavelength=energy_2_wavelength(min_energy
                                                    - energy_step),
                        degrees=self.polar_units == 'deg'
                        ).astype(self.dtype).T
            
            high_qs = get_q_vect(
                        tth[high_mask],
                        chi[high_mask],
                        wavelength=energy_2_wavelength(max_energy
                                                    + energy_step),
                        degrees=self.polar_units == 'deg'
                        ).astype(self.dtype).T

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


    # def plot_3D_isosurfaces(self,
    #                         q_vectors=None,
    #                         intensity=None,
    #                         gridstep=0.01,
    #                         isomin=None,
    #                         isomax=None,
    #                         min_offset=None,
    #                         max_offset=None,
    #                         opacity=0.1,
    #                         surface_count=20,
    #                         colorscale='viridis',
    #                         renderer='browser'):
    #     if q_vectors is None:
    #         if hasattr(self, 'q_vectors'):
    #             q_vectors = self.q_vectors
    #         else:
    #             err_str = 'Must provide or already have q_vectors.'
    #             raise ValueError(err_str)
    #     else:
    #         q_vectors = np.asarray(q_vectors)

    #     if intensity is None:
    #         if (hasattr(self, 'intensity')
    #             and len(self.intensity) == len(q_vectors)):
    #             intensity = self.intensity
    #         else:
    #             err_str = 'Must provide or already have intensity.'
    #             raise ValueError(err_str)
    #     else:
    #         intensity = np.asarray(intensity)

    #     # Copy values; they will be modified.
    #     plot_qs = q_vectors.copy()
    #     plot_ints = intensity.copy()
    #     min_int = np.min(plot_ints)

    #     # Check given q_ext
    #     for axis in range(3):
    #         q_ext = (np.max(q_vectors[:, axis])
    #                 - np.min(q_vectors[:, axis]))
    #         if  q_ext < gridstep:
    #             err_str = (f'Gridstep ({gridstep}) is smaller than '
    #                     + f'q-vectors range along axis {axis} '
    #                     + f'({q_ext:.4f}).')
    #             raise ValueError(err_str)

    #     tth, chi, wavelength = q_2_polar(
    #                         plot_qs,
    #                         degrees=self.polar_units == 'deg')
    #     energy = wavelength_2_energy(wavelength)
    #     # print(energy)

    #     # Assumes energy is rocking axis...
    #     energy_step = np.abs(np.mean(np.gradient(self.energy)))
    #     min_energy = np.min(self.energy)
    #     max_energy = np.max(self.energy)
    #     # print(energy_step)

    #     low_mask = energy <= min_energy + energy_step
    #     high_mask = energy >= max_energy - energy_step
    #     # print(f'{np.min(self.energy)=}')
    #     # print(f'{np.max(self.energy)=}')

    #     # If there are bounded pixels, padd with zeros
    #     # print(np.sum([low_mask, high_mask]))
    #     if np.sum([low_mask, high_mask]) > 0:
    #         low_qs = get_q_vect(
    #                     tth[low_mask],
    #                     chi[low_mask],
    #                     wavelength=energy_2_wavelength(min_energy
    #                                                 - energy_step),
    #                     degrees=self.polar_units == 'deg'
    #                     ).astype(self.dtype).T
            
    #         high_qs = get_q_vect(
    #                     tth[high_mask],
    #                     chi[high_mask],
    #                     wavelength=energy_2_wavelength(max_energy
    #                                                 + energy_step),
    #                     degrees=self.polar_units == 'deg'
    #                     ).astype(self.dtype).T

    #         # Extend plot qs and ints
    #         plot_qs = np.vstack([plot_qs, low_qs, high_qs])
    #         plot_ints = np.hstack([
    #                         plot_ints,
    #                         np.ones(low_mask.sum()) * min_int,
    #                         np.ones(high_mask.sum()) * min_int])
        
    #     # return plot_qs, plot_ints

    #     # Interpolate data for isosurface generation
    #     (x_grid,
    #     y_grid,
    #     z_grid,
    #     int_grid) = map_2_grid(plot_qs,
    #                         plot_ints,
    #                         gridstep=gridstep)

    #     gen_offset = ((np.max(int_grid) - np.min(int_grid))
    #                     / (2 * surface_count))
    #     if isomin is None:
    #         if min_offset is None:
    #             isomin = np.min(int_grid) + gen_offset
    #         else:
    #             isomin = np.min(int_grid) + min_offset
        
    #     if isomax is None:
    #         if max_offset is None:
    #             isomax = np.max(int_grid) - gen_offset
    #         else:
    #             isomax = np.max(int_grid) - max_offset
        
    #     # Generate isosurfaces from plotly graph object
    #     data = go.Volume(
    #         x=x_grid.flatten(),
    #         y=y_grid.flatten(),
    #         z=z_grid.flatten(),
    #         value=int_grid.flatten(),
    #         isomin=isomin,
    #         isomax=isomax,
    #         opacity=opacity,
    #         surface_count=surface_count,
    #         colorscale=colorscale
    #     )

    #     # Find data extent
    #     x_range = np.max(x_grid) - np.min(x_grid)
    #     y_range = np.max(y_grid) - np.min(y_grid)
    #     z_range = np.max(z_grid) - np.min(z_grid)

    #     # Generate figure and plot
    #     fig = go.Figure(data=data)
    #     fig.update_layout(scene_aspectmode='manual',
    #                         scene_aspectratio=dict(
    #                         x=x_range,
    #                         y=y_range,
    #                         z=z_range))
    #     fig.show(renderer=renderer)
        
    
    def plot_sampled_outline(self,
                             edges=None,
                             return_plot=False):

        if edges is None:
            if hasattr(self, 'edges'):
                edges = self.edges
            else:
                err_st= ('Cannot plot sampled volume '
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




# def plot_3D_isosurfaces(self,
#                         q_vectors=None,
#                         intensity=None,
#                         gridstep=0.005,
#                         isomin=None,
#                         isomax=None,
#                         min_offset=None,
#                         max_offset=None,
#                         opacity=0.1,
#                         surface_count=20,
#                         colorscale='viridis',
#                         renderer='browser'):

#     if q_vectors is None:
#         if hasattr(self, 'q_vectors'):
#             q_vectors = self.q_vectors
#         else:
#             err_str = 'Must provide or already have q_vectors.'
#             raise ValueError(err_str)
#     else:
#         q_vectors = np.asarray(q_vectors)

#     if intensity is None:
#         if (hasattr(self, 'intensity')
#             and len(self.intensity) == len(q_vectors)):
#             intensity = self.intensity
#         else:
#             err_str = 'Must provide or already have intensity.'
#             raise ValueError(err_str)
#     else:
#         intensity = np.asarray(intensity)

#     # Copy values; they will be modified.
#     plot_qs = q_vectors.copy()
#     plot_ints = intensity.copy()
#     min_int = np.min(plot_ints)

#     # Check given q_ext
#     for axis in range(3):
#         q_ext = (np.max(q_vectors[:, axis])
#                  - np.min(q_vectors[:, axis]))
#         if  q_ext < gridstep:
#             err_str = (f'Gridstep ({gridstep}) is smaller than '
#                        + f'q-vectors range along axis {axis} '
#                        + f'({q_ext:.4f}).')
#             raise ValueError(err_str)

#     tth, chi, wavelength = q_2_polar(
#                         plot_qs,
#                         degrees=self.polar_units == 'deg')
#     energy = wavelength_2_energy(wavelength)
#     # print(energy)

#     # Assumes energy is rocking axis...
#     energy_step = np.abs(np.mean(np.gradient(self.energy)))
#     min_energy = np.min(self.energy)
#     max_energy = np.max(self.energy)
#     # print(energy_step)

#     low_mask = energy <= min_energy + energy_step
#     high_mask = energy >= max_energy - energy_step
#     # print(f'{np.min(self.energy)=}')
#     # print(f'{np.max(self.energy)=}')

#     # If there are bounded pixels, padd with zeros
#     # print(np.sum([low_mask, high_mask]))
#     if np.sum([low_mask, high_mask]) > 0:
#         low_qs = get_q_vect(
#                     tth[low_mask],
#                     chi[low_mask],
#                     wavelength=energy_2_wavelength(min_energy
#                                                    - energy_step),
#                     degrees=self.polar_units == 'deg'
#                     ).astype(self.dtype).T
        
#         high_qs = get_q_vect(
#                     tth[high_mask],
#                     chi[high_mask],
#                     wavelength=energy_2_wavelength(max_energy
#                                                    + energy_step),
#                     degrees=self.polar_units == 'deg'
#                     ).astype(self.dtype).T

#         # Extend plot qs and ints
#         plot_qs = np.vstack([plot_qs, low_qs, high_qs])
#         plot_ints = np.hstack([
#                         plot_ints,
#                         np.ones(low_mask.sum()) * min_int,
#                         np.ones(high_mask.sum()) * min_int])
    
#     # return plot_qs, plot_ints

#     # Interpolate data for isosurface generation
#     (x_grid,
#     y_grid,
#     z_grid,
#     int_grid) = map_2_grid(plot_qs,
#                            plot_ints,
#                            gridstep=gridstep)

#     gen_offset = ((np.max(int_grid) - np.min(int_grid))
#                     / (2 * surface_count))
#     if isomin is None:
#         if min_offset is None:
#             isomin = np.min(int_grid) + gen_offset
#         else:
#             isomin = np.min(int_grid) + min_offset
    
#     if isomax is None:
#         if max_offset is None:
#             isomax = np.max(int_grid) - gen_offset
#         else:
#             isomax = np.max(int_grid) - max_offset
    
#     # Generate isosurfaces from plotly graph object
#     data = go.Volume(
#         x=x_grid.flatten(),
#         y=y_grid.flatten(),
#         z=z_grid.flatten(),
#         value=int_grid.flatten(),
#         isomin=isomin,
#         isomax=isomax,
#         opacity=opacity,
#         surface_count=surface_count,
#         colorscale=colorscale
#     )

#     # Find data extent
#     x_range = np.max(x_grid) - np.min(x_grid)
#     y_range = np.max(y_grid) - np.min(y_grid)
#     z_range = np.max(z_grid) - np.min(z_grid)

#     # Generate figure and plot
#     fig = go.Figure(data=data)
#     fig.update_layout(scene_aspectmode='manual',
#                         scene_aspectratio=dict(
#                         x=x_range,
#                         y=y_range,
#                         z=z_range))
#     fig.show(renderer=renderer)