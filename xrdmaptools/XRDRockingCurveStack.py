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
import matplotlib.pyplot as plt

# from .XRDMap import XRDMap
# from .utilities.utilities import timed_iter

from xrdmaptools.XRDBaseScan import XRDBaseScan
from xrdmaptools.utilities.math import (
    energy_2_wavelength,
    wavelength_2_energy
)
from xrdmaptools.geometry.geometry import (
    get_q_vect,
    q_2_polar
)
from xrdmaptools.io.db_io import (
    get_scantype,
    load_energy_rc_data,
    load_extended_energy_rc_data,
    load_angle_rc_data,
    load_flying_angle_rc_data
)
from xrdmaptools.io.hdf_io_rev import (
    load_xrdbase_hdf
    )
from xrdmaptools.reflections.spot_blob_search import (
    find_blobs
    )
from xrdmaptools.reflections.spot_blob_search_3d import (
    rsm_blob_search,
    rsm_spot_search
)


class XRDRockingCurveStack(XRDBaseScan):

    # Class variables
    _hdf_type = 'rsm'

    def __init__(self,
                 image_data=None,
                 sclr_dict=None,
                 **xrdbasekwargs
                 ):
        
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
        self.num_images = np.prod(image_data.shape[:-2])
        
        # Parse sclr_dict into useble format
        if sclr_dict is not None and isinstance(sclr_dict, dict):
            for key, value in sclr_dict.items():
                sclr_dict[key] = value.reshape(image_data.shape[:2])

        XRDBaseScan.__init__(
            self,
            image_data=image_data,
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
        lines[4] = f'\tEnergy Range:\t\t{min(self.energy):.3f}-{max(self.energy):.3f} keV'
        ostr = '\n'.join(lines)
        return ostr


    #############################
    ### Re-Written Properties ###
    #############################

    @XRDBaseScan.energy.setter
    def energy(self, energy):
        if np.any(energy is None) or np.any(np.isnan(energy)):
            self._energy = [np.nan,] * self.num_images
            self._wavelength = [np.nan,] * self.num_images
        elif isinstance(energy, (list, np.ndarray)):
            if len(energy) != self.num_images:
                raise ValueError('Energy must have length equal to number of images.')
            else:
                self._energy = list(energy)
                self._wavelength = [energy_2_wavelength(energy_i)
                                    for energy_i in self._energy]
        elif not isinstance(energy, list):
            self._energy = [energy,] * self.num_images
            self._wavelength = [energy_2_wavelength(energy_i)
                                for energy_i in self._energy]
        else:
            raise RuntimeError('Unable to handle energy input. Provide list or value.')

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


    @XRDBaseScan.wavelength.setter
    def wavelength(self, wavelength):
        if np.any(wavelength is None) or np.any(np.isnan(wavelength)):
            self._wavelength = [np.nan,] * self.num_images
            self._energy = [np.nan,] * self.num_images
        elif isinstance(wavelength, (list, np.ndarray)):
            if len(wavelength) != self.num_images:
                raise ValueError('Wavelength must have length equal to number of images.')
            else:
                self._wavelength = list(wavelength)
                self._energy = [wavelength_2_energy(wavelength_i)
                                for wavelength_i in self._wavelength]
        elif not isinstance(wavelength, list):
            self._wavelength = [wavelength,] * self.num_images
            self._energy = [wavelength_2_energy(wavelength_i)
                            for wavelength_i in self._wavelength]
        else:
            raise RuntimeError('Unable to handle wavelength input. Provide list or value.')

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
        if np.any(theta is None) or np.any(np.isnan(theta)):
            print('WARNING: No theta value provided. Assuming 0 deg.')
            self._theta = [0,] * self.num_images
        elif isinstance(theta, (list, np.ndarray)):
            if len(theta) != self.num_images:
                raise ValueError('Theta must have length equal to number of images.')
            else:
                self._theta = list(theta)
        elif not isinstance(theta, list):
            self._theta = [theta,] * self.num_images
        else:
            raise RuntimeError('Unable to handle theta input. Provide list or value.')

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

    # This currently breaks integrations, which are not particularly useful...
    def set_calibration(self, *args, **kwargs):
        super().set_calibration(*args,
                                energy=self.energy[0],
                                **kwargs)    

    # Convenience function
    def load_map_parameters():
        raise NotImplementedError()
    
    # q_arr might get called wrong
    def initial_spot_analysis():
        raise NotImplementedError()
    
    # Replacement for pixel_spots()
    def selected_spots(self,
                       index=None,
                       energy=None,
                       wavelength=None,
                       theta=None):
        raise NotImplementedError()
    
    # ImageMap overrides, must be called explicitly in __init__
    # Override of ImageMap absorption correction
    def updated_apply_absorption_correction():
        raise NotImplementedError()

    def updated_save_labels():
        raise NotImplementedError()

     
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
        else:
            # Get scantype information from check...
            if broker == 'manual':
                temp_broker = 'tiled'
            else:
                temp_broker = broker
            scantype = get_scantype(scanid,
                                    broker=temp_broker)

        if scantype == 'ENERGY_RC':
            load_func = load_energy_rc_data
        elif scantype == 'EXTENDED_ENERGY_RC':
            load_func = load_extended_energy_rc_data
        elif scantype == 'ANGLE_RC':
            load_func = load_angle_rc_data
        elif scantype == 'XRF_FLY':
            load_func = load_flying_angle_rc_data
        else:
            err_str = f'Unable to handle scan type of {scantype}.'
            raise RuntimeError(err_str)
        
        data_dict, scan_md, xrd_dets = load_func(
                              scanid=scanid,
                              broker=broker,
                              returns=['xrd_dets'],
                              repair_method=repair_method
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
                    energy=scan_md['energy'],
                    dwell=scan_md['dwell'],
                    theta=scan_md['theta'],
                    poni_file=poni_file,
                    sclr_dict=sclr_dict,
                    beamline='5-ID (SRX)',
                    facility='NSLS-II',
                    time_stamp=scan_md['time_str'],
                    extra_metadata=None,
                    save_hdf=save_hdf,
                    null_map=null_map
                    )
            
            rocking_curves.append(rc)

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
                         override=False):

        if not override and not hasattr(self, 'blob_masks'):
            raise ValueError('Must find 2D blobs first to avoid overly large datasets.')
        
        edges = ([[] for _ in range(12)])

        # Resever memory, a little faster and throws errors sooner
        q_vectors = np.zeros((np.sum(self.blob_masks), 3), dtype=self.dtype)

        filled_indices = 0
        for i, wavelength in tqdm(enumerate(self.wavelength), total=self.num_images):
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


    #######################
    ### Blobs and Spots ###
    #######################

    # Same as XRDMap.find_blobs()
    def find_2D_blobs(self,
                      threshold_method='minimum',
                      multiplier=5,
                      size=3,
                      expansion=10,
                      override_rescale=False):
    
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
        self.save_images(images=self.blob_masks,
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
                      override=False):

        if not hasattr(self, 'q_vectors') or not hasattr(self, 'intensity'):
            raise AttributeError('Cannot performe 3D spot search without first vectorizing images.')

        int_mask = (self.intensity
                    >= np.min(self.intensity) + intensity_cutoff
                    * (np.max(self.intensity) - np.min(self.intensity)))
                    
        labels = rsm_blob_search(self.q_vectors[int_mask],
                                 max_dist=max_dist,
                                 max_neighbors=max_neighbors,
                                 subsample=subsample)
        
        self.blob_labels = labels
        self.blob_int_mask = int_mask
        
    

    def find_3D_spots(self,
                      nn_dist=0.005,
                      significance=0.1,
                      subsample=1,
                      intensity_cutoff=0,
                      label_int_method='mean'
                      ):

        if not hasattr(self, 'q_vectors') or not hasattr(self, 'intensity'):
            raise AttributeError('Cannot performe 3D spot search without first vectorizing images.')

        int_mask = (self.intensity
                    >= np.min(self.intensity) + intensity_cutoff
                    * (np.max(self.intensity) - np.min(self.intensity)))

        (spot_labels,
        spots,
        label_ints) = rsm_spot_search(self.q_vectors[int_mask],
                                        self.intensity[int_mask],
                                        nn_dist=nn_dist,
                                        significance=significance,
                                        subsample=subsample)

        tth, chi, wavelength = q_2_polar(spots, degrees=(rsm.polar_units == 'deg'))

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

        # Save 3d spots similar to 3d spots
        spots_3d = pd.DataFrame.from_dict(temp_dict)
        rsm.spots_3d = spots_3d
        del temp_dict, spots, label_ints

        # Information for rebuilding spots
        # from vectorized images
        self.spot_labels = spot_labels
        self.spot_int_mask = int_mask


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
                         return_plot=False):
        raise NotImplementedError()
        
        fig, ax = base_slider_plot()

        if return_plot:
            return fig, ax
        else:
            fig.show()

    
    def plot_3D_scatter(self,
                        skip=None,
                        q_vectors=None,
                        intensity=None,
                        edges=None,
                        return_plot=False,
                        **kwargs
                        ):

        fig, ax = plt.subplots(1, 1, 
                               figsize=(5, 5),
                               dpi=200,
                               subplot_kw={'projection':'3d'})

        if q_vectors is None:
            if hasattr(self, 'q_vectors'):
                q_vectors = self.q_vectors
            else:
                err_str = 'Must provide or already have q_vectors.'
                raise ValueError(err_str)
        else:
            q_vectors = np.asarray(q_vectors)
        
        if intensity is None:
            if hasattr(self, 'intensity'):
                intensity = self.intensity
            else:
                err_str = 'Must provide or already have intensity.'
                raise ValueError(err_str)
        else:
            intensity = np.asarray(intensity)
        
        if edges is None and hasattr(self, 'edges'):
            edges = self.edges
        else:
            edges = []
        
        if skip is None:
            skip = np.round(len(q_vectors) / 5000, 0).astype(int) # skips to about 5000 points
            if skip == 0:
                skip = 1
        
        kwargs.setdefault('s', 1)
        kwargs.setdefault('cmap', 'viridis')
        kwargs.setdefault('alpha', 0.1)

        if 'title' in kwargs:
            title = kwargs.pop('title')
            ax.set_title(title)

        ax.scatter(*q_vectors[::skip].T, c=intensity[::skip], **kwargs)

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


    def plot_isosurfaces_volume(self,):
        raise NotImplementedError()

    
    def plot_sampled_volume_edges(self,
                                  edges=None,
                                  return_plot=False):

        if edges is None:
            if hasattr(self, edges):
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