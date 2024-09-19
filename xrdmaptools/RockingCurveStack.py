# Note: This class is highly experimental
# It is intended to wrap the XRDMap class and convert every method to apply iteratively across a stack of XRDMaps
# This is instended for 3D RSM mapping where each method / parameter may change between maps

import os
import h5py
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm

# from .XRDMap import XRDMap
# from .utilities.utilities import timed_iter

from xrdmaptools.XRDMap import XRDMap
from xrdmaptools.ImageMap import ImageMap
from xrdmaptools.utilities.math import (
    energy_2_wavelength,
    wavelength_2_energy
)
from xrdmaptools.geometry.geometry import get_q_vect
from xrdmaptools.io.db_io import (
    get_scantype,
    load_energy_rc_data,
    load_extended_energy_rc_data,
    load_angle_rc_data,
    load_flying_angle_rc_data
)


class RockingCurveStack(XRDMap):

    def __init__(self,
                 scanid=None,
                 wd=None,
                 filename=None,
                 hdf_filename=None,
                 hdf=None,
                 image_data=None,
                 map_shape=None,
                 image_shape=None,
                 map_title=None,
                 energy=None,
                 wavelength=None,
                 dwell=None,
                 theta=None,
                 poni_file=None,
                 sclr_dict=None,
                 beamline='5-ID (SRX)',
                 facility='NSLS-II',
                 time_stamp=None,
                 extra_metadata=None,
                 save_hdf=True
                 ):
        
        # Parse image_data into useble format
        if isinstance(image_data, ImageMap):
            data_ndim = image_data.images.ndim
            data_shape = image_data.images.shape  
        else:
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
                if isinstance(image_data, ImageMap):
                    image_data.images = image_data.images.swapaxes(0, 1)
                    self.map_shape = image_data.images.shape[:2]
                else:
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
        
        # Parse sclr_dict into useble format
        if sclr_dict is not None and isinstance(sclr_dict, dict):
            for key, value in sclr_dict.items():
                sclr_dict[key] = value.reshape(image_data.shape[:2])

        # Check for rocking dimensions
        if energy is None and wavelength is None:
            raise ValueError('Must define either energy or wavelength.')

        XRDMap.__init__(
            self,
            scanid=scanid,
            wd=wd,
            filename=filename,
            hdf_filename=hdf_filename,
            hdf=hdf,
            image_data=image_data,
            integration_data=None,
            map_shape=map_shape,
            image_shape=image_shape,
            map_title=map_title,
            dwell=dwell,
            poni_file=poni_file,
            sclr_dict=sclr_dict,
            beamline=beamline,
            facility=facility,
            time_stamp=time_stamp,
            extra_metadata=extra_metadata,
            save_hdf=save_hdf,
            object_type='rsm' # hard-coded differentiator in hdf
        )

        # Update values
        if energy is not None:
            self.energy = energy
        elif wavelength is not None:
            self.wavelength = wavelength
        else: # Should be redundant
            raise ValueError('Must define either energy or wavelength.')
        
        # Parse theta. energy and wavelength are done in property definition
        if theta is None:
            print('WARNING: No theta value provided. Assuming 0 deg.')
            theta = [0,] * self.map.num_images
        elif isinstance(theta, (list, np.ndarray)):
            if len(theta) != self.map.num_images:
                raise ValueError('Theta must have length equal to number of images.')
            else:
                theta = list(theta)
        elif not isinstance(theta, list):
            theta = [theta,] * self.map.num_images
        else:
            raise RuntimeError('Unable to handle theta input. Provide list or value.')
        self.theta = theta

        # Check for changes. Wavelength will depend on energy
        if all([
            all(np.round(i, 4) == np.round(energy[0], 4) for i in energy),
            all(np.round(i, 4) == np.round(theta[0], 4) for i in theta)
        ]):
            err_str = ('Rocking curves must be constructed by varying '
                       + 'energy/wavelength or theta. Given values are constant.')
            raise ValueError(err_str)
        
        # Other re-define values
        self.map.map_labels = ['rocking_ind',
                               'null_ind']
        
        # Re-write certain values:
        if self.hdf_path is not None:
            with h5py.File(self.hdf_path, 'a') as f:
                f['rsm'].attrs['energy'] = self.energy
                f['rsm'].attrs['wavelength'] = self.wavelength
                f['rsm'].attrs['theta'] = self.theta

        # Bad methods for rocking curves, but cannot delete...
        # del (
        #     # Attributes
        #     self.xrf_path,
        #     # Methods
        #     self.fracture_large_map,
        #     self.integrate1d_map,
        #     self.integrate2d_map,
        #     self.integrate1d_image,
        #     self.integrate2d_image,
        #     self.save_reciprocal_positions, # Maybe keep this one?
        #     self.set_positions,
        #     self.map_extent,
        #     self.swap_axes,
        #     self.interpolate_positions,
        #     self.pixel_spots,
        #     self.load_xrfmap,
        #     # Plotting Methods
        #     self.plot_integration,
        #     self.plot_map,
        #     self.plot_interactive_map,
        #     self.plot_interactive_integration_map,
        #     # ImageMap Functions
        #     self.map.estimate_integration_background,
        #     self.map.remove_integration_background,
        #     self.map.rescale_integrations,
        #     self.map.save_integrations
        # )
        

    #############################
    ### Re-Written Properties ###
    #############################

    @XRDMap.energy.setter
    def energy(self, energy):
        if energy is None:
            self._energy = None
        elif isinstance(energy, (list, np.ndarray)):
            if len(energy) != self.map.num_images:
                raise ValueError('Energy must have length equal to number of images.')
            else:
                energy = list(energy)
        elif not isinstance(energy, list):
            energy = [energy,] * self.map.num_images
        else:
            raise RuntimeError('Unable to handle energy input. Provide list or value.')
        
        wavelength = [energy_2_wavelength(energy_i) for energy_i in energy]
        self._energy = energy
        self._wavelength = wavelength

        # Propogate changes...
        if hasattr(self, 'ai') and self.ai is not None:
            self.ai.energy = self._energy[0]
        if hasattr(self, '_q_arr'):
            delattr(self, '_q_arr')


    @XRDMap.wavelength.setter
    def wavelength(self, wavelength):
        if wavelength is None:
            self._wavelength = None
        elif isinstance(wavelength, (list, np.ndarray)):
            if len(wavelength) != self.map.num_images:
                raise ValueError('Wavelength must have length equal to number of images.')
            else:
                wavelength = list(wavelength)
        elif not isinstance(wavelength, list):
            wavelength = [wavelength,] * self.map.num_images
        else:
            raise RuntimeError('Unable to handle wavelength input. Provide list or value.')
        
        energy = [wavelength_2_energy(wavelength_i) for wavelength_i in wavelength]
        self._energy = energy
        self._wavelength = wavelength

        # Propogate changes...
        if hasattr(self, 'ai') and self.ai is not None:
            self.ai.energy = self._energy[0]
        if hasattr(self, '_q_arr'):
            delattr(self, '_q_arr')


    # Full q-vector, not just magnitude
    @XRDMap.q_arr.getter
    def q_arr(self):
        if hasattr(self, '_q_arr'):
            return self._q_arr
        elif not hasattr(self, 'ai'):
            raise RuntimeError('Cannot calculate q-space without calibration.')
        else:
            self._q_arr = np.empty((self.map.num_images, 3, *self.map.image_shape),
                                    dtype=self.map.dtype)
            for i, wavelength in tqdm(enumerate(self.wavelength),
                                    total=self.map.num_images):
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
    def from_hdf(cls,
                 hdf_filename,
                 wd=None,
                 image_data_key='recent',
                 map_shape=None,
                 image_shape=None,
                 **kwargs):
        
        inst = XRDMap.from_hdf(hdf_filename,
                               wd=wd,
                               dask_enabled=False,
                               image_data_key=image_data_key,
                               integration_data_key=None,
                               map_shape=map_shape,
                               image_shape=image_shape,
                               **kwargs)
    
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
                    save_hdf=save_hdf
                    )
            
            rocking_curves.append(rc)

        if len(rocking_curves) > 1:
            return tuple(rocking_curves)
        else:
            # Don't bother returning a tuple or list of xrdmaps
            return rocking_curves[0]
        


    
    ##########################################
    ### RockingCurveStack Specific Methods ###
    ##########################################

    def vectorize_images(self):
        raise NotImplementedError()
        if not hasattr(self.map, 'blob_masks'):
            pass
    
    
    # Def get 3D q-coordinates

    # Combined spots...Useful for not 3D RSM analysis
    
    # Segment 3D data

    # Center of mass

    # Indexing

    # Strain math

    # 3D plotting

    ##########################
    ### Plotting Functions ###
    ##########################

    def plot_image_stack(self,):
        raise NotImplementedError()

    
    def plot_3D_scatter(self,):
        raise NotImplementedError()


    def plot_3D_volume(self,):
        raise NotImplementedError()

    
    def plot_3D_sampled_outline(self,):
        raise NotImplementedError()
