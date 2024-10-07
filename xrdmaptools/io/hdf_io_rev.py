import numpy as np
import h5py
from collections import OrderedDict
import time as ttime
import pandas as pd
import dask.array as da

# Local imports
from xrdmaptools.crystal.Phase import Phase
from xrdmaptools.reflections.SpotModels import _load_peak_function
from xrdmaptools.utilities.utilities import pathify


##################
### HDF Format ###
##################

def initialize_xrdbase_hdf(xrdbase,
                           hdf_file):
    
    with h5py.File(hdf_file, 'w-') as f:
        base_grp = f.require_group(xrdbase._hdf_type) # xrdmap or rsm
        base_grp.attrs['scanid'] = xrdbase.scanid
        base_grp.attrs['beamline'] = xrdbase.beamline #'5-ID (SRX)'
        base_grp.attrs['facility'] = xrdbase.facility #'NSLS-II'
        base_grp.attrs['energy'] = xrdbase.energy
        base_grp.attrs['wavelength'] = xrdbase.wavelength
        base_grp.attrs['theta'] = xrdbase.theta
        base_grp.attrs['time_stamp'] = ''

        # Record diffraction data
        curr_grp = base_grp.require_group('image_data') # naming the group after the detector may be a bad idea...
        curr_grp.attrs['detector'] = '' #'dexela'
        curr_grp.attrs['detector_binning'] = '' #(4, 4)
        curr_grp.attrs['dwell_units'] = 's'

        # Generate emtpy dataset of extra_metadata
        extra_md = base_grp.create_dataset('extra_metadata',
                                            data=h5py.Empty("f"))
        for key, value in xrdbase.extra_metadata.items():
            extra_md.attrs[key] = value

        if (not hasattr(xrdbase, 'scan_input')
            or xrdbase.scan_input is None):
            scan_input = []
        else:
            scan_input = xrdbase.scan_input
        base_grp.attrs['scan_input'] = scan_input

        # Special consideration for when dwell wasn't being recorded...
        if (not hasattr(xrdbase, 'dwell')
            or xrdbase.dwell is None):
            dwell = ''
        else:
            dwell = xrdbase.dwell
        base_grp.attrs['dwell'] = dwell
        curr_grp.attrs['dwell'] = dwell # Used to be exposure time, but not accurate


def load_xrdbase_hdf(filename,
                     hdf_type, # xrdmap or rsm
                     wd,
                     image_data_key='recent',
                     integration_data_key='recent',
                     map_shape=None,
                     image_shape=None,
                     dask_enabled=False,
                     ):
    
    # Figuring out hdf file stuff
    hdf_path = pathify(wd, filename, '.h5')

    if not dask_enabled:
        hdf = h5py.File(hdf_path, 'r')
    else:
        hdf = h5py.File(hdf_path, 'a')
    base_grp = hdf[hdf_type]

    # Load reflection first
    # Built-in pandas hdf support needs this
    spots = None
    spot_model = None
    if 'reflections' in base_grp.keys():
        print('Loading reflection spots...', end='', flush=True)
        # I really dislike that pandas cannot handle an already open file
        hdf.close()
        spots = pd.read_hdf(hdf_path,
                            key=f'{hdf_type}/reflections/spots')
        if not dask_enabled:
            hdf = h5py.File(hdf_path, 'r')
        else:
            hdf = h5py.File(hdf_path, 'a')
        base_grp = hdf[hdf_type]

        # Load peak model
        if 'spot_model' in hdf[f'{hdf_type}/reflections'].attrs.keys():
            spot_model_name = hdf[f'{hdf_type}/reflections'].attrs['spot_model']
            spot_model = _load_peak_function(spot_model_name)

    # Load base metadata
    base_md = dict(base_grp.attrs.items())

    # Load extra metadata
    extra_md = {}
    if 'extra_metadata' in base_grp.keys(): # Check for backwards compatibility
        for key, value in base_grp['extra_metadata'].attrs.items():
            extra_md[key] = value

    # Load image data
    (image_data,
     image_attrs,
     image_corrections) = _load_xrd_hdf_image_data(
                                base_grp,
                                image_data_key=image_data_key,
                                map_shape=map_shape,
                                image_shape=image_shape,
                                dask_enabled=dask_enabled)

    # Load integration data
    (integration_data,
     integration_attrs,
     integration_corrections) = _load_xrd_hdf_integration_data(
                                    base_grp,
                                    integration_data_key=integration_data_key,
                                    map_shape=map_shape)

    # Load recipricol positions
    recip_pos, poni_od = _load_xrd_hdf_reciprocal_positions(base_grp)

    # Load phases
    phase_dict = _load_xrd_hdf_phases(base_grp,
                                      energy=base_md['energy'],
                                      tth=recip_pos['tth'])

    # Load scalers
    sclr_dict = _load_xrd_hdf_scalers(base_grp)

    # Load pixel positions
    pos_dict = _load_xrd_hdf_positions(base_grp) 

    # Final hdf considerations
    if not dask_enabled:
        hdf = None
    
    # Return dictionary of useful values
    ouput_dict = {'base_md' : base_md,
                  'extra_metadata' : extra_md,
                  'image_data' : image_data,
                  'image_attrs' : image_attrs,
                  'image_corrections' : image_corrections,
                  'integration_data' : integration_data,
                  'integration_attrs' : integration_attrs,
                  'integration_corrections' : integration_corrections,
                  'recip_pos' : recip_pos,
                  'poni_file' : poni_od,
                  'phases' : phase_dict,
                  'spots' : spots,
                  'spot_model' : spot_model,
                  'sclr_dict' : sclr_dict,
                  'pos_dict' : pos_dict,
                  'hdf' : hdf}

    return ouput_dict


def _load_xrd_hdf_image_data(base_grp,
                             image_data_key='recent',
                             map_shape=None,
                             image_shape=None,
                             dask_enabled=False):
    
    if 'image_data' in base_grp.keys():
        img_grp = base_grp['image_data']

        if image_data_key is not None:
            if (str(image_data_key).lower() != 'recent'
                and image_data_key not in img_grp.keys()):
                warn_str = (f'WARNING: Requested image_data_key ({image_data_key}) '
                            + 'not found in hdf. Looking for most recent image_data instead...')
                image_data_key = 'recent'
            
            # Determine image_data key in 
            if str(image_data_key).lower() == 'recent':
                time_stamps, img_keys = [], []
                for key in img_grp.keys():
                    if key[0] != '_':
                        time_stamps.append(img_grp[key].attrs['time_stamp'])
                        img_keys.append(key)
                if len(img_keys) < 1:
                    raise RuntimeError('Could not find recent image data from hdf.')
                time_stamps = [ttime.mktime(ttime.strptime(x)) for x in time_stamps]
                image_data_key = img_keys[np.argmax(time_stamps)]

            print(f'Loading images from ({image_data_key})...', end='', flush=True)
            if dask_enabled:
                # Lazy loads data
                image_dset = img_grp[image_data_key]
                image_data = da.from_array(image_dset, chunks=image_dset.chunks)
            else:
                # Fully loads data
                image_data = img_grp[image_data_key][:]

            # Rebuild correction dictionary
            image_corrections = {}
            for key, value in img_grp[image_data_key].attrs.items():
                if key[0] == '_' and key[-11:] == '_correction':
                    image_corrections[key[1:-11]] = value

            # Collect XRDData attributes that are not instantiated...
            # This includes correction references...
            image_attrs = {}
            for key in ['dark_field',
                        'flat_field',
                        'air_scatter',
                        'scaler_intensity',
                        'lorentz_correction',
                        'polarization_correction',
                        'solidangle_correction',
                        'absorption_correction',
                        'custom_mask',
                        'defect_mask',
                        'calibration_mask']:
                if f'_{key}' in img_grp.keys():
                    image_attrs[key] = img_grp[f'_{key}'][:]

            if '_static_background' in img_grp.keys():
                image_attrs['background'] = img_grp['_static_background'][:]
            
            if map_shape is not None and image_data.shape[:2] != map_shape:
                warn_str = (f'WARNING: Input map_shape {map_shape} does '
                            + f'not match loaded image data map shape of '
                            + f'{image_data.shape[:2]}.'
                            + '\nDefaulting to loaded data map shape.')
                print(warn_str)
                map_shape = image_data.shape[:2]
            if image_shape is not None and image_data.shape[2:] != image_shape:
                warn_str = (f'WARNING: Input image_shape {image_shape} does '
                            + f'not match loaded image data image shape of '
                            + f'{image_data.shape[2:]}.'
                            + '\nDefaulting to loaded data image shape.')
                print(warn_str)
                image_shape = image_data.shape[2:]
            
            print('done!')
        
        else:
            # No data requested
            image_data = None
            image_attrs = {}
            image_corrections = None

        # Mapped attributes can be useful with our without images; load them
        # Not given to XRDData.__init__ so it won't try to rewrite the data
        if '_null_map' in img_grp.keys():
            image_attrs['null_map'] = img_grp['_null_map'][:]
        
        # Deprecated tag, but kept for backwards compatibility
        if '_spot_masks' in img_grp.keys():
            image_attrs['blob_masks'] = img_grp['_spot_masks'][:]
        
        if '_blob_masks' in img_grp.keys():
            image_attrs['blob_masks'] = img_grp['_blob_masks'][:]
    
    else:
        if image_data_key is not None:
            warn_str = ('WARNING: Image data requested, but not found in hdf!'
                        + '\nProceeding without image data.')
            print(warn_str)
        image_data = None
        image_attrs = {}
        image_corrections = None
    
    return image_data, image_attrs, image_corrections


def _load_xrd_hdf_integration_data(base_grp,
                                   integration_data_key='recent',
                                   map_shape=None):
    if 'integration_data' in base_grp.keys():
        int_grp = base_grp['integration_data']

        if integration_data_key is not None:
            # Check integration_data_key in hdf
            if (str(integration_data_key).lower() != 'recent'
                and integration_data_key not in int_grp.keys()):
                warn_str = (f'WARNING: Requested integration_data_key ({integration_data_key}) '
                            + 'not found in hdf. Looking for most recent integration_data instead...')
            
            # Determine image_data key in 
            if str(integration_data_key).lower() == 'recent':
                time_stamps, int_keys = [], []
                for key in int_grp.keys():
                    if key[0] != '_':
                        time_stamps.append(int_grp[key].attrs['time_stamp'])
                        int_keys.append(key)
                if len(int_keys) < 1:
                    raise RuntimeError('Could not find recent image data to construct ImageMap from hdf.')
                time_stamps = [ttime.mktime(ttime.strptime(x)) for x in time_stamps]
                integration_data_key = int_keys[np.argmax(time_stamps)]

            print(f'Loading integrations from ({integration_data_key})...', end='', flush=True)
            integration_data = int_grp[integration_data_key][:]

            if map_shape is not None and integration_data.shape[:2] != map_shape:
                warn_str = (f'WARNING: Input map_shape {map_shape} does '
                            + f'not match loaded integration data map shape of '
                            + f'{integration_data.shape[:2]}.'
                            + '\nDefaulting to loaded data map shape.')
                print(warn_str)
                map_shape = integration_data.shape[:2]

            integration_corrections = {}
            for key, value in int_grp[integration_data_key].attrs.items():
                if key[0] == '_' and key[-11:] == '_correction':
                    integration_corrections[key[1:-11]] = value
            print('done!')
            
            # No current integration attributes. This may change
            integration_attrs = {}
            for key in []:
                if f'_{key}' in int_grp.keys():
                    integration_attrs[key] = int_grp[f'_{key}'][:]

        else:
            integration_data = None
            integration_attrs = {}
            integration_corrections = {}
    
    else:
        # No data requested or none available
        # No warning if no data available
        integration_data = None
        integration_attrs = {}
        integration_corrections = {}
    
    return integration_data, integration_attrs, integration_corrections


def _load_xrd_hdf_reciprocal_positions(base_grp):
    if 'reciprocal_positions' in base_grp.keys():
        print('Loading reciprocal positions...', end='', flush=True)
        recip_grp = base_grp['reciprocal_positions']

        recip_pos = {}
        for key in ['tth', 'chi']:
            if key in recip_grp.keys():
                recip_pos[key] = recip_grp[key][:]
                # None cannot be stored in hdf, empty values act as placeholder
                if len(recip_pos[key]) == 0:
                    recip_pos[key] = None
                recip_pos[f'{key}_resolution'] = recip_grp[key].attrs[f'{key}_resolution']
            else:
                recip_pos[key] = None
                recip_pos[f'{key}_resolution'] = None

        # Load poni_file calibration
        poni_grp = recip_grp['poni_file']
        ordered_keys = ['poni_version',
                        'detector',
                        'detector_config',
                        'dist',
                        'poni1',
                        'poni2',
                        'rot1',
                        'rot2',
                        'rot3',
                        'wavelength']
        poni_od = OrderedDict()
        for key in ordered_keys:
            if key == 'detector_config':
                detector_od = OrderedDict()
                detector_od['pixel1'] = poni_grp['detector_config'].attrs['pixel1']
                detector_od['pixel2'] = poni_grp['detector_config'].attrs['pixel2']
                detector_od['max_shape'] = list(poni_grp['detector_config'].attrs['max_shape'])
                detector_od['orientation'] = poni_grp['detector_config'].attrs['orientation']
                poni_od[key] = detector_od
            else:
                poni_od[key] = poni_grp.attrs[key]
        print('done!')
    
    else: 
        recip_pos = {'tth' : None,
                     'tth_resolution' : None,
                     'chi' : None,
                     'chi_resolution' : None,
                     }
        poni_od = None
    
    return recip_pos, poni_od


def _load_xrd_hdf_phases(base_grp,
                         energy=None,
                         tth=None):
    phase_dict = {}
    if 'phase_list' in base_grp.keys():
        print('Loading saved phases...', end='', flush=True)
        phase_grp = base_grp['phase_list']
        if len(phase_grp) > 0:
            for phase in phase_grp.keys():
                phase_dict[phase] = Phase.from_hdf(phase_grp[phase],
                                                   energy=energy,
                                                   tth=tth)
        print('done!')
    return phase_dict


def _load_xrd_hdf_scalers(base_grp):
    sclr_dict = None
    if 'scalers' in base_grp.keys():
        print('Loading scalers...', end='', flush=True)
        sclr_grp = base_grp['scalers']
        
        sclr_dict = {}
        for key in sclr_grp.keys():
            sclr_dict[key] = sclr_grp[key][:]
        print('done!')
    return sclr_dict


def _load_xrd_hdf_positions(base_grp):
    pos_dict = None
    if 'positions' in base_grp.keys():
        print('Loading positions...', end='', flush=True)
        pos_grp = base_grp['positions']
        
        pos_dict = {}
        for key in pos_grp.keys():
            pos_dict[key] = pos_grp[key][:]
        print('done!')  
    return pos_dict






def initialize_xrdmapstack_hdf():
    raise NotImplementedError()


def load_xrdmapstack_hdf():
    raise NotImplementedError()