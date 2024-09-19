import numpy as np
import h5py
from collections import OrderedDict
import time as ttime
import pandas as pd
import dask.array as da

# Local imports
from ..ImageMap import ImageMap
from ..crystal.Phase import Phase
from ..reflections.SpotModels import _load_peak_function
from ..utilities.utilities import pathify


##################
### HDF Format ###
##################

def initialize_xrdmap_hdf(xrdmap,
                          hdf_file):
    
    with h5py.File(hdf_file, 'w-') as f:
        base_grp = f.require_group(xrdmap._object_type) # xrdmap or rsm
        base_grp.attrs['scanid'] = xrdmap.scanid
        base_grp.attrs['beamline'] = xrdmap.beamline #'5-ID (SRX)'
        base_grp.attrs['facility'] = xrdmap.facility #'NSLS-II'
        base_grp.attrs['energy'] = xrdmap.energy
        base_grp.attrs['wavelength'] = xrdmap.wavelength
        base_grp.attrs['theta'] = xrdmap.theta
        base_grp.attrs['time_stamp'] = ''

        # Record diffraction data
        curr_grp = base_grp.require_group('image_data') # naming the group after the detector may be a bad idea...
        curr_grp.attrs['detector'] = '' #'dexela'
        curr_grp.attrs['detector_binning'] = '' #(4, 4)
        curr_grp.attrs['dwell_units'] = 's'

        # Generate emtpy dataset of extra_metadata
        extra_md = base_grp.create_dataset('extra_metadata',
                                            data=h5py.Empty("f"))
        for key, value in xrdmap.extra_metadata.items():
            extra_md.attrs[key] = value

        if not hasattr(xrdmap, 'scan_input') or xrdmap.scan_input is None:
            scan_input = []
        else:
            scan_input = xrdmap.scan_input
        base_grp.attrs['scan_input'] = scan_input

        # Special consideration for when dwell wasn't being recorded...
        if not hasattr(xrdmap, 'dwell') or xrdmap.dwell is None:
            dwell = ''
        else:
            dwell = xrdmap.dwell
        base_grp.attrs['dwell'] = dwell
        curr_grp.attrs['dwell'] = dwell # Used to be exposure time, but not accurate


def load_xrdmap_hdf(filename,
                    wd=None,
                    image_data_key='recent',
                    integration_data_key='recent',
                    dask_enabled=False,
                    map_shape=None,
                    image_shape=None,
                    object_type='xrdmap' # or rsm
                    ):

    # Figuring out hdf file stuff
    hdf_path = pathify(wd, filename, '.h5')
    if not dask_enabled:
        hdf = h5py.File(hdf_path, 'r')
    else:
        hdf = h5py.File(hdf_path, 'a')
    base_grp = hdf[object_type]

    # Load base metadata
    base_md = dict(base_grp.attrs.items())

    # Load extra metadata
    extra_md = {}
    if 'extra_metadata' in base_grp.keys(): # Check for backwards compatibility
        for key, value in base_grp['extra_metadata'].attrs.items():
            extra_md[key] = value

    # Image data
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
                    raise RuntimeError('Could not find recent image data to construct ImageMap from hdf.')
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

            # Collect ImageMap attributes that are not instantiated...
            # This includes correction references...
            image_map_attrs = {}
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
                    image_map_attrs[key] = img_grp[f'_{key}'][:]

            if '_static_background' in img_grp.keys():
                image_map_attrs['background'] = img_grp['_static_background'][:]
            
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
            image_map_attrs = {}
            image_corrections = None

        # Mapped attributes can be useful with our without images; load them
        # Not given to ImageMap.__init__ so it won't try to rewrite the data
        if '_null_map' in img_grp.keys():
            image_map_attrs['null_map'] = img_grp['_null_map'][:]
        
        # Deprecated tag, but kept for backwards compatibility
        if '_spot_masks' in img_grp.keys():
            image_map_attrs['blob_masks'] = img_grp['_spot_masks'][:]
        
        if '_blob_masks' in img_grp.keys():
            image_map_attrs['blob_masks'] = img_grp['_blob_masks'][:]
    
    else:
        if image_data_key is not None:
            warn_str = ('WARNING: Image data requested, but not found in hdf!'
                        + '\nProceeding without image data.')
            print(warn_str)
        image_data = None
        image_map_attrs = {}
        image_corrections = None

    # Integration data
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
            print('done!')
        else:
            integration_data = None
    
    else:
        # No data requested or none available
        # No warning if no data available
        integration_data = None

    # Recipricol positions
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
                #recip_pos[f'{key}_units'] = recip_grp[key].attrs['units']
                recip_pos[f'{key}_resolution'] = recip_grp[key].attrs[f'{key}_resolution']
            else:
                recip_pos[key] = None
                #recip_pos[f'{key}_units'] = None
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
                     #'tth_units' : None,
                     'chi' : None,
                     'chi_resolution' : None,
                     #'chi_units' : None
                     }
        #image_map_attrs['calibrated_shape'] = None
        poni_od = None

    # Load phases
    phase_dict = {}
    if 'phase_list' in base_grp.keys():
        print('Loading saved phases...', end='', flush=True)
        phase_grp = base_grp['phase_list']
        if len(phase_grp) > 0:
            for phase in phase_grp.keys():
                phase_dict[phase] = Phase.from_hdf(phase_grp[phase],
                                                   energy=base_md['energy'],
                                                   tth=recip_pos['tth'])
        print('done!')

    # Load scalers
    sclr_dict = None
    if 'scalers' in base_grp.keys():
        print('Loading scalers...', end='', flush=True)
        sclr_grp = base_grp['scalers']
        
        sclr_dict = {}
        for key in sclr_grp.keys():
            sclr_dict[key] = sclr_grp[key][:]
        print('done!')

    # Load pixel positions
    pos_dict = None
    if 'positions' in base_grp.keys():
        print('Loading positions...', end='', flush=True)
        pos_grp = base_grp['positions']
        
        pos_dict = {}
        for key in pos_grp.keys():
            pos_dict[key] = pos_grp[key][:]
        print('done!')    

    # Load spots dataframe
    spots = None
    spot_model = None
    if 'reflections' in base_grp.keys():
        print('Loading reflection spots...', end='', flush=True)
        # I really dislike that pandas cannot handle an already open file
        hdf.close()
        spots = pd.read_hdf(hdf_path, key='xrdmap/reflections/spots')

        if dask_enabled:
            # Re-point towards correct dataset
            hdf = h5py.File(hdf_path, 'a')
            img_grp = hdf[f'{object_type}/image_data']
            image_dset = img_grp[image_data_key]
            image_data = da.from_array(image_dset, chunks=image_dset.chunks)
        else:
            hdf = h5py.File(hdf_path, 'r')
        print('done!')

        # Load peak model
        if 'spot_model' in hdf[f'{object_type}/reflections'].attrs.keys():
            spot_model_name = hdf[f'{object_type}/reflections'].attrs['spot_model']
            spot_model = _load_peak_function(spot_model_name)
            
    if not dask_enabled:
        hdf.close()
        hdf = None

    # Instantiate ImageMap!
    print(f'Instantiating ImageMap...')
    image_map = ImageMap(image_data=image_data,
                         integration_data=integration_data,
                         map_shape=map_shape,
                         image_shape=image_shape,
                         title=image_data_key,
                         wd=wd,
                         hdf_path=hdf_path,
                         hdf=hdf,
                         corrections=image_corrections,
                         null_map=None, # Not given to __init__
                         dask_enabled=dask_enabled)
    
    # Add extra ImageMap attributes
    for key, value in image_map_attrs.items():
        setattr(image_map, key, value)

    print(f'ImageMap loaded! Shape is {image_map.shape}.')
    
    # return dictionary of useful values
    ouput_dict = {'base_md' : base_md,
                  'extra_md' : extra_md,
                  'image_data': image_map,
                  'recip_pos' : recip_pos,
                  'poni_od' : poni_od,
                  'phase_dict' : phase_dict,
                  'spots' : spots,
                  'spot_model' : spot_model,
                  'sclr_dict' : sclr_dict,
                  'pos_dict' : pos_dict}

    return ouput_dict



def initialize_xrdmapstack_hdf():
    raise NotImplementedError()


def load_xrdmapstack_hdf():
    raise NotImplementedError()