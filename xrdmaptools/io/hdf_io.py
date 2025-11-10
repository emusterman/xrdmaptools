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


###################
### XRDBaseScan ###
###################

def initialize_xrdbase_hdf(xrdbase,
                           hdf_file):
    
    # Is this with statement still necessary?
    with h5py.File(hdf_file, 'w-') as f:
        base_grp = f.require_group(xrdbase._hdf_type) # xrdmap or rsm
        base_grp.attrs['scan_id'] = xrdbase.scan_id
        base_grp.attrs['beamline'] = xrdbase.beamline #'5-ID (SRX)'
        base_grp.attrs['facility'] = xrdbase.facility #'NSLS-II'
        base_grp.attrs['detector'] = xrdbase.detector #'dexela'
        base_grp.attrs['energy'] = xrdbase.energy
        base_grp.attrs['wavelength'] = xrdbase.wavelength
        base_grp.attrs['theta'] = xrdbase.theta
        base_grp.attrs['use_stage_rotation'] = int(xrdbase.use_stage_rotation)
        if xrdbase.time_stamp is None:
            base_grp.attrs['time_stamp'] = ttime.ctime()
        else:
            base_grp.attrs['time_stamp'] = xrdbase.time_stamp

        # Record diffraction data
        curr_grp = base_grp.require_group('image_data') # naming the group after the detector may be a bad idea...
        curr_grp.attrs['detector'] = '' #'dexela'
        curr_grp.attrs['detector_binning'] = '' #(4, 4)
        curr_grp.attrs['dwell_units'] = 's'

        # Generate emtpy dataset of extra_metadata
        extra_md = base_grp.create_dataset('extra_metadata',
                                           data=h5py.Empty("f"))
        for key, value in xrdbase.extra_metadata.items():
            # print(f'Writing to hdf for {key} with {value} of type {type(value)}')
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
                     load_blob_masks=True,
                     load_vector_map=False,
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

    # Protect hdf
    try:
        if hdf_type in hdf:
            base_grp = hdf[hdf_type]
        else:
            if len(hdf.keys()) == 1:
                found_hdf_type = hdf.keys()[0]
                err_str = ('Found HDF of incorrect type. Expected '
                           + f'{hdf_type}, but found {found_hdf_type}'
                           + 'instead.')
                if found_hdf_type == 'xrfmap':
                    err_str += ('\nXRF maps should be handled with '
                                + 'pyxrf or similar packages.')
            else:
                err_str = ('Found HDF of incorrect type. Expected '
                           + f'{hdf_type}, but found unknown type '
                           + 'instead.')
            raise TypeError(err_str)

        # Load reflection first
        # Built-in pandas hdf support needs this
        spots = None
        spots_3D = None
        spot_model = None
        if 'reflections' in base_grp.keys():
            print('Loading reflection spots...', end='', flush=True)
            # I really dislike that pandas cannot handle an already open file
            has_spots, has_spots_3D = False, False
            if 'spots' in base_grp['reflections'].keys():
                has_spots = True
            if 'spots_3D' in base_grp['reflections'].keys():
                has_spots_3D = True

            hdf.close()
            if has_spots:
                spots = pd.read_hdf(hdf_path,
                                    key=f'{hdf_type}/reflections/spots')
            
            if has_spots_3D:
                spots_3D = pd.read_hdf(hdf_path,
                                    key=f'{hdf_type}/reflections/spots_3D')

            if not dask_enabled:
                hdf = h5py.File(hdf_path, 'r')
                base_grp = hdf[hdf_type]
            else:
                hdf = h5py.File(hdf_path, 'a')
                base_grp = hdf[hdf_type]

            # Load spot model
            if (has_spots
                and 'spot_model' in base_grp['reflections/spots'].attrs.keys()):
                spot_model_name = base_grp['reflections/spots'].attrs['spot_model']
                spot_model = _load_peak_function(spot_model_name)
            print('done!')

        # Load base metadata
        base_md = dict(base_grp.attrs.items())

        # Backwards compatability
        if 'scanid' in base_md:
            # Change dictionary
            base_md['scan_id'] = base_md['scanid']
            del base_md['scanid']
            # Overwrite hdf values
            write_flag = (hdf.mode == 'a')
            if not write_flag:
                hdf.close()
                hdf = h5py.File(hdf_path, 'a')
                base_grp = hdf[hdf_type]
            base_grp.attrs['scan_id'] = base_md['scan_id']
            del base_grp.attrs['scanid']
            if not write_flag:
                hdf.close()
                hdf = h5py.File(hdf_path, 'r')
                base_grp = hdf[hdf_type]

        # Load extra metadata
        extra_md = {}
        if 'extra_metadata' in base_grp.keys(): # Check for backwards compatibility
            for key, value in base_grp['extra_metadata'].attrs.items():
                extra_md[key] = value

        # Load image data
        (image_data,
        image_attrs,
        image_corrections,
        image_data_key,
        map_shape,
        image_shape) = _load_xrd_hdf_image_data(
                                    base_grp,
                                    image_data_key=image_data_key,
                                    load_blob_masks=load_blob_masks,
                                    map_shape=map_shape,
                                    image_shape=image_shape,
                                    dask_enabled=dask_enabled)

        # Load integration data
        (integration_data,
        integration_attrs,
        integration_corrections,
        integration_data_key,
        map_shape) = _load_xrd_hdf_integration_data(
                                    base_grp,
                                    integration_data_key=integration_data_key,
                                    map_shape=map_shape)

        # Load recipricol positions
        recip_pos, poni_od = _load_xrd_hdf_reciprocal_positions(base_grp)

        # Load phases
        phase_dict = _load_xrd_hdf_phases(base_grp)

        # Load scalers
        sclr_dict = _load_xrd_hdf_scalers(base_grp)

        # Load pixel positions
        pos_dict = _load_xrd_hdf_positions(base_grp)

        # Load vector data. Function handles whether from rsm or xrdmap
        vector_dict = _load_xrd_hdf_vector_data(
                                    base_grp,
                                    load_vector_map=load_vector_map)

        # Final hdf considerations
        if not dask_enabled:
            hdf.close()
            hdf = None

    # Catch errors and close hdf
    except Exception as e:
        if hdf is not None:
            hdf.close()
        raise e

    # Return dictionary of useful values
    ouput_dict = {'base_md' : base_md,
                  'extra_metadata' : extra_md,
                  'map_shape' : map_shape,
                  'image_shape' : image_shape,
                  'image_data' : image_data,
                  'image_attrs' : image_attrs,
                  'image_corrections' : image_corrections,
                  'image_data_key' : image_data_key,
                  'integration_data' : integration_data,
                  'integration_attrs' : integration_attrs,
                  'integration_corrections' : integration_corrections,
                  'integration_data_key' : integration_data_key,
                  'recip_pos' : recip_pos,
                  'poni_file' : poni_od,
                  'phases' : phase_dict,
                  'spots' : spots,
                  'spots_3D' : spots_3D,
                  'spot_model' : spot_model,
                  'sclr_dict' : sclr_dict,
                  'pos_dict' : pos_dict,
                  'vector_dict' : vector_dict,
                  'hdf' : hdf}

    return ouput_dict


def _load_xrd_hdf_image_data(base_grp,
                             image_data_key='recent',
                             load_blob_masks=True,
                             map_shape=None,
                             image_shape=None,
                             dask_enabled=False):
    
    if 'image_data' in base_grp.keys():
        img_grp = base_grp['image_data']

        if image_data_key is not None:
            # Check valid image_data_key
            if str(image_data_key).lower() != 'recent':
                if image_data_key not in img_grp.keys():
                    if f'{image_data_key}_images' in img_grp.keys():
                        image_data_key = f'{image_data_key}_images'
                    elif f'_{image_data_key}' in img_grp.keys():
                        image_data_key = f'_{image_data_key}'
                    else:
                        warn_str = ('WARNING: Requested image_data_key'
                                    + f' ({image_data_key}) not found '
                                    + 'in hdf. Looking for most recent'
                                    + ' image_data instead...')
                        print(warn_str)
                        image_data_key = 'recent'
            
            # Set recent image data key
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
                    image_corrections[key[1:-11]] = bool(value)

            # Collect XRDData attributes that are not instantiated...
            image_attrs = {}
            for key in ['saturated_pixels',
                        'dark_field',
                        'flat_field',
                        'air_scatter',
                        'scaler_map',
                        'lorentz_correction',
                        'polarization_correction',
                        'solidangle_correction',
                        'absorption_correction',
                        'custom_mask',
                        'defect_mask',
                        'polar_mask']:
                if f'_{key}' in img_grp.keys():
                    image_attrs[key] = img_grp[f'_{key}'][:]

            if '_static_background' in img_grp.keys():
                image_attrs['background'] = img_grp['_static_background'][:]
            
            if map_shape is None:
                map_shape = image_data.shape[:2]
            elif image_data.shape[:2] != map_shape:
                warn_str = (f'WARNING: Input map_shape {map_shape} does '
                            + f'not match loaded image data map shape of '
                            + f'{image_data.shape[:2]}.'
                            + '\nDefaulting to loaded data map shape.')
                print(warn_str)
                map_shape = image_data.shape[:2]
            
            if image_shape is None:
                image_shape = image_data.shape[2:]
            elif image_data.shape[2:] != image_shape:
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
            image_data_key = None

            # Get map_shape and image_shape from most recent dataset
            if map_shape is None or image_shape is None:
                # Find most recent data key
                time_stamps, img_keys = [], []
                for key in img_grp.keys():
                    if key[0] != '_':
                        time_stamps.append(img_grp[key].attrs['time_stamp'])
                        img_keys.append(key)
                if len(img_keys) < 1:
                    err_str = ('Map_shape or image_shape not provided '
                            + 'and could not find recent image data'
                            + ' from hdf to determine values.')
                    raise RuntimeError(err_str)
                time_stamps = [ttime.mktime(ttime.strptime(x))
                            for x in time_stamps]
                recent_data_key = img_keys[np.argmax(time_stamps)]

                if map_shape is None:
                    map_shape = img_grp[recent_data_key].shape[:2]
                if image_shape is None:
                    image_shape = img_grp[recent_data_key].shape[2:]

        # Mapped attributes can be useful with our without images; load them
        # Not given to XRDData.__init__
        if '_null_map' in img_grp.keys():
            image_attrs['null_map'] = img_grp['_null_map'][:]
        
        # Can be bulky and take up too much memory
        if load_blob_masks: 
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
        image_data_key = None
    
    return (image_data,
            image_attrs,
            image_corrections,
            image_data_key,
            map_shape,
            image_shape)


def _load_xrd_hdf_integration_data(base_grp,
                                   integration_data_key='recent',
                                   map_shape=None):
    if 'integration_data' in base_grp.keys():
        int_grp = base_grp['integration_data']

        if integration_data_key is not None:
            # Check valid integration_data_key
            if str(integration_data_key).lower() != 'recent':
                if integration_data_key not in int_grp.keys():
                    if f'{integration_data_key}_images' in int_grp.keys():
                        integration_data_key = f'{integration_data_key}_images'
                    elif f'_{integration_data_key}' in imt_grp.keys():
                        integration_data_key = f'_{integration_data_key}'
                    else:
                        warn_str = (
                            'WARNING: Requested integration_data_key '
                            + f'({integration_data_key}) not found in '
                            + 'hdf. Looking for most recent '
                            + 'integration_data instead...')
                        print(warn_str)
                        integration_data_key = 'recent'
            
            # Determine image_data key in 
            if str(integration_data_key).lower() == 'recent':
                time_stamps, int_keys = [], []
                for key in int_grp.keys():
                    if key[0] != '_':
                        time_stamps.append(int_grp[key].attrs['time_stamp'])
                        int_keys.append(key)
                if len(int_keys) < 1:
                    raise RuntimeError('Could not find recent integration data from hdf.')
                time_stamps = [ttime.mktime(ttime.strptime(x)) for x in time_stamps]
                integration_data_key = int_keys[np.argmax(time_stamps)]

            print(f'Loading integrations from ({integration_data_key})...', end='', flush=True)
            integration_data = int_grp[integration_data_key][:]
            
            # Get map_shape is None
            if map_shape is None:
                map_shape = integration_data.shape[:2]
            elif integration_data.shape[:2] != map_shape:
                warn_str = (f'WARNING: Input map_shape {map_shape} does '
                            + f'not match loaded integration data map shape of '
                            + f'{integration_data.shape[:2]}.'
                            + '\nAssuming original map shape is from '
                            + 'image_data and proceeding without changes.')
                print(warn_str)
                # map_shape = integration_data.shape[:2]

            integration_corrections = {}
            for key, value in int_grp[integration_data_key].attrs.items():
                if key[0] == '_' and key[-11:] == '_correction':
                    integration_corrections[key[1:-11]] = bool(value)
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
            integration_data_key = None

            # Get map_shape and image_shape from most recent dataset
            if map_shape is None:
             # Find most recent data key
                time_stamps, img_keys = [], []
                for key in img_grp.keys():
                    if key[0] != '_':
                        time_stamps.append(img_grp[key].attrs['time_stamp'])
                        img_keys.append(key)
                if len(img_keys) < 1:
                    err_str = ('Map_shape or image_shape not provided '
                            + 'and could not find recent image data'
                            + ' from hdf to determine values.')
                    raise RuntimeError(err_str)
                time_stamps = [ttime.mktime(ttime.strptime(x))
                            for x in time_stamps]
                integration_data_key = int_keys[np.argmax(time_stamps)]
                map_shape = int_grp[integration_data_key].shape[:2]
    
    else:
        # No data requested or none available
        # No warning if no data available
        integration_data = None
        integration_attrs = {}
        integration_corrections = {}
        integration_data_key = None
    
    return (integration_data,
            integration_attrs,
            integration_corrections,
            integration_data_key,
            map_shape)


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


def _load_xrd_hdf_phases(base_grp):
    phase_dict = {}
    if 'phase_list' in base_grp.keys():
        print('Loading saved phases...', end='', flush=True)
        phase_grp = base_grp['phase_list']
        if len(phase_grp) > 0:
            for phase in phase_grp.keys():
                phase_dict[phase] = Phase.from_hdf(phase_grp[phase])
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


# Load vector data. Different for 'vectorized_data' from rocking curves
# and 'vectorized_map' for xrdmap and xrdmapstack
def _load_xrd_hdf_vector_data(base_grp, load_vector_map=True):
    vector_dict = None

    # XRDRockingCurve Data
    if 'vectorized_data' in base_grp:
        print('Loading vectorized_data...', end='', flush=True)
        vector_dict = {}
        vector_grp = base_grp['vectorized_data']

        q_vectors, intensity = None, None
        for key, value in vector_grp.items():
            if key == 'edges':
                edges = []
                for edge_title, edge_dset in value.items():
                    edges.append(edge_dset[:])
                vector_dict[key] = edges
            elif key == 'q_vectors': # backwards compatibility
                q_vectors = value
            elif key == 'intensity': # backwards compatibility
                intensity = value
            else:
                vector_dict[key] = value[:]
            
            # Collect datset attrs
            for attr_key, attr_value in value.attrs.items():
                if attr_key != 'time_stamp':
                    vector_dict[attr_key] = attr_value
        
        # Collect group attrs
        for attr_key, attr_value in vector_grp.attrs.items():
            if attr_key != 'time_stamp':
                vector_dict[attr_key] = attr_value

        # Construct vectors for backwards compatibility
        if q_vectors is not None and intensity is not None:
            vector_dict['vectors'] = np.hstack([
                                        q_vectors[:],
                                        intensity[:].reshape(-1, 1)])
        
        # Blank group returns nothing
        if len(vector_dict.keys()) == 0:
            vector_dict = None

        print('done!')
    
    # XRDMap and XRDMapStack Data
    elif 'vectorized_map' in base_grp and load_vector_map:
        vector_map = None
        print('Loading vectorized map...', end='', flush=True)
        vector_dict = {}
        vector_grp = base_grp['vectorized_map']

        # Conditional for new format
        if 'vector_map' in vector_grp:
            map_grp = vector_grp['vector_map']
        else:
            map_grp = vector_grp
        
        map_shape = map_grp.attrs['vectorized_map_shape']
        vector_map = np.empty(map_shape, dtype=object)

        for index in range(np.prod(map_shape)):
            indices = np.unravel_index(index, map_shape)
            title = ','.join([str(ind) for ind in indices])
            vector_map[indices] = map_grp[title][:]
        vector_dict['vector_map'] = vector_map

        if 'edges' in vector_grp:
            edges = []
            for edge_title, edge_dset in vector_grp['edges'].items():
                edges.append(edge_dset[:])
            vector_dict['edges'] = edges
        
        # Collect extra attrs from datasets
        # May take awhile for previously saved vector maps
        for key, value in vector_grp.items():
            for attr_key, attr_value in value.attrs.items():
                if attr_key != 'time_stamp':
                    vector_dict[attr_key] = attr_value

        # Collect extra attrs from full group
        for attr_key, attr_value in vector_grp.attrs.items():
            if attr_key != 'time_stamp':
                vector_dict[attr_key] = attr_value
        
        # Remove unwanted values
        if 'vectorized_map_shape' in vector_dict:
            del vector_dict['vectorized_map_shape']

        print('done!')

    return vector_dict


###################
### XRDMapStack ###
###################

def initialize_xrdmapstack_hdf(xrdmapstack,
                               hdf_file):
    
    with h5py.File(hdf_file, 'w-') as f:
        # Base metadata from xrdmaps
        base_grp = f.require_group(xrdmapstack._hdf_type)
        base_grp.attrs['scan_id'] = xrdmapstack.scan_id
        base_grp.attrs['beamline'] = xrdmapstack.beamline #'5-ID (SRX)'
        base_grp.attrs['facility'] = xrdmapstack.facility #'NSLS-II'
        base_grp.attrs['detector'] = xrdmapstack.detector #'dexela'
        base_grp.attrs['energy'] = xrdmapstack.energy
        base_grp.attrs['wavelength'] = xrdmapstack.wavelength
        base_grp.attrs['theta'] = xrdmapstack.theta
        base_grp.attrs['use_stage_rotation'] = int(xrdmapstack.use_stage_rotation)
        base_grp.attrs['time_stamp'] = ttime.ctime()

        # Metadata for only xrdmapstack
        base_grp.attrs['hdf_path'] = xrdmapstack.hdf_path
        base_grp.attrs['rocking_axis'] = xrdmapstack.rocking_axis
        # Might be useful for completely ignoring individual xrdmaps. Have not yet decided
        # base_grp.attrs['shape'] = xrdmapstack.shape
        # base_grp.attrs['image_shape'] = xrdmapstack.image_shape
        # base_grp.attrs['map_shape'] = xrdmapstack.map_shape
        # base_grp.attrs['swapped_axes'] = xrdmapstack._swapped_axes

        # Generate emtpy dataset of extra_metadata
        extra_md = base_grp.create_dataset('extra_metadata',
                                            data=h5py.Empty("f"))
        for key, value in xrdmapstack.xdms_extra_metadata.items():
            extra_md.attrs[key] = value

        if (not hasattr(xrdmapstack, 'scan_input')
            or xrdmapstack.scan_input is None):
            scan_input = []
        else:
            scan_input = xrdmapstack.scan_input
        base_grp.attrs['scan_input'] = scan_input

        # Special consideration for when dwell wasn't being recorded...
        if (not hasattr(xrdmapstack, 'dwell')
            or xrdmapstack.dwell is None):
            dwell = ['',] * len(xrdmapstack)
        else:
            dwell = xrdmapstack.dwell
        base_grp.attrs['dwell'] = dwell


# Does NOT load the individual XRDMaps
def load_xrdmapstack_hdf(filename,
                         wd,
                         load_xdms_vector_map=True,
                         ):
    
    # Figuring out hdf file stuff
    hdf_type = 'xrdmapstack' # hard-coded
    hdf_path = pathify(wd, filename, '.h5')

    hdf = h5py.File(hdf_path, 'r')

    # Protect hdf
    try:
        if hdf_type in hdf:
            base_grp = hdf[hdf_type]
        else:
            if len(hdf.keys()) == 1:
                found_hdf_type = hdf.keys()[0]
                err_str = ('Found HDF of incorrect type. Expected '
                           + f'{hdf_type}, but found {found_hdf_type}'
                           + 'instead.')
                if found_hdf_type == 'xrfmap':
                    err_str += ('\nXRF maps should be handled with '
                                + 'pyxrf or similar packages.')
            else:
                err_str = ('Found HDF of incorrect type. Expected '
                           + f'{hdf_type}, but found unknown type '
                           + 'instead.')
            raise TypeError(err_str)
        
        # Load reflections first
        # Built-in pandas hdf support needs this
        spots_3D = None
        if 'reflections' in base_grp.keys():
            print('Loading reflection spots...', end='', flush=True)

            if 'spots_3D' in base_grp['reflections'].keys():
                has_spots_3D = True

            hdf.close()
            if has_spots_3D:
                spots_3D = pd.read_hdf(hdf_path,
                                    key=f'{hdf_type}/reflections/spots_3D')

            hdf = h5py.File(hdf_path, 'a')  
            base_grp = hdf[hdf_type]
            print('done!')

        # Load base metadata
        base_md = dict(base_grp.attrs.items())

        # Backwards compatability
        if 'scanid' in base_md:
            # Change dictionary
            base_md['scan_id'] = base_md['scanid']
            del base_md['scanid']
            # Overwrite hdf values
            write_flag = (hdf.mode == 'a')
            if not write_flag:
                hdf.close()
                hdf = h5py.File(hdf_path, 'a')
                base_grp = hdf[hdf_type]
            base_grp.attrs['scan_id'] = base_md['scan_id']
            del base_grp.attrs['scanid']
            if not write_flag:
                hdf.close()
                hdf = h5py.File(hdf_path, 'r')
                base_grp = hdf[hdf_type]

        # Load extra metadata
        extra_md = {}
        # Check for backwards compatibility
        if 'extra_metadata' in base_grp.keys(): 
            for key, value in base_grp['extra_metadata'].attrs.items():
                extra_md[key] = value

        # Backwards compatibility
        if ('vectorized_map' in base_grp
            and 'virtual_shape' in base_grp['vectorized_map'].attrs):
            map_shape = base_grp['vectorized_map'].attrs['virtual_shape']
            write_flag = (hdf.mode == 'a')
            if not write_flag:
                hdf.close()
                hdf = h5py.File(hdf_path, 'a')
                base_grp = hdf[hdf_type]
            del base_grp['vectorized_map'].attrs['virtual_shape']
            hdf[f'{hdf_type}/vectorized_map'].attrs[
                                    'vectorized_map_shape'] = map_shape
            if not write_flag:
                hdf.close()
                hdf = h5py.File(hdf_path, 'r')
                base_grp = hdf[hdf_type]
        
        # 'vector_map' is renamed later to 'xdms_vector_map'
        vector_dict = _load_xrd_hdf_vector_data(
                                base_grp,
                                load_vector_map=load_xdms_vector_map)
            
        # if load_xdms_vector_map:
        #     # Same functions works for XRDMaps and XRDMapStacks
        #     vector_dict = _load_xrd_hdf_vectorized_map_data(base_grp)
        # else:
        #     vector_dict = None
    
    # Catch error, close hdf, re-raise
    except Exception as e:
        if hdf is not None:
            hdf.close()
        raise e

    # Return dictionary of useful values
    ouput_dict = {'base_md' : base_md,
                  'xdms_extra_metadata' : extra_md,
                  'spots_3D' : spots_3D,
                  'vector_dict' : vector_dict
                  }

    return ouput_dict
