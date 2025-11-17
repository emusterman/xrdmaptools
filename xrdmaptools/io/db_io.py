import numpy as np
from skimage import io
import h5py
import os
from scipy.stats import mode
from collections import OrderedDict


# Database variables
c, db = None, None


# Working at the beamline...
def _load_tiled_catalog():
    global c
    if c is None:
        print('Connecting to database...', end='', flush=True)
        try:
            from tiled.profiles import ProfileNotFound
        except ModuleNotFoundError:
            print('failed.')
            err_str = ('Cannot load tiled catalog at this time. '
                       + 'Try another database.')
            raise ModuleNotFoundError(err_str)        

        try:
            from tiled.client import from_profile
            c = from_profile('srx')
            print('done!')
        except (ModuleNotFoundError, ProfileNotFound):
            print('failed.')
            err_str = ('Cannot load tiled catalog at this time. '
                    + 'Try another database.')
            raise RuntimeError(err_str)



def _load_databroker():
    global db
    if db is None:
        print('Connecting to database...', end='', flush=True)
        try:
            from tiled.profiles import ProfileNotFound
        except ModuleNotFoundError:
            print('failed.')
            err_str = ('Cannot load tiled catalog at this time. '
                       + 'Try another database.')
            raise ModuleNotFoundError(err_str)   
        
        try:
            from databroker.v1 import Broker
            db = Broker.named('srx')
            print('done!')
        except (ModuleNotFoundError, ProfileNotFound):
            print('failed.')
            err_str = ('Cannot load databroker catalog at this time. '
                       + 'Try another database.')
            raise RuntimeError(err_str)


# Useful detectors
supported_xrd_dets = ['dexela', 'merlin', 'eiger']


# Wrapper for all load data options...
# Might only work for fly scans
def load_data(scan_id=-1,
              broker='manual',
              detectors=None,
              data_keys=None,
              xrd_dets=None,
              returns=None,
              repair_method='replace',
              verbose=True):

    # Load data from tiled
    if str(broker).lower() in ['tiled']:
        _load_tiled_catalog()
        bs_run = c[int(scan_id)]

        out = load_tiled_data(scan_id=scan_id,
                              detectors=detectors,
                              data_keys=data_keys,
                              xrd_dets=xrd_dets,
                              returns=returns,
                              verbose=verbose)

    # Load data from databroker
    elif str(broker).lower() in ['db', 'databroker', 'broker']:
        _load_databroker()
        bs_run = db[int(scan_id)]

        out = load_db_data(scan_id=scan_id, 
                           detectors=detectors,
                           data_keys=data_keys,
                           xrd_dets=xrd_dets,
                           returns=returns,
                           verbose=verbose)
    
    # Load data manually
    elif str(broker).lower() in ['manual']:
        try:
            _load_tiled_catalog()
            bs_run = c[int(scan_id)] # defualt basic data from tiled for future proofing
            broker = 'tiled'
        except: # what error should this throw???
            _load_databroker()
            bs_run = db[int(scan_id)]
            broker = 'db'

        out = manual_load_data(scan_id=scan_id,
                               broker=broker,
                               detectors=detectors,
                               data_keys=data_keys,
                               xrd_dets=xrd_dets,
                               returns=returns,
                               repair_method=repair_method,
                               verbose=verbose)

    return out


# Base Tiled loading function
def load_tiled_data(scan_id=-1,
                    detectors=None,
                    data_keys=None,
                    xrd_dets=None,
                    returns=None,
                    verbose=True):
    
    _load_tiled_catalog()
    bs_run = c[int(scan_id)]

    # Load detectors
    if detectors is None:
        detectors = bs_run.start['scan']['detectors']
    else:
        detectors = [detector.name if type(detector) is not str else str(detector)
                     for detector in detectors]
    if xrd_dets is None:
        xrd_dets = [detector for detector in detectors if detector in supported_xrd_dets]    

    scan_md = _load_scan_metadata(bs_run)
    scan_md.update(_load_baseline_metadata(bs_run))
    print(f'Loading data from tiled for scan {scan_md["scan_id"]}...')

    data_dict = {}
    # enc1 is x for nano_scan_and_fly. Is it always the fast axis or always x??
    if data_keys is None:
        data_keys = ['enc1', 'enc2', 'xs_fluor', 'i0', 'i0_time', 'im', 'it']

    for detector in xrd_dets:
        data_keys.append(f'{detector}_image')
    
    for key in data_keys:
        print(f'Loading data from {key}...', end='', flush=True)
        data_dict[key] = np.array(bs_run['stream0']['data'][key])
        print('done!')
    
    if 'dark' in bs_run:
        for detector in xrd_dets:
            if f'{detector}_image' in bs_run['dark']['data']:
                print(f'Loading data from {detector}_dark...', end='', flush=True)
                dark = np.array(bs_run['dark']['data'][f'{detector}_image'])
                dark = dark.squeeze().reshape(-1, *dark.shape[-2:])
                data_dict[f'{detector}_dark'] = np.median(dark, axis=0)
                print('done!')

    out = [data_dict, scan_md]

    if returns is not None:
        if 'data_keys' in returns:
            out.append(data_keys)
        if 'xrd_dets' in returns:
            out.append(xrd_dets)

    print(f'Data loaded for scan {scan_md["scan_id"]}!')
    return out


# Base function for loading data from DataBroker
def load_db_data(scan_id=-1,
                 detectors=None,
                 data_keys=None,
                 xrd_dets=None,
                 returns=None,
                 verbose=True):

    _load_databroker()
    bs_run = db[int(scan_id)]

    # Load detectors
    if detectors is None:
        detectors = bs_run.start['scan']['detectors']
    else:
        detectors = [detector.name if type(detector) is not str else str(detector)
                     for detector in detectors]
    if xrd_dets is None:
        xrd_dets = [detector for detector in detectors if detector in supported_xrd_dets]

    scan_md = _load_scan_metadata(bs_run)
    scan_md.update(_load_baseline_metadata(bs_run))
    print(f'Loading data from databroker for scan {scan_md["scan_id"]}...')

    data_dict = {}
    # enc1 is x for nano_scan_and_fly. Is it always the fast axis or always x??
    if data_keys is None:
        data_keys = ['enc1', 'enc2', 'xs_fluor', 'i0', 'i0_time', 'im', 'it']

    for detector in xrd_dets:
        data_keys.append(f'{detector}_image')
    for key in data_keys:
        print(f'Loading data from {key}...', end='', flush=True)
        d = bs_run.data(key, stream_name='stream0', fill=True)
        data_dict[key] = np.array(list(d))
        print('done!')

    if 'dark' in bs_run:
        for detector in xrd_dets:
            if f'{detector}_image' in bs_run['dark']['data']:
                print(f'Loading data from {detector}_dark...', end='', flush=True)
                dark = np.array(bs_run['dark']['data'][f'{detector}_image'])
                dark = dark.squeeze().reshape(-1, *dark.shape[-2:])
                data_dict[f'{detector}_dark'] = np.median(dark, axis=0, dtype=np.float32)
                print('done!')

    out = [data_dict, scan_md]
    
    if returns is not None:
        if 'data_keys' in returns:
            out.append(data_keys)
        if 'xrd_dets' in returns:
            out.append(xrd_dets)

    print(f'Data loaded for scan {scan_md["scan_id"]}!')
    return out


# Manual load data
def manual_load_data(scan_id=-1,
                     broker='tiled',
                     data_keys=None,
                     detectors=None,
                     xrd_dets=None,
                     returns=None,
                     repair_method='replace',
                     verbose=True):
    
    # Setup verbosity
    vprint = print if verbose else lambda *a, **k: None

    if str(broker).lower() in ['tiled']:
        _load_tiled_catalog()
        bs_run = c[int(scan_id)]
    elif str(broker).lower() in ['db', 'databroker', 'broker']:
        _load_databroker()
        bs_run = db[int(scan_id)]

    # Load detectors
    if detectors is None:
        detectors = bs_run.start['scan']['detectors']
    else:
        detectors = [detector.name if type(detector) is not str else str(detector)
                     for detector in detectors]
    if xrd_dets is None:
        xrd_dets = [detector for detector in detectors if detector in supported_xrd_dets]      

    scan_md = _load_scan_metadata(bs_run)
    scan_md.update(_load_baseline_metadata(bs_run))
    vprint(f'Manually loading data for scan {scan_md["scan_id"]}...')

    # Add default data keys
    if data_keys is None:
        data_keys = ['enc1', 'enc2', 'xs_fluor', 'i0', 'i0_time', 'im', 'it']

    # Append XRD data keys
    for det in xrd_dets:
        # Data
        data_keys.append(f'{det}_image')
        # Null map
        data_keys.append(f'{det}_null_map')

    # Get relevant hdf_information
    r_paths, r_specs, r_shapes = _get_resources(bs_run, data_keys)

    # Load data
    _empty_lists = [[] for _ in range(len(data_keys))]
    data_dict = dict(zip(data_keys, _empty_lists))
    dropped_rows, broken_rows = [], []

    # Parse r_paths
    supp_dict = {}
    if 'dark' in r_paths:
        # Only dexela uses dark-field
        with h5py.File(r_paths['dark']['dexela_image'][0]) as f:
            dark = f['entry/data/data'][:].squeeze()
            dark = dark.reshape((-1, *dark.shape[-2:]))
            supp_dict['dexela_dark'] = np.median(dark, axis=0)
    if 'stream0' in r_paths:
        r_paths = r_paths['stream0']
        r_specs = r_specs['stream0']
        r_shapes = r_shapes['stream0']
    elif 'primary' in r_paths:
        r_paths = r_paths['primary']
        r_specs = r_specs['primary']
        r_shapes = r_shapes['primary']
    
    # Convert from data_key keys to spec keys
    r_paths = {r_specs[key] : r_paths[key] for key in r_paths}
    spec_shapes = {r_specs[key] : r_shapes[key] for key in r_shapes}
    
    # These are all essentially handlers with more conditionals for fixing data
    # Encoders
    r_key = 'ZEBRA_HDF51'
    if r_key in r_paths.keys():
        vprint('Loading encoders...')
        enc_keys = [value for value in data_keys if value in ['enc1', 'enc2', 'enc3', 'zebra_time']]
        for r_path in r_paths[r_key]:
            with h5py.File(r_path, 'r') as f:
                for key in enc_keys:
                    data_dict[key].append(np.array(f[key]))
        # Stack data into array
        for key in enc_keys:
            dr_rows, br_rows = _flag_broken_rows(data_dict[key],
                                                 key,
                                                 expected_shape=spec_shapes[r_key])
            dropped_rows += dr_rows
            broken_rows += br_rows

    # Scalers
    r_key = 'SIS_HDF51'
    if r_key in r_paths.keys():
        vprint('Loading scalers...')
        sclr_keys = [value for value in data_keys if value in ['i0', 'im', 'it', 'time']]
        if 'i0_time' in data_keys and 'time' not in sclr_keys:
            sclr_keys.append('time')
        time_key = None
        for r_path in r_paths[r_key]:
            with h5py.File(r_path, 'r') as f:
                if 'time' in sclr_keys:
                    time_key = [key for key in f.keys() if 'time' in key][0]
                    sclr_keys[sclr_keys.index('time')] = time_key
                    data_dict[time_key] = []
                for key in sclr_keys:
                    data_dict[key].append(np.array(f[key]))
        # Stack data into array
        if 'i0_time' in data_keys:
            data_dict['i0_time'] = [d / 50e6 for d in data_dict[time_key]] # 50 MHz clock
        if time_key not in data_keys and time_key in data_dict.keys():
            del data_dict[time_key]
        for key in ['i0', 'im', 'it', 'i0_time']:
            if key not in data_dict:
                continue
            dr_rows, br_rows = _flag_broken_rows(data_dict[key],
                                                 key,
                                                 expected_shape=spec_shapes[r_key])
            dropped_rows += dr_rows
            broken_rows += br_rows

    # Xpress3
    for r_key in ['XSP3_FLY', 'XPS3_FLY']:
        if r_key in r_paths.keys():
            vprint('Loading xspress3...')
            key = 'xs_fluor'
            for r_path in r_paths[r_key]:
                with h5py.File(r_path, 'r') as f:
                    data_dict[key].append(np.array(f['entry/data/data']))
            # Stack data into array
            dr_rows, br_rows = _flag_broken_rows(data_dict[key],
                                                 key,
                                                 expected_shape=spec_shapes[r_key])
            dropped_rows += dr_rows
            broken_rows += br_rows
            break

    # Dexela
    r_key = 'DEXELA_FLY_V1'
    if r_key in r_paths.keys():
        vprint('Loading dexela...')
        key = 'dexela_image'
        for r_path in r_paths[r_key]:
            f = h5py.File(r_path, 'r')
            data_dict[key].append(f['entry/data/data'])
        # Stack data into array
        dr_rows, br_rows = _flag_broken_rows(data_dict[key],
                                             key,
                                             expected_shape=spec_shapes[r_key])
        dropped_rows += dr_rows
        broken_rows += br_rows

    # Not tested!!!
    # Merlin
    for r_key in ['MERLIN_FLY_V1', 'MERLIN_FLY_STREAM_V1']:
        if r_key in r_paths.keys():
            vprint('Loading merlin...')
            key = 'merlin_image'
            for r_path in r_paths[r_key]:
                with h5py.File(r_path, 'r') as f:
                    data_dict[key].append(np.array(f['entry/data/data']))
            # Stack data into array
            dr_rows, br_rows = _flag_broken_rows(data_dict[key],
                                                 key,
                                                 expected_shape=spec_shapes[r_key])
            dropped_rows += dr_rows
            broken_rows += br_rows
            break
    
    # Eiger
    for r_key in ['AD_HDF5']:
        if r_key in r_paths.keys():
            vprint('Loading eiger...')
            key = 'eiger_image'
            for row_i, r_path in enumerate(r_paths[r_key]):
                f = h5py.File(r_path, 'r')
                row_data = f['entry/data/data']
                dropped_inds = np.array(f['entry/instrument/NDAttributes/NDArrayUniqueId'][:])
                # Built-in eiger fixes first
                filled_pts = spec_shapes[r_key][0] - len(dropped_inds)
                null_row = np.zeros(spec_shapes[r_key], dtype=np.bool_)
                if filled_pts > 0:
                    print(f'Auto-filled {filled_pts} points in row {row_i} for {key}.')
                    full_row = np.zeros(spec_shapes[r_key], dtype=row_data.dtype)
                    full_row[dropped_inds] = row_data
                    null_row[dropped_inds] = True
                    row_data = full_row
                    del full_row
                data_dict[key].append(row_data)
                data_dict['eiger_null_map'].append(null_row)

            # Stack data into array
            dr_rows, br_rows = _flag_broken_rows(data_dict[key],
                                                 key,
                                                 expected_shape=spec_shapes[r_key])
            dropped_rows += dr_rows
            broken_rows += br_rows
            break

    # Repair data
    dropped_rows = sorted(list(np.unique(dropped_rows)))
    broken_rows = sorted(list(np.unique(broken_rows)))
    data_dict = _repair_data_dict(data_dict,
                                  dropped_rows,
                                  broken_rows,
                                  repair_method=repair_method,
                                  shape_dict=r_shapes)
    data_dict.update(supp_dict)

    out = [data_dict, scan_md]

    if returns is not None:
        if 'data_keys' in returns:
            out.append(data_keys)
        if 'xrd_dets' in returns:
            out.append(xrd_dets)

    vprint(f'Data loaded for scan {scan_md["scan_id"]}!')
    return out


# Helper function
def get_scantype(scan_id=-1,
                 broker='tiled'):
    # Load data from tiled
    if str(broker).lower() in ['tiled']:
        _load_tiled_catalog()
        bs_run = c[int(scan_id)]

    # Load data from databroker
    elif str(broker).lower() in ['db', 'databroker', 'broker']:
        _load_databroker()
        bs_run = db[int(scan_id)]

    else:
        raise ValueError(f"Unknown broker type: {broker}. Only 'tiled' and 'databroker' suppported.")
    
    scantype = bs_run.start['scan']['type']

    return scantype


# TODO: implement way of getting units
def _load_scan_metadata(bs_run, keys=None):

    if keys is None:
        keys = ['scan_id',
                'uid',
                'beamline_id',
                'type',
                'detectors',
                'motors',
                'energy',
                'dwell',
                'time_str',
                'sample_name',
                'theta',
                'scan_input']

    remaining_keys = []
    scan_md = {}
    for key in keys:
        if key in bs_run.start.keys():
            scan_md[key] = bs_run.start[key]
        elif key in bs_run.start['scan'].keys():
            scan_md[key] = bs_run.start['scan'][key]
            if key == 'theta': # Not very generalizable...
                scan_md[key] = bs_run.start['scan'][key]['val'] / 1000
        elif key == 'motors':
            scan_md['motors'] = [
                bs_run.start['scan']['fast_axis']['motor_name'],
                bs_run.start['scan']['slow_axis']['motor_name']
            ]
        else:
            remaining_keys.append(key)
        
    for key in remaining_keys:
        if key == 'theta':
            theta = bs_run['baseline']['data']['nano_stage_th'][0]
            scan_md['theta'] = theta / 1000
        if key == 'energy':
            energy = bs_run['baseline']['data']['energy_energy'][0]
            scan_md['energy'] = np.round(energy, 3)
    
    # Unpack detector metatdata if available
    if ('detectors' in scan_md
        and isinstance(scan_md['detectors'], dict)):
        for det_key, det_md in scan_md['detectors'].items():
            for md_key, md_value in det_md.items():
                scan_md[f'{det_key}_{md_key}'] = md_value
        scan_md['detectors'] = list(scan_md['detectors'].keys())

    return scan_md


def _load_baseline_metadata(bs_run, keys=None):

    baseline_md = {}

    # Hard-coded look for preamp settings in baseline
    for scaler_key in ['i0', 'im', 'it']:
        if all([f'{scaler_key}_preamp_sens_{suffix}' in bs_run.baseline['data']
                for suffix in ['num', 'unit']]):
            sensitivity = float(bs_run.baseline['data'][f'{scaler_key}_preamp_sens_num'][0])
            unit = bs_run.baseline['data'][f'{scaler_key}_preamp_sens_unit'][0]

            # Apply unit
            if 'pA' in unit:
                sensitivity /= 1e12
            elif 'nA' in unit:
                sensitivity /= 1e9
            elif 'uA' in unit:
                sensitivity /= 1e6
            elif 'mA' in unit:
                sensitivity /= 1e3
        
            baseline_md[f'{scaler_key}_sensitivity'] = float(sensitivity) # another float conversion??

    return baseline_md


def _flag_broken_rows(data_list, msg, expected_shape=None):
    # Majority of row shapes
    if expected_shape is None:
        expected_shape = tuple(mode([d.shape for d in data_list])[0])

    dropped_rows, broken_rows = [], []
    for row, d in enumerate(data_list):
        # Check for image shape issues
        if d.ndim == 4 and d.shape[-2:] != expected_shape[-2:]:
            print(f'WARNING: {msg} data from row {row} has an incorrect image shape.')
            dropped_rows.append(row)
            broken_rows.append(row)
        elif d.shape[0] != expected_shape[0]:
            print(f'WARNING: {msg} data from row {row} captured only {d.shape[0]}/{expected_shape[0]} points.')
            broken_rows.append(row)

    return dropped_rows, broken_rows


def _repair_data_dict(data_dict,
                      dropped_rows,
                      broken_rows,
                      repair_method='replace',
                      shape_dict=None):
    if repair_method.lower() not in ['flatten', 'fill', 'replace']:
        err_str = 'Only "flatten", "fill", and "replace" repair methods are supported.'
        raise ValueError()
    if repair_method.lower() == 'fill' and shape_dict is None:
        err_str = 'repair_method of "fill" requires a shape_dict input.'
        raise ValueError(err_str)

    # Get keys
    keys, null_keys = [], []
    for k in data_dict.keys():
        if 'null_map' in k:
            null_keys.append(k)
        else:
            keys.append(k)

    # Check data shape
    num_rows = -1
    for key in keys:
        if num_rows == -1:
            num_rows = len(data_dict[key])
        elif len(data_dict[key]) != num_rows:
            raise ValueError('Data keys have different number of rows!')

    # Add key to track filled images
    if repair_method == 'fill':
        # Fill null maps with placeholders
        for k in null_keys:
            if len(data_dict[k]) == 0:
                data_dict[k] = [[] for _ in range(num_rows)]

    # Check to see if there are any repairs
    if len(dropped_rows) > 0 or len(broken_rows) > 0:
        print(f'Repairing data with "{repair_method}" method...')
    else:
        # If all rows are loaded, convert to numpy array
        # I am not sure if this condition will ever be met
        for key in keys:
            if not np.any([isinstance(row, h5py._hl.dataset.Dataset)
                           for row in data_dict[key]]):
                data_dict[key] = np.asarray(data_dict[key])
        return data_dict  
        
    last_good_row = -1
    queued_rows = []
    for row in range(num_rows):
        # Only drop rows when flattening data. Do not fill data when flattening either
        if repair_method == 'flatten':
            if row in dropped_rows:
                print(f'Removing data from row {row}.')
                for key in keys:
                    data_dict[key].pop(row)
                # Adjust remaining rows indices
                for ind, rem_row in enumerate(dropped_rows):
                    if rem_row > row:
                        dropped_rows[ind] -= 1
        
        # Only Replace data for broken rows
        elif repair_method == 'replace':
            if row in broken_rows:
                if last_good_row == -1:
                    queued_rows.append(row)
                else:
                    print(f'Data in row {row} replaced with data in row {last_good_row}.')
                    for key in keys:
                        data_dict[key][row] = np.asarray(data_dict[key][last_good_row])
        
            else:
                last_good_row = row
                if len(queued_rows) > 0:
                    for q_row in queued_rows:
                        print(f'Data in row {q_row} replaced with data in row {last_good_row}.')
                        for key in keys:
                            data_dict[key][q_row] = np.asarray(data_dict[key][last_good_row])
                        queued_rows = []

        # Only Fill data for broken rows
        elif repair_method == 'fill':
            for k in null_keys:
                if len(data_dict[k][row]) == 0:
                    data_dict[k][row] = np.zeros(len(list(data_dict.values())[0][last_good_row]),
                                                  dtype=np.bool_)

            if row in broken_rows:
                for key in keys:
                    filled_pts = shape_dict[key][0] - len(data_dict[key][row])
                    if filled_pts > 0:
                        print(f'Filled {filled_pts} points in row {row} for {key}.')

                        zero_row = np.zeros(shape_dict[key],
                                            dtype=data_dict[key][row].dtype)
                        zero_row[:len(data_dict[key][row])] = data_dict[key][row]
                        data_dict[key][row] = zero_row

                        null_key = f'{key.split('_')[0]}_null_map'
                        if null_key in null_keys:
                            data_dict[null_key][row][-filled_pts:] = [True,] * filled_pts
        
        else:
            raise ValueError('Unknown repair method indicated.')
    
    # Do not fully flatten if only dropping some rows I guess
    if repair_method == 'flatten' and len(broken_rows) > 0:
        print('Map is missing pixels. Flattening data.')
        for key in keys:
            data_shape = data_dict[key][0].shape[1:]  # Grap data_shape from first not dropped row
            flat_d = np.array([x for y in data_dict[key] for x in y]) # List comprehension is dumb
            data_dict[key] = flat_d.reshape(flat_d.shape[0], 1, *data_shape)

    # Broad conversion to all arrays
    # There is a better way to do this...
    for key in data_dict.keys():
        # If at least one row is still lazy loaded, then leave it that way
        if not np.any([isinstance(row, h5py._hl.dataset.Dataset)
                       for row in data_dict[key]]):
            data_dict[key] = np.asarray(data_dict[key])
    
    return data_dict


def _get_resources(bs_run, data_keys):

    stream_uids = {}
    descriptors = []
    event_pages = []
    resources = {}

    for i, (name, doc) in enumerate(bs_run.documents()):
        
        # Add all streams with data_keys matching those requested
        if (name == 'descriptor'
            and any([key in data_keys for key in doc['data_keys'].keys()])):
            descriptors.append((
                doc['uid'],
                doc['name'],
                {key : doc['data_keys'][key]['shape'] for key in data_keys if key in doc['data_keys']}
            ))
            stream_uids[doc['uid']] = doc['name']
    
        # Add all event_pages with data_keys matching those requested
        if (name == 'event_page'
            and any([key in data_keys for key in doc['data'].keys()])):
            event_pages.append((
                doc['descriptor'],
                doc['seq_num'][0],
                {key : doc['data'][key][0].split('/')[0] for key in data_keys if key in doc['data']}
            ))

        # Add all resouces. Cannot distinguish yet
        if name == 'resource':
            resources[doc['uid']] = (
                doc['spec'],
                doc['resource_path']
            )

    # Build container
    r_paths, r_specs, r_shapes = {}, {}, {}
    for uid, name, shapes in descriptors:
        data_dict = {}
        for key in shapes:
            data_dict[key] = [None,] * sum([page[0] == uid for page in event_pages])
        r_paths[name] = data_dict
        r_specs[name] = {key : None for key in shapes}
        r_shapes[name] = shapes

    # Populate container with uids
    for uid, seq_num, resource_uids in event_pages:
        for key, resource_uid in resource_uids.items():
            r_paths[stream_uids[uid]][key][seq_num - 1] = resources[resource_uid][1]
            r_specs[stream_uids[uid]][key] = resources[resource_uid][0]
    
    # Reduce duplicates
    for name, path_dict in r_paths.items():
        for data_key, paths in path_dict.items():
            path_dict[data_key] = list(OrderedDict.fromkeys(paths))

    return r_paths, r_specs, r_shapes

###########################
### Data Pre-processing ###
###########################


# General function for making composite patterns
def make_composite_pattern(xrd_data,
                           method='sum',
                           subtract=None):

    if not isinstance(xrd_data, list):
        xrd_data = [xrd_data]

    method = method.lower()

    # Can the axis variable be made to always create images along the last two images??
    comps = []
    for xrd in xrd_data:
        xrd = np.asarray(xrd)
        axis = tuple(range(xrd.ndim - 2))

        if method in ['sum', 'total']:
            comp = np.sum(xrd, axis=axis) / np.prod(xrd.shape[:-2])
        elif method in ['mean', 'avg', 'average']:
            comp = np.mean(xrd, axis=axis)
        elif method in ['max', 'maximum']:
            comp = np.max(xrd, axis=axis)
        elif method in ['med', 'median']:
            if subtract in ['med', 'median']:
                print("Composite method and subtract cannot both be 'median'!")
                print("Changing subract to None")
                subtract = None
            comp = np.median(xrd, axis=axis)
        else:
            raise ValueError("Composite method not allowed.")

        if subtract is None:
            pass
        elif type(subtract) is np.ndarray:
            comp = comp - subtract
        elif subtract.lower() in ['min', 'minimum']:
            comp = comp - np.min(xrd, axis=axis)
        elif subtract.lower() in ['med', 'median']:
            comp = comp - np.median(xrd, axis=axis)
        else:
            print('Unknown subtract input. Proceeding without any subtraction.')

        comps.append(np.squeeze(comp))
    
    return comps


# Wrapper for composite function, specifically for calibration data
def make_calibration_pattern(xrd_data): 

    comps = make_composite_pattern(xrd_data, method='max', subtract='min')

    return comps


##################################
### Base Saving Data Functions ###
##################################


# General function to save xrd data to tifs
def _save_xrd_tifs(xrd_data,
                   xrd_dets=None,
                   scan_id=None,
                   wd=None,
                   filenames=None):

    if (scan_id is None and filenames is None):
        raise ValueError('Must define scan_id or filename to name the save file.')
    elif (xrd_dets is None and filenames is None):
        raise ValueError('Must define xrd_dets or filename to name the save file.')
    
    if not isinstance(xrd_data, list):
        xrd_data = [xrd_data]

    if wd is None:
        wd = os.getcwd()   
    if filenames is None:
        filenames = []
        for detector in xrd_dets:
            filenames.append(f'scan{scan_id}_{detector}_xrd.tif')

    if len(filenames) != len(xrd_data):
        raise ValueError('Length of filenames does not match length of xrd_data')
    
    for i, xrd in enumerate(xrd_data):
        io.imsave(f'{wd}{filenames[i]}',
                  np.asarray(xrd).astype(np.uint16),
                  check_contrast=False)
        # I think the data should already be an unsigned integer
        #io.imsave(f'{wd}{filenames[i]}', xrd, check_contrast=False)
    print(f'Saved pattern(s) for scan {scan_id}!')


# General function to save composite pattern as tif
def _save_composite_pattern(comps,
                            method,
                            subtract,
                            xrd_dets=None,
                            scan_id=None,
                            wd=None,
                            filenames=None):

    if subtract is None:
        subtract = ''
    elif isinstance(subtract, np.ndarray):
        subtract = '-custom'
    else:
        subtract = f'-{subtract}'

    if filenames is None:
        filenames = []
        for detector in xrd_dets:
            filenames.append(f'scan{scan_id}_{detector}_{method}{subtract}_composite.tif')
    
    print('Saving composite pattern...')
    _save_xrd_tifs(comps,
                   xrd_dets=xrd_dets,
                   scan_id=scan_id,
                   wd=wd,
                   filenames=filenames)


def _save_calibration_pattern(xrd_data,
                              xrd_dets,
                              scan_id=None,
                              wd=None):

    if not isinstance(xrd_data, list):
        xrd_data = [xrd_data]
    
    if wd is None:
        wd = os.getcwd()

    filenames = []
    for detector in xrd_dets:
        filenames.append(f'scan{scan_id}_{detector}_calibration.tif')
    
    print('Saving calibration pattern...')
    _save_xrd_tifs(xrd_data,
                   xrd_dets=xrd_dets,
                   scan_id=scan_id,
                   wd=wd,
                   filenames=filenames)


# General function save scan metadata 
def _save_map_parameters(data_dict,
                         scan_id,
                         data_keys=[],
                         wd=None,
                         filename=None):

    if wd is None:
        wd = os.getcwd()
    if filename is None:
        filename = f'scan{scan_id}_map_parameters.txt'
    if data_keys == []:
        data_keys = ['enc1', 'enc2', 'i0', 'i0_time', 'im', 'it']
        for key in data_keys:
            if key not in data_dict:
                data_keys.remove(key)
                print(f'WARNING: data_key ({key}) requested but not in data_dict.')

    map_data = np.stack([data_dict[key].ravel() for key in data_keys])
    np.savetxt(f'{wd}{filename}', map_data)
    print(f'Saved map parameters for scan {scan_id}!')


def _save_scan_md(scan_md,
                  scan_id,
                  wd=None,
                  filename=None):
    import json

    if not isinstance(scan_md, dict):
        raise TypeError('Scan metadata not provided as a dictionary.')
    
    if wd is None:
        wd = os.getcwd()
    if filename is None:
        filename = f'scan{scan_id}_scan_md.txt'

    with open(f'{wd}{filename}', 'w') as f:
        f.write(json.dumps(scan_md))
    print(f'Saved metadata for scan {scan_id}!')


##############################
### General Save Functions ###
##############################


# Function to load and save xrd data to tifs
def save_xrd_tifs(scan_id=-1,
                  broker='tiled',
                  detectors=None,
                  data_keys=[],
                  wd=None,
                  filenames=None,
                  repair_method='replace'):

    data_dict, scan_md, data_keys, xrd_dets = load_data(scan_id=scan_id,
                                                        broker=broker,
                                                        detectors=detectors,
                                                        data_keys=data_keys,
                                                        returns=['data_keys',
                                                                 'xrd_dets'],
                                                        repair_method=repair_method)

    xrd_data = [data_dict[f'{xrd_det}_image'] for xrd_det in xrd_dets]
    
    _save_xrd_tifs(xrd_data,
                   xrd_dets=xrd_dets,
                   scan_id=scan_md['scan_id'], # Will return the correct value
                   wd=wd,
                   filenames=filenames)


# Function to load and save composite pattern as tif
def save_composite_pattern(scan_id=-1,
                           broker='manual',
                           method='sum',
                           subtract=None,
                           detectors=None,
                           data_keys=[], 
                           wd=None,
                           filenames=None):

    data_dict, scan_md, data_keys, xrd_dets = load_data(scan_id=scan_id,
                                                        broker=broker,
                                                        detectors=detectors,
                                                        data_keys=data_keys,
                                                        returns=['data_keys',
                                                                 'xrd_dets'],
                                                        repair_method='flatten')

    xrd_data = [data_dict[f'{xrd_det}_image'] for xrd_det in xrd_dets]

    # Mask out eiger hot pixels
    # May have some redundant operations
    for xrd, det in zip(xrd_data, xrd_dets):
        if det == 'eiger':
            xrd = np.asarray(xrd)
            mask = np.min(xrd, axis=range(xrd.ndim - 2)) == 2**32 - 1 # Saturated 32 bit unsigned integer
            xrd[..., mask] = 0

    comps = make_composite_pattern(xrd_data, method=method, subtract=subtract)

    _save_composite_pattern(comps,
                            method,
                            subtract,
                            xrd_dets=xrd_dets,
                            scan_id=scan_md['scan_id'],
                            wd=wd,
                            filenames=filenames)


# Function to load and save calibration pattern as tif
def save_calibration_pattern(scan_id=-1,
                             broker='manual',
                             detectors=None,
                             data_keys=[], 
                             wd=None):

    data_dict, scan_md, data_keys, xrd_dets = load_data(scan_id=scan_id,
                                                        broker=broker,
                                                        detectors=detectors,
                                                        data_keys=data_keys,
                                                        returns=['data_keys',
                                                                 'xrd_dets'],
                                                        repair_method='flatten')

    xrd_data = [data_dict[f'{xrd_det}_image'] for xrd_det in xrd_dets]
    
    comps = make_calibration_pattern(xrd_data)

    _save_calibration_pattern(comps,
                             xrd_dets=xrd_dets,
                             scan_id=scan_md['scan_id'],
                             wd=wd)


# Function load and save map parameters 
def save_map_parameters(scan_id=-1,
                        broker='tiled',
                        detectors=[],
                        data_keys=['enc1',
                                   'enc2',
                                   'i0',
                                   'i0_time',
                                   'im',
                                   'it'], 
                        wd=None,
                        filename=None):

    data_dict, scan_md, data_keys = load_data(scan_id=scan_id,
                                              broker=broker,
                                              detectors=detectors,
                                              data_keys=data_keys,
                                              returns=['data_keys'])

    _save_map_parameters(data_dict,
                         scan_md['scan_id'],
                         data_keys=[],
                         wd=wd,
                         filename=filename)


# Function to load and save scan metatdata
def save_scan_md(scan_id=-1,
                 broker='tiled',
                 detectors=[],
                 data_keys=[], 
                 wd=None,
                 filename=None):

    data_dict, scan_md, data_keys = load_data(scan_id=scan_id,
                                              broker=broker,
                                              detectors=detectors,
                                              data_keys=data_keys,
                                              returns=['data_keys'])

    _save_scan_md(scan_md,
                  scan_md['scan_id'],
                  wd=wd,
                  filename=filename)

# Function to load and save all scan data sans xrf for now
def save_full_scan(scan_id=-1,
                   broker='tiled',
                   detectors=None,
                   data_keys=['enc1',
                              'enc2',
                              'i0',
                              'i0_time',
                              'im',
                              'it'], 
                   wd=None,
                   repair_method='replace'):
    
    data_dict, scan_md, data_keys, xrd_dets = load_data(scan_id=scan_id,
                                                        broker=broker,
                                                        detectors=detectors,
                                                        data_keys=data_keys,
                                                        returns=['data_keys',
                                                                 'xrd_dets'],
                                                        repair_method=repair_method)
    
    xrd_data = [data_dict[f'{xrd_det}_image'] for xrd_det in xrd_dets]
    
    # Save xrd tif data
    _save_xrd_tifs(xrd_data,
                   xrd_dets=xrd_dets,
                   scan_id=scan_md['scan_id'], # Will return the correct value
                   wd=wd)
    
    # Save map parameters (x, y, i0, etc...)
    _save_map_parameters(data_dict,
                         scan_md['scan_id'],
                         data_keys=[],
                         wd=wd)
    
    # Save metadata (scaid, uid, energy, etc...)
    _save_scan_md(scan_md,
                  scan_md['scan_id'],
                  wd=wd)

######################
### Rocking Curves ###
######################


def load_step_rc_data(scan_id=-1,
                      extra_data_keys=None,
                      returns=None,
                      verbose=True):

    # Setup verbosity
    vprint = print if verbose else lambda *a, **k: None
    
    _load_tiled_catalog()
    bs_run = c[int(scan_id)]

    scan_md = {
        'scan_id' : bs_run.start['scan_id'],
        'scan_uid' : bs_run.start['uid'],
        'beamline_id' : bs_run.start['beamline_id'],
        'scantype' : bs_run.start['scan']['type'],
        'detectors' : [det for det in bs_run.start['scan']['detectors']],
        'dwell' : bs_run.start['scan']['dwell'],
        'start_time' : bs_run.start['time_str']
    }
    scan_md.update(_load_baseline_metadata(bs_run))
    scantype = scan_md['scantype'].lower()

    # Special for backwards compatability
    for key in ['theta', 'energy']:
        if key in bs_run.start['scan'].keys():
            scan_md[key] = bs_run.start['scan'][key]
        else:
            scan_md[key] = None

    # Find area detectors
    xrd_dets = [det for det in scan_md['detectors']
                if det in supported_xrd_dets]

    data_keys = [
        'sclr_i0',
        'sclr_im',
        'sclr_it',
        'energy_energy',
        'nano_stage_th',
        # 'xs_fluor' # not default, but supported
    ]
    # Add extras
    if extra_data_keys is not None:
        for key in extra_data_keys:
            data_keys.append(key)
    # Grab event-based data_keys
    event_data_keys = [key for key in data_keys if key not in ['xs_fluor']]
    # Add XRD resources
    xrd_data_keys = [f'{det}_image' for det in xrd_dets]
    for key in xrd_data_keys:
        data_keys.append(key)
    
    vprint(f'Loading data for {scantype} scan {scan_md["scan_id"]}...')
    
    # Load data
    _empty_lists = [[] for _ in range(len(data_keys))]
    data_dict = dict(zip(data_keys, _empty_lists))
    docs = bs_run.documents()
    
    vprint('Loading scalers and rocking data...')
    for doc in docs:
        if doc[0] == 'event_page':
            if 'dexela_image' in doc[1]['filled'].keys():
                for key in event_data_keys:
                    if key in doc[1]['data'].keys():
                        data_dict[key].append(doc[1]['data'][key][0])

    # Check for empty (static) values
    for motor_key, data_key in zip(['energy_energy', 'nano_stage_th'],
                                   ['energy', 'theta']):
        if len(data_dict[motor_key]) == 0:
            if scan_md[data_key] is not None:
                value = scan_md[data_key] # nominal is good enough
            else:
                value = bs_run['baseline']['data'][motor_key][0]

            data_dict[motor_key] = np.repeat(np.asarray(value),
                                       len(data_dict['sclr_i0']))
                                       # Not my favorite method

    # Fix keys (motor names to data key names)
    data_dict['energy'] = np.array(data_dict['energy_energy'])
    data_dict['theta'] = np.asarray(data_dict['nano_stage_th'])
    data_dict['i0'] = np.array(data_dict['sclr_i0'])
    data_dict['im'] = np.array(data_dict['sclr_im'])
    data_dict['it'] = np.array(data_dict['sclr_it'])
    del (data_dict['energy_energy'],
         data_dict['nano_stage_th'],
         data_dict['sclr_i0'],
         data_dict['sclr_im'],
         data_dict['sclr_it'])

    # Convert from mdeg to deg
    data_dict['theta'] /= 1000

    # Get relevant hdf_information
    r_paths, r_specs, r_shapes = _get_resources(bs_run, ['xs_fluor', *[f'{det}_image' for det in supported_xrd_dets]])

    # Parse r_paths
    supp_dict = {}
    if 'dark' in r_paths:
        # Only dexela uses dark-field
        with h5py.File(r_paths['dark']['dexela_image'][0]) as f:
            supp_dict['dexela_dark'] = np.median(f['entry/data/data'][:],
                                                 axis=0,
                                                 ).astype(np.float32)
    if 'stream0' in r_paths:
        r_paths = r_paths['stream0']
        r_specs = r_specs['stream0']
        r_shapes = r_shapes['stream0']
    elif 'primary' in r_paths:
        r_paths = r_paths['primary']
        r_specs = r_specs['primary']
        r_shapes = r_shapes['primary']
    
    # key = 'dexela_image'
    # if key in data_keys:
    #     vprint('Loading dexela...')
    #     if 'AD_HDF5' in r_paths:
    #         for r_path in r_paths['AD_HDF5']:
    #             with h5py.File(r_path, 'r') as f:
    #                 data_dict[key].append(np.asarray(f['entry/data/data']))
    #     # Incorrect spec, but left in for backwards compatibility
    #     elif 'TPX_HDF5' in r_paths:
    #         for r_path in r_paths['AD_HDF5']:
    #             with h5py.File(r_path, 'r') as f:
    #                 data_dict[key].append(np.asarray(f['entry/data/data']))

    # Dexela
    key = 'dexela_image'
    if key in data_keys:
        vprint('Loading dexela...')
        if r_specs[key] in ['AD_HDF5' or 'TPX_HDF5']:
            for r_path in r_paths[key]:
                with h5py.File(r_path, 'r') as f:
                    data_dict[key].append(np.asarray(f['entry/data/data']))
    
    # Merlin
    key = 'merlin_image'
    if key in data_keys:
        vprint('Loading merlin...')
        if r_specs[key] in ['AD_HDF5' or 'TPX_HDF5']:
            for r_path in r_paths[key]:
                with h5py.File(r_path, 'r') as f:
                    data_dict[key].append(np.asarray(f['entry/data/data']))

    # Eiger
    key = 'eiger_image'
    if key in data_keys:
        vprint('Loading eiger...')
        if r_specs[key] in ['AD_HDF5']:
            for r_path in r_paths[key]:
                with h5py.File(r_path, 'r') as f:
                    row_data = np.asarray(f['entry/data/data'])
                    dropped_inds = np.array(f['entry/instrument/NDAttributes/NDArrayUniqueId'][:])
                    # Built-in eiger fixes
                    filled_pts = r_shapes[key][0] - len(dropped_inds)
                    if filled_pts > 0:
                        print(f'Auto-filled {filled_pts} points for {key}.')
                        full_row = np.zeros(r_shapes[key], dtype=row_data.dtype)
                        full_row[dropped_inds] = row_data
                        row_data = full_row
                        del full_row
                    data_dict[key].append(row_data)
    
    # xpress3
    key = 'xs_fluor'
    if key in data_keys:
        if len(r_paths['XSP3_FLY']) > 0:
            vprint('Loading xspress3...')
            for r_path in r_paths['XSP3_FLY']:
                with h5py.File(r_path, 'r') as f:
                    data_dict[key].append(np.asarray(f['entry/data/data']))
        else:
            vprint('WARNING: xspress3 data requested, but none found.')
            del data_dict['xs_fluor']


    # Update data_keys
    data_keys = list(data_dict.keys())
    data_dict.update(supp_dict)

    out = [data_dict, scan_md]

    if returns is not None:
        if 'data_keys' in returns:
            out.append(data_keys)
        if 'xrd_dets' in returns:
            out.append(xrd_dets)

    vprint(f'Data loaded for {scantype} scan {scan_md["scan_id"]}!')
    return out


def save_step_rc_data(scan_id=-1,
                      extra_data_keys=None,
                      wd=None,
                      filenames=None):

    if scantype is None:
        scantype = get_scantype(scan_id).lower()

    (data_dict,
     scan_md,
     data_keys,
     xrd_dets
     ) = load_step_rc_data(
                scan_id=scan_id,
                extra_data_keys=extra_data_keys,
                returns=['data_keys',
                         'xrd_dets'])
    
    scantype = scan_md['scantype'].lower()
    
    # Extract data
    xrd_data = [data_dict[f'{xrd_det}_image']
                for xrd_det in xrd_dets]
    # Update keys
    for det in xrd_dets:
        data_keys.remove(f'{det}_image')
        del data_dict[f'{det}_image']
    if 'xs_fluor' in data_keys:
        data_keys.remove('xs_fluor')

    if filenames is None:
        filenames = []
        for detector in xrd_dets:
            filenames.append(f'scan{scan_id}_{detector}_{scantype}.tif')
    
    _save_xrd_tifs(xrd_data,
                   xrd_dets=xrd_dets,
                   scan_id=scan_md['scan_id'], # Will return the correct value
                   wd=wd,
                   filenames=filenames)

    param_filename = f'scan{scan_id}_{scantype}_parameters.txt'
    _save_map_parameters(data_dict, scan_id, data_keys=data_keys,
                         wd=wd, filename=param_filename)

    md_filename = f'scan{scan_id}_{scantype}_metadata.txt'
    _save_scan_md(scan_md, scan_id,
                  wd=wd, filename=md_filename)
    
    # Probably nicer ways to do this
    # TODO: convert to pyXRF format
    if 'xs_fluor' in data_dict.keys():
        xrf_data = np.asarray(data_dict['xs_fluor'])
        xrf_data = xrf_data.squeeze().sum(axis=1)
        np.savetxt(f'{wd}scan{scan_id}_{scantype}_xrf.txt',
                   xrf_data)
        print(f'Saved XRF data for scan {scan_id}!')


### Energy Rocking Curve Scans ###    

def load_extended_energy_rc_data(start_id,
                                 end_id,
                                 extra_data_keys=None,
                                 returns=None):

    _load_tiled_catalog()
    all_scan_ids = list(range(int(start_id),
                              int(end_id) + 1))
    energy_rc_ids = [scan_id for scan_id in all_scan_ids
                     if c[scan_id].start['scan']['type'] == 'ENERGY_RC']

    data_dicts, scan_mds = [], []

    for scan_id in energy_rc_ids:
        print(f'Loading data for scan {scan_id}...')
        (data_dict,
         scan_md,
         data_keys,
         xrd_dets
         ) = load_step_rc_data(scan_id=scan_id,
                               extra_data_keys=extra_data_keys,
                               returns=['data_keys',
                                        'xrd_dets']
                                 )
        data_dicts.append(data_dict)
        scan_mds.append(scan_md)
    
    # Create empty dicts
    all_data_keys = [key for key in data_dicts[0].keys() if 'dark' not in key]
    _empty_lists = [[] for _ in range(len(all_data_keys))]
    all_data_dict = dict(zip(all_data_keys, _empty_lists))

    all_md_keys = list(scan_mds[0].keys())
    _empty_lists = [[] for _ in range(len(all_md_keys))]
    all_md_dict = dict(zip(all_md_keys, _empty_lists))

    # This seems inefficient to reiterate through the data
    for data_dict, scan_md in zip(data_dicts, scan_mds):
        for key in all_data_keys:
            all_data_dict[key].extend(list(data_dict[key]))
        
        for key in all_md_keys:
            # Must be a better way to do this
            if (all_md_dict[key] != [scan_md[key]]
               and all_md_dict[key] != scan_md[key]):

                if isinstance(scan_md[key], list):
                    all_md_dict[key].extend(scan_md[key])
                else:
                    all_md_dict[key].append(scan_md[key])
    
    # Convert md to arrays for consistency with hdf format
    for key in all_md_dict.keys():
        all_md_dict[key] = np.asarray(all_md_dict[key])
    
    for key in data_dicts[0].keys():
        if 'dark' in key:
            all_data_dict[key] = data_dicts[0][key]
    
    out = [all_data_dict, all_md_dict]

    if returns is not None:
        if 'data_keys' in returns:
            out.append(all_data_keys)
        if 'xrd_dets' in returns:
            out.append(xrd_dets)

    return out


def save_extended_energy_rc_data(start_id,
                                 end_id,
                                 extra_data_keys=None,
                                 wd=None,
                                 filenames=None):
    
    (all_data_dict,
     all_md_dict,
     all_data_keys,
     xrd_dets
     ) = load_extended_energy_rc_data(start_id,
                                      end_id,
                                      extra_data_keys=extra_data_keys,
                                      returns=['data_keys',
                                               'xrd_dets'])

    # Stack image data
    xrd_data = [np.vstack(all_data_dict[f'{xrd_det}_image'])
                for xrd_det in xrd_dets]
    # Reshape image data into 4D
    xrd_data = [data.reshape((data.shape[0], 1, *data.shape[-2:]))
                for data in xrd_data]
    # Remove xrd_data from all_data_dict
    for xrd_det in xrd_dets:
        del all_data_dict[f'{xrd_det}_image']
        all_data_keys.remove(f'{xrd_det}_image')
    # Remove any xrf data
    if 'xs_fluor' in all_data_keys:
        xrf_data = all_data_dict['xs_fluor']
        del all_data_dict['xs_fluor']
        all_data_keys.remove('xs_fluor')
    else:
        xrf_data = None
    # Reformat other data streams into arrays
    for key in all_data_dict.keys():
        all_data_dict[key] = np.asarray(all_data_dict[key])

    scan_range_str = f"{all_md_dict['scan_id'][0]}-{all_md_dict['scan_id'][-1]}"

    if filenames is None:
        filenames = []
        for detector in xrd_dets:
            filenames.append(f'scan{scan_range_str}_{detector}_energy_rc.tif')
    
    _save_xrd_tifs(xrd_data,
                   xrd_dets=xrd_dets,
                   scan_id=scan_range_str,
                   wd=wd,
                   filenames=filenames)

    param_filename = f'scan{scan_range_str}_energy_rc_parameters.txt'
    _save_map_parameters(all_data_dict,
                         scan_range_str,
                         data_keys=all_data_keys, # This might fail
                         wd=wd,
                         filename=param_filename)

    md_filename = f'scan{scan_range_str}_energy_rc_metadata.txt'                  
    _save_scan_md(all_md_dict, scan_range_str,
                  wd=wd, filename=md_filename)

    if xrf_data is not None:
        xrf_data = np.asarray(data_dict['xs_fluor'])
        xrf_data = xrf_data.squeeze().sum(axis=1)
        np.savetxt(f'{wd}scan{scan_id}_extended_energy_rc_xrf.txt',
                   xrf_data)
        print(f'Saved XRF data for scans {scan_range_str}!')

### Angle Rocking Curve Scans ###  

def load_flying_angle_rc_data(scan_id=-1,
                              broker='manual',
                              detectors=None,
                              extra_data_keys=None,
                              returns=None,
                              repair_method='fill'):

    data_keys = [
        'i0',
        'im',
        'it',
    ]

    if extra_data_keys is not None:
        for key in extra_data_keys:
            data_keys.append(key)

    (data_dict,
     scan_md,
     data_keys,
     xrd_dets
     ) = load_data(scan_id=scan_id,
                   broker=broker,
                   detectors=detectors,
                   data_keys=data_keys,
                   returns=['data_keys',
                            'xrd_dets'],
                   repair_method=repair_method)

    # Interpolate angular positions
    _load_tiled_catalog()
    thetas = np.linspace(*c[int(scan_id)].start['scan']['scan_input'][:3])
    #thetas = thetas.reshape(map_shape)
    thetas /= 1000 # mdeg to deg
    data_dict['theta'] = thetas
    data_keys.append('theta')

    # Extend energy
    data_dict['energy'] = np.repeat(
                            np.asarray(scan_md['energy']),
                            len(thetas))
    data_keys.append('energy')

    out = [data_dict, scan_md]

    if returns is not None:
        if 'data_keys' in returns:
            out.append(data_keys)
        if 'xrd_dets' in returns:
            out.append(xrd_dets)

    return out


def save_flying_angle_rc_data(scan_id=-1,
                              broker='manual',
                              detectors=None,
                              extra_data_keys=None,
                              wd=None,
                              filenames=None,
                              repair_method='fill'):

    # Retrieve and format data
    (data_dict,
     scan_md,
     data_keys,
     xrd_dets
     ) = load_flying_angle_rc_data(
                    scan_id=scan_id,
                    broker=broker,
                    detectors=detectors,
                    extra_data_keys=extra_data_keys,
                    returns=['data_keys',
                             'xrd_dets'],
                    repair_method=repair_method)

    if filenames is None:
        filenames = []
        for detector in xrd_dets:
            filenames.append(f'scan{scan_id}_{detector}_flying_angle_rc.tif')

    # Convert from dictionary to list
    xrd_data = [data_dict[f'{xrd_det}_image'] for xrd_det in xrd_dets]
    # And remove data_keys
    for xrd_det in xrd_dets:
        if f'{xrd_det}_image' in data_keys:
            data_keys.remove(f'{xrd_det}_image')

    # XRF data key will cause issue with _save_map_parameters
    if 'xs_fluor' in data_keys:
        data_keys.remove('xs_fluor')
       
    _save_xrd_tifs(xrd_data,
                   xrd_dets=xrd_dets,
                   scan_id=scan_md['scan_id'], # Will return the correct value
                   wd=wd,
                   filenames=filenames)

    # return data_dict, data_keys
    
    param_filename = f'scan{scan_id}_flying_angle_rc_parameters.txt'
    _save_map_parameters(data_dict, scan_id, data_keys=data_keys,
                         wd=wd, filename=param_filename)
    
    md_filename = f'scan{scan_id}_flying_angle_rc_metadata.txt'                  
    _save_scan_md(scan_md, scan_id,
                  wd=wd, filename=md_filename)
    
    # This one can be mapped with pyXRF...
    if 'xs_fluor' in data_dict.keys():
        xrf_data = np.asarray(data_dict['xs_fluor'])
        xrf_data = xrf_data.squeeze().sum(axis=1)
        np.savetxt(f'{wd}scan{scan_id}_flying_angle_rc_xrf.txt',
                   xrf_data)
        print(f'Saved XRF data for scan {scan_id}!')


#############################
### Convenience Functions ###
#############################

# Convenience/utility function for generating record of scan_ids
def generate_scan_logfile(start_id,
                          end_id=-1,
                          filename=None,
                          wd=None,
                          include_peakups=False):

    if wd is None:
        raise ValueError('No defualt wd and none specified. Please define wd.')

    _load_tiled_catalog()

    if end_id == -1:
        bs_run = c[-1]
        end_id = int(bs_run.start['scan_id'])

    if filename is None:
        filename = f'logfile_{start_id}-{end_id}'

    id_list = np.arange(int(start_id), int(end_id) + 1, 1)

    scan_ids = []
    scan_types = []
    scan_statuses = []
    scan_detectors = []
    scan_inputs = []

    for scan_id in id_list:
        try:
            bs_run = c[int(scan_id)]
            start = bs_run.start
            stop = bs_run.stop

            if 'scan' in start.keys():
                if not include_peakups and str(start['scan']['type']) in ['PEAKUP']:
                    continue

                scan_ids.append(str(start['scan_id']))    
                scan_types.append(str(start['scan']['type']))
                
                if 'scan_input' in start['scan'].keys():
                    scan_input = start['scan']['scan_input']
                    if not isinstance(scan_input, str):
                        scan_input = [np.round(x, 3) for x in scan_input ] # Cleans up floating point errors?
                    # print(scan_id)
                    # print(scan_input)
                    scan_inputs.append(scan_input)
                else:
                    scan_inputs.append(str([]))
                
                if 'detectors' in start['scan'].keys():
                    detectors = start['scan']['detectors']
                    # Remove common detectors that are almost always included
                    detectors = [det for det in detectors
                                 if det not in ['nanoZebra', 'sclr1']]
                    scan_detectors.append(f"[{', '.join(detectors)}]")
                else:
                    scan_detectors.append(str([]))
            else:
                scan_ids.append(str(start['scan_id']))
                scan_types.append('UNKOWN')
                scan_detectors.append(str([]))
                scan_inputs.append(str([]))
            
            if stop is None:
                scan_statuses.append('NONE')
            else:
                scan_statuses.append(stop['exit_status'])

        except KeyError:
            print(f'scan_id={scan_id} not found!')

    logfile = f'{wd}{filename}.txt'

    with open(logfile, 'w') as log:
        for (scan_id,
             scan_type,
             scan_status,
             scan_detector,
             scan_input) in zip(scan_ids,
                                scan_types, 
                                scan_statuses,
                                scan_detectors,
                                scan_inputs):
            log.write(f'{scan_id}\t{scan_type}\t{scan_status}\t{scan_detector}\t{scan_input}\n')

