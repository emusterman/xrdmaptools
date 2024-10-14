import numpy as np
from skimage import io
import h5py
import os
from scipy.stats import mode


# Working at the beamline...
try:
    print('Connecting to databrokers...', end='', flush=True)
    from tiled.client import from_profile
    from databroker.v1 import Broker

    c = from_profile('srx')
    db = Broker.named('srx')
    print('done!')
except ModuleNotFoundError:
    print('failed.')
    pass


# Wrapper for all load data options...
# Might only work for fly scans
def load_data(scanid=-1,
              broker='manual',
              detectors=None,
              data_keys=None,
              returns=None,
              repair_method='replace'):

    # Load data from tiled
    if str(broker).lower() in ['tiled']:
        bs_run = c[int(scanid)]

        out = load_tiled_data(scanid=scanid,
                              detectors=detectors,
                              data_keys=data_keys,
                              returns=returns)

    # Load data from databroker
    elif str(broker).lower() in ['db', 'databroker', 'broker']:
        bs_run = db[int(scanid)]

        out = load_db_data(scanid=scanid, 
                           detectors=detectors,
                           data_keys=data_keys,
                           returns=returns)
    
    # Load data manually
    elif str(broker).lower() in ['manual']:
        try:
            bs_run = c[int(scanid)] # defualt basic data from tiled for future proofing
            broker = 'tiled'
        except: # what error should this throw???
            bs_run = db[int(scanid)]
            broker = 'db'

        out = manual_load_data(scanid=scanid,
                               broker=broker,
                               detectors=detectors,
                               data_keys=data_keys,
                               returns=returns,
                               repair_method=repair_method)

    return out


# Base Tiled loading function
def load_tiled_data(scanid=-1,
                    detectors=None,
                    data_keys=None,
                    returns=None):
    
    bs_run = c[int(scanid)]

    if detectors is None:
        detectors = bs_run.start['scan']['detectors']
    else:
        detectors = [detector.name if type(detector) is not str else str(detector).lower()
                     for detector in detectors]
    xrd_dets = [detector for detector in detectors if detector in ['merlin', 'dexela']]

    scan_md = _load_scan_metadata(bs_run)

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

    out = [data_dict, scan_md]

    if returns is not None:
        if 'data_keys' in returns:
            out.append(data_keys)
        if 'xrd_dets' in returns:
            out.append(xrd_dets)

    return tuple(out)


# Base function for loading data from DataBroker
def load_db_data(scanid=-1,
                 detectors=None,
                 data_keys=None,
                 returns=None):

    bs_run = db[int(scanid)]

    if detectors is None:
        detectors = bs_run.start['scan']['detectors']
    else:
        detectors = [detector.name if type(detector) is not str else str(detector)
                     for detector in detectors]
    xrd_dets = [detector for detector in detectors if detector in ['merlin', 'dexela']]

    scan_md = _load_scan_metadata(bs_run)

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

    out = [data_dict, scan_md]
    
    if returns is not None:
        if 'data_keys' in returns:
            out.append(data_keys)
        if 'xrd_dets' in returns:
            out.append(xrd_dets)

    return out


# Manual load data
def manual_load_data(scanid=-1,
                     broker='tiled',
                     data_keys=None,
                     detectors=None,
                     returns=None,
                     repair_method='replace'):

    if str(broker).lower() in ['tiled']:
        bs_run = c[int(scanid)]
    elif str(broker).lower() in ['db', 'databroker', 'broker']:
        bs_run = db[int(scanid)]

    print(f'Manually loading data for scan {scanid}...')
    
    scan_md = _load_scan_metadata(bs_run)

    # Get relevant hdf information
    data_keys, resource_keys, xrd_dets = _get_resource_keys(bs_run,
                                                  data_keys=data_keys,
                                                  detectors=detectors,
                                                  returns=['xrd_dets'])                             
    r_paths = _get_resource_paths(bs_run, resource_keys)

    # Load data
    _empty_lists = [[] for _ in range(len(data_keys))]
    data_dict = dict(zip(data_keys, _empty_lists))

    dropped_rows, broken_rows = [], []
    r_key = 'ZEBRA_HDF51'
    if r_key in r_paths.keys():
        print('Loading encoders...')
        enc_keys = [value for value in data_keys if value in ['enc1', 'enc2', 'enc3', 'zebra_time']]
        for r_path in r_paths[r_key]:
            with h5py.File(r_path, 'r') as f:
                for key in enc_keys:
                    data_dict[key].append(np.array(f[key]))
        # Stack data into array
        for key in enc_keys:
            #data_dict[key] = np.stack(data_dict[key])
            dr_rows, br_rows = _flag_broken_rows(data_dict[key], key)
            dropped_rows += dr_rows
            broken_rows += br_rows

    r_key = 'SIS_HDF51'
    if r_key in r_paths.keys():
        print('Loading scalers...')
        sclr_keys = [value for value in data_keys if value in ['i0', 'im', 'it', 'sis_time']]
        if 'i0_time' in data_keys and 'sis_time' not in sclr_keys:
            sclr_keys.append('sis_time')
            data_dict['sis_time'] = []
        for r_path in r_paths[r_key]:
            with h5py.File(r_path, 'r') as f:
                for key in sclr_keys:
                    data_dict[key].append(np.array(f[key]))
        # Stack data into array
        #for key in sclr_keys:
            #data_dict[key] = np.stack(data_dict[key])
        if 'i0_time' in data_keys:
            #data_dict['i0_time'] = data_dict['sis_time'] / 50e6 # 50 MHz clock
            data_dict['i0_time'] = [d / 50e6 for d in data_dict['sis_time']] # 50 MHz clock
        if 'sis_time' not in data_keys and 'sis_time' in data_dict.keys():
            del data_dict['sis_time']
        for key in ['i0', 'im', 'it', 'i0_time']:
            dr_rows, br_rows = _flag_broken_rows(data_dict[key], key)
            dropped_rows += dr_rows
            broken_rows += br_rows

    r_key = 'XSP3_FLY'
    if r_key in r_paths.keys():
        print('Loading xspress3...')
        key = 'xs_fluor'
        for r_path in r_paths[r_key]:
            with h5py.File(r_path, 'r') as f:
                data_dict[key].append(np.array(f['entry/data/data']))
        # Stack data into array
        #data_dict[key] = np.stack(data_dict[key])
        dr_rows, br_rows = _flag_broken_rows(data_dict[key], key)
        dropped_rows += dr_rows
        broken_rows += br_rows

    r_key = 'DEXELA_FLY_V1'
    if r_key in r_paths.keys():
        print('Loading dexela...')
        key = 'dexela_image'
        for r_path in r_paths[r_key]:
            f = h5py.File(r_path, 'r')
            data_dict[key].append(f['entry/data/data'])
            #with h5py.File(r_path, 'r') as f:
            #    data_dict[key].append(np.array(f['entry/data/data']))
        # Stack data into array
        #data_dict[key] = _check_xrd_data_shape(data_dict[key],
        #                                       repair_method=repair_method)
        dr_rows, br_rows = _flag_broken_rows(data_dict[key], key)
        dropped_rows += dr_rows
        broken_rows += br_rows

    # Not sure if this is correct
    r_key = 'MERLIN_FLY_V1'
    if r_key in r_paths.keys():
        print('Loading merlin...')
        key = 'merlin_image'
        for r_path in r_paths[r_key]:
            with h5py.File(r_path, 'r') as f:
                data_dict[key].append(np.array(f['entry/data/data']))
        # Stack data into array
        #data_dict[key] = _check_xrd_data_shape(data_dict[key],
        #                                       repair_method=repair_method)
        dr_rows, br_rows = _flag_broken_rows(data_dict[key], key)
        dropped_rows += dr_rows
        broken_rows += br_rows

    # Repair data
    dropped_rows = sorted(list(np.unique(dropped_rows)))
    broken_rows = sorted(list(np.unique(broken_rows)))
    #print(dropped_rows)
    #print(broken_rows)
    data_dict = _repair_data_dict(data_dict,
                                  dropped_rows,
                                  broken_rows,
                                  repair_method=repair_method)

    out = [data_dict, scan_md]

    if returns is not None:
        if 'data_keys' in returns:
            out.append(data_keys)
        if 'xrd_dets' in returns:
            out.append(xrd_dets)

    print(f'Data loaded for scan {scanid}!')
    return out


# Helper function
def get_scantype(scanid=-1,
                 broker='tiled'):
        # Load data from tiled
    if str(broker).lower() in ['tiled']:
        bs_run = c[int(scanid)]

    # Load data from databroker
    elif str(broker).lower() in ['db', 'databroker', 'broker']:
        bs_run = db[int(scanid)]

    else:
        raise ValueError(f"Unknown broker type: {broker}. Only 'tiled' and 'databroker' suppported.")
    
    scantype = bs_run.start['scan']['type']

    return scantype


# TODO: implement way of getting units
# for positions, theta, and energy
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

    return scan_md


def _flag_broken_rows(data_list, msg):
    # Majority of row shapes
    mode_shape = tuple(mode([d.shape for d in data_list])[0])

    dropped_rows, broken_rows = [], []
    for row, d in enumerate(data_list):
        # Check for image shape issues
        if d.ndim == 4 and d.shape[-2:] != mode_shape[-2:]:
            print(f'WARNING: {msg} data from row {row} has an incorrect image shape.')
            dropped_rows.append(row)
            broken_rows.append(row)
        elif d.shape[0] != mode_shape[0]:
            print(f'WARNING: {msg} data from row {row} captured only {d.shape[0]}/{mode_shape[0]} points.')
            broken_rows.append(row)

    return dropped_rows, broken_rows


def _repair_data_dict(data_dict,
                      dropped_rows,
                      broken_rows,
                      repair_method='replace'):
    if repair_method.lower() not in ['flatten', 'fill', 'replace']:
            raise ValueError('Only "flatten", "fill", and "replace" repair methods are supported.')
        
    keys = list(data_dict.keys())

    # Check data shape
    num_rows = -1
    for key in keys:
        if num_rows == -1:
            num_rows = len(data_dict[key])
        elif len(data_dict[key]) != num_rows:
            raise ValueError('Data keys have different number of rows!')

    # Add key to track filled images
    if repair_method == 'fill':
        # Generate map with first two indices of shape of first key. Should be map_shape
        data_dict['null_map'] = [[] for _ in range(num_rows)]

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
            data_dict['null_map'][row] = np.zeros(len(list(data_dict.values())[0][last_good_row]), dtype=np.bool_)
            if row in broken_rows:
                if last_good_row == -1:
                    queued_rows.append(row)
                else:
                    for key in keys:
                        filled_pts = (len(data_dict[key][last_good_row])
                                     - len(data_dict[key][row]))
                        if filled_pts > 0:
                            print(f'Filled {filled_pts} points in row {row} for {key}.')

                            zero_row = np.zeros_like(np.asarray(data_dict[key][last_good_row]))
                            #print(f'{zero_row.shape=}')

                            data_dict['null_map'][row][-filled_pts:] = (
                                                            [True,] * filled_pts)

                            zero_row[:len(data_dict[key][row])] = data_dict[key][row]
                            data_dict[key][row] = zero_row
                    
            else:
                last_good_row = row
                if len(queued_rows) > 0:
                    for q_row in queued_rows:
                        for key in keys:
                            filled_pts = (len(data_dict[key][last_good_row])
                                        - len(data_dict[key][q_row]))
                            if filled_pts > 0:
                                print(f'Filled {filled_pts} points in row {q_row} for {key}.')

                                zero_row = np.zeros_like(np.asarray(
                                                            data_dict[key][last_good_row]
                                                            ))
                                #print(f'{zero_row.shape=}')

                                data_dict['null_map'][q_row][-filled_pts:] = (
                                                            [True,] * filled_pts)
                                
                                zero_row[:len(data_dict[key][q_row])] = np.asarray(
                                                                            data_dict[key][q_row]
                                                                            )
                                data_dict[key][q_row] = zero_row
                        queued_rows = []           
        
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


def _get_resource_paths(bs_run, resource_keys):

    docs = list(bs_run.documents())

    # Create dictionary of empty lists
    _empty_lists = [[] for _ in range(len(resource_keys))]
    resource_paths = dict(zip(resource_keys, _empty_lists))
    
    for doc in docs:
        if doc[0] == 'resource':
            r_key = doc[1]['spec']
            if r_key in resource_keys:
                resource_paths[r_key].append(doc[1]['resource_path'])

    return resource_paths


def _get_resource_keys(bs_run,
                    data_keys=None,
                    detectors=None,
                    returns=None):
    
    # Find detectors if not specified
    if detectors is None:
        detectors = bs_run.start['scan']['detectors']
    else:
        detectors = [detector.name if type(detector) is not str else str(detector)
                     for detector in detectors]
    xrd_dets = [detector for detector in detectors if detector in ['merlin', 'dexela']]

    # Add default data keys
    if data_keys is None:
        data_keys = ['enc1', 'enc2', 'xs_fluor', 'i0', 'i0_time', 'im', 'it']

    # Append XRD data keys
    for detector in xrd_dets:
        data_keys.append(f'{detector}_image')

    # Determine actual resource keys
    resource_keys = []
    if any(key in ['enc1', 'enc2', 'enc3'] for key in data_keys):
        resource_keys.append('ZEBRA_HDF51')

    if any(key in ['i0', 'im', 'it'] for key in data_keys):
        resource_keys.append('SIS_HDF51')

    if any(key in ['xs_fluor'] for key in data_keys):
        resource_keys.append('XSP3_FLY')

    if any(key in ['dexela_image'] for key in data_keys):
        resource_keys.append('DEXELA_FLY_V1')

    # Not sure if this is correct???
    if any(key in ['merlin_image'] for key in data_keys):
        resource_keys.append('MERLIN_FLY_V1')

    # Check for issues:
    if len(resource_keys) < 1:
        raise ValueError('No valid data resources requested.')

    out = [data_keys, resource_keys]

    if returns is not None:
        if 'xrd_dets' in returns or returns == 'xrd_dets':
            out.append(xrd_dets)

    return out


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
                   scanid=None,
                   filedir=None,
                   filenames=None):

    if (scanid is None and filenames is None):
        raise ValueError('Must define scanid or filename to name the save file.')
    elif (xrd_dets is None and filenames is None):
        raise ValueError('Must define xrd_dets or filename to name the save file.')
    
    if not isinstance(xrd_data, list):
        xrd_data = [xrd_data]

    if filedir is None:
        filedir = os.getcwd()   
    if filenames is None:
        filenames = []
        for detector in xrd_dets:
            filenames.append(f'scan{scanid}_{detector}_xrd.tif')

    if len(filenames) != len(xrd_data):
        raise ValueError('Length of filenames does not match length of xrd_data')
    
    for i, xrd in enumerate(xrd_data):
        io.imsave(f'{filedir}{filenames[i]}',
                  np.asarray(xrd).astype(np.uint16),
                  check_contrast=False)
        # I think the data should already be an unsigned integer
        #io.imsave(f'{filedir}{filenames[i]}', xrd, check_contrast=False)
    print(f'Saved pattern(s) for scan {scanid}!')


# General function to save composite pattern as tif
def _save_composite_pattern(comps,
                            method,
                            subtract,
                            xrd_dets=None,
                            scanid=None,
                            filedir=None,
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
            filenames.append(f'scan{scanid}_{detector}_{method}{subtract}_composite.tif')
    
    print('Saving composite pattern...')
    _save_xrd_tifs(comps,
                   xrd_dets=xrd_dets,
                   scanid=scanid,
                   filedir=filedir,
                   filenames=filenames)


def _save_calibration_pattern(xrd_data,
                              xrd_dets,
                              scanid=None,
                              filedir=None):

    if not isinstance(xrd_data, list):
        xrd_data = [xrd_data]
    
    if filedir is None:
        filedir = os.getcwd()

    filenames = []
    for detector in xrd_dets:
        filenames.append(f'scan{scanid}_{detector}_calibration.tif')
    
    print('Saving calibration pattern...')
    _save_xrd_tifs(xrd_data,
                   xrd_dets=xrd_dets,
                   scanid=scanid,
                   filedir=filedir,
                   filenames=filenames)


# General function save scan metadata 
def _save_map_parameters(data_dict,
                         scanid,
                         data_keys=[],
                         filedir=None,
                         filename=None):

    if filedir is None:
        filedir = os.getcwd()
    if filename is None:
        filename = f'scan{scanid}_map_parameters.txt'
    if data_keys == []:
        data_keys = ['enc1', 'enc2', 'i0', 'i0_time', 'im', 'it']

    map_data = np.stack([data_dict[key].ravel() for key in data_keys])
    np.savetxt(f'{filedir}{filename}', map_data)
    print(f'Saved map parameters for scan {scanid}!')


def _save_scan_md(scan_md,
                  scanid,
                  filedir=None,
                  filename=None):
    import json

    if not isinstance(scan_md, dict):
        raise TypeError('Scan metadata not provided as a dictionary.')
    
    if filedir is None:
        filedir = os.getcwd()
    if filename is None:
        filename = f'scan{scanid}_scan_md.txt'

    with open(f'{filedir}{filename}', 'w') as f:
        f.write(json.dumps(scan_md))
    print(f'Saved metadata for scan {scanid}!')


##############################
### General Save Functions ###
##############################


# Function to load and save xrd data to tifs
def save_xrd_tifs(scanid=-1,
                  broker='tiled',
                  detectors=None,
                  data_keys=[],
                  filedir=None,
                  filenames=None,
                  repair_method='replace'):

    data_dict, scan_md, data_keys, xrd_dets = load_data(scanid=scanid,
                                                        broker=broker,
                                                        detectors=detectors,
                                                        data_keys=data_keys,
                                                        returns=['data_keys',
                                                                 'xrd_dets'],
                                                        repair_method=repair_method)

    xrd_data = [data_dict[f'{xrd_det}_image'] for xrd_det in xrd_dets]
    
    _save_xrd_tifs(xrd_data,
                   xrd_dets=xrd_dets,
                   scanid=scan_md['scan_id'], # Will return the correct value
                   filedir=filedir,
                   filenames=filenames)


# Function to load and save composite pattern as tif
def save_composite_pattern(scanid=-1,
                           broker='tiled',
                           method='sum',
                           subtract=None,
                           detectors=None,
                           data_keys=[], 
                           filedir=None,
                           filenames=None):

    data_dict, scan_md, data_keys, xrd_dets = load_data(scanid=scanid,
                                                        broker=broker,
                                                        detectors=detectors,
                                                        data_keys=data_keys,
                                                        returns=['data_keys',
                                                                 'xrd_dets'],
                                                        repair_method='flatten')

    xrd_data = [data_dict[f'{xrd_det}_image'] for xrd_det in xrd_dets]
    
    comps = make_composite_pattern(xrd_data, method=method, subtract=subtract)

    _save_composite_pattern(comps,
                            method,
                            subtract,
                            xrd_dets=xrd_dets,
                            scanid=scan_md['scan_id'],
                            filedir=filedir,
                            filenames=filenames)


# Function to load and save calibration pattern as tif
def save_calibration_pattern(scanid=-1,
                             broker='tiled',
                             detectors=None,
                             data_keys=[], 
                             filedir=None):

    data_dict, scan_md, data_keys, xrd_dets = load_data(scanid=scanid,
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
                             scanid=scan_md['scan_id'],
                             filedir=filedir)


# Function load and save map parameters 
def save_map_parameters(scanid=-1,
                        broker='tiled',
                        detectors=[],
                        data_keys=['enc1',
                                   'enc2',
                                   'i0',
                                   'i0_time',
                                   'im',
                                   'it'], 
                        filedir=None,
                        filename=None):

    data_dict, scan_md, data_keys = load_data(scanid=scanid,
                                              broker=broker,
                                              detectors=detectors,
                                              data_keys=data_keys,
                                              returns=['data_keys'])

    _save_map_parameters(data_dict,
                        scan_md['scan_id'],
                        data_keys=[],
                        filedir=filedir,
                        filename=filename)


# Function to load and save scan metatdata
def save_scan_md(scanid=-1,
                 broker='tiled',
                 detectors=[],
                 data_keys=[], 
                 filedir=None,
                 filename=None):

    data_dict, scan_md, data_keys = load_data(scanid=scanid,
                                              broker=broker,
                                              detectors=detectors,
                                              data_keys=data_keys,
                                              returns=['data_keys'])

    _save_scan_md(scan_md,
                  scan_md['scan_id'],
                  filedir=filedir,
                  filename=filename)

# Function to load and save all scan data sans xrf for now
def save_full_scan(scanid=-1,
                   broker='tiled',
                   detectors=None,
                   data_keys=['enc1',
                               'enc2',
                               'i0',
                               'i0_time',
                               'im',
                               'it'], 
                   filedir=None,
                   repair_method='replace'):
    
    data_dict, scan_md, data_keys, xrd_dets = load_data(scanid=scanid,
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
                   scanid=scan_md['scan_id'], # Will return the correct value
                   filedir=filedir)
    
    # Save map parameters (x, y, i0, etc...)
    _save_map_parameters(data_dict,
                         scan_md['scan_id'],
                         data_keys=[],
                         filedir=filedir)
    
    # Save metadata (scaid, uid, energy, etc...)
    _save_scan_md(scan_md,
                  scan_md['scan_id'],
                  filedir=filedir)


### Energy Rocking Curve Scans ###

def load_energy_rc_data(scanid=-1,
                        data_keys=None,
                        returns=None):

    bs_run = c[int(scanid)]

    scan_md = {
        'scan_id' : bs_run.start['scan_id'],
        'scan_uid' : bs_run.start['uid'],
        'beamline' : bs_run.start['beamline_id'],
        'scantype' : bs_run.start['scan']['type'],
        'detectors' : bs_run.start['scan']['detectors'],
        'energy' : bs_run.start['scan']['energy'],
        'dwell' : bs_run.start['scan']['dwell'],
        'start_time' : bs_run.start['time_str']
    }

    docs = bs_run.documents()

    data_keys = [
        'sclr_i0',
        'sclr_im',
        'sclr_it',
        'energy_energy'
    ]

    # Load data
    _empty_lists = [[] for _ in range(len(data_keys))]
    data_dict = dict(zip(data_keys, _empty_lists))

    print('Loading scalers and energies...')
    for doc in docs:
        if doc[0] == 'event_page':
            if 'dexela_image' in doc[1]['filled'].keys():
                for key in data_keys:
                    data_dict[key].append(doc[1]['data'][key][0])

    # Fix keys
    data_dict['energy'] = np.array(data_dict['energy_energy'])
    data_dict['i0'] = np.array(data_dict['sclr_i0'])
    data_dict['im'] = np.array(data_dict['sclr_im'])
    data_dict['it'] = np.array(data_dict['sclr_it'])
    del (data_dict['energy_energy'],
         data_dict['sclr_i0'],
         data_dict['sclr_im'],
         data_dict['sclr_it'])
    data_keys = list(data_dict.keys())
    #print(data_keys)

    # Dexela images saved differently with count...
    r_paths = _get_resource_paths(bs_run, ['TPX_HDF5'])

    print('Loading dexela...')
    key = 'dexela_image'
    data_dict[key] = []
    for r_path in r_paths['TPX_HDF5']:
        with h5py.File(r_path, 'r') as f:
            data_dict[key].append(np.array(f['entry/data/data']))
    print('done!')

    out = [data_dict, scan_md]

    if returns is not None:
        if 'data_keys' in returns:
            out.append(data_keys)
        if 'xrd_dets' in returns:
            xrd_dets = [det for det in scan_md['detectors']
                        if det in ['merlin', 'dexela']]
            out.append(xrd_dets)

    return out


def save_energy_rc_data(scanid=-1,
                        filedir=None,
                        filenames=None):

    (data_dict,
     scan_md,
     data_keys,
     xrd_dets
     ) = load_energy_rc_data(scanid=scanid,
                             returns=['data_keys',
                                      'xrd_dets'])
    
    xrd_data = [data_dict[f'{xrd_det}_image']
                for xrd_det in xrd_dets]

    if filenames is None:
        filenames = []
        for detector in xrd_dets:
            filenames.append(f'scan{scanid}_{detector}_energy_rc.tif')
    
    _save_xrd_tifs(xrd_data,
                   xrd_dets=xrd_dets,
                   scanid=scan_md['scan_id'], # Will return the correct value
                   filedir=filedir,
                   filenames=filenames)

    param_filename = f'scan{scanid}_energy_rc_parameters.txt'
    _save_map_parameters(data_dict, scanid, data_keys=data_keys,
                         filedir=filedir, filename=param_filename)
    
    md_filename = f'scan{scanid}_energy_rc_metadata.txt'                  
    _save_scan_md(scan_md, scanid,
                  filedir=filedir, filename=md_filename)
    

def load_extended_energy_rc_data(start_id,
                                 end_id,
                                 returns=None):

    all_scan_ids = list(range(start_id, end_id + 1))
    energy_rc_ids = [scan_id for scan_id in all_scan_ids
                     if c[scan_id].start['scan']['type'] == 'ENERGY_RC']

    data_dicts, scan_mds = [], []

    for scan_id in energy_rc_ids:
        print(f'Loading data for scan {scan_id}...')
        (data_dict,
         scan_md,
         data_keys,
         xrd_dets
         ) = load_energy_rc_data(scanid=scan_id,
                                 returns=['data_keys',
                                          'xrd_dets']
                                 )
        data_dicts.append(data_dict)
        scan_mds.append(scan_md)
    
    # Create empty dicts
    all_data_keys = list(data_dicts[0].keys())
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
    
    out = [all_data_dict, all_md_dict]

    if returns is not None:
        if 'data_keys' in returns:
            out.append(all_data_keys)
        if 'xrd_dets' in returns:
            out.append(xrd_dets)

    return out


def save_extended_energy_rc_data(start_id,
                                 end_id,
                                 filedir=None,
                                 filenames=None):
    
    (all_data_dict,
     all_md_dict,
     all_data_keys,
     xrd_dets
     ) = load_extended_energy_rc_data(start_id,
                                      end_id,
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
                   scanid=scan_range_str,
                   filedir=filedir,
                   filenames=filenames)

    param_filename = f'scan{scan_range_str}_energy_rc_parameters.txt'
    _save_map_parameters(all_data_dict,
                         scan_range_str,
                         data_keys=all_data_keys, # This might fail
                         filedir=filedir,
                         filename=param_filename)

    md_filename = f'scan{scan_range_str}_energy_rc_metadata.txt'                  
    _save_scan_md(all_md_dict, scan_range_str,
                  filedir=filedir, filename=md_filename)


# def save_extended_energy_rc_data(start_id,
#                                  end_id,
#                                  filedir=None,
#                                  filenames=None):

#     all_scan_ids = list(range(start_id, end_id + 1))
#     energy_rc_ids = [scan_id for scan_id in all_scan_ids
#                      if c[scan_id].start['scan']['type'] == 'ENERGY_RC']

#     data_dicts, scan_mds = [], []

#     for scan_id in energy_rc_ids:
#         print(f'Loading data for scan {scan_id}...')
#         data_dict, scan_md, data_keys = load_energy_rc_data(
#                                             scanid=scan_id,
#                                             returns=['data_keys']
#                                             )
#         data_dicts.append(data_dict)
#         scan_mds.append(scan_md)
    
#     # Create empty dicts
#     all_data_keys = list(data_dicts[0].keys())
#     _empty_lists = [[] for _ in range(len(all_data_keys))]
#     all_data_dict = dict(zip(all_data_keys, _empty_lists))

#     all_md_keys = list(scan_mds[0].keys())
#     _empty_lists = [[] for _ in range(len(all_md_keys))]
#     all_md_dict = dict(zip(all_md_keys, _empty_lists))

#     # This seems inefficient to reiterate through the data
#     for data_dict, scan_md in zip(data_dicts, scan_mds):
#         for key in all_data_keys:
#             all_data_dict[key].extend(list(data_dict[key]))
        
#         for key in all_md_keys:
#             # Must be a better way to do this
#             if (all_md_dict[key] != [scan_md[key]]
#                and all_md_dict[key] != scan_md[key]):

#                 if isinstance(scan_md[key], list):
#                     all_md_dict[key].extend(scan_md[key])
#                 else:
#                     all_md_dict[key].append(scan_md[key])

#     # Get area detectors
#     xrd_dets = [detector for detector in scan_md['detectors']
#                 if detector in ['merlin', 'dexela']]
#     # Stack image data
#     xrd_data = [np.vstack(all_data_dict[f'{xrd_det}_image'])
#                 for xrd_det in xrd_dets]
#     # Reshape image data into 4D
#     xrd_data = [data.reshape((data.shape[0], 1, data.shape[-2:]))
#                 for data in xrd_data]
#     # Remove xrd_data from all_data_dict
#     for xrd_det in xrd_dets:
#         del all_data_dict[f'{xrd_det}_image']
#     # Reformat other data streams into arrays
#     for key in all_data_dict.keys():
#         all_data_dict[key] = np.asarray(all_data_dict[key])
    
#     #return all_data_dict, all_md_dict, xrd_data

#     scan_range_str = f"{all_md_dict['scan_id'][0]}-{all_md_dict['scan_id'][-1]}"

#     if filenames is None:
#         filenames = []
#         for detector in xrd_dets:
#             filenames.append(f'scan{scan_range_str}_{detector}_energy_rc.tif')
    
#     _save_xrd_tifs(xrd_data,
#                    xrd_dets=xrd_dets,
#                    scanid=scan_range_str,
#                    filedir=filedir,
#                    filenames=filenames)

#     param_filename = f'scan{scan_range_str}_energy_rc_parameters.txt'
#     _save_map_parameters(all_data_dict, scan_range_str, data_keys=data_keys,
#                          filedir=filedir, filename=param_filename)

#     md_filename = f'scan{scan_range_str}_energy_rc_metadata.txt'                  
#     _save_scan_md(all_md_dict, scan_range_str,
#                   filedir=filedir, filename=md_filename)


### Angle Rocking Curve Scans ###

def load_angle_rc_data(scanid=-1,
                       detectors=None,
                       data_keys=None,
                       returns=None):

    bs_run = c[int(scanid)]

    scan_md = {
        'scan_id' : bs_run.start['scan_id'],
        'scan_uid' : bs_run.start['uid'],
        'beamline' : bs_run.start['beamline_id'],
        'scantype' : bs_run.start['scan']['type'],
        'detectors' : bs_run.start['scan']['detectors'],
        'energy' : bs_run.start['scan']['energy'],
        'dwell' : bs_run.start['scan']['dwell'],
        'start_time' : bs_run.start['time_str']
    }

    docs = bs_run.documents()

    data_keys = [
        'sclr_i0',
        'sclr_im',
        'sclr_it',
        'nano_stage_th'
    ]

    # Load data
    _empty_lists = [[] for _ in range(len(data_keys))]
    data_dict = dict(zip(data_keys, _empty_lists))

    print('Loading scalers and energies...', end='', flush=True)
    for doc in docs:
        if doc[0] == 'event_page':
            if 'dexela_image' in doc[1]['filled'].keys():
                for key in data_keys:
                    data_dict[key].append(doc[1]['data'][key][0])
    print('done!')

    # Fix keys
    data_dict['th'] = np.array(data_dict['nano_stage_th'])
    data_dict['i0'] = np.array(data_dict['sclr_i0'])
    data_dict['im'] = np.array(data_dict['sclr_im'])
    data_dict['it'] = np.array(data_dict['sclr_it'])
    del data_dict['nano_stage_th'], data_dict['sclr_i0'], data_dict['sclr_im'], data_dict['sclr_it']
    data_keys = list(data_dict.keys())
    #print(data_keys)

    # Dexela images saved differently with count...
    r_paths = _get_resource_paths(bs_run, ['TPX_HDF5'])

    print('Loading dexela...', end='', flush='True')
    key = 'dexela_image'
    data_dict[key] = []
    for r_path in r_paths['TPX_HDF5']:
        with h5py.File(r_path, 'r') as f:
            data_dict[key].append(np.array(f['entry/data/data']))
    # Stack data into array
    print('')
    #data_dict[key] = _check_xrd_data_shape(data_dict[key])
    print('done!')

    out = [data_dict, scan_md]

    if returns is not None:
        if 'data_keys' in returns:
            out.append(data_keys)
        if 'xrd_dets' in returns:
            xrd_dets = [det for det in scan_md['detectors']
                        if det in ['merlin', 'dexela']]
            out.append(xrd_dets)

    return out


def save_angle_rc_data(scanid=-1,
                       filedir=None,
                       filenames=None):

    data_dict, scan_md, data_keys = load_angle_rc_data(scanid=scanid,
                                                        returns=['data_keys'])
    
    xrd_dets = ['dexela']
    xrd_data = [data_dict[f'{xrd_det}_image'] for xrd_det in xrd_dets]

    if filenames is None:
        filenames = []
        for detector in xrd_dets:
            filenames.append(f'scan{scanid}_{detector}_angle_rc.tif')
    
    _save_xrd_tifs(xrd_data,
                  xrd_dets=xrd_dets,
                  scanid=scan_md['scan_id'], # Will return the correct value
                  filedir=filedir,
                  filenames=filenames)

    param_filename = f'scan{scanid}_angle_rc_parameters.txt'
    _save_map_parameters(data_dict, scanid, data_keys=data_keys,
                         filedir=filedir, filename=param_filename)
    

def load_flying_angle_rc_data(scanid=-1,
                              broker='manual',
                              detectors=None,
                              data_keys=['i0',
                                         'im',
                                         'it'],
                              returns=None,
                              repair_method='fill'):

    (data_dict,
     scan_md,
     data_keys,
     xrd_dets
     ) = load_data(scanid=scanid,
                   broker=broker,
                   detectors=detectors,
                   data_keys=data_keys,
                   returns=['data_keys',
                            'xrd_dets'],
                   repair_method=repair_method)

    # Interpolate angular positions
    thetas = np.linspace(*c[int(scanid)].start['scan']['scan_input'][:3])
    thetas /= 1000 # mdeg to deg
    data_dict['theta'] = thetas

    out = [data_dict, scan_md]

    if returns is not None:
        if 'data_keys' in returns:
            out.append(data_keys)
        if 'xrd_dets' in returns:
            out.append(xrd_dets)

    return out


def save_flying_angle_rc_data(scanid=-1,
                              broker='manual',
                              detectors=None,
                              data_keys=['i0',
                                         'im',
                                         'it'],
                              filedir=None,
                              filenames=None,
                              repair_method='fill'):

    # Retrieve and format data
    (data_dict,
     scan_md,
     xrd_dets
     ) = load_flying_angle_rc_data(
                    scanid=-1,
                    broker=broker,
                    detectors=detectors,
                    data_keys=data_keys,
                    returns=['xrd_dets'],
                    repair_method=repair_method)

    if filenames is None:
        filenames = []
        for detector in xrd_dets:
            filenames.append(f'scan{scanid}_{detector}_flying_angle_rc.tif')

    # Convert from dictionary to list
    xrd_data = [data_dict[f'{xrd_det}_image'] for xrd_det in xrd_dets]
       
    _save_xrd_tifs(xrd_data,
                   xrd_dets=xrd_dets,
                   scanid=scan_md['scan_id'], # Will return the correct value
                   filedir=filedir,
                   filenames=filenames)
    
    param_filename = f'scan{scanid}_flying_angle_rc_parameters.txt'
    _save_map_parameters(data_dict, scanid, data_keys=['i0',
                                                       'i0_time',
                                                       'im',
                                                       'it',
                                                       'theta'],
                         filedir=filedir, filename=param_filename)
    
    md_filename = f'scan{scanid}_flying_angle_rc_metadata.txt'                  
    _save_scan_md(scan_md, scanid,
                  filedir=filedir, filename=md_filename)


# def save_flying_angle_rc_data(scanid=-1,
#                               broker='manual',
#                               detectors=None,
#                               data_keys=['i0',
#                                          'im',
#                                          'it'],
#                               filedir=None,
#                               filenames=None,
#                               repair_method='fill'):

#     data_dict, scan_md, data_keys, xrd_dets = load_data(scanid=scanid,
#                                                         broker=broker,
#                                                         detectors=detectors,
#                                                         data_keys=data_keys,
#                                                         returns=['data_keys',
#                                                                  'xrd_dets'],
#                                                         repair_method=repair_method)

#     # Format xrd data as list from each detector
#     # Convert to 
#     xrd_data = [np.asarray(data_dict[f'{xrd_det}_image'])
#                 for xrd_det in xrd_dets]

#     # Interpolate angular positions
#     thetas = np.linspace(*c[int(scanid)].start['scan']['scan_input'][:3])
#     thetas /= 1000 # mdeg to deg
#     data_dict['theta'] = thetas

#     if filenames is None:
#         filenames = []
#         for detector in xrd_dets:
#             filenames.append(f'scan{scanid}_{detector}_flying_angle_rc.tif')
    
#     _save_xrd_tifs(xrd_data,
#                    xrd_dets=xrd_dets,
#                    scanid=scan_md['scan_id'], # Will return the correct value
#                    filedir=filedir,
#                    filenames=filenames)
    
#     param_filename = f'scan{scanid}_flying_angle_rc_parameters.txt'
#     _save_map_parameters(data_dict, scanid, data_keys=['i0',
#                                                        'i0_time',
#                                                        'im',
#                                                        'it',
#                                                        'theta'],
#                          filedir=filedir, filename=param_filename)
    
#     md_filename = f'scan{scanid}_flying_angle_rc_metadata.txt'                  
#     _save_scan_md(scan_md, scanid,
#                   filedir=filedir, filename=md_filename)




# Convenience/utility function for generating record of scan_ids

def generate_scan_logfile(start_id,
                          end_id=-1,
                          filename=None,
                          filedir=None,
                          include_peakups=False):

    if filedir is None:
        raise ValueError('No defualt filedir and none specified. Please define filedir.')

    if end_id == -1:
        bs_run = c[-1]
        end_id = int(c['scan_id'])

    if filename is None:
        filename = f'logfile_{start_id}-{end_id}'

    id_list = np.arange(int(start_id), int(end_id) + 1, 1)

    scan_ids = []
    scan_types = []
    scan_statuses = []
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
                    scan_inputs.append(str(start['scan']['scan_input']))
                else:
                    scan_inputs.append(str([]))
            else:
                scan_ids.append(str(start['scan_id']))
                scan_types.append('UNKOWN')
                scan_inputs.append(str([]))
            
            if stop is None:
                scan_statuses.append('NONE')
            else:
                scan_statuses.append(stop['exit_status'])

        except KeyError:
            print(f'scan_id={scan_id} not found!')

    logfile = f'{filedir}{filename}.txt'

    with open(logfile, 'a') as log:
        for scan_id, scan_type, scan_status, scan_input in zip(scan_ids, scan_types, scan_statuses, scan_inputs):
            log.write(f'{scan_id}\t{scan_type}\t{scan_status}\t{scan_input}\n')