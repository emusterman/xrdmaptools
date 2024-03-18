import numpy as np
from skimage import io
import h5py
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
              repair_method='fill'):

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
        except: # what error should this through???
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

    #data = []
    #for i, detector in enumerate(detectors):       
    #    if 'FLY' in scantype:
    #        d = h.data(f'{detector}_image', stream_name='stream0', fill=True)
    #    elif 'STEP' in scantype:
    #        d = h.data(f'{detector}_image', fill=True)
    #    d = np.array(list(d))
    #
    #    if (d.size == 0):
    #        print('Error collecting dexela data...')
    #        return
    #    elif len(d.shape) == 1:
    #        print('Map is missing pixels!\nStacking all patterns together.')
    #        flat_d = np.array([x for y in d for x in y]) # List comprehension is dumb
    #        d = flat_d.reshape(flat_d.shape[0], 1, *flat_d.shape[-2:])
    #    
    #    data.append(d)
    
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
                     repair_method='fill'):

    if str(broker).lower() in ['tiled']:
        bs_run = c[int(scanid)]
    elif str(broker).lower() in ['db', 'databroker', 'broker']:
        bs_run = db[int(scanid)]
    
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
            data_dict[key] = np.stack(data_dict[key])
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
        for key in sclr_keys:
            data_dict[key] = np.stack(data_dict[key])
        if 'i0_time' in data_keys:
            data_dict['i0_time'] = data_dict['sis_time'] / 50e6 # 50 MHz clock
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
        data_dict[key] = np.stack(data_dict[key])
        dr_rows, br_rows = _flag_broken_rows(data_dict[key], key)
        dropped_rows += dr_rows
        broken_rows += br_rows

    r_key = 'DEXELA_FLY_V1'
    if r_key in r_paths.keys():
        print('Loading dexela...')
        key = 'dexela_image'
        for r_path in r_paths[r_key]:
            with h5py.File(r_path, 'r') as f:
                data_dict[key].append(np.array(f['entry/data/data']))
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
    dropped_rows = list(np.unique(dropped_rows)).sort()
    broken_rows = list(np.unique(broken_rows)).sort()
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

    return out


def _load_scan_metadata(bs_run, keys=None):

    if keys is None:
        keys = ['scan_id',
                'uid',
                'beamline_id',
                'type',
                'detectors',
                'energy',
                'dwell',
                'time_str',
                'sample_name',
                'theta']

    remaining_keys = []
    scan_md = {}
    for key in keys:
        if key in bs_run.start.keys():
            scan_md[key] = bs_run.start[key]
        elif key in bs_run.start['scan'].keys():
            scan_md[key] = bs_run.start['scan'][key]
            if key == 'theta': # Not very generalizable...
                scan_md[key] = bs_run.start['scan'][key]['val'] / 1000
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
            print(f'WARNING: {msg} data from row {row} captured only {d.shape[[0]]}/{mode_shape[0]} points.')
            broken_rows.append(row)

    return dropped_rows, broken_rows

def _repair_data_dict(data_dict, dropped_rows, broken_rows, repair_method='fill'):
    if len(dropped_rows) > 0 and len(broken_rows) > 0:
        print(f'Repairing data with "{repair_method}" method.')
    else:
        return data_dict

    if repair_method.lower() not in ['flatten', 'fill']:
        raise ValueError('Only "flatten" and "fill" repair methods are supported.')
    
    keys = list(data_dict.keys())

    # Check data shape
    num_rows = -1
    for key in keys:
        if num_rows == -1:
            num_rows = len(data_dict[key])
        elif len(data_dict[key]) != num_rows:
            raise ValueError('Data keys have different number of rows!')
        
    last_good_row = -1
    queued_rows = []
    for row in range(num_rows):
        # Only drop rows when flattening data. Do not fill data when flattening either
        if repair_method == 'flatten' and row in dropped_rows:
            print(f'Removing data from row {row}.')
            for key in keys:
                data_dict[key].pop(row)
            # Adjust remaining rows indices
            for ind, rem_row in enumerate(dropped_rows):
                if rem_row > row:
                    dropped_rows[ind] -= 1
        
        # Only replace data when filling and only for broken rows
        elif repair_method == 'fill':
            if row in broken_rows:
                if last_good_row == -1:
                    queued_rows.append(row)
                else:
                    print(f'Data in row {row} replaced with data in row {last_good_row}.')
                    for key in keys:
                        data_dict[key][row] = data_dict[key][last_good_row]
        
            else:
                last_good_row = row
                if len(queued_rows) > 0:
                    for q_row in queued_rows:
                        print(f'Data in row {q_row} replaced with data in row {last_good_row}.')
                        for key in keys:
                            data_dict[key][q_row] = data_dict[key][last_good_row]
    
    # Do not fully flatten if only dropping some rows I guess
    if repair_method == 'flatten' and len(broken_rows) > 0:
        print('Map is missing pixels. Flattening data.')
        for key in keys:
            flat_d = np.array([x for y in data_dict[key] for x in y]) # List comprehension is dumb
            data_dict[key] = flat_d.reshape(flat_d.shape[0], 1, *flat_d.shape[-2:])
    
    return data_dict



def _check_xrd_data_shape(data_list, repair_method='fill'):

    if repair_method.lower() not in ['flatten', 'fill']:
        raise ValueError('Only "flatten" and "fill" repair methods are supported.')

    # Majority of row shapes
    mode_shape = tuple(mode([d.shape for d in data_list])[0])

    # Find and fix broken data
    last_good_row = -1
    broken_rows = []
    FLATTEN_FLAG = False
    for row, d in enumerate(data_list):

        # Check for broken data
        if d.shape != mode_shape:
            # Check image shape
            if d.shape[-2:] != mode_shape[-2:]:
                print(f'WARNING: XRD data from row {row} has an incorrect image shape.')
                if repair_method == 'flatten':
                    print(f'Removing data from row {row}.')
                    data_list.pop(row)

            # Check for dropped frames
            elif d.shape[0] != mode_shape[0]:
                print(f'WARNING: XRD data from row {row} captured only {d.shape[[0]]}/{mode_shape[0]} frames.')

            # Figure out how to fix the data
            if last_good_row == -1:
                # Wait to correct rows until a good row is located
                broken_rows.append(row)
            else:
                if repair_method == 'fill':
                    data_list[row] = data_list[last_good_row]
                    print(f'XRD data in row {row} is replaced with data in row {last_good_row}.')
                elif repair_method == 'flatten':
                    FLATTEN_FLAG = True

        else:
            last_good_row = row
            if len(broken_rows) > 0:
                if repair_method == 'fill':
                    for b_row in broken_rows:
                        data_list[b_row] = data_list[last_good_row]
                        print(f'XRD data in row {b_row} is replaced with data in row {last_good_row}.')
                elif repair_method == 'flatten':
                    FLATTEN_FLAG = True
    
    if repair_method == 'flatten' and FLATTEN_FLAG:
        print('Map is missing pixels!\nStacking all patterns together.')
        flat_d = np.array([x for y in data_list for x in y]) # List comprehension is dumb
        data_list = flat_d.reshape(flat_d.shape[0], 1, *flat_d.shape[-2:])

    data = np.asarray(data_list)

    return data


'''def old_check_xrd_data_shape(data_list, repair_method='flatten'):


    if repair_method.lower() not in ['flatten', 'fill']:
        raise ValueError('Only "flatten" and "fill" repair methods are supported.')

    # Majority of row shape
    mode_shape = tuple(mode([d.shape for d in data_list])[0])

    # Flag rows with broken data
    dropped_frames_rows = []
    bad_image_rows = []
    for row, d in enumerate(data_list):
        # Check for bad shaped images
        # This eliminates first row of data when bad shape issues occure
        if d.shape[-2:] != mode_shape[-2:]:
            print(f'WARNING: XRD data from row {row} will be removed due to incorrect image shape.')
            bad_image_rows.append(row)
            #data_list.pop(row)

        # Then check and flag dropped frames
        elif d.shape[0] != mode_shape[0]:
            dropped_frames_rows.append(row)

    # Drop rows with bad image shapes. Should only be first row, but can handle any
    for row in bad_image_rows:
        data_list.pop(row)
        for index, other_row in enumerate(dropped_frames_rows):
            if other_row == row:
                # Remove from dropped frames too
                dropped_frames_rows.pop(index)
            if other_row > row:
                # Shift indices down
                dropped_frames_rows[index] -= 1

    # Repair rows with dropped frames. Decently common unfortunately
    # Flatten the data. Useful when spatial correlations do not matter (e.g., calibration, dark_field, etc.)
    if repair_method == 'flatten' and len(dropped_frames_rows) > 0:
        print('Map is missing pixels!\nStacking all patterns together.')
        flat_d = np.array([x for y in data_list for x in y]) # List comprehension is dumb
        data_list = flat_d.reshape(flat_d.shape[0], 1, *flat_d.shape[-2:])
    
    # Fill data. Useful when spatial correlations do matter (e.g., maps)
    elif repair_method == 'fill' and len(dropped_frames_rows) > 0:
        for row in dropped_frames_rows:
            pass

    data = np.asarray(data_list)

    return data'''


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
def make_composite_pattern(xrd_data, method='sum', subtract=None):

    if not isinstance(xrd_data, list):
        xrd_data = [xrd_data]

    method = method.lower()

    # Can the axis variable be made to always create images along the last two images??
    comps = []
    for xrd in xrd_data:
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
def _save_xrd_tifs(xrd_data, xrd_dets=None, scanid=None, filedir=None, filenames=None):

    if (scanid is None and filenames is None):
        raise ValueError('Must define scanid or filename to name the save file.')
    elif (xrd_dets is None and filenames is None):
        raise ValueError('Must define xrd_dets or filename to name the save file.')
    
    if not isinstance(xrd_data, list):
        xrd_data = [xrd_data]

    if filedir is None:
        filedir = '/home/xf05id1/current_user_data/'   
    if filenames is None:
        filenames = []
        for detector in xrd_dets:
            filenames.append(f'scan{scanid}_{detector}_xrd.tif')

    if len(filenames) != len(xrd_data):
        raise ValueError('Length of filenames does not match length of xrd_data')
    
    for i, xrd in enumerate(xrd_data):
        io.imsave(f'{filedir}{filenames[i]}', xrd.astype(np.uint16), check_contrast=False)
        # I think the data should already be an unsigned integer
        #io.imsave(f'{filedir}{filenames[i]}', xrd, check_contrast=False)
    print(f'Saved pattern(s) for scan {scanid}!')


# General function to save composite pattern as tif
def _save_composite_pattern(comps, method, subtract,
                           xrd_dets=None, scanid=None,
                           filedir=None, filenames=None):

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
    _save_xrd_tifs(comps, xrd_dets=xrd_dets, scanid=scanid, filedir=filedir, filenames=filenames)


def _save_calibration_pattern(xrd_data, xrd_dets, scanid=None, filedir=None):

    if not isinstance(xrd_data, list):
        xrd_data = [xrd_data]
    
    if filedir is None:
        filedir = '/home/xf05id1/current_user_data/'

    filenames = []
    for detector in xrd_dets:
        filenames.append(f'scan{scanid}_{detector}_calibration.tif')
    
    print('Saving calibration pattern...')
    _save_xrd_tifs(xrd_data, xrd_dets=xrd_dets, scanid=scanid, filedir=filedir, filenames=filenames)


# General function save scan metadata 
def _save_map_parameters(data_dict, scanid, data_keys=[], filedir=None, filename=None):

    if filedir is None:
        filedir = '/home/xf05id1/current_user_data/'
    if filename is None:
        filename = f'scan{scanid}_map_parameters.txt'
    if data_keys == []:
        data_keys = ['enc1', 'enc2', 'i0', 'i0_time', 'im', 'it']

    map_data = np.stack([data_dict[key].ravel() for key in data_keys])
    np.savetxt(f'{filedir}{filename}', map_data)
    print(f'Saved map parameters for scan {scanid}!')


def _save_scan_md(scan_md, scanid, filedir=None, filename=None):
    import json

    if not isinstance(scan_md, dict):
        raise TypeError('Scan metadata not provided as a dictionary.')
    
    if filedir is None:
        filedir = '/home/xf05id1/current_user_data/'
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
                  repair_method='fill'):

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
                  scanid=scan_md['scanid'], # Will return the correct value
                  filedir=filedir,
                  filenames=filenames)


# Function to load and save composite pattern as tif
def save_composite_pattern(scanid=-1,
                           broker='tiled',
                           method='sum', subtract=None,
                           detectors=None, data_keys=[], 
                           filedir=None, filenames=None):

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
                            scanid=scan_md['scanid'],
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
                             scanid=scan_md['scanid'],
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
                        scan_md['scanid'],
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
                  scan_md['scanid'],
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
                   repair_method='fill'):
    
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
                   scanid=scan_md['scanid'], # Will return the correct value
                   filedir=filedir)
    
    # Save map parameters (x, y, i0, etc...)
    _save_map_parameters(data_dict,
                         scan_md['scanid'],
                         data_keys=[],
                         filedir=filedir)
    
    # Save metadata (scaid, uid, energy, etc...)
    _save_scan_md(scan_md,
                  scan_md['scanid'],
                  filedir=filedir)


### Energy Rocking Curve Scans ###

def load_energy_rc_data(scanid=-1,
                        detectors=None,
                        data_keys=None,
                        returns=None):

    bs_run = c[int(scanid)]

    scan_md = {
        'scanid' : bs_run.start['scan_id'],
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

    print('Loading scalers and energies...', end='', flush=True)
    for doc in docs:
        if doc[0] == 'event_page':
            if 'dexela_image' in doc[1]['filled'].keys():
                for key in data_keys:
                    data_dict[key].append(doc[1]['data'][key][0])
    print('done!')

    # Fix keys
    data_dict['energy'] = np.array(data_dict['energy_energy'])
    data_dict['i0'] = np.array(data_dict['sclr_i0'])
    data_dict['im'] = np.array(data_dict['sclr_im'])
    data_dict['it'] = np.array(data_dict['sclr_it'])
    del data_dict['energy_energy'], data_dict['sclr_i0'], data_dict['sclr_im'], data_dict['sclr_it']
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
    data_dict[key] = _check_xrd_data_shape(data_dict[key])
    print('done!')

    out = [data_dict, scan_md]

    if returns is not None:
        if 'data_keys' in returns:
            out.append(data_keys)

    return out


def save_energy_rc_data(scanid=-1,
                        filedir=None,
                        filenames=None):

    data_dict, scan_md, data_keys = load_energy_rc_data(scanid=scanid,
                                                        returns=['data_keys'])
    
    xrd_dets = ['dexela']
    xrd_data = [data_dict[f'{xrd_det}_image'] for xrd_det in xrd_dets]

    if filenames is None:
        filenames = []
        for detector in xrd_dets:
            filenames.append(f'scan{scanid}_{detector}_energy_rc.tif')
    
    _save_xrd_tifs(xrd_data,
                  xrd_dets=xrd_dets,
                  scanid=scan_md['scanid'], # Will return the correct value
                  filedir=filedir,
                  filenames=filenames)

    param_filename = f'scan{scanid}_energy_rc_parameters.txt'
    _save_map_parameters(data_dict, scanid, data_keys=data_keys,
                         filedir=filedir, filename=param_filename)


### Angle Rocking Curve Scans ###

def load_angle_rc_data(scanid=-1,
                        detectors=None,
                        data_keys=None,
                        returns=None):

    bs_run = c[int(scanid)]

    scan_md = {
        'scanid' : bs_run.start['scan_id'],
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
    data_dict[key] = _check_xrd_data_shape(data_dict[key])
    print('done!')

    out = [data_dict, scan_md]

    if returns is not None:
        if 'data_keys' in returns:
            out.append(data_keys)

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
                  scanid=scan_md['scanid'], # Will return the correct value
                  filedir=filedir,
                  filenames=filenames)

    param_filename = f'scan{scanid}_angle_rc_parameters.txt'
    _save_map_parameters(data_dict, scanid, data_keys=data_keys,
                         filedir=filedir, filename=param_filename)