import numpy as np
import os
from skimage import io
import h5py
from collections import OrderedDict
import time as ttime
import pandas as pd

#################
### H5 Format ###
#################

def initialize_xrdmap_h5(xrdmap, h5_file):
    with h5py.File(h5_file, 'w-') as f:
        base_grp = f.require_group('xrdmap')
        base_grp.attrs['scanid'] = xrdmap.scanid
        base_grp.attrs['beamline'] = xrdmap.beamline #'5-ID (SRX)'
        base_grp.attrs['facility'] = xrdmap.facility #'NSLS-II'
        base_grp.attrs['energy'] = xrdmap.energy
        base_grp.attrs['wavelength'] = xrdmap.wavelength
        base_grp.attrs['time_stamp'] = '' # Not sure why I cannot assign None

        # Record diffraction data
        curr_grp = base_grp.require_group('image_data') # naming the group after the detector may be a bad idea...
        curr_grp.attrs['detector'] = '' #'dexela'
        curr_grp.attrs['detector_binning'] = '' #(4, 4)
        curr_grp.attrs['exposure_time'] = '' # 0.1
        curr_grp.attrs['expsure_time_units'] = 's'

        # Add pixel spatial positions

        # Add pixel scalar values


def check_h5_current_images(xrdmap, h5_file):
    with h5py.File(h5_file, 'r') as f:
        return xrdmap.map.title in f['/xrdmap/image_data']
    

def load_XRD_h5(filename, wd=None):
    # TODO: Add conditional to check for .h5 at end of file name

    h5_path = f'{wd}{filename}'
    with h5py.File(h5_path, 'r') as f:
        base_grp = f['/xrdmap']

        # Load base metadata
        base_md = dict(base_grp.attrs.items())

        # Load most recent image data
        if 'image_data' not in base_grp.keys():
            raise RuntimeError("No image data in h5 file! Try loading from image stack or from databroker.")
        img_grp = base_grp['image_data']

        time_stamps, img_keys = [], []
        for key in img_grp.keys():
            if key[0] != '_':
                time_stamps.append(img_grp[key].attrs['time_stamp'])
                img_keys.append(key)
        time_stamps = [ttime.mktime(ttime.strptime(x)) for x in time_stamps]
        recent_index = np.argmax(time_stamps)
        print(f'Loading most recent images ({img_keys[recent_index]})...', end='', flush=True)
        image_data = img_grp[img_keys[recent_index]][:]

        # Rebuild correction dictionary
        corrections = {}
        for key, value in img_grp[img_keys[recent_index]].attrs.items():
            if key[0] == '_' and key[-11:] == '_correction':
                corrections[key[1:-11]] = value

        image_data = ImageMap(image_data, title=img_keys[recent_index],
                              h5=h5_path, corrections=corrections)
        image_data.image_shape = img_grp['raw_images'].shape[2:]
        
        # I should look for other composite images???
        if '_processed_images_composite' in img_grp.keys():
            image_data._processed_images_composite = img_grp['_processed_images_composite'][:]

        if '_calibration_mask' in img_grp.keys():
            image_data.calibration_mask = img_grp['_calibration_mask'][:]

        if '_defect_mask' in img_grp.keys():
            image_data.defect_mask = img_grp['_defect_mask'][:]

        if '_custom_mask' in img_grp.keys():
            image_data.custom_mask = img_grp['_custom_mask'][:]
        
        if '_spot_masks' in img_grp.keys():
            image_data.spot_masks = img_grp['_spot_masks'][:]
        
        print('done!')

        # Recipricol positions
        if 'reciprocal_positions' in base_grp.keys():
            print('Loading reciprocal positions...', end='', flush=True)
            recip_grp = base_grp['reciprocal_positions']

            if 'tth' in recip_grp.keys() and 'chi' in recip_grp.keys():
                tth = recip_grp['tth'][:]
                chi = recip_grp['chi'][:]

                recip_pos = {
                    'tth' : tth,
                    'chi' : chi,
                    'calib_units' : recip_grp['tth'].attrs['units']
                }

                # Add some extra attributes to image_data
                image_data.tth_resolution = recip_grp['tth'].attrs['tth_resolution']
                image_data.chi_resolution = recip_grp['chi'].attrs['chi_resolution']
                image_data.extent = recip_grp.attrs['extent']
                image_data.calibrated_shape = (len(chi), len(tth))
                image_data.chi_num = image_data.calibrated_shape[0]
                image_data.tth_num = image_data.calibrated_shape[1]
                
            else:
                recip_pos = {'tth' : None,
                             'chi' : None,
                             'calib_units' : None}
                image_data.tth_resolution = None
                image_data.chi_resolution = None
                image_data.extent = None
                image_data.chi_num = None
                image_data.tth_num = None

            # Load poni_file calibration
            poni_grp = recip_grp['poni_file']
            ordered_keys = ['poni_version', 'detector', 'detector_config', 'dist', 'poni1', 'poni2', 'rot1', 'rot2', 'rot3', 'wavelength']
            poni_od = OrderedDict()
            for key in ordered_keys:
                if key == 'detector_config':
                    detector_od = OrderedDict()
                    detector_od['pixel1'] = poni_grp['detector_config'].attrs['pixel1']
                    detector_od['pixel2'] = poni_grp['detector_config'].attrs['pixel2']
                    detector_od['max_shape'] = list(poni_grp['detector_config'].attrs['max_shape'])
                    poni_od[key] = detector_od
                else:
                    poni_od[key] = poni_grp.attrs[key]
            print('done!')
        else:
            recip_pos = {'tth' : None,
                         'chi' : None,
                         'calib_units' : None}
            poni_od = None
        image_data.tth = recip_pos['tth']
        image_data.chi = recip_pos['chi']


        # Load phases
        phase_dict = {}
        if 'phase_list' in base_grp.keys():
            print('Loading saved phases...', end='', flush=True)
            phase_grp = base_grp['phase_list']
            if len(phase_grp) > 0:
                for phase in phase_grp.keys():
                    phase_dict[phase] = Phase.from_h5(phase_grp[phase],
                                                      energy=base_md['energy'],
                                                      tth=recip_pos['tth'])
            print('done!')


        # Load spots dataframe
        spots = None
        spot_model = None
        if 'reflections' in base_grp.keys():
            print('Loading reflection spots...', end='', flush=True)
            spots = pd.read_hdf(h5_path, key='xrdmap/reflections/spots')
            print('done!')

            # Load peak model
            if 'spot_model' in f['xrdmap/reflections'].attrs.keys():
                spot_model_name = f['xrdmap/reflections'].attrs['spot_model']
                spot_model = _load_peak_function(spot_model_name)
        
        # Load scalars

        # Load pixel positions
    
    
    # return dictionary of useful values
    ouput_dict = {'base_md' : base_md,
                  'image_data': image_data,
                  'recip_pos' : recip_pos,
                  'poni_od' : poni_od,
                  'phase_dict' : phase_dict,
                  'spots' : spots,
                  'spot_model' : spot_model}

    return ouput_dict

        



    
    





####################
### Image Stacks ###
####################


def load_xrd_patterns(scanid, detectors=[], return_detectors=False,
                      filedir=None, filenames=None):
    # This could be useful for the base class
    h = db[int(scanid)]

    if detectors == []:
        detectors = h.start['scan']['detectors']
    else:
        detectors = [detector.name if type(detector) is not str else str(detector)
                     for detector in detectors]
    detectors = [detector for detector in detectors if detector in ['merlin', 'dexela']]

    scantype = h.start['scan']['type']

    if filedir is None:
        filedir = '/home/xf05id1/current_user_data/'
    if filenames is None:
        filenames = []
        for detector in detectors:
            filenames.append(f'scan{scanid}_{detector}_xrd.tif')
    
    data = []
    for i, detector in enumerate(detectors):       
        # Check if data has already been loaded from the databroker
        if os.path.exists(f'{filedir}{filenames[i]}'):
            print('Found data from local drive. Loading from there.')
            d = io.imread(f'{filedir}{filenames[i]}')
        # Otherwise load data from databroker
        else:
            if 'FLY' in scantype:
                d = h.data(f'{detector}_image', stream_name='stream0', fill=True)
            elif 'STEP' in scantype:
                d = h.data(f'{detector}_image', fill=True)
            d = np.array(list(d))

            if (d.size == 0):
                print('Error collecting dexela data...')
                return
            elif len(d.shape) == 1:
                print('Map is missing pixels!\nStacking all patterns together.')
                flat_d = np.array([x for y in d for x in y]) # List comprehension is dumb
                d = flat_d.reshape(flat_d.shape[0], 1, *flat_d.shape[-2:])
        
        data.append(d)
    
    # Helps to clear out files that somehow do not close
    # Has yet to be tested...
    # db._catalog._entries.cache_clear()
    # gc.collect()

    # Return the data
    if return_detectors:
        return tuple(data), detectors # Apparently it's bad form having different types of returns
    else:
        return tuple(data)



def load_map_parameters(scanid):
    h = db[int(scanid)]

    x = np.array(list(h.data('enc1', stream_name='stream0', fill=True)))
    y = np.array(list(h.data('enc2', stream_name='stream0', fill=True)))
    I0= np.array(list(h.data('i0', stream_name='stream0', fill=True)))

    return (x, y, I0)


def make_composite_pattern(scanid, detectors=[],
                           method='sum', subtract=None,
                           return_detectors=False,
                           filedir=None, filenames=None):
    if return_detectors:
        data, detectors = load_xrd_patterns(scanid, detectors=detectors,
                                            return_detectors=return_detectors,
                                            filedir=filedir, filenames=filenames)
    else:
        data = load_xrd_patterns(scanid, detectors=detectors,
                                 filedir=filedir, filenames=filenames)

    method = method.lower()
    if subtract is not None:
        subtract = subtract.lower()

    comps = []
    for xrd in data:
        if method in ['sum', 'total']:
            comp = np.sum(xrd, axis=(0, 1)) / np.multiply(*xrd.shape[:2])
        elif method in ['mean', 'avg', 'average']:
            comp = np.mean(xrd, axis=(0, 1))
        elif method in ['max', 'maximum']:
            comp = np.max(xrd, axis=(0, 1))
        elif method in ['med', 'median']:
            if subtract in ['med', 'median']:
                print("Composite method and subtract cannot both be 'median'!")
                print("Changing subract to None")
                subract = None
            comp = np.median(xrd, axis=(0, 1))
        else:
            raise ValueError("Composite method not allowed.")

        if subtract is None:
            pass
        elif type(subtract) is np.ndarray:
            comp = comp - subtract
        elif subtract in ['min', 'minimum']:
            comp = comp - np.min(xrd, axis=(0, 1))
        elif subtract in ['med', 'median']:
            comp = comp - np.median(xrd, axis=(0, 1))
        else:
            print('Unknown subtract input. Proceeding without any subtraction.')

        comps.append(np.squeeze(comp))
    
    if return_detectors:
        return tuple(comps), detectors
    else:
        return tuple(comps)


def extract_calibration_pattern(scanid, detectors=[],
                                filedir=None, filenames=None):
    if filedir is None:
        filedir = '/home/xf05id1/current_user_data/'
    if scanid == -1:
        h = db[scanid]
        scanid = int(h.start['scan_id'])

    comps, detectors = make_composite_pattern(scanid, detectors=detectors,
                                              method='max', subtract='min',
                                              return_detectors=True,
                                              filedir=filedir, filenames=filenames)
    if filenames is None:
        filenames = []
        for detector in detectors:
            filenames.append(f'scan{scanid}_{detector}_calibration.tif')

    for i, comp in enumerate(comps):
        io.imsave(f'{filedir}{filenames[i]}', comp, check_contrast=False)


def save_xrd_tiffs(scanid, filedir=None, filenames=None, detectors=[]):
    if filedir is None:
        filedir = '/home/xf05id1/current_user_data/'
    if scanid == -1:
        h = db[scanid]
        scanid = int(h.start['scan_id'])

    xrd_maps, detectors = load_xrd_patterns(scanid, detectors=detectors,
                                            return_detectors=True,
                                            filedir=filedir, filenames=filenames)

    if filenames is None:
        filenames = []
        for detector in detectors:
            filenames.append(f'scan{scanid}_{detector}_xrd.tif')
    
    for i, xrd in enumerate(xrd_maps):
        io.imsave(f'{filedir}{filenames[i]}', xrd.astype(np.uint16), check_contrast=False)



def save_map_parameters(scanid, filedir=None, filename=None):
    if filedir is None:
        filedir = '/home/xf05id1/current_user_data/'
    if scanid == -1:
        h = db[scanid]
        scanid = int(h.start['scan_id'])
    if filename is None:
        filename = f'scan{scanid}_map_parameters.txt'
    
    x, y, I0 = load_map_parameters(scanid)

    (N, M) = x.shape
    x_flat = np.reshape(x, (N * M, ))
    y_flat = np.reshape(y, (N * M, ))
    I0_flat = np.reshape(I0, (N * M, ))

    np.savetxt(f'{filedir}{filename}', np.array((x_flat, y_flat, I0_flat)))



def save_composite_pattern(scanid, detectors=[],
                           method='sum', subtract=None,
                           filedir=None, filenames=None):
    if filedir is None:
        filedir = '/home/xf05id1/current_user_data/'
    if scanid == -1:
        h = db[scanid]
        scanid = int(h.start['scan_id'])
    
    comps, detectors = make_composite_pattern(scanid, detectors=detectors,
                                              method=method, subtract=subtract,
                                              return_detectors=True,
                                              filedir=filedir, filenames=filenames)
    
    if filenames is None:
        filenames = []
        for detector in detectors:
            filenames.append(f'scan{scanid}_{detector}_{method}{subtract}_composite.tif')
    
    for i, comp in enumerate(comps):
        io.imsave(f'{filedir}{filenames[i]}', comp, check_contrast=False)


##########################
### Original Functions ###
##########################


''''def make_tiff(scanid, *, scantype='', fn=''):
        h = db[int(scanid)]
        if scanid == -1:
            scanid = int(h.start['scan_id'])
    
        if scantype == '':
            scantype = h.start['scan']['type']
    
        if ('FLY' in scantype):
            d = list(h.data('dexela_image', stream_name='stream0', fill=True))
            d = np.array(d)
    
            (row, col, imgY, imgX) = d.shape
            if (d.size == 0):
                print('Error collecting dexela data...')
                return
            d = np.reshape(d, (row * col, imgY, imgX))
        elif ('STEP' in scantype):
            d = list(h.data('dexela_image', fill=True))
            d = np.array(d)
            d = np.squeeze(d)
        # elif (scantype == 'count'):
        #     d = list(h.data('dexela_image', fill=True))
        #     d = np.array(d)
        else:
            print('I don\'t know what to do.')
            return
    
        if fn == '':
            fn = f"scan{scanid}_xrd.tiff"
        try:
            io.imsave(fn, d.astype('uint16'))
        except:
            print(f'Error writing file!')
    

def export_flying_merlin2tiff(scanid=-1, wd=None):
        if wd is None:
            wd = '/home/xf05id1/current_user_data/'
    
        print('Loading data...')
        h = db[int(scanid)]
        d = h.data('merlin_image', stream_name='stream0', fill=True)
        d = np.array(list(d))
        d = np.squeeze(d)
        d = np.array(d, dtype='float32')
        x = np.array(list(h.data('enc1', stream_name='stream0', fill=True)))
        y = np.array(list(h.data('enc2', stream_name='stream0', fill=True)))
        I0= np.array(list(h.data('i0', stream_name='stream0', fill=True)))
    
        # Flatten arrays
        (N, M) = x.shape
        x_flat = np.reshape(x, (N * M, ))
        y_flat = np.reshape(y, (N * M, ))
        I0_flat = np.reshape(I0, (N * M, ))
    
        # Get scanid
        if (scanid < 0):
            scanid = h.start['scan_id']
    
        print('Writing data...')
        fn = 'scan%d.tif' % scanid
        fn_txt = 'scan%d.txt' % scanid
        io.imsave(wd + fn, d)
        np.savetxt(wd + fn_txt, np.array((x_flat, y_flat, I0_flat)))
           # HACK to make sure we clear the cache.  The cache size is 1024 so
        # this would eventually clear, however on this system the maximum
        # number of open files is 1024 so we fail from resource exaustion before
        # we evict anything.
        db._catalog._entries.cache_clear()
        gc.collect()'''