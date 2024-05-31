import numpy as np
import h5py
from collections import OrderedDict
import time as ttime
import pandas as pd
import dask.array as da

# Local imports
#from ..XRDMap import XRDMap
from ..ImageMap import ImageMap
from ..crystal.Phase import Phase
from ..reflections.SpotModels import _load_peak_function


##################
### HDF Format ###
##################

def initialize_xrdmap_hdf(xrdmap, hdf_file):
    with h5py.File(hdf_file, 'w-') as f:
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
    

def load_XRD_hdf(filename, wd=None, dask_enabled=False):
    # TODO: Add conditional to check for .h5 at end of file name

    # Figuring out hdf file stuff
    hdf_path = f'{wd}{filename}'
    if not dask_enabled:
        hdf = h5py.File(hdf_path, 'r')
    else:
        hdf = h5py.File(hdf_path, 'a')
    base_grp = hdf['xrdmap']

    # Load base metadata
    base_md = dict(base_grp.attrs.items())

    # Load most recent image data
    if 'image_data' not in base_grp.keys():
        raise RuntimeError("No image data in hdf file! Try loading from image stack or from databroker.")
    img_grp = base_grp['image_data']

    time_stamps, img_keys = [], []
    for key in img_grp.keys():
        if key[0] != '_':
            time_stamps.append(img_grp[key].attrs['time_stamp'])
            img_keys.append(key)
    time_stamps = [ttime.mktime(ttime.strptime(x)) for x in time_stamps]
    recent_index = np.argmax(time_stamps)

    print(f'Loading most recent images ({img_keys[recent_index]})...', end='', flush=True)
    if dask_enabled:
        # Lazy loads data
        image_dset = img_grp[img_keys[recent_index]]
        image_data = da.from_array(image_dset, chunks=image_dset.chunks)
    else:
        # Fully loads data
        image_data = img_grp[img_keys[recent_index]][:]

    # Rebuild correction dictionary
    corrections = {}
    for key, value in img_grp[img_keys[recent_index]].attrs.items():
        if key[0] == '_' and key[-11:] == '_correction':
            corrections[key[1:-11]] = value

    # Collect ImageMap attributes that are not instantiated...
    image_map_attrs = {}
    
    #image_map.image_shape = img_grp['raw_images'].shape[2:]
    image_map_attrs['image_shape'] = img_grp['raw_images'].shape[2:]
    
    # I should look for other composite images???
    if '_processed_images_composite' in img_grp.keys():
        #image_map._processed_images_composite = img_grp['_processed_images_composite'][:]
        image_map_attrs['_processed_images_composite'] = img_grp['_processed_images_composite'][:]

    if '_calibration_mask' in img_grp.keys():
        #image_map.calibration_mask = img_grp['_calibration_mask'][:]
        image_map_attrs['calibration_mask'] = img_grp['_calibration_mask'][:]

    if '_defect_mask' in img_grp.keys():
        #image_map.defect_mask = img_grp['_defect_mask'][:]
        image_map_attrs['defect_mask'] = img_grp['_defect_mask'][:]

    if '_custom_mask' in img_grp.keys():
        #image_map.custom_mask = img_grp['_custom_mask'][:]
        image_map_attrs['custom_mask'] = img_grp['_custom_mask'][:]
    
    if '_spot_masks' in img_grp.keys():
        #image_map.spot_masks = img_grp['_spot_masks'][:]
        image_map_attrs['spot_masks'] = img_grp['_spot_masks'][:]
    
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

            # Add some extra attributes to image_map
            #image_map.tth_resolution = recip_grp['tth'].attrs['tth_resolution']
            #image_map.chi_resolution = recip_grp['chi'].attrs['chi_resolution']
            #image_map.extent = recip_grp.attrs['extent']
            #image_map.calibrated_shape = (len(chi), len(tth))
            #image_map.chi_num = image_map.calibrated_shape[0]
            #image_map.tth_num = image_map.calibrated_shape[1]
            image_map_attrs['tth_resolution'] = recip_grp['tth'].attrs['tth_resolution']
            image_map_attrs['chi_resolution'] = recip_grp['chi'].attrs['chi_resolution']
            image_map_attrs['extent'] = recip_grp.attrs['extent']
            image_map_attrs['calibrated_shape'] = (len(chi), len(tth))
            image_map_attrs['chi_num'] = len(chi)
            image_map_attrs['tth_num'] = len(tth)

        else:
            recip_pos = {'tth' : None,
                         'chi' : None,
                         'calib_units' : None}
            #image_map.tth_resolution = None
            #image_map.chi_resolution = None
            #image_map.extent = None
            #image_map.chi_num = None
            #image_map.tth_num = None
            image_map_attrs['tth_resolution'] = None
            image_map_attrs['chi_resolution'] = None
            #image_map_attrs['extent'] = None
            image_map_attrs['chi_num'] = None
            image_map_attrs['tth_num'] = None

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
                detector_od['orientation'] = poni_grp['detector_config'].attrs['orientation']
                poni_od[key] = detector_od
            else:
                poni_od[key] = poni_grp.attrs[key]
        print('done!')
    else:
        recip_pos = {'tth' : None,
                     'chi' : None,
                     'calib_units' : None}
        #image_map.tth_resolution = None
        #image_map.chi_resolution = None
        image_map_attrs['tth_resolution'] = None
        image_map_attrs['chi_resolution'] = None
        poni_od = None
    #image_map.tth = recip_pos['tth']
    #image_map.chi = recip_pos['chi']
    image_map_attrs['tth'] = recip_pos['tth']
    image_map_attrs['chi'] = recip_pos['chi']

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
    # Should be last since it closed base_grp
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
            img_grp = hdf['xrdmap/image_data']
            image_dset = img_grp[img_keys[recent_index]]
            image_data = da.from_array(image_dset, chunks=image_dset.chunks)
        else:
            hdf = h5py.File(hdf_path, 'r')
        print('done!')

        # Load peak model
        if 'spot_model' in hdf['xrdmap/reflections'].attrs.keys():
            spot_model_name = hdf['xrdmap/reflections'].attrs['spot_model']
            spot_model = _load_peak_function(spot_model_name)
            
    if not dask_enabled:
        hdf.close()
        hdf = None

    # Instantiate ImageMap!
    print(f'Instantiating ImageMap...', end='', flush=True)
    image_map = ImageMap(image_data,
                         title=img_keys[recent_index],
                         wd=wd,
                         hdf_path=hdf_path,
                         hdf=hdf,
                         corrections=corrections,
                         dask_enabled=dask_enabled)
    
    # Add extra ImageMap attributes
    for key, value in image_map_attrs.items():
        setattr(image_map, key, value)
    print('done!')
    
    # return dictionary of useful values
    ouput_dict = {'base_md' : base_md,
                  'image_data': image_map,
                  'recip_pos' : recip_pos,
                  'poni_od' : poni_od,
                  'phase_dict' : phase_dict,
                  'spots' : spots,
                  'spot_model' : spot_model,
                  'sclr_dict' : sclr_dict,
                  'pos_dict' : pos_dict}

    return ouput_dict


# Just a convenience wrapper wihout returning the class
'''def make_xrdmap_hdf(scanid=-1,
                    broker='manual',
                    filedir=None,
                    filename=None,
                    poni_file=None):
    
    XRDMap.from_db(scanid=scanid,
                   broker=broker,
                   filedir=filedir,
                   filename=filename,
                   poni_file=poni_file,
                   save_hdf=True)'''


'''def make_xrdmap_hdf(scanid=-1,
                    broker='manual',
                    filedir=None,
                    filename=None,
                    poni_file=None,
                    return_xrdmap=False,
                    save_hdf=True):
    
    pos_keys = ['enc1', 'enc2']
    sclr_keys = ['i0', 'i0_time', 'im', 'it']
    
    if data_keys is None:
        data_keys = pos_keys + sclr_keys

    data_dict, scan_md, data_keys, xrd_dets = load_data(scanid=scanid,
                                                        broker=broker,
                                                        detectors=None,
                                                        data_keys=None,
                                                        returns=['data_keys',
                                                                 'xrd_dets'])

    xrd_data = [data_dict[f'{xrd_det}_image'] for xrd_det in xrd_dets]

    # Make position dictionary
    pos_dict = {key:value for key, value in data_dict.items() if key in pos_keys}

    # Make scaler dictionary
    sclr_dict = {key:value for key, value in data_dict.items() if key in sclr_keys}

    if len(xrd_data) > 1:
        pass
        # Add more to filename to prevent overwriting...

    extra_md = {}
    for key in scan_md.keys():
        if key not in ['scanid', 'beamline', 'energy', 'dwell', 'start_time']:
            extra_md[key] = scan_md[key]
    
    xrdmaps = []
    for xrd_data_i in xrd_data:
        xrdmap = XRDMap(scanid=scan_md['scanid'],
                        wd=filedir,
                        filename=filename,
                        #hdf_filename=None, # ???
                        #hdf=None,
                        image_map=xrd_data_i,
                        #map_title=None,
                        #map_shape=None,
                        energy=scan_md['energy'],
                        #wavelength=None,
                        dwell=scan_md['dwell'],
                        poni_file=poni_file,
                        sclr_dict=sclr_dict,
                        pos_dict=pos_dict,
                        #tth_resolution=None,
                        #chi_resolution=None,
                        #tth=None,
                        #chi=None,
                        beamline=scan_md['beamline'],
                        facility='NSLS-II',
                        time_stamp=scan_md['start_time'],
                        #extra_metadata=None,
                        save_hdf=save_hdf,
                        #dask_enabled=False
                        )
        
        xrdmaps.append(xrdmap)

    if return_xrdmap:
        if len(xrdmaps) > 1:
            return tuple(xrdmaps)
        else:
            # Don't bother returning a tuple or list of xrdmaps
            return xrdmaps[0]'''