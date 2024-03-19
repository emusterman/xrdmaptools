# This entire module is to separate the make_crdmap_hdf to avoid circular imports
# I could not figure another way around this...


# Local imports
from ..XRDMap import XRDMap

# Just a convenience wrapper wihout returning the class
def make_xrdmap_hdf(scanid=-1,
                    broker='manual',
                    filedir=None,
                    filename=None,
                    poni_file=None,
                    repair_method='replace'):
    
    print('*' * 72)
    XRDMap.from_db(scanid=scanid,
                   broker=broker,
                   filedir=filedir,
                   filename=filename,
                   poni_file=poni_file,
                   save_hdf=True,
                   repair_method=repair_method)
    print('*' * 72)
    


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


def make_xrdmap_composite():
    raise NotImplementedError()