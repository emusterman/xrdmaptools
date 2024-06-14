import numpy as np
import os
import h5py
import psutil


##################
### HDF Format ###
##################


def check_hdf_current_images(title, hdf_file=None, hdf=None):
    if hdf is None and hdf_file is not None:
        with h5py.File(hdf_file, 'r') as f:
            return title in f['/xrdmap/image_data']
    elif hdf is not None:
        return title in hdf['/xrdmap/image_data']
    else:
        raise ValueError('Must specify hdf_file or hdf.')
    

def get_optimal_chunks(data, approx_chunk_size=None):
    
    data_shape = data.shape
    data_nbytes = data[(0,) * data.ndim].nbytes

    if approx_chunk_size is None:
        available_memory = psutil.virtual_memory()[1] / (2**20) # In MB
        cpu_count = os.cpu_count()
        approx_chunk_size = (available_memory * 0.85) / cpu_count # 15% wiggle room
        #approx_chunk_size = np.round(approx_chunk_size)
        if approx_chunk_size > 2**10:
            approx_chunk_size = 2**10
    elif approx_chunks_size > 2**10:
        print('WARNING: Chunk sizes above 1 GB may start to perform poorly.')

    # Split images up by data size in MB that seems reasonable
    images_per_chunk = (approx_chunk_size * 2**20) / np.prod([*data_shape[-2:], data_nbytes], dtype=np.int64)

    # Try to make square chunks if possible
    square_chunks = np.sqrt(images_per_chunk)

    num_chunk_x = np.round(data_shape[0] / square_chunks, 0).astype(np.int32)
    num_chunk_x = min(max(num_chunk_x, 1), data_shape[0])
    chunk_x = data_shape[0] // num_chunk_x

    num_chunk_y = np.round(data_shape[1] / (data_shape[0] / num_chunk_x), 0).astype(np.int32)
    num_chunk_y = min(max(num_chunk_y, 1), data_shape[1])
    chunk_y = data_shape[1] // num_chunk_y

    # String togther, maintaining full images per chunk
    chunk_size = (chunk_x, chunk_y, *data_shape[-2:])

    return chunk_size


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