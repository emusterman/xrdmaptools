import numpy as np
import os
from tqdm import tqdm
from skimage import io
import h5py
import time as ttime
from scipy.optimize import curve_fit
from matplotlib import colors
import matplotlib.pyplot as plt

# Local imports
from xrdmaptools.XRDRockingCurve import XRDRockingCurve
from xrdmaptools.XRDMap import XRDMap
from xrdmaptools.XRDMapStack import XRDMapStack
from xrdmaptools.io.db_io import *
# from xrdmaptools.reflections.spot_blob_indexing_3D import *
# from xrdmaptools.crystal.strain import *

from xrdmaptools.utilities.utilities import (
    timed_iter,
    pathify
)
from xrdmaptools.utilities.math import compute_r_squared, tth_2_q
from xrdmaptools.reflections.SpotModels import (
    GaussianFunctions,
    generate_bounds
)
from xrdmaptools.utilities.background_estimators import masked_bruckner_image_background

from tiled.client import from_profile

c = from_profile('srx')


def process_map(scan_id,
                base_wd,
                dark_field,
                poni_file,
                swapped_axes=False):

    if not isinstance(scan_id, XRDMap):
        if os.path.exists(f'{base_wd}xrdmaps/scan{scan_id}_xrdmap.h5'):
            print('File found! Loading from HDF...')
            xdm = XRDMap.from_hdf(f'scan{scan_id}_xrdmap.h5', wd=f'{base_wd}xrdmaps/', image_data_key='raw', swapped_axes=swapped_axes)
        else:
            print('Loading data from server...')
            xdm = XRDMap.from_db(scan_id, wd=f'{base_wd}xrdmaps/', swapped_axes=swapped_axes)
    else:
        xdm = scan_id
    
    if xdm.title == 'raw':
        # Basic corrections
        xdm.correct_dark_field(dark_field)
        xdm.correct_air_scatter(xdm.med_image, applied_corrections=xdm.corrections)
        xdm.correct_scaler_energies(scaler_key='i0')
        xdm.convert_scalers_to_flux(scaler_key='i0')
        xdm.correct_scaler_energies(scaler_key='im')
        xdm.convert_scalers_to_flux(scaler_key='im')
        xdm.normalize_scaler()
        xdm.correct_outliers(tolerance=10)

        # Geometric corrections
        xdm.set_calibration(poni_file, wd=f'{base_wd}calibrations/')
        xdm.apply_polarization_correction()
        xdm.apply_solidangle_correction()

        # Background correction
        xdm.estimate_background(method='bruckner',
                                binning=8,
                                min_prominence=0.1)
        
        # Rescale and saving
        xdm.rescale_images(arr_max=xdm.estimate_saturated_pixel())
        xdm.finalize_images()

    # Integrate map
    xdm.integrate1D_map()

    # Find blobs
    xdm.find_blobs(threshold_method='minimum',
                   multiplier=5,
                   size=3,
                   expansion=10)
    
    # Vectorize blobs
    xdm.vectorize_map_data()

    # Convert to 1D integrations
    tth, intensity = xdm.integrate1D_image(xdm.max_image)
    q = tth_2_q(tth, wavelength=xdm.wavelength)

    np.savetxt(f'{base_wd}/max_1D_integrations/scan{xdm.scan_id}_max_1D_integration.txt',
               np.asarray([q, tth, intensity]))
    fig, ax = xdm.plot_integration(intensity, tth=tth, title='Max Integration', return_plot=True)
    fig.savefig(f'{base_wd}max_1D_integrations/scan{xdm.scan_id}_max_integration.png')
    plt.close('all')

    io.imsave(f'{base_wd}integrated_maps/scan{xdm.scan_id}_max_map.tif', xdm.max_map)

    images = xdm.images.copy()
    xdm.dump_images()
    images[~xdm.blob_masks] = 0
    blob_sum_map = np.sum(images, axis=(2, 3))

    io.imsave(f'{base_wd}integrated_maps/scan{xdm.scan_id}_blob_sum.tif', blob_sum_map)

    return xdm


def process_rsm(scan_id,
                base_wd,
                dark_field,
                poni_file):

    if not isinstance(scan_id, XRDRockingCurve):
        if os.path.exists(f'{base_wd}rocking_curves/scan{scan_id}_rsm.h5'):
            print('File found! Loading from HDF...')
            rsm = XRDRockingCurve.from_hdf(f'scan{scan_id}_rsm.h5', wd=f'{base_wd}rocking_curves/', image_data_key='raw')
        else:
            print('Loading data from server...')
            rsm = XRDRockingCurve.from_db(scan_id, wd=f'{base_wd}rocking_curves/')
    else:
        rsm = scan_id

    if rsm.title == 'raw':
        # Basic corrections
        rsm.correct_dark_field(dark_field)
        rsm.correct_air_scatter(rsm.med_image, applied_corrections=rsm.corrections) # Using the median image does not make as much sense here...
        rsm.correct_scaler_energies(scaler_key='i0')
        rsm.convert_scalers_to_flux(scaler_key='i0')
        rsm.correct_scaler_energies(scaler_key='im')
        rsm.convert_scalers_to_flux(scaler_key='im')
        rsm.normalize_scaler()
        rsm.correct_outliers(tolerance=10)

        # Geometric corrections
        rsm.set_calibration(poni_file, wd=f'{base_wd}calibrations/')
        rsm.apply_polarization_correction()
        rsm.apply_solidangle_correction()

        # Background correction
        rsm.estimate_background(method='bruckner',
                                binning=8,
                                min_prominence=0.1)
        
        # Rescale and saving
        rsm.rescale_images() # No estimating arr_max. It already exists in the dataset!
        rsm.finalize_images()

    # Find blobs
    rsm.find_2D_blobs(threshold_method='minimum',
                      multiplier=5,
                      size=3,
                      expansion=10)
    
    # Vectorize blobs
    rsm.vectorize_images()

    # Find 3D spots, no indexing yet...
    rsm.find_3D_spots(nn_dist=0.05, int_cutoff=0.01, relative_cutoff=True)

    return rsm


# Batch processing xrdmaps
def batch_processing(start_id, stop_id=None):

    base_wd = '/nsls2/data/srx/proposals/2025-2/pass-316224/'
    swapped_axes = False

    # Start scan_id
    scan_id = start_id
    while True:

        # STOP
        if stop_id is not None and scan_id >= stop_id:
            print(f'Batch processing reached stop limit of {stop_id}!')
            break

        print(f'Checking for scan {scan_id}...')
        try:
            # Wait until scan is finished
            if c[-1].start['scan_id'] == scan_id:
                if (not hasattr(c[scan_id], 'stop')
                     or c[scan_id].stop is None
                     or 'time' not in c[scan_id].stop):
                    print(f'Scan {scan_id} has yet to finish. Waiting 5 min...')
                    ttime.sleep(300)
                    continue       
            
            # Make sure it is XRF_FLY
            if c[scan_id].start['scan']['type'] != 'XRF_FLY':
                print(f'Scan {scan_id} is not type XRF_FLY. Waiting 1 min and trying next scan ID.')
                ttime.sleep(60)
                scan_id += 1
                continue
                # Check for rocking curves???
            
            # Check to see if the scan uses the dexela
            if 'dexela' not in c[scan_id].start['scan']['detectors']:
                print(f'Scan {scan_id} did not use the dexela. Waiting and trying next scan ID.')
                ttime.sleep(300)
                scan_id += 1
                continue

            # Check if scan is dark-field
            if c[scan_id].start['scan']['shape'] == [5, 5]:
                print(f'Scan {scan_id} is probably a dark-field. Generating and saving composite pattern.')
                save_composite_pattern(scan_id, broker='manual', wd=f'{base_wd}dark/', method='median')
                ttime.sleep(10)
                scan_id += 1
                continue
            
            # Check for already processed scans. Kind of
            if os.path.exists(f'{base_wd}xrdmaps/scan{scan_id}_xrdmap.h5'):
                with h5py.File(f'{base_wd}xrdmaps/scan{scan_id}_xrdmap.h5', 'r') as f:
                    PROCESSED = False
                    if 'final_images' in f['xrdmap/image_data']:
                        PROCESSED = True
                if PROCESSED:
                    print(f'Scan {scan_id} already processed. Moving to next.')
                    ttime.sleep(10)
                    scan_id += 1
                    continue
                else:
                    print(f'XRDMap for scan {scan_id} already generated, but not fully processed.')
                    print('Finishing processing...')
                    pass

            print(f'Processing scan {scan_id}...')

            # Find most recent dark field prior to this scan ID
            dark_list = os.listdir(f'{base_wd}dark/')
            dark_scan_ids = [int(name[4:10]) for name in dark_list]
            dark_diff_list = scan_id - np.array(dark_scan_ids, dtype=float)
            dark_ind = np.nonzero(dark_diff_list > 0)[0][np.argmin(dark_diff_list[dark_diff_list > 0])]
            dark_field = io.imread(f'{base_wd}dark/{dark_list[dark_ind]}')

            # Find most recent poni_file prior to this scan ID
            poni_list = [file for file in os.listdir(f'{base_wd}calibrations/') if file[-4:] == 'poni']
            poni_scan_ids = [int(name[4:10]) for name in poni_list if name[-4:] == 'poni']
            poni_diff_list = scan_id - np.array(poni_scan_ids, dtype=float)
            poni_ind = np.nonzero(poni_diff_list > 0)[0][np.argmin(poni_diff_list[poni_diff_list > 0])]
            poni_file = poni_list[poni_ind]

            xdm = process_map(scan_id,
                              base_wd,
                              dark_field,
                              poni_file,
                              swapped_axes=swapped_axes)

            print('Waiting 1 min to check for next scan...')
            ttime.sleep(60)
            scan_id += 1
        
        except KeyboardInterrupt:
            break
        
        except Exception as e:
            print(f'Batching processing failed for scan {scan_id}.\n{e}')
            print('Waiting 5 sec to check for next scan...')
            ttime.sleep(5)
            scan_id += 1


def fix_integrated_maps():
    base_wd = '/nsls2/data/srx/proposals/2025-2/pass-316224/'
    from xrdmaptools.utilities.utilities import get_int_vector_map, get_max_vector_map

    map_list = os.listdir(f'{base_wd}xrdmaps/')
    map_scan_ids = [int(name[4:10]) for name in map_list]

    for scan_id in np.unique(map_scan_ids):
        xdm = None

        if f'scan{scan_id}_max_map.tif' not in map_list:
            xdm = XRDMap.from_hdf(f'scan{scan_id}_xrdmap.h5',
                                  wd=f'{base_wd}xrdmaps/',
                                  image_data_key=None,
                                  integration_data_key=None,
                                  load_blob_masks=None,
                                  load_vector_map=True)
            
            if hasattr(xdm, 'vector_map'):
                max_map = get_max_vector_map(xdm.vector_map).astype(np.float32)
                io.imsave(f'{base_wd}integrated_maps/scan{xdm.scan_id}_max_map.tif', max_map)

        if f'scan{scan_id}_blob_sum_map.tif' not in map_list:
            if xdm is None:
                xdm = XRDMap.from_hdf(f'scan{scan_id}_xrdmap.h5',
                                      wd=f'{base_wd}xrdmaps/',
                                      image_data_key=None,
                                      integration_data_key=None,
                                      load_blob_masks=None,
                                      load_vector_map=True)
            
            if hasattr(xdm, 'vector_map'):
                blob_sum_map = get_int_vector_map(xdm.vector_map).astype(np.float32)
                io.imsave(f'{base_wd}integrated_maps/scan{xdm.scan_id}_blob_sum.tif', blob_sum_map)