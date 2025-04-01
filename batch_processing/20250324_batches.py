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
from xrdmaptools.reflections.spot_blob_indexing_3D import *
from xrdmaptools.reflections.spot_blob_indexing_3D import _get_connection_indices
from xrdmaptools.crystal.strain import *

from xrdmaptools.utilities.utilities import (
    timed_iter,
    pathify
)
from xrdmaptools.utilities.math import compute_r_squared
from xrdmaptools.reflections.SpotModels import (
    GaussianFunctions,
    generate_bounds
)
from xrdmaptools.utilities.background_estimators import masked_bruckner_image_background

from tiled.client import from_profile

c = from_profile('srx')


def process_map(scan_id, wd, dark_field, poni_file, swapped_axes=False):

    if os.path.exists(f'{wd}xrdmaps/scan{scan_id}_xrdmap.h5'):
        print('File found! Loading from HDF...')
        xdm = XRDMap.from_hdf(f'scan{scan_id}_xrdmap.h5', wd=f'{wd}xrdmaps/', image_data_key='raw', swapped_axes=swapped_axes)
    else:
        print('Loading data from server...')
        xdm = XRDMap.from_db(scan_id, wd=f'{wd}xrdmaps/', swapped_axes=swapped_axes)
    
    # Basic corrections
    xdm.correct_dark_field(dark_field)
    xdm.correct_scaler_energies(scaler_key='i0')
    xdm.correct_scaler_energies(scaler_key='im')
    xdm.normalize_scaler()
    xdm.correct_outliers(tolerance=10)

    # Geometric corrections
    xdm.set_calibration(poni_file, wd=f'{wd}calibrations/')
    xdm.apply_polarization_correction()
    xdm.apply_solidangle_correction()

    # Background correction
    xdm.estimate_background(method='bruckner',
                            binning=8,
                            min_prominence=0.1)
    
    # Rescale and saving
    xdm.rescale_images(arr_max=xdm.estimate_saturated_pixel())
    xdm.finalize_images()

    # Find blobs
    xdm.find_blobs(threshold_method='minimum',
                   multiplier=5,
                   size=3,
                   expansion=10)
        
    # Vectorize data
    xdm.vectorize_map_data(rewrite_data=True)

    return xdm



def batch1():

    base_wd = '/nsls2/data/srx/proposals/2025-1/pass-316224/'

    dark_field = io.imread(f'{base_wd}dark/scan166787_dexela_median_composite.tif')
    poni_file = 'scan166760_dexela_calibration.poni'

    # scanlist = np.arange(166798, 166871)
    scanlist = np.arange(166871, 166883 + 1)

    for i, scan_id in timed_iter(enumerate(scanlist), total=len(scanlist)):
        
        if c[scan_id].start['scan']['type'] != 'XRF_FLY':
            continue

        print(f'Processing scan {scan_id}...')
        print(f'Processing index {i}...')

        # if i < 20:
            # continue

        xdm = process_map(scan_id,
                          base_wd,
                          dark_field,
                          poni_file,
                          swapped_axes=True)

        max_map = xdm.max_map
        images = xdm.images.copy()
        xdm.dump_images()
        images[~xdm.blob_masks] = 0
        # blob_max = np.max(images, axis=(2, 3))
        blob_sum = np.sum(images, axis=(2, 3))

        io.imsave(f'{base_wd}integrated_maps/scan{xdm.scan_id}_max_map.tif', max_map)
        # io.imsave(f'{base_wd}integrated_maps/scan{xdm.scan_id}_blob_max.tif', blob_max)
        io.imsave(f'{base_wd}integrated_maps/scan{xdm.scan_id}_blob_sum.tif', blob_sum)


def batch2():

    base_wd = '/nsls2/data/srx/proposals/2025-1/pass-316224/'

    dark_field = io.imread(f'{base_wd}dark/scan166952_dexela_median_composite.tif')
    poni_file = 'scan166760_dexela_calibration.poni'

    scan_id = 167024
    while True:
        print(f'Checking for scan {scan_id}...')
        try:
            if scan_id > 167073:
                print('All done!!!')
                break

            # Wait until scan is finished
            if c[-1].start['scan_id'] == scan_id:
                print(f'Scan {scan_id} has yet to finish. Waiting 5 min...')
                ttime.sleep(600)
                continue
            
            # Make sure it is XRF_FLY
            if c[scan_id].start['scan']['type'] != 'XRF_FLY':
                ttime.sleep(120)
                scan_id += 1
                continue
            
            # Check for already processed scans. Kind of
            if os.path.exists(f'{base_wd}xrdmaps/scan{scan_id}_xrdmap.h5'):
                print(f'Scan {scan_id} already processed. Moving to next.')
                ttime.sleep(10)
                scan_id += 1
                continue

            print(f'Processing scan {scan_id}...')
            xdm = process_map(scan_id,
                              base_wd,
                              dark_field,
                              poni_file,
                              swapped_axes=True)
            
            max_map = xdm.max_map
            images = xdm.images.copy()
            xdm.dump_images()
            images[~xdm.blob_masks] = 0
            blob_sum = np.sum(images, axis=(2, 3))

            io.imsave(f'{base_wd}integrated_maps/scan{xdm.scan_id}_max_map.tif', max_map)
            io.imsave(f'{base_wd}integrated_maps/scan{xdm.scan_id}_blob_sum.tif', blob_sum)

            print('Waiting 1 min to check for next scan...')
            ttime.sleep(60)
            scan_id += 1
        
        except KeyboardInterrupt:
            break
        
        except Exception as e:
            print(f'Batching processing failed for scan {scan_id}.\n{e}')
            print('Waiting 5 min to check for next scan...')
            ttime.sleep(600)
            scan_id += 1
        
        
def get_alignment_maps():
    
    base_wd = '/nsls2/data/srx/proposals/2025-1/pass-316224/'

    scanlist = np.arange(166799, 166871)

    for i, scan_id in timed_iter(enumerate(scanlist), total=len(scanlist)):

        if not os.path.exists(f'{base_wd}xrdmaps/scan{scan_id}_xrdmap.h5'):
            continue

        xdm = XRDMap.from_hdf(f'scan{scan_id}_xrdmap.h5', wd=f'{base_wd}xrdmaps/')
        
        max_map = xdm.max_map
        images = xdm.images.copy()
        xdm.dump_images()
        images[~xdm.blob_masks] = 0
        # blob_max = np.max(images, axis=(2, 3))
        blob_sum = np.sum(images, axis=(2, 3))

        io.imsave(f'{base_wd}integrated_maps/scan{xdm.scan_id}_max_map.tif', max_map)
        # io.imsave(f'{base_wd}integrated_maps/scan{xdm.scan_id}_blob_max.tif', blob_max)
        io.imsave(f'{base_wd}integrated_maps/scan{xdm.scan_id}_blob_sum.tif', blob_sum)



def setup_batch1_xdms():

    hdf_filenames = []
    for scan_id in np.arange(166795, 166883 + 1):
        hdf_filenames.extend([fname for fname in os.listdir(f'{base_wd}xrdmaps/')
                              if str(scan_id) in fname and 'stack' not in fname])

    # return hdf_filenames

    xdms = XRDMapStack.from_XRDMap_hdfs(hdf_filenames, wd=f'{base_wd}xrdmaps/')

    return xdms


def setup_batch2_xdms():

    hdf_filenames = []
    for scan_id in np.arange(166954, 167073 + 1):
        hdf_filenames.extend([fname for fname in os.listdir(f'{base_wd}xrdmaps/')
                              if str(scan_id) in fname and 'stack' not in fname])

    xdms = XRDMapStack.from_XRDMap_hdfs(hdf_filenames, wd=f'{base_wd}xrdmaps/')

    for xdm in xdms:
        xdm.load_vector_map()
        xdm.vector_map = xdm.vector_map.swapaxes(0, 1)
        xdm.save_vector_map(rewrite_data=True)

    return xdms


def setup_batch3_xdms():

    hdf_filenames = []
    for scan_id in np.arange(167103, 167175 + 1):
        hdf_filenames.extend([fname for fname in os.listdir(f'{base_wd}xrdmaps/')
                              if str(scan_id) in fname and 'stack' not in fname])

    xdms = XRDMapStack.from_XRDMap_hdfs(hdf_filenames, wd=f'{base_wd}xrdmaps/')

    for xdm in xdms:
        xdm.swap_axes()
        # xdm.load_vector_map()
        # if xdm.vector_map != xdm.map_shape:
        #     xdm.save_vector_map(rewrite_data=True)

    return xdms


def batch3():

    base_wd = '/nsls2/data/srx/proposals/2025-1/pass-316224/'

    dark_field = io.imread(f'{base_wd}dark/scan166952_dexela_median_composite.tif')
    poni_file = 'scan166760_dexela_calibration.poni'

    scan_id = 167103
    while True:
        print(f'Checking for scan {scan_id}...')
        try:
            # Wait until scan is finished
            if c[-1].start['scan_id'] == scan_id:
                print(f'Scan {scan_id} has yet to finish. Waiting 5 min...')
                ttime.sleep(600)
                continue
            
            # Make sure it is XRF_FLY
            if c[scan_id].start['scan']['type'] != 'XRF_FLY':
                ttime.sleep(120)
                scan_id += 1
                continue
            
            # Check for already processed scans. Kind of
            if os.path.exists(f'{base_wd}xrdmaps/scan{scan_id}_xrdmap.h5'):
                print(f'Scan {scan_id} already processed. Moving to next.')
                ttime.sleep(10)
                scan_id += 1
                continue

            print(f'Processing scan {scan_id}...')
            xdm = process_map(scan_id,
                              base_wd,
                              dark_field,
                              poni_file,
                              swapped_axes=True)
            
            max_map = xdm.max_map
            images = xdm.images.copy()
            xdm.dump_images()
            images[~xdm.blob_masks] = 0
            blob_sum = np.sum(images, axis=(2, 3))

            io.imsave(f'{base_wd}integrated_maps/scan{xdm.scan_id}_max_map.tif', max_map)
            io.imsave(f'{base_wd}integrated_maps/scan{xdm.scan_id}_blob_sum.tif', blob_sum)

            print('Waiting 1 min to check for next scan...')
            ttime.sleep(60)
            scan_id += 1
        
        except KeyboardInterrupt:
            break
        
        except Exception as e:
            print(f'Batching processing failed for scan {scan_id}.\n{e}')
            print('Waiting 5 min to check for next scan...')
            ttime.sleep(600)
            scan_id += 1
        

def batch4():

    base_wd = '/nsls2/data/srx/proposals/2025-1/pass-316224/'

    dark_field = io.imread(f'{base_wd}dark/scan167196_dexela_median_composite.tif')
    poni_file = 'scan166760_dexela_calibration.poni'

    scan_id = 167198
    while True:
        print(f'Checking for scan {scan_id}...')
        try:
            # Wait until scan is finished
            if c[-1].start['scan_id'] == scan_id:
                print(f'Scan {scan_id} has yet to finish. Waiting 5 min...')
                ttime.sleep(600)
                continue
            
            # Make sure it is XRF_FLY
            if c[scan_id].start['scan']['type'] != 'XRF_FLY':
                ttime.sleep(120)
                scan_id += 1
                continue
            
            # Check for already processed scans. Kind of
            if os.path.exists(f'{base_wd}xrdmaps/scan{scan_id}_xrdmap.h5'):
                print(f'Scan {scan_id} already processed. Moving to next.')
                ttime.sleep(10)
                scan_id += 1
                continue

            print(f'Processing scan {scan_id}...')
            xdm = process_map(scan_id,
                              base_wd,
                              dark_field,
                              poni_file,
                              swapped_axes=False)
            
            max_map = xdm.max_map
            xdm.images[~xdm.blob_masks] = 0
            blob_sum = np.sum(xdm.images, axis=(2, 3))
            xdm.dump_images()

            io.imsave(f'{base_wd}integrated_maps/scan{xdm.scan_id}_max_map.tif', max_map)
            io.imsave(f'{base_wd}integrated_maps/scan{xdm.scan_id}_blob_sum.tif', blob_sum)

            print('Waiting 1 min to check for next scan...')
            ttime.sleep(60)
            scan_id += 1
        
        except KeyboardInterrupt:
            break
        
        except Exception as e:
            print(f'Batching processing failed for scan {scan_id}.\n{e}')
            print('Waiting 5 min to check for next scan...')
            ttime.sleep(600)
            scan_id += 1