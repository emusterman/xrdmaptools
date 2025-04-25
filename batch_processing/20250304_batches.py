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

# from pyxrf.api import *

from tiled.client import from_profile

c = from_profile('srx')


def reprocess_xdms():

    base_wd = '/nsls2/data/srx/proposals/2024-1/pass-314118/processed_xrdmaps/'

    xdms = XRDMapStack.from_hdf('scan153253-153297_xrdmapstack.h5', wd=base_wd, load_xdms_vector_map=False)

    for xdm in timed_iter(xdms):
        print(f'Processing scan {xdm.scan_id}...')

        xdm.load_images_from_hdf(image_data_key='raw')
        xdm.load_images_from_hdf(image_data_key='dark_field')
        
        # Basic correction.
        xdm.correct_dark_field()
        xdm.correct_scaler_energies(scaler_key='i0')
        xdm.correct_scaler_energies(scaler_key='im')
        xdm.normalize_scaler()
        xdm.correct_outliers(tolerance=10)

        # Geometric corrections
        xdm.apply_polarization_correction()
        xdm.apply_solidangle_correction()

        # Background correction
        xdm.estimate_background(method='bruckner',
                                binning=4,
                                min_prominence=0.1)

        # Rescale and saving
        xdm.rescale_images(arr_max=xdm.estimate_saturated_pixel())
        xdm.finalize_images()

        # Find blobs and spots while were at it
        xdm.find_blobs(threshold_method='minimum',
                          multiplier=3,
                          size=3,
                          expansion=10)
        
        # Vectorize data and then dump it
        xdm.vectorize_map_data(rewrite_data=True)
        xdm.dump_images()
        del xdm.blob_masks
        xdm.blob_masks = None


def re_reprocess_xdms():

    base_wd = '/nsls2/data/srx/proposals/2024-1/pass-314118/processed_xrdmaps/'

    xdms = XRDMapStack.from_hdf('scan153253-153297_xrdmapstack.h5', wd=base_wd, load_xdms_vector_map=False)

    for xdm in timed_iter(xdms):
        print(f'Processing scan {xdm.scan_id}...')

        if not np.any(xdm.null_map):
            print(f'Null map empty for scan {xdm.scan_id}. Proceeding to next...')
            continue
        
        # xdm.load_images_from_hdf() # Should be final images!
        # xdm.load_images_from_hdf('blob_masks')
        
        # # Nullify loaded values. Is this necessary??
        # for attr in ['images', 'blob_masks', 'integrations']:
        #     if hasattr(xdm, attr) and getattr(xdm, attr) is not None:
        #         # This should all be done in place
        #         getattr(xdm, attr)[xdm.null_map] = 0
        
        # if hasattr(xdm, 'vector_map'):
        #     xdm.vector_map[xdm.null_map] = np.empty((0, 4), dtype=np.float32)

        xdm.open_hdf()
        

        for indices in xdm.indices:
            if not xdm.null_map[indices]:
                continue
            
            xdm.hdf['xrdmap/image_data/final_images'][indices] = 0
            xdm.hdf['xrdmap/image_data/_blob_masks'][indices] = 0

            # e.g., '1,2'
            title = ','.join([str(ind) for ind in indices])

            del xdm.hdf['xrdmap/vectorized_map'][title]
            xdm.hdf['xrdmap/vectorized_map'].require_dataset(
                title,
                data=np.empty((0, 4), dtype=np.float32),
                shape=(0, 4),
                dtype=np.float32)
        
        xdm.close_hdf()



def reprocess_xdms_2():

    base_wd = '/nsls2/data/srx/proposals/2024-1/pass-314118/processed_xrdmaps/'

    xdms = XRDMapStack.from_hdf('scan153102-153145_xrdmapstack.h5', wd=base_wd, load_xdms_vector_map=False)

    for xdm in timed_iter(xdms):
        print(f'Processing scan {xdm.scan_id}...')

        xdm.load_images_from_hdf(image_data_key='raw')
        xdm.load_images_from_hdf(image_data_key='dark_field')
        
        # Basic correction.
        xdm.correct_dark_field()
        xdm.correct_scaler_energies(scaler_key='i0')
        xdm.correct_scaler_energies(scaler_key='im')
        xdm.normalize_scaler()
        xdm.correct_outliers(tolerance=10)

        # Geometric corrections
        xdm.apply_polarization_correction()
        xdm.apply_solidangle_correction()

        # Background correction
        xdm.estimate_background(method='bruckner',
                                binning=4,
                                min_prominence=0.1)

        # Rescale and saving
        xdm.rescale_images(arr_max=xdm.estimate_saturated_pixel())
        xdm.finalize_images()

        # Find blobs
        xdm.find_blobs(threshold_method='minimum',
                       multiplier=3,
                       size=3,
                       expansion=10)
        
        # Vectorize data and then dump it
        xdm.vectorize_map_data(rewrite_data=True)
        xdm.dump_images()
        del xdm.blob_masks
        xdm.blob_masks = None


def reprocess_xdms_3():

    base_wd = '/nsls2/data/srx/proposals/2024-1/pass-314118/processed_xrdmaps/'

    xdms = XRDMapStack.from_hdf('scan153102-153145_xrdmapstack.h5', wd=base_wd, load_xdms_vector_map=False)

    for i, xdm in timed_iter(enumerate(xdms), total=len(xdms)):
        print(f'Processing scan {xdm.scan_id}...')
        print(f'Processing index {i}...')

        if i < 20:
            continue

        xdm.load_images_from_hdf(image_data_key='raw')
        xdm.load_images_from_hdf(image_data_key='dark_field')
        
        # Basic corrections
        xdm.correct_dark_field()
        xdm.correct_scaler_energies(scaler_key='i0')
        xdm.correct_scaler_energies(scaler_key='im')

        air_scatter = estimate_map_air_scatter(xdm)
        xdm.correct_air_scatter(air_scatter,
                                applied_corrections=xdm.corrections)
        del air_scatter

        xdm.normalize_scaler()
        xdm.correct_outliers(tolerance=10)

        # Geometric corrections
        xdm.apply_polarization_correction()
        xdm.apply_solidangle_correction()

        # Background corrections
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
        
        # Vectorize data and then dump it
        xdm.vectorize_map_data(rewrite_data=True)
        xdm.dump_images()
        del xdm.blob_masks
        xdm.blob_masks = None
        del xdm.vector_map





def estimate_map_air_scatter(xdm):
    
    air_scatter = np.zeros_like(xdm.images)

    def inverse_function(x, m, p, I0):
        return I0 / (m * x**p)

    print('Estimating air scatter per pixel...')
    for indices in tqdm(xdm.indices):
        # Get image
        img = xdm.images[indices]
        # Integrate to 1D
        tth, intensity = xdm.integrate1D_image(img)
        # Strip peaks
        bkg = masked_bruckner_image_background(intensity.reshape(-1, 1),
                                               size=20,
                                               iterations=5000,
                                               mask=np.ones_like(intensity, dtype=np.bool_).reshape(-1, 1),
                                               min_prominence=1e-3).squeeze()
        # Fit scattering curve
        popt, pcov = curve_fit(inverse_function, tth, bkg, p0=[1, 1, 1e3])

        # Extrapole scatter curve into 2D
        air_scatter[indices] = inverse_function(xdm.tth_arr, *popt)
    
    return air_scatter


def process_map(scan_id, wd):

    xdm = XRDMap.from_hdf(f'scan{scan_id}_xrdmap.h5', wd=f'{wd}xrdmaps/', image_data_key='raw')

    # Fix position transpose issues
    for key in xdm.pos_dict.keys():
        if xdm.pos_dict[key].shape != xdm.map_shape:
            xdm.pos_dict[key] = xdm.pos_dict[key].swapaxes(0, 1)
    xdm.set_positions(xdm.pos_dict)

    # Fix sclr transpose issues
    for key in xdm.sclr_dict.keys():
        if xdm.sclr_dict[key].shape != xdm.map_shape:
            xdm.sclr_dict[key] = xdm.sclr_dict[key].swapaxes(0, 1)
    xdm.set_scalers(xdm.sclr_dict)

    if hasattr(xdm, 'spots'):
        del xdm.spots
        xdm.open_hdf()
        del xdm.hdf['xrdmap/reflections']
        xdm.close_hdf()
    
    # Basic corrections
    xdm.correct_dark_field()
    xdm.correct_scaler_energies(scaler_key='i0')
    xdm.correct_scaler_energies(scaler_key='im')
    xdm.normalize_scaler()
    xdm.correct_outliers(tolerance=10)

    # Geometric corrections
    xdm.apply_polarization_correction()
    xdm.apply_solidangle_correction()

    # Background correction
    xdm.estimate_background(method='bruckner',
                            binning=4,
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


def reprocess1():

    base_wd = '/nsls2/data/srx/proposals/2024-3/pass-316224/'

    scanlist = np.arange(162081, 162137 + 1)

    for i, scan_id in timed_iter(enumerate(scanlist), total=len(scanlist)):
        
        if c[scan_id].start['scan']['type'] != 'XRF_FLY':
            continue

        print(f'Processing scan {scan_id}...')
        print(f'Processing index {i}...')

        xdm = process_map(scan_id,
                          base_wd)

        max_map = xdm.max_map
        images = xdm.images.copy()
        xdm.dump_images()
        images[~xdm.blob_masks] = 0
        # blob_max = np.max(images, axis=(2, 3))
        blob_sum = np.sum(images, axis=(2, 3))

        io.imsave(f'{base_wd}integrated_maps/scan{xdm.scan_id}_max_map.tif', max_map)
        # io.imsave(f'{base_wd}integrated_maps/scan{xdm.scan_id}_blob_max.tif', blob_max)
        io.imsave(f'{base_wd}integrated_maps/scan{xdm.scan_id}_blob_sum.tif', blob_sum)
    

def reprocess2():

    base_wd = '/nsls2/data/srx/proposals/2024-3/pass-316224/'

    scanlist = np.arange(162161, 162225 + 1)

    for i, scan_id in timed_iter(enumerate(scanlist), total=len(scanlist)):
        
        if c[scan_id].start['scan']['type'] != 'XRF_FLY':
            continue

        print(f'Processing scan {scan_id}...')
        print(f'Processing index {i}...')

        xdm = process_map(scan_id,
                          base_wd)

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

    base_wd = '/nsls2/data/srx/proposals/2024-3/pass-316224/'

    hdf_filenames = []
    for scan_id in np.arange(162073, 162137 + 1):
        hdf_filenames.extend([fname for fname in os.listdir(f'{base_wd}xrdmaps/')
                              if str(scan_id) in fname and 'stack' not in fname])

    # return hdf_filenames

    xdms = XRDMapStack.from_XRDMap_hdfs(hdf_filenames, wd=f'{base_wd}xrdmaps/')

    return xdms


def setup_batch2_xdms():

    base_wd = '/nsls2/data/srx/proposals/2024-3/pass-316224/'

    hdf_filenames = []
    for scan_id in np.arange(162161, 162224 + 1):
        hdf_filenames.extend([fname for fname in os.listdir(f'{base_wd}xrdmaps/')
                              if str(scan_id) in fname and 'stack' not in fname])

    # return hdf_filenames

    xdms = XRDMapStack.from_XRDMap_hdfs(hdf_filenames, wd=f'{base_wd}xrdmaps/')

    return xdms