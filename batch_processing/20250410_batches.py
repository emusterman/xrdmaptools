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
        xdm = XRDMap.from_hdf(f'scan{scan_id}_xrdmap.h5', wd=wd, image_data_key='raw', swapped_axes=swapped_axes)
    else:
        print('Loading data from server...')
        xdm = XRDMap.from_db(scan_id, wd=wd, swapped_axes=swapped_axes)
    
    # Basic corrections
    xdm.correct_dark_field(dark_field)
    xdm.normalize_scaler()
    xdm.correct_outliers(tolerance=10)

    # Geometric corrections
    xdm.set_calibration(poni_file, wd=wd)
    xdm.apply_polarization_correction()
    xdm.apply_solidangle_correction()

    # Background correction
    xdm.estimate_background(method='bruckner',
                            binning=4,
                            min_prominence=0.1)
    
    # Rescale and saving
    xdm.rescale_images()
    xdm.finalize_images()

    # Integrate map
    xdm.integrate1d_map()

    # Find blobs
    xdm.find_blobs(threshold_method='minimum',
                   multiplier=5,
                   size=3,
                   expansion=10)

    return xdm



def batch1():

    base_wd = '/nsls2/data/srx/proposals/2025-1/pass-316535/'

    dark_field = io.imread(f'{base_wd}scan167615_dexela_median_composite.tif')
    poni_file = 'scan167612_dexela_calibration.poni'

    scanlist = np.arange(167613, 167629 + 1)

    for i, scan_id in timed_iter(enumerate(scanlist), total=len(scanlist)):
        
        if c[scan_id].start['scan']['type'] != 'XRF_FLY':
            continue
        
        if c[scan_id].stop['exit_status'] != 'success':
            continue

        print(f'Processing scan {scan_id}...')
        print(f'Processing index {i}...')

        xdm = process_map(scan_id,
                          base_wd,
                          dark_field,
                          poni_file,
                          swapped_axes=True)

        fig, ax = xdm.plot_map(xdm.max_map, title='Max Map', return_plot=True)
        fig.savefig(f'{base_wd}figures/scan{xdm.scan_id}_max_map.png')

        fig, ax = xdm.plot_image(xdm.max_image, vmax=1, title='Max Image', return_plot=True)
        fig.savefig(f'{base_wd}figures/scan{xdm.scan_id}_max_image.png')

        tth, intensity = xdm.integrate1d_image(xdm.max_image)
        fig, ax = xdm.plot_integration(intensity, tth=tth, title='Max Integration', return_plot=True)
        fig.savefig(f'{base_wd}figures/scan{xdm.scan_id}_max_integration.png')

        images = xdm.images.copy()
        xdm.dump_images()
        images[~xdm.blob_masks] = 0
        blob_sum_map = np.sum(images, axis=(2, 3))

        fig, ax = xdm.plot_map(blob_sum_map, title='Blob Sum Map', return_plot=True)
        fig.savefig(f'{base_wd}figures/scan{xdm.scan_id}_blob_sum_map.png')

        blob_sum_image = np.sum(images, axis=(0, 1))

        tth, intensity = xdm.integrate1d_image(blob_sum_image)
        fig, ax = xdm.plot_integration(intensity, tth=tth, title='Blob Sum Integration', return_plot=True)
        fig.savefig(f'{base_wd}figures/scan{xdm.scan_id}_blob_sum_integration.png')

