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
