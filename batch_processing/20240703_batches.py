import numpy as np
import os
from tqdm import tqdm
from skimage import io

# Local imports
from xrdmaptools.XRDMap import XRDMap
from xrdmaptools.io.db_io import (
    save_full_scan,
    save_xrd_tifs,
    save_map_parameters,
    save_scan_md,
    save_composite_pattern,
    save_calibration_pattern,
    save_energy_rc_data,
    save_angle_rc_data
)
from xrdmaptools.io.db_utils import make_xrdmap_hdf
from xrdmaptools.reflections.spot_blob_search import spot_search
from xrdmaptools.crystal.Phase import phase_selector
from xrdmaptools.utilities.math import (
    energy_2_wavelength,
    wavelength_2_energy,
    tth_2_d,
    d_2_tth,
    convert_qd,
    q_2_tth,
    tth_2_q
)
from xrdmaptools.utilities.utilities import (
    label_nearest_spots,
    arbitrary_center_of_mass,
    vector_angle,
    timed_iter
)

# Basic batch functioning for processing xrdmaps at single energy
def xmt_batch1():

    base_wd = ''
    scanlist = [

    ]

    dark_field = io.imread()
    flat_field = io.imread()
    poni_file = ''

    for scan in timed_iter(scanlist):
        
        # Load map and set calibration
        xrdmap = XRDMap.from_hdf(f'scan{scan}_xrd.h5', wd=f'{base_wd}', save_hdf=True)
        xrdmap.set_calibration(poni_file, filedir=xrdmap.wd)
        
        # Basic correction. No outliers
        xrdmap.map.correct_dark_field(dark_field=dark_field)
        xrdmap.map.correct_flat_field(flat_field=flat_field)
        xrdmap.map.normalize_scaler() # Assumed information in sclr_dict
        #xrdmap.map.correct_outliers() # Too slow!

        # Geometric corrections
        xrdmap.map.apply_polarization_correction()
        xrdmap.map.apply_solidangle_correction()
        xrdmap.map.images *= np.radians(xrdmap.tth_arr)
        xrdmap.corrections['lorentz'] = True

        # Background correction
        xrdmap.map.estimate_background(method='bruckner', binning=4, min_prominence=0.1)
        xrdmap.map.remove_background()

        # Rescale and saving
        xrdmap.map.rescale_images(upper=100, lower=0, arr_min=0)
        xrdmap.map.finalize_images()

        # Find blobs and spots while were at it
        xrdmap.find_spots(threshold_method='minimum',
                          multiplier=5, size=3,
                          radius=5, expansion=5)


    