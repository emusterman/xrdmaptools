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

    base_wd = '/nsls2/data/srx/proposals/2024-1/pass-314118/'
    scanlist = [
        #153102,
        #153104,
        #153106,
        #153108,
        #153110,
        #153112,
        #153114,
        #153116,
        #153118,
        #153120,
        #153122,
        #153124,
        #153126,
        #153128,
        #153130,
        #153132,
        #153134,
        #153136,
        #153138,
        #153140,
        153142,
        153143,
        153145,
    ]

    dark_field = io.imread(f'{base_wd}scan153086_dexela_median_composite.tif')
    #flat_field = io.imread()
    poni_file = 'scan153043_dexela_calibration.poni'

    for scan in timed_iter(scanlist):
        print(f'Batch processing scan {scan}...')
        
        # Load map and set calibration
        xrdmap = XRDMap.from_hdf(f'scan{scan}_xrd.h5', wd=f'{base_wd}processed_xrdmaps/', save_hdf=True)
        xrdmap.set_calibration(poni_file, filedir=base_wd)
        
        # Basic correction. No outliers
        xrdmap.map.correct_dark_field(dark_field=dark_field)
        #xrdmap.map.correct_flat_field(flat_field=flat_field)
        xrdmap.map.normalize_scaler() # Assumed information in sclr_dict
        #xrdmap.map.correct_outliers() # Too slow!

        # Geometric corrections
        xrdmap.map.apply_polarization_correction()
        xrdmap.map.apply_solidangle_correction()
        xrdmap.map.images *= np.radians(xrdmap.tth_arr)
        xrdmap.map.corrections['lorentz'] = True

        # Background correction
        xrdmap.map.estimate_background(method='bruckner', binning=4, min_prominence=0.1)
        xrdmap.map.remove_background()

        # Rescale and saving
        xrdmap.map.rescale_images(upper=100, lower=0, arr_min=0)
        xrdmap.map.finalize_images()

        # Integrations for good measure
        xrdmap.tth_resolution = 0.01
        xrdmap.chi_resolution = 0.05
        xrdmap.integrate1d_map()

        # Find blobs and spots while were at it
        xrdmap.find_spots(threshold_method='minimum',
                          multiplier=3, size=3,
                          radius=10, expansion=10)

        

# Basic batch functioning for processing xrdmaps at single energy
def xmt_batch2():

    base_wd = '/nsls2/data/srx/proposals/2024-1/pass-314118/'
    scanlist = [
        153157,
        153159,
        153161,
        153163,
        153165,
        153167,
        153169,
        153171,
        153173,
        153175,
        153177,
        153179,
        153181,
        153183,
        153185,
        153187,
        153189,
        153191,
        153193,
        153195,
        153197,
        153199,
        153201,
        153203,
        153205,
        153207,
        153209,
        153211,
        153213,
        153215,
    ]

    dark_field = io.imread(f'{base_wd}scan153155_dexela_median_composite.tif')
    #flat_field = io.imread()
    poni_file = 'scan153219_dexela_calibration.poni'

    for scan in timed_iter(scanlist):
        print(f'Batch processing scan {scan}...')

        if not os.path.exists(f'{base_wd}processed_xrdmaps/scan{scan}_xrd.h5'):
            print('No raw file found. Generating new file!')
            make_xrdmap_hdf(scan, filedir=base_wd + 'processed_xrdmaps/')
        
        # Load map and set calibration
        xrdmap = XRDMap.from_hdf(f'scan{scan}_xrd.h5', wd=f'{base_wd}processed_xrdmaps/', save_hdf=True)
        xrdmap.set_calibration(poni_file, filedir=base_wd)
        
        # Basic correction. No outliers
        xrdmap.map.correct_dark_field(dark_field=dark_field)
        #xrdmap.map.correct_flat_field(flat_field=flat_field)
        xrdmap.map.normalize_scaler() # Assumed information in sclr_dict
        #xrdmap.map.correct_outliers() # Too slow!

        # Geometric corrections
        xrdmap.map.apply_polarization_correction()
        xrdmap.map.apply_solidangle_correction()
        xrdmap.map.images *= np.radians(xrdmap.tth_arr)
        xrdmap.map.corrections['lorentz'] = True

        # Background correction
        xrdmap.map.estimate_background(method='bruckner', binning=4, min_prominence=0.1)
        xrdmap.map.remove_background()

        # Rescale and saving
        xrdmap.map.rescale_images(upper=100, lower=0, arr_min=0)
        xrdmap.map.finalize_images()

        # Integrations for good measure
        xrdmap.tth_resolution = 0.01
        xrdmap.chi_resolution = 0.05
        xrdmap.integrate1d_map()

        # Find blobs and spots while were at it
        xrdmap.find_spots(threshold_method='minimum',
                          multiplier=3, size=3,
                          radius=10, expansion=10)


# Basic batch functioning for processing xrdmaps at single energy
def xmt_batch3():

    base_wd = '/nsls2/data/srx/proposals/2024-1/pass-314118/'
    scanlist = [
        153253,
        153255,
        153257,
        153259,
        153261,
        153263,
        153265,
        153267,
        153269,
        153271,
        153273,
        153175,
        153277,
        153279,
        153281,
        153283,
        153285,
        153287,
        153289,
        153291,
        153293,
        153295,
        153297,
    ]

    dark_field = io.imread(f'{base_wd}scan153247_dexela_median_composite.tif')
    #flat_field = io.imread()
    poni_file = 'scan153219_dexela_calibration.poni'

    for scan in timed_iter(scanlist):
        print(f'Batch processing scan {scan}...')

        if not os.path.exists(f'{base_wd}processed_xrdmaps/scan{scan}_xrd.h5'):
            print('No raw file found. Generating new file!')
            make_xrdmap_hdf(scan, filedir=base_wd + 'processed_xrdmaps/')
        
        # Load map and set calibration
        xrdmap = XRDMap.from_hdf(f'scan{scan}_xrd.h5', wd=f'{base_wd}processed_xrdmaps/', save_hdf=True)
        xrdmap.set_calibration(poni_file, filedir=base_wd)
        
        # Basic correction. No outliers
        xrdmap.map.correct_dark_field(dark_field=dark_field)
        #xrdmap.map.correct_flat_field(flat_field=flat_field)
        xrdmap.map.normalize_scaler() # Assumed information in sclr_dict
        #xrdmap.map.correct_outliers() # Too slow!

        # Geometric corrections
        xrdmap.map.apply_polarization_correction()
        xrdmap.map.apply_solidangle_correction()
        xrdmap.map.images *= np.radians(xrdmap.tth_arr)
        xrdmap.map.corrections['lorentz'] = True

        # Background correction
        xrdmap.map.estimate_background(method='bruckner', binning=4, min_prominence=0.1)
        xrdmap.map.remove_background()

        # Rescale and saving
        xrdmap.map.rescale_images(upper=100, lower=0, arr_min=0)
        xrdmap.map.finalize_images()

        # Integrations for good measure
        xrdmap.tth_resolution = 0.01
        xrdmap.chi_resolution = 0.05
        xrdmap.integrate1d_map()

        # Find blobs and spots while were at it
        xrdmap.find_spots(threshold_method='minimum',
                          multiplier=3, size=3,
                          radius=10, expansion=10)
    