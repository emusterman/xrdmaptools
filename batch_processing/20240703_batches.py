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


def xmt_batch4():

    base_wd = '/nsls2/data/srx/proposals/2024-1/pass-314118/'
    scanlist = [
        153485,
        153487,
        153489,
        153491,
        153493,
        153495,
        153502,
        153504,
        153506,
        153508,
        153510
    ]

    dark_field = io.imread(f'{base_wd}scan153481_dexela_median_composite.tif')
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



def xmt_batch5():

    base_wd = '/nsls2/data/srx/proposals/2024-1/pass-314118/'
    scanlist = [
        153442,
        153443,
        153444,
        153445,
        153446,
        153448,
        153449,
        153450,
        153451,
        153452,
        153454,
        153455,
        153456,
        153457,
        153458,
        153460,
        153461,
        153462,
        153463,
        153464

    ]

    dark_field = io.imread(f'{base_wd}scan153436_dexela_median_composite.tif')
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
        #test.map.images[0, 0, 0, 0] = 100 # to trick the scaled image check
        xrdmap.find_spots(threshold_method='minimum',
                          multiplier=5, size=2,
                          radius=10, expansion=10)


# Everything else...
def xmt_batch6():

    base_wd = '/nsls2/data/srx/proposals/2024-1/pass-314118/'
    scanlist = [
        153175,
        153275,
        153076,
        153088,
        153092,
        153094,
        153098,
        153154,
        153248
    ]

    darklist = [
        153155,
        153247,
        153074,
        153086,
        153086,
        153086,
        153086,
        153155,
        153247
    ]

    ponilist = [
        153219,
        153219,
        153043,
        153043,
        153043,
        153043,
        153043,
        153219,
        153219
    ]

    

    for i in timed_iter(range(len(scanlist))):
        scan, dark, poni = scanlist[i], darklist[i], ponilist[i]

        print(f'Batch processing scan {scan}...')

        if not os.path.exists(f'{base_wd}processed_xrdmaps/scan{scan}_xrd.h5'):
            print('No raw file found. Generating new file!')
            make_xrdmap_hdf(scan, filedir=base_wd + 'processed_xrdmaps/')

        dark_field = io.imread(f'{base_wd}scan{dark}_dexela_median_composite.tif')
        #flat_field = io.imread()
        poni_file = f'scan{poni}_dexela_calibration.poni'
        
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
        #test.map.images[0, 0, 0, 0] = 100 # to trick the scaled image check
        xrdmap.find_spots(threshold_method='minimum',
                          multiplier=5, size=3,
                          radius=10, expansion=10)


# Everything else...
def xmt_batch7():

    base_wd = '/nsls2/data/srx/proposals/2024-2/pass-314118/'

    dark_field = io.imread(f'{base_wd}dark_fields/scan156843_dexela_median_composite.tif')
    #flat_field = io.imread()
    poni_file = f'scan156160_dexela_calibration.poni'
    air_scatter = io.imread(f'{base_wd}air_scatter/scan156859_dexela_median_composite.tif')
    dark_air = io.imread(f'{base_wd}dark_fields/scan156858_dexela_median_composite.tif')
    air = (air_scatter.astype(np.float32) - dark_air.astype(np.float32)) / 10

    for folder in [
        #156844,
        156846,
        156848,
        156855
        ]:
        folder_name = f'scan{folder}_fractured_maps'

        new_base = base_wd + f'xrdmaps/{folder_name}/'
        scanlist = [f'{folder}-{i + 1}' for i in range(len(os.listdir(new_base)))]
    

        for i in timed_iter(range(len(scanlist))):
            scan = scanlist[i]

            print(f'Batch processing scan {scan}...')

            # if not os.path.exists(f'{base_wd}processed_xrdmaps/scan{scan}_xrd.h5'):
            #     print('No raw file found. Generating new file!')
            #     make_xrdmap_hdf(scan, filedir=base_wd + 'processed_xrdmaps/')
            
            # Load map and set calibration
            xrdmap = XRDMap.from_hdf(f'scan{scan}_xrd.h5', wd=new_base, save_hdf=True)
            xrdmap.set_calibration(poni_file, filedir=base_wd + 'calibrations/')
            
            # Basic correction. No outliers
            xrdmap.map.correct_dark_field(dark_field=dark_field)
            #xrdmap.map.correct_flat_field(flat_field=flat_field)
            xrdmap.map.images -= air

            xrdmap.map.normalize_scaler() # Assumed information in sclr_dict
            #xrdmap.map.correct_outliers() # Too slow!

            # Geometric corrections
            xrdmap.map.apply_polarization_correction()
            xrdmap.map.apply_solidangle_correction()
            xrdmap.map.apply_lorentz_correction()

            # Background correction
            xrdmap.map.estimate_background(method='bruckner', binning=4, min_prominence=0.1)
            xrdmap.map.remove_background()

            # Rescale and saving
            xrdmap.map.rescale_images(
                upper=100,
                lower=0,
                arr_min=0,
                arr_max=xrdmap.map.estimate_saturated_pixel())
            xrdmap.map.finalize_images()

            # Integrations for good measure
            xrdmap.tth_resolution = 0.01
            xrdmap.chi_resolution = 0.05
            xrdmap.integrate1d_map()

            # Find blobs and spots while were at it
            #test.map.images[0, 0, 0, 0] = 100 # to trick the scaled image check
            xrdmap.find_spots(
                threshold_method='minimum',
                multiplier=3,
                size=3,
                radius=10,
                expansion=10,
                override_rescale=True)
            

# With absorption
def xmt_batch8():

    base_wd = '/nsls2/data/srx/proposals/2024-1/pass-314118/'

    dark_field = io.imread(f'{base_wd}dark_fields/scan153481_dexela_median_composite.tif')
    #flat_field = io.imread()
    poni_file = f'scan153219_dexela_calibration.poni'

    scanlist = [
        153485,
        153487,
        153489,
        153491,
        153493,
        153495,
        153502,
        153504,
        153506,
        153508,
        153510
    ]

    for i in timed_iter(range(len(scanlist))):
        scan = scanlist[i]

        print(f'Batch processing scan {scan}...')

        # if not os.path.exists(f'{base_wd}processed_xrdmaps/scan{scan}_xrd.h5'):
        #     print('No raw file found. Generating new file!')
        #     make_xrdmap_hdf(scan, filedir=base_wd + 'processed_xrdmaps/')
        
        # Load map and set calibration
        xrdmap = XRDMap.from_hdf(f'scan{scan}_xrd.h5', wd=base_wd + 'processed_xrdmaps/', save_hdf=True)
        xrdmap.interpolate_positions()
        xrdmap.load_phase('AMCSD\\Stibnite_0008636.cif', filedir=cif_dir, phase_name="stibnite")
        xrdmap.set_calibration(poni_file, filedir=base_wd + 'calibrations/')
        
        # Basic correction. No outliers
        xrdmap.map.correct_dark_field(dark_field=dark_field)
        #xrdmap.map.correct_flat_field(flat_field=flat_field)

        xrdmap.map.normalize_scaler() # Assumed information in sclr_dict
        #xrdmap.map.correct_outliers() # Too slow!

        # Geometric corrections
        xrdmap.map.apply_polarization_correction()
        xrdmap.map.apply_solidangle_correction()
        xrdmap.map.apply_lorentz_correction()
        
        # Apply absorption corrections
        exp_dict = {
        'attenuation_length' : 0,
        'mode' : 'transmission',
        'thickness' : 200, # microns # Horrible guess...
        'theta' : 0
        }
        exp_dict['attenuation_length'] = xrdmap.phases['stibnite'].absorption_length(en=xrdmap.energy * 1e3)
        xrdmap.map.apply_absorption_correction(exp_dict=exp_dict, apply=True)

        # Background correction
        xrdmap.map.estimate_background(method='bruckner', binning=4, min_prominence=0.1)
        xrdmap.map.remove_background()

        # Rescale and saving
        xrdmap.map.rescale_images(
            upper=100,
            lower=0,
            arr_min=0,
            arr_max=xrdmap.map.estimate_saturated_pixel())
        xrdmap.map.finalize_images()

        # Integrations for good measure
        xrdmap.tth_resolution = 0.01
        xrdmap.chi_resolution = 0.05
        xrdmap.integrate1d_map()

        # Find blobs and spots while were at it
        #test.map.images[0, 0, 0, 0] = 100 # to trick the scaled image check
        xrdmap.find_spots(
            threshold_method='minimum',
            multiplier=3,
            size=3,
            radius=10,
            expansion=10,
            override_rescale=True)