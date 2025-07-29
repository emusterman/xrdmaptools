import numpy as np
import os
from tqdm import tqdm
from skimage import io
import h5py
import time as ttime
import matplotlib.pyplot as plt

# Local imports
from xrdmaptools.XRDRockingCurve import XRDRockingCurve
from xrdmaptools.XRDMap import XRDMap
from xrdmaptools.XRDMapStack import XRDMapStack
from xrdmaptools.io.db_io import *
from xrdmaptools.utilities.utilities import pathify
from xrdmaptools.utilities.math import tth_2_q


# Working at the beamline...
try:
    print('Connecting to databrokers...', end='', flush=True)
    from tiled.client import from_profile
    c = from_profile('srx')
    print('done!')
except ModuleNotFoundError:
    print('failed.')
    pass


# TODO:
# Automatically check for swapped axes
# Check dark-field shapes
# Add auto processing for rsm
# How to alter kwargs for each individual function?
# Check for most recent scan_id and remove delays when lagging behind


def standard_process_xdm(scan_id,
                         wd,
                         dark_field,
                         poni_file,
                         swapped_axes=False):

    if not isinstance(scan_id, XRDMap):
        if os.path.exists(f'{wd}xrdmaps/scan{scan_id}_xrdmap.h5'):
            print('File found! Loading from HDF...')
            xdm = XRDMap.from_hdf(f'scan{scan_id}_xrdmap.h5',
                                  wd=f'{wd}xrdmaps/', image_data_key='raw', swapped_axes=swapped_axes)
        else:
            print('Loading data from server...')
            xdm = XRDMap.from_db(scan_id, wd=f'{wd}xrdmaps/', swapped_axes=swapped_axes)
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

    np.savetxt(f'{wd}/max_1D_integrations/scan{xdm.scan_id}_max_1D_integration.txt',
               np.asarray([q, tth, intensity]))
    fig, ax = xdm.plot_integration(intensity, tth=tth, title='Max Integration', return_plot=True)
    fig.savefig(f'{wd}max_1D_integrations/scan{xdm.scan_id}_max_integration.png')
    plt.close('all')

    io.imsave(f'{wd}integrated_maps/scan{xdm.scan_id}_max_map.tif', xdm.max_map)

    images = xdm.images.copy()
    xdm.dump_images()
    images[~xdm.blob_masks] = 0
    blob_sum_map = np.sum(images, axis=(2, 3))

    io.imsave(f'{wd}integrated_maps/scan{xdm.scan_id}_blob_sum.tif', blob_sum_map)

    return xdm


def standard_process_rsm(scan_id,
                         wd,
                         dark_field,
                         poni_file):

    if not isinstance(scan_id, XRDRockingCurve):
        if os.path.exists(f'{wd}reciprocal_space_maps/scan{scan_id}_rsm.h5'):
            print('File found! Loading from HDF...')
            rsm = XRDRockingCurve.from_hdf(f'scan{scan_id}_rsm.h5', wd=f'{wd}reciprocal_space_maps/', image_data_key='raw')
        else:
            print('Loading data from server...')
            rsm = XRDRockingCurve.from_db(scan_id, wd=f'{wd}reciprocal_space_maps/')
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
        rsm.set_calibration(poni_file, wd=f'{wd}calibrations/')
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


def find_most_recent_dark_field(dark_wd, shape=None):
    dark_list = os.listdir(dark_wd)
    dark_scan_ids = [int(name[4:10]) for name in dark_list]
    dark_diff_list = scan_id - np.array(dark_scan_ids, dtype=float)
    dark_ind = np.nonzero(dark_diff_list > 0)[0][np.argmin(dark_diff_list[dark_diff_list > 0])]
    dark_field = io.imread(f'{wd}dark_fields/{dark_list[dark_ind]}')

    return dark_field


def find_most_recent_calibration(calibration_wd):
    poni_list = [file for file in os.listdir(calibration_wd) if file[-4:] == 'poni']
    poni_scan_ids = [int(name[4:10]) for name in poni_list if name[-4:] == 'poni']
    poni_diff_list = scan_id - np.array(poni_scan_ids, dtype=float)
    poni_ind = np.nonzero(poni_diff_list > 0)[0][np.argmin(poni_diff_list[poni_diff_list > 0])]
    poni_file = poni_list[poni_ind]

    return poni_file


def prepare_standard_directories(wd):

    # Generate standard directory structure
    # Should this be wrapped in an xrd folder?
    os.makedirs(wd, exist_ok=True)
    os.makedirs(os.path.join(wd, 'dark_fields/'), exist_ok=True)
    os.makedirs(os.path.join(wd, 'calibrations/'), exist_ok=True)
    os.makedirs(os.path.join(wd, 'air_scatter/'), exist_ok=True)
    os.makedirs(os.path.join(wd, 'xrdmaps/'), exist_ok=True)
    os.makedirs(os.path.join(wd, 'reciprocal_space_maps/'), exist_ok=True)
    os.makedirs(os.path.join(wd, 'integrated_maps/'), exist_ok=True)
    os.makedirs(os.path.join(wd, 'max_1D_integrations/'), exist_ok=True)


# Batch processing xrdmaps
def auto_process_xdm(start_id,
                     stop_id=None,
                     wd=None,
                     swapped_axes=False):

    # Prepare save locations
    prepare_standard_directories(wd)

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
                save_composite_pattern(scan_id, broker='manual', wd=f'{wd}dark_fields/', method='median')
                ttime.sleep(10)
                scan_id += 1
                continue
            
            # Check for already processed scans. Kind of
            if os.path.exists(f'{wd}xrdmaps/scan{scan_id}_xrdmap.h5'):
                with h5py.File(f'{wd}xrdmaps/scan{scan_id}_xrdmap.h5', 'r') as f:
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
            dark_list = os.listdir(f'{wd}dark_fields/')
            dark_scan_ids = [int(name[4:10]) for name in dark_list]
            dark_diff_list = scan_id - np.array(dark_scan_ids, dtype=float)
            dark_ind = np.nonzero(dark_diff_list > 0)[0][np.argmin(dark_diff_list[dark_diff_list > 0])]
            dark_field = io.imread(f'{wd}dark_fields/{dark_list[dark_ind]}')

            # Find most recent poni_file prior to this scan ID
            poni_list = [file for file in os.listdir(f'{wd}calibrations/') if file[-4:] == 'poni']
            poni_scan_ids = [int(name[4:10]) for name in poni_list if name[-4:] == 'poni']
            poni_diff_list = scan_id - np.array(poni_scan_ids, dtype=float)
            poni_ind = np.nonzero(poni_diff_list > 0)[0][np.argmin(poni_diff_list[poni_diff_list > 0])]
            poni_file = poni_list[poni_ind]

            xdm = standard_process_xdm(scan_id,
                                       wd,
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