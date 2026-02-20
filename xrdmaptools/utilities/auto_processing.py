import numpy as np
import os
from tqdm import tqdm
from skimage import io
import h5py
import time as ttime
import matplotlib.pyplot as plt

# Local imports
import xrdmaptools
from .. import XRDRockingCurve
from .. import XRDMap
# from .. import XRDMapStack
from ..io.db_io import _load_tiled_catalog
from .utilities import pathify
from .math import tth_2_q

# Load database if not already. This may fail...
c = xrdmaptools.io.db_io.c
# if c is None:
#     _load_tiled_catalog()



# TODO:
# Automatically check for swapped axes
# Check dark-field shapes
# Add auto processing for rsm
# How to alter kwargs for each individual function?
# Check for most recent scan_id and remove delays when lagging behind


def standard_process_xdm(scan_id,
                         wd,
                         dark_field=None,
                         poni_file=None,
                         air_scatter=False,
                         proc_dict=None,
                         swapped_axes=False,
                         reprocess=False):

    
    # Standard. Rewrite as needed
    _proc_dict = {
        'images' : True,
        'save_images' : True,
        'nullify_bad_rows' : False,
        'integrations' : True,
        'blobs' : True,
        'vectors' : True,
        'maximums' : True,
        'reduced' : False
    }
    _proc_dict.update(proc_dict or {})

    if not isinstance(scan_id, XRDMap):
        if isinstance(scan_id, str):
            fname = scan_id
        else:
            fname = f'scan{int(scan_id)}_xrdmap.h5'

        if os.path.exists(f'{wd}xrdmaps/{fname}'):
            print('File found! Loading from HDF...')
            
            load_kwargs = {
                'wd' : f'{wd}xrdmaps/',
                'image_data_key' : 'raw',
                'swapped_axes' : swapped_axes
            }
            if not reprocess:
                # Check to avoid reprocessing
                temp_hdf = h5py.File(f'{wd}xrdmaps/{fname}')
                if 'final_images' in temp_hdf['xrdmap/image_data']:
                    _proc_dict['images'] = False
                    _proc_dict['save_images'] = False
                    load_kwargs['image_data_key'] = 'final'
                if 'integration_data' in temp_hdf['xrdmap'] and 'final_integrations' in temp_hdf['xrdmap/integration_data']:
                    _proc_dict['integrations'] = False
                    load_kwargs['integration_data_key'] = None
                if '_blob_masks' in temp_hdf['xrdmap/image_data']:
                    _proc_dict['blobs'] = False
                if 'vectorized_map' in temp_hdf['xrdmap']:
                    _proc_dict['vectors'] = False
                temp_hdf.close()

                if all([os.path.exists(p) for p in [f'{wd}/max_1D_integrations/scan{fname[4:10]}_max_1D_integration.txt',
                                                    f'{wd}max_1D_integrations/scan{fname[4:10]}_max_integration.png',
                                                    f'{wd}integrated_maps/scan{fname[4:10]}_max_map.tif',
                                                    f'{wd}integrated_maps/scan{fname[4:10]}_blob_sum.tif']]):
                    _proc_dict['maximums'] = False
            
            if any(_proc_dict.values()):
                xdm = XRDMap.from_hdf(fname, **load_kwargs)
            else:
                xdm = None
        else:
            if not isinstance(scan_id, str):
                print('Loading data from server...')
                xdm = XRDMap.from_db(scan_id, wd=f'{wd}xrdmaps/', swapped_axes=swapped_axes)
            else:
                raise ValueError(f'Scan ID must be string not {scan_id}')
        
        print(f'Starting processing for XRDMap Scan ID {xdm.scan_id}.')
    else:
        # TODO: Look for processing conditions
        xdm = scan_id
    
    if _proc_dict['images']:
        # Basic corrections
        if xdm.detector == 'dexela':
            if dark_field is None:
                if (not hasattr(xdm, 'dark_field')
                    or xdm.dark_field is None):
                    dark_field  = find_most_recent_dark_field(xdm.scan_id,
                                                            f'{wd}dark_fields/',
                                                            shape=xdm.image_shape)
            xdm.correct_dark_field(dark_field)


        elif xdm.detector == 'eiger':
            xdm.apply_defect_mask()

        xdm.convert_scalers_to_flux(scaler_key='i0')
        xdm.convert_scalers_to_flux(scaler_key='im')
        xdm.normalize_scaler()
        if air_scatter == False:
            pass
        elif isinstance(air_scatter, np.ndarray):
            xdm.correct_air_scatter(air_scatter)
        elif air_scatter == True:
            xdm.correct_air_scatter(xdm.med_image, applied_corrections=xdm.corrections)
        else:
            err_str = 'Error handling air_scatter. Designate array, None, or bool.'
            raise RuntimeError(err_str)
        xdm.correct_outliers(tolerance=5)

        # Geometric corrections
        if poni_file is None:
            if not hasattr(xdm, 'ai') or xdm.ai is None:
                poni_file = find_most_recent_calibration(xdm.scan_id,
                                                         f'{wd}calibrations/')
                xdm.set_calibration(poni_file, wd=f'{wd}calibrations/')
        else:
            xdm.set_calibration(poni_file, wd=f'{wd}calibrations/')
        xdm.apply_polarization_correction()
        xdm.apply_solidangle_correction()

        # Background correction
        xdm.estimate_background(method='bruckner',
                                binning=8,
                                min_prominence=0.01)

        # Rescale and saving
        xdm.rescale_images(arr_max=xdm.estimate_saturated_pixel())
        if _proc_dict['save_images']:
            if xdm.images.dtype != xdm.dtype:
                print('WARNING: Data was somehow upcast!!!!')
                print('Trying to downcasting data to np.float32')
                xdm.images = xdm.images.astype(xdm.dtype)
                xdm._dtype = xdm.images.dtype
            
            xdm.images[:, :, ~xdm.mask] = 0
            xdm.finalize_images()

    # Remove enitire bad rows
    if _proc_dict['nullify_bad_rows'] and xdm is not None:
        bad_rows = np.any(xdm.null_map, axis=1)
        xdm.images[bad_rows] = 0  

    # Integrate map
    if _proc_dict['integrations']:
        xdm.integrate1D_map()

    # Find blobs
    if _proc_dict['blobs']:
        xdm.find_blobs(filter_method='minimum',
                       multiplier=5,
                       size=3,
                       expansion=10)

    # Vectorize blobs
    if _proc_dict['vectors']:
        xdm.vectorize_map_data()

    # Convert to 1D integrations
    if _proc_dict['maximums'] and xdm is not None:
        tth, intensity = xdm.integrate1D_image(xdm.max_image)
        q = tth_2_q(tth, wavelength=xdm.wavelength)

        np.savetxt(f'{wd}/max_1D_integrations/scan{xdm.scan_id}_max_1D_integration.txt',
                np.asarray([q, tth, intensity]))
        fig, ax = xdm.plot_integration(intensity, tth=tth, title='Max Integration', return_plot=True)
        fig.savefig(f'{wd}max_1D_integrations/scan{xdm.scan_id}_max_integration.png')
        plt.close('all')

        # io.imsave(f'{wd}integrated_maps/scan{xdm.scan_id}_max_map.tif', xdm.max_map)

        # images = xdm.images.copy()
        # xdm.dump_images()
        # images[~xdm.blob_masks] = 0
        # blob_sum_map = np.sum(images, axis=(2, 3))

        # io.imsave(f'{wd}integrated_maps/scan{xdm.scan_id}_blob_sum.tif', blob_sum_map)
    
    if _proc_dict['reduced']:
        xdm = save_and_reduce_xdm(xdm)

    return xdm


def standard_process_rsm(scan_id,
                         wd,
                         dark_field=None,
                         poni_file=None,
                         air_scatter=True,
                         proc_dict=None,
                         swapped_axes=False,
                         reprocess=False):

    
    # Standard. Rewrite as needed
    _proc_dict = {
        'images' : True,
        'save_images' : True,
        'blobs' : True,
        'vectors' : True,
        'spots' : True
    }
    _proc_dict.update(proc_dict or {})

    if not isinstance(scan_id, XRDRockingCurve):
        if isinstance(scan_id, str):
            fname = scan_id
        else:
            fname = f'scan{int(scan_id)}_rsm.h5'

        if os.path.exists(f'{wd}reciprocal_space_maps/{fname}'):
            print('File found! Loading from HDF...')
            
            load_kwargs = {
                'wd' : f'{wd}reciprocal_space_maps/',
                'image_data_key' : 'raw',
                'swapped_axes' : swapped_axes
            }
            if not reprocess:
                # Check to avoid reprocessing
                temp_hdf = h5py.File(f'{wd}reciprocal_space_maps/{fname}')
                if 'final_images' in temp_hdf['xrdmap/image_data']:
                    _proc_dict['images'] = False
                    _proc_dict['save_images'] = False
                    load_kwargs['image_data_key'] = 'final'
                if 'integration_data' in temp_hdf['xrdmap'] and 'final_integrations' in temp_hdf['xrdmap/integration_data']:
                    _proc_dict['integrations'] = False
                    load_kwargs['integration_data_key'] = None
                if '_blob_masks' in temp_hdf['xrdmap/image_data']:
                    _proc_dict['blobs'] = False
                if 'vectorized_map' in temp_hdf['xrdmap']:
                    _proc_dict['vectors'] = False
                temp_hdf.close()

                if all([os.path.exists(p) for p in [f'{wd}/max_1D_integrations/scan{fname[4:10]}_max_1D_integration.txt',
                                                    f'{wd}max_1D_integrations/scan{fname[4:10]}_max_integration.png',
                                                    f'{wd}integrated_maps/scan{fname[4:10]}_max_map.tif',
                                                    f'{wd}integrated_maps/scan{fname[4:10]}_blob_sum.tif']]):
                    _proc_dict['maximums'] = False
            
            if any(_proc_dict.values()):
                xdm = XRDMap.from_hdf(fname, **load_kwargs)
            else:
                xdm = None
        else:
            if not isinstance(scan_id, str):
                print('Loading data from server...')
                xdm = XRDMap.from_db(scan_id, wd=f'{wd}xrdmaps/', swapped_axes=swapped_axes)
            else:
                raise ValueError(f'Scan ID must be string not {scan_id}')
    else:
        # TODO: Look for processing conditions
        xdm = scan_id
    
    if _proc_dict['images']:
        # Basic corrections
        if dark_field is None:
            if (not hasattr(xdm, 'dark_field')
                or xdm.dark_field is None):
                dark_field  = find_most_recent_dark_field(xdm.scan_id,
                                                          f'{wd}dark_fields/',
                                                          shape=xdm.image_shape)
        xdm.correct_dark_field(dark_field)
        # xdm.correct_scaler_energies(scaler_key='i0')
        xdm.convert_scalers_to_flux(scaler_key='i0')
        # xdm.correct_scaler_energies(scaler_key='im')
        xdm.convert_scalers_to_flux(scaler_key='im')
        xdm.normalize_scaler()
        if air_scatter == False:
            pass
        elif isinstance(air_scatter, np.ndarray):
            xdm.correct_air_scatter(air_scatter)
        elif air_scatter == True:
            xdm.correct_air_scatter(xdm.med_image, applied_corrections=xdm.corrections)
        else:
            err_str = 'Error handling air_scatter. Designate array, None, or bool.'
            raise RuntimeError(err_str)
        xdm.correct_outliers(tolerance=10)

        # Geometric corrections
        if poni_file is None:
            if not hasattr(xdm, 'ai') or xdm.ai is None:
                poni_file = find_most_recent_calibration(xdm.scan_id,
                                                         f'{wd}calibrations/')
                xdm.set_calibration(poni_file, wd=f'{wd}calibrations/')
        else:
            xdm.set_calibration(poni_file, wd=f'{wd}calibrations/')
        xdm.apply_polarization_correction()
        xdm.apply_solidangle_correction()

        # Background correction
        xdm.estimate_background(method='bruckner',
                                binning=8,
                                min_prominence=0.01)

        # Rescale and saving
        xdm.rescale_images(arr_max=xdm.estimate_saturated_pixel())
        if _proc_dict['save_images']:
            if xdm.images.dtype != xdm.dtype:
                print('WARNING: Data was somehow upcast!!!!')
                print('Trying to downcasting data to np.float32')
                xdm.images = xdm.images.astype(xdm.dtype)
                xdm._dtype = xdm.images.dtype
            xdm.finalize_images()

    # Remove enitire bad rows
    if _proc_dict['nullify_bad_rows'] and xdm is not None:
        bad_rows = np.any(xdm.null_map, axis=1)
        xdm.images[bad_rows] = 0  

    # Integrate map
    if _proc_dict['integrations']:
        xdm.integrate1D_map()

    # Find blobs
    if _proc_dict['blobs']:
        xdm.find_blobs(filter_method='minimum',
                       multiplier=5,
                       size=3,
                       expansion=10)

    # Vectorize blobs
    if _proc_dict['vectors']:
        xdm.vectorize_map_data()

    # Convert to 1D integrations
    if _proc_dict['maximums'] and xdm is not None:
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


def find_most_recent_dark_field(scan_id, dark_wd, shape=None):
    dark_list = os.listdir(dark_wd)
    dark_scan_ids = [int(name[4:10]) for name in dark_list]
    dark_diff_list = scan_id - np.array(dark_scan_ids, dtype=float)
    dark_ind = np.nonzero(dark_diff_list > 0)[0][np.argmin(dark_diff_list[dark_diff_list > 0])]
    dark_field = io.imread(f'{wd}dark_fields/{dark_list[dark_ind]}')

    return dark_field


def find_most_recent_calibration(scan_id, calibration_wd, detector=None):
    poni_list = [file for file in os.listdir(calibration_wd) if file[-4:] == 'poni']
    if detector is not None:
        poni_list = [file for file in poni_list if detector in file]
    poni_scan_ids = [int(name[4:10]) for name in poni_list if name[-4:] == 'poni']
    poni_diff_list = scan_id - np.array(poni_scan_ids, dtype=float)
    poni_ind = np.nonzero(poni_diff_list > 0)[0][np.argmin(poni_diff_list[poni_diff_list > 0])]
    poni_file = poni_list[poni_ind]

    return poni_file


def prepare_standard_directories(wd):

    # Generate standard directory structure
    # Should this be wrapped in an xrd folder?
    os.makedirs(wd, exist_ok=True)
    # os.makedirs(os.path.join(wd, 'dark_fields/'), exist_ok=True)
    os.makedirs(os.path.join(wd, 'calibrations/'), exist_ok=True)
    # os.makedirs(os.path.join(wd, 'air_scatter/'), exist_ok=True)
    os.makedirs(os.path.join(wd, 'xrdmaps/'), exist_ok=True)
    # os.makedirs(os.path.join(wd, 'reciprocal_space_maps/'), exist_ok=True)
    # os.makedirs(os.path.join(wd, 'integrated_maps/'), exist_ok=True)
    os.makedirs(os.path.join(wd, 'max_1D_integrations/'), exist_ok=True)


def save_and_reduce_xdm(xdm):
    
    new_wd = f'{xdm.wd}reduced/'
    xdm.wd = new_wd
    xdm.dump_images()
    xdm.switch_hdf(hdf_filename=f'{xdm.filename}_reduced',
                   save_current=True,
                   verbose=False)
    xdm.save_images(title='empty')

    return xdm


def auto_process_xrd(start_id,
                     stop_id=None,
                     wd=None,
                     xrd_dets=None,
                     generate_directories=True,
                     verbose=False,
                     **kwargs):
    
    if verbose:
        vprint = lambda *args, **kwargs : print(*args, **kwargs)
    else:
        vprint = lambda *args, **kwargs : None

    if xrd_dets is None:
        xrd_dets = ['dexela', 'merlin', 'eiger']
    
    # # Check for database
    global c
    if c is None:
        _load_tiled_catalog()
        c = xrdmaptools.io.db_io.c

    # Prepare save locations
    if generate_directories:
        prepare_standard_directories(wd)

    if stop_id is not None and start_id > stop_id:
        err_str = 'Start ID after stop ID!'
        raise ValueError(err_str)

    # Start scan_id
    scan_id = start_id
    wait_iter = 0
    while True:
        try:
            # First check if beyond stop ID
            if stop_id is not None and scan_id > stop_id:
                print(f'Auto processing reached stop limit of {stop_id}!')
                break
            
            # Second check to see if the current scan_id is finished
            WAIT = False
            recent_id = c[-1].start['scan_id']
            # scan id has not been started
            if scan_id > recent_id:
                WAIT = True
            else:
                # Does the scan_id exist?
                if scan_id not in c:
                    err_str = (f'Scan ID {scan_id} not found in '
                               + 'tiled. Currently no provisions for '
                               + 'this error.\nSkipping and moving to '
                               + 'next scan ID.')
                    scan_id += 1
                    continue
                # Is this the most recent scan and has it finished?
                elif scan_id == recent_id:
                    if (not hasattr(c[scan_id], 'stop')
                        or c[scan_id].stop is None
                        or 'time' not in c[scan_id].stop):
                        WAIT = True

            # Current scan has yet to finish. Give it some time.
            if WAIT:
                wait_time = 60
                ostr = (f'Current scan ID {scan_id} not yet finished. '
                        + f'{wait_iter} minutes since last scan added...')
                print(ostr, end='\r', flush=True)
                ttime.sleep(wait_time)
                wait_iter += 1
                continue
            else:
                if wait_iter > 0:
                    print('', flush=True)
                    wait_iter = 0

            # Third check for area detectors
            if ('scan' not in c[scan_id].start
                or 'detectors' not in c[scan_id].start['scan']
                or not any([det in c[scan_id].start['scan']['detectors'] for det in xrd_dets])):
                vprint(f'Missing XRD det for scan ID {scan_id}.')
                scan_id += 1
                continue
            
            # Fourth determine exit status
            if c[scan_id].stop['exit_status'] != 'success':
                err_str = (f'Scan failed for scan ID {scan_id}.'
                           + '\nSkipping and moving to next scan ID.')
                print(err_str)
                scan_id += 1
                continue

            # Fifth determine processing type
            scan_type = None
            if 'type' in c[scan_id].start['scan']:
                scan_type = c[scan_id].start['scan']['type']
            else:
                vprint(f'Scan type of {scan_type} does not have XRD for scan ID {scan_id}.')
                scan_id += 1
                continue
            
            if scan_type == 'XRF_FLY':
                # First, is this a rocking curve
                if c[scan_id].start['scan']['fast_axis'] == 'nano_stage_th':
                    rsm = standard_process_rsm(scan_id,
                                               wd,
                                               **kwargs)
                    scan_id += 1
                    continue
                
                # Second check if shape looks like a dark-field
                elif c[scan_id].start['scan']['shape'] == [5, 5]:
                    ostr = (f'Scan {scan_id} is probably a dark-field. '
                            + 'Generating and saving composite pattern.')
                    print(ostr)
                    save_composite_pattern(scan_id,
                                           broker='manual',
                                           wd=f'{wd}dark_fields/',
                                           method='median')
                    scan_id += 1
                    continue
                
                else:
                    xdm = standard_process_xdm(scan_id,
                                               wd,
                                               **kwargs)
                    scan_id += 1
                    continue
            
            elif scan_type == 'ENERGY_RC':
                rsm = standard_process_rsm(scan_id,
                                           wd,
                                           **kwargs)
                scan_id += 1
                continue
                

            elif scan_type == 'ANGLE_RC':
                rsm = standard_process_rsm(scan_id,
                                           wd,
                                           **kwargs)
                scan_id += 1
                continue

            else:
                if scan_type is not None:
                    err_str = (f'Unknown scan type of {scan_type} for '
                    + f'scan ID {scan_id}.\nSkipping and moving to '
                    + 'next scan ID.')
                    print(err_str)
                scan_id += 1
                continue
                
        except KeyboardInterrupt:
            print(f'Stopping auto processing on scan {scan_id}.')
            break
        
        except Exception as e:
            print(f'Auto processing failed for scan {scan_id}.\n{e}')
            scan_id += 1
            # continue
            break

    