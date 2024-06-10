import numpy as np
import os
import time as ttime
from itertools import product
from skimage import io
import h5py
from scipy.stats import mode
import matplotlib.pyplot as plt

import pyFAI
from pyFAI.io import ponifile
from enum import IntEnum # Only for ponifile orientation
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator


# Working at the beamline...
try:
    print('Connecting to databrokers...', end='', flush=True)
    from tiled.client import from_profile
    from databroker.v1 import Broker

    c = from_profile('srx')
    db = Broker.named('srx')
    print('done!')
except ModuleNotFoundError:
    print('failed.')
    pass
from tqdm import tqdm
import matplotlib.pyplot as plt

# Local imports
import xrdmaptools as xmt
from xrdmaptools.XRDMap import XRDMap
from xrdmaptools.utilities.db_io import (
    _get_resource_keys,
    _get_resource_paths,
    _flag_broken_rows,
    _repair_data_dict
) 


def pseudolive_orientation_plots(scanid, phase):
    raise NotImplementedError()


def base_proc_func(images, ai=None, dark=None, flat=None, scalers=None):

    # Check inputs
    if images.shape[1:] != dark.shape:
        raise ValueError('Shape of images and dark field do not match.')
    if images.shape[1:] != flat.shape:
        raise ValueError('Shape of images and flat field do not match.')
    if images.shape[0] != scalers.shape:
        raise ValueError('Nmber of images and scaler values do not match')

    # Increase data depth
    images = np.asarray(images).astype(np.float32)

    # Quick pre_processing
    if dark is not None:
        images -= dark.astype(np.float32)
    if flat is not None:
        images /= flat
    if scalers is not None:
        images /= scalers

    # Geometric processing
    if ai is not None:
        tth_arr = ai.twoThetaArray().astype(np.float32)
        lorentz = 1 / np.sin(tth_arr / 2)  # close enough
        polar = ai.polarization(factor=0.9).astype(np.float32)
        solidangle_correction = ai.solidAngleArray().astype(np.float32)

        images /= lorentz
        images /= polar
        images /= solidangle_correction

    return images


def stats_proc_func(images, ai=None, dark=None, flat=None, scalers=None): 
    images = base_proc_func(images, ai=ai, dark=dark, flat=flat, scalers=scalers)

    proc_data = {}
    proc_data['sum'] = np.sum(images, axis=0)
    proc_data['max'] = np.max(images, axis=0)

    return proc_data


def phase_corr_func(images, simulated_phases, ai=None, dark=None, flat=None, scalers=None):
    images = base_proc_func(images, ai=ai, dark=dark, flat=flat, scalers=scalers)

    tth_arr = np.degrees(ai.twoThetaArray().astype(np.float32))

    tth_min = np.min

    tth_min = np.min(tth_arr)
    tth_max = np.max(tth_arr)
    tth_num = int(np.round((tth_max - tth_min) / 0.01))

    integrations = np.empty((images.shape[0], tth_num), dtype=np.float32)
    for i, image in enumerate(images):
        integration = ai.integrate1d_ng(image, tth_num,
                                        unit='2th_deg',
                                      correctSolidAngle=False,
                                      polarization_factor=None)
        integrations[i] = integration


        




def base_pseudolive_plot(scanid, plot_keys, process_fuction):

    # Initial processing
    bs_run = c[int(scanid)]
    map_params, plot_dicts = build_map_from_start(bs_run, plot_keys)

    # Initial blank images
    figax = {}
    for key in plot_keys:
        fig, ax = plot_map(map_params,
                           np.empty(map_params['shape']),
                           key)
        figax['key'] = [fig, ax]
        
    # Track rows
    finished_rows = []
    UPDATING_PLOTS = True

    while UPDATING_PLOTS:
        new_rows, new_data = load_finished_rows(bs_run, finished_rows)

        for i, new_row in enumerate(new_rows):
            # Consolidate new data
            for key in new_data.keys():
                plot_dicts[key][new_row] = new_data[i][key]

            # Process new data
            # This will change on what needs to be done...
            # Does not currently support simultaneous imaging
            # plot_keys must match the keys of the process_function...
            for det in ['dexela', 'merlin']:
                if det in plot_dicts.keys():
                    if plot_dicts[det] is None:
                        plot_dicts[det] = np.empty((*map_params['shape'],
                                                        *new_data[i][f'{det}_image'].shape),
                                                        dtype=np.float32)
                    proc_data = process_fuction(new_data[i][f'{det}_image'])
                    for key in proc_data:
                        plot_dicts[key][new_row] = proc_data
        
        # Update and plot_figures
        figax = {}
        for key in plot_keys:
            fig, ax = plot_map(map_params,
                            np.empty(map_params['shape'],
                                     dtype=np.float32),
                            key,
                            fig=figax[key][0],
                            ax=figax[key][1])
        
        # Update finished rows
        finished_rows.append(new_row)

        # Check if already finished
        if len(finished_rows) == map_params['shape'][-1]:
            UPDATING_PLOTS = False

        # If nothing was added wait to avoid too many iterations
        elif len(new_data) == 0:
            ttime.sleep(0.1)
            
        


def load_finished_rows(bs_run, finished_rows, map_params):

    # Get all available resource documents
    # r_paths is a dictionary with resource_keys as keys
    data_keys, resource_keys, xrd_dets = _get_resource_keys(bs_run)                             
    r_paths = _get_resource_paths(bs_run, resource_keys)

    # Check if there are new rows!
    if len(finished_rows) == len(r_paths[resource_keys[0]]):
        return None, None # Not sure the best way to do this...


    new_rows, new_data = [], []
    # Iterate throgh rows and add unfinished data
    for row in range(len(r_paths[resource_keys[0]])):
        # Skip finished rows
        if row in finished_rows:
            continue

        # Load data
        # Copied most from manual_load_data in db_io. Maybe rewrite into indpendent function
        _empty_lists = [[] for _ in range(len(data_keys))]
        data_dict = dict(zip(data_keys, _empty_lists))

        # No implementation for broken rows
        # Goal is to load one row at a time

        r_key = 'SIS_HDF51'
        if r_key in r_paths.keys():
            #print('Loading scalers...')
            sclr_keys = [value for value in data_keys if value in ['i0', 'im', 'it', 'sis_time']]
            if 'i0_time' in data_keys and 'sis_time' not in sclr_keys:
                sclr_keys.append('sis_time')
                data_dict['sis_time'] = []
            for r_path in r_paths[r_key]:
                with h5py.File(r_path, 'r') as f:
                    for key in sclr_keys:
                        data_dict[key].append(np.array(f[key]))
            if 'i0_time' in data_keys:
                data_dict['i0_time'] = [d / 50e6 for d in data_dict['sis_time']] # 50 MHz clock
            if 'sis_time' not in data_keys and 'sis_time' in data_dict.keys():
                del data_dict['sis_time']

        r_key = 'DEXELA_FLY_V1'
        if r_key in r_paths.keys():
            #print('Loading dexela...')
            key = 'dexela_image'
            for r_path in r_paths[r_key]:
                with h5py.File(r_path, 'r') as f:
                    data_dict[key].append(np.array(f['entry/data/data']))

        # Not sure if this is correct
        r_key = 'MERLIN_FLY_V1'
        if r_key in r_paths.keys():
            #print('Loading merlin...')
            key = 'merlin_image'
            for r_path in r_paths[r_key]:
                with h5py.File(r_path, 'r') as f:
                    data_dict[key].append(np.array(f['entry/data/data']))

        
        # Convert data to arrays and fix any errors
        for key in data_dict.keys():
            if len(data_dict[key]) < map_params.shape[1]:
                short_by = map_params.shape[1] - map_params.shape[1]
                # Fixing data by extending the last value out until the end
                data_dict[key].extend([data_dict[key][-1],] * short_by)
            elif len(data_dict[key]) > map_params.shape[1]:
                raise RuntimeError('Got more data than expected. Something is very wrong!')
            
            data_dict[key] = np.asarray(data_dict[key])
        
        # If we get this far, should be successful and counted
        new_rows.append(row)
        new_data.append(data_dict)

    return new_rows, new_data



def plot_map(map_params, map_data, map_title, fig=None, ax=None, **kwargs):
    raise NotImplementedError()
    # Generate new figure if this is the first update...
    if fig is None and ax is None:
        fig, ax = plt.subplots()

    #ax.clear()
    ax.set_title(f"scan{map_params['scanid']} {map_title} map")
    im = ax.imshow(map_data, extent=map_params['extent'], **kwargs)
    fig.colorbar(im, ax=ax)

    fig.canvas.draw()
    fig.show()  # I am not sure if this is needed beyond the first time...


# Read start document to build map parameters and empty arrays for plotting
def build_map_from_start(bs_run, plot_keys):
    raise NotImplementedError()

    start_dict = bs_run.start

    # Build map_params
    map_params = {}
    map_params['scanid'] = start_dict['scan_id']
    map_params['energy'] = start_dict['scan']['energy']
    scan_input = start_dict['scan']['scan_input']
    map_params['map_extent'] = list(np.array(scan_input)[0, 1, 3, 4]) # Horrible way to do this
    map_params['map_shape'] = start_dict['scan']['shape']
    map_params['units'] = start_dict['scan']['fast_axis']['units']
    map_params['detectors'] = start_dict['detectors']

    # Build scalar plot_dictionaries
    plot_dicts['i0'] = np.empty(tuple(map_params['shape']), dtype=np.float32)
    plot_dicts['im'] = np.empty(tuple(map_params['shape']), dtype=np.float32)
    plot_dicts['it'] = np.empty(tuple(map_params['shape']), dtype=np.float32)
    plot_dicts['i0_time'] = np.empty(tuple(map_params['shape']), dtype=np.float32)

    # Build place-holders for diffraction
    if 'dexela' in map_params['detectors']:
        plot_dicts['dexela'] = None
    if 'merlin' in map_params['detectors']:
        plot_dicts['merlin'] = None

    # Initialize plot_keys
    # Plot keys will be used for actually plotting
    for key in plot_keys:
        plot_dicts[key] = np.empty(tuple(map_params['shape']), dtype=np.float32)

    # map_params are plotting inputs
    # plot_dicts are empty dicts with appropriate shapes for raw data
    return map_params, plot_dicts # Loading ImageMaps may be better...




def set_calibration(poni_file, filedir, image_shape):
    if isinstance(poni_file, str):
        if not os.path.exists(f'{filedir}{poni_file}'):
            raise FileNotFoundError(f"{filedir}{poni_file} does not exist")

        if poni_file[-4:] != 'poni':
            raise RuntimeError("Please provide a .poni file.")

        print('Setting detector calibration...')
        ai = pyFAI.load(f'{filedir}{poni_file}')
    
    else:
        raise TypeError(f"{type(poni_file)} is unknown and not supported!")

    if ai.detector.shape != image_shape:
        print('Calibration performed under different settings. Adjusting calibration.')

        # Exctract old values
        poni_shape = ai.detector.shape
        poni_pixel1 = ai.detector.pixel1
        poni_pixel2 = ai.detector.pixel2

        bin_est = np.array(image_shape) / np.array(poni_shape)
        # Forces whole number binning, either direction
        # This would prevent custom resizing for preprocessing
        if all([any(bin_est != np.round(bin_est, 0)),
                any(1 / bin_est != np.round(1 / bin_est, 0))]):
            err_str = ("Calibration file was performed with an "
                        + "image that is not an integral multiple "
                        + "of the current map's images."
                        + "\n\t\tEnsure the calibration is for the "
                        + "correct detector with the appropriate binning.")
            raise ValueError(err_str)

        # Overwrite values
        ai.detector.shape = image_shape
        ai.detector.max_shape = image_shape # Not exactly correct, but more convenient
        ai.detector.pixel1 = poni_pixel1 / bin_est[0]
        ai.detector.pixel2 = poni_pixel2 / bin_est[1]

    else:
        print('Warning: Could not find any images to compare calibration!')
        print('Defaulting to detectors settings used for calibration.')

    return ai