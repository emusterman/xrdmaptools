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


def pseudolive_orientation_plots(scan_id, phase):
    raise NotImplementedError()



def base_proc_func(images, map_params, poni_file=None, dark=None, flat=None, scalers=None):


    # Increase data depth
    images = np.asarray(images).astype(np.float16)

    # Quick pre_processing
    if dark is not None:
        if not isinstance(dark, np.ndarray):
            dark = io.imread(dark)
        if images.shape[1:] != dark.shape:
            raise ValueError('Shape of images and dark field do not match.')
        images -= dark.astype(np.float16)
    if flat is not None:
        if not isinstance(flat, np.ndarray):
            flat = io.imread(flat)
        if images.shape[1:] != flat.shape:
            raise ValueError('Shape of images and flat field do not match.')
        images /= flat
    if scalers is not None:
        if not isinstance(scalers, np.ndarray):
            raise ValueError(f'Unknown scaler type {type(scalers)}')
        if images.shape[0] != scalers.shape[0]:
            raise ValueError('Nmber of images and scaler values do not match')
        images /= scalers.reshape(-1, 1, 1)

    # Geometric processing
    if poni_file is not None:
        if isinstance(poni_file, AzimuthalIntegrator):
            ai = poni_file
        else:
            ai = set_calibration(poni_file,
                                map_params['energy'],
                                map_params['image_shape'])

        tth_arr = ai.twoThetaArray().astype(np.float16)
        lorentz = 1 / np.sin(tth_arr / 2)  # close enough
        polar = ai.polarization(factor=0.9).astype(np.float16)
        solidangle_correction = ai.solidAngleArray().astype(np.float16)

        images /= lorentz
        images /= polar
        images /= solidangle_correction

    return images


def stats_proc_func(images,
                    map_params,
                    poni_file=None,
                    dark=None,
                    flat=None,
                    scalers=None): 
    images = base_proc_func(images,
                            map_params,
                            poni_file=poni_file,
                            dark=dark,
                            flat=flat,
                            scalers=scalers)

    proc_data = {}
    proc_data['sum'] = np.sum(images, axis=(1, 2))
    proc_data['max'] = np.max(images, axis=(1, 2))

    return proc_data


def phase_corr_func(images, ai=None, dark=None, flat=None, scalers=None):
    plot_keys = [phase.name for phase in simulated_phases]
    images = base_proc_func(images, ai=ai, dark=dark, flat=flat, scalers=scalers)

    tth_arr = np.degrees(ai.twoThetaArray().astype(np.float16))

    tth_min = np.min

    tth_min = np.min(tth_arr)
    tth_max = np.max(tth_arr)
    tth_num = int(np.round((tth_max - tth_min) / 0.01))

    integrations = np.empty((images.shape[0], tth_num), dtype=np.float16)
    for i, image in enumerate(images):
        integration = ai.integrate1D_ng(image, tth_num,
                                        unit='2th_deg',
                                      correctSolidAngle=False,
                                      polarization_factor=None)
        integrations[i] = integration
       




def base_pseudo_live_plot(scan_id, plot_keys, process_function, process_kwargs={}):

    # Initial processing
    bs_run = c[int(scan_id)]
    map_params, temp_dict = build_map_from_start(bs_run, plot_keys)

    # Initial blank images
    plot_params = {}
    for key in plot_keys:
        map_dict = plot_map(map_params,
                           np.nan * np.empty(map_params['map_shape']),
                           key)
        plot_params[key] = map_dict
    
    plt.ion()
    fig.show()
    #plt.draw()
        
    # Track rows
    finished_rows = []
    UPDATING_PLOTS = True

    while UPDATING_PLOTS:
        new_rows, new_data = load_finished_rows(bs_run, finished_rows, map_params, data_keys=['i0', 'dexela_image'])

        if len(new_rows) > 0:
            for i, new_row in enumerate(new_rows):
                print(f'Found new row ({new_row})! Updating data and plot...')
                # Consolidate new data
                #for key in new_data[i].keys():
                #    temp_dict[key][new_row] = new_data[i][key]

                # Process new data
                # This will change on what needs to be done...
                # Does not currently support simultaneous imaging
                # plot_keys must match the keys of the process_function...
                for det in ['dexela', 'merlin']:
                    if det in temp_dict.keys():
                        if temp_dict[det] is None:
                            map_params[f'image_shape'] = new_data[i][f'{det}_image'].shape[-2:]
                            #temp_dict[det] = np.nan * np.empty((*map_params['map_shape'],
                            #                                    *map_params[f'image_shape']),
                            #                                    dtype=np.float16)
                        
                        if 'scalers' not in process_kwargs.keys():
                            scalers = new_data[i]['i0']
                        else:
                            scalers = proces_kwargs['scalers']

                        #return new_data[i][f'{det}_image'], scalers
                        
                        proc_dict = process_function(new_data[i][f'{det}_image'],
                                                     map_params,
                                                     scalers=scalers,
                                                     **process_kwargs)
                        for key in proc_dict.keys():
                            if key in plot_keys:
                                temp_dict[key][:, new_row] = proc_dict[key]

                # Update finished rows
                finished_rows.append(new_row)      

            for key in plot_params.keys():
                map_dict = plot_map(map_params,
                                    temp_dict[key].T, # Transpose for matplotlib (V, H) but data is (X, Y)
                                    key,
                                    map_dict=plot_params[key])
                plot_params[key] = map_dict

            # Check if already finished
            if len(finished_rows) == map_params['map_shape'][-1]:
                UPDATING_PLOTS = False

        # If nothing was added wait to avoid too many iterations
        elif len(new_data) == 0:
            #print('No new rows found. Sleeping before checking again...')
            ttime.sleep(1)
    
    print(f'Finished pseudo livePlot for scan {map_params["scan_id"]}!')
            
        


def load_finished_rows(bs_run, finished_rows, map_params, data_keys=None):

    # Get all available resource documents
    # r_paths is a dictionary with resource_keys as keys
    data_keys, resource_keys = _get_resource_keys(bs_run,
                                                  data_keys=data_keys) 
    
    #data_keys = ['i0', 'i0_time', 'im', 'it', 'dexela_image']                       
    r_paths = _get_resource_paths(bs_run, resource_keys)
    #return r_paths

    # Check if there are new rows!
    if len(finished_rows) == len(r_paths[resource_keys[0]]):
        return [], [] # Not sure the best way to do this...


    new_rows, new_data = [], []
    # Iterate throgh rows and add unfinished data
    for row in range(len(r_paths[resource_keys[0]])):
        # Skip finished rows
        if row in finished_rows:
            continue

        # Load data
        # Copied most from manual_load_data in db_io. Maybe rewrite into indpendent function
        _empty_lists = [[] for _ in range(len(data_keys))]
        data_dict = {}

        # No implementation for broken rows
        # Goal is to load one row at a time

        r_key = 'SIS_HDF51'
        if r_key in r_paths.keys():
            r_path = r_paths[r_key][row]
            #print('Loading scalers...')
            sclr_keys = [value for value in data_keys if value in ['i0', 'im', 'it', 'sis_time']]
            if 'i0_time' in data_keys and 'sis_time' not in sclr_keys:
                sclr_keys.append('sis_time')
                data_dict['sis_time'] = []
            with h5py.File(r_path, 'r') as f:
                for key in sclr_keys:
                    data_dict[key] = np.array(f[key])
            if 'i0_time' in data_keys:
                data_dict['i0_time'] = [d / 50e6 for d in data_dict['sis_time']] # 50 MHz clock
            if 'sis_time' not in data_keys and 'sis_time' in data_dict.keys():
                del data_dict['sis_time']

        r_key = 'DEXELA_FLY_V1'
        if r_key in r_paths.keys():
            r_path = r_paths[r_key][row]
            #print('Loading dexela...')
            key = 'dexela_image'
            with h5py.File(r_path, 'r') as f:
                data_dict[key] = np.array(f['entry/data/data'])

        # Not sure if this is correct
        r_key = 'MERLIN_FLY_V1'
        if r_key in r_paths.keys():
            r_path = r_paths[r_key][row]
            #print('Loading merlin...')
            key = 'merlin_image'
            with h5py.File(r_path, 'r') as f:
                data_dict[key] = np.array(f['entry/data/data'])

        
        # Find and fix errors
        #return data_dict
        for key in data_dict.keys():
            if len(data_dict[key]) < map_params['map_shape'][0]:
                short_by = map_params['map_shape'][0] - len(data_dict[key])
                # Fixing data by extending the last value out until the end
                #data_dict[key].extend([data_dict[key][-1],] * short_by)
                data_dict[key] = np.vstack([data_dict[key], np.stack([data_dict[key][-1],] * short_by)])
            elif len(data_dict[key]) > map_params['map_shape'][0]:
                raise RuntimeError(f'Got {len(data_dict[key])} / {map_params["map_shape"][0]} data. Something is very wrong!')
            
            #data_dict[key] = np.asarray(data_dict[key])
        
        # If we get this far, should be successful and counted
        new_rows.append(row)
        new_data.append(data_dict)

    return new_rows, new_data



def plot_map(map_params, map_data, map_title, map_dict=None, **kwargs):
    # Generate new figure if this is the first update...
    if map_dict is None:
        fig, ax = plt.subplots()
        ax.set_title(f"scan{map_params['scan_id']} {map_title} map")
        im = ax.imshow(map_data, extent=map_params['map_extent'], **kwargs)
        cbar = fig.colorbar(im, ax=ax)

        map_dict = {
            'fig' : fig,
            'ax' : ax,
            'im' : im,
            'cbar' : cbar
        }
    else:
        vmin = np.nanmin(map_data)
        vmax = np.nanmax(map_data)
        map_dict['im'].set_data(map_data)
        map_dict['im'].set_clim(vmin, vmax)

    #if cbar is not None:
    #    cbar.remove()


    #ax.clear()
    
    #ax.figure.canvas.draw_idle()
    map_dict['fig'].canvas.draw()
    map_dict['fig'].canvas.flush_events()
    #fig.show()  # I am not sure if this is needed beyond the first time...
    return map_dict


# Read start document to build map parameters and empty arrays for plotting
def build_map_from_start(bs_run, plot_keys):
    start_doc = bs_run.start

    # Build map_params
    map_params = {}
    map_params['scan_id'] = start_doc['scan_id']
    map_params['energy'] = start_doc['scan']['energy']
    map_params['map_extent'] = [bs_run.start['scan']['scan_input'][i] for i in [0, 1, 3, 4]]
    map_params['map_shape'] = start_doc['scan']['shape']
    map_params['units'] = start_doc['scan']['fast_axis']['units']
    map_params['detectors'] = start_doc['scan']['detectors']

    # Build scaler temp_dictionaries
    temp_dict = {}
    temp_dict['i0'] = np.nan * np.empty(tuple(map_params['map_shape']), dtype=np.float16)
    temp_dict['im'] = np.nan * np.empty(tuple(map_params['map_shape']), dtype=np.float16)
    temp_dict['it'] = np.nan * np.empty(tuple(map_params['map_shape']), dtype=np.float16)
    temp_dict['i0_time'] = np.nan * np.empty(tuple(map_params['map_shape']), dtype=np.float16)

    # Build place-holders for diffraction
    if 'dexela' in map_params['detectors']:
        temp_dict['dexela'] = None
    if 'merlin' in map_params['detectors']:
        temp_dict['merlin'] = None

    # Initialize plot_keys
    # Plot keys will be used for actually plotting
    for key in plot_keys:
        temp_dict[key] = np.nan * np.empty(tuple(map_params['map_shape']), dtype=np.float16)

    # map_params are plotting inputs
    # temp_dict are empty dicts with appropriate shapes for raw data
    return map_params, temp_dict # Loading xrddatas may be better...




def set_calibration(poni_file, energy, image_shape):
    if isinstance(poni_file, str):
        if not os.path.exists(poni_file):
            raise FileNotFoundError(f"{poni_file} does not exist")

        if poni_file[-4:] != 'poni':
            raise RuntimeError("Please provide a .poni file.")

        print('Setting detector calibration...')
        ai = pyFAI.load(poni_file)
    
    #elif isinstance(poni_file, pyFAI.AzimuthalIntegrator):
        ai = poni_file
    
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
        print('WARNING: Could not find any images to compare calibration!')
        print('Defaulting to detectors settings used for calibration.')

    return ai