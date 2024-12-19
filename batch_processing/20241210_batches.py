import numpy as np
import os
from tqdm import tqdm
from skimage import io
import h5py
import time as ttime
from scipy.optimize import curve_fit

# Local imports
from xrdmaptools.XRDRockingCurve import XRDRockingCurve
from xrdmaptools.XRDMap import XRDMap
from xrdmaptools.reflections.spot_blob_indexing_3D import *
from xrdmaptools.reflections.spot_blob_indexing_3D import _get_connection_indices
from xrdmaptools.crystal.strain import *

from xrdmaptools.utilities.utilities import (
    timed_iter
)
from xrdmaptools.utilities.math import compute_r_squared
from xrdmaptools.reflections.SpotModels import (
    GaussianFunctions,
    generate_bounds
)

# from pyxrf.api import *

from tiled.client import from_profile

c = from_profile('srx')




def rsm_batch1():

    base_wd = '/nsls2/data/srx/proposals/2024-2/pass-314118/'
    scanlist = [
        # '156179-156201',
        # '156205-156227',
        # '156229-156251',
        # '156253-156275',
        # '156277-156299',
        # '156301-156323',
        # '156325-156347',
        # '156349-156371',
        # '156373-156395',
        # '156397-156419',
        # '156421-156443',
        # '156445-156467', # new error
        # '156469-156491', # error
        # '156493-156515',
        # '156517-156539',
        # '156541-156563',
        # '156565-156587', # error, # new error
        # '156589-156611', # error, # new error
        # '156613-156635', # new error
        # '156637-156659', # error
        # '156661-156683', # error
        # '156685-156707', # new error
        # '156709-156731', # completely different error
        # '156733-156755',
        # '156757-156775', # error, # new error
        ]


    for scan in timed_iter(scanlist):
        print(f'Batch processing scan {scan}...')
        rsm = XRDRockingCurve.from_hdf(f'scan{scan}_rsm.h5', wd=f'{base_wd}energy_rc/')

        rsm.find_3D_spots(intensity_cutoff=0.075, nn_dist=0.1, significance=1, subsample=1)
        rsm.index_all_spots(0.1, 2.5)

        grain_id = rsm.spots_3D['grain_id'].values[np.argmax(rsm.spots_3D['qof'])]

        q_vectors = rsm.spots_3D[['qx', 'qy', 'qz']][rsm.spots_3D['grain_id'] == grain_id].values
        hkls = rsm.spots_3D[['h', 'k', 'l']][rsm.spots_3D['grain_id'] == grain_id].values.astype(int)
        
        e, u, _ = phase_get_strain_orientation(q_vectors, hkls, rsm.phases['stibnite'])

        np.savetxt(f'{rsm.wd}scan{rsm.scan_id}_coarse_eij2.txt', e)



def process_tomo_maps():

    base_wd = '/nsls2/data/srx/proposals/commissioning/pass-315950/Musterman_20241008/'

    scanlist = list(range(163553, 163644 + 1, 1))
    scanlist.remove(163627)

    dark_field = io.imread(f'{base_wd}scan163645_dexela_median_composite.tif')
    poni_file = 'scan163493_dexela_calibration.poni'
    

    for scan in timed_iter(scanlist):
        print(f'Batch Processing scan {scan}...')

        xdm = XRDMap.from_db(scan, wd=f'{base_wd}xrd_tomo/')

        xdm.correct_dark_field(dark_field)
        xdm.normalize_scaler()
        xdm.correct_outliers(tolerance=5)

        xdm.set_calibration(poni_file, wd=base_wd)
        xdm.apply_polarization_correction()
        xdm.apply_solidangle_correction()

        xdm.estimate_background(method='bruckner', binning=8, min_prominence=0.01)
        xdm.rescale_images(arr_max=xdm.estimate_saturated_pixel())
        xdm.nullify_images()

        xdm.finalize_images()
        xdm.integrate1d_map()

        xdm.find_blobs()
        # xdm.find_spots(radius=5)


# def process_tomo_xrf():

#     base_wd = '/nsls2/data/srx/proposals/commissioning/pass-315950/Musterman_20241008/'

#     os.chdir(f'{base_wd}xrd_tomo/')

#     scanlist = list(range(163553, 163644 + 1, 1))
#     scanlist.remove(163627)

#     param_file = f'{base_wd}pyxrf_model_parameters_163553.json'

#     for scan in timed_iter(scanlist):
#         print(f'Batch Processing scan {scan}...')

#         make_hdf(scan)
#         fit_pixel_data_and_save(f'{base_wd}xrd_tomo/',
#                                 f'scan2D_{scan}_xs_sum8ch.h5',
#                                 param_file_name=param_file)


def open_elements(el_key, data_files, wd):

    fit_list = []
    scaler_list = []

    for file in tqdm(data_files):

        f = h5py.File(wd + file)

        xrf = {}

        xrf_fit_names = [d.decode('utf-8')
                            for d
                            in f['xrfmap/detsum/xrf_fit_name'][:]]
        xrf_fit = f['xrfmap/detsum/xrf_fit'][:]

        scaler_names = [d.decode('utf-8')
                        for d
                        in f['xrfmap/scalers/name'][:]]
        scalers = np.moveaxis(f['xrfmap/scalers/val'][:], -1, 0)

        # i0 = f['xrfmap/scalers/val'][..., 0]
        # xrf_fit = np.concatenate((xrf_fit,
        #                           np.expand_dims(i0, axis=0)),
        #                           axis=0)
        # xrf_fit_names.append('i0')

        for key, value in zip(xrf_fit_names + scaler_names,
                                np.vstack([xrf_fit, scalers])):
            xrf[key] = value
        
        fit_list.append(xrf[el_key])
        scaler_list.append(xrf['i0'])

    return np.asarray(fit_list), np.asarray(scaler_list)



def process_broken_tomo_maps():

    base_wd = '/nsls2/data/srx/proposals/commissioning/pass-315950/Musterman_20241008/'

    scanlist = list(range(163495, 163495 + 47, 1)) # up to 45 deg


    dark_field = io.imread(f'{base_wd}scan163494_dexela_median_composite.tif')
    poni_file = 'scan163493_dexela_calibration.poni'
    

    for scan in timed_iter(scanlist):
        print(f'Batch Processing scan {scan}...')

        xdm = XRDMap.from_db(scan, wd=f'{base_wd}broken_xrd_tomo/')

        xdm.correct_dark_field(dark_field)
        xdm.normalize_scaler()
        xdm.correct_outliers(tolerance=5)

        xdm.set_calibration(poni_file, wd=base_wd)
        xdm.apply_polarization_correction()
        xdm.apply_solidangle_correction()

        xdm.estimate_background(method='bruckner', binning=8, min_prominence=0.01)
        xdm.rescale_images(arr_max=xdm.estimate_saturated_pixel())
        xdm.nullify_images()

        xdm.finalize_images()
        xdm.integrate1d_map()

        xdm.find_blobs()
        # xdm.find_spots(radius=5)


def get_blob_integrations():
    
    base_wd = '/nsls2/data/srx/proposals/commissioning/pass-315950/Musterman_20241008/'

    scanlist = list(range(163553, 163644 + 1, 1))
    scanlist.remove(163627)

    all_blob_ints = []

    for scan in timed_iter(scanlist):
        print(f'Batch Processing scan {scan}...')

        xdm = XRDMap.from_hdf(f'scan{scan}_xrdmap.h5', wd=f'{base_wd}xrd_tomo/')

        int_map, tth, extent, tth_res = xdm.integrate1d_map(
                        mask=xdm.blob_masks,
                        return_values=True)

        all_blob_ints.append(int_map)
    
    return all_blob_ints



def multi_phase_fit(tth, intensity, phases):

    for phase in phases:
        phase.get_hkl_reflections(
                        energy=18,
                        tth_range=(np.min(tth), np.max(tth)))

    p0 = [np.median(intensity)]
    peak_identities = ['offset']

    for phase in phases:
        for i in range(len(phase.reflections['tth'])):
            tth_i = phase.reflections['tth'][i]
            tth_ind = (np.abs(tth - tth_i)).argmin()
            amp = intensity[tth_ind] - p0[0]
            if amp <= 0:
                amp = 0.001

            p0.append(amp)
            p0.append(tth_i) # tth
            p0.append(0.2) # sigma guess
            peak_identities.extend([f'{phase.name}_{phase.reflections["hkl"][i]}',] * 3)

    bounds = generate_bounds(p0[1:], GaussianFunctions.func_1d, tth_step=np.diff(tth)[0])
    bounds[0].insert(0, np.min(intensity)) # offset lower bound
    bounds[1].insert(0, np.max(intensity)) # offset upper bound

    return p0, bounds

    popt, _ = curve_fit(GaussianFunctions.multi_1d, tth, intensity, p0=p0, bounds=bounds)
    r_squared = compute_r_squared(intensity, GaussianFunctions.multi_1d(tth, *popt))

    return popt, r_squared


def get_windowed_integrations(tth, integrations, phases, window=0.25):

    for phase in phases:
        phase.get_hkl_reflections(
                        energy=18,
                        tth_range=(np.min(tth), np.max(tth)))

    wind_st, wind_en = [], []
    peak_identities = []

    for phase in phases:
        for i in range(len(phase.reflections['tth'])):
            tth_i = phase.reflections['tth'][i]
            tth_ind = (np.abs(tth - tth_i)).argmin()

            tth_st = tth_i - window / 2
            tth_en = tth_i + window / 2

            wind_st.append((np.abs(tth - tth_st)).argmin())
            wind_en.append((np.abs(tth - tth_en)).argmin())

            peak_identities.append(f'{phase.name}_{phase.reflections["hkl"][i]}')

    map_shape = integrations.shape[:2]
    peak_int_map = np.zeros((len(peak_identities), *map_shape))

    for index in tqdm(range(np.prod(map_shape))):
        indices = np.unravel_index(index, map_shape)
        for i in range(len(peak_identities)):
            peak_int_map[(i, *indices)] = np.sum(integrations[indices][wind_st[i] : wind_en[i]])

    return peak_int_map, peak_identities


def window_integrate_all_maps():

    base_wd = '/nsls2/data/srx/proposals/commissioning/pass-315950/Musterman_20241008/'

    scanlist = list(range(163553, 163644 + 1, 1))
    scanlist.remove(163627)

    all_peak_maps = []

    for scan in timed_iter(scanlist):
        print(f'Batch Processing scan {scan}...')

        xdm = XRDMap.from_hdf(f'scan{scan}_xrdmap.h5', wd=f'{base_wd}xrd_tomo/', image_data_key=None, save_hdf=False)
        xdm.load_phase('Aluminum_0011136.cif', wd='/nsls2/users/emusterma/Documents/cif', phase_name='aluminum')
        xdm.load_phase('Copper_0011145.cif', wd='/nsls2/users/emusterma/Documents/cif', phase_name='copper')

        peak_int_map, peak_identities = get_windowed_integrations(xdm.tth, xdm.integrations, list(xdm.phases.values()))

        all_peak_maps.append(peak_int_map)
    
    return all_peak_maps, peak_identities


def interpolate_shifts(image_stack, shifts):

    interp_list = []
    y_arr, x_arr = np.meshgrid(*[np.array(range(axis), dtype=np.float32) for axis in image_stack[0].shape], indexing='ij')


    for i in range(len(image_stack)):
        out = griddata(np.array([x_arr.ravel() + shifts[i][1], y_arr.ravel() + shifts[i][0]]).T,
                    image_stack[i].ravel(),
                    np.array([x_arr.ravel(), y_arr.ravel()]).T, 
                    method='linear')
        interp_list.append(out.reshape(xdm.map_shape))
    
    interp_list = np.asarray(interp_list)
    interp_list[np.isnan(interp_list)] = 0

    return interp_list


def fit_rotation(th, data):

    def rot_func(th, r, th0, x0):
        return r * np.cos(th + th0) + x0

    jitter_list = []
    popt_list = []

    for ind in range(71):
        exp = np.asarray([arbitrary_center_of_mass(data[i, ind], range(91)) for i in range(91)]).squeeze()
        
        # low = np.argmin(np.gradient(data[:, ind].astype(np.float32), axis=1), axis=1)
        # high = np.argmax(np.gradient(data[:, ind].astype(np.float32), axis=1), axis=1)
        # exp = np.mean([low, high], axis=0)

        popt, _ = curve_fit(rot_func, th, exp, bounds=[[-np.inf, -2*np.pi, -np.inf],
                                                    [np.inf, 2*np.pi, np.inf]])

        pred = rot_func(th, *popt)

        jitter_list.append(pred - exp)
        popt_list.append(popt)
    
    return np.asarray(jitter_list), np.asarray(popt_list)




# class Projected(PseudoPositioner):

#     proj_x = Cpt(PsuedoSingle)
#     proj_z = 
