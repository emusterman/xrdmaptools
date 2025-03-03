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


def process_leaf_xrd():

    base_wd = '/nsls2/data/srx/proposals/2025-1/pass-317736/'
    dark_field = io.imread(f'{base_wd}/scan164296_dexela_median_composite.tif')
    air_scatter = io.imread(f'{base_wd}/scan164294_dexela_median_composite.tif')

    # new_hdf = h5py.File(f'{base_wd}scan{np.min(scanlist)}-{np.max(scanlist)}_selected_images.h5', 'w+')

    scanlist = [
        # 164298,
        # 164299,
        # 164300,
        # 164301,
        # 164302,
        # 164303,
        # 164304,
        # 164305,
        # 164306,
        # 164307,
        # 164308,
        # 164309,
        # 164310,
        # 164311,
        164312,
        164313,
        164314,
        164315,
        164316,
        164317,
        164318,
        164319
    ]

    for scan in timed_iter(scanlist):

        xdm = XRDMap.from_hdf(f'scan{scan}_xrdmap.h5', wd=f'{base_wd}')
        xdm.correct_dark_field(dark_field)
        xdm.correct_air_scatter(air_scatter)
        xdm.normalize_scaler()
        xdm.correct_outliers(tolerance=15)
        xdm.set_calibration('scan164261_dexela_calibration.poni', wd=base_wd)
        xdm.load_xrfmap()

        selected_images = xdm.images[xdm.xrf['Ca_K'] / xdm.xrf['i0'] > 0.075]
        io.imsave(f'{base_wd}scan{scan}_selected_images.tif', selected_images)

        del selected_images
        del xdm


def load_selected_images():

    base_wd = '/nsls2/data/srx/proposals/2025-1/pass-317736/'

    scanlist = [
        164298,
        164299,
        164300,
        164301,
        164302,
        164303,
        164304,
        164305,
        164306,
        164307,
        164308,
        164309,
        164310,
        164311,
        164312,
        164313,
        164314,
        164315,
        164316,
        164317,
        164318,
        164319
    ]

    all_images = []

    for scan in scanlist:
        all_images.append(io.imread(f'{base_wd}scan{scan}_selected_images.tif'))

    return np.vstack(all_images)




def load_xrf():

    base_wd = '/nsls2/data/srx/proposals/2025-1/pass-317736/'
    
    scanlist = [
        164298,
        164299,
        164300,
        164301,
        164302,
        164303,
        164304,
        164305,
        164306,
        164307,
        164308,
        164309,
        164310,
        164311,
        164312,
        164313,
        164314,
        164315,
        164316,
        164317,
        164318,
        164319
    ]

    all_xrf = []

    for scan in scanlist:

        xrf_name =  f'scan2D_{scan}_xs_sum8ch'    
        xrf_path = pathify(base_wd, xrf_name, '.h5')

        # Load the data
        xrf = {}
        with h5py.File(xrf_path, 'r') as f:
            
            if 'xrf_fit_name' in f['xrfmap/detsum'].keys():
                xrf_fit_names = [d.decode('utf-8')
                                    for d
                                    in f['xrfmap/detsum/xrf_fit_name'][:]]
                xrf_fit = f['xrfmap/detsum/xrf_fit'][:]

                scaler_names = [d.decode('utf-8')
                                for d
                                in f['xrfmap/scalers/name'][:]]
                scalers = np.moveaxis(f['xrfmap/scalers/val'][:], -1, 0)

                for key, value in zip(xrf_fit_names + scaler_names,
                                        np.vstack([xrf_fit, scalers])):
                    xrf[key] = value

                md_key = 'xrfmap/scan_metadata'
                E0_key = 'instrument_mono_incident_energy'
                xrf['E0'] = f[md_key].attrs[E0_key]

                all_xrf.append(xrf)

    return all_xrf


def get_coarse_scan_positions():

    base_wd = '/nsls2/data/srx/proposals/2025-1/pass-317736/'
    
    scanlist = [
        164298,
        164299,
        164300,
        164301,
        164302,
        164303,
        164304,
        164305,
        164306,
        164307,
        164308,
        164309,
        164310,
        164311,
        164312,
        164313,
        164314,
        164315,
        164316,
        164317,
        164318,
        164319
    ]

    x, y = [], []
    for scan in scanlist:
        bs_run = c[scan]

        x.append(bs_run['baseline']['data']['nano_stage_topx'][0])
        y.append(bs_run['baseline']['data']['nano_stage_y'][0])
    
    return x, y



def plot_all_combined(xrf, xrf_key, x, y, sclr_key='i0'):

    fig, ax = plt.subplots()

    norm = colors.Normalize(vmin=np.min(xrf[xrf_key] / xrf[sclr_key]),
                            vmax=np.max(xrf[xrf_key] / xrf[sclr_key]))

    extents, images = [], []
    for i in range(len(xrf[xrf_key])):
        
        data = xrf[xrf_key][i] / xrf[sclr_key][i]
        shape = np.asarray(data.shape, dtype=float) * 0.5 # hard-coded step size
        
        #print(shape)

        # extent = [x[i] - (shape[1] / 2),
        #           x[i] + (shape[1] / 2),
        #           y[i] - (shape[0] / 2),
        #           y[i] + (shape[0] / 2)]
        
        extent = [x[i] - (shape[1] / 2),
                  x[i] + (shape[1] / 2),
                  y[i] + (shape[0] / 2),
                  y[i] - (shape[0] / 2)]
        
        extents.append(extent)
                  
        # print(extent)
        images.append(ax.imshow(data, extent=extent, norm=norm))

    extents = np.asarray(extents)

    ax.set_xlim(np.min(extents[:, 0]), np.max(extents[:, 1]))
    ax.set_ylim(np.min(extents[:, 3]), np.max(extents[:, 2]))

    def sync_cmaps(changed_image):
        for im in images:
            if changed_image.get_cmap() != im.get_cmap():
                im.set_cmap(changed_image.get_cmap())

    for im in images:
        im.callbacks.connect('changed', sync_cmaps)

    fig.colorbar(images[0], ax=ax)
    #fig.canvas.draw_idle()
    fig.show()





def process_xrd_batch():

    base_wd = '/nsls2/data/srx/proposals/commissioning/pass-315950/Musterman_20250224/'
    dark_field = io.imread(f'{base_wd}/scan164554_dexela_median_composite.tif')

    scanlist = [
        164536,
        164537,
        164538,
        164539,
        164540,
        164541,
        164542,
        164549,
        164550,
        164551,
        164552,
        164553,
        164554,
        164556,
        164558,
        164560
    ]

    for scan in timed_iter(scanlist):

        xdm = XRDMap.from_db(scan, wd=f'{base_wd}')
        xdm.correct_dark_field(dark_field)
        xdm.normalize_scaler()
        xdm.correct_outliers(tolerance=10)
        xdm.set_calibration('scan164261_dexela_calibration.poni', wd=base_wd)
        xdm.apply_polarization_correction()
        xdm.apply_solidangle_correction()
        xdm.estimate_background(method='bruckner')
        xdm.rescale_images()
        xdm.finalize_images()
        xdm.integrate1d_map()
        del xdm


def load_xrf_dict(scan_id, wd):

        xrf_name =  f'scan2D_{scan_id}_xs_sum8ch'    
        xrf_path = pathify(wd, xrf_name, '.h5')

        # Load the data
        xrf = {}
        with h5py.File(xrf_path, 'r') as f:
            
            if 'xrf_fit_name' in f['xrfmap/detsum'].keys():
                xrf_fit_names = [d.decode('utf-8')
                                    for d
                                    in f['xrfmap/detsum/xrf_fit_name'][:]]
                xrf_fit = f['xrfmap/detsum/xrf_fit'][:]

                scaler_names = [d.decode('utf-8')
                                for d
                                in f['xrfmap/scalers/name'][:]]
                scalers = np.moveaxis(f['xrfmap/scalers/val'][:], -1, 0)

                position_names = [d.decode('utf-8')
                                for d
                                in f['xrfmap/positions/name'][:]]
                positions = f['xrfmap/positions/pos'][:]

                for key, value in zip(xrf_fit_names + scaler_names + position_names,
                                        np.vstack([xrf_fit, scalers, positions])):
                    xrf[key] = value

                md_key = 'xrfmap/scan_metadata'
                E0_key = 'instrument_mono_incident_energy'
                xrf['E0'] = f[md_key].attrs[E0_key]

        return xrf


def plot_map(data, x_pos, y_pos, title=None, wd=None, savefig=True):

    fig, ax = plt.subplots()

    extent = [
        np.min(x_pos),
        np.max(x_pos),
        np.max(y_pos),
        np.min(y_pos)
    ]

    im = ax.imshow(data, extent=extent)
    fig.colorbar(im, ax=ax)
    
    ax.set_aspect('equal')
    ax.set_xlabel('x-axis [μm]')
    ax.set_ylabel('y-axis [μm]')
    if title is not None:
        ax.set_title(title)
        figname = title
    else:
        figname = 'figure'
    
    if savefig:
        if wd is not None:
            fig.savefig(f'{wd}{figname}.png')
        else:
            raise ValueError('Must provide wd to save figure.')
    else:
        fig.show()


def quick_figure_processing(scan_id, wd, element_keys):

    xrf = load_xrf_dict(scan_id, wd)

    for el in element_keys:
        plot_map(
            xrf[el] / xrf['i0'],
            xrf['x_pos'],
            xrf['y_pos'],
            title=f'scan{scan_id}_{el}_norm_map',
            wd=f'{wd}figures/',
            savefig=True
        )


def batch_process_xrf_figures():

    wd = '/nsls2/data/srx/proposals/commissioning/pass-315950/Musterman_20250224/'

    scanlist = [
        164522,
        164523,
        164525,
        164526,
        164527,
        164528,
        164529,
        164531,
        164532,
        164533,
        164535,
        164536,
        164537,
        164538,
        164539,
        164540,
        164541,
        164542,
        164543,
        164544,
        164545,
        164546,
        164547,
        164548,
        164549,
        164550,
        164551,
        164552,
        164553,
        164555,
        164556,
        164557,
        164558,
        164559,
        164560,
        164572,
        164576,
        164578,
        164579,
        164580,
        164582,
        164583,
        164585,
        164586,
        164588,
        164589,
        164590,
        164591,
        164592,
        164593,
        164594,
        164595,
        164596,
        164597,
        164598,
        164599,
        164600,
        164601,
        164602,
        164603,
        164604,
        164605,
        164606,
        164607,
        164608,
        164609,
        164610,
        164611,
        164612,
        164613,
        164614,
        164615,
        164616,
        164617,
        164618,
        164619,
        164620,
        164621,
        164622,
        164623,
        164641,
        164642,
        164643
    ]

    for scan in scanlist:
        try:
            quick_figure_processing(scan, wd, ['Fe_K', 'Ge_K'])
        except Exception as e:
            print(f'Scan ID {scan} failed!')
            print(e)

def batch_process_xrf_figures2():

    wd = '/nsls2/data/srx/proposals/commissioning/pass-315950/Musterman_20250224/'

    scanlist = [
        164504,
        164505,
        164506,
        164507,
        164508,
        164509,
        164509,
        164510,
        164511
    ]

    for scan in scanlist:
        try:
            quick_figure_processing(scan, wd, ['Au_L'])
        except Exception as e:
            print(f'Scan ID {scan} failed!')
            print(e)


Ge_roi = slice(977, 997)
Fe_roi = slice(631, 651)


def generate_rois(scan_id,
                  data_slice,
                  data_cutoff,
                  feature_type,
                  analysis_args,
                  move_for_analysis=True,
                  analysis_motors=[DummyMotor(limits=(-52.5, 52.5)), DummyMotor(limits=(-52.5, 52.5))]):
    data = _get_processed_data(c[scan_id], data_slice=data_slice)
    rois = _get_rois(data, data>=data_cutoff, feature_type=feature_type)

    #analysis_motors = [DummyMotor(limits=(-52.5, 52.5)), DummyMotor(limits=(-52.5, 52.5))]
    fast_motor, slow_motor, fast_values, slow_values = _generate_positions_and_motors(c[scan_id])
    # fast_values += 5
    # slow_values += 8

    analysis_args_list, new_positions_list, valid_rois, fixed_rois = [], [], [], []
    for roi in rois:
        out = _generate_analysis_args(
            roi,
            analysis_args,
            fast_values,
            slow_values,
            analysis_motors,
            move_for_analysis,
            False,
            feature_type
        )
        analysis_args_list.append(out[0])
        new_positions_list.append(out[1])
        valid_rois.append(out[2])
        fixed_rois.append(out[3])

    # return analysis_args_list, new_positions_list, valid_rois, fixed_rois

    _plot_analysis_args(c[scan_id].start['scan_id'],
                        data,
                        rois,
                        analysis_args_list,
                        valid_rois,
                        fixed_rois,
                        fast_values,
                        slow_values,
                        new_positions_list,
                        feature_type=feature_type,
                        analysis_motors=analysis_motors)
