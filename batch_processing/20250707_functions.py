import numpy as np
import os
from tqdm import tqdm
from skimage import io
import h5py
import time as ttime
from matplotlib import colors
import matplotlib.pyplot as plt

from scipy.spatial import KDTree

from xrdmaptools.crystal.rsm import map_2_grid


def decorate_vectors(vectors, gridstep=0.01):

    min_int = np.min(vectors[:, -1])

    qx = vectors[:, 0]
    qy = vectors[:, 1]
    qz = vectors[:, 2]

    # Find bounds
    x_min = np.min(qx) - gridstep
    x_max = np.max(qx) + gridstep
    y_min = np.min(qy) - gridstep
    y_max = np.max(qy) + gridstep
    z_min = np.min(qz) - gridstep
    z_max = np.max(qz) + gridstep

    # Generate q-space grid
    xx = np.linspace(x_min, x_max, int((x_max - x_min) / gridstep))
    yy = np.linspace(y_min, y_max, int((y_max - y_min) / gridstep))
    zz = np.linspace(z_min, z_max, int((z_max - z_min) / gridstep))

    grid = np.array(np.meshgrid(xx, yy, zz, indexing='ij'))
    grid = grid.reshape(3, -1).T

    kdtree = KDTree(vectors[:, :3])

    nns = kdtree.query_ball_point(grid, gridstep)

    blank_vectors = []
    for point, nn in zip(grid, nns):
        if len(nn) == 0:
            blank_vectors.append([*point, min_int])

    return np.vstack([vectors, np.asarray(blank_vectors)])

from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def plot_single_isosurface(vectors, gridstep=0.01):

    # Interpolate data for isosurface generation
    (x_grid,
     y_grid,
     z_grid,
     int_grid) = map_2_grid(vectors[:, :3],
                            vectors[:, -1],
                            gridstep=gridstep)

    verts, faces, _, _ = measure.marching_cubes(int_grid, 0, spacing=(gridstep, gridstep, gridstep))

    return verts, faces

    fig = plt.figure(figsize=(5, 5), dpi=200)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2], lw=0)

    ax.set_xlabel('qx [Å⁻¹]')
    ax.set_ylabel('qy [Å⁻¹]', labelpad=10)
    ax.set_zlabel('qz [Å⁻¹]')
    ax.set_aspect('equal')
    
    fig.show()



def plot_phase_and_xrf_maps(xdm):
    from scipy.signal import find_peaks
    from scipy.ndimage import gaussian_filter

    tth, intensity = xdm.integrate1D_image(xdm.max_image)
    gauss_int = gaussian_filter(intensity, sigma=5)
    # gauss_int = intensity

    peaks = find_peaks(gauss_int, prominence=1, width=5)[0]

    phase_maps = {phase_name : np.zeros(xdm.map_shape)
                  for phase_name in xdm.phases.keys()}
    
    all_tth = []
    all_phases = []

    for phase in xdm.phases.values():
        phase.get_hkl_reflections(energy=xdm.energy)
        all_tth.extend(phase.reflections['tth'])
        all_phases.extend([phase.name,] * len(phase.reflections['tth']))
    all_tth = np.asarray(all_tth)

    wind = 20
    for peak in peaks:
        best_match = np.argmin(np.abs(tth[peak] - all_tth))
        tth_cen = np.argmin(np.abs(np.asarray(tth) - all_tth[best_match]))
        phase_maps[all_phases[best_match]] += np.sum(xdm.integrations[..., tth_cen - wind : tth_cen + wind], axis=-1)

    fig, ax = plt.subplots(2, 3, figsize=(12, 6), dpi=300, layout='tight')

    # Element maps
    for i, el_key in enumerate(['Fe', 'Ni', 'Cr']):
        axi = ax[0, i]
        
        im = axi.imshow(xdm.xrf[f'{el_key}_K'] / xdm.xrf['Ar_K'], extent=xdm.map_extent())
        fig.colorbar(im, ax=axi)
        axi.set_title(f'XRF {el_key} K-emission')
        axi.set_xticks([])
        if i != 0:
            axi.set_yticks([])
        else:
            axi.set_ylabel(f'y-axis [{xdm.position_units}]')

    # Phase maps
    for i, phase_key in enumerate(['Hematite', 'Ferrite', 'Austenite']):
        axi = ax[1, i]

        im = axi.imshow(phase_maps[phase_key], extent=xdm.map_extent())
        fig.colorbar(im, ax=axi)
        axi.set_title(f'XRD {phase_key} Sum')
        axi.set_xlabel(f'x-axis [{xdm.position_units}]')
        if i != 0:
            axi.set_yticks([])
        else:
            axi.set_ylabel(f'y-axis [{xdm.position_units}]')
    
    fig.show()


import matplotlib.ticker as mticker
# 161738
def plot_map_and_xrd(xdm):

    tth, intensity = xdm.integrate1D_image(xdm.max_image)
    peaks = find_peaks(intensity, prominence=0.1, width=7, height=0.1)[0]

    # rescale_array(intensity, upper=100)

    fig, ax = plt.subplots(1, 2, dpi=300, figsize=(8, 3), layout='tight')

    # XRD
    ax[0].plot(tth, intensity, c='k')
    ax[0].set_xlabel(f'Scattering Angle, 2θ [{xdm.scattering_units}]')
    # ax[0].set_yticks([])
    ax[0].set_ylabel('Intensity [a.u.]')
    ax[0].set_title('Maximum Integration')
    # ax[0].yaxis.set_major_locator(mticker.MultipleLocator(0.2))

    # for peak in peaks:
    #     ax[0].text(tth[peak], intensity[peak], 'hkl', fontsize=8, color='k')

    # Map
    cen, wind = 526, 20
    im = ax[1].imshow(np.max(xdm.integrations[..., cen - wind : cen + wind], axis=-1), extent=xdm.map_extent(), vmin=0, vmax=0.1)
    fig.colorbar(im, ax=ax[1])
    ax[1].set_xlabel(f'x-axis [{xdm.position_units}]')
    ax[1].set_ylabel(f'y-axis [{xdm.position_units}]')
    ax[1].set_title('(012) Intensity')
    ax[1].set_aspect('equal')

    # asp0 = np.abs(np.diff(ax[0].get_xlim())[0] / np.diff(ax[0].get_ylim())[0])
    # print(f'{asp0=}')
    # asp1 = np.abs(np.diff(ax[1].get_xlim())[0] / np.diff(ax[0].get_ylim())[0])
    # print(f'{asp1=}')
    # ax[0].set_aspect(asp0)

    fig.show()

from xrdmaptools.geometry.geometry import modular_azimuthal_shift, modular_azimuthal_reshift
def plot_azimuth(image, tth_arr, chi_arr, tth_range, chi_range=None, plotme=True, **kwargs):

    tth_mask = (tth_arr >= tth_range[0]) & (tth_arr <= tth_range[1])
    new_chi_arr, new_max_arr, shifted = modular_azimuthal_shift(chi_arr)

    n, bins = np.histogram(new_chi_arr[tth_mask], 
                           weights=image[tth_mask],
                           **kwargs)
    
    if plotme:
        fig, ax = plt.subplots()
        ax.plot(bins[:-1], n)
        fig.show()

    return n, bins, np.max(image[tth_mask])

from scipy.stats import circmean
from xrdmaptools.utilities.math import arbitrary_center_of_mass
from xrdmaptools.utilities.utilities import Iterable2D
def plot_hsv(xdm):

    hsv_plot = np.zeros((*xdm.map_shape, 3))
    
    max_full = np.max(xdm.images)
    print(max_full)

    for indices in tqdm(Iterable2D(xdm.map_shape)):

        # n, bins, max_int = plot_azimuth(xdm.images[indices], xdm.tth_arr, xdm.chi_arr, tth_range=(11.8, 12.3), plotme=False, bins=180)
        n, bins, max_int = plot_azimuth(xdm.images[indices], xdm.tth_arr, xdm.chi_arr, tth_range=(10.15, 10.3), plotme=False, bins=180)
        
        bins -= np.min(bins)

        com = arbitrary_center_of_mass(n, bins[:-1])[0]

        hsv = np.asarray(plt.cm.hsv(com / np.max(bins))[:3])
        # hsv *= (max_int / max_full)

        hsv_plot[indices] = hsv
    
    return hsv_plot



