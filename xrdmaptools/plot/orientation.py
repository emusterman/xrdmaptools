import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import orix
from orix import plot as orix_plot
from matplotlib import color_sequences

# Local imports
from xrdmaptools.plot import config
from xrdmaptools.plot.general import plot_image_gallery
from xrdmaptools.plot.volume import plot_3D_scatter
from xrdmaptools.utilities.math import rescale_array
from xrdmaptools.geometry.geometry import (
    _parse_rotation_input
)

# This module is intended for plotting/visualizing orientation information
# This will likely make heavy use of the orix module


#################################
### Single Pixel Orientations ###
#################################


def plot_3D_indexing(indexings,
                     all_spot_qs,
                     all_spot_ints,
                     all_ref_qs,
                     all_ref_hkls,
                     all_ref_fs,
                     qmask,
                     edges=[],
                     c_seq=None,
                     title=None,
                     **kwargs,
                     ):
    
    # All numpy arrays
    all_spot_qs = np.asarray(all_spot_qs)
    all_spot_ints = np.asarray(all_spot_ints)
    all_ref_qs = np.asarray(all_ref_qs)
    all_ref_hkls = np.asarray(all_ref_hkls)
    all_ref_fs = np.asarray(all_ref_fs)
    
    # Defaults
    kwargs.setdefault('s', 1)

    # Indexing colors
    if c_seq is None:
        c_seq = color_sequences['tab10']
    
    # Transparency based on intensity
    alpha = all_spot_ints.copy()
    rescale_array(alpha, lower=0.5)
    colors = [(0, 0, 0, a) for a in alpha]

    fig, ax = plot_3D_scatter(all_spot_qs,
                              colors=colors,
                              alpha=None,
                              edges=edges,
                              label='Measured',
                              **kwargs)
    
    # Plot part of othe reciprocal lattice
    for i, indexing in enumerate(indexings):
        spot_inds, ref_inds = np.asarray(indexing).T
        orientation = Rotation.align_vectors(
                        all_spot_qs[spot_inds],
                        all_ref_qs[ref_inds])[0]
        all_rot_qs = orientation.apply(all_ref_qs)
        temp_qmask = qmask.generate(all_rot_qs)

        alpha = all_ref_fs[temp_qmask].copy()
        rescale_array(alpha, lower=0.5)
        color = [(*c_seq[i], a) for a in alpha]

        # Plot reference lattice
        ax.scatter(*all_rot_qs[temp_qmask].T,
                   alpha=None,
                   color=color,
                   label=f'Grain ID {i}',
                   **kwargs)
        
        # Label indexed measured spots
        uvw = []
        for spot, ref, hkl in zip(all_spot_qs[spot_inds],
                                  all_rot_qs[ref_inds],
                                  all_ref_hkls[ref_inds]):
            uvw.append(ref - spot)
            ax.text(*spot.T,
                    str(tuple([int(index) for index in hkl])),
                    fontsize=8,
                    c=c_seq[i])

        ax.quiver(*all_spot_qs[spot_inds].T,
                  *np.asarray(uvw).T,
                  arrow_length_ratio=0,
                  linewidth=0.25,
                  color=c_seq[i])

    ax.legend(prop={'size': 8})
    ax.set_title(title)

    return fig, ax


def plot_unit_cell(phase,
                   orientation=None,
                   degrees=False):
    # Re-write of phase.show_unitcell
    if orientation is None:
        orientation = np.eye(3)

    ori = _parse_rotation_input(
                    orientation,
                    'orientation',
                    degrees=degrees)

    # Sample axes
    s_axes = np.asarray([
        ((0, 0, 0), (1.0, 0, 0)), # x-axis
        ((0, 0, 0), (0, 1.0, 0)), # y-axis
        ((0, 0, 0), (0, 0, 1.0))  # z-axis
    ])
    s_axes *= 1.15 * np.max([phase.a, phase.b, phase.c])
    
    # Unit cell edges
    # a1, a2, a3 = LatticeParameters.from_Phase(phase).Amat.T
    a1, a2, a3 = phase.a1, phase.a2, phase.a3
    # print(a1, a2, a3)

    # [First point (x, y, z), second point]
    edges = np.asarray([
        ((0, 0, 0), a1), # basis vectors
        ((0, 0, 0), a2), # basis vectors
        ((0, 0, 0), a3), # basis vectors
        (a1, a1 + a2),
        (a1, a1 + a3),
        (a2, a2 + a1),
        (a2, a2 + a3),
        (a3, a3 + a1),
        (a3, a3 + a2),
        (a1 + a2, a1 + a2 + a3),
        (a1 + a3, a1 + a2 + a3),
        (a2 + a3, a1 + a2 + a3)
    ])

    edges[:, 0] = ori.apply(edges[:, 0], inverse=True)
    edges[:, 1] = ori.apply(edges[:, 1], inverse=True)

    fig, ax = plt.subplots(subplot_kw={'projection':'3d'})

    for line, label in zip(s_axes, ['x', 'y', 'z']):
        ax.quiver(*line.ravel(), color='k', lw=2)
        ax.text(*line[1] * 1.05, f'{label}-axis', zdir=tuple(line[1]), c='k', fontsize=12)

    for line, label, c in zip(edges[:3],
                                  ['a', 'b', 'c'],
                                  [[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
        ax.quiver(*line.ravel(), color=c, lw=2.5)
        ax.text(*line[1] * 1.05, f'{label}-axis', zdir=tuple(line[1]), c=c, fontsize=12)

    for line in edges[3:]:
        ax.plot(*line.T, color='gray', lw=1)

    ax.set_xlabel('x [Å]')
    ax.set_ylabel('y [Å]')
    ax.set_zlabel('z [Å]')
    ax.set_aspect('equal')

    fig.show()


###########################
### Orientation Mapping ###
###########################

# Utility functions

# TODO: Generalize, move elsewhere, link to other use cases
def pixel_spots(spots_3D, indices):
    return spots_3D[((spots_3D['map_x'] == indices[1]) & (spots_3D['map_y'] == indices[0]))]


def map_shape_from_spots(spots_3D):
    return (np.max(spots_3D['map_y']) + 1,
            np.max(spots_3D['map_x']) + 1)


# Build functions

def build_mis_mag_map(ori_map, ref=None):

    mag_map = np.zeros(ori_map.shape[:2])

    if ref is None:
        ref = np.eye(3)
    elif isinstance(ref, Rotation):
        ref = ref.as_matrix()

    for index in tqdm(range(np.prod(mag_map.shape))):
        indices = np.unravel_index(index, mag_map.shape)
        mag_map[indices] = np.degrees(Rotation.from_matrix(ref @ ori_map[indices].T).magnitude())
    
    return mag_map


def build_grain_map(spots_3D, grain_id, color=None, intensity=False, metric='int_corr'):

    map_shape = map_shape_from_spots(spots_3D)
        
    if intensity and metric not in spots_3D:
        raise KeyError(f'{metric} not in spots_3D')

    if color is None:
        grain_map = np.empty(map_shape)
    else:
        grain_map = np.empty((*map_shape, *np.asarray(color).shape))
    int_map = np.zeros(map_shape)
    grain_map[:] = np.nan
    int_map[:] = np.nan
    
    for index in range(np.prod(map_shape)):
        indices = np.unravel_index(index, map_shape)
        df = pixel_spots(spots_3D, indices)
        if grain_id in df['grain_id'].values:
            grain_mask = df['grain_id'] == grain_id
            if intensity:
                int_map[indices] = np.sum(df[grain_mask][metric].values)
            if color is None:
                grain_map[indices] = True
            else:
                grain_map[indices] = color
    
    # int_map /= np.nanmax(int_map)
    
    if intensity:
        if color:
            if len(color) == 3:
                grain_map = np.concatenate((grain_map, int_map[:, :, np.newaxis]), axis=-1)
            elif len(color) == 4:
                grain_map[:, :, -1] = int_map
        else:
            grain_map = int_map

    return grain_map


def build_ori_map(spots_3D, phase, grain_id=0):

    map_shape = map_shape_from_spots(spots_3D)
    ori_map = np.empty((*map_shape, 3, 3))

    for index in tqdm(range(np.prod(map_shape))):
        indices = np.unravel_index(index, map_shape)
        pixel_df = pixel_spots(spots_3D, indices)
        grain_mask = pixel_df['grain_id'] == grain_id
        if sum(grain_mask) > 1:
            spots = pixel_df[grain_mask][['qx', 'qy', 'qz']].values
            hkls = pixel_df[grain_mask][['h', 'k', 'l']].values
            ori, _ = Rotation.align_vectors(spots, phase.Q(hkls))
            ori_map[indices] = ori.as_matrix()
        else:
            ori_map[indices] = np.nan
    
    return ori_map


def build_ori_list(spots_3D, phase, grain_id=0, intensity=True, metric='int_corr'):

    int_flag = intensity
    ori_list = []
    intensity = []

    if int_flag and metric not in spots_3D:
        raise KeyError(f'{metric} not in spots_3D')

    map_shape = map_shape_from_spots(spots_3D)
    null_ori = np.eye(3)
    null_ori[:] = np.nan

    for index in tqdm(range(np.prod(map_shape))):
        indices = np.unravel_index(index, map_shape)
        pixel_df = pixel_spots(spots_3D, indices)
        grain_mask = pixel_df['grain_id'] == grain_id
        if sum(grain_mask) > 1:
            spots = pixel_df[grain_mask][['qx', 'qy', 'qz']]
            hkls = pixel_df[grain_mask][['h', 'k', 'l']]
            ori, _ = Rotation.align_vectors(spots, phase.Q(hkls))
            ori_list.append(ori.as_matrix())
            if int_flag:
                intensity.append(pixel_df[grain_mask][metric].values.sum())
        else:
            ori_list.append(null_ori)
            if int_flag:
                intensity.append(np.nan)
    
    if int_flag:
        return np.asarray(ori_list), np.asarray(intensity)
    else:
        return np.asarray(ori_list)


def build_full_ori_list(spots_3D, phase, intensity=True, metric='int_corr'):
    int_flag = intensity
    ori_list = []
    intensity = []

    if int_flag and metric not in spots_3D:
        raise KeyError(f'{metric} not in spots_3D')

    map_shape = map_shape_from_spots(spots_3D)

    for index in tqdm(range(np.prod(map_shape))):
        indices = np.unravel_index(index, map_shape)
        pixel_df = pixel_spots(spots_3D, indices)

        for grain_id in np.unique(pixel_df['grain_id'].values):
            grain_mask = pixel_df['grain_id'] == grain_id
            if sum(grain_mask) > 1:
                spots = pixel_df[grain_mask][['qx', 'qy', 'qz']]
                hkls = pixel_df[grain_mask][['h', 'k', 'l']]
                ori, _ = Rotation.align_vectors(spots, phase.Q(hkls))
                ori_list.append(ori.as_matrix())
                if int_flag:
                    intensity.append(pixel_df[grain_mask][metric].values.sum())
    
    if int_flag:
        return np.asarray(ori_list), np.asarray(intensity)
    else:
        return np.asarray(ori_list)


# Plotting functions

def plot_grain_map_gallery(spots_3D,
                           n_grains,
                           plot_intensity=False,
                           plot_color=False,
                           color_sequence=color_sequences['tab20'],
                           **kwargs):

    grain_maps, titles = [], []
    for n in range(n_grains):
        if plot_color:
            color = color_sequence[n]
        else:
            color = None

        grain_map = build_grain_map(spots_3D,
                                    n,
                                    intensity=plot_intensity,
                                    color=color)

        grain_maps.append(grain_map)
        titles.append(f'Grain {n}')
    
    if plot_color:
        grain_maps = np.asarray(grain_maps)
        grain_maps[..., -1] /= np.nanmax(grain_maps[..., -1])

    plot_image_gallery(grain_maps,
                       titles=titles,
                       plot_colorbar=plot_intensity and not plot_color,
                       **kwargs)


def plot_grain_map_overlay(spots_3D,
                           n_grains,
                           color_sequence=color_sequences['Set1'],
                           **kwargs):

    grain_maps = []
    for n in range(n_grains):
        grain_maps.append(build_grain_map(spots_3D,
                                          n,
                                          intensity=True,
                                          color=color_sequence[n]))

    # return grain_maps

    # Normalize
    grain_maps = np.asarray(grain_maps)
    grain_maps[..., -1] /= np.nanmax(grain_maps[..., -1])

    # Combine
    rgbs = grain_maps[..., :3]
    alphas = grain_maps[..., 3]
    
    alpha_weights = np.expand_dims(alphas, axis=-1)
    sum_alphas = np.nansum(alpha_weights, axis=0)
    weighted_rgbs = np.nansum(rgbs * alpha_weights, axis=0)
    out_rgb = weighted_rgbs / sum_alphas
    overlay = np.concatenate([out_rgb, np.expand_dims(np.nanmax(alphas, axis=0), axis=-1)], axis=-1)
    # grain_maps = np.nanmax(grain_maps, axis=0)

    # return overlay

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    ax.imshow(overlay, **kwargs)

    # for n in range(n_grains):
    #     ax.scatter([], [], c=color_sequence[n], label=f'Grain {n}', marker='s', s=150)
    
    # ax.legend(loc='upper left',
    #           bbox_to_anchor=(1.05, 1),
    #           borderaxespad=0)
    fig.show()




def plot_all_ori_IPF(spots_3D,
                     phase,
                     direction=None,
                     intensity=True,
                     **kwargs):

    if direction is not None:
        direction = orix.vector.vector3d.Vector3d(direction)
                     
    full_ori_list, intensity = build_full_ori_list(spots_3D,
                                                   phase,
                                                   intensity=intensity)
    
    symmetry = get_point_group(phase.lattice.space_group_nr).laue
    ori = Orientation.from_matrix(full_ori_list, symmetry=symmetry)

    fig = ori.scatter('ipf',
                      c=intensity,
                      return_figure=True,
                      direction=direction,
                      **kwargs)
    
    fig.show()


def plot_grain_IPF(spots_3D,
                   n_grains,
                   phase,
                   direction=None,
                   intensity=False,
                   color_sequence=color_sequences['tab20'],
                   **kwargs):
    
    int_flag = intensity
    grain_list = []
    intensity_list = []
    colors = []

    if direction is not None:
        direction = orix.vector.vector3d.Vector3d(direction)

    for grain_id in range(n_grains):
        ori_list, intensity = build_ori_list(spots_3D,
                                             phase,
                                             grain_id=grain_id,
                                             intensity=True)
        
        ori_list = ori_list[~np.isnan(intensity)]
        intensity = intensity[~np.isnan(intensity)]

        grain_list.append(ori_list)
        intensity_list.extend(list(intensity))
        colors.extend([color_sequence[grain_id],] * len(intensity))

    # return grain_list

    symmetry = get_point_group(phase.lattice.space_group_nr).laue
    ori = Orientation.from_matrix(np.vstack(grain_list),
                                  symmetry=symmetry)
    
    if int_flag:
        intensity_list = np.asarray(intensity_list)
        s = intensity_list / intensity_list.max()
    else:
        s = None


    fig = ori.scatter('ipf',
                      c=colors,
                      alpha=s,
                      return_figure=True,
                      direction=direction,
                      **kwargs)
    
    fig.show()



def plot_IPF_map(ori_map,
                 phase,
                 int_map=None,
                 direction=None,
                 **kwargs):

    if direction is not None:
        direction = orix.vector.vector3d.Vector3d(direction)
    
    nan_mask = np.any(np.isnan(ori_map), axis=(2, 3))

    ori_list = ori_map[~nan_mask]

    symmetry = get_point_group(phase.lattice.space_group_nr).laue
    ori = Orientation.from_matrix(ori_list, symmetry=symmetry)

    ipf_key = orix_plot.IPFColorKeyTSL(symmetry, direction=direction)
    ori.symmetry = ipf_key.symmetry
    rgb = ipf_key.orientation2color(ori)

    rgb_map = np.zeros((*ori_map.shape[:-2], 3))
    rgb_map[:] = np.nan

    rgb_map[~nan_mask] = rgb
    plotting_map = rgb_map

    if int_map is not None:
        int_map /= np.nanmax(int_map)
        plotting_map = np.concatenate([plotting_map, np.expand_dims(int_map, axis=-1)], axis=-1)

    fig, ax = plt.subplots()
    ax.imshow(plotting_map)
    ax.set_aspect('equal')
    fig.show()





def plot_interactive_misorientation(ori_map, x_ticks=None, y_ticks=None, vmax=5, cmap='viridis'):

    if x_ticks is None:
        x_ticks = list(range(ori_map.shape[1]))
    if y_ticks is None:
        y_ticks = list(range(ori_map.shape[0]))

    # Setup variables
    row, col = 0, 0
    map_x = x_ticks[col]
    map_y = y_ticks[row]

    fig, ax = plt.subplots()

    im = ax.imshow(np.zeros(ori_map.shape[:2]), vmin=0, vmax=vmax, cmap=cmap)
    fig.colorbar(im, ax=ax)


    def update_coordinates(event):
        nonlocal row, col, map_x, map_y
        old_row, old_col = row, col
        #col, row = event.xdata, event.ydata
        map_x, map_y = event.xdata, event.ydata

        col = np.argmin(np.abs(x_ticks - map_x))
        map_x = x_ticks[col]

        row = np.argmin(np.abs(y_ticks - map_y))
        map_y = y_ticks[row]
        # row = len(y_ticks) - row - 1 # reverse the index

        # Check if the pixel is in data
        if ((col >= ori_map.shape[1])
            and (row >= ori_map.shape[0])):
            return False

        elif ((event.name == 'motion_notify_event')
            and (old_row == row and old_col == col)):
            return False
        
        else:
            return True

    def update_axes(event):
        if update_coordinates(event):
            pix_ori = ori_map[row, col]
            mis_map = np.empty(ori_map.shape[:2])
            mis_map[:] = np.nan
            
            if not np.any(np.isnan(pix_ori)):
                for index in range(np.prod(mis_map.shape)):
                    indices = np.unravel_index(index, mis_map.shape)
                    if not np.any(np.isnan(ori_map[indices])):
                        mis = Rotation.from_matrix(pix_ori @ ori_map[indices].T)
                        mis_map[indices] = np.degrees(mis.magnitude())

            im.set_data(mis_map)
            fig.canvas.draw_idle()

    # Make interactive
    def onclick(event):
        if event.inaxes == ax:
            update_axes(event)
    
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    fig.show()