import numpy as np
import os
import dask
from tqdm import tqdm
from skimage import io
import h5py
import time as ttime
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib import patches
from matplotlib.collections import PatchCollection
import matplotlib
from matplotlib import cm
from scipy.spatial.transform import Rotation
from tqdm.dask import TqdmCallback
import gc
from collections import Counter
import itertools
from matplotlib import color_sequences
from scipy.spatial import distance_matrix, KDTree

from orix.quaternion.orientation import Orientation
from orix.quaternion.symmetry import get_point_group
from orix.vector import Vector3d

# Local imports
from xrdmaptools.utilities.utilities import (
#   rescale_array,
  timed_iter,
  memory_iter
)
from xrdmaptools.XRDBaseScan import XRDBaseScan
from xrdmaptools.utilities.utilities import (
    generate_intensity_mask
)
# from xrdmaptools.reflections.spot_blob_indexing_3D import pair_casting_index_full_pattern
from xrdmaptools.reflections.spot_blob_search_3D import rsm_spot_search
from xrdmaptools.crystal.crystal import are_collinear, are_coplanar




def interactive_misorientation_plot(ori_map, x_ticks=None, y_ticks=None, vmax=5, cmap='viridis'):

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


def reindex_grains_by_row(all_spots,
                          phase,
                          mis_thresh=0.5,
                          degrees=True,
                        #   debug_stop=None
                          ):

    # Collect orientation maps of new grain ids
    all_grains = {}
    next_grain_id = 0

    # Effective map shape
    map_shape = (np.max(all_spots['map_y']) + 1,
                 np.max(all_spots['map_x']) + 1)

    # Working containers
    curr_row_oris = [{} for _ in range(map_shape[1])]

    # Setup progressbar
    pbar = tqdm(total=np.prod(map_shape))

    # Iterate trough rows
    for row in range(map_shape[0]):
        # Update containers
        prev_row_oris = curr_row_oris.copy()
        curr_row_oris = [{} for _ in range(map_shape[1])]

        # Iterate through columns
        for col in range(map_shape[1]):
            
            # if debug_stop is not None:
            #     if row == debug_stop[0] and col == debug_stop[1]: 
            #         return all_grains, prev_row_oris, curr_row_oris

            # Get pixel grain information
            df = pixel_spots(all_spots, (row, col))
            grain_ids = df['grain_id'].values
            grain_ids = np.unique(grain_ids[~np.isnan(grain_ids)])
            
            # Kick out early
            if len(grain_ids) == 0:
                pbar.update(1)
                # print(f'No grain for pixel {(row, col)}.')
                continue
            
            # Determine all orientations within pixel
            pixel_oris = []
            for grain_id in grain_ids:
                grain_mask = df['grain_id'] == grain_id
                grain_spots = df[grain_mask][['qx', 'qy', 'qz']].values
                grain_hkls = df[grain_mask][['h', 'k', 'l']].values
                grain_ori = Rotation.align_vectors(grain_spots,
                                                   phase.Q(grain_hkls))[0]
                pixel_oris.append(grain_ori)

            pixel_oris = Rotation.concatenate(pixel_oris)
            
            # Pixel orientations within mis_thresh?
            if len(grain_ids) > 1:
                dropped_inds = []
                for grain_id in range(len(grain_ids) - 1):
                    misori_mags = (pixel_oris[grain_id] * pixel_oris[grain_id + 1:].inv()).magnitude()
                    if degrees:
                        misori_mags = np.degrees(misori_mags)

                    for mis_i, mag in enumerate(misori_mags):
                        if mag < mis_thresh:
                            dropped_inds.append(grain_id + mis_i + 1)
                
                # Drop overlapping orientations
                if len(dropped_inds) > 0:
                    mask = [True,] * len(pixel_oris)
                    print(f'WARNING: Overlapping indexing dropped in pixel {(row, col)}.')
                    for i, dropped_ind in enumerate(np.unique(dropped_inds)):
                        spot_inds = df[df['grain_id'] == dropped_ind].index
                        all_spots.loc[spot_inds, 'grain_id'] = -(i + 1)
                        mask[dropped_ind] = False
                    grain_ids = grain_ids[mask]
                    pixel_oris = Rotation.from_matrix(pixel_oris.as_matrix()[mask])

            # Collect neighbor orientations
            near_keys = list(prev_row_oris[col].keys())
            near_oris = list(prev_row_oris[col].values())
            if col > 0:
                near_keys.extend(list(curr_row_oris[col - 1].keys()))
                near_oris.extend(list(curr_row_oris[col - 1].values()))                
            
            # Construct misorientation magnitude matrix
            if len(near_keys) > 0:
                near_oris = Rotation.concatenate(near_oris)
                mis_mag_mat = np.asarray([(ori * near_oris.inv()).magnitude() for ori in pixel_oris])
                if degrees:
                    mis_mag_mat = np.degrees(mis_mag_mat)
                matches = mis_mag_mat < mis_thresh
            else:
                matches = np.array([[]])

            # Check if two pixel orientations match the same nearby grain
            overlap = np.sum(matches, axis=0) > 1
            if np.any(overlap):
                print(f'WARNING: Overlapping indexing found during stiching in pixel {(row, col)}.')

                # Eliminate the less good orientation from the pixel like before
                dropped_inds = []
                for o_ind in np.nonzero(overlap)[0]:
                    dropped_inds.extend(list(np.nonzero(matches[:, o_ind])[0][1:])) # Ignore first

                # Drop pixel orientations
                mask = [True,] * len(pixel_oris)
                for i, dropped_ind in enumerate(np.unique(dropped_inds)):
                    spot_inds = df[df['grain_id'] == dropped_ind].index
                    all_spots.loc[spot_inds, 'grain_id'] = np.nanmin(df['grain_id'].values) - 1
                    mask[dropped_ind] = False
                grain_ids = grain_ids[mask]
                pixel_oris = Rotation.from_matrix(pixel_oris.as_matrix()[mask])

                # Re-evaluate matrix
                mis_mag_mat = np.asarray([(ori * near_oris.inv()).magnitude() for ori in pixel_oris])
                if degrees:
                    mis_mag_mat = np.degrees(mis_mag_mat)
                matches = mis_mag_mat < mis_thresh

            for ind in range(len(grain_ids)):

                grain_id = grain_ids[ind]
                spot_inds = list(df[df['grain_id'] == grain_id].index)
                ori = pixel_oris[ind]
                if matches.size > 0:
                    match_inds = np.nonzero(matches[ind])[0]
                    match_keys = np.unique(np.asarray(near_keys)[match_inds])
                else:
                    match_keys = []
                
                # No matches. Register new grain
                if len(match_keys) == 0:
                    # print(f'Registering new grain with index {next_grain_id}.')
                    all_grains[next_grain_id] = spot_inds
                    curr_row_oris[col][next_grain_id] = ori
                    next_grain_id += 1

                # One match. Assign orientation to grain
                elif len(match_keys) == 1:
                    all_grains[match_keys[0]].extend(spot_inds)
                    curr_row_oris[col][match_keys[0]] = ori
                
                # # Two matches. Combine previous grains
                # elif len(match_keys) == 2:
                #     keep_key = match_keys[0]
                #     drop_key = match_keys[1]

                #     # Update all_grains
                #     all_grains[keep_key].extend(all_grains[drop_key])
                #     all_grains[keep_key].extend(spot_inds)
                #     # print(f'Dropping grain with index {drop_key}.')
                #     del all_grains[drop_key]

                #     # Update other stored orientations
                #     for ori_dict in curr_row_oris + prev_row_oris:
                #         if drop_key in ori_dict:
                #             if keep_key in ori_dict:
                #                 pass
                #                 print(f'WARNING: Convergent crystal grain found at pixel {(row, col)}.')
                #             temp_ori = ori_dict[drop_key]
                #             ori_dict[keep_key] = temp_ori
                #             del ori_dict[drop_key]
                    
                #     # Update current
                #     curr_row_oris[col][keep_key] = ori
                
                # Multiple matches. Stitch previous grains
                elif len(match_keys) > 1:
                    keep_key = match_keys[0]
                    drop_keys = match_keys[1:]

                    for drop_key in drop_keys:
                        # Update all_grains
                        all_grains[keep_key].extend(all_grains[drop_key])
                        all_grains[keep_key].extend(spot_inds)
                        # print(f'Dropping grain with index {drop_key}.')
                        del all_grains[drop_key]

                        # Update other stored orientations
                        for ori_dict in curr_row_oris + prev_row_oris:
                            if drop_key in ori_dict:
                                if keep_key in ori_dict:
                                    pass
                                    print(f'WARNING: Convergent crystal grain found at pixel {(row, col)}.')
                                temp_ori = ori_dict[drop_key]
                                ori_dict[keep_key] = temp_ori
                                del ori_dict[drop_key]
                
                # # More matches. Something went wrong with overlapping orientation
                # else: # Theoretically possible. Better way to handle?
                #     warn_str = ('More matched orientations than reasonably expected...')
                #     print(warn_str)
                #     # raise RuntimeError
                #     return near_oris, pixel_oris, grain_ids, all_spots
                else:
                    raise ValueError('Something went terribly wrong...')
            
                # if np.any([spot_ind in [11356, 17616, 20840, 59224, 59243, 64114, 64128] for spot_ind in spot_inds]):
                #     print(f'Spots dropped for grain {grain_id} in pixel {(row, col)}.')
            
            # Update progress
            pbar.update(1)

    # Remove progress!!
    pbar.close()

    # Count all grain orientations
    counts = [len(v) for v in all_grains.values()]
    sorted_keys = [x for _, x in sorted(zip(counts, all_grains.keys()),
                                        key=lambda pair: pair[0],
                                        reverse=True)]

    # Re-label grains
    sorted_grains = {}
    for new_grain_id, sorted_key in enumerate(sorted_keys):
        all_spots.loc[all_grains[sorted_key], 'grain_id'] = new_grain_id # sorted
        # all_spots.loc[all_grains[sorted_key], 'grain_id'] = sorted_key
        sorted_grains[new_grain_id] = all_grains[sorted_key]

    # return all_grains
    return sorted_grains



def pixel_spots(spots, indices):
    ps = spots[((spots['map_x'] == indices[1]) & (spots['map_y'] == indices[0]))]
    return ps


def build_mis_mag_map(ori_map, ref=None):

    mag_map = np.zeros(ori_map.shape[:2])

    if ref is None:
        ref = np.eye(3)
    elif isinstance(ref, Rotation):
        ref = ref.as_matrix()

    for index in tqdm(range(np.prod(mag_map.shape))):
        indices = np.unravel_index(index, mag_map.shape)
        mag_map[indices] = np.degrees(Rotation.from_matrix(ref.T @ ori_map[indices]).magnitude())
    
    return mag_map


def build_grain_map(map_shape, all_spots, grain_id, color=None, intensity=False):
        
        if color is None:
            grain_map = np.empty(map_shape)
        else:
            grain_map = np.empty((*map_shape, *np.asarray(color).shape))
        int_map = np.zeros(map_shape)
        grain_map[:] = np.nan
        int_map[:] = np.nan
        
        for index in range(np.prod(map_shape)):
            indices = np.unravel_index(index, map_shape)
            ps = pixel_spots(all_spots, indices)
            if grain_id in ps['grain_id'].values:
                grain_mask = ps['grain_id'] == grain_id
                int_map[indices] = np.sum(ps[grain_mask]['intensity'].values)
                if color is None:
                    grain_map[indices] = True
                else:
                    grain_map[indices] = color
        
        int_map /= np.nanmax(int_map)
        
        if intensity:
            if color:
                if len(color) == 3:
                    grain_map = np.concatenate((grain_map, int_map[:, :, np.newaxis]), axis=-1)
                elif len(color) == 4:
                    grain_map[:, :, -1] = int_map
            else:
                grain_map = int_map

        return grain_map
        


def build_ori_map(spots_3D, map_shape, phase, grain_id=0):
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


def build_ori_list(spots_3D, map_shape, phase, grain_id=0, intensity=True):

    int_flag = intensity
    ori_list = []
    intensity = []

    for index in tqdm(range(np.prod(map_shape))):
        indices = np.unravel_index(index, map_shape)
        pixel_df = pixel_spots(spots_3D, indices)
        grain_mask = pixel_df['grain_id'] == grain_id
        if sum(grain_mask) > 1:
            spots = pixel_df[grain_mask][['qx', 'qy', 'qz']]
            hkls = pixel_df[grain_mask][['h', 'k', 'l']]
            ori, _ = Rotation.align_vectors(spots, phase.Q(hkls))
            ori_list.append(ori.as_matrix())
            intensity.append(pixel_df[grain_mask]['intensity'].values.sum())
    
    if int_flag:
        return np.asarray(ori_list), np.asarray(intensity)
    else:
        return np.asarray(ori_list)


def build_full_ori_list(spots_3D, map_shape, phase, intensity=True):
    int_flag = intensity
    ori_list = []
    intensity = []

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
                intensity.append(pixel_df[grain_mask]['intensity'].values.sum())
    
    if int_flag:
        return np.asarray(ori_list), np.asarray(intensity)
    else:
        return np.asarray(ori_list)


def plot_all_ori_IPF(spots_3D,
                     map_shape,
                     phase,
                     direction=None,
                     intensity=True,
                     **kwargs):
                     
    full_ori_list, intensity = build_full_ori_list(spots_3D,
                                                   map_shape,
                                                   phase,
                                                   intensity=intensity)
    
    symmetry = get_point_group(phase.lattice.space_group_nr).laue
    # symmetry = laue
    ori = Orientation.from_matrix(full_ori_list, symmetry=symmetry)

    fig = ori.scatter('ipf',
                      c=intensity,
                      return_figure=True,
                      direction=direction,
                      **kwargs)
    
    fig.show()



def plot_grain_map_gallery(n_maps,
                           map_shape,
                           spots3D,
                           color_sequence=color_sequences['tab20']):

    plot_image_gallery([build_grain_map(map_shape,
                                        spots3D,
                                        n,
                                        color=color_sequence[n])
                        for n in range(n_maps)])


def plot_grain_IPF(n_grains,
                   spots_3D,
                   map_shape,
                   phase,
                   direction=None,
                   intensity=False,
                   color_sequence=color_sequences['tab20'],
                   **kwargs):
    
    int_flag = intensity
    grain_list = []
    intensity_list = []
    colors = []

    for grain_id in range(n_grains):
        ori_list, intensity = build_ori_list(spots_3D,
                                             map_shape,
                                             phase,
                                             grain_id=grain_id,
                                             intensity=True)

        grain_list.append(ori_list)
        intensity_list.extend(list(intensity))
        colors.extend([color_sequence[grain_id],] * len(intensity))

    # return grain_list

    symmetry = get_point_group(phase.lattice.space_group_nr).laue
    # symmetry = laue
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


from xrdmaptools.plot.interactive import interactive_3D_labeled_plot
from xrdmaptools.utilities.utilities import get_max_vector_map
def interactive_indexing_plot(xdms,
                              spots_3D=None,
                              map_shape=None,
                              use_grains=False,
                              color_sequence=color_sequences['tab10'],
                              **kwargs):

    map_kw = {}
    dyn_kw = {}

    if map_shape is None:
        if hasattr(xdms, 'xdms_vector_map') and xdms.xdms_vector_map is not None:
            map_shape = xdms.xdms_vector_map.shape
        else:
            map_shape = xdms.map_shape

    if spots_3D is None:
        if hasattr(xdms, 'spots_3D') and xdms.spots_3D is not None:
            spots_3D = xdms.spots_3D
    
    if not hasattr(xdms, 'edges') or xdms.edges is not None:
        xdms.get_sampled_edges()
    dyn_kw['edges'] = xdms.edges
    
    dyn_kw['data'] = xdms._spots_3D_to_vectors(spots_3D=spots_3D,
                                     map_shape=map_shape)
    map_kw['map'] = get_max_vector_map(dyn_kw['data'])

    labels = np.empty(map_shape, dtype=object)
    label_spots = labels.copy()
    label_colors = labels.copy()

    max_inds = len(color_sequence)

    for index in range(np.prod(map_shape)):
        indices = np.unravel_index(index, map_shape)
        df = pixel_spots(spots_3D, indices)
        grain_ids = df['grain_id'].values
        grain_ids = np.unique(grain_ids[~np.isnan(grain_ids)]).astype(int)
        if use_grains:
            grain_ids = grain_ids[grain_ids < max_inds]
        elif len(grain_ids) > max_inds:
            grain_ids = grain_ids[:max_inds]
        
        # Build labels
        hkl_list = []
        spot_list = []
        color_list = []

        for i, grain_id in enumerate(grain_ids):
            grain_mask = df['grain_id'] == grain_id
            spots = df[grain_mask][['qx', 'qy', 'qz']].values
            hkls = df[grain_mask][['h', 'k', 'l']].values
            hkl_list.extend([f'({int(h)} {int(k)} {int(l)})' for h, k, l in hkls])
            spot_list.extend(spots)
            if use_grains:
                colors = [color_sequence[grain_id],] * sum(grain_mask)
            else:
                colors = [color_sequence[i],] * sum(grain_mask)
            color_list.extend(colors)
        
        labels[indices] = hkl_list
        label_spots[indices] = spot_list
        label_colors[indices] = color_list
    
    dyn_kw['labels'] = labels
    dyn_kw['label_spots'] = label_spots
    dyn_kw['label_colors'] = label_colors

    fig, ax = interactive_3D_labeled_plot(map_kw=map_kw,
                                          dyn_kw=dyn_kw,
                                          **kwargs)
    
    # return dyn_kw

    fig.show()


def unit_vecs(vecs):

    vecs = np.asarray(vecs)

    if vecs.ndim != 2:
        vecs = vecs.reshape(1, -1)
    
    if vecs.shape[1] != 3:
        raise ValueError(f'Vectors must be of shape (n, 3) not {vecs.shape}')
    
    return vecs / np.linalg.norm(vecs, axis=1).reshape(-1, 1)
    


from orix.vector import Vector3d
from xrdmaptools.utilities.math import wavelength_2_energy

# def symmeterize_indexing(pixel_df,
#                          phase,
#                          symmetry,
#                          grain_id=0,
#                          metric='intensity',
#                          verbose=False):

#     # Mask out single orientation
#     grain_mask = pixel_df['grain_id'] == grain_id

#     # Collect values from dataframe
#     spots = pixel_df[grain_mask][['qx', 'qy', 'qz']].values
#     hkls = pixel_df[grain_mask][['h', 'k', 'l']].values

#     energies = wavelength_2_energy(pixel_df[grain_mask]['wavelength'].values) * 1e3

#     # Apply symmetry to data
#     laue = symmetry.laue
#     laue = laue[:laue.shape[0] // 2]
#     sym_qs = laue.outer(Vector3d(phase.Q(hkls))).data
#     # sym_qs = phase.Q(hkls) @ laue.to_matrix()
#     sym_calc = np.abs(phase.StructureFactorForEnergy(sym_qs, energies))**2
    
#     # Measure fitness
#     if metric:
#         ints = pixel_df[grain_mask][metric].values
#         sym_corr = [np.dot(ints / np.linalg.norm(ints), calc / np.linalg.norm(calc)) for calc in sym_calc]
#         sym_corr = np.round(sym_corr, 10)
#         sym_uniq = np.unique(sym_corr)
#         # if len(sym_uniq) > 1:
#         #     corr_mask = sym_corr >= sym_uniq[-2]
#         # else:
#         #     corr_mask = sym_corr >= sym_uniq[-1]
#         corr_mask = sym_corr == np.max(sym_corr)
        
#         # corr_mask = np.isclose(sym_corr, np.max(sym_corr), rtol=1e-1)
#         if verbose:
#             print(f'Found {corr_mask.sum()} symmetrically equivalent orientations.')
#     else:
#         corr_mask = np.asarray([True,] * len(sym_qs))

#     # Determine magnitude of orientation (minimum is definition of fundamental zone, kind of)
#     sym_mags = [Rotation.align_vectors(spots, qs)[0].magnitude() for qs in sym_qs]
#     # proper_mask = [l.is_proper for l in laue]
#     proper_mask = np.asarray([True,] * len(sym_qs))
#     mask = corr_mask & np.asarray(proper_mask)

#     # Choose best orientation for minimum fitness and minimum orientation magnitude
#     best_ind = np.nonzero(mask)[0][np.argmin(np.asarray(sym_mags)[mask])]

#     # Switch to new hkls
#     new_hkls = np.round(phase.HKL(sym_qs[best_ind])).astype(np.float32)

#     return new_hkls


# def symmeterize_map(spots_3D, phase, symmetry, metric='intensity', verbose=False):

#     # Effective map shape
#     map_shape = (np.max(all_spots['map_y']) + 1,
#                  np.max(all_spots['map_x']) + 1)

    
#     for index in tqdm(range(np.prod(map_shape))):
#         indices = np.unravel_index(index, map_shape)

#         df = pixel_spots(spots_3D, indices)
#         grain_ids = np.unique(df['grain_id'].values)
        
#         for grain_id in grain_ids[~np.isnan(grain_ids)]:
#             try:
#                 new_hkls = symmeterize_indexing(df, phase, symmetry, grain_id=grain_id, metric=metric, verbose=verbose)
#             except Exception as e:
#                 print(e)
#                 print(indices, grain_id)
#             spot_inds = df[df['grain_id'] == grain_id].index.values
#             spots_3D.loc[spot_inds, ['h', 'k', 'l']] = new_hkls
    


def symmeterize_and_stitch(spots_3D,
                           phase,
                        #    symmetry,
                           mis_thresh=0.5,
                           metric='intensity',
                           degrees=True,
                           verbose=False,
                           debug_stop=None):

    # Setup symmetry
    symmetry = get_point_group(phase.lattice.space_group_nr)
    laue = symmetry.laue
    laue = laue[:laue.shape[0] // 2] # Ignore inversion for now...
    syms = laue.to_matrix()

    # Collect orientation maps of new grain ids
    all_grains = {}
    next_grain_id = 0

    # Effective map shape
    map_shape = (np.max(spots_3D['map_y']) + 1,
                 np.max(spots_3D['map_x']) + 1)

    # Working containers
    curr_row_oris = [{} for _ in range(map_shape[1])]

    # Setup progressbar
    # pbar = tqdm(total=np.prod(map_shape))

    # Iterate trough rows
    with tqdm(total=np.prod(map_shape)) as pbar:
        for row in range(map_shape[0]):
            # Update containers
            prev_row_oris = curr_row_oris.copy()
            curr_row_oris = [{} for _ in range(map_shape[1])]

            # Iterate through columns
            for col in range(map_shape[1]):

                if debug_stop is not None:
                    if row == debug_stop[0] and col == debug_stop[1]: 
                        return all_grains, prev_row_oris, curr_row_oris

                # Get pixel grain information
                df = pixel_spots(spots_3D, (row, col))
                grain_ids = df['grain_id'].values
                grain_ids = np.unique(grain_ids[~np.isnan(grain_ids)])
                
                # Kick out early
                if len(grain_ids) == 0:
                    pbar.update(1)
                    # print(f'No grain for pixel {(row, col)}.')
                    continue
                
                # Determine all orientations within a pixel from given indexing
                pixel_oris = []
                for grain_id in grain_ids:
                    grain_mask = df['grain_id'] == grain_id
                    grain_spots = df[grain_mask][['qx', 'qy', 'qz']].values
                    grain_hkls = df[grain_mask][['h', 'k', 'l']].values
                    grain_ori = Rotation.align_vectors(grain_spots,
                                                    phase.Q(grain_hkls))[0]
                    pixel_oris.append(grain_ori.as_matrix())
                
                # Symmeterize orientations. shape = (# orientations in pixel, # of symmetry, 3, 3)
                pixel_oris = np.asarray(pixel_oris)[:, np.newaxis] @ syms[np.newaxis]
                
                # Do these orientations overlap by falling within the defined mis_thresh value?
                if len(grain_ids) > 1:
                    dropped_inds = []
                    for grain_id in range(len(grain_ids) - 1):
                        mis_oris = pixel_oris[grain_id][np.newaxis] @ pixel_oris.swapaxes(-1, -2)[grain_id + 1:, : np.newaxis]
                        mis_mags = np.arccos(0.5 * (np.trace(mis_oris, axis1=-2, axis2=-1) - 1))

                        if degrees:
                            mis_mags = np.degrees(mis_mags)

                        for mis_i, mag in enumerate(mis_mags):
                            if np.any(mag < mis_thresh):
                                dropped_inds.append(grain_id + mis_i + 1)
                    
                    # If yes, then drop them. TODO: Check if they are worth combining
                    if len(dropped_inds) > 0:
                        mask = [True,] * len(pixel_oris)
                        if verbose:
                            print(f'WARNING: Overlapping indexing dropped in pixel {(row, col)}.')
                        for i, dropped_ind in enumerate(np.unique(dropped_inds)):
                            spot_inds = df[df['grain_id'] == dropped_ind].index
                            spots_3D.loc[spot_inds, 'grain_id'] = -(i + 1)
                            mask[dropped_ind] = False
                        grain_ids = grain_ids[mask]
                        pixel_oris = pixel_oris[mask]

                # Collect nearby orientations
                # No symmetry applied to these. Should already be optimized
                near_keys = list(prev_row_oris[col].keys())
                near_oris = list(prev_row_oris[col].values())
                if col > 0:
                    near_keys.extend(list(curr_row_oris[col - 1].keys()))
                    near_oris.extend(list(curr_row_oris[col - 1].values()))
                near_keys = np.asarray(near_keys) # for indexing             
                
                # Construct misorientation magnitude matrix
                if len(near_keys) > 0:
                    near_oris = np.stack(near_oris)
                    
                    # All misorientations. shape = (# orientations in pixel, # nearby orientations, # of symmetry, 3, 3)
                    mis_mat_ori = pixel_oris[:, np.newaxis] @ near_oris.swapaxes(-1, -2)[:, np.newaxis]
                    mis_mat_mag = np.arccos(0.5 * (np.trace(mis_mat_ori, axis1=-2, axis2=-1) - 1))

                    if degrees:
                        mis_mat_mag = np.degrees(mis_mat_mag)
                    matches = np.any(mis_mat_mag < mis_thresh, axis=-1)
                else:
                    matches = np.array([[[]]]) # so many brackets!

                # Check if two pixel orientations match the same nearby grain
                overlap = np.sum(matches, axis=0) > 1
                if np.any(overlap):
                    if verbose:
                        print(f'WARNING: Overlapping indexing found during stiching in pixel {(row, col)}.')

                    # Eliminate the less good orientation from the pixel like before
                    dropped_inds = []
                    for o_ind in np.nonzero(overlap)[0]:
                        dropped_inds.extend(list(np.nonzero(matches[:, o_ind])[0][1:])) # Ignore first

                    # Drop pixel orientations
                    mask = [True,] * len(pixel_oris)
                    for i, dropped_ind in enumerate(np.unique(dropped_inds)):
                        spot_inds = df[df['grain_id'] == dropped_ind].index
                        spots_3D.loc[spot_inds, 'grain_id'] = np.nanmin(df['grain_id'].values) - 1
                        mask[dropped_ind] = False
                    grain_ids = grain_ids[mask]
                    pixel_oris = pixel_oris[mask]
                    mis_mat_mag = mis_mat_mag[mask]
                    matches = matches[mask]

                # For each pixel orientation, check if it matches any nearby orientations
                for ind in range(len(grain_ids)):

                    grain_id = grain_ids[ind]
                    spot_inds = list(df[df['grain_id'] == grain_id].index)
                    sym_oris = pixel_oris[ind]
                    if matches.size > 0:
                        match_inds = np.nonzero(matches[ind])[0]
                        match_keys = np.unique(near_keys[match_inds])
                    else:
                        match_keys = []
                    
                    # No matches. Register new grain
                    if len(match_keys) == 0:
                        # Choose pixel orientation in fundamental zone (closest to reference orientation)
                        ori_mags = np.arccos(0.5 * (np.trace(sym_oris, axis1=-2, axis2=-1) - 1))
                        sym_ind = np.nanargmin(ori_mags)

                        # Update containers
                        all_grains[next_grain_id] = spot_inds
                        curr_row_oris[col][next_grain_id] = sym_oris[sym_ind]
                        next_grain_id += 1

                        # Update hkls
                        hkls = spots_3D.loc[spot_inds, ['h', 'k', 'l']].values
                        new_hkls = phase.HKL(phase.Q(hkls) @ syms[sym_ind]).round()
                        spots_3D.loc[spot_inds, ['h', 'k', 'l']] = new_hkls

                    # One match. Assign orientation to grain
                    elif len(match_keys) == 1:
                        # Choose equivalent orientation closest to nearby orientation
                        min_index = np.nanargmin(mis_mat_mag[ind])
                        sym_ind = np.unravel_index(min_index, mis_mat_mag.shape[1:])[1]

                        # Update containers
                        all_grains[match_keys[0]].extend(spot_inds)
                        curr_row_oris[col][match_keys[0]] = sym_oris[sym_ind]

                        # Update hkls
                        hkls = spots_3D.loc[spot_inds, ['h', 'k', 'l']].values
                        new_hkls = phase.HKL(phase.Q(hkls) @ syms[sym_ind]).round()
                        spots_3D.loc[spot_inds, ['h', 'k', 'l']] = new_hkls

                    # Multiple matches. Stitch previous grains
                    elif len(match_keys) > 1:
                        if verbose:
                            print(f'Stitching grains for grain {grain_id} in pixel {(row, col)}.')

                        # Get index and pointer values
                        min_index = np.nanargmin(mis_mat_mag[ind])
                        keep_ind, sym_ind = np.unravel_index(min_index, mis_mat_mag[ind].shape)
                        all_sym_inds = np.nanargmin(mis_mat_mag[ind][match_inds], axis=1)

                        # Convert to grain keys
                        keep_key = near_keys[keep_ind]

                        # Update current containers
                        all_grains[keep_key].extend(spot_inds)
                        curr_row_oris[col][keep_key] = sym_oris[sym_ind]
                        if col > 0:
                            curr_row_oris[col - 1][keep_key] = sym_oris[sym_ind]
                        # prev_row_oris does not need to be updated...

                        # Update current hkls
                        hkls = spots_3D.loc[spot_inds, ['h', 'k', 'l']].values
                        new_hkls = phase.HKL(phase.Q(hkls) @ syms[sym_ind]).round()
                        spots_3D.loc[spot_inds, ['h', 'k', 'l']] = new_hkls

                        # print(f'Drop keys are {[near_keys[ind] for ind in match_inds]}')
                        for drop_ind, drop_sym_ind in zip(match_inds, all_sym_inds):
                            drop_key = near_keys[drop_ind]
                            near_keys[np.nonzero(near_keys == drop_key)[0]] = keep_key
                            
                            # Solves issues of duplicates (which is physical)
                            if drop_key == keep_key:
                                continue

                            # Find all spots and update labels                            
                            # try:
                            #     updated_inds = all_grains[drop_key]
                            # except:
                            #     print(near_keys)
                            #     print(drop_ind)
                            #     continue
                            updated_inds = all_grains[drop_key]
                            all_grains[keep_key].extend(updated_inds)
                            del all_grains[drop_key]
                            # Remove key from row orientations
                            # prev_row_oris does not need to be updated
                            # if drop_key in prev_row_oris[col]:
                            #     del prev_row_oris[col][drop_key]
                            if col > 0 and drop_key in curr_row_oris[col - 1]:
                                del curr_row_oris[col - 1][drop_key]
    
                            # Update hkls if different
                            if drop_sym_ind != sym_ind:
                                hkls = spots_3D.loc[updated_inds, ['h', 'k', 'l']].values
                                new_hkls = phase.HKL(phase.Q(hkls)@ syms[drop_sym_ind].T @ syms[sym_ind]).round()
                                spots_3D.loc[updated_inds, ['h', 'k', 'l']] = new_hkls

                            # Update other stored orientations
                            for ori_dict in curr_row_oris + prev_row_oris:
                                if drop_key in ori_dict:
                                    temp_ori = ori_dict[drop_key]
                                    
                                    # Revert orientation and update to new
                                    if drop_sym_ind != sym_ind:
                                        temp_ori = temp_ori @ syms[drop_sym_ind].T @ syms[sym_ind]

                                    ori_dict[keep_key] = temp_ori
                                    del ori_dict[drop_key]
                
                    else:
                        raise ValueError('Something went terribly wrong...')
                
                # Update progress
                pbar.update(1)

    # Count all grain orientations
    counts = [len(v) for v in all_grains.values()]
    sorted_keys = [x for _, x in sorted(zip(counts, all_grains.keys()),
                                        key=lambda pair: pair[0],
                                        reverse=True)]

    # In-place, re-label grains
    sorted_grains = {}
    for new_grain_id, sorted_key in enumerate(sorted_keys):
        spots_3D.loc[all_grains[sorted_key], 'grain_id'] = new_grain_id # sorted
        sorted_grains[new_grain_id] = all_grains[sorted_key]
    
    # Re-evaluate whole grains for space group and average minimum rotation

    # Should they be split?

    # return all_grains
    return sorted_grains






def dilate_grains(spots_3D,
                  near_q,
                  phase,
                  qmask,
                  iterations=1,
                  min_connections=1,
                  max_total_grains=None,
                  ):
    
    # Effective map shape
    map_shape = (np.max(spots_3D['map_y']) + 1,
                 np.max(spots_3D['map_x']) + 1)
    
    # Get phase information
    max_q = np.max(spots_3D['q_mag'])
    phase.generate_reciprocal_lattice(1.15 * max_q)
    all_ref_qs = phase.all_qs.copy()
    all_ref_hkls = phase.all_hkls.copy()
    all_ref_fs = phase.all_fs.copy()

    ref_mags = np.linalg.norm(all_ref_qs, axis=1)

    upd_spot_inds, new_phase, new_grains, new_hkls, new_qofs = [], [], [], [], []
    clear_spot_inds = []
    for index in tqdm(range(np.prod(map_shape))):
        # Get current pixel
        indices = np.unravel_index(index, map_shape)
        pixel_df = pixel_spots(spots_3D, indices)
        
        # Ignore orientationally indeterminant grains
        if len(pixel_df) < 2:
            continue

        # Collect useful values
        pixel_grain_ids = pixel_df['grain_id'].values
        pixel_grain_ids = np.unique(pixel_grain_ids[~np.isnan(pixel_grain_ids)]).astype(int)
        all_spot_qs = pixel_df[['qx', 'qy', 'qz']].values
        spot_mags = pixel_df['q_mag'].values
        all_spot_ints = pixel_df['intensity'].values
        ext = 0.15
        ref_mask = ((ref_mags > spot_mags.min() * (1 - ext))
                    & (ref_mags < spot_mags.max() * (1 + ext)))
        kdtree = KDTree(all_spot_qs)            
        
        # Collect neighbor grains
        next_indices = np.asarray([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]]) + indices
        next_grains = [{} for _ in range(5)] # Current pixel and 1st nearest neighbor in rectilinear grid

        # next_indices = np.asarray([[0, 0]]) + indices
        # next_grains = [{} for _ in range(1)] # Current pixel and 1st nearest neighbor in rectilinear grid

        for next_ind, n_indices in enumerate(next_indices):
            next_df = pixel_spots(spots_3D, n_indices)
            grain_ids = next_df['grain_id'].values
            grain_ids = np.unique(grain_ids[~np.isnan(grain_ids)]).astype(int)
            if max_total_grains is not None:
                grain_ids = grain_ids[grain_ids < max_total_grains]
            
            # Setup grain information
            for grain_id in grain_ids:
                # Skip those already indexed
                if next_ind != 0 and grain_id in pixel_grain_ids:
                    continue
                grain_mask = next_df['grain_id'] == grain_id
                ori = Rotation.align_vectors(
                        next_df[grain_mask][['qx', 'qy', 'qz']].values,
                        phase.Q(next_df[grain_mask][['h', 'k', 'l']].values))[0]
                next_grains[next_ind][grain_id] = ori

        # if indices == (26, 26):
        #     return next_grains

        # Count all grains to determine connection numbers
        all_next_grains = []
        for next_grain in next_grains:
            all_next_grains.extend(list(next_grain.keys()))
        grain_counts = Counter(all_next_grains)

        # Collect neighbor orientations worth measuring
        all_grain_ids, all_orientations = [], []
        for i, next_grain in enumerate(next_grains):
            for key, value in next_grain.items():
                if i == 0 or grain_counts[key] >= min_connections:
                    all_grain_ids.append(key)
                    all_orientations.append(value)

        # if indices == (27, 19):
        #     print(all_grain_ids)
        #     return all_grain_ids, all_orientations, all_spot_qs, all_ref_qs, ref_mask, kdtree

        # Setup connections
        start_conns = []
        for grain_id, orientation in zip(all_grain_ids, all_orientations):
            # Taking from pair_casting function. Possibly generalized
            connection = [np.nan,] * len(all_spot_qs)
            all_rot_qs = orientation.apply(all_ref_qs[ref_mask], inverse=False)
            temp_qmask = qmask.generate(all_rot_qs)

            # Query kdtree for find closest reference lattice points
            pot_conn = kdtree.query_ball_point(all_rot_qs[temp_qmask],
                                                r=near_q)
            
            # Cast and fill into blank connection
            for conn_i, conn in enumerate(pot_conn):
                if len(conn) > 0:
                    if len(conn) == 0:
                        continue
                    elif len(conn) == 1:
                        # Add candidate reflection
                        connection[conn[0]] = np.nonzero(temp_qmask)[0][conn_i]
                    else:
                        # Add closest of multiple candidate reflections
                        _, spot_idx = kdtree.query(all_rot_qs[temp_qmask][conn_i])
                        connection[spot_idx] = np.nonzero(temp_qmask)[0][conn_i]
            
            if np.sum(~np.isnan(connection)) >= 2:
                start_conns.append(connection)
        
        if len(start_conns) < 1:
            continue
        # Possibility of updating grains, so record for clearing
        clear_spot_inds.extend(list(pixel_df.index.values))

        # if indices == (27, 19):
        #     print(start_conns)

        # Test all connections
        try:
            (best_connections, 
             best_qofs) = decaying_pattern_decomposition(
                            np.asarray(start_conns).astype(np.float32),
                            all_spot_qs,
                            all_spot_ints,
                            all_ref_qs[ref_mask],
                            all_ref_fs[ref_mask],
                            qmask,
                            near_q,
                            verbose=False)
        except:
            print(f'Decomposition failed for indices {indices}')
            return np.asarray(start_conns), all_spot_qs, all_spot_ints, all_ref_qs[ref_mask], all_ref_fs[ref_mask], qmask, near_q
        
        # if indices == (27, 19):
        #     print(start_conns)
        #     return best_connections, best_qofs, start_conns
        
        for connection, qof in zip(best_connections, best_qofs):
            # Parse spots, hkls, and redetermine closest grain_id
            spot_inds, ref_inds = _get_connection_indices(connection)
            conn_spots = all_spot_qs[spot_inds]
            conn_refs = all_ref_qs[ref_mask][ref_inds]
            pixel_ori = Rotation.align_vectors(conn_spots, conn_refs)[0]
            # misori = np.arccos(0.5 * (np.trace(all_orientations @ pixel_ori.T, axis1=-2, axis2=-1) - 1))
            misori = (pixel_ori * Rotation.concatenate(all_orientations).inv()).magnitude()
            best_grain_id = all_grain_ids[np.nanargmin(misori)]
            
            # Add to list to update later
            upd_spot_inds.extend(list(pixel_df.index.values[spot_inds]))
            new_phase.extend([phase.name,] * len(spot_inds))
            new_grains.extend([best_grain_id,] * len(spot_inds))
            new_hkls.extend(list(phase.HKL(conn_refs).astype(int)))
            new_qofs.extend([qof,] * len(spot_inds))
    
    # Clear modified values
    spots_3D.loc[clear_spot_inds, 'phase'] = ''
    spots_3D.loc[clear_spot_inds, 'grain_id'] = np.nan
    spots_3D.loc[clear_spot_inds, ['h', 'k', 'l']] = [np.nan, np.nan, np.nan]
    spots_3D.loc[clear_spot_inds, 'qof'] = np.nan

    # Update all values
    spots_3D.loc[upd_spot_inds, 'phase'] = new_phase
    spots_3D.loc[upd_spot_inds, 'grain_id'] = new_grains
    spots_3D.loc[upd_spot_inds, ['h', 'k', 'l']] = new_hkls
    spots_3D.loc[upd_spot_inds, 'qof'] = new_qofs








# from xrdmaptools.reflections.spot_blob_indexing_3D import (
#     pair_casting_index_full_pattern,
#     _get_connection_indices,
#     find_all_valid_pairs,
#     reduce_symmetric_equivalents,
#     decaying_pattern_decomposition
# )

# def dask_index_all_3D_spots(self,
#                         near_q,
#                         near_angle,
#                         phase=None,
#                         degrees=None,
#                         save_to_hdf=True,
#                         verbose=False):

#     if not hasattr(self, 'spots_3D') or self.spots_3D is None:
#         err_str = 'Spots must be found before they can be indexed.'
#         raise AttributeError(err_str)
    
#     if phase is None:
#         if len(self.phases) == 1:
#             phase = list(self.phases.values())[0]
#         else:
#             err_str = 'Phase must be provided for indexing.'
#             raise ValueError(err_str)

#     # Get phase information
#     all_q_mags = np.linalg.norm(self.spots_3D[['qx',
#                                                'qy',
#                                                'qz']], axis=1)
#     max_q = np.max(all_q_mags)

#     phase.generate_reciprocal_lattice(1.15 * max_q)
#     all_ref_qs = phase.all_qs.copy()
#     all_ref_hkls = phase.all_hkls.copy()
#     all_ref_fs = phase.all_fs.copy()
#     all_ref_mags = np.linalg.norm(all_ref_qs, axis=1)

#     # Find minimum q vector step size from reference phase
#     min_q = np.min(np.linalg.norm(phase.Q([[1, 0, 0],
#                                            [0, 1, 0],
#                                            [0, 0, 1]]),
#                                                 axis=0))

#     # Function for scheduled indexing
#     # Removed all external references in hopes of increased speed...
#     @dask.delayed
#     def delayed_indexing(spot_indices, spot_qs, spot_ints):
#         # pixel_df = self.pixel_3D_spots(indices, copied=False)
#         # spots = pixel_df[['qx', 'qy', 'qz']].values
        
#         all_rel_inds = []
#         all_grain_ids = []
#         all_qofs = []
#         all_hkls = []

#         if len(spot_indices) > 1 and not are_collinear(spot_qs):

#             spot_q_mags = np.linalg.norm(spot_qs, axis=1)
#             max_spot_q = np.max(spot_q_mags)
#             min_spot_q = np.min(spot_q_mags)

#             ext = 0.15
#             ref_mask = ((min_spot_q * (1 - ext) < all_ref_mags)
#                         & (all_ref_mags < max_spot_q * (1 + ext)))

#             (conns,
#              qofs) = pair_casting_index_full_pattern(
#                                         all_ref_qs[ref_mask],
#                                         all_ref_hkls[ref_mask],
#                                         all_ref_fs[ref_mask],
#                                         min_q,
#                                         spot_qs,
#                                         spot_ints,
#                                         near_q,
#                                         near_angle,
#                                         self.qmask,
#                                         degrees=degrees,
#                                         verbose=verbose)
            
#             return all_rel_inds, all_grain_ids, all_qofs, all_hkls

#             for grain_id, (conn, qof) in enumerate(zip(conns, qofs)):
#                 # Get values
#                 (spot_inds,
#                     hkl_inds) = _get_connection_indices(conn)

#                 # Collect results
#                 rel_ind = spot_indices[spot_inds]
#                 all_rel_inds.extend(rel_ind)
#                 all_grain_ids.extend([grain_id,] * len(spot_inds))
#                 all_qofs.extend([qof,] * len(spot_inds))
#                 all_hkls.extend(hkls[hkl_inds])

#         return all_rel_inds, all_grain_ids, all_qofs, all_hkls
            
#     # Update spots dataframe with new columns
#     self.spots_3D['phase'] = ''
#     self.spots_3D[['grain_id', 'h', 'k', 'l', 'qof']] = np.nan

#     # Iterate through each spatial pixel of map
#     delayed_list = []
#     for index in range(np.prod(self.xdms_vector_map.shape)):
#         indices = np.unravel_index(index, self.xdms_vector_map.shape)
#         pixel_df = self.pixel_3D_spots(indices)
#         spot_indices = pixel_df.index.values
#         spot_qs = pixel_df[['qx', 'qy', 'qz']].values
#         spot_ints = pixel_df['intensity'].values
#         if verbose:
#             print(f'Indexing for map indices {indices}.')
        
#         # Collect scheduled calls
#         delayed_list.append(delayed_indexing(spot_indices, spot_qs, spot_ints))

#     # Compute scheduled operations
#     if not verbose:
#         with TqdmCallback(tqdm_class=tqdm):
#             proc_list = dask.compute(*delayed_list)
#     else:
#         proc_list = dask.compute(*delayed_list)
    
#     return
    
#     # Unpack data into useable format
#     all_inds = list(itertools.chain(*[res[0] for res in proc_list]))
#     self.spots_3D.loc[all_inds, 'phase'] = phase.name
#     self.spots_3D.loc[all_inds, 'grain_id'] = list(itertools.chain(*[res[1] for res in proc_list]))
#     self.spots_3D.loc[all_inds, 'qof'] = list(itertools.chain(*[res[2] for res in proc_list]))
#     self.spots_3D.loc[all_inds, ['h', 'k', 'l']] = list(itertools.chain(*[res[3] for res in proc_list]))

#     # Write to hdf
#     if save_to_hdf:
#         self.save_3D_spots(
#                 extra_attrs={'near_q' : near_q,
#                              'near_angle' : near_angle,
#                              'degrees' : degrees})



# def new_dask_index_all_3D_spots(self,
#                            near_q,
#                            near_angle,
#                            phase=None,
#                            degrees=None,
#                            save_to_hdf=True,
#                            verbose=False,
#                            half_mask=True):

#     if not hasattr(self, 'spots_3D') or self.spots_3D is None:
#         err_str = 'Spots must be found before they can be indexed.'
#         raise AttributeError(err_str)
    
#     if phase is None:
#         if len(self.phases) == 1:
#             phase = list(self.phases.values())[0]
#         else:
#             err_str = 'Phase must be provided for indexing.'
#             raise ValueError(err_str)
    
#     map_shape = (np.max(self.spots_3D['map_y'] + 1),
#                  np.max(self.spots_3D['map_x'] + 1))

#     # Get phase information
#     max_q = np.max(self.spots_3D['q_mag'])

#     phase.generate_reciprocal_lattice(1.15 * max_q)
#     all_ref_qs = phase.all_qs.copy()
#     all_ref_hkls = phase.all_hkls.copy()
#     all_ref_fs = phase.all_fs.copy()

#     # Ignore half...
#     if half_mask:
#         half_mask = all_ref_hkls[:, -1] <= 0
#         all_ref_qs = all_ref_qs[half_mask]
#         all_ref_hkls = all_ref_hkls[half_mask]
#         all_ref_fs = all_ref_fs[half_mask]

#     ref_mags = np.linalg.norm(all_ref_qs, axis=1)

#     # Find minimum q vector step size from reference phase
#     min_q = phase.min_q

#     @dask.delayed
#     def delayed_indexing(pixel_df):
#         spots = pixel_df[['qx', 'qy', 'qz']].values
        
#         if len(spots) > 1 and not are_collinear(spots):
#             spot_mags = pixel_df['q_mag'].values
#             spot_ints = pixel_df['intensity'].values
#             ext = 0.15
#             ref_mask = ((ref_mags > spot_mags.min() * (1 - ext))
#                         & (ref_mags < spot_mags.max() * (1 + ext)))

#             (conns,
#              qofs) = pair_casting_index_full_pattern(
#                                         all_ref_qs[ref_mask],
#                                         all_ref_hkls[ref_mask],
#                                         all_ref_fs[ref_mask],
#                                         min_q,
#                                         spots,
#                                         spot_ints,
#                                         near_q,
#                                         near_angle,
#                                         self.qmask,
#                                         degrees=degrees,
#                                         verbose=verbose)
            
#             return
            
#             for grain_id, (conn, qof) in enumerate(zip(conns, qofs)):
#                 # Get values
#                 (spot_inds,
#                  hkl_inds) = _get_connection_indices(conn)
#                 hkls = all_ref_hkls[ref_mask][hkl_inds]

#                 # Assign values
#                 rel_ind = pixel_df.index[spot_inds]
#                 # self.spots_3D.loc[rel_ind, 'phase'] = phase.name
#                 # self.spots_3D.loc[rel_ind, 'grain_id'] = grain_id
#                 # self.spots_3D.loc[rel_ind, 'qof'] = qof
#                 # self.spots_3D.loc[rel_ind, ['h', 'k', 'l']] = hkls
    
#     # Update spots dataframe with new columns
#     self.spots_3D['phase'] = ''
#     self.spots_3D[['grain_id', 'h', 'k', 'l', 'qof']] = np.nan

#     # Iterate through each spatial pixel of map
#     delayed_results = []
#     for index in range(np.prod(map_shape)):
#         indices = np.unravel_index(index, map_shape)        
#         pixel_df = self.pixel_3D_spots(indices, copied=True)
        
#         if len(pixel_df) > 1:
#             delayed_results.append(delayed_indexing(pixel_df))
    
#     # Compute scheduled operations
#     if not verbose:
#         with TqdmCallback(tqdm_class=tqdm):
#             proc_list = dask.compute(*delayed_results)
#     else:
#         proc_list = dask.compute(*delayed_results)

#     # Write to hdf
#     # if save_to_hdf:
#     #     self.save_3D_spots(
#     #             extra_attrs={'near_q' : near_q,
#     #                             'near_angle' : near_angle,
#     #                             'degrees' : int(degrees)})

# def delayed_pair_indexing(all_ref_qs,
#                           all_ref_hkls,
#                                     all_ref_fs,
#                                     min_q,
#                                     all_spot_qs,
#                                     all_spot_ints,
#                                     near_q,
#                                     near_angle,
#                                     qmask,
#                                     degrees=False,
#                                     qof_minimum=0.2,
#                                     max_ori_refine_iter=50,
#                                     max_ori_decomp_count=20,
#                                     keep_initial_pair=False,
#                                     generate_reciprocal_lattice=False,
#                                     verbose=True):

#     # Find all valid pairs within near_q and near_angle
#     pairs = dask.delayed(find_all_valid_pairs)(
#             all_spot_qs,
#             all_ref_qs,
#             near_q,
#             near_angle,
#             min_q,
#             degrees=degrees,
#             verbose=verbose)
    
#     if len(pairs) > 0:
#         # Symmetrically reduce pairs
#         red_pairs = dask.delayed(reduce_symmetric_equivalents)(
#                 pairs,
#                 all_spot_qs,
#                 all_ref_qs,
#                 all_ref_hkls,
#                 near_angle,
#                 min_q,
#                 degrees=degrees,
#                 verbose=verbose)
        
#         if len(red_pairs) > 0:
#             # Iteratively decompose patterns
#             (best_connections, 
#              best_qofs) = dask.delayed(decaying_pattern_decomposition)(
#                     red_pairs,
#                     all_spot_qs,
#                     all_spot_ints,
#                     all_ref_qs,
#                     all_ref_fs,
#                     qmask,
#                     near_q,
#                     qof_minimum=qof_minimum,
#                     keep_initial_pair=keep_initial_pair,
#                     max_ori_refine_iter=max_ori_refine_iter,
#                     max_ori_decomp_count=max_ori_decomp_count,
#                     verbose=verbose)
#         else:
#             # This is where all nans are coming from!
#             best_connections = [[np.nan,] * len(all_spot_qs)]
#             best_qofs = [np.nan]
#     else:
#         # This is where all nans are coming from!
#         best_connections = [[np.nan,] * len(all_spot_qs)]
#         best_qofs = [np.nan]

#     return best_connections, best_qofs


# def new_new_dask_index_all_3D_spots(self,
#                            near_q,
#                            near_angle,
#                            phase=None,
#                            degrees=None,
#                            save_to_hdf=True,
#                            verbose=False,
#                            half_mask=True):

#     if not hasattr(self, 'spots_3D') or self.spots_3D is None:
#         err_str = 'Spots must be found before they can be indexed.'
#         raise AttributeError(err_str)
    
#     if phase is None:
#         if len(self.phases) == 1:
#             phase = list(self.phases.values())[0]
#         else:
#             err_str = 'Phase must be provided for indexing.'
#             raise ValueError(err_str)
    
#     map_shape = (np.max(self.spots_3D['map_y'] + 1),
#                  np.max(self.spots_3D['map_x'] + 1))

#     # Get phase information
#     max_q = np.max(self.spots_3D['q_mag'])

#     phase.generate_reciprocal_lattice(1.15 * max_q)
#     all_ref_qs = phase.all_qs.copy()
#     all_ref_hkls = phase.all_hkls.copy()
#     all_ref_fs = phase.all_fs.copy()

#     # Ignore half...
#     if half_mask:
#         half_mask = all_ref_hkls[:, -1] <= 0
#         all_ref_qs = all_ref_qs[half_mask]
#         all_ref_hkls = all_ref_hkls[half_mask]
#         all_ref_fs = all_ref_fs[half_mask]

#     ref_mags = np.linalg.norm(all_ref_qs, axis=1)

#     # Find minimum q vector step size from reference phase
#     min_q = phase.min_q

#     # Construct iterable
#     if verbose:
#         iterable = timed_iter(range(np.prod(map_shape)))
#     else:
#         iterable = tqdm(range(np.prod(map_shape)))

#     # Iterate through each spatial pixel of map
#     delayed_results = []
#     for index in iterable:
#         indices = np.unravel_index(index, map_shape)
#         if verbose:
#             print(f'Indexing for map indices {indices}.')
        
#         pixel_df = self.pixel_3D_spots(indices, copied=False)
#         spots = pixel_df[['qx', 'qy', 'qz']].values
        
#         if len(spots) > 1 and not are_collinear(spots):
#             spot_mags = pixel_df['q_mag'].values
#             spot_ints = pixel_df['intensity'].values
#             ext = 0.15
#             ref_mask = ((ref_mags > spot_mags.min() * (1 - ext))
#                         & (ref_mags < spot_mags.max() * (1 + ext)))

#             delayed_results.append(delayed_pair_indexing(
#                                         all_ref_qs[ref_mask],
#                                         all_ref_hkls[ref_mask],
#                                         all_ref_fs[ref_mask],
#                                         min_q,
#                                         spots,
#                                         spot_ints,
#                                         near_q,
#                                         near_angle,
#                                         self.qmask,
#                                         degrees=degrees,
#                                         verbose=verbose))
    
#     # Update spots dataframe with new columns
#     self.spots_3D['phase'] = ''
#     self.spots_3D[['grain_id', 'h', 'k', 'l', 'qof']] = np.nan

#     # Iterate through each spatial pixel of map
#     delayed_results = []
#     for index in range(np.prod(map_shape)):
#         indices = np.unravel_index(index, map_shape)        
#         pixel_df = self.pixel_3D_spots(indices, copied=True)
        
#         if len(pixel_df) > 1:
#             delayed_results.append(delayed_indexing(pixel_df))
    
#     # Compute scheduled operations
#     if not verbose:
#         with TqdmCallback(tqdm_class=tqdm):
#             proc_list = dask.compute(*delayed_results)
#     else:
#         proc_list = dask.compute(*delayed_results)

#     for proc in tqdm(proc_list):
#         for grain_id, (conn, qof) in enumerate(zip(*proc)):
#             # Get values
#             (spot_inds,
#                 hkl_inds) = _get_connection_indices(conn)
#             hkls = all_ref_hkls[ref_mask][hkl_inds]

#             # Assign values
#             rel_ind = pixel_df.index[spot_inds]
#             self.spots_3D.loc[rel_ind, 'phase'] = phase.name
#             self.spots_3D.loc[rel_ind, 'grain_id'] = grain_id
#             self.spots_3D.loc[rel_ind, 'qof'] = qof
#             self.spots_3D.loc[rel_ind, ['h', 'k', 'l']] = hkls

#     # Write to hdf
#     # if save_to_hdf:
#     #     self.save_3D_spots(
#     #             extra_attrs={'near_q' : near_q,
#     #                             'near_angle' : near_angle,
#     #                             'degrees' : int(degrees)})


from xrdmaptools.crystal.strain import phase_get_strain_orientation
from xrdmaptools.crystal.strain import get_strain_orientation
from xrdmaptools.crystal.crystal import LatticeParameters

def get_strain_and_orientation_maps(spots_3D, phase, grain_id=0):

    # Effective map shape
    map_shape = (np.max(spots_3D['map_y']) + 1,
                 np.max(spots_3D['map_x']) + 1)

    # Setup containers
    ori_map = np.empty((*map_shape, 3, 3), dtype=float)
    ori_map[:] = np.nan
    e_map = ori_map.copy()
    lat_map = np.empty((*map_shape, 6), dtype=float)
    lat_map[:] = np.nan

    for index in tqdm(range(np.prod(map_shape))):
        indices = np.unravel_index(index, map_shape)
        pixel_df = pixel_spots(spots_3D, indices)
        grain_mask = pixel_df['grain_id'] == grain_id

        q_vectors = pixel_df[grain_mask][['qx', 'qy', 'qz']].values
        hkls = pixel_df[grain_mask][['h', 'k', 'l']].values

        if len(q_vectors) < 3 or are_coplanar(hkls):
            continue

        # out = phase_get_strain_orientation(q_vectors,
        #                                    hkls,
        #                                    phase)

        out = get_strain_orientation(q_vectors,
                                           hkls,
                                           phase)

        eij, ori, strained = out

        ori_map[indices] = ori
        e_map[indices] = eij
        lat_map[indices] = strained.a, strained.b, strained.c, strained.alpha, strained.beta, strained.gamma
    
    return ori_map, e_map, lat_map



# def quick_processing():
#     wd = '/nsls2/data/srx/proposals/2025-1/pass-316224/figures/'
#     scan = 'scan167103-167175'
    
#     lat_params = [phase.a, phase.b, phase.c, phase.alpha, phase.beta, phase.gamma]
#     title_stubs = [f'{i}_lattice_parameter' for i in ['a', 'b', 'c', 'alpha', 'beta', 'gamma']]

#     strain_ext = 5e-2
#     angle_ext = 2
#     strain_ext_func = lambda x : (x * (1 - strain_ext), x * (1 + strain_ext))
#     angle_ext_func = lambda x : (x - angle_ext, x + angle_ext)
#     ext_funcs = ([strain_ext_func,] * 3) + ([angle_ext_func,] * 3)

#     def gen_plots(lat_map):
#         for i in range(6):
#             param_map = lat_map[..., i]
#             vmin, vmax = ext_funcs[i](lat_params[i])
#             fig, ax = xdms.plot_map(param_map,
#                                     vmin=vmin,
#                                     vmax=vmax,
#                                     title=title_stubs[i],
#                                     return_plot=True)
#             fig.savefig(f'{wd}{title}{title_stubs[i]}.png')


#     title = f'{scan}_full_'
#     ori_map, e_map, lat_map = get_strain_and_orientation_maps(xdms.spots_3D, LatticeParameters.from_Phase(phase), grain_id=0)
#     lat_map[..., 3:] = np.degrees(lat_map[..., 3:])
#     gen_plots(lat_map)

#     title = f'{scan}_grain0_'
#     ori_map, e_map, lat_map = get_strain_and_orientation_maps(all_spots, LatticeParameters.from_Phase(phase), grain_id=0)
#     lat_map[..., 3:] = np.degrees(lat_map[..., 3:])
#     gen_plots(lat_map)

#     title = f'{scan}_grain1_'
#     ori_map, e_map, lat_map = get_strain_and_orientation_maps(all_spots, LatticeParameters.from_Phase(phase), grain_id=1)
#     lat_map[..., 3:] = np.degrees(lat_map[..., 3:])
#     gen_plots(lat_map)

#     title = f'{scan}_grain2_'
#     ori_map, e_map, lat_map = get_strain_and_orientation_maps(all_spots, LatticeParameters.from_Phase(phase), grain_id=2)
#     lat_map[..., 3:] = np.degrees(lat_map[..., 3:])
#     gen_plots(lat_map)

#     plt.close('all')



# Testing Graph theory
import networkx as nx

def connections_to_graph(pairs):

    point_conversions = [] # (spot_ind, ref_ind)

    # Find all graph points
    for spot_ind in range(pairs.shape[1]):
        ref_inds = np.unique(pairs[:, spot_ind][~np.isnan(pairs[:, spot_ind])])

        for ref_ind in ref_inds:
            point_conversions.append((int(spot_ind), int(ref_ind)))

    # Decomposes connections into points and find edges
    edge_list= []
    for pair in pairs:
        spot_inds = np.nonzero(~np.isnan(pair))[0]
        ref_inds = pair[spot_inds]

        points = [point_conversions.index((int(spot_ind), int(ref_ind)))
                  for spot_ind, ref_ind in zip(spot_inds, ref_inds)]
        
        edge_list.append(tuple(points))

        # G.add_edge(points)
    
    G = nx.Graph(edge_list)
    cliques = list(nx.find_cliques(G))

    qofs = []
    point_conversions = np.asarray(point_conversions)
    for clique in tqdm(cliques):

        spot_inds, ref_inds = point_conversions[clique].T

        conn_spots = all_spot_qs[spot_inds]
        conn_refs = all_ref_qs[ref_mask][ref_inds]
        orientation, _ = Rotation.align_vectors(conn_spots,
                                                conn_refs)
        all_rot_qs = orientation.apply(all_ref_qs[ref_mask], inverse=False)
        temp_qmask = qmask.generate(all_rot_qs)

        qof = get_quality_of_fit(
            all_spot_qs[spot_inds], # fit_spot_qs
            all_spot_ints[spot_inds], # fit_spot_ints
            all_rot_qs[ref_inds], # fit_rot_qs
            all_ref_fs[ref_mask][ref_inds], # fit_ref_fs
            all_spot_qs, # all_spot_qs
            all_spot_ints, # all_spot_ints
            all_rot_qs[temp_qmask], # all_rot_qs
            all_ref_fs[ref_mask][temp_qmask], # all_ref_fs
            sigma=0.1)
        qofs.append(qof)
    
    
    return point_conversions, edge_list, cliques, qofs, G


def connection_from_clique(clique, point_conversions, spot_num):

    conn = [np.nan,] * spot_num

    for spot_ind, ref_ind in point_conversions[clique]:
        conn[spot_ind] = ref_ind
    
    return np.asarray(conn)


from xrdmaptools.utilities.utilities import tic, toc
spot_nums = []
times = []

def tic(output=False):
    global _t0
    _t0 = ttime.monotonic()
    if output:
        return _t0


def toc(string='', print_msg=True, output=False):
    global _t0
    dt = ttime.monotonic() - _t0
    s = f'{string}\ndt = {dt:.3f} sec'
    if print_msg:
        print(s, end='\n')
    if output:
        return dt

def timed(func):
    def wrapper(*args, **kwargs):
        tic()
        out = func(*args, **kwargs)
        times.append(toc(output=True, print_msg=False))
        return out
    return wrapper

# from xrdmaptools.reflections.spot_blob_indexing_3D import pair_casting_index_best_grain, pair_casting_index_full_pattern
# pair_casting_index_best_grain = timed(pair_casting_index_best_grain)
# pair_casting_index_full_pattern = timed(pair_casting_index_full_pattern)

@timed
def test_indexing(all_spot_qs,
                  all_spot_ints,
                  all_ref_qs,
                  all_ref_hkls,
                  all_ref_fs,
                  near_q,
                  near_angle,
                  min_q,
                  qmask,
                  qof_minimum=0,
                  max_grains=20,
                  degrees=False,
                  symmeterize=True,
                  verbose=True):
    
    if near_q > min_q * 0.85:
        err_str = ("'near_q' threshold is greater than 85% of minimum "
                   + "lattice spacing. This seems unwise.")
        raise ValueError(err_str)    
    
    # Find vector magnitudes for measured and reference reflections
    spot_q_mags = np.linalg.norm(all_spot_qs, axis=1)
    ref_q_mags = np.linalg.norm(all_ref_qs, axis=1)

    # Find difference between measured and calculated q magnitudes
    mag_diff_arr = np.abs(spot_q_mags[:, np.newaxis]
                          - ref_q_mags[np.newaxis, :])
    
    # Eliminate any reflections outside of phase-allowed spots
    phase_mask = np.any(mag_diff_arr < near_q, axis=1)
    phase_inds = np.nonzero(phase_mask)[0]
    mag_diff_arr = mag_diff_arr[phase_mask]
    all_spot_qs = all_spot_qs[phase_mask]
    spot_q_mags = spot_q_mags[phase_mask]

    # Generate all pairs of spots which are crystallographically feasible
    spot_pair_indices = list(combinations(range(len(all_spot_qs)), 2))
    spot_pair_dist = distance_matrix(all_spot_qs, all_spot_qs) 
    allowed_pairs = [spot_pair_dist[tuple(indices)] > min_q * 0.85
                      for indices in spot_pair_indices]
    spot_pair_indices = np.asarray(spot_pair_indices)[allowed_pairs]
    
    # Determine all angles
    spot_angles = multi_vector_angles(all_spot_qs, all_spot_qs, degrees=degrees)
    ref_angles = multi_vector_angles(all_ref_qs, all_ref_qs, degrees=degrees)

    # Construct iterable
    if verbose:
        iterate = lambda x : tqdm(x, position=0, leave=True)
    else:
        iterate = lambda x : x

    if verbose:
        print('Finding all valid pairs...', flush=True)

    blank_connection = np.array([np.nan,] * len(phase_mask))
    connection_pairs = []
    pairs = []    
    for pair in iterate(spot_pair_indices):
        ref_combos = list(product(*[np.nonzero(mag_diff_arr[i] < near_q)[0]
                          for i in pair]))

        angle_mask = [np.abs(spot_angles[tuple(pair)]
                      - ref_angles[tuple(combo)]) < near_angle
                      for combo in ref_combos]
        doublet_mask = [combo[0] != combo[1] for combo in ref_combos]
        collinear_mask = [ref_angles[tuple(combo)] > 0 for combo in ref_combos]

        ref_combos = np.asarray(ref_combos)[(np.asarray(angle_mask)
                                             & np.asarray(doublet_mask)
                                             & np.asarray(collinear_mask))]
        
        # Rotation based symmeterization
        if len(ref_combos) > 0:
            temp_connections = []
            temp_pairs, temp_pair_mask = [], []
            pair_chars, pair_mags, pair_matches = [], [], []
            match_id = 0
            
            # Check all possible combintations from pair
            for combo in ref_combos:
                
                # Remove orientationally invariant combinations
                if are_collinear(all_ref_hkls[combo]):
                    continue

                # Add pairs
                temp_pairs.append([(s, r) for s, r in zip(pair, combo)])
                temp_pair_mask.append(True)

                # Add connections (MAY BE REMOVED LATER)
                connection = blank_connection.copy()
                connection[phase_inds[pair[0]]] = combo[0]
                connection[phase_inds[pair[1]]] = combo[1]
                temp_connections.append(connection)

                # Qualify potential fits and their orientationally magnitude
                if symmeterize:
                    # Characterize combo
                    combo_char = (ref_q_mags[combo[0]].round(10),
                                  ref_q_mags[combo[1]].round(10),
                                  ref_angles[tuple(combo)].round(10))
                    if combo_char not in pair_chars:
                        match_id += 1
                        pair_matches.append(match_id)
                    else:
                        pair_matches.append(pair_chars.index(combo_char))
                    pair_chars.append(combo_char)
                    
                    # Orientation magnitude of combo. Will chose smallest-ish later
                    combo_mag = Rotation.align_vectors(
                                        all_spot_qs[pair],
                                        all_ref_qs[combo])[0].magnitude()
                    if degrees:
                        combo_mag = np.degrees(combo_mag)
                    pair_mags.append(combo_mag)
            
            # Pick 
            if symmeterize:
                temp_pair_mask = np.asarray([False,] * len(temp_pair_mask))
                for idx in np.unique(pair_matches):
                    equi_mask = pair_matches == idx
                    min_angle = np.min(np.asarray(pair_mags)[equi_mask])
                    keep_mask = np.asarray(pair_mags)[equi_mask] < min_angle + near_angle
                    temp_pair_mask[np.nonzero(equi_mask)[0][keep_mask]] = True
            
            pairs.extend(np.asarray(temp_pairs)[temp_pair_mask])
            connection_pairs.extend(np.asarray(temp_connections)[temp_pair_mask])


        # Magnitude base symmeterization - BAD        
        # if len(ref_combos) > 0:
        #     temp_pairs, temp_pair_mask = [], []
        #     pair_characteristics, pair_matches, pair_evaluators = [], [], []
        #     match_id = 0
        #     for combo in ref_combos:
        #         temp_pairs.append([(s, r) for s, r in zip(pair, combo)])
        #         temp_pair_mask.append(True)
        #         if symmeterize:
        #             characteristics = (ref_q_mags[combo[0]].round(10),
        #                                ref_q_mags[combo[1]].round(10),
        #                                ref_angles[tuple(combo)].round(10))
        #             if characteristics not in pair_characteristics:
        #                 match_id += 1
        #             pair_characteristics.append(characteristics)
        #             pair_matches.append(match_id)
        #             # Simple way to get close to fundamental zone
        #             pair_evaluators.append(np.sum([mag_diff_arr[m] for m in temp_pairs[-1]]))
                
        #     if symmeterize:
        #         temp_pair_mask = [False,] * len(temp_pair_mask)
        #         for idx in np.unique(pair_matches):
        #             equi_mask = pair_matches == idx
        #             keep_ind = np.argmin(np.asarray(pair_evaluators)[equi_mask])
        #             temp_pair_mask[np.nonzero(equi_mask)[0][keep_ind]] = True
            
        #     pairs.extend(np.asarray(temp_pairs)[temp_pair_mask])

        # # No symmeterization
        # if len(ref_combos) > 0:
        #     pair_characteristics = []
        #     for combo in ref_combos:
        #         pairs.append([(s, r) for s, r in zip(pair, combo)])
            
        #         connection = blank_connection.copy()
        #         connection[phase_inds[pair[0]]] = combo[0]
        #         connection[phase_inds[pair[1]]] = combo[1]
        #         connection_pairs.append(connection)

    # if len(connection_pairs) > 0:
    #     red_pairs = reduce_symmetric_equivalents(np.asarray(connection_pairs),
    #                                             all_spot_qs,
    #                                             all_ref_qs,
    #                                             all_ref_hkls,
    #                                             near_angle,
    #                                             min_q,
    #                                             degrees=degrees,
    #                                             verbose=verbose)

    #     pairs = []
    #     for pair in red_pairs:
    #         (spot_inds, ref_inds) = _get_connection_indices(pair)
    #         pairs.append([(s, r) for s, r in zip(spot_inds, ref_inds)])

    # return pairs

    if verbose:
        print('Constructing graph and finding best cliques...')
    
    # Find all unique graph points and construct from edges
    point_conversions = [] # (spot_ind, ref_ind)
    edge_list = []
    for pair in pairs:
        edge_points = []
        for point_values in pair:
            if tuple(point_values) not in point_conversions:
                point_conversions.append(tuple(point_values))
            edge_points.append(point_conversions.index(tuple(point_values)))
        edge_list.append(tuple(edge_points))

    G = nx.Graph(edge_list)
    cliques = list(nx.find_cliques(G))

    # Remove symmetrically equivalent cliques
    if symmeterize:
        pass
        # clique_character = []
        # clique_ids = []
        # match_id = 0
        # for i, clique in enumerate(cliques):
        #     character = []
        #     for point in clique:
        #         spot_ind, ref_ind = point_conversions[point]
        #         character.append(ref_q_mags[ref_ind].round(10))
        #     character = tuple(character)

        #     if character in clique_character:
        #         clique_ids.append(clique_ids[clique_character.index(character)])
        #     else:
        #         match_id += 1
        #         clique_ids.append(match_id)
        #     clique_character.append(character)

        # sym_cliques = []
        # clique_ids = np.asarray(clique_ids)
        # for idx in np.unique(clique_ids):
        #     sym_inds = np.nonzero(clique_ids == idx)[0]
        #     min_ind = np.argmin([clique_character[ind][-1] for ind in sym_inds])

        #     sym_cliques.append(cliques[sym_inds[min_ind]])

        # cliques = sym_cliques

        # clique_character = []
        # sym_cliques = []
        # for clique in cliques:
        #     character = []
        #     for point in clique:
        #         spot_ind, ref_ind = point_conversions[point]
        #         character.append([spot_ind, ref_q_mags[ref_ind].round(10)])
        #     character = tuple(character)

        #     if character not in clique_character:
        #         sym_cliques.append(clique)
        #         clique_character.append(character)
    
        # cliques = sym_cliques

        # clique_character = []
        # clique_mags = []
        # sym_cliques = []
        # for clique in cliques:
        #     character, spots = [], []
        #     for point in clique:
        #         spot_ind, ref_ind = point_conversions[point]
        #         character.extend([spot_ind, ref_q_mags[ref_ind].round(10)])
        #     character = tuple(character)
        #     mag = np.sum([mag_diff_arr[point_conversions[point]] for point in clique])

        #     if character not in clique_character:
        #         sym_cliques.append(clique)
        #         clique_character.append(character)
        #         clique_mags.append(mag)
        #     else:
        #         ind = clique_character.index(character)
        #         if clique_mags[ind] > mag:
        #             clique_mags[ind] = mag
        #             sym_cliques[ind] = clique
    
        # cliques = sym_cliques

    # return cliques, point_conversions

    qofs = []
    point_conversions = np.asarray(point_conversions)

    for clique in iterate(cliques):

        spot_inds, ref_inds = point_conversions[clique].T

        conn_spots = all_spot_qs[spot_inds]
        conn_refs = all_ref_qs[ref_inds]
        orientation, _ = Rotation.align_vectors(conn_spots,
                                                conn_refs)
        all_rot_qs = orientation.apply(all_ref_qs, inverse=False)
        temp_qmask = qmask.generate(all_rot_qs)

        qof = get_quality_of_fit(
            all_spot_qs[spot_inds], # fit_spot_qs
            all_spot_ints[spot_inds], # fit_spot_ints
            all_rot_qs[ref_inds], # fit_rot_qs
            all_ref_fs[ref_inds], # fit_ref_fs
            all_spot_qs, # all_spot_qs
            all_spot_ints, # all_spot_ints
            all_rot_qs[temp_qmask], # all_rot_qs
            all_ref_fs[temp_qmask], # all_ref_fs
            sigma=0.1)
        qofs.append(qof)
    
    # return point_conversions, cliques, qofs

    # Dummy check
    if len(cliques) < 1:
        return [], []

    # Decompose cliques into best fitting
    best_cliques, best_qofs = [], []
    iterations = 1
    while True:

        # Add best clique
        best_ind = np.argmax(qofs)
        points = cliques[best_ind]
        best_cliques.append(points)
        best_qofs.append(qofs[best_ind])

        # Update cliques and qofs
        used_spots = [point_conversions[point][0] for point in points]
        removed_cliques = []
        for i, clique in enumerate(cliques):
            for point in clique:
                if point_conversions[point][0] in used_spots:
                    removed_cliques.append(i)
                    break
        cliques = [c for i, c in enumerate(cliques) if i not in removed_cliques]
        qofs = [q for i, q in enumerate(qofs) if i not in removed_cliques]

        if (len(cliques) < 1
            or best_qofs[-1] < qof_minimum
            or iterations > max_grains):
            break
            # pass
        else:
            iterations += 1

    # return best_cliques, best_qofs

    best_connections = [connection_from_clique(c, point_conversions, len(all_spot_qs)) for c in best_cliques]

    return best_connections, best_qofs





def test_index_all_3D_spots(self,
                        near_q,
                        near_angle,
                        phase=None,
                        degrees=None,
                        save_to_hdf=True,
                        verbose=False,
                        half_mask=True):

    if not hasattr(self, 'spots_3D') or self.spots_3D is None:
        err_str = 'Spots must be found before they can be indexed.'
        raise AttributeError(err_str)
    
    if phase is None:
        if len(self.phases) == 1:
            phase = list(self.phases.values())[0]
        else:
            err_str = 'Phase must be provided for indexing.'
            raise ValueError(err_str)
    
    # Effective map shape
    map_shape = (np.max(self.spots_3D['map_y']) + 1,
                 np.max(self.spots_3D['map_x']) + 1)

    # Get phase information
    max_q = np.max(self.spots_3D['q_mag'])
    phase.generate_reciprocal_lattice(1.15 * max_q)
    all_ref_qs = phase.all_qs.copy()
    all_ref_hkls = phase.all_hkls.copy()
    all_ref_fs = phase.all_fs.copy()

    # Ignore half...
    if half_mask:
        half_mask = all_ref_hkls[:, -1] <= 0
        all_ref_qs = all_ref_qs[half_mask]
        all_ref_hkls = all_ref_hkls[half_mask]
        all_ref_fs = all_ref_fs[half_mask]

    ref_mags = np.linalg.norm(all_ref_qs, axis=1)

    # Find minimum q vector step size from reference phase
    min_q = phase.min_q
    
    # Update spots dataframe with new columns
    self.spots_3D['phase'] = ''
    self.spots_3D[['grain_id', 'h', 'k', 'l', 'qof']] = np.nan

    # Construct iterable
    if verbose:
        iterable = timed_iter(range(np.prod(map_shape)))
    else:
        iterable = tqdm(range(np.prod(map_shape)))

    # Iterate through each spatial pixel of map
    for index in iterable:
        indices = np.unravel_index(index, map_shape)
        pixel_df = self.pixel_3D_spots(indices, copied=False)
        spots = pixel_df[['qx', 'qy', 'qz']].values

        if verbose:
            print(f'Indexing for map indices {indices}.')
            print(f'Pixel has {len(spots)} spots.')            
        
        if len(spots) > 1 and not are_collinear(spots):
            spot_mags = pixel_df['q_mag'].values
            spot_ints = pixel_df['intensity'].values
            ext = 0.15
            ref_mask = ((ref_mags > spot_mags.min() * (1 - ext))
                        & (ref_mags < spot_mags.max() * (1 + ext)))

            spot_nums.append(len(spots))  

            (conns,
                qofs) = pair_casting_index_full_pattern(
                                        all_ref_qs[ref_mask],
                                        all_ref_hkls[ref_mask],
                                        all_ref_fs[ref_mask],
                                        min_q,
                                        spots,
                                        spot_ints,
                                        near_q,
                                        near_angle,
                                        self.qmask,
                                        degrees=True,
                                        verbose=verbose)


            # conns, qofs = test_indexing(spots,
            #                         spot_ints,
            #                         all_ref_qs[ref_mask],
            #                         all_ref_hkls[ref_mask],
            #                         all_ref_fs[ref_mask],
            #                         near_q,
            #                         near_angle,
            #                         min_q,
            #                         self.qmask,
            #                         degrees=True,
            #                         verbose=verbose)
            
            for grain_id, (conn, qof) in enumerate(zip(conns, qofs)):
                # Get values
                (spot_inds,
                    hkl_inds) = _get_connection_indices(conn)
                hkls = all_ref_hkls[ref_mask][hkl_inds]

                # Assign values
                rel_ind = pixel_df.index[spot_inds]
                self.spots_3D.loc[rel_ind, 'phase'] = phase.name
                self.spots_3D.loc[rel_ind, 'grain_id'] = grain_id
                self.spots_3D.loc[rel_ind, 'qof'] = qof
                self.spots_3D.loc[rel_ind, ['h', 'k', 'l']] = hkls


def dask_index_all_3D_spots(self,
                        near_q,
                        near_angle,
                        phase=None,
                        degrees=None,
                        save_to_hdf=True,
                        verbose=False,
                        half_mask=True):

    if not hasattr(self, 'spots_3D') or self.spots_3D is None:
        err_str = 'Spots must be found before they can be indexed.'
        raise AttributeError(err_str)
    
    if phase is None:
        if len(self.phases) == 1:
            phase = list(self.phases.values())[0]
        else:
            err_str = 'Phase must be provided for indexing.'
            raise ValueError(err_str)
    
    # Effective map shape
    map_shape = (np.max(self.spots_3D['map_y']) + 1,
                    np.max(self.spots_3D['map_x']) + 1)

    # Get phase information
    max_q = np.max(self.spots_3D['q_mag'])
    phase.generate_reciprocal_lattice(1.15 * max_q)
    all_ref_qs = phase.all_qs.copy()
    all_ref_hkls = phase.all_hkls.copy()
    all_ref_fs = phase.all_fs.copy()

    # Ignore half...
    if half_mask:
        half_mask = all_ref_hkls[:, -1] <= 0
        all_ref_qs = all_ref_qs[half_mask]
        all_ref_hkls = all_ref_hkls[half_mask]
        all_ref_fs = all_ref_fs[half_mask]

    ref_mags = np.linalg.norm(all_ref_qs, axis=1)

    # Find minimum q vector step size from reference phase
    min_q = phase.min_q
    
    # Update spots dataframe with new columns
    self.spots_3D['phase'] = ''
    self.spots_3D[['grain_id', 'h', 'k', 'l', 'qof']] = np.nan

    @dask.delayed
    def delayed_indexing(spots, spot_ints, ref_mask):
        ttime.sleep(5)
        return [], []
        return test_indexing(spots,
                            spot_ints,
                            all_ref_qs[ref_mask],
                            all_ref_fs[ref_mask],
                            near_q,
                            near_angle,
                            min_q,
                            self.qmask,
                            degrees=True,
                            verbose=False)

    # Iterate through each spatial pixel of map
    delayed_list = []
    for index in range(np.prod(map_shape)):
        indices = np.unravel_index(index, map_shape)

        pixel_df = self.pixel_3D_spots(indices, copied=False)
        spots = pixel_df[['qx', 'qy', 'qz']].values     
        
        if len(spots) > 1 and not are_collinear(spots):
            spot_mags = pixel_df['q_mag'].values
            spot_ints = pixel_df['intensity'].values
            ext = 0.15
            ref_mask = ((ref_mags > spot_mags.min() * (1 - ext))
                        & (ref_mags < spot_mags.max() * (1 + ext)))

            delayed_list.append(delayed_indexing(spots, spot_ints, ref_mask))

    # Compute delayed results
    if verbose:
        with TqdmCallback(tqdm_class=tqdm):
            proc_list = dask.compute(*delayed_list)
    else:
        proc_list = dask.compute(*delayed_list)

    
    return proc_list


            # conns, qofs = test_indexing(spots,
            #                         spot_ints,
            #                         all_ref_qs[ref_mask],
            #                         all_ref_fs[ref_mask],
            #                         near_q,
            #                         near_angle,
            #                         min_q,
            #                         self.qmask,
            #                         degrees=True,
            #                         verbose=True)
            
            # for grain_id, (conn, qof) in enumerate(zip(conns, qofs)):
            #     # Get values
            #     (spot_inds,
            #         hkl_inds) = _get_connection_indices(conn)
            #     hkls = all_ref_hkls[ref_mask][hkl_inds]

            #     # Assign values
            #     rel_ind = pixel_df.index[spot_inds]
            #     self.spots_3D.loc[rel_ind, 'phase'] = phase.name
            #     self.spots_3D.loc[rel_ind, 'grain_id'] = grain_id
            #     self.spots_3D.loc[rel_ind, 'qof'] = qof
            #     self.spots_3D.loc[rel_ind, ['h', 'k', 'l']] = hkls