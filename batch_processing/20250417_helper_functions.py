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
from matplotlib import color_sequences

from orix.quaternion.orientation import Orientation
from orix.quaternion.symmetry import get_point_group
from orix.vector import Vector3d

# Local imports
from xrdmaptools.utilities.utilities import (
  rescale_array,
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
    # Pull global variables
        if update_coordinates(event):
            pix_ori = ori_map[row, col]
            mis_map = np.empty(ori_map.shape[:2])

            for index in range(np.prod(mis_map.shape)):
                indices = np.unravel_index(index, mis_map.shape)
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



def reindex_grains(xdms, all_spots, mis_thresh=0.5, degrees=True, map_shape=None):

    # Setup useful values
    if map_shape is None:
        map_shape = xdms.xdms_vector_map.shape
    # all_spots = xdms.spots_3D

    # Collect orientation maps of new grain ids
    ori_map_list = []
    matched_grain_ids = []
    blank_ori_map = np.empty((*map_shape, 3, 3))
    blank_ori_map[:] = np.nan

    # Function to find neighbor pixels
    # indices_vars = np.asarray(list(itertools.permutations([-1, 0, 1], 2)))
    # indices_vars = np.asarray([(-1, 0), (0, -1), (1, 0), (0, 1)])
    indices_vars = np.asarray([(-1, 0), (0, -1)])

    def get_near_indices(indices):
        new_indices = indices + indices_vars
        valid = ((new_indices[:, 0] >= 0)
                 & (new_indices[:, 0] < map_shape[0])
                 & (new_indices[:, 1] >= 0)
                 & (new_indices[:, 1] < map_shape[1]))

        return new_indices[valid]        

    for index in tqdm(range(np.prod(map_shape))):
        indices = np.unravel_index(index, map_shape)
        near_indices = get_near_indices(indices)

        df = xdms.pixel_3D_spots(indices)
        grain_ids = df['grain_id'].values
        grain_ids = np.unique(grain_ids[~np.isnan(grain_ids)])

        for grain_id in grain_ids:
            grain_mask = df['grain_id'] == grain_id
            grain_inds = df[grain_mask].index.values
            grain_spots = df[grain_mask][['qx', 'qy', 'qz']]
            grain_hkls = df[grain_mask][['h', 'k', 'l']]

            grain_ori = Rotation.align_vectors(grain_spots,
                                               grain_hkls)[0]

            matched_oris = [False,] * len(ori_map_list)
            ori_i = -1
            for ori_i, ori in enumerate(ori_map_list):
                # if ori is None:
                #     continue

                for near in near_indices:
                    # If orientation has been determined
                    # print(ori[tuple(near)])
                    if not np.any(np.isnan(ori[tuple(near)])):
                        # print('Found nearby index!')
                        # Find misorientation
                        misori = grain_ori.as_matrix() @ ori[tuple(near)].T
                        mis_mag = Rotation.from_matrix(misori).magnitude()
                        if degrees:
                            mis_mag = np.degrees(mis_mag)
                        
                        # Check if within threshold
                        if mis_mag < mis_thresh:
                            ori_map_list[ori_i][indices] = grain_ori.as_matrix()
                            matched_oris[ori_i] = True
                            # Update grain ids
                            all_spots.loc[grain_inds, 'grain_id'] = ori_i
                            # print('Threshold achieved!!!')
                            continue
            
            # No matches, create new!
            if np.sum(matched_oris) == 0:
                ori_map_list.append(blank_ori_map.copy())
                ori_map_list[-1][indices] = grain_ori.as_matrix()
                matched_grain_ids.append([ori_i + 1])
                # Update grain_ids
                all_spots.loc[grain_inds, 'grain_id'] = ori_i + 1

            # Mark stiched grains to be combined later
            elif np.sum(matched_oris) > 1:
                matched_inds = np.nonzero(matched_oris)[0]
                for ind in matched_inds:
                    if ind not in matched_grain_ids[matched_inds[0]]:
                        matched_grain_ids[matched_inds[0]].append(ind)
        
    # Check for combined grains and stitch them back together
    for matched_ids in matched_grain_ids:
        if len(matched_ids) > 1:
            # Check for other stitchings
            for other_id in matched_ids[1:]:
                if len(matched_grain_ids[other_id]) > 1:
                    # Check and add other stitchings
                    for next_id in matched_grain_ids[other_id]:
                        if next_id not in matched_ids:
                            matched_ids.append(next_id)
                    # Reset values
                    matched_grain_ids[other_id] = [matched_grain_ids[other_id][0]]
            
            # Re-label actual grains
            for other_id in matched_ids[1:]:
                all_spots.loc[all_spots['grain_id'] == other_id, 'grain_id'] = matched_ids[0]

    # return 

    # Count all grain orientations and 
    all_ids = all_spots['grain_id'].values
    all_ids = all_ids[~np.isnan(all_ids)]
    unique_ids, counts = np.asarray(list(Counter(all_ids).items()), dtype=int).T
    sorted_ids = [x for _, x in sorted(zip(counts, unique_ids), key=lambda pair: pair[0], reverse=True)]

    # Reindex based on number of pixels (size)
    old_grain_ids = all_spots['grain_id'].values
    new_grain_ids = old_grain_ids.copy()
    
    for new_grain_id, grain_id in enumerate(sorted_ids):
        new_grain_ids[old_grain_ids == grain_id] = new_grain_id
    all_spots['grain_id'] = new_grain_ids


def new_reindex_grains(xdms, all_spots, mis_thresh=0.5, degrees=True, map_shape=None):

    # Setup useful values
    if map_shape is None:
        map_shape = xdms.xdms_vector_map.shape
    # all_spots = xdms.spots_3D

    # Collect orientation maps of new grain ids
    ori_map_list = []
    matched_grain_ids = []
    blank_ori_map = np.empty((*map_shape, 3, 3))
    blank_ori_map[:] = np.nan

    # Function to find neighbor pixels
    indices_vars = np.asarray([(-1, 0), (0, -1)])

    def get_near_indices(indices):
        new_indices = indices + indices_vars
        valid = ((new_indices[:, 0] >= 0)
                 & (new_indices[:, 0] < map_shape[0])
                 & (new_indices[:, 1] >= 0)
                 & (new_indices[:, 1] < map_shape[1]))

        return new_indices[valid]        

    for index in memory_iter(tqdm(range(np.prod(map_shape)))):
        indices = np.unravel_index(index, map_shape)
        near_indices = get_near_indices(indices)

        df = xdms.pixel_3D_spots(indices)
        grain_ids = df['grain_id'].values
        grain_ids = np.unique(grain_ids[~np.isnan(grain_ids)])

        for grain_id in grain_ids:
            grain_mask = df['grain_id'] == grain_id
            grain_inds = df[grain_mask].index.values
            grain_spots = df[grain_mask][['qx', 'qy', 'qz']]
            grain_hkls = df[grain_mask][['h', 'k', 'l']]

            grain_ori = Rotation.align_vectors(grain_spots,
                                               grain_hkls)[0]

            matched_oris = [False,] * len(ori_map_list)
            ori_i = -1
            for ori_i, ori in enumerate(ori_map_list):
                if ori is None:
                    continue

                for near in near_indices:
                    # If orientation has been determined
                    if not np.any(np.isnan(ori[tuple(near)])):
                        # Find misorientation
                        misori = grain_ori.as_matrix() @ ori[tuple(near)].T
                        mis_mag = Rotation.from_matrix(misori).magnitude()
                        if degrees:
                            mis_mag = np.degrees(mis_mag)
                        
                        # Check if within threshold
                        if mis_mag < mis_thresh:
                            ori_map_list[ori_i][indices] = grain_ori.as_matrix()
                            matched_oris[ori_i] = True
                            # Update grain ids
                            all_spots.loc[grain_inds, 'grain_id'] = ori_i
                            # print('Threshold achieved!!!')
                            continue
            
            # No matches, create new!
            if np.sum(matched_oris) == 0:
                ori_map_list.append(blank_ori_map.copy())
                ori_map_list[-1][indices] = grain_ori.as_matrix()
                matched_grain_ids.append([ori_i + 1])
                # Update grain_ids
                all_spots.loc[grain_inds, 'grain_id'] = ori_i + 1

            # Mark stiched grains to be combined later
            elif np.sum(matched_oris) > 1:
                matched_inds = np.nonzero(matched_oris)[0]
                new_ori_map = ori_map_list[matched_inds[0]].copy()
                for ind in matched_inds:
                    if ind not in matched_grain_ids[matched_inds[0]]:
                        matched_grain_ids[matched_inds[0]].append(ind)
                        matched_ori_map = ori_map_list[ind]
                        mask = ~np.all(np.isnan(matched_ori_map), axis=(2, 3))
                        ori_map_list[matched_inds[0]][mask] = matched_ori_map[mask]
                        ori_map_list[ind] = None
                        del matched_ori_map 

        
    # Check for combined grains and stitch them back together
    for matched_ids in matched_grain_ids:
        if len(matched_ids) > 1:
            # Check for other stitchings
            for other_id in matched_ids[1:]:
                if len(matched_grain_ids[other_id]) > 1:
                    # Check and add other stitchings
                    for next_id in matched_grain_ids[other_id]:
                        if next_id not in matched_ids:
                            matched_ids.append(next_id)
                    # Reset values
                    matched_grain_ids[other_id] = [matched_grain_ids[other_id][0]]
            
            # Re-label actual grains
            for other_id in matched_ids[1:]:
                all_spots.loc[all_spots['grain_id'] == other_id, 'grain_id'] = matched_ids[0]

    # return 

    # Count all grain orientations and 
    all_ids = all_spots['grain_id'].values
    all_ids = all_ids[~np.isnan(all_ids)]
    unique_ids, counts = np.asarray(list(Counter(all_ids).items()), dtype=int).T
    sorted_ids = [x for _, x in sorted(zip(counts, unique_ids), key=lambda pair: pair[0], reverse=True)]

    # Reindex based on number of pixels (size)
    old_grain_ids = all_spots['grain_id'].values
    new_grain_ids = old_grain_ids.copy()
    
    for new_grain_id, grain_id in enumerate(sorted_ids):
        new_grain_ids[old_grain_ids == grain_id] = new_grain_id
    all_spots['grain_id'] = new_grain_ids


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
    map_shape = (np.max(all_spots['map_y']),
                 np.max(all_spots['map_x']))

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
                        if degrees:
                            mag = np.degrees(mag)
                        if mag < mis_thresh:
                            dropped_inds.append(grain_id + mis_i + 1)
                
                # Drop overlapping orientations
                if len(dropped_inds) > 0:
                    mask = [True,] * len(pixel_oris)
                    print(f'WARNING: Overlapping grain dropped in pixel {(row, col)}.')
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
                print(f'WARNING: Overlapping grain found during stiching in pixel {(row, col)}.')

                # Eliminate the less good orientation from the pixel like before
                dropped_inds = []
                for o_ind in np.nonzero(overlap)[0]:
                    dropped_inds.extend(list(np.nonzero(matches[:, o_ind])[0][1:])) # Ignore first

                # Drop pixel orientations
                mask = [True,] * len(pixel_oris)
                for i, dropped_ind in enumerate(np.unique(dropped_inds)):
                    spot_inds = df[df['grain_id'] == dropped_ind].index
                    all_spots.loc[spot_inds, 'grain_id'] = np.min(df['grain_id'].values) - 1
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
                
                # Two matches. Combine previous grains
                elif len(match_keys) == 2:
                    keep_key = match_keys[0]
                    drop_key = match_keys[1]

                    # Update all_grains
                    all_grains[keep_key].extend(all_grains[drop_key])
                    all_grains[keep_key].extend(spot_inds)
                    # print(f'Dropping grain with index {drop_key}.')
                    # print(all_grains[drop_key])
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
                    
                    # Update current
                    curr_row_oris[col][keep_key] = ori
                
                # More matches. Something went wrong with overlapping orientation
                else: # Theoretically possible. Better way to handle?
                    warn_str = ('More matched orientations than reasonably expected...')
                    print(warn_str)
                    # raise RuntimeError
            
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


def build_grain_map(map_shape, all_spots, grain_id, color=None):
        
        if color is None:
            grain_map = np.empty(map_shape)
        else:
            grain_map = np.empty((*map_shape, *np.asarray(color).shape))
        grain_map[:] = np.nan
        
        for index in range(np.prod(map_shape)):
            indices = np.unravel_index(index, map_shape)
            ps = pixel_spots(all_spots, indices)
            if grain_id in ps['grain_id'].values:
                if color is None:
                    grain_map[indices] = True
                else:
                    grain_map[indices] = color
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
    # labels[:] = []
    label_spots = labels.copy()
    label_colors = labels.copy()

    max_inds = len(color_sequence)

    for index in range(np.prod(map_shape)):
        indices = np.unravel_index(index, map_shape)
        df = pixel_spots(spots_3D, indices)
        grain_ids = df['grain_id'].values
        grain_ids = np.unique(grain_ids[~np.isnan(grain_ids)])
        if len(grain_ids) > max_inds:
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
            if use_grains and grain_id < max_ind:
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
