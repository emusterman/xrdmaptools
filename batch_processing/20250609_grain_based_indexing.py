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

from matplotlib.animation import FuncAnimation

# Local imports
from xrdmaptools.utilities.utilities import (
  timed_iter,
  memory_iter,
  get_int_vector_map,
  get_num_vector_map,
  get_max_vector_map
)
from xrdmaptools.utilities.math import rescale_array
from xrdmaptools.XRDBaseScan import XRDBaseScan
from xrdmaptools.utilities.utilities import (
    generate_intensity_mask
)
# from xrdmaptools.reflections.spot_blob_indexing_3D import pair_casting_index_full_pattern
from xrdmaptools.reflections.spot_blob_search_3D import rsm_spot_search
from xrdmaptools.crystal.crystal import are_collinear, are_coplanar


def pixel_spots(spots, indices):
    ps = spots[((spots['map_x'] == indices[1]) & (spots['map_y'] == indices[0]))]
    return ps




def index_map_by_grains(xdms,
                        near_q,
                        near_angle,
                        degrees=False,
                        phase=None,
                        half_mask=True,
                        qof_cutoff=0.1,
                        verbose=True):

    # Create temporary dataframe to alter
    # This protects the original in case of errors
    spots_3D = xdms.spots_3D.copy()
    
    # Update spots dataframe with new columns
    spots_3D['phase'] = ''
    spots_3D[['grain_id', 'h', 'k', 'l', 'qof']] = np.nan

    if phase is None:
        if len(xdms.phases) == 1:
            phase = list(xdms.phases.values())[0]
        else:
            err_str = 'Phase must be provided for indexing.'
            raise ValueError(err_str)

    # Effective map shape
    map_shape = (np.max(spots_3D['map_y']) + 1,
                 np.max(spots_3D['map_x']) + 1)

    # TEMP: figure information for visualization
    plot_map = np.empty(map_shape)
    plot_map[:] = np.nan
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.ion()
    im = ax.imshow(plot_map, vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)
    plt.pause(1)
    
    # Vectorize spots
    print('Setting up indexing...')
    vector_map = xdms._spots_3D_to_vectors(spots_3D=spots_3D,
                                           map_shape=map_shape)
    sum_map = get_int_vector_map(vector_map)
    mutable_map = sum_map.astype(np.float64)
    # num_map = get_num_vector_map(vector_map)
    # mutable_map = num_map.astype(np.float64)

    # Setup reference information
    all_q_mags = np.linalg.norm(spots_3D[['qx', 'qy', 'qz']], axis=1)
    max_q = np.max(all_q_mags)
    min_q = phase.min_q

    phase.generate_reciprocal_lattice(1.15 * max_q)
    all_ref_qs = phase.all_qs.copy()
    all_ref_hkls = phase.all_hkls.copy()
    all_ref_fs = phase.all_fs.copy()
    all_ref_mags = np.linalg.norm(all_ref_qs, axis=1)

    # Ignore half...
    if half_mask:
        half_mask = all_ref_hkls[:, -1] <= 0
        all_ref_qs = all_ref_qs[half_mask]
        all_ref_hkls = all_ref_hkls[half_mask]
        all_ref_fs = all_ref_fs[half_mask]
    
    # Setup progress tracker
    pbar = tqdm(total=np.prod(map_shape), position=0, leave=True)

    # Decompose map in order of max total spot intensities
    while not np.all(np.isnan(mutable_map)):
        index = np.nanargmax(mutable_map)
        indices = np.unravel_index(index, map_shape)
        print(f'Indices are {indices}')

        # Setup containers
        all_indexing = []
        all_qofs = []

        # Setup current pixel information
        pixel_df = pixel_spots(spots_3D, indices)
        # Ignore orientationally indeterminant pixels
        if len(pixel_df) < 2:
            mutable_map[indices] = np.nan
            pbar.update(1)
            continue
        all_spot_qs = pixel_df[['qx', 'qy', 'qz']].values
        all_spot_ints = pixel_df['intensity'].values

        # Generate mask for reference values
        spot_q_mags = np.linalg.norm(all_spot_qs, axis=1)
        max_spot_q = np.max(spot_q_mags)
        min_spot_q = np.min(spot_q_mags)
        ext = 0.15
        ref_mask = ((min_spot_q * (1 - ext) < all_ref_mags)
                    & (all_ref_mags < max_spot_q * (1 + ext)))

        # Generate neighbor pixel indices
        poss_n_indices = np.asarray(indices) + [[1, 0], [0, 1], [-1, 0], [0, -1]]
        n_oris = []

        for n_indices in poss_n_indices:
            # if (np.any(n_indices < 0)
                # or np.any(map_shape - n_indices < 0)):
                # continue

            n_df = pixel_spots(spots_3D, n_indices)
            if len(n_df) > 1:
                grain_ids = n_df['grain_id'].values
                grain_ids = np.unique(grain_ids[~np.isnan(grain_ids)])

                for grain_id in grain_ids:
                    grain_mask = n_df['grain_id'] == grain_id
                    grain_spots = n_df[grain_mask][['qx', 'qy', 'qz']].values
                    grain_hkls = n_df[grain_mask][['h', 'k', 'l']].values
                    grain_ori = Rotation.align_vectors(grain_spots,
                                                    phase.Q(grain_hkls))[0]
                    n_oris.append(grain_ori)

        # Attempt indexing by neighbor orientations first
        spot_mask = np.ones(len(all_spot_qs), dtype=np.bool_)
        if len(n_oris) > 0:
            print('Grain casting')

            # if indices == (24, 118):
            #     return n_oris, all_spot_qs, all_spot_ints, all_ref_qs, all_ref_fs, xdms.qmask, near_q

            (grain_indexing,
             grain_qofs) = pattern_decomposition_from_seeds(
                                n_oris,
                                all_spot_qs,
                                all_spot_ints,
                                all_ref_qs,
                                all_ref_fs,
                                xdms.qmask,
                                near_q,
                                verbose=verbose
                            )
            all_indexing.extend(grain_indexing)
            all_qofs.extend(grain_qofs)
            
            # Which spots ares left?
            for indexing in grain_indexing:
                spot_mask[indexing[:, 0]] = False
        
        # Enough spots to continue?
        if spot_mask.sum() > 1:

            # Find pairs
            print('Finding pairs')
            pairs, connection_pairs = find_valid_pairs(
                                all_spot_qs[spot_mask],
                                all_ref_qs,
                                all_ref_hkls,
                                near_q,
                                near_angle,
                                phase.min_q,
                                degrees=degrees,
                                symmeterize=True,
                                verbose=verbose)
            
            # Enough pairs to continue?
            if len(pairs) > 0:

                # Index
                print('Full casting')

                # if indices == (26, 117):
                #     return pairs, all_spot_qs[spot_mask], all_spot_ints, all_ref_qs, all_ref_fs, xdms.qmask, near_q

                (rem_indexing,
                rem_qofs) = pattern_decomposition_from_seeds(
                                    pairs,
                                    all_spot_qs[spot_mask],
                                    all_spot_ints, # Leave full for comparable qof
                                    all_ref_qs,
                                    all_ref_fs,
                                    xdms.qmask,
                                    near_q,
                                    verbose=verbose)

                # if indices == (26, 117):
                #     return all_spot_qs, spot_mask, rem_indexing

                # Up-convert indexing references to full spot list
                full_spot_indices = np.array(range(len(all_spot_qs)))[spot_mask]
                for indexing in rem_indexing:
                    indexing[:, 0] = full_spot_indices[indexing[:, 0]]

                # Add new values
                all_indexing.extend(rem_indexing)
                all_qofs.extend(rem_qofs)

        # Record values
        for grain_id, (indexing, qof) in enumerate(zip(all_indexing, all_qofs)):
            
            # TEMP:
            if grain_id == 0:
                plot_map[indices] = qof
                im.set_data(plot_map)
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.0001)

            df_indices = pixel_df.index[indexing[:, 0]]

            spots_3D.loc[df_indices, 'grain_id'] = grain_id
            spots_3D.loc[df_indices, 'phase'] = phase.name
            spots_3D.loc[df_indices, 'qof'] = qof
            spots_3D.loc[df_indices, ['h', 'k', 'l']] = all_ref_hkls[indexing[:, 1]]        
        
        # Update map and progress
        mutable_map[indices] = np.nan
        pbar.update(1)
    
    # Stop progress
    pbar.close()

    return spots_3D