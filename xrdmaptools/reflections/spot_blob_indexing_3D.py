import numpy as np
from tqdm import tqdm
import scipy
from scipy.spatial.transform import Rotation
from scipy.spatial import distance_matrix, KDTree
from scipy.optimize import curve_fit
from itertools import combinations, product
from collections import Counter
import functools

from xrdmaptools.utilities.math import (
    multi_vector_angles,
    general_polynomial
)
from xrdmaptools.geometry.geometry import (
    q_2_polar,
    QMask,
)
from xrdmaptools.crystal.crystal import (
    are_collinear,
    are_coplanar
)

##########################
### Combined Functions ###
##########################


def index_best_grain(all_ref_qs,
                     all_ref_hkls,
                     all_ref_fs,
                     min_q,
                     all_spot_qs,
                     all_spot_ints,
                     near_q,
                     near_angle,
                     qmask,
                     degrees=False,
                     symmeterize=True,
                     exclude_found_seeds=False,
                     max_ori_refine_iter=50,
                     max_ori_decomp_count=20,
                     verbose=True):

    # Find all valid pairs within near_q and near_angle
    pairs = find_valid_pairs(
                all_spot_qs,
                all_ref_qs,
                all_ref_hkls,
                near_q,
                near_angle,
                min_q,
                degrees=degrees,
                symmeterize=symmeterize,
                verbose=verbose)

    if len(pairs) > 0:        
        # Index spots
        (indexings,
         qofs) = multiple_seed_casting(
                    pairs,
                    all_spot_qs,
                    all_spot_ints,
                    all_ref_qs,
                    all_ref_fs,
                    qmask,
                    near_q,
                    iter_max=max_ori_refine_iter,
                    exclude_found_seeds=exclude_found_seeds,
                    verbose=verbose)
    else:
        indexings = [np.asarray([[], []])]
        qofs = [np.nan]

    return indexings[0], qofs[0]


def index_all_grains(all_ref_qs,
                     all_ref_hkls,
                     all_ref_fs,
                     min_q,
                     all_spot_qs,
                     all_spot_ints,
                     near_q,
                     near_angle,
                     qmask,
                     degrees=False,
                     symmeterize=True,
                     qof_minimum=0.2,
                     max_ori_refine_iter=50,
                     max_ori_decomp_count=20,
                     verbose=True):

    # Find all valid pairs within near_q and near_angle
    pairs = find_valid_pairs(
                all_spot_qs,
                all_ref_qs,
                all_ref_hkls,
                near_q,
                near_angle,
                min_q,
                degrees=degrees,
                symmeterize=symmeterize,
                verbose=verbose)
        
    if len(pairs) > 0:
        # Iteratively decompose patterns
        (best_indexings, 
            best_qofs) = pattern_decomposition_from_seeds(
                pairs,
                all_spot_qs,
                all_spot_ints,
                all_ref_qs,
                all_ref_fs,
                qmask,
                near_q,
                qof_minimum=qof_minimum,
                max_ori_refine_iter=max_ori_refine_iter,
                max_ori_decomp_count=max_ori_decomp_count,
                verbose=verbose)

    else:
        best_indexings = [np.asarray([[], []])]
        best_qofs = [np.nan]

    return best_indexings, best_qofs


# Replaces reference reflection arguments with phase object instead
def phase_indexing_wrapper(function):
    @functools.wraps(function)
    def phase_wrapped(phase,
                      all_spot_qs,
                      *args,
                      half_mask=True,
                      **kwargs):

        spot_q_mags = np.linalg.norm(all_spot_qs, axis=1)
        max_q = np.max(spot_q_mags)

        phase.generate_reciprocal_lattice(1.15 * max_q)
        all_ref_qs = phase.all_qs.copy()
        all_ref_hkls = phase.all_hkls.copy()
        all_ref_fs = phase.all_fs.copy()

        # Only use hk-l values as those are closest to fundamental zone
        # Only Laue groups are indexable anyway - should be faster
        if half_mask:
            half_mask = all_ref_hkls[:, -1] <= 0
            all_ref_qs = all_ref_qs[half_mask]
            all_ref_hkls = all_ref_hkls[half_mask]
            all_ref_fs = all_ref_fs[half_mask]

        return function(all_ref_qs,
                        all_ref_hkls,
                        all_ref_fs,
                        phase.min_q,
                        all_spot_qs,
                        *args,
                        **kwargs)
    
    return phase_wrapped

phase_index_best_grain = phase_indexing_wrapper(index_best_grain)
phase_index_all_grains = phase_indexing_wrapper(index_all_grains)
        

def phase_based_index_best_pattern(phase, 
                                   all_spot_qs,
                                   all_spot_ints,
                                   near_q,
                                   near_angle,
                                   qmask,
                                   **kwargs):
    
    # Find q vector magnitudes and max for spots
    spot_q_mags = np.linalg.norm(all_spot_qs, axis=1)
    max_q = np.max(spot_q_mags)

    phase.generate_reciprocal_lattice(1.15 * max_q)
    all_ref_qs = phase.all_qs.copy()
    all_ref_hkls = phase.all_hkls.copy()
    all_ref_fs = phase.all_fs.copy()

    # Find minimum q vector step size from reference phase
    min_q = np.min(np.linalg.norm(phase.Q([[1, 0, 0],
                                           [0, 1, 0],
                                           [0, 0, 1]]),
                                           axis=0))

    return pair_casting_index_best_grain(all_ref_qs,
                                         all_ref_hkls,
                                         all_ref_fs,
                                         min_q,
                                         all_spot_qs,
                                         all_spot_ints,
                                         near_q,
                                         near_angle,
                                         qmask,
                                         **kwargs)


#####################
### Sub-Functions ###
#####################


def find_valid_pairs(all_spot_qs,
                     all_ref_qs,
                     all_ref_hkls,
                     near_q,
                     near_angle,
                     min_q,
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

    # Determine all angles
    spot_angles = multi_vector_angles(all_spot_qs, all_spot_qs, degrees=degrees)
    ref_angles = multi_vector_angles(all_ref_qs, all_ref_qs, degrees=degrees)
    min_angle = ref_angles[ref_angles > 0].min()

    # Generate all pairs of spots which are crystallographically feasible
    spot_pair_indices = list(combinations(range(len(all_spot_qs)), 2))
    spot_pair_dist = distance_matrix(all_spot_qs, all_spot_qs)
    # Check feasibility against reference phase spot distances and angles
    allowed_pairs = [spot_pair_dist[tuple(indices)] > min_q * 0.85
                     and spot_angles[tuple(indices)] > min_angle * 0.85
                     for indices in spot_pair_indices]
    spot_pair_indices = np.asarray(spot_pair_indices)[allowed_pairs]

    # Construct iterable
    if verbose:
        iterate = lambda x : tqdm(x, position=0, leave=True)
    else:
        iterate = lambda x : x

    if verbose:
        print('Finding all valid pairs...', flush=True)

    #blank_connection = np.array([np.nan,] * len(phase_mask))
    #connection_pairs = []
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
            #temp_connections = []
            temp_pairs, temp_mask = [], []
            pair_chars, pair_mags, pair_matches = [], [], []
            match_id = 0
            
            # Check all possible combintations from pair
            for combo in ref_combos:
                
                # Remove orientationally indeterminate combinations
                if are_collinear(all_ref_hkls[combo]):
                    continue

                # Add pairs
                temp_pairs.append([(phase_inds[s], r)
                                   for s, r in zip(pair, combo)])
                temp_mask.append(True)

                # Qualify potential fits and their orientation magnitude
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
            
            # Pick minimum orientation magnitude from reference within range
            if symmeterize:
                temp_mask = np.asarray([False,] * len(temp_mask))
                for idx in np.unique(pair_matches):
                    equi_mask = pair_matches == idx
                    min_angle = np.min(np.asarray(pair_mags)[equi_mask])
                    keep_mask = np.asarray(pair_mags)[equi_mask] < min_angle + near_angle
                    temp_mask[np.nonzero(equi_mask)[0][keep_mask]] = True
            
            pairs.extend(np.asarray(temp_pairs)[temp_mask])
            #connection_pairs.extend(np.asarray(temp_connections)[temp_mask])
    
    return pairs #, connection_pairs


def seed_casting(seed,
                 all_spot_qs,
                 all_spot_ints,
                 all_ref_qs,
                 all_ref_fs,
                 qmask,
                 near_q,
                 iter_max=50):

    # Perform input checks
    if isinstance(seed, (list, np.ndarray)):
        seed = np.asarray(seed)
        if seed.shape[1] != 2:
            err_str = ('Seed must have shape of (N, 2) not '
                       + f'{seed.shape}.')
            raise ValueError(err_str)
        if len(seed) < 2:
            # Seed is orientationally indeterminate
            return seed, 0 # Force return of input with 0 qof

        prev_indexing = seed
        orientation = Rotation.align_vectors(
                        all_spot_qs[np.asarray(seed)[:, 0]],
                        all_ref_qs[np.asarray(seed)[:, 1]])[0]

    elif isinstance(seed, Rotation):
        orientation = seed
        prev_indexing = []
    
    else:
        err_str = ('Only seeds of indexing as type list or '
                   + 'numpy.ndarray or seeds of a starting orientation'
                   + ' of type scipy.spatail.transform.Rotation are '
                   + f'supported. Type {type(seed)} is not.')
        raise TypeError(err_str)

    # Build spot kdtree
    kdtree = KDTree(all_spot_qs)

    # Start iterating
    iter_count = 0
    while True:
        # Rotate reference based on previous orientation
        all_rot_qs = orientation.apply(all_ref_qs, inverse=False)
        temp_qmask = qmask.generate(all_rot_qs)

        # Query kdtree for find closest reference lattice points
        pot_conn = kdtree.query_ball_point(all_rot_qs[temp_qmask],
                                           r=near_q)

        # Find closests points
        indexing = []
        for conn_i, conn in enumerate(pot_conn):
            if len(conn) > 0:
                if len(conn) == 0:
                    continue
                elif len(conn) == 1:
                    # Add candidate reflection
                    indexing.append([conn[0], np.nonzero(temp_qmask)[0][conn_i]])
                else:
                    # Add closest of multiple candidate reflections
                    _, spot_idx = kdtree.query(all_rot_qs[temp_qmask][conn_i])
                    indexing.append([spot_idx, np.nonzero(temp_qmask)[0][conn_i]])
        indexing = np.asarray(indexing)

        # Break if indexing becomes orientationally indeterminant
        if len(indexing) < 2:
            # Has the original seed failed?
            if iter_count == 0:
                if isinstance(seed, Rotation):
                    return [], 0
                else:
                    indexing = seed
                    curr_spots, curr_refs = indexing.T
                    # Re-determine orientation
                    orientation = Rotation.align_vectors(
                                    all_spot_qs[curr_spots],
                                    all_ref_qs[curr_refs])[0]
                    break
            # Otherwise prev_indexing should have worked
            else:
                indexing = prev_indexing
                curr_spots, curr_refs = indexing.T
                break
        else:
            curr_spots, curr_refs = indexing.T
            if are_collinear(all_ref_qs[curr_refs]):
                curr_spots, curr_refs = prev_indexing.T
                break
        
        # Parse indexing
        if len(prev_indexing) > 0:
            prev_spots, prev_refs = np.asarray(prev_indexing).T

            # Check to see if indexing has converged
            if len(indexing) == len(prev_indexing):
                if all([curr_spots[i] == prev_spots[i]
                        and curr_refs[i] == prev_refs[i]
                        for i in range(len(indexing))]):
                    break

        # Update for next round
        prev_indexing = indexing
        orientation = Rotation.align_vectors(
                            all_spot_qs[curr_spots],
                            all_ref_qs[curr_refs])[0]

        # Or break if max iterations reached
        iter_count += 1
        if iter_count >= iter_max:
            break
   
    # Find qof
    all_rot_qs = orientation.apply(all_ref_qs, inverse=False)
    temp_qmask = qmask.generate(all_rot_qs)
    # Forces bad seeds to be included in qmask
    temp_qmask[curr_refs] = True

    # qof = get_quality_of_fit(
    #     all_spot_qs[curr_spots], # fit_spot_qs
    #     all_spot_ints[curr_spots], # fit_spot_ints
    #     all_rot_qs[curr_refs], # fit_rot_qs
    #     all_ref_fs[curr_refs], # fit_ref_fs
    #     all_spot_ints, # all_spot_ints
    #     all_ref_fs[temp_qmask], # all_ref_fs
    #     sigma=near_q)

    filled_ints = np.zeros(sum(temp_qmask))
    for i, ind in enumerate(temp_qmask.nonzero()[0]):
        if ind in curr_refs:
            spot_ind = curr_spots[np.nonzero(curr_refs == ind)[0]][0]
            filled_ints[i] = all_spot_ints[spot_ind]


    qof = int_corr_qof(all_spot_qs[curr_spots],
                        all_rot_qs[curr_refs],
                        all_spot_ints[curr_spots],
                        all_spot_ints,
                        filled_ints,
                        all_ref_fs[temp_qmask],
                        sigma=near_q,
                        ratio=0.5)
    
    return indexing, qof


def multiple_seed_casting(seeds,
                          all_spot_qs,
                          all_spot_ints,
                          all_ref_qs,
                          all_ref_fs,
                          qmask,
                          near_q,
                          iter_max=50,
                          exclude_found_seeds=False,
                          sort_results=True,
                          verbose=True):

    # Modify and set up some values
    evaluated_seed_mask = np.array([False,] * len(seeds))
    all_spot_qs = np.asarray(all_spot_qs)
    all_ref_qs = np.asarray(all_ref_qs)
    all_ref_fs = np.asarray(all_ref_fs)

    # Construct iterator
    if verbose:
        iterate = lambda x : tqdm(enumerate(x),
                                  total=len(x),
                                  position=0,
                                  leave=True)
    else:
        iterate = lambda x : enumerate(x)

    indexings = []
    qofs = []
    if verbose:
        print('Casting valid seeds...')
    for i, seed in iterate(seeds):
        # Check if the pair has already been included
        if (exclude_found_seeds
            and evaluated_seed_mask[i]):
            continue

        indexing, qof = seed_casting(seed,
                                     all_spot_qs,
                                     all_spot_ints,
                                     all_ref_qs,
                                     all_ref_fs,
                                     qmask,
                                     near_q,
                                     iter_max=iter_max)

        indexings.append(indexing)
        qofs.append(qof)

        # Find and exclude included pairs
        if exclude_found_seeds:
            print('Checking future seeds...')
            evaluated_seed_mask[i] = True # Should be redundant
            if len(indexing) < 2:
                continue
            
            # Check all seeds to see if they were identified
            indexed = [tuple(c) for c in indexing]
            found_seed_mask = evaluated_seed_mask.copy()
            for ind, remaining_seed in enumerate(seeds[i:]):
                # Cannot perform this check with rotation seeds
                if isinstance(remaining_seed, Rotation):
                    continue
                seeded = [tuple(c) for c in remaining_seed]
                if all([s in indexed for s in seeded]):
                    found_seed_mask[ind] = True

            # Exclude from further evaluations
            evaluated_seed_mask[found_seed_mask] = True

    # Sort by qof
    if sort_results:
        indexings = [x for _, x in sorted(zip(qofs, indexings),
                                            key=lambda pair: pair[0],
                                            reverse=True)]
        qofs = sorted(qofs, reverse=True)
                
    return indexings, np.asarray(qofs)


###########################
### Iterative Functions ###
###########################

def pattern_decomposition_from_seeds(start_seeds,
                                     all_spot_qs,
                                     all_spot_ints,
                                     all_ref_qs,
                                     all_ref_fs,
                                     qmask,
                                     near_q,
                                     qof_minimum=0,
                                     max_ori_refine_iter=50,
                                     max_ori_decomp_count=20,
                                     verbose=True):

    # Setup containers and working values
    best_indexings, best_qofs = [], []
    excluded_spots = set()
    included_spot_mask = np.asarray([True,] * len(all_spot_qs))
    included_indexing_mask = np.asarray([True,] * len(start_seeds))

    ORIENTATION_SEEDS = isinstance(start_seeds[0], Rotation)

    # Internal wrapper for indexing method
    def _internal_indexing(seeds, spots, verbose=verbose):
        out = multiple_seed_casting(
            seeds,
            spots,
            all_spot_ints,
            all_ref_qs,
            all_ref_fs,
            qmask,
            near_q,
            sort_results=False,
            iter_max=max_ori_refine_iter,
            verbose=verbose
            )
        return out

    # First pass at indexing with scrubbed bad rotations
    indexings, qofs = _internal_indexing(start_seeds, all_spot_qs)

    # Mask out not useful indexing. May switch to qof_minimum
    mask = np.array([len(ind) > 1 for ind in indexings])
    indexings = [ind for ind, b in zip(indexings, mask) if b]
    qofs = np.asarray(qofs)[mask]
    included_indexing_mask = included_indexing_mask[mask]
    
    # All seeds happen to be excluded
    if len(indexings) < 1:
        return best_indexings, np.asarray(best_qofs)

    # Iteratively decompose pattern based on seeds
    iter_count = 0
    while True:
        # Find and record best results
        best_ind = np.nanargmax(qofs)
        best_indexings.append(indexings[best_ind].astype(int))
        best_qofs.append(qofs[best_ind])

        # Remove already indexed spots from further analysis
        excluded_spots.update(set(indexings[best_ind][:, 0]))
        included_spot_mask[np.array(list(excluded_spots))] = False
        valid_mask = np.asarray([True,] * len(indexings))
        recalc_mask = np.asarray([False,] * len(indexings))
        
        for i in range(len(indexings)):
            indexed_spots = set(indexings[i][:, 0])

            if ORIENTATION_SEEDS:
                # index results are invalid
                if len(indexed_spots - excluded_spots) < 2:
                    valid_mask[i] = False
                # Modified, but still indexable
                elif len(indexed_spots - excluded_spots) != len(indexed_spots):
                    recalc_mask[i] = True

            else:
                start_spots = set(start_seeds[np.nonzero(included_indexing_mask)[0][i]][:, 0])
                # Cannot reindex from start
                if len(start_spots - excluded_spots) < 2:
                    valid_mask[i] = False
                # Modified, but still indexable
                elif len(indexed_spots - excluded_spots) != len(indexed_spots):
                    recalc_mask[i] = True

        # Remove invalid connections
        included_indexing_mask[np.nonzero(included_indexing_mask)[0][~valid_mask]] = False
        indexings = [ind for ind, b in zip(indexings, valid_mask) if b]
        qofs = qofs[valid_mask]
        recalc_mask = recalc_mask[valid_mask]

        # Recalculate indexing results as necessary
        if recalc_mask.sum() > 0:
            spot_indices = list(np.array(range(len(all_spot_qs)))[included_spot_mask])
            idxs = np.nonzero(included_indexing_mask)[0][recalc_mask]
            
            if ORIENTATION_SEEDS:
                recalc_seeds = [start_seeds[idx] for idx in idxs] # from orientations
            else:
                recalc_seeds = []
                for i, seed in enumerate(start_seeds):
                    if i in idxs:
                        # Overwrite with new seed indices
                        seed[:, 0] = [spot_indices.index(s) for s in seed[:, 0]]
                        recalc_seeds.append(seed)
        
            # Re-index
            new_indexing, new_qofs = _internal_indexing(
                                recalc_seeds,
                                all_spot_qs[included_spot_mask],
                                verbose=False)

            # Why is this needed. Maybe temporary
            # new_mask = [len(i) != 0 for i in new_indexing]
            # new_indexing = [i for i, b in zip(new_indexing, new_mask) if b]
            # new_qofs = [i for i, b in zip(new_qofs, new_mask) if b]
            # if len(new_indexing) > 0:
            # print(f'{new_indexing=}')
            for idx, new_indexed, new_qof in zip(recalc_mask.nonzero()[0], new_indexing, new_qofs):
                # print(f'{new_indexed=}')
                if len(new_indexed) > 0:
                    new_indexed[:, 0] = included_spot_mask.nonzero()[0][new_indexed[:, 0]]
                    indexings[idx] = new_indexed
                    qofs[idx] = new_qof

            # Remove invalid seeds again
            if ORIENTATION_SEEDS:
                new_valid_mask = np.ones_like(recalc_mask, dtype=np.bool_)
                new_valid_mask[recalc_mask] = [len(ind) >= 2 for ind in new_indexing]
                if np.any(~new_valid_mask):
                    included_indexing_mask[np.nonzero(included_indexing_mask)[0][~new_valid_mask]] = False
                    indexings = [ind for ind, b in zip(indexings, new_valid_mask) if b]
                    qofs = qofs[new_valid_mask]
                    recalc_mask = recalc_mask[new_valid_mask]
        
        # Conditionals to kill iteration
        iter_count += 1
        if (len(indexings) < 1 # Nothing left to compare
            or len(all_spot_qs) - len(excluded_spots) < 1 # Cannot solve orientations
            or qofs.max() < qof_minimum # Quality of fit has gotten too poor
            or iter_count >= max_ori_decomp_count): # Reach maxed allowed orientations
            break
  
    return best_indexings, np.asarray(best_qofs)


#########################
### Utility Functions ###
#########################


def get_quality_of_fit(fit_spot_qs,
                       fit_spot_ints,
                       fit_rot_qs,
                       fit_ref_fs,
                       all_spot_ints,
                       all_ref_fs,
                       **kwargs):

    # Ideal conditions
    # 1. Penalize missing reference reflections weighted according to their expected intensity
    # 2. Do not penalize extra measured reflections which are not indexing (allows for overlapping orientations)
    # 3. Penalize reflections weighted by their distance from expected postions

    kwargs.setdefault('sigma', 0.1)
    sigma = kwargs.pop('sigma')

    # qof = get_rmse(fit_spot_qs,
    #                fit_rot_qs)

    # qof = weighted_distance_qof(fit_spot_qs,
    #                             fit_rot_qs,
    #                             fit_ref_fs,
    #                             all_ref_fs,
    #                             int_weight=0.5)

    # qof = complete_distance_qof(fit_spot_qs,
    #                             fit_rot_qs,
    #                             all_rot_fs,
    #                             sigma=1,
    #                             int_weight=0.1)

    # qof = explained_intensity_qof(fit_spot_ints,
    #                               fit_ref_fs,
    #                               all_spot_ints,
    #                               all_ref_fs,
    #                               ratio=0.75)
    
    qof = dist_int_qof(fit_spot_qs,
                       fit_rot_qs,
                       fit_spot_ints,
                       all_spot_ints,
                       fit_ref_fs,
                       all_ref_fs,
                       sigma=sigma,
                       ratio=0.5)

    # qof = len(fit_spot_qs) / len(all_spot_ints)

    return qof


def weighted_distance_qof(fit_spot_qs,
                          fit_rot_qs,
                          fit_ref_fs,
                          all_ref_fs,
                          sigma=1,
                          int_weight=0.5):

    # # Determine which reflections are indexed
    # found_spot_mask = [tuple(ref) in [tuple(x) for x in ref_qs] for ref in all_ref_qs]

    dist = [np.sqrt(np.sum([(p - q)**2 for p, q in zip(v1, v2)]))
            for v1, v2 in zip(fit_spot_qs, fit_rot_qs)]

    # Gaussian with specified standard deviation
    # centered at zero sampled at distance
    dist_val = np.sum(np.exp(-(np.asarray(dist))**2 / (2 * sigma**2)))
    norm_dist_val = dist_val / len(all_ref_fs)

    int_val = (np.sum(fit_ref_fs)
              / np.sum(all_ref_fs))

    qof = (int_weight * int_val) + ((1 - int_weight) * norm_dist_val)

    # If not all inputs are given, the whole thing can fail
    # Return an operable value
    if np.isnan(qof):
        qof = 0

    return qof


def complete_distance_qof(fit_spot_qs,
                          fit_rot_qs,
                          all_rot_fs,
                          sigma=1,
                          int_weight=0.5): # Gaussian standard deviation to evaluate


    # Determine which reflections are indexed
    # found_spot_mask = [tuple(ref) in [tuple(x) for x in ref_qs] for ref in all_ref_qs]

    dist = [np.sqrt(np.sum([(p - q)**2 for p, q in zip(v1, v2)]))
            for v1, v2 in zip(fit_spot_qs, fit_rot_qs)]

    # Gaussian with specified standard deviation
    # centered at zero sampled at distance
    qof = np.sum(np.exp(-(np.asarray(dist))**2 / (2 * sigma**2)))

    max_qof = len(all_rot_fs)
    norm_qof = qof / max_qof

    return norm_qof


def explained_intensity_qof(fit_spot_ints,
                            fit_ref_fs,
                            all_spot_ints,
                            all_ref_fs,
                            ratio=0.5,
                            ):

    exp_val = np.sum(fit_spot_ints) / np.sum(all_spot_ints)
    ref_val = np.sum(fit_ref_fs) / np.sum(all_ref_fs)

    # print(f'{exp_val=}')
    # print(f'{ref_val=}')

    return (ratio * exp_val) + ((1 - ratio) * ref_val)



def dist_int_qof(fit_spot_qs,
                 fit_rot_qs,
                 fit_spot_ints,
                 all_spot_ints,
                 fit_ref_fs,
                 all_ref_fs,
                 sigma=1,
                 ratio=0.5):


    dist = [np.sqrt(np.sum([(p - q)**2 for p, q in zip(v1, v2)]))
            for v1, v2 in zip(fit_spot_qs, fit_rot_qs)]
    
    # Gaussian with specified standard deviation
    # centered at zero sampled at distance
    gauss_int = fit_spot_ints * np.exp(-(np.asarray(dist))**2 / (2 * sigma**2))

    exp_val = np.sum(gauss_int) / np.sum(all_spot_ints)
    ref_val = np.sum(fit_ref_fs) / np.sum(all_ref_fs)

    return (ratio * exp_val) + ((1 - ratio) * ref_val)


def int_corr_qof(fit_spot_qs,
                 fit_rot_qs,
                 fit_spot_ints,
                 all_spot_ints,
                 filled_ints,
                 all_ref_fs,
                 sigma=1,
                 ratio=0.5):


    dist = [np.sqrt(np.sum([(p - q)**2 for p, q in zip(v1, v2)]))
            for v1, v2 in zip(fit_spot_qs, fit_rot_qs)]
    
    # Gaussian with specified standard deviation
    # centered at zero sampled at distance
    gauss_int = fit_spot_ints * np.exp(-(np.asarray(dist))**2 / (2 * sigma**2))

    exp_val = np.sum(gauss_int) / np.sum(all_spot_ints)

    ref_val = np.dot(filled_ints, all_ref_fs) / (np.linalg.norm(filled_ints) * np.linalg.norm(all_ref_fs))

    # return (ratio * exp_val) + ((1 - ratio) * ref_val)
    return exp_val * ref_val # Both must be large


def get_rmse(fit_spot_qs,
             fit_rot_qs):
    rmse = np.mean([np.sqrt(np.sum([(p - q)**2 for p, q in zip(v1, v2)]))
                    for v1, v2 in zip(fit_spot_qs, fit_rot_qs)])

    return rmse
