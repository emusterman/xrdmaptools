import numpy as np
from tqdm import tqdm
import scipy
from scipy.spatial.transform import Rotation
from scipy.spatial import distance_matrix, KDTree
from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit
from itertools import combinations, product
from collections import Counter
import functools
import dask
from tqdm.dask import TqdmCallback

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
from xrdmaptools.reflections.qualify_indexing import (
    reduce_indexings,
    qof_from_indexing,
    qofs_from_indexings,
    sort_by_qofs,
    desymmetrize,
    multi_desymmetrize
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
                     phase=None,
                     symmetrize='lattice',
                     space_group_nr=230,
                     exclude_found_seeds=False,
                     max_ori_refine_iter=50,
                     max_ori_decomp_count=20,
                     verbose=True):

    # Find all valid pairs within near_q and near_angle
    pairs = find_valid_pairs(
                all_spot_qs,
                all_ref_qs,
                all_ref_hkls,
                all_ref_fs,
                near_q,
                near_angle,
                min_q,
                degrees=degrees,
                symmetrize=symmetrize,
                verbose=verbose)

    if len(pairs) > 0:        
        # Index spots
        indexings = multiple_seed_casting(
                    pairs,
                    all_spot_qs,
                    all_spot_ints,
                    all_ref_qs,
                    all_ref_fs,
                    near_q,
                    iter_max=max_ori_refine_iter,
                    exclude_found_seeds=exclude_found_seeds,
                    verbose=verbose)
        
        # Eliminate equivalent indexings
        indexings = reduce_indexings(indexings)
        
        # Determine quality of fit
        qofs = qofs_from_indexings(indexings,
                                   all_ref_qs,
                                   all_ref_fs,
                                   all_spot_qs,
                                   all_spot_ints,
                                   near_q,
                                   qmask)
        
        # Sort results
        qofs, indexings = sort_by_qofs(qofs, indexings)
        
        # De-symmetrize
        if symmetrize is not None:
            if len(indexings) > 1:
                # De-symmetrize best indexings.
                # May include pseudosymmetry      
                qof_mask = qofs >= np.max(qofs) - 0.15 * np.max(qofs)
                if qof_mask.sum() < 6:
                    if len(qofs) >= 6:
                        qof_mask = qofs >= sorted(np.unique(qofs))[-6]
                    else:
                        qof_mask = np.ones_like(qofs, dtype=np.bool_)
            else:
                qof_mask = [True]

            indexing, qof = multi_desymmetrize(
                                    # indexings[qof_mask],
                                    [indexing for indexing, b in zip(indexings, qof_mask) if b],
                                    all_ref_qs,
                                    all_ref_fs,
                                    all_spot_qs,
                                    all_spot_ints,
                                    near_q,
                                    qmask,
                                    phase=phase,
                                    all_ref_hkls=all_ref_hkls,
                                    space_group_nr=space_group_nr,
                                    symmetrize=symmetrize)
            
        else:
            # Pick best
            indexing = indexings[0]
            qof = qofs[0]
                    
    else:
        indexing = np.asarray([[], []])
        qof = np.nan

    return indexing, qof


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
                     phase=None,
                     symmetrize='lattice',
                     space_group_nr=230,
                     qof_minimum=0,
                     max_ori_refine_iter=50,
                     max_ori_decomp_count=20,
                     reduce_seeds=True,
                     verbose=True):

    # Find all valid pairs within near_q and near_angle
    pairs = find_valid_pairs(
                all_spot_qs,
                all_ref_qs,
                all_ref_hkls,
                all_ref_fs,
                near_q,
                near_angle,
                min_q,
                degrees=degrees,
                symmetrize=symmetrize,
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
                phase=phase,
                all_ref_hkls=all_ref_hkls,
                symmetrize=symmetrize,
                space_group_nr=space_group_nr,
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
                      space_group_nr=None,
                      verbose=False,
                      **kwargs):

        spot_q_mags = np.linalg.norm(all_spot_qs, axis=1)
        max_q = np.max(spot_q_mags)
        if not phase._has_reciprocal_lattice:    
            phase.generate_reciprocal_lattice(1.15 * max_q)
        elif verbose:
            if phase._reciprocal_lattice_qmax < 1.15 * max_q:
                warn_str = ('WARNING: Phase already has a reciprocal '
                            + 'lattice, but it was generated with a '
                            + 'maximum q-vector lower than expected. '
                            + 'Some reflections may be missing for '
                            + 'indexing.')
                print(warn_str)

        all_ref_qs = phase.all_qs.copy()
        all_ref_hkls = phase.all_hkls.copy()
        all_ref_fs = phase.all_fs.copy()

        if space_group_nr is None:
            space_group_nr = phase.lattice.space_group_nr

        if verbose and space_group_nr != phase.lattice.space_group_nr:
            warn_str = ('WARNING: Given space group number of '
                        + f'{space_group_nr} does not match phase '
                        + 'space group_number of '
                        + f'{phase.lattice.space_group_nr}.'
                        + f'\nUsing given number of {space_group_nr}')
            print(warn_str)

        return function(all_ref_qs,
                        all_ref_hkls,
                        all_ref_fs,
                        phase.min_q,
                        all_spot_qs,
                        *args,
                        phase=phase,
                        space_group_nr=space_group_nr,
                        verbose=verbose,
                        **kwargs)
    
    return phase_wrapped

phase_index_best_grain = phase_indexing_wrapper(index_best_grain)
phase_index_all_grains = phase_indexing_wrapper(index_all_grains)


#####################
### Sub-Functions ###
#####################

def kabsch_orientation_matrix(S, T):
    # Compute cross-covariance matrix
    H = np.dot(S.T, T)

    # Singular Value Decomposition
    U, S_vals, Vt = np.linalg.svd(H)

    # Calculator Rotation Matrix
    R = np.dot(Vt.T, U.T)

    # Reflection correction for right-handed coordinates
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    
    return R


def triad_orientation_matrix(s1, s2, t1, t2):

    # Standardize dimensions: coerces (3,) to (1, 3) without changing (N, 3)
    s1, s2 = np.atleast_2d(s1), np.atleast_2d(s2)
    t1, t2 = np.atleast_2d(t1), np.atleast_2d(t2)
    
    # Vectorized Source Basis
    u1 = s1 / np.linalg.norm(s1, axis=1, keepdims=True)
    cross_s = np.cross(s1, s2)
    u2 = cross_s / np.linalg.norm(cross_s, axis=1, keepdims=True)
    u3 = np.cross(u1, u2)
    M_source = np.stack((u1, u2, u3), axis=1) # Shape: (N, 3, 3)
    
    # Vectorized Target Basis
    v1 = t1 / np.linalg.norm(t1, axis=1, keepdims=True)
    cross_t = np.cross(t1, t2)
    v2 = cross_t / np.linalg.norm(cross_t, axis=1, keepdims=True)
    v3 = np.cross(v1, v2)
    M_target = np.stack((v1, v2, v3), axis=1) # Shape: (N, 3, 3)

    # Batch Transpose and Multiply: (N, 3, 3) @ (N, 3, 3)
    R_matrices = np.matmul(M_target, np.swapaxes(M_source, -1, -2))
    
    # If the original input was a single pair, squeeze out the extra dim
    if R_matrices.shape[0] == 1:
        return R_matrices[0]
        
    return R_matrices



def find_valid_pairs(all_spot_qs,
                     all_ref_qs,
                     all_ref_hkls,
                     all_ref_fs,
                     near_q,
                     near_angle,
                     min_q,
                     degrees=False,
                     symmetrize=None,
                     verbose=True):

    # Padding value for masking
    min_ext = 0.85

    # Guard block(s)
    if near_q > min_q * min_ext:
        err_str = (f"'near_q' threshold is greater than {min_ext * 100}% "
                   "of minimum lattice spacing. This seems unwise.")
        raise ValueError(err_str)

    # Tracking references to original lists
    num_spots = len(all_spot_qs)
    num_refs = len(all_ref_qs)

    if verbose:
        vprint = lambda *a, **k : print(*a, **k)
    else:
        vprint = lambda *a, **k : None

    vprint('Finding all valid pairs...', flush=True)

    # Increasingly aggressive culling steps to reduce data size
    
    # 1. Compare direct magnitudes and keep only feasible 
    # Find vector magnitudes for measured and reference reflections
    spot_q_mags = np.linalg.norm(all_spot_qs, axis=1)
    ref_q_mags = np.linalg.norm(all_ref_qs, axis=1)

    # Find difference between measured and calculated q magnitudes
    mag_diff_matrix = np.abs(spot_q_mags[:, np.newaxis]
                             - ref_q_mags[np.newaxis, :])
    valid_mag_mask = mag_diff_matrix < near_q
    spot_map, ref_map = np.nonzero(valid_mag_mask)

    # Indexing values for pairwise comparisons
    num_map = len(spot_map)
    idx1, idx2 = np.triu_indices(num_map, k=1)
    s1 = spot_map[idx1]
    r1 = ref_map[idx1]
    s2 = spot_map[idx2]
    r2 = ref_map[idx2]
    vprint(f'Full Set:\t\tNo. Valid Pairs = {len(s1)}')

    def get_unq(l1, l2, num):
        pair_keys = l1 * num + l2
        unq_keys, inverse = np.unique(pair_keys, return_inverse=True)
        unq_l1 = unq_keys // num
        unq_l2 = unq_keys % num
        return unq_l1, unq_l2, inverse, unq_keys

    # 2. Immediately remove all doublets (cheapest)
    mask = ((s1 != s2) | (r1 != r2))
    s1, r1, s2, r2 = s1[mask], r1[mask], s2[mask], r2[mask]
    vprint(f'Doublet:\t\tNo. Valid Pairs = {len(s1)}')

    # 3. Only consider feasible inter-spot distances
    # Downcast to only spot_pairs
    unq_s1, unq_s2, spot_inverse, spot_keys = get_unq(s1, s2, num_spots)
    unq_spot_dist = np.linalg.norm(all_spot_qs[unq_s1] - all_spot_qs[unq_s2], axis=1)
    unq_spot_dist_mask = (unq_spot_dist > min_q * min_ext)
    mask = unq_spot_dist_mask[spot_inverse]
    s1, r1, s2, r2 = s1[mask], r1[mask], s2[mask], r2[mask]
    vprint(f'Dist Min:\t\tNo. Valid Pairs = {len(s1)}')

    # 4. Only consider non-collinear pairs
    # Downcast refs pairs
    unq_r1, unq_r2, ref_inverse, ref_keys = get_unq(r1, r2, num_refs)
    unq_cross_norms = np.linalg.norm(np.cross(all_ref_hkls[unq_r1], all_ref_hkls[unq_r2]), axis=1)
    unq_cross_norms_mask = unq_cross_norms > 1e-5
    mask = unq_cross_norms_mask[ref_inverse]
    s1, r1, s2, r2 = s1[mask], r1[mask], s2[mask], r2[mask]
    vprint(f'Collinearity:\t\tNo. Valid Pairs = {len(s1)}')

    # 5. Match inter-distances
    # Downcast both datasets again
    unq_s1, unq_s2, spot_inverse, spot_keys = get_unq(s1, s2, num_spots)
    unq_r1, unq_r2, ref_inverse, ref_keys = get_unq(r1, r2, num_refs)
    multiplier = len(unq_r1)
    flat_keys = spot_inverse * multiplier + ref_inverse
    unq_keys, combo_inverse = np.unique(flat_keys, return_inverse=True)

    mapped_spot_idx = unq_keys // multiplier
    mapped_ref_idx = unq_keys % multiplier

    unq_spot_dist = np.linalg.norm(all_spot_qs[unq_s1] - all_spot_qs[unq_s2], axis=1)
    unq_ref_dist = np.linalg.norm(all_ref_qs[unq_r1] - all_ref_qs[unq_r2], axis=1)

    inter_dist_mask = np.abs(unq_spot_dist[mapped_spot_idx] - unq_ref_dist[mapped_ref_idx]) < near_q
    mask = inter_dist_mask[combo_inverse]
    s1, r1, s2, r2 = s1[mask], r1[mask], s2[mask], r2[mask]
    vprint(f'Pair Dist:\t\tNo. Valid Pairs = {len(s1)}')

    # 6. Match inter-angles (slowest)
    # Downcast both datasets again
    unq_s1, unq_s2, spot_inverse, spot_keys = get_unq(s1, s2, num_spots)
    unq_r1, unq_r2, ref_inverse, ref_keys = get_unq(r1, r2, num_refs)
    multiplier = len(unq_r1)
    flat_keys = spot_inverse * multiplier + ref_inverse
    unq_keys, combo_inverse = np.unique(flat_keys, return_inverse=True)

    mapped_spot_idx = unq_keys // multiplier
    mapped_ref_idx = unq_keys % multiplier

    cos_spot = np.einsum('ij, ij->i', all_spot_qs[unq_s1], all_spot_qs[unq_s2]) / (spot_q_mags[unq_s1] * spot_q_mags[unq_s2])
    unq_spot_ang = np.arccos(np.clip(cos_spot, -1.0, 1.0))

    cos_ref = np.einsum('ij, ij->i', all_ref_qs[unq_r1], all_ref_qs[unq_r2]) / (ref_q_mags[unq_r1] * ref_q_mags[unq_r2])
    unq_ref_ang = np.arccos(np.clip(cos_ref, -1.0, 1.0))

    if degrees:
        rad_near_ang = np.radians(near_angle)
    else:
        rad_near_ang = near_angle

    inter_ang_mask = np.abs(unq_spot_ang[mapped_spot_idx] - unq_ref_ang[mapped_ref_idx]) < rad_near_ang
    mask = inter_ang_mask[combo_inverse] # Keep angles for symmetrizing
    ref_angles = unq_ref_ang[ref_inverse]
    ref_angles = ref_angles[mask]
    s1, r1, s2, r2 = s1[mask], r1[mask], s2[mask], r2[mask]
    vprint(f'Pair Angle:\t\tNo. Valid Pairs = {len(s1)}')

    if symmetrize:
        prec = 8

        # Precompute approximate orientation matrices for each pair
        oris = triad_orientation_matrix(all_spot_qs[s1],
                                        all_spot_qs[s2],
                                        all_ref_qs[r1],
                                        all_ref_qs[r2])
        traces = np.einsum('nii->n', oris)

        # Vectorize rouding characteristics first
        q1_round = ref_q_mags[r1].round(prec)
        q2_round = ref_q_mags[r2].round(prec)
        ang_round = ref_angles.round(prec)
        
        # Get pair characteristics
        if symmetrize == 'lattice':
            sort_idx = np.lexsort((-traces,
                                   ang_round,
                                   q1_round,
                                   q2_round,))
            stacked_keys = np.column_stack((q1_round[sort_idx],
                                            q2_round[sort_idx],
                                            ang_round[sort_idx]))
                                   
        elif symmetrize == 'point_group':
            f1_round = all_ref_fs[r1].round(prec)
            f2_round = all_ref_fs[r2].round(prec)
            sort_idx = np.lexsort((-traces,
                                   ang_round,
                                   q1_round,
                                   f1_round,
                                   q2_round,
                                   f2_round,))
            stacked_keys = np.column_stack((q1_round[sort_idx],
                                            f1_round[sort_idx],
                                            q2_round[sort_idx],
                                            f2_round[sort_idx],
                                            ang_round[sort_idx]))
        else:
            raise ValueError(f"'symmetrize' must be 'point_group' or 'lattice', not {symmetrize}.")
        
        # Sort groups contiguously
        # stacked_keys_sorted = stacked_keys[sort_idx]
        s1_sort, r1_sort, s2_sort, r2_sort = s1[sort_idx], r1[sort_idx], s2[sort_idx], r2[sort_idx]
        traces_sort = traces[sort_idx]
        
        # Allow for experimental error within the near_angle parameter
        trace_tolerance = 3 - (1 + 2 * np.cos(rad_near_ang))

        # Find the start of sorted groups
        group_boundaries = np.any(stacked_keys[1:] != stacked_keys[:-1], axis=1)
        group_starts = np.flatnonzero(np.concatenate(([True], group_boundaries)))

        # Generate an array where each group only include the max
        idx = np.zeros(len(traces_sort), dtype=int)
        idx[group_starts] = group_starts
        group_mapping = np.maximum.accumulate(idx)
        max_traces = traces_sort[group_mapping]

        # Extract the filtered data matching our tolerance condition
        tolerance_mask = traces_sort >= (max_traces - trace_tolerance)
        final_s1 = s1_sort[tolerance_mask]
        final_r1 = r1_sort[tolerance_mask]
        final_s2 = s2_sort[tolerance_mask]
        final_r2 = r2_sort[tolerance_mask]
        
        # Build pairs list
        pairs = [np.asarray([[ps1, pr1], [ps2, pr2]]) for ps1, pr1, ps2, pr2 in zip(final_s1, final_r1, final_s2, final_r2)]
        
    else:
        # Build pairs list
        pairs = [np.asarray([[ps1, pr1], [ps2, pr2]]) for ps1, pr1, ps2, pr2 in zip(s1, r1, s2, r2)]

    vprint(f'Symmetry:\t\tNo. Valid Pairs = {len(pairs)}')

    return pairs


# def old_find_valid_pairs(all_spot_qs,
#                      all_ref_qs,
#                      all_ref_hkls,
#                      all_ref_fs,
#                      near_q,
#                      near_angle,
#                      min_q,
#                      degrees=False,
#                      symmetrize=None,
#                      verbose=True):

#     if near_q > min_q * 0.85:
#         err_str = ("'near_q' threshold is greater than 85% of minimum "
#                    + "lattice spacing. This seems unwise.")
#         raise ValueError(err_str)    
    
#     # Find vector magnitudes for measured and reference reflections
#     spot_q_mags = np.linalg.norm(all_spot_qs, axis=1)
#     ref_q_mags = np.linalg.norm(all_ref_qs, axis=1)

#     # Find difference between measured and calculated q magnitudes
#     mag_diff_arr = np.abs(spot_q_mags[:, np.newaxis]
#                           - ref_q_mags[np.newaxis, :])
    
#     # Eliminate any reflections outside of phase-allowed spots
#     phase_mask = np.any(mag_diff_arr < near_q, axis=1)
#     phase_inds = np.nonzero(phase_mask)[0]
#     mag_diff_arr = mag_diff_arr[phase_mask]
#     all_spot_qs = all_spot_qs[phase_mask]
#     spot_q_mags = spot_q_mags[phase_mask]

#     # Determine all angles
#     spot_angles = multi_vector_angles(all_spot_qs, all_spot_qs, degrees=degrees)
#     ref_angles = multi_vector_angles(all_ref_qs, all_ref_qs, degrees=degrees)
#     min_angle = ref_angles[ref_angles > 0].min()

#     # Generate all pairs of spots which are crystallographically feasible
#     spot_pair_indices = list(combinations(range(len(all_spot_qs)), 2))
#     spot_pair_dist = distance_matrix(all_spot_qs, all_spot_qs)

#     # Check feasibility against reference phase spot distances and angles
#     allowed_pairs = [spot_pair_dist[tuple(indices)] > min_q * 0.85
#                      and spot_angles[tuple(indices)] > min_angle * 0.85
#                      for indices in spot_pair_indices]
#     spot_pair_indices = np.asarray(spot_pair_indices)[allowed_pairs]

#     # Construct iterable
#     if verbose:
#         iterate = lambda x : tqdm(x, position=0, leave=True)
#     else:
#         iterate = lambda x : x

#     if verbose:
#         print('Finding all valid pairs...', flush=True)

#     pairs = []    
#     for pair in iterate(spot_pair_indices):
#         possible_refs = [np.nonzero(mag_diff_arr[i] < near_q)[0] for i in pair]

#         if any(len(refs) == 0 for refs in possible_refs):
#             print('First check!')
#             continue

#         pair_chars = {}
#         for combo in product(*possible_refs):
#             if (combo[0] == combo[1] # No doublets
#                 or np.abs(spot_angles[tuple(pair)] - ref_angles[tuple(combo)]) > near_angle # Angle check
#                 or ref_angles[tuple(combo)] == 0): # No collinear combos)
#                 continue
                
#             # Second conlinearity check on HKLs
#             if are_collinear(all_ref_hkls[np.array(combo)]):
#                 continue

#             temp_pair = np.asarray([[phase_inds[s], r] for s, r in zip(pair, combo)])

#             # Qualify potential fits and their orientation magnitude
#             if symmetrize:
#                 # Characterize combo
#                 prec = 8
#                 if symmetrize == 'lattice':
#                     combo_char = (ref_q_mags[combo[0]].round(prec), # spot1 q-magnitude
#                                   ref_q_mags[combo[1]].round(prec), # spot2 q-magnitude
#                                   ref_angles[tuple(combo)].round(prec) # interplanar angle
#                                   )
#                 elif symmetrize == 'point_group':
#                     combo_char = (ref_q_mags[combo[0]].round(prec), # spot1 q-magnitude
#                                   all_ref_fs[combo[0]].round(prec), # spot1 structure factor
#                                   ref_q_mags[combo[1]].round(prec), # spot2 q-magnitude
#                                   all_ref_fs[combo[1]].round(prec), # spot2 structure factor
#                                   ref_angles[tuple(combo)].round(prec) # interplanar angle
#                                   )
#                 else:
#                     raise ValueError(f"'symmetrize' must be 'point_group' or 'lattice', not {symmetrize}.")

#                 # Add pair if it has unique characteristics
#                 if combo_char not in pair_chars:
#                     # pair_chars.append(combo_char)
#                     pair_chars[combo_char] = None
#                     pairs.append(temp_pair)
#                     continue
            
#             else:
#                 # Include everything, even symmetrically equivalent duplicates
#                 pairs.append(temp_pair)
    
#     return pairs


# # Unused now. 
# def masked_seed_casting(seed,
#                  all_spot_qs,
#                  all_spot_ints,
#                  all_ref_qs,
#                  all_ref_fs,
#                  qmask,
#                  near_q,
#                  iter_max=50):

#     # Perform input checks
#     if isinstance(seed, (list, np.ndarray)):
#         seed = np.asarray(seed)
#         if seed.shape[1] != 2:
#             err_str = ('Seed must have shape of (N, 2) not '
#                        + f'{seed.shape}.')
#             raise ValueError(err_str)
#         if len(seed) < 2:
#             # Seed is orientationally indeterminate
#             return seed # Force return of input with 0 qof

#         prev_indexing = seed
#         orientation = Rotation.align_vectors(
#                         all_spot_qs[np.asarray(seed)[:, 0]],
#                         all_ref_qs[np.asarray(seed)[:, 1]])[0]

#     elif isinstance(seed, Rotation):
#         orientation = seed
#         prev_indexing = []
    
#     else:
#         err_str = ('Only seeds of indexing as type list or '
#                    + 'numpy.ndarray or seeds of a starting orientation'
#                    + ' of type scipy.spatail.transform.Rotation are '
#                    + f'supported. Type {type(seed)} is not.')
#         raise TypeError(err_str)

#     # Build spot kdtree
#     kdtree = KDTree(all_spot_qs)

#     # Start iterating
#     iter_count = 0
#     while True:
#         # Rotate reference based on previous orientation
#         all_rot_qs = orientation.apply(all_ref_qs, inverse=False)
#         temp_qmask = qmask.generate(all_rot_qs)

#         # Query kdtree to find closest reference lattice points
#         pot_conn = kdtree.query_ball_point(all_rot_qs[temp_qmask],
#                                            r=near_q)

#         # Find closests points
#         indexing = []
#         for conn_i, conn in enumerate(pot_conn):
#             if len(conn) == 0:
#                 continue
#             elif len(conn) == 1:
#                 # Add candidate reflection
#                 indexing.append([conn[0], np.nonzero(temp_qmask)[0][conn_i]])
#             else:
#                 # Add closest of multiple candidate reflections
#                 _, spot_idx = kdtree.query(all_rot_qs[temp_qmask][conn_i])
#                 indexing.append([spot_idx, np.nonzero(temp_qmask)[0][conn_i]])
#         indexing = np.asarray(indexing)

#         # Break if indexing becomes orientationally indeterminant
#         if len(indexing) < 2:
#             # Has the original seed failed?
#             if iter_count == 0:
#                 if isinstance(seed, Rotation):
#                     return []
#                 else:
#                     indexing = seed
#                     curr_spots, curr_refs = indexing.T
#                     # Re-determine orientation
#                     orientation = Rotation.align_vectors(
#                                     all_spot_qs[curr_spots],
#                                     all_ref_qs[curr_refs])[0]
#                     break
#             # Otherwise prev_indexing should have worked
#             else:
#                 indexing = prev_indexing
#                 curr_spots, curr_refs = indexing.T
#                 break
#         else:
#             curr_spots, curr_refs = indexing.T
#             if are_collinear(all_ref_qs[curr_refs]):
#                 if len(prev_indexing) > 0:
#                     curr_spots, curr_refs = prev_indexing.T
#                     break
#                 else:
#                     # Only collinear solution found...
#                     return []
        
#         # Parse indexing
#         if len(prev_indexing) > 0:
#             prev_spots, prev_refs = np.asarray(prev_indexing).T

#             # Check to see if indexing has converged
#             if len(indexing) == len(prev_indexing):
#                 if all([curr_spots[i] == prev_spots[i]
#                         and curr_refs[i] == prev_refs[i]
#                         for i in range(len(indexing))]):
#                     break

#         # Update for next round
#         prev_indexing = indexing
#         orientation = Rotation.align_vectors(
#                             all_spot_qs[curr_spots],
#                             all_ref_qs[curr_refs])[0]

#         # Or break if max iterations reached
#         iter_count += 1
#         if iter_count >= iter_max:
#             break
    
#     return indexing


def seed_casting(seed,
                 all_spot_qs,
                 all_spot_ints,
                 all_ref_qs,
                 all_ref_fs,
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
            return seed # Force return of input with 0 qof

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

        # Query kdtree to find closest reference lattice points
        pot_conn = kdtree.query_ball_point(all_rot_qs,
                                           r=near_q)

        # Find closests points
        indexing = []
        for conn_i, conn in enumerate(pot_conn):
            if len(conn) == 0:
                continue
            elif len(conn) == 1:
                # Add candidate reflection
                indexing.append([conn[0], conn_i])
            else:
                # Add closest of multiple candidate reflections
                _, spot_idx = kdtree.query(all_rot_qs[conn_i])
                indexing.append([spot_idx, conn_i])
        indexing = np.asarray(indexing)

        # Break if indexing becomes orientationally indeterminant
        if len(indexing) < 2:
            # Has the original seed failed?
            if iter_count == 0:
                if isinstance(seed, Rotation):
                    return []
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
                if len(prev_indexing) > 0:
                    indexing = prev_indexing # revert
                    break
                else:
                    # Only collinear solution found...
                    return []
        
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

    return indexing


# def dask_multiple_seed_casting(seeds,
#                           all_spot_qs,
#                           all_spot_ints,
#                           all_ref_qs,
#                           all_ref_fs,
#                           near_q,
#                           iter_max=50,
#                           verbose=True):
    
#     # Modify and set up some values
#     evaluated_seed_mask = np.array([False,] * len(seeds))
#     all_spot_qs = np.asarray(all_spot_qs)
#     all_ref_qs = np.asarray(all_ref_qs)
#     all_ref_fs = np.asarray(all_ref_fs)

#     @dask.delayed
#     def dask_seed_casting(seed):
#         return seed_casting(seed,
#                             all_spot_qs,
#                             all_spot_ints,
#                             all_ref_qs,
#                             all_ref_fs,
#                             near_q,
#                             iter_max=iter_max)

#     indexings = []
#     for i, seed in enumerate(seeds):
#         indexings.append(dask_seed_casting(seed))

#     if verbose:
#         print('Casting valid seeds...')
#         with TqdmCallback(tqdm_class=tqdm):
#                 dask.compute(*indexings)
#     else:
#         dask.compute(*indexings)
    
#     return indexings


def multiple_seed_casting(seeds,
                          all_spot_qs,
                          all_spot_ints,
                          all_ref_qs,
                          all_ref_fs,
                          near_q,
                          iter_max=50,
                          exclude_found_seeds=False,
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
    if verbose:
        print('Casting valid seeds...')
    for i, seed in iterate(seeds):
        # Check if the pair has already been included
        if (exclude_found_seeds
            and evaluated_seed_mask[i]):
            continue

        indexing = seed_casting(seed,
                                all_spot_qs,
                                all_spot_ints,
                                all_ref_qs,
                                all_ref_fs,
                                near_q,
                                iter_max=iter_max)
        indexings.append(indexing)

        # Find and exclude included pairs
        # This actually slows it down while checking!
        if exclude_found_seeds:
            # print('Checking future seeds...')
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
    
    return indexings


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
                                     phase=None,
                                     all_ref_hkls=None,
                                     symmetrize='lattice',
                                     space_group_nr=230,
                                     desymmetrize_seeds=True,
                                     verbose=True):

    # Setup containers and working values
    best_indexings, best_qofs = [], []
    excluded_spots = set()
    included_spot_mask = np.asarray([True,] * len(all_spot_qs))
    valid_seed_mask = np.asarray([True,] * len(start_seeds))

    ORIENTATION_SEEDS = isinstance(start_seeds[0], Rotation)

    # Internal wrapper for indexing method
    def internal_indexing(seeds,
                          spots,
                          verbose=verbose):

        indexings = multiple_seed_casting(
            seeds,
            spots,
            all_spot_ints, # All spot ints passed to maintain qof
            all_ref_qs,
            all_ref_fs,
            near_q,
            iter_max=max_ori_refine_iter,
            verbose=verbose
            )
        
        return indexings

    def qualify_indexings(indexings,
                          reduce_seeds=True):

        if reduce_seeds:
            indexings, seed_groups = reduce_indexings(
                                            indexings,
                                            return_groups=True)
        else:
            seed_groups = [[i] for i in range(len(indexings))]

        # Determine quality of fit
        qofs = qofs_from_indexings(indexings,
                                   all_ref_qs,
                                   all_ref_fs,
                                   all_spot_qs,
                                   all_spot_ints,
                                   near_q,
                                   qmask)

        return indexings, qofs, seed_groups

    # First pass at indexing with scrubbed bad seeds
    indexings = internal_indexing(start_seeds, all_spot_qs)
    mask = np.array([len(ind) > 1 for ind in indexings])
    if np.any(~mask):
        indexings = [ind for ind, b in zip(indexings, mask) if b]
        start_seeds = [seed for seed, b in zip(start_seeds, mask) if b]

    # All seeds happen to be excluded
    if len(indexings) < 1:
        return best_indexings, np.asarray(best_qofs)

    # Reduce and qualify indexings
    indexings, qofs, seed_groups = qualify_indexings(indexings,
                                                     reduce_seeds=True)

    # Iteratively decompose pattern based on seeds
    iter_count = 0
    while True:
        # Desymmetrize only top results
        if symmetrize is not None and (iter_count > 0 or desymmetrize_seeds):
            # Trying to catch pseudosymmtery conditions
            qof_mask = qofs >= np.max(qofs) - 0.15 * np.max(qofs)
            if qof_mask.sum() < 6:
                if len(qofs) >= 6:
                    qof_mask = qofs >= sorted(np.unique(qofs))[-6]
                else:
                    qof_mask = np.ones_like(qofs, dtype=np.bool_)

            syms = [desymmetrize(indexing, 
                                 all_ref_qs,
                                 all_ref_fs,
                                 all_spot_qs,
                                 all_spot_ints,
                                 near_q,
                                 qmask,
                                 phase=phase,
                                 all_ref_hkls=all_ref_hkls,
                                 space_group_nr=space_group_nr,
                                 symmetrize=symmetrize)
                    for indexing, b in zip(indexings, qof_mask) if b]
            sym_indexings, sym_qofs = zip(*syms)

            for sym_i, ind_i in enumerate(np.nonzero(qof_mask)[0]):
                indexings[ind_i] = sym_indexings[sym_i]
                qofs[ind_i] = sym_qofs[sym_i]

        # Find best results
        best_ind = np.nanargmax(qofs)
        best_indexings.append(indexings[best_ind].astype(int))
        best_qofs.append(qofs[best_ind])

        # Remove already indexed spots from further analysis
        excluded_spots.update(set(indexings[best_ind][:, 0].astype(int)))
        included_spot_mask[np.array(list(excluded_spots))] = False
        valid_mask = np.asarray([False,] * len(indexings))
        recalc_mask = np.asarray([False,] * len(indexings))
        recalc_seeds = []

        # Find modified start seeds to recalculate
        for i in range(len(indexings)):
            indexed_spots = set(indexings[i][:, 0])

            # An indexed spot has been excluded
            if (indexed_spots - excluded_spots) != len(indexed_spots):
                # Determine if new indexing can be found
                if ORIENTATION_SEEDS:
                    # All orientations have potential
                    recalc_mask[i] = True
                    # recalc_seeds.extend(index_seeds)
                    recalc_seeds.extend(start_seeds)
                else:
                    index_seeds = [start_seeds[ind].copy() for ind in seed_groups[i]]
                    seed_mask = np.array([True,] * len(index_seeds))
                    for seed_i, seed in enumerate(index_seeds):
                        start_spots = set(seed[:, 0])
                        # Only recalculate seeds that are still valid
                        if len(start_spots - excluded_spots) >= 2:
                            recalc_mask[i] = True
                            recalc_seeds.append(seed)
                        else:
                            seed_mask[seed_i] = False
                    if not np.any(seed_mask):
                        valid_mask[i] = False
                    else:
                        seed_groups[i] = [s for s, b in zip(seed_groups[i], seed_mask) if b]

        indexings = [ind for ind, b in zip(indexings, valid_mask) if b]
        qofs = [q for q, b in zip(qofs, valid_mask) if b]
        seed_groups = [seed for seed, b in zip(seed_groups, valid_mask) if b]
        
        # Re-index
        if len(recalc_seeds) > 1:
            # Re-label spots to masked
            spot_indices = list(included_spot_mask.nonzero()[0])
            if not ORIENTATION_SEEDS:
                for i in range(len(recalc_seeds)):
                    recalc_seeds[i][:, 0] = [spot_indices.index(s) for s in recalc_seeds[i][:, 0]]
        
            # Re-index
            new_indexings = internal_indexing(
                                    recalc_seeds, 
                                    all_spot_qs[included_spot_mask],
                                    verbose=False)
        
            # Mask out failed indexing
            mask = np.array([len(ind) > 1 for ind in new_indexings])
            new_indexings = [ind for ind, b in zip(new_indexings, mask) if b]

            # Re-label spots to original
            for i in range(len(new_indexings)):
                new_indexings[i][:, 0] = np.array(spot_indices)[new_indexings[i][:, 0]]

            # Do not reduce subsequent iterations
            (new_indexings,
             new_qofs,
             new_seed_groups) = qualify_indexings(new_indexings)
            indexings += new_indexings
            qofs += new_qofs
            seed_groups += new_seed_groups
        
        # Conditionals to kill iteration
        iter_count += 1
        if (len(indexings) < 1 # Nothing left to compare
            or len(all_spot_qs) - len(excluded_spots) < 1 # Cannot solve orientations
            or np.max(qofs) < qof_minimum # Quality of fit has gotten too poor
            or iter_count >= max_ori_decomp_count): # Reach maxed allowed orientations
            break
  
    return best_indexings, np.asarray(best_qofs)
