import numpy as np
import scipy
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist
from orix.quaternion.symmetry import get_point_group
from tqdm import tqdm

from xrdmaptools.crystal.phase import get_default_sym_ops




##############################
### Base Utility Functions ###
##############################


def reduce_indexings(indexings, return_groups=False):

    groups = {}
    for idx, indexing in enumerate(indexings):
        # key = tuple(indexing[np.argsort(indexing[:, 0])].ravel())
        key = frozenset(map(tuple, indexing))

        groups.setdefault(key, []).append(idx)
    
    red_indexings = [indexings[groups[k][0]] for k in groups.keys()]

    if return_groups:
        return red_indexings, list(groups.values())
    return red_indexings



def qof_from_indexing(indexing,
                      all_ref_qs,
                      all_ref_fs,
                      all_spot_qs,
                      all_spot_ints,
                      near_q,
                      qmask,
                      orientation=None):

    # Parse indexing
    spot_inds, ref_inds = indexing.T
    
    # Rotate reference frame
    if orientation is None:
        orientation, _, _, _ = align_vectors_with_inversion(
                                    all_spot_qs[spot_inds],
                                    all_ref_qs[ref_inds])

    # all_rot_qs = orientation.apply(all_ref_qs, inverse=False)
    all_rot_qs = all_ref_qs @ orientation.T
    temp_qmask = qmask.generate(all_rot_qs)
    # Forces bad seeds to be included in qmask
    temp_qmask[ref_inds] = True

    filled_ints = np.zeros(sum(temp_qmask))
    for i, ind in enumerate(temp_qmask.nonzero()[0]):
        if ind in ref_inds:
            spot_ind = spot_inds[np.nonzero(ref_inds == ind)[0]][0]
            filled_ints[i] = all_spot_ints[spot_ind]

    qof = int_corr_qof(all_spot_qs[spot_inds],
                       all_rot_qs[ref_inds],
                       all_spot_ints[spot_inds],
                       all_spot_ints,
                       filled_ints,
                       all_ref_fs[temp_qmask],
                       sigma=near_q
                       )
    
    return qof


def qofs_from_indexings(indexings,
                        *args,
                        **kwargs):

    qofs = [qof_from_indexing(indexing,
                              *args,
                              **kwargs)
            for indexing in indexings]
    
    return qofs


def sort_by_qofs(qofs, *args):

        sorted_args = []
        for arg in args:
            sorted_args.append(
                [x for _, x in sorted(zip(qofs, arg),
                                      key=lambda pair: pair[0],
                                      reverse=True)]
            )
        
        qofs = sorted(qofs, reverse=True)

        return qofs, *sorted_args



def desymmetrize(indexing,
                 all_ref_qs,
                 all_ref_fs,
                 all_spot_qs,
                 all_spot_ints,
                 near_q,
                 qmask,
                 phase=None,
                 all_ref_hkls=None,
                 space_group_nr=230,
                 symmetrize='lattice',
                 return_all=False,
                 allow_improper=False):

    # Determine all symmetrically equivalent orientations
    # 'point_group' considers only Laue group
    if symmetrize is None:
        qof = qof_from_indexing(indexing,
                                all_ref_qs,
                                all_ref_fs,
                                all_spot_qs,
                                all_spot_ints,
                                near_q,
                                qmask)
        return indexing, qof
    
    # Get symmetry operators
    if phase is None:
        syms = get_default_sym_ops(space_group_nr, symmetrize=symmetrize)
    else:
        syms = phase.get_sym_ops(symmetrize=symmetrize)
    
    if not allow_improper:
        syms = syms[np.linalg.det(syms) > 0.5]

    # Parse symmetrized indexing
    spot_inds, ref_inds = indexing.T

    if phase is None:
        sym_qs = all_ref_qs[ref_inds] @ syms
        sym_ref_inds = []
        for i in range(len(sym_qs)):
            temp_ref_inds = []
            for q_vect in sym_qs[i]:
                dist = np.linalg.norm(all_ref_qs - q_vect, axis=1)
                if dist.min() <= near_q:
                    temp_ref_inds.append(np.argmin(dist))
                else:
                    break
            else:
                sym_ref_inds.append(temp_ref_inds)

    else:
        sym_hkls = phase.HKL(all_ref_qs[ref_inds] @ syms).round(0)
        sym_ref_inds = []
        for i in range(len(sym_hkls)):
            temp_ref_inds = []
            for hkl in sym_hkls[i]:
                match_mask = np.all(all_ref_hkls == hkl, axis=1)
                if match_mask.sum() > 0:
                    temp_ref_inds.append(match_mask.nonzero()[0][0])
                else:
                    break
            else:
                sym_ref_inds.append(temp_ref_inds)

    
    sym_indexings = []
    for ref_inds in sym_ref_inds:
        sym_indexings.append(np.asarray([spot_inds, ref_inds]).T)
    
    # Break to determine orientation magnitudes
    sym_qofs, sym_mags, sym_oris = [], [], []
    for indexing in sym_indexings:
        spot_inds, ref_inds = indexing.T
        M, R, _, _ = align_vectors_with_inversion(
                                    all_spot_qs[spot_inds],
                                    all_ref_qs[ref_inds])

        sym_qof = qof_from_indexing(indexing,
                                    all_ref_qs,
                                    all_ref_fs,
                                    all_spot_qs,
                                    all_spot_ints,
                                    near_q,
                                    qmask,
                                    orientation=M)
        
        sym_qofs.append(sym_qof)
        sym_mags.append(R.magnitude())
        sym_oris.append(M)
    
    if return_all:
        return sym_indexings, sym_qofs, sym_mags, sym_oris
    
    # Pick the best
    best_qof_mask = np.array(sym_qofs == np.max(sym_qofs))
    min_mag_ind = np.argmin(np.asarray(sym_mags)[best_qof_mask])
    best_ind = best_qof_mask.nonzero()[0][min_mag_ind]

    return sym_indexings[best_ind], sym_qofs[best_ind]


def multi_desymmetrize(indexings, *args, **kwargs):
    """
    Combined method to account for potential pseudosymmetery
    """

    best_indexing = np.asarray([[], []])
    best_qof = np.nan

    for indexing in indexings:
        sym_indexing, sym_qof = desymmetrize(indexing,
                                              *args,
                                              **kwargs)
        if (np.isnan(best_qof)
            or sym_qof > best_qof):
            best_indexing = sym_indexing
            best_qof = sym_qof
        
    return best_indexing, best_qof


def align_vectors_with_inversion(a, b, weights=None):
    """
    Version of scipy Rotation class align_vectors method which allows
    for improper rotations present in real crystal symmetry groups.
    """

    if weights is None:
        w = np.ones(len(a))
    else:
        w = np.array(weights, dtype=float)
    w /= np.sum(w)

    H = np.dot(a.T * w, b)

    U, S, Vt = np.linalg.svd(H)
    M = np.dot(U, Vt)
    det = np.linalg.det(M)

    is_inverted = det < 0

    if is_inverted:
        R_proper = Rotation.from_matrix(-1.0 * M)
    else:
        R_proper = Rotation.from_matrix(M)
    
    b_transformed = np.dot(b, M.T)
    rssd = np.sqrt(np.sum(w * np.sum((a - b_transformed)**2, axis=1)))

    return M, R_proper, rssd, is_inverted
    

    





##################################
### Quality of Fit Evaluations ###
##################################




# def get_quality_of_fit(fit_spot_qs,
#                        fit_spot_ints,
#                        fit_rot_qs,
#                        fit_ref_fs,
#                        all_spot_ints,
#                        all_ref_fs,
#                        **kwargs):

#     # Ideal conditions
#     # 1. Penalize missing reference reflections weighted according to their expected intensity
#     # 2. Do not penalize extra measured reflections which are not indexing (allows for overlapping orientations)
#     # 3. Penalize reflections weighted by their distance from expected postions

#     kwargs.setdefault('sigma', 0.1)
#     sigma = kwargs.pop('sigma')

#     # qof = get_rmse(fit_spot_qs,
#     #                fit_rot_qs)

#     # qof = weighted_distance_qof(fit_spot_qs,
#     #                             fit_rot_qs,
#     #                             fit_ref_fs,
#     #                             all_ref_fs,
#     #                             int_weight=0.5)

#     # qof = complete_distance_qof(fit_spot_qs,
#     #                             fit_rot_qs,
#     #                             all_rot_fs,
#     #                             sigma=1,
#     #                             int_weight=0.1)

#     # qof = explained_intensity_qof(fit_spot_ints,
#     #                               fit_ref_fs,
#     #                               all_spot_ints,
#     #                               all_ref_fs,
#     #                               ratio=0.75)
    
#     qof = dist_int_qof(fit_spot_qs,
#                        fit_rot_qs,
#                        fit_spot_ints,
#                        all_spot_ints,
#                        fit_ref_fs,
#                        all_ref_fs,
#                        sigma=sigma,
#                        ratio=0.5)

#     # qof = len(fit_spot_qs) / len(all_spot_ints)

#     return qof


# def weighted_distance_qof(fit_spot_qs,
#                           fit_rot_qs,
#                           fit_ref_fs,
#                           all_ref_fs,
#                           sigma=1,
#                           int_weight=0.5):

#     # # Determine which reflections are indexed
#     # found_spot_mask = [tuple(ref) in [tuple(x) for x in ref_qs] for ref in all_ref_qs]

#     dist = [np.sqrt(np.sum([(p - q)**2 for p, q in zip(v1, v2)]))
#             for v1, v2 in zip(fit_spot_qs, fit_rot_qs)]

#     # Gaussian with specified standard deviation
#     # centered at zero sampled at distance
#     dist_val = np.sum(np.exp(-(np.asarray(dist))**2 / (2 * sigma**2)))
#     norm_dist_val = dist_val / len(all_ref_fs)

#     int_val = (np.sum(fit_ref_fs)
#               / np.sum(all_ref_fs))

#     qof = (int_weight * int_val) + ((1 - int_weight) * norm_dist_val)

#     # If not all inputs are given, the whole thing can fail
#     # Return an operable value
#     if np.isnan(qof):
#         qof = 0

#     return qof


# def complete_distance_qof(fit_spot_qs,
#                           fit_rot_qs,
#                           all_rot_fs,
#                           sigma=1,
#                           int_weight=0.5): # Gaussian standard deviation to evaluate


#     # Determine which reflections are indexed
#     # found_spot_mask = [tuple(ref) in [tuple(x) for x in ref_qs] for ref in all_ref_qs]

#     dist = [np.sqrt(np.sum([(p - q)**2 for p, q in zip(v1, v2)]))
#             for v1, v2 in zip(fit_spot_qs, fit_rot_qs)]

#     # Gaussian with specified standard deviation
#     # centered at zero sampled at distance
#     qof = np.sum(np.exp(-(np.asarray(dist))**2 / (2 * sigma**2)))

#     max_qof = len(all_rot_fs)
#     norm_qof = qof / max_qof

#     return norm_qof


# def explained_intensity_qof(fit_spot_ints,
#                             fit_ref_fs,
#                             all_spot_ints,
#                             all_ref_fs,
#                             ratio=0.5,
#                             ):

#     exp_val = np.sum(fit_spot_ints) / np.sum(all_spot_ints)
#     ref_val = np.sum(fit_ref_fs) / np.sum(all_ref_fs)

#     # print(f'{exp_val=}')
#     # print(f'{ref_val=}')

#     return (ratio * exp_val) + ((1 - ratio) * ref_val)


# def dist_int_qof(fit_spot_qs,
#                  fit_rot_qs,
#                  fit_spot_ints,
#                  all_spot_ints,
#                  fit_ref_fs,
#                  all_ref_fs,
#                  sigma=1,
#                  ratio=0.5):


#     dist = [np.sqrt(np.sum([(p - q)**2 for p, q in zip(v1, v2)]))
#             for v1, v2 in zip(fit_spot_qs, fit_rot_qs)]
    
#     # Gaussian with specified standard deviation
#     # centered at zero sampled at distance
#     gauss_int = fit_spot_ints * np.exp(-(np.asarray(dist))**2 / (2 * sigma**2))

#     exp_val = np.sum(gauss_int) / np.sum(all_spot_ints)
#     ref_val = np.sum(fit_ref_fs) / np.sum(all_ref_fs)

#     return (ratio * exp_val) + ((1 - ratio) * ref_val)


def int_corr_qof(fit_spot_qs,
                 fit_rot_qs,
                 fit_spot_ints,
                 all_spot_ints,
                 filled_ints,
                 all_ref_fs,
                 sigma=1):

    # dist = [np.sqrt(np.sum([(p - q)**2 for p, q in zip(v1, v2)]))
    #         for v1, v2 in zip(fit_spot_qs, fit_rot_qs)]

    dist = np.linalg.norm(fit_spot_qs - fit_rot_qs, axis=-1)
    
    # Gaussian with specified standard deviation
    # centered at zero sampled at distance
    gauss_int = fit_spot_ints * np.exp(-(np.asarray(dist))**2 / (2 * sigma**2))

    # Gaussian-weighted explained intensity fraction
    exp_val = np.sum(gauss_int) / np.sum(all_spot_ints)
    # print(f'{exp_val=}')

    # Correlation of measured and calculated intensities
    ref_val = np.dot(filled_ints, all_ref_fs) / (np.linalg.norm(filled_ints) * np.linalg.norm(all_ref_fs))
    # print(f'{ref_val=}')

    # return (ratio * exp_val) + ((1 - ratio) * ref_val)
    return exp_val**2 * ref_val # Both must be large


def get_rmse(fit_spot_qs,
             fit_rot_qs):
    rmse = np.mean([np.sqrt(np.sum([(p - q)**2 for p, q in zip(v1, v2)]))
                    for v1, v2 in zip(fit_spot_qs, fit_rot_qs)])

    return rmse





#############################
### Large Level Functions ###
#############################


def pixel_spots(spots, indices):
    ps = spots[((spots['map_x'] == indices[1]) & (spots['map_y'] == indices[0]))]
    return ps


def symmetrize_and_stitch_v2(spots_3D,
                             phase,
                             near_q,
                             qmask,
                             mis_thresh=0.5,
                             metric='int_corr',
                             degrees=True,
                             verbose=False):

    if metric not in spots_3D:
        raise KeyError(f'{metric} not in spots_3D')

    map_shape = (np.max(spots_3D['map_y']) + 1,
                 np.max(spots_3D['map_x']) + 1)
    if degrees:
        mis_thresh = np.radians(mis_thresh)

    all_oris = []
    next_grain_id = 0
    curr_row_oris = [{} for _ in range(map_shape[1])]
    with tqdm(total=np.prod(map_shape)) as pbar:
        if verbose:
            pbar.disable = True
        # Iterate by row
        for row in range(map_shape[0]):
            prev_row_oris = curr_row_oris.copy()
            curr_row_oris = [{} for _ in range(map_shape[1])]

            # Then iteratre for each pixel in row
            for col in range(map_shape[1]):
                pixel_grain_symmetries = {}

                indices = (row, col)
                pixel_df = pixel_spots(spots_3D, indices)
                grain_ids = np.unique(pixel_df['grain_id'].values[~np.isnan(pixel_df['grain_id'].values)])
                if verbose:
                    print(f'Grain assignment for pixel {indices}.')

                if len(grain_ids) == 0:
                    if verbose:
                        print(f'No grains found.')
                    pbar.update(1)
                    continue

                # Extract current pixel grains and their symmetries
                for grain_id in grain_ids:
                    grain_mask = pixel_df['grain_id'].values == grain_id
                    spot_inds = grain_mask.nonzero()[0]
                    hkls = pixel_df.loc[grain_mask, ['h', 'k', 'l']].values
                    ref_inds = np.array([np.all(phase.all_hkls == hkl, axis=1).nonzero()[0][0] for hkl in hkls])
                    spot_qs = pixel_df[['qx', 'qy', 'qz']].values
                    spot_ints = pixel_df[metric].values

                    indexing = np.asarray([spot_inds, ref_inds]).T

                    (sym_indexings,
                     sym_qofs,
                     sym_mags,
                     sym_oris) = desymmetrize(indexing,
                                              phase.all_qs,
                                              phase.all_fs,
                                              spot_qs,
                                              spot_ints,
                                              near_q,
                                              qmask,
                                              space_group_nr=phase.lattice.space_group_nr,
                                              symmetrize='lattice',
                                              return_all=True)
                    
                    # Reduce symmetry operations to equivalents in fundamental zone?

                    spots_3D.loc[pixel_df.index[grain_mask], 'grain_id'] = next_grain_id
                    pixel_df.loc[pixel_df.index[grain_mask], 'grain_id'] = next_grain_id
                    pixel_grain_symmetries[next_grain_id] = (np.array(sym_qofs), np.asarray(sym_oris), np.array(sym_mags), np.asarray(sym_indexings))
                    next_grain_id += 1

                # Check neighboring symmetries and stitch together if similar enough
                top_keys = list(prev_row_oris[col].keys())
                top_oris = [v[1] for v in prev_row_oris[col].values()] # (n_neighbors, 3, 3)
                if col > 0:
                    left_keys = list(curr_row_oris[col - 1].keys())
                    left_oris = [v[1] for v in curr_row_oris[col - 1].values()] # (n_neighbors, 3, 3)
                else:
                    left_keys, left_oris = [], []
                # Could have redundant keys for the same grain 
                near_keys = top_keys + left_keys 
                near_oris = top_oris + left_oris

                # No neighbors; sort symmetries and move on    
                if len(near_oris) == 0:
                    for grain_id, (sym_qofs, sym_oris, sym_mags, sym_indexings) in pixel_grain_symmetries.items():
                        sym_qofs, sym_oris, sym_mags, sym_indexings = sort_by_qofs(sym_qofs, sym_oris, sym_mags, sym_indexings)
                        curr_row_oris[col][grain_id] = (np.array(sym_qofs), np.asarray(sym_oris), np.array(sym_mags), np.asarray(sym_indexings))
                    if verbose:
                        print(f'Assignment Condition 1: No neighboring grain IDs. Keeping current ID {grain_id}.')
                    pbar.update(1)
                    continue

                try:
                    # Check misorientation across all symmetries and neighbors
                    grain_sym_oris = np.asarray([v[1] for v in pixel_grain_symmetries.values()]) # (n_grains, n_syms, 3, 3)
                    mis_oris = grain_sym_oris[:, :, np.newaxis, np.newaxis] @ np.asarray(near_oris).transpose(0, 1, 3, 2)[np.newaxis, np.newaxis] # (n_grains, n_grain_syms, n_neighbors, 3, 3)
                    mis_mags = Rotation.from_matrix(mis_oris.reshape(-1, 3, 3)).magnitude().reshape(mis_oris.shape[:-2]) # (n_grains, n_grain_syms, n_neighbors, n_neighbor_syms)          
                    matches = mis_mags < mis_thresh # (n_grains, n_grain_syms, n_neighbors, n_neighbor_syms) 
                except Exception as e:
                    print(e)
                    raise e
                    return pixel_grain_symmetries, near_oris

                # Symmetry invariant versions
                sym_mis_mags = np.min(mis_mags, axis=(1, 3)) # (n_grains, n_neighbors)
                sym_matches = sym_mis_mags < mis_thresh # (n_grains, n_neighbors)

                if sym_matches.sum() == 0:
                    for grain_id, (sym_qofs, sym_oris, sym_mags, sym_indexings) in pixel_grain_symmetries.items():
                        sym_qofs, sym_oris, sym_mags, sym_indexings = sort_by_qofs(sym_qofs, sym_oris, sym_mags, sym_indexings)
                        curr_row_oris[col][grain_id] = (np.array(sym_qofs), np.asarray(sym_oris), np.array(sym_mags), np.asarray(sym_indexings))
                    if verbose:
                        print(f'Assignment Condition 2: No matching neighboring grain IDs. Keeping current ID {grain_id}.')
                    pbar.update(1)
                    continue

                # Grain assignment logic
                assigned_grain_mask = np.zeros(len(pixel_grain_symmetries), dtype=bool)
                assigned_neighbor_mask = np.zeros(len(near_oris), dtype=bool)
                while (any(~assigned_grain_mask) # current pixel orientations remain
                       and any(~assigned_neighbor_mask) # current neighboring orientations remain
                       and sym_matches.sum() > 0): # matches between the two remain

                    min_index = np.argmin(sym_mis_mags[np.ix_(~assigned_grain_mask, ~assigned_neighbor_mask)])
                    min_mis_mag = np.min(sym_mis_mags[np.ix_(~assigned_grain_mask, ~assigned_neighbor_mask)])
                    min_grain_idx, min_near_idx = np.unravel_index(min_index, sym_mis_mags[np.ix_(~assigned_grain_mask, ~assigned_neighbor_mask)].shape)
                    grain_id = np.array(list(pixel_grain_symmetries.keys()))[~assigned_grain_mask][min_grain_idx]

                    # Find matrix of all symmetry combinations
                    sym_matrix = mis_mags[~assigned_grain_mask][:, :, ~assigned_neighbor_mask][min_grain_idx, :, min_near_idx] # (n_syms, n_syms)
                    sym_indexing = np.argmin(sym_matrix, axis=0)

                    # Single match; add to grain, sort sym_oris and move on
                    if sym_matches[np.ix_(~assigned_grain_mask, ~assigned_neighbor_mask)][min_grain_idx].sum() == 1:
                        near_id = np.array(near_keys)[~assigned_neighbor_mask][min_near_idx]
                        assigned_neighbor_mask[np.array(near_keys) == near_id] = True
                        sym_matches[:, assigned_neighbor_mask] = 0
                        if verbose:
                            print(f'Assignment Condition 3: Grain ID {grain_id} assigned to single neighbor ID {near_id}.')                     
              
                    # Multiple neighbor matches
                    else:
                        match_idxs = sym_matches[np.ix_(~assigned_grain_mask, ~assigned_neighbor_mask)][min_grain_idx].nonzero()[0]
                        near_ids = np.array(near_keys)[~assigned_neighbor_mask][match_idxs]
                        pixel_ids = np.array([0,] * len(top_keys) + [1,] * len(left_keys))[~assigned_neighbor_mask][match_idxs]
                        best_pixel_id = np.array([0,] * len(top_keys) + [1,] * len(left_keys))[~assigned_neighbor_mask][min_near_idx]
                        near_id = None

                        # First, simple check if the multiple matches connnect to the same neighbor grain in both pixels
                        # There should never be the same grain twice in the same pixel, so this has to come from both pixels
                        if len(np.unique(near_ids)) == 1:
                            near_id = near_ids[0]
                            # All matches have been assigned
                            assigned_neighbor_mask[np.array(near_keys) == near_id] = True
                            sym_matches[:, assigned_neighbor_mask] = 0
                            if verbose:
                                print(f'Assignment Condition 4: Grain ID {grain_id} assigned to bordering neighbor grain with ID {near_id}.')
                        
                                                  

                        # Second, check if the best match exists in both neighboring pixels
                        if near_id is None:
                            if ((best_pixel_id == 0 and np.array(near_keys)[~assigned_neighbor_mask][min_near_idx] in left_keys) # Best match in top pixel
                                or (best_pixel_id == 1 and np.array(near_keys)[~assigned_neighbor_mask][min_near_idx] in top_keys)): # Best match in left pixel
                                near_id = np.array(near_keys)[~assigned_neighbor_mask][min_near_idx]

                                # Update mask of neighboring grains
                                assigned_neighbor_mask[np.array(near_keys) == near_id] = True
                                sym_matches[:, assigned_neighbor_mask] = 0
                                
                                if verbose:
                                    print(f'Assignment Condition 5: Grain ID {grain_id} assigned from multiple neighbors to ID {near_id}.')

                        # Stitching
                        # Third, find the best match in both pixels that does NOT create overlapping grains
                        if near_id is None:
                            best_near_id = np.array(near_keys)[~assigned_neighbor_mask][min_near_idx]
                            keys_to_check = []
                            for poss_near_id, pixel_id, match_id in zip(near_ids, pixel_ids, match_idxs):
                                if pixel_id == best_pixel_id:
                                    continue
                                
                                # Check if the possible matches come from seperate pixels
                                if ((best_pixel_id == 0 and poss_near_id not in top_keys)
                                    or (best_pixel_id == 1 and poss_near_id not in left_keys)):
                                    
                                    # Final check to make sure the two grains do not overlap anywhere else in the map
                                    overlap = False
                                    for row_oris in [curr_row_oris, prev_row_oris, *all_oris]:
                                        for d in row_oris:
                                            if poss_near_id in d and best_near_id in d:
                                                overlap = True
                                                break
                                        if overlap:
                                            break
                                    if overlap:
                                        continue

                                    sym_mis_mag = sym_mis_mags[np.ix_(~assigned_grain_mask, ~assigned_neighbor_mask)][min_grain_idx][match_id]
                                    keys_to_check.append((poss_near_id, sym_mis_mag, match_id))
                                
                            if len(keys_to_check) > 0:
                                keys_to_check = np.asarray(keys_to_check)
                                match_near_ids = [best_near_id, # Best match
                                                  keys_to_check[np.argmin(keys_to_check[:, 1]), 0]] # Best in neighbor without overlap
                                
                                # The second of match_near_keys will need the symmetry order updated to match the other key's order
                                min_non_overlap_idx = int(keys_to_check[np.argmin(keys_to_check[:, 1]), -1]) 
                                new_sym_matrix = mis_mags[~assigned_grain_mask][:, :, ~assigned_neighbor_mask][min_grain_idx, :, min_non_overlap_idx] # (n_syms, n_syms)
                                new_sym_indexing = np.argmin(new_sym_matrix, axis=1)                            
                                                  
                                # Get new unique grain id; will help prevent errors
                                near_id = next_grain_id
                                next_grain_id += 1
                                if verbose:
                                    print(f'Assignment Condition 6: Stitching grain IDs {match_near_ids} into new ID {near_id}.')

                                # Update near_keys with new grain_id
                                near_keys = [k if k not in match_near_ids else near_id for k in near_keys]                              

                                # Convert old near_ids to the same new near_id
                                for i, old_id in enumerate(match_near_ids):
                                    # Update pandas dataframes
                                    spots_3D.loc[spots_3D['grain_id'].values == old_id, 'grain_id'] = near_id
                                    pixel_df.loc[pixel_df['grain_id'].values == old_id, 'grain_id'] = near_id

                                    # Update symmetry tracking along rows
                                    for row_oris in [curr_row_oris, prev_row_oris, *all_oris]:
                                        for d in row_oris:
                                            if old_id in d:
                                                sym_info = d.pop(old_id)
                                                if i == 1:
                                                    sym_info = tuple([np.array(vals)[new_sym_indexing[sym_indexing]] for vals in sym_info])
                                                d[near_id] = sym_info
    
                                # Update mask of neighboring grains
                                assigned_neighbor_mask[np.array(near_keys) == near_id] = True
                                sym_matches[:, assigned_neighbor_mask] = 0

                            # If no option without overlapp, then only assign to closest match
                            else:
                                near_id = np.array(near_keys)[~assigned_neighbor_mask][min_near_idx]
                                assigned_neighbor_mask[np.array(near_keys) == near_id] = True
                                sym_matches[:, assigned_neighbor_mask] = 0
                                if verbose:
                                    print(f'Assignment Condition 7: Grain ID {grain_id} assigned to only non-overlapping neighbor grain ID {near_id}.')

                    # Update values
                    sym_qofs, sym_oris, sym_mags, sym_indexings = pixel_grain_symmetries[grain_id]

                    curr_row_oris[col][near_id] = (
                        sym_qofs[sym_indexing],
                        sym_oris[sym_indexing],
                        sym_mags[sym_indexing],
                        sym_indexings[sym_indexing]
                    )

                    grain_mask = pixel_df['grain_id'] == grain_id
                    spots_3D.loc[pixel_df.index[grain_mask], 'grain_id'] = near_id
                    pixel_df.loc[pixel_df.index[grain_mask], 'grain_id'] = near_id
                    assignment_index = np.nonzero(~assigned_grain_mask)[0][min_grain_idx]
                    assigned_grain_mask[assignment_index] = True
                    sym_matches[assigned_grain_mask] = 0

                # Add any unassigned grains as new grains
                if (~assigned_grain_mask).sum() > 0:
                    for grain_idx in np.nonzero(~assigned_grain_mask)[0]:
                        grain_id, (grain_qofs, grain_oris, grain_mags, grain_indexings) = list(pixel_grain_symmetries.items())[grain_idx]
                        curr_row_oris[col][grain_id] = sort_by_qofs(grain_qofs, grain_oris, grain_mags, grain_indexings)
                        if verbose:
                            print(f'Assignment Condition 8: No matching neighboring grain IDs. Keeping current ID {grain_id}.')                
                
                nan_mask = ~np.isnan(pixel_df['grain_id'].values)
                if not all(pixel_df['grain_id'].values[nan_mask] == spots_3D.loc[pixel_df.index]['grain_id'].values[nan_mask]):
                    raise
                pbar.update(1)

            all_oris.append(curr_row_oris)
    all_oris = np.asarray(all_oris, dtype=object)
    
    # return all_oris

    rem_grains = {}
    for row in range(map_shape[0]):
        for col in range(map_shape[1]):
            indices = (row, col)
            for grain_id, sym_info in all_oris[indices].items():
                if grain_id in rem_grains:
                    rem_grains[grain_id].append((indices, *sym_info))
                else:
                    rem_grains[grain_id] = [(indices, *sym_info)]
    
    # Determine best symmetry across full grain
    # sym_info = (indices, sym_qofs, sym_oris, sym_mags, sym_indexings)
    for grain_id, grain_info in rem_grains.items():
        qofs = np.asarray([gi[1] for gi in grain_info])
        mags = np.asarray([gi[3] for gi in grain_info])
        med_qofs = np.median(qofs, axis=0)
        med_mags = np.median(mags, axis=0)

        # TODO: This is where a step to check for multi-modal distributions
        # Current assumption is mono-modal

        # Mask to check for actual symmetry and not just Laue symmetry
        qof_sym_mask = med_qofs.round(10) == np.min(med_qofs).round(10)
        # Find the symmetry condition with smallest misorientation from reference frame
        # This is treated as the fundamental zone
        best_sym_index = qof_sym_mask.nonzero()[0][np.argmin(med_mags[qof_sym_mask])]

        all_spot_inds = []
        all_hkls = []
        for gi in grain_info:
            pixel_df = pixel_spots(spots_3D, gi[0])
            grain_mask = pixel_df['grain_id'].values == grain_id
            
            all_spot_inds.extend(list(pixel_df[grain_mask].index))
            all_hkls.extend(phase.all_hkls[gi[4][best_sym_index][:, 1]])

            if len(gi[4][best_sym_index][:, 1]) != grain_mask.sum():
                print(gi[0], pixel_df, gi[4][best_sym_index][:, 1])
        
        # Update hkls to best symmetry condition across full grain
        if len(np.asarray(all_hkls)) != len(all_spot_inds):
            print('Grain Assignment Error!')
            return rem_grains

        try:
            spots_3D.loc[all_spot_inds, ['h', 'k', 'l']] = np.asarray(all_hkls)
        except ValueError as e:
            print(e)
            raise ValueError# surface level for debug
    
    if verbose:
        print('Sorting grain IDs')
    # Finally re-label grains with lowest number having the greatest number of pixels
    counts = {k : len(v) for k, v in rem_grains.items()}
    sorted_grain_ids = [x for x, _ in sorted(counts.items(),
                                        key=lambda pair: pair[1],
                                        reverse=True)]

    # In-place, re-label grains
    all_grain_id_labels = np.array([np.nan,] * len(spots_3D))
    for new_grain_id, sorted_grain_id in enumerate(sorted_grain_ids):
        if verbose:
            print(f'Old grain ID {sorted_grain_id} includes {counts[sorted_grain_id]} pixels and will be re-labeled as {new_grain_id}.')
        all_grain_id_labels[spots_3D['grain_id'].values == sorted_grain_id] = new_grain_id
    spots_3D['grain_id'] = all_grain_id_labels

    return counts




def symmetrize_and_stitch(spots_3D,
                          phase,
                          mis_thresh=0.5,
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
                
                # symmetrize orientations. shape = (# orientations in pixel, # of symmetry, 3, 3)
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