import numpy as np
from tqdm import tqdm
import scipy
from scipy.spatial.transform import Rotation
from scipy.spatial import distance_matrix, KDTree
from scipy.optimize import curve_fit
from itertools import combinations, product
from collections import Counter

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


def pair_casting_index_best_grain(all_ref_qs,
                                  all_ref_hkls,
                                  all_ref_fs,
                                  min_q,
                                  all_spot_qs,
                                  all_spot_ints,
                                  near_q,
                                  near_angle,
                                  qmask,
                                  degrees=False,
                                  max_ori_refine_iter=50,
                                  max_ori_decomp_count=20,
                                  keep_initial_pair=False,
                                  generate_reciprocal_lattice=False,
                                  verbose=True):

    # Find all valid pairs within near_q and near_angle
    pairs = find_all_valid_pairs(
            all_spot_qs,
            all_ref_qs,
            near_q,
            near_angle,
            min_q,
            degrees=degrees,
            verbose=verbose)

    if len(pairs) > 0:
        # Symmetrically reduce pairs
        red_pairs = reduce_symmetric_equivalents(
                pairs,
                all_spot_qs,
                all_ref_qs,
                all_ref_hkls,
                near_angle,
                min_q,
                degrees=degrees,
                verbose=verbose)
        
        if len(red_pairs) > 0:
            # Index spots
            connections, qofs, _ = pair_casting_indexing(
                    red_pairs,
                    all_spot_qs,
                    all_spot_ints,
                    all_ref_qs,
                    all_ref_fs,
                    qmask,
                    near_q,
                    iter_max=max_ori_refine_iter,
                    keep_initial_pair=keep_initial_pair,
                    exclude_found_pairs=False,
                    verbose=verbose)
        else:
            # This is where all nans are coming from!
            best_connections = [[np.nan,] * len(all_spot_qs)]
            best_qofs = [np.nan,] * len(all_spot_qs)
    else:
        # This is where all nans are coming from!
        best_connections = [[np.nan,] * len(all_spot_qs)]
        best_qofs = [np.nan,] * len(all_spot_qs)

    return connections[np.argmax(qofs)], qofs[np.argmax(qofs)]


def pair_casting_index_full_pattern(all_ref_qs,
                                    all_ref_hkls,
                                    all_ref_fs,
                                    min_q,
                                    all_spot_qs,
                                    all_spot_ints,
                                    near_q,
                                    near_angle,
                                    qmask,
                                    degrees=False,
                                    qof_minimum=0.2,
                                    max_ori_refine_iter=50,
                                    max_ori_decomp_count=20,
                                    keep_initial_pair=False,
                                    generate_reciprocal_lattice=False,
                                    verbose=True):

    # Find all valid pairs within near_q and near_angle
    pairs = find_all_valid_pairs(
            all_spot_qs,
            all_ref_qs,
            near_q,
            near_angle,
            min_q,
            degrees=degrees,
            verbose=verbose)
    
    if len(pairs) > 0:
        # Symmetrically reduce pairs
        red_pairs = reduce_symmetric_equivalents(
                pairs,
                all_spot_qs,
                all_ref_qs,
                all_ref_hkls,
                near_angle,
                min_q,
                degrees=degrees,
                verbose=verbose)
        
        if len(red_pairs) > 0:
            # Iteratively decompose patterns
            (best_connections, 
             best_qofs) = decaying_pattern_decomposition(
                    red_pairs,
                    all_spot_qs,
                    all_spot_ints,
                    all_ref_qs,
                    all_ref_fs,
                    qmask,
                    near_q,
                    qof_minimum=qof_minimum,
                    keep_initial_pair=keep_initial_pair,
                    max_ori_refine_iter=max_ori_refine_iter,
                    max_ori_decomp_count=max_ori_decomp_count,
                    verbose=verbose)
        else:
            # This is where all nans are coming from!
            best_connections = [[np.nan,] * len(all_spot_qs)]
            best_qofs = [np.nan]
    else:
        # This is where all nans are coming from!
        best_connections = [[np.nan,] * len(all_spot_qs)]
        best_qofs = [np.nan]

    return best_connections, best_qofs



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


def phase_based_index_full_pattern(phase, 
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

    return pair_casting_index_full_pattern(all_ref_qs,
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


def find_all_valid_pairs(all_spot_qs,
                         all_ref_qs,
                         near_q,
                         near_angle,
                         min_q,
                         degrees=False,
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

    blank_connection = np.array([np.nan,] * len(phase_mask))
    connection_pairs = []
    if verbose:
        print('Finding all valid pairs...', flush=True)
    for pair in iterate(spot_pair_indices):
        ref_combos = list(product(*[np.nonzero(mag_diff_arr[i] < near_q)[0]
                          for i in pair]))

        angle_mask = [np.abs(spot_angles[tuple(pair)]
                      - ref_angles[tuple(combo)]) < near_angle
                      for combo in ref_combos]
        doublet_mask = [combo[0] != combo[1] for combo in ref_combos]

        ref_combos = np.asarray(ref_combos)[(np.asarray(angle_mask)
                                             & np.asarray(doublet_mask))]
        
        if len(ref_combos) > 0:
            for combo in ref_combos:
                connection = blank_connection.copy()
                connection[phase_inds[pair[0]]] = combo[0]
                connection[phase_inds[pair[1]]] = combo[1]
                connection_pairs.append(connection)

    return np.asarray(connection_pairs)


# Xrayutilities isequivalent function might be useful...
def reduce_symmetric_equivalents(connection_pairs,
                                 all_spot_qs,
                                 all_ref_qs,
                                 all_ref_hkls,
                                 near_angle,
                                 min_q,
                                 degrees=False,
                                 verbose=True):
    
    # near_angle = 0

    # Convert to arrays
    all_spot_qs = np.asarray(all_spot_qs)
    all_ref_qs = np.asarray(all_ref_qs)
    all_ref_hkls = np.asarray(all_ref_hkls)

    # All internal comparisons in radians
    if degrees:
        near_angle = np.radians(near_angle)

    # Construct iterator. Generators cannot be reused
    if verbose:
        iterate = lambda x : tqdm(enumerate(x),
                                  total=len(x),
                                  position=0,
                                  leave=True)
    else:
        iterate = lambda x : enumerate(x)
    
    # Evaluating valid pairs
    pair_orientations = []
    pair_mis_mag = []
    pair_rmse = []
    if verbose:
        print('Evaluating all valid pairs...')
    for pair_i, pair_connection in iterate(connection_pairs):
        (spot_indices,
        ref_indices) = _get_connection_indices(pair_connection)
        pair_ref_hkls = all_ref_hkls[ref_indices]
        pair_ref_qs = all_ref_qs[ref_indices]
        pair_spot_qs = all_spot_qs[spot_indices]

        # Check collinearity.
        # 3D orientation cannot be determined from collinear pairs
        if are_collinear(pair_ref_hkls):
            pair_orientations.append(np.nan) # assumes validity
            pair_rmse.append(np.nan)
            pair_mis_mag.append(np.nan)
            continue
        
        orientation, _ = Rotation.align_vectors(pair_spot_qs,
                                                pair_ref_qs)
        rmse = get_rmse(pair_spot_qs,
                        orientation.apply(pair_ref_qs,
                                          inverse=False))

        pair_orientations.append(orientation)
        pair_mis_mag.append(orientation.magnitude())
        pair_rmse.append(rmse)
    
    pair_mis_mag = np.asarray(pair_mis_mag)
    pair_rmse = np.asarray(pair_rmse).round(10)

    # Reducing symmetrically equivalent pairs
    eval_pair_mask = np.array([True,] * len(connection_pairs))
    keep_pair_mask = eval_pair_mask.copy()
    for pair_i, pair_connection in enumerate(connection_pairs):
        # Immediately kick out already evaluated pairs
        if not eval_pair_mask[pair_i]:
            continue

        # Ignore collinear pairs and very bad fittings
        if (np.isnan(pair_rmse[pair_i])
            or pair_rmse[pair_i] > min_q):
            eval_pair_mask[pair_i] = False
            keep_pair_mask[pair_i] = False
            continue
        
        # Find equivalent orientations
        similar_pair_mask = pair_rmse == pair_rmse[pair_i]
        eval_pair_mask[similar_pair_mask] = False
        keep_pair_mask[similar_pair_mask] = False
        if np.sum(similar_pair_mask) > 1: # isolated pairs are ignored. Probably collinear
            min_mis_mag = np.min(pair_mis_mag[similar_pair_mask])
            min_indices = np.nonzero(pair_mis_mag[similar_pair_mask]
                                     <= min_mis_mag + near_angle)[0] # some wiggle room

            keep_pair_mask[np.nonzero(similar_pair_mask)[0][min_indices]] = True
        
    return connection_pairs[keep_pair_mask]


def pair_casting(connection_pair,
                 all_spot_qs,
                 all_spot_ints,
                 all_ref_qs,
                 all_ref_fs,
                 qmask,
                 near_q,
                 keep_initial_pair=False,
                 iter_max=50):

    prev_connection = connection_pair.copy()
    (pair_spot_inds,
     pair_ref_inds) = _get_connection_indices(connection_pair)
        
    kdtree = KDTree(all_spot_qs)

    iter_count = 0
    while True:
        connection = connection_pair.copy()
        recalc_flag = False
        if not keep_initial_pair:
            connection[:] = np.nan # blank the connection

        # Find orientation and rotate reference lattice
        (conn_spots,
         conn_refs) = _decompose_connection(prev_connection,
                                            all_spot_qs,
                                            all_ref_qs)
        orientation, _ = Rotation.align_vectors(conn_spots,
                                                conn_refs)
        all_rot_qs = orientation.apply(all_ref_qs, inverse=False)
        temp_qmask = qmask.generate(all_rot_qs)

        # Query kdtree for find closest reference lattice points
        pot_conn = kdtree.query_ball_point(all_rot_qs[temp_qmask],
                                            r=near_q)
        
        if keep_initial_pair:
            # Remove original pair reflections from pot_conn
            for ind in pair_ref_inds:
                if ind in np.nonzero(temp_qmask)[0]:
                    pot_conn[np.nonzero(np.nonzero(temp_qmask)[0] == ind)[0][0]] = []
        
        # Cast and fill into blank connection
        multi = []
        for conn_i, conn in enumerate(pot_conn):
            if len(conn) > 0:
                if keep_initial_pair:
                    # Remove reflections near original pair
                    for ind in pair_spot_inds:
                        if ind in conn:
                            conn.remove(ind)
                if len(conn) == 0:
                    continue
                elif len(conn) == 1:
                    # Add candidate reflection
                    connection[conn[0]] = np.nonzero(temp_qmask)[0][conn_i]
                else:
                    # Add closest of multiple candidate reflections
                    _, spot_idx = kdtree.query(all_rot_qs[temp_qmask][conn_i])
                    connection[spot_idx] = np.nonzero(temp_qmask)[0][conn_i]
                    multi += conn # concatenate lists
        multi = tuple(np.unique(multi))
        
        # Compare connection with previous connection
        curr_spot_inds, curr_ref_inds = _get_connection_indices(connection)
        prev_spot_inds, prev_ref_inds = _get_connection_indices(prev_connection)

        if len(curr_spot_inds) == len(prev_spot_inds):
            if (np.all(curr_spot_inds == prev_spot_inds)
                and np.all (curr_ref_inds == prev_ref_inds)):
                break

        # Kick out any casting that is orientationally indeterminant
        if len(curr_spot_inds) < 2:
            # Revert to previous solution
            connection = prev_connection.copy()
            curr_spot_inds, curr_ref_inds = _get_connection_indices(connection)
            break

        # Prepare for next iteration
        prev_connection = connection.copy()
        iter_count += 1
        if iter_count >= iter_max:
            # Re-update orientation
            conn_spots = all_spot_qs[curr_spot_inds]
            conn_refs = all_ref_qs[curr_ref_inds]
            orientation, _ = Rotation.align_vectors(conn_spots,
                                                    conn_refs)
            # print('Max iterations reached in pair casting.')
            # print(f'{connection_pair}')
            break
    
    if np.sum(~np.isnan(connection)) == 0:
        print('Casting found empty connection')
        connection = connection_pair
    
    # Find qof
    all_rot_qs = orientation.apply(all_ref_qs, inverse=False)
    temp_qmask = qmask.generate(all_rot_qs)

    if np.sum(temp_qmask) == 0:
        # print('Empty qmask in pair casting')
        qof = 0
    else:
        qof = get_quality_of_fit(
            all_spot_qs[curr_spot_inds], # fit_spot_qs
            all_spot_ints[curr_spot_inds], # fit_spot_ints
            all_rot_qs[curr_ref_inds], # fit_rot_qs
            all_ref_fs[curr_ref_inds], # fit_ref_fs
            all_spot_qs, # all_spot_qs
            all_spot_ints, # all_spot_ints
            all_rot_qs[temp_qmask], # all_rot_qs
            all_ref_fs[temp_qmask], # all_ref_fs
            sigma=near_q)
    
    return connection, qof, multi


def multiple_pair_casting(connection_pairs,
                          all_spot_qs,
                          all_spot_ints,
                          all_ref_qs,
                          all_ref_fs,
                          qmask,
                          near_q,
                          iter_max=50,
                          keep_initial_pair=False,
                          exclude_found_pairs=False,
                          sort_results=True,
                          verbose=True):

    # Modify and set up some values
    connection_pairs = np.asarray(connection_pairs)
    evaluated_pair_mask = np.array([False,] * len(connection_pairs))
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

    connections = []
    qofs = []
    multi_reflections = []
    if verbose:
        print('Casting valid pairs...')
    for i, pair in iterate(connection_pairs):
        # Check if the pair has already been included
        if (exclude_found_pairs
            and evaluated_pair_mask[i]):
            continue

        connection, qof, multi = pair_casting(
                                    pair,
                                    all_spot_qs,
                                    all_spot_ints,
                                    all_ref_qs,
                                    all_ref_fs,
                                    qmask,
                                    near_q,
                                    keep_initial_pair=keep_initial_pair,
                                    iter_max=iter_max)

        connections.append(connection)
        qofs.append(qof)
        multi_reflections.append(multi)

        # Find and exclude included pairs
        if exclude_found_pairs:
            evaluated_pair_mask[i] = True # Should be redundant

            if np.sum(~np.isnan(connection)) > 2:
                found_pair_mask = (
                    np.sum([connection_pairs[:, si] == ri
                    for si, ri in zip(
                        *_get_connection_indices(connection))],
                    axis=0) >= 2)
                # print(f'Removing {np.sum(~evaluated_pair_mask[found_pair_mask])} evaluated pairs')   
                evaluated_pair_mask[found_pair_mask] = True
    
    # Sort by qof
    if sort_results:
        connections = [x for _, x in sorted(zip(qofs, connections),
                                            key=lambda pair: pair[0],
                                            reverse=True)]
        multi_reflections = [x for _, x in sorted(zip(qofs, multi_reflections),
                                                key=lambda pair: pair[0],
                                                reverse=True)]
        qofs = sorted(qofs, reverse=True)
                
    return connections, np.asarray(qofs), multi_reflections


# Alias for backwards compatibility
pair_casting_indexing = multiple_pair_casting


# def pair_voting(connection_pairs,
#                 approximate_max_pairs=10):

#     # Creat blank to fill
#     blank_connection = connection_pairs[0].copy()
#     blank_connection[:] = np.nan

#     # Convert pairs in sorted list of votes. List comprehension is a bit too messy
#     spot_votes_list = []
#     all_votes_list = []
#     for i in range(np.asarray(connection_pairs).shape[1]):
#         spot_votes = Counter(red_pairs[:, i][~np.isnan(red_pairs[:, i])].astype(int))
#         spot_votes = np.asarray(list(spot_votes.items()))
#         spot_votes_list.append(spot_votes)
#         if len(spot_votes) > 0:
#             all_votes_list.extend(spot_votes[:, 1])
    
#     # Determine number of popular spots to use
#     for num_spots in range(2, 16):
#         if scipy.special.comb(num_spots, 2) > approximate_max_pairs:
#             num_spots -= 1 # Went too far
#             break

#     # Determine most popular spot indexing
#     # Equally popular spots will inflate the number of spots used
#     vote_num_cutoff = sorted(all_votes_list, reverse=True)[num_spots]

#     # Build list of single spots
#     single_spots = []
#     for spot_i, spot_votes in enumerate(spot_votes_list):
#         for votes in spot_votes:
#             if votes[1] >= vote_num_cutoff:
#                 single_spots.append((spot_i, votes[0]))
    
#     # Combine spots into indexable pairs
#     popular_pairs = []
#     for combo in combinations(range(len(single_spots)), 2):
#         # Check if they are the same spot
#         if single_spots[combo[0]][0] != single_spots[combo[1]][0]:
#             pop_pair = blank_connection.copy()
#             pop_pair[single_spots[combo[0]][0]] = single_spots[combo[0]][1]
#             pop_pair[single_spots[combo[1]][0]] = single_spots[combo[1]][1]
#             popular_pairs.append(pop_pair)

#     return popular_pairs


# Works best with small near_q values. Time saving is thus limited
# def pair_voting_indexing(connection_pairs,
#                          all_spot_qs,
#                          all_spot_ints,
#                          all_ref_qs,
#                          all_ref_fs,
#                          qmask,
#                          near_q,
#                          keep_initial_pair=False,
#                          iter_max=50,
#                          approximate_max_pairs=10):

#     popular_pairs = pair_voting(connection_pairs,
#                         approximate_max_pairs=approximate_max_pairs)
    
#     # Index most popular pairs
#     connections, qofs, _ = pair_casting_indexing(
#                             popular_pairs,
#                             all_spot_qs,
#                             all_spot_ints,
#                             phase.all_qs,
#                             phase.all_fs,
#                             qmask,
#                             near_q,
#                             iter_max=iter_max,
#                             keep_initial_pair=keep_initial_pair,
#                             exclude_found_pairs=False,
#                             verbose=False)
    
#     return connections[np.argmax(qofs)], qofs[np.argmax(qofs)]


###########################
### Iterative Functions ###
###########################

# Deprecated. Slow
# def iterative_pattern_decomposition(connection_pairs,
#                                     all_spot_qs,
#                                     all_spot_ints,
#                                     all_ref_qs,
#                                     all_ref_fs,
#                                     qmask,
#                                     near_q,
#                                     keep_initial_pair=False,
#                                     max_ori_refine_iter=50,
#                                     max_ori_decomp_count=20,
#                                     verbose=True):
    
#     best_connections = []
#     best_qofs = []
#     excluded_spot_indices = []
#     included_spot_mask = np.asarray([True,] * len(connection_pairs[0]))
#     blank_full_connection = np.asarray([np.nan,] * len(included_spot_mask))
#     current_pair_list = connection_pairs.copy()

#     iter_count = 0
#     ITERATE = True
#     while ITERATE:

#         # Evaluate all pairs
#         connections, qofs, _ = pair_casting_indexing(
#                                     current_pair_list,
#                                     all_spot_qs[included_spot_mask],
#                                     all_spot_ints,
#                                     all_ref_qs,
#                                     all_ref_fs,
#                                     qmask,
#                                     near_q,
#                                     iter_max=max_ori_refine_iter,
#                                     keep_initial_pair=keep_initial_pair,
#                                     verbose=verbose)
        
#         best_connection = connections[np.argmax(qofs)]

#         # Condition to catch catastrophic failures
#         if np.sum(~np.isnan(best_connection)) <= 1:
#             print('ERROR: All indexing failed!')
#             print('Returning previously successful indexing.')
#             return best_connections, best_qofs
        
#         # Expand best connection to reference all spots
#         full_best_connection = blank_full_connection.copy()
#         full_best_connection[included_spot_mask] = best_connection
#         best_connections.append(full_best_connection)
#         best_qofs.append(np.max(qofs))

#         # Update connections
#         full_spot_inds, full_ref_inds = _get_connection_indices(full_best_connection)
#         excluded_spot_indices.extend(full_spot_inds)
#         included_spot_mask[excluded_spot_indices] = False

#         # Remove pairs where spots have already been indexed
#         new_pairs = []
#         curr_spot_inds, curr_ref_inds = _get_connection_indices(best_connection)
#         for pair in current_pair_list:
#             # All nan means the pair does not use any of the excluded indices
#             if np.all([np.isnan(pair[index]) for index in curr_spot_inds]):
#                 new_pairs.append(pair[np.isnan(best_connection)]) # reversed??
#         current_pair_list = np.asarray(new_pairs)

#         # Conditionals to kill iteration
#         iter_count += 1
#         if (len(all_spot_qs) - len(excluded_spot_indices) < 1 # Cannot solve orientations
#             or len(current_pair_list) < 1 # No more valid pairs to solve
#             or iter_count >= max_ori_decomp_count): # Reach maxed allowed orientations
#             ITERATE = False
#             break
    
#     return best_connections, np.asarray(best_qofs)


# More intelligent. Only re-evaluates pairs of connections which are no longer valid
def decaying_pattern_decomposition(start_connections,
                                   all_spot_qs,
                                   all_spot_ints,
                                   all_ref_qs,
                                   all_ref_fs,
                                   qmask,
                                   near_q,
                                   qof_minimum=0,
                                   keep_initial_pair=False,
                                   max_ori_refine_iter=50,
                                   max_ori_decomp_count=20,
                                   verbose=True):

    best_connections, best_qofs = [], []
    excluded_spot_indices = []
    included_conn_mask = np.asarray([True,] * len(start_connections))
    included_spot_mask = np.asarray([True,] * len(start_connections[0]))
    track_conns = np.asarray(start_connections).copy()

    # Internal wrapper for indexing method
    # Can redefine for other methods as desired
    def _internal_indexing(pairs,
                           spots,
                           verbose=verbose):
        out = multiple_pair_casting(
                    pairs,
                    spots,
                    all_spot_ints, # must always be full amount for proper comparison
                    all_ref_qs,
                    all_ref_fs,
                    qmask,
                    near_q,
                    sort_results=False,
                    iter_max=max_ori_refine_iter,
                    keep_initial_pair=keep_initial_pair,
                    verbose=verbose)
        return out
    
    (connections,
     qofs,
     multi_reflections) = _internal_indexing(
                                start_connections,
                                all_spot_qs)
    connections = np.asarray(connections)

    iter_count = 0
    while True:        
        # Find best connection
        best_ind = np.nanargmax(qofs)
        
        # Record best parameters
        best_connections.append(connections[best_ind].copy())
        best_qofs.append(qofs[best_ind])

        # Update connections
        spot_inds, ref_inds = _get_connection_indices(best_connections[-1])
        excluded_spot_indices.extend(spot_inds)

        # Record changes and remove already indexed spots
        changed_mask = np.any(~np.isnan(connections[:, excluded_spot_indices]), axis=1)
        track_conns[:, excluded_spot_indices] = np.nan
        included_spot_mask[excluded_spot_indices] = False

        # Determine which connections are still valid
        valid_mask = np.sum(~np.isnan(track_conns), axis=1) >= 2
        included_conn_mask[np.nonzero(included_conn_mask)[0][~valid_mask]] = False
        
        # Remove invalid connections, and determine which should be recalculated
        track_conns = track_conns[valid_mask]
        connections = connections[valid_mask]
        qofs = qofs[valid_mask]
        recalc_mask = changed_mask[valid_mask]

        # Recalculate as necessary
        if recalc_mask.sum() > 0:
            # Re-index connections
            (new_connections,
             new_qofs,
             new_multi_reflections) = _internal_indexing(
                # connections[np.ix_(recalc_mask, included_spot_mask)],
                start_connections[included_conn_mask][np.ix_(recalc_mask, included_spot_mask)],
                all_spot_qs[included_spot_mask],
                verbose=False)

            # Expand new connections
            full_new_connections = np.empty((recalc_mask.sum(),
                                            len(included_spot_mask)))
            full_new_connections[:] = np.nan
            full_new_connections[:, included_spot_mask] = new_connections

            # Update values
            connections[recalc_mask] = full_new_connections
            qofs[recalc_mask] = new_qofs

        # Conditionals to kill iteration
        iter_count += 1
        if (len(connections) < 1 # Nothing left to compare
            or len(all_spot_qs) - len(excluded_spot_indices) < 1 # Cannot solve orientations
            or qofs.max() < qof_minimum # Quality of fit has gotten too poor
            or iter_count >= max_ori_decomp_count): # Reach maxed allowed orientations
            break

    # Trim bad connections.
    best_qofs = np.asarray(best_qofs)
    if len(best_connections) > 1:
        # I don't like this, but I want to keep it as a list
        best_connections = list(np.asarray(best_connections)[best_qofs >= qof_minimum])
        best_qofs = best_qofs[best_qofs >= qof_minimum]
    else:
        if verbose and best_qofs.squeeze() < qof_minimum:
            warn_str = ('WARNING: Indexing quality '
                        + f'({best_qofs.squeeze():.4f}) below '
                        + f'designated minimum ({qof_minimum:.4f}). '
                        + 'Stopping indexing.')
            print(warn_str)
        
    return best_connections, best_qofs


#########################
### Utility Functions ###
#########################



def get_quality_of_fit(fit_spot_qs,
                       fit_spot_ints,
                       fit_rot_qs,
                       fit_ref_fs,
                       all_spot_qs,
                       all_spot_ints,
                       all_rot_qs,
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
    #                             all_spot_qs,
    #                             all_rot_qs,
    #                             all_ref_fs,
    #                             int_weight=0.5)

    # qof = complete_distance_qof(fit_spot_qs,
    #                             fit_rot_qs,
    #                             all_spot_qs,
    #                             all_rot_qs,
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

    # qof = len(fit_spot_qs) / len(all_spot_qs)

    return qof


def weighted_distance_qof(fit_spot_qs,
                          fit_rot_qs,
                          fit_ref_fs,
                          all_spot_qs, # unused
                          all_rot_qs,
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
    # norm_dist_val = dist_val / len(all_spot_qs)
    norm_dist_val = dist_val / len(all_rot_qs)

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
                          all_spot_qs,
                          all_rot_qs,
                          sigma=1,
                          int_weight=0.5): # Gaussian standard deviation to evaluate


    # Determine which reflections are indexed
    # found_spot_mask = [tuple(ref) in [tuple(x) for x in ref_qs] for ref in all_ref_qs]

    dist = [np.sqrt(np.sum([(p - q)**2 for p, q in zip(v1, v2)]))
            for v1, v2 in zip(fit_spot_qs, fit_rot_qs)]

    # Gaussian with specified standard deviation
    # centered at zero sampled at distance
    qof = np.sum(np.exp(-(np.asarray(dist))**2 / (2 * sigma**2)))

    max_qof = len(all_rot_qs)
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
                 ratio=0.75):


    dist = [np.sqrt(np.sum([(p - q)**2 for p, q in zip(v1, v2)]))
            for v1, v2 in zip(fit_spot_qs, fit_rot_qs)]
    
    # Gaussian with specified standard deviation
    # centered at zero sampled at distance
    gauss_int = fit_spot_ints * np.exp(-(np.asarray(dist))**2 / (2 * sigma**2))

    exp_val = np.sum(gauss_int) / np.sum(all_spot_ints)
    ref_val = np.sum(fit_ref_fs) / np.sum(all_ref_fs)

    return (ratio * exp_val) + ((1 - ratio) * ref_val)



    

def get_rmse(fit_spot_qs,
             fit_rot_qs):
    rmse = np.mean([np.sqrt(np.sum([(p - q)**2 for p, q in zip(v1, v2)]))
                    for v1, v2 in zip(fit_spot_qs, fit_rot_qs)])

    return rmse


def _get_connection_indices(connection):

    connection = np.asarray(connection)
    spot_indices = np.nonzero(~np.isnan(connection))[0]
    ref_indices = connection[spot_indices].astype(int)

    return spot_indices, ref_indices


def _decompose_connection(connection,
                          all_spot_qs,
                          all_ref_qs):

    all_spot_qs = np.asarray(all_spot_qs)
    all_ref_qs = np.asarray(all_ref_qs)

    (spot_indices,
    ref_indices) = _get_connection_indices(connection)

    conn_spots = all_spot_qs[spot_indices]
    conn_refs = all_ref_qs[ref_indices]

    return conn_spots, conn_refs

