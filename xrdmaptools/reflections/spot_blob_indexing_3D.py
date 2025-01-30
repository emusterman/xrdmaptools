import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from scipy.spatial import distance_matrix, KDTree
from scipy.optimize import curve_fit
from itertools import combinations, product

from xrdmaptools.utilities.math import (
    multi_vector_angles,
    general_polynomial
)
from xrdmaptools.geometry.geometry import (
    q_2_polar,
    modular_azimuthal_shift
)

##########################
### Combined Functions ###
##########################

def pair_casting_index_best_grain(
                    all_spot_qs,
                    phase,
                    near_q,
                    near_angle,
                    qmask,
                    degrees=False,
                    max_ori_refine_iter=50,
                    max_ori_decomp_count=20,
                    keep_initial_pair=False):
    
    # Find q vector magnitudes and max for spots
    spot_q_mags = np.linalg.norm(all_spot_qs, axis=1)
    max_q = np.max(spot_q_mags)

    # Find phase reciprocal lattice
    phase.generate_reciprocal_lattice(1.15 * max_q)
    all_ref_qs = phase.all_qs
    all_ref_fs = phase.all_fs
    all_ref_hkls = phase.all_hkls

    # Find minimum q vector step size from reference phase
    min_q = np.min(np.linalg.norm(phase.Q([[1, 0, 0],
                                           [0, 1, 0],
                                           [0, 0, 1]]),
                                           axis=0))

    # Find all valid pairs within near_q and near_angle
    pairs = find_all_valid_pairs(
            all_spot_qs,
            phase.all_qs,
            near_q,
            near_angle,
            min_q,
            degrees=degrees)
    if len(pairs) > 0:
        # Symmetrically reduce pairs
        red_pairs = reduce_symmetric_equivalents(
                pairs,
                all_spot_qs,
                phase.all_qs,
                phase.all_hkls,
                near_angle,
                min_q)
        
        # Index spots
        connections, qofs, _ = pair_casting_indexing(
                red_pairs,
                all_spot_qs,
                phase.all_qs,
                phase.all_fs,
                qmask,
                near_q,
                iter_max=max_ori_refine_iter,
                keep_initial_pair=keep_initial_pair,
                exclude_found_pairs=False,
                verbose_iterator=True)
    else:
        best_connections = [[np.nan,] * len(all_spot_qs)]
        best_qofs = [np.nan,] * len(all_spot_qs)

    return connections[np.argmax(qofs)], qofs[np.argmax(qofs)]


def pair_casting_index_full_pattern(
                    all_spot_qs,
                    phase,
                    near_q,
                    near_angle,
                    qmask,
                    degrees=False,
                    qof_minimum=0.2,
                    max_ori_refine_iter=50,
                    max_ori_decomp_count=20,
                    keep_initial_pair=False):
    
    # Find q vector magnitudes and max for spots
    spot_q_mags = np.linalg.norm(all_spot_qs, axis=1)
    max_q = np.max(spot_q_mags)

    # Find phase reciprocal lattice
    phase.generate_reciprocal_lattice(1.15 * max_q)
    all_ref_qs = phase.all_qs
    all_ref_fs = phase.all_fs
    all_ref_hkls = phase.all_hkls

    # Find minimum q vector step size from reference phase
    min_q = np.min(np.linalg.norm(phase.Q([[1, 0, 0],
                                           [0, 1, 0],
                                           [0, 0, 1]]),
                                           axis=0))

    # Find all valid pairs within near_q and near_angle
    pairs = find_all_valid_pairs(
            all_spot_qs,
            phase.all_qs,
            near_q,
            near_angle,
            min_q,
            degrees=degrees)
    
    if len(pairs) > 0:
        # Symmetrically reduce pairs
        red_pairs = reduce_symmetric_equivalents(
                pairs,
                all_spot_qs,
                phase.all_qs,
                phase.all_hkls,
                near_angle,
                min_q)
        
        # Iteratively decompose patterns
        best_connections, best_qofs = decaying_pattern_decomposition(
                red_pairs,
                all_spot_qs,
                phase.all_qs,
                phase.all_fs,
                qmask,
                near_q,
                qof_minimum=qof_minimum,
                keep_initial_pair=keep_initial_pair,
                max_ori_refine_iter=max_ori_refine_iter,
                max_ori_decomp_count=max_ori_decomp_count)
    else:
        best_connections = [[np.nan,] * len(all_spot_qs)]
        best_qofs = [np.nan,] * len(all_spot_qs)

    return best_connections, best_qofs


#####################
### Sub-Functions ###
#####################


def find_all_valid_pairs(all_spot_qs,
                         all_ref_qs,
                         near_q,
                         near_angle,
                         min_q,
                         degrees=False):
    
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

    blank_connection = np.array([np.nan,] * len(phase_mask))
    connection_pairs = []
    for pair in tqdm(spot_pair_indices):
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


def reduce_symmetric_equivalents(connection_pairs,
                                 all_spot_qs,
                                 all_ref_qs,
                                 all_ref_hkls,
                                 near_angle,
                                 min_q):

    # Convert to arrays
    all_spot_qs = np.asarray(all_spot_qs)
    all_ref_qs = np.asarray(all_ref_qs)
    all_ref_hkls = np.asarray(all_ref_hkls)
    
    # Evaluating valid pairs
    pair_orientations = []
    pair_mis_mag = []
    pair_rmse = []
    for pair_connection in tqdm(connection_pairs):
        (spot_indices,
        ref_indices) = _get_connection_indices(pair_connection)
        pair_ref_hkls = all_ref_hkls[ref_indices]
        pair_ref_qs = all_ref_qs[ref_indices]
        pair_spot_qs = all_spot_qs[spot_indices]

        # Check colinearity.
        # 3D orientation cannot be determined from colinear pairs
        pair_divs = pair_ref_hkls[0] / pair_ref_hkls[1]
        if len(np.unique(pair_divs[~np.isnan(pair_divs)])) < 2:
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
        pair_mis_mag.append(np.degrees(orientation.magnitude()))
        pair_rmse.append(rmse)
    
    pair_mis_mag = np.asarray(pair_mis_mag)
    pair_rmse = np.asarray(pair_rmse).round(10)

    # Reducing symmetrically equivalent pairs
    eval_pair_mask = np.array([True,] * len(connection_pairs))
    keep_pair_mask = eval_pair_mask.copy()
    for pair_i in tqdm(range(len(connection_pairs))):
        # Immediately kick out already evaluated pairs
        if not eval_pair_mask[pair_i]:
            continue

        # Cannot symmetrically reduce colinear pairs
        # And remove really bad fitting
        if (np.isnan(pair_rmse[pair_i])
            or pair_rmse[pair_i] > min_q):
            eval_pair_mask[pair_i] = False
            keep_pair_mask[pair_i] = False # colinear pairs are not useful for casting
            continue
        
        # Find equivalent orientations
        similar_pair_mask = pair_rmse == pair_rmse[pair_i]
        eval_pair_mask[similar_pair_mask] = False
        keep_pair_mask[similar_pair_mask] = False
        if np.sum(similar_pair_mask) > 1: # isolated pairs are ignored. probably colinear
            min_mis_mag = np.min(pair_mis_mag[similar_pair_mask])
            min_indices = np.nonzero(pair_mis_mag[similar_pair_mask]
                                     < min_mis_mag + near_angle)[0] # some wiggle room

            keep_pair_mask[np.nonzero(similar_pair_mask)[0][min_indices]] = True
        
    return connection_pairs[keep_pair_mask]


def pair_casting_indexing(connection_pairs,
                          all_spot_qs,
                          all_ref_qs,
                          all_ref_fs,
                          qmask,
                          near_q,
                          iter_max=50,
                          keep_initial_pair=False,
                          exclude_found_pairs=False,
                          verbose_iterator=True):

    # Modify and set up some values
    connection_pairs = np.asarray(connection_pairs)
    evaluated_pair_mask = np.array([False,] * len(connection_pairs))
    all_spot_qs = np.asarray(all_spot_qs)
    all_ref_qs = np.asarray(all_ref_qs)
    all_ref_fs = np.asarray(all_ref_fs)

    first_best = True
    bad_indices = []

    if verbose_iterator:
        iterated = tqdm(enumerate(connection_pairs),
                        total=len(connection_pairs))
    else:
        iterated = enumerate(connection_pairs)

    connections = []
    qofs = []
    multi_reflections = []
    evaluated_pairs = 0
    for i, pair in iterated:
        
        # Check if the pair has already be included
        if (exclude_found_pairs
            and evaluated_pair_mask[i]):
            continue

        if i in bad_indices:
            print('Bad index evaluated')

        prev_connection = pair.copy()
        (pair_spot_inds,
         pair_ref_inds) = _get_connection_indices(pair)
         
        kdtree = KDTree(all_spot_qs)

        iter_count = 0
        ITERATE = True
        while ITERATE:
            connection = pair.copy()
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
                        #multi.append(spot_idx)
                        multi += conn # concatenate lists
                        #recalc_flag = True
            multi = tuple(np.unique(multi))
            
            # Compare connection with previous connection
            curr_spot_inds, curr_ref_inds = _get_connection_indices(connection)
            prev_spot_inds, prev_ref_inds = _get_connection_indices(prev_connection)

            if len(curr_spot_inds) == len(prev_spot_inds):
                if (np.all(curr_spot_inds == prev_spot_inds)
                    and np.all (curr_ref_inds == prev_ref_inds)):
                    ITERATE = False
                    break

            # Kick out any casting that is orientationally indeterminant
            if len(curr_spot_inds) < 2:
                # Revert to previous solution
                connection = prev_connection.copy()
                ITERATE = False
                break

            # Prepare for next iteration
            prev_connection = connection.copy()
            iter_count += 1
            if iter_count >= iter_max:
                ITERATE = False
                # Re-update orientation
                conn_spots = all_spot_qs[curr_spot_inds]
                conn_refs = all_ref_qs[curr_ref_inds]
                orientation, _ = Rotation.align_vectors(conn_spots,
                                                        conn_refs)
        
        # Find qof
        all_rot_qs = orientation.apply(all_ref_qs, inverse=False)
        temp_qmask = qmask.generate(all_rot_qs)
        qof = get_quality_of_fit(
                    all_spot_qs[curr_spot_inds],
                    all_rot_qs[curr_ref_inds],
                    all_ref_fs[curr_ref_inds],
                    all_spot_qs, # Already within qmask
                    all_rot_qs[temp_qmask],
                    all_ref_fs[temp_qmask],
                    sigma=near_q)

        connections.append(connection)
        qofs.append(qof)
        multi_reflections.append(multi)
        evaluated_pairs += 1

        # Find and exclude included pairs
        if exclude_found_pairs:
            evaluated_pair_mask[i] = True # Should be redundant
            if len(curr_spot_inds) > 2:
                found_pair_mask = (np.sum([connection_pairs[:, si] == ri
                                   for si, ri in zip(*_get_connection_indices(connection))],
                                   axis=0) >= 2)
                # print(evaluated_pair_mask.shape)
                # print(found_pair_mask.shape)
                print(f'{len(curr_spot_inds)=}', f'{found_pair_mask.sum()=}')
                evaluated_pair_mask[found_pair_mask] = True
            if len(curr_spot_inds) > 9 and first_best:
                first_best = False
                bad_indices = np.nonzero(found_pair_mask)[0]
            
    # print(evaluated_pairs)
    return connections, np.asarray(qofs), multi_reflections


# Deprecated. Slow
def iterative_pattern_decomposition(connection_pairs,
                                    all_spot_qs,
                                    all_ref_qs,
                                    all_ref_fs,
                                    qmask,
                                    near_q,
                                    keep_initial_pair=False,
                                    max_ori_refine_iter=50,
                                    max_ori_decomp_count=20):
    
    best_connections = []
    best_qofs = []
    excluded_spot_indices = []
    included_spot_mask = np.asarray([True,] * len(connection_pairs[0]))
    blank_full_connection = np.asarray([np.nan,] * len(included_spot_mask))
    current_pair_list = connection_pairs.copy()

    iter_count = 0
    ITERATE = True
    while ITERATE:

        # Evaluate all pairs
        connections, qofs, _ = pair_casting_indexing(
                                    current_pair_list,
                                    all_spot_qs[included_spot_mask],
                                    all_ref_qs,
                                    all_ref_fs,
                                    qmask,
                                    near_q,
                                    iter_max=max_ori_refine_iter,
                                    keep_initial_pair=keep_initial_pair)
        
        best_connection = connections[np.argmax(qofs)]

        # Condition to catch catastrophic failures
        if np.sum(~np.isnan(best_connection)) <= 1:
            print('ERROR: All indexing failed!')
            print('Returning previously successful indexing.')
            return best_connections, best_qofs
        
        # Expand best connection to reference all spots
        full_best_connection = blank_full_connection.copy()
        full_best_connection[included_spot_mask] = best_connection
        best_connections.append(full_best_connection)
        best_qofs.append(np.max(qofs))

        # Update connections
        full_spot_inds, full_ref_inds = _get_connection_indices(full_best_connection)
        excluded_spot_indices.extend(full_spot_inds)
        included_spot_mask[excluded_spot_indices] = False

        # Remove pairs where spots have already been indexed
        new_pairs = []
        curr_spot_inds, curr_ref_inds = _get_connection_indices(best_connection)
        for pair in current_pair_list:
            # All nan means the pair does not use any of the excluded indices
            if np.all([np.isnan(pair[index]) for index in curr_spot_inds]):
                new_pairs.append(pair[np.isnan(best_connection)]) # reversed??
        current_pair_list = np.asarray(new_pairs)

        # Conditionals to kill iteration
        iter_count += 1
        if (len(all_spot_qs) - len(excluded_spot_indices) < 1 # Cannot solve orientations
            or len(current_pair_list) < 1 # No more valid pairs to solve
            or iter_count >= max_ori_decomp_count): # Reach maxed allowed orientations
            ITERATE = False
            break
    
    return best_connections, np.asarray(best_qofs)


def decaying_pattern_decomposition(connection_pairs,
                                   all_spot_qs,
                                   all_ref_qs,
                                   all_ref_fs,
                                   qmask,
                                   near_q,
                                   qof_minimum=0,
                                   keep_initial_pair=False,
                                   max_ori_refine_iter=50,
                                   max_ori_decomp_count=20):

    best_connections, best_qofs = [], []
    excluded_spot_indices = []
    included_conn_mask = np.asarray([True,] * len(connection_pairs))
    included_spot_mask = np.asarray([True,] * len(connection_pairs[0]))
    blank_full_connection = np.asarray([np.nan,] * len(included_spot_mask))

    def _internal_indexing(pairs,
                           spots,
                           verbose_iterator=True):
        out = pair_casting_indexing(
                    pairs,
                    spots,
                    all_ref_qs,
                    all_ref_fs,
                    qmask,
                    near_q,
                    iter_max=max_ori_refine_iter,
                    keep_initial_pair=keep_initial_pair,
                    verbose_iterator=verbose_iterator)
        return out
    
    (orig_connections,
    orig_qofs,
    orig_multi_reflections) = _internal_indexing(
                                connection_pairs,
                                all_spot_qs)

    connections = np.asarray(orig_connections.copy())
    qofs = np.asarray(orig_qofs.copy())
    multi_reflections = orig_multi_reflections.copy()

    iter_count = 0
    ITERATE = True
    while ITERATE:
        
        # Find best connection
        best_ind = np.nanargmax(qofs[included_conn_mask]) # Should not be nan???
        best_connection = connections[included_conn_mask][best_ind]
        best_qof = qofs[included_conn_mask][best_ind]

        # Conditional to catch catastrophic failures
        if np.sum(~np.isnan(best_connection)) <= 1:
            print('ERROR: All indexing failed!')
            print('Returning previously successful indexing.')
            print('override: returning connections, qofs, included_conn_mask, best_connections, best_qofs')
            return connections, qofs, included_conn_mask, best_connections, np.asarray(best_qofs)
            return best_connections, np.asarray(best_qofs)
        
        # Record best parameters
        best_connections.append(best_connection)
        best_qofs.append(best_qof)

        # Update connections
        full_spot_inds, full_ref_inds = _get_connection_indices(best_connection)
        excluded_spot_indices.extend(full_spot_inds)
        included_spot_mask[excluded_spot_indices] = False

        # Update masks for different types of results
        valid_conn_mask = ~np.any([~np.isnan(connections[:, idx])
                                   for idx in excluded_spot_indices],
                                   axis=0)
        valid_pair_mask = ~np.any([~np.isnan(connection_pairs[:, idx])
                                   for idx in excluded_spot_indices],
                                   axis=0)

        exclude_ambig_num = np.array([np.sum([ind in multi
                                        for ind in full_spot_inds])
                                      for multi in multi_reflections])
        all_ambig_num = np.array([len(multi)
                                  for multi in multi_reflections])

        ambig_mask = ((exclude_ambig_num > 0) # Has an excluded index
                       & ((all_ambig_num
                           - exclude_ambig_num) > 0)) # And has other valid indices

        # Useful masks
        recalc_mask = ambig_mask & ~(~valid_pair_mask & ~valid_conn_mask)
        included_conn_mask = valid_conn_mask.copy()
        included_conn_mask[recalc_mask] = True

        # Conditionals to kill iteration
        iter_count += 1
        if (len(all_spot_qs) - len(excluded_spot_indices) < 1 # Cannot solve orientations
            or best_qof < qof_minimum
            or included_conn_mask.sum() < 1
            or iter_count >= max_ori_decomp_count): # Reach maxed allowed orientations
            ITERATE = False
            break

        # Evaluate new pairs as needed
        if recalc_mask.sum() > 0:
            # Down-cast connections
            new_pairs = []
            # for pair in connection_pairs[recalc_mask]:
            #     new_pairs.append(pair[included_spot_mask])
            # new_pairs = np.asarray(new_pairs)

            for idx in range(len(connection_pairs)):
                if recalc_mask[idx]:
                    if (~valid_pair_mask & valid_conn_mask)[idx]:
                        # Original pair is invalid, but connection is valid.
                        # Append connection
                        new_pairs.append(connections[idx][included_spot_mask])
                    else:
                        # Append original pair
                        new_pairs.append(connection_pairs[idx][included_spot_mask])

            # Re-index connections
            (new_connections,
            new_qofs,
            new_multi_reflections) = _internal_indexing(
                                        new_pairs,
                                        all_spot_qs[included_spot_mask],
                                        verbose_iterator=False)
            
            # Expand new connections
            full_new_connections = []
            for conn in new_connections:
                full_new_connection = blank_full_connection.copy()
                full_new_connection[included_spot_mask] = conn
                full_new_connections.append(full_new_connection)
            
            if np.any(np.array([np.sum(~np.isnan(conn))
                                for conn in full_new_connections]) < 2):
                print('Found error.')
                return (recalc_mask,
                        new_pairs,
                        included_spot_mask,
                        included_conn_mask,
                        full_new_connections,
                        new_qofs)
            
            # Update values
            connections[recalc_mask] = full_new_connections
            qofs[recalc_mask] = new_qofs
            # multi_reflections cannot be converted to array
            for i, idx in enumerate(np.nonzero(recalc_mask)[0]):
                multi_reflections[idx] = new_multi_reflections[i]
    
    # Trim bad connections. May be worth keeping since they have 
    # already been calculated.
    best_qofs = np.asarray(best_qofs)
    if len(best_connections) > 1:
        # I don't like this, but I want to keep it as a list
        best_connections = list(np.asarray(best_connections)[best_qofs >= qof_minimum])
        # best_connections = [best_connections[idx] if best_qofs[idx] >= qof_minimum for idx in range(len(best_connections))]
        best_qofs = best_qofs[best_qofs >= qof_minimum]
    else:
        if best_qofs.squeeze() < qof_minimum:
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
                       fit_rot_qs,
                       fit_ref_fs,
                       all_spot_qs,
                       all_rot_qs,
                       all_ref_fs,
                       **kwargs):

    # Ideal conditions
    # 1. Penalize missing reference reflections weighted according to their expected intensity
    # 2. Do not penalize extra measured reflections which are not indexing (allows for overlapping orientations)
    # 3. Penalize reflections weighted by their distance from expected postions

    # qof = get_rmse(all_spot_qs[fit_spot_inds],
    #                  all_rot_qs[fit_ref_inds])
    qof = weighted_distance_qof(fit_spot_qs,
                                fit_rot_qs,
                                fit_ref_fs,
                                all_spot_qs,
                                all_rot_qs,
                                all_ref_fs,
                                **kwargs)

    return qof


def weighted_distance_qof(fit_spot_qs,
                          fit_rot_qs,
                          fit_ref_fs,
                          all_spot_qs,
                          all_rot_qs,
                          all_ref_fs,
                          sigma=1,
                          int_weight=0):

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

    return qof


def complete_distance_qof(fit_spot_qs,
                          fit_rot_qs,
                          all_spot_qs,
                          all_rot_qs,
                          sigma=1,
                          int_weight=0.5): # Gaussian standard deviation to evaluate

    # # Determine which reflections are indexed
    # found_spot_mask = [tupl                              returns=['xrd_dets'],

    return norm_qof
    

def get_rmse(fit_spot_qs,
             fit_rot_qs):
    rmse = np.mean([np.sqrt(np.sum([(p - q)**2 for p, q in zip(v1, v2)]))
                    for v1, v2 in zip(fit_spot_qs, fit_rot_qs)])

    return rmse


# Class for bounding reciprocal space volumes mapped by energy rocking curves
# Angle rocking curves are not currently supported, although they should be simpler
class QMask():

    def __init__(self,
                 tth_arr,
                 chi_arr,
                 wavelength_vals,
                 theta_vals=0,
                 poly_order=6,
                 degrees=False,
                 use_stage_rotation=False):
        
        # Check for azimuthal discontintuites
        chi_arr, max_arr, shifted = modular_azimuthal_shift(chi_arr)
        self.chi_max_arr = max_arr # determines units...
        self.chi_shifted = shifted # conditional

        # Determine simple extents
        self.tth_min = np.min(tth_arr)
        self.tth_max = np.max(tth_arr)
        self.chi_min = np.min(chi_arr)
        self.chi_max = np.max(chi_arr)
        self.wavelength_min = np.min(wavelength_vals)
        self.wavelength_max = np.max(wavelength_vals)
        self.theta_min = np.min(theta_vals)
        self.theta_max = np.max(theta_vals)
        self.degrees = degrees
        self.use_stage_rotation = use_stage_rotation

        # Get rocking axis
        energy_rc = self.wavelength_min != self.wavelength_max
        angle_rc = self.theta_min != self.theta_max

        if energy_rc and angle_rc:
            err_str = ('Both energy and angle are changing. '
                      + '\nOne must be static to generate Qmask.')
            raise RuntimeError(err_str)
        elif not energy_rc and not angle_rc:
            err_str = ('Neither energy or angle are changing. '
                      + '\nOne must be rock to generate Qmask.')
            raise RuntimeError(err_str)
        elif energy_rc:
            self.rocking_axis = 'energy'
        else:
            self.rocking_axis = 'angle'


        # Determine edges
        # Inelegant method of throwing higher
        # order polynomials at the problem
        p0 = np.zeros(poly_order + 1)
        for indexing, poly in zip([(0), # [0, :] 
                                   (-1), # [-1, :]
                                   (slice(None), 0), # [:, 0]
                                   (slice(None), -1)], # [:, -1]
                                   ['upper_poly',
                                   'lower_poly',
                                   'left_poly',
                                   'right_poly']):
            
            # Determine functional direction
            tth_grad = np.gradient(tth_arr[indexing])
            if np.all(tth_grad > 0) or np.all(tth_grad < 0):
                first_arr = tth_arr
                second_arr = chi_arr
                setattr(self, f'{poly}_first', 'tth')
            else:
                first_arr = chi_arr
                second_arr = tth_arr
                setattr(self, f'{poly}_first', 'chi')
            
            # Fit edge functions
            popt, _ = curve_fit(general_polynomial,
                                first_arr[indexing],
                                second_arr[indexing],
                                p0=p0)
            
            setattr(self, poly, popt)

    
    @classmethod
    def from_XRDRockingCurve(cls,
                             rsm,
                             **kwargs):
        
        inst = cls(rsm.tth_arr,
                   rsm.chi_arr,
                   rsm.wavelength,
                   rsm.theta,
                   degrees=rsm.polar_units == 'deg',
                   use_stage_rotation=rsm.use_stage_rotation,
                   **kwargs)
        
        return inst


    def generate(self,
                 q_vectors,
                 ext=0):

        # Convert vectors to polar
        if self.rocking_axis == 'energy':
            if self.use_stage_rotation:
                theta = self.theta_min
            else:
                theta = 0

            tth, chi, wavelength = q_2_polar(q_vectors,
                                             stage_rotation=theta,
                                             degrees=self.degrees)

            rocking_mask = np.all([
                            wavelength >= self.wavelength_min * (1 - ext),
                            wavelength <= self.wavelength_max * (1 + ext)],
                            axis=0)
        
        else:
            tth, chi, rotation = q_2_polar(q_vectors,
                                           wavelength=self.wavelength_min,
                                           degrees=self.degrees)

            rocking_mask = np.all([
                            rotation >= self.theta_min * (1 - ext),
                            rotation <= self.theta_max * (1 + ext)],
                            axis=0)

        # Shift chi values if discontinuiteies
        chi, _, _ = modular_azimuthal_shift(
                            chi,
                            max_arr=self.chi_max_arr,
                            force_shift=self.chi_shifted)

        # Vertically bounded mask
        if getattr(self, 'upper_poly_first') == 'tth':
            first = tth
            second = chi
        else:
            first = chi
            second = tth

        upper = second - general_polynomial(first, *self.upper_poly)
        lower = second - general_polynomial(first, *self.lower_poly)
        vertical_mask = np.sign(upper) != np.sign(lower)
        
        # Horizontally bounded mask
        if getattr(self, 'left_poly_first') == 'tth':
            first = tth
            second = chi
        else:
            first = chi
            second = tth

        left = second - general_polynomial(first, *self.left_poly)
        right = second - general_polynomial(first, *self.right_poly)
        horizontal_mask = np.sign(left) != np.sign(right)
        
        # Extent masks
        tth_mask = np.all([tth >= self.tth_min * (1 - ext),
                           tth <= self.tth_max * (1 + ext)],
                           axis=0)
        chi_mask = np.all([chi >= self.chi_min * (1 - ext),
                           chi <= self.chi_max * (1 + ext)],
                           axis=0)
        
        qmask = np.all([
            vertical_mask,
            horizontal_mask,
            tth_mask,
            chi_mask,
            rocking_mask
        ], axis=0)

        return qmask


def _get_connection_indices(connection):

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

