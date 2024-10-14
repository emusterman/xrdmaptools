
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


# def mutli_vector_angles(v1s, v2s, degrees=False):
#     v1_units = v1s / np.linalg.norm(v1s, axis=1).reshape(-1, 1)
#     v1_units = v2s / np.linalg.norm(v2s, axis=1).reshape(-1, 1)

#     # Not happy about the round. This is not perfect...
#     angles = np.arccos(np.inner(v1_units, v2_units).round(6))

#     if degrees:
#         angles = np.degrees(angles)
    
#     return angles


def pair_casting_index_pattern(all_spot_qs,
                               phase,
                               near_q,
                               near_angle,
                               q_mask,
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
    connection_pair_list = find_all_valid_pairs(
            all_spot_qs,
            all_ref_qs,
            near_q,
            near_angle,
            degrees=degrees)

    # Symmetrically reduce pairs
    connection_pair_list = reduce_symmetric_equivalents(
            connection_pair_list,
            all_spot_qs,
            all_ref_qs,
            all_ref_hkls,
            near_angle,
            min_q)

    # Iteratively decompose patterns
    best_connections, best_qofs = iterative_pattern_decomposition(
            connection_pair_list,
            all_spot_qs,
            all_ref_qs,
            all_ref_fs,
            q_mask,
            near_q,
            keep_initial_pair=keep_initial_pair,
            max_ori_refine_iter=max_ori_refine_iter,
            max_ori_decomp_count=max_ori_decomp_count)

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
    connection_pair_list = []
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
                connection_pair_list.append(connection)

    return np.asarray(connection_pair_list)


def reduce_symmetric_equivalents(connection_pair_list,
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
    for pair_connection in tqdm(connection_pair_list):
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
    eval_pair_mask = np.array([True,] * len(connection_pair_list))
    keep_pair_mask = eval_pair_mask.copy()
    for pair_i in tqdm(range(len(connection_pair_list))):
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
        
    return connection_pair_list[keep_pair_mask]


def pair_casting_indexing(connection_pair_list,
                          all_spot_qs,
                          all_ref_qs,
                          all_ref_fs,
                          q_mask,
                          near_q,
                          iter_max=50,
                          keep_initial_pair=False):

    all_spot_qs = np.asarray(all_spot_qs)
    all_ref_qs = np.asarray(all_ref_qs)
    all_ref_fs = np.asarray(all_ref_fs)

    connections = []
    qofs = []
    for i, pair in tqdm(enumerate(connection_pair_list),
                        total=len(connection_pair_list)):

        prev_connection = pair.copy()
        (pair_spot_inds,
         pair_ref_inds) = _get_connection_indices(pair)
         
        kdtree = KDTree(all_spot_qs)

        iter_count = 0
        ITERATE = True
        while ITERATE:
            connection = pair.copy()
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
            temp_q_mask = q_mask.generate(all_rot_qs)

            # Build kdtree from measured spots and query to referene
            # lattice to avoids non-crystallographic indexing
            #kdtree = KDTree(conn_spots)
            pot_conn = kdtree.query_ball_point(all_rot_qs[temp_q_mask],
                                               r=near_q)
            
            if keep_initial_pair:
                # Remove original pair reflections
                for ind in pair_ref_inds:
                    if ind in np.nonzero(temp_q_mask)[0]:
                        pot_conn[np.nonzero(np.nonzero(temp_q_mask)[0] == ind)[0][0]] = []
            
            # Cast and expand connection
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
                        connection[conn[0]] = np.nonzero(temp_q_mask)[0][conn_i]
                    else:
                        # Add closest of multiple candidate reflections
                        _, ref_idx = kdtree.query(all_rot_qs[temp_q_mask][conn_i])
                        connection[ref_idx] = np.nonzero(temp_q_mask)[0][conn_i]
            
            # Compare connection with previous connection
            curr_spot_inds, curr_ref_inds = _get_connection_indices(connection)
            prev_spot_inds, prev_ref_inds = _get_connection_indices(prev_connection)

            if len(curr_spot_inds) == len(prev_spot_inds):
                if (np.all(curr_spot_inds == prev_spot_inds)
                    and np.all (curr_ref_inds == prev_ref_inds)):
                    ITERATE = False

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
        temp_q_mask = q_mask.generate(all_rot_qs)
        qof = get_quality_of_fit(
                    all_spot_qs[curr_spot_inds],
                    all_rot_qs[curr_ref_inds],
                    all_ref_fs[curr_ref_inds],
                    all_spot_qs, # Already within q_mask
                    all_rot_qs[temp_q_mask],
                    all_ref_fs[temp_q_mask],
                    sigma=near_q)

        connections.append(connection)
        qofs.append(qof)

    return connections, qofs


def iterative_pattern_decomposition(connection_pair_list,
                                    all_spot_qs,
                                    all_ref_qs,
                                    all_ref_fs,
                                    q_mask,
                                    near_q,
                                    keep_initial_pair=False,
                                    max_ori_refine_iter=50,
                                    max_ori_decomp_count=20):
    
    best_connections = []
    best_qofs = []
    excluded_spot_indices = []
    included_spot_mask = np.asarray([True,] * len(connection_pair_list[0]))
    blank_full_connection = np.asarray([np.nan,] * len(included_spot_mask))
    current_pair_list = connection_pair_list.copy()

    iter_count = 0
    ITERATE = True
    while ITERATE:

        # Evaluate all pairs
        connections, qofs = pair_casting_indexing(
                                    current_pair_list,
                                    all_spot_qs[included_spot_mask],
                                    all_ref_qs,
                                    all_ref_fs,
                                    q_mask,
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
    # 1. Penalize missing reflections weighted according to their expected intensity
    # 2. Do not penalize extra reflections which are not indexing (allows for overlapping orientations)
    # 3. Penalize reflections wighted by their distance from expected postions

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
                          int_weight=0.5):

    # # Determine which reflections are indexed
    # found_spot_mask = [tuple(ref) in [tuple(x) for x in ref_qs] for ref in all_ref_qs]

    dist = [np.sqrt(np.sum([(p - q)**2 for p, q in zip(v1, v2)]))
            for v1, v2 in zip(fit_spot_qs, fit_rot_qs)]

    # Gaussian with specified standard deviation
    # centered at zero sampled at distance
    dist_val = np.sum(np.exp(-(np.asarray(dist))**2 / (2 * sigma**2)))
    norm_dist_val = dist_val / len(all_spot_qs)

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
    # found_spot_mask = [tuple(ref) in [tuple(x) for x in ref_qs] for ref in all_ref_qs]

    dist = [np.sqrt(np.sum([(p - q)**2 for p, q in zip(v1, v2)]))
            for v1, v2 in zip(fit_spot_qs, fit_rot_qs)]

    # Gaussian with specified standard deviation
    # centered at zero sampled at distance
    qof = np.sum(np.exp(-(np.asarray(dist))**2 / (2 * sigma**2)))

    max_qof = len(all_rot_qs)
    norm_qof = qof / max_qof

    return norm_qof
    

def get_rmse(fit_spot_qs,
             fit_rot_qs):
    rmse = np.mean([np.sqrt(np.sum([(p - q)**2 for p, q in zip(v1, v2)]))
                    for v1, v2 in zip(fit_spot_qs, fit_rot_qs)])

    return rmse


def fit_orientation_index(connection,
                          spot_qs,
                          ref_qs,
                          **qof_kwargs):

    fit_spot_qs, fit_ref_qs = _decompose_connection(
                                connection,
                                spot_qs,
                                ref_qs)

    # Find rotation
    # reference then spots give passive rotation...I think
    fit_orientation, _ = Rotation.align_vectors(fit_ref_qs,
                                                fit_ref_spots) 

    qof = get_quality_of_fit(spot_qs,
                             ref_qs,
                             **qof_kwargs)
                            
    return fit_orienation, qof

    
def generate_q_mask(q_vectors,
                    tth_ext,
                    chi_ext,
                    wavelength_ext,
                    poly_args,
                    degrees=False,
                    ext=0):
    
    # Check extent parameters
    for param in [tth_ext, chi_ext, wavelength_ext]:
        if len(param) != 2:
            raise ValueError('Input extents must be of length 2.')
        if param[0] > param[1]:
            raise ValuError('Input extents must be (minimum, maximum).')

    tth, chi, wavelength = q_2_polar(q_vectors,
                                     degrees=degrees)
    
    chi_upr_mask = chi <= general_polynomial(tth, *poly_args[0]) * (1 + ext)
    chi_lwr_mask = chi >= general_polynomial(tth, *poly_args[1]) * (1 - ext)
    tth_upr_mask = tth <= general_polynomial(chi, *poly_args[2]) * (1 + ext)
    tth_lwr_mask = tth >= general_polynomial(chi, *poly_args[3]) * (1 - ext)

    tth_mask = np.all([tth >= tth_ext[0] * (1 - ext),
                       tth <= tth_ext[1] * (1 + ext)],
                       axis=0)
    chi_mask = np.all([chi >= chi_ext[0] * (1 - ext),
                       chi <= chi_ext[1] * (1 + ext)],
                       axis=0)
    wavelength_mask = np.all([wavelength >= wavelength_ext[0] * (1 - ext),
                              wavelength <= wavelength_ext[1] * (1 + ext)],
                              axis=0)
    
    q_mask = np.all([
        chi_lwr_mask,
        chi_upr_mask,
        tth_lwr_mask,
        tth_upr_mask,
        tth_mask,
        chi_mask,
        wavelength_mask
        ], axis=0)

    return q_mask


class QMask():

    def __init__(self,
                 tth_arr,
                 chi_arr,
                 wavelength_vals,
                 # theta_vals,
                 poly_order=6,
                 degrees=False):
        
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
        # self.theta_min = np.min(theta_vals)
        # self.theta_max = np.max(theta_vals)
        self.degrees = degrees

        # Determine edges
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
        
        # upper_poly, _ = curve_fit(general_polynomial, tth_arr[0],
        #                           chi_arr[0], p0=p0)
        # lower_poly, _ = curve_fit(general_polynomial, tth_arr[-1],
        #                           chi_arr[-1], p0=p0)
        # left_poly, _ = curve_fit(general_polynomial, chi_arr[:, 0],
        #                          tth_arr[:, 0], p0=p0)
        # right_poly, _ = curve_fit(general_polynomial, chi_arr[:, -1],
        #                           tth_arr[:, -1], p0=p0)


    
    @classmethod
    def from_XRDRockingCurveStack(cls,
                                  rsm,
                                  **kwargs):

        degrees = rsm.polar_units == 'deg'
        
        inst = cls(rsm.tth_arr,
                   rsm.chi_arr,
                   rsm.wavelength,
                   # rsm.theta,
                   degrees=degrees,
                   **kwargs)
        
        return inst


    def generate(self,
                 q_vectors,
                 ext=0):

        # Convert vectors to polar
        tth, chi, wavelength = q_2_polar(q_vectors,
                                         degrees=self.degrees)

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
        wavelength_mask = np.all([wavelength >= self.wavelength_min * (1 - ext),
                                  wavelength <= self.wavelength_max * (1 + ext)],
                                  axis=0)
        
        q_mask = np.all([
            vertical_mask,
            horizontal_mask,
            tth_mask,
            chi_mask,
            wavelength_mask
        ], axis=0)

        return q_mask


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







### Deprecated ###

'''def fixed_pair_casting_indexing(connection_pair_list,
                                all_spot_qs,
                                all_ref_qs,
                                all_ref_fs,
                                q_mask,
                                near_q
                                iter_max=50):

    connections = []
    qof = []
    for i, pair in tqdm(enumerate(connection_pair_list)):

        prev_connection = pair.copy()
        (pair_spot_inds,
         pair_ref_inds) = _get_connection_indices(pair)

        iter_count = 0
        ITERATE = True
        while ITERATE:
            # Forces original pair
            connection = pair.copy()

            # Find orientation and rotate reference lattice
            (conn_spots,
             conn_refs) = _decompose_connection(prev_connection,
                                                all_spot_qs,
                                                all_ref_qs)
            orientation, _ = Rotation.align_vectors(conn_spots,
                                                    conn_refs)
            all_rot_qs = orientation.apply(all_ref_qs, inverse=False)
            temp_q_mask = q_mask.generate(all_rot_qs)

            # Build kdtree from measured spots and query to referene
            # lattice to avoids non-crystallographic indexing
            kdtree = KDTree(conn_spots)
            pot_conn = kdtree.query_ball_point(all_rot_qs[temp_q_mask],
                                               r=near_q)
            
            # Remove original pair reflections
            for ind in pair_ref_inds:
                if ind in np.nonzero(q_mask)[0]:
                    pot_conn[np.nonzero(np.nonzero(q_mask)[0] == ind)[0][0]] = []
            
            # Cast and expand connection
            for conn_i, conn in enumerate(pot_conn):
                if len(conn) > 0:
                    # Remove reflections near original pair
                    for ind in pair_spot_inds:
                        if ind in conn:
                            conn.remove(ind)
                    if len(conn) == 0:
                        continue
                    elif len(conn) == 1:
                        # Add candidate reflection
                        connection[conn[0]] = np.nonzero(q_mask)[0][conn_i]
                    else:
                        # Add closest of multiple candidate reflections
                        _, ref_idx = kdtree.query(all_rot_qs[q_mask][conn_i])
                        connection[ref_idx] = np.nonzero(q_mask)[0][conn_i]
            
            # Compare connection with previous connection
            curr_spot_inds, curr_ref_inds = _get_connection_indices(connection)
            prev_spot_inds, prev_ref_inds = _get_connection_indices(prev_connection)

            if len(curr_spot_inds) == len(prev_spot_inds):
                if (np.all(curr_spot_inds == prev_spot_inds)
                    and np.all (curr_ref_inds == prev_ref_inds)):
                    ITERATE = False

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
        temp_q_mask = q_mask.generate(all_rot_qs)
        qof = get_quality_of_fit(all_spot_qs[curr_spot_inds],
                                 all_rot_qs[curr_ref_inds],
                                 all_rot_qs,
                                 all_ref_fs,
                                 sigma=near_q)

        connections.append(connection)
        qofs.append(qof)

    return connections, qofs



def initial_pair_casting_indexing(connection_pair_list,
                                  all_spot_qs,
                                  all_ref_qs,
                                  all_ref_fs,
                                  near_q,
                                  q_mask,
                                  iter_max=50):

    connections = []
    qofs = []
    for i, pair in tqdm(enumerate(connection_pair_list)):

        prev_connection = pair.copy()
        (pair_spot_inds,
         pair_ref_inds) = _get_connection_indices(pair)

        iter_count = 0
        ITERATE = True
        while ITERATE:
            # Blank baseline connection
            connection = pair.copy()
            connection[:] = np.nan

            # Find orientation and rotate reference lattice
            (conn_spots,
             conn_refs) = _decompose_connection(prev_connection,
                                                all_spot_qs,
                                                all_ref_qs)
            orientation, _ = Rotation.align_vectors(conn_spots,
                                                    conn_refs)
            all_rot_qs = orientation.apply(all_ref_qs, inverse=False)
            temp_q_mask = q_mask.generate(all_rot_qs)

            # Build kdtree from measured spots and query to referene
            # lattice to avoids non-crystallographic indexing
            kdtree = KDTree(conn_spots)
            pot_conn = kdtree.query_ball_point(all_rot_qs[temp_q_mask],
                                               r=near_q)
            
            # Cast and expand connection
            for conn_i, conn in enumerate(pot_conn):
                if len(conn) > 0:
                    if len(conn) == 0:
                        continue
                    elif len(conn) == 1:
                        # Add candidate reflection
                        connection[conn[0]] = np.nonzero(q_mask)[0][conn_i]
                    else:
                        # Add closest of multiple candidate reflections
                        _, ref_idx = kdtree.query(all_rot_qs[q_mask][conn_i])
                        connection[ref_idx] = np.nonzero(q_mask)[0][conn_i]
            
            # Eliminate invalid connections and replace with original pair
            if np.sum(~np.isnan(connection)) <= 1:
                connection = pair.copy()
                ITERATE = False
                (conn_spots,
                 conn_refs) = _decompose_connection(connection,
                                                    all_spot_qs,
                                                    all_ref_qs)
                orientation, _ = Rotation.align_vectors(conn_spots,
                                                        conn_refs)
                break

            # Compare connection with previous connection
            curr_spot_inds, curr_ref_inds = _get_connection_indices(connection)
            prev_spot_inds, prev_ref_inds = _get_connection_indices(prev_connection)

            if len(curr_spot_inds) == len(prev_spot_inds):
                if (np.all(curr_spot_inds == prev_spot_inds)
                    and np.all (curr_ref_inds == prev_ref_inds)):
                    ITERATE = False

            # Prepare for next iteration
            prev_connection = connection.copy()
            iter_count += 1
            if iter_count >= iter_max:
                ITERATE = False
                # Re-update orientation
                (conn_spots,
                 conn_refs) = _decompose_connection(prev_connection,
                                                    all_spot_qs,
                                                    all_ref_qs)
                orientation, _ = Rotation.align_vectors(conn_spots,
                                                        conn_refs)
        
        # Find qof
        all_rot_qs = orientation.apply(all_ref_qs, inverse=False)
        temp_q_mask = q_mask.generate(all_rot_qs)
        qof = get_quality_of_fit(all_spot_qs[curr_spot_inds],
                                 all_rot_qs[curr_ref_inds],
                                 all_rot_qs,
                                 all_ref_fs,
                                 sigma=near_q)

        connections.append(connection)
        qofs.append(qof)
    
    return connections, qofs'''