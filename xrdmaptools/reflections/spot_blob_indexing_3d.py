
import numpy as np
from scipy.spatial.transofrm import Rotation
from scipy.spatial import distance_matrix
from itertools import combinations, product



from xrdmaptools.utilities.math import (
    multi_vector_angles,
    general_polynomial
)
from xrdmaptools.geometry.geometry import q_2_polar


def pair_casting_index_pattern(fixed_pairs=False):
    raise NotImplementedError()


#####################
### Sub-Functions ###
#####################


def find_all_valid_pairs(spot_qs,
                         phase,
                         near_q,
                         near_angle,
                         degrees=False):
    

    # Find q vector magnitudes and max for spots
    spot_q_mags = np.linalg.norm(spot_qs, axis=1)
    max_q = np.max(spot_q_mags)

    # Find q vector magnitudes for reference phase
    phase.generate_reciprocal_lattice(1.15 * max_q)
    ref_qs = phase.all_ref_qs
    ref_q_mags = np.linalg.norm(ref_qs, axis=1)

    # Find minimum q vector step size from reference phase
    min_q = np.min(np.linalg.norm(phase.Q([[1, 0, 0],
                                           [0, 1, 0],
                                           [0, 0, 1]]),
                                           axis=0))
    if near_q > min_q * 0.85:
        err_str = ("'near_q' threshold is greater than 85% of minimum "
                   + "lattice spacing. This seems unwise.")
        raise ValueError(err_str)

    # Find difference between measured and calculated q magnitudes
    mag_diff_arr = np.abs(spot_q_mags[:, np.newaxis]
                          - ref_q_mags[np.newaxis, :])
    
    # Eliminate any reflections outside of phase-allowed spots
    phase_mask = np.any(mag_diff_arr < near_q, axis=1)
    mag_diff_arr = mag_diff_arr[phase_mask]
    spot_qs = spot_qs[phase_mask]
    spot_q_mags = spot_q_mags[phase_mask]

    # Generate all pairs of spots which are crystallographically feasible
    spot_pair_indices = list(combinations(range(len(spot_qs)), 2))
    spot_pair_dist = [spot_pair_dist[tuple(indices)] > min_q * 0.85
                      for indices in spot_pair_indices]
    
    # Determine all angles
    spot_angles = multi_vector_angles(spot_qs, spot_qs, degrees=degrees)
    ref_angles = multi_vector_angles(ref_qs, ref_qs, degrees=degrees)

    blank_connection = [np.nan,] * len(spot_qs)
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
                connection[pair[0]] = combo[0]
                connection[pair[1]] = combo[1]
                connection_pair_list.append(connection)

    return np.asarray(connection_pair_list)


def reduce_symmetric_equivalents(connection_pair_list,
                                 all_spot_qs,
                                 all_ref_qs,
                                 all_ref_hkls,
                                 near_angle,
                                 q_mask):
    
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
        pair_divs = np.array(pair_ref_hkls[0]) / np.array(pair_ref_hkls[1])
        if len(np.unique(pair_divs[~np.isnan(pair_divs)])) < 2:
            pair_orientations.append(np.nan) # assumes validity
            pair_rmse.append(np.nan)
            pair_mis_mag.append(np.nan)
        
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
        if np.isnan(pair_rmse[pair_i]):
            eval_pair_mask[pair_i] = False
            keep_pair_mask[pair_i] = False # colinear pairs are not useful for casting
            continue

        # Discard poorly fitting pairs
        if pair_rmse[pair_i] < min_q:
            eval_pair_mask[pair_i] = False
            keep_pair_mask[pair_i] = False
            continue
        
        # Find equivalent orientations
        similar_pair_mask = pair_rmse == pair_rmse[pair_i]
        if np.sum(similar_pair_mask) > 1: # isolated pairs are ignored. probably colinear
            min_mis_mag = np.min(pair_mis_mag[similar_pair_mask])
            min_indices = np.nonzero(pair_mis_mag[similar_pair_mask]
                                     < min_mis_mag + near_angle[0]) # some wiggle room

            keep_pair_mask[np.nonzero(similar_pair_mask)[0][min_indices]] = True
        
    return connection_pair_list[keep_pair_mask]


def fixed_pair_casting_indexing(connection_pair_list,
                                all_spot_qs,
                                all_ref_qs,
                                q_mask
                                iter_max=50):

    connections = []

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
            rot_qs = orientation.apply(all_ref_qs, inverse=False)
            temp_q_mask = q_mask.generate(rot_qs)

            # Build kdtree from measured spots and query to referene
            # lattice to avoids non-crystallographic indexing
            kdtree = KDTree(conn_spots)
            pot_conn = kdtree.query_ball_point(rot_qs[temp_q_mask],
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
                        _, ref_idx = kdtree.query(rot_qs[q_mask][conn_i])
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
                (conn_spots,
                 conn_refs) = _decompose_connection(prev_connection,
                                                    all_spot_qs,
                                                    all_ref_qs)
                orientation, _ = Rotation.align_vectors(conn_spots,
                                                        conn_refs)
        
        connections.append(connection)



def initial_pair_casting_indexing(connection_pair_list,
                                all_spot_qs,
                                all_ref_qs,
                                q_mask
                                iter_max=50):

    connections = []

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
            rot_qs = orientation.apply(all_ref_qs, inverse=False)
            temp_q_mask = q_mask.generate(rot_qs)

            # Build kdtree from measured spots and query to referene
            # lattice to avoids non-crystallographic indexing
            kdtree = KDTree(conn_spots)
            pot_conn = kdtree.query_ball_point(rot_qs[temp_q_mask],
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
                        _, ref_idx = kdtree.query(rot_qs[q_mask][conn_i])
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
        
        connections.append(connection)


def iterative_pattern_decomposition():
    raise NotImplementedError()


#########################
### Utility Functions ###
#########################


def get_quality_of_fit(spot_qs,
                       ref_qs,
                       all_ref_qs=None,
                       all_ref_fs=None
                       **kwargs):

    # Ideal conditions
    # 1. Penalize missing reflections weighted according to their expected intensity
    # 2. Do not penalize extra reflections which are not indexing (allows for overlapping orientations)
    # 3. Penalize reflections wighted by their distance from expected postions

    # qof = get_rmse(spot_qs, ref_qs)
    qof = custom_quality_of_fit(spot_qs
                                ref_qs,
                                all_ref_qs,
                                sigma=kwargs['sigma'])

    return qof


def custom_quality_of_fit(spot_qs, # Found spots
                          ref_qs, # Indexed refs
                          all_ref_qs, # All refs in mask
                          # all_ref_fs, # All fs in mask
                          sigma=1): # Gaussian standard deviation to evaluate

    # # Determine which reflections are indexed
    # found_spot_mask = [tuple(ref) in [tuple(x) for x in ref_qs] for ref in all_ref_qs]

    dist = [np.sqrt(np.sum([(p - q)**2 for p, q in zip(v1, v2)]))
            for v1, v2 in zip(spot_qs, ref_qs)]

    # Gaussian with specified standard deviation
    # centered at zero sampled at distance
    qof = np.sum(np.exp(-(np.asarray(dist))**2 / (2 * sigma**2)))

    max_qof = len(all_ref_qs)
    norm_qof = qof / max_qof

    return norm_qof
    

def get_rmse(spot_qs,
             ref_qs):
    rmse = np.mean([np.sqrt(np.sum([(p - q)**2 for p, q in zip(v1, v2)]))
                    for v1, v2 in zip(spot_qs, ref_qs)])

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
                 ext=0,
                 degrees=False):

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
        p0 = np.ones(poly_order + 1)
        for indexing, poly in zip([(0), # [0, :] 
                                   (-1), # [-1, :]
                                   (slice(None), 0), # [:, 0]
                                   (slice(None), -1)], # [:, -1]
                                   ['upper_poly',
                                   'lower_poly',
                                   'left_poly',
                                   'right_poly'])
            
            # Determine functional direction
            tth_gradient = np.gradent(tth_arr[indexing])
            if np.all(tth-grad > 0) or np.all(tth_grd < 0):
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
        
        inst = cls(rsm.tth_arr,
                   rsm.chi_arr,
                   rsm.wavelength,
                   # rsm.theta,
                   **kwargs)
        
        return inst


    def generate(self,
                 q_vectors):

        # Convert vectors to polar
        tth, chi, wavelength = q_2_polar(q_vectors,
                                         degrees=self.degrees)
        
        edge_masks = []
        for poly in ['upper_poly',
                     'lower_poly',
                     'left_poly',
                     'right_poly']:
            
            if getattr(self, f'{poly}_first') == 'tth':
                first = tth
                second = chi
            else:
                first = chi
                second = tth
            
            mask = second <= general_polynomial(first, *getattr(self, poly)) * (1 + ext)
            edge_masks.append(mask)
        
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
            *edge_masks,
            tth_mask,
            chi_mask,
            wavelength_mask
        ], axis=0)

        return q_mask



def _get_connection_indices(connection):

    spot_indices = np.nonzero(~np.isnan(connection))
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