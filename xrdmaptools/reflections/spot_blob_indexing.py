import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.transform import Rotation
from itertools import product, combinations

# Local imports
from .SpotModels import GaussianFunctions
from ..crystal.orientation import euler_rotation
from ..geometry.geometry import get_q_vect



def _initial_spot_analysis(xrdmap, SpotModel=None):
    # TODO: rewrite with spots dataframe and wavelength as inputs...

    # Extract fit stats
    print('Extracting more information from spot parameters...')
    if SpotModel is not None and any([x[:3] == 'fit' for x in xrdmap.spots.loc[0].keys()]):
        interested_params = [x for x in xrdmap.spots.iloc[0].keys()
                             if x[:3] == 'fit'][:6]
        prefix = 'fit'
    elif SpotModel is None or SpotModel == 'guess':
        interested_params = [x for x in xrdmap.spots.iloc[0].keys()
                             if x[:5] == 'guess']
        prefix = 'guess'

    for i in tqdm(xrdmap.spots.index):
        spot = xrdmap.spots.loc[i]

        if prefix == 'fit':
            fit_params = spot[interested_params]

            fwhm = SpotModel.get_2d_fwhm(*fit_params)
            volume = SpotModel.get_volume(*fit_params)

        elif prefix == 'guess':
            guess_params = spot[['guess_height',
                                 'guess_cen_tth',
                                 'guess_cen_chi',
                                 'guess_fwhm_tth',
                                 'guess_fwhm_chi']].values
            fwhm = GaussianFunctions.get_2d_fwhm(*guess_params, 0) # zero for theta
            volume = spot['guess_int']

        more_params = [volume, *fwhm]

        labels = ['integrated',
                  'fwhm_a',
                  'fwhm_b',
                  'rot_fwhm_tth',
                  'rot_fwhm_chi']
        labels = [f'{prefix}_{label}' for label in labels]
        for ind, label in enumerate(labels):
            xrdmap.spots.loc[i, label] = more_params[ind]
    
    
    # Find q-space coordinates
    print('Converting peaks positions to q-space...', end='', flush=True)
    if prefix == 'fit':
        spot_tth = xrdmap.spots['fit_tth0'].values
        spot_chi = xrdmap.spots['fit_chi0'].values

    elif prefix == 'guess':
        spot_tth = xrdmap.spots['guess_cen_tth'].values
        spot_chi = xrdmap.spots['guess_cen_chi'].values
    
    q_values = get_q_vect(spot_tth, spot_chi, xrdmap.wavelength, degrees=True)

    for key, value in zip(['qx', 'qy', 'qz'], q_values):
        xrdmap.spots[key] = value
    print('done!')


# Blind brute force approach to indexing diffraction patterns
# Uninformed symmetry reductions of euler space
# Cannot handle multiple grains
# Does not handle missing reflections
# def iterative_dictionary_indexing(spot_qs,
#                                   Phase,
#                                   tth_range,
#                                   cut_off=0.1,
#                                   start_angle=10,
#                                   angle_resolution=0.001,
#                                   euler_bounds=[[-180, 180],
#                                                 [0, 180],
#                                                 [-180, 180]]):
#     from itertools import product

#     #spot_qs = pixel_df[['qx', 'qy', 'qz']].values
#     all_hkls, all_qs, all_fs = generate_reciprocal_lattice(Phase, tth_range=tth_range)

#     dist = euclidean_distances(all_qs)
#     min_q = np.min(dist[dist > 0])

#     step = start_angle
#     #print(f'Finding orientations with {step} deg resolution...')
#     phi1_list = np.arange(*euler_bounds[0], step)
#     PHI_list = np.arange(*euler_bounds[1], step)
#     phi2_list = np.arange(*euler_bounds[2], step)
#     orientations = list(product(phi1_list, PHI_list, phi2_list))

#     fit_ori = []
#     fit_min = []
#     ITERATE = True
#     while ITERATE:
#         step /= 2 # half the resolution each time
#         if step <= angle_resolution:
#             ITERATE = False

#         #print(f'Evaluating the current Euler space...')
#         min_list = []
#         for orientation in orientations:
#             dist = euclidean_distances(spot_qs, euler_rotation(all_qs, *orientation))
#             min_list.append(np.sum(np.min(dist, axis=1)**2))
        
#         fit_min.append(np.min(min_list))
#         fit_ori.append(orientations[np.argmin(min_list)])
        
#         min_mask = min_list < cut_off * (np.max(min_list) - np.min(min_list)) + np.min(min_list)
#         best_orientations = np.asarray(orientations)[min_mask]

#         #print(f'Finding new orientations with {step:.4f} deg resolution...')
#         new_orientations = []
#         for orientation in best_orientations:
#             phi1, PHI, phi2 = orientation
#             new_phi1 = [phi1 - step, phi1, phi1 + step]
#             new_PHI = [PHI - step, PHI, PHI + step]
#             new_phi2 = [phi2 - step, phi2, phi2 + step]

#             sub_orientations = product(new_phi1, new_PHI, new_phi2)

#             for sub_orientation in sub_orientations:
#                 if sub_orientation not in new_orientations:
#                     new_orientations.append(sub_orientation)
            
#             orientations = new_orientations

#     #print(f'Evaluating the last Euler space...')
#     min_list = []
#     for orientation in orientations:
#         dist = euclidean_distances(spot_qs, euler_rotation(all_qs, *orientation))
#         min_list.append(np.sum(np.min(dist, axis=1)**2))
    
#     fit_min.append(np.min(min_list))
#     fit_ori.append(orientations[np.argmin(min_list)])

#     return fit_ori, fit_min


def ewald_iterative_indexing(spot_qs,
                             Phase,
                             tth_range,
                             cut_off=0.9,
                             start_angle=10,
                             angle_resolution=0.001,
                             euler_bounds=[[-180, 180],
                                           [0, 180],
                                           [-180, 180]]):
    from itertools import product

    #spot_qs = pixel_df[['qx', 'qy', 'qz']].values
    all_hkls, all_qs, all_fs = generate_reciprocal_lattice(Phase, tth_range=tth_range)

    dist = euclidean_distances(all_qs)
    min_q = np.min(dist[dist > 0])

    step = start_angle
    print(f'Finding orientations with {step} deg resolution...')
    phi1_list = np.arange(*euler_bounds[0], step)
    PHI_list = np.arange(*euler_bounds[1], step)
    phi2_list = np.arange(*euler_bounds[2], step)
    orientations = list(product(phi1_list, PHI_list, phi2_list))

    fit_ori = []
    fit_min = []
    ITERATE = True
    while ITERATE:
        step /= 2 # half the resolution each time
        if step <= angle_resolution:
            ITERATE = False

        print(f'Evaluating the current Euler space...')
        min_list = []
        for orientation in tqdm(orientations):
            img_coords = nearest_pixels_on_ewald(euler_rotation(all_qs, *orientation),
                                                 test.wavelength, test.tth_arr, test.chi_arr,
                            near_thresh=0.25, degrees=True)
            min_list.append(test.map.images[*pixel_indices, *img_coords[:, ::-1].T])
        
        fit_min.append(np.max(min_list))
        fit_ori.append(orientations[np.argmax(min_list)])
        
        min_mask = min_list > cut_off * (np.max(min_list) - np.min(min_list)) + np.min(min_list)
        best_orientations = np.asarray(orientations)[min_mask]

        print(f'Finding new orientations with {step:.4f} deg resolution...')
        new_orientations = []
        for orientation in best_orientations:
            phi1, PHI, phi2 = orientation
            new_phi1 = [phi1 - step, phi1, phi1 + step]
            new_PHI = [PHI - step, PHI, PHI + step]
            new_phi2 = [phi2 - step, phi2, phi2 + step]

            sub_orientations = product(new_phi1, new_PHI, new_phi2)

            for sub_orientation in sub_orientations:
                if sub_orientation not in new_orientations:
                    new_orientations.append(sub_orientation)
            
            orientations = new_orientations

    print(f'Evaluating the last Euler space...')
    min_list = []
    for orientation in orientations:
        img_coords = nearest_pixels_on_ewald(euler_rotation(all_qs, *orientation),
                                                test.wavelength, test.tth_arr, test.chi_arr,
                        near_thresh=0.25, degrees=True)
        min_list.append(test.map.images[*pixel_indices, *img_coords[:, ::-1].T])
    
    fit_min.append(np.max(min_list))
    fit_ori.append(orientations[np.argmax(min_list)])

    return fit_ori, fit_min


def explicit_indexing(spot_qs, # Must already be sorted...
                      phase,
                      near_thresh=0.05,
                      required_spots=2,
                      check_equivalent=True):

    # find potential matces within a threshold
    phase_q_vals = phase.reflections['q']
    spot_q_vals = np.linalg.norm(spot_qs, axis=1)
    diff_arr = np.abs(spot_q_vals[:, np.newaxis] - phase_q_vals[np.newaxis, :])
    pot_match = diff_arr < near_thresh

    # Distinguish by phase kind of
    phase_mask = np.any(pot_match, axis=1)
    pot_match = pot_match[phase_mask]
    spot_qs = np.array(spot_qs)[phase_mask]
    #spot_int = spot_ints[phase_mask]

    fit_q_meas = spot_qs[:required_spots]
    fit_q_calc = []
    for i, pot_match_i in enumerate(pot_match[:required_spots]):
        match_ind = np.where(pot_match_i)[0]
        hkls = [phase.reflections['hkl'][ind] for ind in match_ind]

        # Add equivalent potential indices. Only first index restricted
        # Not sure how much this actually helps...
        if check_equivalent and i > 0:
            ext_hkls = []
            [ext_hkls.extend(list(phase.lattice.equivalent_hkls(hkl))) for hkl in hkls];
            hkls = ext_hkls

        q_calc = phase.Q(hkls)
        fit_q_calc.append(q_calc)

    if len(fit_q_calc) > 0:
        combos = list(product(*fit_q_calc))

        rot_list, rssd_list = [], []
        for q_calc in combos:
            rot, rssd = Rotation.align_vectors(q_calc, fit_q_meas)
            rot_list.append(rot.as_matrix())
            rssd_list.append(rssd)

        min_ind = np.argmin(rssd_list)
        min_rssd = rssd_list[min_ind]
        best_orientation = rot_list[min_ind]

    else:
        min_rssd = np.nan
        best_orientation = np.nan
    #print(f'Best orientation found at {min_ind} with rssd of {rssd_list[min_ind]:.3f}')

    # This was worth it!
    #for combo in combos[min_ind][1:]:
    #    if phase.HKL(combo) not in phase.reflections['hkl']:
    #        print('Equivalent hkl used successfully. The extra computation was worth it!')
    #        break

    return best_orientation, min_rssd


def _index_spots(spot_qs,
                 rotation,
                 ref_qs,
                 ref_hkls,
                 near_thresh,
                 indexed_spots=None,
                 indexed_hkls=None):

    if indexed_spots is None:
        indexed_spots = []
    if indexed_hkls is None:
        indexed_hkls = []
    
    rot_spot_qs = spot_qs @ rotation.T

    dist = euclidean_distances(rot_spot_qs, ref_qs)
    # higher threshold to account for chi value changes
    dist[dist > 10 * near_thresh] = np.inf
    matches = np.argmin(dist, axis=1)
    dist_mins = np.min(dist, axis=1)

    #print(f'{len(indexed_spots)=}')
    #print(f'{len(indexed_hkls)=}')
    # This is to add pair indices without know what the hkls necessarily are
    if len(indexed_spots) > 0 and indexed_hkls == []:
        for i in indexed_spots:
            indexed_hkls.append(ref_hkls[matches[i]])
        if indexed_hkls[0] == indexed_hkls[1]:
            raise RuntimeError

    #print(f'{len(indexed_spots)=}')
    #print(f'{len(indexed_hkls)=}')
    for i, val in enumerate(matches):
        if (ref_hkls[val] in indexed_hkls
            or i in indexed_spots
            or np.isinf(dist_mins[i])):
            continue
        
        if np.sum(matches == val) == 1:
            indexed_spots.append(i)
            indexed_hkls.append(ref_hkls[val])
        elif np.sum(matches == val) > 1:
            mask = ((matches == val)
                    & (dist_mins == np.min(dist_mins[matches == val])))
            ind = np.where(mask)[0][0]
            indexed_spots.append(ind)
            indexed_hkls.append(ref_hkls[val])

    return indexed_spots, indexed_hkls


def pairwise_indexing(xrdmap,
                      pixel_indices,
                      phase,
                      near_thresh=0.05,
                      min_rssd=0.15,
                      max_iterations=50):

    pixel_df = xrdmap.pixel_spots(pixel_indices).dropna().sort_values('fit_integrated', ascending=False)

    if len(pixel_df) < 2:
        raise ValueError('Not enough spots for indexing!')

    spot_qs = pixel_df[['qx', 'qy', 'qz']].values

    if phase.reflections is None:
        phase.get_hkl_reflections(tth_range=(np.min(test.tth_arr), np.max(test.tth_arr)))

    # Semi-useful...
    all_hkls, all_qs, all_fs = generate_reciprocal_lattice(phase, tth_range=(np.min(test.tth_arr), np.max(test.tth_arr)))

    # Minimum step size in q-space. Maybe...
    min_q = np.min(np.linalg.norm(phase.Q([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), axis=0))

    # find potential matces within a threshold
    phase_q_vals = phase.reflections['q']
    spot_q_vals = np.linalg.norm(spot_qs, axis=1)
    diff_arr = np.abs(spot_q_vals[:, np.newaxis] - phase_q_vals[np.newaxis, :])
    pot_match = diff_arr < near_thresh

    # Distinguish by phase kind of
    phase_mask = np.any(pot_match, axis=1)
    pot_match = pot_match[phase_mask]
    spot_qs = np.array(spot_qs)[phase_mask]

    # Find matching reflections and convert to q-space (with 2pi factor)
    simple_q_calc = []
    all_q_calc = []
    for pot_match_i in pot_match:
        if not np.any(pot_match_i):
            continue
        match_ind = np.where(pot_match_i)[0]
        hkls = [phase.reflections['hkl'][ind] for ind in match_ind]
        simple_q_calc.append(phase.Q(hkls))

        # Add equivalent hkls
        ext_hkls = []
        [ext_hkls.extend(list(phase.lattice.equivalent_hkls(hkl))) for hkl in hkls];
        hkls = ext_hkls

        q_calc = phase.Q(hkls)
        all_q_calc.append(q_calc)

    if len(all_q_calc) < 1:
        raise ValueError('No reflections close to found spots!')
        grain_list = [[[], [], []]]
        return grain_list

    grain_list = []
    remaining_spots = list(range(len(spot_qs)))
    while len(remaining_spots) > 0:
        #print('Indexing new grain')
        rem_spot_qs = spot_qs[remaining_spots]

        # Indexing only possible on two spots
        # If only one remaining, mark as unindexed
        if len(remaining_spots) == 1:
            grain_list.append((remaining_spots, [], []))
            remaining_spots = []

        #print('Generating new pairs')
        pairs_indices = list(combinations(range(len(remaining_spots)), 2))

        for pair in pairs_indices:
            spot0, spot1 = rem_spot_qs[list(pair)]

            dist = np.sqrt((spot0[0] - spot1[0])**2
                        + (spot0[1] - spot1[1])**2
                        + (spot0[2] - spot1[2])**2)
            
            # Only consider crystallographically feasible distnaces
            if dist < (min_q * 0.85): # within 15 % error
                if pair == pairs_indices[-1]:
                    grain_list.append((remaining_spots, [], []))
                    remaining_spots = []
                continue

            fit_q_calc = [simple_q_calc[remaining_spots[pair[0]]],
                        all_q_calc[remaining_spots[pair[1]]]]

            combos = list(product(*fit_q_calc))

            rot_list, rssd_list = [], []
            valid_combos = []
            for q_calc in combos:
                if np.sum(np.abs(np.cross(*q_calc))) < 1e-8:
                    continue
                rot, rssd = Rotation.align_vectors(q_calc, rem_spot_qs[list(pair)])
                rot_list.append(rot.as_matrix())
                rssd_list.append(rssd)
                valid_combos.append(q_calc)

            # Collect best fit guess hkls
            rssd_list = np.asarray(rssd_list)
            if not np.any(rssd_list < min_rssd):
                if pair == pairs_indices[-1]:
                    grain_list.append((remaining_spots, [], []))
                    remaining_spots = []
                continue
            min_rssd_list = np.round(rssd_list[rssd_list < min_rssd], 10)
            min_q_calcs = np.round(np.array(valid_combos)[rssd_list < min_rssd], 10)
            min_rot_list = np.array(rot_list)[rssd_list < min_rssd]

            # Sort unique fits by minimum rssd
            # This biases towards equivalent positive values
            sorted_min_rssd_list = sorted(np.unique(min_rssd_list))
            sorted_min_q_calcs = [min_q_calcs[np.where(min_rssd_list == val)[0][0]]
                                for val in sorted_min_rssd_list]
            sorted_min_rot_list = [min_rot_list[np.where(min_rssd_list == val)[0][0]]
                                for val in sorted_min_rssd_list]

            # Setup features for refinement
            rotation_list = sorted_min_rot_list
            indexed_spots_catalog = []
            indexed_hkls_catalog = []
            rssd_list = []
            misorientation_list = [] # not currently used
            best_indexed = 0

            # Refine orientation!
            REFINE = True
            iteration = 0
            while REFINE:
                # Setup comparison features for refinement
                new_rotation_list = []
                new_indexed_spots_catalog = []
                new_indexed_hkls_catalog = []
                new_rssd_list = []
                new_misorientation_list = [] # not currently used
                #print(f'{iteration=}')
                for rot in rotation_list:
                    #print(rot)
                    
                    # Forces pair to be included
                    indexed_spots = list(pair)
                    indexed_spots = []
                    indexed_spots, indexed_hkls = _index_spots(
                                                    rem_spot_qs,
                                                    rot,
                                                    all_qs,
                                                    all_hkls,
                                                    indexed_spots=indexed_spots)
                    
                    new_indexed_spots_catalog.append(indexed_spots)
                    new_indexed_hkls_catalog.append(indexed_hkls)

                    new_rot, rssd = Rotation.align_vectors(phase.Q(indexed_hkls), rem_spot_qs[indexed_spots])
                    new_rotation_list.append(new_rot.as_matrix())
                    new_rssd_list.append(rssd)
                    misori = rot @ new_rot.as_matrix().T
                    new_misorientation_list.append(np.degrees(np.arccos(0.5 * (np.trace(misori) - 1))))

                    #if len(remaining_spots) == 4:
                    #    raise
                
                # Best candidate of current refinment
                new_best_indexed = np.argmin(new_rssd_list)
                
                # Force at least two iterations
                if iteration == 0:
                    pass
                elif iteration == max_iterations:
                    #print('WARNING: Refinement reached max number of iterations.')
                    REFINE = False
                elif len(new_indexed_spots_catalog[new_best_indexed]) > len(indexed_spots_catalog[best_indexed]):
                #elif new_rssd_list[new_best_indexed] < rssd_list[best_indexed]:
                    # Keep updating
                    pass
                else:
                    REFINE = False
                    #print(f'Refinement complete after {iteration + 1} iterations.')
                    # Do not redefine values
                    break
                #else:
                #    # Compare values and stop if no new spots
                #    if (best_indexed == new_best_indexed
                #        and len(indexed_spots_catalog[best_indexed])
                #        <= len(new_indexed_spots_catalog[new_best_indexed])):
                #        REFINE = False # No changes, stop refinement
                #        print(f'Refinement complete after {iteration + 1} iterations.')

                # Redefine values
                rotation_list = new_rotation_list.copy()
                indexed_spots_catalog = new_indexed_spots_catalog.copy()
                indexed_hkls_catalog = new_indexed_hkls_catalog.copy()
                rssd_list = new_rssd_list.copy()
                misorientation_list = new_misorientation_list.copy() # not currently used
                best_indexed = new_best_indexed
                iteration += 1

            grain_list.append((np.array(remaining_spots)[indexed_spots_catalog[best_indexed]],
                            indexed_hkls_catalog[best_indexed],
                            rotation_list[best_indexed]))
            
            for spot in np.array(remaining_spots)[indexed_spots_catalog[best_indexed]]:
                remaining_spots.remove(spot)

            if pair == pairs_indices[-1]:
                grain_list.append((remaining_spots, [], []))
                remaining_spots = []
            else:
                break

    return grain_list