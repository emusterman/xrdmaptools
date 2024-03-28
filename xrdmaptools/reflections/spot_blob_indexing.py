import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.transform import Rotation


# Local imports
from .SpotModels import GaussianFunctions
from ..crystal.Phase import generate_reciprocal_lattice
from ..crystal.orientation import euler_rotation
from ..geometry.geometry import get_q_vect



def _initial_spot_analysis(xrdmap, SpotModel=None):
    # TODO: rewrite with spots dataframe and wavelength as inputs...

    # Extract fit stats
    print('Extracting more information from peak parameters...')
    if SpotModel is not None and any([x[:3] == 'fit' for x in xrdmap.spots.loc[0].keys()]):
        interested_params = [x for x in xrdmap.spots.iloc[0].keys()
                             if x[:3] == 'fit'][:6]
        prefix='fit'
    elif SpotModel is None or SpotModel == 'guess':
        interested_params = [x for x in xrdmap.spots.iloc[0].keys()
                             if x[:5] == 'guess']
        prefix='guess'

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
    
    q_values = get_q_vect(spot_tth, spot_chi, xrdmap.wavelength)

    for key, value in zip(['qx', 'qy', 'qz'], q_values):
        xrdmap.spots[key] = value
    print('done!')


# Blind brute force approach to indexing diffraction patterns
# Uninformed symmetry reductions of euler space
# Cannot handle multiple grains
# Does not handle missing reflections
def iterative_dictionary_indexing(spot_qs, Phase, tth_range, cut_off=0.1, start_angle=10, angle_resolution=0.001,
                                  euler_bounds=[[-180, 180], [0, 180], [-180, 180]]):
    from itertools import product

    #spot_qs = pixel_df[['qx', 'qy', 'qz']].values
    all_hkls, all_qs, all_fs = generate_reciprocal_lattice(Phase, tth_range=tth_range)

    dist = euclidean_distances(all_qs)
    min_q = np.min(dist[dist > 0])

    step = start_angle
    #print(f'Finding orientations with {step} deg resolution...')
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

        #print(f'Evaluating the current Euler space...')
        min_list = []
        for orientation in orientations:
            dist = euclidean_distances(spot_qs, euler_rotation(all_qs, *orientation))
            min_list.append(np.sum(np.min(dist, axis=1)**2))
        
        fit_min.append(np.min(min_list))
        fit_ori.append(orientations[np.argmin(min_list)])
        
        min_mask = min_list < cut_off * (np.max(min_list) - np.min(min_list)) + np.min(min_list)
        best_orientations = np.asarray(orientations)[min_mask]

        #print(f'Finding new orientations with {step:.4f} deg resolution...')
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

    #print(f'Evaluating the last Euler space...')
    min_list = []
    for orientation in orientations:
        dist = euclidean_distances(spot_qs, euler_rotation(all_qs, *orientation))
        min_list.append(np.sum(np.min(dist, axis=1)**2))
    
    fit_min.append(np.min(min_list))
    fit_ori.append(orientations[np.argmin(min_list)])

    return fit_ori, fit_min