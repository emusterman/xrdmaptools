import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.transform import Rotation

# Local imports
from .Phase import generate_reciprocal_lattice

# TODO:
# Build-up the code around more general rotations using the scipy Rotation class



###########################
### Reference Functions ###
###########################

# Returns orientation matrix from euler angles
def g_func(phi1, phi, phi2, rad=False):

    '''
    phi1        (float) First Euler angle in Bunge definition
    phi         (float) Second Euler angle in Bunge definition
    phi2        (float) Third Euler angle in Bunge definition
    rad         (bool)  True if angles given in radians
    '''

    # Check and convert input angles
    if rad & (np.max([phi1, phi, phi2]) > 30): # Assumes some random pixel will exceed threshold in dataset
        raise ValueError('Euler angle values in radians exceed expected bounds!')
    elif not rad:
        phi1 = np.radians(phi1)
        phi = np.radians(phi)
        phi2 = np.radians(phi2)
        
    # Just a bit of shorthand
    from numpy import cos as c
    from numpy import sin as s

    # Passive Bunge definition
    g_T =   ([c(phi1)*c(phi2)-s(phi1)*s(phi2)*c(phi), s(phi1)*c(phi2)+c(phi1)*s(phi2)*c(phi), s(phi2)*s(phi)],
            [-c(phi1)*s(phi2)-s(phi1)*c(phi2)*c(phi), -s(phi1)*s(phi2)+c(phi1)*c(phi2)*c(phi), c(phi2)*s(phi)],
            [s(phi1)*s(phi), -c(phi1)*s(phi), c(phi)])
    g_T = np.asarray(g_T).T # not sure if this needs to be an array

    # Transpose array to be correct. Not sure why it is wrong, but this works
    g = np.array([g_T[i].T for i in range(len(g_T))])

    return g


# Rotates an array from euler angles
# Same as multplying a vector by g
# The order of rotations may not be the same as g_func above
def euler_rotation(arr, phi1, PHI, phi2, radians=False):

    if not radians:
        phi1 = np.radians(phi1)
        PHI = np.radians(PHI)
        phi2 = np.radians(phi2)

    R1 = np.array([[np.cos(phi1), -np.sin(phi1), 0],
                   [np.sin(phi1), np.cos(phi1), 0],
                   [0, 0, 1]])

    R2 = np.array([[1, 0, 0],
                   [0, np.cos(PHI), -np.sin(PHI)],
                   [0, np.sin(PHI), np.cos(PHI)]])

    R3 = np.array([[np.cos(phi2), -np.sin(phi2), 0],
                   [np.sin(phi2), np.cos(phi2), 0],
                   [0, 0, 1]])
    
    # One of these has to be right
    return arr @ R1 @ R2 @ R3
    #return arr @ R3 @ R2 @ R1


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




#########################
### Orientation Plots ###
#########################

# TODO: Import generic version from my old code?
# I'll need to generalize to more than just orthorhombic point group
# Consider the orix module instead???