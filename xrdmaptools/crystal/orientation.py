import numpy as np
from scipy.spatial.transform import Rotation

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




#########################
### Orientation Plots ###
#########################

# TODO: Import generic version from my old code?
# I'll need to generalize to more than just orthorhombic point group
# Consider the orix module instead???