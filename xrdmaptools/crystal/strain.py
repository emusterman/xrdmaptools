# Submodule for calculated and applying strain corrections (measurements)

import numpy as np
from scipy import linalg

from xrdmaptools.crystal.crystal import LatticeParameters


def get_strain_orientation(q_vectors,
                           hkls,
                           unstrained):

    q_vectors = np.asarray(q_vectors)
    hkls = np.asarray(hkls)

    if len(q_vectors) != len(hkls):
        err_str = ('Number of spots and assigned hkl '
                   + 'indices must be equal.')
        raise ValueError(err_str)

    # Fit deformation (displacement?) tensor
    # x carries orientation and lattice parameter information
    x, res, rnk, s = linalg.lstsq(hkls, q_vectors)

    # Convert to Busing and Levy UB matrix. Remove 2pi factor
    UBmat = x.T / (2 * np.pi)

    # Polar decomposition to remove rotation components
    # U is the active rotation and the inverse (transpose)
    # is required for passive rotation
    # B is the right-stretch tensor and is related to the B
    # matrix defined by Busing and Levy
    U, B = linalg.polar(UBmat, side='right')

    # Build strained lattice paremeters from right-stretch tensor
    strained = LatticeParameters.from_UBmat(B) # rename this!!!

    # Get transformation matrix between strained and unstrained lattices
    Tij = np.dot(strained.Amat, linalg.inv(unsrained.Amat))
    # This can be acquired in reciprocal space too, but switched positions
    # to account for the changed sign
    # Tij = np.dot(unstrained.Bmat, np.linalg.inv(strained.Bmat))

    # Decompose transformation matrix into strain components
    # This is in the crystal reference frame!
    # Is this Eulerian, Lagrangian, or infinitesimal strain???
    eij_full = 0.5 * (Tij + Tij.T) / 2

    return eij_full, U.T # Convert from active to passive rotation



def decompose_full_strain_tensor(eij_full):

    eij_hydro = np.trace(eij_full) / 3
    eij_dev = eij_full - eij_hydro * np.eye(3)
    
    return eij_dev, eij_hydro


# Less often used
def apply_strain_orientation(q_vectors,
                             eij_full,
                             U):
    raise NotImplementedError()


def CS_2_SS_strain(eij, U):
    raise NotImplementedError()
    return U.T @ eij @ U


def SS_2_CS_strain(eij, U):
    raise NotImplementedError()
    return U @ eij @ U.T