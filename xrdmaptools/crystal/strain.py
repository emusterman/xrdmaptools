# Submodule for calculated and applying strain corrections (measurements)

import numpy as np
from scipy import linalg
from copy import deepcopy

from xrdmaptools.crystal.crystal import LatticeParameters
from xrdmaptools.utilities.math import vector_angle



def phase_get_strain_orientation(q_vectors,
                                 hkls,
                                 unstrained_phase):
    
    q_vectors = np.asarray(q_vectors)
    hkls = np.asarray(hkls)

    if len(q_vectors) != len(hkls):
        err_str = ('Number of spots and assigned hkl '
                   + 'indices must be equal.')
        raise ValueError(err_str)

    # Fit deformation (displacement?) tensor
    # x carries orientation and lattice parameter information
    # Remove 2pi factor
    x, res, rnk, s = linalg.lstsq(hkls, q_vectors / (2 * np.pi))

    # Convert to Busing and Levy UB matrix. 
    UBmat = x.T

    # Get B. U is lost when converting to lattice constants
    Ur_vec = LatticeParameters.mat_2_vec(UBmat)
    r_params = LatticeParameters.vec_2_const(*Ur_vec)
    d_params = LatticeParameters.convert_lat_const(*r_params)
    B = LatticeParameters.get_mat(*r_params[:3],
                                  *r_params[-2:],
                                  d_params[3],
                                  d_params[2])
    A = np.linalg.inv(B)

    # print(*d_params[:3], *np.degrees(d_params[3:]))
    # print(B)

    # U, B = linalg.polar(UBmat, side='right')

    # b1, b2, b3 = x
    # a = np.linalg.norm(b1)
    # b = np.linalg.norm(b2)
    # c = np.linalg.norm(b3)
    # alpha = vector_angle(b2, b3)
    # beta = vector_angle(b1, b3)
    # gamma = vector_angle(b1, b2)

    # rec_V = b1 @ (np.cross(b2, b3))

    # ca = np.cos(alpha)
    # cb = np.cos(beta)
    # cg = np.cos(gamma)
    # sa = np.sin(alpha)
    # sb = np.sin(beta)
    # sg = np.sin(gamma)

    # new_a = b * c * sa / rec_V
    # new_b = a * c * sb / rec_V
    # new_c = a * b * sg / rec_V
    # new_alpha = np.arccos((cb * cg - ca) / (sb * sg))
    # new_beta = np.arccos((ca * cg - cb) / (sa * sg))
    # new_gamma = np.arccos((ca * cb - cg) / (sa * sb))

    # B = np.array([[a, b * np.cos(gamma), c * np.cos(beta)],
    #               [0, b * np.sin(gamma), -c * np.sin(beta) * np.cos(new_alpha)],
    #               [0, 0, 1 / new_c]])

    # Get active rotation from UBmat
    # Following Busing and Levy standard Cartesian frame
    U = np.dot(UBmat, linalg.inv(B))

    # Get transformation matrix between strained and unstrained lattices
    # Tij is also deformation gradient matrix from infinitesimal strain theory
    Tij = np.dot(np.linalg.inv(unstrained_phase.lattice.ai), A)
    # Tij = np.dot(unstrained_phase.B / (2 * np.pi),
    #              np.linalg.inv(B))

    # Decompose transformation matrix into strain components
    # This is in the crystal reference frame!
    # Infinitesimal strain!
    eij_full = 0.5 * (Tij + Tij.T) - np.eye(3)

    # # Strain according to xrayutilities
    # # Diagonal is the same, but shear componenent are different
    # eps = np.dot((A - phase.lattice.ai).T, np.linalg.inv(phase.lattice.ai.T))
    # print(eps)


    # # strained_phase.ApplyStrain(eps)
    # Strain from xrayutilities does not take symmetrized input
    strained_phase = deepcopy(unstrained_phase)
    strained_phase.ApplyStrain((Tij - np.eye(3)).T)

    # strained_phase.convert_to_P1()
    # strained_phase.a = d_params[0]
    # strained_phase.b = d_params[1]
    # strained_phase.c = d_params[2]
    # strained_phase.alpha = np.degrees(d_params[3])
    # strained_phase.beta = np.degrees(d_params[4])
    # strained_phase.gamma = np.degrees(d_params[5])

    return eij_full, U.T, strained_phase


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
    # Remove 2pi factor
    x, res, rnk, s = linalg.lstsq(hkls, q_vectors / (2 * np.pi))

    # Convert to Busing and Levy UB matrix. 
    UBmat = x.T 

    # Build strained lattice parameters from UBmat
    strained = LatticeParameters.from_UBmat(UBmat)

    # Get active rotation from UBmat
    U = np.dot(UBmat, linalg.inv(strained.Bmat))

    # Get transformation matrix between strained and unstrained lattices
    Tij = np.dot(strained.Amat, linalg.inv(unstrained.Amat))
    # This can be acquired in reciprocal space too, but switched positions
    # to account for the changed sign
    # Tij = np.dot(unstrained.Bmat, np.linalg.inv(strained.Bmat))

    # Decompose transformation matrix into strain components
    # This is in the crystal reference frame!
    # Is this Eulerian, Lagrangian, or infinitesimal strain???
    eij_full = 0.5 * (Tij + Tij.T) - np.eye(3)

    return (eij_full,
            U.T, # Convert from active to passive rotation
            strained)


# def get_strain(q_vectors,
#                hkls,
#                unstrained):

#     q_vectors = np.asarray(q_vectors)
#     hkls = np.asarray(hkls)

#     if len(q_vectors) != len(hkls):
#         err_str = ('Number of spots and assigned hkl '
#                    + 'indices must be equal.')
#         raise ValueError(err_str)

#     # Fit deformation (displacement?) tensor
#     # x carries orientation and lattice parameter information
#     x, res, rnk, s = linalg.lstsq(hkls, q_vectors / (2 * np.pi))

#     # Convert to Busing and Levy UB matrix. Remove 2pi factor
#     # UBmat = x.T / (2 * np.pi)

#     # UB_star = np.dot(UBmat, unstrained.Bmat)
#     # strained = LatticeParameters.from_reciprocal_stretch_tensor(UB_star)
#     strained = LatticeParameters.from_Bmat(x.T)

#     # # Polar decomposition to remove rotation components
#     # # U is the active rotation and the inverse (transpose)
#     # # is required for passive rotation
#     # # B is the right-stretch tensor and is related to the B
#     # # matrix defined by Busing and Levy
#     # U, B = linalg.polar(UBmat, side='right')

#     # # Build strained lattice paremeters from right-stretch tensor
#     # strained = LatticeParameters.from_reciprocal_stretch_tensor(B)

#     # Get transformation matrix between strained and unstrained lattices
#     Tij = np.dot(strained.Amat, linalg.inv(unstrained.Amat))
#     # This can be acquired in reciprocal space too, but switched positions
#     # to account for the changed sign
#     # Tij = np.dot(unstrained.Bmat, np.linalg.inv(strained.Bmat))

#     # Decompose transformation matrix into strain components
#     # This is in the crystal reference frame!
#     # Is this Eulerian, Lagrangian, or infinitesimal strain???
#     eij_full = 0.5 * (Tij + Tij.T) - np.eye(3)

#     return (eij_full,
#             # U.T, # Convert from active to passive rotation
#             strained)


# def get_strain_orientation(q_vectors,
#                            hkls,
#                            unstrained):

#     q_vectors = np.asarray(q_vectors)
#     hkls = np.asarray(hkls)

#     if len(q_vectors) != len(hkls):
#         err_str = ('Number of spots and assigned hkl '
#                    + 'indices must be equal.')
#         raise ValueError(err_str)

#     # Fit deformation (displacement?) tensor
#     # x carries orientation and lattice parameter information
#     x, res, rnk, s = linalg.lstsq(hkls, q_vectors)

#     # Convert to Busing and Levy UB matrix. Remove 2pi factor
#     UBmat = x.T / (2 * np.pi)

#     # Polar decomposition to remove rotation components
#     # U is the active rotation and the inverse (transpose)
#     # is required for passive rotation
#     # B is the right-stretch tensor and is related to the B
#     # matrix defined by Busing and Levy
#     U, B = linalg.polar(UBmat, side='right')

#     # Build strained lattice paremeters from right-stretch tensor
#     strained = LatticeParameters.from_reciprocal_stretch_tensor(B)

#     # Get transformation matrix between strained and unstrained lattices
#     Tij = np.dot(strained.Amat, linalg.inv(unstrained.Amat))
#     # This can be acquired in reciprocal space too, but switched positions
#     # to account for the changed sign
#     # Tij = np.dot(unstrained.Bmat, np.linalg.inv(strained.Bmat))

#     # Decompose transformation matrix into strain components
#     # This is in the crystal reference frame!
#     # Is this Eulerian, Lagrangian, or infinitesimal strain???
#     eij_full = 0.5 * (Tij + Tij.T) - np.eye(3)

#     return (eij_full,
#             U.T, # Convert from active to passive rotation
#             strained)


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