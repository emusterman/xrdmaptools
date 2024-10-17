import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# Local imports
from xrdmaptools.utilities.math import vector_angle

# Assumption that all angles are in radians...
# Reciprocal space WITHOUT the 2 * pi factor of q-space
# The 2 * pi factor is stored in the Phase class reciprocal lattice vectors


class LatticeParameters():

    def __init__(self,
                 a=None, b=None, c=None,
                 alpha=None, beta=None, gamma=None,
                 a1=None, a2=None, a3=None,
                 Amat=None,
                 a_star=None, b_star=None, c_star=None,
                 alpha_star=None, beta_star=None, gamma_star=None,
                 b1=None, b2=None, b3=None,
                 Bmat=None):
        
        # Consolidate connected parameters
        d_params = np.array([a, b, c,
                             alpha, beta, gamma])
        d_vec = np.array([a1, a2, a3])

        r_params = np.array([a_star, b_star, c_star,
                             alpha_star, beta_star, gamma_star])
        r_vec = np.array([b1, b2, b3])
        
        # Check for complete sets of information, starting with direct lattice
        # Use this information to construct everything else
        if all([d is not None for d in d_params]):
            all([d is not None for d in d_params])
            volume = self.const_2_cell_volume(*d_params)
            r_params = self.convert_lat_const(*d_params)
            rec_volume = self.const_2_cell_volume(*r_params)
            Amat = self.get_mat(*d_params[:3], *d_params[4:],
                                r_params[3], r_params[2])
            Bmat = self.get_mat(*r_params[:3], *r_params[4:],
                                d_params[3], d_params[2])
            d_vec = self.mat_2_vec(Amat)
            r_vec = self.mat_2_vec(Bmat)
        
        elif all([d is not None for d in d_vec]):
            volume = self.vec_2_cell_volume(*d_vec)
            r_vec = self.convert_lat_vec(*d_vec)
            rec_volume = self.vec_2_cell_volume(*r_vec)
            d_params = self.vec_2_const(*d_vec)
            r_params = self.vec_2_const(*r_vec)
            Amat = self.get_mat(*d_params[:3], *d_params[4:],
                                r_params[3], r_params[2])
            Bmat = self.get_mat(*r_params[:3], *r_params[4:],
                                d_params[3], d_params[2])

        elif Amat is not None:
            d_vec = self.mat_2_vec(Amat)
            volume = self.vec_2_cell_volume(*d_vec)
            r_vec = self.convert_lat_vec(*d_vec)
            rec_volume = self.vec_2_cell_volume(*r_vec)
            d_params = self.vec_2_const(*d_vec)
            r_params = self.vec_2_const(*r_vec)
            Bmat = self.get_mat(*r_params[:3], *r_params[4:],
                                d_params[3], d_params[2])
        
        elif all([r is not None for r in r_params]):
            rec_volume = self.const_2_cell_volume(*r_params)
            d_params = self.convert_lat_const(*r_params)
            volume = self.const_2_cell_volume(*d_params)
            Amat = self.get_mat(*d_params[:3], *d_params[4:],
                                r_params[3], r_params[2])
            Bmat = self.get_mat(*r_params[:3], *r_params[4:],
                                d_params[3], d_params[2])
            d_vec = self.mat_2_vec(Amat)
            r_vec = self.mat_2_vec(Bmat)
        
        elif all([r is not None for r in r_vec]):
            rec_volume = self.vec_2_cell_volume(*r_vec)
            d_vec = self.convert_lat_vec(*r_vec)
            volume = self.vec_2_cell_volume(*d_vec)
            d_params = self.vec_2_const(*d_vec)
            r_params = self.vec_2_const(*r_vec)
            Amat = self.get_mat(*d_params[:3], *d_params[4:],
                                r_params[3], r_params[2])
            Bmat = self.get_mat(*r_params[:3], *r_params[4:],
                                d_params[3], d_params[2])

        elif Bmat is not None:
            r_vec = self.mat_2_vec(Bmat)
            rec_volume = self.vec_2_cell_volume(*r_vec)
            d_vec = self.convert_lat_vec(*r_vec)
            volume = self.vec_2_cell_volume(*d_vec)
            d_params = self.vec_2_const(*d_vec)
            r_params = self.vec_2_const(*r_vec)
            Amat = self.get_mat(*d_params[:3], *d_params[4:],
                                r_params[3], r_params[2])

        else:
            raise ValueError('Too little information provided to construct full lattice parameters.')
        
        # Unpack constants
        a, b, c, alpha, beta, gamma = d_params
        a1, a2, a3 = d_vec
        a_star, b_star, c_star, alpha_star, beta_star, gamma_star = r_params
        b1, b2, b3 = r_vec
        
        # A lot of redundant information. Could condense with properties
        # Direct lattice
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.Amat = Amat
        self.volume = volume

        # Reciprocal lattice
        self.a_star = a_star
        self.b_star = b_star
        self.c_star = c_star
        self.alpha_star = alpha_star
        self.beta_star = beta_star
        self.gamma_star = gamma_star
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.Bmat = Bmat
        self.rec_volume = rec_volume

    def __repr__(self):
        ostr = f'|a = {self.a:.6f}\t|b = {self.b:.6f}\t|c = {self.c:.6f}'
        ostr += (f'\n|alpha = {np.degrees(self.alpha):.3f}'
                 + f'\t|beta = {np.degrees(self.beta):.3f}'
                 + f'\t|gamma = {np.degrees(self.gamma):.3f}')
        return ostr
    
    ####################
    ### Classmethods ###
    ####################

    # Convenience ClassMethods for instantiating
    @classmethod
    def from_lat_const(cls, a, b, c, alpha, beta, gamma):
        return cls(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
    

    @classmethod
    def from_lat_vec(cls, a1, a2, a3):
        return cls(a1=a1, a2=a2, a3=a3)
    

    @classmethod
    def from_Amat(cls, Amat):
        return cls(Amat=Amat)
    

    @classmethod
    def from_rec_const(cls, a_star, b_star, c_star,
                       alpha_star, beta_star, gamma_star):
        return cls(a_star=a_star, b_star=b_star, c_star=c_star,
                   alpha_star=alpha_star, beta_star=beta_star, gamma_star=gamma_star)
    

    @classmethod
    def from_rec_vec(cls, b1, b2, b3):
        return cls(b1=b1, b2=b2, a3=b3)
    

    @classmethod
    def from_Bmat(cls, Bmat):
        return cls(Bmat=Bmat)
    

    @classmethod
    def from_Phase(cls, Phase):
        # Just noting the two matrices are already stored...
        Amat = Phase.lattice._ai.T
        Bmat = Phase.lattice._bi.T # with 2 * pi factor
        a = Phase.a
        b = Phase.b
        c = Phase.c
        alpha = np.radians(Phase.alpha)
        beta = np.radians(Phase.beta)
        gamma = np.radians(Phase.gamma)
        return cls(a=a,
                   b=b,
                   c=c,
                   alpha=alpha,
                   beta=beta,
                   gamma=gamma)
    

    @classmethod
    def from_reciprocal_stretch_tensor(cls, B):
        Ur_vec = cls.mat_2_vec(B)
        r_params = cls.vec_2_const(*Ur_vec)
        return cls.from_rec_const(*r_params)
    
    ########################
    ### Base Conversions ###
    ########################

    @staticmethod
    def const_2_cell_volume(a, b, c, alpha, beta, gamma):
        # Inputs can be direct or reciprocal lattice constants and will return the associated volume

        ca = np.cos(alpha)
        cb = np.cos(beta)
        cg = np.cos(gamma)

        #V = a * b * c * np.sqrt(1 - (np.cos(alpha))**2 - (np.cos(beta))**2 - (np.cos(gamma))**2 + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma))
        V = a * b * c * np.sqrt(1 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg)
        
        return V

    @staticmethod
    def vec_2_cell_volume(a1, a2, a3):
        # Inputs can be direct or reciprocal lattice vectors

        V = a1 @ (np.cross(a2, a3))
        return V

    @staticmethod
    def convert_lat_const(a, b, c, alpha, beta, gamma):

        volume = LatticeParameters.const_2_cell_volume(a, b, c, alpha, beta, gamma)

        ca = np.cos(alpha)
        cb = np.cos(beta)
        cg = np.cos(gamma)
        sa = np.sin(alpha)
        sb = np.sin(beta)
        sg = np.sin(gamma)

        new_a = b * c * sa / volume
        new_b = a * c * sb / volume
        new_c = a * b * sg / volume
        new_alpha = np.arccos((cb * cg - ca) / (sb * sg))
        new_beta = np.arccos((ca * cg - cb) / (sa * sg))
        new_gamma = np.arccos((ca * cb - cg) / (sa * sb))

        return new_a, new_b, new_c, new_alpha, new_beta, new_gamma

    @staticmethod
    def convert_lat_vec(a1, a2, a3):

        volume = LatticeParameters.vec_2_cell_volume(a1, a2, a3)

        # It seems the 2 * pi is dropped in these methods. I do not know why
        #b1 = (2 * np.pi * np.cross(a2, a3)) / volume
        #b2 = (2 * np.pi * np.cross(a3, a1)) / volume
        #b3 = (2 * np.pi * np.cross(a1, a2)) / volume

        b1 = np.cross(a2, a3) / volume
        b2 = np.cross(a3, a1) / volume
        b3 = np.cross(a1, a2) / volume

        return b1, b2, b3

    @staticmethod
    def get_mat(a1, a2, a3, alpha2, alpha3, beta1, b3):

        mat = np.array([[a1, a2 * np.cos(alpha3), a3 * np.cos(alpha2)],
                        [0, a2 * np.sin(alpha3), -a3 * np.sin(alpha2) * np.cos(beta1)],
                        [0, 0, 1 / b3]])
        
        return mat

    @staticmethod
    def mat_2_vec(mat):

        a1, a2, a3 = mat.T

        return a1, a2, a3

    @staticmethod
    def vec_2_const(a1, a2, a3):
        from numpy.linalg import norm

        a = norm(a1)
        b = norm(a2)
        c = norm(a3)

        alpha = vector_angle(a2, a3)
        beta = vector_angle(a1, a3)
        gamma = vector_angle(a1, a2)

        return a, b, c, alpha, beta, gamma
    

    def metric_tensor(self):
        return self._metric_tensor(self.a,
                                   self.b,
                                   self.c,
                                   self.alpha,
                                   self.beta,
                                   self.gamma)
    

    def reciprocal_metric_tensor(self):
        return self._metric_tensor(self.a_star,
                                   self.b_star,
                                   self.c_star,
                                   self.alpha_star,
                                   self.beta_star,
                                   self.gamma_star)


    def _metric_tensor(a, b, c, alpha, beta, gamma):
        G11 = a**2
        G22 = b**2
        G33 = c**2
        G12 = a * b * np.cos(gamma)
        G13 = a * c * np.cos(beta)
        G23 = b * c * np.cos(alpha)

        return np.array([[G11, G12, G13],
                         [G12, G22, G23],
                         [G23, G13, G33]])

        # Vector definition
        #np.array([[np.dot(a1, a1), np.dot(a1, a2), np.dot(a1, a3)],
        #          [np.dot(a2, a1), np.dot(a2, a2), np.dot(a2, a3)],
        #          [np.dot(a3, a1), np.dot(a3, a2), np.dot(a3, a3)]])


# This is just math...
def are_coplanar(vectors, return_volume=False):
    vecs = np.asarray(vectors)

    if vecs.ndim != 2:
        raise ValueError('Input vectors must be iterable with at least three 3D vectors.')
    else:
        # Attempt to fix transpose
        if vecs.shape[1] != 3:
            vecs = vecs.T
        if vecs.shape[1] != 3:
            raise ValueError('Vectors are not 3D.')
    
    combos = list(combinations(vecs, 3))
    vols = []
    coplanar_flag = True
    for combo in combos:
        vec1, vec2, vec3 = combo

        # Compute volume with triple scaler product
        vol = vec1 @ (np.cross(vec2, vec3))
        #vol = vec3 @ (np.cross(vec1, vec2))
        #vol = vec2 @ (np.cross(vec3, vec1))
        
        vols.append(vol)
        vol = np.round(vol, 8)

        if not return_volume and vol != 0:
            coplanar_flag = False
            break
        
    # If the volume of the 3 vectors is 0, then they are coplanar
    if return_volume:
        return vols
    else:
        return coplanar_flag
    

# This assumes list of vectors which may be different than are_coplanar()
def are_collinear(vectors):
    vecs = np.asarray(vectors)
    collinear_flag = True

    # Probably faster. Not easy to perform pairwise
    if len(vecs) == 2:
        if np.sum(np.abs(np.cross(*vecs))) > 1e-8:
            collinear_flag = False
        return collinear_flag

    # Pairwise analysis fo list of vectors
    const_list = []
    for ind in range(vecs.shape[1]):
        vecs_axis = vecs[:, ind]
        if not np.any(vecs_axis == 0):
            const = np.abs(vecs_axis[:, np.newaxis] / vecs_axis[np.newaxis, :])
            const_list.append(np.round(const, 3))

    combos = list(combinations(range(vecs.shape[1]), 2))
    if len(combos) > 1:
        combos.pop(-1) # last index is redundant

    for combo in combos:
        if np.any(const_list[combo[0]] != const_list[combo[1]]):
            collinear_flag = False
            break
        
    return collinear_flag
    

def hkl_2_hkil(hkls):
    hkls = np.asarray(hkls)

    if hkls.ndim == 1:
        pass
    elif hkls.ndim == 2:
        # Attempt to handle transpose lists
        if hkls.shape[1] != 3:
            hkls = hkls.T
        if hkls.shape[1] != 3:
            raise ValueError('Input values are not of lenght 3 (h, k, l).')
    else:
        raise ValueError('Input must be hkl or iterable of hkls.')
    
    hkils = []
    for hkl in hkls:
        h, k, l = hkl
        i = -(h + k)
        hkils.append((h, k, i, l))

    return hkils


def hkil_2_hkl(hkils):
    hkils = np.asarray(hkils)

    if hkils.ndim == 1:
        pass
    elif hkils.ndim == 2:
        # Attempt to handle transpose lists
        if hkils.shape[1] != 4:
            hkils = hkils.T
        if hkils.shape[1] != 4:
            raise ValueError('Input values are not of lenght 4 (h, k, i, l).')
    else:
        raise ValueError('Input must be hkil or iterable of hkils.')
    
    hkls = []
    for hkil in hkils:
        h, k, i, l = hkil
        hkils.append((h, k, l))

    return hkls
