import numpy as np
import scipy.constants as constants


###################################
### Fundamental X-ray Relations ###
###################################

def energy_2_wavelength(energy):
    # Convert energy into keV
    if np.any(energy > 1000):
        energy = energy / 1000
    factor = 1e7 * constants.h * constants.c / constants.e
    return factor / energy


def wavelength_2_energy(wavelength):
    factor = 1e7 * constants.h * constants.c / constants.e
    return factor / wavelength

# Bragg relations
def tth_2_d(tth, wavelength, radians=False, n=1):
    if radians:
        tth = np.degrees(tth)
    return n * wavelength / (2 * np.sin(np.radians(tth / 2)))

def d_2_tth(d, wavelength, radians=False, n=1):
    tth = 2 * np.arcsin(n * wavelength / (2 * d))
    if not radians:
        tth = np.degrees(tth)
    return tth

# Reciprocal space TODO: Add reciprocal space vector components
def convert_qd(input):
    return 2 * np.pi / input # Returns |q|

def tth_2_q(tth, wavelength, radians=False, n=1):
    return convert_qd(tth_2_d(tth, wavelength, radians=radians, n=n)) # Returns |q|

def q_2_tth(q, wavelength, radians=False, n=1):
    return d_2_tth(convert_qd(q), wavelength, radians=radians, n=n) # Returns |q|

########################
### Vector Functions ###
########################

def vector_angle(v1, v2, degrees=False):
    angle = np.arccos(np.dot(v1, v2)
                      / (np.linalg.norm(v1, axis=-1)
                         *  np.linalg.norm(v2, axis=-1)))
    if degrees:
        angle = np.degrees(angle)
    return angle


def mutli_vector_angles(v1s, v2s, degrees=False):
    v1_units = v1s / np.linalg.norm(v1s, axis=1).reshape(-1, 1)
    v2_units = v2s / np.linalg.norm(v2s, axis=1).reshape(-1, 1)

    # Not happy about the round. This is not perfect...
    angles = np.arccos(np.inner(v1_units, v2_units).round(6))

    if degrees:
        angles = np.degrees(angles)
    
    return angles


########################
### Useful Functions ###
########################

def arbitrary_center_of_mass(weights, *args):

    weights = np.asarray(weights)
    for i, arg in enumerate(args):
        arg = np.asarray(arg)
        if weights.shape != arg.shape:
            raise ValueError(f'Shape of arg {i + 1} does not match shape of weights!')
        
    val_list = []
    for arg in args:
        arg = np.asarray(arg)
        val = np.dot(weights.ravel(), arg.ravel()) / np.sum(weights)
        val_list.append(val)

    return tuple(val_list)


#####################################
### Fitting Convenience Functions ###
#####################################

def compute_r_squared(actual, predicted):
    residuals = actual - predicted
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared


def check_precision(*dtypes):
    info = []
    for d in [*dtypes]:
        d = np.dtype(d)
        if np.issubdtype(d, np.integer):
            val = np.iinfo(d)
        elif np.issubdtype(d, np.floating):
            val = np.finfo(d)
        info.append(val)
    return info

#######################
### Other Functions ###
#######################

def circular_mask(shape, center, radius):

    Y, X = np.ogrid[:shape[0], :shape[1]]
    dist_from_center = np.sqrt((X - center[1])**2 + (Y - center[0])**2)

    mask = dist_from_center <= radius
    return mask


def plane_eq(xy, a, b, c):

    if len(xy) != 2:
        raise IOError("xy input must be length 2.")
        
    x = xy[0]
    y = xy[1]

    return a * x + b * y + c


# Higher order polynomial for generic curve fitting
def general_polynomial(x, *args):

    order = len(args) - 1

    y = 0
    for i, arg in enumerate(args):
        y += arg * x**(order - i)

    return y



#####################
### Ray Functions ###
#####################

class k_vector():
    # Class for working diffracted rays as parametric lines

    def __init__(self,
                 point,
                 tth,
                 chi,
                 degrees=False):
        
        self.x0, self.y0, self.z0 = point

        self.degrees = degrees
        if self.degrees:
            tth = np.radians(tth)
            chi = np.radians(chi)

        self.a = np.sin(tth) * np.cos(chi)
        self.b = np.sin(tth) * np.sin(chi)
        self.c = np.cos(tth)
    
        
        def __call__(self, t):
            return (self.x0 + self.a * t,
                    self.y0 + self.b * t,
                    self.z0 + self.c * t)
                
        
        def get_planar_intercept(self, a, b, c):
            t = ((a * (d[0] - self.x0) + b * (d[1] - self.y0) + c * (d[2] - self.z0))
                / (self.a * a + self.b * b + self.c * c))
            return self(t)
        

        def replicate(self,
                      point=None,
                      tth=None,
                      chi=None,
                      degrees=None):
            if point is None:
                point = (self.x0, self.y0, self.z0)
            if tth is None:
                tth = self.tth
            if chi is None:
                chi = self.chi
            if degrees is None:
                degrees = self.degrees
            
            return self.__class__(point,
                                  tth,
                                  chi,
                                  degrees=degrees)


def lstsq_line_intersect(P0, P1):
    # From Traa, Johannes "Least-Squares intersection of Lines" (2013)

    # Generate all line direction vectors
    n = (P1 - P0) / np.linalg.norm(P1 - P0, axis=1)[:, np.newaxis] # normalized

    # Generate array of all projectors
    projs = np.eye(n.shape[1] - n[:, :, np.newaxis] * n[:, np.newaxis]) # I - n * n.T

    # Generate R matrix and q vector
    R = projs.sum(axis=0)
    q = (projs @ p0[:, :, np.newaxis]).sum(axis=0)

    # Solve the least square problem for the 
    # intersection point p: Rp = q
    p = np.linalg.lstsq(R, q, rcond=None)[0]

    return p0


def det_plane_from_ai(ai, skip=None):

    num_pixels = np.prod(*ai.detector.shape)

    if skip is None:
        # skip to about 2500 points
        skip = np.round(np.sqrt(num_pixels
                                / 2500), 0).astype(int)

    # 2, 1, 0 order to adjust to NSLS-II coordinate system
    points = np.asarray([ai.position_array()[::skip, ::skip, i].ravel()
                         for i in [2, 1, 0]])

    d = np.mean(points, axis=1, keepdims=True)
    svd = np.linalg.svd(points - d)

    # Return plane normal n = (a, b, c) and point (d)
    return svd[0][:, -1], d.squeeze()

