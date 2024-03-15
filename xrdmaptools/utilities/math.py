import numpy as np
import scipy.constants as constants


#############################
### Fundamental Relations ###
#############################
# TODO: Wrap conversion functions inside a class
# Or maybe not...

def energy_2_wavelength(energy):
    # Convert energy into keV
    if energy > 1000:
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

#def tth_2_q(tth, wavelength, radians=False, n=1):
#    if radians:
#        tth = np.degrees(tth)
#    return (4 * np.pi * np.sin(np.radians(tth / 2))) / (n * wavelength)

#def q_2_tth(q, wavelength, radians=False, n=1):
#    tth = 2 * np.arcsin(n * wavelength * q / (4 * np.pi))
#    if not radians:
#        tth = np.degrees(tth)
#    return tth


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


def circular_mask(shape, center, radius):

    #if center is None:
    #    center = (int(shape[0] / 2), int(shape[1]/2))

    #if radius is None: 
    #    radius = np.min(center[0], center[1], shape[0] - center[0], shape[1] - center[1])

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
    


'''# Works for 1d and 2d peaks. Ignores arguments with defualt values
def multi_peak_fitting(x, peak_function, *params):
    # Get function input variable names, excluding 'x' or '(xy)' and any default arguments
    inputs = peak_function.__code__.co_varnames[1:peak_function.__code__.co_argcount]
    if peak_function.__defaults__ != None:
        inputs = inputs[:-len(peak_function.__defaults__)]

    print(type(x))

    # Set starting values
    if len(x) == 1:
        z = np.zeros_like(x)
    else:
        z = np.zeros(np.asarray(x).shape[1])

    # Iterate through input paramters for every peak of interest
    arg_dict = {}
    for i in range(0, len(params), len(inputs)):
        for j, arg in enumerate(inputs):
            arg_dict[arg] = params[i + j]
        z += peak_function(x, *arg_dict.values())
    
    return z'''

'''def multi_gaussian_1d(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 4):
        amp = params[i]
        y0 = params[i + 1]
        x0 = params[i + 2]
        sigma = params[i + 3]
        y = y + gaussian_1d(x, amp, y0, x0, sigma)
    return y'''

