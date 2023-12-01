import numpy as np
from scipy.special import wofz
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


############################
### Classes of Functions ###
############################


class PeakFunctionBase():
    # Must define self.func_1d() and self.func_2d()
    # Relies on standard argument input order

    def multi_1d(self, *args):
        return multi_peak_fitting(self.func_1d, *args)

    def multi_2d(self, *args, **kwargs):
        return multi_peak_fitting(self.func_2d, *args, **kwargs)
    
    def func_1d_offset(self, *args):
        return args[-1] + self.func_1d(*args[:-1])
    
    def func_2d_offset(self, *args, **kwargs):     
        return args[-1] + self.func_2d(*args[:-1], **kwargs)

    def generate_guess():
        return

    def generate_bounds():
        return

    def qualify_fit():
        return


class GaussianFunctions(PeakFunctionBase):
    name = 'Gaussian'
    abbr = 'gauss'

    @staticmethod
    def func_1d(x, amp, x0, sigma):
        return amp * (np.exp(-(x - x0)**2 / (2 * sigma**2)))
    
    @staticmethod
    def func_2d(xy, amp, x0, y0, sigma_x, sigma_y, theta, radians=False):
        if len(xy) != 2:
            raise IOError("xy input must be length 2.")

        x = xy[0]
        y = xy[1]

        if not radians:
            theta = np.radians(theta)

        a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
        b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
        c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
        
        return amp * np.exp(-(a * ((x - x0) ** 2) + 2 * b * (x - x0) * (y - y0) + c *((y - y0) **2)))
    
    def get_area():
        return

    def get_FWHM():
        return


class LorentzianFunctions(PeakFunctionBase):
    name = 'Lorentzian'
    abbr = 'lorentz'

    @staticmethod
    def func_1d(x, amp, x0, gamma):
        # FWHM = 2 * gamma
        return amp * (gamma**2 / ((x - x0)**2 + gamma**2))
    
    # Not sure about this one...
    @staticmethod
    def func_2d(xy, amp, x0, y0, gamma_x, gamma_y, theta, radians=False):
        if len(xy) != 2:
            raise IOError("xy input must be length 2.")
        
        x = xy[0]
        y = xy[1]

        if not radians:
            theta = np.radians(theta)
        
        xp = (x - x0) * np.cos(theta) - (y - y0) * np.sin(theta)
        yp = (x - x0) * np.sin(theta) + (y - y0) * np.cos(theta)
        return amp / (1.0 + 4 * xp ** 2 / gamma_x + 4 * yp ** 2 / gamma_y)
    
    def get_area():
        return
    
    def get_FWHM():
        return
    

class VoigtFunctions(PeakFunctionBase):

    def __inti__(self):
        self.G = GaussianFunctions()
        self.L = LorentzianFunctions()
        
    def func_1d(self, x, amp, x0, sigma, gamma, eta):
        # FWHM = 2 * gamma
        return eta * self.G.func_1d(x, amp, x0, sigma) + (1 - eta) * self.L.func_1d(x, amp, x0, gamma)
    
    # Not sure about this one...
    def func_2d(self, xy, amp, x0, y0, sigma_x, sigma_y, gamma_x, gamma_y, theta, eta, radians=False):
        if len(xy) != 2:
            raise IOError("xy input must be length 2.")
        
        x = xy[0]
        y = xy[1]

        if not radians:
            theta = np.radians(theta)
        
        return (eta * self.G.func_2d(x, amp, x0, y0, sigma_x, sigma_y, theta, radians=radians) + 
                (1 - eta) * self.L.func_2d(x, amp, x0, y0, gamma_x, gamma_y, theta, radians=radians))
    
    def get_area():
        return

    def get_FWHM():
        return



####################
### 1D Functions ###
####################


def voigt_1d(x, amp, y0, x0, sigma, gamma):
    # Actual voigt function. Difficult to define FWHM, so less practical for fitting
    return y0 + amp * (np.real(wofz((x - x0 + 1j * gamma) / sigma / np.sqrt(2))) / sigma / np.sqrt(2 * np.pi))


def psuedo_voigt_1d(x, amp, y0, x0, sigma, gamma):
    fg = 2 * sigma * np.sqrt(2 * np.log(2))
    fl = 2 * gamma
    f = ((fg ** 5) + 2.69269 * (fg ** 4) * fl + 2.42843 * (fg ** 3) * (fl ** 2) + 4.47163 * (fg ** 2) * (fl ** 3) + 0.07842 * fg * (fl **4) + (fl ** 5)) ** (1/5)
    eta = 1.36603 * (fl / f) - 0.47719 * ((fl / f) ** 2) + 0.11116 * ((fl / f) ** 3)
    return y0 + amp * ((eta * lorentzian_1d(x, 1, 0, x0, gamma)) + ((1 - eta) * gaussian_1d(x, 1, 0, x0, sigma)))


def quasi_voigt_1d(x, amp, y0, x0, sigma, gamma, eta):
    # psuedo_voigt_1d function with eta as an input. Might combine in future iterations
    return (eta * lorentzian_1d(x, amp, y0, x0, gamma)) + ((1 - eta) * gaussian_1d(x, amp, y0, x0, sigma))


def parabola_1d(x, x0, a, b, y0):
    # b = 0 for symmetric parabola
    return a * ((x - x0) ** 2) + b * (x - x0) + y0


def polar_ellispe_1d(chi, a, e, phi):
    # chi is the azimuthal angle
    # a is the major axis length
    # e is the eccentricity
    # phi is the rotation angle...I think...
    return a * (1 - (e ** 2)) / (1 - e * np.cos(chi - phi))

#def sym_parabola_1d(x, x0, a, y0):
#    return a * ((x - x0) ** 2) + y0

# Not sure how to fit an ellipse, but the rotated equation is:
# 


####################
### 2D Functions ###
####################

def gaussian_2d(xy, amp, y0, x0, z0, sigma_x, sigma_y, theta, radians=False):
    x = xy[0]
    y = xy[1]

    if not radians:
        theta = np.radians(theta)

    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta))/(4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2)/(2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
    
    return z0 + amp * np.exp(-(a * ((x - x0) ** 2) + 2 * b * (x - x0) * (y - y0) + c *((y - y0) **2)))


#######################
### Mutli-Functions ###
#######################

# Works for 1d and 2d peaks.
def multi_peak_fitting(peak_function, *args):
    # arg[0] is the x independent variable
    # arg[1] is the background offset
    # arg[>1] are the arguments of the individual peaks models

    # Get function input variable names, excluding 'self', 'x' or '(xy)' and any default arguments
    inputs = list(peak_function.__code__.co_varnames[:peak_function.__code__.co_argcount])
    if 'self' in inputs: inputs.remove('self')
    inputs = inputs[1:] # remove the x, or xy inputs
    if peak_function.__defaults__ is not None:
        inputs = inputs[:-len(peak_function.__defaults__)] # remove defualts

    x = args[0]

    # Set starting values
    if len(np.asarray(x).shape) == 1:
        z = np.zeros_like(x)
    else:
        z = np.zeros(np.asarray(x).shape[1])
    
    # Add background
    z += args[1]

    # Iterate through input paramters for every peak of interest
    arg_dict = {}
    for i in range(2, len(args) - 2, len(inputs)):
        for j, arg in enumerate(inputs):
            #print(f'{i + j=}')
            #print(f'{arg=}')
            arg_dict[arg] = args[i + j]
        
        #print(arg_dict.values())
        z += peak_function(x, *arg_dict.values())
    
    return z


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

