import numpy as np
from scipy.special import wofz
import scipy.stats as stats
from numba import njit


############################
### Classes of Functions ###
############################


def _load_peak_function(name):

    for SpotModel in [GaussianFunctions,
                     LorentzianFunctions,
                     PseudoVoigtFunctions]:
        if name in [SpotModel.name, SpotModel.abbr]:
            return SpotModel
        
    print(f'Given peak name: {name} is not yet supported.')


class SpotModelBase():
    # Must define self.func_1d() and self.func_2d()
    # Relies on standard argument input order

    # Useful parameters
    par_1d = ['amp', 'x0', 'fwhm_x']
    par_2d = ['amp', 'x0', 'y0', 'fwhm_x', 'fwhm_y', 'theta']


    @classmethod
    def multi_1d(cls, *args):
        return multi_peak_fitting(cls.func_1d, *args)

    @classmethod
    def multi_2d(cls, *args, radians=False):
        return multi_peak_fitting(cls.func_2d, *args, radians=radians)
    
    @classmethod
    def func_1d_offset(cls, *args):
        return args[-1] + cls.func_1d(*args[:-1])
    
    @classmethod
    def func_2d_offset(cls, *args, radians=False):     
        return args[-1] + cls.func_2d(*args[:-1], radians=radians)
    
    @classmethod
    def get_1d_fwhm(cls, *args):
        arg_dict = dict(zip(cls.par_1d, args))
        return arg_dict['fwhm']
    
    @classmethod
    def get_2d_fwhm(cls, *args, radians=False):
        arg_dict = dict(zip(cls.par_2d, args))
        fwhm_x = arg_dict['fwhm_x']
        fwhm_y = arg_dict['fwhm_y']
        theta = arg_dict['theta']
                        
        if not radians:
            theta = np.radians(theta)

        fwhm_rx = fwhm_x * np.cos(-theta) - fwhm_y * np.sin(-theta)
        fwhm_ry = fwhm_x * np.sin(-theta) + fwhm_y * np.cos(-theta)

        return fwhm_rx, fwhm_ry, fwhm_x, fwhm_y

    # def generate_guess():
    #     raise NotImplementedError
    
    # def generate_bounds():
    #     raise NotImplementedError

    # def qualify_fit():
    #     raise NotImplementedError


class GaussianFunctions(SpotModelBase):
    # Useful parameters
    name = 'Gaussian'
    abbr = 'gauss'
    #par_1d = ['amp', 'x0', 'fwhm']
    #par_2d = ['amp', 'x0', 'y0', 'fwhm_x', 'fwhm_y', 'theta']
    par_3d = ['amp', 'x0', 'y0', 'z0',
              'fwhm_x', 'fwhm_y','fwhm_z',
              'theta', 'phi']

    @staticmethod
    # @njit
    def func_1d(x, amp, x0, fwhm_x):
        sigma = fwhm_x / (2 * np.sqrt(2 * np.log(2)))
        return amp * np.exp(-(x - x0)**2 / (2 * sigma**2))
    
    @staticmethod
    # @njit
    def func_2d(xy, amp, x0, y0, fwhm_x, fwhm_y, theta, radians=False):
        if len(xy) != 2:
            raise IOError("xy input must be length 2.")

        x = xy[0]
        y = xy[1]

        if not radians:
            theta = np.radians(theta)

        #a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
        #b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
        #c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
        #return amp * np.exp(-(a * ((x - x0) ** 2) + 2 * b * (x - x0) * (y - y0) + c *((y - y0) **2)))
        
        # Coordinate rotation and translation
        # Rewrite in matrix notation??
        xp = (x - x0) * np.cos(theta) - (y - y0) * np.sin(theta)
        yp = (x - x0) * np.sin(theta) + (y - y0) * np.cos(theta)

        sigma_x = fwhm_x / (2 * np.sqrt(2 * np.log(2)))
        sigma_y = fwhm_y / (2 * np.sqrt(2 * np.log(2)))

        return amp * np.exp(-((xp**2 / (2 * sigma_x**2))
                            + (yp**2 / (2 * sigma_y**2))))
    
    @staticmethod
    # @njit
    def func_3d(xyz, amp,
                x0, y0, z0,
                fwhm_x, fwhm_y, fwhm_z,
                theta, phi, radians=False):
        
        if len(xyz) != 3:
            raise IOError("xyz input must be length 3.")

        x = xyz[0]
        y = xyz[1]
        z = xyz[2]

        if not radians:
            theta = np.radians(theta)
            phi = np.radians(phi)
        
        xp = ((x - x0) * np.cos(phi)
              - (y - y0) * np.sin(phi) * np.cos(theta)
              + (z - z0) * np.sin(phi) * np.sin(theta))
        yp = ((x - x0) * np.sin(phi)
              + (y - y0) * np.cos(phi) * np.cos(theta)
              - (z - z0) * np.cos(phi) * np.sin(theta))
        zp = ((y - y0) * np.sin(theta)
              + (z - z0) * np.cos(theta))
        
        sigma_x = fwhm_x / (2 * np.sqrt(2 * np.log(2)))
        sigma_y = fwhm_y / (2 * np.sqrt(2 * np.log(2)))
        sigma_z = fwhm_z / (2 * np.sqrt(2 * np.log(2)))

        return amp * np.exp(-((xp**2 / (2 * sigma_x**2))
                            + (yp**2 / (2 * sigma_y**2))
                            + (zp**2 / (2 * sigma_z**2))))
    
    @staticmethod
    # @njit
    def get_area(amp, x0, fwhm_x):
        # Returns area under 1d gaussian function
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        return amp * np.sqrt(2 * np.pi) * sigma
    
    @staticmethod
    # @njit
    def get_volume(amp, x0, y0, fwhm_x, fwhm_y, theta, radians=False):
        # Returns volume under 2d gaussian function
        sigma_x = fwhm_x / (2 * np.sqrt(2 * np.log(2)))
        sigma_y = fwhm_y / (2 * np.sqrt(2 * np.log(2)))
        return amp * 2 * np.pi * sigma_x * sigma_y
    
    @staticmethod
    # @njit
    def get_hyper_volume(amp, x0, y0, z0, fwhm_x, fwhm_y, fwhm_z, theta, phi, radians=False):
        raise NotImplementedError()
        sigma_x = fwhm_x / (2 * np.sqrt(2 * np.log(2)))
        sigma_y = fwhm_y / (2 * np.sqrt(2 * np.log(2)))
        sigma_z = fwhm_z / (2 * np.sqrt(2 * np.log(2)))
        return amp * sigma_x * sigma_y * sigma_z * (2 * np.pi)**1.5

    def get_3d_fwhm():
        raise NotImplementedError()


class LorentzianFunctions(SpotModelBase):
    # Useful parameters
    name = 'Lorentzian'
    abbr = 'lorentz'
    #par_1d = ['amp', 'x0', 'fwhm']
    #par_2d = ['amp', 'x0', 'y0', 'fwhm_x', 'fwhm_y', 'theta']

    @staticmethod
    # @njit
    def func_1d(x, amp, x0, fwhm_x):
        gamma = 0.5 * fwhm_x
        return amp * (gamma**2 / ((x - x0)**2 + gamma**2))
    
    # Not sure about this one...
    @staticmethod
    # @njit
    def func_2d(xy, amp, x0, y0, fwhm_x, fwhm_y, theta, radians=False):
        if len(xy) != 2:
            raise IOError("xy input must be length 2.")
        
        x = xy[0]
        y = xy[1]

        if not radians:
            theta = np.radians(theta)
        
        # Coordinate rotation and translation
        # Rewrite in matrix notation??
        xp = (x - x0) * np.cos(theta) - (y - y0) * np.sin(theta)
        yp = (x - x0) * np.sin(theta) + (y - y0) * np.cos(theta)

        gamma_x = 0.5 * fwhm_x
        gamma_y = 0.5 * fwhm_y

        return amp / (1 + (xp / gamma_x)**2 + (yp/ gamma_y)**2)
    
    @staticmethod
    # @njit
    def get_area(amp, x0, fwhm_x):
        gamma = 0.5 * fwhm
        return amp * np.pi * gamma
    
    @staticmethod
    # @njit
    def get_volume(amp, x0, y0, fwhm_x, fwhm_y, theta):
        # Current 2D Lorentzian function does not have a finite volume

        # We approximate volume with a 2D gaussian based on equal areas of the major axes
        gamma_x = 0.5 * fwhm_x
        gamma_y = 0.5 * fwhm_y
        sigma_x = gamma_x * np.sqrt(np.pi / 2)
        sigma_y = gamma_y * np.sqrt(np.pi / 2)

        return amp * 2 * np.pi * sigma_x * sigma_y


# This class is still untested...
# njit might break on these functions...
class PseudoVoigtFunctions(SpotModelBase):
    # Note: This is the pseudo-voigt profile!
    # Useful parameters
    name = 'PseudoVoigt'
    abbr = 'p_voigt'
    par_1d = ['amp', 'x0', 'G_fwhm_x', 'L_fwhm_x']
    par_2d = ['amp', 'x0', 'y0',
              'G_fwhm_x', 'G_fwhm_y',
              'L_fwhm_x', 'L_fwhm_y',
              'theta']
    
    G = GaussianFunctions
    L = LorentzianFunctions

    @staticmethod
    # @njit
    def _get_eta_fwhm(G_fwhm_x, L_fwhm_x):
        # Convolved fwhm
        fwhm_x = ((G_fwhm_x**5
                 + 2.69269 * G_fwhm_x**4 * L_fwhm_x**1
                 + 2.42843 * G_fwhm_x**3 * L_fwhm_x**2
                 + 4.47163 * G_fwhm_x**2 * L_fwhm_x**3
                 + 0.07842 * G_fwhm_x**1 * L_fwhm_x**4
                 + L_fwhm_x**5
                 )**(1 / 5))
        # Fraction
        eta = (1.36603 * (L_fwhm_x / fwhm_x)
               -0.47719 * (L_fwhm_x / fwhm_x)**2
               +0.11116 * (L_fwhm_x / fwhm_x)**3)
        
        return eta, fwhm_x
    
    @staticmethod
    # @njit
    def func_1d(x, amp, x0, G_fwhm_x, L_fwhm_x):
        V = PseudoVoigtFunctions
        eta, fwhm_x = V._get_eta_fwhm(G_fwhm_x, L_fwhm_x) # This feels wrong
        return (eta * V.G.func_1d(x, amp, x0, fwhm_x)
                + (1 - eta) * V.L.func_1d(x, amp, x0, fwhm_x))
    
    # Not sure about this one...
    @staticmethod
    # @njit
    def func_2d(xy, amp, x0, y0, G_fwhm_x, G_fwhm_y,
                L_fwhm_x, L_fwhm_y, theta, radians=False):
        V = PseudoVoigtFunctions
        eta_x, fwhm_x = V._get_eta_fwhm(G_fwhm_x, L_fwhm_x)
        eta_y, fwhm_y = V._get_eta_fwhm(G_fwhm_y, L_fwhm_y)
        eta = np.mean([eta_x, eta_y]) # might not be needed  
        return (eta * V.G.func_2d(xy, amp, x0, y0,
                                  fwhm_x, fwhm_y,
                                  theta, radians=radians)
                + (1 - eta) * V.L.func_2d(xy, amp, x0, y0,
                                          fwhm_x, fwhm_y,
                                          theta, radians=radians))
    
    @staticmethod
    # @njit
    def get_area(amp, x0, G_fwhm_x, L_fwhm_x):
        V = PseudoVoigtFunctions
        eta, fwhm_x = V._get_eta_fwhm(G_fwhm_x, L_fwhm_x)
        return (eta * V.G.get_area(amp, x0, fwhm_x)
                + (1 - eta) * V.L.get_area(amp, x0, fwhm_x))
    
    @staticmethod
    # @njit
    def get_volume(amp, x0, y0, G_fwhm_x, G_fwhm_y,
                   L_fwhm_x, L_fwhm_y, theta):
        V = PseudoVoigtFunctions
        eta_x, fwhm_x = V._get_eta_fwhm(G_fwhm_x, L_fwhm_x)
        eta_y, fwhm_y = V._get_eta_fwhm(G_fwhm_y, L_fwhm_y)
        eta = np.mean([eta_x, eta_y]) # might not be needed
        return (eta * V.G.get_volume(amp, x0, y0, fwhm_x, fwhm_y, theta)
               + (1 - eta) * V.L.get_volume(amp, x0, y0, fwhm_x, fwhm_y, theta))

    @staticmethod
    def get_1d_fwhm(amp, x0, G_fwhm_x, L_fwhm_x):
        # TODO: Go back and find the original references...
        V = PseudoVoigtFunctions
        eta, fwhm_x = V._get_eta_fwhm(G_fwhm_x, L_fwhm_x)
        return fwhm_x
    
    @staticmethod
    def get_2d_fwhm(amp, x0, y0, G_fwhm_x, G_fwhm_y, L_fwhm_x, L_fwhm_y, theta, radians=False):

        V = PseudoVoigtFunctions
        eta_x, fwhm_x = V._get_eta_fwhm(G_fwhm_x, L_fwhm_x)
        eta_y, fwhm_y = V._get_eta_fwhm(G_fwhm_y, L_fwhm_y)

        return SpotModelBase.get_2d_fwhm(amp, x0, y0, fwhm_x, fwhm_y, theta, radians=radians)
    

####################
### 1D Functions ###
#################### 


def voigt_1d(x, amp, y0, x0, sigma, gamma):
    # Actual voigt function. Difficult to define FWHM, so less practical for fitting
    return y0 + amp * (np.real(wofz((x - x0 + 1j * gamma) / sigma / np.sqrt(2))) / sigma / np.sqrt(2 * np.pi))


def parabola_1d(x, x0, a, b, y0):
    # b = 0 for symmetric parabola
    return a * ((x - x0) ** 2) + b * (x - x0) + y0


def polar_ellispe_1d(chi, a, e, phi):
    # chi is the azimuthal angle
    # a is the major axis length
    # e is the eccentricity
    # phi is the rotation angle...I think...
    return a * (1 - (e ** 2)) / (1 - e * np.cos(chi - phi))


#######################
### Mutli-Functions ###
#######################

# Works for 1d and 2d peaks.
def multi_peak_fitting(peak_function, *args, **kwargs):
    # TODO: Reewrite with inspect signature

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
        z += peak_function(x, *arg_dict.values(), **kwargs)
    
    return z

#########################
### Related Functions ###
#########################

# Not currently used.
def qualify_gaussian_2d_fit(r_squared, sig_threshold, guess_params, fit_params, return_reason=False):
    # Determine euclidean distance between intitial guess and peak fit. TODO change to angular misorientation
    offset_distance = np.sqrt((fit_params[1] - guess_params[1])**2 + (fit_params[2] - guess_params[2])**2)
    
    keep_fit = np.all([r_squared >= 0, # Qualify peak fit. Decent fits still do not have great r_squared, so cutoff is low
                       fit_params[0] > sig_threshold,
                       offset_distance < 50, # Useless when using bounded curve fit...
                       np.abs(fit_params[3]) < 0.5, # Qualifies sigma_x to not be too diffuse
                       np.abs(fit_params[4]) < 1.5, # Qualifies sigma_y to not be too diffuse
                       ])

    # Add qualifiers to determine if the gaussian fit is:
    # Sufficiently good fit (in r_squared)
    # Has sufficient amplitude to be worthwhile
    # Has not moved too far from initial guess position
    if return_reason:
        return keep_fit, offset_distance
    return keep_fit


def generate_bounds(p0, peak_function, tth_step=None, chi_step=None):
    
    # Get function input variable names, excluding 'self', 'x' or '(xy)' and any default arguments
    inputs = list(peak_function.__code__.co_varnames[:peak_function.__code__.co_argcount])
    if 'self' in inputs: inputs.remove('self')
    inputs = inputs[1:] # remove the x, or xy inputs
    if peak_function.__defaults__ is not None:
        inputs = inputs[:-len(peak_function.__defaults__)] # remove defualts

    if tth_step is None: tth_step = 0.02
    if chi_step is None: chi_step = 0.05

    tth_resolution, chi_resolution = 0.01, 0.01 # In degrees...
    tth_range = np.max([tth_resolution, tth_step]) * 3 # just a bit more wiggle room
    chi_range = np.max([chi_resolution, chi_step]) * 3 # just a bit more wiggle room

    low_bounds, upr_bounds = [], []
    for i in range(0, len(p0), len(inputs)):
        for j, arg in enumerate(inputs):
            if arg == 'amp':
                low_bounds.append(0)
                upr_bounds.append(p0[i + j] * 10) # guess intensity is probably not off by more than this
            elif arg == 'x0':
                low_bounds.append(p0[i + j] - tth_range) # Restricting peak shift based on guess position uncertainty
                upr_bounds.append(p0[i + j] + tth_range)
            elif arg == 'y0':
                low_bounds.append(p0[i + j] - chi_range) # Restricting peak shift based on guess position uncertainty
                upr_bounds.append(p0[i + j] + chi_range)
            elif 'fwhm_x' in arg:
                low_bounds.append(0.001)
                upr_bounds.append(p0[i + j] * 1.15) # should be based off of intrument resolution
            elif 'fwhm_y' in arg:
                low_bounds.append(0.001)
                upr_bounds.append(p0[i + j] * 3) # should be based off of intrument resolution
            elif arg == 'theta':
                low_bounds.append(-45) # Prevents spinning
                upr_bounds.append(45) # Only works for degrees...
            else:
                raise ValueError(f"{peak_function} inputs {arg} not defined by the generate_bounds function!")
            
    return [low_bounds, upr_bounds]


def get_confidence_interval(data, params, pcov, alpha=0.05):
    n = len(data)
    p = len(params)

    # Degrees of freedom
    dof = np.max(0, n - p)
    
    # student-t value for the dof and confidence level
    tval = stats.t.ppf(1 - float(alpha) / 2, dof)

    ci_list = []
    for variance in np.diag(pcov):
        sigma = variance**2
        ci_list.append(sigma * tval)

    return ci_list


# Confidence interval estimation
# Nonlinear curve fit with confidence interval
"""import numpy as np
from scipy.optimize import curve_fit
from scipy.stats.distributions import  t

x = np.array([0.5, 0.387, 0.24, 0.136, 0.04, 0.011])
y = np.array([1.255, 1.25, 1.189, 1.124, 0.783, 0.402])

# this is the function we want to fit to our data
def func(x, a, b):
    'nonlinear function in a and b to fit to data'
    return a * x / (b + x)

initial_guess = [1.2, 0.03]
pars, pcov = curve_fit(func, x, y, p0=initial_guess)

alpha = 0.05 # 95% confidence interval = 100*(1-alpha)

n = len(y)    # number of data points
p = len(pars) # number of parameters

dof = max(0, n - p) # number of degrees of freedom

# student-t value for the dof and confidence level
tval = t.ppf(1.0-alpha/2., dof) 

for i, p,var in zip(range(n), pars, np.diag(pcov)):
    sigma = var**0.5
    print 'p{0}: {1} [{2}  {3}]'.format(i, p,
                                  p - sigma*tval,
                                  p + sigma*tval)

import matplotlib.pyplot as plt
plt.plot(x,y,'bo ')
xfit = np.linspace(0,1)
yfit = func(xfit, pars[0], pars[1])
plt.plot(xfit,yfit,'b-')
plt.legend(['data','fit'],loc='best')
plt.savefig('images/nonlin-curve-fit-ci.png')"""