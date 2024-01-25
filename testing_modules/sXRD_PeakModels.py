import numpy as np
from scipy.special import wofz
import scipy.constants as constants


############################
### Classes of Functions ###
############################


def _load_peak_function(name):

    for PeakModel in [GaussianFunctions,
                     LorentzianFunctions,
                     PseudoVoigtFunctions]:
        if name in [PeakModel.name, PeakModel.abbr]:
            return PeakModel
        
    print(f'Given peak name: {name} is not yet supported.')


class PeakFunctionBase():
    # Must define self.func_1d() and self.func_2d()
    # Relies on standard argument input order

    # Useful parameters
    par_1d = ['amp', 'x0', 'fwhm']
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

    def generate_guess():
        raise NotImplementedError
    
    def generate_bounds():
        raise NotImplementedError

    def qualify_fit():
        raise NotImplementedError


class GaussianFunctions(PeakFunctionBase):
    # Useful parameters
    name = 'Gaussian'
    abbr = 'gauss'
    #par_1d = ['amp', 'x0', 'fwhm']
    #par_2d = ['amp', 'x0', 'y0', 'fwhm_x', 'fwhm_y', 'theta']
    par_3d = ['amp', 'x0', 'y0', 'z0',
              'fwhm_x', 'fwhm_y','fwhm_z',
              'theta', 'phi']

    @staticmethod
    def func_1d(x, amp, x0, fwhm):
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        return amp * np.exp(-(x - x0)**2 / (2 * sigma**2))
    
    @staticmethod
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
    def get_area(amp, x0, fwhm):
        # Returns area under 1d gaussian function
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        return amp * np.sqrt(2 * np.pi) * sigma
    
    @staticmethod
    def get_volume(amp, x0, y0, fwhm_x, fwhm_y, theta, radians=False):
        # Returns volume under 2d gaussian function
        sigma_x = fwhm_x / (2 * np.sqrt(2 * np.log(2)))
        sigma_y = fwhm_y / (2 * np.sqrt(2 * np.log(2)))
        return amp * 2 * np.pi * sigma_x * sigma_y

    #@staticmethod
    #def get_1d_fwhm(amp, x0, fwhm):
        # sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        # return 2 * np.sqrt(2 * np.log(2)) * sigma
    #    return fwhm
    
    #@staticmethod
    #def get_2d_fwhm(amp, x0, y0, fwhm_x, fwhm_y, theta, radians=False):
    #            
    #    if not radians:
    #        theta = np.radians(theta)
    #
    #    fwhm_rx = fwhm_x * np.cos(-theta) - fwhm_y * np.sin(-theta)
    #    fwhm_ry = fwhm_x * np.sin(-theta) + fwhm_y * np.cos(-theta)
    #
    #    return fwhm_rx, fwhm_ry, fwhm_x, fwhm_y

    def get_3d_fwhm():
        raise NotImplementedError()


class LorentzianFunctions(PeakFunctionBase):
    # Useful parameters
    name = 'Lorentzian'
    abbr = 'lorentz'
    #par_1d = ['amp', 'x0', 'fwhm']
    #par_2d = ['amp', 'x0', 'y0', 'fwhm_x', 'fwhm_y', 'theta']

    @staticmethod
    def func_1d(x, amp, x0, fwhm):
        gamma = 0.5 * fwhm
        return amp * (gamma**2 / ((x - x0)**2 + gamma**2))
    
    # Not sure about this one...
    @staticmethod
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
    def get_area(amp, x0, fwhm):
        gamma = 0.5 * fwhm
        return amp * np.pi * gamma
    
    @staticmethod
    def get_volume(amp, x0, y0, fwhm_x, fwhm_y, theta):
        # Current 2D Lorentzian function does not have a finite volume

        # We approximate volume with a 2D gaussian based on equal areas of the major axes
        gamma_x = 0.5 * fwhm_x
        gamma_y = 0.5 * fwhm_y
        sigma_x = gamma_x * np.sqrt(np.pi / 2)
        sigma_y = gamma_y * np.sqrt(np.pi / 2)

        return amp * 2 * np.pi * sigma_x * sigma_y

    
    #@staticmethod
    #def get_1d_fwhm(amp, x0, fwhm):
        #gamma = 0.5 * fwhm
        #return 2 * gamma
    #    return fwhm
    
    #@staticmethod
    #def get_2d_fwhm(amp, x0, y0, fwhm_x, fwhm_y, theta, radians=False):
    #            
    #    if not radians:
    #        theta = np.radians(theta)
    #
    #    fwhm_rx = fwhm_x * np.cos(-theta) - fwhm_y * np.sin(-theta)
    #    fwhm_ry = fwhm_x * np.sin(-theta) + fwhm_y * np.cos(-theta)
    #
    #    return fwhm_rx, fwhm_ry, fwhm_x, fwhm_y


class PseudoVoigtFunctions(PeakFunctionBase):
    # Note: This is the pseudo-voigt profile!
    # Useful parameters
    name = 'PseudoVoigt'
    abbr = 'p_voigt'
    par_1d = ['amp', 'x0', 'G_fwhm', 'L_fwhm']
    par_2d = ['amp', 'x0', 'y0',
              'G_fwhm_x', 'G_fwhm_y',
              'L_fwhm_x', 'L_fwhm_y',
              'theta']
    
    G = GaussianFunctions
    L = LorentzianFunctions

    @staticmethod
    def _get_eta_fwhm(G_fwhm, L_fwhm):
        # Convolved fwhm
        fwhm = ((G_fwhm**5
                 + 2.69269 * G_fwhm**4 * L_fwhm**1
                 + 2.42843 * G_fwhm**3 * L_fwhm**2
                 + 4.47163 * G_fwhm**2 * L_fwhm**3
                 + 0.07842 * G_fwhm**1 * L_fwhm**4
                 + L_fwhm**5
                 )**(1 / 5))
        # Fraction
        eta = (1.36603 * (L_fwhm / fwhm)
               -0.47719 * (L_fwhm / fwhm)**2
               +0.11116 * (L_fwhm / fwhm)**3)
        
        return eta, fwhm
    
    @staticmethod
    def func_1d(x, amp, x0, G_fwhm, L_fwhm):
        V = PseudoVoigtFunctions
        eta, fwhm = V._get_eta_fwhm(G_fwhm, L_fwhm)
        return (eta * V.G.func_1d(x, amp, x0, fwhm)
                + (1 - eta) * V.L.func_1d(x, amp, x0, fwhm))
    
    # Not sure about this one...
    @staticmethod
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
    def get_area(amp, x0, G_fwhm, L_fwhm):
        V = PseudoVoigtFunctions
        eta, fwhm = V._get_eta_fwhm(G_fwhm, L_fwhm)
        return (eta * V.G.get_area(amp, x0, fwhm)
                + (1 - eta) * V.L.get_area(amp, x0, fwhm))
    
    @staticmethod
    def get_volume(amp, x0, y0, G_fwhm_x, G_fwhm_y,
                   L_fwhm_x, L_fwhm_y, theta):
        V = PseudoVoigtFunctions
        eta_x, fwhm_x = V._get_eta_fwhm(G_fwhm_x, L_fwhm_x)
        eta_y, fwhm_y = V._get_eta_fwhm(G_fwhm_y, L_fwhm_y)
        eta = np.mean([eta_x, eta_y]) # might not be needed
        return (eta * V.G.get_volume(amp, x0, y0, fwhm_x, fwhm_y, theta)
               + (1 - eta) * V.L.get_volume(amp, x0, y0, fwhm_x, fwhm_y, theta))

    @staticmethod
    def get_1d_fwhm(amp, x0, G_fwhm, L_fwhm):
        # TODO: Go back and find the original references...
        V = PseudoVoigtFunctions
        eta, fwhm = V._get_eta_fwhm(G_fwhm, L_fwhm)
        return eta
    
    @staticmethod
    def get_2d_fwhm(amp, x0, y0, G_fwhm_x, G_fwhm_y, L_fwhm_x, L_fwhm_y, theta, radians=False):

        V = PseudoVoigtFunctions
        eta_x, fwhm_x = V._get_eta_fwhm(G_fwhm_x, L_fwhm_x)
        eta_y, fwhm_y = V._get_eta_fwhm(G_fwhm_y, L_fwhm_y)

        return PeakFunctionBase.get_2d_fwhm(amp, x0, y0, fwhm_x, fwhm_y, theta, radians=radians)
        
    

class old_PseudoVoigtFunctions(PeakFunctionBase):
    # Note: This is the pseudo-voigt profile!
    # Useful parameters
    name = 'PseudoVoigt'
    abbr = 'p_voigt'
    par_1d = ['amp', 'x0', 'G_fwhm', 'L_fwhm', 'eta']
    par_2d = ['amp', 'x0', 'y0',
              'G_fwhm_x', 'G_fwhm_y',
              'L_fwhm_x', 'L_fwhm_y',
              'theta', 'eta']
    
    G = GaussianFunctions()
    L = LorentzianFunctions()
    
    @staticmethod
    def func_1d(x, amp, x0, G_fwhm, L_fwhm, eta):
        V = PseudoVoigtFunctions
        return (eta * V.G.func_1d(x, amp, x0, G_fwhm)
                + (1 - eta) * V.L.func_1d(x, amp, x0, L_fwhm))
    
    # Not sure about this one...
    @staticmethod
    def func_2d(xy, amp, x0, y0, G_fwhm_x, G_fwhm_y,
                L_fwhm_x, L_fwhm_y, theta, eta, radians=False):
        V = PseudoVoigtFunctions        
        return (eta * V.G.func_2d(xy, amp, x0, y0,
                                  G_fwhm_x, G_fwhm_y,
                                  theta, radians=radians)
                + (1 - eta) * V.L.func_2d(xy, amp, x0, y0,
                                          L_fwhm_x, L_fwhm_y,
                                          theta, radians=radians))
    
    @staticmethod
    def get_area(amp, x0, G_fwhm, L_fwhm, eta):
        V = PseudoVoigtFunctions
        return (eta * V.G.get_area(amp, x0, G_fwhm)
                + (1 - eta) * V.L.get_area(amp, x0, L_fwhm))
    
    @staticmethod
    def get_volume(amp, x0, y0, G_fwhm_x, G_fwhm_y,
                   L_fwhm_x, L_fwhm_y, theta, eta):
        V = PseudoVoigtFunctions
        return (eta * V.G.get_volume(amp, x0, y0, G_fwhm_x, G_fwhm_y, theta)
               + (1 - eta) * V.L.get_volume(amp, x0, y0, L_fwhm_x, L_fwhm_y, theta))

    @staticmethod
    def get_1d_fwhm(amp, x0, G_fwhm, L_fwhm, eta):
        # All from the wikipedia article.
        # TODO: Go back and find the original references...

        '''fwhm = ((G_fwhm**5
                 + 2.69269 * G_fwhm**4 * L_fwhm**1
                 + 2.42843 * G_fwhm**3 * L_fwhm**2
                 + 4.47163 * G_fwhm**2 * L_fwhm**3
                 + 0.07842 * G_fwhm**1 * L_fwhm**4
                 + L_fwhm**5
                 )**(1 / 5))

        eta = (1.36603 * (L_fwhm / fwhm)
               -0.47719 * (L_fwhm / fwhm)**2
               +0.11116 * (L_fwhm / fwhm)**3)'''


        # True Voigt FWHM poor approximation (error = 1.2%)
        #FWHM = FWHM_L / 2 + np.sqrt(((FWHM_L**2) / 4) + (FWHM_G**2))

        # True Voigt FWHM better approximation (error = 0.2%)
        #FWHM = 0.5346 * FWHM_L + np.sqrt((0.2166 * (FWHM_L**2)) + (FWHM_G**2))

        # Maybe true Voigt FWHM approximation (unknown error)
        #

        # Not sure this is correct...
        return eta * G_fwhm + (1 - eta) * L_fwhm
    
    @staticmethod
    def get_2d_fwhm(amp, x0, y0, G_fwhm_x, G_fwhm_y, L_fwhm_x, L_fwhm_y, theta, eta, radians=False):
        V = PseudoVoigtFunctions
        fwhm_x = V.get_1d_fwhm(amp, x0, G_fwhm_x, L_fwhm_x, eta)
        fwhm_y = V.get_1d_fwhm(amp, y0, G_fwhm_y, L_fwhm_y, eta)

        return PeakFunctionBase.get_2d_fwhm(amp, x0, y0, fwhm_x, fwhm_y, theta, radians=radians)

        # Gaussian and Lorentzian grab from the same
        return V.G.get_2d_fwhm(amp, x0, y0, fwhm_x, fwhm_y, theta, radians=radians)
        
        #if not radians:
        #    theta = np.radians(theta)
        #
        #fwhm_rx = fwhm_x * np.cos(-theta) - fwhm_y * np.sin(-theta)
        #fwhm_ry = fwhm_x * np.sin(-theta) + fwhm_y * np.cos(-theta)
        #
        #return fwhm_rx, fwhm_ry, fwhm_x, fwhm_y

    #@staticmethod
    #def get_2d_fwhm(amp, x0, y0, sigma_x, sigma_y, gamma_x, gamma_y, theta, eta, radians=False):
    #   
    #    if not radians:
    #        theta = np.radians(theta)
    #
    #    gamma_rx = gamma_x * np.cos(-theta) - gamma_y * np.sin(-theta)
    #    gamma_ry = gamma_x * np.sin(-theta) + gamma_y * np.cos(-theta)
    #    sigma_rx = sigma_x * np.cos(-theta) - sigma_y * np.sin(-theta)
    #    sigma_ry = sigma_x * np.sin(-theta) + sigma_y * np.cos(-theta)
    #    
    #    # 1D cuts of the rotated profile
    #    V = PseudoVoigtFunctions
    #    fwhm_rx = V.get_1d_fwhm(amp, x0, sigma_rx, gamma_rx, eta)
    #    fwhm_ry = V.get_1d_fwhm(amp, y0, sigma_ry, gamma_ry, eta)
    #    fwhm_x = V.get_1d_fwhm(amp, x0, sigma_x, gamma_x, eta)
    #    fwhm_y = V.get_1d_fwhm(amp, y0, sigma_y, gamma_y, eta)
    #
    #    return fwhm_rx, fwhm_ry, fwhm_x, fwhm_y
    




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