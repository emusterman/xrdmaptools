from scipy.special import wofz
import numpy as np

####################
### 1D Functions ###
####################


def gaussian_1d(x, a, y0, x0, sigma):
    # FWHM = 2 * np.sqrt(2 * np.log(2)) * sigma
    return y0 + a * (np.exp(-(x - x0)**2 / (2 * sigma**2)))


def lorentzian_1d(x, a, y0, x0, gamma):
    # FWHM = 2 * gamma
    return y0 + a * (gamma / np.pi / ((x - x0)**2 + gamma**2))


def voigt_1d(x, a, y0, x0, sigma, gamma):
    # Actual voigt function. Difficult to define FWHM, so less practical for fitting
    return y0 + a * (np.real(wofz((x - x0 + 1j * gamma) / sigma / np.sqrt(2))) / sigma / np.sqrt(2 * np.pi))


def psuedo_voigt_1d(x, a, y0, x0, sigma, gamma):
    fg = 2 * sigma * np.sqrt(2 * np.log(2))
    fl = 2 * gamma
    f = ((fg ** 5) + 2.69269 * (fg ** 4) * fl + 2.42843 * (fg ** 3) * (fl ** 2) + 4.47163 * (fg ** 2) * (fl ** 3) + 0.07842 * fg * (fl **4) + (fl ** 5)) ** (1/5)
    eta = 1.36603 * (fl / f) - 0.47719 * ((fl / f) ** 2) + 0.11116 * ((fl / f) ** 3)
    return y0 + a * ((eta * lorentzian_1d(x, 1, 0, x0, gamma)) + ((1 - eta) * gaussian_1d(x, 1, 0, x0, sigma)))


def quasi_voigt_1d(x, a, y0, x0, sigma, gamma, eta):
    # psuedo_voigt_1d function with eta as an input. Might combine in future iterations
    return (eta * lorentzian_1d(x, a, y0, x0, gamma)) + ((1 - eta) * gaussian_1d(x, a, y0, x0, sigma))


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