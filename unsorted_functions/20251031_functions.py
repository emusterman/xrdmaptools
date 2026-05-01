import numpy as np
import os
from skimage import io
from skimage.restoration import rolling_ball
import matplotlib.pyplot as plt
import h5py
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.stats import t


from xrdmaptools.reflections.SpotModels import (
    GaussianFunctions,
    LorentzianFunctions,
    PseudoVoigtFunctions
)
from xrdmaptools.utilities.math import rescale_array
from xrdmaptools.utilities.background_estimators import masked_bruckner_image_background



def plot_fit(tth, raw, fit, guess=None, title=None):
    fig, ax = plt.subplots()

    ax.plot(tth, raw, '.-', c='r', label='measured')
    ax.plot(tth, fit, c='k', label='fit')
    if guess is not None:
        ax.plot(tth, guess, '--', c='k', alpha=0.5, label='guess')

    ax.plot(tth, fit - raw - 10, c='k')

    if title is not None:
        ax.set_title(title)

    ax.legend()
    fig.show()



def find_and_fit_peaks(tth,
                       intensity,
                       normalized_prominence=0.25, # In percent
                       minimum_peak_width=0.05, # In tth units
                       SpotModel=GaussianFunctions,
                       plot_results=True,
                       plot_title='Fit'):

    intensity = rescale_array(intensity.copy(), arr_min=0, upper=100)

    width = int(round(minimum_peak_width / np.median(np.diff(tth))))
    # bkg = rolling_ball(intensity, radius=width)
    bkg = masked_bruckner_image_background(intensity.reshape(1, -1), 25, 1000, np.ones((1, len(intensity)), dtype=np.bool_), min_prominence=0.0001)[0]
    intensity = intensity - bkg

    peaks = find_peaks(intensity,
                       prominence=normalized_prominence,
                       width=width,
                       )[0]
    
    p0 = [np.min(intensity)]
    for i in range(len(peaks)):
        peak = peaks[i]

        # A
        p0.append(intensity[peak])
        
        # x0
        p0.append(tth[peak])

        # fwhm
        wind = width * 4
        tth_wind = tth[peak - wind : peak + wind]
        aweights = rescale_array(intensity[peak - wind : peak + wind].copy(), lower=0, upper=1)
        std = np.sqrt(np.cov(tth_wind, aweights=aweights))
        fwhm = std * 2 * np.sqrt(2 * np.log(2))

        p0.append(fwhm)
        if len(SpotModel.par_1d) == 4:
            p0[-1] = p0[-1] / 2
            p0.append(fwhm / 2)

    # Fitting    
    popt, pcov = curve_fit(SpotModel.multi_1d, tth, intensity, p0=p0)

    if plot_results:
        plot_fit(tth, intensity + bkg, SpotModel.multi_1d(tth, *popt) + bkg, title=plot_title)

    return popt, p0


# Good enough for now
def fit_cubic_lattice(a0, hkls, d_meas, wavelength,
                      con_int=0.99):

    def cube_lattice(hkls, a):
        return [a / np.sqrt(sum([h**2 for h in hkl])) for hkl in hkls]

    d_calc = cube_lattice(hkls, a0)

    popt, pcov = curve_fit(cube_lattice, hkls, d_meas, p0=[a0])
    se = np.sqrt(pcov[0]) # Approximate. Does not consider error in d_meas
    me = se * t.ppf((1 + con_int) / 2, len(d_meas) - 1)

    return popt[0], me[0]


def process_pattern(scan_id, wd, energy=18, SpotModel=PseudoVoigtFunctions):
    
    # Load data
    q, tth, intensity = np.genfromtxt(f'{wd}max_1D_integrations/scan{scan_id}_max_1D_integration.txt')

    # Fit and plot data
    popt, p0 = find_and_fit_peaks(tth, intensity, SpotModel=SpotModel, plot_title=f'Scan {scan_id} Fit')

    amps = popt[1::len(SpotModel.par_1d)]
    tths = popt[2::len(SpotModel.par_1d)]
    amp_max = sorted(amps, reverse=True)[3]

    d_meas = np.asarray(tth_2_d(np.asarray(tths)[amps >= amp_max], wavelength=energy_2_wavelength(energy)))

    hkls = [[1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 0, 0]]
    a, da = fit_cubic_lattice(2.24, hkls, d_meas, energy_2_wavelength(energy))

    return a, da

