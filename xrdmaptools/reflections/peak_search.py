import dask
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter, maximum_filter, minimum_filter
from scipy.optimize import curve_fit
from tqdm import tqdm
from scipy.signal import find_peaks

# Local imports
# from ..utilities.math import (
#     circular_mask,
#     compute_r_squared,
#     rescale_array
#     arbitrary_center_of_mass
# )

from xrdmaptools.utilities.math import (
    rescale_array,
    arbitrary_center_of_mass,
    compute_r_squared
)
from xrdmaptools.utilities.utilities import (
    iterative_background
)
from xrdmaptools.reflections.SpotModels import (
    _load_peak_function
)


def peak_search(xrd,
                tth=None,
                plotme=False,
                spotmodel='gauss',
                **kwargs):

    kwargs.setdefault('prominence', 0.1)

    if 'distance' not in kwargs:
        if tth is not None:
            tth_step = np.mean(np.diff(tth))
            distance = 0.05 // tth_step # 0.05 deg
            kwargs.setdefault('distance', distance)
        
    if tth is None:
        tth = list(range(len(xrd)))
    tth = np.asarray(tth)

    norm_xrd = rescale_array(xrd, upper=1, arr_min=0, copy=True)
    norm_xrd = gaussian_filter(norm_xrd, sigma=3)
    rois = iterative_background(norm_xrd)
    norm_xrd[~rois] = 0

    peaks, _ = find_peaks(norm_xrd, **kwargs)

    spotmodel = _load_peak_function(spotmodel.lower())

    p0 = [np.mean(xrd[~rois])]
    for peak in peaks:
        peak_mask = slice(int(peak - distance),
                          int(peak + distance))
        aweights = rescale_array(xrd[peak_mask], lower=0, upper=1, copy=True)
        std_tth = np.sqrt(np.cov(tth[peak_mask],
                                 aweights=aweights))
        fwhm_tth = std_tth * 2 * np.sqrt(2 * np.log(2))

        p0.extend([
            np.max(xrd[peak_mask]),
            arbitrary_center_of_mass(xrd[peak_mask], tth[peak_mask])[0],
            fwhm_tth,
        ])
        if len(spotmodel.par_1d) > 3:
            p0.append(fwhm_tth)
    
    popt, pcov = curve_fit(spotmodel.multi_1d, tth, xrd, p0=p0)
    r_squared = compute_r_squared(xrd, spotmodel.multi_1d(tth, *popt))
    print(f'R squared is {r_squared}')

    if plotme:
        fig, ax = plt.subplots()
        ax.plot(tth, xrd, label='Data')
        ax.plot(tth, spotmodel.multi_1d(tth, *p0), label='Guess')
        ax.plot(tth, spotmodel.multi_1d(tth, *popt), label='Fit')
        ax.scatter(tth[peaks], xrd[peaks], c='r', s=10, marker='*', label='Peaks')
        ax.set_xlabel('Scattering Angle')
        ax.set_ylabel('Intensity')
        ax.legend()
        fig.show()

    return popt
