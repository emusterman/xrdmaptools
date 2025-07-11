import dask
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter, maximum_filter, minimum_filter
from scipy.optimize import curve_fit
import scipy.stats as st
from skimage.measure import label, find_contours
from skimage.segmentation import expand_labels
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from tqdm.dask import TqdmCallback
from tqdm import tqdm
import dask.array as da
from collections import OrderedDict
from skimage.feature import peak_local_max
from scipy.signal import find_peaks

# Local imports
from ..utilities.math import (
    circular_mask,
    compute_r_squared,
    rescale_array
    arbitrary_center_of_mass
)


def peak_search():
    raise NotImplementedError()
    from skimage.feature import peak_local_max
    from scipy.signal import find_peaks

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=200)

    plot_int = rescale_array(np.max(test.map.integrations, axis=(0, 1)))
    #peaks = peak_local_max(plot_int,
    #                       threshold_rel=0.05)

    peaks, _ = find_peaks(plot_int, prominence=0.01, distance=4)