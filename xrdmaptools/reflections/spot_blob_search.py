import dask
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter,
    median_filter,
    uniform_filter,
    maximum_filter,
    minimum_filter
    )
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

# Local imports
from ..utilities.math import circular_mask, compute_r_squared
from ..utilities.utilities import arbitrary_center_of_mass, label_nearest_spots
from ..utilities.image_corrections import rescale_array
from ..geometry.geometry import (
    estimate_image_coords,
    estimate_polar_coords,
    modular_azimuthal_shift,
    modular_azimuthal_reshift
)

from .SpotModels import generate_bounds


def resize_blobs(blob_image, distance=0):

    if distance is None or distance == 0:
        return blob_image

    elif distance > 0:
        distances = distance_transform_edt(blob_image == 0)
        resize_mask = distances <= distance
    elif distance < 0:
        distances = distance_transform_edt(blob_image == 1)
        resize_mask = distances <= -distance
        resize_mask = ~resize_mask

    return resize_mask


def blob_search(scaled_image,
                mask=None,
                threshold_method='gaussian',
                multiplier=5,
                size=3,
                expansion=None):
    
    if mask is None or np.all(mask):
       temp_mask = np.ones_like(scaled_image, dtype=np.bool_)
    else:
        temp_mask = mask

    # Estimate individual image offset and noise
    pseudo_blob_mask = scaled_image < 0.01
    pseudo_blob_mask *= temp_mask
    image_noise = np.std(scaled_image[pseudo_blob_mask])
    image_offset = np.median(scaled_image[pseudo_blob_mask])

    # Mask image
    mask_thresh = image_offset + multiplier * image_noise

    # Setup filter for the images
    # Works okay, could take no inputs...
    if str(threshold_method).lower() in ['gaussian', 'gauss']:
        image_filter = lambda image : gaussian_filter(image, sigma=size)
    # Works best
    elif str(threshold_method).lower() in ['minimum', 'min']:
        image_filter = lambda image : minimum_filter(gaussian_filter(image, sigma=1), size=size)
    # Gaussian is better and faster
    elif str(threshold_method).lower() in ['median', 'med']:
        image_filter = lambda image : median_filter(image, size=size)
    else:
        raise ValueError('Unknown threshold method requested.')
    
    if mask is not None:
        # Smooth image to reduce noise contributions
        zero_image = np.copy(scaled_image)
        zero_image[~temp_mask] = 0 # should be redundant
        zero_filter = image_filter(zero_image)
        
        div_image = np.ones_like(scaled_image)
        div_image[~temp_mask] = 0
        filter_div = image_filter(div_image)
        
        blurred_image = zero_filter / filter_div
        # Clean up some NaNs from median filters
        temp_mask[np.isnan(blurred_image)] = False
        # Clear image from masked values. Should not matter....
        blurred_image[~temp_mask] = 0

        # Create mask for peak search
        blob_mask = blurred_image > mask_thresh
        blob_mask *= temp_mask
    else:
        zero_image = np.copy(scaled_image)
        zero_image[~temp_mask] = 0 # should be redundant
        blurred_image = image_filter(zero_image)

        blob_mask = blurred_image > mask_thresh

    # Exclude edges from analysis
    # They can be erroneous from filters
    blob_mask[0] = 0
    blob_mask[-1] = 0
    blob_mask[:, 0] = 0
    blob_mask[:, -1] = 0

    if expansion is not None and expansion != 0:
        blob_mask = resize_blobs(blob_mask, distance=expansion)

    return blob_mask, blurred_image


def spot_search(scaled_image,
                    blob_mask,
                    blurred_image,
                    min_distance=3,
                    plotme=False):        
        
    spots = peak_local_max(blurred_image,
                #threshold_rel=image_noise,
                min_distance=min_distance, # in pixel units...
                labels=blob_mask,
                num_peaks_per_label=np.inf)
        
    if plotme:
        fig, ax = plt.subplots(1, 1, figsize=(9, 6), dpi=200)

        im = ax.imshow(scaled_image,
                       vmin=0,
                       vmax=0.1, # images should be scaled
                       aspect='auto')
        fig.colorbar(im, ax=ax)
        ax.scatter(spots[:, 1], spots[:, 0], s=1, c='r')

        fig.show()
    
    return spots, blob_mask, blurred_image


def blob_spot_search(scaled_image,
                     mask=None,
                     threshold_method='gaussian',
                     multiplier=5,
                     size=3,
                     min_distance=3,
                     expansion=None,
                     plotme=False):
    
    (blob_mask,
     blurred_image) = blob_search(
        scaled_image,
        mask=mask,
        threshold_method=threshold_method,
        multiplier=multiplier,
        size=size,
        expansion=0 # Do not expand until after spot search!
        )
    
    (spots,
     blob_mask,
     blurred_image) = spot_search(
        scaled_image,
        blob_mask,
        blurred_image,
        min_distance=min_distance,
        plotme=plotme
        )
    
    # Expand blobs after spot search. Makes spot search more selective
    if expansion is not None and expansion != 0:
        blob_mask = resize_blobs(blob_mask, distance=expansion)

    return spots, blob_mask, blurred_image


# Must take scaled images!!!
def old_spot_search(scaled_image,
                mask=None,
                threshold_method='gaussian',
                multiplier=5,
                size=3,
                min_distance=3,
                expansion=None,
                plotme=False):
    
    '''
    Returns spots in image coordinates.
    '''

    if mask is None:
       mask = (scaled_image != 0)

    # Estimate individual image offset and noise
    pseudo_peak_mask = scaled_image < 0.01
    pseudo_peak_mask *= mask
    image_noise = np.std(scaled_image[pseudo_peak_mask])
    image_offset = np.median(scaled_image[pseudo_peak_mask])

    # Mask image
    mask_thresh = image_offset + multiplier * image_noise

    # Setup filter for the images
    # Works okay, could take no inputs...
    if str(threshold_method).lower() in ['gaussian', 'gauss']:
        image_filter = lambda image : gaussian_filter(image, sigma=size)
    # Works best
    elif str(threshold_method).lower() in ['minimum', 'min']:
        image_filter = lambda image : minimum_filter(gaussian_filter(image, sigma=1), size=size)
    # Gaussian is better and faster
    elif str(threshold_method).lower() in ['median', 'med']:
        image_filter = lambda image : median_filter(image, size=size)
    else:
        raise ValueError('Unknown threshold method requested.')
    
    # Smooth image to reduce noise contributions
    zero_image = np.copy(scaled_image)
    zero_image[~mask] = 0 # should be redundant
    gauss_zero = image_filter(zero_image)

    div_image = np.ones_like(scaled_image)
    div_image[~mask] = 0
    gauss_div = image_filter(div_image)

    thresh_img = gauss_zero / gauss_div
    # Clean up some NaNs from median filters
    mask[np.isnan(thresh_img)] = False
    # Clear image from masked values. Should not matter....
    thresh_img[~mask] = 0

    # Create mask for peak search
    peak_mask = thresh_img > mask_thresh
    peak_mask *= mask

    # Exclude edges from analysis
    # They can be erroneous from filters
    peak_mask[0] = 0
    peak_mask[-1] = 0
    peak_mask[:, 0] = 0
    peak_mask[:, -1] = 0

    spots = peak_local_max(thresh_img,
                           #threshold_rel=image_noise,
                           min_distance=min_distance, # in pixel units...
                           labels=peak_mask,
                           num_peaks_per_label=np.inf)
    
    # Expand blobs for better fitting
    if expansion is not None:
        peak_mask = expand_labels(peak_mask, distance=expansion)
    
    if plotme:
        fig, ax = plt.subplots(1, 1, figsize=(9, 6), dpi=200)

        im = ax.imshow(scaled_image * mask, vmin=np.min([mask_thresh, 0]), vmax=10 * mask_thresh, aspect='auto')
        fig.colorbar(im, ax=ax)
        ax.scatter(spots[:, 1], spots[:, 0], s=1, c='r')

        fig.show()

    return spots, peak_mask, thresh_img


# Parallelized function to find only blobs in 4D images
def find_blobs(images,
               mask=None,
               threshold_method='gaussian',
               multiplier=5,
               size=3,
               expansion=None):

    # Dask wrapper to work wtih spot search function
    @dask.delayed
    def dask_blob_search(image):
        blob_mask, _ = blob_search(image,
                            mask=mask,
                            threshold_method=threshold_method,
                            multiplier=multiplier,
                            size=size,
                            expansion=expansion)
        return blob_mask, # thesh_image

    # Create list of delayed tasks
    map_shape = images.shape[:-2]
    num_images = np.prod(map_shape)
    delayed_list = []
    for index in range(num_images):
        indices = np.unravel_index(index, map_shape)
        image = images[indices]

        # Convert dask to numpy arrays
        if isinstance(image, da.core.Array):
            image = image.compute()

        output = dask_blob_search(image)
        delayed_list.append(output)

    # Process delayed tasks with callback
    print('Searching images for blobs...')
    with TqdmCallback(tqdm_class=tqdm):
        blob_mask_list = dask.compute(*delayed_list)
    
    return blob_mask_list

# Parallelized function to find blobs and spots in 4D images
def find_blobs_spots(images,
                     mask=None,
                     threshold_method='gaussian',
                     multiplier=5,
                     size=3,
                     expansion=None,
                     min_distance=3):

    # Dask wrapper to work wtih spot search function
    @dask.delayed
    def dask_blob_spot_search(image):
        spots, blob_mask, _ = blob_spot_search(
                            image,
                            mask=mask,
                            threshold_method=threshold_method,
                            multiplier=multiplier,
                            size=size,
                            min_distance=min_distance,
                            expansion=expansion,
                            plotme=False)
        return spots, blob_mask

    # Create list of delayed tasks
    map_shape = images.shape[:-2]
    num_images = np.prod(map_shape)
    delayed_list = []
    for index in range(num_images):
        indices = np.unravel_index(index, map_shape)
        image = images[indices]

        # Convert dask to numpy arrays
        if isinstance(image, da.core.Array):
            image = image.compute()

        output = dask_blob_spot_search(image)
        delayed_list.append(output)

    # Process delayed tasks with callback
    print('Searching images for blobs and spots...')
    with TqdmCallback(tqdm_class=tqdm):
        proc_list = dask.compute(*delayed_list)

    # Separate outputs of original spot search function
    spot_list, blob_mask_list = [], []
    for proc in proc_list:
        spot_list.append(proc[0])
        blob_mask_list.append(proc[1])
    
    return spot_list, blob_mask_list


# Old parallelized function for finding blobs and spots in imagemap
# Deprecated
def old_find_spots(imagemap,
               mask=None,
               threshold_method='gaussian',
               multiplier=5,
               size=3,
               expansion=None):

    # Converient way to iterate through image map
    iter_image = imagemap.images.reshape(imagemap.num_images, *imagemap.images.shape[-2:])

    # Dask wrapper to work wtih spot search function
    @dask.delayed
    def dask_spot_search(image,
                         mask=mask,
                         threshold_method=threshold_method,
                         multiplier=multiplier,
                         size=size,
                         expansion=expansion):
        
        spots, spot_mask, thresh_image = blob_spot_search(image,
                                                mask=mask,
                                                threshold_method=threshold_method,
                                                multiplier=multiplier,
                                                size=size,
                                                expansion=expansion)
        return spots, spot_mask, # thesh_image

    # Create list of delayed tasks
    delayed_list = []
    for image in iter_image:
        # Convert dask to numpy arrays
        if isinstance(image, da.core.Array):
            image = image.compute()

        output = dask_spot_search(image, mask=mask, threshold_method=threshold_method,
                                                multiplier=multiplier, size=size)
        delayed_list.append(output)

    # Process delayed tasks with callback
    print('Searching images for spots...')
    with TqdmCallback(tqdm_class=tqdm):
        proc_list = dask.compute(*delayed_list)

    # Separate outputs of original spot search function
    spot_list, mask_list = [], []
    #thresh_image_list = []
    for proc in proc_list:
        spot_list.append(proc[0])
        mask_list.append(proc[1])
        #thresh_image_list.append(proc[2])
    
    return spot_list, mask_list, #thresh_image_list


def spot_stats(spot, image, tth_arr, chi_arr, radius=5):
    if tth_arr is None or chi_arr is None:
        raise ValueError('Cannot estimate all spot stats without angular values.')
    
    # Shift azimuthal discontinuities
    chi_arr, max_arr, _ = modular_azimuthal_shift(chi_arr)

    # spot should be iterable of [img_x, img_y]
    spot_mask = circular_mask(image.shape, [*spot], radius)
    spot_image = image * spot_mask

    height = np.max(spot_image) - np.min(spot_image)
    img_x = int(spot[0])
    img_y = int(spot[1])
    tth = tth_arr[*spot]
    chi = chi_arr[*spot]
    intensity = np.sum(spot_image)

    center = list(arbitrary_center_of_mass(spot_image, tth_arr, chi_arr))

    aweights = rescale_array(image[spot_mask], lower=0, upper=1)
    std_tth = np.sqrt(np.cov(tth_arr[spot_mask],
                                aweights=aweights))
    fwhm_tth = std_tth * 2 * np.sqrt(2 * np.log(2))
    
    std_chi = np.sqrt(np.cov(chi_arr[spot_mask],
                                aweights=aweights))
    fwhm_chi = std_chi * 2 * np.sqrt(2 * np.log(2))

    # Reshift absolute azimuthal units
    chi, center[1] = modular_azimuthal_reshift([chi, center[1]], max_arr=max_arr)
    
    # 'guess_height', 'img_x', 'img_y',
    # 'guess_tth', 'guess_chi', 'guess_cen_tth', 'guess_cen_chi',
    # 'guess_fwhm_tth', 'guess_fwhm_chi', 'guess_int'
    return [height, img_x, img_y,
            tth, chi, center[0], center[1],
            fwhm_tth, fwhm_chi, intensity]


def find_spot_stats(imagemap, spot_list, tth_arr, chi_arr, radius=5):
    # Convenient way to iterate through image map
    iter_image = imagemap.images.reshape(imagemap.num_images, *imagemap.images.shape[-2:])

    # Dask wrapper to work wtih spot search function
    @dask.delayed
    def dask_spot_stats(spot, image, tth_arr, chi_arr, radius=radius):
        stats = spot_stats(spot, image, tth_arr, chi_arr, radius=radius)
        return stats

    delayed_list = []
    num_spots = np.sum([len(spots) for spots in spot_list])
    for spots, image in zip(spot_list, iter_image):
        # Convert dask to numpy arrays
        if isinstance(image, da.core.Array):
            image = image.compute()

        delayed_stats = []
        if len(spots) > 0: # Ignore images without spots
            for spot in spots:
                stats = dask_spot_stats(spot, image, tth_arr, chi_arr, radius=radius)
                delayed_stats.append(stats)
        delayed_list.append(delayed_stats)

    print('Estimating spot characteristics...')
    with TqdmCallback(tqdm_class=tqdm):
        stats_list = dask.compute(*delayed_list)
    
    return stats_list


def make_stat_dict(stat_list, map_shape):

    # This might be backwards
    map_x, map_y = np.unravel_index(range(len(stat_list)), map_shape)
    stat_keys = ['guess_height', 'img_x', 'img_y',
                 'guess_tth', 'guess_chi', 'guess_cen_tth', 'guess_cen_chi',
                 'guess_fwhm_tth', 'guess_fwhm_chi', 'guess_int']

    #stat_dict = dict(zip(stat_keys, [[] for i in range(len(stat_keys))]))
    stat_dict = OrderedDict(zip(stat_keys, [[] for i in range(len(stat_keys))]))
    stat_dict['map_x'] = []
    stat_dict['map_y'] = []

    for i, spots in enumerate(stat_list):
        if len(spots) > 0:
            for spot in spots:
                stat_dict['map_x'].append(map_x[i])
                stat_dict['map_y'].append(map_y[i])
                for stat, key in zip(spot, stat_keys):
                    stat_dict[key].append(stat)

    return stat_dict


def make_stat_df(stats, map_shape=None):

    if isinstance(stats, (list, tuple)) and map_shape is not None:
        stat_dict = make_stat_dict(stats, map_shape)
    elif isinstance(stats, (list, tuple)) and map_shape is None:
        raise TypeError('Map shape must be specified when making dataframe from list.')
    elif isinstance(stats, (dict, OrderedDict)):
        stat_dict = stats
    else:
        raise TypeError('Unknown stats input type.')

    stat_df = pd.DataFrame.from_dict(stat_dict)
    return stat_df.reindex(columns=['map_x', 'map_y', *stat_df.keys()[:-2]])


def remake_spot_list(stat_df, map_shape):

    if isinstance(stat_df, (dict, OrderedDict)):
        stat_df = make_stat_df(stat_df)
    
    spot_list = [ [] for _ in range(np.prod(map_shape))]

    for index in range(len(spot_list)):
        indices = np.unravel_index(index, map_shape)
        pixel_df = stat_df[(stat_df['map_x'] == indices[0])
                            & (stat_df['map_y'] == indices[1])]
        spot_list[index] = np.asarray([list(pixel_df['img_x'].values[:]),
                            list(pixel_df['img_y'].values[:])]).T
        
    return spot_list


# are spots in image or polar coordinates?? Affects the scale of max_dist
# Should take tth_arr and chi_arr arguments
def combine_nearby_spots(spots, max_dist=0.5, max_neighbors=np.inf):

    data = label_nearest_spots(spots, max_dist=max_dist,
                               max_neighbors=max_neighbors)

    new_spots = []
    for label in np.unique(data[:, -1]):
        if label in data[:, -1]:
            new_spot = np.mean(data[data[:, -1] == label][:, :-1], axis=0)
            #new_spot = np.round(new_spot, 0)
            new_spots.append(new_spot)

    return np.asarray(new_spots)


# Not used, but decent reference
'''def watershed_blob_segmentation(spots, mask):
    # Generate distance image from binary blob_img
    distance = ndi.distance_transform_edt(mask)

    # Generate markers upon which to segment blobs
    # Number of input spots may need to be reduceds depending on blob morphology
    coords = np.asarray(spots)
    watershed_mask = np.zeros(distance.shape, dtype=bool)
    watershed_mask[*coords.T] = True
    markers, _ = ndi.label(watershed_mask)

    # Futher segment blob image
    new_blob_image = watershed(-distance, markers, mask=mask)

    # I don't like this, but it fixes some labeling issues
    new_blob_image = median_filter(new_blob_image, size=3)

    # Just to make sure the background is not included
    new_blob_image *= mask

    return new_blob_image'''


def gaussian_watershed_segmentation(blurred_image, spots, mask):
    # Generate markers upon which to segment blobs
    # Number of input spots may need to be reduceds depending on blob morphology
    coords = np.asarray(spots)
    watershed_mask = np.zeros(blurred_image.shape, dtype=bool)
    #watershed_mask[tuple(coords.T)] = True
    watershed_mask[*coords.T] = True
    markers, _ = ndi.label(watershed_mask)

    # Futher segment blob image
    new_blob_image = watershed(-blurred_image, markers, mask=mask)

    # I don't like this, but it fixes some labeling issues
    new_blob_image = median_filter(new_blob_image, size=3)

    # Just to make sure the background is not included
    new_blob_image *= mask

    return new_blob_image   


def append_watershed_blobs(spots, mask, new_blob_image):
    old_blob_image = label(mask)
    for spot in spots:
        if new_blob_image[*spot] == 0:
            new_blob_image[old_blob_image == old_blob_image[*spot]] = (np.max(new_blob_image) + 1)

    # Just to make sure the background is not included
    new_blob_image *= mask

    return new_blob_image


def segment_blobs(xrdmap, map_indices, mask, blurred_image, max_dist=0.75):
    # TODO:
    # Rewrite to not require xrdmap and not have separate map_indices and spots
    tth_arr = xrdmap.tth_arr
    chi_arr = xrdmap.chi_arr

    # Shift azimuthal discontinuities
    _, max_arr, _ = modular_azimuthal_shift(chi_arr) # Do not reuse shifted chi_arr

    pixel_df = xrdmap.spots[(xrdmap.spots['map_x'] == map_indices[0])
                        & (xrdmap.spots['map_y'] == map_indices[1])]
    
    spots = pixel_df[['guess_cen_tth', 'guess_cen_chi']].values
    
    if len(spots) < 1:
        #print('Pixel has no spots!')
        return

    # condense spots
    new_spots = combine_nearby_spots(spots, max_dist=max_dist)
    reduced_coords = estimate_image_coords(new_spots, tth_arr, chi_arr, method='nearest')
    full_coords = estimate_image_coords(spots, tth_arr, chi_arr, method='nearest')

    # Segment blobs based on low intensity regions
    blobs = gaussian_watershed_segmentation(blurred_image, reduced_coords[:, ::-1], mask)
    blobs = append_watershed_blobs(full_coords[:, ::-1], mask, blobs)

    spot_labels = np.array([blobs[*spot] for spot in full_coords[:, ::-1]])

    output_list = []
    for blob in np.unique(blobs):
        if blob == 0:
            continue

        spot_indices = pixel_df.index[spot_labels == blob]

        num_spots = len(spot_indices)
        if num_spots == 0:
            #print(f'No spots in blob {blob}. Proceeding to next blob.')
            continue # Not sure about how to handle this...
        
        bkg_mask = [blobs != 0][0]
        blob_mask = [blobs[bkg_mask] == blob][0]

        blob_int = xrdmap.map.images[map_indices][bkg_mask][blob_mask]
        blob_tth = tth_arr[bkg_mask][blob_mask]
        blob_chi = chi_arr[bkg_mask][blob_mask]

        # Reshift azimuthal units
        blob_chi = modular_azimuthal_reshift(blob_chi, max_arr=max_arr)

        output_list.append([blob_int, blob_tth, blob_chi, spot_indices])
    
    return output_list


# Combined function for lower memory requirements
def fit_spots(xrdmap, SpotModel, max_dist=0.75, sigma=1):
    
    # Check for azimuthal discontinuities
    _, max_arr, shifted = modular_azimuthal_shift(xrdmap.chi_arr)

    # Add fit results columns 
    nan_list = [np.nan,] * len(xrdmap.spots)
    guess_labels = ['height',
                    'cen_tth',
                    'cen_chi',
                    'fwhm_tth',
                    'fwhm_chi']
    guess_labels = [f'guess_{guess_label}' for guess_label in guess_labels]
    fit_labels = ['amp',
                  'tth0',
                  'chi0',
                  'fwhm_tth',
                  'fwhm_chi',
                  'theta',
                  'offset',
                  'r_squared']
    fit_labels = [f'fit_{fit_label}' for fit_label in fit_labels]
    for fit_label in fit_labels:
        xrdmap.spots[fit_label] = nan_list

    # Fit segmented spots with blobs
    def blob_fit(xrdmap, fit):

        fit_int = fit[0]
        fit_x = fit[1]
        fit_y = fit[2]

        # Shift azimuthal disconinuity
        fit_y, _, _ = modular_azimuthal_shift(fit_y, max_arr=max_arr, force_shift=shifted)

        spots_df = xrdmap.spots.iloc[fit[3]]
        num_spots = len(spots_df)

        if len(fit_int) / (6 * len(spots_df)) <= 1.5:
            #print(f'More unknowns than pixels in blob {blob_num}!')
            return [np.nan,] * (6 * len(spots_df) + 2) # Fit variables plus offset and r_squared

        p0 = [np.min(fit_int)] # Guess offset
        for index in spots_df.index:
            p0.extend(list(spots_df.loc[index][guess_labels].values))
            p0.append(0) # theta

        # Shift azimuthal discontinuity, only guess cen_chi
        p0[3::5], _, _ = modular_azimuthal_shift(p0[3::5], max_arr=max_arr, force_shift=shifted)

        bounds = generate_bounds(p0[1:], SpotModel.func_2d, tth_step=None, chi_step=None)
        bounds[0].insert(0, -np.inf) # offset lower bound
        bounds[1].insert(0, np.max(fit_int)) # offset upper bound

        try:
            popt, _ = curve_fit(SpotModel.multi_2d, [fit_x, fit_y], fit_int, p0=p0, bounds=bounds)
            r_squared = compute_r_squared(fit_int, SpotModel.multi_2d([fit_x, fit_y], *popt))
            #print(f'done! R² is {r_squared:.4f}')
        except RuntimeError:
            #print('Fitting failed!')
            popt = [np.nan,] * len(p0)
            r_squared = np.nan

        # Write updates to dataframe|
        offset = popt[0]
        fit_arr = np.array(popt[1:]).reshape(num_spots, 6)

        # Reshift azimuthal discontinuities
        fit_arr[:, 2] = modular_azimuthal_reshift(fit_arr[:, 2], max_arr=max_arr)

        for i, spot_index in enumerate(spots_df.index):
            xrdmap.spots.loc[spot_index, fit_labels] = [*fit_arr[i], offset, r_squared]
    
    # Preparing fit information
    @dask.delayed
    def segment_and_fit(map_indices):
        mask = xrdmap.map.blob_masks[map_indices]
        image = xrdmap.map.images[map_indices]

        if isinstance(image, da.core.Array):
                    image = image.compute()
                    
        # Hard-coded sigma in pixel units. Not the best...
        blurred_image = gaussian_filter(median_filter(image, size=2), sigma=sigma)

        spot_fits = segment_blobs(xrdmap,
                                  map_indices,
                                  mask,
                                  blurred_image,
                                  max_dist=max_dist)
    
        # Iterate through the prepared fits
        for fit in spot_fits:
            if fit is not None:
                delayed_list.append(blob_fit(xrdmap, fit))

    # Scheduling blob segmentation and spot fitting
    delayed_list = []
    for index in range(xrdmap.map.num_images):
        indices = np.unravel_index(index, xrdmap.map.map_shape)
        delayed_list.append(segment_and_fit(indices))        

    # Calculation
    print('Parsing images, segmenting blobs, and fitting spots...')
    with TqdmCallback(tqdm_class=tqdm):
            dask.compute(*delayed_list)

    # Report some stats
    fit_succ = sum(~np.isnan(xrdmap.spots["fit_r_squared"]))
    num_spots = len(xrdmap.spots)
    print(f'Successfully fit {fit_succ} / {num_spots} spots ( {100 * fit_succ / num_spots:.1f} % ).')


# Deprecated
def prepare_fit_spots(xrdmap, max_dist=0.75, sigma=1):
    map_shape = xrdmap.map.map_shape

    @dask.delayed
    def delayed_segment_blobs(xrdmap,
                              map_indices,
                              mask,
                              blurred_image,
                              max_dist=max_dist):
        return segment_blobs(xrdmap,
                             map_indices,
                             mask,
                             blurred_image,
                             max_dist=max_dist)

    delayed_list = []
    print('Scheduling blob segmentation for spot fits...')
    for index in tqdm(range(xrdmap.map.num_images)):
        indices = np.unravel_index(index, map_shape)
        mask = xrdmap.map.blob_masks[indices]
        image = xrdmap.map.images[indices]

        if isinstance(image, da.core.Array):
            image = image.compute()
            
        # Hard-coded sigma in pixel units. Not the best...
        blurred_image = gaussian_filter(median_filter(image, size=2), sigma=sigma)
        
        spot_fits = delayed_segment_blobs(xrdmap,
                                          indices,
                                          mask,
                                          blurred_image,
                                          max_dist)
        delayed_list.append(spot_fits)

    print('Segmenting blobs for spot fits...')
    with TqdmCallback(tqdm_class=tqdm):
        result_list = dask.compute(*delayed_list)

    # Clean up the output list
    spot_fit_info_list = []
    for result in result_list:
        if result is not None:
            spot_fit_info_list.extend(result)

    return spot_fit_info_list


# Deprecated
def old_fit_spots(xrdmap, spot_fit_info_list, SpotModel):

    # Check for azimuthal discontinuities
    _, max_arr, shifted = modular_azimuthal_shift(xrdmap.chi_arr)

    # Add fit results columns 
    nan_list = [np.nan,] * len(xrdmap.spots)
    guess_labels = ['height',
                    'cen_tth',
                    'cen_chi',
                    'fwhm_tth',
                    'fwhm_chi']
    guess_labels = [f'guess_{guess_label}' for guess_label in guess_labels]
    fit_labels = ['amp',
                  'tth0',
                  'chi0',
                  'fwhm_tth',
                  'fwhm_chi',
                  'theta',
                  'offset',
                  'r_squared']
    fit_labels = [f'fit_{fit_label}' for fit_label in fit_labels]
    for fit_label in fit_labels:
        xrdmap.spots[fit_label] = nan_list

    @dask.delayed
    def delayed_fit(xrdmap, fit):

        fit_int = fit[0]
        fit_x = fit[1] # tth
        fit_y = fit[2] # chi

        # Shift azimuthal disconinuity
        fit_y, _, _ = modular_azimuthal_shift(fit_y, max_arr=max_arr, force_shift=shifted)

        spots_df = xrdmap.spots.iloc[fit[3]]
        num_spots = len(spots_df)

        if len(fit_int) / (6 * len(spots_df)) <= 1.5:
            #print(f'More unknowns than pixels in blob {blob_num}!')
            return [np.nan,] * (6 * len(spots_df) + 2) # Fit variables plus offset and r_squared

        p0 = [np.min(fit_int)] # Guess offset
        for index in spots_df.index:
            p0.extend(list(spots_df.loc[index][guess_labels].values))
            p0.append(0) # theta

        # Shift azimuthal discontinuity, only guess cen_chi
        p0[3::5], _, _ = modular_azimuthal_shift(p0[3::5], max_arr=max_arr, force_shift=shifted)

        bounds = generate_bounds(p0[1:], SpotModel.func_2d, tth_step=None, chi_step=None)
        bounds[0].insert(0, -np.inf) # offset lower bound
        bounds[1].insert(0, np.max(fit_int)) # offset upper bound

        try:
            popt, _ = curve_fit(SpotModel.multi_2d, [fit_x, fit_y], fit_int, p0=p0, bounds=bounds)
            r_squared = compute_r_squared(fit_int, SpotModel.multi_2d([fit_x, fit_y], *popt))
            #print(f'done! R² is {r_squared:.4f}')
        except RuntimeError:
            #print('Fitting failed!')
            popt = [np.nan,] * len(p0)
            r_squared = np.nan

        # Write updates to dataframe|
        offset = popt[0]
        fit_arr = np.array(popt[1:]).reshape(num_spots, 6)

        # Reshift azimuthal discontinuities
        fit_arr[:, 2] = modular_azimuthal_reshift(fit_arr[:, 2], max_arr=max_arr)

        for i, spot_index in enumerate(spots_df.index):
            xrdmap.spots.loc[spot_index, fit_labels] = [*fit_arr[i], offset, r_squared]

    # Scheduling delayed fits
    delayed_list = []
    for fit in spot_fit_info_list:
        delayed_list.append(delayed_fit(xrdmap, fit))

    # Computation
    print('Fitting spots in blobs...')
    with TqdmCallback(tqdm_class=tqdm):
            dask.compute(*delayed_list)

    fit_succ = sum(~np.isnan(xrdmap.spots["fit_r_squared"]))
    num_spots = len(xrdmap.spots)
    print(f'Successfully fit {fit_succ} / {num_spots} spots ( {100 * fit_succ / num_spots:.1f} % ).')


















####################
### Still useful ###
####################

def find_blob_contours(blob_img):
    contours = []
    for i in np.unique(blob_img):
        if i == 0: # Skip insignificant regions
            continue
        single_blob = np.zeros_like(blob_img)
        single_blob[blob_img == i] = 1
        #print(len(find_contours(single_blob, 0)))
        contours.append(find_contours(single_blob, 0)[0])
    return [contour[:, ::-1].T for contour in contours] # Rearranging for convenience


# No longer used, but still useful
def coord_mask(coords, image, masked_image=False):
    # Convert coordinates into mask
    mask = np.zeros_like(image)
    mask[*coords.T] = 1
    if masked_image:
        return image[np.bool_(mask)]
    else:
        return np.bool_(mask)