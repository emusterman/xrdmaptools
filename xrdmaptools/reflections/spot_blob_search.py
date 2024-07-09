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
from ..geometry.geometry import estimate_image_coords, estimate_polar_coords

from .SpotModels import generate_bounds



'''def estimate_map_noise(imagemap, sample_number=200):
    if imagemap.num_images > sample_number:
        indices = np.unravel_index(np.random.choice(range(imagemap.num_images), size=sample_number), imagemap.map_shape)
        median_image = np.nanmedian(imagemap.images[indices], axis=(0))
    else:
        median_image = np.nanmedian(imagemap.images, axis=(0, 1))
    bkg_noise = np.nanstd(median_image[imagemap.mask])
    return bkg_noise'''

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

    # Expand blobs for better fitting
    if expansion is not None:
        expanded_mask = expand_labels(blob_mask, distance=expansion)
    else:
        expanded_mask = blob_mask

    return blob_mask, expanded_mask, blurred_image


def new_spot_search(scaled_image,
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
                       vmax=np.max(scaled_image) * 0.1,
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
     expanded_mask,
     blurred_image) = blob_search(
        scaled_image,
        mask=mask,
        threshold_method=threshold_method,
        multiplier=multiplier,
        size=size,
        expansion=expansion
        )
    
    (spots,
     blob_mask,
     blurred_image) = new_spot_search(
        scaled_image,
        blob_mask,
        blurred_image,
        min_distance=min_distance,
        plotme=plotme
        )

    return spots, expanded_mask, blurred_image


# Must take scaled images!!!
def spot_search(scaled_image,
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

        plt.show()

    return spots, peak_mask, thresh_img


# depracated
'''def old_spot_search(image, bkg_noise=None, mask=None,
                multiplier=10, sigma=3, expansion=None, plotme=False):

    if mask is None:
       mask = (image != 0)
    
    # Consider other threshold methods
    mask_thresh = np.max([np.median(image[mask]), 0]) + multiplier * bkg_noise
    image = median_filter(image, size=2)
    #thresh_img = gaussian_filter(median_filter(image, size=2), sigma=sigma)

    # Remove some outliers
    med_image = median_filter(image, size=2)
    
    # Smooth image to reduce noise contributions
    zero_image = np.copy(med_image)
    zero_image[~mask] = 0 # should be redundant
    gauss_zero = gaussian_filter(zero_image, sigma=sigma)

    div_image = np.ones_like(med_image)
    div_image[~mask] = 0
    gauss_div = gaussian_filter(div_image, sigma=sigma)

    thresh_img = gauss_zero / gauss_div
    thresh_img[~mask] = 0

    # Create mask for peak search
    peak_mask = thresh_img > mask_thresh
    peak_mask *= mask
    if expansion is not None:
        peak_mask = expand_labels(peak_mask, distance=expansion)

    #spot_img = gaussian_filter(median_filter(pixel, size=2), sigma=5)
    spots = peak_local_max(thresh_img,
                           #threshold_rel=multiplier * bkg_noise,
                           min_distance=2, # Just a pinch more spacing
                           labels=peak_mask,
                           num_peaks_per_label=np.inf)
    
    if plotme:
        fig, ax = plt.subplots(1, 1, figsize=(9, 6), dpi=200)

        im = ax.imshow(image * mask, vmin=np.min([mask_thresh, 0]), vmax=10 * mask_thresh, aspect='auto')
        fig.colorbar(im, ax=ax)
        ax.scatter(spots[:, 1], spots[:, 0], s=1, c='r')

        plt.show()

    return spots, peak_mask, thresh_img'''


def find_spots(imagemap, mask=None,
               threshold_method='gaussian',
               multiplier=5, size=3,
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

    # spot should be iterable of [img_x, img_y]
    spot_mask = circular_mask(image.shape, [*spot], radius)
    spot_image = image * spot_mask

    height = np.max(spot_image) - np.min(spot_image)
    img_x = int(spot[0])
    img_y = int(spot[1])
    tth = tth_arr[*spot]
    chi = chi_arr[*spot]
    intensity = np.sum(spot_image)

    center = arbitrary_center_of_mass(spot_image, tth_arr, chi_arr)
    #center = estimate_polar_coords(center, tth_arr, chi_arr, method='linear')

    aweights = rescale_array(image[spot_mask], lower=0, upper=1)
    std_tth = np.sqrt(np.cov(tth_arr[spot_mask],
                                aweights=aweights))
    fwhm_tth = std_tth * 2 * np.sqrt(2 * np.log(2))
    
    std_chi = np.sqrt(np.cov(chi_arr[spot_mask],
                                aweights=aweights))
    fwhm_chi = std_chi * 2 * np.sqrt(2 * np.log(2))
    
    return [height, img_x, img_y, tth, chi, center[0], center[1], fwhm_tth, fwhm_chi, intensity]


# deprecated
'''def old_spot_stats(spot, image, radius=5, tth=None, chi=None):
    #y_coords, x_coords = np.meshgrid(chi[::-1], tth)
    x_coords, y_coords = np.meshgrid(tth, chi[::-1])
    spot_mask = circular_mask(image.shape, [*spot], radius)
    spot_image = image * spot_mask

    height = np.max(spot_image) - np.min(spot_image)
    indx = int(spot[0])
    indy = int(spot[1])
    x = x_coords[*spot]
    y = y_coords[*spot]
    intensity = np.sum(spot_image)

    center = center_of_mass(spot_image)
    center = estimate_reciprocal_coords(center[::-1], image.shape, tth=tth, chi=chi)

    aweights = rescale_array(image[spot_mask], lower=0, upper=1)
    stdx = np.sqrt(np.cov(x_coords[spot_mask],
                                aweights=aweights))
    fwhm_x = stdx * 2 * np.sqrt(2 * np.log(2))
    stdy = np.sqrt(np.cov(y_coords[spot_mask],
                                aweights=aweights))
    fwhm_y = stdy * 2 * np.sqrt(2 * np.log(2))
    
    return [height, indx, indy, x, y, center[0], center[1], fwhm_x, fwhm_y, intensity]'''


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
    stat_keys = ['guess_height', 'guess_img_x', 'guess_img_y',
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
    
    spot_list = [ [] for _ in range(np.multiply(*map_shape))]

    for index in range(len(spot_list)):
        indices = np.unravel_index(index, map_shape)
        pixel_df = stat_df[(stat_df['map_x'] == indices[0])
                            & (stat_df['map_y'] == indices[1])]
        spot_list[index] = np.asarray([list(pixel_df['guess_cen_tth'].values[:]),
                            list(pixel_df['guess_cen_chi'].values[:])]).T
        
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


def segment_blobs(xrdmap, map_indices, spots, mask, blurred_image, max_dist=0.75):
    # TODO:
    # Rewrite to not require xrdmap and not have separate map_indices and spots
    tth_arr = xrdmap.tth_arr
    chi_arr = xrdmap.chi_arr

    pixel_df = xrdmap.spots[(xrdmap.spots['map_x'] == map_indices[0])
                        & (xrdmap.spots['map_y'] == map_indices[1])]
    
    if len(pixel_df) < 1:
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

        output_list.append([blob_int, blob_tth, blob_chi, spot_indices])
    
    return output_list


# deprecated
'''def old_segment_blobs(xrdmap, map_indices, spots, mask, blurred_image, max_dist=0.75):
    x_coords, y_coords = np.meshgrid(xrdmap.tth, xrdmap.chi[::-1])
    pixel_df = xrdmap.spots[(xrdmap.spots['map_x'] == map_indices[0])
                        & (xrdmap.spots['map_y'] == map_indices[1])]
    
    if len(pixel_df) < 1:
        #print('Pixel has no spots!')
        return

    new_spots = combine_nearby_spots(spots, max_dist=max_dist)
    reduced_coords = estimate_img_coords(new_spots.T, xrdmap.map.calibrated_shape, tth=xrdmap.tth, chi=xrdmap.chi).T
    full_coords = estimate_img_coords(spots.T, xrdmap.map.calibrated_shape, tth=xrdmap.tth, chi=xrdmap.chi).T

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
        blob_x = x_coords[bkg_mask][blob_mask]
        blob_y = y_coords[bkg_mask][blob_mask]

        output_list.append([blob_int, blob_x, blob_y, spot_indices])
    
    return output_list'''

# TODO: Combine prepare_fit_spots and fit_spots in one big function with a couple of delay calls
# in the current iteration, the the spot_fit_info_list variable can be 75% the size of the full dataset
# this needs to be smaller or only transiently in memory
def prepare_fit_spots(xrdmap, max_dist=0.75, sigma=1):
    map_shape = xrdmap.map.map_shape
    spot_list = remake_spot_list(xrdmap.spots, map_shape)

    @dask.delayed
    def delayed_segment_blobs(xrdmap, map_indices, spots, mask, blurred_image, max_dist=max_dist):
        return segment_blobs(xrdmap, map_indices, spots, mask, blurred_image, max_dist=max_dist)

    delayed_list = []
    print('Scheduling blob segmentation for spot fits...')
    for index in tqdm(range(len(spot_list))):
        indices = np.unravel_index(index, map_shape)
        mask = xrdmap.map.spot_masks[indices]
        image = xrdmap.map.images[indices]

        if isinstance(image, da.core.Array):
            image = image.compute()
            
        # Hard-coded sigma in pixel units. Not the best...
        blurred_image = gaussian_filter(median_filter(image, size=2), sigma=sigma)
        #blurred_image = xrdmap.map.blurred_images[indices]
        
        spot_fits = delayed_segment_blobs(xrdmap, indices, spot_list[index], mask, blurred_image, max_dist)
        delayed_list.append(spot_fits)

    print('Segmenting blobs for spot fits...')
    with TqdmCallback(tqdm_class=tqdm):
        result_list = dask.compute(*delayed_list)

    # Clean up the output list
    output_list = []
    for result in result_list:
        if result is not None:
            output_list.extend(result)

    return output_list


def fit_spots(xrdmap, spot_fit_info_list, SpotModel):

    # Add fit results columns 
    nan_list = [np.nan,] * len(xrdmap.spots)
    guess_labels = ['height', 'cen_tth', 'cen_chi', 'fwhm_tth', 'fwhm_chi']
    guess_labels = [f'guess_{guess_label}' for guess_label in guess_labels]
    fit_labels = ['amp', 'tth0', 'chi0', 'fwhm_tth', 'fwhm_chi', 'theta', 'offset', 'r_squared']
    fit_labels = [f'fit_{fit_label}' for fit_label in fit_labels]
    for fit_label in fit_labels:
        xrdmap.spots[fit_label] = nan_list

    @dask.delayed
    def delayed_fit(xrdmap, fit):

        fit_int = fit[0]
        fit_x = fit[1]
        fit_y = fit[2]
        spots_df = xrdmap.spots.iloc[fit[3]]
        num_spots = len(spots_df)

        if len(fit_int) / (6 * len(spots_df)) <= 1.5:
            #print(f'More unknowns than pixels in blob {blob_num}!')
            return [np.nan,] * (6 * len(spots_df) + 2) # Fit variables plus offset and r_squared

        p0 = [np.min(fit_int)] # Guess offset
        for index in spots_df.index:
            p0.extend(list(spots_df.loc[index][guess_labels].values))
            p0.append(0) # theta

        bounds = generate_bounds(p0[1:], SpotModel.func_2d, tth_step=None, chi_step=None)
        bounds[0].insert(0, -np.inf) # offset lower bound
        bounds[1].insert(0, np.max(fit_int)) # offset upper bound

        #print(f'Fitting {num_spots} spots...')
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
        for i, spot_index in enumerate(spots_df.index):
            xrdmap.spots.loc[spot_index, fit_labels] = [*fit_arr[i], offset, r_squared]

    # Scheduling delayed fits
    delayed_list = []
    for fit in spot_fit_info_list:
        delayed_list.append(delayed_fit(xrdmap, fit))

    # Computation
    #print('Fitting spots...', end='', flush=True)
    print('Fitting spots in blobs...')
    with TqdmCallback(tqdm_class=tqdm):
            dask.compute(*delayed_list)

    #print('done!')
    fit_succ = sum(~np.isnan(xrdmap.spots["fit_r_squared"]))
    num_spots = len(xrdmap.spots)
    print(f'Successfully fit {fit_succ} / {num_spots} spots ( {100 * fit_succ / num_spots:.1f} % ).')
    #return proc_list


'''def fit_spots_blob(xrdmap, fit_int, fit_x, fit_y, spot_indices, SpotModel, guess_labels, fit_labels):

        spots_df = xrdmap.spots.iloc[spot_indices]
        num_spots = len(spots_df)

        if len(fit_int) / (6 * len(spots_df)) <= 1.5:
            #print(f'More unknowns than pixels in blob {blob_num}!')
            return [np.nan,] * (6 * len(spots_df) + 2) # Fit variables plus offset and r_squared

        p0 = [np.min(fit_int)] # Guess offset
        for index in spots_df.index:
            p0.extend(list(spots_df.loc[index][guess_labels].values))
            p0.append(0) # theta

        bounds = generate_bounds(p0[1:], SpotModel.func_2d, tth_step=None, chi_step=None)
        bounds[0].insert(0, -np.inf) # offset lower bound
        bounds[1].insert(0, np.max(fit_int)) # offset upper bound

        #print(f'Fitting {num_spots} spots...')
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
        for i, spot_index in enumerate(spots_df.index):
            xrdmap.spots.loc[spot_index, fit_labels] = [*fit_arr[i], offset, r_squared]'''



# Combined function for lower memory requirements
def new_fit_spots(xrdmap, SpotModel, max_dist=0.75, sigma=1):
    # Reformmating
    map_shape = xrdmap.map.map_shape
    # OPTIMIZE ME: this doubles spot memory and is just reformatting
    spot_list = remake_spot_list(xrdmap.spots, map_shape)

    # Add fit results columns 
    nan_list = [np.nan,] * len(xrdmap.spots)
    guess_labels = ['height', 'cen_tth', 'cen_chi', 'fwhm_tth', 'fwhm_chi']
    guess_labels = [f'guess_{guess_label}' for guess_label in guess_labels]
    fit_labels = ['amp', 'tth0', 'chi0', 'fwhm_tth', 'fwhm_chi', 'theta', 'offset', 'r_squared']
    fit_labels = [f'fit_{fit_label}' for fit_label in fit_labels]
    for fit_label in fit_labels:
        xrdmap.spots[fit_label] = nan_list

    # Preparing fit information
    @dask.delayed
    def delayed_segment_blobs(xrdmap, map_indices, spots, mask, blurred_image, max_dist=max_dist):
        return segment_blobs(xrdmap, map_indices, spots, mask, blurred_image, max_dist=max_dist)
    
    # Fitting information from above
    # Could send the innner operations to its own function like segment_blobs
    @dask.delayed
    def delayed_fit(xrdmap, fit):

        fit_int = fit[0]
        fit_x = fit[1]
        fit_y = fit[2]
        spots_df = xrdmap.spots.iloc[fit[3]]
        num_spots = len(spots_df)

        if len(fit_int) / (6 * len(spots_df)) <= 1.5:
            #print(f'More unknowns than pixels in blob {blob_num}!')
            return [np.nan,] * (6 * len(spots_df) + 2) # Fit variables plus offset and r_squared

        p0 = [np.min(fit_int)] # Guess offset
        for index in spots_df.index:
            p0.extend(list(spots_df.loc[index][guess_labels].values))
            p0.append(0) # theta

        bounds = generate_bounds(p0[1:], SpotModel.func_2d, tth_step=None, chi_step=None)
        bounds[0].insert(0, -np.inf) # offset lower bound
        bounds[1].insert(0, np.max(fit_int)) # offset upper bound

        #print(f'Fitting {num_spots} spots...')
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
        for i, spot_index in enumerate(spots_df.index):
            xrdmap.spots.loc[spot_index, fit_labels] = [*fit_arr[i], offset, r_squared]

    # Scheduling blob segmentation and spot fitting
    delayed_list = []
    print('Scheduling blob segmentation for spot fits...')
    for index in tqdm(range(len(spot_list))):
        indices = np.unravel_index(index, map_shape)
        mask = xrdmap.map.spot_masks[indices]
        image = xrdmap.map.images[indices]

        if isinstance(image, da.core.Array):
            image = image.compute()
            
        # Hard-coded sigma in pixel units. Not the best...
        blurred_image = gaussian_filter(median_filter(image, size=2), sigma=sigma)
        
        # Segment and prepare blobs for fitting
        spot_fits = delayed_segment_blobs(xrdmap, indices, spot_list[index], mask, blurred_image, max_dist)

        # Iterate through the prepared fits
        for fit in spot_fits:
            if fit is not None:
                delayed_list.append(delayed_fit(xrdmap, fit))

    # Calculation
    print('Segmenting blobs and fitting spots...')
    with TqdmCallback(tqdm_class=tqdm):
            dask.compute(*delayed_list)

    # Report some stats
    fit_succ = sum(~np.isnan(xrdmap.spots["fit_r_squared"]))
    num_spots = len(xrdmap.spots)
    print(f'Successfully fit {fit_succ} / {num_spots} spots ( {100 * fit_succ / num_spots:.1f} % ).')


































'''def old_find_spots(image, bkg_noise, multiplier=10, sigma=3,
               radius=5, tth=None, chi=None,
               plotme=False):

    
    image = np.asarray(image)

    spots, mask, _ = old_spot_search(image, bkg_noise,
                              multiplier=multiplier, sigma=sigma,
                              plotme=plotme)
    
    x_label, y_label = 'tth', 'chi'
    if tth is None:
        tth = range(image.shape[1])
        x_label = 'img_x'
    if chi is None:
        chi = range(image.shape[0])
        y_label = 'img_y'
    x_coords, y_coords = np.meshgrid(tth, chi[::-1])
    

    height = []
    indx, indy = [], []
    x, y = [], []
    cenx, ceny = [], []
    stdx, stdy = [], []
    int_lst = []

    for spot in spots:
        spot_mask = circular_mask(image.shape, [*spot], radius)
        #spot_mask = np.zeros_like(image)
        #spot_mask[spot[0] - radius:spot[0] + radius, spot[1] - radius:spot[1] + radius] = np.ones((2 * radius, 2 * radius))
        #spot_mask = spot_mask.astype(np.bool_)
        spot_image = image * spot_mask

        height.append(np.max(spot_image) - np.min(spot_image))
        indx.append(int(spot[0]))
        indy.append(int(spot[1]))
        x.append(x_coords[*spot])
        y.append(y_coords[*spot])

        center = center_of_mass(spot_image)
        center = estimate_recipricol_coords(center[::-1], image.shape, tth=tth, chi=chi)
        cenx.append(center[0])
        ceny.append(center[1])

        stdx.append(np.sqrt(np.cov(x_coords[spot_mask],
                                   aweights=image[spot_mask])))
        stdy.append(np.sqrt(np.cov(y_coords[spot_mask],
                                   aweights=image[spot_mask])))

        intensity = np.sum(spot_image)
        int_lst.append(intensity)
    
    spot_array = np.stack([height,
                           indx, indy,
                           x, y,
                           cenx, ceny,
                           stdx, stdy,
                           int_lst])
    
    dict_labels = [f'guess_height',
                   f'guess_indx',
                   f'guess_indy',
                   f'guess_x',
                   f'guess_y',
                   f'guess_cenx',
                   f'guess_ceny',
                   f'guess_stdx',
                   f'guess_stdy',
                   f'guess_int']

    spot_dict = dict(zip(dict_labels, spot_array))

    return spot_dict, mask'''



'''def old_spot_search(image, bkg_noise, multiplier=10, sigma=3, plotme=False):


    # Consider other threshold methods
    mask_thresh = np.median(image[image != 0]) + multiplier * bkg_noise
    thresh_img = gaussian_filter(median_filter(image, size=2), sigma=sigma)
    mask = thresh_img > mask_thresh

    #spot_img = gaussian_filter(median_filter(pixel, size=1), sigma=5)
    spots = peak_local_max(thresh_img,
                           threshold_rel=multiplier * bkg_noise,
                           min_distance=2, # Just a pinch more spacing
                           labels=mask,
                           num_peaks_per_label=np.inf)
    
    if plotme:
        fig, ax = plt.subplots(1, 1, figsize=(9, 6), dpi=200)

        im = ax.imshow(image, vmin=0, vmax=10 * mask_thresh)
        fig.colorbar(im, ax=ax)
        ax.scatter(spots[:, 1], spots[:, 0], s=1, c='r')

        plt.show()

    return spots, mask, thresh_img'''


'''def old_combine_nearby_spots(spots, max_dist=20, max_neighbors=np.inf):

    data = label_nearest_spots(spots, max_dist=max_dist,
                               max_neighbors=max_neighbors)

    new_spots = []
    for label in np.unique(data[:, -1]):
        if label in data[:, -1]:
            new_spot = np.mean(data[data[:, -1] == label][:, :-1], axis=0)
            #new_spot = np.round(new_spot, 0)
            new_spots.append(new_spot)

    return np.asarray(new_spots)'''


'''def old_watershed_blob_segmentation(spots, mask):
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


'''def old_gaussian_watershed_segmentation(blurred_image, spots, mask):
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

    return new_blob_image'''


'''def old_append_watershed_blobs(spots, mask, new_blob_image):
    old_blob_image = label(mask)
    for spot in spots:
        if new_blob_image[*spot] == 0:
            new_blob_image[old_blob_image == old_blob_image[*spot]] = (np.max(new_blob_image) + 1)

    # Just to make sure the background is not included
    new_blob_image *= mask

    return new_blob_image'''


# Currently only works for Gaussian and Lorentzian peak models(based on number of inputs)
'''def old_fit_spots(image, blob_image, spots, SpotModel, tth=None, chi=None, verbose=False):
    global _verbose
    _verbose = verbose


    x_label, y_label = 'tth', 'chi'
    if tth is None:
        tth = range(image.shape[1])
        x_label = 'img_x'
    if chi is None:
        chi = range(image.shape[0])
        y_label = 'img_y'

    x_coords, y_coords = np.meshgrid(tth, chi[::-1])

    # Add watershed to further separate blobs...
    spot_labels = np.array([blob_image[*spot] for spot in spots])
    model_fits = []
    offsets = []

    for blob in np.unique(blob_image):
        if blob == 0:
            continue

        num_spots = len(spots[spot_labels == blob])
        if num_spots == 0:
            vprint(f'No spots in blob {blob}. Proceeding to next blob.')
            continue

        bkg_mask = [blob_image != 0][0]
        blob_mask = [blob_image[bkg_mask] == blob][0]
        blob_size = np.sum(blob_mask)

        if blob_size / (6 * num_spots) <= 1.5:
            vprint(f'More unknowns than pixels in blob {blob}!')
            vprint('Proceeding to next blob.')
            continue

        vprint(f'Fitting {len(spots[spot_labels == blob])} spots within blob {blob}...', end='', flush=True)

        blob_int = image[bkg_mask][blob_mask]
        blob_x = x_coords[bkg_mask][blob_mask]
        blob_y = y_coords[bkg_mask][blob_mask]

        p0 = [np.median(image)] # Guess offset
        for i, spot in enumerate(spots[spot_labels == blob]):
            p0.append(image[*spot]) # amp
            p0.append(x_coords[*spot]) # x0
            p0.append(y_coords[*spot]) # y0
            p0.append(0.05) # sigma_x
            p0.append(0.05) # sigma_y
            p0.append(0) # theta

        bounds = generate_bounds(p0[1:], SpotModel.func_2d, tth_step=None, chi_step=None)
        bounds[0].insert(0, np.min(image)) # offset lower bound
        bounds[1].insert(0, np.max(image)) # offset upper bound

        try:
            popt, pcov = curve_fit(SpotModel.multi_2d, [blob_x, blob_y], blob_int, p0=p0, bounds=bounds)
        except RuntimeError:
            vprint('\nPrevious sigma guess too far off. Guessing again...')
            try:
                p0[1:][3::6] = [0.1,] * num_spots
                p0[1:][4::6] = [0.1,] * num_spots
                popt, pcov = curve_fit(SpotModel.multi_2d, [blob_x, blob_y], blob_int, p0=p0, bounds=bounds)
            except RuntimeError:
                vprint('Previous sigma guess too far off. Guessing again...')
                try:
                    p0[1:][3::6] = [1,] * num_spots
                    p0[1:][4::6] = [1,] * num_spots
                    popt, pcov = curve_fit(SpotModel.multi_2d, [blob_x, blob_y], blob_int, p0=p0, bounds=bounds)
                except RuntimeError:
                    vprint('Cannot fit spots!')
                    popt = [np.nan,] * len(p0)

        r_squared = compute_r_squared(blob_int, SpotModel.multi_2d([blob_x, blob_y], *popt))
        #residual = blob_int - SpotModel.multi_2d([blob_x, blob_y], *popt)
        vprint('done!')
        vprint(f'Fitting R²: {r_squared:.4f}')


        model_fits.extend(popt[1:])
        offsets.extend([popt[0],] * num_spots)

    model_fits = np.asarray(model_fits).reshape(len(model_fits) // 6, 6)
    model_fits = np.hstack((model_fits, np.asarray(offsets).reshape(len(offsets), 1)))

    #abbr = SpotModel.abbr
    dict_labels = [f'fit_amp',
                   f'fit_{x_label}',
                   f'fit_{y_label}',
                   f'fit_sigma_x', # Not labeled due to potential rotation
                   f'fit_sigma_y', # Not labeled due to potential rotation
                   f'fit_theta',
                   f'fit_offset']

    spot_dict = dict(zip(dict_labels, model_fits.T))

    vprint(f'{len(model_fits)}/{len(spots)} spots have been successfully fitted!')

    return spot_dict'''


# Big combined function!
# This is what would be parallelized...
'''def old_find_and_fit_spots(image, bkg_noise, SpotModel,
                       multiplier=10, sigma=3,
                       max_dist=20, max_neighbors=np.inf,
                       tth=None, chi=None, verbose=False):
    
    # Find spots and mask significant regions
    spots, mask, _ = old_spot_search(image, bkg_noise,
                              multiplier=multiplier, sigma=sigma)
    blob_image = label(mask)
    print(f'Found {len(spots)} spots.')

    # If too many spots, segment blobs further
    if len(spots) > 25: # 25 is arbitrary distinction for time
        print(f'Too many spots found for effective fitting. Combining and segmenting data.')
        new_spots = combine_nearby_spots(spots, max_dist=max_dist,
                                         max_neighbors=max_neighbors)
        
        # This one gets new_spots (blob_combined values)
        new_blob_image = watershed_blob_segmentation(new_spots, mask)
        # This one gets old spots (real values)
        new_blob_image = append_watershed_blobs(spots, mask, new_blob_image)
        blob_image = new_blob_image

    # Fit spots within each blob
    print('Fitting spots...', end='', flush=True)
    spot_dict = old_fit_spots(image, blob_image, spots, SpotModel,
                          tth=tth, chi=chi, verbose=verbose)
    print('done!')
    
    return spot_dict'''




##############################################################################

# Depracted in favor of spot search path
'''def old_find_blobs(img, thresh, method='gaussian_threshold',
               sigma=3,
               tth=None, chi=None,
               blob_expansion=5, min_blob_significance=500,
               find_contours=True,
               listme=False, plotme=False):
    
    img                     ()
    thresh                  ()
    method                  ()
    sigma                   ()
    tth                     ()
    chi                     ()
    blob_expansion          ()
    min_blob_significance   ()
    list_blobs              ()
    plot_blobs              ()
    

    proc_img = np.copy(img).astype(np.float32)

    if method == 'simple_threshold':
        pass

    elif method == 'gaussian_threshold':
        proc_img = gaussian_filter(proc_img, sigma=sigma)
        
    else:
        print(f"Method {method} not implemented. Please choose 'simple_threshold' or 'gaussian_threshold'.")
        return None # Not sure I am supposed to do this...
    
    proc_img[proc_img < thresh] = np.nan
    proc_img[~np.isnan(proc_img)] = 1
    proc_img[np.isnan(proc_img)] = 0 # This forces the background/isignificant regions to be zero
    # Adds a buffer to each blob. In theory this should hurt the fitting since it adds insignificant noise
    # Helps connect blobs that get separated though
    proc_img = expand_labels(proc_img, distance=blob_expansion)
    blob_img = label(proc_img + 1).astype(np.uint16)

    blob_labels = np.unique(blob_img)

    blob_masks = []
    blob_values =  []
    blob_areas = []
    blob_intensities = []
    blob_significances = []
    blob_max_coords = []
    blob_centers_of_mass = []

    # Characterize individual blobs
    if len(blob_labels) > 1:
        for i, blob in enumerate(blob_labels):
            blob_mask = np.copy(blob_img)
            blob_mask[blob_mask != blob] = False
            blob_mask[blob_mask != 0] = True
            blob_mask = np.bool_(blob_mask)

            blob_value = img * blob_mask
            # TODO: Convert blob area to actual solid angle...
            blob_area = np.sum(blob_mask) * np.diff(tth[:2])[0] * np.diff(chi[:2])[0] # in angle squared
            blob_intensity = np.sum(blob_value)
            blob_significance = blob_intensity / blob_area
            
            # Remove isignificant blobs. Earlier is better
            if blob_significance < min_blob_significance:
                # Remove blob from blob_img
                #return blob_img, blob_mask
                blob_img[blob_mask] = 0
                continue

            blob_max_coord = np.unravel_index(np.argmax(blob_value), blob_value.shape)
            blob_center_of_mass = center_of_mass(blob_value)

            # Reduced memory method for storing mask coords
            blob_mask_coords = np.asarray(np.where(blob_mask)).astype(np.uint16).T
            # Reconstruct mask with:
                # mask = np.zeros(test.map.image_shape)
                # mask[*coords.T] = 1
            blob_pixel_values = img[*blob_mask_coords.T]

            blob_masks.append(blob_mask_coords)
            blob_values.append(blob_pixel_values)
            blob_areas.append(blob_area)
            blob_intensities.append(blob_intensity)
            blob_significances.append(blob_significance)
            blob_max_coords.append(blob_max_coord)
            blob_centers_of_mass.append(blob_center_of_mass)

    blob_df = pd.DataFrame.from_dict({
        'intensity':blob_intensities,
        'area':blob_areas,
        'significance':blob_significances,
        'values':blob_values,
        'mask':blob_masks,
        'max_coords':list(estimate_reciprocal_coords(np.asarray(blob_max_coords).T[::-1], img.shape, tth=tth, chi=chi).T),
        'center_of_mass':list(estimate_reciprocal_coords(np.asarray(blob_centers_of_mass).T[::-1], img.shape, tth=tth, chi=chi).T)
    })

    if find_contours:
        if len(np.unique(blob_img)) > 1:
            blob_contours = find_blob_contours(blob_img)
            for i in range(len(blob_contours)):
                blob_contours[i] = estimate_recipricol_coords(blob_contours[i], img.shape, tth=tth, chi=chi)
        else:
            blob_contours = (None,) * len(blob_masks)

        if 'contour' not in blob_df:
            blob_df.insert(len(blob_df.columns), 'contour', blob_contours, True)

    # Remove insignificant regions. Mainly aiming for background "blobs"
    # Might be worth adding and area qualifier later...
    blob_df.drop(blob_df[np.array(np.isnan(blob_df['significance'].to_list()))
                         | np.array(blob_df['significance'] < min_blob_significance)].index, inplace=True)

    # Resort values by significance for convensence
    blob_df.sort_values('significance', ascending=False, inplace=True)
    blob_df.reset_index(inplace=True, drop=True)
    
    if plotme:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=200)
        im = ax.imshow(img, extent=[tth[0], tth[-1], chi[0], chi[-1]], vmin=0, aspect='auto')
        fig.colorbar(im, ax=ax)
        ax.set_xlabel('Scattering Angle, 2θ [°]')
        ax.set_ylabel('Azimuthal Angle, χ [°]')
        if find_contours and len(blob_df) > 0:
            for i in range(len(blob_df)):
                if blob_df['contour'][i] is not None:
                    ax.plot(*blob_df['contour'][i], c='r', lw=0.5)
                    ax.scatter(*blob_df['center_of_mass'][i], c='r', s=1)
            
    if listme:
        print(blob_df.loc[:, ['intensity', 'area', 'significance','center_of_mass']].head())
    
    return blob_df'''


# Depracated in favor of spot search path
'''def old_fit_blobs(img, thresh, blob_df, SpotModel,  tth=None, chi=None, plotme=False):

    x_coords, y_coords = np.meshgrid(tth, chi[::-1])

    if plotme:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=200)
        im = ax.imshow(img, extent=[tth[0], tth[-1], chi[0], chi[-1]], vmin=0)
        fig.colorbar(im, ax=ax)
        ax.set_xlabel('Scattering Angle, 2θ [°]')
        ax.set_ylabel('Azimuthal Angle, χ [°]')
        for i in range(len(blob_df)):
            ax.plot(*blob_df['contour'][i], c='r', lw=0.5)

    # This only works for gaussain and lorentzian. Voigt has one more function...
    blob_gaussian_lst = []
    for i in range(len(blob_df)):
        p0 = [
            np.max(blob_df['value'][i]), # amp
            blob_df['center_of_mass'][i][0], # x0
            blob_df['center_of_mass'][i][1], # y0
            0.1, # sigma_x
            0.1, # sigma_y
            0 # theta
        ]

        bounds = generate_bounds(p0, SpotModel.func_2d, tth_step=None, chi_step=None)
        
        try:
            popt, pcov = curve_fit(SpotModel.func_2d, (x_coords[blob_df['mask'][i]], y_coords[blob_df['mask'][i]]), img[blob_df['mask'][i]], p0=p0, bounds=bounds)
            r_squared = compute_r_squared(img[blob_df['mask'][i]], SpotModel.func_2d((x_coords[blob_df['mask'][i]], y_coords[blob_df['mask'][i]]), *popt))
            if not qualify_gaussian_2d_fit(r_squared, 2.5 * np.std(img), p0, popt):
                print(f'Peak fitting exceeds predicted behavior for blob {i}. It may not be significant.')
                raise RuntimeError("Peak fitting exceeds expected behavior.")
            blob_gaussian_lst.append(np.append(popt, [r_squared, SpotModel])) 

            if plotme:
                fit_data = SpotModel.func_2d((x_coords, y_coords), *popt)
                ax.contour(x_coords, y_coords, fit_data.reshape(*img.shape), np.linspace(thresh, popt[0], 5), colors='w', linewidths=0.5)
                ax.scatter(popt[1], popt[2], s=1, c='r')

                props = dict(boxstyle='round', facecolor='white', alpha=0.5)
                arrow_props = dict(facecolor='black', shrink=0.1, width=1, headwidth=3, headlength=5)
                ax.annotate(f'R² = {r_squared:.2f}', xy=(popt[1], popt[2]), xytext=(popt[1] + 1, popt[2] + 1),
                            arrowprops=arrow_props, bbox=props, fontsize=8)
                
        except RuntimeError:
            blob_gaussian_lst.append([np.nan])
    
    blob_df['fit_parameters'] = blob_gaussian_lst
    return blob_df'''


# Depracated in favor of spot search path
'''def old_find_blob_spots(img, blob_df, tth=None, chi=None,
                    size=2, sigma=0.5, min_distance=3, threshold_rel=0.15,
                    plotme=False):

    filt_img = gaussian_filter(median_filter(img, size=size), sigma=sigma)
    local_spots = peak_local_max(filt_img, min_distance=min_distance, threshold_rel=threshold_rel)

    # Assign spots to blobs
    blob_spot_lst = []
    blob_spot_int_lst = []
    for i in range(len(blob_df)):
        spot_lst = []
        spot_int_lst = []
        for spot in local_spots:
            if blob_df['mask'][i][*spot]:
                spot_lst.append(spot)
                spot_int_lst.append(img[*spot])
        if len(spot_lst) > 0:
            blob_spot_lst.append(list(estimate_recipricol_coords(np.asarray(spot_lst).T[::-1], img.shape, tth=tth, chi=chi).T))
            blob_spot_int_lst.append(spot_int_lst)

        else:
            blob_spot_lst.append([blob_df['max_coords'][i]])
            blob_spot_int_lst.append([np.max(blob_df['value'][i])])

    # Add spot information to blob dataframe
    if 'spots' not in blob_df:
        blob_df.insert(len(blob_df.columns), 'spots', blob_spot_lst, True)
        blob_df.insert(len(blob_df.columns), 'spots_intensity', blob_spot_int_lst, True)
    else:
        blob_df['spots'] = pd.DataFrame.from_dict({'spots':blob_spot_lst})
        blob_df['spots_intensity'] = pd.DataFrame.from_dict({'spots_intensity':blob_spot_int_lst})

    #local_spots = list(calibrate_img_coords(np.asarray(local_spots).T[::-1], img, tth=tth, chi=chi).T)
    if plotme:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=200)
        im = ax.imshow(img, extent=[tth[0], tth[-1], chi[0], chi[-1]], vmin=0)
        fig.colorbar(im, ax=ax)
        ax.set_xlabel('Scattering Angle, 2θ [°]')
        ax.set_ylabel('Azimuthal Angle, χ [°]')
        for i in range(len(blob_df)):
            ax.plot(*blob_df['contour'][i], c='r', lw=0.5)
        ax.scatter(*np.array([item for sublist in blob_df['spots'].to_list() for item in sublist]).T, c='r', s=1)

    return blob_df'''


# Depracated in favor of spot search path
# WIP
'''def old_adaptive_spot_fitting(img, blob_df, thresh, SpotModel, tth=None, chi=None,
                          plotme=False):
    # Adaptive peak fitting within blobs
    # TODO: Allow for peak addition with residual?
    x_coords, y_coords = np.meshgrid(tth, chi[::-1])

    if plotme:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=200)
        im = ax.imshow(img, extent=[tth[0], tth[-1], chi[0], chi[-1]], vmin=0)
        fig.colorbar(im, ax=ax)
        for i in range(len(blob_df)):
            ax.plot(*blob_df['contour'][i], c='r', lw=0.5)

    spot_fits = []
    parent_blobs = []
    for blob_num in range(len(blob_df)):
        
        z_fit = blob_df['value'][blob_num][blob_df['mask'][blob_num]]
        x_fit = x_coords[blob_df['mask'][blob_num]]
        y_fit = y_coords[blob_df['mask'][blob_num]]

        # Only works for Gaussian and Lorentzian. Create function to generate these...
        p0 = []
        for i, spot in enumerate(blob_df['spots'][blob_num]):
            p0.append(blob_df['spots_intensity'][blob_num][i]) # amp
            p0.append(spot[0]) # x0
            p0.append(spot[1]) # y0
            p0.append(1) # sigma_x
            p0.append(1) # sigma_y
            p0.append(0) # theta

        bounds = generate_bounds(p0, SpotModel.func_2d, tth_step=None, chi_step=None)

        adaptive_fitting = True
        while adaptive_fitting:
            peaks_discounted = False
            popt, pcov = curve_fit(SpotModel.multi_2d, np.vstack([x_fit, y_fit]), z_fit, p0=p0, bounds=bounds)
            r_squared = compute_r_squared(z_fit, SpotModel.multi_2d((x_fit, y_fit), *popt))
            
            old_p0 = p0.copy()
            for i in range(len(popt) // 6):
                if not qualify_gaussian_2d_fit(0.5, 4 * np.std(img), old_p0[6 * i:6 * i + 6], popt[6 * i:6 * i + 6]):
                    print(f'Spot {i} discounted.')
                    #print(old_p0[6 * i:6 * i + 6])
                    #print(popt[6 * i:6 * i + 6])
                    del(p0[6 * i:6 * i + 6])
                    del(bounds[0][6 * i:6 * i + 6])
                    del(bounds[1][6 * i:6 * i + 6])
                    #print(len(p0) // 7)
                    peaks_discounted = True
            if len(p0) == 0:
                adaptive_fitting = False
                print(f'Eliminated all local maxima. Defaulting to full blob fit for {blob_num}')
                popt = blob_df['fit_parameters'][blob_num]
                if len(popt) > 1:
                    popt = popt[:-2].astype(float)
                else:
                    print('No blob fit either!. Blob appears to be insignificant. Consider removing?')
            else:
                adaptive_fitting = peaks_discounted
            if adaptive_fitting:
                print('Refitting...')
            elif np.all(~np.isnan(popt)) and (len(popt) > 1):
                print(f'Fitting successful for {len(popt) // 6} spot(s) within blob {blob_num}!')
                spot_fits.append(popt)
                parent_blobs.append([blob_num]*(len(popt) // 6))

        # Adds contours and peak centers if spot fitting successful and plotme=True
        if plotme and np.all(~np.isnan(popt)) and (len(popt) > 1):       
            fit_data = SpotModel.multi_2d((x_coords.ravel(), y_coords.ravel()), *popt)
            ax.contour(x_coords, y_coords, fit_data.reshape(*img.shape), np.linspace(thresh, popt[0], 5), colors='w', linewidths=0.5)
            ax.scatter(np.asarray(popt).reshape(len(popt) // 6, 6)[:, 1], np.asarray(popt).reshape(len(popt) // 6, 6)[:, 2], s=1, c='r')
            ax.set_xlabel('Scattering Angle, 2θ [°]')
            ax.set_ylabel('Azimuthal Angle, χ [°]')

            props = dict(boxstyle='round', facecolor='white', alpha=0.5)
            arrow_props = dict(facecolor='black', shrink=0.1, width=1, headwidth=5, headlength=5)
            ax.annotate(f'R² = {r_squared:.2f}', xy=(popt[1], popt[2]), xytext=(popt[1] + 1, popt[2] + 1),
                        arrowprops=arrow_props, bbox=props, fontsize=8)
            
    parent_blobs = np.array([item for sublist in parent_blobs for item in sublist])
    spot_fits = np.array([item for sublist in spot_fits for item in sublist])
    spot_fits = spot_fits.reshape(len(parent_blobs), len(spot_fits) // len(parent_blobs))

    spot_df = pd.DataFrame.from_dict({
        'height' : spot_fits[:, 0], # Will add integrated intensity too
        'tth' : spot_fits[:, 1],
        'chi' : spot_fits[:, 2],
        'tth_width' : spot_fits[:, 3], # Not accurate yet. Will change to FWHM
        'chi_width' : spot_fits[:, 4], # Not accurate yet. Will change to FWHM
        'rotation' : spot_fits[:, 5], # Not accurate yet. Need to apply to above widths
        'parent_blobs' : parent_blobs
        })
    
    spot_df.sort_values('height', ascending=False, inplace=True)
    spot_df.reset_index(inplace=True, drop=True)
    return spot_df'''


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



'''def generate_bounds(p0, peak_function, tth_step=None, chi_step=None):
    
    # Get function input variable names, excluding 'self', 'x' or '(xy)' and any default arguments
    inputs = list(peak_function.__code__.co_varnames[:peak_function.__code__.co_argcount])
    if 'self' in inputs: inputs.remove('self')
    inputs = inputs[1:] # remove the x, or xy inputs
    if peak_function.__defaults__ is not None:
        inputs = inputs[:-len(peak_function.__defaults__)] # remove defualts

    if tth_step is None: tth_step = 0.02
    if chi_step is None: chi_step = 0.05

    tth_resolution, chi_resolution = 0.01, 0.01 # In degrees...
    tth_range = np.max([tth_resolution, tth_step]) * 2 # just a bit more wiggle room
    chi_range = np.max([chi_resolution, chi_step]) * 2 # just a bit more wiggle room

    low_bounds, upr_bounds = [], []
    for i in range(0, len(p0), len(inputs)):
        for j, arg in enumerate(inputs):
            if arg == 'amp':
                low_bounds.append(0)
                upr_bounds.append(np.inf)
            elif arg == 'x0':
                low_bounds.append(p0[i + j] - tth_range) # Restricting peak shift based on guess position uncertainty
                upr_bounds.append(p0[i + j] + tth_range)
            elif arg == 'y0':
                low_bounds.append(p0[i + j] - chi_range) # Restricting peak shift based on guess position uncertainty
                upr_bounds.append(p0[i + j] + chi_range)
            elif arg == 'sigma':
                low_bounds.append(0.005)
                upr_bounds.append(2) # Base off of intrument resolution
            elif arg == 'sigma_x':
                low_bounds.append(0.005)
                upr_bounds.append(5) # Base off of intrument resolution
            elif arg == 'sigma_y':
                low_bounds.append(0.005)
                upr_bounds.append(5) # Base off of intrument resolution
            elif arg == 'theta':
                low_bounds.append(-45) # Prevents spinning
                upr_bounds.append(45) # Only works for degrees...
            elif arg == 'gamma':
                low_bounds.append(0.005)
                upr_bounds.append(1) # Base off of intrument resolution
            elif arg == 'gamma_x':
                low_bounds.append(0.005)
                upr_bounds.append(1) # Base off of intrument resolution
            elif arg == 'gamma_y':
                low_bounds.append(0.005)
                upr_bounds.append(1) # Base off of intrument resolution
            elif arg == 'eta':
                low_bounds.append(0)
                upr_bounds.append(1)
            else:
                raise ValueError(f"{peak_function} uses inputs not defined by the generate_bounds function!")
            
    return [low_bounds, upr_bounds]'''









'''def filter_time_series(time_series, size=(3, 1, 1), sigma=2):
    median_xrd = median_filter(time_series, size=size)
    gaussian_xrd = gaussian_filter(median_xrd, sigma)
    return median_xrd, gaussian_xrd'''


'''def find_max_coord(median, gaussian, wind=5):
    max_coord = []
    spot_int, gauss_int = [], []

    for i, img in enumerate(median):
        
        # Pixel accurate peak maxima
        ind = np.unravel_index(np.argmax(gaussian[i], axis=None), median.shape)
        #ind = np.unravel_index(np.argmax(img, axis=None), xrd.shape)
        x, y = ind[1], ind[2]

        # Sub-pixel accurate peak maxima
        bbox = [x - wind, x + wind, y - wind, y + wind]
        bbox = [d if (d > 0 and d < median.shape[1]) else 0 for d in bbox]
        dx, dy = center_of_mass(img[bbox[0]:bbox[1], bbox[2]:bbox[3]])
        new_x = x - wind + dx
        new_y = y - wind + dy
        if np.isnan(new_x) or new_x < 0 or new_x > median.shape[1]:
            new_x = x
        if np.isnan(new_y) or new_y < 0 or new_y > median.shape[1]:
            new_y = y
        max_coord.append(np.array([new_x, new_y]))

        # XRD spot intensity
        try:
            bbox = [int(new_x) - wind, int(new_x) + wind, int(new_y) - wind, int(new_y) + wind]
            bbox = [d if (d > 0 and d < median.shape[1]) else 0 for d in bbox]
            spot_int.append(np.sum(img[bbox[0]:bbox[1], bbox[2]:bbox[3]]))
            gauss_int.append(np.sum(gaussian[i][bbox[0]:bbox[1], bbox[2]:bbox[3]]))
        except ValueError:
            bbox = [x - wind, x + wind, y - wind, y + wind]
            bbox = [d if (d > 0 and d < median.shape[1]) else 0 for d in bbox]
            spot_int.append(np.sum(img[bbox[0]:bbox[1], bbox[2]:bbox[3]]))
            gauss_int.append(np.sum(gaussian[i][bbox[0]:bbox[1], bbox[2]:bbox[3]]))

    max_coord = np.asarray(max_coord)
    spot_int, gauss_int = np.asarray(spot_int), np.asarray(gauss_int)
    return max_coord, spot_int, gauss_int'''