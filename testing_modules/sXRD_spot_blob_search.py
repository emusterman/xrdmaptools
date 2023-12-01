import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass, gaussian_filter, median_filter
from scipy.optimize import curve_fit
import scipy.stats as st
from skimage.measure import label, find_contours
from skimage.segmentation import expand_labels
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi




def spot_search(image, bkg_noise, multiplier=10, sigma=3, plotme=False):


    # Consider other threshold methods
    mask_thresh = np.median(image[image != 0]) + multiplier * bkg_noise
    thresh_img = gaussian_filter(median_filter(image, size=1), sigma=sigma)
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

    return spots, mask


def watershed_blob_segmentation(blob_image, spots):
    old_blob_image = np.copy(blob_image)

    if blob_image.dtype != np.bool_:
        blob_image = (blob_image != 0)

    # Generate distance image from binary blob_img
    distance = ndi.distance_transform_edt(blob_image)

    #coords = peak_local_max(distance, footprint=np.ones((3, 3)),
    #                        labels=[blob_img != 0][0], num_peaks_per_label=np.inf,
    #                        threshold_rel=0.5)
    
    # Generate markers upon which to segment blobs
    # Number of input spots may need to be reduceds depending on blob morphology
    coords = np.asarray(spots)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)

    # Futher segment blob image
    new_blob_image = watershed(-distance, markers, mask=blob_image)

    # I don't like this, but it fixes some labeling issues
    new_blob_image = median_filter(new_blob_image, size=3)

    # Re-adding very small blobs eliminated by watershed filter
    for spot in spots:
        if new_blob_image[*spot] == 0:
            new_blob_image[old_blob_image == old_blob_image[*spot]] = (np.max(new_blob_image) + 1)

    return new_blob_image


def combine_nearby_spots(spots, max_dist=25, max_neighbors=np.inf):

    data = label_nearest_spots(spots, max_dist=max_dist,
                               max_neighbors=max_neighbors)

    new_spots = []
    for label in np.unique(data[:, -1]):
        if label in data[:, -1]:
            new_spot = np.mean(data[data[:, -1] == label][:, :-1], axis=0)
            new_spot = np.round(new_spot, 0).astype(np.int32)
            new_spots.append(new_spot)

    return np.asarray(new_spots)


# Currently only works for Gaussian and Lorentzian peak models(based on number of inputs)
def fit_spots(image, blob_image, spots, PeakModel, tth=None, chi=None, verbose=False):
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
            p0.append(0.1) # sigma_y
            p0.append(0) # theta

        bounds = generate_bounds(p0[1:], PeakModel.func_2d, tth_step=None, chi_step=None)
        bounds[0].insert(0, np.min(image)) # offset lower bound
        bounds[1].insert(0, np.max(image)) # offset upper bound

        popt, pcov = curve_fit(PeakModel.multi_2d, [blob_x, blob_y], blob_int, p0=p0, bounds=bounds)
        r_squared = compute_r_squared(blob_int, PeakModel.multi_2d([blob_x, blob_y], *popt))
        #residual = blob_int - PeakModel.multi_2d([blob_x, blob_y], *popt)
        vprint('done!')
        vprint(f'Fitting R²: {r_squared:.4f}')


        model_fits.extend(popt[1:])
        offsets.extend([popt[0],] * num_spots)

    vprint(f'All {len(spots)} spots have been fit!')

    model_fits = np.asarray(model_fits).reshape(len(model_fits) // 6, 6)
    model_fits = np.hstack((model_fits, np.asarray(offsets).reshape(len(offsets), 1)))


    #abbr = PeakModel.abbr
    dict_labels = [f'fit_amp',
                   f'fit_{x_label}',
                   f'fit_{y_label}',
                   f'fit_sigma_x', # Not labeled due to potential rotation
                   f'fit_sigma_y', # Not labeled due to potential rotation
                   f'fit_theta',
                   f'fit_offset']

    spot_dict = dict(zip(dict_labels, model_fits.T))
    
    return spot_dict


# Big combined function!
# This is what would be parallelized...
def find_and_fit_spots(image, bkg_noise, PeakModel,
                       multiplier=10, sigma=3,
                       max_dist=25, max_neighbors=np.inf,
                       tth=None, chi=None, verbose=False):
    
    # Find spots and mask significant regions
    spots, mask = spot_search(image, bkg_noise,
                              multiplier=multiplier, sigma=sigma)
    # Generate blob_image
    blob_image = label(mask)

    # If too many spots, segment blobs further
    if len(spots) > 25: # 25 is arbitrary distinction for time
        new_spots = combine_nearby_spots(spots, max_dist=max_dist,
                                         max_neighbors=max_neighbors)
        
        new_blob_image = watershed_blob_segmentation(blob_image, new_spots)

        spots = new_spots
        blob_image = new_blob_image

    # Fit spots within each blob
    spot_dict = fit_spots(image, blob_image, spots, PeakModel,
                          tth=tth, chi=chi, verbose=verbose)
    
    return spot_dict




##############################################################################

# Depracted in favor of spot search path
def find_blobs(img, thresh, method='gaussian_threshold',
               sigma=3,
               tth=None, chi=None,
               blob_expansion=5, min_blob_significance=500,
               find_contours=True,
               listme=False, plotme=False):
    '''
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
    '''

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
        'max_coords':list(estimate_recipricol_coords(np.asarray(blob_max_coords).T[::-1], img.shape, tth=tth, chi=chi).T),
        'center_of_mass':list(estimate_recipricol_coords(np.asarray(blob_centers_of_mass).T[::-1], img.shape, tth=tth, chi=chi).T)
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
    
    return blob_df


# Depracated in favor of spot search path
def fit_blobs(img, thresh, blob_df, PeakModel,  tth=None, chi=None, plotme=False):

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

        bounds = generate_bounds(p0, PeakModel.func_2d, tth_step=None, chi_step=None)
        
        try:
            popt, pcov = curve_fit(PeakModel.func_2d, (x_coords[blob_df['mask'][i]], y_coords[blob_df['mask'][i]]), img[blob_df['mask'][i]], p0=p0, bounds=bounds)
            r_squared = compute_r_squared(img[blob_df['mask'][i]], PeakModel.func_2d((x_coords[blob_df['mask'][i]], y_coords[blob_df['mask'][i]]), *popt))
            if not qualify_gaussian_2d_fit(r_squared, 2.5 * np.std(img), p0, popt):
                print(f'Peak fitting exceeds predicted behavior for blob {i}. It may not be significant.')
                raise RuntimeError("Peak fitting exceeds expected behavior.")
            blob_gaussian_lst.append(np.append(popt, [r_squared, PeakModel])) 

            if plotme:
                fit_data = PeakModel.func_2d((x_coords, y_coords), *popt)
                ax.contour(x_coords, y_coords, fit_data.reshape(*img.shape), np.linspace(thresh, popt[0], 5), colors='w', linewidths=0.5)
                ax.scatter(popt[1], popt[2], s=1, c='r')

                props = dict(boxstyle='round', facecolor='white', alpha=0.5)
                arrow_props = dict(facecolor='black', shrink=0.1, width=1, headwidth=3, headlength=5)
                ax.annotate(f'R² = {r_squared:.2f}', xy=(popt[1], popt[2]), xytext=(popt[1] + 1, popt[2] + 1),
                            arrowprops=arrow_props, bbox=props, fontsize=8)
                
        except RuntimeError:
            blob_gaussian_lst.append([np.nan])
    
    blob_df['fit_parameters'] = blob_gaussian_lst
    return blob_df


# Depracated in favor of spot search path
def find_blob_spots(img, blob_df, tth=None, chi=None,
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

    return blob_df


# Depracated in favor of spot search path
# WIP
def adaptive_spot_fitting(img, blob_df, thresh, PeakModel, tth=None, chi=None,
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
            p0.append(0.05) # sigma_x
            p0.append(0.05) # sigma_y
            p0.append(0) # theta

        bounds = generate_bounds(p0, PeakModel.func_2d, tth_step=None, chi_step=None)

        adaptive_fitting = True
        while adaptive_fitting:
            peaks_discounted = False
            popt, pcov = curve_fit(PeakModel.multi_2d, np.vstack([x_fit, y_fit]), z_fit, p0=p0, bounds=bounds)
            r_squared = compute_r_squared(z_fit, PeakModel.multi_2d((x_fit, y_fit), *popt))
            
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
            fit_data = PeakModel.multi_2d((x_coords.ravel(), y_coords.ravel()), *popt)
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
    return spot_df



# No longer used, but still useful
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
    tth_range = np.max([tth_resolution, tth_step])
    chi_range = np.max([chi_resolution, chi_step])

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
                low_bounds.append(0)
                upr_bounds.append(1) # Base off of intrument resolution
            elif arg == 'sigma_x':
                low_bounds.append(0)
                upr_bounds.append(1) # Base off of intrument resolution
            elif arg == 'sigma_y':
                low_bounds.append(0)
                upr_bounds.append(1) # Base off of intrument resolution
            elif arg == 'theta':
                low_bounds.append(-45) # Prevent spinning
                upr_bounds.append(45) # Only works for degrees...
            elif arg == 'gamma':
                low_bounds.append(0)
                upr_bounds.append(1) # Base off of intrument resolution
            elif arg == 'gamma_x':
                low_bounds.append(0)
                upr_bounds.append(1) # Base off of intrument resolution
            elif arg == 'gamma_y':
                low_bounds.append(0)
                upr_bounds.append(1) # Base off of intrument resolution
            elif arg == 'eta':
                low_bounds.append(0)
                upr_bounds.append(1)
            else:
                raise IOError(f"{peak_function} uses inputs not defined by the generate_bounds function!")
            
    return [low_bounds, upr_bounds]









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