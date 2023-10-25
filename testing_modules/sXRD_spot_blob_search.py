import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass, gaussian_filter, median_filter
from scipy.optimize import curve_fit
from skimage.measure import label, find_contours
from skimage.segmentation import expand_labels
from skimage.feature import peak_local_max


def find_blobs(img, thresh, method='gaussian_threshold',
               sigma=3,
               tth=tth, chi=chi,
               blob_expansion=5, min_blob_significance=500,
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

    if method == 'simple_threshold':
        proc_img = np.copy(img)
        proc_img[proc_img < thresh] = np.nan
        proc_img[~np.isnan(proc_img)] = 1
        blob_img = label(proc_img).astype(np.float32)

    elif method == 'gaussian_threshold':
        proc_img = gaussian_filter(img, sigma=3)
        proc_img[proc_img < thresh] = np.nan
        proc_img[~np.isnan(proc_img)] = 1
        blob_img = label(proc_img).astype(np.float32)

    else:
        print("Method {mehtod} not implemented. Please choose 'simple_threshold' or 'gaussian_threshold'.")
        return None
    blob_labels = np.unique(blob_img)

    # Adds a buffer to each blob. In theory this should hurt the fitting since it adds insignificant noise
    # Helps connect blobs that get separated
    temp_image = np.copy(blob_img)
    temp_image[temp_image == 1] = 0
    blob_img = expand_labels(temp_image, distance=blob_expansion)

    blob_masks = []
    blob_values =  []
    blob_areas = []
    blob_intensities = []
    blob_significances = []
    blob_max_coords = []
    blob_centers_of_mass = []

    # Includes pieces of background. These are removed latter
    blob_contours = find_blob_contours(blob_img)
    for i in range(len(blob_contours)):
        blob_contours[i] = estimate_recipricol_coords(blob_contours[i], img, tth=tth, chi=chi)

    # Characterize individual blobs
    for i, blob in enumerate(blob_labels):
        blob_mask = np.copy(blob_img)
        blob_mask[blob_mask != blob] = False
        blob_mask[blob_mask != 0] = True
        blob_mask = np.bool_(blob_mask)

        blob_value = img * blob_mask
        # TODO: Convert blob area to actual solid angle...
        blob_area = np.sum(blob_mask) * np.diff(tth[:2]) * np.diff(chi[:2]) # in angle squared
        blob_intensity = np.sum(blob_value)
        blob_significance = blob_intensity / blob_area
        blob_max_coord = np.unravel_index(np.argmax(blob_value), blob_value.shape)
        blob_center_of_mass = center_of_mass(blob_value)

        blob_masks.append(blob_mask)
        blob_values.append(blob_value)
        blob_areas.append(blob_area)
        blob_intensities.append(blob_intensity)
        blob_significances.append(blob_significance)
        blob_max_coords.append(blob_max_coord)
        blob_centers_of_mass.append(blob_center_of_mass)

    blob_df = pd.DataFrame.from_dict({
        'intensity':blob_intensities,
        'area':blob_areas,
        'significance':blob_significances,
        'value':blob_values,
        'mask':blob_masks,
        'contour':blob_contours,
        'max_coords':list(estimate_recipricol_coords(np.asarray(blob_max_coords).T[::-1], img, tth=tth, chi=chi).T),
        'center_of_mass':list(estimate_recipricol_coords(np.asarray(blob_centers_of_mass).T[::-1], img, tth=tth, chi=chi).T)
    })

    # Remove insignificant regions. Mainly aiming for background "blobs"
    # Might be worth adding and area qualifier later...
    blob_df.drop(blob_df[np.array(np.isnan(blob_df['significance'].to_list())) | np.array(blob_df['significance'] < min_blob_significance)].index, inplace=True)

    # Resort values by significance for convensence
    blob_df.sort_values('significance', ascending=False, inplace=True)
    blob_df.reset_index(inplace=True, drop=True)
    
    if plotme:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=200)
        im = ax.imshow(img, extent=[tth[0], tth[-1], chi[0], chi[-1]], vmin=0)
        fig.colorbar(im, ax=ax)
        ax.set_xlabel('Scattering Angle, 2θ [°]')
        ax.set_ylabel('Azimuthal Angle, χ [°]')
        for i in range(len(blob_df)):
            ax.plot(*blob_df['contour'][i], c='r', lw=0.5)
            ax.scatter(*blob_df['center_of_mass'][i], c='r', s=1)
            
    if listme:
        print(blob_df.loc[:, ['intensity', 'area', 'significance','center_of_mass']].head())
    
    return blob_df


def fit_blobs(img, thresh, blob_df, PeakModel,  tth=tth, chi=chi, plotme=False):

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
                
        except:
            blob_gaussian_lst.append([np.nan])
    
    blob_df['fit_parameters'] = blob_gaussian_lst
    return blob_df


def find_blob_spots(img, blob_df, tth=tth, chi=chi,
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
            blob_spot_lst.append(list(estimate_recipricol_coords(np.asarray(spot_lst).T[::-1], img, tth=tth, chi=chi).T))
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


def adaptive_spot_fitting(img, blob_df, tth=tth, chi=chi):
    x_img_lst = np.linspace(0, tth_num - 1, tth_num)
    y_img_lst = np.linspace(0, chi_num - 1, chi_num)
    x_img, y_img = np.meshgrid(x_img_lst, y_img_lst)

    blob_bbox = [int(np.min(x_img[blob_df['mask'][blob_num]])), int(np.max(x_img[blob_df['mask'][blob_num]])) + 1,
                int(np.min(y_img[blob_df['mask'][blob_num]])), int(np.max(y_img[blob_df['mask'][blob_num]])) + 1]

    return



def find_blob_contours(blob_img):
    contours = []
    for i in np.unique(blob_img).astype(int):
        single_blob = np.zeros_like(blob_img)
        single_blob[blob_img == i] = 1
        #print(len(find_contours(single_blob, 0)))
        contours.append(find_contours(single_blob, 0)[0])
    return [contour[:, ::-1].T for contour in contours] # Rearranging for convenience


def qualify_gaussian_2d_fit(r_squared, sig_threshold, guess_params, fit_params, return_reason=False):
    # Determine euclidean distance between intitial guess and peak fit. TODO change to angular misorientation
    offset_distance = np.sqrt((fit_params[1] - guess_params[1])**2 + (fit_params[2] - guess_params[2])**2)
    
    keep_fit = np.all([r_squared >= 0, # Qualify peak fit. Decent fits still do not have great r_squared, so cutoff is low
                       fit_params[0] > sig_threshold,
                       offset_distance < 50,
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
    if peak_function.__defaults__ != None:
        inputs = inputs[:-len(peak_function.__defaults__)] # remove defualts

    if tth_step == None: tth_step = 0.02
    if chi_step == None: chi_step = 0.05

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
                upr_bounds.append(np.inf) # Base off of intrument resolution
            elif arg == 'sigma_x':
                low_bounds.append(0)
                upr_bounds.append(np.inf) # Base off of intrument resolution
            elif arg == 'sigma_y':
                low_bounds.append(0)
                upr_bounds.append(np.inf) # Base off of intrument resolution
            elif arg == 'theta':
                low_bounds.append(-45) # Prevent spinning
                upr_bounds.append(45) # Only works for degrees...
            elif arg == 'gamma':
                low_bounds.append(0)
                upr_bounds.append(np.inf) # Base off of intrument resolution
            elif arg == 'gamma_x':
                low_bounds.append(0)
                upr_bounds.append(np.inf) # Base off of intrument resolution
            elif arg == 'gamma_y':
                low_bounds.append(0)
                upr_bounds.append(np.inf) # Base off of intrument resolution
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