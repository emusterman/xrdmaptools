import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter, median_filter
from skimage.transform import resize
from scipy.optimize import curve_fit
from scipy.interpolate import RectBivariateSpline
from tqdm import tqdm


#############################
### Background Estimators ###
#############################

### ImageMap ###

# Unused general masked filter function
def masked_filter(imagemap, filter, mask=None, **kwargs):
    if mask is None:
        mask = np.empty_like(imagemap.images, dtype=np.bool_)
        mask[:, :] = imagemap.mask
    elif mask.ndim == 2:
        image_mask = mask.copy()
        mask = np.empty_like(imagemap.images, dtype=np.bool_)
        mask[:, :] = image_mask

    for key, value in kwargs.items():
        if isinstance(value, (int, float)):
            kwargs[key] = (0, 0, value, value)

    zero_image = imagemap.images.copy()
    zero_image[~mask] = 0 # should be redundant
    gauss_zero = filter(zero_image, **kwargs)

    # This block is redundant when calling this serveral several times...
    div_image = np.ones_like(imagemap.images)
    div_image[~mask] = 0
    gauss_div = filter(div_image, **kwargs)

    overlap = gauss_zero / gauss_div
    overlap[~mask] = 0

    return overlap


def masked_gaussian_background(imagemap, sigma=100, mask=None):

    if mask is None:
        mask = np.empty_like(imagemap.images, dtype=np.bool_)
        mask[:, :] = imagemap.mask
    elif mask.ndim == 2:
        image_mask = mask.copy()
        mask = np.empty_like(imagemap.images, dtype=np.bool_)
        mask[:, :] = image_mask

    zero_image = imagemap.images.copy()
    zero_image[~mask] = 0 # should be redundant
    gauss_zero = gaussian_filter(zero_image, sigma=(0, 0, sigma, sigma))

    div_image = np.ones_like(imagemap.images)
    div_image[~mask] = 0
    gauss_div = gaussian_filter(div_image, sigma=(0, 0, sigma, sigma))

    overlap = gauss_zero / gauss_div
    overlap[~mask] = 0
    
    return overlap # Always returned as ndarray. Something with gaussian_filter()


# Very slow and not very accurate
def fit_poly_bkg(imagemap, order=3, mask=None):

    if mask is None:
        if np.any(imagemap.mask != 1):
            mask = imagemap.mask
        else:
            mask = np.ones_like(imagemap.images.shape[-2:])
            print('Warning: No mask could be constructed.')
    

    bkg_map = np.zeros_like(imagemap.images) 

    x_coords, y_coords = np.meshgrid(imagemap.tth, imagemap.chi[::-1])
    x_fit = x_coords[mask]
    y_fit = y_coords[mask]

    p0 = np.ones(2 * order**2)

    for index in tqdm(range(imagemap.num_images)):
        indices = np.unravel_index(index, imagemap.map_shape)
        image = imagemap.images[indices]
        z_fit = image[mask]

        popt, pcov = curve_fit(poly_bkg, [x_fit, y_fit], z_fit, p0=p0)
        bkg_map[indices] = poly_bkg([x_coords, y_coords], *popt)
        p0 = popt

    return bkg_map


def fit_spline_bkg(imagemap, mask=None, sparsity=0.5, s=5000):

    if mask is None:
        if np.any(imagemap.mask != 1):
            mask = imagemap.mask
        else:
            mask = np.ones_like(imagemap.images)
            print('Warning: No mask could be constructed.')

    tth = imagemap.tth
    chi = imagemap.chi

    tth_step = np.diff(tth[:2])[0]
    tth_skip = int(sparsity / tth_step)
    sparse_tth = tth[::tth_skip]

    chi_step = np.diff(chi[:2])[0]
    chi_skip = int(sparsity / chi_step)
    sparse_chi = chi[::chi_skip]
    
    sparse_mask = mask[::chi_skip, ::tth_skip]
    med_filt_size = tuple(np.asarray(sparse_mask.shape) // 20)
    med_filt_size = tuple([x if x > 0 else 1 for x in med_filt_size])

    bkg_map = np.zeros_like(imagemap.images)

    for index in tqdm(range(imagemap.num_images)):
        indices = np.unravel_index(index, imagemap.map_shape)
        image = imagemap.images[indices]
        sparse_image = image[::chi_skip, ::tth_skip]
        sparse_image = median_filter(sparse_image, size=med_filt_size)

        zero_image = np.copy(sparse_image)
        zero_image[~sparse_mask] = 0 # should be redundant
        zero_func = RectBivariateSpline(sparse_tth, sparse_chi, zero_image.T, s=s)
        zero_spline = zero_func(tth, chi).T

        div_image = 0 * np.copy(sparse_image) + 1
        div_image[~sparse_mask] = 0
        div_func = RectBivariateSpline(sparse_tth, sparse_chi, div_image.T, s=s)
        div_spline = div_func(tth, chi).T

        overlap = zero_spline / div_spline
        overlap[~mask] = 0

        bkg_map[indices] = overlap

    return bkg_map


def masked_bruckner_background(imagemap, size=10, max_iterations=100,
                                     binning=2, min_prominence=0.1,
                                     mask=None, verbose=False):

    image_shape = imagemap.images.shape[-2:]

    # Determine mask
    if mask is None:
        if np.any(imagemap.mask != 1):
            mask = imagemap.mask
        else:
            mask = np.ones(image_shape, dtype=np.bool_)
            print('Warning: No mask could be constructed.')

    if binning is not None:
        new_shape = tuple(i // binning for i in image_shape)
        mask = resize(mask, new_shape)
    else:
        new_shape = image_shape

    # Divisor image for ignoring mask contributions
    div_image = np.ones(new_shape)
    div_image[~mask] = 0
    div_filter = uniform_filter(div_image, size=size)
    
    bkg_map = np.zeros_like(imagemap.images)

    # Cycle through all map images. TODO: Parallelize this
    for index in tqdm(range(imagemap.num_images)):
        indices = np.unravel_index(index, imagemap.map_shape)
        
        # Initial image
        init_image = np.copy(imagemap.images[indices])
        init_image = resize(init_image, new_shape)

        # remove outliers
        #init_image = median_filter(init_image, size=1)

        # Chop off high intensity signals and grab stats
        init_min = np.min(init_image[mask])
        init_avg = np.mean(init_image[mask])
        #init_std = np.std(init_image[mask])
        init_med = np.median(init_image[mask])
        #init_max = np.max(init_image[mask])
        init_thresh = init_avg + 2 * (init_avg - init_min)
        init_mask = (init_image >= init_thresh)
        init_image[init_mask] = init_thresh

        # remove masked pixels for good measure
        init_image[~mask] = 0

        old_image = np.copy(init_image)

        # Filter before iterative approach
        zero_image = np.copy(old_image)
        #return zero_image, mask
        zero_image[~mask] = 0 # should be redundant
        zero_filter = uniform_filter(zero_image, size=size)

        old_image = zero_filter / div_filter
        old_image[~mask] = 0

        # Bruckner Algorithm
        for i in range(max_iterations):
            zero_image = np.copy(old_image)
            zero_image[~mask] = 0 # should be redundant
            zero_filter = uniform_filter(zero_image, size=size)

            avg_image = zero_filter / div_filter
            avg_image[~mask] = 0    

            min_image = np.min(np.array([old_image, avg_image]), axis=0)
            
            if np.max(np.abs(old_image - min_image)) > (min_prominence * init_med):
                old_image = np.copy(min_image)
            else:
                if verbose:
                    print(f'Background converged after {i + 1} iterations!')
                break

            if (i == max_iterations - 1) & verbose:
                print('Max number of iterations reached.')
        
        bkg_map[indices] = resize(min_image, image_shape)

    return bkg_map


'''def masked_bruckner_background(imagemap, size=10, iterations=100, mask=None, verbose=False):

    # Determine mask
    if mask is None:
        if np.any(imagemap.mask != 1):
            mask = imagemap.mask
        else:
            mask = np.ones(imagemap.images.shape[-2:], dtype=np.bool_)
            print('Warning: No mask could be constructed.')

    # Divisor image for ignoring mask contributions
    div_image = np.ones(imagemap.images.shape[-2:])
    div_image[~mask] = 0
    div_filter = uniform_filter(div_image, size=size)
    
    bkg_map = np.zeros_like(imagemap.images)

    # Cycle through all map images. TODO: Parallelize this
    for index in tqdm(range(imagemap.num_images)):
        indices = np.unravel_index(index, imagemap.map_shape)

        # Inititial cleanup
        old_image = np.copy(imagemap.images[indices])
        init_min = np.min(old_image)
        init_avg = np.mean(old_image)
        init_thresh = init_avg + 2 * (init_avg - init_min)
        init_mask = (old_image >= init_thresh)
        old_image[init_mask] = init_thresh

        # Filter before iterative approach
        zero_image = np.copy(old_image)
        #return zero_image, mask
        zero_image[~mask] = 0 # should be redundant
        zero_filter = uniform_filter(zero_image, size=size)

        old_image = zero_filter / div_filter
        old_image[~mask] = 0

        # Bruckner Algorithm
        for i in range(iterations):
            zero_image = np.copy(old_image)
            zero_image[~mask] = 0 # should be redundant
            zero_filter = uniform_filter(zero_image, size=size)

            avg_image = zero_filter / div_filter
            avg_image[~mask] = 0    

            min_image = np.min(np.array([old_image, avg_image]), axis=0)
            
            if np.max(np.abs(old_image - min_image)) > 5e-4:
                old_image = np.copy(min_image)
            else:
                if verbose:
                    print(f'Background converged after {i + 1} iterations!')
                break
        
        bkg_map[indices] = min_image

    return bkg_map'''




### Single Image ###

def masked_image_filter(image, mask, filter, **kwargs):

    zero_image = np.copy(image)
    zero_image[~mask] = 0 # should be redundant
    gauss_zero = filter(zero_image, **kwargs)

    div_image = 0 * np.copy(image) + 1
    div_image[~mask] = 0
    gauss_div = filter(div_image, **kwargs)

    overlap = gauss_zero / gauss_div
    overlap[~mask] = 0

    return overlap


def masked_gaussian_image_background(image, mask, sigma=100):
    nan_image = np.copy(image)
    nan_image[~mask] = np.nan

    zero_image = np.copy(image)
    zero_image[~mask] = 0 # should be redundant
    gauss_zero = gaussian_filter(zero_image, sigma=sigma)

    div_image = 0 * np.copy(nan_image) + 1
    div_image[~mask] = 0
    gauss_div = gaussian_filter(div_image, sigma=sigma)

    overlap = gauss_zero / gauss_div
    overlap[~mask] = 0
    
    return overlap


def poly_bkg(xy, *args):
    import itertools
    order = int(np.sqrt(len(args) / 2))

    x = np.asarray(xy[0])
    y = np.asarray(xy[1])

    # All possible terms and cross-terms with sum powers of order
    powers = list(itertools.product(range(order), repeat=2))

    z = 0
    for i, power in enumerate(powers):
        z += (args[2 * i] * x ** power[0]
              + args[2 * i + 1] * y ** power[1])

    return z



def spline_image_bkg(image, mask, tth, chi, sparsity=0.5, s=5000):

    tth_step = np.diff(tth[:2])[0]
    tth_skip = int(sparsity / tth_step)
    chi_step = np.diff(chi[:2])[0]
    chi_skip = int(sparsity / chi_step)

    zero_image = np.copy(image)
    zero_image[~mask] = 0 # should be redundant
    zero_func = RectBivariateSpline(tth[::tth_skip], chi[::chi_skip], zero_image[::chi_skip, ::tth_skip].T, s=s)
    zero_spline = zero_func(tth, chi).T

    div_image = 0 * np.copy(image) + 1
    div_image[~mask] = 0
    div_func = RectBivariateSpline(tth[::tth_skip], chi[::chi_skip], div_image[::chi_skip, ::tth_skip].T, s=s)
    div_spline = div_func(tth, chi).T

    overlap = zero_spline / div_spline
    overlap[~mask] = 0

    return overlap


def masked_bruckner_image_background(image, size, iterations, mask):

    # Divisor image for ignoring mask contributions
    div_image = np.ones_like(image)
    div_image[~mask] = 0
    div_filter = uniform_filter(div_image, size=size)

    # Inititial cleanup
    old_image = np.copy(image)
    init_min = np.min(old_image)
    init_avg = np.mean(old_image)
    init_thresh = init_avg + 2 * (init_avg - init_min)
    init_mask = (old_image >= init_thresh)
    old_image[init_mask] = init_thresh

    # Filter before iterative approach
    zero_image = np.copy(old_image)
    zero_image[~mask] = 0 # should be redundant
    zero_filter = uniform_filter(zero_image, size=size)

    old_image = zero_filter / div_filter
    old_image[~mask] = 0      

    # Bruckner Algorithm
    for i in range(iterations):
        zero_image = np.copy(old_image)
        zero_image[~mask] = 0 # should be redundant
        zero_filter = uniform_filter(zero_image, size=size)

        avg_image = zero_filter / div_filter
        avg_image[~mask] = 0    

        min_image = np.min(np.array([old_image, avg_image]), axis=0)
        
        if np.max(np.abs(old_image - min_image)) > 5e-4:
            old_image = np.copy(min_image)
        else:
            print(f'Background converged after {i + 1} iterations!')
            break

    return min_image