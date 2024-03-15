import numpy as np
import scipy.ndimage as ndi
from dask_image import ndfilters as dask_ndi
from tqdm import tqdm
import dask
from tqdm.dask import TqdmCallback
import dask.array as da



def interpolate_merlin_mask(masked_img):
    # TODO apply interpolation based adaptively on masked image
    # This approach is very clunky and too specific
    # Cannot handle higher dimensional data...
    healed_img = np.copy(masked_img)
    healed_img[256, :] = masked_img[255, :]
    healed_img[257, :] = masked_img[258, :]
    healed_img[:, 255] = masked_img[:, 254]
    healed_img[:, 256] = masked_img[:, 257]
    return healed_img


# Now with dask support!
def find_outlier_pixels(images, size=2, tolerance=2):
    '''
    images      (arr)   Input ND images. Last two dimensions should be the 2D image   
    size        (float) Size of median window. 2 by default only accounts for nearest neighbor pixels
    tolerance   (float) Multiplier value above which to consider a pixel as an outlier. By default 2 times the median value
    TODO better account for significance.
    '''

    data = images.copy()

    size_iter = np.ones(data.ndim, dtype=np.int8)
    size_iter[-2:] = size, size

    if isinstance(data, da.core.Array):
        med_image = dask_ndi.median_filter(data, size=size_iter)
    else:
        med_image = ndi.median_filter(data, size=size_iter)

    ratio_image = np.abs(data / med_image)

    replace_mask = (ratio_image > tolerance)

    # Replace values with median values
    # Direct replace fails with dask arrays
    data[replace_mask] = 0
    med_image[~replace_mask] = 0
    data += med_image

    #masked_data = da.ma.masked_array(data, mask=replace_mask, fill_value=0)
    #masked_med = da.ma.masked_array(med_image, mask=~replace_mask, fill_value=0)
    #new_data = da.ma.filled(masked_data + masked_med)
    # Conditional for dask arrays
    #if isinstance(data, da.core.Array):
    #    data = data.compute()
    #    replace_mask = replace_mask.compute()

    # I dont't like this, but it works
    #replace_mask = replace_mask.compute()

    #for index in zip(*np.where(replace_mask)):
    #    data[index] = med_image[index]

    return data


#def old_find_outlier_pixels(data, size=2, tolerance=3, significance=None):
#    '''
#   data        (arr)   Input 2D image. Cannot handle higher dimensional data yet...   
#   size        (float) Size of median window. 2 by default only accounts for nearest neighbor pixels
#   tolerance   (float) Multiplier value above which to consider a pixel as an outlier. By default 3 times the median value
#   signficance (float) Signficance value below which to ignore contributions from noise fluctuations. 
#                        Set to 5 * data standard deviation. Causes issues with Bragg peaks currently.
#    TODO better account for significance. Better account for multidimensional data...
#    '''
#
#    med_img = ndi.median_filter(data, size=size, mode='mirror')
#
#    ratio_img = np.abs(data / med_img)
#    diff_img = np.abs(data - med_img)
#
#    if significance is None:
#        significance = 5 * np.std(data)
#
#    replace_mask = (ratio_img > tolerance) & (diff_img > significance)
#    fixed_img = np.copy(data)
#    fixed_img[replace_mask] = med_img[replace_mask]
#    return fixed_img


def rebin(arr, new_shape=None, bin_size=(2, 2), method='sum', keep_range=False):
    # TODO: Make sure this can be applied across a full image stack...
    '''
    arr         (arr)   Original image
    new_shape   (tuple) (n, m) shape of new image
    bin_size    (tuple) (n, m) size of bins
    method      (str)   Method of rebinning. Accepts 'sum' or 'mean'. Mean maintains range. Default is 'sum'.
    keep_range  (bool)  NOT IMPLEMENTED. Rescales output to the same range as the original image.
    '''

    # Determine new image shape
    if type(new_shape) in [tuple, list, np.ndarray] and len(new_shape) == 2:
        shape = (new_shape[0], int(arr.shape[0] // new_shape[0]),
                 new_shape[1], int(arr.shape[1] // new_shape[1]))
    elif len(bin_size) == 2:
        new_shape = (int(arr.shape[0] // bin_size[0]), int(arr.shape[1] // bin_size[1]))
        shape = (new_shape[0], int(arr.shape[0] // new_shape[0]),
                 new_shape[1], int(arr.shape[1] // new_shape[1]))
    else:
        raise AttributeError("Input variable either new_shape or bin_size must be tuple, list, or numpy.ndarray of length 2")
    
    # Actual rebinning
    if method == 'sum':
        new_img = arr.reshape(shape).sum(-1).sum(1)
    elif method == 'mean':
        new_img = arr.reshape(shape).mean(-1).mean(1)
    else:
        raise IOError('Must input a valid method for rebinning')
    
    if keep_range:
        new_img = np.min(arr) + np.max(arr) * (new_img - np.min(new_img)) / (np.max(new_img) - np.min(new_img))

    return new_img


def rescale_array(arr, lower=0, upper=1, arr_min=None, arr_max=None, mask=None):
    # Works for arrays of any size including images!

    if mask is not None:
        arr[~mask] = np.nan
    if arr_min is None:
        arr_min = np.nanmin(arr)
    if arr_max is None:
        arr_max = np.nanmax(arr)
    ext = upper - lower
    
    scaled_arr = lower + ext * ((arr - arr_min) / (arr_max - arr_min))

    if mask is not None:
        scaled_arr[~mask] = 0

    return scaled_arr


def iter_rescale_array(arr, lower=0, upper=1, arr_min=None, arr_max=None):
    # Works for arrays of any size including images!
    if arr_min is None:
        arr_min = np.min(arr)
    if arr_max is None:
        arr_max = np.max(arr)
    ext = upper - lower

    @dask.delayed
    def delayed_rescale_array(arr, lower, ext, arr_min, arr_max):
        return lower + ext * ((arr - arr_min) / (arr_max - arr_min))

    for index in tqdm(range(np.multiply(*arr.shape[:2])), desc='Scheduling...'):
        indices = np.unravel_index(index, arr.shape[:2])

        arr[indices] = delayed_rescale_array(arr[indices], lower, ext, arr_min, arr_max)

    with TqdmCallback(desc='Computing...', tqdm_class=tqdm):
        dask.compute(*arr.ravel())
        



'''def rebin(arr, new_shape, method='sum'):
    # TODO: Generalize further based on binning number and not image shape...
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    if method == 'sum':
        return arr.reshape(shape).sum(-1).sum(1)
    elif method == 'mean':
        return arr.reshape(shape).mean(-1).mean(1)
    else:
        raise IOError('Must input a valid method for rebinning')


def rebin_2(arr, new_shape):
    new_shape = (arr.shape[0] // 2, arr.shape[1] // 2)
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).sum(-1).sum(1)'''


