import numpy as np
from skimage import restoration
from scipy.ndimage import median_filter
from tqdm import tqdm
import dask
from tqdm.dask import TqdmCallback






def approximate_flat_field():
    # median of full map
    # Add some qualifier to reject the output if not good
    raise NotImplementedError()

def extract_dexela_dark_field():
    raise NotImplementedError()


def determine_img_background():
    # for determining individual backgrounds. Should account for variations in beam intensity and non-Bragg X-ray scattering
    raise NotImplementedError()


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


def find_outlier_pixels(data, size=2, tolerance=3, significance=None):
    '''
    data        (arr)   Input 2D image. Cannot handle higher dimensional data yet...   
    size        (float) Size of median window. 2 by default only accounts for nearest neighbor pixels
    tolerance   (float) Multiplier value above which to consider a pixel as an outlier. By default 3 times the median value
    signficance (float) Signficance value below which to ignore contributions from noise fluctuations. 
                        Set to 5 * data standard deviation. Causes issues with Bragg peaks currently.
    TODO better account for significance. Better account for multidimensional data...
    '''

    med_img = median_filter(data, size=size, mode='mirror')

    ratio_img = np.abs(data / med_img)
    diff_img = np.abs(data - med_img)

    if significance is None:
        significance = 5 * np.std(data)

    replace_mask = (ratio_img > tolerance) & (diff_img > significance)
    fixed_img = np.copy(data)
    fixed_img[replace_mask] = med_img[replace_mask]
    return fixed_img


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


def rescale_array(arr, lower=0, upper=1, arr_min=None, arr_max=None):
    # Works for arrays of any size including images!
    if arr_min is None:
        arr_min = np.min(arr)
    if arr_max is None:
        arr_max = np.max(arr)
    ext = upper - lower
    
    scaled_arr = lower + ext * ((arr - arr_min) / (arr_max - arr_min))
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







'''def find_outlier_pixels1(data,tolerance=3,worry_about_edges=True):
    #This function finds the hot or dead pixels in a 2D dataset. 
    #tolerance is the number of standard deviations used to cutoff the hot pixels
    #If you want to ignore the edges and greatly speed up the code, then set
    #worry_about_edges to False.
    #
    #The function returns a list of hot pixels and also an image with with hot pixels removed

    from scipy.ndimage import median_filter
    blurred = median_filter(data, size=2)
    difference = data - blurred
    threshold = tolerance*np.std(difference)

    #find the hot pixels, but ignore the edges
    hot_pixels = np.nonzero((np.abs(difference[1:-1,1:-1])>threshold) )
    hot_pixels = np.array(hot_pixels) + 1 #because we ignored the first row and first column

    fixed_image = np.copy(data) #This is the image with the hot pixels removed
    for y,x in zip(hot_pixels[0],hot_pixels[1]):
        fixed_image[y,x]=blurred[y,x]
        
    if worry_about_edges == True:
        height,width = np.shape(data)
    
        ###Now get the pixels on the edges (but not the corners)###

        #left and right sides
        for index in range(1,height-1):
            #left side:
            med  = np.median(data[index-1:index+2,0:2])
            diff = np.abs(data[index,0] - med)
            if diff>threshold: 
                hot_pixels = np.hstack(( hot_pixels, [[index],[0]]  ))
                fixed_image[index,0] = med
            
            #right side:
            med  = np.median(data[index-1:index+2,-2:])
            diff = np.abs(data[index,-1] - med)
            if diff>threshold: 
                hot_pixels = np.hstack(( hot_pixels, [[index],[width-1]]  ))
                fixed_image[index,-1] = med

        #Then the top and bottom
        for index in range(1,width-1):
            #bottom:
            med  = np.median(data[0:2,index-1:index+2])
            diff = np.abs(data[0,index] - med)
            if diff>threshold: 
                hot_pixels = np.hstack(( hot_pixels, [[0],[index]]  ))
                fixed_image[0,index] = med
            
            #top:
            med  = np.median(data[-2:,index-1:index+2])
            diff = np.abs(data[-1,index] - med)
            if diff>threshold: 
                hot_pixels = np.hstack(( hot_pixels, [[height-1],[index]]  ))
                fixed_image[-1,index] = med
                  
        ###Then the corners###

        #bottom left
        med  = np.median(data[0:2,0:2])
        diff = np.abs(data[0,0] - med)
        if diff>threshold: 
            hot_pixels = np.hstack(( hot_pixels, [[0],[0]]  ))
            fixed_image[0,0] = med
        
        #bottom right
        med  = np.median(data[0:2,-2:])
        diff = np.abs(data[0,-1] - med)
        if diff>threshold: 
            hot_pixels = np.hstack(( hot_pixels, [[0],[width-1]]  ))
            fixed_image[0,-1] = med

        #top left
        med  = np.median(data[-2:,0:2])
        diff = np.abs(data[-1,0] - med)
        if diff>threshold: 
            hot_pixels = np.hstack(( hot_pixels, [[height-1],[0]]  ))
            fixed_image[-1,0] = med

        #top right
        med  = np.median(data[-2:,-2:])
        diff = np.abs(data[-1,-1] - med)
        if diff>threshold: 
            hot_pixels = np.hstack(( hot_pixels, [[height-1],[width-1]]  ))
            fixed_image[-1,-1] = med
    
    return hot_pixels,fixed_image'''