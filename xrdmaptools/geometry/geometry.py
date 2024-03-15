import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator


# TODO: a lot
# Should be an exact transoform between image and polar coordinates. Possibly in pyFAI for individual coordinates
# Not sure if it is worth the effort


def estimate_polar_coords(coords, tth_arr, chi_arr, method='linear'):
    #coords = np.array([[0, 0], [767, 485], [x, y], ...])
    # TODO:
    # Check coord values to make sure they are in range
    if tth_arr.shape != chi_arr.shape:
        raise ValueError(f"tth_arr shape {tth_arr.shape} does not match chi_arr shape {chi_arr}")

    x_coords, y_coords = np.asarray(coords).T

    # Image shapes are VxH
    image_shape = tth_arr.shape
    img_x = np.arange(0, image_shape[1])
    img_y = np.arange(0, image_shape[0])

    # Regular grid rather than griddata
    # These can probably combined, but this works fine
    tth_interp = RegularGridInterpolator((img_x, img_y), tth_arr.T, method=method)
    chi_interp = RegularGridInterpolator((img_x, img_y), chi_arr.T, method=method)

    est_tth = tth_interp((x_coords, y_coords))
    est_chi = chi_interp((x_coords, y_coords))

    return np.array([est_tth, est_chi]).T


def estimate_image_coords(coords, tth_arr, chi_arr, method='nearest'):
    # Warning: Any method except 'nearest' is fairly slow and not recommended
    #coords = np.array([[0, 0], [767, 485], [x, y], ...])
    # TODO:
    # Check coord values to make sure they are in range
    if tth_arr.shape != chi_arr.shape:
        raise ValueError(f"tth_arr shape {tth_arr.shape} does not match chi_arr shape {chi_arr}")

    polar_arr = np.array([tth_arr.ravel(), chi_arr.ravel()]).T

    # Image shapes are VxH
    image_shape = tth_arr.shape
    img_x = np.arange(0, image_shape[1])
    img_y = np.arange(0, image_shape[0])
    img_xx, img_yy = np.meshgrid(img_x, img_y)
    img_arr = np.array([img_xx.ravel(), img_yy.ravel()]).T

    # griddata for unstructured data. Fairly slow with any method but nearest
    est_img_coords = griddata(polar_arr, img_arr, coords, method=method)
    est_img_coords = np.round(est_img_coords).astype(np.int32)
    return est_img_coords


# deprecated
'''def estimate_img_coords(coords, image_shape, tth=None, chi=None):
    if len(coords) == 0:
        return coords
    
    # Estimate image coordinates from tth and chi values
    tth_i = np.asarray(coords[0])
    chi_i = np.asarray(coords[1])
    x = (tth_i - np.min(tth)) / (np.max(tth) - np.min(tth)) * image_shape[1]
    y = image_shape[0] - (chi_i-  np.min(chi)) / (np.max(chi) - np.min(chi)) * image_shape[0]
    return np.array([x.astype(np.int32), y.astype(np.int32)])'''


# deprecated
'''def estimate_reciprocal_coords(coords, image_shape, tth=None, chi=None):
    if len(coords) == 0:
        return coords
    
    # Convert image coordinates to tth and chi values
    x_i = np.asarray(coords[0])
    y_i = np.asarray(coords[1])

    # Convert image coordinates to tth and chi values
    tth_lst = np.min(tth) + (np.max(tth) - np.min(tth)) * x_i / image_shape[1]
    chi_lst = np.min(chi) + (np.max(chi) - np.min(chi)) * (image_shape[0] - y_i) / image_shape[0]
    return np.array([tth_lst, chi_lst])'''