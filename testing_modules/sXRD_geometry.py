import numpy as np
from collections import OrderedDict


# TODO: a lot



def estimate_img_coords(coords, image_shape, tth=None, chi=None):
    if len(coords) == 0:
        return coords
    
    # Estimate image coordinates from tth and chi values
    tth_i = np.asarray(coords[0])
    chi_i = np.asarray(coords[1])
    x = (tth_i - np.min(tth)) / (np.max(tth) - np.min(tth)) * image_shape[1]
    y = image_shape[0] - (chi_i-  np.min(chi)) / (np.max(chi) - np.min(chi)) * image_shape[0]
    return np.array([x.astype(np.int32), y.astype(np.int32)])


def estimate_reciprocal_coords(coords, image_shape, tth=None, chi=None):
    if len(coords) == 0:
        return coords
    
    # Convert image coordinates to tth and chi values
    x_i = np.asarray(coords[0])
    y_i = np.asarray(coords[1])

    # Convert image coordinates to tth and chi values
    tth_lst = np.min(tth) + (np.max(tth) - np.min(tth)) * x_i / image_shape[1]
    chi_lst = np.min(chi) + (np.max(chi) - np.min(chi)) * (image_shape[0] - y_i) / image_shape[0]
    return np.array([tth_lst, chi_lst])