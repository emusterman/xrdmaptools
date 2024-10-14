import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator

# Local imports
from xrdmaptools.utilities.math import vector_angle
from xrdmaptools.utilities.utilities import delta_array


# TODO: a lot
# Should be an exact transoform between image and polar coordinates. Possibly in pyFAI for individual coordinates
# Not sure if it is worth the effort

# Create function to get wavelength, tth, and chi from q_vect
# Create function to estimate nearest pixel from q_vect
# Create function to estimate nearest tth and chi on detector from nearest 


def get_q_vect(tth, chi, wavelength, return_kf=False, degrees=False):
    # Calculate q-vector from arrays of tth and chi polar coordinates and wavelength
    if not isinstance(tth, (list, tuple, np.ndarray)):
        tth = np.asarray([tth])
        chi = np.asarray([chi])
    if len(tth) != len(chi):
        raise ValueError("Length of tth does not match length of chi.")
    
    if degrees:
        tth = np.radians(tth)
        chi = np.radians(chi)

    # Incident wavevector
    ki_unit = np.broadcast_to(np.array([0, 0, 1]).reshape(3, *([1,] * len(tth.shape))),
                              (3, *tth.shape))

    # Diffracted wavevector
    kf_unit = np.array([np.sin(tth) * np.cos(chi),
                        np.sin(tth) * np.sin(chi),
                        np.cos(tth)])
    
    if return_kf:
        return 2 * np.pi / wavelength * kf_unit
    
    delta_k = kf_unit - ki_unit

    # Scattering vector with origin set at transmission (0, 0, 0)
    q_vect = 2 * np.pi / wavelength * delta_k

    return q_vect


def q_2_polar(q_vect, wavelength=None, degrees=False):
    q_vect = np.asarray(q_vect)
    q_norm = np.linalg.norm(q_vect, axis=-1)

    # Find tth and chi
    theta = np.pi / 2 - vector_angle(q_vect, [0, 0, -1], degrees=False) # always false
    tth = 2 * theta
    chi = np.arctan2(q_vect[..., 1], q_vect[..., 0])

    if degrees:
        tth = np.degrees(tth)
        chi = np.degrees(chi)

    # Get radius of Ewald sphere and wavelength
    if wavelength is None:
        r = 0.5 * q_norm / np.sin(theta)
        wavelength = 2 * np.pi / r

    return tth, chi, wavelength


# Find nearest q-space position on Ewald sphere within a certain threshold
def nearest_q_on_ewald(q_vect, wavelength, near_thresh=None):
    # Currently no way to bound within surface measured by detector
    
    q_vect = np.asarray(q_vect)

    r = 2 * np.pi / wavelength # radius of Ewald sphere
    qx = q_vect[..., 0] 
    qy = q_vect[..., 1]
    qz = q_vect[..., 2] + r # shift vector origin to center of Ewald sphere
    q_norm = np.linalg.norm([qx, qy, qz], axis=0) # evil things with axis ordering

    # Solving parametric equation of a vector intercepting the Ewald sphere
    t = r / q_norm

    # Nearest q_vect, with transmission center
    qx_near = t * qx
    qy_near = t * qy
    qz_near = t * qz - r

    # Distance normal to Ewald sphere
    # Positive means input is outside of Ewald sphere
    q_dist = q_norm - np.linalg.norm([qx_near, qy_near, qz_near + r], axis=0)

    # Mask out values too far away from Ewald sphere
    if near_thresh is not None:
        dist_mask = np.abs(q_dist) > near_thresh
        qx_near[dist_mask] = np.nan
        qy_near[dist_mask] = np.nan
        qz_near[dist_mask] = np.nan
        q_dist[dist_mask] = np.nan

    # Transpose necessary to maintain input shape. Maybe best to shift all q_vectors to be consistent
    return np.array([qx_near, qy_near, qz_near]).T, q_dist


def nearest_polar_on_ewald(q_vect, wavelength,
                           near_thresh=None, degrees=False):

    q_near, q_dist = nearest_q_on_ewald(q_vect,
                                        wavelength,
                                        near_thresh=near_thresh)
    
    tth, chi, wavelength = q_2_polar(q_near,
                                     wavelength=wavelength,
                                     degrees=degrees)
    
    return tth, chi


def nearest_pixels_on_ewald(q_vect, wavelength, tth_arr, chi_arr,
                            near_thresh=None, degrees=False, method='nearest'):
    # degrees bool must match tth_arr and chi_arr input values
    
    tth, chi = nearest_polar_on_ewald(q_vect,
                                      wavelength,
                                      near_thresh=near_thresh,
                                      degrees=degrees)
    
    if near_thresh is not None:
        nan_mask = np.any([np.isnan(tth), np.isnan(chi)], axis=0)
        tth = tth[~nan_mask]
        chi = chi[~nan_mask]

    est_img_coords = estimate_image_coords(np.array([tth, chi]).T, tth_arr, chi_arr, method)

    bound_mask = np.any([
        est_img_coords[:, 0] <= 0,
        est_img_coords[:, -1] <= 0,
        est_img_coords[:, 0] >= tth_arr.shape[1] - 1,
        est_img_coords[:, -1] >= tth_arr.shape[0] - 1
    ], axis=0)

    return est_img_coords[~bound_mask]
    

def estimate_polar_coords(coords, tth_arr, chi_arr, method='linear'):
    #coords = np.array([[0, 0], [767, 485], [x0, y0], ...])
    # TODO:
    # Check coord values to make sure they are in range
    if tth_arr.shape != chi_arr.shape:
        raise ValueError(f"tth_arr shape {tth_arr.shape} does not match chi_arr shape {chi_arr}")

    # Not strictly necessary
    coords = np.asarray(coords)

    # Shift azimuthal discontinuties
    chi_arr, max_arr, _ = modular_azimuthal_shift(chi_arr)

    # Separate coords
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
    est_chi = modular_azimuthal_reshift(est_chi, max_arr=max_arr)

    return np.array([est_tth, est_chi]).T # Given as np.array([[tth0, chi0], [tth1, chi1], ...])


def estimate_image_coords(coords, tth_arr, chi_arr, method='nearest'):
    # Warning: Any method except 'nearest' is fairly slow and not recommended
    #coords = np.array([[tth0, chi0], [tth1, chi1], ...])
    # TODO:
    # Check coord values to make sure they are in range
    if tth_arr.shape != chi_arr.shape:
        raise ValueError(f"tth_arr shape {tth_arr.shape} does not match chi_arr shape {chi_arr}")

    # For better indexing
    coords = np.asarray(coords)

    # Shift azimuthal discontinuities
    chi_arr, max_arr, shifted = modular_azimuthal_shift(chi_arr)
    coords[:, 1], _, _ = modular_azimuthal_shift(coords[:, 1], max_arr=max_arr, force_shift=shifted)

    # Combine into large polar vector
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
    return est_img_coords # Given as np.array([[x0, y0], [x1, y1], ...])


# Does not yet figure wavelength
# Intent is to find exact wavelength, tth, and chi values 
'''def _q_2_polar_old(q_vect, wavelength, degrees=True):

    q_vect = np.asarray(q_vect)
    factor = 2 * np.pi / wavelength
    norm_q = q_vect / factor

    tth = np.asarray(np.arccos(norm_q[..., 2] + 1))

    #chi0 = np.asarray(-np.arcsin(norm_q[..., 1] / -np.sin(tth)))
    #chi0[q_vect[..., 0] < 0] = np.pi + chi0[q_vect[..., 0] < 0]
    #chi0[q_vect[..., 1] < 0] = -np.abs(chi0[q_vect[..., 1] < 0])

    #chi1 = np.asarray(-np.arccos(norm_q[..., 0] / -np.sin(tth)))
    #chi1[q_vect[..., 1] < 0] = -np.abs(chi1[q_vect[..., 1] < 0])

    # negative qx to switch to pyFAI coordinate system
    chi = np.arctan2(norm_q[..., 1], -norm_q[..., 0])

    if degrees:
        tth = np.degrees(tth)
        chi = np.degrees(chi)
        #chi1 = np.degrees(chi1)

    return tth, chi'''


def modular_azimuthal_shift(arr, max_arr=None, force_shift=None):
    arr = np.asarray(arr).copy()
    if max_arr is None:
        max_arr = np.max(np.abs(arr))

    if force_shift is None:
        delta_arr = delta_array(arr)
        force_shift = np.max(delta_arr) > max_arr

    # Modular shift values if there is a discontinuity
    if force_shift:
        shifted=True
        # Degrees
        if max_arr > np.pi: shift_value = 2 * 180
        # Radians
        else: shift_value = 2 * np.pi
        # Shift and recalculate
        arr[arr < 0] += shift_value

        new_max_arr = np.max(np.abs(arr))
    else:
        shifted = False
        new_max_arr = max_arr
    
    return arr, new_max_arr, shifted


def modular_azimuthal_reshift(arr, max_arr=None):
    arr = np.asarray(arr).copy()
    if max_arr is None:
        max_arr = np.max(np.abs(arr))

    # Degrees and shifted
    if max_arr > 180:
        shift_value = 2 * 180
        arr[arr > 180] -= shift_value

    # Degrees, but not shifted
    elif max_arr > 4 * np.pi:
        pass

    # Radians and shifted
    elif max_arr > np.pi:
        shift_value = 2 * np.pi
        arr[arr > np.pi] -= shift_value
    
    # Radians, but not shifted
    elif max_arr <= np.pi:
        pass

    return arr


def det_plane_from_ai(ai, skip=None):


    num_pixels = np.prod(*ai.detector.shape)

    if skip is None:
        # skip to about 2500 points
        skip = np.round(np.sqrt(num_pixels
                                / 2500), 0).astype(int)

    # 2, 1, 0 order to adjust to NSLS-II coordinate system
    points = np.asarray([ai.position_array()[::skip, ::skip, i].ravel()
                         for i in [2, 1, 0]])

    d = np.mean(points, axis=1, keepdims=True)
    svd = np.linalg.svd(points - d)

    # Return plane normal n = (a, b, c) and point (d)
    return svd[0][:, -1], d.squeeze()



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