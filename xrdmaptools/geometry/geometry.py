import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.transform import Rotation

# Local imports
from xrdmaptools.utilities.math import vector_angle
from xrdmaptools.utilities.utilities import delta_array


# TODO:
# Should be an exact transform between image and polar coordinates.
# Not sure if it is worth the effort

def _parse_rotation_input(rotation,
                          input_name='rotation',
                          rotation_axis='y',
                          degrees=False):

    if rotation_axis.lower() not in ['x', 'y', 'z']:
        err_str = (f"Unknown rotation_axis {rotation_axis}. Only 'x', "
                   + "'y', and 'z' axes supported.")
        raise ValueError(err_str)
    axis_index = ['x', 'y', 'z'].index(rotation_axis.lower())
    rotvec = [0, 0, 0]

    if isinstance(rotation, Rotation):
        pass
    elif isinstance(rotation, np.ndarray):
        if rotation.squeeze().shape == ():
            rotvec[axis_index] = rotation
            rotation = Rotation.from_rotvec(rotvec, degrees=degrees)
        elif rotation.shape != (3, 3):
            err_str = (f'{input_name} as array must have shape '
                        + f'(3, 3) not {rotation.shape}.')
            raise ValueError(err_str)
        else:
            rotation = Rotation.from_matrix(rotation)
    elif isinstance(rotation, (float, int)):
        rotvec[axis_index] = rotation
        rotation = Rotation.from_rotvec(rotvec, degrees=degrees)
    else:
        err_str = (f'Unknown {input_name} of type '
                   + f'({type(rotation)}). Must be given as '
                   + '(3, 3) array or scipy Rotation class.')
    return rotation


def get_q_vect(tth,
               chi,
               wavelength,
               stage_rotation=None,
               rotation_axis='y',
               return_kf=False,
               degrees=False):
    # Calculate q-vector from arrays of tth and chi polar coordinates and wavelength
    if not isinstance(tth, (list, tuple, np.ndarray)):
        tth = np.asarray([tth])
        chi = np.asarray([chi])
    if tth.shape != chi.shape:
        err_str = (f'tth shape of {tth.shape} does not match chi of '
                   + f'shape {chi.shape}')
        raise ValueError(err_str)
    
    if degrees:
        tth = np.radians(tth)
        chi = np.radians(chi)

    # Incident wavevector
    ki_unit = np.broadcast_to(np.array([0, 0, 1]).reshape(
                                       3,
                                       *([1,] * len(tth.shape))),
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
    q_vect = np.moveaxis(q_vect, 0, -1) # copies data...

    # stage rotation is rotation OUT of sample reference frame
    if stage_rotation is not None:
        rotation = _parse_rotation_input(
                            stage_rotation,
                            'stage_rotation',
                            rotation_axis=rotation_axis,
                            degrees=degrees)
        # Bring rotated q_vect back into sample reference frame
        q_vect = rotation.apply(q_vect.reshape(-1, 3),
                                inverse=True).reshape(q_vect.shape)

    return q_vect


# Will give polar coordinates and wavelength or stage rotation 
# to bring a q_vector incident with the Ewald sphere
# Could probably be better generalized...
def q_2_polar(q_vect,
              wavelength=None,
              stage_rotation=None,
              degrees=False,
              rotation_axis='y'):

    if wavelength is None and stage_rotation is None:
        warn_str = ('WARNING: Neither wavelength nor stage_rotation '
                    + 'are indicated. Assuming stage rotation is '
                    + '0 deg and finding wavelength.')
        print(warn_str)
        stage_rotation = Rotation.from_array(np.eye(3))
    
    # stage rotation is rotation OUT of sample reference frame
    if stage_rotation is not None:
        if rotation_axis.lower() not in ['x', 'y']:
            if rotation_axis.lower() == 'z':
                err_str = ("Rotation about the z-axis (beam direction)"
                           + " only changes the azimuthal angle. "
                           + "Rotate about the 'x'- or 'y' to change "
                           + "the Bragg condition.")
            else:
                err_str = ("Axis rotation only supported for 'x'- or "
                           + "'y'-axes; not "
                           + f"({rotation_axis.lower()}).")
            raise ValueError(err_str)
            
        stage_rotation = _parse_rotation_input(
                                stage_rotation,
                                'stage_rotation',
                                rotation_axis=rotation_axis,
                                degrees=degrees)
        
        # Bring rotated q_vect back into sample reference frame
        q_vect = stage_rotation.apply(q_vect)
    
    # Probably a shape check that should be made on the last axis
    q_vect = np.asarray(q_vect)
    if q_vect.shape[-1] != 3:
        err_str = (f'q_vect has shape {q_vect.shape}, '
                   + 'but last axis should be 3.')
        raise ValueError(err_str)
    q_norm = np.linalg.norm(q_vect, axis=-1)

    # Find transformed q_vect if stage is rotated
    if stage_rotation is None:
        # Find tth
        tth = 2 * np.arcsin(q_norm * wavelength / (4 * np.pi))
        new_qz = (2 * np.pi / wavelength) * (np.cos(tth) - 1)

        # TODO: Determine sign of rotation
        if rotation_axis == 'y':
            new_qy = q_vect[..., 1] # Constant qy
            new_qx = np.sqrt(q_norm**2 - new_qz**2 - new_qy**2)

            # Rotation magnitude
            output = np.arccos((q_vect[..., 2] * new_qz 
                                + q_vect[..., 0] * new_qx)
                            / (q_vect[..., 2]**2 + q_vect[..., 0]**2))

            # Get the sign from the curl about y-axis
            # Probably a better way to do this
            sign = np.sign(((new_qx - q_vect[..., 0]) / (new_qz + q_vect[..., 2]))
                           - ((new_qz - q_vect[..., 2]) / (new_qx + q_vect[..., 0])))
            output *= sign

            q_vect = np.array([new_qx, new_qy, new_qz]).T

        elif rotation_axis == 'x':
            new_qx = q_vect[..., 0] # Constant qx
            new_qy = np.sqrt(q_norm**2 - new_qz**2 - new_qx**2)

            # Rotation magnitude
            output = np.arccos((q_vect[..., 2] * new_qz
                                + q_vect[..., 1] * new_qy)
                            / (q_vect[..., 2]**2 + q_vect[..., 1]**2))
            
            # Get the sign from the curl about y-axis
            # Probably a better way to do this
            sign = np.sign(((new_qz - q_vect[..., 2]) / (new_qy + q_vect[..., 1]))
                           - ((new_qy - q_vect[..., 1]) / (new_qz + q_vect[..., 2])))
            output *= sign

            q_vect = np.array([new_qx, new_qy, new_qz]).T
    
    else:
        # Determine tth
        tth = 2 * ((np.pi / 2)
                   - np.arccos(-q_vect[..., 2] / q_norm))

        # Negative tth values are nonsensical
        if isinstance(tth, np.ndarray):
            tth[tth <= 0] = np.nan
        elif tth <= 0: # Single q_vect evaluation
            tth = np.nan

        # Determine wavelength from Ewald sphere radius
        output = (4 * np.pi / q_norm) * np.sin(tth / 2)

    # Find chi
    chi = np.arctan2(q_vect[..., 1],
                     q_vect[..., 0], dtype=np.float64)
    if degrees:
        tth = np.degrees(tth)
        chi = np.degrees(chi)
        if stage_rotation is None:
            output = np.degrees(output)

    return tth, chi, output



# Deprecated. Only finds wavelength
def old_q_2_polar(q_vect, wavelength=None, degrees=False):
    q_vect = np.asarray(q_vect)
    q_norm = np.linalg.norm(q_vect, axis=-1)

    # Find tth and chi
    theta = np.pi / 2 - vector_angle(q_vect,
                                     [0, 0, -1],
                                     degrees=False) # always false
    tth = 2 * theta
    chi = np.arctan2(q_vect[..., 1],
                     q_vect[..., 0])

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
    
    tth, chi, _ = q_2_polar(q_near,
                            stage_rotation=0,
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

    est_img_coords = estimate_image_coords(np.array([tth, chi]).T,
                                           tth_arr,
                                           chi_arr,
                                           method)

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
        err_str = (f'tth_arr shape {tth_arr.shape} does not match '
                   + f'chi_arr shape {chi_arr}')
        raise ValueError(err_str)

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
    tth_interp = RegularGridInterpolator((img_x, img_y),
                                         tth_arr.T,
                                         method=method)
    chi_interp = RegularGridInterpolator((img_x, img_y),
                                         chi_arr.T,
                                         method=method)

    est_tth = tth_interp((x_coords, y_coords))
    est_chi = chi_interp((x_coords, y_coords))
    est_chi = modular_azimuthal_reshift(est_chi, max_arr=max_arr)
    
    # Given as np.array([[tth0, chi0], [tth1, chi1], ...])
    return np.array([est_tth, est_chi]).T 


def estimate_image_coords(coords, tth_arr, chi_arr, method='nearest'):
    # Warning: Any method except 'nearest' is fairly slow and not recommended
    #coords = np.array([[tth0, chi0], [tth1, chi1], ...])
    # TODO:
    # Check coord values to make sure they are in range
    if tth_arr.shape != chi_arr.shape:
        err_str = (f'tth_arr shape {tth_arr.shape} does not match '
                   + f'chi_arr shape {chi_arr}')
        raise ValueError(err_str)

    # For better indexing
    coords = np.asarray(coords)

    # Shift azimuthal discontinuities
    chi_arr, max_arr, shifted = modular_azimuthal_shift(chi_arr)
    coords[:, 1], _, _ = modular_azimuthal_shift(coords[:, 1],
                                                 max_arr=max_arr,
                                                 force_shift=shifted)

    # Combine into large polar vector
    polar_arr = np.array([tth_arr.ravel(), chi_arr.ravel()]).T

    # Image shapes are VxH
    image_shape = tth_arr.shape
    img_x = np.arange(0, image_shape[1])
    img_y = np.arange(0, image_shape[0])
    img_xx, img_yy = np.meshgrid(img_x, img_y)
    img_arr = np.array([img_xx.ravel(), img_yy.ravel()]).T

    # griddata for unstructured data. Fairly slow with any method but nearest
    est_img_coords = griddata(polar_arr,
                              img_arr,
                              coords,
                              method=method)
    est_img_coords = np.round(est_img_coords).astype(np.int32)
    return est_img_coords # Given as np.array([[x0, y0], [x1, y1], ...])


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