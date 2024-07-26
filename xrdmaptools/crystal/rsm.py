# Submodule for processing and plotting 3D reciprocal space mapping.
# Strain analysis will be reserved for strain.py in this same submodule.
# Some 3D spot and blob search functions may be found here or strain.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from skimage.filters import window, difference_of_gaussians
from scipy.fft import fft2, fftshift
from skimage.transform import warp_polar, rotate
from skimage.registration import phase_cross_correlation




# Function for mapping vectorized data onto regular grid
def map_2_grid(q_dset, gridstep=0.005):
    
    all_qx = q_dset[0]
    all_qy = q_dset[1]
    all_qz = q_dset[2]
    all_int = q_dset[3]

    # Find bounds
    x_min = np.min(all_qx)
    x_max = np.max(all_qx)
    y_min = np.min(all_qy)
    y_max = np.max(all_qy)
    z_min = np.min(all_qz)
    z_max = np.max(all_qz)

    # Generate q-space grid
    xx = np.linspace(x_min, x_max, int((x_max - x_min) / gridstep))
    yy = np.linspace(y_min, y_max, int((y_max - y_min) / gridstep))
    zz = np.linspace(z_min, z_max, int((z_max - z_min) / gridstep))

    grid = np.array(np.meshgrid(xx, yy, zz, indexing='ij'))
    grid = grid.reshape(3, -1).T

    points = np.array([all_qx, all_qy, all_qz]).T

    int_grid = griddata(points, all_int, grid, method='nearest')
    #int_grid = int_grid.reshape(yy.shape[0], xx.shape[0], zz.shape[0]).T
    int_grid = int_grid.reshape(xx.shape[0], yy.shape[0], zz.shape[0])

    return np.array([*np.meshgrid(xx, yy, zz, indexing='ij'), int_grid])


def map_registration():
    raise NotImplementedError()



def rotation_scale_translation(ref_img,
                               mov_img,
                               rotation_upsample=1000,
                               shift_upsample=1000,
                               bandpass=(0, 1),
                               fix_rotation=False):
    
    # Get rotation component
    if not fix_rotation:
        # Image bandpass
        ref_img = difference_of_gaussians(ref_img, *bandpass)
        mov_img = difference_of_gaussians(mov_img, *bandpass)

        # Window images
        ref_wimage = ref_img * window('hann', ref_img.shape)
        mov_wimage = mov_img * window('hann', ref_img.shape)

        # work with shifted FFT magnitudes
        ref_fs = np.abs(fftshift(fft2(ref_wimage)))
        mov_fs = np.abs(fftshift(fft2(mov_wimage)))

        # Create log-polar transform
        shape = ref_fs.shape
        radius = shape[0] // 8  # only take lower frequencies
        warped_ref_fs = warp_polar(ref_fs, radius=radius, output_shape=shape,
                                scaling='log', order=0)
        warped_mov_fs = warp_polar(mov_fs, radius=radius, output_shape=shape,
                                scaling='log', order=0)
        
        # Register shift in polar space with half of FFT
        warped_ref_fs = warped_ref_fs[:shape[0] // 2, :]  # only use half of FFT
        warped_mov_fs = warped_mov_fs[:shape[0] // 2, :]
        (polar_shift,
        polar_error,
        polar_phasediff) = phase_cross_correlation(
                                        warped_ref_fs,
                                        warped_mov_fs,
                                        upsample_factor=rotation_upsample,
                                        normalization=None)
        
        # Convert to useful parameters
        shiftr, shiftc = polar_shift[:2]
        angle = (360 / shape[0]) * shiftr
        klog = shape[1] / np.log(radius)
        scale = np.exp(shiftc / klog)

        # Correct image and register translation
        # adj_image = rescale(moving_image, 2 - scale) # this seem wrong
        adj_img = rotate(mov_img, -angle)
        # adj_image = rescale
    else:
        angle = 0
        scale = 0
        adj_img = mov_img

    # Get translation component
    (shift,
    trans_error,
    trans_phasediff) = phase_cross_correlation(
                                        ref_img,
                                        adj_img,
                                        upsample_factor=shift_upsample,
                                        normalization=None)
    
    return angle, scale, shift


# When registering a stack to a reference image, poor quality maps may fail
# This method uses each map as the reference
# The median of the relative distances between images is used to reconstruct the shifts
# Relative to the first image
def relative_align_maps(image_stack, **kwargs):
    # I do not yet know how to do this for rotation. Maybe just rotate the x and y shifts...
    
    shifts_list = []
    for i in range(len(image_stack)):
        shifts = [rotation_scale_translation(image_stack[i],
                                             arr,
                                             fix_rotation=True,
                                             **kwargs)[-1]
                    for arr in image_stack]
        shifts_list.append(shifts)
    
    shifts_arr = np.asarray(shifts_list)

    y_diff_list = []
    for i in range(len(image_stack)):
        diff = [shifts_arr[i, idx, 0] - shifts_arr[i, idx - 1, 0]
                    for idx in range(1, len(image_stack))]
        y_diff_list.append(diff)
    med_y_diffs = np.median(np.asarray(y_diff_list), axis=0)

    x_diff_list = []
    for i in range(len(image_stack)):
        diff = [shifts_arr[i, idx, 1] - shifts_arr[i, idx - 1, 1]
                    for idx in range(1, len(image_stack))]
        x_diff_list.append(diff)
    med_x_diffs = np.median(np.asarray(x_diff_list), axis=0)

    med_x_shifts = [0]
    _ = [med_x_shifts.append(med_x_shifts[-1] + diff) for diff in med_x_diffs]
    med_x_shifts = np.round(med_x_shifts, 3)

    med_y_shifts = [0]
    _ = [med_y_shifts.append(med_y_shifts[-1] + diff) for diff in med_y_diffs]
    med_y_shifts = np.round(med_y_shifts, 3)

    return tuple([(y, x) for y, x in zip(med_y_shifts, med_x_shifts)])
    

