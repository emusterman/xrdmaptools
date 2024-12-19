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
from matplotlib.widgets import Slider

from xrdmaptools.plot.image_stack import base_slider_plot
from xrdmaptools.utilities.math import arbitrary_center_of_mass


def rotation_scale_translation_registration(
                    ref_img,
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
def relative_correlation_auto_alignment(image_stack, **kwargs):
    # I do not yet know how to do this for rotation. Maybe just rotate the x and y shifts...
    
    shifts_list = []
    for i in range(len(image_stack)):
        shifts = [rotation_scale_translation_registration(
                            image_stack[i],
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


def neighbor_correlation_auto_alignment(image_stack,
                                        center_index=None,
                                        **kwargs):
    
    if (center_index is None
        or center_index > len(image_stack)):
        center_index = int(len(image_stack) / 2)
    
    shifts_list = []
    for i in range(len(image_stack) - 1):
        shifts = rotation_scale_translation_registration(
                            image_stack[i],
                            image_stack[i + 1],
                            fix_rotation=True,
                            **kwargs)[-1]
        shifts_list.append(shifts)
    
    shifts_arr = np.asarray(shifts_list)

    # return shifts_arr

    full_shifts = [np.array([0, 0])]
    for i in range(len(image_stack) - 1):
        full_shifts.append(np.round(full_shifts[-1] + shifts_arr[i], 4))

    return tuple([(x, y) for y, x in full_shifts])

    # return full_shifts, shifts_arr

    # y_diff_list = []
    # for i in range(len(image_stack)):
    #     diff = [shifts_arr[i, idx, 0] - shifts_arr[i, idx - 1, 0]
    #                 for idx in range(1, len(image_stack))]
    #     y_diff_list.append(diff)
    # med_y_diffs = np.median(np.asarray(y_diff_list), axis=0)

    # x_diff_list = []
    # for i in range(len(image_stack)):
    #     diff = [shifts_arr[i, idx, 1] - shifts_arr[i, idx - 1, 1]
    #                 for idx in range(1, len(image_stack))]
    #     x_diff_list.append(diff)
    # med_x_diffs = np.median(np.asarray(x_diff_list), axis=0)

    # med_x_shifts = [0]
    # _ = [med_x_shifts.append(med_x_shifts[-1] + diff) for diff in med_x_diffs]
    # med_x_shifts = np.round(med_x_shifts, 3)

    # med_y_shifts = [0]
    # _ = [med_y_shifts.append(med_y_shifts[-1] + diff) for diff in med_y_diffs]
    # med_y_shifts = np.round(med_y_shifts, 3)

    # return tuple([(y, x) for y, x in zip(med_y_shifts, med_x_shifts)])


def com_auto_alignment(image_stack):

    yy, xx = np.meshgrid(*[range(dim) for dim in image_stack[0].shape], indexing='ij')

    coms = [arbitrary_center_of_mass(image, yy, xx) for image in image_stack]

    shifts = np.round([np.asarray(coms[0]) - np.asarray(com) for com in coms], 3)
    shifts = tuple([(shift_y, shift_x) for shift_y, shift_x in shifts])

    return shifts
    

def manual_alignment(image_stack,
                     slider_vals=None,
                     slider_label='Index',
                     vmin=None,
                     vmax=None,
                     **kwargs):
    # Built on interactive plotting functionality    

    fig = plt.figure(figsize=(5, 5), dpi=200)
    fig.suptitle('Manual Map Alignment')
    ax = fig.add_axes([0.1, 0.1, 0.5, 0.8])

    # ax.set_xlim(0, image_stack[0].shape[1])
    # ax.set_ylim(0, image_stack[0].shape[0])

    if vmin is None:
        img_vmin = np.min(image_stack)
    else:
        img_vmin = vmin
    if vmax is None:
        img_vmax = np.max(image_stack)
    else:
        img_vmax = vmax

    # Define placeholder values
    image = ax.imshow(image_stack[0],
                      vmin=img_vmin,
                      vmax=img_vmax,
                      **kwargs)
    image_index = 0
    marker_list = []
    marker_coords = []
    for i in range(len(image_stack)):
        marker = ax.scatter([], [],
                            marker='+',
                            linewidth=1,
                            color='red')
        marker.set_visible(False)
        marker_list.append(marker)
        marker_coords.append((np.nan, np.nan))
    
    if slider_vals is None:
        slider_vals = np.asarray((range(len(image_stack))))
    else:
        slider_vals = np.asarray(slider_vals)

        is_sorted = all(a <= b for a, b in zip(slider_vals, slider_vals[1:]))
        if not is_sorted:
            raise ValueError('Slider values must be sorted sequentially.')
        
    ax.set_title(f'{slider_vals[0]}')
    slider_ax = fig.add_axes([0.7, 0.1, 0.03, 0.8])
    slider = Slider(
        ax=slider_ax,
        label=slider_label,
        valmin=slider_vals[0],
        valmax=slider_vals[-1],
        valinit=slider_vals[0],
        valstep=slider_vals,
        orientation='vertical'
    )

    # The function to be called anytime a slider's value changes
    def update_image(val):
        nonlocal image, image_index, marker_list, marker_coords, slider

        new_index = np.argmin(np.abs(slider_vals - val))
        if new_index == image_index:
            return
        
        # Hide old marker
        marker_list[image_index].set_visible(False)

        # Get new image index
        image_index = new_index

        # Update image
        image.set_data(image_stack[image_index])
        
        if vmin is None:
            img_vmin = np.min(image_stack[image_index])
        else:
            img_vmin = vmin
        if vmax is None:
            img_vmax = np.max(image_stack[image_index])
        else:
            img_vmax = vmax
        
        image.set_clim(img_vmin, img_vmax)

        # Unhide new marker, if already found
        if not np.all(np.isnan(marker_coords[image_index])):
            marker_list[image_index].set_visible(True)

        ax.set_title(f'{slider_vals[image_index]}')
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes == ax:
            nonlocal image_index, marker_list, marker_coords
            marker_coords[image_index] = event.ydata, event.xdata

            # Having no datapoints may already hide the markers
            if not marker_list[image_index]._visible:
                marker_list[image_index].set_visible(True)

            marker_list[image_index].set_offsets(
                [*marker_coords[image_index][::-1]] # reverse to make (x, y)
                )
            fig.canvas.draw_idle()

    def on_close(event):
        nonlocal marker_coords
        if np.any(np.isnan(np.asarray(marker_coords))):
            warn_str = ('WARNING: manual map alignment plot closed '
                        + 'before all maps were properly aligned.')
            print(warn_str)
        
        manual_shifts = np.round([np.asarray(marker_coords[0]) - np.asarray(coords)
                                  for coords in marker_coords], 3)
        marker_coords = tuple([(shift_y, shift_x) for shift_y, shift_x in manual_shifts])
        #return marker_coords

    slider.on_changed(update_image)
    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('close_event', on_close)

    #plt.show()
    plt.show(block=True)
    plt.pause(0.01)
    return marker_coords


# Can this be built on base_slider_plot???
def test_manual_alignment(image_stack,
                          slider_vals=None,
                          **kwargs):
    # Built on interactive plotting functionality

    fig, ax, slider = base_slider_plot(
                        image_stack,
                        slider_vals=slider_vals,
                        **kwargs
                        )   
    fig.suptitle('Manual Map Alignment')

    # Define placeholder values
    image = ax.imshow(image_stack[0])
    marker_list = []
    marker_coords = []
    for i in range(len(image_stack)):
        marker = ax.scatter([], [],
                            marker='+',
                            linewidth=1,
                            color='red')
        marker.set_visible(False)
        marker_list.append(marker)
        marker_coords.append((np.nan, np.nan))

    def on_click(event):
        if event.inaxes == ax:
            nonlocal slider, marker_list, marker_coords
            image_index = np.nonzero(slider_vals == slider.val)[0][0]
            marker_coords[image_index] = event.ydata, event.xdata

            # Having no datapoints may already hide the markers
            if not marker_list[image_index]._visible:
                marker_list[image_index].set_visible(True)

            marker_list[image_index].set_offsets(
                [*marker_coords[image_index][::-1]] # reverse to make (x, y)
                )
            fig.canvas.draw_idle()

    def on_close(event):
        nonlocal marker_coords
        if np.any(np.isnan(np.asarray(marker_coords))):
            warn_str = ('WARNING: manual map alignment plot closed '
                        + 'before all maps were properly aligned.')
            print(warn_str)
        
        manual_shifts = np.round([np.asarray(marker_coords[0]) - np.asarray(coords)
                                  for coords in marker_coords], 3)
        marker_coords = tuple([(shift_y, shift_x) for shift_y, shift_x in manual_shifts])
        #return marker_coords

    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('close_event', on_close)

    #plt.show()
    plt.show(block=True)
    plt.pause(0.01)
    return marker_coords