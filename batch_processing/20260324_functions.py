import numpy as np
import os
from skimage import io
from tqdm import tqdm
from scipy.spatial.transform import Rotation


from xrdmaptools.plot.image_stack import base_slider_plot



def get_chi_window_stack(xdm, tth_window, chi_window, chi_step, **kwargs):
    
    tth_mask = (xdm.tth_arr >= tth_window[0]) & (xdm.tth_arr <= tth_window[1])

    chi_min = np.min(xdm.chi_arr[tth_mask])
    chi_max = np.max(xdm.chi_arr[tth_mask])
    chi_ext = chi_max - chi_min
    chi_num = int(np.round(chi_ext / chi_step))


    chi_cens = np.linspace(chi_min, chi_max, chi_num)


    masked_maps = []
    for chi_cen in chi_cens:

        chi_low = chi_cen - (chi_window / 2)
        chi_high = chi_cen + (chi_window / 2)
        chi_mask = (xdm.chi_arr >= chi_low) & (xdm.chi_arr <= chi_high)

        full_mask = tth_mask & chi_mask

        masked_maps.append(np.sum(xdm.images[:, :, full_mask], axis=-1))

    # return masked_maps

    if 'vmin' in kwargs:
        vmin = kwargs.pop('vmin')
    else:
        vmin = np.min(masked_maps)
    if 'vmax' in kwargs:
        vmax = kwargs.pop('vmax')
    else:
        vmax = np.max(masked_maps)

    fig, ax, slider = base_slider_plot(masked_maps,
                     slider_vals=chi_cens,
                     slider_label='Azimuthal Angle [deg]',
                     vmin=vmin,
                     vmax=vmax,
                     title=xdm._title_with_scan_id('Azimuthal Maps'),
                     **kwargs)

    fig.show()

    return slider