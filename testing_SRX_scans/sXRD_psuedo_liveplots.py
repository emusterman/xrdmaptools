import numpy as np
import time as ttime
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt

# Local imports
import xrdmaptools as xmt
from xrdmaptools.XRDMap import XRDMap
from xrdmaptools.utilities.image_corrections import rescale_array
from xrdmaptools.reflections.spot_blob_search import spot_search, spot_stats, make_stat_df


def _instantiate_xrdmap(scan_md, dark_field, calibration):

    # From scan_md, estimate and create blank map size
    xrdmap = XRDMap()

    arr_max = 16000 # max of uint with processing...

    return xrdmap, arr_max


def _find_finished_rows(scanid):
    raise NotImplementedError()


def _load_image_row(scanid):
    raise NotImplementedError()
    return image_row # as a 3D array


# OPTIMIZE ME: use dask.delayed...
def _inline_process_row(image_row, xrdmap, arr_max,
                        phases,
                        mask=None,
                        threshold_method='minimum',
                        multiplier=5,
                        size=3,
                        min_distance=3,
                        expansion=10,
                        radius=10,
                        ):

    image_row -= xrdmap.map.dark_field
    rescale_array(image_row, arr_min=0, arr_max=arr_max)

    out_list = []
    for image in tqdm(image_row):
        out = spot_search(image,
                          mask=None,
                          threshold_method='gaussian',
                          multiplier=5,
                          size=3,
                          min_distance=3,
                          expansion=10,
                          plotme=False)
        
        stat_list = []
        for spot in out:
        
            stat = spot_stats(spot,
                            image,
                            xrdmap.tth_arr,
                            xrdmap.chi_arr,
                            radius=5)
            stat_list.append(stat)

        stat_df = make_stat_df(stat_list)

    # Find phases in the row
        
    # Find orientation in the row (for a given phase)
            
        


def _replot():
    raise NotImplementedError()



def _inititial_live_plot(scanid, num_plots):
    raise NotImplementedError()

    xrdmap, arr_max = _instantiate_xrdmap(scan_md, dark_field, calibration)


    # Generate blank plots
    fig_ax_list = []
    for i in range(num_plots):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        fig_ax_list.append([fig, ax])

        im = ax.imshow()
        fig.colorbar(im, ax=ax)
        fig.show()

    return xrdmap, fig_ax_list



def live_phase_plots(scanid, phases, method):
    raise NotImplementedError()

    xrdmap, fig_ax_list = _inititial_live_plots(scanid, len(phases))

    SCAN_ACTIVE = True
    processed_rows = []
    num_rows = value
    while SCAN_ACTIVE:
        finished_rows = _find_finished_rows()
        _load_image_row(finished_rows[-1])
        processed_rows.append(value)

        if len(processed_rows) >= num_rows:
            SCAN_ACTIVE = False
        
        


def live_orientation_plots(scanid, phase):
    raise NotImplementedError()


def live_roi_plots(scanid, roi):
    raise NotImplementedError()

