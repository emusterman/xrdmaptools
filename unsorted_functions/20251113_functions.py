import numpy as np
import os
from skimage import io
from scipy.ndimage import (
    minimum_filter,
    gaussian_filter,
    median_filter
)

from xrdmaptools.utilities.utilities import (pathify, iterative_background)
from xrdmaptools.utilities.math import tth_2_q
from xrdmaptools.reflections.spot_blob_search import resize_blobs
from xrdmaptools import XRDMap

def standard_process_xdm(scan_id,
                         wd):

    if not isinstance(scan_id, XRDMap):
        if isinstance(scan_id, str):
            fname = scan_id
        else:
            fname = f'scan{int(scan_id)}_xrdmap.h5'

        
    xdm = XRDMap.from_hdf(fname, wd=f'{wd}xrdmaps/')
    outlier_map = iterative_background(xdm.sum_map)
    xdm.null_map = np.any([xdm.null_map, outlier_map], axis=0)
    xdm.finalize_images()

    # Integrate map
    xdm.integrate1D_map()

    # Find blobs
    xdm.find_blobs(filter_method='minimum',
                    multiplier=5,
                    size=3,
                    expansion=10)

    # Convert to 1D integrations
    tth, intensity = xdm.integrate1D_image(xdm.max_image)
    q = tth_2_q(tth, wavelength=xdm.wavelength)

    np.savetxt(f'{wd}/max_1D_integrations/scan{xdm.scan_id}_max_1D_integration.txt',
            np.asarray([q, tth, intensity]))
    fig, ax = xdm.plot_integration(intensity, tth=tth, title='Max Integration', return_plot=True)
    fig.savefig(f'{wd}max_1D_integrations/scan{xdm.scan_id}_max_integration.png')
    plt.close('all')

    io.imsave(f'{wd}integrated_maps/scan{xdm.scan_id}_max_map.tif', xdm.max_map)

    images = xdm.images.copy()
    xdm.dump_images()
    images[~xdm.blob_masks] = 0
    blob_sum_map = np.sum(images, axis=(2, 3))

    io.imsave(f'{wd}integrated_maps/scan{xdm.scan_id}_blob_sum.tif', blob_sum_map)

    del xdm




def eiger_process_xdm(scan_id,
                         wd, 
                         poni_file=None):

    if not isinstance(scan_id, XRDMap):
        if isinstance(scan_id, str):
            fname = scan_id
        else:
            fname = f'scan{int(scan_id)}_xrdmap.h5'

    xdm = XRDMap.from_hdf(fname, wd=f'{wd}xrdmaps/', image_data_key='raw')

    xdm.construct_null_map(override=True)
    xdm.apply_defect_mask()
    xdm.correct_scaler_energies(scaler_key='i0')
    xdm.convert_scalers_to_flux(scaler_key='i0')
    xdm.correct_scaler_energies(scaler_key='im')
    xdm.convert_scalers_to_flux(scaler_key='im')
    xdm.normalize_scaler()
    xdm.correct_outliers(tolerance=10)

    # Geometric corrections
    xdm.set_calibration(poni_file, wd=f'{wd}calibrations/')
    xdm.apply_polarization_correction()
    xdm.apply_solidangle_correction()

    # Background correction
    xdm.estimate_background(method='bruckner',
                            binning=4,
                            min_prominence=0.1)

    # Rescale and saving
    xdm.rescale_images(arr_max=xdm.estimate_saturated_pixel())
    xdm.finalize_images()

    # Integrate map
    xdm.integrate1D_map()

    # Find blobs
    xdm.find_blobs(filter_method='minimum',
                    multiplier=5,
                    size=3,
                    expansion=10)

    # Convert to 1D integrations
    tth, intensity = xdm.integrate1D_image(xdm.max_image)
    q = tth_2_q(tth, wavelength=xdm.wavelength)

    np.savetxt(f'{wd}/max_1D_integrations/scan{xdm.scan_id}_max_1D_integration.txt',
            np.asarray([q, tth, intensity]))
    fig, ax = xdm.plot_integration(intensity, tth=tth, title='Max Integration', return_plot=True)
    fig.savefig(f'{wd}max_1D_integrations/scan{xdm.scan_id}_max_integration.png')
    plt.close('all')

    io.imsave(f'{wd}integrated_maps/scan{xdm.scan_id}_max_map.tif', xdm.max_map)

    images = xdm.images.copy()
    xdm.dump_images()
    images[~xdm.blob_masks] = 0
    blob_sum_map = np.sum(images, axis=(2, 3))

    io.imsave(f'{wd}integrated_maps/scan{xdm.scan_id}_blob_sum.tif', blob_sum_map)

    del xdm



def blob_search(image,
                mask=None,
                filter_method='minimum',
                multiplier=5,
                size=3,
                expansion=None):
    
    if mask is None or np.all(mask):
       temp_mask = np.ones_like(image, dtype=np.bool_)
    else:
        temp_mask = mask

    # Setup filter for the images
    # Works okay, could take no inputs...
    if str(filter_method).lower() in ['gaussian', 'gauss']:
        image_filter = lambda image : gaussian_filter(image, sigma=size)
    # Works best
    elif str(filter_method).lower() in ['minimum', 'min']:
        image_filter = lambda image : minimum_filter(gaussian_filter(image, sigma=1), size=size)
    # Gaussian is better and faster
    elif str(filter_method).lower() in ['median', 'med']:
        image_filter = lambda image : median_filter(image, size=size)
    else:
        raise ValueError('Unknown threshold method requested.')
    
    if mask is not None:
        # Smooth image to reduce noise contributions
        zero_image = np.copy(image)
        zero_image[~temp_mask] = 0 # should be redundant
        zero_filter = image_filter(zero_image)
        
        div_image = np.ones_like(image)
        div_image[~temp_mask] = 0
        filter_div = image_filter(div_image)
        
        blurred_image = zero_filter / filter_div
        # Clean up some NaNs from median filters
        temp_mask[np.isnan(blurred_image)] = False
        # Clear image from masked values. Should not matter....
        blurred_image[~temp_mask] = 0
    
    else:
        zero_image = np.copy(image)
        zero_image[~temp_mask] = 0 # should be redundant
        blurred_image = image_filter(zero_image)

    # Create and trim mask
    blob_mask = iterative_map_background(blurred_image,
                                         multiplier=multiplier)
    # blob_mask = blurred_image > mask_thresh
    blob_mask[~temp_mask] = 0

    if (expansion is not None
        and expansion != 0
        and np.any(blob_mask)):
        blob_mask = resize_blobs(blob_mask, distance=expansion)
        blob_mask[~temp_mask] = 0

    # Exclude edges from analysis
    # They can be erroneous from filters
    blob_mask[0] = 0
    blob_mask[-1] = 0
    blob_mask[:, 0] = 0
    blob_mask[:, -1] = 0

    return blob_mask, blurred_image
