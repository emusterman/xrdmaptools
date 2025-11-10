import numpy as np
import os
from skimage import io
import matplotlib.pyplot as plt
import h5py


from xrdmaptools import XRDMap
from xrdmaptools.utilities.math import tth_2_q, rescale_array


def process_xdm(scan_id,
                wd,
                dark_field=None,
                poni_file=None,
                air_scatter=False,
                swapped_axes=False,
                reprocess=False):

    proc_dict = {
        'images' : True,
        'integrations' : True,
        'blobs' : True,
        'analysis' : True
    }
    if not isinstance(scan_id, XRDMap):
        if os.path.exists(f'{wd}xrdmaps/scan{scan_id}_xrdmap.h5'):
            print('File found! Loading from HDF...')
            
            load_kwargs = {
                'wd' : f'{wd}xrdmaps/',
                'image_data_key' : 'raw',
                'swapped_axes' : swapped_axes
            }
            if not reprocess:
                temp_hdf = h5py.File(f'{wd}xrdmaps/scan{scan_id}_xrdmap.h5')
                if 'final_images' in temp_hdf['xrdmap/image_data']:
                    proc_dict['images'] = False
                    load_kwargs['image_data_key'] = 'final'
                if 'integration_data' in temp_hdf['xrdmap'] and 'final_integrations' in temp_hdf['xrdmap/integration_data']:
                    proc_dict['integrations'] = False
                    load_kwargs['integration_data_key'] = None
                if '_blob_masks' in temp_hdf['xrdmap/image_data']:
                    proc_dict['blobs'] = False
                if 'vectorized_map' in temp_hdf['xrdmap']:
                    proc_dict['vectors'] = False
                temp_hdf.close()
            
            print(proc_dict)
            if any(proc_dict.values()):
                xdm = XRDMap.from_hdf(f'scan{scan_id}_xrdmap.h5',
                                      **load_kwargs)
                # Update formatting
                xdm.open_hdf()
                if 'extra_metadata' not in xdm.hdf['xrdmap']:
                    xdm.hdf['xrdmap'].create_dataset('extra_metadata', data=h5py.Empty("f"))
                xdm.close_hdf()
            else:
                xdm = None
        else:
            print('Loading data from server...')
            xdm = XRDMap.from_db(scan_id, wd=f'{wd}xrdmaps/', swapped_axes=swapped_axes)
    else:
        # TODO: Look for processing conditions
        xdm = scan_id

        if not reprocess:
            if xdm.title == 'final':
                proc_dict['images'] = False
            if hasattr(xdm, 'integrations') and xdm.title == 'final':
                proc_dict['integrations'] = False
            if hasattr(xdm, 'blob_masks') and xdm.blob_masks is not None:
                proc_dict['blobs'] = False
    
    if proc_dict['images']:
        if xdm.title != 'raw':
            xdm.load_images_from_hdf('raw')

        xdm.correct_dark_field(dark_field)
        xdm.normalize_scaler()
        if isinstance(air_scatter, np.ndarray):
            xdm.correct_air_scatter(air_scatter, applied_corrections={'dark_field':True, 'scaler_intensity':True})
        elif air_scatter == False:
            pass
        elif air_scatter == True:
            xdm.correct_air_scatter(xdm.med_image, applied_corrections=xdm.corrections)
        else:
            err_str = 'Error handling air_scatter. Designate array, None, or bool.'
            raise RuntimeError(err_str)

        xdm.correct_outliers(tolerance=10)

        # Geometric corrections
        xdm.set_calibration(poni_file, wd=f'{wd}calibrations/')
        xdm.apply_polarization_correction()
        xdm.apply_solidangle_correction()

        # Background correction
        xdm.estimate_background(method='bruckner',
                                binning=8,
                                min_prominence=0.1)

        # Rescale and saving
        xdm.rescale_images()
        xdm.finalize_images()

    # Integrate map
    if proc_dict['integrations']:
        xdm.integrate1D_map()

    # Find blobs
    if proc_dict['blobs']:
        xdm.find_blobs(filter_method='minimum',
                    multiplier=5,
                    size=3,
                    expansion=10)

    # Convert to 1D integrations
    if xdm is not None:

        os.makedirs(f'{wd}new_processed/scan{xdm.scan_id}/', exist_ok=True)

        # Load elmental fitting and generate masks
        xdm.load_xrfmap(wd=f'{wd}xrfmaps/')
        if len(np.unique(xdm.sclr_dict['i0'])) > 10:
            # print("Using scaler 'i0'.")
            sclr = xdm.sclr_dict['i0']
        else:
            sclr = xdm.sclr_dict['im']
            # print("Using scaler 'im'.")
        Ce_mask = xdm.xrf['Ce_L'] / sclr > 0.01
        La_mask = xdm.xrf['La_L'] / sclr > 0.04
        Nd_mask = xdm.xrf['Nd_L'] / sclr > 0.01
        Y_mask = xdm.xrf['Y_K'] / sclr > 0.075
        all_mask = np.ones(Ce_mask.shape, dtype=np.bool_)

        # Plot mask...
        for title, mask, xrf_map in zip(['Ce', 'La', 'Nd', 'Y'],
                                        [Ce_mask, La_mask, Nd_mask, Y_mask],
                                        [xdm.xrf['Ce_L'], xdm.xrf['La_L'], xdm.xrf['Nd_L'], xdm.xrf['Y_K']]):
            if mask.sum() == 0:
                continue
            
            xrf_map = xrf_map.copy() / sclr
            xrf_map[~mask] = np.nan                          
            fig, ax = xdm.plot_map(xrf_map, title=f'{title} Masked Region', return_plot=True)
            fig.savefig(f'{wd}new_processed/scan{xdm.scan_id}/scan{xdm.scan_id}_{title}_mask_map.png')
            plt.close('all')

        # Plot full max integration
        for title, mask in zip(['all', 'Ce', 'La', 'Nd', 'Y'],
                               [all_mask, Ce_mask, La_mask, Nd_mask, Y_mask]):
            if mask.sum() == 0:
                continue
            out_path = f'{wd}new_processed/scan{xdm.scan_id}/scan{xdm.scan_id}_{title}_full_max_1D_integration'

            # Plot masked integration
            tth, intensity = xdm.integrate1D_image(np.max(xdm.images[mask], axis=0))
            q = tth_2_q(tth, wavelength=xdm.wavelength)

            np.savetxt(f'{out_path}.txt',
                       np.asarray([q, tth, intensity]))
            fig, ax = xdm.plot_integration(intensity, tth=tth, title=f'Full Max {title} Integration', return_plot=True)
            fig.savefig(f'{out_path}.png')
            plt.close('all')

        # Plot blob max integration
        xdm.images[~xdm.blob_masks] = 0
        for title, mask in zip(['all', 'Ce', 'La', 'Nd', 'Y'],
                               [all_mask, Ce_mask, La_mask, Nd_mask, Y_mask]):
            if mask.sum() == 0:
                continue
            out_path = f'{wd}new_processed/scan{xdm.scan_id}/scan{xdm.scan_id}_{title}_blob_max_1D_integration'

            tth, intensity = xdm.integrate1D_image(np.max(xdm.images[mask], axis=0))
            q = tth_2_q(tth, wavelength=xdm.wavelength)

            np.savetxt(f'{out_path}.txt',
                       np.asarray([q, tth, intensity]))
            fig, ax = xdm.plot_integration(intensity, tth=tth, title=f'Blob Max {title} Integration', return_plot=True)
            fig.savefig(f'{out_path}.png')
            plt.close('all')

    return xdm 