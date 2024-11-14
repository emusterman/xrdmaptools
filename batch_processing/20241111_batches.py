import numpy as np
import os
from tqdm import tqdm
from skimage import io
import h5py
import time as ttime

# Local imports
from xrdmaptools.XRDRockingCurveStack import XRDRockingCurveStack
from xrdmaptools.reflections.spot_blob_indexing_3D import *
from xrdmaptools.reflections.spot_blob_indexing_3D import _get_connection_indices
from xrdmaptools.crystal.strain import *

from xrdmaptools.utilities.utilities import (
    timed_iter
)

from tiled.client import from_profile

c = from_profile('srx')


def check_for_new_maps(wd):

    start_id = 162163
    MAPS_LEFT = True

    while MAPS_LEFT:
        try:
            current_id = c[-1].start['scan_id']
            for scan_id in np.arange(start_id, current_id, 1):
                if (c[scan_id].start['scan']['type'] == 'XRF_FLY'
                    and not os.path.exists(f'{wd}xrdmaps/scan{scan_id}_xrdmap.h5')):
                    _ = process_map(scan_id, transpose=True)

                    if scan_id == 162230:
                        _ = process_map(scan_id + 1, transpose=True) # Won't reach last one otherwise
                        MAPS_LEFTS = False
                        print('FINISHED!!!')
                        return
            
            print('Iterated through all possible new maps. Sleeping for 5 min...')
            ttime.sleep(300)
        
        except KeyboardInterrupt:
            print('Interrupting!')
            return


def rsm_batch1():

    base_wd = '/nsls2/data/srx/proposals/2024-2/pass-314118/'
    scanlist = [
        '156179-156201'
        # '156205-156227',
        # '156229-156251',
        # '156253-156275',
        # '156277-156299',
        # '156301-156323',
        # '156325-156347',
        # '156349-156371',
        # '156373-156395',
        # '156397-156419',
        # '156421-156443',
        # '156445-156467',
        # '156469-156491',
        # '156493-156515',
        # '156517-156539',
        # '156541-156563',
        # '156565-156587',
        # '156589-156611',
        # '156613-156635',
        # '156637-156659',
        # '156661-156683',
        # '156685-156707',
        # '156709-156731',
        # '156733-156755',
        # '156757-156775',
        ]
    
    dark_field = io.imread(f'{base_wd}dark_fields/scan156203_dexela_median_composite.tif')


    for scan in timed_iter(scanlist):
        print(f'Batch processing scan {scan}...')

        if not os.path.exists(f'{base_wd}energy_rc/scan{scan}_rsm.h5'):
            rsm = XRDRockingCurveStack.from_image_stack(f'scan{scan}_dexela_energy_rc.tif',
                                                        wd=f'{base_wd}energy_rc/',
                                                        scan_id=scan)
            rsm.load_parameters_from_txt()
            rsm.load_metadata_from_txt()

            rsm.correct_dark_field(dark_field=dark_field)
            rsm.normalize_scaler()
            rsm.correct_outliers()
            rsm.set_calibration('scan156160_dexela_calibration.poni', filedir=f'{base_wd}calibrations/')
            rsm.apply_polarization_correction()
            rsm.apply_solidangle_correction()
            rsm.estimate_background(method='bruckner', binning=8, min_prominence=0.1)
            rsm.remove_background()
            rsm.rescale_images(arr_max=rsm.estimate_saturated_pixel())
            rsm.finalize_images()
        else:
            print('HDF already exists. Loading that!')
            rsm = XRDRockingCurveStack.from_hdf(f'scan{scan}_rsm.h5', wd=f'{base_wd}energy_rc/')

        rsm.load_phase('Stibnite_0008636.cif', filedir='/nsls2/users/emusterma/Documents/cif/', phase_name='stibnite')
        rsm.update_phases()

        rsm.find_2D_blobs(threshold_method='minimum', multiplier=5, size=3, expansion=10, override_rescale=True)
        rsm.vectorize_images()
        rsm.find_3D_spots(intensity_cutoff=0.05, nn_dist=0.1, significance=1, subsample=1, save_to_hdf=True)

        rsm.phases['stibnite'].generate_reciprocal_lattice(qmax=8)
        unstrained = LatticeParameters.from_Phase(rsm.phases['stibnite'])

        all_spot_qs = np.asarray(rsm.spots_3D[['qx', 'qy', 'qz']])
        all_ref_qs = np.asarray(rsm.phases['stibnite'].all_qs)
        all_ref_hkls = np.asarray(rsm.phases['stibnite'].all_hkls)
        all_ref_fs = np.asarray(rsm.phases['stibnite'].all_fs)
        
        min_q = np.min(np.linalg.norm(rsm.phases['stibnite'].Q([[1, 0, 0],
                                                                [0, 1, 0],
                                                                [0, 0, 1]]), axis=0))
        near_q = 0.1
        near_angle = 2 # degrees
        qmask = QMask.from_XRDRockingCurveStack(rsm)

        connection_pair_list = find_all_valid_pairs(all_spot_qs,
                                                    all_ref_qs,
                                                    near_q=near_q,
                                                    near_angle=near_angle,
                                                    degrees=True,
                                                    min_q=min_q)
        
        new_pair_list = reduce_symmetric_equivalents(connection_pair_list,
                                                     all_spot_qs,
                                                     all_ref_qs,
                                                     all_ref_hkls,
                                                     near_angle,
                                                     min_q)

        best_connections, best_qofs = decaying_pattern_decomposition(new_pair_list,
                                                                     all_spot_qs,
                                                                     all_ref_qs,
                                                                     all_ref_fs,
                                                                     qmask,
                                                                     near_q,
                                                                     qof_minimum=0.5)

        spot_inds, ref_inds = _get_connection_indices(best_connections[0])
        eij, orientation, strained = get_strain_orientation(all_spot_qs[spot_inds], all_ref_hkls[ref_inds], unstrained)
        np.savetxt(f'{rsm.wd}scan{rsm.scan_id}_coarse_eij.txt', eij)
        




def get_rsm_baseline_positions1():

    scanlist = [
        '156205-156227',
        '156229-156251',
        '156253-156275',
        '156277-156299',
        '156301-156323',
        '156325-156347',
        '156349-156371',
        '156373-156395',
        '156397-156419',
        '156421-156443',
        '156445-156467',
        '156469-156491',
        '156493-156515',
        '156517-156539',
        '156541-156563',
        '156565-156587',
        '156589-156611',
        '156613-156635',
        '156637-156659',
        '156661-156683',
        '156685-156707',
        '156709-156731',
        '156733-156755',
        '156757-156775',
        ]


    x_vals, y_vals, z_vals = [], [], []
    for scanrange in scanlist:
        print(f'Finding data for scans {scanrange}.')
        scan_start = int(scanrange[:6])
        scan_end = int(scanrange[-6:])

        x_pos, y_pos, z_pos = [], [], []
        for scan in range(scan_start, scan_end + 1):
            bs_run = c[scan]

            # pos_keys = ['x', 'y', 'z']
            stage_keys = ['nano_stage_sx', 'nano_stage_sy', 'nano_stage_z']
            pos_values = [bs_run['baseline']['data'][key][0] for key in stage_keys]

            x_pos.append(pos_values[0])
            y_pos.append(pos_values[1])
            z_pos.append(pos_values[2])
        
        x_vals.append(np.mean(x_pos))
        y_vals.append(np.mean(y_pos))
        z_vals.append(np.mean(z_pos))
    
    return scanlist, x_vals, y_vals, z_vals

    with open(f'{base_wd}energy_rc/scan_positions.txt', "w") as f:
        for lines in [scanlist, x_vals, y_vals, z_vals]:
            for line in lines:
                f.write(f'{line}\t')
            f.write('\n')



def get_rsm_baseline_positions2():

    scanlist = [
        '156205-156227',
        '156229-156251',
        '156253-156275',
        '156277-156299',
        '156301-156323',
        '156325-156347',
        '156349-156371',
        '156373-156395',
        '156397-156419',
        '156421-156443',
        '156445-156467',
        '156469-156491',
        '156493-156515',
        '156517-156539',
        '156541-156563',
        '156565-156587',
        '156589-156611',
        '156613-156635',
        '156637-156659',
        '156661-156683',
        '156685-156707',
        '156709-156731',
        '156733-156755',
        '156757-156775',
        ]


    fine_x, coarse_x, fine_y, coarse_y, coarse_z = [], [], [], [], []
    for scanrange in scanlist:
        print(f'Finding data for scans {scanrange}.')
        scan_start = int(scanrange[:6])
        scan_end = int(scanrange[-6:])

        sx, x, sy, y, z = [], [], [], [], []
        for scan in range(scan_start, scan_end + 1):
            bs_run = c[scan]

            # pos_keys = ['x', 'y', 'z']
            stage_keys = ['nano_stage_sx',
                          'nano_stage_topx',
                          'nano_stage_sy',
                          'nano_stage_y',
                          'nano_stage_z']
            pos_values = [bs_run['baseline']['data'][key][0] for key in stage_keys]

            sx.append(pos_values[0])
            x.append(pos_values[1])
            sy.append(pos_values[2])
            y.append(pos_values[3])
            z.append(pos_values[4])

        fine_x.append(np.mean(sx))
        coarse_x.append(np.mean(x))
        fine_y.append(np.mean(sy))
        coarse_y.append(np.mean(y))
        coarse_z.append(np.mean(z))

    with open(f'{base_wd}energy_rc/scan_positions.txt', "w") as f:
        for lines in [scanlist, fine_x, coarse_x, fine_y, coarse_y, coarse_z]:
            for line in lines:
                f.write(f'{line}\t')
            f.write('\n')