import numpy as np
import os
from tqdm import tqdm
from skimage import io
import h5py
import time as ttime

# Local imports
from xrdmaptools.XRDRockingCurve import XRDRockingCurve
from xrdmaptools.reflections.spot_blob_indexing_3D import *
from xrdmaptools.reflections.spot_blob_indexing_3D import _get_connection_indices
from xrdmaptools.crystal.strain import *

from xrdmaptools.utilities.utilities import (
    timed_iter
)

from tiled.client import from_profile

c = from_profile('srx')




def rsm_batch1():

    base_wd = '/nsls2/data/srx/proposals/2024-2/pass-314118/'
    scanlist = [
        # '156179-156201',
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
        # '156445-156467', # new error
        # '156469-156491', # error
        # '156493-156515',
        # '156517-156539',
        # '156541-156563',
        # '156565-156587', # error, # new error
        # '156589-156611', # error, # new error
        # '156613-156635', # new error
        # '156637-156659', # error
        # '156661-156683', # error
        # '156685-156707', # new error
        # '156709-156731', # completely different error
        # '156733-156755',
        # '156757-156775', # error, # new error
        ]


    for scan in timed_iter(scanlist):
        print(f'Batch processing scan {scan}...')
        rsm = XRDRockingCurve.from_hdf(f'scan{scan}_rsm.h5', wd=f'{base_wd}energy_rc/')

        rsm.find_3D_spots(intensity_cutoff=0.075, nn_dist=0.1, significance=1, subsample=1)
        rsm.index_all_spots(0.1, 2.5)

        grain_id = rsm.spots_3D['grain_id'].values[np.argmax(rsm.spots_3D['qof'])]

        q_vectors = rsm.spots_3D[['qx', 'qy', 'qz']][rsm.spots_3D['grain_id'] == grain_id].values
        hkls = rsm.spots_3D[['h', 'k', 'l']][rsm.spots_3D['grain_id'] == grain_id].values.astype(int)
        
        e, u, _ = phase_get_strain_orientation(q_vectors, hkls, rsm.phases['stibnite'])

        np.savetxt(f'{rsm.wd}scan{rsm.scan_id}_coarse_eij2.txt', e)
        