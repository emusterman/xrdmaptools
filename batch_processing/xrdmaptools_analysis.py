import numpy as np
import os
from tqdm import tqdm
from skimage import io

# Local imports
from xrdmaptools.XRDMap import XRDMap
from xrdmaptools.io.db_io import (
    save_full_scan,
    save_xrd_tifs,
    save_map_parameters,
    save_scan_md,
    save_composite_pattern,
    save_calibration_pattern,
    save_energy_rc_data,
    save_angle_rc_data
)
from xrdmaptools.io.db_utils import make_xrdmap_hdf
from xrdmaptools.reflections.spot_blob_search import spot_search
from xrdmaptools.crystal.Phase import phase_selector
from xrdmaptools.utilities.math import (
    energy_2_wavelength,
    wavelength_2_energy,
    tth_2_d,
    d_2_tth,
    convert_qd,
    q_2_tth,
    tth_2_q
)
from xrdmaptools.utilities.utilities import (
    label_nearest_spots,
    arbitrary_center_of_mass,
    vector_angle,
    timed_iter
)