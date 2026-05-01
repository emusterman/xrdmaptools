import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm

from matplotlib.widgets import EllipseSelector, RectangleSelector

from xrdmaptools.plot.interactive import _check_missing_key, _figsize, _dpi, _update_map, _set_globals, _update_axes

def pixel_spots(spots, indices):
    ps = spots[((spots['map_x'] == indices[1]) & (spots['map_y'] == indices[0]))]
    return ps





from xrdmaptools.utilities.math import vector_angle
def build_tracking_map(spots_3D, q_target=None, hkl_target=None):


    # Effective map shape
    map_shape = (np.max(spots_3D['map_y']) + 1,
                 np.max(spots_3D['map_x']) + 1)

    mag_map = np.empty(map_shape)
    mag_map[:] = np.nan
    phi_map = mag_map.copy()
    chi_map = mag_map.copy()
    int_map = mag_map.copy()

    for index in tqdm(range(np.prod(map_shape))):
        indices = np.unravel_index(index, map_shape)
        pixel_df = pixel_spots(spots_3D, indices)

        spot_ints = pixel_df['intensity'].values
        sorted_spot_ints = sorted(list(spot_ints), reverse=True)

        for val in sorted_spot_ints:
            ind = np.nonzero(spot_ints == val)[0][0]
            chi = pixel_df.iloc[ind]['chi']
            q_mag = pixel_df.iloc[ind]['q_mag']
            q_vec = pixel_df.iloc[ind][['qx', 'qy', 'qz']].values
            # hkls = pixel_df.iloc[ind][['h', 'k', 'l']].values
            # if np.abs(q_mag - 5.15628987) < 0.025:
            # if np.abs(q_mag - 4.1422022472) < 0.025:
            # if np.abs(q_mag - 4.22354026) < 0.025:
            # if np.abs(q_mag - 5.18799) < 0.025:


            if np.abs(q_mag - q_target) < 0.025:
            # if np.all(hkls == [3, -2, -4]):
                mag_map[indices] = q_mag
                phi_map[indices] = vector_angle([0, 0, -1], q_vec, degrees=True)
                chi_map[indices] = vector_angle([-1, 0], q_vec[:-1], degrees=True)
                # chi_map[indices] = chi
                int_map[indices] = val
                break
    
    return mag_map, phi_map, chi_map, int_map


def build_simple_ori_map(spots_3D):


    # Effective map shape
    map_shape = (np.max(spots_3D['map_y']) + 1,
                 np.max(spots_3D['map_x']) + 1)

    ori_map = np.empty((*map_shape, 3, 3))
    ori_map[:] = np.nan

    for index in tqdm(range(np.prod(map_shape))):
        indices = np.unravel_index(index, map_shape)
        pixel_df = pixel_spots(spots_3D, indices)

        spot_ints = pixel_df['intensity'].values
        sorted_spot_ints = sorted(list(spot_ints), reverse=True)

        spot1 = None
        spot2 = None

        for val in sorted_spot_ints:
            ind = np.nonzero(spot_ints == val)[0][0]
            q_mag = pixel_df.iloc[ind]['q_mag']
            q_vec = pixel_df.iloc[ind][['qx', 'qy', 'qz']].values
           
            if np.abs(q_mag - 5.15628987) < 0.025:
                spot1 = q_vec
                break

        for val in sorted_spot_ints:
            ind = np.nonzero(spot_ints == val)[0][0]
            q_mag = pixel_df.iloc[ind]['q_mag']
            q_vec = pixel_df.iloc[ind][['qx', 'qy', 'qz']].values

            # Unlikely check
            if np.all(q_vec == spot1):
                continue
           
            if np.abs(q_mag - 4.1422022472) < 0.025:
                spot2 = q_vec
                break
        
        if spot1 is not None and spot2 is not None:
            ori = Rotation.align_vectors(phase.Q([[-4, 1, -2], [-3, 1, -4]]),
                                            [spot1, spot2])[0]
            ori_map[indices] = ori.as_matrix()
    
    return ori_map