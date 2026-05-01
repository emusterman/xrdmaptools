import numpy as np
import os
from skimage import io
from tqdm import tqdm
from scipy.spatial.transform import Rotation


import xrdmaptools as xmt






def track_q_vector_com(xdms, q_target, cutoff=0.25):

    map_shape = xdms.xdms_vector_map.shape
    fill_map = np.empty((*map_shape, 3), dtype=float)
    fill_map[:] = np.nan

    for index in tqdm(range(np.prod(map_shape))):
        indices = np.unravel_index(index, map_shape)
        
        q_vectors = xdms.xdms_vector_map[indices][:, :3]
        intensity = xdms.xdms_vector_map[indices][:, -1]

        if len(q_vectors) == 0:
            continue 

        maskx = np.abs(q_vectors[:, 0] - q_target[0]) < cutoff
        masky = np.abs(q_vectors[:, 1] - q_target[1]) < cutoff
        maskz = np.abs(q_vectors[:, 2] - q_target[2]) < cutoff
        mask = maskx & masky & maskz

        com = xmt.utilities.math.arbitrary_center_of_mass(
            intensity[mask],
            *q_vectors[mask].T
        )

        fill_map[indices] = com
    
    return fill_map


def track_q_vector_sum(xdms, q_target, cutoff=0.25):

    map_shape = xdms.xdms_vector_map.shape
    fill_map = np.empty(map_shape, dtype=float)
    fill_map[:] = np.nan

    for index in tqdm(range(np.prod(map_shape))):
        indices = np.unravel_index(index, map_shape)
        
        q_vectors = xdms.xdms_vector_map[indices][:, :3]
        intensity = xdms.xdms_vector_map[indices][:, -1]

        if len(q_vectors) == 0:
            continue 

        maskx = np.abs(q_vectors[:, 0] - q_target[0]) < cutoff
        masky = np.abs(q_vectors[:, 1] - q_target[1]) < cutoff
        maskz = np.abs(q_vectors[:, 2] - q_target[2]) < cutoff
        mask = maskx & masky & maskz

        fill_map[indices] = np.sum(intensity[mask])
    
    return fill_map


def track_q_vector_max(xdms, q_target, cutoff=0.25):

    map_shape = xdms.xdms_vector_map.shape
    fill_map = np.empty((*map_shape, 3), dtype=float)
    fill_map[:] = np.nan

    for index in tqdm(range(np.prod(map_shape))):
        indices = np.unravel_index(index, map_shape)
        
        q_vectors = xdms.xdms_vector_map[indices][:, :3]
        intensity = xdms.xdms_vector_map[indices][:, -1]

        if len(q_vectors) == 0:
            continue 

        maskx = np.abs(q_vectors[:, 0] - q_target[0]) < cutoff
        masky = np.abs(q_vectors[:, 1] - q_target[1]) < cutoff
        maskz = np.abs(q_vectors[:, 2] - q_target[2]) < cutoff
        mask = maskx & masky & maskz

        if mask.sum() == 0:
            continue

        max_q = q_vectors[mask][np.argmax(intensity[mask])]

        fill_map[indices] = max_q
    
    return fill_map



def get_max_vectors(xdms, q_target, cutoff=0.25):

    map_shape = xdms.xdms_vector_map.shape

    max_q_vectors = []
    max_intensity = []

    for index in tqdm(range(np.prod(map_shape))):
        indices = np.unravel_index(index, map_shape)
        
        q_vectors = xdms.xdms_vector_map[indices][:, :3]
        intensity = xdms.xdms_vector_map[indices][:, -1]

        if len(q_vectors) == 0:
            continue 

        maskx = np.abs(q_vectors[:, 0] - q_target[0]) < cutoff
        masky = np.abs(q_vectors[:, 1] - q_target[1]) < cutoff
        maskz = np.abs(q_vectors[:, 2] - q_target[2]) < cutoff
        mask = maskx & masky & maskz

        if mask.sum() == 0:
            continue

        for i in range(mask.sum()):
            if tuple(q_vectors[mask][i]) not in max_q_vectors:
                max_q_vectors.append(tuple(q_vectors[mask][i]))
                max_intensity.append(intensity[mask][i])
            else:
                idx = max_q_vectors.index(tuple(q_vectors[mask][i]))
                if max_intensity[idx] < intensity[mask][i]:
                    max_intensity[idx] = intensity[mask][i]
    
    return max_q_vectors, max_intensity


def get_max_vectors_fast(xdms, q_target, cutoff=0.25):
    # 1. Flatten the map cells and stack them into a single continuous array
    all_cells = xdms.xdms_vector_map.ravel()
    valid_cells = [cell for cell in all_cells if cell is not None and len(cell) > 0]
    
    if not valid_cells:
        return [], []
        
    all_data = np.vstack(valid_cells)
    
    q_vectors = all_data[:, :3]
    intensity = all_data[:, -1]

    # 2. Vectorized spatial filtering
    diff = np.abs(q_vectors - q_target)
    mask = np.all(diff < cutoff, axis=1)
    
    filtered_vectors = q_vectors[mask]
    filtered_intensities = intensity[mask]
    
    if len(filtered_vectors) == 0:
        return [], []

    # 3. Sort by intensity in ascending order
    # Dicts overwrite existing keys; the last write will naturally be the maximum!
    sort_idx = np.argsort(filtered_intensities)
    sorted_vectors = filtered_vectors[sort_idx]
    sorted_intensities = filtered_intensities[sort_idx]

    # 4. Use a dict comprehension to reduce to unique vectors at lightning speed
    max_map = {tuple(vec): inten for vec, inten in zip(sorted_vectors, sorted_intensities)}
    
    return list(max_map.keys()), list(max_map.values())



def get_matching_indices(q0=5.26, q1=5.2, phase0=None, phase1=None, cutoff=0.1):


    q0_mag = np.abs(np.linalg.norm(phase0.all_qs, axis=-1) - q0)
    q1_mag = np.abs(np.linalg.norm(phase1.all_qs, axis=-1) - q1)

    q0_idx = [idx for mag, idx in sorted(zip(q0_mag, range(len(q0_mag))), key=lambda pair: pair[0], reverse=True) if mag < cutoff]
    q1_idx = [idx for mag, idx in sorted(zip(q1_mag, range(len(q1_mag))), key=lambda pair: pair[0], reverse=True) if mag < cutoff]

    for idx0 in q0_idx:
        if phase0.all_hkls[idx0][1] != 0:
            continue

        l = phase0.all_hkls[idx0][-1]
        l_mask = phase1.all_hkls[:, -1] == l

        for idx1 in q1_idx:
            if not l_mask[idx1]:
                continue
            
            print(f'q0 has HKL {tuple(phase0.all_hkls[idx0])}')
            print(f'q1 has HKL {tuple(phase1.all_hkls[idx1])}')
            print('')