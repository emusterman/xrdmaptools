import numpy as np
import os
from skimage import io
from tqdm import tqdm
from scipy.spatial.transform import Rotation


import xrdmaptools as xmt






def track_q_vector_com(vector_map, q_target, cutoff=0.25):

    map_shape = vector_map.shape
    fill_map = np.empty((*map_shape, 3), dtype=float)
    fill_map[:] = np.nan

    for index in tqdm(range(np.prod(map_shape))):
        indices = np.unravel_index(index, map_shape)
        
        q_vectors = vector_map[indices][:, :3]
        intensity = vector_map[indices][:, -1]

        if len(q_vectors) == 0:
            continue 

        # maskx = np.abs(q_vectors[:, 0] - q_target[0]) < cutoff
        # masky = np.abs(q_vectors[:, 1] - q_target[1]) < cutoff
        # maskz = np.abs(q_vectors[:, 2] - q_target[2]) < cutoff
        # mask = maskx & masky & maskz

        mask = np.linalg.norm(q_vectors - q_target, axis=1) < cutoff

        com = xmt.utilities.math.arbitrary_center_of_mass(
            intensity[mask],
            *q_vectors[mask].T
        )

        fill_map[indices] = com
    
    return fill_map


def track_q_vector_sum(vector_map, q_target, cutoff=0.25):

    map_shape = vector_map.shape
    fill_map = np.empty(map_shape, dtype=float)
    fill_map[:] = np.nan

    for index in tqdm(range(np.prod(map_shape))):
        indices = np.unravel_index(index, map_shape)
        
        q_vectors = vector_map[indices][:, :3]
        intensity = vector_map[indices][:, -1]

        if len(q_vectors) == 0:
            continue

        maskx = np.abs(q_vectors[:, 0] - q_target[0]) < cutoff
        masky = np.abs(q_vectors[:, 1] - q_target[1]) < cutoff
        maskz = np.abs(q_vectors[:, 2] - q_target[2]) < cutoff
        mask = maskx & masky & maskz

        # mask = np.linalg.norm(q_vectors - q_target, axis=1) < cutoff

        if mask.sum() == 0:
            continue

        fill_map[indices] = np.sum(intensity[mask])
    
    return fill_map


def track_q_vector_max(vector_map, q_target, cutoff=0.25):

    map_shape = vector_map.shape
    fill_map = np.empty((*map_shape, 3), dtype=float)
    fill_map[:] = np.nan

    for index in tqdm(range(np.prod(map_shape))):
        indices = np.unravel_index(index, map_shape)
        
        q_vectors = vector_map[indices][:, :3]
        intensity = vector_map[indices][:, -1]

        if len(q_vectors) == 0:
            continue

        maskx = np.abs(q_vectors[:, 0] - q_target[0]) < cutoff
        masky = np.abs(q_vectors[:, 1] - q_target[1]) < cutoff
        maskz = np.abs(q_vectors[:, 2] - q_target[2]) < cutoff
        mask = maskx & masky & maskz

        # mask = np.linalg.norm(q_vectors - q_target, axis=1) < cutoff

        if mask.sum() == 0:
            continue

        max_q = q_vectors[mask][np.argmax(intensity[mask])]

        fill_map[indices] = max_q
    
    return fill_map



# def get_max_vectors(xdms, q_target, cutoff=0.25):

#     map_shape = xdms.xdms_vector_map.shape

#     max_q_vectors = []
#     max_intensity = []

#     for index in tqdm(range(np.prod(map_shape))):
#         indices = np.unravel_index(index, map_shape)
        
#         q_vectors = xdms.xdms_vector_map[indices][:, :3]
#         intensity = xdms.xdms_vector_map[indices][:, -1]

#         if len(q_vectors) == 0:
#             continue 

#         maskx = np.abs(q_vectors[:, 0] - q_target[0]) < cutoff
#         masky = np.abs(q_vectors[:, 1] - q_target[1]) < cutoff
#         maskz = np.abs(q_vectors[:, 2] - q_target[2]) < cutoff
#         mask = maskx & masky & maskz

#         if mask.sum() == 0:
#             continue

#         for i in range(mask.sum()):
#             if tuple(q_vectors[mask][i]) not in max_q_vectors:
#                 max_q_vectors.append(tuple(q_vectors[mask][i]))
#                 max_intensity.append(intensity[mask][i])
#             else:
#                 idx = max_q_vectors.index(tuple(q_vectors[mask][i]))
#                 if max_intensity[idx] < intensity[mask][i]:
#                     max_intensity[idx] = intensity[mask][i]
    
#     return max_q_vectors, max_intensity


def get_max_vectors_fast(vector_map, q_target, cutoff=0.25):
    all_cells = vector_map.ravel()
    valid_cells = [cell for cell in all_cells if cell is not None and len(cell) > 0]
    
    if not valid_cells:
        return [], []
        
    all_data = np.vstack(valid_cells)
    
    q_vectors = all_data[:, :3]
    intensity = all_data[:, -1]

    diff = np.abs(q_vectors - q_target)
    mask = np.all(diff < cutoff, axis=1)
    
    filtered_vectors = q_vectors[mask]
    filtered_intensities = intensity[mask]
    
    if len(filtered_vectors) == 0:
        return [], []

    sort_idx = np.argsort(filtered_intensities)
    sorted_vectors = filtered_vectors[sort_idx]
    sorted_intensities = filtered_intensities[sort_idx]

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


def mask_vector_map(vector_map, q_target, cutoff=0.25):

    map_shape = vector_map.shape
    fill_map = np.empty(map_shape, dtype=object)

    for index in tqdm(range(np.prod(map_shape))):
        indices = np.unravel_index(index, map_shape)
        
        q_vectors = vector_map[indices][:, :3]
        intensity = vector_map[indices][:, -1]

        if len(q_vectors) == 0:
            fill_map[indices] = []

        maskx = np.abs(q_vectors[:, 0] - q_target[0]) < cutoff
        masky = np.abs(q_vectors[:, 1] - q_target[1]) < cutoff
        maskz = np.abs(q_vectors[:, 2] - q_target[2]) < cutoff
        mask = maskx & masky & maskz

        # mask = np.linalg.norm(q_vectors - q_target, axis=1) < cutoff

        if mask.sum() == 0:
            fill_map[indices] = []

        fill_map[indices] = vector_map[indices][mask]
    
    return fill_map



def analyze_q_vector(vector_map,
                     q_target,
                     q_cutoff=0.25,
                     int_cutoff=0,
                     map_cutoff=0,
                     plot=True,
                     title=None):

    map_shape = vector_map.shape
    sum_map = np.empty(map_shape, dtype=float)
    sum_map[:] = np.nan    
    com_map = np.empty((*map_shape, 3), dtype=float)
    com_map[:] = np.nan

    for index in tqdm(range(np.prod(map_shape))):
        # Get vectors
        indices = np.unravel_index(index, map_shape)
        q_vectors = vector_map[indices][:, :3]
        intensity = vector_map[indices][:, -1]

        # Check for vectors
        if len(q_vectors) == 0:
            continue

        # Mask intensity first
        mask = intensity > int_cutoff
        if mask.sum() == 0:
            continue
        q_vectors = q_vectors[mask]
        intensity = intensity[mask]

        # Then mask for q-space
        mask = np.linalg.norm(q_vectors - q_target, axis=1) < q_cutoff
        # maskx = np.abs(q_vectors[:, 0] - q_target[0]) < q_cutoff
        # masky = np.abs(q_vectors[:, 1] - q_target[1]) < q_cutoff
        # maskz = np.abs(q_vectors[:, 2] - q_target[2]) < q_cutoff
        # mask = maskx & masky & maskz

        com = xmt.utilities.math.arbitrary_center_of_mass(
            intensity[mask],
            *q_vectors[mask].T
        )

        com_map[indices] = com
        sum_map[indices] = np.sum(intensity[mask])
        map_mask = sum_map > map_cutoff

    mag_map = np.linalg.norm(com_map, axis=-1)
    phi_map = np.degrees(com_map[..., -1] / mag_map)
    chi_map = np.degrees(np.arctan2(com_map[..., 1], com_map[..., 0]))

    if not plot:
        return sum_map, com_map, mag_map, phi_map, chi_map

    # Plot
    fig, ax = plt.subplots(2, 2, figsize=(5, 5), dpi=200, layout='constrained')
    fig.suptitle(title)
    
    def plot_axis(ax, img, title, mask=map_mask):
        img = img.copy()
        if mask is not None:
            img[~mask] = np.nan
        image = ax.imshow(img)
        fig.colorbar(image, ax=ax)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plot_axis(ax[0, 0], sum_map, 'Sum Intensity [a.u.]', mask=None)
    plot_axis(ax[0, 1], (2 * np.pi) / mag_map, 'd-spacing [Å]')
    plot_axis(ax[1, 0], phi_map, 'Polar Angle [deg.]')
    plot_axis(ax[1, 1], chi_map, 'Azimuthal Angle [deg.]')
    fig.show()

    # # Plot
    # fig = plt.figure(figsize=(14, 6), layout="constrained")
    # gs = fig.add_gridspec(nrows=2, ncols=6)

    # # Create the top large image axis spanning the entire first row (all 6 columns)
    # ax_top = fig.add_subplot(gs[0, :])
    # ax_top.imshow(sum_map)
    # ax_top.set_title("Sum Intensity")

    # # Create the six smaller image axes lined up side-by-side in the second row
    # for idx, img, title in enumerate([(mag)]):
    #     ax_sub = fig.add_subplot(gs[1, idx])
    #     ax_sub.imshow(img, cmap='plasma')
    #     ax_sub.set_title(f"Sub {idx + 1}")
    #     ax_sub.axis('off')

    # plt.show()





def add_plane(ax, normal_vector, multiplier, grid_size=5, grid_resolution=10):
    """
    Draws a 3D plane on a given matplotlib 3D axis.
    The plane has the specified normal vector and is offset from the origin by a 
    distance equal to: multiplier * ||normal_vector||.
    """
    # Convert normal vector to a numpy array and ensure it is float
    n = np.array(normal_vector, dtype=float)
    a, b, c = n
    
    # Calculate the norm (length) of the normal vector
    norm_length = np.linalg.norm(n)
    if norm_length == 0:
        raise ValueError("Normal vector cannot be the zero vector.")
        
    # The algebraic constant D in the plane equation: ax + by + cz = D
    # Perpendicular distance d = multiplier * ||n||
    # For a plane facing direction n, D = d * ||n|| = multiplier * ||n||^2
    D = multiplier * (norm_length ** 2)
    
    # Generate a grid for the plane plotting
    # We choose the two coordinates with the largest normal components to avoid division by zero
    max_idx = np.argmax(np.abs(n))
    
    span = np.linspace(-grid_size, grid_size, grid_resolution)
    u, v = np.meshgrid(span, span)
    
    if max_idx == 2:  # c is the largest, solve for z: z = (D - ax - by) / c
        x, y = u, v
        z = (D - a*x - b*y) / c
    elif max_idx == 1:  # b is the largest, solve for y: y = (D - ax - cz) / b
        x, z = u, v
        y = (D - a*x - c*z) / b
    else:  # a is the largest, solve for x: x = (D - by - cz) / a
        y, z = u, v
        x = (D - b*y - c*z) / a
        
    # Plot the surface representing the 3D plane
    ax.plot_surface(x, y, z, alpha=0.25, label='Plane')