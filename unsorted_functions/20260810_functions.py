import numpy as np
import os
from skimage import io
from tqdm import tqdm



std_2_fwhm = 2 * np.sqrt(2 * np.log(2))



def characterize_3D_spots(xdms,
                          spots_3D=None,
                          vector_int_cutoff=None,
                          q_target=None,
                          q_cutoff=None,
                          int_cutoff=None):

    if spots_3D is None:
        if not hasattr(xdms, 'spots_3D') or xdms.spots_3D is None:
            err_str = ("XRDMapStack must first find 3D spots before "
                    + "they can be characterized.")
            raise ValueError(err_str)
        else:
            spots_3D = xdms.spots_3D
    
    # Try to find this information
    if vector_int_cutoff is None:
        try:
            xdms.open_xdms_hdf()
            vector_int_cutoff = xdms.xdms_hdf['xrdmapstack/reflections'].attrs['abs_int_cutoff']
        finally:
            xdms.close_xdms_hdf()

    # One more guard block following q-target logic
    if q_target is not None:
        if q_cutoff is None:
            err_str = "q_cutoff required when using q_target."
            raise ValueError(err_str)
    
    # Effective map shape
    map_shape = (np.max(spots_3D['map_y']) + 1,
                 np.max(spots_3D['map_x']) + 1)

    # Pre-fill expected data
    # Zero moment analysis
    int_map = np.empty(map_shape)
    int_map[:] = np.nan
    int_std_map = int_map.copy()
    # First moment analysis
    com_map = np.empty((*map_shape, 3))
    com_map[:] = np.nan
    com_std_map = com_map.copy()
    com_mag_map = int_map.copy()
    com_mag_std_map = com_mag_map.copy()
    # Second moment analysis
    fwhm_map = com_map.copy()
    fwhm_std_map = fwhm_map.copy()
    # Third moment analysis
    skew_map = com_map.copy()
    skew_std_map = skew_map.copy()
    # Rotated second moment analysis
    rot_fwhm_map = fwhm_map.copy()
    rot_fwhm_std_map = fwhm_std_map.copy()

    # Iterate through all pixels
    for index in tqdm(range(np.prod(map_shape))):
        indices = np.unravel_index(index, map_shape)

        # Get vectors used for spot search
        vectors = xdms.xdms_vector_map[indices]
        vectors = vectors[vectors[:, -1] > float(vector_int_cutoff)]
        spots_labels = xdms.xdms_spot_labels_map[indices]
        unique_spots_labels = np.unique(spots_labels)

        # Find spots
        df = xdms.pixel_3D_spots(indices, spots_3D=spots_3D, copied=True)

        if len(df) == 0:
            continue

        q_mag = df['q_mag'].values
        intensity = df['intensity'].values

        # Target a specific hkl ring
        q_mag_mask = np.ones_like(q_mag, dtype=np.bool_)
        if q_target is not None:
            q_mag_mask = np.abs(q_mag - q_target) < q_cutoff
        
        # More intensity filtering
        int_mask = np.ones_like(intensity, dtype=np.bool_)
        if int_cutoff is not None:
            int_mask = intensity > int_cutoff
        
        unique_spots_labels = unique_spots_labels[q_mag_mask & int_mask]
        if len(unique_spots_labels) == 0:
            continue
        
        # Iterate through all found spots and characterize
        int_list, com_list, com_mag_list, fwhm_list, skew_list, rot_fwhm_list = [], [], [], [], [], []
        for label in unique_spots_labels:
            label_mask = spots_labels == label
            spot_vectors = vectors[label_mask]

            if len(spot_vectors) <= 2:
                continue

            positions = spot_vectors[:, :3]  # Shape: (N, 3)
            w = spot_vectors[:, -1]           # Shape: (N,)

            # Add a new axis to weights for broadcasting across columns: (N, 1)
            w_grid = w[:, np.newaxis]
            total_w = np.sum(w)

            # 2. Vectorized 3D Mean (Center of Mass)
            # Multiplies each X, Y, Z by its weight and sums down the rows
            mean = np.sum(positions * w_grid, axis=0) / total_w  # Shape: (3,)

            # 3. Vectorized 3D Standard Deviation
            # Subtracts the mean vector from each row using broadcasting
            variance = np.sum(w_grid * (positions - mean) ** 2, axis=0) / total_w  # Shape: (3,)
            std = np.sqrt(variance)  # Shape: (3,)
            fwhm = std_2_fwhm * std

            # 4. Vectorized 3D Skewness (Third Standardized Moment)
            # Divides the deviations by the standard deviation vector before cubing
            standardized_deviations = (positions - mean) / std
            skewness = np.sum(w_grid * (standardized_deviations ** 3), axis=0) / total_w  # Shape: (3,)

            # Rotate reference frame
            R = get_rotation_matrix(mean)
            rot_positions = spot_vectors[:, :3] @ R
            rot_mean = np.sum(rot_positions * w_grid, axis=0) / total_w  # Shape: (3,)
            rot_variance = np.sum(w_grid * (rot_positions - mean) ** 2, axis=0) / total_w  # Shape: (3,)
            rot_std = np.sqrt(rot_variance)  # Shape: (3,)
            rot_fwhm = std_2_fwhm * rot_std
            
            int_list.append(total_w)
            com_list.append(mean)
            com_mag_list.append(np.linalg.norm(mean))
            fwhm_list.append(fwhm)
            skew_list.append(skewness)
            rot_fwhm_list.append(rot_fwhm)

            # raise RuntimeError

            # int_list.append(np.sum(spot_vectors[:, -1]))

            # com = xmt.utilities.math.arbitrary_center_of_mass(spot_vectors[:, -1], *spot_vectors[:, :3].T)
            # com_list.append(com)
            # com_mag_list.append(np.linalg.norm(com))
            
            # aweights = xmt.utilities.math.rescale_array(spot_vectors[:, -1], upper=1, lower=0, copy=True)
            # std = np.sqrt(np.asarray([np.cov(spot_vectors[:, i], aweights=aweights) for i in range(3)]))
            # fwhm_list.append(std_2_fwhm * std)    

            # # Rotation
            # R = get_rotation_matrix(com)
            # # rot_spots = (R.T @ spot_vectors[:, :3].T).T
            # rot_spots = spot_vectors[:, :3] @ R
            # rot_std = np.asarray([np.cov(rot_spots[:, i], aweights=aweights) for i in range(3)])
            # rot_fwhm_list.append(std_2_fwhm * rot_std)

        # raise RuntimeError
        
        if len(int_list) > 1:
            int_map[indices] = np.mean(int_list)
            int_std_map[indices] = np.std(int_list)
        
        if len(com_list) > 1:
            com_map[indices] = np.mean(com_list, axis=0)
            com_std_map[indices] = np.std(com_list, axis=0)
            com_mag_map[indices] = np.mean(com_mag_list, axis=0)
            com_mag_std_map[indices] = np.std(com_mag_list, axis=0)
        
        if len(fwhm_list) > 1:
            fwhm_map[indices] = np.mean(fwhm_list, axis=0)
            fwhm_std_map[indices] = np.std(fwhm_list, axis=0)
            rot_fwhm_map[indices] = np.mean(rot_fwhm_list, axis=0)
            rot_fwhm_std_map[indices] = np.std(rot_fwhm_list, axis=0)
        
        if len(skew_list) > 1:
            skew_map[indices] = np.mean(skew_list, axis=0)
            skew_std_map[indices] = np.std(skew_list, axis=0)

    return int_map, int_std_map, com_map, com_std_map, com_mag_map, com_mag_std_map, fwhm_map, fwhm_std_map, skew_map, skew_std_map, rot_fwhm_map, rot_fwhm_std_map

import xrdmaptools

@xrdmaptools.XRDMapStack._protect_xdms_hdf()
def load_xdms_spot_labels(xdms):

    base_grp = xdms.xdms_hdf['xrdmapstack']

    if not hasattr(xdms, 'xdms_vector_map') or xdms.xdms_vector_map is None:
        err_str = 'Data must first be vectorized!'
        raise ValueError(err_str)
    if 'vectorized_map' not in base_grp:
        err_str = 'Data must first be vectorized!'
        raise ValueError(err_str)
    elif 'spot_labels_map' not in base_grp['vectorized_map']:
        err_str = '3D spots must first be found!'
        raise ValueError(err_str)

    map_shape = xdms.xdms_vector_map.shape
    xdms_spot_labels_map = np.empty(map_shape, dtype=object)

    map_grp = base_grp['vectorized_map/spot_labels_map']
    for index in range(np.prod(map_shape)):
        indices = np.unravel_index(index, map_shape)
        title = ','.join([str(ind) for ind in indices])
        xdms_spot_labels_map[indices] = map_grp[title][:]

    xdms.xdms_spot_labels_map = xdms_spot_labels_map

        




def get_rotation_matrix(point):
    
    # radial component
    z_local = point / np.linalg.norm(point)

    # xy plane component
    if np.allclose(z_local[0:2], 0.0):
        x_local = np.array([1.0, 0.0, 0.0])
    else:
        x_local = np.array([-z_local[1], z_local[0], 0.0])
        x_local /= np.linalg.norm(x_local)
    
    # Orthogonal component
    y_local = np.cross(z_local, x_local)

    return np.column_stack((x_local, y_local, z_local))




def convert_1D(data, std=None):

    data_1D = np.nanmean(data, axis=0)
    std_1D = np.asarray([
        np.sqrt(np.mean(np.square(std_list[~np.isnan(std_list)])))
        for std_list in std.T
    ])

    return data_1D, std_1D

def plot_1D(data, std=None, x=None, step=0.1, title=None, wd='/nsls2/users/emusterma/Figures/Ru patterns/20260817_subset/'):

    if std is not None:
        data_1D, std_1D = convert_1D(data, std)
    else:
        data_1D = np.nanmean(data, axis=0)

    if x is None:
        x = np.array(range(len(data_1D))) * step
        x -= x[len(x) // 2]

    fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
    ax.plot(x, data_1D, label='Mean', linewidth=2)
    if std is not None:
        ax.fill_between(x, data_1D - std_1D, data_1D + std_1D, alpha=0.2, label='±1 Std Dev')
    ax.set_xlabel('Relative Position [um]')
    ax.set_title(title)
    fig.show()
    # fig.savefig(f'{wd}{title} plot.png')
    


def plot_2D(data, step=0.1, title=None, wd='/nsls2/users/emusterma/Figures/Ru patterns/20260817_subset/'):

    x = data.shape[1] * step
    y = data.shape[0] * step
    extent = [-(x / 2) - (step / 2),
              (x / 2) + (step / 2),
              (y / 2) + (step / 2),
              -(y / 2) - (step / 2)]

    fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
    img = ax.imshow(data, extent=extent, interpolation='none')
    fig.colorbar(img, ax=ax)
    ax.set_aspect('equal')
    
    ax.set_xlabel('Relative Position [um]')
    ax.set_xlabel('Relative Position [um]')

    ax.set_title(title)
    fig.show()
    # fig.savefig(f'{wd}{title} map.png')


def get_and_generate_plots(xdms):

    xdms[0].load_xrfmap()
    ru_map = xdms[0].xrf['Ru_L'] / xdms[0].sclr_dict['i0']
    title = f'Ru L XRF @ {xdms[0].energy:.2f} keV'
    plot_2D(ru_map, title=title)
    plot_1D(ru_map, title=title)

    def generate_plots(data, prefix=''):

        title = f'{prefix} Average Intensity'
        plot_2D(data[0], title=title)
        plot_1D(data[0], data[1], title=title)

        title = f'{prefix} Average Q-Magnitude'
        plot_2D(data[4], title=title)
        plot_1D(data[4], data[5], title=title)

        plot_1D(data[6][..., 0], data[7][..., 0], title=f'{prefix} FWHM Qx')
        plot_1D(data[6][..., 1], data[7][..., 1], title=f'{prefix} FWHM Qy')
        plot_1D(data[6][..., 2], data[7][..., 2], title=f'{prefix} FWHM Qz')

        plot_1D(data[8][..., 2], data[9][..., 2], title=f'{prefix} FWHM Q-radial')
        plot_1D(data[8][..., 0], data[9][..., 0], title=f'{prefix} FWHM Q-tangent xy')
        plot_1D(data[8][..., 1], data[9][..., 1], title=f'{prefix} FWHM Q-tangent ortho')

    q_cutoff = 0.04

    data = characterize_3D_spots(xdms, q_target=2.683, q_cutoff=q_cutoff)
    generate_plots(data, prefix='(1 0 0)')

    data = characterize_3D_spots(xdms, q_target=2.935, q_cutoff=q_cutoff)
    generate_plots(data, prefix='(0 0 2)')

    data = characterize_3D_spots(xdms, q_target=3.058, q_cutoff=q_cutoff)
    generate_plots(data, prefix='(1 0 1)')



def plot_multi_orientations(xdms, measured_spots, index_list, phase, **kwargs):

    fig, ax = xmt.plot.volume.plot_3D_scatter(measured_spots,
                                  colors='k',
                                  skip=1,
                                  s=5,
                                  edges=xdms.edges,
                                  alpha=1,
                                  **kwargs)
    ax.collections[0].set_label('Measured')

    ori_list = []
    for indices in index_list:
        df = xdms.pixel_3D_spots(indices, copied=True)
        df = df[df['grain_id'] == 0]
        df.reset_index(drop=True, inplace=True)
        spots = df[['qx', 'qy', 'qz']].values.copy()
        hkls = df[['h', 'k', 'l']].values

        # spot_inds = df.index.values
        # ref_inds = [np.all(hkl == phase.all_hkls, axis=1).nonzero()[0][0] for hkl in hkls]
        # indexing = np.asarray([spot_inds, ref_inds]).T

        orientation = Rotation.align_vectors(
                        spots,
                        phase.Q(hkls))[0]
        rot_qs = orientation.apply(phase.all_qs)
        temp_qmask = xdms.qmask.generate(rot_qs)

        ax.scatter(*rot_qs[temp_qmask].T, s=5, label=str(tuple(indices)))
        ori_list.append(orientation)

    ax.legend()
    fig.show()

    return ori_list
    



def get_hkl_comparisons(phase_list, q_vector, d_cutoff=0.1, keep_all=False, angle_tolerance=15):

    d_target = xmt.utilities.math.convert_qd(np.linalg.norm(q_vector))

    theta_i_x = xmt.utilities.math.vector_angle(q_vector, [1, 0, 0], degrees=True)
    theta_i_y = xmt.utilities.math.vector_angle(q_vector, [0, 1, 0], degrees=True)
    theta_i_z = xmt.utilities.math.vector_angle(q_vector, [0, 0, 1], degrees=True)

    comps = []
    for phase in phase_list:
        phase.get_hkl_reflections()

        cands = []
        for d, hkl in zip(*[phase.reflections[key] for key in ['d', 'hkl']]):
            if abs(d - d_target) < d_cutoff:
                hkls = tuple(phase.lattice.equivalent_hkls(hkl))
                valid_mask = np.ones(len(hkls), dtype=np.bool_)

                axes_angles = []
                for i, hkl in enumerate(hkls):

                    theta_c_a = xmt.utilities.math.vector_angle(phase.Q(hkl), phase.Q([1, 0, 0]), degrees=True)
                    theta_c_b = xmt.utilities.math.vector_angle(phase.Q(hkl), phase.Q([0, 1, 0]), degrees=True)
                    theta_c_c = xmt.utilities.math.vector_angle(phase.Q(hkl), phase.Q([0, 0, 1]), degrees=True)

                    min_angle_ax = np.abs(np.min([abs(theta_c_a - theta_i_x), 180 - (theta_c_a + theta_i_x)]))
                    min_angle_by = np.abs(np.min([abs(theta_c_b - theta_i_y), 180 - (theta_c_b + theta_i_y)]))
                    min_angle_cz = np.abs(np.min([abs(theta_c_c - theta_i_z), 180 - (theta_c_c + theta_i_z)]))

                    angles = np.array([min_angle_ax, min_angle_by, min_angle_cz])

                    if not keep_all:
                        valid_mask[i] = np.all(angles < angle_tolerance)

                    axes_angles.append(angles)
                
                if np.any(valid_mask):
                    cands.append({
                        'd' : d,
                        'hkls' : tuple([hkl for hkl, b in zip(hkls, valid_mask) if b]),
                        'axes_angles' : np.asarray(axes_angles)[valid_mask]
                    })

        comps.append(cands)

    return comps


def find_matches(comps):

    combined = []
    for comp in comps:
        combined_cand = {
            'd' : [],
            'hkls' : [],
            'axes_angles' : []
        }
        for dic in comp:
            for hkl, angles in zip(dic['hkls'], dic['axes_angles']):
                combined_cand['d'].append(dic['d'])
                combined_cand['hkls'].append(tuple(hkl))
                combined_cand['axes_angles'].append(angles)
        
        # combined_cand['hkls'] = np.asarray(combined_cand['hkls'])
        # combined_cand['axes_angles'] = np.asarray(combined_cand['axes_angles'])

        combined.append(combined_cand)
    
    mono = combined[0]
    hexa = combined[1]

    matches = []
    for d, hkl, axes_angles in zip(*mono.values()):
        
        predicted_hkl = np.array([
            0.5 * (hkl[1] - hkl[0]),
            -hkl[1], 
            hkl[2]
        ])

        if np.any(predicted_hkl % 1) > 0:
            continue
        
        predicted_hkl = tuple([int(r) for r in predicted_hkl])
        # print(f'{hkl} --> {predicted_hkl}')

        if predicted_hkl in hexa['hkls']:
            matches.append((hkl, predicted_hkl))
    
    return matches
        

def pairwise_analysis(all_spot_qs, all_ref_qs, all_ref_hkls, all_ref_fs, near_q, near_angle, min_q, degrees=True, symmetrize='lattice'):
    pairs = find_valid_pairs(all_spot_qs, all_ref_qs, all_ref_hkls, all_ref_fs, near_q, near_angle, min_q, degrees=degrees, symmetrize=symmetrize)

    possible_pairs = []
    for pair in pairs:
        spot_inds, ref_inds = np.asarray(pair).T
        orientation = Rotation.align_vectors(
                        all_spot_qs[spot_inds],
                        all_ref_qs[ref_inds])[0]

        axes_angles = np.degrees(np.arccos(np.diag(orientation.as_matrix())))
        axes_angles = np.asarray([[a, 180 - a] for a in axes_angles])
        if np.all(np.any(axes_angles < 20, axis=1)):
            print(axes_angles)
            possible_pairs.append(pair)


    return possible_pairs



def reduce_vector_map(vector_map, q_target=None, q_cutoff=None, int_cutoff=None):

    if q_target is None and int_cutoff is None:
        raise ValueError('Must define q_target or int_cutoff!')
    if q_target is not None and q_cutoff is None:
        raise ValueError('Must define q_cutoff with q_target!')


    reduced_vector_map = np.empty(vector_map.shape, dtype=object)

    # for index in tqdm(range(np.product(vector_map.shape))):
    #     indices = np.unravel_index(vector_map.shape)
    for indices in tqdm(np.ndindex(vector_map.shape), total=vector_map.size):
        reduced_vector_map[indices] = np.empty((0, 4))

        vectors = vector_map[indices]

        if len(vectors) == 0:
            continue
        
        if int_cutoff is not None:
            int_mask = vectors[:, -1] > int_cutoff
        else:
            int_mask = np.ones(len(vectors), dtype=np.bool_)

        if int_mask.sum() == 0:
            continue
        vectors = vectors[int_mask]

        if q_target is not None:
            q_mask = np.linalg.norm(vectors[:, :3] - q_target, axis=-1) < q_cutoff
        else:
            q_mask = np.ones(len(vectors), dtype=np.bool_)

        if q_mask.sum() == 0:
            continue
        reduced_vector_map[indices] = vectors[q_mask]

    return reduced_vector_map
        