import numpy as np
import os
from skimage import io
from tqdm import tqdm
from scipy.spatial.transform import Rotation


from xrdmaptools.crystal.strain import phase_get_strain_orientation
from xrdmaptools.utilities.math import convert_qd
from xrdmaptools.crystal.crystal import are_collinear, are_coplanar


def pixel_spots(spots, indices):
    ps = spots[((spots['map_x'] == indices[1]) & (spots['map_y'] == indices[0]))]
    return ps


def get_maps(spots_3D, phase, grain_id=0):

    # Effective map shape
    map_shape = (np.max(spots_3D['map_y']) + 1,
                 np.max(spots_3D['map_x']) + 1)

    ec_map = np.empty((*map_shape, 3, 3))
    ec_map[:] = np.nan
    es_map = ec_map.copy()
    ori_map = ec_map.copy()
    lattice_map = np.empty((*map_shape, 6))
    lattice_map[:] = np.nan

    for index in tqdm(range(np.prod(map_shape))):

        try:
            indices = np.unravel_index(index, map_shape)
            pixel_df = pixel_spots(spots_3D, indices).sort_values(by='intensity', ascending=False)
            spot_qs = pixel_df[['qx', 'qy', 'qz']].values
            spot_hkls = pixel_df[['h', 'k', 'l']].values.astype(int)

            mask = pixel_df['grain_id'].values == grain_id
            if (np.sum(mask) >= 3
                and not xmt.crystal.crystal.are_coplanar(spot_hkls[mask])):

                eij, ori, strained = phase_get_strain_orientation(spot_qs[mask],
                                                                  spot_hkls[mask],
                                                                  phase)
                ec_map[indices] = eij
                ori_map[indices] = ori
                lattice_map[indices] = [strained.a, strained.b, strained.c,
                                        strained.alpha, strained.beta, strained.gamma]
                
                es_map[indices] = ori @ eij @ ori.T
            
            elif np.sum(mask) == 2:
                ori = Rotation.align_vectors(phase.Q(spot_hkls[mask]), spot_qs[mask])[0]
                ori_map[indices] = ori.as_matrix()
        except Exception as e:
            print(f'Error at index {indices}: {e}')
            print(spot_qs)
            print(spot_hkls)
            raise e
    
    return lattice_map, ori_map, ec_map, es_map



def build_indexing(spots_3D, phase, grain_id=0):

    spots = spots_3D[['qx', 'qy', 'qz']].values
    hkls = spots_3D[['h', 'k', 'l']].values
    grain_ids = spots_3D['grain_id'].values

    indexing = []
    for i, (spot, hkl, i_grain_id) in enumerate(zip(spots, hkls, grain_ids)):
        if i_grain_id != grain_id:
            continue

        hkl_index = np.all(np.round(hkl).astype(int) == phase.all_hkls, axis=1).nonzero()[0][0]
        indexing.append([i, hkl_index])
    
    return np.asarray(indexing)



def desymmetrize_full_map(spots_3D, phase, near_q, xdms):

    # Effective map shape
    map_shape = (np.max(spots_3D['map_y']) + 1,
                 np.max(spots_3D['map_x']) + 1)

    for index in tqdm(range(np.prod(map_shape))):
        
        indices = np.unravel_index(index, map_shape)
        pixel_df = pixel_spots(spots_3D, indices)
        spots = pixel_df[['qx', 'qy', 'qz']].values
        spot_ints = pixel_df['intensity'].values

        for grain_id in pixel_df['grain_id'].unique():
            if np.isnan(grain_id):
                continue
            
            grain_mask = pixel_df['grain_id'] == grain_id
            if grain_mask.sum() < 2:
                continue

            indexing = build_indexing(pixel_df[grain_mask], phase, grain_id)
            deindexing, qof = xmt.reflections.spot_blob_indexing_3D.desymmetrize(
                indexing,
                phase.all_qs,
                phase.all_fs,
                spots,
                spot_ints,
                near_q,
                xdms.qmask,
                space_group_nr=phase.lattice.space_group_nr
            )
            new_hkls = phase.all_hkls[deindexing[:, 1]]
            # print("assigning", pixel_df[grain_mask].index, new_hkls)
            # print("before", spots_3D.loc[pixel_df[grain_mask].index, ['h','k','l']].values)
            spots_3D.loc[pixel_df[grain_mask].index, ['h', 'k', 'l']] = new_hkls
            # print("after", spots_3D.loc[pixel_df[grain_mask].index, ['h','k','l']].values)



def seeded_map_indexing(xdms, near_q, phase, spots_3D=None, seeds=None, **kwargs):
    
    # Some quick defaults
    kwargs.setdefault('verbose', False)

    if spots_3D is None:
        if (not hasattr(self, 'spots_3D')
            or self.spots_3D is None):
            err_str = "spots_3D must be given or already found before they can be indexed."
            raise AttributeError(err_str)
        else:
            spots_3D = self.spots_3D

    # Get all reference values from the phase to index
    spot_q_mags = np.linalg.norm(all_spot_qs, axis=1)
    max_q = np.max(spot_q_mags)
    if not phase._has_reciprocal_lattice:    
        phase.generate_reciprocal_lattice(1.15 * max_q)
    elif kwargs['verbose']:
        if phase._reciprocal_lattice_qmax < 1.15 * max_q:
            warn_str = ('WARNING: Phase already has a reciprocal '
                        + 'lattice, but it was generated with a '
                        + 'maximum q-vector lower than expected. '
                        + 'Some reflections may be missing for '
                        + 'indexing.')
            print(warn_str)
    ref_mags = np.linalg.norm(phase.all_qs, axis=1)

    # Effective map shape
    map_shape = (np.max(spots_3D['map_y']) + 1,
                 np.max(spots_3D['map_x']) + 1)

    # Update spots dataframe with new columns
    spots_3D['phase'] = ''
    spots_3D[['grain_id', 'h', 'k', 'l', 'qof']] = np.nan

    # Construct iterable
    if kwargs['verbose']:
        iterable = timed_iter(range(np.prod(map_shape)))
    else:
        iterable = tqdm(range(np.prod(map_shape)))

    for index in iterable:
        
        # Get the individual map pixel's spots
        indices = np.unravel_index(index, map_shape)
        pixel_df = pixel_spots(spots_3D, indices)

        # Get values
        spot_qs = pixel_df[['qx', 'qy', 'qz']].values
        spot_ints = pixel_df['intensity'].values
        spot_mags = pixel_df['q_mag'].values

        # Check for indexability
        if len(spot_qs) < 2 or are_collinear(spot_qs):
            continue

        # Quick mask for better speed
        ext = 0.15
        ref_mask = ((ref_mags > spot_mags.min() * (1 - ext))
                    & (ref_mags < spot_mags.max() * (1 + ext)))
        
        indexings, qofs = xmt.reflections.spot_blob_indexing_3D.pattern_decomposition_from_seeds(
            seeds,
            spot_qs,
            spot_ints,
            phase.all_qs[ref_mask],
            phase.all_fs[ref_mask],
            xdms.qmask,
            near_q,
            space_group_nr=phase.lattice.space_group_nr,
            **kwargs
        )

        for gid, (ind, qof) in enumerate(zip(indexings, qofs)):
            if np.isnan(qof):
                continue

            spot_inds, ref_inds = ind.T
            hkls = phase.all_hkls[ref_mask][ref_inds]

            # Assign values
            rel_ind = pixel_df.index[spot_inds]
            spots_3D.loc[rel_ind, 'phase'] = phase.name
            spots_3D.loc[rel_ind, 'grain_id'] = gid
            spots_3D.loc[rel_ind, 'qof'] = qof
            spots_3D.loc[rel_ind, ['h', 'k', 'l']] = hkls



def get_strain_orientation_maps(spots_3D,
                                phase,
                                grain_id=0,
                                int_cutoff=0):

    # Effective map shape
    map_shape = (np.max(spots_3D['map_y']) + 1,
                 np.max(spots_3D['map_x']) + 1)
    
    int_map = np.empty(map_shape)
    int_map[:] = np.nan
    ec_map = np.empty((*map_shape, 3, 3))
    ec_map[:] = np.nan
    es_map = ec_map.copy()
    ori_map = ec_map.copy()
    lattice_map = np.empty((*map_shape, 6))
    lattice_map[:] = np.nan

    for index in tqdm(range(np.prod(map_shape))):

        indices = np.unravel_index(index, map_shape)
        pixel_df = pixel_spots(spots_3D, indices).sort_values(by='intensity', ascending=False)
        spot_mags = pixel_df['q_mag'].values
        spot_qs = pixel_df[['qx', 'qy', 'qz']].values
        spot_ints = pixel_df['intensity'].values
        spot_hkls = pixel_df[['h', 'k', 'l']].values

        grain_mask = pixel_df['grain_id'] == grain_id
        int_mask = np.sum(spot_ints) > int_cutoff
        mask = grain_mask & int_mask

        if np.sum(mask) >= 3 and not are_coplanar(spot_hkls[mask]):

            eij, ori, strained = phase_get_strain_orientation(spot_qs[mask],
                                                              spot_hkls[mask],
                                                              phase)
            es_map[indices] = eij
            ori_map[indices] = ori
            lattice_map[indices] = [strained.a, strained.b, strained.c,
                                    strained.alpha, strained.beta, strained.gamma]
            
            ec_map[indices] = ori @ eij @ ori.T
        
        elif np.sum(mask) == 2 and not are_collinear(spot_hkls[mask]):
            ori = Rotation.align_vectors(phase.Q(spot_hkls[mask]), spot_qs[mask])[0]
            ori_map[indices] = ori.as_matrix()
        
        if np.sum(mask) > 0:
            int_map[indices] = np.sum(spot_ints[mask])


    
    return int_map, lattice_map, ori_map, ec_map, es_map


def get_all_strain_orientation_maps(spots_3D,
                                    phase,
                                    grain_id=0,
                                    int_cutoff=0):

    # Effective map shape
    map_shape = (np.max(spots_3D['map_y']) + 1,
                 np.max(spots_3D['map_x']) + 1)
    
    int_map = np.empty(map_shape)
    int_map[:] = np.nan
    ec_map = np.empty((*map_shape, 3, 3))
    ec_map[:] = np.nan
    es_map = ec_map.copy()
    ori_map = ec_map.copy()
    lattice_map = np.empty((*map_shape, 6))
    lattice_map[:] = np.nan

    for index in tqdm(range(np.prod(map_shape))):

        indices = np.unravel_index(index, map_shape)
        pixel_df = pixel_spots(spots_3D, indices).sort_values(by='intensity', ascending=False)
        spot_mags = pixel_df['q_mag'].values
        spot_qs = pixel_df[['qx', 'qy', 'qz']].values
        spot_ints = pixel_df['intensity'].values
        spot_hkls = pixel_df[['h', 'k', 'l']].values
        
        int_mask = spot_ints > int_cutoff
        nan_mask = ~np.isnan(pixel_df['grain_id'])

        if int_mask.sum() == 0:
            continue
        elif nan_mask.sum() == 0:
            int_map[indices] = np.sum(spot_ints[int_mask])
            continue

        # Find grain with highest intensity
        grain_intensity = [np.sum(spot_ints[int_mask][pixel_df['grain_id'][int_mask] == gi]) for gi in np.unique(pixel_df['grain_id'][nan_mask])]
        try:
            grain_id = np.unique(pixel_df['grain_id'][nan_mask])[np.argmax(grain_intensity)]
        except Exception:
            raise Exception

        grain_mask = pixel_df['grain_id'] == grain_id
        mask = grain_mask & int_mask

        if np.sum(mask) >= 3 and not are_coplanar(spot_hkls[mask]):

            eij, ori, strained = phase_get_strain_orientation(spot_qs[mask],
                                                              spot_hkls[mask],
                                                              phase)
            es_map[indices] = eij
            ori_map[indices] = ori
            lattice_map[indices] = [strained.a, strained.b, strained.c,
                                    strained.alpha, strained.beta, strained.gamma]
            
            ec_map[indices] = ori @ eij @ ori.T
        
        elif np.sum(mask) == 2 and not are_collinear(spot_hkls[mask]):
            ori = Rotation.align_vectors(phase.Q(spot_hkls[mask]), spot_qs[mask])[0]
            ori_map[indices] = ori.as_matrix()
        
        if np.sum(mask) > 0:
            int_map[indices] = np.sum(spot_ints[mask])


    
    return int_map, lattice_map, ori_map, ec_map, es_map


def plot_maps(map_list,
              title_list,
              shape,
              step=1,
              figsize=(5, 5),
              fig_title=None,
              vmin=None,
              vmax=None,
              cmap='viridis'):

    fig, ax = plt.subplots(*shape, figsize=figsize, layout='constrained')
    axes = ax.ravel()

    map_shape = map_list[0].shape
    map_extent = [
        -(map_shape[1] * step / 2),
        (map_shape[1] * step / 2),
        (map_shape[0] * step / 2),
        -(map_shape[0] * step / 2)
    ]

    # Create each map
    for i, (data, title) in enumerate(zip(map_list, title_list)):

        if (cmap in {'Spectral_r', 'Spectral', 'bwr', 'bwr_r'}
            and vmin is None and vmax is None):
            ext = np.max(np.abs([np.nanmin(data), np.nanmax(data)]))
            i_vmin = -ext
            i_vmax = ext
        else:
            i_vmin = vmin
            i_vmax = vmax

        im = axes[i].imshow(data,
                            extent=map_extent,
                            vmin=i_vmin,
                            vmax=i_vmax,
                            cmap=cmap)
        fig.colorbar(im, ax=axes[i])
        axes[i].set_title(title)
        axes[i].set_aspect('equal')
    
    for axi in ax.flat:
        axi.set(xlabel='x-position [μm]',
                ylabel='y-position [μm]')
        axi.label_outer()

    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=14, fontweight='bold')

    fig.show()


def get_strain_profiles(es_map, func=np.median):


    strain_maps = [es_map[..., *indices] for indices in [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]]
    # titles = ['e_xx', 'e_yy', 'e_zz', 'e_xy', 'e_xz', 'e_yz']

    strain_profiles = [func(data, axis=0) for data in strain_maps]
    strain_err = [np.std(data, axis=0) for data in strain_maps]

    return strain_profiles, strain_err



def plot_strain_profiles(es_map):

    strain_maps = [es_map[..., *indices] for indices in [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]]
    titles = ['e_xx', 'e_yy', 'e_zz', 'e_xy', 'e_xz', 'e_yz']

    strain_plots = [np.median(data, axis=0) for data in strain_maps]
    strain_err = [np.std(data, axis=0) for data in strain_maps]

    fig, ax = plt.subplots()

    for i, (data, title, err) in enumerate(zip(strain_plots, titles, strain_err)):
        # raise Exception
        ax.errorbar(range(len(data)), data, label=title, yerr=err)

    ax.legend()
    fig.show()


def plot_selected_strains(all_es_map, all_int_map, strain_indices=0, int_cutoff=1e3):

    fig, ax = plt.subplots(figsize=(5, 5))

    for i in range(len(strains_list)):

        strain_profiles, strain_errs = strains_list[i]
        strain_profile = strain_profiles[ind]
        strain_err = strain_errs[ind]

        ax.errorbar(range(len(strain_profile)), strain_profile, yerr=strain_err, label=i)

    ax.legend()
    fig.show()


def get_profile_distance(profile, step=1):

    nan_mask = np.isnan(profile)

    edges = {}

    in_feature = False
    if not nan_mask[0]:
        in_feature = True
        edges['left_edge_left_feature'] = np.nan

    for i, val in enumerate(nan_mask):
        if not in_feature and not val:
            in_feature = True
            if 'left_edge_left_feature' not in edges:
                edges['left_edge_left_feature'] = i
            else:
                edges['left_edge_right_feature'] = i
        
        if in_feature and val:
            in_feature = False
            if 'right_edge_left_feature' not in edges:
                edges['right_edge_left_feature'] = i
            else:
                edges['right_edge_right_feature'] = i
    
    if 'right_edge_right_feature' not in edges:
        edges['right_edge_right_feature'] = np.nan
    
    # return edges

    dist = edges['left_edge_right_feature'] - edges['right_edge_left_feature']

    return (dist * step,
            slice(edges['left_edge_right_feature'], edges['right_edge_right_feature']))


def plot_selected_strains(all_es_maps, all_int_maps, strain_indices=(0, 0), int_cutoff=1e3, step=0.5):

    titles = dict(zip([(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)],
                      ['ε_xx', 'ε_yy', 'ε_zz', 'ε_xy', 'ε_xz', 'ε_yz']))
    title = titles[strain_indices]

    def int_mask(arr, ints, int_cutoff=int_cutoff):
        arr = arr.copy()
        arr[ints < int_cutoff] = np.nan
        return arr
    
    distances, profiles, errors = [], [], []
    # Get strain profiles
    for es_map, int_map in zip(all_es_maps, all_int_maps):
        strain_map = int_mask(es_map[..., *strain_indices], int_map)

        strain_map *= 1e3

        full_profile = np.median(strain_map, axis=0)
        full_err = np.std(strain_map, axis=0)
        
        dist, feature_slice = get_profile_distance(full_profile, step=step)

        # raise Exception
        if np.isnan(feature_slice.stop):
            feature_slice = slice(feature_slice.start, None)

        distances.append(dist)
        profiles.append(full_profile[feature_slice])
        errors.append(full_err[feature_slice])
    
    profiles = [p for _, p in sorted(zip(distances, profiles))]
    errors = [e for _, e in sorted(zip(distances, errors))]
    distances = sorted(distances)

    fig, ax = plt.subplots(figsize=(10, 5))

    for i in range(len(distances)):

        ax.errorbar(np.array(range(len(profiles[i]))) * step, profiles[i], yerr=errors[i], label=f'Separation = {round(distances[i], 1):.1f} μm')
    
    ax.set_xlabel('x-axis [μm]')
    ax.set_ylabel(f'Absolute {title} Strain [x10^3]')

    ax.legend()
    fig.show()


def map_shape_from_spots(spots_3D):
    return (np.max(spots_3D['map_y']) + 1,
            np.max(spots_3D['map_x']) + 1)




def get_custom_maps(xdms, q_targets, hkls, phase, cutoff=0.1):


    map_shape = xdms.map_shape
    
    int_map = np.empty(map_shape)
    int_map[:] = np.nan
    ec_map = np.empty((*map_shape, 3, 3))
    ec_map[:] = np.nan
    es_map = ec_map.copy()
    ori_map = ec_map.copy()
    lattice_map = np.empty((*map_shape, 7))
    lattice_map[:] = np.nan

    # q_dict = {}
    # int_dict = {}
    # for q_target, hkl in zip(q_targets, hkls):
    #     q_dict[tuple(hkl)] = track_q_vector_com(xdms, q_target, cutoff=0.1)
    #     int_dict[tuple(hkl)] = track_q_vector_sum(xdms, q_target, cutoff=0.1)
    

    for index in range(np.prod(xdms.map_shape)):
        indices = np.unravel_index(index, xdms.map_shape)
        pixel_df = pixel_spots(xdms.spots_3D, indices)

        if len(pixel_df) < 1:
            continue

        all_spot_qs = pixel_df[['qx', 'qy', 'qz']].values
        all_spot_ints = pixel_df['intensity'].values

        spot_qs = []
        spot_hkls = []
        for q_target, hkl in zip(q_targets, hkls):
            diff_q = np.linalg.norm(all_spot_qs - q_target, axis=1)
            if np.min(diff_q) < cutoff:
                spot_qs.append(all_spot_qs[np.argmin(diff_q)])
                spot_hkls.append(hkl)
        
        # spot_hkls = []
        # spot_qs = []
        # for hkl in hkls:
        #     spot_q = q_dict[tuple(hkl)][indices]
        #     if not np.any(np.isnan(spot_q)):
        #         spot_qs.append(spot_q)
        #         spot_hkls.append(hkl)

        if len(spot_qs) > 3:

            eij, ori, strained = phase_get_strain_orientation(spot_qs,
                                                              spot_hkls,
                                                              phase)
            es_map[indices] = eij
            ori_map[indices] = ori
            lattice_map[indices] = [strained.a, strained.b, strained.c,
                                    strained.alpha, strained.beta, strained.gamma,
                                    strained.lattice.UnitCellVolume()]
            
            ec_map[indices] = ori @ eij @ ori.T
        
        # if len(spots) > 2:
        #     ori = Rotation.align_vectors(phase.Q(spot_hkls), spot_qs)[0]
        #     ori_map[indices] = ori.as_matrix()
    
    return lattice_map, ori_map, ec_map, es_map