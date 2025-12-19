import numpy as np
import os
from skimage import io
from tqdm import tqdm
from scipy.spatial.transform import Rotation

from xrdmaptools.crystal.strain import phase_get_strain_orientation
from xrdmaptools.utilities.math import convert_qd

def pixel_spots(spots, indices):
    ps = spots[((spots['map_x'] == indices[1]) & (spots['map_y'] == indices[0]))]
    return ps


hkls = np.array([[-5, 0, 0],
                    [-7, 1, -2],
                #  [-9, 2, -2],
                    [-5, 0, 1]])


def get_maps(spots_3D, phase):
    
    ref_qs = phase.Q(hkls)
    ref_mags = np.linalg.norm(ref_qs, axis=1)

    # Effective map shape
    map_shape = (np.max(spots_3D['map_y']) + 1,
                 np.max(spots_3D['map_x']) + 1)

    ec_map = np.empty((*map_shape, 3, 3))
    ec_map[:] = np.nan
    es_map = ec_map.copy()
    ori_map = ec_map.copy()
    lattice_map = np.empty((*map_shape, 6))
    lattice_map[:] = np.nan
    q_map = np.empty((*map_shape, len(hkls)))
    q_map[:] = np.nan

    for index in tqdm(range(np.prod(map_shape))):

        indices = np.unravel_index(index, map_shape)
        pixel_df = pixel_spots(spots_3D, indices).sort_values(by='intensity', ascending=False)
        spot_mags = pixel_df['q_mag'].values
        spot_qs = pixel_df[['qx', 'qy', 'qz']].values

        exp_qs = np.empty((len(hkls), 3))
        exp_qs[:] = np.nan
        for i, mag in enumerate(spot_mags):
            diff_mags = np.abs(mag - ref_mags)
            if (np.any(np.isnan(exp_qs[np.argmin(diff_mags)]))
                and np.min(diff_mags) < 0.025):
                exp_qs[np.argmin(diff_mags)] = spot_qs[i]
            
            if not np.any(np.isnan(exp_qs)):
                break
        
        q_map[indices] = np.linalg.norm(exp_qs, axis=1)
        
        # if indices == (18, 21):
        #     return hkls, ref_qs, exp_qs

        mask = np.any(~np.isnan(exp_qs), axis=1)
        if np.sum(mask) >= 3:
            eij, ori, strained = phase_get_strain_orientation(exp_qs[mask],
                                                              hkls[mask],
                                                              phase)
            es_map[indices] = eij
            ori_map[indices] = ori
            lattice_map[indices] = [strained.a, strained.b, strained.c,
                                    strained.alpha, strained.beta, strained.gamma]
            
            ec_map[indices] = ori @ eij @ ori.T
        
        elif np.sum(mask) == 2:
            ori = Rotation.align_vectors(ref_qs[mask], exp_qs[mask])[0]
            ori_map[indices] = ori.as_matrix()

        

    
    return q_map, lattice_map, ori_map, ec_map, es_map


def convert_ori_map(ori_map):
    rot_map = np.empty((*ori_map.shape[:2], 3))
    rot_map[:] = np.nan
    for index in range(np.prod(ori_map.shape[:2])):
        indices = np.unravel_index(index, ori_map.shape[:2])
        if np.any(np.isnan(ori_map[indices])):
            continue
        rot = Rotation.from_matrix(ori_map[indices])
        rot_map[indices] = rot.as_rotvec()
    return rot_map


def plot_maps(map_list,
              title_list,
              shape,
              step=1,
              figsize=None,
              fig_title=None,
              vmin=None,
              vmax=None,
              cmap=None):

    fig, ax = plt.subplots(*shape, figsize=figsize, layout='constrained')
    axes = ax.ravel()

    map_shape = map_list[0].shape
    map_extent = [
        -(map_shape[0] * step / 2),
        (map_shape[0] * step / 2),
        (map_shape[1] * step / 2),
        -(map_shape[1] * step / 2)
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




# def generate_all_plots(out):
#     plot_maps([out[-1][..., 0, 0], out[-1][..., 1, 1], out[-1][..., 2, 2], out[-1][..., 0, 1], out[-1][..., 0, 2], out[-1][..., 1, 2]],
#               ['e_xx', 'e_yy', 'e_zz', 'e_xy', 'e_xz', 'e_yz'],
#               (2, 3),
#               step=0.75,
#               figsize=(10, 5),
#               fig_title='Strain',
#               vmin=-1.5e-2,
#               vmax=1.5e-2,
#               cmap='Spectral_r')
    
#     rot_map = np.degrees(convert_ori_map(out[2]))
#     plot_maps([rot_map[..., 0], rot_map[..., 1], rot_map[..., 2]],
#               ['w_1', 'w_2', 'w_3'],
#               (1, 3),
#               step=0.75,
#               figsize=(10, 2.5),
#               fig_title='Rotation',
#               cmap='Spectral_r')


def generate_all_difference_plots(out1, out2, phase):

    proc_dict = {
        'q magnitude' : False,
        'hkl strain' : True,
        'lattice orientation' : False,
        'sample strain' : True,
    }

    # Get q_mag plots
    if proc_dict['q magnitude']:
        q1 = out1[0]
        q2 = out2[0]
        dq = q2 - q1

        for i, (map_list, fig_title) in enumerate(zip([q1, q2, dq],
                                                      ['Q Magnitude : Before Crack',
                                                       'Q Magnitude : After Crack',
                                                       'Q Magnitude : Difference'])):
            cmap, vmin, vmax = None, None, None
            if i == 2:
                cmap='Spectral_r'
                vmin = -1e-2
                vmax = 1e-2
            
            plot_maps([map_list[..., 0], map_list[..., 1], map_list[..., 2]],
                      ['(-5 0 0)', '(-7 1 -2)', '(-5 0 1)'],
                      (1, 3),
                      step=0.75,
                      figsize=(10, 2.5),
                      fig_title=fig_title,
                      vmin=vmin,
                      vmax=vmax,
                      cmap=cmap)

    # Get hkl strain plots
    if proc_dict['hkl strain']:
        d_calc = phase.planeDistance(hkls)
        d1 = convert_qd(out1[0])
        d2 = convert_qd(out2[0])

        e1, e2 = d1.copy(), d2.copy()
        for i in range(len(d_calc)):
            e1[..., i] = (d1[..., i] - d_calc[i]) / d_calc[i]
            e2[..., i] = (d2[..., i] - d_calc[i]) / d_calc[i]
        de = e2 - e1

        for i, (map_list, fig_title) in enumerate(zip([e1, e2, de],
                                                      ['HKL Strain : Before Crack',
                                                      'HKL Strain : After Crack',
                                                      'HKL Strain : Difference'])):
            cmap = 'Spectral_r'
            vmin, vmax = None, None
            if i == 2:
                vmin = -1.5
                vmax = 1.5
            
            map_list *= 1e3
            
            plot_maps([map_list[..., 0], map_list[..., 1], map_list[..., 2]],
                      ['(-5 0 0)', '(-7 1 -2)', '(-5 0 1)'],
                      (1, 3),
                      step=0.75,
                      figsize=(10, 2.5),
                      fig_title=fig_title,
                      vmin=vmin,
                      vmax=vmax,
                      cmap=cmap)

    # Get rotation plots
    if proc_dict['lattice orientation']:
        rot1 = np.degrees(convert_ori_map(out1[2]))
        rot2 = np.degrees(convert_ori_map(out2[2]))
        drot = rot2 - rot1

        for i, (map_list, fig_title) in enumerate(zip([rot1, rot2, drot],
                                                      ['Lattice Orientation : Before Crack',
                                                       'Lattice Orientation : After Crack',
                                                       'Lattice Orientation : Difference'])):
            cmap, vmin, vmax = None, None, None
            if i == 2:
                cmap = 'Spectral_r'
                vmin = -0.5
                vmax = 0.5
            
            plot_maps([map_list[..., 0], map_list[..., 1], map_list[..., 2]],
                      ['w_x', 'w_y', 'w_z'],
                      (1, 3),
                      step=0.75,
                      figsize=(10, 2.5),
                      fig_title=fig_title,
                      vmin=vmin,
                      vmax=vmax,
                      cmap=cmap)

    # Get sample strain plots
    if proc_dict['sample strain']:
        es1 = out1[-1]
        es2 = out2[-1]
        des = es2 - es1

        for i, (map_list, fig_title) in enumerate(zip([es1, es2, des],
                                                      ['Sample Strain : Before Crack',
                                                       'Sample Strain : After Crack',
                                                       'Sample Strain : Difference'])):
            cmap = 'Spectral_r'
            vmin = None
            vmax = None
            if i == 2:
                pass
            
            map_list *= 1e3

            plot_maps([map_list[..., 0, 0],
                       map_list[..., 1, 1],
                       map_list[..., 2, 2],
                       map_list[..., 0, 1], 
                       map_list[..., 0, 2],
                       map_list[..., 1, 2]],
                      ['e_xx', 'e_yy', 'e_zz', 'e_xy', 'e_xz', 'e_yz'],
                      (2, 3),
                      step=0.75,
                      figsize=(10, 5),
                      fig_title=fig_title,
                      vmin=vmin,
                      vmax=vmax,
                      cmap='Spectral_r')
    
