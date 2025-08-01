import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import orix

# Local imports
from xrdmaptools.plot import config
from xrdmaptools.geometry.geometry import (
    _parse_rotation_input
)
from xrdmaptools.reflections.spot_blob_indexing_3D import (
    _get_connection_indices
)

# This module is intended for plotting/visualizing orientation information
# This will likely make heavy use of the orix module

def plot_3D_indexing(indexings,
                     all_spot_qs,
                     all_ref_qs,
                     all_ref_hkls,
                     all_ref_fs,
                     qmask,
                     edges=[],
                     colors=None,
                     return_plot=False):
    
    all_spot_qs = np.asarray(all_spot_qs)
    all_ref_qs = np.asarray(all_ref_qs)
    all_ref_hkls = np.asarray(all_ref_hkls)
    all_ref_fs = np.asarray(all_ref_fs)

    if colors is not None:
        c = colors
    else:
        c = ['k',] * len(indexings)

    fig, ax = plt.subplots(1, 1, 
                           figsize=config.figsize,
                           dpi=config.dpi,
                           subplot_kw={'projection':'3d'})

    # Iterate and plot all connections
    for i, indexing in enumerate(indexings):
        spot_inds, ref_inds = indexing.T
        # spot_inds, ref_inds = _get_connection_indices(
        #                         indexing)
        orientation, _ = Rotation.align_vectors(
                            all_spot_qs[spot_inds],
                            all_ref_qs[ref_inds])
        all_rot_qs = orientation.apply(all_ref_qs)
        temp_q_mask = qmask.generate(all_rot_qs)

        ax.scatter(*all_rot_qs[temp_q_mask].T,
                   s=all_ref_fs[temp_q_mask] * 0.1,
                   c=c[i])
        ax.scatter(*all_spot_qs.T,
                   s=1,
                   c='r')

        for spot, hkl in zip(all_spot_qs[spot_inds], all_ref_hkls[ref_inds]):
            ax.text(*spot.T, str(tuple(hkl)), fontsize=8, c=c[i])
    
    # Plot bounding edges if they are given
    for edge in edges:
        ax.plot(*edge.T, c='gray', lw=1)
    
    ax.set_xlabel('qx [Å⁻¹]')
    ax.set_ylabel('qy [Å⁻¹]')
    ax.set_zlabel('qz [Å⁻¹]')
    ax.set_aspect('equal')

    if return_plot:
        return fig, ax
    else:
        fig.show()   


def plot_unit_cell(phase,
                   orientation=None,
                   degrees=False):
    # Re-write of phase.show_unitcell
    if orientation is None:
        orientation = np.eye(3)

    ori = _parse_rotation_input(
                    orientation,
                    'orientation',
                    degrees=degrees)

    # Sample axes
    s_axes = np.asarray([
        ((0, 0, 0), (1.0, 0, 0)), # x-axis
        ((0, 0, 0), (0, 1.0, 0)), # y-axis
        ((0, 0, 0), (0, 0, 1.0))  # z-axis
    ])
    s_axes *= 1.15 * np.max([phase.a, phase.b, phase.c])
    
    # Unit cell edges
    # a1, a2, a3 = LatticeParameters.from_Phase(phase).Amat.T
    a1, a2, a3 = phase.a1, phase.a2, phase.a3
    # print(a1, a2, a3)

    # [First point (x, y, z), second point]
    edges = np.asarray([
        ((0, 0, 0), a1), # basis vectors
        ((0, 0, 0), a2), # basis vectors
        ((0, 0, 0), a3), # basis vectors
        (a1, a1 + a2),
        (a1, a1 + a3),
        (a2, a2 + a1),
        (a2, a2 + a3),
        (a3, a3 + a1),
        (a3, a3 + a2),
        (a1 + a2, a1 + a2 + a3),
        (a1 + a3, a1 + a2 + a3),
        (a2 + a3, a1 + a2 + a3)
    ])

    edges[:, 0] = ori.apply(edges[:, 0], inverse=True)
    edges[:, 1] = ori.apply(edges[:, 1], inverse=True)

    fig, ax = plt.subplots(subplot_kw={'projection':'3d'})

    for line, label in zip(s_axes, ['x', 'y', 'z']):
        ax.quiver(*line.ravel(), color='k', lw=2)
        ax.text(*line[1] * 1.05, f'{label}-axis', zdir=tuple(line[1]), c='k', fontsize=12)

    for line, label, c in zip(edges[:3],
                                  ['a', 'b', 'c'],
                                  [[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
        ax.quiver(*line.ravel(), color=c, lw=2.5)
        ax.text(*line[1] * 1.05, f'{label}-axis', zdir=tuple(line[1]), c=c, fontsize=12)

    for line in edges[3:]:
        ax.plot(*line.T, color='gray', lw=1)

    ax.set_xlabel('x [Å]')
    ax.set_ylabel('y [Å]')
    ax.set_zlabel('z [Å]')
    ax.set_aspect('equal')

    fig.show()

