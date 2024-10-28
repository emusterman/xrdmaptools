import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from xrdmaptools.reflections.spot_blob_indexing_3D import (
    _get_connection_indices
)




#########################################
### 3D Reciprocal Space Visualization ###
#########################################


def plot_point_cloud():
    raise NotImplementedError()


def plot_iso_surfaces():
    raise NotImplementedError()


def _interpolate_grid():
    raise NotImplementedError()


def plot_3D_indexing(connections,
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
        c = 'k'


    fig, ax = plt.subplots(1, 1, 
                            figsize=(5, 5),
                            dpi=200,
                            subplot_kw={'projection':'3d'})

    # Iterate and plot all connections
    for connection in connections:
        spot_inds, ref_inds = _get_connection_indices(
                                connection)
        orientation, _ = Rotation.align_vectors(
                            all_spot_qs[spot_inds],
                            all_ref_qs[ref_inds])
        all_rot_qs = orientation.apply(all_ref_qs)
        temp_q_mask = qmask.generate(all_rot_qs)

        ax.scatter(*all_rot_qs[temp_q_mask].T,
                   s=all_ref_fs[temp_q_mask] * 0.1,
                   c=c)
        ax.scatter(*all_spot_qs.T,
                   s=1,
                   c='r')

        for spot, hkl in zip(all_spot_qs[spot_inds], all_ref_hkls[ref_inds]):
            ax.text(*spot.T, str(tuple(hkl)), fontsize=8, c=c)
    
    # Plot bounding edges if they are given
    for edge in rsm.edges:
        ax.plot(*edge.T, c='gray', lw=1)
    
    ax.set_xlabel('qx [Å⁻¹]')
    ax.set_ylabel('qy [Å⁻¹]')
    ax.set_zlabel('qz [Å⁻¹]')
    ax.set_aspect('equal')

    if return_plot:
        return fig, ax
    else:
        fig.show()        
