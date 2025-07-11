import numpy as np
import os
from tqdm import tqdm
from skimage import io
import h5py
import time as ttime
from matplotlib import colors
import matplotlib.pyplot as plt

from scipy.spatial import KDTree

from xrdmaptools.crystal.rsm import map_2_grid


def decorate_vectors(vectors, gridstep=0.01):

    min_int = np.min(vectors[:, -1])

    qx = vectors[:, 0]
    qy = vectors[:, 1]
    qz = vectors[:, 2]

    # Find bounds
    x_min = np.min(qx) - gridstep
    x_max = np.max(qx) + gridstep
    y_min = np.min(qy) - gridstep
    y_max = np.max(qy) + gridstep
    z_min = np.min(qz) - gridstep
    z_max = np.max(qz) + gridstep

    # Generate q-space grid
    xx = np.linspace(x_min, x_max, int((x_max - x_min) / gridstep))
    yy = np.linspace(y_min, y_max, int((y_max - y_min) / gridstep))
    zz = np.linspace(z_min, z_max, int((z_max - z_min) / gridstep))

    grid = np.array(np.meshgrid(xx, yy, zz, indexing='ij'))
    grid = grid.reshape(3, -1).T

    kdtree = KDTree(vectors[:, :3])

    nns = kdtree.query_ball_point(grid, gridstep)

    blank_vectors = []
    for point, nn in zip(grid, nns):
        if len(nn) == 0:
            blank_vectors.append([*point, min_int])

    return np.vstack([vectors, np.asarray(blank_vectors)])

from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def plot_single_isosurface(vectors, gridstep=0.01):

    # Interpolate data for isosurface generation
    (x_grid,
     y_grid,
     z_grid,
     int_grid) = map_2_grid(vectors[:, :3],
                            vectors[:, -1],
                            gridstep=gridstep)

    verts, faces, _, _ = measure.marching_cubes(int_grid, 0, spacing=(gridstep, gridstep, gridstep))

    return verts, faces

    fig = plt.figure(figsize=(5, 5), dpi=200)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2], lw=0)

    ax.set_xlabel('qx [Å⁻¹]')
    ax.set_ylabel('qy [Å⁻¹]', labelpad=10)
    ax.set_zlabel('qz [Å⁻¹]')
    ax.set_aspect('equal')
    
    fig.show()
