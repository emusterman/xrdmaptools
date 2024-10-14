# Submodule for processing and plotting 3D reciprocal space mapping.
# Strain analysis will be reserved for strain.py in this same submodule.
# Some 3D spot and blob search functions may be found here or strain.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


# Function for mapping vectorized data onto regular grid
def map_2_grid(q_vectors,
               intensity,
               gridstep=0.005):
    
    qx = q_vectors[:, 0]
    qy = q_vectors[:, 1]
    qz = q_vectors[:, 2]

    # Find bounds
    x_min = np.min(qx)
    x_max = np.max(qx)
    y_min = np.min(qy)
    y_max = np.max(qy)
    z_min = np.min(qz)
    z_max = np.max(qz)

    # Generate q-space grid
    xx = np.linspace(x_min, x_max, int((x_max - x_min) / gridstep))
    yy = np.linspace(y_min, y_max, int((y_max - y_min) / gridstep))
    zz = np.linspace(z_min, z_max, int((z_max - z_min) / gridstep))

    grid = np.array(np.meshgrid(xx, yy, zz, indexing='ij'))
    grid = grid.reshape(3, -1).T

    points = np.array([qx, qy, qz]).T

    int_grid = griddata(points, intensity, grid, method='nearest')
    #int_grid = int_grid.reshape(yy.shape[0], xx.shape[0], zz.shape[0]).T
    int_grid = int_grid.reshape(xx.shape[0], yy.shape[0], zz.shape[0])

    return *np.meshgrid(xx, yy, zz, indexing='ij'), int_grid