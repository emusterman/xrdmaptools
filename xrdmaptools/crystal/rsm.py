# Submodule for processing and plotting 3D reciprocal space mapping.
# Strain analysis will be reserved for strain.py in this same submodule.
# Some 3D spot and blob search functions may be found here or strain.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


# Function for mapping vectorized data onto regular grid
def map_2_grid(q_dset, gridstep=0.005):
    
    all_qx = q_dset[0]
    all_qy = q_dset[1]
    all_qz = q_dset[2]
    all_int = q_dset[3]

    # Find bounds
    x_min = np.min(all_qx)
    x_max = np.max(all_qx)
    y_min = np.min(all_qy)
    y_max = np.max(all_qy)
    z_min = np.min(all_qz)
    z_max = np.max(all_qz)

    # Generate q-space grid
    xx = np.linspace(x_min, x_max, int((x_max - x_min) / gridstep))
    yy = np.linspace(y_min, y_max, int((y_max - y_min) / gridstep))
    zz = np.linspace(z_min, z_max, int((z_max - z_min) / gridstep))

    grid = np.array(np.meshgrid(xx, yy, zz, indexing='ij'))
    grid = grid.reshape(3, -1).T

    points = np.array([all_qx, all_qy, all_qz]).T

    int_grid = griddata(points, all_int, grid, method='nearest')
    #int_grid = int_grid.reshape(yy.shape[0], xx.shape[0], zz.shape[0]).T
    int_grid = int_grid.reshape(xx.shape[0], yy.shape[0], zz.shape[0])

    return np.array([*np.meshgrid(xx, yy, zz, indexing='ij'), int_grid])