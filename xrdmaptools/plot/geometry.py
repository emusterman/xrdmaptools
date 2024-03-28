import matplotlib.pyplot as plt
import numpy as np



def plot_q_space(self, pixel_indices=None, skip=500, return_plot=False):
    raise NotImplementedError

    q = get_q_vect(self.tth_arr, self.chi_arr, wavelength=self.wavelength)

    if pixel_indices is not None:
        pixel_df = self.spots[(self.spots['map_x'] == pixel_indices[0])
                                & (self.spots['map_y'] == pixel_indices[1])].copy()

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=200, subplot_kw={'projection':'3d'})

    # Plot sampled Ewald sphere
    q_mask = q[:, self.map.mask]
    ax.plot_trisurf(q_mask[0].ravel()[::skip],
                    q_mask[1].ravel()[::skip],
                    q_mask[2].ravel()[::skip],
                    alpha=0.5, label='detector')

    # Plot full Ewald sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    radius = 2 * np.pi / self.wavelength
    x =  radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z - radius, alpha=0.2, color='k', label='Ewald sphere')

    if pixel_indices is not None:
        ax.scatter(*pixel_df[['qx', 'qy', 'qz']].values.T, s=1, c='r', label='spots')

    # Sample geometry
    ax.quiver([0, 0], [0, 0], [-2 * radius, -radius], [0, 0], [0, 0], [radius, radius], colors='k')
    ax.scatter(0, 0, 0, marker='o', s=10, facecolors='none', edgecolors='k', label='transmission')
    ax.scatter(0, 0, -radius, marker='h', s=10, c='b', label='sample')

    ax.set_xlabel('qx [Å⁻¹]')
    ax.set_ylabel('qy [Å⁻¹]')
    ax.set_zlabel('qz [Å⁻¹]')
    ax.set_aspect('equal')

    # Initial view
    ax.view_init(elev=-45, azim=90, roll=0)

    if return_plot:
        return fig, ax

    plt.show()


def plot_detector_geometry(ai, skip=300, return_plot=False):
    raise NotImplementedError()

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=200, subplot_kw={'projection':'3d'})

    # Plot detector position
    pos_arr = self.ai.position_array()

    x = pos_arr[:, :, 2].ravel()[::skip] # Switch to positive outboard
    y = pos_arr[:, :, 1].ravel()[::skip]
    z = pos_arr[:, :, 0].ravel()[::skip]

    ax.plot_trisurf(x, y, z,
                    alpha=0.5, label='detector')

    # X-ray beam
    radius = self.ai.dist
    ax.quiver([0], [0], [-radius], [0], [0], [radius], colors='k')
    ax.scatter(0, 0, 0, marker='h', s=10, c='b', label='sample')

    # Detector
    corner_indices = np.array([[0, 0], [-1, 0], [0, -1], [-1, -1]]).T
    corn = pos_arr[*corner_indices].T
    ax.quiver([0,] * 4,
                [0,] * 4,
                [0,] * 4,
                corn[2],
                corn[1],
                corn[0], colors='gray', lw=0.5)

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.set_aspect('equal')

    # Initial view
    ax.view_init(elev=-60, azim=90, roll=0)

    if return_plot:
        return fig, ax

    plt.show()


def plot_recprocal_lattice(phase, tth_range):
    raise NotImplementedError()