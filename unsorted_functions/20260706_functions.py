import numpy as np
import os
from skimage import io
from tqdm import tqdm



def pixel_spots(spots, indices):
    ps = spots[((spots['map_x'] == indices[1]) & (spots['map_y'] == indices[0]))]
    return ps


def get_rgb_map(spots_3D,
                phase,
                q_mag_target=0,
                q_mag_cutoff=0.1,
                int_cutoff=0):

    # Effective map shape
    map_shape = (np.max(spots_3D['map_y']) + 1,
                 np.max(spots_3D['map_x']) + 1)

    q_map = np.empty((*map_shape, 3))
    q_map[:] = np.nan
    int_map = np.empty(map_shape)
    int_map[:] = np.nan

    for index in tqdm(range(np.prod(map_shape))):

        indices = np.unravel_index(index, map_shape)
        pixel_df = pixel_spots(spots_3D, indices).sort_values(by='intensity', ascending=False)
        spot_mags = pixel_df['q_mag'].values
        spot_qs = pixel_df[['qx', 'qy', 'qz']].values
        spot_ints = pixel_df['intensity'].values

        int_mask = spot_ints > int_cutoff
        q_mask = np.abs(spot_mags - q_mag_target) < q_mag_cutoff
        mask = int_mask & q_mask
        
        if np.sum(mask) > 0:
            int_map[indices] = spot_ints[mask][0]
            q_map[indices] = spot_qs[mask][0]

    # Create rgb_map
    rgb_map = q_map.copy()
    for ch in [0, 1, 2]:
        rgb_map[..., ch] -= np.nanmin(rgb_map[..., ch])
        rgb_map[..., ch] /= np.nanmax(rgb_map[..., ch])
    rgb_map /= np.max(rgb_map, axis=-1, keepdims=True)
    
    return int_map, q_map, rgb_map



# def transform_qvectors(qvectors, w_chi=0, stage_theta=0, silicon_theta=0, degrees=True):
#     """
#     Transform q_vectors into a coordinate system defined by the wire axis, where:
#     - q_magnitude is the magnitude of the q-vector
#     - alpha is the angle between the q-vector and the wire axis
#     - phi is the angle of the q-vector in the plane perpendicular to the wire axis

#     Parameters
#     ----------
#     qvectors : array-like, shape (N, 3) or (N, 4)
#         The q-vectors to be transformed, given in lab coordinates.
#     w_chi : float
#         The angle between the positive x-axis and the projection of the wire axis onto the xy-plane (in degrees or radians).
#     stage_theta : float
#         The angle of the stage rotation (in degrees or radians).
#     silicon_theta : float, optional
#         The angle of the silicon rotation (in degrees or radians).
#     degrees : bool, optional
#         If True, the input angles are in degrees and the output angles will also be in degrees.

#     Returns   
#     -------
#     q_magnitudes : array, shape (N,)
#         The magnitudes of the q-vectors.
#     alpha : array, shape (N,)
#         The angles between the q-vectors and the wire axis.
#     phi : array, shape (N,)
#         The angles of the q-vectors in the plane perpendicular to the wire axis.
#     """

#     # Convert angles to radians if in degrees
#     if degrees is True:
#         w_chi = np.radians(w_chi)
#         stage_theta = np.radians(stage_theta)
#         silicon_theta = np.radians(silicon_theta)

#     # Calculate wire theta as sum of stage and silicon thetas
#     w_theta = stage_theta + silicon_theta

#     # Get unit vector of wire axis direction in lab coordinates
#     w_vector = np.array([
#         np.cos(w_chi) * np.cos(w_theta), # x-component
#         np.sin(w_chi), # y-component
#         np.cos(w_chi) * -np.sin(w_theta) # z-component
#     ])

#     # Calculate angle between q-vectors and wire axis
#     q_magnitudes = np.linalg.norm(qvectors, axis=1)
#     alpha = np.arccos(np.dot(qvectors, w_vector) / q_magnitudes)

#     q_parallel = q_magnitudes * (np.cos(alpha)[:, np.newaxis] * w_vector).T
#     q_parallel = q_parallel.T
#     q_perpendicular = qvectors - q_parallel

#     # Define orthogonal basis in plane perpendicular to wire axis
#     e0 = np.cross(w_vector, [0, 0, 1])
#     e90 = np.cross(w_vector, e0)

#     phi = np.arctan2(np.dot(q_perpendicular, e90), np.dot(q_perpendicular, e0))

#     if degrees is True:
#         alpha = np.degrees(alpha)
#         phi = np.degrees(phi)
    
#     return q_magnitudes, alpha, phi



def transform_qvectors(qvectors, w_chi=0, stage_theta=0, silicon_theta=0, degrees=True):

    # Convert angles to radians if in degrees
    if degrees is True:
        w_chi = np.radians(w_chi)
        stage_theta = np.radians(stage_theta)
        silicon_theta = np.radians(silicon_theta)

    # Calculate wire theta as sum of stage and silicon thetas
    w_theta = stage_theta + silicon_theta

    # Get unit vector of wire axis direction in lab coordinates
    w_vector = np.array([
        np.cos(w_chi) * np.cos(w_theta), # x-component
        np.sin(w_chi), # y-component
        np.cos(w_chi) * -np.sin(w_theta) # z-component
    ])

    # Calculate angle between q-vectors and wire axis
    q_magnitudes = np.linalg.norm(qvectors, axis=1)

    alpha = np.arccos(np.dot(qvectors, w_vector) / q_magnitudes)

    q_parallel = q_magnitudes * (np.cos(alpha)[:, np.newaxis] * w_vector).T
    q_parallel = q_parallel.T
    q_perpendicular = qvectors - q_parallel

    # Define orthogonal basis in plane perpendicular to wire axis
    e90 = np.cross(w_vector, [0, 0, 1])
    e0 = np.cross(w_vector, e90)

    phi = np.arctan2(np.dot(q_perpendicular, e90), np.dot(q_perpendicular, e0))

    if degrees is True:
        alpha = np.degrees(alpha)
        phi = np.degrees(phi)
    
    return q_magnitudes, alpha, phi


def get_tilt_maps(spots_3D,
                  phase,
                  q_mag_target=0,
                  q_mag_cutoff=0.1,
                  int_cutoff=0,
                  **kwargs):

    # Effective map shape
    map_shape = (np.max(spots_3D['map_y']) + 1,
                 np.max(spots_3D['map_x']) + 1)

    q_mag_map = np.empty(map_shape)
    q_mag_map[:] = np.nan
    alpha_map = q_mag_map.copy()
    phi_map = q_mag_map.copy()

    for index in tqdm(range(np.prod(map_shape))):

        indices = np.unravel_index(index, map_shape)
        pixel_df = pixel_spots(spots_3D, indices).sort_values(by='intensity', ascending=False)
        spot_mags = pixel_df['q_mag'].values
        spot_qs = pixel_df[['qx', 'qy', 'qz']].values
        spot_ints = pixel_df['intensity'].values

        int_mask = spot_ints > int_cutoff
        q_mask = np.abs(spot_mags - q_mag_target) < q_mag_cutoff
        mask = int_mask & q_mask
        
        if np.sum(mask) > 0:

            q_mags, alpha, phi = transform_qvectors(spot_qs[mask], **kwargs)

            alpha_map[indices] = alpha[0]
            phi_map[indices] = phi[0]
            q_mag_map[indices] = q_mags[0]
    
    return q_mag_map, alpha_map, phi_map



def get_2D_color(norm_x, norm_y):

    # c_bl = [0., 0., 1.]
    # c_br = [0., 1., 1.]
    # c_tl = [1., 0., 0.]
    # c_tr = [1., 1., 0.]

    c_bl = [0., 1., 1.]
    c_br = [0., 0., 1.]
    c_tl = [1., 1., 0.]
    c_tr = [1., 0., 0.]

    w_bl = (1.0 - norm_x) * (1.0 - norm_y)
    w_br = norm_x * (1.0 - norm_y)
    w_tl = (1.0 - norm_x) * norm_y
    w_tr = norm_x * norm_y

    # Calculate combined channels
    rgb = (w_bl[..., np.newaxis] * c_bl
           + w_br[..., np.newaxis] * c_br
           + w_tl[..., np.newaxis] * c_tl
           + w_tr[..., np.newaxis] * c_tr)

    return rgb


def plot_tilts(alpha_map, phi_map, xdms=None):

    norm_alpha = alpha_map.copy()
    norm_alpha -= np.nanmin(norm_alpha)
    norm_alpha /= np.nanmax(norm_alpha)

    norm_phi = phi_map.copy()
    norm_phi -= np.nanmin(norm_phi)
    norm_phi /= np.nanmax(norm_phi)

    rgb = get_2D_color(norm_alpha, norm_phi)

    fig, ax = plt.subplots(1, 2, figsize=(5, 5), dpi=200)
    
    # Plot map
    rgb_legend = get_2D_color(norm_alpha, norm_phi)
    ax[0].imshow(rgb_legend)
    ax[0].set_aspect('equal')

    # Plot scatter
    if xdms is None:
        angle_extent = [np.nanmin(alpha_map), np.nanmax(alpha_map),
                        np.nanmin(phi_map), np.nanmax(phi_map)]
    else:
        print('Finding measured tilt boundaries...')
        bounds = np.vstack([xdm.q_arr.reshape(-1, 3) for xdm in xdms])
        bound_q_mag, bound_alpha, bound_phi = transform_qvectors(bounds)
        mask = np.abs(bound_q_mag - phase.reflections['q'][1]) < 0.1
        bound_alpha = bound_alpha[mask]
        bound_phi = bound_phi[mask]
        angle_extent = [np.nanmin(bound_alpha), np.nanmax(bound_alpha),
                        np.nanmin(bound_phi), np.nanmax(bound_phi)]

    dx = angle_extent[1] - angle_extent[0]
    dy = angle_extent[3] - angle_extent[2]
    pixel_aspect = (dx / dy)

    ax[1].imshow(get_color_key(),
                 extent=angle_extent,
                 aspect=pixel_aspect,
                 interpolation='none')
    # ax[1].set_aspect('equal')
    ax[1].set_xlabel('Alpha [deg]')
    ax[1].set_ylabel('Phi [deg]')

    ax[1].scatter(alpha_map.ravel(), phi_map.ravel(), s=5, c='k')

    # Define edges
    if xdms is not None:
        edges = find_concave_outline(np.stack([bound_alpha, bound_phi]).T)
        edges = np.asarray([bound_alpha[edges[:, 0]], bound_phi[edges[:, 1]]]).T
        edges = create_ordered_points(edges)
        ax[1].plot(*edges.T, '--', lw=1, c='k', alpha=0.5)


    fig.show()



def plot_map(*args, **kwargs):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
    img = ax.imshow(*args, **kwargs)
    if args[0].squeeze().ndim != 3:
        fig.colorbar(img, ax=ax)
    ax.set_aspect('equal')
    fig.show()


def get_color_key():

    grid_res = 200
    legend_x, legend_y = np.meshgrid(np.linspace(0, 1, grid_res), np.linspace(0, 1, grid_res))
    legend_rgb = get_2D_color(legend_x, legend_y)

    for ch in [0, 1, 2]:
        legend_rgb[..., ch] -= np.min(legend_rgb[..., ch])
        legend_rgb[..., ch] /= np.max(legend_rgb[..., ch])
    legend_rgb /= np.max(legend_rgb, axis=-1, keepdims=True)

    return legend_rgb


def plot_legend(alpha, phi, s=1, c='k', **kwargs):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=200)

    # Plot scatter
    angle_extent = [np.nanmin(alpha), np.nanmax(alpha),
                    np.nanmin(phi), np.nanmax(phi)]
    dx = angle_extent[1] - angle_extent[0]
    dy = angle_extent[3] - angle_extent[2]
    pixel_aspect = (dx / dy)

    ax.imshow(get_color_key(),
                 extent=angle_extent,
                 aspect=pixel_aspect,
                 interpolation='none')
    # ax[1].set_aspect('equal')
    ax.set_xlabel('Alpha [deg]')
    ax.set_ylabel('Phi [deg]')

    ax.scatter(alpha.ravel(), phi.ravel(), s=s, c=c, **kwargs)

    fig.show()


def find_concave_outline(points, alpha=1):
    if len(points) < 4:
        return points

    # Delaunay Triangulation
    tri = Delaunay(points)
    triangles = points[tri.simplices]

    # Formula for side lengths of each triangle
    a = np.linalg.norm(triangles[:, 0] - triangles[:, 1], axis=1)
    b = np.linalg.norm(triangles[:, 1] - triangles[:, 2], axis=1)
    c = np.linalg.norm(triangles[:, 2] - triangles[:, 0], axis=1)

    s = (a + b + c) / 2.0
    area = np.sqrt(s * (s - a) * (s - b) * (s - c) + 1e-10)
    circumradius = (a * b * c) / (4.0 * area)

    # A smaller circumradius threshold creates a tighter fit
    radius_threshold = 1.0 / alpha
    valid_triangles = tri.simplices[circumradius < radius_threshold]

    edges = set()
    for simplex in valid_triangles:
        # Check all 3 edges of the triangle
        for i in range(3):
            edge = tuple(sorted((simplex[i], simplex[(i + 1) % 3])))
            if edge in edges:
                edges.remove(edge)  # Shared edge (internal), remove it
            else:
                edges.add(edge)  # Unique edge (boundary), keep it

    return np.asarray(list(edges))



def create_ordered_points(points):

    cutoff = 2

    ordered_points = [points[-1]]
    points = np.asarray(points[:-1])

    mask = [True,] * len(points)
    for _ in tqdm(range(len(points))):

        last_point = ordered_points[-1]
        distances = np.sqrt(np.sum((points[mask] - last_point)**2, axis=-1))

        if distances.min() < cutoff:
            min_ind = np.argmin(distances)
            ordered_points.append(points[mask][min_ind])
            mask[np.nonzero(mask)[0][min_ind]] = False
        else:
            # print('Breaking iterations!')
            break
    
    return np.asarray(ordered_points)


    
