import matplotlib.pyplot as plt
import numpy as np
from plotly import graph_objects as go





#########################################
### 3D Reciprocal Space Visualization ###
#########################################


def plot_3D_scatter(q_vectors,
                    intensity,
                    skip=None,
                    edges=None,
                    **kwargs
                    ):

    # Check matching inputs
    if len(q_vectors) != len(intensity):
        err_str = (f'Length of q_vectors ({len(q_vectors)}) does not '
                   + 'match length of intensity ({len(intensity)}).')
        raise ValueError(err_str)

    q_vectors = np.asarray(q_vectors)
    intensity = np.asarray(intensity)

    # Check q_vectors shape
    if (q_vectors.ndim != 2 
        or q_vectors.shape[1] != 3):
        err_str = ('q_vectors must have shape (n, 3) not '
                   + f'({q_vector.shape}).')
        raise ValueError(err_str)

    if edges is None:
        edges = []
    
    if skip is None:
        skip = np.round(len(q_vectors) / 5000, 0).astype(int) # skips to about 5000 points
        if skip == 0:
            skip = 1
    
    kwargs.setdefault('s', 1)
    kwargs.setdefault('cmap', 'viridis')
    kwargs.setdefault('alpha', 0.1)

    fig, ax = plt.subplots(1, 1, 
                           figsize=(5, 5),
                           dpi=200,
                           subplot_kw={'projection':'3d'})
    
    if 'title' in kwargs:
        title = kwargs.pop('title')
        ax.set_title(title)

    ax.scatter(*q_vectors[::skip].T, c=intensity[::skip], **kwargs)

    for edge in edges:
        ax.plot(*edge.T, c='gray', lw=1)

    ax.set_xlabel('qx [Å⁻¹]')
    ax.set_ylabel('qy [Å⁻¹]')
    ax.set_zlabel('qz [Å⁻¹]')
    ax.set_aspect('equal')

    return fig, ax      


def plot_3D_isosurfaces(q_vectors,
                        intensity,
                        gridstep=0.01,
                        isomin=None,
                        isomax=None,
                        min_offset=None,
                        max_offset=None,
                        opacity=0.1,
                        surface_count=20,
                        colorscale='viridis',
                        renderer='browser'):

    # Check matching inputs
    if len(q_vectors) != len(intensity):
        err_str = (f'Length of q_vectors ({len(q_vectors)}) does not '
                   + 'match length of intensity ({len(intensity)}).')
        raise ValueError(err_str)

    q_vectors = np.asarray(q_vectors)
    intensity = np.asarray(intensity)

    # Check q_vectors shape
    if (q_vectors.ndim != 2 
        or q_vectors.shape[1] != 3):
        err_str = ('q_vectors must have shape (n, 3) not '
                   + f'({q_vector.shape}).')
        raise ValueError(err_str)

    # Check given q_ext
    for axis in range(3):
        q_ext = (np.max(q_vectors[:, axis])
                - np.min(q_vectors[:, axis]))
        if  q_ext < gridstep:
            err_str = (f'Gridstep ({gridstep}) is smaller than '
                    + f'q-vectors range along axis {axis} '
                    + f'({q_ext:.4f}).')
            raise ValueError(err_str)

    # Interpolate data for isosurface generation
    (x_grid,
     y_grid,
     z_grid,
     int_grid) = map_2_grid(q_vectors,
                            intensity,
                            gridstep=gridstep)

    gen_offset = ((np.max(int_grid) - np.min(int_grid))
                    / (2 * surface_count))
    if isomin is None:
        if min_offset is None:
            isomin = np.min(int_grid) + gen_offset
        else:
            isomin = np.min(int_grid) + min_offset
    
    if isomax is None:
        if max_offset is None:
            isomax = np.max(int_grid) - gen_offset
        else:
            isomax = np.max(int_grid) - max_offset
    
    # Generate isosurfaces from plotly graph object
    data = go.Volume(
        x=x_grid.flatten(),
        y=y_grid.flatten(),
        z=z_grid.flatten(),
        value=int_grid.flatten(),
        isomin=isomin,
        isomax=isomax,
        opacity=opacity,
        surface_count=surface_count,
        colorscale=colorscale
    )

    # Find data extent
    x_range = np.max(x_grid) - np.min(x_grid)
    y_range = np.max(y_grid) - np.min(y_grid)
    z_range = np.max(z_grid) - np.min(z_grid)

    # Generate figure and plot
    fig = go.Figure(data=data)
    fig.update_layout(scene_aspectmode='manual',
                        scene_aspectratio=dict(
                        x=x_range,
                        y=y_range,
                        z=z_range))
    fig.show(renderer=renderer)
