import numpy as np
import os
from skimage import io
from tqdm import tqdm



# Goal: Two new interactive plotting functions
# 1 - Allow for integrating 2D or 1D data within a user-defined box, and then plotting the integrated intensity.
# 2 - Allow for integrating a map from a sub-section user-defined box in 3D reciprocal space.

from matplotlib.widgets import Slider, Button, SpanSelector, RectangleSelector

from matplotlib.widgets import Slider, Button

def interactive_masked_3D_plot(dyn_kw={},
                               map_kw={},
                               aggregation='sum',
                               cmap='viridis',
                               marker_color='red'):
    """
    Interactive 3D scatter plot with masking controls.
    
    Parameters
    ----------
    dyn_kw : dict
        Must contain 'data': 2D array of lists, each containing [x, y, z, intensity]
        Optional keys: 'x_label', 'y_label', 'z_label', 'x_min', 'x_max', etc.
    map_kw : dict
        Map configuration (x_ticks, y_ticks, x_label, y_label, etc.)
    aggregation : str
        'sum' or 'max' - how to aggregate masked intensities across the map
    cmap : str
        Colormap for scatter plot
    marker_color : str
        Color for map marker
    """
    
    # Validate input
    if 'data' not in dyn_kw:
        raise ValueError("dyn_kw must contain 'data' key")
    
    dyn_kw = _fill_kwargs(
        dyn_kw,
        ['data', 'x_label', 'y_label', 'z_label',
         'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max',
         'edges', 'skip', 'int_cutoff']
    )
    
    # State variables for masking
    mask_state = {
        'offset': 0.0,
        'width': 1.0,
        'direction': 'z',  # 'x', 'y', or 'z'
        'active_directions': {'x': False, 'y': False, 'z': True}
    }
    
    # Create figure with space for sliders and buttons
    fig = plt.figure(figsize=(16, 8), dpi=config.dpi)
    
    # Create GridSpec for better layout control
    gs = fig.add_gridspec(3, 2, left=0.1, right=0.85, top=0.9, bottom=0.25,
                          wspace=0.3, hspace=0.4)
    
    # Main plots
    ax_map = fig.add_subplot(gs[0:2, 0])
    ax_3d = fig.add_subplot(gs[0:2, 1], projection='3d')
    
    dyn_kw['axes'] = [ax_map, ax_3d]
    
    # Initial map plot
    _update_map(dyn_kw['data'],
                map_kw=map_kw,
                axis=ax_map,
                cmap=cmap)
    
    # Set globals
    _set_globals(ax_map, map_kw)
    
    # Compute initial masked intensity map
    def compute_masked_map():
        """Calculate map based on current mask settings"""
        masked_map = np.zeros(dyn_kw['data'].shape[:2])
        
        for i in range(dyn_kw['data'].shape[0]):
            for j in range(dyn_kw['data'].shape[1]):
                vectors = dyn_kw['data'][i, j]
                if len(vectors) == 0:
                    continue
                
                vectors = np.array(vectors)
                
                # Apply direction-based masking
                dir_index = {'x': 0, 'y': 1, 'z': 2}[mask_state['direction']]
                coord = vectors[:, dir_index]
                
                # Create mask based on offset and width
                mask = (coord >= mask_state['offset']) & \
                       (coord < mask_state['offset'] + mask_state['width'])
                
                masked_intensities = vectors[mask, -1]
                
                if len(masked_intensities) > 0:
                    if aggregation == 'sum':
                        masked_map[i, j] = np.sum(masked_intensities)
                    elif aggregation == 'max':
                        masked_map[i, j] = np.max(masked_intensities)
        
        return masked_map
    
    # Update function for scatter plot with mask visualization
    def update_scatter():
        """Update 3D scatter with mask visualization"""
        global row, col
        
        plot_scatter = np.array(dyn_kw['data'][row, col])
        
        if len(plot_scatter) == 0:
            for child in ax_3d.get_children():
                if isinstance(child, Path3DCollection):
                    child.remove()
            return
        
        # Filter by intensity cutoff
        if dyn_kw['int_cutoff'] is None:
            dyn_kw['int_cutoff'] = 0
        int_mask = plot_scatter[:, -1] > dyn_kw['int_cutoff']
        plot_scatter = plot_scatter[int_mask]
        
        if len(plot_scatter) == 0:
            for child in ax_3d.get_children():
                if isinstance(child, Path3DCollection):
                    child.remove()
            return
        
        # Apply skip for point density
        if dyn_kw['skip'] is None:
            skip = max(1, int(np.round(len(plot_scatter) / 5000)))
        else:
            skip = dyn_kw['skip']
        
        # Create mask for highlighted points
        dir_index = {'x': 0, 'y': 1, 'z': 2}[mask_state['direction']]
        coord = plot_scatter[:, dir_index]
        highlight_mask = (coord >= mask_state['offset']) & \
                        (coord < mask_state['offset'] + mask_state['width'])
        
        # Clear previous scatter
        for child in ax_3d.get_children():
            if isinstance(child, Path3DCollection):
                child.remove()
        
        # Plot all points
        colors = np.array(['red' if highlight_mask[i] else 'blue' 
                          for i in range(len(plot_scatter))])
        
        ax_3d.scatter(*plot_scatter[::skip, :3].T,
                     c=colors[::skip],
                     alpha=0.6,
                     s=30)
        
        # Plot edges if provided
        if dyn_kw['edges'] is not None:
            for edge in dyn_kw['edges']:
                ax_3d.plot(*edge.T, c='gray', lw=1, alpha=0.3)
        
        # Set labels and limits
        if dyn_kw['x_label'] is None:
            dyn_kw['x_label'] = 'x'
        if dyn_kw['y_label'] is None:
            dyn_kw['y_label'] = 'y'
        if dyn_kw['z_label'] is None:
            dyn_kw['z_label'] = 'z'
        
        ax_3d.set_xlabel(dyn_kw['x_label'])
        ax_3d.set_ylabel(dyn_kw['y_label'])
        ax_3d.set_zlabel(dyn_kw['z_label'])
        ax_3d.set_aspect('equal')
    
    # Slider callback
    def on_offset_change(val):
        mask_state['offset'] = val
        masked_map = compute_masked_map()
        map_kw['map'] = masked_map
        
        if ax_map.get_images():
            ax_map.get_images()[0].set_data(masked_map)
            ax_map.figure.canvas.draw_idle()
        
        update_scatter()
        fig.canvas.draw_idle()
    
    def on_width_change(val):
        mask_state['width'] = val
        masked_map = compute_masked_map()
        map_kw['map'] = masked_map
        
        if ax_map.get_images():
            ax_map.get_images()[0].set_data(masked_map)
            ax_map.figure.canvas.draw_idle()
        
        update_scatter()
        fig.canvas.draw_idle()
    
    # Direction button callbacks
    def on_direction_button(direction_char):
        def callback(event):
            mask_state['direction'] = direction_char
            masked_map = compute_masked_map()
            map_kw['map'] = masked_map
            
            if ax_map.get_images():
                ax_map.get_images()[0].set_data(masked_map)
            
            update_scatter()
            fig.canvas.draw_idle()
        return callback
    
    # Create sliders
    ax_offset = fig.add_axes([0.15, 0.15, 0.3, 0.03])
    ax_width = fig.add_axes([0.15, 0.10, 0.3, 0.03])
    
    slider_offset = Slider(ax_offset, 'Offset', -3, 3, valinit=0, valstep=0.0001)
    slider_width = Slider(ax_width, 'Width', 0.0001, 0.01, valinit=1, valstep=0.0001)
    
    slider_offset.on_changed(on_offset_change)
    slider_width.on_changed(on_width_change)
    
    # Create direction buttons
    button_positions = {
        'X': [0.15, 0.05],
        'Y': [0.35, 0.05],
        'Z': [0.55, 0.05]
    }
    
    buttons = {}
    for direction, pos in button_positions.items():
        ax_btn = fig.add_axes(pos + [0.08, 0.03])
        btn = Button(ax_btn, f'{direction} Direction')
        btn.on_clicked(on_direction_button(direction.lower()))
        buttons[direction] = btn
    
    # Initial scatter plot
    update_scatter()
    
    # Map interaction
    def update_map_axes(event):
        global row, col
        if _update_coordinates(event, map_kw):
            update_scatter()
            ax_3d.set_title(f'Row = {row}, Col = {col}\ny = {map_y:.2f}, x = {map_x:.2f}')
            _update_marker(ax_map, marker_color)
            fig.canvas.draw_idle()
    
    return fig, [ax_map, ax_3d], {'offset': slider_offset, 'width': slider_width, 'buttons':buttons}




def get_max_vectors_fast(vector_map, q_target, cutoff=0.25):
    all_cells = vector_map.ravel()
    valid_cells = [cell for cell in all_cells if cell is not None and len(cell) > 0]
    
    if not valid_cells:
        return [], []
        
    all_data = np.vstack(valid_cells)
    
    q_vectors = all_data[:, :3]
    intensity = all_data[:, -1]

    diff = np.abs(q_vectors - q_target)
    mask = np.all(diff < cutoff, axis=1)
    
    filtered_vectors = q_vectors[mask]
    filtered_intensities = intensity[mask]
    
    if len(filtered_vectors) == 0:
        return [], []

    sort_idx = np.argsort(filtered_intensities)
    sorted_vectors = filtered_vectors[sort_idx]
    sorted_intensities = filtered_intensities[sort_idx]

    max_map = {tuple(vec): inten for vec, inten in zip(sorted_vectors, sorted_intensities)}
    
    return list(max_map.keys()), list(max_map.values())



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Path3DCollection
from matplotlib.colors import Normalize

def interactive_masked_3D_plot(dyn_kw={},
                               map_kw={},
                               aggregation='sum',
                               cmap='viridis',
                               marker_color='red'):
    """
    Interactive 3D scatter plot with masking controls.
    
    Static scatter plot of all vectors from the map, with mask visualization as lines.
    
    Parameters
    ----------
    dyn_kw : dict
        Must contain 'data': 2D array of lists, each containing [x, y, z, intensity]
        Optional keys: 'x_label', 'y_label', 'z_label', 'x_min', 'x_max', etc.
    map_kw : dict
        Map configuration (x_ticks, y_ticks, x_label, y_label, etc.)
    aggregation : str
        'sum' or 'max' - how to aggregate masked intensities across the map
    cmap : str
        Colormap for scatter plot
    marker_color : str
        Color for map marker
    """
    
    # Validate input
    if 'data' not in dyn_kw:
        raise ValueError("dyn_kw must contain 'data' key")
    
    # Helper function to fill kwargs
    def _fill_kwargs(kwargs, keys):
        for key in keys:
            if key not in tuple(kwargs.keys()):
                kwargs[key] = None
        return kwargs
    
    dyn_kw = _fill_kwargs(
        dyn_kw,
        ['data', 'x_label', 'y_label', 'z_label',
         'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max',
         'edges', 'skip', 'int_cutoff']
    )
    
    # State variables for masking
    mask_state = {
        'offset': 0.0,
        'width': 0.01,
        'direction': 'z',  # 'x', 'y', or 'z'
    }
    
    # Extract all vectors from the map
    def extract_all_vectors():
        """Extract all vectors and intensities from the entire vector map"""
        all_vectors = []
        all_intensities = []
        
        for i in range(dyn_kw['data'].shape[0]):
            for j in range(dyn_kw['data'].shape[1]):
                vectors = dyn_kw['data'][i, j]
                if vectors is not None and len(vectors) > 0:
                    vectors_arr = np.array(vectors)
                    all_vectors.append(vectors_arr[:, :3])  # x, y, z
                    all_intensities.append(vectors_arr[:, -1])  # intensity
        
        if len(all_vectors) == 0:
            return np.array([]), np.array([])
        
        all_vectors = np.vstack(all_vectors)
        all_intensities = np.hstack(all_intensities)
        
        return all_vectors, all_intensities
    
    # Compute masked intensity map
    def compute_masked_map():
        """Calculate map based on current mask settings"""
        masked_map = np.zeros(dyn_kw['data'].shape[:2])
        
        for i in range(dyn_kw['data'].shape[0]):
            for j in range(dyn_kw['data'].shape[1]):
                vectors = dyn_kw['data'][i, j]
                if vectors is None or len(vectors) == 0:
                    continue
                
                vectors_arr = np.array(vectors)
                
                # Apply direction-based masking
                dir_index = {'x': 0, 'y': 1, 'z': 2}[mask_state['direction']]
                coord = vectors_arr[:, dir_index]
                
                # Create mask based on offset and width
                mask = (coord >= mask_state['offset']) & \
                       (coord < mask_state['offset'] + mask_state['width'])
                
                masked_intensities = vectors_arr[mask, -1]
                
                if len(masked_intensities) > 0:
                    if aggregation == 'sum':
                        masked_map[i, j] = np.sum(masked_intensities)
                    elif aggregation == 'max':
                        masked_map[i, j] = np.max(masked_intensities)
        
        return masked_map
    
    # Extract all vectors for static plot
    all_vectors, all_intensities = extract_all_vectors()
    
    if len(all_vectors) == 0:
        raise ValueError("No valid vectors found in data")
    
    # Normalize intensities for colormap
    norm = Normalize(vmin=np.min(all_intensities), vmax=np.max(all_intensities))
    
    # Create figure with space for sliders and buttons
    fig = plt.figure(figsize=(16, 8), dpi=100)
    
    # Create GridSpec for better layout control
    gs = fig.add_gridspec(3, 2, left=0.1, right=0.85, top=0.9, bottom=0.25,
                          wspace=0.3, hspace=0.4)
    
    # Main plots
    ax_map = fig.add_subplot(gs[0:2, 0])
    ax_3d = fig.add_subplot(gs[0:2, 1], projection='3d')
    
    # Initial map plot
    initial_masked_map = compute_masked_map()
    im = ax_map.imshow(initial_masked_map, cmap='viridis', aspect='auto')
    ax_map.set_title('Masked Intensity Map')
    fig.colorbar(im, ax=ax_map)
    
    # Plot static 3D scatter
    scatter = ax_3d.scatter(*all_vectors.T, c=all_intensities, cmap=cmap, 
                           norm=norm, alpha=0.6, s=30)
    fig.colorbar(scatter, ax=ax_3d, label='Intensity')
    
    # Set default labels
    if dyn_kw['x_label'] is None:
        dyn_kw['x_label'] = 'qx'
    if dyn_kw['y_label'] is None:
        dyn_kw['y_label'] = 'qy'
    if dyn_kw['z_label'] is None:
        dyn_kw['z_label'] = 'qz'
    
    ax_3d.set_xlabel(dyn_kw['x_label'])
    ax_3d.set_ylabel(dyn_kw['y_label'])
    ax_3d.set_zlabel(dyn_kw['z_label'])
    ax_3d.set_aspect('equal')
    
    # Store mask lines reference
    mask_lines = []
    
    def draw_mask_region():
        """Draw the mask region as lines on the 3D plot"""
        nonlocal mask_lines
        
        # Clear previous lines
        for line in mask_lines:
            line.remove()
        mask_lines = []
        
        # Get axis limits
        x_min, x_max = ax_3d.get_xlim()
        y_min, y_max = ax_3d.get_ylim()
        z_min, z_max = ax_3d.get_zlim()
        
        offset = mask_state['offset']
        width = mask_state['width']
        direction = mask_state['direction']
        
        # Draw planes as rectangles for the mask region
        corners = np.array([
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
        ])
        
        if direction == 'z':
            # Draw two horizontal planes at offset and offset+width
            for z_val in [offset, offset + width]:
                plane_corners = corners.copy()
                plane_corners[:, 2] = z_val
                # Draw edges of plane
                edges = [
                    [plane_corners[0], plane_corners[1]],
                    [plane_corners[1], plane_corners[2]],
                    [plane_corners[2], plane_corners[3]],
                    [plane_corners[3], plane_corners[0]],
                ]
                for edge in edges:
                    line, = ax_3d.plot(*np.array(edge).T, 'r-', linewidth=2, alpha=0.7)
                    mask_lines.append(line)
        
        elif direction == 'x':
            # Draw two vertical planes at offset and offset+width
            for x_val in [offset, offset + width]:
                plane_corners = corners.copy()
                plane_corners[:, 0] = x_val
                edges = [
                    [plane_corners[0], plane_corners[1]],
                    [plane_corners[1], plane_corners[2]],
                    [plane_corners[2], plane_corners[3]],
                    [plane_corners[3], plane_corners[0]],
                ]
                for edge in edges:
                    line, = ax_3d.plot(*np.array(edge).T, 'r-', linewidth=2, alpha=0.7)
                    mask_lines.append(line)
        
        elif direction == 'y':
            # Draw two vertical planes at offset and offset+width
            for y_val in [offset, offset + width]:
                plane_corners = corners.copy()
                plane_corners[:, 1] = y_val
                edges = [
                    [plane_corners[0], plane_corners[1]],
                    [plane_corners[1], plane_corners[2]],
                    [plane_corners[2], plane_corners[3]],
                    [plane_corners[3], plane_corners[0]],
                ]
                for edge in edges:
                    line, = ax_3d.plot(*np.array(edge).T, 'r-', linewidth=2, alpha=0.7)
                    mask_lines.append(line)
    
    # Slider callback
    def on_offset_change(val):
        mask_state['offset'] = val
        masked_map = compute_masked_map()
        
        # Update map image
        ax_map.images[0].set_data(masked_map)
        
        # Redraw mask region
        draw_mask_region()
        fig.canvas.draw_idle()
    
    def on_width_change(val):
        mask_state['width'] = val
        masked_map = compute_masked_map()
        
        # Update map image
        ax_map.images[0].set_data(masked_map)
        
        # Redraw mask region
        draw_mask_region()
        fig.canvas.draw_idle()
    
    # Direction button callbacks
    def on_direction_button(direction_char):
        def callback(event):
            mask_state['direction'] = direction_char
            masked_map = compute_masked_map()
            
            # Update map image
            ax_map.images[0].set_data(masked_map)
            
            # Redraw mask region
            draw_mask_region()
            fig.canvas.draw_idle()
        return callback
    
    # Create sliders
    ax_offset = fig.add_axes([0.15, 0.15, 0.3, 0.03])
    ax_width = fig.add_axes([0.15, 0.10, 0.3, 0.03])
    
    # Determine slider ranges based on data
    coord_ranges = {
        'x': [np.min(all_vectors[:, 0]), np.max(all_vectors[:, 0])],
        'y': [np.min(all_vectors[:, 1]), np.max(all_vectors[:, 1])],
        'z': [np.min(all_vectors[:, 2]), np.max(all_vectors[:, 2])],
    }
    
    slider_offset = Slider(ax_offset, 'Offset', 
                          coord_ranges['z'][0], coord_ranges['z'][1], 
                          valinit=coord_ranges['z'][0], valstep=0.001)
    slider_width = Slider(ax_width, 'Width', 
                         0.001, coord_ranges['z'][1] - coord_ranges['z'][0], 
                         valinit=0.01, valstep=0.001)
    
    slider_offset.on_changed(on_offset_change)
    slider_width.on_changed(on_width_change)
    
    # Create direction buttons
    button_positions = {
        'X': [0.15, 0.05],
        'Y': [0.35, 0.05],
        'Z': [0.55, 0.05]
    }
    
    buttons = {}
    for direction, pos in button_positions.items():
        ax_btn = fig.add_axes(pos + [0.08, 0.03])
        btn = Button(ax_btn, f'{direction} Direction')
        btn.on_clicked(on_direction_button(direction.lower()))
        buttons[direction] = btn
    
    # Draw initial mask region
    draw_mask_region()
    
    return fig, [ax_map, ax_3d], {'offset': slider_offset, 'width': slider_width, 'buttons': buttons}




import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Path3DCollection
from matplotlib.colors import Normalize
import numpy as np

def interactive_masked_3D_plot(dyn_kw={},
                               map_kw={},
                               aggregation='sum',
                               cmap='viridis',
                               marker_color='red'):
    """
    Interactive 3D scatter plot with masking controls (OPTIMIZED).
    
    Static scatter plot of all vectors from the map, with mask visualization as lines.
    """
    
    if 'data' not in dyn_kw:
        raise ValueError("dyn_kw must contain 'data' key")
    
    def _fill_kwargs(kwargs, keys):
        for key in keys:
            if key not in tuple(kwargs.keys()):
                kwargs[key] = None
        return kwargs
    
    dyn_kw = _fill_kwargs(
        dyn_kw,
        ['data', 'x_label', 'y_label', 'z_label',
         'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max',
         'edges', 'skip', 'int_cutoff']
    )
    
    # PRE-COMPUTE all vectors once
    print("Pre-computing vectors...")
    all_vectors = []
    all_intensities = []
    vector_row_indices = []  # Track which row each vector came from
    vector_col_indices = []  # Track which col each vector came from
    
    for i in range(dyn_kw['data'].shape[0]):
        for j in range(dyn_kw['data'].shape[1]):
            vectors = dyn_kw['data'][i, j]
            if vectors is not None and len(vectors) > 0:
                vectors_arr = np.array(vectors)
                n_vecs = len(vectors_arr)
                all_vectors.append(vectors_arr[:, :3])
                all_intensities.append(vectors_arr[:, -1])
                vector_row_indices.extend([i] * n_vecs)
                vector_col_indices.extend([j] * n_vecs)
    
    if len(all_vectors) == 0:
        raise ValueError("No valid vectors found in data")
    
    # Stack into single arrays
    all_vectors = np.vstack(all_vectors)
    all_intensities = np.hstack(all_intensities)
    vector_row_indices = np.array(vector_row_indices)
    vector_col_indices = np.array(vector_col_indices)
    
    # Pre-compute coordinate ranges
    coord_ranges = {
        'x': [np.min(all_vectors[:, 0]), np.max(all_vectors[:, 0])],
        'y': [np.min(all_vectors[:, 1]), np.max(all_vectors[:, 1])],
        'z': [np.min(all_vectors[:, 2]), np.max(all_vectors[:, 2])],
    }
    
    # Normalize intensities for colormap
    norm = Normalize(vmin=np.min(all_intensities), vmax=np.max(all_intensities))
    
    # State variables
    mask_state = {
        'offset': coord_ranges['z'][0],
        'width': 0.01,
        'direction': 'z',
    }
    
    print("Creating figure...")
    
    # Create figure
    fig = plt.figure(figsize=(16, 8), dpi=100)
    gs = fig.add_gridspec(3, 2, left=0.1, right=0.85, top=0.9, bottom=0.25,
                          wspace=0.3, hspace=0.4)
    
    ax_map = fig.add_subplot(gs[0:2, 0])
    ax_3d = fig.add_subplot(gs[0:2, 1], projection='3d')
    
    # FAST vectorized compute_masked_map function
    def compute_masked_map_fast():
        """Compute masked map using vectorized operations"""
        dir_index = {'x': 0, 'y': 1, 'z': 2}[mask_state['direction']]
        coords = all_vectors[:, dir_index]
        
        # Create mask (vectorized)
        mask = (coords >= mask_state['offset']) & \
               (coords < mask_state['offset'] + mask_state['width'])
        
        masked_intensities = all_intensities[mask]
        masked_rows = vector_row_indices[mask]
        masked_cols = vector_col_indices[mask]
        
        # Create empty map
        masked_map = np.zeros(dyn_kw['data'].shape[:2])
        
        if aggregation == 'sum':
            # Use np.add.at for fast aggregation
            np.add.at(masked_map, (masked_rows, masked_cols), masked_intensities)
        elif aggregation == 'max':
            # Manual max aggregation
            for row, col, intensity in zip(masked_rows, masked_cols, masked_intensities):
                masked_map[row, col] = max(masked_map[row, col], intensity)
        
        return masked_map
    
    # Initial map plot
    print("Computing initial map...")
    initial_masked_map = compute_masked_map_fast()
    im = ax_map.imshow(initial_masked_map, cmap='viridis', aspect='auto')
    ax_map.set_title('Masked Intensity Map')
    fig.colorbar(im, ax=ax_map)
    
    # Plot static 3D scatter (once!)
    print("Plotting scatter...")
    scatter = ax_3d.scatter(*all_vectors.T, c=all_intensities, cmap=cmap, 
                           norm=norm, alpha=0.6, s=30)
    fig.colorbar(scatter, ax=ax_3d, label='Intensity')
    
    # Set labels
    if dyn_kw['x_label'] is None:
        dyn_kw['x_label'] = 'qx'
    if dyn_kw['y_label'] is None:
        dyn_kw['y_label'] = 'qy'
    if dyn_kw['z_label'] is None:
        dyn_kw['z_label'] = 'qz'
    
    ax_3d.set_xlabel(dyn_kw['x_label'])
    ax_3d.set_ylabel(dyn_kw['y_label'])
    ax_3d.set_zlabel(dyn_kw['z_label'])
    ax_3d.set_aspect('equal')
    
    # Mask lines
    mask_lines = []
    
    def draw_mask_region():
        """Draw the mask region as lines on the 3D plot"""
        nonlocal mask_lines
        
        # Clear previous lines
        for line in mask_lines:
            line.remove()
        mask_lines = []
        
        # Get axis limits
        x_min, x_max = ax_3d.get_xlim()
        y_min, y_max = ax_3d.get_ylim()
        z_min, z_max = ax_3d.get_zlim()
        
        offset = mask_state['offset']
        width = mask_state['width']
        direction = mask_state['direction']
        
        corners = np.array([
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
        ])
        
        if direction == 'z':
            for z_val in [offset, offset + width]:
                plane_corners = corners.copy()
                plane_corners[:, 2] = z_val
                edges = [
                    [plane_corners[0], plane_corners[1]],
                    [plane_corners[1], plane_corners[2]],
                    [plane_corners[2], plane_corners[3]],
                    [plane_corners[3], plane_corners[0]],
                ]
                for edge in edges:
                    line, = ax_3d.plot(*np.array(edge).T, 'r-', linewidth=2, alpha=0.7)
                    mask_lines.append(line)
        
        elif direction == 'x':
            for x_val in [offset, offset + width]:
                plane_corners = corners.copy()
                plane_corners[:, 0] = x_val
                edges = [
                    [plane_corners[0], plane_corners[1]],
                    [plane_corners[1], plane_corners[2]],
                    [plane_corners[2], plane_corners[3]],
                    [plane_corners[3], plane_corners[0]],
                ]
                for edge in edges:
                    line, = ax_3d.plot(*np.array(edge).T, 'r-', linewidth=2, alpha=0.7)
                    mask_lines.append(line)
        
        elif direction == 'y':
            for y_val in [offset, offset + width]:
                plane_corners = corners.copy()
                plane_corners[:, 1] = y_val
                edges = [
                    [plane_corners[0], plane_corners[1]],
                    [plane_corners[1], plane_corners[2]],
                    [plane_corners[2], plane_corners[3]],
                    [plane_corners[3], plane_corners[0]],
                ]
                for edge in edges:
                    line, = ax_3d.plot(*np.array(edge).T, 'r-', linewidth=2, alpha=0.7)
                    mask_lines.append(line)
    
    # Slider callbacks
    def on_offset_change(val):
        mask_state['offset'] = val
        masked_map = compute_masked_map_fast()
        ax_map.images[0].set_data(masked_map)
        draw_mask_region()
        fig.canvas.draw_idle()
    
    def on_width_change(val):
        mask_state['width'] = val
        masked_map = compute_masked_map_fast()
        ax_map.images[0].set_data(masked_map)
        draw_mask_region()
        fig.canvas.draw_idle()
    
    def on_direction_button(direction_char):
        def callback(event):
            mask_state['direction'] = direction_char
            # Update slider ranges
            new_min, new_max = coord_ranges[direction_char]
            slider_offset.set_val(new_min)
            slider_offset.valmin = new_min
            slider_offset.valmax = new_max
            slider_width.valmax = new_max - new_min
            
            masked_map = compute_masked_map_fast()
            ax_map.images[0].set_data(masked_map)
            draw_mask_region()
            fig.canvas.draw_idle()
        return callback
    
    # Create sliders
    ax_offset = fig.add_axes([0.15, 0.15, 0.3, 0.03])
    ax_width = fig.add_axes([0.15, 0.10, 0.3, 0.03])
    
    slider_offset = Slider(ax_offset, 'Offset', 
                          coord_ranges['z'][0], coord_ranges['z'][1], 
                          valinit=coord_ranges['z'][0], valstep=0.001)
    slider_width = Slider(ax_width, 'Width', 
                         0.001, coord_ranges['z'][1] - coord_ranges['z'][0], 
                         valinit=0.01, valstep=0.001)
    
    slider_offset.on_changed(on_offset_change)
    slider_width.on_changed(on_width_change)
    
    # Direction buttons
    button_positions = {
        'X': [0.15, 0.05],
        'Y': [0.35, 0.05],
        'Z': [0.55, 0.05]
    }
    
    buttons = {}
    for direction, pos in button_positions.items():
        ax_btn = fig.add_axes(pos + [0.08, 0.03])
        btn = Button(ax_btn, f'{direction} Direction')
        btn.on_clicked(on_direction_button(direction.lower()))
        buttons[direction] = btn
    
    # Draw initial mask region
    draw_mask_region()
    
    print("Done!")
    return fig, [ax_map, ax_3d], {'offset': slider_offset, 'width': slider_width, 'buttons': buttons}