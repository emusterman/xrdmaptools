import numpy as np

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.widgets import EllipseSelector, RectangleSelector

from xrdmaptools.plot.interactive import _check_missing_key, _figsize, _dpi, _update_map, _set_globals, _update_axes

def static_window_stats_2D_plot(dyn_kw={},
                                map_kw={},
                                cmap='viridis',
                                marker_color='red'):
    '''
    
    '''

    # Check axes range
    if _check_missing_key(dyn_kw, 'x_ticks'):
        dyn_kw['x_ticks'] = range(dyn_kw['data'].shape[-1])

    # Generate plot
    fig = plt.figure(figsize=_figsize, dpi=_dpi, layout='tight')
    map_ax = [fig.add_axes(321), fig.add_axes(323), fig.add_axes(325)]
    dyn_ax = fig.add_axes(122)


    dyn_kw['axes'] = [map_ax[0], dyn_ax]

    for ax in map_ax:
        _update_map(dyn_kw['data'],
                    map_kw=map_kw,
                    axis=ax,
                    cmap=cmap)
    
    map_vmin = map_kw['vmin']
    map_vmax = map_kw['vmax']
    
    _set_globals(map_ax[0], map_kw)
    _update_axes(dyn_kw,
                 dimensions=2,
                 fig=fig,
                 cmap=cmap,
                 marker_color=marker_color)

    def onselect(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        ymin = np.min([y1, y2]).round().astype(int)
        ymax = np.max([y1, y2]).round().astype(int)
        xmin = np.min([x1, x2]).round().astype(int)
        xmax = np.max([x1, x2]).round().astype(int)

        if ymax != ymin or xmax != xmin:
            rect_slice = (slice(ymin, ymax), slice(xmin, xmax))

            sum_map = np.sum(dyn_kw['data'][..., *rect_slice],
                                            axis = (-2, -1))

            weight_rect = dyn_kw['data'][..., *rect_slice]
            tth_rect = dyn_kw['tth_arr'][rect_slice]
            chi_rect = dyn_kw['chi_arr'][rect_slice]
            div_map = np.sum(dyn_kw['data'][..., *rect_slice], axis=(-2, -1))

            tth_map = np.sum(weight_rect * tth_rect, axis=(-2, -1)) / div_map
            chi_map = np.sum(weight_rect * chi_rect, axis=(-2, -1)) / div_map

            for ax, val_map, val_rect, title in zip(map_ax,
                                          [sum_map, tth_map, chi_map],
                                          [[map_vmin, map_vmax], tth_rect, chi_rect],
                                          ['ROI Sum',
                                           'ROI CoM Scattering Angle',
                                           'ROI CoM Azimuthal Angle']):
            
                map_kw['map'] = val_map
                map_kw['title'] = title

                map_kw['vmin'] = np.min(val_rect)
                map_kw['vmax'] = np.max(val_rect)
                ax.images[0].set_clim(map_kw['vmin'], map_kw['vmax'])

                _update_map(
                    map_kw=map_kw,
                    axis=ax,
                    cmap=cmap,
                    update=True)
            fig.canvas.draw_idle()

    def toggle_selector(event):
        if event.key == 't':
            if rect.active:
                print(f'ROI selector deactivated.')
                rect.set_active(False)
            else:
                print(f'ROI selector activated.')
                rect.set_active(True)

    # Manually set image to max image
    axi = dyn_kw['axes'][1].images[0]
    max_img = np.max(dyn_kw['data'], axis=(0, 1))
    axi.set_data(max_img)
    dyn_kw['axes'][1].set_title("Max Image\nDraw box for ROI here. Toggle 't' to turn off.")
    dyn_kw['vmin'] = max_img.min()
    dyn_kw['vmax'] = max_img.max()
    axi.set_clim(dyn_kw['vmin'], dyn_kw['vmax'])

    rect = RectangleSelector(
        dyn_ax,
        onselect,
        useblit=True,
        button=[1, 3],  # disable middle button
        minspanx=1,
        minspany=1,
        spancoords='pixels',
        interactive=True,
        props=dict(alpha=0.5, facecolor="tab:red"),
    )

    fig.canvas.mpl_connect('key_press_event', toggle_selector)
    fig.show()
    return fig, ax, rect



def static_window_sum_2D_plot(dyn_kw={},
                              map_kw={},
                              cmap='viridis',
                              marker_color='red'):
    '''
    
    '''

    # Check axes range
    if _check_missing_key(dyn_kw, 'x_ticks'):
        dyn_kw['x_ticks'] = range(dyn_kw['data'].shape[-1])

    # Generate plot
    fig, ax = plt.subplots(1, 2, figsize=_figsize, dpi=_dpi)
    
    # Fill and save set values
    map_kw = _fill_kwargs(map_kw,
            ['vmin',
             'vmax'])
    map_vmin = map_kw['vmin']
    map_vmax = map_kw['vmax']

    dyn_kw['axes'] = ax
    _update_map(dyn_kw['data'],
                map_kw=map_kw,
                axis=ax[0],
                cmap=cmap)
    
    _set_globals(ax[0], map_kw)
    _update_axes(dyn_kw,
                 dimensions=2,
                 fig=fig,
                 cmap=cmap,
                 marker_color=marker_color)

    def onselect(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        ymin = np.min([y1, y2]).round().astype(int)
        ymax = np.max([y1, y2]).round().astype(int)
        xmin = np.min([x1, x2]).round().astype(int)
        xmax = np.max([x1, x2]).round().astype(int)

        if ymax != ymin or xmax != xmin:
            rect_slice = (slice(ymin, ymax), slice(xmin, xmax))

            sum_map = np.sum(dyn_kw['data'][..., *rect_slice],
                                            axis = (-2, -1))
            
            map_kw['map'] = sum_map
            map_kw['title'] = 'ROI Sum'

            map_kw['vmin'] = np.min(map_vmin)
            map_kw['vmax'] = np.max(map_vmax)
            ax[0].images[0].set_clim(map_kw['vmin'], map_kw['vmax'])

            _update_map(
                map_kw=map_kw,
                axis=ax[0],
                cmap=cmap,
                update=True)
            fig.canvas.draw_idle()

    def toggle_selector(event):
        if event.key == 't':
            name = type(rect).__name__
            if rect.active:
                print(f'ROI selector deactivated.')
                rect.set_active(False)
            else:
                print(f'ROI selector activated.')
                rect.set_active(True)

    # Manually set image to max image
    axi = dyn_kw['axes'][1].images[0]
    max_img = np.max(dyn_kw['data'], axis=(0, 1))
    axi.set_data(max_img)
    dyn_kw['axes'][1].set_title("Max Image\nDraw ROI here. Toggle 't' to turn off.")
    dyn_kw['vmin'] = max_img.min()
    dyn_kw['vmax'] = max_img.max()
    axi.set_clim(dyn_kw['vmin'], dyn_kw['vmax'])

    rect = RectangleSelector(
        ax[1],
        onselect,
        useblit=True,
        button=[1, 3],  # disable middle button
        minspanx=1,
        minspany=1,
        spancoords='pixels',
        interactive=True,
        props=dict(alpha=0.5, facecolor="tab:red"),
    )

    fig.canvas.mpl_connect('key_press_event', toggle_selector)
    fig.show()
    return fig, ax, rect





from xrdmaptools.utilities.math import vector_angle
def build_tracking_map(spots_3D):


    # Effective map shape
    map_shape = (np.max(spots_3D['map_y']) + 1,
                 np.max(spots_3D['map_x']) + 1)

    mag_map = np.empty(map_shape)
    mag_map[:] = np.nan
    phi_map = mag_map.copy()
    chi_map = mag_map.copy()
    int_map = mag_map.copy()

    for index in tqdm(range(np.prod(map_shape))):
        indices = np.unravel_index(index, map_shape)
        pixel_df = pixel_spots(spots_3D, indices)

        spot_ints = pixel_df['intensity'].values
        sorted_spot_ints = sorted(list(spot_ints), reverse=True)

        for val in sorted_spot_ints:
            ind = np.nonzero(spot_ints == val)[0][0]
            chi = pixel_df.iloc[ind]['chi']
            q_mag = pixel_df.iloc[ind]['q_mag']
            q_vec = pixel_df.iloc[ind][['qx', 'qy', 'qz']].values
            # if np.abs(q_mag - 5.15628987) < 0.025:
            if np.abs(q_mag - 4.1422022472) < 0.025:
            # if np.abs(q_mag - 4.22354026) < 0.025:
                mag_map[indices] = q_mag
                phi_map[indices] = vector_angle([0, 0, -1], q_vec, degrees=True)
                chi_map[indices] = vector_angle([-1, 0], q_vec[:-1], degrees=True)
                # chi_map[indices] = chi
                int_map[indices] = val
                break
    
    return mag_map, phi_map, chi_map, int_map


def build_simple_ori_map(spots_3D):


    # Effective map shape
    map_shape = (np.max(spots_3D['map_y']) + 1,
                 np.max(spots_3D['map_x']) + 1)

    ori_map = np.empty((*map_shape, 3, 3))
    ori_map[:] = np.nan

    for index in tqdm(range(np.prod(map_shape))):
        indices = np.unravel_index(index, map_shape)
        pixel_df = pixel_spots(spots_3D, indices)

        spot_ints = pixel_df['intensity'].values
        sorted_spot_ints = sorted(list(spot_ints), reverse=True)

        spot1 = None
        spot2 = None

        for val in sorted_spot_ints:
            ind = np.nonzero(spot_ints == val)[0][0]
            q_mag = pixel_df.iloc[ind]['q_mag']
            q_vec = pixel_df.iloc[ind][['qx', 'qy', 'qz']].values
           
            if np.abs(q_mag - 5.15628987) < 0.025:
                spot1 = q_vec
                break

        for val in sorted_spot_ints:
            ind = np.nonzero(spot_ints == val)[0][0]
            q_mag = pixel_df.iloc[ind]['q_mag']
            q_vec = pixel_df.iloc[ind][['qx', 'qy', 'qz']].values

            # Unlikely check
            if np.all(q_vec == spot1):
                continue
           
            if np.abs(q_mag - 4.1422022472) < 0.025:
                spot2 = q_vec
                break
        
        if spot1 is not None and spot2 is not None:
            ori = Rotation.align_vectors(phase.Q([[-4, 1, -2], [-3, 1, -4]]),
                                            [spot1, spot2])[0]
            ori_map[indices] = ori.as_matrix()
    
    return ori_map