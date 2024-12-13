import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, LogNorm

'''
Preliminary interactive plotting for scanning XRD maps from SRX beamline.
Ultimate goal is to allow for quick preliminary analysis during beamtime to allow diagnostic and on-the-fly testing and analysis. 
More formalized, detailed, and conventional analysis will be saved for other modules.
Short-term goal is to write down several iterations of interactive functions, not necessarilly following best coding practices.
This will keep them all in one place to ease access later.
'''

'''
Terms:
xticks              (arr)   List of two theta radial angle values (in degrees or radians). Could be different between 1D and 2D outputs
yticks              (arr)   List of y_ticks azimuthal angle values (in degrees or radians)
integrated_data     (arr)   (x, y, x_ticks) array of integrated mapped data from the pyFAI 1D azimuthal integrator
image_data          (arr)   (x, y, x_ticks, y_ticks) array of calibrated mapped data from pyFAI 2D azimuthal integrator
'''


# map_kw = {
#     map : 2D array,
#     title : str,
#     vmin : float,
#     vmax : float,
#     norm : normalizer,
#     x_ticks : list,
#     y_ticks : list,
#     x_label : str,
#     y_label : str,
# }




### Support Functions ###

def _update_coordinates(event,
                        map_kw):
    
    # Pull global variables
    global row, col, map_x, map_y
    old_row, old_col = row, col
    #col, row = event.xdata, event.ydata
    map_x, map_y = event.xdata, event.ydata

    col = np.argmin(np.abs(map_kw['x_ticks'] - map_x))
    map_x = map_kw['x_ticks'][col]

    row = np.argmin(np.abs(map_kw['y_ticks'] - map_y))
    map_y = map_kw['y_ticks'][row]
    row = len(map_kw['y_ticks']) - row - 1 # reverse the index

    # Check if the pixel is in data
    if ((col >= map_kw['map'].shape[1])
        and (row >= map_kw['map'].shape[0])):
        return False

    elif ((event.name == 'motion_notify_event')
          and (old_row == row and old_col == col)):
        return False
    
    else:
        return True


def _update_axes(dyn_kw,
                 fig=None,
                 cmap='viridis',
                 marker_color='red'):
    '''
        
    '''

    dyn_kw = _fill_kwargs(
        dyn_kw,
        ['data',
         'axes', # [display map, and dynamic ax]
         'vmin',
         'vmax',
         'scale',
         'x_ticks',
         'x_min',
         'x_max',
         'x_label',
         'y_ticks',
         'y_min',
         'y_max',
         'y_label']
    )
    
    dyn_kw['axes'][1].clear()
    if len(dyn_kw['data'].shape) == 3:
        _update_plot(dyn_kw=dyn_kw)

    elif len(dyn_kw['data'].shape) == 4:
        _update_image(dyn_kw=dyn_kw,
                      cmap=cmap)
        
    dyn_kw['axes'][1].set_title((f'Row = {row}, Col = {col}\n'
                                 + f'y = {map_y:.2f}, x = {map_x:.2f}'))
    
    _update_marker(axi=dyn_kw['axes'][0], marker_color=marker_color)    
    fig.canvas.draw_idle()


def _update_plot(dyn_kw):
    '''
    
    '''
    #print('Updating Plot!')
    x_tick_range = np.max(dyn_kw['x_ticks']) - np.min(dyn_kw['x_ticks'])
    if dyn_kw['x_min'] is None:
        dyn_kw['x_min'] = np.min(dyn_kw['x_ticks']) - 0.05 * x_tick_range
    if dyn_kw['x_max'] is None:
        dyn_kw['x_max'] = np.max(dyn_kw['x_ticks']) + 0.05 * x_tick_range

    y_tick_range = np.max(dyn_kw['data']) - np.min(dyn_kw['data'])
    if dyn_kw['y_min'] is None:
        dyn_kw['y_min'] = np.min(dyn_kw['data']) - 0.05 * y_tick_range
    if dyn_kw['y_max'] is None:
        dyn_kw['y_max'] = np.max(dyn_kw['data']) + 0.05 * y_tick_range

    if dyn_kw['scale'] in [None, 'linear', Normalize]:
        dyn_kw['scale'] = 'linear'
    elif dyn_kw['scale'] in ['log', 'logrithmic', LogNorm]:
        dyn_kw['scale'] = 'log'

    if len(dyn_kw['x_ticks']) == len(dyn_kw['data'][row, col]):
        dyn_kw['axes'][1].plot(dyn_kw['x_ticks'], dyn_kw['data'][row, col])
    else:
        plot_x = np.linspace(np.min(dyn_kw['x_ticks']),
                             np.max(dyn_kw['x_ticks']),
                             len(dyn_kw['data'][row, col]))
        dyn_kw['axes'][1].plot(plot_x, dyn_kw['data'][row, col])
    
    dyn_kw['axes'][1].set_yscale(dyn_kw['scale'])
    dyn_kw['axes'][1].set_xlim(dyn_kw['x_min'], dyn_kw['x_max'])
    dyn_kw['axes'][1].set_ylim(dyn_kw['y_min'], dyn_kw['y_max'])
    dyn_kw['axes'][1].set_xlabel(dyn_kw['x_label'])
    dyn_kw['axes'][1].set_ylabel(dyn_kw['y_label'])


def _update_image(dyn_kw,
                  cmap='viridis'):
    '''
    
    '''

    plot_img = dyn_kw['data'][row, col]
    extent = _find_image_extent(dyn_kw['x_ticks'], dyn_kw['y_ticks'])
    
    if dyn_kw['vmin'] is None:
        dyn_kw['vmin'] = np.min(plot_img)
    if dyn_kw['vmax'] is None:
        dyn_kw['vmax'] = np.max(plot_img)
    
    if dyn_kw['scale'] in [Normalize, LogNorm]:
        pass
    elif dyn_kw['scale'] in [None, 'linear']:
        dyn_kw['scale'] = Normalize
    elif dyn_kw['scale'] in ['log', 'logrithmic']:
        dyn_kw['scale'] = LogNorm

    #print(f'row is {row}, col is {col}')
    img = dyn_kw['axes'][1].imshow(
        plot_img,
        extent=extent,
        aspect='auto',
        cmap=cmap,
        norm=dyn_kw['scale'](
            vmin=dyn_kw['vmin'],
            vmax=dyn_kw['vmax']
        )
    )
    
    dyn_kw['axes'][1].set_xlabel(dyn_kw['x_label'])
    dyn_kw['axes'][1].set_ylabel(dyn_kw['y_label'])


def _update_marker(axi=None,
                   marker_color='red'):
    '''
    
    '''

    global marker
    marker.remove()
    marker = axi.scatter(map_x, map_y, marker='+', s=25, linewidth=1, color=marker_color)
    if dynamic_toggle:
        marker.set_visible(False)


def _display_map(data=None,
                 map_kw={},
                 axes=None,
                 cmap='viridis',
                 update=False):
    '''
        
    '''

    map_kw = _fill_kwargs(map_kw,
                ['map',
                 'title',
                 'vmin',
                 'vmax',
                 'scale',
                 'x_ticks',
                 'y_ticks',
                 'x_label',
                 'y_label'])

    # Estimate map if not given
    if map_kw['map'] is None:
        map_kw['title'] = 'Summed Intensity'
        if len(data.shape) == 3:
            map_kw['map'] = np.sum(data, axis=2)
        elif len(data.shape) == 4:   
            map_kw['map'] = np.sum(data, axis=(2, 3))
        
    # Check axes range
    if (map_kw['x_ticks'] is None
        or len(map_kw['x_ticks']) != map_kw['map'].shape[1]):
        map_kw['x_ticks'] = list(range(map_kw['map'].shape[1]))
    
    if (map_kw['y_ticks'] is None
        or len(map_kw['y_ticks']) != map_kw['map'].shape[0]):
        # Reverse order to acquiesce to matplotlib
        map_kw['y_ticks'] = list(range(map_kw['map'].shape[0]))[::-1]
    
    map_extent = _find_image_extent(map_kw['x_ticks'], map_kw['y_ticks'])    

    # Set color depth
    if map_kw['vmin'] is None:
        map_kw['vmin'] = np.min(map_kw['map'])
    if map_kw['vmax'] is None:
        map_kw['vmax'] = np.max(map_kw['map'])

    if map_kw['scale'] in [Normalize, LogNorm]:
        pass
    elif map_kw['scale'] in [None, 'linear']:
        map_kw['scale'] = Normalize
    elif map_kw['scale'] in ['log', 'logrithmic']:
        map_kw['scale'] = LogNorm

    
    # Plot Image! Add colorbar???
    axes[0].imshow(map_kw['map'],
                cmap=cmap,
                extent=map_extent,
                norm=map_kw['scale'](
                    vmin=map_kw['vmin'],
                    vmax=map_kw['vmax']
                    ))
    
    # Set map title
    if map_kw['title'] != None:
        axes[0].set_title(map_kw['title'])
    else:
        axes[0].set_title('Custom Map')

    # Set label titles
    axes[0].set_xlabel(map_kw['x_label'])
    axes[0].set_ylabel(map_kw['y_label'])

    
def _set_globals(ax):
    # Plot display map with marker
    global row, col, marker, dynamic_toggle, cbar, map_x, map_y
    marker = ax[0].scatter([], [])
    marker.set_visible(False)
    dynamic_toggle = False
    cbar = None
    row, col = -1, -1
    map_x, map_y = np.nan, np.nan


### Variations of Plotting Functions ###

def interactive_1D_plot(dyn_kw={},
                        map_kw={},
                        cmap='viridis',
                        marker_color='red'):
    '''
    
    '''

    # Check axes range
    if _check_missing_key(dyn_kw, 'x_ticks'):
        dyn_kw['x_ticks'] = range(dyn_kw['data'].shape[-1])

    # Generate plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=200)
    dyn_kw['axes'] = ax
    _display_map(dyn_kw['data'],
                 map_kw=map_kw,
                 axes=ax,
                 cmap=cmap)
    
    _set_globals(ax)

    def update_axes(event):
        if _update_coordinates(event,
                               map_kw):
            _update_axes(dyn_kw,
                         fig=fig,
                         cmap=cmap,
                         marker_color=marker_color)

    # Make interactive
    def onclick(event):
        if event.inaxes == ax[0]:
            global dynamic_toggle, marker
            dynamic_toggle = not dynamic_toggle
            update_axes(event)

    # Make dynamic
    def onmove(event):
        global dynamic_toggle
        if dynamic_toggle:
            if event.inaxes == ax[0]:
                update_axes(event)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    binding_id = plt.connect('motion_notify_event', onmove)
    return fig, ax


def interactive_2D_plot(dyn_kw={},
                        map_kw={},
                        cmap='viridis',
                        marker_color='red'):
    '''
    
    '''

    # Check axes range
    if _check_missing_key(dyn_kw, 'x_ticks'):
        dyn_kw['x_ticks'] = range(dyn_kw['data'].shape[-1])
    if _check_missing_key(dyn_kw, 'y_ticks'):
        dyn_kw['y_ticks'] = range(dyn_kw['data'].shape[-2])

    # Generate plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=200)
    dyn_kw['axes'] = ax
    _display_map(dyn_kw['data'],
                 map_kw=map_kw,
                 axes=ax,
                 cmap=cmap)

    _set_globals(ax)

    def update_axes(event):
        if _update_coordinates(event,
                               map_kw):
            _update_axes(dyn_kw=dyn_kw,
                         fig=fig,
                         cmap=cmap,
                         marker_color=marker_color)

    # Make interactive
    def onclick(event):
        if event.inaxes == ax[0]:
            global dynamic_toggle, marker
            dynamic_toggle = not dynamic_toggle
            update_axes(event)
    
    # Make dynamic
    def onmove(event):
        global dynamic_toggle
        if dynamic_toggle:
            if event.inaxes == ax[0]:
                update_axes(event)
    
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    binding_id = plt.connect('motion_notify_event', onmove)
    return fig, ax


def interactive_2D_1D_plot(dyn_2D_kw={},
                           dyn_1D_kw={},
                           map_kw={},
                           cmap='viridis',
                           marker_color='red'):
    '''
    
    '''

    # Check 2D axes range
    if _check_missing_key(dyn_2D_kw, 'x_ticks'):
        dyn_2D_kw['x_ticks'] = range(dyn_2D_kw['data'].shape[-1])
    if _check_missing_key(dyn_2D_kw, 'y_ticks'):
        dyn_2D_kw['y_ticks'] = range(dyn_2D_kw['data'].shape[-2])
    # Check 1D axes range
    if _check_missing_key(dyn_1D_kw, 'x_ticks'):
        dyn_1D_kw['x_ticks'] = range(dyn_1D_kw['data'].shape[-1])

    # Generate plot
    fig = plt.figure(figsize=(10, 5), dpi=200)
    subfigs = fig.subfigures(1, 2)
    ax = [subfigs[0].subplots(1, 1),
          *subfigs[1].subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})]
    subfigs[1].subplots_adjust(hspace=0.5)
    dyn_2D_kw['axes'] = [ax[0], ax[1]]
    dyn_1D_kw['axes'] = [ax[0], ax[2]]

    _display_map(dyn_1D_kw['data'],
                 map_kw=map_kw,
                 axes=ax,
                 cmap=cmap)

    _set_globals(ax)

    def update_axes(event):
        if _update_coordinates(event,
                               map_kw):
            _update_axes(dyn_kw=dyn_2D_kw,
                         fig=fig,
                         cmap=cmap,
                         marker_color=marker_color)
            _update_axes(dyn_kw=dyn_1D_kw,
                         fig=fig,
                         cmap=cmap,
                         marker_color=marker_color)
            ax[2].set_title('')

    # Make interactive
    def onclick(event):
        if event.inaxes == ax[0]:
            global dynamic_toggle, marker
            dynamic_toggle = not dynamic_toggle
            update_axes(event)     

    # Make dynamic
    def onmove(event):
        global dynamic_toggle
        if dynamic_toggle:
            if event.inaxes == ax[0]:
                update_axes(event)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    binding_id = plt.connect('motion_notify_event', onmove)
    return fig, ax


def interactive_1D_1D_plot(dyn_kw1={},
                           dyn_kw2={},
                           map_kw={},
                           cmap='viridis',
                           marker_color='red'):
    '''
    
    '''

    # Check axes range
    for dyn_kw in [dyn_kw1, dyn_kw2]:
        # Check 1D axes range
        if _check_missing_key(dyn_kw, 'x_ticks'):
            dyn_kw['x_ticks'] = range(dyn_kw['data'].shape[-1])

    # Generate plot
    fig = plt.figure(figsize=(10, 5), dpi=200)
    subfigs = fig.subfigures(1, 2)
    ax = [subfigs[0].subplots(1, 1),
          *subfigs[1].subplots(2, 1)]
    subfigs[1].subplots_adjust(hspace=0.5)
    dyn_kw1['axes'] = [ax[0], ax[1]]
    dyn_kw2['axes'] = [ax[0], ax[2]]

    _display_map(dyn_kw1['data'],
                 map_kw=map_kw,
                 axes=ax,
                 cmap=cmap)

    _set_globals(ax)

    def update_axes(event):
        if _update_coordinates(event,
                               map_kw):
            for dyn_kw in [dyn_kw1, dyn_kw2]:
                _update_axes(dyn_kw,
                             fig=fig,
                             cmap=cmap,
                             marker_color=marker_color)
            ax[2].set_title('')

    # Make interactive
    def onclick(event):
        if event.inaxes == ax[0]:
            global dynamic_toggle, marker
            dynamic_toggle = not dynamic_toggle
            update_axes(event)     

    # Make dynamic
    def onmove(event):
        global dynamic_toggle
        if dynamic_toggle:
            if event.inaxes == ax[0]:
                update_axes(event)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    binding_id = plt.connect('motion_notify_event', onmove)
    return fig, ax


# TODO: Add checks to make sure all shared axes information matches
def interactive_shared_2D_1D_plot(dyn_2D_kw={},
                                  dyn_1D_kw={},
                                  map_kw={},
                                  cmap='viridis',
                                  marker_color='red'):
    '''
    
    '''

    # Check 2D axes range
    if _check_missing_key(dyn_2D_kw, 'x_ticks'):
        dyn_2D_kw['x_ticks'] = range(dyn_2D_kw['data'].shape[-1])
    if _check_missing_key(dyn_2D_kw, 'y_ticks'):
        dyn_2D_kw['y_ticks'] = range(dyn_2D_kw['data'].shape[-2])
    # Check 1D axes range
    if _check_missing_key(dyn_1D_kw, 'x_ticks'):
        dyn_1D_kw['x_ticks'] = range(dyn_1D_kw['data'].shape[-1])

    # Generate plot
    fig = plt.figure(figsize=(10, 5), dpi=200)
    subfigs = fig.subfigures(1, 2, wspace=0.1)
    ax = [subfigs[0].subplots(1, 1),
          *subfigs[1].subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)]
    subfigs[1].subplots_adjust(hspace=0)
    dyn_2D_kw['axes'] = [ax[0], ax[1]]
    dyn_1D_kw['axes'] = [ax[0], ax[2]]

    _display_map(dyn_1D_kw['data'],
                 map_kw=map_kw,
                 axes=ax,
                 cmap=cmap)

    _set_globals(ax)

    def update_axes(event):
        if _update_coordinates(event,
                               map_kw):
            _update_axes(dyn_kw=dyn_1D_kw,
                         fig=fig,
                         cmap=cmap,
                         marker_color=marker_color,)
            _update_axes(dyn_kw=dyn_2D_kw,
                         fig=fig,
                         cmap=cmap,
                         marker_color=marker_color)
            ax[2].set_title('')

    # Make interactive
    def onclick(event):
        if event.inaxes == ax[0]:
            global dynamic_toggle, marker
            dynamic_toggle = not dynamic_toggle
            update_axes(event)     

    # Make dynamic
    def onmove(event):
        global dynamic_toggle
        if dynamic_toggle:
            if event.inaxes == ax[0]:
                update_axes(event)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    binding_id = plt.connect('motion_notify_event', onmove)
    return fig, ax



### Small Helper Functions ###

def _fill_kwargs(kwargs, keys):
    
    # Auto populate kwargs
    for key in keys:
        if key not in tuple(kwargs.keys()):
            kwargs[key] = None
    
    return kwargs


def _check_missing_key(dict, key):
    return key not in dict.keys() or key is None


def _find_image_extent(x_ticks, y_ticks):
    # y_ticks should be given in proper, descending order!

    x_step = np.mean(np.diff(x_ticks))
    y_step = np.mean(np.diff(y_ticks))

    x_start = x_ticks[0] - x_step / 2
    x_end = x_ticks[-1] + x_step / 2
    y_start = y_ticks[0] - y_step / 2
    y_end = y_ticks[-1] + y_step / 2

    return [x_start, x_end, y_start, y_end]




### WIP Functions ###

from matplotlib.widgets import SpanSelector

def integrateable_static_1D_plot():
    raise NotImplementedError()

def integrateable_dynamic_1D_plot(dyn_kw={},
                                  map_kw={},
                                  cmap='viridis',
                                  marker_color='red'):
    '''
    
    '''

    # Check axes range
    if _check_missing_key(dyn_kw, 'x_ticks'):
        dyn_kw['x_ticks'] = range(dyn_kw['data'].shape[-1])

    # Generate plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=200)
    

    # Fill and save set values
    map_kw = _fill_kwargs(map_kw,
            ['vmin',
             'vmax'])
    map_vmin = map_kw['vmin']
    map_vmax = map_kw['vmax']

    dyn_kw['axes'] = ax
    _display_map(dyn_kw['data'],
                 map_kw=map_kw,
                 axes=ax,
                 cmap=cmap)
    
    _set_globals(ax)

    def update_axes(event):
        if _update_coordinates(event,
                               map_kw):
            _update_axes(dyn_kw,
                         fig=fig,
                         cmap=cmap,
                         marker_color=marker_color)


    def onselect(xmin, xmax):
        indmin, indmax = np.searchsorted(dyn_kw['x_ticks'],
                                         (xmin, xmax))
        indmax = min(len(dyn_kw['x_ticks']) - 1, indmax)

        if indmax - indmin >= 1:
            new_map = np.sum(dyn_kw['data'][:, :, indmin : indmax],
                         axis=(-1))
            
            map_kw['map'] = new_map
            # map_kw['title'] = 'Selected ROI'
            map_kw['title'] = f'Sum from {xmin:.2f}-{xmax:.2f}'
            map_kw['vmin'] = map_vmin
            map_kw['vmax'] = map_vmax

            _display_map(
                # dyn_kw['data'],
                map_kw=map_kw,
                axes=ax,
                cmap=cmap,
                update=True)
            fig.canvas.draw_idle()

    # Make interactive
    def onclick(event):
        if event.inaxes == ax[0]:
            global dynamic_toggle, marker
            dynamic_toggle = not dynamic_toggle
            update_axes(event)

    # Make dynamic
    def onmove(event):
        global dynamic_toggle
        if dynamic_toggle:
            if event.inaxes == ax[0]:
                update_axes(event)

    span = SpanSelector(
        ax[1],
        onselect,
        "horizontal",
        useblit=True,
        props=dict(alpha=0.5, facecolor="tab:red"),
        interactive=True,
        drag_from_anywhere=True
    )

    # ax[1].plot(dyn_kw['data'][0, 0])

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    binding_id = plt.connect('motion_notify_event', onmove)
    return fig, ax, span