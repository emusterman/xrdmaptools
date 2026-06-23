import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, LogNorm
from mpl_toolkits.mplot3d.art3d import Path3DCollection
from matplotlib.widgets import SpanSelector, RectangleSelector

# Local imports
from xrdmaptools.plot import config

"""
Interactive plotting for scanning XRD maps from SRX beamline.
"""




def BaseInteractivePlot:

    def __init__(self,
                 map_kw,
                 data_kw,
                 cmap=config.cmap,
                 marker_color='red'):

        # Input map and data information
        self.map_kw = self._fill_kwargs(map_kw,
                                        ['map',
                                         'title',
                                         'vmin',
                                         'vmax',
                                         'facecolor',
                                         'scale',
                                         'x_ticks',
                                         'y_ticks',
                                         'x_label',
                                         'y_label'])

        self.data_kw = self._fill_kwargs(data_kw,
                                         ['data',
                                          'title', # Only used for windowed plots
                                          'vmin', # For images only
                                          'vmax', # For images only
                                          'scale', # For plot and images
                                          'alpha', # For 3D scatter
                                          'edges', # For 3D scatter
                                          'skip', # For 3D scatter
                                          'int_cutoff', # For 3D scatter
                                          'x_ticks', # For plot, images, and 3D scatter
                                          'x_min',
                                          'x_max',
                                          'x_label',
                                          'y_ticks', # For images and 3D scatter
                                          'y_min',
                                          'y_max',
                                          'y_label',
                                          'z_ticks', # Unused 
                                          'z_min', # For 3D scatter
                                          'z_max',
                                          'z_label'])
        
        # Matplotlib parameters
        self.fig = None
        self.axes = None
        self.cmap = cmap
        self.marker_color = marker_color

        # State information
        self.row = 0
        self.col = 0
        self.map_x = self.map_kw['x_ticks'][self.col]
        self.map_y = self.map_kw['y_ticks'][self.row]
        self.marker = None
        self.dynamic_toggle = True


    ##############################
    ### Initial Plot Functions ###
    ##############################

    def set_initial_map(self, axis):

        # Estimate map if not given
        if self.map_kw['map'] is None:
            self.map_kw['title'] = 'Summed Intensity'
            if self.data_kw['data'].ndim in (3, 4):
                self.map_kw=['map'] = np.sum(self.data_kw['data'],
                                             axis=tuple(range(2, self.data_kw['data'].ndim)))
            
        # Check axes range
        if (self.map_kw['x_ticks'] is None
            or len(self.map_kw['x_ticks']) != self.map_kw['map'].shape[1]):
            self.map_kw['x_ticks'] = list(range(self.map_kw['map'].shape[1]))
        
        if (self.map_kw['y_ticks'] is None
            or len(self.map_kw['y_ticks']) != self.map_kw['map'].shape[0]):
            # Reverse order to acquiesce to matplotlib
            self.map_kw['y_ticks'] = list(range(self.map_kw['map'].shape[0]))[::-1]
        
        map_extent = self._find_image_extent(self.map_kw['x_ticks'],
                                             self.map_kw['y_ticks'])    

        # Set color depth
        if self.map_kw['vmin'] is None:
            self.map_kw['vmin'] = np.min(self.map_kw['map'])
        if self.map_kw['vmax'] is None:
            self.map_kw['vmax'] = np.max(self.map_kw['map'])

        # Set color scale
        if self.map_kw['scale'] in [Normalize, LogNorm]:
            pass
        elif self.map_kw['scale'] in [None, 'linear']:
            self.map_kw['scale'] = Normalize
        elif self.map_kw['scale'] in ['log', 'logrithmic']:
            self.map_kw['scale'] = LogNorm

        # Plot the map!
        im = axis.imshow(self.map_kw['map'],
                         cmap=self.cmap,
                         extent=map_extent,
                         norm=self.map_kw['scale'](
                         vmin=self.map_kw['vmin'],
                         vmax=self.map_kw['vmax']
                         ))
        
        # Add colorbar
        axis.figure.colorbar(im, ax=axis)
        
        # Set map title
        if self.map_kw['title'] != None:
            axis.set_title(self.map_kw['title'])
        else:
            axis.set_title('Custom Map')

        # Set label titles
        axis.set_xlabel(self.map_kw['x_label'])
        axis.set_ylabel(self.map_kw['y_label'])

        # Set facecolor
        if (isinstance(self.map_kw['facecolor'], str)
            and self.map_kw['facecolor'].lower() == 'transparent'):
            axis.set_facecolor((1., 1., 1., 0.))
        if self.map_kw['facecolor'] is not None:
            axis.set_facecolor(self.map_kw['facecolor'])

    
    def set_initial_1D(self, axis):

        # Check x_ticks
        if (data_kw['x_ticks'] is None:
            or len(data_kw['x_ticks']) != data_kw['data'].shape[-1]):
            if data_kw['x_ticks'] is not None:
                warn_str = ("WARNING: Initial x_ticks length of "
                            + f"{self.data_kw['x_ticks'].shape[-1]} "
                            + "does not match data shape of "
                            + f"{self.data_kw['data'].shape}.")
                print(warn_str)
            self.data_kw['x_ticks'] = list(range(self.data_kw['data'].shape[-1]))

        # Confirm extents
        x_tick_range = np.max(self.data_kw['x_ticks']) - np.min(self.data_kw['x_ticks'])
        if self.data_kw['x_min'] is None:
            self.data_kw['x_min'] = np.min(self.data_kw['x_ticks']) - 0.05 * x_tick_range
        if self.data_kw['x_max'] is None:
            self.data_kw['x_max'] = np.max(self.data_kw['x_ticks']) + 0.05 * x_tick_range

        # Determine y_ticks
        y_tick_range = np.max(self.data_kw['data']) - np.min(self.data_kw['data'])
        if self.data_kw['y_min'] is None:
            self.data_kw['y_min'] = np.min(self.data_kw['data']) - 0.05 * y_tick_range
        if self.data_kw['y_max'] is None:
            self.data_kw['y_max'] = np.max(self.data_kw['data']) + 0.05 * y_tick_range

        # Set scale
        if self.data_kw['scale'] in [None, 'linear', Normalize]:
            self.data_kw['scale'] = 'linear'
        elif self.data_kw['scale'] in ['log', 'logrithmic', LogNorm]:
            self.data_kw['scale'] = 'log'

        # Plot the plot!
        axis.plot(self.data_kw['x_ticks'], self.data_kw['data'][row, col])
        axis.set_yscale(self.data_kw['scale'])
        axis.set_xlim(self.data_kw['x_min'], self.data_kw['x_max'])
        axis.set_ylim(self.data_kw['y_min'], self.data_kw['y_max'])
        axis.set_xlabel(self.data_kw['x_label'])
        axis.set_ylabel(self.data_kw['y_label'])


    def set_initial_2D(self, axis):

        # Check axes range
        if (self.data_kw['x_ticks'] is None
            or len(self.data_kw['x_ticks']) != self.data_kw['data'].shape[-1]):
            self.data_kw['x_ticks'] = list(range(self.data_kw['data'].shape[-1]))
        
        if (self.data_kw['y_ticks'] is None
            or len(self.data_kw['y_ticks']) != self.data_kw['data'].shape[-2]):
            # Reverse order to acquiesce to matplotlib
            self.data_kw['y_ticks'] = list(range(self.data_kw['data'].shape[-2]))[::-1]
        extent = _find_image_extent(self.data_kw['x_ticks'], self.data_kw['y_ticks'])
        
        if self.data_kw['vmin'] is None:
            self.data_kw['vmin'] = np.min(self.data_kw['data'])
        if self.data_kw['vmax'] is None:
            self.data_kw['vmax'] = np.max(self.data_kw['data'])
        
        if self.data_kw['scale'] in [Normalize, LogNorm]:
            pass
        elif self.data_kw['scale'] in [None, 'linear']:
            self.data_kw['scale'] = Normalize
        elif self.data_kw['scale'] in ['log', 'logrithmic']:
            self.data_kw['scale'] = LogNorm

        #print(f'row is {row}, col is {col}')
        im = axis.imshow(
                self.data_kw['data'][row, col],
                extent=extent,
                aspect='auto',
                cmap=cmap,
                norm=self.data_kw['scale'](
                    vmin=self.data_kw['vmin'],
                    vmax=self.data_kw['vmax']
                )
            )

        # Add colorbar
        axis.figure.colorbar(im, ax=self.data_kw['axes'][1])
        
        # Add labels
        axis.set_xlabel(self.data_kw['x_label'])
        axis.set_ylabel(self.data_kw['y_label'])

    
    def set_initial_3D(self, data, skip, axis):
def _update_3D_scatter(self.data_kw,
                       cmap='viridis'):

    

    plot_scatter = self.data_kw['data'][row, col]

    if self.data_kw['int_cutoff'] is None:
        self.data_kw['int_cutoff'] = 0
    int_mask = plot_scatter[:, -1] > self.data_kw['int_cutoff']
    plot_scatter = plot_scatter[int_mask]

    if len(plot_scatter) == 0:
        # Remove data and skip
        for child in self.data_kw['axes'][1].get_children():
            if isinstance(child, Path3DCollection):
                child.remove()
        return

    if self.data_kw['skip'] is None:
        skip = np.round(len(plot_scatter)
                        / 5000, 0).astype(int) # skips to about 5000 points
    else:
        skip = self.data_kw['skip']
    if skip == 0: # Default to at least 1
        skip = 1

    if not self.data_kw['axes'][1].has_data():
        self.data_kw['axes'][1].scatter(
                *plot_scatter[::skip, :3].T,
                c=plot_scatter[::skip, -1],
                cmap=cmap)

        if self.data_kw['edges'] is not None:
            # print('Plotting edges!!!')
            for edge in self.data_kw['edges']:
                self.data_kw['axes'][1].plot(*edge.T, c='gray', lw=1)
        
        # Define limits
        x_min, x_max = self.data_kw['axes'][1].get_xlim()
        y_min, y_max = self.data_kw['axes'][1].get_ylim()
        z_min, z_max = self.data_kw['axes'][1].get_zlim()
        for key, value in zip(['x_min', 'x_max',
                               'y_min', 'y_max',
                               'z_min', 'z_max'],
                               [x_min, x_max,
                                y_min, y_max,
                                z_min, z_max]):
            if self.data_kw[key] is None:
                self.data_kw[key] = value
        
        # Set limits
        self.data_kw['axes'][1].set_xlim(self.data_kw['x_min'], self.data_kw['x_max'])
        self.data_kw['axes'][1].set_ylim(self.data_kw['y_min'], self.data_kw['y_max'])
        self.data_kw['axes'][1].set_zlim(self.data_kw['z_min'], self.data_kw['z_max'])

        # Set default labels
        if self.data_kw['x_label'] is None:
            self.data_kw['x_label'] = 'qx [Å⁻¹]'
        if self.data_kw['y_label'] is None:
            self.data_kw['y_label'] = 'qy [Å⁻¹]'
        if self.data_kw['z_label'] is None:
            self.data_kw['z_label'] = 'qz [Å⁻¹]'

        # Set plot labels
        self.data_kw['axes'][1].set_xlabel(self.data_kw['x_label'])
        self.data_kw['axes'][1].set_ylabel(self.data_kw['y_label'])
        self.data_kw['axes'][1].set_zlabel(self.data_kw['z_label'])
        self.data_kw['axes'][1].set_aspect('equal')

    else:
        for child in self.data_kw['axes'][1].get_children():
            if isinstance(child, Path3DCollection):
                child.remove()

        self.data_kw['axes'][1].scatter(
                *plot_scatter[::skip, :3].T,
                c=plot_scatter[::skip, -1],
                cmap=cmap)

    
    #############################
    ### Update Plot Functions ###
    #############################

    def update_marker(self, axis):

        if self.marker is not None:
            self.marker.remove()

        self.marker = axi.scatter()
        self.marker = axi.scatter(self.map_x,
                                  self.map_y,
                                  marker='+',
                                  s=25,
                                  linewidth=1,
                                  color=self.marker_color)
        if self.dynamic_toggle:
            self.marker.set_visible(False)    
    
    
    def update_map(self, axis):
        if not axis.has_data():
            self.set_initial_map(axis)
        else:
            axis.get_images()[0].set_data(self.map_kw['map'])


    def update_1D(self, axis):
        if not axis.has_data():
            self.set_initial_1D(axis)
        else:
            axis.lines[0].set_data(self.data_kw['x_ticks'],
                                   self.data_kw['data'][row, col]) 


    def update_2D(self, axis):
        if not axis.has_data():
            self.set_initial_2D(axis)
        else:
            axis.images[0].set_data(self.data_kw['data'][row, col])


    def update_3D(self, axis):
def _update_3D_scatter(self.data_kw,
                       cmap='viridis'):

    

    data = self.data_kw['data'][row, col]

    if self.data_kw['int_cutoff'] is None:
        self.data_kw['int_cutoff'] = 0
    int_mask = data[:, -1] > self.data_kw['int_cutoff']
    data = data[int_mask]

    if len(data) == 0:
        # Remove data and skip
        for child in self.data_kw['axes'][1].get_children():
            if isinstance(child, Path3DCollection):
                child.remove()
        return

    if self.data_kw['skip'] is None:
        skip = np.round(len(data)
                        / 5000, 0).astype(int) # skips to about 5000 points
    else:
        skip = self.data_kw['skip']
    if skip == 0: # Default to at least 1
        skip = 1

    if not axis.has_data():
        self.set_initial_3d(data, skip, axis)
    else:
        for child in axis.get_children():
            if isinstance(child, Path3DCollection):
                child.remove()
        
        axis.scatter(*data[::skip, :3].T,
                     c=data[::skip, -1],
                     alpha=self.data_kw['alpha']
                     cmap=self.cmap)
    
    
    #############################
    ### Interactive Functions ###
    #############################

    def update_coordinates(event):
        pass

    
    def onclick(event):
        pass

    
    def onmove(event):
        pass

    

    
    ##############################
    ### Base Utility Functions ###
    ##############################

    @staticmethod
    def _fill_kwargs(kwargs, keys):
    
    # Auto populate kwargs
    for key in keys:
        if key not in tuple(kwargs.keys()):
            kwargs[key] = None
    
    return kwargs


    @staticmethod
    def _check_missing_key(dict, key):
        return key not in dict.keys() or key is None


    @staticmethod
    def _find_image_extent(x_ticks, y_ticks):
        # y_ticks should be given in proper, descending order!

        x_step = np.mean(np.diff(x_ticks))
        y_step = np.mean(np.diff(y_ticks))

        x_start = x_ticks[0] - x_step / 2
        x_end = x_ticks[-1] + x_step / 2
        y_start = y_ticks[0] - y_step / 2
        y_end = y_ticks[-1] + y_step / 2

        return [x_start, x_end, y_start, y_end]







####################################################################


### Support Functions ###

def _update_coordinates(event,
                        self.map_kw):
    
    # Pull global variables
    global row, col, map_x, map_y
    old_row, old_col = row, col
    #col, row = event.xdata, event.ydata
    map_x, map_y = event.xdata, event.ydata

    col = np.argmin(np.abs(self.map_kw['x_ticks'] - map_x))
    map_x = self.map_kw['x_ticks'][col]

    row = np.argmin(np.abs(self.map_kw['y_ticks'] - map_y))
    map_y = self.map_kw['y_ticks'][row]
    row = len(self.map_kw['y_ticks']) - row - 1 # reverse the index

    # Check if the pixel is in data
    if ((col >= self.map_kw['map'].shape[1])
        and (row >= self.map_kw['map'].shape[0])):
        return False

    elif ((event.name == 'motion_notify_event')
          and (old_row == row and old_col == col)):
        return False
    
    else:
        return True


def _update_axes(self.data_kw,
                 dimensions=None,
                 fig=None,
                 cmap='viridis',
                 marker_color='red',
                 update_title=True):
    """
        
    """

    self.data_kw = _fill_kwargs(
        self.data_kw,
        ['data',
         'title', # Only used for windowed plots
         'axes', # [display map, and dynamic ax]
         'vmin', # For images only
         'vmax', # For images only
         'scale', # For plot and images
         'edges', # For 3D scatter
         'skip', # For 3D scatter
         'int_cutoff', # For 3D scatter
         'x_ticks', # For plot, images, and 3D scatter
         'x_min',
         'x_max',
         'x_label',
         'y_ticks', # For images and 3D scatter
         'y_min',
         'y_max',
         'y_label',
         # 'z_ticks', # Unused 
         'z_min', # For 3D scatter
         'z_max',
         'z_label',
         ]
    )

    if dimensions is None:
        if len(self.data_kw['data'].shape) == 3:
            dimensions = 1
        elif len(self.data_kw['data'].shape) == 4:
            dimensions = 2
    
    if dimensions == 1:
        _update_plot(self.data_kw=self.data_kw)

    elif dimensions == 2:
        _update_image(self.data_kw=self.data_kw,
                      cmap=cmap)

    elif dimensions == 3:
        _update_3D_scatter(self.data_kw=self.data_kw,
                           cmap=cmap)
    
    if update_title:
        self.data_kw['axes'][1].set_title((f'Row = {row}, Col = {col}\n'
                                    + f'y = {map_y:.2f}, x = {map_x:.2f}'))
    
    _update_marker(axi=self.data_kw['axes'][0], marker_color=marker_color)    
    fig.canvas.draw_idle()



def _update_3D_scatter(self.data_kw,
                       cmap='viridis'):

    

    plot_scatter = self.data_kw['data'][row, col]

    if self.data_kw['int_cutoff'] is None:
        self.data_kw['int_cutoff'] = 0
    int_mask = plot_scatter[:, -1] > self.data_kw['int_cutoff']
    plot_scatter = plot_scatter[int_mask]

    if len(plot_scatter) == 0:
        # Remove data and skip
        for child in self.data_kw['axes'][1].get_children():
            if isinstance(child, Path3DCollection):
                child.remove()
        return

    if self.data_kw['skip'] is None:
        skip = np.round(len(plot_scatter)
                        / 5000, 0).astype(int) # skips to about 5000 points
    else:
        skip = self.data_kw['skip']
    if skip == 0: # Default to at least 1
        skip = 1

    if not self.data_kw['axes'][1].has_data():
        self.data_kw['axes'][1].scatter(
                *plot_scatter[::skip, :3].T,
                c=plot_scatter[::skip, -1],
                cmap=cmap)

        if self.data_kw['edges'] is not None:
            # print('Plotting edges!!!')
            for edge in self.data_kw['edges']:
                self.data_kw['axes'][1].plot(*edge.T, c='gray', lw=1)
        
        # Define limits
        x_min, x_max = self.data_kw['axes'][1].get_xlim()
        y_min, y_max = self.data_kw['axes'][1].get_ylim()
        z_min, z_max = self.data_kw['axes'][1].get_zlim()
        for key, value in zip(['x_min', 'x_max',
                               'y_min', 'y_max',
                               'z_min', 'z_max'],
                               [x_min, x_max,
                                y_min, y_max,
                                z_min, z_max]):
            if self.data_kw[key] is None:
                self.data_kw[key] = value
        
        # Set limits
        self.data_kw['axes'][1].set_xlim(self.data_kw['x_min'], self.data_kw['x_max'])
        self.data_kw['axes'][1].set_ylim(self.data_kw['y_min'], self.data_kw['y_max'])
        self.data_kw['axes'][1].set_zlim(self.data_kw['z_min'], self.data_kw['z_max'])

        # Set default labels
        if self.data_kw['x_label'] is None:
            self.data_kw['x_label'] = 'qx [Å⁻¹]'
        if self.data_kw['y_label'] is None:
            self.data_kw['y_label'] = 'qy [Å⁻¹]'
        if self.data_kw['z_label'] is None:
            self.data_kw['z_label'] = 'qz [Å⁻¹]'

        # Set plot labels
        self.data_kw['axes'][1].set_xlabel(self.data_kw['x_label'])
        self.data_kw['axes'][1].set_ylabel(self.data_kw['y_label'])
        self.data_kw['axes'][1].set_zlabel(self.data_kw['z_label'])
        self.data_kw['axes'][1].set_aspect('equal')

    else:
        for child in self.data_kw['axes'][1].get_children():
            if isinstance(child, Path3DCollection):
                child.remove()

        self.data_kw['axes'][1].scatter(
                *plot_scatter[::skip, :3].T,
                c=plot_scatter[::skip, -1],
                cmap=cmap)


def _update_marker(axi=None,
                   marker_color='red'):
    """
    
    """

    global marker
    marker.remove()
    marker = axi.scatter(map_x,
                         map_y,
                         marker='+',
                         s=25,
                         linewidth=1,
                         color=marker_color)
    if dynamic_toggle:
        marker.set_visible(False)


def _update_map(data=None,
                self.map_kw={},
                axis=None,
                cmap='viridis'):
    """
        
    """

    self.map_kw = _fill_kwargs(self.map_kw,
                ['map',
                 'title',
                 'vmin',
                 'vmax',
                 'facecolor',
                 'scale',
                 'x_ticks',
                 'y_ticks',
                 'x_label',
                 'y_label'])

    if not axis.has_data():
        # Estimate map if not given
        if self.map_kw['map'] is None:
            self.map_kw['title'] = 'Summed Intensity'
            if len(data.shape) == 3:
                self.map_kw['map'] = np.sum(data, axis=2)
            elif len(data.shape) == 4:   
                self.map_kw['map'] = np.sum(data, axis=(2, 3))
            
        # Check axes range
        if (self.map_kw['x_ticks'] is None
            or len(self.map_kw['x_ticks']) != self.map_kw['map'].shape[1]):
            self.map_kw['x_ticks'] = list(range(self.map_kw['map'].shape[1]))
        
        if (self.map_kw['y_ticks'] is None
            or len(self.map_kw['y_ticks']) != self.map_kw['map'].shape[0]):
            # Reverse order to acquiesce to matplotlib
            self.map_kw['y_ticks'] = list(range(self.map_kw['map'].shape[0]))[::-1]
        
        map_extent = _find_image_extent(self.map_kw['x_ticks'], self.map_kw['y_ticks'])    

        # Set color depth
        if self.map_kw['vmin'] is None:
            self.map_kw['vmin'] = np.min(self.map_kw['map'])
        if self.map_kw['vmax'] is None:
            self.map_kw['vmax'] = np.max(self.map_kw['map'])

        if self.map_kw['scale'] in [Normalize, LogNorm]:
            pass
        elif self.map_kw['scale'] in [None, 'linear']:
            self.map_kw['scale'] = Normalize
        elif self.map_kw['scale'] in ['log', 'logrithmic']:
            self.map_kw['scale'] = LogNorm

        # Plot Image!
        im = axis.imshow(self.map_kw['map'],
                            cmap=cmap,
                            extent=map_extent,
                            norm=self.map_kw['scale'](
                                vmin=self.map_kw['vmin'],
                                vmax=self.map_kw['vmax']
                                ))
        
        # Add colorbar
        if self.map_kw['map'].ndim == 2:
            axis.figure.colorbar(im, ax=axis)
        
        # Set map title
        if self.map_kw['title'] != None:
            axis.set_title(self.map_kw['title'])
        else:
            axis.set_title('Custom Map')

        # Set label titles
        axis.set_xlabel(self.map_kw['x_label'])
        axis.set_ylabel(self.map_kw['y_label'])

        # Set facecolor
        if (isinstance(self.map_kw['facecolor'], str)
            and self.map_kw['facecolor'].lower() == 'transparent'):
            axis.set_facecolor((1., 1., 1., 0.))
        if self.map_kw['facecolor'] is not None:
            axis.set_facecolor(self.map_kw['facecolor'])
    
    else:
        axi = axis.get_images()[0]
        axi.set_data(self.map_kw['map'])
        axis.set_title(self.map_kw['title'])

    
def _set_globals(axis, self.map_kw):
    # Plot display map with marker
    global row, col, marker, dynamic_toggle, map_x, map_y
    marker = axis.scatter([], [])
    marker.set_visible(False)
    dynamic_toggle = True
    row, col = 0, 0
    map_x = self.map_kw['x_ticks'][col]
    map_y = self.map_kw['y_ticks'][row]


### Interactive plotting functions ###

def interactive_1D_plot(self.data_kw={},
                        self.map_kw={},
                        cmap='viridis',
                        marker_color='red'):
    """
    
    """

    # Check axes range
    if _check_missing_key(self.data_kw, 'x_ticks'):
        self.data_kw['x_ticks'] = range(self.data_kw['data'].shape[-1])

    # Generate plot
    fig, ax = plt.subplots(1, 2, figsize=config.ext_figsize, dpi=config.dpi)
    self.data_kw['axes'] = ax
    _update_map(self.data_kw['data'],
                self.map_kw=self.map_kw,
                axis=ax[0],
                cmap=cmap)
    
    # Set globals and first plot
    _set_globals(ax[0], self.map_kw)
    _update_axes(self.data_kw,
                 dimensions=1,
                 fig=fig,
                 cmap=cmap,
                 marker_color=marker_color)

    def update_axes(event):
        if _update_coordinates(event,
                               self.map_kw):
            _update_axes(self.data_kw,
                         dimensions=1,
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


def interactive_2D_plot(self.data_kw={},
                        self.map_kw={},
                        cmap='viridis',
                        marker_color='red'):
    """
    
    """

    # Check axes range
    if _check_missing_key(self.data_kw, 'x_ticks'):
        self.data_kw['x_ticks'] = range(self.data_kw['data'].shape[-1])
    if _check_missing_key(self.data_kw, 'y_ticks'):
        self.data_kw['y_ticks'] = range(self.data_kw['data'].shape[-2])

    # Generate plot
    fig, ax = plt.subplots(1, 2, figsize=config.ext_figsize, dpi=config.dpi)
    self.data_kw['axes'] = ax
    _update_map(self.data_kw['data'],
                self.map_kw=self.map_kw,
                axis=ax[0],
                cmap=cmap)

    # Set globals and first image
    _set_globals(ax[0], self.map_kw)
    _update_axes(self.data_kw,
                 dimensions=2,
                 fig=fig,
                 cmap=cmap,
                 marker_color=marker_color)

    def update_axes(event):
        if _update_coordinates(event,
                               self.map_kw):
            _update_axes(self.data_kw=self.data_kw,
                         dimensions=2,
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


def interactive_3D_plot(self.data_kw={},
                        self.map_kw={},
                        cmap='viridis',
                        marker_color='red'):
    """
    
    """

    # Generate plot
    fig = plt.figure(figsize=config.ext_figsize, dpi=config.dpi)
    ax = [fig.add_axes(121), fig.add_axes(122, projection='3d')]
    self.data_kw['axes'] = ax
    _update_map(self.data_kw['data'],
                self.map_kw=self.map_kw,
                axis=ax[0],
                cmap=cmap)

    # Set globals and first scatter
    _set_globals(ax[0], self.map_kw)
    _update_axes(self.data_kw=self.data_kw,
                 dimensions=3,
                 fig=fig,
                 cmap=cmap,
                 marker_color=marker_color)

    def update_axes(event):
        if _update_coordinates(event,
                               self.map_kw):
            _update_axes(self.data_kw=self.data_kw,
                         dimensions=3,
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


### 1D Windowed Plots ###

def _interactive_1D_window_plot(self.data_kw={},
                                self.map_kw={},
                                func=None,
                                map_label=None,
                                cmap='viridis',
                                marker_color='red'):
    """
    
    """

    # Check axes range
    if _check_missing_key(self.data_kw, 'x_ticks'):
        self.data_kw['x_ticks'] = range(self.data_kw['data'].shape[-1])

    # Generate plot
    fig, ax = plt.subplots(1, 2, figsize=config.ext_figsize, dpi=config.dpi)

    self.data_kw['axes'] = ax
    _update_map(self.data_kw['data'],
                self.map_kw=self.map_kw,
                axis=ax[0],
                cmap=cmap)
    
    _set_globals(ax[0], self.map_kw)
    _update_axes(self.data_kw,
                 dimensions=1,
                 fig=fig,
                 cmap=cmap,
                 marker_color=marker_color,
                 update_title=False)

    def onselect(xmin, xmax):
        indmin, indmax = np.searchsorted(self.data_kw['x_ticks'],
                                         (xmin, xmax))
        indmax = min(len(self.data_kw['x_ticks']) - 1, indmax)

        if indmax - indmin >= 1:
            new_map = func(self.data_kw, self.map_kw, slice(indmin, indmax))
            
            self.map_kw['map'] = new_map
            self.map_kw['title'] = f'{map_label} from {xmin:.2f}-{xmax:.2f}'

            self.data_kw['axes'][0].images[0].set_norm(
                self.map_kw['scale'](self.map_kw['vmin'], self.map_kw['vmax']))
            _update_map(
                self.map_kw=self.map_kw,
                axis=ax[0],
                cmap=cmap)
            fig.canvas.draw_idle()
    
    # Manually set integration to max value
    axi = self.data_kw['axes'][1].lines[0]
    axi.set_data(self.data_kw['x_ticks'], np.max(self.data_kw['data'], axis=(0, 1)))
    # self.data_kw['axes'][1].set_title('Max Integration')
    self.data_kw['axes'][1].set_title(self.data_kw['title']
                                + "Max Integration")

    span = SpanSelector(
        ax[1],
        onselect,
        "horizontal",
        useblit=True,
        props=dict(alpha=0.5, facecolor="tab:red"),
        interactive=True,
        drag_from_anywhere=True
    )

    fig.show()
    return fig, ax, span


def interactive_1D_window_sum_plot(**kwargs):

    def func(self.data_kw, self.map_kw, inds):
        new_map = np.sum(self.data_kw['data'][:, :, inds], axis=-1)
        # Keep current values of plot
        self.map_kw['vmin'], self.map_kw['vmax'] = self.data_kw['axes'][0].images[0].get_clim()

        return new_map

    return _interactive_1D_window_plot(**kwargs,
                                       func=func,
                                       map_label='Sum')


def interactive_1D_window_com_plot(**kwargs):
    
    # def func(vals, inds):
    #     return (np.sum(vals * inds, axis=-1) / np.sum(vals, axis=-1))
    def func(self.data_kw, self.map_kw, inds):
        new_map = (np.sum(self.data_kw['data'][..., inds]
                          * self.data_kw['x_ticks'][inds], axis=-1)
                   / np.sum(self.data_kw['data'][..., inds], axis=-1))
        self.map_kw['vmin'] = np.min(self.data_kw['x_ticks'][inds])
        self.map_kw['vmax'] = np.max(self.data_kw['x_ticks'][inds])
        return new_map
    
    return _interactive_1D_window_plot(**kwargs,
                                       func=func,
                                       map_label='2θ CoM')


### 2D Windowed Plots ###

def _interactive_2D_window_plot(self.data_kw={},
                                self.map_kw={},
                                func=None,
                                map_label=None,
                                cmap='viridis',
                                marker_color='red'):
    """
    
    """

    # Check axes range
    if _check_missing_key(self.data_kw, 'x_ticks'):
        self.data_kw['x_ticks'] = range(self.data_kw['data'].shape[-1])

    # Generate plot
    fig, ax = plt.subplots(1, 2, figsize=config.ext_figsize, dpi=config.dpi)

    self.data_kw['axes'] = ax
    _update_map(self.data_kw['data'],
                self.map_kw=self.map_kw,
                axis=ax[0],
                cmap=cmap)
    
    _set_globals(ax[0], self.map_kw)
    _update_axes(self.data_kw,
                 dimensions=2,
                 fig=fig,
                 cmap=cmap,
                 marker_color=marker_color,
                 update_title=False)

    def onselect(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        ymin = np.min([y1, y2]).round().astype(int)
        ymax = np.max([y1, y2]).round().astype(int)
        xmin = np.min([x1, x2]).round().astype(int)
        xmax = np.max([x1, x2]).round().astype(int)

        if ymax != ymin and xmax != xmin:
            rect_slice = (slice(ymin, ymax), slice(xmin, xmax))

            new_map = func(self.data_kw, self.map_kw, rect_slice)
            
            self.map_kw['map'] = new_map
            self.map_kw['title'] = f'ROI {map_label}'

            self.data_kw['axes'][0].images[0].set_norm(
                self.map_kw['scale'](self.map_kw['vmin'], self.map_kw['vmax']))         
            _update_map(
                self.map_kw=self.map_kw,
                axis=ax[0],
                cmap=cmap)

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
    axi = self.data_kw['axes'][1].images[0]
    max_img = np.max(self.data_kw['data'], axis=(0, 1))
    axi.set_data(max_img)
    # self.data_kw['axes'][1].set_title("Max Image\nDraw box for ROI here. Toggle 't' to turn off.")
    self.data_kw['axes'][1].set_title(self.data_kw['title']
                                + "Max Image\nDraw box for ROI here. Toggle 't' to turn off.")
    self.data_kw['vmin'] = max_img.min()
    self.data_kw['vmax'] = max_img.max()
    axi.set_clim(self.data_kw['vmin'], self.data_kw['vmax'])

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


def interactive_2D_window_sum_plot(**kwargs):

    def func(self.data_kw, self.map_kw, rect_slice):
        self.map_kw['vmin'], self.map_kw['vmax'] = self.data_kw['axes'][0].images[0].get_clim()
        return np.sum(self.data_kw['data'][..., *rect_slice], axis = (-2, -1))

    return _interactive_2D_window_plot(**kwargs,
                                       func=func,
                                       map_label='Sum')


def interactive_2D_window_tth_com_plot(self.data_kw={},
                                       **kwargs):
    # Specific check
    if 'tth_arr' not in self.data_kw:
        raise ValueError("Must provide 'tth_arr' in self.data_kw!")

    def func(self.data_kw, self.map_kw, rect_slice):
        weight_rect = self.data_kw['data'][..., *rect_slice]
        tth_rect = self.data_kw['tth_arr'][rect_slice]
        div_map = np.sum(self.data_kw['data'][..., *rect_slice], axis=(-2, -1))
        tth_map = np.sum(weight_rect * tth_rect, axis=(-2, -1)) / div_map

        self.map_kw['vmin'] = np.min(tth_rect)
        self.map_kw['vmax'] = np.max(tth_rect)

        return tth_map

    return _interactive_2D_window_plot(self.data_kw=self.data_kw,
                                       **kwargs,
                                       func=func,
                                       map_label='2θ CoM')


def interactive_2D_window_chi_com_plot(self.data_kw={},
                                       **kwargs):
    # Specific check
    if 'chi_arr' not in self.data_kw:
        raise ValueError("Must provide 'chi_arr' in self.data_kw!")

    def func(self.data_kw, self.map_kw, rect_slice):
        weight_rect = self.data_kw['data'][..., *rect_slice]
        chi_rect = self.data_kw['chi_arr'][rect_slice]
        div_map = np.sum(self.data_kw['data'][..., *rect_slice], axis=(-2, -1))
        chi_map = np.sum(weight_rect * chi_rect, axis=(-2, -1)) / div_map

        self.map_kw['vmin'] = np.min(chi_rect)
        self.map_kw['vmax'] = np.max(chi_rect)

        return chi_map

    return _interactive_2D_window_plot(self.data_kw=self.data_kw,
                                       **kwargs,
                                       func=func,
                                       map_label='χ CoM')





### Not fully implemented into dataset classes

def interactive_2D_1D_plot(dyn_2D_kw={},
                           dyn_1D_kw={},
                           self.map_kw={},
                           cmap='viridis',
                           marker_color='red'):
    """
    
    """

    # Check 2D axes range
    if _check_missing_key(dyn_2D_kw, 'x_ticks'):
        dyn_2D_kw['x_ticks'] = range(dyn_2D_kw['data'].shape[-1])
    if _check_missing_key(dyn_2D_kw, 'y_ticks'):
        dyn_2D_kw['y_ticks'] = range(dyn_2D_kw['data'].shape[-2])
    # Check 1D axes range
    if _check_missing_key(dyn_1D_kw, 'x_ticks'):
        dyn_1D_kw['x_ticks'] = range(dyn_1D_kw['data'].shape[-1])

    # Generate plot
    fig = plt.figure(figsize=config.ext_figsize, dpi=config.dpi)
    subfigs = fig.subfigures(1, 2)
    ax = [subfigs[0].subplots(1, 1),
          *subfigs[1].subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})]
    subfigs[1].subplots_adjust(hspace=0.5)
    dyn_2D_kw['axes'] = [ax[0], ax[1]]
    dyn_1D_kw['axes'] = [ax[0], ax[2]]

    _update_map(dyn_1D_kw['data'],
                self.map_kw=self.map_kw,
                axis=ax[0],
                cmap=cmap)

    # Set globals and first plots
    _set_globals(ax[0], self.map_kw)
    _update_axes(self.data_kw=dyn_2D_kw,
                 dimensions=2,
                 fig=fig,
                 cmap=cmap,
                 marker_color=marker_color)
    _update_axes(self.data_kw=dyn_1D_kw,
                 dimensions=1,
                 fig=fig,
                 cmap=cmap,
                 marker_color=marker_color)

    def update_axes(event):
        if _update_coordinates(event,
                               self.map_kw):
            _update_axes(self.data_kw=dyn_2D_kw,
                         dimensions=2,
                         fig=fig,
                         cmap=cmap,
                         marker_color=marker_color)
            _update_axes(self.data_kw=dyn_1D_kw,
                         dimensions=1,
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


def interactive_1D_1D_plot(self.data_kw1={},
                           self.data_kw2={},
                           self.map_kw={},
                           cmap='viridis',
                           marker_color='red'):
    """
    
    """

    # Check axes range
    for self.data_kw in [self.data_kw1, self.data_kw2]:
        # Check 1D axes range
        if _check_missing_key(self.data_kw, 'x_ticks'):
            self.data_kw['x_ticks'] = range(self.data_kw['data'].shape[-1])

    # Generate plot
    fig = plt.figure(figsize=config.ext_figsize, dpi=config.dpi)
    subfigs = fig.subfigures(1, 2)
    ax = [subfigs[0].subplots(1, 1),
          *subfigs[1].subplots(2, 1)]
    subfigs[1].subplots_adjust(hspace=0.5)
    self.data_kw1['axes'] = [ax[0], ax[1]]
    self.data_kw2['axes'] = [ax[0], ax[2]]

    _update_map(self.data_kw1['data'],
                self.map_kw=self.map_kw,
                axis=ax[0],
                cmap=cmap)

    # Set globals and first plots
    _set_globals(ax[0], self.map_kw)
    for self.data_kw in [self.data_kw1, self.data_kw2]:
        _update_axes(self.data_kw,
                     dimensions=1,
                     fig=fig,
                     cmap=cmap,
                     marker_color=marker_color)

    def update_axes(event):
        if _update_coordinates(event,
                               self.map_kw):
            for self.data_kw in [self.data_kw1, self.data_kw2]:
                _update_axes(self.data_kw,
                             dimensions=1,
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
                                  self.map_kw={},
                                  cmap='viridis',
                                  marker_color='red'):
    """
    
    """

    # Check 2D axes range
    if _check_missing_key(dyn_2D_kw, 'x_ticks'):
        dyn_2D_kw['x_ticks'] = range(dyn_2D_kw['data'].shape[-1])
    if _check_missing_key(dyn_2D_kw, 'y_ticks'):
        dyn_2D_kw['y_ticks'] = range(dyn_2D_kw['data'].shape[-2])
    # Check 1D axes range
    if _check_missing_key(dyn_1D_kw, 'x_ticks'):
        dyn_1D_kw['x_ticks'] = range(dyn_1D_kw['data'].shape[-1])

    # Generate plot
    fig = plt.figure(figsize=config.ext_figsize, dpi=config.dpi)
    subfigs = fig.subfigures(1, 2, wspace=0.1)
    ax = [subfigs[0].subplots(1, 1),
          *subfigs[1].subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)]
    subfigs[1].subplots_adjust(hspace=0)
    dyn_2D_kw['axes'] = [ax[0], ax[1]]
    dyn_1D_kw['axes'] = [ax[0], ax[2]]

    _update_map(dyn_1D_kw['data'],
                self.map_kw=self.map_kw,
                axis=ax[0],
                cmap=cmap)

    # Set globals and first plots
    _set_globals(ax[0], self.map_kw)
    _update_axes(self.data_kw=dyn_1D_kw,
                 dimensions=1,
                 fig=fig,
                 cmap=cmap,
                 marker_color=marker_color,)
    _update_axes(self.data_kw=dyn_2D_kw,
                 dimensions=2,
                 fig=fig,
                 cmap=cmap,
                 marker_color=marker_color)

    def update_axes(event):
        if _update_coordinates(event,
                               self.map_kw):
            _update_axes(self.data_kw=dyn_1D_kw,
                         dimensions=1,
                         fig=fig,
                         cmap=cmap,
                         marker_color=marker_color,)
            _update_axes(self.data_kw=dyn_2D_kw,
                         dimensions=2,
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

# Written quickly
def interactive_3D_labeled_plot(self.data_kw={},
                                self.map_kw={},
                                cmap='viridis',
                                marker_color='red'):
    """
    
    """

    # Generate plot
    fig = plt.figure(figsize=config.ext_figsize, dpi=config.dpi)
    ax = [fig.add_axes(121), fig.add_axes(122, projection='3d')]
    self.data_kw['axes'] = ax
    _update_map(self.data_kw['data'],
                self.map_kw=self.map_kw,
                axis=ax[0],
                cmap=cmap)

    # Set globals and first scatter
    _set_globals(ax[0], self.map_kw)
    _update_axes(self.data_kw=self.data_kw,
                    dimensions=3,
                    fig=fig,
                    cmap=cmap,
                    marker_color=marker_color)

    label_list = []    
    def update_axes(event):
        nonlocal label_list
        if _update_coordinates(event,
                               self.map_kw):
            _update_axes(self.data_kw=self.data_kw,
                         dimensions=3,
                         fig=fig,
                         cmap=cmap,
                         marker_color=marker_color)
            
            for a in label_list:
                a.remove()
            label_list = []
            for hkls, spots, colors in zip(
                                self.data_kw['labels'][row, col],
                                self.data_kw['label_spots'][row, col],
                                self.data_kw['label_colors'][row, col]):
                a = self.data_kw['axes'][1].text(*spots.T, hkls, fontsize=8, c=colors)
                label_list.append(a)

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