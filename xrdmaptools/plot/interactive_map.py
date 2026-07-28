import matplotlib.pyplot as plt
import numpy as np
from abc import abstractmethod
from matplotlib.colors import Normalize, LogNorm
from mpl_toolkits.mplot3d.art3d import Path3DCollection
from matplotlib.widgets import SpanSelector, RectangleSelector

# Local imports
from xrdmaptools.plot import config

"""
Interactive plotting for scanning XRD maps from SRX beamline.
"""



class BaseInteractivePlot:

    def __init__(self,
                 map_kw,
                 data_kw,
                 cmap=config.cmap,
                 marker_color='red'):

        # Input map and data information
        self.map_kw = self.fill_map_kwargs(map_kw)
        self.data_kw = self.fill_data_kwargs(data_kw)

        # Check for minimum data
        if self.data_kw['data'] is None:
            err_str = "Must provide data!"
            raise ValueError(err_str)

        # Matplotlib parameters
        self.fig = None
        self.axes = None
        self.cmap = cmap
        self.marker_color = marker_color

        # State information
        self.row = 0
        self.col = 0
        self.map_x = 0
        self.map_y = 0
        self.marker = None
        self.dynamic_toggle = True
        
        # Check for missing data than cannot be guessed
        self.check_inputs()

    
    def return_plot(self):
        self.fig.canvas.draw()
        self.fig.plot_instance = self
        return self.fig, self.ax


    def check_inputs(self):
        pass


    ##############################
    ### Initial Plot Functions ###
    ##############################

    def set_initial_map(self, axis, data_kw):

        # Estimate map if not given
        if self.map_kw['map'] is None:
            self.map_kw['title'] = 'Summed Intensity'
            if data_kw['data'].ndim in (3, 4):
                self.map_kw['map'] = np.sum(data_kw['data'],
                                            axis=tuple(range(2, data_kw['data'].ndim)))
            
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
                         ),
                         interpolation=None)
        
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

    
    def set_initial_1D(self, axis, data_kw):

        # Check x_ticks
        if (data_kw['x_ticks'] is None
            or len(data_kw['x_ticks']) != data_kw['data'].shape[-1]):
            if data_kw['x_ticks'] is not None:
                warn_str = ("WARNING: Initial x_ticks length of "
                            + f"{data_kw['x_ticks'].shape[-1]} "
                            + "does not match data shape of "
                            + f"{data_kw['data'].shape}.")
                print(warn_str)
            data_kw['x_ticks'] = list(range(data_kw['data'].shape[-1]))

        # Confirm extents
        x_tick_range = np.max(data_kw['x_ticks']) - np.min(data_kw['x_ticks'])
        if data_kw['x_min'] is None:
            data_kw['x_min'] = np.min(data_kw['x_ticks']) - 0.05 * x_tick_range
        if data_kw['x_max'] is None:
            data_kw['x_max'] = np.max(data_kw['x_ticks']) + 0.05 * x_tick_range

        # Determine y_ticks
        y_tick_range = np.max(data_kw['data']) - np.min(data_kw['data'])
        if data_kw['y_min'] is None:
            data_kw['y_min'] = np.min(data_kw['data']) - 0.05 * y_tick_range
        if data_kw['y_max'] is None:
            data_kw['y_max'] = np.max(data_kw['data']) + 0.05 * y_tick_range

        # Set scale
        if data_kw['scale'] in [None, 'linear', Normalize]:
            data_kw['scale'] = 'linear'
        elif data_kw['scale'] in ['log', 'logrithmic', LogNorm]:
            data_kw['scale'] = 'log'

        # Plot the plot!
        axis.plot(data_kw['x_ticks'], data_kw['data'][self.row, self.col])
        axis.set_yscale(data_kw['scale'])
        axis.set_xlim(data_kw['x_min'], data_kw['x_max'])
        axis.set_ylim(data_kw['y_min'], data_kw['y_max'])
        axis.set_xlabel(data_kw['x_label'])
        axis.set_ylabel(data_kw['y_label'])


    def set_initial_2D(self, axis, data_kw):

        # Check axes range
        if (data_kw['x_ticks'] is None
            or len(data_kw['x_ticks']) != data_kw['data'].shape[-1]):
            data_kw['x_ticks'] = list(range(data_kw['data'].shape[-1]))
        
        if (data_kw['y_ticks'] is None
            or len(data_kw['y_ticks']) != data_kw['data'].shape[-2]):
            # Reverse order to acquiesce to matplotlib
            data_kw['y_ticks'] = list(range(data_kw['data'].shape[-2]))[::-1]
        extent = self._find_image_extent(data_kw['x_ticks'], data_kw['y_ticks'])
        
        if data_kw['vmin'] is None:
            data_kw['vmin'] = np.min(data_kw['data'])
        if data_kw['vmax'] is None:
            data_kw['vmax'] = np.max(data_kw['data'])
        
        if data_kw['scale'] in [Normalize, LogNorm]:
            pass
        elif data_kw['scale'] in [None, 'linear']:
            data_kw['scale'] = Normalize
        elif data_kw['scale'] in ['log', 'logrithmic']:
            data_kw['scale'] = LogNorm

        #print(f'row is {row}, col is {col}')
        im = axis.imshow(
                data_kw['data'][self.row, self.col],
                extent=extent,
                aspect='auto',
                cmap=self.cmap,
                norm=data_kw['scale'](
                    vmin=data_kw['vmin'],
                    vmax=data_kw['vmax']
                ),
                interpolation=None
            )

        # Add colorbar
        axis.figure.colorbar(im, ax=axis)
        
        # Add labels
        axis.set_xlabel(data_kw['x_label'])
        axis.set_ylabel(data_kw['y_label'])

    
    def set_initial_3D(self, axis, data_kw):

        data = data_kw['data'][self.row, self.col]

        if data_kw['int_cutoff'] is None:
            data_kw['int_cutoff'] = 0
        int_mask = data[:, -1] > data_kw['int_cutoff']
        data = data[int_mask]

        if data_kw['skip'] is None:
            skip = np.round(len(data)
                            / 5000, 0).astype(int) # skips to about 5000 points
        else:
            skip = data_kw['skip']
        # Default to at least 1
        skip = max([1, skip])

        axis.scatter(*data[::skip, :3].T,
                     c=data[::skip, -1],
                     cmap=self.cmap)

        if data_kw['edges'] is not None:
            for edge in data_kw['edges']:
                axis.plot(*edge.T, c='gray', lw=1)
        
        # Define limits
        x_min, x_max = axis.get_xlim()
        y_min, y_max = axis.get_ylim()
        z_min, z_max = axis.get_zlim()
        for key, value in zip(['x_min', 'x_max',
                               'y_min', 'y_max',
                               'z_min', 'z_max'],
                               [x_min, x_max,
                                y_min, y_max,
                                z_min, z_max]):
            if data_kw[key] is None:
                data_kw[key] = value
        
        # Set limits
        axis.set_xlim(data_kw['x_min'], data_kw['x_max'])
        axis.set_ylim(data_kw['y_min'], data_kw['y_max'])
        axis.set_zlim(data_kw['z_min'], data_kw['z_max'])

        # Set default labels
        if data_kw['x_label'] is None:
            data_kw['x_label'] = 'qx [Å⁻¹]'
        if data_kw['y_label'] is None:
            data_kw['y_label'] = 'qy [Å⁻¹]'
        if data_kw['z_label'] is None:
            data_kw['z_label'] = 'qz [Å⁻¹]'

        # Set plot labels
        axis.set_xlabel(data_kw['x_label'])
        axis.set_ylabel(data_kw['y_label'])
        axis.set_zlabel(data_kw['z_label'])
        axis.set_aspect('equal')


    #############################
    ### Update Plot Functions ###
    #############################

    def update_marker(self, axis):

        if self.marker is not None:
            self.marker.remove()

        self.marker = axis.scatter(self.map_x,
                                   self.map_y,
                                   marker='+',
                                   s=25,
                                   linewidth=1,
                                   color=self.marker_color)
        if self.dynamic_toggle:
            self.marker.set_visible(False)    
    
    
    def update_map(self, axis, data_kw=None):

        if data_kw is None:
            data_kw = self.data_kw

        if not axis.has_data():
            self.set_initial_map(axis, data_kw)
        else:
            axis.get_images()[0].set_data(self.map_kw['map'])


    def update_1D(self, axis, data_kw):
        if not axis.has_data():
            self.set_initial_1D(axis, data_kw)
        else:
            axis.lines[0].set_data(data_kw['x_ticks'],
                                   data_kw['data'][self.row, self.col]) 


    def update_2D(self, axis, data_kw):
        if not axis.has_data():
            self.set_initial_2D(axis, data_kw)
        else:
            axis.images[0].set_data(data_kw['data'][self.row, self.col])


    def update_3D(self, axis, data_kw):
        if not axis.has_data():
            self.set_initial_3D(axis, data_kw)
        else:
            for child in axis.get_children():
                if isinstance(child, Path3DCollection):
                    child.remove()

            data = data_kw['data'][self.row, self.col]

            if data_kw['int_cutoff'] is None:
                data_kw['int_cutoff'] = 0
            int_mask = data[:, -1] > data_kw['int_cutoff']
            data = data[int_mask]

            if len(data) == 0:
                return

            if data_kw['skip'] is None:
                skip = np.round(len(data)
                                / 5000, 0).astype(int) # skips to about 5000 points
            else:
                skip = data_kw['skip']
            # Default to at least 1
            skip = max([1, skip])
            
            axis.scatter(*data[::skip, :3].T,
                         c=data[::skip, -1],
                         alpha=data_kw['alpha'],
                         cmap=self.cmap)
    
    
    #############################
    ### Interactive Functions ###
    #############################

    def update_coordinates(self, event):    
        # Set previous values
        old_row, old_col = self.row, self.col
        map_x, map_y = event.xdata, event.ydata
        
        # Update col
        self.col = np.argmin(np.abs(self.map_kw['x_ticks'] - map_x))
        self.map_x = self.map_kw['x_ticks'][self.col]
        
        # Update row
        self.row = np.argmin(np.abs(self.map_kw['y_ticks'] - map_y))
        self.map_y = self.map_kw['y_ticks'][self.row]
        self.row = len(self.map_kw['y_ticks']) - self.row - 1 # reverse the index

        # Check if the pixel is in data
        if ((self.col >= self.map_kw['map'].shape[1])
            and (self.row >= self.map_kw['map'].shape[0])):
            return False

        elif ((event.name == 'motion_notify_event')
            and (old_row == self.row and old_col == self.col)):
            return False
        
        else:
            return True


    def update_axes(self,
                    axis,
                    map_axis,
                    dimensions=None,
                    data_kw=None,
                    update_title=True):

        # Use internal attribute
        if data_kw is None:
            data_kw = self.data_kw
        
        if dimensions == 1:
            self.update_1D(axis, data_kw)

        elif dimensions == 2:
            self.update_2D(axis, data_kw)

        elif dimensions == 3:
            self.update_3D(axis, data_kw)
        
        if update_title:
            axis.set_title((f'Row = {self.row}, Col = {self.col}\n'
                            + f'y = {self.map_y:.2f}, x = {self.map_x:.2f}'))
        
        self.update_marker(map_axis)    
        self.fig.canvas.draw_idle()

    
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

    ##########################
    ### Utiliity Functions ###
    ##########################

    def fill_map_kwargs(self, map_kw):
        return  self._fill_kwargs(map_kw,
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


    def fill_data_kwargs(self, data_kw):
        return self._fill_kwargs(data_kw,
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


class BaseInteractiveNDPlot(BaseInteractivePlot):

    dimensions = None

    def __init__(self,
                 *args,
                 figsize=config.ext_figsize,
                 dpi=config.dpi,
                 **kwargs):
        super().__init__(*args, **kwargs)

        # Generate plot
        self.fig, self.ax = self.generate_plots(figsize, dpi)

        # Generate the map
        self.update_map(self.ax[0])

        # Initialize axes
        self.update_axes(self.ax[1],
                         self.ax[0],
                         dimensions=self.dimensions)
        
        cid_onclick = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        cid_onmove = self.fig.canvas.mpl_connect('motion_notify_event', self.onmove)


    # Make interactive
    def internal_update_axes(self, event):
        if self.update_coordinates(event):
            self.update_axes(self.ax[1],
                             self.ax[0],
                             dimensions=self.dimensions)


    def onclick(self, event):
        if event.inaxes == self.ax[0]:
            self.dynamic_toggle = not self.dynamic_toggle
            self.internal_update_axes(event)
    

    def onmove(self, event):
        if self.dynamic_toggle and event.inaxes == self.ax[0]:
            self.internal_update_axes(event)        


    def generate_plots(self, figsize, dpi):
        return plt.subplots(1, 2, figsize=figsize, dpi=dpi)


class BaseInteractiveNDWindowPlot(BaseInteractivePlot):

    dimensions = None
    map_label = None
    data_label = None

    def __init__(self,
                 *args,
                 figsize=config.ext_figsize,
                 dpi=config.dpi,
                 **kwargs):
        super().__init__(*args, **kwargs)

        # Check inputs
        self.check_inputs()

        # Generate plot
        self.fig, self.ax = self.generate_plots(figsize, dpi)

        # Generate the map
        self.update_map(self.ax[0])
        self.ax[0].set_title

        # Initialize axes
        self.update_axes(self.ax[1],
                         self.ax[0],
                         dimensions=self.dimensions,
                         update_title=False)
        
        self.selector = self.get_selector()


    def generate_plots(self, figsize, dpi):
        return plt.subplots(1, 2, figsize=figsize, dpi=dpi)


    @abstractmethod
    def func(self):
        pass

    @abstractmethod
    def onselect(self):
        pass

    @abstractmethod
    def get_selector(self):
        pass


class BaseInteractive1DWindowPlot(BaseInteractiveNDWindowPlot):

    dimensions = 1
    data_label = 'Max Integration'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Manually set integration to max value
        axi = self.ax[1].lines[0]
        axi.set_data(self.data_kw['x_ticks'], np.max(self.data_kw['data'], axis=(0, 1)))
        self.ax[1].set_title(self.data_kw['title'] + " " + self.data_label)

    def onselect(self, xmin, xmax):
        indmin, indmax = np.searchsorted(self.data_kw['x_ticks'],
                                         (xmin, xmax))
        indmax = min(len(self.data_kw['x_ticks']) - 1, indmax)

        if indmax - indmin >= 1:
            new_map = self.func(slice(indmin, indmax))
            
            self.map_kw['map'] = new_map
            self.map_kw['title'] = f'{self.map_label} from {xmin:.2f}-{xmax:.2f}'

            self.ax[0].images[0].set_norm(
                self.map_kw['scale'](self.map_kw['vmin'], self.map_kw['vmax']))
            self.update_map(axis=self.ax[0])
            self.fig.canvas.draw_idle()

    
    def get_selector(self):
        return SpanSelector(self.ax[1],
                            self.onselect,
                            "horizontal",
                            useblit=True,
                            props=dict(alpha=0.5, facecolor="tab:red"),
                            interactive=True,
                            drag_from_anywhere=True
                            )


class BaseInteractive2DWindowPlot(BaseInteractiveNDWindowPlot):

    dimensions = 2
    data_label = 'Max Image'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Manually set image to max image
        axi = self.ax[1].images[0]
        max_img = np.max(self.data_kw['data'], axis=(0, 1))
        axi.set_data(max_img)
        self.ax[1].set_title(self.data_kw['title'] + " " + self.data_label)

        if self.data_kw['vmin'] is None:
            self.data_kw['vmin'] = max_img.min()
        if self.data_kw['vmax'] is None:
            self.data_kw['vmax'] = max_img.max()
        axi.set_clim(self.data_kw['vmin'], self.data_kw['vmax'])


    def onselect(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        ymin = np.min([y1, y2]).round().astype(int)
        ymax = np.max([y1, y2]).round().astype(int)
        xmin = np.min([x1, x2]).round().astype(int)
        xmax = np.max([x1, x2]).round().astype(int)

        if ymax != ymin and xmax != xmin:
            inds = (slice(ymin, ymax), slice(xmin, xmax))

            new_map = self.func(inds)
            
            self.map_kw['map'] = new_map
            self.map_kw['title'] = f'ROI {self.map_label}'

            self.ax[0].images[0].set_norm(
                self.map_kw['scale'](self.map_kw['vmin'], self.map_kw['vmax']))         
            self.update_map(self.ax[0])

            self.fig.canvas.draw_idle()

    
    def get_selector(self):
        return RectangleSelector(self.ax[1],
                                 self.onselect,
                                 useblit=True,
                                 button=[1, 3],  # disable middle button
                                 minspanx=1,
                                 minspany=1,
                                 spancoords='pixels',
                                 interactive=True,
                                 props=dict(alpha=0.5, facecolor="tab:red"),
                                 )


#########################
### Plotting Fuctions ###
#########################

class Interactive1DPlot(BaseInteractiveNDPlot):
    dimensions = 1


class Interactive2DPlot(BaseInteractiveNDPlot):
    dimensions = 2


class Interactive3DPlot(BaseInteractiveNDPlot):
    dimensions = 3

    def generate_plots(self, figsize, dpi):
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = [fig.add_axes(121), fig.add_axes(122, projection='3d')]
        return fig, ax


class Interactive3DLabeledPlot(Interactive3DPlot):
    label_list = []

    def internal_update_axes(self, event):
        super().internal_update_axes(event)

        for a in self.label_list:
            a.remove()
        self.label_list = []
        for hkls, spots, colors in zip(
                            self.data_kw['labels'][self.row, self.col],
                            self.data_kw['label_spots'][self.row, self.col],
                            self.data_kw['label_colors'][self.row, self.col]):
            a = self.ax[1].text(*spots.T, hkls, fontsize=8, c=colors)
            self.label_list.append(a)


    def check_inputs(self):
        for key in ['labels', 'label_spots', 'label_colors']:
            if (key not in self.data_kw or
                self.data_kw[key] is None):
                err_str = f"Missing data from data_kw: '{key}'."
                raise ValueError(err_str)


### Window Functions ###

class Interactive1DWindowSumPlot(BaseInteractive1DWindowPlot):
    map_label = 'Sum'

    def func(self, inds):
        self.map_kw['vmin'], self.map_kw['vmax'] = self.ax[0].images[0].get_clim()
            # self.map_kw['vmin'] = self.ax[0].images[0].get_clim()[0]
            # self.map_kw['vmax'] = self.ax[0].images[0].get_clim()[1]
        return np.sum(self.data_kw['data'][:, :, inds], axis=-1)


class Interactive1DWindowCOMPlot(BaseInteractive1DWindowPlot):
    map_label = '2θ CoM'
    
    def func(self, inds):
        self.map_kw['vmin'] = np.min(self.data_kw['x_ticks'][inds])
        self.map_kw['vmax'] = np.max(self.data_kw['x_ticks'][inds])        
        return (np.sum(self.data_kw['data'][..., inds]
                       * self.data_kw['x_ticks'][inds], axis=-1)
                / np.sum(self.data_kw['data'][..., inds], axis=-1))


class Interactive2DWindowSumPlot(BaseInteractive2DWindowPlot):
    map_label = 'Sum'

    def func(self, inds):
        self.map_kw['vmin'], self.map_kw['vmax'] = self.ax[0].images[0].get_clim()
        # if self.map_kw['vmin'] is None:
        #     self.map_kw['vmin'] = self.ax[0].images[0].get_clim()[0]
        # if self.map_kw['vmax'] is None:
        #     self.map_kw['vmax'] = self.ax[0].images[0].get_clim()[1]
        return np.sum(self.data_kw['data'][..., *inds], axis = (-2, -1))


class Interactive2DWindowTTHCOMPlot(BaseInteractive2DWindowPlot):
    map_label = '2θ CoM'

    def check_inputs(self):
        if 'tth_arr' not in self.data_kw:
            raise ValueError("Must provide 'tth_arr' in data_kw!")

    def func(self, inds):
        weight_rect = self.data_kw['data'][..., *inds]
        tth_rect = self.data_kw['tth_arr'][inds]
        div_map = np.sum(self.data_kw['data'][..., *inds], axis=(-2, -1))
        tth_map = np.sum(weight_rect * tth_rect, axis=(-2, -1)) / div_map

        self.map_kw['vmin'] = np.min(tth_rect)
        self.map_kw['vmax'] = np.max(tth_rect)

        return tth_map


class Interactive2DWindowCHICOMPlot(BaseInteractive2DWindowPlot):
    map_label = 'χ CoM'

    def check_inputs(self):
        if 'chi_arr' not in self.data_kw:
            raise ValueError("Must provide 'chi_arr' in data_kw!")

    def func(self, inds):
        weight_rect = self.data_kw['data'][..., *inds]
        chi_rect = self.data_kw['chi_arr'][inds]
        div_map = np.sum(self.data_kw['data'][..., *inds], axis=(-2, -1))
        chi_map = np.sum(weight_rect * chi_rect, axis=(-2, -1)) / div_map

        self.map_kw['vmin'] = np.min(chi_rect)
        self.map_kw['vmax'] = np.max(chi_rect)

        return chi_map


### Multiple Dynamic Plots ###

class BaseInteractiveNDNDPlot(BaseInteractivePlot):

    dimensions1 = None
    dimensions2 = None

    def __init__(self,
                 map_kw,
                 data_kw1,
                 data_kw2,
                 figsize=config.ext_figsize,
                 dpi=config.dpi,
                 **kwargs):
        super().__init__(map_kw, data_kw1, **kwargs)
        
        # Relabel and fill
        self.data_kw1 = self.data_kw
        del self.data_kw
        self.data_kw2 = self.fill_data_kwargs(data_kw2)

        # Generate plot
        self.fig, self.ax = self.generate_plots(figsize, dpi)

        # Generate the map
        self.update_map(self.ax[0], self.data_kw1)

        # Initialize 1st data axis
        self.update_axes(self.ax[1],
                         self.ax[0],
                         data_kw=self.data_kw1,
                         dimensions=self.dimensions1)

        # Initialize 2nd data axis
        self.update_axes(self.ax[2],
                         self.ax[0],
                         data_kw=self.data_kw2,
                         dimensions=self.dimensions2)
        
        cid_onclick = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        cid_onmove = self.fig.canvas.mpl_connect('motion_notify_event', self.onmove)



    # Make interactive
    def internal_update_axes(self, event):
        if self.update_coordinates(event):
            # 1st data axis
            self.update_axes(self.ax[1],
                             self.ax[0],
                             data_kw=self.data_kw,
                             dimensions=self.dimensions1)
            # 2nd data axis
            self.update_axes(self.ax[2],
                             self.ax[0],
                             data_kw=self.data_kw2,
                             dimensions=self.dimensions2)


    def onclick(self, event):
        if event.inaxes == self.ax[0]:
            self.dynamic_toggle = not self.dynamic_toggle
            self.internal_update_axes(event)
    

    def onmove(self, event):
        if self.dynamic_toggle and event.inaxes == self.ax[0]:
            self.internal_update_axes(event)        


    @abstractmethod
    def generate_plots(self, figsize, dpi):
        pass


class Interactive1D1DPlot(BaseInteractiveNDNDPlot):
    dimensions1 = 1
    dimensions2 = 1

    def generate_plots(self, figsize, dpi):
        fig = plt.figure(figsize=figsize, dpi=dpi)
        subfigs = fig.subfigures(1, 2)
        ax = [subfigs[0].subplots(1, 1),
              *subfigs[1].subplots(2, 1)]
        subfigs[1].subplots_adjust(hspace=0.5)
        
        return fig, ax


class Interactive2D1DPlot(BaseInteractiveNDNDPlot):
    dimensions1 = 2
    dimensions2 = 1

    def generate_plots(self, figsize, dpi):
        fig = plt.figure(figsize=figsize, dpi=figsize)
        subfigs = fig.subfigures(1, 2)
        ax = [subfigs[0].subplots(1, 1),
              *subfigs[1].subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})]
        subfigs[1].subplots_adjust(hspace=0.5)

        return fig, ax


class Interactive2D1DPlotShared(BaseInteractiveNDNDPlot):
    dimensions1 = 2
    dimensions2 = 1

    def generate_plots(self, figsize, dpi):
        fig = plt.figure(figsize=figsize, dpi=figsize)
        subfigs = fig.subfigures(1, 2)
        ax = [subfigs[0].subplots(1, 1),
              *subfigs[1].subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)]
        subfigs[1].subplots_adjust(hspace=0.5)

        return fig, ax


####################
### WIP Plotting ###
####################

class InteractiveIntegrated2DPlot(BaseInteractivePlot):

    dimensions = 2
    data_label = 'Sum'

    def __init__(self,
                 *args,
                 figsize=config.ext_figsize,
                 dpi=config.dpi,
                 **kwargs):
        super().__init__(*args, **kwargs)

        # Set some specific globals
        self.press_pos = (0, 0)
        self.release_pos = (0, 0)

        # Generate plot
        self.fig, self.ax = self.generate_plots(figsize, dpi)

        # Generate the map
        self.update_map(self.ax[0])

        # Initialize axes
        self.update_axes(self.ax[1],
                         self.ax[0],
                         dimensions=self.dimensions)
        
        self.selector = self.get_selector()

        # cid_press = self.fig.canvas.mpl_connect('button_press_event', self.onpress)
        # cid_release = self.fig.canvas.mpl_connect('button_release_event', self.onrelease)
        cid_onmove = self.fig.canvas.mpl_connect('motion_notify_event', self.onmove)


    # Make interactive
    def internal_update_axes(self, event):
        if self.update_coordinates(event):
            self.update_axes(self.ax[1],
                             self.ax[0],
                             dimensions=self.dimensions)


    def generate_plots(self, figsize, dpi):
        return plt.subplots(1, 2, figsize=figsize, dpi=dpi)


    def get_selector(self):
        self.ax[0].set_autoscale_on(False) 
        return InclusivePixelSnapSelector(self.ax[0],
                                 self.onselect,
                                 useblit=True,
                                 button=[1, 3],  # disable middle button
                                 minspanx=1,
                                 minspany=1,
                                 spancoords='pixels',
                                 interactive=True,
                                 props=dict(alpha=0.5, facecolor="tab:red"),
                                 )

    # def onpress(self, event):
    #     if event.inaxes != self.ax[0]:
    #         return
    #     self.press_pos = np.array([event.x, event.y])


    # def onrelease(self, event):
    #     if event.inaxes != self.ax[0]:
    #         return

    #     self.dynamic_toggle = not self.dynamic_toggle
    #     self.selector.set_active(not self.dynamic_toggle)
    #     self.release_pos = np.array([event.x, event.y])

    #     # Default to rectangle selector behavior
    #     if any(np.abs(self.press_pos - self.release_pos)) > 1:
    #         return
        
    #     self.internal_update_axes(event)
    

    def onmove(self, event):
        if self.dynamic_toggle and event.inaxes == self.ax[0]:
            self.internal_update_axes(event)

    
    def onselect(self, eclick, erelease):
        
        left, right, bottom, top = self.ax[0].get_images()[0].get_extent()
        left_start, top_start, width, height = self.selector._rect_bbox
        pixel_w, pixel_h = self.selector.pixel_w, self.selector.pixel_h
        
        xmin = np.floor(np.abs(left - left_start - (pixel_w / 2)) / pixel_w)
        xmax = xmin + (width / pixel_w)

        ymin = np.floor(np.abs(top - top_start - (pixel_h / 2)) / pixel_h)
        ymax = ymin + (height / pixel_h)

        xmin = int(round(xmin))
        xmax = int(round(xmax))
        ymin = int(round(ymin))
        ymax = int(round(ymax))

        if xmax - xmin > 1 or ymax - ymin > 1:
            self.dynamic_toggle = False
            self.update_marker(self.ax[0])

            inds = (slice(ymin, ymax), slice(xmin, xmax))
            
            new_data = self.func(inds)

            self.ax[1].images[0].set_data(new_data)
            self.ax[1].set_title(f'{self.data_label} of row = {ymin}:{ymax}, col = {xmin}:{xmax}\n'
                                 + f'y = {self.map_kw['y_ticks'][ymin]:.2f}:{self.map_kw['y_ticks'][ymax - 1]:.2f},'
                                 + f' x = {self.map_kw['x_ticks'][xmin]:.2f}:{self.map_kw['x_ticks'][xmax - 1]:.2f}')

            self.fig.canvas.draw_idle()  
        else:
            self.dynamic_toggle = not self.dynamic_toggle
            self.selector._clear_without_update()
            self.internal_update_axes(eclick)


    def func(self, inds):
        return np.sum(self.data_kw['data'][*inds], axis=(0, 1))




class InclusivePixelSnapSelector(RectangleSelector):

    def __init__(self,
                 ax,
                 onselect,
                 **kwargs):
        
        self.extent = ax.get_images()[0].get_extent()
        self.left, self.right, self.bottom, self.top = self.extent
        self.img_h, self.img_w = ax.get_images()[0].get_array().shape

        self.pixel_w = abs(self.right - self.left) / self.img_w
        self.pixel_h = abs(self.top - self.bottom) / self.img_h
        
        super().__init__(ax, onselect, **kwargs)


    @property
    def extents(self):
        return super().extents


    @extents.setter
    def extents(self, extents):

        x0, x1, y0, y1 = extents
        
        # Map x data to pixel columns
        col0 = int(np.floor((x0 - self.left) / self.pixel_w))
        col1 = int(np.floor((x1 - self.left) / self.pixel_w))
        col_min = max(0, min(col0, col1))
        col_max = min(self.img_w, max(col0, col1))
        # self.col_min, self.col_max = col_min, col_max
        
        # Map y data to pixel rows
        row0 = int(np.floor((y0 - self.top) / self.pixel_h))
        row1 = int(np.floor((y1 - self.top) / self.pixel_h))
        row_min = max(0, min(row0, row1))
        row_max = min(self.img_h, max(row0, row1))
        # self.row_min, self.row_max = row_min, row_max

        # Convert back to data coordinates
        snap_x_min = self.left + col_min * self.pixel_w
        snap_x_max = self.left + (col_max + 1) * self.pixel_w
        
        snap_y_min = self.top + row_min * self.pixel_h
        snap_y_max = self.top + (row_max + 1) * self.pixel_h

        # Re-sort to preserve mouse drag orientation mapping internally
        new_x0 = snap_x_min if x0 <= x1 else snap_x_max
        new_x1 = snap_x_max if x0 <= x1 else snap_x_min
        new_y0 = snap_y_min if y0 <= y1 else snap_y_max
        new_y1 = snap_y_max if y0 <= y1 else snap_y_min
        
        # Send snapped coordinates down to base artist renderer
        # super(PixelSnapSelector, type(self)).extents.fset(self, (new_x0, new_x1, new_y0, new_y1))
        # self._rect_bbox = [new_x0, new_x1, new_y0, new_y1]
        # self._extents_on_press = self.extents
        # self._selection_artist.get_bbox().bounds = (new_x0, new_x1, new_y0, new_y1)
        super(InclusivePixelSnapSelector, InclusivePixelSnapSelector).extents.fset(
            self, (new_x0, new_x1, new_y0, new_y1)
        )