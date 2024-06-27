import matplotlib.pyplot as plt
import numpy as np
from skimage import restoration
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
xticks                 (arr)   List of two theta radial angle values (in degrees or radians). Could be different between 1D and 2D outputs
yticks                 (arr)   List of yticks azimuthal angle values (in degrees or radians)
integrated_data     (arr)   (x, y, xticks) array of integrated mapped data from the pyFAI 1D azimuthal integrator
image_data          (arr)   (x, y, xticks, yticks) array of calibrated mapped data from pyFAI 2D azimuthal integrator
'''


### Utility Plotting Function ###

def update_axes(event, data,
                xticks=None, yticks=None,
                fig=None, axes=None,
                cmap='viridis', marker_color='red',
                img_vmin=None, img_vmax=None,
                y_min=None, y_max=None,
                img_norm=Normalize):
    '''
        
    '''

    # Pull global variables
    global row, col, marker, dynamic_toggle
    old_row, old_col = row, col
    col, row = event.xdata, event.ydata

    # Check to pixel is in data
    if col >= data.shape[1] and row >= data.shape[0]:
        return
    
    # Check for new pixel if mouse motion
    col = int(np.round(col))
    row = int(np.round(row))
    if ((event.name == 'motion_notify_event')
        and (old_row == row and old_col == col)):
        return
    
    axes[1].clear()
    if len(data.shape) == 3:
        update_plot(data,
                    xticks,
                    axi=axes[1],
                    y_min=y_min,
                    y_max=y_max)
    elif len(data.shape) == 4:
        update_img(data,
                   xticks,
                   yticks,
                   axi=axes[1],
                   cmap=cmap,
                   img_vmin=img_vmin,
                   img_vmax=img_vmax,
                   img_norm=img_norm)
    axes[1].set_title(f'Row = {row}, Col = {col}')
    #axes[1].set_title(f'Col: {col}, Row: {row}')
    
    update_marker(axi=axes[0], marker_color=marker_color)    
    fig.canvas.draw_idle()


def update_plot(data,
                xticks,
                y_min=None,
                y_max=None,
                axi=None):
    '''
    
    '''

    if y_min is None: y_min = np.min(data) - 0.15 * np.abs(np.min(data))
    if y_max is None: y_max = np.max(data) + 0.15 * np.abs(np.max(data))

    if len(xticks) == len(data[row, col]):
        axi.plot(xticks, data[row, col])
    else:
        axi.plot(np.linspace(xticks[0], xticks[-1], len(data[row, col])), data[row, col])
    axi.set_ylim(y_min, y_max)


def update_img(data,
               xticks, yticks,
               axi=None,
               cmap='viridis',
               img_vmin=None, img_vmax=None,
               img_norm=Normalize):
    '''
    
    '''
    global row, col, cbar
    plot_img = data[row, col]
    extent = [xticks[0], xticks[-1], yticks[0], yticks[-1]]
    
    if img_vmin is None: img_vmin = np.min(plot_img)
    if img_vmax is None: img_vmax = np.max(plot_img)

    #print(f'row is {row}, col is {col}')
    img = axi.imshow(plot_img,
                     extent=extent,
                     aspect='auto',
                     cmap=cmap,
                     norm=img_norm(vmin=img_vmin, vmax=img_vmax))
    #if cbar is not None and cbar.ax._axes is not None:
    #    cax = cbar.ax
    #    print('cbar removed!')
    #    cbar.remove()
    #    cbar = fig.colorbar(img, cax=cax)
    #elif cbar is None:
    #    cbar = fig.colorbar(img, ax=axi)


def update_marker(axi=None,
                  marker_color='red'):
    '''
    
    '''

    global row, col, marker
    marker.remove()
    marker = axi.scatter(col, row, marker='+', s=25, linewidth=1, color=marker_color)
    if dynamic_toggle:
        marker.set_visible(False)


def display_plot(data,
                 axes=None,
                 display_map=None,
                 display_title=None,
                 cmap='viridis',
                 map_vmin=None,
                 map_vmax=None,
                 map_norm=Normalize):
    '''
        
    '''
    # Generate plot
    if display_map is not None:
        if map_vmin is None: map_vmin = np.min(display_map)
        if map_vmax is None: map_vmax = np.max(display_map)

        axes[0].imshow(display_map,
                       cmap=cmap,
                       norm=map_norm(vmin=map_vmin, vmax=map_vmax))
        
        if display_title != None:
            axes[0].set_title(display_title)
        else:
            axes[0].set_title('Custom Map')
    
    else:
        if len(data.shape) == 3:
            sum_plot = np.sum(data, axis=2)
        elif len(data.shape) == 4:   
            sum_plot = np.sum(data, axis=(2, 3))
        if map_vmin is None: map_vmin = np.min(sum_plot)
        if map_vmax is None: map_vmax = np.max(sum_plot)
        axes[0].imshow(sum_plot, cmap=cmap,
                       norm=map_norm(vmin=map_vmin, vmax=map_vmax))
        axes[0].set_title('Summed Intensity')

    
def set_globals(ax):
    # Plot display map with marker
    global row, col, marker, dynamic_toggle, cbar
    marker = ax[0].scatter(0, 0)
    marker.set_visible(False)
    dynamic_toggle = False
    cbar = None
    row, col = -1, -1


### Variations of Plotting Functions ###

def interactive_1d_plot(integrated_data, xticks=None,
                        display_map=None, display_title=None,
                        y_min=None, y_max=None,
                        map_vmin=None, map_vmax=None, map_norm=Normalize,
                        cmap='viridis', marker_color='red'):
    '''
    
    '''
    
    # Check axes range
    if xticks is None:
        xticks = range(integrated_data.shape[-1])

    # Generate plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=200)
    display_plot(integrated_data,
                 axes=ax,
                 display_map=display_map,
                 display_title=display_title,
                 cmap=cmap,
                 map_vmin=map_vmin,
                 map_vmax=map_vmax,
                 map_norm=map_norm)

    # Plot display map with marker
    global marker, dynamic_toggle
    marker = ax[0].scatter(0, 0)
    marker.set_visible(False)
    dynamic_toggle = False

    # Make interactive
    def onclick(event):
        if event.inaxes == ax[0]:
            global dynamic_toggle, marker
            update_axes(event,
                        integrated_data,
                        xticks=xticks,
                        fig=fig,
                        axes=ax, 
                        cmap=cmap,
                        marker_color=marker_color,
                        y_min=y_min,
                        y_max=y_max,
                        map_norm=map_norm)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    return fig, ax


def interactive_2d_plot(image_data, xticks=None, yticks=None,
                        display_map=None, display_title=None,
                        map_vmin=None, map_vmax=None, map_norm=Normalize,
                        cmap='viridis', marker_color='red',
                        img_vmin=None, img_vmax=None, img_norm=Normalize):
    '''
    
    '''

    # Check axes range
    if xticks is None:
        xticks = range(image_data.shape[-1])
    if yticks is None:
        yticks = range(image_data.shape[-2])

    # Generate plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=200)
    display_plot(image_data,
                 axes=ax,
                 display_map=display_map,
                 display_title=display_title,
                 cmap=cmap,
                 map_vmin=map_vmin,
                 map_vmax=map_vmax,
                 map_norm=map_norm)

    # Plot display map with marker
    global marker, dynamic_toggle
    marker = ax[0].scatter(0, 0)
    marker.set_visible(False)
    dynamic_toggle = False

    # Make interactive
    def onclick(event):
        if event.inaxes == ax[0]:
            global dynamic_toggle, marker
            update_axes(event,
                        image_data,
                        xticks=xticks,
                        yticks=yticks,
                        fig=fig,
                        axes=ax,
                        cmap=cmap,
                        marker_color=marker_color,
                        img_vmin=img_vmin,
                        img_vmax=img_vmax,
                        img_norm=img_norm)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    return fig, ax


def interactive_combined_plot(integrated_data, image_data, xticks=None, yticks=None,
                              display_map=None, display_title=None,
                              map_vmin=None, map_vmax=None, map_norm=Normalize,
                              y_min=None, y_max=None,
                              cmap='viridis', marker_color='red',
                              img_vmin=None, img_vmax=None, img_norm=Normalize):
    '''
    
    '''

    # Check axes range
    if xticks is None:
        xticks = range(image_data.shape[-1])
    if yticks is None:
        yticks = range(image_data.shape[-2])

    # Generate plot
    fig = plt.figure(figsize=(8, 7), dpi=200)
    subfigs = fig.subfigures(1, 2, wspace=0.1)
    ax = [subfigs[0].subplots(1, 1),
          *subfigs[1].subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)]
    subfigs[1].subplots_adjust(hspace=0)
    display_plot(integrated_data,
                 axes=ax,
                 display_map=display_map,
                 display_title=display_title,
                 cmap=cmap,
                 map_vmin=map_vmin,
                 map_vmax=map_vmax,
                 map_norm=map_norm)

    # Plot display map with marker
    global marker, dynamic_toggle
    marker = ax[0].scatter(0, 0)
    marker.set_visible(False)
    dynamic_toggle = False

    # Make interactive
    def onclick(event):
        if event.inaxes == ax[0]:
            global dynamic_toggle, marker
            update_axes(event,
                        integrated_data,
                        xticks=xticks,
                        fig=fig,
                        axes=[ax[0], ax[2]],
                        cmap=cmap,
                        marker_color=marker_color,
                        y_min=y_min,
                        y_max=y_max)
            update_axes(event,
                        image_data,
                        xticks=xticks,
                        yticks=yticks,
                        fig=fig,
                        axes=ax,
                        cmap=cmap,
                        marker_color=marker_color,
                        img_vmin=img_vmin,
                        img_vmax=img_vmax,
                        img_norm=img_norm)
            ax[2].set_title('')          

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    return fig, ax


def dynamic_1d_plot(integrated_data, xticks=None,
                    display_map=None, display_title=None,
                    map_vmin=None, map_vmax=None, map_norm=Normalize,
                    y_min=None, y_max=None,
                    cmap='viridis', marker_color='red'):
    '''
    
    '''

    # Check axes range
    if xticks is None:
        xticks = range(integrated_data.shape[-1])

    # Generate plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=200)
    display_plot(integrated_data,
                 axes=ax,
                 display_map=display_map,
                 display_title=display_title,
                 cmap=cmap,
                 map_vmin=map_vmin,
                 map_vmax=map_vmax,
                 map_norm=map_norm)

    # Plot display map with marker
    global marker, dynamic_toggle
    marker = ax[0].scatter(0, 0)
    marker.set_visible(False)
    dynamic_toggle = True

    # Make dynamic
    def onmove(event):
        global dynamic_toggle
        if dynamic_toggle:
            if event.inaxes == ax[0]:
                update_axes(event,
                            integrated_data,
                            xticks=xticks,
                            fig=fig,
                            axes=ax,
                            cmap=cmap,
                            marker_color=marker_color,
                            y_min=y_min,
                            y_max=y_max)
                fig.canvas.draw_idle()

    binding_id = plt.connect('motion_notify_event', onmove)
    return fig, ax


def dynamic_2d_plot(image_data, xticks=None, yticks=None,
                        display_map=None, display_title=None,
                        cmap='viridis', marker_color='red',
                        map_vmin=None, map_vmax=None, map_norm=Normalize,
                        img_vmin=None, img_vmax=None, img_norm=Normalize):
    '''
    
    '''

    # Check axes range
    if xticks is None:
        xticks = range(image_data.shape[-1])
    if yticks is None:
        yticks = range(image_data.shape[-2])

    # Generate plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=200)
    display_plot(image_data,
                 axes=ax,
                 display_map=display_map,
                 display_title=display_title,
                 cmap=cmap,
                 map_vmin=map_vmin,
                 map_vmax=map_vmax,
                 map_norm=map_norm)

    # Plot display map with marker
    global marker, dynamic_toggle
    marker = ax[0].scatter(0, 0)
    marker.set_visible(False)
    dynamic_toggle = True
    
    # Make dynamic
    def onmove(event):
        global dynamic_toggle
        if dynamic_toggle:
            if event.inaxes == ax[0]:
                update_axes(event,
                            image_data,
                            xticks=xticks, 
                            yticks=yticks,
                            fig=fig,
                            axes=ax,
                            cmap=cmap,
                            marker_color=marker_color,
                            img_vmin=img_vmin,
                            img_vmax=img_vmax,
                            img_norm=img_norm)
                fig.canvas.draw_idle()

    binding_id = plt.connect('motion_notify_event', onmove)
    return fig, ax


def dynamic_combined_plot(integrated_data, image_data, xticks=None, yticks=None,
                              display_map=None, display_title=None,
                              map_vmin=None, map_vmax=None, map_norm=Normalize,
                              cmap='viridis', marker_color='red',
                              img_vmin=None, img_vmax=None, img_norm=Normalize,
                              y_min=None, y_max=None):
    '''
    
    '''

    # Check axes range
    if xticks is None:
        xticks = range(image_data.shape[-1])
    if yticks is None:
        yticks = range(image_data.shape[-2])

    # Generate plot
    fig = plt.figure(figsize=(8, 7), dpi=200)
    subfigs = fig.subfigures(1, 2, wspace=0.1)
    ax = [subfigs[0].subplots(1, 1),
          *subfigs[1].subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)]
    subfigs[1].subplots_adjust(hspace=0)
    display_plot(integrated_data,
                 axes=ax,
                 display_map=display_map,
                 display_title=display_title,
                 cmap=cmap,
                 map_vmin=map_vmin,
                 map_vmax=map_vmax,
                 map_norm=map_norm)

    # Plot display map with marker
    global marker, dynamic_toggle
    marker = ax[0].scatter(0, 0)
    marker.set_visible(False)
    dynamic_toggle = True

    # Make dynamic
    def onmove(event):
        global dynamic_toggle
        if dynamic_toggle:
            if event.inaxes == ax[0]:
                update_axes(event,
                            integrated_data,
                            xticks=xticks,
                            fig=fig,
                            axes=[ax[0], ax[2]],
                            cmap=cmap,
                            marker_color=marker_color,
                            y_min=y_min,
                            y_max=y_max)
                update_axes(event,
                            image_data,
                            xticks=xticks,
                            yticks=yticks,
                            fig=fig,
                            axes=ax,
                            cmap=cmap,
                            marker_color=marker_color,
                            img_vmin=img_vmin,
                            img_vmax=img_vmax,
                            img_norm=img_norm)
                ax[2].set_title('')  

    binding_id = plt.connect('motion_notify_event', onmove)
    return fig, ax


def interactive_dynamic_1d_plot(integrated_data, xticks=None,
                                display_map=None, display_title=None,
                                map_vmin=None, map_vmax=None, map_norm=Normalize,
                                y_min=None, y_max=None,
                                cmap='viridis', marker_color='red'):
    '''
    
    '''

    # Check axes range
    if xticks is None:
        xticks = range(integrated_data.shape[-1])

    # Generate plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=200)
    display_plot(integrated_data,
                 axes=ax,
                 display_map=display_map, 
                 display_title=display_title,
                 cmap=cmap,
                 map_vmin=map_vmin,
                 map_vmax=map_vmax,
                 map_norm=map_norm)

    # Plot display map with marker
    #global marker, dynamic_toggle
    #marker = ax[0].scatter(0, 0)
    #marker.set_visible(False)
    #dynamic_toggle = False
    set_globals(ax)

    # Make interactive
    def onclick(event):
        if event.inaxes == ax[0]:
            global dynamic_toggle, marker
            dynamic_toggle = not dynamic_toggle
            update_axes(event,
                        integrated_data,
                        xticks=xticks,
                        fig=fig,
                        axes=ax,
                        cmap=cmap,
                        marker_color=marker_color,
                        y_min=y_min,
                        y_max=y_max)

    # Make dynamic
    def onmove(event):
        global dynamic_toggle
        if dynamic_toggle:
            if event.inaxes == ax[0]:
                update_axes(event,
                            integrated_data,
                            xticks=xticks,
                            fig=fig,
                            axes=ax,
                            cmap=cmap,
                            marker_color=marker_color,
                            y_min=y_min,
                            y_max=y_max)
                fig.canvas.draw_idle()


    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    binding_id = plt.connect('motion_notify_event', onmove)
    return fig, ax


def interactive_dynamic_2d_plot(image_data, xticks=None, yticks=None,
                        display_map=None, display_title=None,
                        map_vmin=None, map_vmax=None, map_norm=Normalize,
                        cmap='viridis', marker_color='red',
                        img_vmin=None, img_vmax=None, img_norm=Normalize):
    '''
    
    '''

    # Check axes range
    if xticks is None:
        xticks = range(image_data.shape[-1])
    if yticks is None:
        yticks = range(image_data.shape[-2])

    # Generate plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=200)
    display_plot(image_data,
                 axes=ax,
                 display_map=display_map,
                 display_title=display_title,
                 cmap=cmap,
                 map_vmin=map_vmin,
                 map_vmax=map_vmax,
                 map_norm=map_norm)

    ## Plot display map with marker
    #global marker, dynamic_toggle
    #marker = ax[0].scatter(0, 0)
    #marker.set_visible(False)
    #dynamic_toggle = False
    set_globals(ax)

    # Make interactive
    def onclick(event):
        if event.inaxes == ax[0]:
            global dynamic_toggle, marker
            dynamic_toggle = not dynamic_toggle
            update_axes(event,
                        image_data,
                        xticks=xticks,
                        yticks=yticks,
                        fig=fig,
                        axes=ax,
                        cmap=cmap,
                        marker_color=marker_color,
                        img_vmin=img_vmin,
                        img_vmax=img_vmax,
                        img_norm=img_norm)
    
    # Make dynamic
    def onmove(event):
        global dynamic_toggle
        if dynamic_toggle:
            if event.inaxes == ax[0]:
                update_axes(event,
                            image_data,
                            xticks=xticks,
                            yticks=yticks,
                            fig=fig,
                            axes=ax,
                            cmap=cmap,
                            marker_color=marker_color,
                            img_vmin=img_vmin,
                            img_vmax=img_vmax,
                            img_norm=img_norm)
                fig.canvas.draw_idle()
    
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    binding_id = plt.connect('motion_notify_event', onmove)
    return fig, ax


def interactive_dynamic_combined_plot(integrated_data, image_data, xticks=None, yticks=None,
                              display_map=None, display_title=None,
                              map_vmin=None, map_vmax=None, map_norm=Normalize,
                              cmap='viridis', marker_color='red',
                              img_vmin=None, img_vmax=None, img_norm=Normalize,
                              y_min=None, y_max=None,):
    '''
    
    '''

    # Check axes range
    if xticks is None:
        xticks = range(image_data.shape[-1])
    if yticks is None:
        yticks = range(image_data.shape[-2])

    # Generate plot
    fig = plt.figure(figsize=(8, 7), dpi=200)
    subfigs = fig.subfigures(1, 2, wspace=0.1)
    ax = [subfigs[0].subplots(1, 1),
          *subfigs[1].subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)]
    subfigs[1].subplots_adjust(hspace=0)
    display_plot(integrated_data,
                 axes=ax,
                 display_map=display_map,
                 display_title=display_title,
                 cmap=cmap,
                 map_vmin=map_vmin,
                 map_vmax=map_vmax,
                 map_norm=map_norm)

    # Plot display map with marker
    global marker, dynamic_toggle
    marker = ax[0].scatter(0, 0)
    marker.set_visible(False)
    dynamic_toggle = False

    # Make interactive
    def onclick(event):
        if event.inaxes == ax[0]:
            global dynamic_toggle, marker
            dynamic_toggle = not dynamic_toggle
            update_axes(event,
                        integrated_data,
                        xticks=xticks,
                        fig=fig,
                        axes=[ax[0], ax[2]],
                        cmap=cmap,
                        marker_color=marker_color,
                        y_min=y_min,
                        y_max=y_max)
            update_axes(event,
                        image_data,
                        xticks=xticks,
                        yticks=yticks,
                        fig=fig,
                        axes=ax,
                        cmap=cmap,
                        marker_color=marker_color,
                        img_vmin=img_vmin,
                        img_vmax=img_vmax,
                        img_norm=img_norm)
            ax[2].set_title('')      

    # Make dynamic
    def onmove(event):
        global dynamic_toggle
        if dynamic_toggle:
            if event.inaxes == ax[0]:
                update_axes(event,
                            integrated_data,
                            xticks=xticks,
                            fig=fig,
                            axes=[ax[0], ax[2]],
                            cmap=cmap,
                            marker_color=marker_color,
                            y_min=y_min,
                            y_max=y_max)
                update_axes(event,
                            image_data,
                            xticks=xticks,
                            yticks=yticks,
                            fig=fig,
                            axes=ax,
                            cmap=cmap,
                            marker_color=marker_color,
                            img_vmin=img_vmin,
                            img_vmax=img_vmax,
                            img_norm=img_norm)
                ax[2].set_title('')  

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    binding_id = plt.connect('motion_notify_event', onmove)
    return fig, ax