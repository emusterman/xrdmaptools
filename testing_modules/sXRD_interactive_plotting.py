import matplotlib.pyplot as plt
import numpy as np
from skimage import restoration

'''
Preliminary interactive plotting for scanning XRD maps from SRX beamline.
Ultimate goal is to allow for quick preliminary analysis during beamtime to allow diagnostic and on-the-fly testing and analysis. 
More formalized, detailed, and conventional analysis will be saved for other modules.
Short-term goal is to write down several iterations of interactive functions, not necessarilly following best coding practices.
This will keep them all in one place to ease access later.
'''

'''
Terms:
tth                 (arr)   List of two theta radial angle values (in degrees or radians). Could be different between 1D and 2D outputs
chi                 (arr)   List of chi azimuthal angle values (in degrees or radians)
integrated_data     (arr)   (x, y, tth) array of integrated mapped data from the pyFAI 1D azimuthal integrator
calibrated_data     (arr)   (x, y, tth, chi) array of calibrated mapped data from pyFAI 2D azimuthal integrator
'''


### Preprocessing Utility Functions ###

def integrated_background_removal(integrated_data, bkg_removal=None, ball_size=None):
    '''
    bkg_removal     (str)   'None', 'local', 'global', or 'both' specifying rolling ball background removel
    ball_size       (float) Ball size for rolling ball algorithm. Default is lenght of tth
    '''

    if ball_size == None:
        ball_size = integrated_data.shape[2]

    if bkg_removal == None:
        return integrated_data
    elif bkg_removal in ['global', 'both']:
        map_bkg = restoration.rolling_ball(np.median(integrated_data, axis=(0, 1)), radius=ball_size)
        integrated_data = integrated_data - map_bkg
    elif bkg_removal in ['local', 'both']:
            plt_bkgs = np.zeros((integrated_data.shape[0] * integrated_data.shape[1], integrated_data.shape[2]))
            for i, pixel in enumerate(integrated_data.reshape(integrated_data.shape[0] * integrated_data.shape[1], integrated_data.shape[2])):
                plt_bkgs[i] = restoration.rolling_ball(pixel, radius=ball_size)
            plt_bkgs = plt_bkgs.reshape(*integrated_data.shape)
            integrated_data = integrated_data - plt_bkgs
    else:
        raise IOError("bkg_removal must be None, 'global', 'local', or 'both'!")
    return integrated_data


def normalize_integrated_data(integrated_data, normalize=None):
    '''
    normalize       (str)   'None', 'full', 'partial' specifying how to normalize the data. 'None' as is, 'full' between (0, 100), 'partial' between (minumum, 100)
    '''

    if normalize == None:
        return integrated_data
    elif normalize == 'full':
        return 100 * (integrated_data - np.min(integrated_data)) / (np.max(integrated_data) - np.min(integrated_data))
    elif normalize == 'partial':
        return 100 * integrated_data / np.max(integrated_data)
    else:
        raise IOError("normalize must be None, 'full', or 'partial'!")


### Utility Plotting Function ###

def update_axes(event, data, tth=None, chi=None, fig=None, axes=None, cmap='viridis', marker_color='red', img_vmin=0, img_vmax=10):
    '''
        
    '''
    # Pull global variables
    global row, col, marker, dynamic_toggle
    col, row = event.xdata, event.ydata
    
    if col <= data.shape[1] and row <= data.shape[0]:
        col = int(np.round(col))
        row = int(np.round(row))
        axes[1].clear()
        if len(data.shape) == 3:
            update_plot(data, tth, axi=axes[1])
        elif len(data.shape) == 4:
            update_img(data, tth, chi, axi=axes[1], cmap=cmap, img_vmin=img_vmin, img_vmax=img_vmax)
        axes[1].set_title(f'Col: {col}, Row: {row}')
        
        update_marker(axi=axes[0], marker_color=marker_color)    
    fig.canvas.draw_idle()

def update_plot(data, tth, axi=None):
    '''
    
    '''

    if len(tth) == len(data[row, col]):
        axi.plot(tth, data[row, col])
    else:
        axi.plot(np.linspace(tth[0], tth[-1], len(data[row, col])), data[row, col])
    axi.set_ylim(np.min(data) - 0.15 * np.abs(np.min(data)), np.max(data) + 0.15 * np.abs(np.max(data)))


def update_img(data, tth, chi, axi=None, cmap='viridis', img_vmin=0, img_vmax=10):
    '''
    
    '''
    global row, col
    extent = [tth[0], tth[-1], chi[0], chi[-1]]
    axi.imshow(data[row, col], extent=extent, aspect='auto', vmin=img_vmin, vmax=img_vmax, cmap=cmap)


def update_marker(axi=None, marker_color='red'):
    '''
    
    '''

    global row, col, marker
    marker.remove()
    marker = axi.scatter(col, row, marker='+', s=25, linewidth=1, color=marker_color)
    if dynamic_toggle:
        marker.set_visible(False)


def display_plot(data, axes=None, display_map=None, display_title=None, cmap='viridis', map_vmin=None, map_vmax=None):
    '''
        
    '''
    # Generate plot
    if display_map != None:
        if map_vmin == None: map_vmin = np.min(display_map)
        if map_vmax == None: map_vmax = np.min(display_map)
        axes[0].imshow(display_map, cmap=cmap, vmin=map_vmin, vmax=map_vmax)
        if display_title != None:
            axes[0].set_title(display_title)
        else:
            axes[0].set_title('Custom Map')
    else:
        if len(data.shape) == 3:
            sum_plot = np.sum(data, axis=2)
        elif len(data.shape) == 4:   
            sum_plot = np.sum(data, axis=(2, 3))
        if map_vmin == None: map_vmin = np.min(sum_plot)
        if map_vmax == None: map_vmax = np.max(sum_plot)
        axes[0].imshow(sum_plot, cmap=cmap, vmin=map_vmin, vmax=map_vmax)
        axes[0].set_title('Summed Intensity')


### Variations of Plotting Functions ###

def interactive_1d_plot(integrated_data, tth, 
                        bkg_removal=None, ball_size=None, normalize=None,
                        display_map=None, display_title=None,
                        map_vmin=None, map_vmax=None,
                        cmap='viridis', marker_color='red'):
    '''
    
    '''

    # Remove background
    integrated_data = integrated_background_removal(integrated_data, bkg_removal=bkg_removal, ball_size=ball_size)

    # Normalize data
    integrated_data = normalize_integrated_data(integrated_data, normalize=normalize)

    # Generate plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=200)
    display_plot(integrated_data, axes=ax, display_map=display_map, display_title=display_title, cmap=cmap, map_vmin=map_vmin, map_vmax=map_vmax)

    # Plot display map with marker
    global marker, dynamic_toggle
    marker = ax[0].scatter(0, 0)
    marker.set_visible(False)
    dynamic_toggle = False

    # Make interactive
    def onclick(event):
        if event.inaxes == ax[0]:
            global dynamic_toggle, marker
            update_axes(event, integrated_data, tth=tth, fig=fig, axes=ax, cmap=cmap, marker_color=marker_color)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)



def interactive_2d_plot(calibrated_data, tth, chi,
                        display_map=None, display_title=None,
                        map_vmin=None, map_vmax=None,
                        cmap='viridis', marker_color='red',
                        img_vmin=0, img_vmax=10):
    '''
    
    '''

    # Generate plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=200)
    display_plot(calibrated_data, axes=ax, display_map=display_map, display_title=display_title, cmap=cmap, map_vmin=map_vmin, map_vmax=map_vmax)

    # Plot display map with marker
    global marker, dynamic_toggle
    marker = ax[0].scatter(0, 0)
    marker.set_visible(False)
    dynamic_toggle = False

    # Make interactive
    def onclick(event):
        if event.inaxes == ax[0]:
            global dynamic_toggle, marker
            update_axes(event, calibrated_data, tth=tth, chi=chi, fig=fig, axes=ax,
                        cmap=cmap, marker_color=marker_color, img_vmin=img_vmin, img_vmax=img_vmax)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)


def interactive_combined_plot(integrated_data, calibrated_data, tth, chi,
                              bkg_removal=None, ball_size=None, normalize=None,
                              display_map=None, display_title=None,
                              map_vmin=None, map_vmax=None,
                              cmap='viridis', marker_color='red',
                              img_vmin=0, img_vmax=10):
    '''
    
    '''

    # Remove background
    integrated_data = integrated_background_removal(integrated_data, bkg_removal=bkg_removal, ball_size=ball_size)

    # Normalize data
    integrated_data = normalize_integrated_data(integrated_data, normalize=normalize)

    # Generate plot
    fig = plt.figure(figsize=(8, 7), dpi=200)
    subfigs = fig.subfigures(1, 2, wspace=0.1)
    ax = [subfigs[0].subplots(1, 1),
          *subfigs[1].subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)]
    subfigs[1].subplots_adjust(hspace=0)
    display_plot(integrated_data, axes=ax, display_map=display_map, display_title=display_title, cmap=cmap, map_vmin=map_vmin, map_vmax=map_vmax)

    # Plot display map with marker
    global marker, dynamic_toggle
    marker = ax[0].scatter(0, 0)
    marker.set_visible(False)
    dynamic_toggle = False

    # Make interactive
    def onclick(event):
        if event.inaxes == ax[0]:
            global dynamic_toggle, marker
            update_axes(event, integrated_data, tth=tth, fig=fig, axes=[ax[0], ax[2]], cmap=cmap, marker_color=marker_color)
            update_axes(event, calibrated_data, tth=tth, chi=chi, fig=fig, axes=ax,
                        cmap=cmap, marker_color=marker_color, img_vmin=img_vmin, img_vmax=img_vmax)
            ax[2].set_title('')          

    cid = fig.canvas.mpl_connect('button_press_event', onclick)


def dynamic_1d_plot(integrated_data, tth, 
                    bkg_removal=None, ball_size=None, normalize=None,
                    display_map=None, display_title=None,
                    map_vmin=None, map_vmax=None,
                    cmap='viridis', marker_color='red'):
    '''
    
    '''

    # Remove background
    integrated_data = integrated_background_removal(integrated_data, bkg_removal=bkg_removal, ball_size=ball_size)

    # Normalize data
    integrated_data = normalize_integrated_data(integrated_data, normalize=normalize)

    # Generate plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=200)
    display_plot(integrated_data, axes=ax, display_map=display_map, display_title=display_title, cmap=cmap, map_vmin=map_vmin, map_vmax=map_vmax)

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
                update_axes(event, integrated_data, tth=tth, fig=fig, axes=ax, cmap=cmap, marker_color=marker_color)
                fig.canvas.draw_idle()

    binding_id = plt.connect('motion_notify_event', onmove)


def dynamic_2d_plot(calibrated_data, tth, chi,
                        display_map=None, display_title=None,
                        cmap='viridis', marker_color='red',
                        map_vmin=None, map_vmax=None,
                        img_vmin=0, img_vmax=10):
    '''
    
    '''

    # Generate plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=200)
    display_plot(calibrated_data, axes=ax, display_map=display_map, display_title=display_title, cmap=cmap, map_vmin=map_vmin, map_vmax=map_vmax)

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
                update_axes(event, calibrated_data, tth=tth, chi=chi, fig=fig, axes=ax,
                            cmap=cmap, marker_color=marker_color, img_vmin=img_vmin, img_vmax=img_vmax)
                fig.canvas.draw_idle()

    binding_id = plt.connect('motion_notify_event', onmove)


def dynamic_combined_plot(integrated_data, calibrated_data, tth, chi,
                              bkg_removal=None, ball_size=None, normalize=None,
                              display_map=None, display_title=None,
                              map_vmin=None, map_vmax=None,
                              cmap='viridis', marker_color='red',
                              img_vmin=0, img_vmax=10):
    '''
    
    '''

    # Remove background
    integrated_data = integrated_background_removal(integrated_data, bkg_removal=bkg_removal, ball_size=ball_size)

    # Normalize data
    integrated_data = normalize_integrated_data(integrated_data, normalize=normalize)

    # Generate plot
    fig = plt.figure(figsize=(8, 7), dpi=200)
    subfigs = fig.subfigures(1, 2, wspace=0.1)
    ax = [subfigs[0].subplots(1, 1),
          *subfigs[1].subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)]
    subfigs[1].subplots_adjust(hspace=0)
    display_plot(integrated_data, axes=ax, display_map=display_map, display_title=display_title, cmap=cmap, map_vmin=map_vmin, map_vmax=map_vmax)

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
                update_axes(event, integrated_data, tth=tth, fig=fig, axes=[ax[0], ax[2]], cmap=cmap, marker_color=marker_color)
                update_axes(event, calibrated_data, tth=tth, chi=chi, fig=fig, axes=ax,
                            cmap=cmap, marker_color=marker_color, img_vmin=img_vmin, img_vmax=img_vmax)
                ax[2].set_title('')  

    binding_id = plt.connect('motion_notify_event', onmove)


def interactive_dynamic_1d_plot(integrated_data, tth, 
                                bkg_removal=None, ball_size=None, normalize=None,
                                display_map=None, display_title=None,
                                map_vmin=None, map_vmax=None,
                                cmap='viridis', marker_color='red'):
    '''
    
    '''

    # Remove background
    integrated_data = integrated_background_removal(integrated_data, bkg_removal=bkg_removal, ball_size=ball_size)

    # Normalize data
    integrated_data = normalize_integrated_data(integrated_data, normalize=normalize)

    # Generate plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=200)
    display_plot(integrated_data, axes=ax, display_map=display_map, display_title=display_title, cmap=cmap, map_vmin=map_vmin, map_vmax=map_vmax)

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
            update_axes(event, integrated_data, tth=tth, fig=fig, axes=ax, cmap=cmap, marker_color=marker_color)

    # Make dynamic
    def onmove(event):
        global dynamic_toggle
        if dynamic_toggle:
            if event.inaxes == ax[0]:
                update_axes(event, integrated_data, tth=tth, fig=fig, axes=ax, cmap=cmap, marker_color=marker_color)
                fig.canvas.draw_idle()


    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    binding_id = plt.connect('motion_notify_event', onmove)



def interactive_dynamic_2d_plot(calibrated_data, tth, chi,
                        display_map=None, display_title=None,
                        map_vmin=None, map_vmax=None,
                        cmap='viridis', marker_color='red',
                        img_vmin=0, img_vmax=10):
    '''
    
    '''

    # Generate plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=200)
    display_plot(calibrated_data, axes=ax, display_map=display_map, display_title=display_title, cmap=cmap, map_vmin=map_vmin, map_vmax=map_vmax)

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
            update_axes(event, calibrated_data, tth=tth, chi=chi, fig=fig, axes=ax,
                        cmap=cmap, marker_color=marker_color, img_vmin=img_vmin, img_vmax=img_vmax)
    
    # Make dynamic
    def onmove(event):
        global dynamic_toggle
        if dynamic_toggle:
            if event.inaxes == ax[0]:
                update_axes(event, calibrated_data, tth=tth, chi=chi, fig=fig, axes=ax,
                            cmap=cmap, marker_color=marker_color, img_vmin=img_vmin, img_vmax=img_vmax)
                fig.canvas.draw_idle()
    
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    binding_id = plt.connect('motion_notify_event', onmove)


def interactive_dynamic_combined_plot(integrated_data, calibrated_data, tth, chi,
                              bkg_removal=None, ball_size=None, normalize=None,
                              display_map=None, display_title=None,
                              map_vmin=None, map_vmax=None,
                              cmap='viridis', marker_color='red',
                              img_vmin=0, img_vmax=10):
    '''
    
    '''

    # Remove background
    integrated_data = integrated_background_removal(integrated_data, bkg_removal=bkg_removal, ball_size=ball_size)

    # Normalize data
    integrated_data = normalize_integrated_data(integrated_data, normalize=normalize)

    # Generate plot
    fig = plt.figure(figsize=(8, 7), dpi=200)
    subfigs = fig.subfigures(1, 2, wspace=0.1)
    ax = [subfigs[0].subplots(1, 1),
          *subfigs[1].subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)]
    subfigs[1].subplots_adjust(hspace=0)
    display_plot(integrated_data, axes=ax, display_map=display_map, display_title=display_title, cmap=cmap, map_vmin=map_vmin, map_vmax=map_vmax)

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
            update_axes(event, integrated_data, tth=tth, fig=fig, axes=[ax[0], ax[2]], cmap=cmap, marker_color=marker_color)
            update_axes(event, calibrated_data, tth=tth, chi=chi, fig=fig, axes=ax,
                        cmap=cmap, marker_color=marker_color, img_vmin=img_vmin, img_vmax=img_vmax)
            ax[2].set_title('')      

    # Make dynamic
    def onmove(event):
        global dynamic_toggle
        if dynamic_toggle:
            if event.inaxes == ax[0]:
                update_axes(event, integrated_data, tth=tth, fig=fig, axes=[ax[0], ax[2]], cmap=cmap, marker_color=marker_color)
                update_axes(event, calibrated_data, tth=tth, chi=chi, fig=fig, axes=ax,
                            cmap=cmap, marker_color=marker_color, img_vmin=img_vmin, img_vmax=img_vmax)
                ax[2].set_title('')  

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    binding_id = plt.connect('motion_notify_event', onmove)