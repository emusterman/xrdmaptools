import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
import matplotlib.patches as patches


def base_slider_plot(image_stack,
                     slider_vals=None,
                     slider_label='Index',
                     shifts=None,
                     vmin=None,
                     vmax=None,
                     title=None,
                     **kwargs
                     ):

    fig = plt.figure(figsize=(5, 5), dpi=200)
    ax = fig.add_axes([0.1, 0.1, 0.5, 0.8])

    image_shape = image_stack[0].shape

    if shifts is None:
        shifts = [(0, 0),] * len(image_stack)

    x_ticks = np.asarray(range(image_shape[1]))
    y_ticks = np.asarray(range(image_shape[0]))

    x_possible = x_ticks[:, np.newaxis] + np.asarray(shifts)[:, 1][np.newaxis, :]
    x_min = np.min(x_possible) - 0.5
    x_max = np.max(x_possible) + 0.5

    y_possible = y_ticks[:, np.newaxis] - np.asarray(shifts)[:, 0][np.newaxis, :]
    y_min = np.min(y_possible) - 0.5
    y_max = np.max(y_possible) + 0.5

    extent = [0 + shifts[0][1] - 0.5,
              image_stack[0].shape[1] + shifts[0][1] - 0.5,
              0 - shifts[0][0] - 0.5,
              image_stack[0].shape[0] - shifts[0][0] - 0.5]

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    if vmin is None:
        img_vmin = np.min(image_stack)
    else:
        img_vmin = vmin
    if vmax is None:
        img_vmax = np.max(image_stack)
    else:
        img_vmax = vmax

    image = ax.imshow(image_stack[0],
                      extent=extent,
                      vmin=img_vmin,
                      vmax=img_vmax,
                      **kwargs)

    if title is None:
        title = ''

    ax.set_title(f'{title} [0]')

    if np.any(np.asarray(shifts) != 0):
        rect_left = x_max - image_stack[0].shape[1]
        rect_right = x_min + image_stack[0].shape[1]
        rect_bot = y_min + image_stack[0].shape[0]
        rect_top = y_max - image_stack[0].shape[0]
        rect = patches.Rectangle((rect_left, rect_bot),
                                rect_right - rect_left,
                                rect_top - rect_bot,
                                linewidth=1,
                                linestyle='--',
                                edgecolor='r',
                                facecolor='none')
        ax.add_patch(rect)

    if slider_vals is None:
        slider_vals = np.asarray((range(len(image_stack))))
    else:
        slider_vals = np.asarray(slider_vals)

        ascending = all(a < b for a, b in zip(slider_vals, slider_vals[1:]))
        descending = all(a > b for a, b in zip(slider_vals, slider_vals[1:]))

        if descending:
            slider_vals = slider_vals[::-1]
            image_stack = image_stack[::-1]
        elif not (ascending or descending):
            raise ValueError('Slider values must be sorted sequentially.')

    slider_ax = fig.add_axes([0.7, 0.1, 0.03, 0.8])
    slider = Slider(
        ax=slider_ax,
        label=slider_label,
        valmin=slider_vals[0],
        valmax=slider_vals[-1],
        valinit=slider_vals[0],
        valstep=slider_vals,
        orientation='vertical'
    )

    # The function to be called anytime a slider's value changes
    def update(val):
        nonlocal image
        val_ind = np.argmin(np.abs(slider_vals - val))

        extent = [0 + shifts[val_ind][1] - 0.5,
                  image_stack[0].shape[1] + shifts[val_ind][1] - 0.5,
                  0 - shifts[val_ind][0] - 0.5,
                  image_stack[0].shape[0] - shifts[val_ind][0] - 0.5]
        
        image.set_data(image_stack[val_ind])

        if vmin is None:
            img_vmin = np.min(image_stack[val_ind])
        else:
            img_vmin = vmin
        if vmax is None:
            img_vmax = np.max(image_stack[val_ind])
        else:
            img_vmax = vmax
        
        image.set_clim(img_vmin, img_vmax)
        image.set_extent(extent)

        ax.set_title(f'{title} [{val_ind}]')
        fig.canvas.draw_idle()
        
    slider.on_changed(update)

    fig.show()
    return fig, ax, slider