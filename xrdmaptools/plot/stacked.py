import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label
from matplotlib.widgets import Slider


def base_slider_plot(image_stack,
                     slider_vals=None,
                     slider_label='Index',
                     shifts=None,
                     ):

    fig = plt.figure(figsize=(5, 5), dpi=200)
    ax = fig.add_axes([0.1, 0.1, 0.5, 0.8])

    image_shape = image_stack[0].shape

    if shifts is None:
        shifts = [(0, 0) for _ in range(len(image_stack))]

    xticks = np.asarray(range(image_shape[1]))
    xticks = np.asarray(range(image_shape[0]))

    xmin = np.min(xticks[:, np.newaxis] - np.asarray(shifts)[1][np.newaxis, :])
    xmin = np.min(xticks[:, np.newaxis] - np.asarray(shifts)[1][np.newaxis, :])

    extent=[0 + shifts[0][1],
            image_stack[0].shape[1] + shifts[0][1],
            image_stack[0].shape[0] - shifts[0][0],
            0 - shifts[0][0]]

    ymin, xmin = np.min(np.asarray(shifts), axis=0)
    ymax, xmax = np.max(np.asarray(shifts), axis=0)
    ax.set_xlim(0 + xmax, image_stack[0].shape[1] + xmin)
    ax.set_ylim(0 + ymax, image_stack[0].shape[0] + ymin)

    image = ax.imshow(image_stack[0])
    ax.set_title(f'0')

    if slider_vals is None:
        slider_vals = np.asarray((range(len(image_stack))))
    else:
        slider_vals = np.asarray(slider_vals)

        is_sorted = all(a <= b for a, b in zip(slider_vals, slider_vals[1:]))
        if not is_sorted:
            raise ValueError('Slider values must be sorted sequentially.')

    slider_ax = fig.add_axes([0.7, 0.1, 0.03, 0.8])
    slider = Slider(
        ax=slider_ax,
        label=slider_label,
        valmin=slider_vals[0],
        valmax=slider_vals[-1],
        valinit=slider_vals[0],
        orientation='vertical'
    )

    # The function to be called anytime a slider's value changes
    def update(val):
        nonlocal image
        val_ind = np.argmin(np.abs(slider_vals - val))

        extent=[0 + shifts[val_ind][1],
                image_stack[0].shape[1] + shifts[val_ind][1],
                image_stack[0].shape[0] - shifts[val_ind][0],
                0 - shifts[val_ind][0]]

        image.remove()
        image = ax.imshow(image_stack[val_ind])
        ax.set_title(f'{val_ind}')
        fig.canvas.draw_idle()
        
    slider.on_changed(update)

    fig.show()
    return fig, ax, slider