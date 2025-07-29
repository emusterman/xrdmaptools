import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Local imports
from xrdmaptools.plot import config
from xrdmaptools.plot.general import plot_integration

# Catch all for other non-interactive analysis plots


# Generic waterfall plot
def plot_waterfall(integrations,
                   tth=None,
                   units=None,
                   title=None,
                   v_offset=0.25,
                   fig=None,
                   ax=None,
                   cmap=None,
                   **kwargs
                   ):

    c = None
    if 'c' in kwargs:
        c = kwargs.pop('c')
    if 'color' in kwargs:
        c = kwargs.pop('color')

    if cmap is None:
        if c is None:
            c = 'k'
        colors = [c,] * len(integrations)
    else:
        if c is not None:
            warn_str = ('WARNING: Specifying both line color and cmap'
                        + 'conflict. Defaulting to use cmap only.')
            print(warn_str)
        cmap = mpl.colormaps[cmap]
        colors = cmap(np.linspace(0, 1, len(integrations)))
    
    if title is None:
        title = 'Waterfall Plot'

    if fig is None and ax is None:
        fig, ax = plot_integration(integrations[0],
                                   tth=tth,
                                   units=units,
                                   title=title,
                                   color=colors[0],
                                   **kwargs)
        start_ind = 1
    elif fig is None and ax is not None or fig is not None and ax is None:
        err_str = 'Fig and ax must both be None or both provided.'
        raise ValueError(err_str)
    else:
        start_ind = 0

    if tth is None:
        tth = range(len(intensity))
    
    # Make v_offset relative
    v_offset *= (np.max(integrations) - np.min(integrations))
    
    # Add each integration with offsets
    for i in range(start_ind, len(integrations)):
        ax.plot(tth, integrations[i] + (v_offset * i), color=colors[i], **kwargs)
    
    # Update y limits
    ax.set_yticklabels([])
    ax.relim()
    ax.autoscale()

    return fig, ax


