import numpy as np
import os
import skimage.io
import h5py
import time as ttime
import dask
from tqdm.dask import TqdmCallback
from tqdm import tqdm
import warnings
import functools
from sklearn.metrics.pairwise import euclidean_distances

# Must define this as something
_t0 = 0
_verbose = False


def tic(output=False):
    global _t0
    _t0 = ttime.monotonic()
    if output:
        return _t0


def toc(string='', output=False):
    global _t0
    dt = ttime.monotonic() - _t0
    s = f'{string}\ndt = {dt:.3f} sec'
    print(s, end='\n')
    if output:
        return dt


def parallel_loop(function, iterable, *args, **kwargs):
    import dask

    # This may break with enumerate functions...
    
    @dask.delayed
    def sub_function(index):
        return function(index, *args, **kwargs)

    results = []
    for index in tqdm(iterable):
        results.append(sub_function(index, *args, **kwargs))

    #with ProgressBar():
    #    output = dask.compute(results)

    with TqdmCallback():
        output = dask.compute(*results)

    return output


def label_nearest_spots(spots, max_dist=25, max_neighbors=np.inf):
    dist = euclidean_distances(spots)

    spot_indices = list(range(len(spots)))

    # Ignore distances greater than bounds
    # Ignore same pairs
    dist[dist > max_dist] = np.nan
    #dist[dist == 0] = np.nan
    # Create dataset
    data = np.empty((len(spots), 3))
    data[:] = np.nan
    data[:, :-1] = spots

    labels=[0]
    for i in spot_indices:
        # This seems unecessary...
        connections = dist[i] < max_dist

        # Useful for connectivity
        if sum(connections) > max_neighbors:
            connections = dist[i] <= sorted(dist[i][connections])[max_neighbors - 1]
            #break
            
        extra_labels = np.unique(data[connections][:, -1])
        extra_labels = extra_labels[~np.isnan(extra_labels)]

        # Find new label if all points are unlabeled
        if np.all(np.isnan(data[connections][:, -1])):
            current_label = labels[-1] + 1
            data[connections, -1] = current_label
            labels.append(current_label)
            #print('New label added')

        # Otherwise relabel all to lowest label value
        else:
            base_label = np.nanmin(data[connections][:, -1])
            data[connections, -1] = base_label
            current_label = base_label

        
        # Relabel previous labels if connected with others...
        if np.any(extra_labels != current_label):
            for extra in extra_labels:
                if np.all([extra != current_label, 
                        not np.isnan(extra), 
                        extra in labels]):
                    #print('I should be here...')
                    
                    data[[data[:, -1] == extra][0], -1] = current_label
                    labels.remove(extra)

    #data = data.astype(np.int32)
    labels = np.array(labels).astype(np.int32)
    # labels is not perfectly sequential. Why???
    return data


def arbitrary_center_or_mass(weights, *args):

    weights = np.asarray(weights)
    for i, arg in enumerate(args):
        arg = np.asarray(arg)
        if weights.shape != arg.shape:
            raise ValueError(f'Shape of arg {i + 1} does not match shape of weights!')
        
    val_list = []
    for arg in args:
        val = np.dot(weights.ravel(), arg.ravel()) / np.sum(weights)
        val_list.append(val)

    return tuple(val_list)


def vprint(message, **kwargs):
    global _verbose
    if _verbose:
        print(message, **kwargs)



def deprecated(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func