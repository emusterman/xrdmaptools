import numpy as np
import time as ttime
import os
from tqdm.dask import TqdmCallback
from tqdm import tqdm
import warnings
import functools
from sklearn.metrics.pairwise import euclidean_distances
from scipy.ndimage import sobel

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

# Methods to scrub directory and file inputs

def check_ext(path, ext):
    root, found_ext = os.path.splitext(path)

    if isinstance(ext, str):
        ext = [ext]
    elif not isinstance(ext, list):
        ext = list(ext)

    if found_ext in ext:
        # Maybe check if path exists?
        return path
    elif found_ext not in ext:
        raise ValueError(f'{path} input extension does not match required extension: {ext}')
    elif found_ext == '':
        for ext_i in ext:
            if os.path.exists(path + ext_i):
                return path + ext_i
            else:
                raise FileNotFoundError(f'File extension not specified and cannot find file with default extension: {ext}')
    else:
        raise RuntimeError(f'Unknown issue with file extentsion for {path}')
    

def pathify(directory, filename, ext):
    directory = os.path.normpath(directory)
    path = os.path.join(directory, filename)
    path = check_ext(path, ext)
    return path

    

# Does not work!
def timed_func(func):
    # Time a function
    
    def wrap_func(*args, **kwargs):
        t0 = ttime()
        result = func(*args, **kwargs)
        tf = ttime()
        print(f'Function {func.__name__!r} executed in {(tf - t0):.4f}s')
        return result
    
    return wrap_func


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
    spots = np.asarray(spots)
    dist = euclidean_distances(spots)

    spot_indices = list(range(len(spots)))

    # Ignore distances greater than bounds
    # Ignore same pairs
    dist[dist > max_dist] = np.nan
    #dist[dist == 0] = np.nan
    # Create dataset
    data = np.empty((len(spots), spots.shape[-1] + 1))
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


def combine_nearby_spots(spots, *weights, max_dist, max_neighbors=np.inf):
    # Spots are weighted by the first weight!
    
    labeled_spots = label_nearest_spots(spots, max_dist=max_dist, max_neighbors=max_neighbors)

    combined_spots = []
    combined_weights = []
    for label in np.unique(labeled_spots[:, -1]):
        label_mask = labeled_spots[:, -1] == label
        #combined_spot = np.mean(labeled_spots[label_mask][:, :-1], axis=0)
        combined_spot = arbitrary_center_of_mass(np.squeeze(np.asarray(weights)[0])[label_mask],
                                                  *labeled_spots[label_mask][:, :-1].T)
        combined_spots.append(combined_spot)
        for weight in weights:
            combined_weight = np.sum(weight[label_mask])
            combined_weights.append(combined_weight)

    return combined_spots, combined_weights


def arbitrary_center_of_mass(weights, *args):

    weights = np.asarray(weights)
    for i, arg in enumerate(args):
        arg = np.asarray(arg)
        if weights.shape != arg.shape:
            raise ValueError(f'Shape of arg {i + 1} does not match shape of weights!')
        
    val_list = []
    for arg in args:
        arg = np.asarray(arg)
        val = np.dot(weights.ravel(), arg.ravel()) / np.sum(weights)
        val_list.append(val)

    return tuple(val_list)


def delta_array(arr):
    # Returns the geometric mean of the delta array
    sobel_h = sobel(arr, 0)  # horizontal gradient
    sobel_v = sobel(arr, 1)  # vertical gradient

    # Approximate edges by nearest pixels
    sobel_h[0] = sobel_h[1] # top row
    sobel_h[-1] = sobel_h[-2] # bottom row
    sobel_v[:, 0] = sobel_v[:, 1] # first col
    sobel_v[:, -1] = sobel_v[:, -2] # last col

    magnitude = np.sqrt(sobel_h**2 + sobel_v**2)

    return magnitude


def vector_angle(v1, v2, degrees=False):
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1, axis=-1) *  np.linalg.norm(v2, axis=-1)))
    if degrees:
        angle = np.degrees(angle)
    return angle


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


# Class for timing large for loops with lots of print statements that do not play nicely with tqdm
# There probably is a way to do this with tqdm, but I have used variations of this method several times
# Example:
# for _ in timed_iter(iterable):
#   some code...
class timed_iter(object):
    import time
    #def __new__(cls, *_, **__):
    #    instance = object.__new__(cls)
    #    return instance

    def __init__(self, iterable=None, iter_name=None, total=None):
        object.__init__(self)

        if total is None and iterable is not None:
            total = len(iterable)

        if iter_name is None:
            iter_name = 'iteration'

        self.iterable = iterable
        self.total = total
        self.iter_name = iter_name


    def __iter__(self):
        self.t_start = self.time.monotonic()
        self.index = 1 # start from 1!

        for obj in self.iterable:
            self.t0 = self.time.monotonic()
            self._print_before()
            yield obj
            self.tf = self.time.monotonic()
            self._print_after()
            self.index += 1


    def __len__(self):
        if self.total is not None:
            out_len = self.total
        else:
            out_len = len(self.iterable)

        return out_len


    def _print_before(self):
        print(f'Iterating though {self.iter_name} {self.index}/{self.total}.')


    def _print_after(self):
        # Current elapsed time
        self.dt = self.tf - self.t0
        self.tot_dt = self.tf - self.t_start
        iter_rem = self.total - self.index
        p_iter_time = self.time.strftime('%H:%M:%S', self.time.gmtime(self.dt))
        print(f'{self.iter_name.capitalize()} {self.index} took {p_iter_time} time. {iter_rem} / {self.total} {self.iter_name}(s) remaining. ')

        # Average time per iteration
        if iter_rem != 0:
            avg_dt = self.tot_dt / self.index
            self.t_rem = avg_dt * iter_rem
            p_t_rem = self.time.strftime('%H:%M:%S', self.time.gmtime(self.t_rem))
            p_t_fin = self.time.strftime('%H:%M:%S', self.time.localtime(self.time.mktime(self.time.localtime()) + self.t_rem))
            print(f'Estimated {p_t_rem} time remaining completing at {p_t_fin}.')
            print('#' * 72)
        else:
            self._print_final()

    def _print_final(self):
        p_t_tot = self.time.strftime('%H:%M:%S', self.time.gmtime(self.tot_dt))
        p_t_fin = self.time.strftime('%H:%M:%S', self.time.localtime())
        print(f'Completed {self.total} / {self.total} {self.iter_name}(s) in {p_t_tot} time at {p_t_fin}')
