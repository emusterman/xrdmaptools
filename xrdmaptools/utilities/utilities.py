import numpy as np
import time as ttime
import os
from tqdm.dask import TqdmCallback
from tqdm import tqdm
import warnings
import functools
from scipy.spatial import distance_matrix
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

def check_ext(path,
              ext,
              check_exists=True):
    root, found_ext = os.path.splitext(path)

    if isinstance(ext, str):
        ext = [ext]
    elif not isinstance(ext, list):
        ext = list(ext)

    # Add periods for redundancy
    ext = [e if e[0] == '.' else f'.{e}' for e in ext]

    if not check_exists:
        if found_ext == '':
            if len(ext) > 1:
                warn_str = ('WARNING: More than one extension '
                            + 'accepted. Defaulting to: '
                            + f'{ext[0]}')
                print(warn_str)
            return path + ext[0]
        elif found_ext in ext:
            return path
        else:
            err_str = (f'{path} input extension does not match '
                       + f'required extension: {ext}')
            raise ValueError(err_str)
    
    else:
        err_str = ''
        if found_ext in ext:
            if os.path.exists(path):
                return path
            err_str = (f'Specified file does not exist: {path}')
        elif found_ext != '':
            err_str = (f'Specified file extension {found_ext} is '
                       + 'not accepted.')

        for e in ext:
            if os.path.exists(root + e):
                if err_str != '':
                    err_str = (f'WARNING: {err_str}'
                               + '\nAnother acceptable file found '
                               + f'instead. Using: {root + e}')
                    print(err_str)
                return root + e
        if err_str != '':
            err_str += '\n'
        err_str += (f'Could not find file {root} with default '
                    + f'extensions in: {ext}')
        # err_str += ('No file exists with default extensions '
        #             + f'in: {ext}')
        raise FileNotFoundError(err_str)
        

def pathify(directory,
            filename,
            ext,
            check_exists=True):
    
    directory = os.path.normpath(directory)
    path = os.path.join(directory, filename)
    path = check_ext(path,
                     ext,
                     check_exists=check_exists)
    
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

# I imagine this function is a simple named clustering alogorithm
# I'm not sure what it would be though.
# Just a very convenient function for mapping connected clusters
# Works fairly fast, but calculating distances takes significant memory
def label_nearest_spots(spots, max_dist=25, max_neighbors=np.inf):
    spots = np.asarray(spots)
    dist = distance_matrix(spots)

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


def rescale_array(arr,
                  lower=0,
                  upper=1,
                  arr_min=None,
                  arr_max=None,
                  mask=None,
                  copy=False):
    # Works for arrays of any size including images!

    if copy:
        arr = arr.copy()

    if mask is not None:
        arr[~mask] = np.nan
    if arr_min is None:
        arr_min = np.nanmin(arr)
    if arr_max is None:
        arr_max = np.nanmax(arr)
        if upper is None:
            upper = arr_max
    
    ext = upper - lower
    
    # Copied array operation
    #scaled_arr = lower + ext * ((arr - arr_min) / (arr_max - arr_min))
    
    # In-place operation. Much faster
    arr -= arr_min
    arr /= (arr_max - arr_min)
    arr *= ext
    arr += lower

    if mask is not None:
        arr[~mask] = 0

    return arr # I don't really need to return the array after this...


def generate_intensity_mask(intensity, intensity_cutoff=0):

        int_mask = (intensity
                    >= np.min(intensity) + intensity_cutoff
                    * (np.max(intensity) - np.min(intensity)))

        return int_mask


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


def _check_dict_key(dict, key):
    return key in dict.keys() and key is not None


# Helper class for iterating though XRDMaps
class Iterable2D:

    def __init__(self, shape):
        self.len = np.prod(shape)
        self.shape = shape


    def __iter__(self):
        for index in range(self.len):
            yield np.unravel_index(index, self.shape)
        
    
    def __len__(self):
        return self.len


def get_vector_map_feature(vector_map,
                           feature_function=len,
                           dtype=float):

    feature_map = np.empty(vector_map.shape, dtype=dtype)

    for indices in Iterable2D(feature_map.shape):
        feature_map[indices] = feature_function(vector_map[indices])
    
    return feature_map

# def get_vector_map_feature(vector_map,
#                            feature_function=len,
#                            dtype=float):
#     feature_map = np.empty(vector_map.shape, dtype=dtype)

#     for index in range(np.prod(vector_map.shape)):
#         indices = np.unravel_index(index, vector_map.shape)
#         feature_map[indices] = feature_function(vector_map[indices])

#     return feature_map

get_int_vector_map = lambda vm : get_vector_map_feature(vm, feature_function=np.sum)
get_max_vector_map = lambda vm : get_vector_map_feature(vm, feature_function=np.max)
get_num_vector_map = lambda vm : get_vector_map_feature(vm, dtype=int)


# Class for timing large for loops with lots of print statements that do not play nicely with tqdm
# There probably is a way to do this with tqdm, but I have used variations of this method several times
# Example:
# for _ in timed_iter(iterable):
#   some code...
class timed_iter(object):
    import time

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
        
        p_t_time, p_t_fin = self._get_string_dt(self.dt)
        print(f'{self.iter_name.capitalize()} {self.index} took {p_t_time} time. {iter_rem} / {self.total} {self.iter_name}(s) remaining. ')

        # Average time per iteration
        if iter_rem != 0:
            avg_dt = self.tot_dt / self.index
            self.t_rem = avg_dt * iter_rem
            p_t_time, p_t_fin = self._get_string_dt(self.t_rem)

            print(f'Estimated {p_t_time} time remaining to complete at {p_t_fin}.')
            print('#' * 72)
        else:
            self._print_final()

    def _print_final(self):

        p_t_time, p_t_fin = self._get_string_dt(self.tot_dt)
        # Overwrite finish time, since self.tot_dt is elapsed, not predicted
        p_t_fin = self.time.strftime('%d %b %Y %H:%M:%S', self.time.localtime())

        print(f'Completed {self.total} / {self.total} {self.iter_name}(s) in {p_t_time} time at {p_t_fin}')

    def _get_string_dt(self, dt):
        # Separate days for special handling.
        # Should be a better way to do this...
        dt_days = int(dt / 86400)
        dt_rem = dt % 86400

        # Only hours, minutes, and seconds    
        p_t_time = self.time.strftime('%H:%M:%S', self.time.gmtime(dt_rem))

        # One day
        if dt_days == 1:
            p_t_time = f'{dt_days} day and ' + p_t_time

        # Multiple days
        elif dt_days > 1:
            p_t_time = f'{dt_days} days and ' + p_t_time

        # Time of completion
        t_now = self.time.localtime()
        t_fin = self.time.localtime(self.time.mktime(self.time.localtime()) + dt)

        # Check to see if new day
        if t_now[7] == t_fin[7]:
            p_t_fin = self.time.strftime('%H:%M:%S',
                                         self.time.localtime(self.time.mktime(self.time.localtime())
                                                             + dt))
        else:
            p_t_fin = self.time.strftime('%d %b %Y %H:%M:%S',
                                         self.time.localtime(self.time.mktime(self.time.localtime())
                                                             + dt))
            
        return p_t_time, p_t_fin
