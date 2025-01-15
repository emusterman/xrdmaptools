import numpy as np
import os
import h5py
import psutil
import numpy.core.numeric as _nx
from itertools import product
from collections.abc import Iterable


##################
### HDF Format ###
##################


def check_hdf_current_images(title,
                             hdf_file=None,
                             hdf=None):
    if hdf is None and hdf_file is not None:
        with h5py.File(hdf_file, 'r') as f:
            return_bool = title in f[f'{list(f.keys())[0]}/image_data']
        return return_bool
    elif hdf is not None:
        return title in hdf[f'{list(hdf.keys())[0]}/image_data']
    else:
        raise ValueError('Must specify hdf_file or hdf.')
    

def get_optimal_chunks(data,
                       approx_chunk_size=None,
                       final_dtype=np.float32):
    
    data_shape = data.shape

    if final_dtype is None:
        data_nbytes = data[(0,) * data.ndim].nbytes
    else:
        try:
            np.dtype(final_dtype)
            data_nbytes = final_dtype().itemsize
        except TypeError as e:
            raise e(f'dtype input of {final_dtype} is not numpy datatype.')

    if approx_chunk_size is None:
        available_memory = psutil.virtual_memory()[1] / (2**20) # In MB
        cpu_count = os.cpu_count()
        approx_chunk_size = ((available_memory * 0.85)
                             / cpu_count) # 15% wiggle room
        #approx_chunk_size = np.round(approx_chunk_size)
        if approx_chunk_size > 2**10:
            approx_chunk_size = 2**10
    elif approx_chunk_size > 2**10:
        warn_str = ('WARNING: Chunk sizes above 1 GB may start to '
                    + 'perform poorly.')
        print(warn_str)

    # Split images up by data size in MB that seems reasonable
    images_per_chunk = ((approx_chunk_size * 2**20)
                         / np.prod([*data_shape[-2:],
                                    data_nbytes],
                         dtype=np.int32))

    # Try to make square chunks if possible
    square_chunks = np.sqrt(images_per_chunk)

    num_chunk_x = np.round(data_shape[0]
                           / square_chunks, 0).astype(np.float32)
    num_chunk_x = min(max(num_chunk_x, 1), data_shape[0])
    chunk_x = int(data_shape[0] // num_chunk_x)

    num_chunk_y = np.round(data_shape[1]
                           / (data_shape[0]
                              / num_chunk_x), 0).astype(np.int32)
    num_chunk_y = min(max(num_chunk_y, 1), data_shape[1])
    chunk_y = int(data_shape[1] // num_chunk_y)

    # String togther, maintaining full images per chunk
    chunk_size = (chunk_x, chunk_y, *data_shape[-2:])

    return chunk_size


# Check to see if the hdf attr needs to be overwritten
def check_attr_overwrite(attrs, attr_key, new_attr):

    # Check if attr exists
    if attr_key not in attrs:
        return True
    else:
        attr = attrs[attr_key]
    
    # Check if attr matches new attr
    if (isinstance(attr, Iterable)
        and isinstance(new_attr, Iterable)):
        if (isinstance(attr, str)
            and isinstance(new_attr, str)):
            return attr != new_attr
        elif len(attr) == len(new_attr):
            return np.any([np.any(attr[i] != new_attr[i])
                           for i in range(len(attr))])
        else:
            return True
    elif (not isinstance(attr, Iterable)
          and not isinstance(new_attr, Iterable)):
        return attr != new_attr
    else:
        return True # Overwriting different object types


# Check and overwrite hdf attr only if changed
# Separate from check_attr_overwrite in case of special processing
def overwrite_attr(attrs, attr_key, new_attr):
    if check_attr_overwrite(attrs, attr_key, new_attr):
        # print(f'Overwriting {attr_key} with {new_attr}')
        attrs[attr_key] = new_attr


# Built on improved code from get_optimal_chunks above
# Intended towards fracturing full maps into smaller manageable pieces,
# but not so small as chunks
# An alternative way to avoid using dask
def get_large_map_slices(datamap,
                         approx_new_map_sizes=10, # In GB
                         final_dtype=np.float32):
    
    data_shape = datamap.shape
    
    if final_dtype is None:
        data_nbytes = datamap[(0,) * datamap.ndim].nbytes
    else:
        try:
            np.dtype(final_dtype)
            data_nbytes = final_dtype().itemsize
        except TypeError as e:
            raise e(f'dtype input of {final_dtype} is not numpy datatype.')

    # Split images up by data size in MB that seems reasonable
    images_per_chunk = ((approx_new_map_sizes * 2**30)
                        / np.prod([*data_shape[-2:], data_nbytes],
                                dtype=np.int64))

    # Try to make square chunks if possible
    side_length = np.sqrt(images_per_chunk)

    num_chunks = []
    chunk_sizes = []
    indices = ([], [])
    for axis in range(2):
        num_chunk = np.round(data_shape[axis]
                        / side_length,
                        0).astype(np.int32)
        num_chunk = min(max(num_chunk, 1), data_shape[axis])
        chunk_size = data_shape[axis] // num_chunk
        side_length = chunk_size

        num_chunks.append(num_chunk)
        chunk_sizes.append(chunk_size)

        # From numpy.array_split
        Neach_section, extras = divmod(data_shape[axis], num_chunk)
        section_sizes = ([0]
                        + extras * [Neach_section+1]
                        + (num_chunk - extras) * [Neach_section])
        div_points = _nx.array(section_sizes, dtype=_nx.intp).cumsum()

        for i in range(num_chunk):
            st = div_points[i]
            end = div_points[i + 1]
            indices[axis].append((st, end))

    slicings = list(product(*indices))

    return slicings, num_chunks, chunk_sizes
