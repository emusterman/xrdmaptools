import numpy as np
import os
import skimage.io
import h5py
import time as ttime
import dask

t0 = 0




def tic(output=False):
    global t0
    t0 = ttime.monotonic()
    if output:
        return t0


def toc(output=False):
    global t0
    dt = ttime.monotonic() - t0
    s = f'dt = {dt:.3f} sec\n'
    print(s, end='')
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
        output = dask.compute(results)

    return output[0]