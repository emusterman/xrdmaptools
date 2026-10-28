import numpy as np
import functools
from scipy.optimize import curve_fit
from scipy.spatial import KDTree, distance_matrix
from tqdm import tqdm
import dask
from tqdm.dask import TqdmCallback

# Local Imports
from xrdmaptools.utilities.math import (
    arbitrary_center_of_mass,
)
from xrdmaptools.geometry.geometry import get_3D_single_crystal_Lorentz
from xrdmaptools.reflections.spot_models import load_spot_function


def _enable_subsample(func):

    """
    Allow subsampling of 3D vectors.
    """

    @functools.wraps(func)
    def wrapped(*args,
                subsample=1,
                max_dist=0.01,
                verbose=False,
                **kwargs):

        # Subsample guard block
        if (not isinstance(subsample, int)
            or subsample < 1):
            err_str = ("'subsample' kwarg must be whole number "
                    + "greater than zero.")
            raise ValueError(err_str)
        
        # Subsample
        num_vectors = len(args[0])
        down_args = [arg[::subsample] for arg in args]

        labels = func(*down_args,
                      verbose=verbose,
                      max_dist=max_dist,
                      **kwargs)

        if subsample > 1:
            kdtree = KDTree(down_args[0])
            indices = list(range(num_vectors))[::subsample]
            full_labels = np.empty(num_vectors)
            full_labels[:] = np.nan
            full_labels[indices] = labels

            # Construct iterable
            if verbose:
                print('Upsampling data...')
                iterable = tqdm(range(len(full_labels)))
            else:
                iterable = range(len(full_labels))
            
            relabels, reindices = [], []
            for i in iterable:
                if np.isnan(full_labels[i]):
                    dist, nn_idxs = kdtree.query(args[0][i],
                                        k=np.max([subsample, 20]),
                                        distance_upper_bound=max_dist * 2)
                    
                    # Remove infinities
                    nn_idxs = nn_idxs[~np.isinf(dist)]
                    dist = dist[~np.isinf(dist)]

                    # Remove nan labels
                    dist[np.isnan(full_labels[nn_idxs])] = np.nan

                    if not all(np.isnan(dist)):
                        # full_labels[i] = full_labels[nn_idxs[np.nanargmin(dist)]]
                        reindex = nn_idxs[np.nanargmin(dist)]
                        reindices.append(reindex)
                        relabels.append(full_labels[reindex])
            
            # Only put labels in afterwards to avoid order-dependent growth
            full_labels[reindices] = relabels

        else:
            full_labels = labels
        
        # # Re-label remaining nans as new blob. Downsize datatype
        # full_labels[np.isnan(full_labels)] = np.nanmax(full_labels) + 1
        # full_labels = full_labels.astype(np.uint16)

        return full_labels
    
    return wrapped


# 3D analog of blob_search
@_enable_subsample
def rsm_blob_search(q_vectors,
                    max_dist=0.01,
                    max_neighbors=4,
                    subsample=1,
                    verbose=True):

    if max_neighbors < 4:
        warn_str = ('WARNING: max_neighbors < 4 can lead to unexpected'
                    + ' behavior.')
        print(warn_str)

    labels = np.array([np.nan,] * len(q_vectors))
    next_label = 0
    kdtree = KDTree(q_vectors)

    @dask.delayed
    def get_nearby_label(q):
        nonlocal next_label
        dist, nn_idxs = kdtree.query(q,
                                k=max_neighbors + 1,
                                distance_upper_bound=max_dist)
        
        nn_idxs = nn_idxs[~np.isinf(dist)]
        dist = dist[~np.isinf(dist)]
        dist[dist == 0] = np.nan

        nan_mask = ~np.isnan(labels[nn_idxs])

        # If nothing is labeled, add new label to all
        if not np.any(nan_mask):
            labels[nn_idxs] = next_label
            next_label += 1
        
        else:
            found_labels = np.unique(labels[nn_idxs])
            found_labels = found_labels[~np.isnan(found_labels)]
            min_label = np.min(found_labels)

            labels[nn_idxs] = min_label
            for label in found_labels:
                labels[labels == label] = min_label

    delayed_list = []
    for i in range(len(q_vectors)):
        # Check to see if already labeled
        if not np.isnan(labels[i]):
            continue
        delayed_list.append(get_nearby_label(q_vectors[i]))

    if verbose:
        print('Finding blobs...')
        with TqdmCallback(tqdm_class=tqdm):
                dask.compute(*delayed_list)
    else:
        dask.compute(*delayed_list)

    # Re-label to ordered sequential
    # Could possibly be faster, but not too slow already
    unique_labels = np.unique(labels)
    new_labels = range(len(unique_labels))
    for unique_label, new_label in zip(unique_labels, new_labels):
        labels[labels == unique_label] = new_label
      
    return labels


# 3D analog of spot search
@_enable_subsample
def rsm_spot_search(q_vectors,
                    intensity,
                    max_dist=0.025,
                    significance=0.1,
                    subsample=1,
                    verbose=False):

    labels = np.array([np.nan,] * len(q_vectors))
    next_label = 0
    kdtree = KDTree(q_vectors)

    mutable_intensity = intensity.copy()

    if verbose:
        print('Finding spots...')
        iterable = tqdm(range(len(mutable_intensity)))
    else:
        iterable = range(len(mutable_intensity))

    for i in iterable:
        max_index = np.nanargmax(mutable_intensity)
        if not np.isnan(labels[max_index]):
            mutable_intensity[max_index] = np.nan
            continue

        nn_idxs = kdtree.query_ball_point(q_vectors[max_index], r=max_dist)

        if np.all(np.isnan(labels[nn_idxs])):
            labels[nn_idxs] = next_label
            next_label += 1

        else:
            # Find intensities
            found_labels = np.unique(labels[nn_idxs])
            found_labels = found_labels[~np.isnan(found_labels)]
            local_intensity = np.sum(intensity[nn_idxs]) / len(nn_idxs)
            nearby_intensities = [np.sum(intensity[labels == label])
                                  / np.sum(labels == label)
                                  for label in found_labels]

            # Single round conversion of nearby labels' intensities
            for i, nearby_intensity in enumerate(nearby_intensities):
                if (nearby_intensity < significance
                                        * np.max(nearby_intensities)):
                    convert_idxs = list(np.nonzero(labels
                                                == found_labels[i])[0])
                    labels[convert_idxs] = found_labels[
                                        np.argmax(nearby_intensities)]

            # Re-find nearby intensities
            found_labels = np.unique(labels[nn_idxs])
            found_labels = found_labels[~np.isnan(found_labels)]
            nearby_intensities = [np.sum(intensity[labels == label])
                                  / np.sum(labels == label)
                                  for label in found_labels]

            # Is the local itensity below the significance of nearby?
            if np.max(nearby_intensities) > (local_intensity / significance):
                labels[np.array(nn_idxs)[np.isnan(labels[nn_idxs])]] = found_labels[np.argmax(nearby_intensities)]

            else:
                # Find closest index
                dist = distance_matrix(q_vectors[max_index].reshape(1, -1),
                                       q_vectors[nn_idxs])
                dist[dist == 0] = np.nan
                dist[:, np.isnan(labels[nn_idxs])] = np.nan
                min_index = np.nanargmin(dist[0, :])

                labels[np.array(nn_idxs)[np.isnan(labels[nn_idxs])]] = labels[nn_idxs[min_index]]
        
        mutable_intensity[max_index] = np.nan

    # Re-label to ordered sequential
    # Could possibly be faster, but not too slow already
    unique_labels = np.unique(labels)
    new_labels = range(len(unique_labels))
    for unique_label, new_label in zip(unique_labels, new_labels):
        labels[labels == unique_label] = new_label

    return labels


def rsm_characterize_spots(q_vectors,
                           intensity,
                           labels,
                           spot_model=None,
                           ext=0.15,
                           verbose=False):
    """

    """

    # Parse spot model input
    spot_model = load_spot_function(spot_model)

    # Disregard nans from spots
    countable_labels = np.unique(labels)
    countable_labels = countable_labels[~np.isnan(countable_labels)]

    if verbose:
        print('Fitting vector data to spot model...')    

    spots, spot_intensities, spot_heights = [], [], []
    for val in countable_labels:
        mask = labels == val
        mask_ints = intensity[mask]
        mask_qs = q_vectors[mask].T

        # For simple fitting first
        if spot_model is None:
            spots.append(arbitrary_center_of_mass(mask_ints, *mask_qs))
            spot_intensities.append(np.sum(mask_ints))
            spot_heights.append(np.max(mask_ints))
            continue

        # Ignore spots with two few vectors for fitting
        if mask.sum() <= len(spot_model.par_3d):
            labels[mask] = np.nan
            continue
        
        # 3D distribution without rotation
        static_func = lambda *args : spot_model.func_3d(*args, 0, 0)
        
        # guess
        p0 = [
            np.max(mask_ints),
            *mask_qs[:, np.argmax(mask_ints)],
            *(2.355 * np.sqrt(np.diag(np.cov(mask_qs, aweights=mask_ints)))),
        ]
        low_bounds = [
            0,
            *[p0[ind] - ext * np.abs(p0[ind]) for ind in range(1, 4)],
            0,
            0,
            0
        ]
        high_bounds = [
            10 * p0[0],
            *[p0[ind] + ext * np.abs(p0[ind]) for ind in range(1, 7)]
        ]

        popt, pcov = curve_fit(static_func,
                               [*mask_qs],
                               mask_ints,
                               p0=p0,
                               bounds=(low_bounds, high_bounds))
        
        spots.append(popt[1:4])
        spot_heights.append(popt[0])
        # Count the predicted values instead
        spot_intensities.append(np.sum(static_func([*mask_qs], *popt)))

        # # Estimate hyper volume
        # hyper_V = spot_model.get_hyper_volume(*popt, 0, 0)
        # # Estimate volume of 2 sigma sphere
        # C = 2 * np.sqrt(2 * np.log(2))
        # V_2sigma = (4 / 3) * np.pi * (2 * popt[-3] / C) * (2 * popt[-2] / C) * (2 * popt[-1] / C)
        # delta_V = V_2sigma / (0.74 * len(mask_ints))
        # # Estimate intensity from sampling density
        # spot_intensities.append(hyper_V / delta_V)
    
    return (np.asarray(spots),
            np.asarray(spot_heights),
            np.asarray(spot_intensities))


# Everything in radians!
def rsm_get_spot_stats(spots,
                       spot_heights,
                       spot_intensities,
                       wavelength,
                       theta,
                       rocking_axis,
                       rotation_axis='y',
                       use_stage_rotation=False):

        # Get the Lorentz Corrections and polar angles
        (lorentz,
         tth,
         chi,
         wavelength,
         theta) = get_3D_single_crystal_Lorentz(
                                spots,
                                wavelength,
                                theta,
                                rocking_axis,
                                rotation_axis=rotation_axis,
                                use_stage_rotation=use_stage_rotation,
                                return_stats=True
                                )

        # Create stats dictionary
        stats_dict = {
            'height' : spot_heights, # height is not corrected
            'int_raw' : spot_intensities,
            'int_corr' : spot_intensities * lorentz,
            'lorentz' : lorentz,
            'qx' : spots[:, 0],
            'qy' : spots[:, 1],
            'qz' : spots[:, 2],
            'q_mag' : np.linalg.norm(spots, axis=-1),
            'tth' : tth,
            'chi' : chi,
            'wavelength': wavelength,
            'theta' : theta
            }

        # Sort by corrected intensity
        sorted_indices = sorted(range(len(stats_dict['int_corr'])),
                            key=lambda i: stats_dict['int_corr'][i],
                            reverse=True)
        for key in stats_dict:
            stats_dict[key] = stats_dict[key][sorted_indices]

        return stats_dict


# # 3D analog of spot search
# def old_rsm_spot_search(qs,
#                     intensity,
#                     max_dist=0.025,
#                     significance=0.1,
#                     subsample=1,
#                     label_int_method='sum',
#                     verbose=False):

#     if (not isinstance(subsample, int)
#         or subsample < 1):
#         err_str = ("'subsample' kwarg must be whole number "
#                    + "greater than zero.")
#         raise ValueError(err_str)
    
#     spot_model = None
#     if label_int_method.lower() in {'mean', 'avg', 'average'}:
#         label_int_func = np.mean
#     elif label_int_method.lower() in {'sum', 'total'}:
#         label_int_func = np.sum
#     elif label_int_method.lower() in {'gauss', 'gaussian',
#                                       'lorentz', 'lorentzian',
#                                       'p_voigt', 'pseudovoigt'}:
#         label_int_func = np.sum
#         spot_model = load_spot_function(name)
#     else:
#         err_str = ("Unknown label_int_method. "
#                    + "'mean' or 'sum' are supported.")
#         raise ValueError(err_str)        
    
#     new_qs = qs[::subsample]
#     new_int = intensity[::subsample]

#     labels = np.array([np.nan,] * len(new_qs))
#     next_label = 0
#     kdtree = KDTree(new_qs)

#     mutable_intensity = new_int.copy()

#     if verbose:
#         print('Finding spots...')
#         iterable = tqdm(range(len(mutable_intensity)))
#     else:
#         iterable = range(len(mutable_intensity))

#     for i in iterable:
#         max_index = np.nanargmax(mutable_intensity)
#         if not np.isnan(labels[max_index]):
#             mutable_intensity[max_index] = np.nan
#             continue

#         nn_idxs = kdtree.query_ball_point(new_qs[max_index], r=max_dist)

#         if np.all(np.isnan(labels[nn_idxs])):
#             labels[nn_idxs] = next_label
#             next_label += 1

#         else:
#             # Find intensities
#             found_labels = np.unique(labels[nn_idxs])
#             found_labels = found_labels[~np.isnan(found_labels)]
#             local_intensity = np.sum(new_int[nn_idxs]) / len(nn_idxs)
#             nearby_intensities = [np.sum(new_int[labels == label])
#                                   / np.sum(labels == label)
#                                   for label in found_labels]

#             # Single round conversion of nearby labels' intensities
#             for i, nearby_intensity in enumerate(nearby_intensities):
#                 # print('First significance call')
#                 if (nearby_intensity < significance
#                                         * np.max(nearby_intensities)):
#                     convert_idxs = list(np.nonzero(labels
#                                                 == found_labels[i])[0])
#                     labels[convert_idxs] = found_labels[
#                                         np.argmax(nearby_intensities)]

#             # Re-find nearby intensities
#             found_labels = np.unique(labels[nn_idxs])
#             found_labels = found_labels[~np.isnan(found_labels)]
#             nearby_intensities = [np.sum(new_int[labels == label])
#                                   / np.sum(labels == label)
#                                   for label in found_labels]

#             # Is the local itensity below the significance of nearby?
#             if np.max(nearby_intensities) > (local_intensity / significance):
#                 # print('Second significance call')
#                 labels[np.array(nn_idxs)[np.isnan(labels[nn_idxs])]] = found_labels[np.argmax(nearby_intensities)]

#             else:
#                 # Find closest index
#                 dist = distance_matrix(new_qs[max_index].reshape(1, -1),
#                                        new_qs[nn_idxs])
#                 dist[dist == 0] = np.nan
#                 dist[:, np.isnan(labels[nn_idxs])] = np.nan
#                 min_index = np.nanargmin(dist[0, :])

#                 labels[np.array(nn_idxs)[np.isnan(labels[nn_idxs])]] = labels[nn_idxs[min_index]]
        
#         mutable_intensity[max_index] = np.nan

#     # Re-label to ordered sequential
#     # Could possibly be faster, but not too slow already
#     unique_labels = np.unique(labels)
#     new_labels = range(len(unique_labels))
#     for unique_label, new_label in zip(unique_labels, new_labels):
#         labels[labels == unique_label] = new_label

#     # Up-sample data back to full range
#     if subsample > 1:
#         full_kdtree = KDTree(qs)
#         indices = list(range(len(qs)))[::subsample]
#         full_labels = np.empty(len(qs))
#         full_labels[:] = np.nan
#         full_labels[indices] = labels

#         # Construct iterable
#         if verbose:
#             print('Upsampling data...')
#             iterable = tqdm(range(len(full_labels)))
#         else:
#             iterable = range(len(full_labels))
        
#         for i in iterable:
#             if np.isnan(full_labels[i]):
#                 dist, nn_idxs = full_kdtree.query(qs[i],
#                                     k=np.max([subsample, 20]),
#                                     distance_upper_bound=max_dist * 2)
                
#                 # Remove infinities
#                 nn_idxs = nn_idxs[~np.isinf(dist)]
#                 dist = dist[~np.isinf(dist)]

#                 # Remove nan labels
#                 dist[np.isnan(full_labels[nn_idxs])] = np.nan

#                 if not all(np.isnan(dist)):
#                     full_labels[i] = full_labels[nn_idxs[np.nanargmin(dist)]]
#     else:
#         full_labels = labels

#     # Disregard nans from spots
#     countable_labels = np.unique(full_labels)
#     countable_labels = countable_labels[~np.isnan(countable_labels)]
    
#     # Get q_vectors and intensity
#     # return full_labels, countable_labels, intensity, qs

#     # TEMP TESTING
#     if verbose:
#         print('Fitting vector data to spot model...')    

#     spots, label_ints, label_maxs = [], [], []
#     for val in countable_labels:
#         mask = full_labels == val
#         mask_ints = intensity[mask]
#         mask_qs = qs[mask].T

#         # For simple fitting first
#         if spot_model is None:
#             spots.append(arbitrary_center_of_mass(mask_ints, *mask_qs))
#             label_ints.append(label_int_func(mask_ints))
#             label_maxs.append(np.max(mask_ints))
#             continue
        
#         # Spot Model fitting : without any rotations

#         # Ignore spots with two few vectors for fitting
#         if mask.sum() <= len(spot_model.par_3d):
#             full_labels[mask] = np.nan
#             continue
        
#         # 3D distribution without rotation
#         fixed_func = lambda *args : spot_model.func_3d(*args, 0, 0)

#         ext = 0.15
#         p0 = [
#             np.max(mask_ints),
#             *mask_qs[:, np.argmax(mask_ints)],
#             *(2.355 * np.sqrt(np.diag(np.cov(mask_qs, aweights=mask_ints)))),
#         ]
#         low_bounds = [
#             0,
#             *[p0[ind] - ext * np.abs(p0[ind]) for ind in range(1, 4)],
#             0,
#             0,
#             0
#         ]
#         high_bounds = [
#             10 * p0[0],
#             *[p0[ind] + ext * np.abs(p0[ind]) for ind in range(1, 7)]
#         ]

#         # return mask_qs, mask_ints, p0, low_bounds, high_bounds

#         popt, pcov = curve_fit(fixed_func,
#                                [*mask_qs],
#                                mask_ints,
#                                p0=p0,
#                                bounds=(low_bounds, high_bounds))
#         spots.append(popt[1:4])
#         label_maxs.append(popt[0])
#         label_ints.append(spotmodel.get_hyper_volume(*popt, 0, 0))


#     # spots = [arbitrary_center_of_mass(intensity[full_labels == val],
#     #                                   *qs[full_labels == val].T)
#     #          for val in countable_labels]
#     # label_ints = [label_int_func(intensity[full_labels == val])
#     #               for val in countable_labels]
#     # label_maxs = [np.max(intensity[full_labels == val])
#     #               for val in countable_labels]

#     # Re-label remaining nans as new blob (without spot info). Downsize datatype
#     if not np.all(np.isnan(full_labels)):
#         full_labels[np.isnan(full_labels)] = np.nanmax(full_labels) + 1
#         full_labels = full_labels.astype(np.uint16)
#     else:
#         full_labels = np.array([])      

#     return full_labels, np.asarray(spots), np.asarray(label_ints), np.asarray(label_maxs)


