import numpy as np
from scipy.spatial import KDTree, distance_matrix
from tqdm import tqdm
import dask
from tqdm.dask import TqdmCallback

# Local Imports
from xrdmaptools.utilities.utilities import arbitrary_center_of_mass



# 3D analog of blob_search
def rsm_blob_search(qs,
                    max_dist=0.01,
                    max_neighbors=4,
                    subsample=1):

    if max_neighbors < 4:
        print('WARNING: max_neighbors < 4 can lead to unexpected behavior.')
    
    new_qs = qs[::subsample]

    labels = np.array([np.nan,] * len(new_qs))
    next_label = 0
    kdtree = KDTree(new_qs)

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
    print('Scheduling blob search...')
    for i in tqdm(range(len(new_qs))):
        # Check to see if already labeled
        if not np.isnan(labels[i]):
            continue
        delayed_list.append(get_nearby_label(new_qs[i]))

    print('Finding blobs...')
    with TqdmCallback(tqdm_class=tqdm):
            dask.compute(*delayed_list)

    # Re-label to ordered sequential
    # Could possibly be faster, but not too slow already
    unique_labels = np.unique(labels)
    new_labels = range(len(unique_labels))
    for unique_label, new_label in zip(unique_labels, new_labels):
        labels[labels == unique_label] = new_label

    if subsample > 1:
        print('Upsampling data...')
        full_kdtree = KDTree(qs)
        indices = list(range(len(qs)))[::subsample]
        full_labels = np.empty(len(qs))
        full_labels[:] = np.nan
        full_labels[indices] = labels
        
        for i in tqdm(range(len(full_labels))):
            if np.isnan(full_labels[i]):
                dist, nn_idxs = full_kdtree.query(qs[i],
                                    k=np.max([max_neighbors + 1, subsample, 20]),
                                    distance_upper_bound=max_dist)
                
                # Remove infinities
                nn_idxs = nn_idxs[~np.isinf(dist)]
                dist = dist[~np.isinf(dist)]

                # Remove nan labels
                dist[np.isnan(full_labels[nn_idxs])] = np.nan

                if not all(np.isnan(dist)):
                    full_labels[i] = full_labels[nn_idxs[np.nanargmin(dist)]]
    else:
        full_labels = labels
      
    return full_labels


# 3D analog of spot search
def rsm_spot_search(qs,
                    intensity,
                    nn_dist=0.025,
                    significance=0.1,
                    subsample=1,
                    label_int_method='mean'):
    
    if label_int_method.lower() in ['mean', 'avg', 'average']:
        label_int_func = np.mean
    elif label_int_method.lower() in ['sum', 'total']:
        label_int_func = np.sum
    else:
        err_str = ("Unknown label_int_method. "
                   + "'mean' or 'sum' are supported.")
        raise ValueError(err_str)        
    
    new_qs = qs[::subsample]
    new_int = intensity[::subsample]

    labels = np.array([np.nan,] * len(new_qs))
    next_label = 0
    kdtree = KDTree(new_qs)

    mutable_intensity = new_int.copy()
    mutable_intensity[np.argmax(mutable_intensity)] = np.nan

    print('Finding spots...')
    for i in tqdm(range(len(mutable_intensity) - 1)):

        max_index = np.nanargmax(mutable_intensity)
        if not np.isnan(labels[max_index]):
            mutable_intensity[max_index] = np.nan
            continue

        nn_idxs = kdtree.query_ball_point(new_qs[max_index], r=nn_dist)

        if np.all(np.isnan(labels[nn_idxs])):
            labels[nn_idxs] = next_label
            next_label += 1

        else:
            # Find intensities
            found_labels = np.unique(labels[nn_idxs])
            found_labels = found_labels[~np.isnan(found_labels)]
            local_intensity = np.sum(new_int[nn_idxs]) / len(nn_idxs)
            nearby_intensities = [np.sum(new_int[labels == label])
                                  / np.sum(labels == label)
                                  for label in found_labels]

            # Single round conversion of nearby labels' intensities
            for i, nearby_intensity in enumerate(nearby_intensities):
                if nearby_intensity < significance * np.max(nearby_intensities):
                    convert_idxs = list(np.nonzero(labels == found_labels[i])[0])
                    labels[convert_idxs] = found_labels[np.argmax(nearby_intensities)]

            # Re-find nearby intensities
            found_labels = np.unique(labels[nn_idxs])
            found_labels = found_labels[~np.isnan(found_labels)]
            nearby_intensities = [np.sum(new_int[labels == label])
                                  / np.sum(labels == label)
                                  for label in found_labels]

            # Is the local itensity below the significance of nearby?
            if np.max(nearby_intensities) > (local_intensity / significance):
                labels[np.array(nn_idxs)[np.isnan(labels[nn_idxs])]] = found_labels[np.argmax(nearby_intensities)]

            else:
                # Find closest index
                dist = distance_matrix(new_qs[max_index].reshape(1, -1),
                                       new_qs[nn_idxs])
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

    if subsample > 1:
        print('Upsampling data...')
        full_kdtree = KDTree(qs)
        indices = list(range(len(qs)))[::subsample]
        full_labels = np.empty(len(qs))
        full_labels[:] = np.nan
        full_labels[indices] = labels
        
        for i in tqdm(range(len(full_labels))):
            if np.isnan(full_labels[i]):
                dist, nn_idxs = full_kdtree.query(qs[i],
                                    k=np.max([subsample, 20]),
                                    distance_upper_bound=nn_dist * 2)
                
                # Remove infinities
                nn_idxs = nn_idxs[~np.isinf(dist)]
                dist = dist[~np.isinf(dist)]

                # Remove nan labels
                dist[np.isnan(full_labels[nn_idxs])] = np.nan

                if not all(np.isnan(dist)):
                    full_labels[i] = full_labels[nn_idxs[np.nanargmin(dist)]]
    else:
        full_labels = labels

    # spots = [arbitrary_center_of_mass(intensity[labels == i], *qs[labels == i].T) for i in range(int(np.max(labels)) + 1)]
    # Ignores nan values
    spots = [arbitrary_center_of_mass(intensity[full_labels == val], *qs[full_labels == val].T) for val in np.unique(full_labels)[:-1]]
    label_ints = [label_int_func(intensity[full_labels == val]) for val in np.unique(full_labels)[:-1]]

    return full_labels, spots, label_ints


def rsm_characterize_spots():
    raise NotImplementedError()