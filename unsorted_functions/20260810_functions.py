import numpy as np
import os
from skimage import io
from tqdm import tqdm



std_2_fwhm = 2 * np.sqrt(2 * np.log(2))



def characterize_3D_spots(xdms,
                          spots_3D=None,
                          vector_int_cutoff=None,
                          q_target=None,
                          q_cutoff=None,
                          int_cutoff=None):

    if spots_3D is None:
        if not hasattr(xdms, 'spots_3D') or xdms.spots_3D is None:
            err_str = ("XRDMapStack must first find 3D spots before "
                    + "they can be characterized.")
            raise ValueError(err_str)
        else:
            spots_3D = xdms.spots_3D
    
    # Try to find this information
    if vector_int_cutoff is None:
        try:
            xdms.open_xdms_hdf()
            vector_int_cutoff = xdms.xdms_hdf['xrdmapstack/reflections'].attrs['abs_int_cutoff']
        finally:
            xdms.close_xdms_hdf()

    # One more guard block following q-target logic
    if q_target is not None:
        if q_cutoff is None:
            err_str = "q_cutoff required when using q_target."
            raise ValueError(err_str)
    
    # Effective map shape
    map_shape = (np.max(spots_3D['map_y']) + 1,
                 np.max(spots_3D['map_x']) + 1)

    # Pre-fill expected data
    # Zero moment analysis
    int_map = np.empty(map_shape)
    int_map[:] = np.nan
    int_std_map = int_map.copy()
    # First moment analysis
    com_map = np.empty((*map_shape, 3))
    com_map[:] = np.nan
    com_std_map = com_map.copy()
    com_mag_map = int_map.copy()
    com_mag_std_map = com_mag_map.copy()
    # Second moment analysis
    fwhm_map = com_map.copy()
    fwhm_std_map = fwhm_map.copy()

    # Iterate through all pixels
    for index in tqdm(range(np.prod(map_shape))):
        indices = np.unravel_index(index, map_shape)

        # Get vectors used for spot search
        vectors = xdms.xdms_vector_map[indices]
        vectors = vectors[vectors[:, -1] > float(vector_int_cutoff)]
        spots_labels = xdms.xdms_spot_labels_map[indices]
        unique_spots_labels = np.unique(spots_labels)

        # Find spots
        df = xdms.pixel_3D_spots(indices, spots_3D=spots_3D, copied=True)

        if len(df) == 0:
            continue

        q_mag = df['q_mag'].values
        intensity = df['intensity'].values

        # Target a specific hkl ring
        q_mag_mask = np.ones_like(q_mag, dtype=np.bool_)
        if q_target is not None:
            q_mag_mask = np.abs(q_mag - q_target) < q_cutoff
        
        # More intensity filtering
        int_mask = np.ones_like(intensity, dtype=np.bool_)
        if int_cutoff is not None:
            int_mask = intensity > int_cutoff
        
        unique_spots_labels = unique_spots_labels[q_mag_mask & int_mask]
        if len(unique_spots_labels) == 0:
            continue
        
        # Iterate through all found spots and characterize
        int_list, com_list, com_mag_list, fwhm_list = [], [], [], []
        for label in unique_spots_labels:
            label_mask = spots_labels == label
            spot_vectors = vectors[label_mask]

            if len(spot_vectors) <= 2:
                continue

            int_list.append(np.sum(spot_vectors[:, -1]))

            com = xmt.utilities.math.arbitrary_center_of_mass(spot_vectors[:, -1], *spot_vectors[:, :3].T)
            com_list.append(com)
            com_mag_list.append(np.linalg.norm(com))
            
            aweights = xmt.utilities.math.rescale_array(spot_vectors[:, -1], upper=1, lower=0, copy=True)
            std = np.asarray([np.cov(spot_vectors[:, i], aweights=aweights) for i in range(3)])
            fwhm_list.append(std_2_fwhm * std)
            
            if any(np.isnan(std)):
                raise RuntimeError
        
        if len(int_list) > 1:
            int_map[indices] = np.mean(int_list)
            int_std_map[indices] = np.std(int_list)
        
        if len(com_list) > 1:
            com_map[indices] = np.mean(com_list, axis=0)
            com_std_map[indices] = np.std(com_list, axis=0)
            com_mag_map[indices] = np.mean(com_mag_list, axis=0)
            com_mag_std_map[indices] = np.std(com_mag_list, axis=0)
        
        if len(fwhm_list) > 1:
            fwhm_map[indices] = np.mean(fwhm_list, axis=0)
            fwhm_std_map[indices] = np.std(fwhm_list, axis=0)

    return int_map, int_std_map, com_map, com_std_map, com_mag_map, com_mag_std_map, fwhm_map, fwhm_std_map








def convert_1D(data, std=None):

    data_1D = np.nanmean(data, axis=0)
    std_1D = np.asarray([
        np.sqrt(np.mean(np.square(std_list[~np.isnan(std_list)])))
        for std_list in std.T
    ])

    return data_1D, std_1D

def plot_1D(data, std=None, x=None):

    if std is not None:
        data_1D, std_1D = convert_1D(data, std)
    else:
        data_1D = np.nanmean(data, axis=0)

    if x is None:
        x = range(len(data_1D))

    fig, ax = plt.subplots()
    ax.plot(x, data_1D, label='Mean', linewidth=2)
    if std is not None:
        ax.fill_between(x, data_1D - std_1D, data_1D + std_1D, alpha=0.2, label='±1 Std Dev')
    fig.show()
