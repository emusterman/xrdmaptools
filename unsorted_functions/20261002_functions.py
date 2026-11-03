


# Testing Graph theory
import networkx as nx



def graph_from_pairs(pairs):

    # Find all unique graph points and construct from edges
    point_conversions = [] # (spot_ind, ref_ind)
    edge_list = []
    for pair in pairs:
        edge_points = []
        for point_values in pair:
            if tuple(point_values) not in point_conversions:
                point_conversions.append(tuple(point_values))
            edge_points.append(point_conversions.index(tuple(point_values)))
        edge_list.append(tuple(edge_points))

    G = nx.Graph(edge_list)
    cliques = list(nx.find_cliques(G))

    return G


def graph_from_pairs(pairs):
    pairs = np.asarray(pairs)
    
    # 1. Map each unique (spot_ind, ref_ind) assignment to a unique point_conversion node
    point_conversions = []
    
    # Flatten the 3D array to easily extract all [spot, ref] rows
    flat_mappings = pairs.reshape(-1, 2)
    for row in flat_mappings:
        point = (int(row[0]), int(row[1]))
        if point not in point_conversions:
            point_conversions.append(point)
            
    # 2. Replicate your edge creation loop using the new structure
    edge_list = []
    for pair in pairs:
        # Extract the two explicit points making up this specific pair
        pt_a = (int(pair[0, 0]), int(pair[0, 1]))
        pt_b = (int(pair[1, 0]), int(pair[1, 1]))
        
        # Look up their index locations exactly like your original code did
        idx_a = point_conversions.index(pt_a)
        idx_b = point_conversions.index(pt_b)
        
        if idx_a != idx_b:  # Avoid self-loops
            edge_list.append((idx_a, idx_b))
        
    # 3. Generate the Graph object
    G = nx.Graph(edge_list)
    
    return G, point_conversions








def connections_to_graph(pairs):

    point_conversions = [] # (spot_ind, ref_ind)

    # Find all graph points
    for spot_ind in range(pairs.shape[1]):
        ref_inds = np.unique(pairs[:, spot_ind][~np.isnan(pairs[:, spot_ind])])

        for ref_ind in ref_inds:
            point_conversions.append((int(spot_ind), int(ref_ind)))

    # Decomposes connections into points and find edges
    edge_list= []
    for pair in pairs:
        spot_inds = np.nonzero(~np.isnan(pair))[0]
        ref_inds = pair[spot_inds]

        points = [point_conversions.index((int(spot_ind), int(ref_ind)))
                  for spot_ind, ref_ind in zip(spot_inds, ref_inds)]
        
        edge_list.append(tuple(points))

        # G.add_edge(points)
    
    G = nx.Graph(edge_list)
    cliques = list(nx.find_cliques(G))

    qofs = []
    point_conversions = np.asarray(point_conversions)
    for clique in tqdm(cliques):

        spot_inds, ref_inds = point_conversions[clique].T

        conn_spots = all_spot_qs[spot_inds]
        conn_refs = all_ref_qs[ref_mask][ref_inds]
        orientation, _ = Rotation.align_vectors(conn_spots,
                                                conn_refs)
        all_rot_qs = orientation.apply(all_ref_qs[ref_mask], inverse=False)
        temp_qmask = qmask.generate(all_rot_qs)

        qof = get_quality_of_fit(
            all_spot_qs[spot_inds], # fit_spot_qs
            all_spot_ints[spot_inds], # fit_spot_ints
            all_rot_qs[ref_inds], # fit_rot_qs
            all_ref_fs[ref_mask][ref_inds], # fit_ref_fs
            all_spot_qs, # all_spot_qs
            all_spot_ints, # all_spot_ints
            all_rot_qs[temp_qmask], # all_rot_qs
            all_ref_fs[ref_mask][temp_qmask], # all_ref_fs
            sigma=0.1)
        qofs.append(qof)
    
    
    return point_conversions, edge_list, cliques, qofs, G


def connection_from_clique(clique, point_conversions, spot_num):

    conn = [np.nan,] * spot_num

    for spot_ind, ref_ind in point_conversions[clique]:
        conn[spot_ind] = ref_ind
    
    return np.asarray(conn)


from xrdmaptools.utilities.utilities import tic, toc
spot_nums = []
times = []

def tic(output=False):
    global _t0
    _t0 = ttime.monotonic()
    if output:
        return _t0


def toc(string='', print_msg=True, output=False):
    global _t0
    dt = ttime.monotonic() - _t0
    s = f'{string}\ndt = {dt:.3f} sec'
    if print_msg:
        print(s, end='\n')
    if output:
        return dt

def timed(func):
    def wrapper(*args, **kwargs):
        tic()
        out = func(*args, **kwargs)
        times.append(toc(output=True, print_msg=False))
        return out
    return wrapper

# from xrdmaptools.reflections.spot_blob_indexing_3D import pair_casting_index_best_grain, pair_casting_index_full_pattern
# pair_casting_index_best_grain = timed(pair_casting_index_best_grain)
# pair_casting_index_full_pattern = timed(pair_casting_index_full_pattern)



def lazy_writing(dset, image_data):
    for row_idx, row in tqdm(enumerate(image_data), total=len(image_data)):
        if isinstance(row, np.ndarray):
            dset[row_idx] = row
        else:
            for img_idx, img in enumerate(row.iter_chunks()):
                mask, comp = row.id.read_direct_chunk((img_idx, 0, 0))
                dset.id.write_direct_chunk((row_idx, img_idx, 0, 0), comp, mask)


def estimate_compression(data):
    _, counts = np.unique(data, return_counts=True)
    probs = counts / data.size
    entropy = -np.sum(probs * np.log2(probs))

    return {
        'entropy_bits' : entropy,
        'estimated_ratio' : 32 / max(entropy, 0.001),
        'should_compress' : entropy < 24
    }



import numpy as np
from tqdm import tqdm

from xrdmaptools.geometry.geometry import modular_azimuthal_shift
from xrdmaptools.utilities.math import rescale_array

def plot_azimuthal_maxima(xdm,
                          tth_range=None,
                          tth_cen=None,
                          tth_ext=0.5,
                          chi_shift=True,
                          min_int=None,
                          scale='linear',
                          facecolor='black',
                          title=None):

    default_title = 'Azimuthal Maxima'
    if tth_range is not None:
        default_title += f'\n2θ = ({tth_range[0]:.2f}-{tth_range[1]:.2f})'
    elif tth_cen is not None:
        if tth_ext is None:
            raise ValueError("Must define 'tth_ext' with 'tth_cen'.")
        tth_range = (tth_cen - tth_ext,
                     tth_cen + tth_ext)
        default_title += f'\n2θ = ({tth_cen:.2f}±{tth_ext:.2f})'
    else:
        raise ValueError("Must define either 'tth_range' or 'tth_cen' and 'tth_ext'.")
    
    if title is None:
        title = default_title 

    tth_mask = ((xdm.tth_arr > tth_range[0])
                & (xdm.tth_arr < tth_range[1]))
    mask_indices = tth_mask.nonzero()

    if chi_shift:
        chi_arr, max_arr, shifted = modular_azimuthal_shift(xdm.chi_arr)
    else:
        chi_arr = xdm.chi_arr
    
    chi_map = np.empty(xdm.map_shape)
    int_map = np.empty(xdm.map_shape)
    for indices in tqdm(xdm.indices):
        
        max_mask_index = np.argmax(xdm.images[indices][tth_mask])
        max_indices = tuple([mask_indices[i][max_mask_index] for i in range(2)])
        chi_map[indices] = chi_arr[max_indices]
        int_map[indices] = xdm.images[indices][max_indices]

    if min_int is not None:
        int_map[int_map < min_int] = np.nan

    int_map = rescale_array(int_map)
    int_map[int_map == 0] = np.nanmin(int_map[int_map > 0])

    if scale == 'linear':
        int_map = rescale_array(int_map)
    elif scale == 'log':
        int_map = rescale_array(int_map)
        int_map[int_map == 0] = np.nanmin(int_map[int_map > 0])
        int_map = np.log(int_map)
        int_map = rescale_array(int_map)
        int_map[int_map == 0] = np.nanmin(int_map[int_map > 0])
    elif scale == 'flat':
        int_map[~np.isnan(int_map)] = 1
    else:
        raise ValueError("Scale must be {'linear' 'log'}")
    int_map[np.isnan(int_map)] = 0

    
    fig, ax = xdm.plot_map(chi_map,
                        #    vmin=np.min(chi_arr),
                        #    vmax=np.max(chi_arr),
                           cmap='hsv',
                           alpha=int_map,
                           title=title,
                           return_plot=True)
    ax.set_facecolor(facecolor)

    chi_label = f'Azimuthal Angle [{xdm.azimuthal_units}]'
    if chi_shift:
        chi_label = 'Shifted ' + chi_label

    cbar = ax.images[-1].colorbar
    cbar.set_label(chi_label, rotation=270, labelpad=20)

    fig.show()

    return chi_map, int_map