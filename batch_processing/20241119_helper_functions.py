import numpy as np
import os
from tqdm import tqdm
from skimage import io
import h5py
import time as ttime
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib import patches
from matplotlib.collections import PatchCollection
import matplotlib
from matplotlib import cm
from scipy.spatial.transform import Rotation

from sklearn.decomposition import PCA, NMF

# Local imports
from xrdmaptools.utilities.utilities import (
  rescale_array,
  timed_iter 
)
from xrdmaptools.XRDBaseScan import XRDBaseScan
from xrdmaptools.utilities.utilities import (
    generate_intensity_mask
)
# from xrdmaptools.reflections.spot_blob_indexing_3D import pair_casting_index_full_pattern
from xrdmaptools.reflections.spot_blob_search_3D import rsm_spot_search
from xrdmaptools.crystal.crystal import are_collinear, are_coplanar


def plot_image_gallery(images, titles=None,
                       vmin=None, vmax=None):


    num_images = len(images)
    img_y, img_x = images[0].shape

    if titles is None:
        titles = range(num_images)

    ncols = int(np.round(np.sqrt(num_images * (img_y / img_x))))
    nrows = int(np.ceil(num_images / ncols))

    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True)
    ax = axes.ravel()

    # if vmin is None:
    #     vmin = np.min(images)
    # if vmax is None:
    #     vmax = np.max(images)

    for i  in range(num_images):
        im = ax[i].imshow(images[i], vmin=vmin, vmax=vmax)
        ax[i].set_title(titles[i])
        fig.colorbar(im, ax=ax[i])

    fig.show()


def plot_plot_gallery(ys,
                      x=None,
                      aspect_ratio=2,
                      titles=None
                      ):

    num_plots = len(ys)
    img_x = aspect_ratio
    img_y = 1

    if x is None:
        x = list(range(len(ys[0])))
    
    if titles is None:
        titles = range(num_plots)

    ncols = int(np.round(np.sqrt(num_plots * (img_y / img_x))))
    nrows = int(np.ceil(num_plots / ncols))

    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True)
    ax = axes.ravel()

    for i in range(num_plots):
        ax[i].plot(x, ys[i])
        ax[i].set_title(titles[i])

    fig.show()
    

def get_element_dict(xrf):

    xrf_keys = list(xrf.keys())

    element_keys = []
    for key in xrf_keys:
        parts = key.split('_')

        if parts[0] in el_abr:
            element_keys.append(key)
    
    element_dict = {key : xrf[key] for key in element_keys}
    return element_dict


def do_nmf(data,
           n_components,
           dims=1):
    # dims are taken from the back going forward
    data_shape = data.shape
    num_shape = np.prod(data_shape[:-dims])
    fit_shape = np.prod(data_shape[-dims:])

    # All positive
    data = data.copy()
    data[data < 0] = 0

    nmf = NMF(n_components=n_components).fit(
                    data.reshape(num_shape, fit_shape))

    weights = nmf.transform(
                data.reshape(
                    num_shape,
                    fit_shape)).reshape(
                        *data_shape[:-dims],
                        n_components)

    components = nmf.components_.reshape(
                    n_components,
                    *data_shape[-dims:])

    weights = np.moveaxis(weights,
                          -1,
                          0)

    return weights, components


def do_pca(data,
           n_components,
           dims=1):
    # dims are taken from the back going forward
    data_shape = data.shape
    num_shape = np.prod(data_shape[:-dims])
    fit_shape = np.prod(data_shape[-dims:])

    pca = PCA(n_components=n_components).fit(
                    data.reshape(num_shape, fit_shape))

    weights = pca.transform(
                data.reshape(
                    num_shape,
                    fit_shape)).reshape(
                        *data_shape[:-dims],
                        n_components)

    components = pca.components_.reshape(
                    n_components,
                    *data_shape[-dims:])

    weights = np.moveaxis(weights,
                          -1,
                          0)

    return weights, components


def vector_correlation(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    return np.dot(v1, v2) / (n1 * n2)


def get_correlation_pattern(images, el_map):

    corr_img = np.zeros(images.shape[-2:])
    corr_img[:] = np.nan

    for index in tqdm(range(corr_img.size)):
        indices = np.unravel_index(index, corr_img.shape)
        corr = np.corrcoef(el_map.ravel(),
                           images[:, :, *indices].ravel())

        corr_img[indices] = corr[0, 1]
    
    return corr_img


def get_projection_pattern(images, el_map):

    proj_img = np.zeros(images.shape[-2:])

    for index in tqdm(range(proj_img.size)):
        indices = np.unravel_index(index, proj_img.shape)
        proj = np.dot(el_map.ravel(),
                      images[:, :, *indices].ravel())

        proj_img[indices] = proj
    
    return proj_img


def interactive_rotation_plot(data,
                              fig=None,
                              ax=None,
                              qmask=None,
                              edges=None,
                              **kwargs):

    if fig is not None and ax is not None:
        if ax.name != '3d':
            raise TypeError('Given axes must have 3d projection!')
    else:
        fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
    orig_data = data
    data = data.copy()

    # Colors
    r = [1, 0, 0]
    g = [0, 1, 0]
    b = [0, 0, 1]
    r_colors = np.array([r for _ in range(len(data))])
    b_colors = np.array([b for _ in range(len(data))])
    if qmask is not None:
        colors = r_colors.copy()
        colors[qmask.generate(data)] = g
    else:
        colors = b_colors

    scatter = ax.scatter(*data.T, c=colors, **kwargs)

    if edges is not None:
        for edge in edges:
            ax.plot(*edge.T, c='gray', lw=1)

    slider_lst, update_lst = [], []
    rot_vec = [0, 0, 0]
    slider_vpos = np.linspace(0.1, 0, 3)
    fig.subplots_adjust(bottom=0.2)
    axes = ['X', 'Y', 'Z']
    for i in range(3):

        # Make a horizontal sliders
        slider_ax = fig.add_axes([0.25, slider_vpos[i], 0.5, 0.05])
        angle_slider = Slider(
            ax=slider_ax,
            label=f'{axes[i]} [deg]',
            valmin=0,
            valmax=360,
            valinit=0,
            orientation='horizontal'
        )

        slider_lst.append(angle_slider)
        rot, rot_data = None, None

        # The function to be called anytime a slider's value changes
        def update_factory(i):
            def update(val):
                nonlocal scatter, slider_lst, data, rot, rot_data, rot_vec, colors
                
                # Set other axes to zero
                for ind in range(3):
                    if ind != i and slider_lst[ind].val != 0:
                        data = rot_data
                        slider_lst[ind].reset()
                        rot_vec[ind] = 0
                
                # Get new rotation and redefine data
                rot_vec[i] = val
                rot = Rotation.from_rotvec(rot_vec, degrees=True)
                rot_data = rot.apply(data)

                # Colors
                if qmask is not None:
                    colors = r_colors.copy()
                    colors[qmask.generate(rot_data)] = g
                else:
                    colors = b_colors

                scatter._offsets3d = rot_data.T
                scatter._facecolors = colors
                fig.canvas.draw_idle()
            return update

        update_lst.append(update_factory(i))
        slider_lst[i].on_changed(update_lst[i])

    resetax = fig.add_axes([0.8, 0.05, 0.075, 0.05])
    button = Button(resetax, 'Reset', hovercolor='0.975')

    def reset(event):
        nonlocal scatter, slider_lst, data
        for slider in slider_lst:
            slider.reset()
        data = orig_data

        if qmask is not None:
            colors = r_colors.copy()
            colors[qmask.generate(rot_data)] = g
        else:
            colors = b_colors

        scatter._offsets3d = data.T
        scatter._facecolors = colors
    button.on_clicked(reset)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')

    fig.show()

    return fig, ax, slider_lst, button


def get_spot_num(xdm):

    plot_map = np.zeros(xdm.map_shape)

    for index in range(xdm.num_images):
        indices = np.unravel_index(index, xdm.map_shape)
        pixel_df = xdm.pixel_spots(indices)
        val = len(pixel_df)
        plot_map[indices] = val
    
    return plot_map


def get_spot_int(xdm):

    plot_map = np.zeros(xdm.map_shape)

    for index in range(xdm.num_images):
        indices = np.unravel_index(index, xdm.map_shape)
        pixel_df = xdm.pixel_spots(indices)
        val = sum(pixel_df['guess_int'].values)
        plot_map[indices] = val
    
    return plot_map


def get_spot_int(xdm):

    plot_map = np.zeros(xdm.map_shape)

    for index in range(xdm.num_images):
        indices = np.unravel_index(index, xdm.map_shape)
        pixel_df = xdm.pixel_spots(indices)
        val = sum(pixel_df['guess_int'].values)
        plot_map[indices] = val
    
    return plot_map


def get_spot_tth_integrations(xdm, function=np.max):

    tth_min = np.min(xdm.tth_arr)
    tth_max = np.max(xdm.tth_arr)

    tth_num = int(np.round((tth_max - tth_min)
                                   / xdm.tth_resolution))

    integration_map = np.zeros((*xdm.map_shape, tth_num))

    for indices in tqdm(xdm.indices):
        pixel_df = xdm.pixel_spots(indices)
        tth = pixel_df['guess_cen_tth'].values

        for i, tth_cen in enumerate(
                       np.linspace(tth_min,
                                   tth_max,
                                   tth_num)):
            tth_st = tth_cen - xdm.tth_resolution
            tth_en = tth_cen + xdm.tth_resolution

            tth_mask = (tth > tth_st) & (tth < tth_en)

            if tth_mask.sum() > 0:
                val = function(pixel_df['guess_int'].values[tth_mask])
            else:
                val = 0

            integration_map[(*indices, i)] = val
    
    return integration_map


def get_spot_elements(xdm):

    min_tth = np.min(xdm.tth)
    max_tth = np.max(xdm.tth)
    tth_num = (max_tth - min_tth) / xdm.tth_resolution

    spot_tth = xdm.spots['guess_tth_cen'].values

    for i in range(tth_num):
        tth_mask = (spot_tth > min_tth + i * xdm.tth_resolution
                   & spot_tth <= min_tth + (i + 1) * xdm.tth_resolution)


# Quick throw-away function
def transpose_dictionaries(self):

    self.open_hdf()
    
    if hasattr(self, 'pos_dict'):
        for key in list(self.pos_dict.keys()):
            self.pos_dict[key] = self.pos_dict[key].swapaxes(
                                                    0, 1)
        self.save_sclr_pos('positions',
                            self.pos_dict,
                            self.position_units)
    
    if hasattr(self, 'sclr_dict'):
        for key in list(self.sclr_dict.keys()):
            self.sclr_dict[key] = self.sclr_dict[key].swapaxes(
                                                        0, 1)
        self.save_sclr_pos('scalers',
                           self.sclr_dict,
                           self.scaler_units)
    # Update spot map_indices
    if hasattr(self, 'spots'):
        map_x_ind = self.spots['map_x'].values
        map_y_ind = self.spots['map_y'].values
        self.spots['map_x'] = map_y_ind
        self.spots['map_y'] = map_x_ind
        self.save_spots()

    self.close_hdf()


def get_vector_map_feature(vector_map,
                           feature_function=len,
                           dtype=float):
    feature_map = np.empty(vector_map.shape, dtype=dtype)

    for index in range(np.prod(vector_map.shape)):
        indices = np.unravel_index(index, vector_map.shape)
        feature_map[indices] = feature_function(vector_map[indices])

    return feature_map

get_int_vector_map = lambda vm : get_vector_feature_map(vm, feature_function=np.sum)
fet_num_vector_map = lambda vm : get_vector_feature_map(vm, dtype=int)


# def get_int_vector_map(vector_map):
#     int_map = np.empty(vector_map.shape, dtype=float)

#     for index in range(np.prod(vector_map.shape)):
#         indices = np.unravel_index(index, vector_map.shape)
#         int_map[indices] = float(np.sum(vector_map[indices]))

#     return int_map


# def get_num_vector_map(vector_map):
#     num_map = np.empty(vector_map.shape, dtype=int)

#     for index in range(np.prod(vector_map.shape)):
#         indices = np.unravel_index(index, vector_map.shape)
#         num_map[indices] = len(vector_map[indices])

#     return num_map


def get_connection_map(xrdmapstack, phase,
                       verbose=False):

    vmap = xrdmapstack.xdms_vector_map
    phase.generate_reciprocal_lattice(1.15 * np.linalg.norm(xrdmapstack.q_arr, axis=-1).max())
    # qmask = QMask.from_XRDRockingScan(xdms)
    # phase = xrdmapstack.phases['stibnite']

    spot_map = np.empty(vmap.shape, dtype=object)
    conn_map = np.empty(vmap.shape, dtype=object)
    qofs_map = np.empty(vmap.shape, dtype=object)
    hkls_map = np.empty(vmap.shape, dtype=object)

    if verbose:
        iterable = timed_iter(range(np.prod(vmap.shape)))
    else:
        iterable = tqdm(range(np.prod(vmap.shape)))

    for index in iterable:
        indices = np.unravel_index(index, vmap.shape)

        q_vectors = vmap[indices][:, :-1]
        intensity = vmap[indices][:, -1]

        # int_mask = generate_intensity_mask(intensity,
        #                                    intensity_cutoff=0.05)
        int_mask = intensity > 1
        
        if np.sum(int_mask) > 0:

            # return q_vectors, intensity, int_mask

            (spot_labels,
             spots,
             label_ints) = rsm_spot_search(q_vectors[int_mask],
                                           intensity[int_mask],
                                           nn_dist=0.05,
                                           significance=0.1,
                                           subsample=1)
            
            # print(f'Number of spots is {len(spots)}')
            if len(spots) > 1:
                
                try:
                    (best_connections,
                    best_qofs
                    ) = pair_casting_index_full_pattern(spots,
                                                        phase,
                                                        0.05,
                                                        5,
                                                        xrdmapstack.qmask,
                                                        degrees=True,
                                                        verbose=verbose)
                    hkls = phase.all_hkls
                    # spots = list(spots)

                    # # Not sure where all nan connections are coming from, but reduce them
                    # for i, conn in enumerate(best_connections):
                    #     if np.all(np.isnan(conn)):
                    #         spots[i] = np.asarray([])
                    #         best_connections[i] = np.asarray([])
                    #         best_qofs[i] = np.asarray([])
                    #         break

                    # record hkls
                    # if len(spots) > 1:            
                    #     hkls = phase.all_hkls


                except IndexError:
                    print('INDEX ERROR')
                    print(indices)
                    return spots
                except Exception as e:
                    print('New Exception')
                    print(indices)
                    raise e
            
            else:
                spots = np.asarray([])
                best_connections = np.asarray([])
                best_qofs = np.asarray([])
                hkls = np.asarray([])
        
        else:
            spots = np.asarray([])
            best_connections = np.asarray([])
            best_qofs = np.asarray([])
            hkls = np.asarray([])
        
        spot_map[indices] = np.asarray(spots)
        conn_map[indices] = np.asarray(best_connections)
        qofs_map[indices] = np.asarray(best_qofs)
        hkls_map[indices] = np.asarray(hkls)

    return spot_map, conn_map, qofs_map, hkls_map


# def get_hkls_map(xrdmapstack, phase):

#     vmap = xdms.xdms_vector_map
#     qmask = QMask.from_XRDRockingCurve(xdms)
#     # phase = xrdmapstack.phases['stibnite']

#     hkls_map = np.empty(vmap.shape, dtype=object)

#     for index in tqdm(range(np.prod(vmap.shape))):
#         indices = np.unravel_index(index, vmap.shape)

#         q_vectors = vmap[indices][:, :-1]
#         intensity = vmap[indices][:, -1]

#         # int_mask = generate_intensity_mask(intensity,
#         #                                    intensity_cutoff=0.05)
#         int_mask = intensity > 0.25
        
#         if np.sum(int_mask) > 0:

#             # return q_vectors, intensity, int_mask

#             (spot_labels,
#              spots,
#              label_ints) = rsm_spot_search(q_vectors[int_mask],
#                                            intensity[int_mask],
#                                            nn_dist=0.25,
#                                            significance=0.1,
#                                            subsample=1)
            
#             # print(f'Number of spots is {len(spots)}')
#             if len(spots) > 1:

#                 # Find q vector magnitudes and max for spots
#                 spot_q_mags = np.linalg.norm(spots, axis=1)
#                 max_q = np.max(spot_q_mags)

#                 # Find phase reciprocal lattice
#                 phase.generate_reciprocal_lattice(1.15 * max_q)                
#                 hkls = phase.all_hkls
            
#             else:
#                 hkls = np.asarray([])
        
#         else:
#             hkls = np.asarray([])
        
#         hkls_map[indices] = np.asarray(hkls)

#     return hkls_map


def construct_map(base_map,
                  func=None,
                  dtype=float,
                  verbose=False,
                  **kwargs):

    if func is None:
        def func(inputs):
            if len(inputs) > 0:
                return inputs[0]
            else:
                return np.nan

    derived_map = np.empty(base_map.shape, dtype=dtype)
    derived_map[:] = np.nan

    for index in range(np.prod(base_map.shape)):
        indices = np.unravel_index(index, base_map.shape)
        if verbose:
            try:
                val = func(base_map[indices], **kwargs)
                derived_map[indices] = val
            except:
                print(f'Indices {indices} failed.')
                continue
        else:
            val = func(base_map[indices], **kwargs)
            derived_map[indices] = val
    
    return derived_map


from xrdmaptools.crystal.strain import phase_get_strain_orientation
from xrdmaptools.reflections.spot_blob_indexing_3D import (
    _get_connection_indices
)

def get_ori_map(conn_map, spot_map, hkls_map, phase):

    map_shape = conn_map.shape
    ori_map = np.empty(map_shape, dtype=object)
    e_map = ori_map.copy()
    strained_map = np.empty(map_shape, dtype=object)

    for index in range(np.prod(map_shape)):
        indices = np.unravel_index(index, map_shape)

        hkls = hkls_map[indices]
        spots = spot_map[indices]

        nan_arr = np.eye(3)
        nan_arr[:] = np.nan

        oris = []
        es = []
        strains = []
        for connection in conn_map[indices]:
            spot_inds, ref_inds = _get_connection_indices(
                                    connection)

            # print(indices)
            # if len(spot_inds) > 2 and not are_coplanar(hkls[ref_inds]):
            if len(spot_inds) > 2 and not are_coplanar(hkls[ref_inds]):
                try:
                    e, ori, strained = phase_get_strain_orientation(
                                                        spots[spot_inds],
                                                        hkls[ref_inds],
                                                        phase
                                                        )
                except:
                    print('Strain orientation fitting failed!')
                    print(indices)
                    return spots[spot_inds], hkls[ref_inds]

            
            elif len(ref_inds) > 1 and not are_collinear(hkls[ref_inds]):
                # Align vectors if possible
                ori = Rotation.align_vectors(
                                spots[spot_inds],
                                phase.Q(hkls[ref_inds]))[0].as_matrix()
                # e = nan_arr.copy()
                e = None
                strained = None
            else:
                # ori = nan_arr.copy()
                # e = nan_arr.copy()
                ori = None
                e = None
                strained = None

            oris.append(ori)
            es.append(e)
            strains.append(strained)
        
        # Catch no connections
        if len(conn_map[indices]) == 0:
            # oris.append(nan_arr.copy())
            # es.append(nan_arr.copy())
            oris.append(None)
            es.append(None)
            strains.append(None)

        ori_map[indices] = oris
        e_map[indices] = es
        strained_map[indices] = strains

    return ori_map, e_map, strained_map


def fit_single_connection(spots, hkls, phase):

    # Full 3D lattice fitting - orientation and strain
    if len(spots) > 2 and not are_coplanar(hkls):
        try:
            e, ori, strained = phase_get_strain_orientation(
                                    spots,
                                    hkls,
                                    phase)
        except:
            print('Strain orientation fitting failed!')
            return spots, hkls
    
    # Alignment of two reflections for orientation only
    elif len(spots) > 1 and not are_collinear(hkls):
        ori = Rotation.align_vectors(spots,
                                     phase.Q(hkls))[0].as_matrix()
        e, strained = None, None
    
    # Insufficient information
    else:
        ori, e, strained = None, None, None
    
    return ori, e, strained


def find_spots_and_index_full_map(xdms, phase, verbose=False):

    # Setup container objects. Hard to estimate memory usage
    map_shape = xdms.xdms_vector_map.shape
    ori_map = np.empty(map_shape, dtype=object)
    e_map = ori_map.copy()
    strained_map = ori_map.copy()
    qofs_map = ori_map.copy()

    # Setup major iterable with verbosity
    if verbose:
        iterable = timed_iter(range(np.prod(xdms.xdms_vector_map.shape)))
    else:
        iterable = tqdm(range(np.prod(xdms.xdms_vector_map.shape)))

    # Iterate through each spatial pixel of map
    for index in iterable:
        indices = np.unravel_index(index, xdms.xdms_vector_map.shape)
        oris, es, strains = [], [], []

        # Break down individual vectors
        q_vectors = xdms.xdms_vector_map[indices][:, :-1]
        intensity = xdms.xdms_vector_map[indices][:, -1]

        # Some level of assigning significance
        int_mask = intensity > 0.1
        
        # Find spots if there are vectors to index
        if np.sum(int_mask) > 0:

            (spot_labels,
             spots,
             label_ints) = rsm_spot_search(q_vectors[int_mask],
                                           intensity[int_mask],
                                           nn_dist=0.05,
                                           significance=0.1,
                                           subsample=1)
            

            # Try to index if sufficient spots found
            if len(spots) > 1:
                try:
                    (best_connections,
                    best_qofs
                    ) = pair_casting_index_full_pattern(spots,
                                                        phase,
                                                        0.05,
                                                        5,
                                                        xdms.qmask,
                                                        degrees=True,
                                                        verbose=verbose)

                # Catch a few errors. Should have been solved...
                except IndexError:
                    print('INDEX ERROR')
                    print(indices)
                    return spots
                except Exception as e:
                    print('New Exception')
                    print(indices)
                    raise e
                
                # Trim bad connections that should not have happened...
                bad_conn_mask = np.array([np.sum(np.isnan(conn)) < 2 for conn in best_connections])
                best_connections = list(np.array(best_connections)[~bad_conn_mask])
                best_qofs = list(np.array(best_qofs)[~bad_conn_mask])
                if np.any(bad_conn_mask) and verbose:
                    print(f'Bad connection found at {indices}.')
                
                # Fit connections
                for conn in best_connections:
                    spot_inds, ref_inds = _get_connection_indices(conn)
                    
                    ori, e, strained = fit_single_connection(
                                            spots[spot_inds],
                                            phase.all_hkls[ref_inds],
                                            phase)
                    
                    # Record values
                    oris.append(ori)
                    es.append(e)
                    strains.append(strained)

            
            else:
                # Fill with None placeholder
                oris.append(None)
                es.append(None)
                strains.append(None)
                best_qofs = [None]
        
        else:
            # Fill with None placeholder
            oris.append(None)
            es.append(None)
            strains.append(None)
            best_qofs = [None]
        
        # Fill map pixel
        ori_map[indices] = oris
        e_map[indices] = es
        strained_map[indices] = strains
        qofs_map[indices] = best_qofs

    return ori_map, e_map, strained_map, qofs_map




def restack_map(unstacked_map):

    stacked_map = np.empty((*unstacked_map.shape,
                            *unstacked_map[0, 0].shape))
    
    for index in range(np.prod(unstacked_map.shape)):
        indices = np.unravel_index(index, unstacked_map.shape)

        stacked_map[indices][:] = unstacked_map[indices]
    
    return stacked_map





def plot_derived_map(arr):

    fig, ax = plt.subplots(figsize=(5, 5), dpi=200)

    im = ax.imshow(arr)
    fig.colorbar(im, ax=ax)

    ax.set_aspect('equal')
    ax.set_facecolor('black')

    fig.show()
        





def transform_coords(x, M):
    # append x with 1
    x = [*x, 1]

    return x @ M




# def process_maps_with_null():

#     scanlist = [
#         # 153442,
#         # 153443,
#         153444,
#         153445,
#         153446,
#         153448,
#         153449,
#         153450,
#         153451,
#         153452,
#         153454,
#         153455,
#         153456,
#         153457,
#         153458,
#         153460,
#         153461,
#         153462,
#         153463,
#         153464
#     ]

#     for scan in timed_iter(scanlist):
#         xdm = XRDMap.from_hdf(f'scan{scan}_xrd.h5',
#                               wd=f'{base_wd}processed_xrdmaps/',
#                               image_data_key='raw_images',
#                               integration_data_key=None)
        
#         xdm.construct_null_map()
#         if not np.any(xdm.null_map):
#             note_str = ('Null map is empty, there are no missing '
#                         + 'pixels. Proceeding without changes')
#             print(note_str)
#         else:
#             xdm.load_images_from_hdf(image_data_key='final_images')
#             xdm.nullify_images()
#             xdm.save_images()
#             xdm.vectorize_map_data(rewrite_data=True)
#             xdm.integrate1d_map()
#             xdm.save_integrations()