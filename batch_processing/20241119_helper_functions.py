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


# elemental names in American English
el_names = [
    'hydrogen', 'helium', 'lithium', 'beryllium', 'boron', 'carbon',
    'nitrogen', 'oxygen', 'fluorine', 'neon', 'sodium', 'magnesium',
    'aluminum', 'silicon', 'phosphorus', 'sulfur', 'chlorine', 'argon',
    'potassium', 'calcium', 'scandium', 'titanium', 'vanadium',
    'chromium', 'manganese', 'iron', 'cobalt', 'nickel', 'copper',
    'zinc', 'gallium', 'germanium', 'arsenic', 'selenium', 'bromine',
    'krypton', 'rubidium', 'strontium', 'yttrium', 'zirconium',
    'niobium', 'molybdenum', 'technetium', 'ruthenium', 'rhodium',
    'palladium', 'silver', 'cadmium', 'indium', 'tin', 'antimony',
    'tellurium', 'iodine', 'xenon', 'cesium', 'barium', 'lanthanum',
    'cerium', 'praseodymium', 'neodymium', 'promethium', 'samarium',
    'europium', 'gadolinium', 'terbium', 'dysprosium', 'holmium',
    'erbium', 'thulium', 'ytterbium', 'lutetium', 'hafnium',
    'tantalum', 'tungsten', 'rhenium', 'osmium', 'iridium', 'platinum',
    'gold', 'mercury', 'thallium', 'lead', 'bismuth', 'polonium',
    'astatine', 'radon', 'francium', 'radium', 'actinium', 'thorium',
    'protactinium', 'uranium', 'neptunium', 'plutonium', 'americium',
    'curium', 'berkelium', 'californium', 'einsteinium', 'fermium',
    'mendelevium', 'nobelium', 'lawrencium', 'rutherfordium',
    'dubnium', 'seaborgium', 'bohrium', 'hassium', 'meitnerium',
    'darmstadtium', 'roentgenium', 'copernicium', 'nihonium',
    'flerovium', 'moscovium', 'livermorium', 'tennessine', 'oganesson'
]

el_abr = [
    'H',  'He', 'Li', 'Be', 'B',  'C',  'N',  'O',  'F',  'Ne', 'Na',
    'Mg', 'Al', 'Si', 'P',  'S',  'Cl', 'Ar', 'K',  'Ca', 'Sc', 'Ti',
    'V',  'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As',
    'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y',  'Zr', 'Nb', 'Mo', 'Tc', 'Ru',
    'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I',  'Xe', 'Cs',
    'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W',  'Re', 'Os', 'Ir',
    'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra',
    'Ac', 'Th', 'Pa', 'U',  'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es',
    'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
    'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Os'
]



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


# TODO: Consider linear interpolation of data, not just nearest
def interpolate_positions(map_shape,
                          shifts,
                          method='nearest', # not currently used...
                          plotme=False):

    # Create generic regular coordinates
    x = np.arange(0, map_shape[1])
    y = np.arange(0, map_shape[0])

    # Redefine y-shifts to match matplotlib axes...
    shifts = np.asarray(shifts)
    shifts[:, 0] *= -1

    # Shifts stats
    x_step = np.mean(np.diff(x))
    y_step = np.mean(np.diff(y))
    ymin, xmin = np.min(shifts, axis=0)
    ymax, xmax = np.max(shifts, axis=0)
    xx, yy = np.meshgrid(x, y[::-1])  # matching matplotlib description

    # Determine virtual grid centers based on mean positions
    mean_shifts = np.mean(shifts, axis=0)
    xx_virt = xx + mean_shifts[1]
    yy_virt = yy + mean_shifts[0]

    # Mask out incomplete virtual pixels
    mask = np.all([xx_virt > np.min(x) + xmax - (x_step / 2), # left edge
                   xx_virt < np.max(x) + xmin + (x_step / 2), # right edge
                   yy_virt > np.min(y) + ymax - (y_step / 2), # bottom edge
                   yy_virt < np.max(y) + ymin + (y_step / 2)], # top edge
                  axis=0)
    xx_virt = xx_virt[mask]
    yy_virt = yy_virt[mask]
    virt_shape = (len(np.unique(yy_virt)), len(np.unique(xx_virt)))
    # print(virt_shape)

    if plotme:
        fig, ax = plt.subplots()

    # Contruct virtual masks of full grids to fill virtual grid
    vmask_list = []
    for i, shift in enumerate(shifts):
        xxi = (xx + shift[1])
        yyi = (yy + shift[0])

        xx_ind, yy_ind = xx_virt[0], yy_virt[0]
        vmask_x0 = np.argmin(np.abs(xxi[0] - xx_ind))
        vmask_y0 = np.argmin(np.abs(yyi[:, 0] - yy_ind))
        y_start = xx.shape[0] - (virt_shape[0] + vmask_y0)
        y_end = xx.shape[0] - vmask_y0
        x_start = vmask_x0
        x_end = vmask_x0 + virt_shape[1]

        # print(vmask_x0, vmask_y0)
        vmask = np.zeros_like(xx, dtype=np.bool_)
        vmask[y_start : y_end,
              x_start : x_end] = True
        vmask_list.append(vmask)

        if plotme:
            ax.scatter(xxi.flatten(),
                       yyi.flatten(),
                       s=5,
                       color=grid_colors[i])

    if plotme:
        ax.scatter(xx_virt,
                   yy_virt,
                   s=20,
                   c='r',
                   marker='*')

        # This can probably be done with RegularPolyCollection but this proved finicky
        rect_list = []
        for xi, yi in zip(xx_virt, yy_virt):
            # Create a Rectangle patch
            rect = patches.Rectangle((xi - (x_step / 2),
                                    yi - (y_step / 2)),
                                    x_step,
                                    y_step,
                                    linewidth=1,
                                    edgecolor='gray',
                                    facecolor='none')
            rect_list.append(rect)
        pc = PatchCollection(rect_list, match_original=True)
        ax.add_collection(pc)

        ax.set_aspect('equal')
        fig.show()

    return vmask_list



# def plot_shifted_points(map_shape, shifts, add_grid=True):

#     x = np.arange(0, map_shape[1])
#     y = np.arange(0, map_shape[0])

#     # Redefine y-shifts to match matplotlib axes...
#     shifts = np.asarray(shifts)
#     shifts[:, 0] *= -1

#     # Get sequential colors for each grid
#     norm = matplotlib.colors.Normalize(vmin=0, vmax=(len(shifts)))
#     mapper = cm.ScalarMappable(norm=norm, cmap='jet')
#     grid_colors = [(r, g, b) for r, g, b, a in mapper.to_rgba(range(len(shifts)))]

#     # Shifts stats
#     x_step = np.mean(np.diff(x))
#     y_step = np.mean(np.diff(y))
#     ymin, xmin = np.min(shifts, axis=0)
#     ymax, xmax = np.max(shifts, axis=0)
#     xx, yy = np.meshgrid(x, y[::-1])  # matching matplotlib description

#     fig, ax = plt.subplots()

#     mean_shifts = np.mean(shifts, axis=0)
#     xx_virt = xx + mean_shifts[1]
#     yy_virt = yy + mean_shifts[0]

#     # Mask out incomplete virtual pixels
#     mask = np.all([xx_virt > np.min(x) + xmax - (x_step / 2), # left edge
#                    xx_virt < np.max(x) + xmin + (x_step / 2), # right edge
#                    yy_virt > np.min(y) + ymax - (y_step / 2), # bottom edge
#                    yy_virt < np.max(y) + ymin + (y_step / 2)], # top edge
#                   axis=0)
#     xx_virt = xx_virt[mask]
#     yy_virt = yy_virt[mask]

#     virt_shape = (len(np.unique(yy_virt)), len(np.unique(xx_virt)))
#     # print(virt_shape)
    
#     vmask_list = []
#     for i, shift in enumerate(shifts):
#         xxi = (xx + shift[1])
#         yyi = (yy + shift[0])

#         xx_ind, yy_ind = xx_virt[0], yy_virt[0]
#         vmask_x0 = np.argmin(np.abs(xxi[0] - xx_ind))
#         vmask_y0 = np.argmin(np.abs(yyi[:, 0] - yy_ind))
#         # print(vmask_x0, vmask_y0)
#         vmask = np.zeros_like(xx, dtype=np.bool_)
#         vmask[vmask_y0 : vmask_y0 + virt_shape[0],
#               vmask_x0 : vmask_x0 + virt_shape[1]] = True
#         vmask_list.append(vmask)

#         ax.scatter(xxi.flatten(),
#                    yyi.flatten(),
#                    s=5,
#                    color=grid_colors[i])
    
#     ax.scatter(xx_virt,
#                yy_virt,
#                s=20,
#                c='r',
#                marker='*')

#     if add_grid: # This can probably be done with RegularPolyCollection but this proved finicky
#         rect_list = []
#         for xi, yi in zip(xx_virt, yy_virt):
#             # Create a Rectangle patch
#             rect = patches.Rectangle((xi - (x_step / 2),
#                                       yi - (y_step / 2)),
#                                      x_step,
#                                      y_step,
#                                      linewidth=1,
#                                      edgecolor='gray',
#                                      facecolor='none')
#             rect_list.append(rect)
#         pc = PatchCollection(rect_list, match_original=True)
#         ax.add_collection(pc)

#     ax.set_aspect('equal')
#     fig.show()
    
#     return vmask_list
#     # return xx_virt.reshape(virt_shape), yy_virt.reshape(virt_shape)
#     # return xx_virt.reshape(virt_shape), yy_virt.reshape(virt_shape)[::-1] # flip y-axes again

@XRDBaseScan.protect_hdf()
def vectorize_map_data(self,
                       image_data_key='recent',
                       keep_images=False):

    # Check required data
    remove_blob_masks_after = False
    if not hasattr(self, 'blob_masks') or self.blob_masks is None:
        if '_blob_masks' in self.hdf[
                            f'{self._hdf_type}/image_data'].keys():
            self.load_images_from_hdf('_blob_masks')
        # For backwards compatibility
        elif '_spot_masks' in self.hdf[
                            f'{self._hdf_type}/image_data'].keys():
            self.load_images_from_hdf('_spot_masks')
        else:
            err_str = (f'{self._hdf_type} does not have blob_masks'
                       + 'attribute.')
            raise AttributeError(err_str)
        remove_blob_masks_after = True

    remove_images_after = False
    if not hasattr(self, 'images') or self.images is None:
        remove_images_after = True
        self.load_images_from_hdf(image_data_key=image_data_key)

    print('Vectorizing data...')
    vector_map = np.empty(self.map_shape, dtype=object)

    for indices in tqdm(self.indices):
        blob_mask = self.blob_masks[indices]
        intensity = self.images[indices][blob_mask]
        q_vectors = self.q_arr[blob_mask]
        
        vector_map[indices] = np.hstack([q_vectors,
                                         intensity.reshape(-1, 1)])

    # Record values
    self.vector_map = vector_map
    
    # Cleaning up XRDMap state
    if not keep_images and remove_images_after:
        self.dump_images()
    if not keep_images and remove_blob_masks_after:
        del self.blob_masks
        self.blob_masks = None


def vectorize_xrdmapstack_data(xrdmapstack, vmask_list, image_data_key='recent'):

    # Quick input check
    for i, xrdmap in enumerate(xrdmapstack):
        active_hdf = xrdmap.hdf is not None
        if not active_hdf:
            xrdmap.open_hdf()

        img_grp = xrdmap.hdf['xrdmap/image_data']

        if (not hasattr(xrdmap, 'q_arr')
             or xrdmap.q_arr is None):
             err_str = f'q_arr not defined for xrdmap[{i}]!'
             raise AttributeError(err_str)
        elif ('_blob_masks' not in img_grp.keys()
              and (not hasattr(xrdmap, 'blob_masks')
                   or xrdmap.blob_masks is None)):
            err_str = f'blob_masks not defined for xrdmap[{i}]!'
            raise AttributeError(err_str)
        
        if not active_hdf:
            xrdmap.close_hdf()

    # Vectorize images
    v_arr_list = []
    for i, xrdmap in timed_iter(enumerate(xrdmapstack),
                                total=len(xrdmapstack),
                                iter_name='XRDMap'):
        print(f'Processing data from scan {xrdmap.scan_id}.')

        vectorize_map_data(xrdmap)

        # Load images
        xrdmap.load_images_from_hdf(image_data_key=image_data_key)
        # Load blob masks
        # Deprecated: _spot_masks key will not be supported
        xrdmap.load_images_from_hdf(image_data_key='_blob_masks')

        v_arr = np.empty(xrdmap.map_shape, dtype=object)

        print('Vectorizing data...')
        for indices in tqdm(xrdmap.indices):
            blob_mask = xrdmap.blob_masks[indices]
            intensity = xrdmap.images[indices][blob_mask]
            q_vectors = xrdmap.q_arr[blob_mask]
            
            v_arr[indices] = np.hstack([q_vectors,
                                        intensity.reshape(-1, 1)])
        
        # Record and then release memory
        v_arr_list.append(v_arr)
        del intensity, q_vectors
        xrdmap.dump_images()
        del xrdmap.blob_masks, blob_mask # Helps with gc
        xrdmap.blob_masks = None
    

    try:
        # Construct full vector array
        print('Combining all vectorized images...', end='', flush=True)
        # vmask_shape = np.max([vmask_list[0].sum(axis=0),
        #                       vmask_list[0].sum(axis=1)],
        #                      axis=1)
        vmask_shape = (np.max(vmask_list[0].sum(axis=0)),
                    np.max(vmask_list[0].sum(axis=1)))

        full_v_arr = np.empty(vmask_shape, dtype=object)

        for v_arr, vmask in zip(v_arr_list, vmask_list):
            virt_index = 0
            for indices in xrdmap.indices:
                # Skip if not in vmask
                if not vmask[indices]:
                    continue
                
                virt_indices = np.unravel_index(virt_index, vmask_shape)

                if full_v_arr[virt_indices] is None:
                    full_v_arr[virt_indices] = v_arr[indices]
                else:
                    full_v_arr[virt_indices] = np.vstack([
                                                full_v_arr[virt_indices],
                                                v_arr[indices]])
                virt_index += 1
    except:
        return v_arr_list, None
    
    # Get edges too
    try:
        edges = get_sampled_edges(xrdmapstack)
    except:
        return full_v_arr, None

    print('done!')
    return full_v_arr, edges


def get_sampled_edges(xrdmapstack):

    edges = ([[] for _ in range(12)])

    for i, xrdmap in enumerate(xrdmapstack):
        q_arr = xrdmap.q_arr

        # Find edges
        if i == 0:
            edges[4] = q_arr[0]
            edges[5] = q_arr[-1]
            edges[6] = q_arr[:, 0]
            edges[7] = q_arr[:, -1]
        elif i == len(xrdmapstack) - 1:
            edges[8] = q_arr[0]
            edges[9] = q_arr[-1]
            edges[10] = q_arr[:, 0]
            edges[11] = q_arr[:, -1]
        # Corners
        edges[0].append(q_arr[0, 0])
        edges[1].append(q_arr[0, -1])
        edges[2].append(q_arr[-1, 0])
        edges[3].append(q_arr[-1, -1])
    
    for i in range(4):
        edges[i] = np.asarray(edges[i])

    return edges


# @XRDBaseScan.protect_hdf()
def save_map_vectorization(self,
                           vector_map=None,
                           edges=None,
                           rewrite_data=False):

    # Adaptability between functions
    hdf = self.xdms_hdf

    # Check input
    if vector_map is None:
        if (hasattr(self, 'vector_map')
            and self.vector_map is not None):
            vector_map = self.vector_map
        else:
            err_str = ('Must provide vector_map or '
                       + f'{self.__class__.__name__} must have '
                       + 'vector_map attribute.')
            raise AttributeError(err_str)

    # Write data to hdf
    print('Saving vectorized map data...')
    vect_grp = hdf[self._hdf_type].require_group(
                                            'vectorized_map')
    vect_grp.attrs['time_stamp'] = ttime.ctime()
    vect_grp.attrs['virtual_shape'] = vector_map.shape

    all_used_indices = [] # For potential vmask shape changes
    for index in range(np.prod(vector_map.shape)):
        indices = np.unravel_index(index, vector_map.shape)
        data = vector_map[indices]
        title = ','.join([str(ind) for ind in indices]) # e.g., '1,2'
        all_used_indices.append(title)

        if title not in vect_grp:
            dset = vect_grp.require_dataset(
                        title,
                        data=data,
                        shape=data.shape,
                        dtype=data.dtype)
        else:
            dset = vect_grp[title]

            if (dset.shape == data.shape
                and dset.dtype == data.dtype):
                dset[...] = data
            else:
                warn_str = 'WARNING:'
                if dset.shape != data.shape:
                    warn_str += (f'{title} shape of'
                                + f' {data.shape} does not '
                                + 'match dataset shape '
                                + f'{dset.shape}. ')
                if dset.dtype != data.dtype:
                    warn_str += (f'{title} dtype of'
                                + f' {data.dtype} does not '
                                + 'match dataset dtype '
                                + f'{dset.dtype}. ')
                if rewrite_data:
                    warn_str += (f'\nOvewriting {title}. This '
                                + 'may bloat the total file size.')
                    print(warn_str)
                    del vect_grp[title]
                    dset = vect_grp.require_dataset(
                        title,
                        data=data,
                        shape=data.shape,
                        dtype=data.dtype)
                else:
                    warn_str += '\nProceeding without changes.'
                    print(warn_str)
    
    # In case virtual shape changed; remove extra datasets
    for dset_key in vect_grp.keys():
        if dset_key not in all_used_indices:
            del vect_grp[dset_key]

    # Only save edge information if given
    if edges is not None:
        edge_grp = vect_grp.require_group('edges')
        edge_grp.attrs['time_stamp'] = ttime.ctime()

        # Check for existenc and compatibility
        for i, edge in enumerate(edges):
            edge = np.asarray(edge)
            edge_title = f'edge_{i}'
            if edge_title not in edge_grp.keys():
                edge_grp.require_dataset(
                    edge_title,
                    data=edge,
                    shape=edge.shape,
                    dtype=edge.dtype)
            else:
                dset = edge_grp[edge_title]

                if (dset.shape == edge.shape
                    and dset.dtype == edge.dtype):
                    dset[...] = edge
                else:
                    warn_str = 'WARNING:'
                    if dset.shape != edge.shape:
                        warn_str += (f'Edge shape for {edge_title}'
                                    + f' {edge.shape} does not '
                                    + 'match dataset shape '
                                    + f'{dset.shape}. ')
                    if dset.dtype != edge.dtype:
                        warn_str += (f'Edge dtype for {edge_title}'
                                    + f' {edge.dtype} does not '
                                    + 'match dataset dtype '
                                    + f'{dset.dtype}. ')
                    if rewrite_data:
                        warn_str += ('\nOvewriting data. This may '
                                    + 'bloat the total file size.')
                        # Shape changes should not happen
                        # except from q_arr changes
                        print(warn_str)
                        del edge_grp[edge_title]
                        edge_grp.require_dataset(
                                edge_title,
                                data=edge,
                                shape=edge.shape,
                                dtype=edge.dtype)
                    else:
                        warn_str += '\nProceeding without changes.'
                        print(warn_str)

# @XRDBaseScan.protect_hdf()
def _load_xrd_hdf_vectorized_map_data(base_grp):

    vector_map = None
    edges = None
    if 'vectorized_map' in base_grp.keys():
        print('Loading vectorized map...', end='', flush=True)
        vector_grp = base_grp['vectorized_map']
        map_shape = vector_grp.attrs['virtual_shape']
        vector_map = np.empty(map_shape, dtype=object)

        for index in range(np.prod(map_shape)):
            indices = np.unravel_index(index, map_shape)
            title = ','.join([str(ind) for ind in indices])
            vector_map[indices] = vector_grp[title][:]
        
        if 'edges' in vector_grp:
            edges = []
            for edge_title, edge_dset in vector_grp['edges'].items():
                edges.append(edge_dset[:])
        print('done!')

    return vector_map, edges


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


def stack_vector_maps(xrdmapstack, vmask_list):
    # Construct full vector array
    print('Combining all vectorized images...')
    # vmask_shape = np.max([vmask_list[0].sum(axis=0),
    #                       vmask_list[0].sum(axis=1)],
    #                      axis=1)
    vmask_shape = (np.max(vmask_list[0].sum(axis=0)),
                   np.max(vmask_list[0].sum(axis=1)))

    full_v_arr = np.empty(vmask_shape, dtype=object)

    for i in range(len(xrdmapstack)):
        # vector_map = xrdmapstack[i].vector_map
        xrdmapstack[i].open_hdf()
        vector_map = _load_xrd_hdf_vectorized_map_data(xrdmapstack[i].hdf['xrdmap'])
        xrdmapstack[i].close_hdf()
        vmask = vmask_list[i]

        virt_index = 0
        for indices in xdms[0].indices:
            # Skip if not in vmask
            if not vmask[indices]:
                continue
            
            virt_indices = np.unravel_index(virt_index, vmask_shape)

            if full_v_arr[virt_indices] is None:
                full_v_arr[virt_indices] = vector_map[indices]
            else:
                full_v_arr[virt_indices] = np.vstack([
                                            full_v_arr[virt_indices],
                                            vector_map[indices]])
            virt_index += 1
        
        # Try to release memory
        del vector_map
    
    edges = get_sampled_edges(xrdmapstack)
    
    print('done!')
    return full_v_arr, edges


def get_int_vector_map(vector_map):
    int_map = np.empty(vector_map.shape, dtype=float)

    for index in range(np.prod(vector_map.shape)):
        indices = np.unravel_index(index, vector_map.shape)
        int_map[indices] = float(np.sum(vector_map[indices]))

    return int_map


def get_num_vector_map(vector_map):
    num_map = np.empty(vector_map.shape, dtype=int)

    for index in range(np.prod(vector_map.shape)):
        indices = np.unravel_index(index, vector_map.shape)
        num_map[indices] = len(vector_map[indices])

    return num_map


def get_connection_map(xrdmapstack):

    vmap = xdms.vector_map
    qmask = QMask.from_XRDRockingCurve(xdms)
    phase = xrdmapstack.phases['stibnite']

    spot_map = np.empty(vmap.shape, dtype=object)
    conn_map = np.empty(vmap.shape, dtype=object)
    qofs_map = np.empty(vmap.shape, dtype=object)

    for index in timed_iter(range(np.prod(vmap.shape))):
        indices = np.unravel_index(index, vmap.shape)

        q_vectors = vmap[indices][:, :-1]
        intensity = vmap[indices][:, -1]

        # int_mask = generate_intensity_mask(intensity,
        #                                    intensity_cutoff=0.05)
        int_mask = intensity > 5
        
        if np.sum(int_mask) > 0:

            # return q_vectors, intensity, int_mask

            (spot_labels,
             spots,
             label_ints) = rsm_spot_search(q_vectors[int_mask],
                                           intensity[int_mask],
                                           nn_dist=0.1,
                                           significance=0.1,
                                           subsample=1)
            
            print(f'Number of spots is {len(spots)}')
            if len(spots) > 1:
                
                try:
                    (best_connections,
                    best_qofs
                    ) = pair_casting_index_full_pattern(spots,
                                                        phase,
                                                        0.1,
                                                        2.5,
                                                        qmask,
                                                        degrees=True)
                except IndexError:
                    print('INDEX ERROR')
                    return spots
            
            else:
                spots = np.asarray([])
                best_connections = np.asarray([])
                best_qofs = np.asarray([])
        
        else:
            spots = np.asarray([])
            best_connections = np.asarray([])
            best_qofs = np.asarray([])
        
        spot_map[indices] = np.asarray(spots)
        conn_map[indices] = np.asarray(best_connections)
        qofs_map[indices] = np.asarray(best_qofs)

    return spot_map, conn_map, qofs_map


def get_hkls_map(xrdmapstack):

    vmap = xdms.vector_map
    qmask = QMask.from_XRDRockingCurve(xdms)
    phase = xrdmapstack.phases['stibnite']

    hkls_map = np.empty(vmap.shape, dtype=object)

    for index in timed_iter(range(np.prod(vmap.shape))):
        indices = np.unravel_index(index, vmap.shape)

        q_vectors = vmap[indices][:, :-1]
        intensity = vmap[indices][:, -1]

        # int_mask = generate_intensity_mask(intensity,
        #                                    intensity_cutoff=0.05)
        int_mask = intensity > 5
        
        if np.sum(int_mask) > 0:

            # return q_vectors, intensity, int_mask

            (spot_labels,
             spots,
             label_ints) = rsm_spot_search(q_vectors[int_mask],
                                           intensity[int_mask],
                                           nn_dist=0.1,
                                           significance=0.1,
                                           subsample=1)
            
            print(f'Number of spots is {len(spots)}')
            if len(spots) > 1:

                # Find q vector magnitudes and max for spots
                spot_q_mags = np.linalg.norm(spots, axis=1)
                max_q = np.max(spot_q_mags)

                # Find phase reciprocal lattice
                phase.generate_reciprocal_lattice(1.15 * max_q)                
                hkls = phase.all_hkls
            
            else:
                hkls = np.asarray([])
        
        else:
            hkls = np.asarray([])
        
        hkls_map[indices] = np.asarray(hkls)

    return hkls_map


def construct_map(base_map, func=None):

    if func is None:
        def func(inputs):
            if len(inputs) > 0:
                return inputs[0]
            else:
                return np.nan

    derived_map = np.empty(base_map.shape, dtype=float)
    derived_map[:] = np.nan

    for index in range(np.prod(base_map.shape)):
        indices = np.unravel_index(index, base_map.shape)
        try:
            val = func(base_map[indices])
            derived_map[indices] = val
        except:
            print(f'Indices {indices} failed.')
            continue
    
    return derived_map


def plot_derived_map(arr):

    fig, ax = plt.subplots(figsize=(5, 5), dpi=200)

    im = ax.imshow(arr)
    fig.colorbar(im, ax=ax)

    ax.set_aspect('equal')

    fig.show()
        





def transform_coords(x, M):
    # append x with 1
    x = [*x, 1]

    return x @ M