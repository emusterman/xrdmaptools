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
from xrdmaptools.utilities.utilities import rescale_array


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


def plot_shifted_points(map_arr, shifts, add_grid=True):

    x = np.arange(0, map_arr.shape[1])
    y = np.arange(0, map_arr.shape[0])

    # Redefine y-shifts to match matplotlib axes...
    shifts = np.asarray(shifts)
    shifts[:, 0] *= -1

    # Get sequential colors for each grid
    norm = matplotlib.colors.Normalize(vmin=0, vmax=(len(shifts)))
    mapper = cm.ScalarMappable(norm=norm, cmap='jet')
    grid_colors = [(r, g, b) for r, g, b, a in mapper.to_rgba(range(len(shifts)))]

    # Shifts stats
    x_step = np.mean(np.diff(x))
    y_step = np.mean(np.diff(y))
    ymin, xmin = np.min(shifts, axis=0)
    ymax, xmax = np.max(shifts, axis=0)
    xx, yy = np.meshgrid(x, y[::-1])  # matching matplotlib description

    fig, ax = plt.subplots()

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
    print(virt_shape)
    
    vmask_list = []
    for i, shift in enumerate(shifts):
        xxi = (xx + shift[1])
        yyi = (yy + shift[0])

        xx_ind, yy_ind = xx_virt[0], yy_virt[0]
        vmask_x0 = np.argmin(np.abs(xxi[0] - xx_ind))
        vmask_y0 = np.argmin(np.abs(yyi[:, 0] - yy_ind))
        # print(vmask_x0, vmask_y0)
        vmask = np.zeros_like(xx, dtype=np.bool_)
        vmask[vmask_y0 : vmask_y0 + virt_shape[0],
              vmask_x0 : vmask_x0 + virt_shape[1]] = True
        vmask_list.append(vmask)

        ax.scatter(xxi.flatten(),
                   yyi.flatten(),
                   s=5,
                   color=grid_colors[i])
    
    ax.scatter(xx_virt,
               yy_virt,
               s=20,
               c='r',
               marker='*')

    if add_grid: # This can probably be done with RegularPolyCollection but this proved finicky
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
    # return xx_virt.reshape(virt_shape), yy_virt.reshape(virt_shape)
    # return xx_virt.reshape(virt_shape), yy_virt.reshape(virt_shape)[::-1] # flip y-axes again


def vectorize_images(xrdmapstack, vmask_list, image_data_key='recent'):

    # Quick input check
    for i, xrdmap in enumerate(xrdmapstack):
        if (not hasattr(xrdmap, 'q_arr')
             or xrdmap.q_arr is None):
             err_str = f'q_arr not defined for xrdmap[{i}]!'
             raise AttributeError(err_str)
        elif (not hasattr(xrdmap, 'blob_masks')
                or xrdmap.blob_masks is None):
            err_str = f'blob_masks not defined for xrdmap[{i}]!'
            raise AttributeError(err_str)

    # Vectorize images

    v_arr_list = []
    for i, xrdmap in enumerate(xrdmapstack):
        print(f'Processing data from scan {xrdmap.scan_id}.')

        # Load images
        xrdmap.load_images_from_hdf(image_data_key=image_data_key)

        v_arr = np.empty(xrdmap.map_shape, dtype=object)
        # v_arr.fill([])

        print('Vectorizing data...')
        for indices in tqdm(xrdmap.indices):
            blob_mask = xrdmap.blob_masks[indices]
            intensity = xrdmap.images[indices][blob_mask]
            q_vectors = xrdmap.q_arr[blob_mask]
            
            v_arr[indices] = np.hstack([q_vectors,
                                        intensity.reshape(-1, 1)])
        
        v_arr_list.append(v_arr)

        # Release memory
        xrdmap.dump_images()
    
    # Combine v_arr_list with vmask_list
    # return v_arr_list

    vmask_shape = np.max([vmask_list[0].sum(axis=0),
                          vmask_list[0].sum(axis=1)],
                         axis=1)

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

    return full_v_arr
        
