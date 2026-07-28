import numpy as np
import os
from skimage import io
from tqdm import tqdm
from matplotlib.colors import LogNorm

def load_xas_slice(scanid, xrd_dets=['eiger']):
    data_dict, scan_md = xmt.io.db_io.load_data(scanid, xrd_dets=xrd_dets, data_keys=['enc1', 'enc2'], repair_method='fill')
    if scan_md['type'] != 'XAS_SLICE':
        raise ValueError(f"Scan must be of type XAS_SLICE. Scan {scanid} is of type {scan_md['type']}")
    pos_data = xmt.io.db_io.c[scanid]['primary']['data']
    energy = pos_data['energy_energy'][:]
    for key in ['nano_stage_x', 'nano_stage_y']:
        if key in pos_data:
            enc = pos_data[key]
            break
    else:
        for key in ['enc1', 'enc2']:
            if np.max(data_dict[key]) - np.min(data_dict[key]) > 1:
                enc = data_dict[key]
                axes = [np.mean(enc, axis=i) for i in range(2)]
                enc = axes[np.argmax([np.gradient(ax).sum() for ax in axes])]
    return data_dict[f'{xrd_dets[0]}_image'], energy, enc


def plot_rc(images, energy, roi=None):
    if roi is None:
        roi = (slice(None, None), slice(None, None))
    print('Loading data and calculating row sums...')
    sums = []
    for row_images in images:
        row_images = row_images[:]
        row_images[row_images > 1e6] = 0
        sums.append(np.sum(row_images[:, *roi]))
    # sums = [np.sum(row_images[:, *roi]) for row_images in images]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(energy, sums)
    ax.set_xlabel('Energy [keV]')
    ax.set_ylabel('ROI Sum [cts]')
    ax.set_title('Energy Rocking Curve')
    fig.show()


def plot_rc_stack(images, energy, roi=None, **kwargs):
    if roi is None:
        roi = (slice(None, None), slice(None, None))
    print('Loading data and calculating row sums...')
    sums = [np.sum(row_images[..., *roi], axis=0) for row_images in images]
    energy = [np.round(en, 3) for en in energy]
    # matplotlib is awful
    sums = [np.flipud(img) for img in sums]
    fig, ax = xmt.plot.image_stack.base_slider_plot(sums,
                                                    slider_vals=energy,
                                                    slider_label='Energy [keV]',
                                                    **kwargs)
    ax.invert_yaxis()
    fig.show()


def plot_space_stack(images, enc, roi=None, **kwargs):
    if roi is None:
        roi = (slice(None, None), slice(None, None))
    print('Loading data and calculating row sums...')
    # sums = [np.sum(row_images, axis=1) for row_images in images]
    enc = [np.round(en, 3) for en in enc]
    sums = np.sum(images[..., *roi], axis=0)
    # matplotlib is awful
    sums = [np.flipud(img) for img in sums]
    fig, ax = xmt.plot.image_stack.base_slider_plot(sums,
                                                    slider_vals=enc,
                                                    slider_label='Axis [um]',
                                                    **kwargs)
    ax.invert_yaxis()
    fig.show()


def plot_rc_map(images, energy, enc, roi=None, **kwargs):
    if roi is None:
        roi = (slice(None, None), slice(None, None))

    print('Loading data and calculating sums...')
    sums = [np.sum(row_images[..., *roi], axis=(-2, -1)) for row_images in images]
    sums = np.asarray(sums)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    enc_step = np.mean(np.diff(np.abs(enc)))
    energy_step = np.mean(np.diff(np.abs(energy)))
    extent = [
        np.min(enc) - enc_step,
        np.max(enc) + enc_step,
        np.min(energy) - energy_step,
        np.max(energy) + energy_step
    ]
    img = ax.imshow(sums, extent=extent, **kwargs)
    ax.set_xlabel('Position [um]')
    ax.set_ylabel('Energy [keV]')
    ax.set_title('XAS Slice ROI Sum')
    ax.set_aspect('auto')
    fig.colorbar(img, ax=ax)
    fig.show()


def plot_rc_maps(images, energy, enc, roi=None, **kwargs):
    if roi is None:
        roi = (slice(None, None), slice(None, None))

    print('Loading data and calculating...')
    sums, comx, comy = [], [], []
    for row_images in tqdm(images):
        image_roi = row_images[..., *roi]
        image_roi[image_roi > 500_000] = 0
        sums.append(np.sum(image_roi, axis=(-2, -1)))
        comx.append(np.sum(image_roi * np.arange(image_roi.shape[-1]), axis = (-2, -1)) / np.sum(image_roi, axis = (-2, -1)))
        comy.append(np.sum(image_roi * np.arange(image_roi.shape[-2]).reshape(-1, 1), axis = (-2, -1)) / np.sum(image_roi, axis = (-2, -1)))

    #print('Loading data and calculating sums...')
    #sums = [np.sum(row_images[..., *roi], axis=(-2, -1)) for row_images in images]
    #sums = np.asarray(sums)

    #print('Computing CoMs in x and y...')
    #comx = [np.sum(row_images[..., *roi] * np.arange(row_images[..., *roi].shape[-1]), axis = (-2, -1)) / np.sum(row_images[..., *roi], axis = (-2, -1)) for row_images in images]
    #comy = [np.sum(row_images[..., *roi] * np.arange(row_images[..., *roi].shape[-2]).reshape(-1, 1), axis = (-2, -1)) / np.sum(row_images[..., *roi], axis = (-2, -1)) for row_images in images]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    enc_step = np.mean(np.diff(np.abs(enc)))
    energy_step = np.mean(np.diff(np.abs(energy)))
    extent = [
        np.min(enc) - enc_step,
        np.max(enc) + enc_step,
        np.min(energy) - energy_step,
        np.max(energy) + energy_step
    ]

    for i, (label, data) in enumerate(zip(['ROI Sum LogNorm', 'CoM x', 'CoM y'], [sums, comx, comy])):
        if i == 0:
            img = axes[i].imshow(data, extent=extent, norm = LogNorm(), **kwargs)
        else:
            img = axes[i].imshow(data, extent=extent, **kwargs)

        axes[i].set_xlabel('Position [um]')
        axes[i].set_ylabel('Energy [keV]')
        axes[i].set_title(label)
        axes[i].set_aspect('auto')
        fig.colorbar(img, ax=axes[i])
        fig.show()





def decompose_q_space(q_vectors):

    mag = np.linalg.norm(q_vectors, axis=-1)

    phi = np.arccos(q_vectors[:, 1] / mag)

    chi = np.arctan2(q_vectors[:, 2], q_vectors[:, 0]) 

    return mag, phi, chi