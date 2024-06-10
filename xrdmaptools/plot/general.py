import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label

# Local imports
from ..geometry.geometry import estimate_image_coords, estimate_polar_coords
from ..reflections.spot_blob_search import find_blob_contours


######################
### Image Plotting ###
######################

# Grabs and formats data from xrdmap if requested
def _plot_parse_xrdmap(xrdmap, indices, mask=False, spots=False, contours=False):

    # Extract mask
    out_mask = None
    if mask and hasattr(xrdmap.map, 'mask'):
        out_mask = xrdmap.map.mask
    elif mask and not hasattr(xrdmap.map, 'mask'):
        print('WARNING: Mask requested, but xrdmap.map does not have a mask!')

    # Extract spots
    out_spots = None
    if spots and hasattr(xrdmap, 'spots'):
        pixel_df = xrdmap.spots[(xrdmap.spots['map_x'] == indices[0])
                                & (xrdmap.spots['map_y'] == indices[1])].copy()
        
        if any([x[:3] == 'fit' for x in pixel_df.keys()]):
            pixel_df.dropna(axis=0, inplace=True)
            out_spots = pixel_df[['fit_chi0', 'fit_tth0']].values
        else:
            out_spots = pixel_df[['guess_cen_chi', 'guess_cen_tth']].values

        if not xrdmap.map.corrections['polar_calibration']:
            out_spots = estimate_image_coords(out_spots[:, ::-1],
                                              xrdmap.tth_arr,
                                              xrdmap.chi_arr)[:, ::-1]

    elif spots and not hasattr(xrdmap, 'spots'):
        print('WARNING: Spots requested, but xrdmap does not have any spots!')

    # Extract contours
    out_contour_list = None
    if contours and hasattr(xrdmap.map, 'spot_masks'):
        blob_img = label(xrdmap.map.spot_masks[indices])
        blob_contours = find_blob_contours(blob_img)
        out_contour_list = []
        for blob_contour in blob_contours:
            if xrdmap.map.corrections['polar_calibration']:
                out_contour_list.append(estimate_polar_coords(blob_contour.T,
                                                              xrdmap.tth_arr,
                                                              xrdmap.chi_arr).T)

    elif contours and hasattr(xrdmap.map, 'spot_masks'):
        print('WARNING: Contours requested, but xrdmap does not have any spot masks to draw contours!')

    return tuple([out_mask, out_spots, out_contour_list])


def _xrdmap_image(xrdmap,
                  image=None,
                  indices=None):
    # Check image type
    if image is not None:
        image = np.asarray(image)
        if len(image.shape) == 1 and len(image) == 2:
            indices = tuple(iter(image))
            image = xrdmap.map.images[indices]
        elif len(image.shape) == 2:
            if indices is not None:
                indices = tuple(indices)
        else:
            raise ValueError(f"Incorrect image shape of {image.shape}. Should be two-dimensional.")
    else:
        # Evaluate images
        xrdmap.map._dask_2_dask()

        if indices is not None:
            indices = tuple(indices)
            image = xrdmap.map.images[indices]
            image = np.asarray(image)
        else:
            i = np.random.randint(xrdmap.map.map_shape[0])
            j = np.random.randint(xrdmap.map.map_shape[1])
            indices = (i, j)
            image = xrdmap.map.images[indices]
            image = np.asarray(image)

    return image, indices


def _xrdmap_integration(xrdmap,
                        integration=None,
                        indices=None):

    # Check image type
    if integration is not None:
        integration = np.asarray(integration)
        if len(integration.shape) == 1 and len(integration) == 2:
            indices = tuple(iter(integration))
            integration = xrdmap.map.integrations[indices]
        elif len(integration.shape) == 1:
            if indices is not None:
                indices = tuple(indices)
        else:
            raise ValueError(f"Incorrect image shape of {integration.shape}. Should be one-dimensional.")
    else:
        if not hasattr(xrdmap.map, 'integrations'):
            raise ValueError("Integration has not been specified and XRDMap does not have any integrations calculated!")

        if indices is not None:
            indices = tuple(indices)
            integration = xrdmap.map.integrations[indices]
            integration = np.asarray(integration)
        else:
            i = np.random.randint(xrdmap.map.map_shape[0])
            j = np.random.randint(xrdmap.map.map_shape[1])
            indices = (i, j)
            integration = xrdmap.map.integrations[indices]
            integration = np.asarray(integration)

    return integration, indices


def plot_image(image,
               indices=None,
               title=None,
               mask=None,
               spots=None,
               contours=None,
               fig=None,
               ax=None,
               aspect='auto',
               **kwargs):
    
    # Plot image
    if fig is None and ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=200)
    elif fig is None and ax is not None or fig is not None and ax is None:
        raise ValueError('Figure and axes must both provided or both None')
    
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape == image.shape:
            image = image * mask
            #image[mask] = np.nan
        else:
            err_str = (f'Mask shape of {mask.shape} does not match '
                       + f'image shape of {image.shape}')
            raise ValueError(err_str)
    
    # Allow some flexibility for kwarg inputs
    plot_kwargs = {'c' : 'r',
                   'lw' : 0.5,
                   's' : 1}
    for key in plot_kwargs.keys():
        if key in kwargs.keys():
            plot_kwargs[key] = kwargs[key]
            del kwargs[key]

    im = ax.imshow(image, aspect=aspect, **kwargs)
    ax.set_xlabel('x index')
    ax.set_ylabel('y index')
    fig.colorbar(im, ax=ax)

    if title is not None:
        ax.set_title(title)
    elif indices is not None:
        ax.set_title(f'Row = {indices[0]}, Col = {indices[1]}')
    else:
        ax.set_title('Input Image')
        
    # Plot spots
    if spots is not None:
        ax.scatter(spots[:, 1], spots[:, 0], s=plot_kwargs['s'], c=plot_kwargs['c'])
    
    if contours is not None:
        for contour in contours:
            ax.plot(*contour, c=plot_kwargs['c'], lw=plot_kwargs['lw'])

    return fig, ax


def plot_reconstruction(self,
                        indices=None,
                        plot_residual=False,
                        fig=None,
                        ax=None,
                        **kwargs):
    raise NotImplementedError()
    if not hasattr(self, 'spots'):
        raise RuntimeError('xrdmap does not have any spots!')

    if indices is None:
        i = np.random.randint(self.map.map_shape[0])
        j = np.random.randint(self.map.map_shape[1])
        indices = (i, j)
    else:
        indices = tuple(indices)
        if (indices[0] < 0 or indices[0] > self.map.map_shape[0]):
            raise IndexError(f'Indices ({indices}) is out of bounds along axis 0 for map shape ({self.map.map_shape})')
        elif (indices[1] < 0 or indices[1] > self.map.map_shape[1]):
            raise IndexError(f'Indices ({indices}) is out of bounds along axis 1 for map shape ({self.map.map_shape})')
    
    if hasattr(self, 'spot_model'):
        spot_model = self.spot_model
    else:
        print('Warning: No spot model saved. Defaulting to Gaussian.')
        spot_model = GaussianFunctions
    
    pixel_df = self.spots[(self.spots['map_x'] == indices[0]) & (self.spots['map_y'] == indices[1])].copy()

    if any([x[:3] == 'fit' for x in pixel_df.keys()]):
        prefix = 'fit'
        pixel_df.dropna(axis=0, inplace=True)
        param_labels = [x for x in self.spots.iloc[0].keys() if x[:3] == 'fit'][:6]
    else:
        prefix = 'guess'
        param_labels = ['height', 'cen_tth', 'cen_chi', 'fwhm_tth', 'fwhm_chi']
        param_labels = [f'guess_{param_label}' for param_label in param_labels]
        spot_model = GaussianFunctions

    fit_args = []
    for index in pixel_df.index:
        fit_args.extend(pixel_df.loc[index, param_labels].values)
        if prefix == 'guess':
            fit_args.append(0) # Filling in theta value

    if len(fit_args) > 0:
        #return fit_args
        recon_image = spot_model.multi_2d([self.tth_arr.ravel(), self.chi_arr.ravel()], 0, *fit_args)
        recon_image = recon_image.reshape(self.map.images.shape[-2:])
    else:
        recon_image = np.zeros(self.map.images.shape[-2:])

    if not plot_residual:
        fig, ax = self.plot_image(recon_image,
                            return_plot=True, indices=indices,
                            **kwargs)
        plt.show()

    else:
        image = self.map.images[indices]
        residual = recon_image - image
        ext = np.max(np.abs(residual[self.map.mask]))
        fig, ax = self.plot_image(residual,
                            title=f'Residual of (Row = {indices[0]}, Col = {indices[1]})',
                            return_plot=True, indices=indices,
                            vmin=-ext, vmax=ext, cmap='bwr', # c='k',
                            **kwargs)
        plt.show()


####################
### Map Plotting ###
####################
        

# Pretty simple...
def plot_map(value,
             map_extent=None,
             position_units=None,
             fig=None,
             ax=None,
             **kwargs):

    if fig is None and ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=200)
    elif fig is None and ax is not None or fig is not None and ax is None:
        raise ValueError('Figure and axes must both provided or both None')

    im = ax.imshow(value, extent=map_extent, **kwargs)
    fig.colorbar(im, ax=ax)

    # Set position_units. Map_extent must be provided to be valid
    if position_units is None or map_extent is None:
        position_units = 'a.u.'

    ax.set_xlabel(f'x position [{position_units}]')
    ax.set_ylabel(f'y position [{position_units}]')

    ax.set_aspect('equal') # in case of non-square pixel size

    return fig, ax


##########################
### Geometry Corrected ###
##########################

def plot_integration(intensity,
                     indices=None,
                     tth=None,
                     units=None,
                     title=None,
                     fig=None,
                     ax=None,
                     y_min=None,
                     y_max=None,
                     **kwargs):
    
    if fig is None and ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=200)
    elif fig is None and ax is not None or fig is not None and ax is None:
        raise ValueError('Figure and axes must both provided or both None')
    
    if units is None:
        units = 'a.u.'

    if tth is None:
        tth = range(len(intensity))
    
    ax.plot(tth, intensity, **kwargs)
    ax.set_xlabel(f'Scattering Angle, 2θ [{units}]')
    # Direct access to scaling!
    ax.set_ylim(y_min, y_max)

    if title is not None:
        ax.set_title(title)
    elif indices is not None:
        ax.set_title(f'Row = {indices[0]}, Col = {indices[1]}')
    else:
        ax.set_title('Input Integration')

    return fig, ax


def plot_cake(intensity,
              tth,
              chi,
              units=None,
              fig=None, 
              ax=None,
              **kwargs):
    
    raise NotImplementedError()

    if fig is None and ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=200)
    elif fig is None and ax is not None or fig is not None and ax is None:
        raise ValueError('Figure and axes must both provided or both None')
        
    extent = [np.min(tth), np.max(tth), np.min(chi), np.max(chi)]

    if units is None:
        tth_units = 'a.u.'
        chi_units = 'a.u.'
    elif isinstance(units, str):
        tth_units = units
        chi_units = units
    elif isinstance(units, (tuple, list, np.ndarray)):
        tth_units = units[0]
        chi_units = units[1]
    else:
        TypeError(f'Unknown units type of {type(units)}.')

    im = ax.imshow(intensity, extent=extent, **kwargs)
    fig.colorbar(im, ax=ax)

    ax.set_xlabel(f'scattering angle, 2θ [{tth_units}]')
    ax.set_ylabel(f'azimuthal angle, χ [{chi_units}]')

    return fig, ax
    
