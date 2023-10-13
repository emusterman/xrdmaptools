import numpy as np
from scipy.ndimage import center_of_mass, gaussian_filter, median_filter


def filter_time_series(time_series, size=(3, 1, 1), sigma=2):
    median_xrd = median_filter(time_series, size=size)
    gaussian_xrd = gaussian_filter(median_xrd, sigma)
    return median_xrd, gaussian_xrd


def find_max_coord(median, gaussian, wind=5):
    max_coord = []
    spot_int, gauss_int = [], []

    for i, img in enumerate(median):
        
        # Pixel accurate peak maxima
        ind = np.unravel_index(np.argmax(gaussian[i], axis=None), median.shape)
        #ind = np.unravel_index(np.argmax(img, axis=None), xrd.shape)
        x, y = ind[1], ind[2]

        # Sub-pixel accurate peak maxima
        bbox = [x - wind, x + wind, y - wind, y + wind]
        bbox = [d if (d > 0 and d < median.shape[1]) else 0 for d in bbox]
        dx, dy = center_of_mass(img[bbox[0]:bbox[1], bbox[2]:bbox[3]])
        new_x = x - wind + dx
        new_y = y - wind + dy
        if np.isnan(new_x) or new_x < 0 or new_x > median.shape[1]:
            new_x = x
        if np.isnan(new_y) or new_y < 0 or new_y > median.shape[1]:
            new_y = y
        max_coord.append(np.array([new_x, new_y]))

        # XRD spot intensity
        try:
            bbox = [int(new_x) - wind, int(new_x) + wind, int(new_y) - wind, int(new_y) + wind]
            bbox = [d if (d > 0 and d < median.shape[1]) else 0 for d in bbox]
            spot_int.append(np.sum(img[bbox[0]:bbox[1], bbox[2]:bbox[3]]))
            gauss_int.append(np.sum(gaussian[i][bbox[0]:bbox[1], bbox[2]:bbox[3]]))
        except ValueError:
            bbox = [x - wind, x + wind, y - wind, y + wind]
            bbox = [d if (d > 0 and d < median.shape[1]) else 0 for d in bbox]
            spot_int.append(np.sum(img[bbox[0]:bbox[1], bbox[2]:bbox[3]]))
            gauss_int.append(np.sum(gaussian[i][bbox[0]:bbox[1], bbox[2]:bbox[3]]))

    max_coord = np.asarray(max_coord)
    spot_int, gauss_int = np.asarray(spot_int), np.asarray(gauss_int)
    return max_coord, spot_int, gauss_int