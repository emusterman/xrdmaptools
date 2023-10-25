import numpy as np
import pyFAI
import pyFAI.azimuthalIntegrator as ai





def estimate_img_coords(coords, img, tth=tth, chi=chi):
    # Estimate image coordinates from tth and chi values
    tth_i = np.asarray(coords[0])
    chi_i = np.asarray(coords[1])
    x = (tth_i - np.min(tth)) / (np.max(tth) - np.min(tth)) * img.shape[1]
    y = img.shape[0] - (chi_i-  np.min(chi)) / (np.max(chi) - np.min(chi)) * img.shape[0]
    return np.array([int(x), int(y)])

def estimate_recipricol_coords(coords, img, tth=tth, chi=chi):
    # Convert image coordinates to tth and chi values
    x_i = np.asarray(coords[0])
    y_i = np.asarray(coords[1])
    # Convert image coordinates to tth and chi values
    tth_lst = np.min(tth) + (np.max(tth) - np.min(tth)) * x_i / img.shape[1]
    chi_lst = np.min(chi) + (np.max(chi) - np.min(chi)) * (img.shape[0] - y_i) / img.shape[0]
    return np.array([tth_lst, chi_lst])


'''def calibrate_img_coords(coords, img, tth=tth, chi=chi):
    # Convert image coordinates to tth and chi values
    axes = [tth, chi]
    
    new_coords = []
    for i, coord in enumerate(coords):
        # Matplotlib uses a different origin for images. This handles that...
        if i == 1:
            coord = img.T.shape[i] - coord
            
        new_coords.append(np.min(axes[i]) + (np.max(axes[i]) - np.min(axes[i])) * coord / img.T.shape[i])
    return np.array(new_coords)'''


def xy_2_tthchi():
    return

def tthchi_2_xy():
    return