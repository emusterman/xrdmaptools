import xrayutilities as xu
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def write_calibration_file(mat, name=None, tt_cutoff=90, ignore_less=1, simulate_convolution=False):
    '''
    
    '''
    if name == None:
        name = mat.name
    header = [f'# Calibrant: {name} ({mat.chemical_composition()})',
              f'# Crystal: a={mat.a} b={mat.b} c={mat.c} alpha={mat.alpha} beta={mat.beta} gamma={mat.gamma}',
              f'# d (Å)\t#|\tInt.(%)\t|\tHKL Reflection']
    #density?? What about x-ray parameters, space group??

    planes_data = xu.simpack.PowderDiffraction(mat, en=18e3, tt_cutoff=tt_cutoff).data

    refl_lst = []
    ang_lst = []
    d_lst = []
    int_lst = []

    for i, refl in enumerate(planes_data):
        ang = planes_data[refl]['ang']
        d = mat.planeDistance(refl)
        int = planes_data[refl]['r']

        refl_lst.append(refl)
        ang_lst.append(ang)
        d_lst.append(d)
        int_lst.append(int)

    d_lst = np.array(d_lst)[np.array(int_lst) > ignore_less]
    refl_lst = np.array(refl_lst)[np.array(int_lst) > ignore_less]
    int_lst = np.array(int_lst)[np.array(int_lst) > ignore_less]

    # Convolves peaks together using xrayutilities built in simulations
    # Helps to convolve close reflections into what will be observed experimentally
    if simulate_convolution:
        two_theta = np.arange(0, tt_cutoff, 0.005)
        powder = xu.simpack.smaterials.Powder(mat, 1, crystallite_size_gauss=100e-9)
        pm = xu.simpack.PowderModel(powder, I0=100, en=18e3)
        intensities = pm.simulate(two_theta)

        norm_int = 100 * (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))
        norm_int[norm_int < ignore_less] = 0

        angles = find_peaks(norm_int)
        d_spacings = [12400 / (2 * 18e3 * np.sin(ang / 2)) for ang in two_theta[angles[0]]]

        new_refl_lst = []
        
        for i in range(len(d_spacings)):
            diff_arr = np.absolute(np.array(d_lst) - d_spacings[i])
            close_refl = [diff_arr > 0.5]
            new_refl_lst.append(refl_lst[close_refl])

        int_lst = intensities(angles[0])
        refl_lst = np.asarray(new_refl_lst)
        d_lst = d_spacings


    data = []
    for i in range(len(refl_lst)):
        data.append(str(np.round(d_lst[i], 3)) + '\t#|\t' + str(np.round(int_lst[i], 1)) + '\t|\t' + str(tuple(refl_lst[i])))

    file = open(filedir + filename, 'w')
    file.writelines(s + '\n' for s in header + data)
    file.close()

    return header + data


def get_HKL_reflections(mat, tt_cutoff=90, en=15e3, ignore_less=1):
    #header = [f'# Calibrant: {name} ({mat.chemical_composition()})',
    #          f'# Crystal: a={mat.a} b={mat.b} c={mat.c} alpha={mat.alpha} beta={mat.beta} gamma={mat.gamma}',
    #          f'# d (Å)\t#|\tInt.(%)\t|\tHKL Reflection']

    planes_data = xu.simpack.PowderDiffraction(mat, en=en, tt_cutoff=tt_cutoff).data

    refl_lst = []
    ang_lst = []
    d_lst = []
    int_lst = []

    for i, refl in enumerate(planes_data):
        ang = planes_data[refl]['ang']
        d = mat.planeDistance(refl)
        int = planes_data[refl]['r']

        refl_lst.append(refl)
        ang_lst.append(ang)
        d_lst.append(d)
        int_lst.append(int)

    refl_lst = np.array(refl_lst)[np.array(int_lst) > ignore_less]
    ang_lst = 2 * np.array(ang_lst)[np.array(int_lst) > ignore_less]
    d_lst = np.array(d_lst)[np.array(int_lst) > ignore_less]
    int_lst = np.array(int_lst)[np.array(int_lst) > ignore_less]

    data = {
        'd-spacing':d_lst,
        '2th Angles':ang_lst,
        'HKL Reflection':refl_lst,
        'Intensities':int_lst
    }

    return data


def phase_selector(xrd_plot, phases):

    fig, ax = plt.subplots()

    xrd_two_theta = xrd_plot[0]
    xrd_intensities = xrd_plot[1]

    phase_two_theta = phases['2th Angles'][:]
    phase_intensities = phases['Intensities'][:]
    
    phase_mask = (phase_two_theta <= np.max(xrd_two_theta)) & (phase_two_theta >= np.min(xrd_two_theta))
    phase_scale = np.max(xrd_intensities) / np.max(phase_intensities[phase_mask])

    xrd_plot = ax.plot(xrd_two_theta, xrd_intensities, label='XRD Spectra')
    phase_plot = ax.vlines(phase_two_theta, ymin=0, ymax=phase_intensities * phase_scale, colors='k', alpha=0.5, label='Phase')

    ax.set_xlim(0.975 * np.min(xrd_two_theta), 1.025 * np.max(xrd_two_theta))
    ax.set_ylim(0, 1.15 * np.max(xrd_intensities))
    ax.set_title("XRD Pattern for " + stibnite.name)
    ax.set_xlabel("2 theta [degrees]")
    ax.set_ylabel("Relative Intensity [%]")
    ax.set_title('Click on legend line to toggle line on/off')
    leg = ax.legend(fancybox=True, shadow=True)

    lines = [xrd_plot, phase_plot]
    lined = {}  # Will map legend lines to original lines.
    for legline, origline in zip(leg.get_lines(), lines):
        legline.set_picker(True)  # Enable picking on the legend line.
        lined[legline] = origline


    def on_pick(event):
        # On the pick event, find the original line corresponding to the legend
        # proxy line, and toggle its visibility.
        legline = event.artist
        origline = lined[legline]
        if origline == xrd_plot:
            return
        visible = not origline.get_visible()
        origline.set_visible(visible)
        # Change the alpha on the line in the legend, so we can see what lines
        # have been toggled.
        legline.set_alpha(1.0 if visible else 0.2)
        fig.canvas.draw()

    fig.canvas.mpl_connect('pick_event', on_pick)


def find_label_peaks(theta, intensity, phase):
    '''
    Assumes uniformly spaced data...
    '''

    # Normalize data
    norm_int = 100 * (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))

    # Find peak indices
    peaks = find_peaks(norm_int, prominence=2, height=1, width=5)[0]

    # Matching peaks to phase information
    # Ignores peaks too far away - not sure what to do with them yet
    # Cannot handle assigning multiple reflections to the same peak yet
    
    for peak in peaks:
        ang = np.array(theta)[peak]
        min_ref_ind = np.argmin(np.abs(ang - phase['2th Angles']))
        min_ref = np.min(np.abs(ang - phase['2th Angles']))
        if min_ref > 0.5:
            continue
        print(f'Peak indexed as\t{tuple(phase["HKL Reflection"][min_ref_ind])}')
        print(f'Peak is {min_ref:.4f}° away.')