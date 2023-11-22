import xrayutilities as xu
import matplotlib
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from matplotlib.widgets import Slider


class Phase(xu.materials.Crystal):
    # TODO: Add specifiic calls useful for this analysis
    # Find way to generate Phase class automatically from XRD card files too

    def __init__(self, name, lat, energy=None, tth=None):
        super().__init__(name, lat, cij=None, thetaDebye=None)
        self.reflections = None # Place holder to check later
        if energy is not None:
            if energy < 1e3:
                energy *= 1e3
            self.energy = energy
        if tth is not None:
            self.tth = tth
            self.tth_range = (np.min(tth), np.max(tth))

    def __str__(self):
        ostr = f'{self.name} crystal phase class\n'
        ostr += super().__repr__()
        return ostr
    
    def __repr__(self): # Do I need this??
        return f'{self.name} crystal phase class\n' # This is probably backwards from what it shoudld be...


    # This might be able to be done without invoking the simpack module...
    # May need to add a conditional to pass this function if Phase generated from XRD card
    def get_hkl_reflections(self, tth_range=(0, 90), energy=15e3, ignore_less=1):
        if hasattr(self, 'energy'):
            energy = self.energy
        if hasattr(self, 'tth_range'):
            energy = self.tth_range

        planes_data = xu.simpack.PowderDiffraction(self, en=energy, tt_cutoff=tth_range[1]).data

        refl_lst = []
        ang_lst = []
        d_lst = []
        q_lst = []
        int_lst = []

        for i, refl in enumerate(planes_data):
            ang = 2 * planes_data[refl]['ang']
            if ang < tth_range[0]: # Skips tth values less than lower bound
                continue
            d = self.planeDistance(refl)
            q = planes_data[refl]['qpos']
            int = planes_data[refl]['r']

            refl_lst.append(refl)
            ang_lst.append(ang)
            d_lst.append(d)
            q_lst.append(q)
            int_lst.append(int)

        refl_lst = np.array(refl_lst)[np.array(int_lst) > ignore_less]
        ang_lst = np.array(ang_lst)[np.array(int_lst) > ignore_less]
        d_lst = np.array(d_lst)[np.array(int_lst) > ignore_less]
        q_lst = np.array(q_lst)[np.array(int_lst) > ignore_less]
        int_lst = np.array(int_lst)[np.array(int_lst) > ignore_less]

        data = {
            'd' : d_lst,
            'q' : q_lst,
            'tth' : ang_lst,
            'hkl' : refl_lst,
            'int' : int_lst
        }

        self.reflections = data
        
    
    def planeDistances(self, hkl_lst):
        # Re-write of planeDistance to accomadate multiple plances at once
        return 2 * np.pi / np.linalg.norm(self.Q(hkl_lst), axis=1)
    
    def planeAngles(self, hkl1, hkl2):
        # Double-check to make sure this is still used in the final version
        a, b, c = list(self.lattice._parameters.values())[:3]
        alpha, beta, gamma = np.radians(list(self.lattice._parameters.values())[3:])

        hkl1 = np.asarray(hkl1)
        hkl2 = np.asarray(hkl2)
        if len(hkl1.shape) == 1:
            hkl1 = hkl1.reshape(1, *hkl1.shape)
        if len(hkl2.shape) == 1:
            hkl2 = hkl2.reshape(1, *hkl2.shape)
        h1, k1, l1 = hkl1.T
        h2, k2, l2 = hkl2.T

        # Useful constants
        V = self.lattice.UnitCellVolume()
        S11 = b**2 * c**2 * np.sin(alpha)**2
        S22 = a**2 * c**2 * np.sin(beta)**2
        S33 = a**2 * b**2 * np.sin(gamma)**2
        S12 = a * b * c**2 * (np.cos(alpha) * np.cos(beta) - np.cos(gamma))
        S23 = a**2 * b * c * (np.cos(beta) * np.cos(gamma) - np.cos(alpha))
        S13 = a * b**2 * c * (np.cos(gamma) * np.cos(alpha) - np.cos(beta))

        d1 = self.planeDistances(hkl1)
        d2 = self.planeDistances(hkl2)

        # The multiple outer functions is incredibly computational wasteful
        # I am not sure how to reduce it though...
        vec = (S11 * np.outer(h1, h2)
        + S22 * np.outer(k1, k2)
        + S33 * np.outer(l1, l2)
        + S23 * (np.outer(k1, l2) + np.outer(k2, l1).T)
        + S13 * (np.outer(l1, h2) + np.outer(l2, h1).T)
        + S12 * (np.outer(h1, k2) + np.outer(h2, k1).T))

        phi = np.degrees(np.arccos((np.outer(np.outer(d1, d2), vec) / V**2)))

        # Output such that phi[0] will yield planar angles for hkl1[0]
        phi = np.diag(phi).reshape(len(hkl1), len(hkl2)).T
        return phi



# filedir and filename need conditionals
# Should probably wrap this into the phase class as well...
# Need to better identify and consolidate peaks with convolution as well
def write_calibration_file(mat, name=None, tt_cutoff=90, ignore_less=1,
                           filedir=None, filename=None,
                           simulate_convolution=False):
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


def approximate_powder_xrd(xrd_map, poni, energy=None, background=None):
    return

def phase_selector(xrd, phases, tth, ignore_less=1):
    # TODO:
    # Add 2d plot for better understanding of peak character and significance
    # Add background subtraction?
        # I shouldn't have to do that if the 2d background is working correctly yes???

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=200)
    colors = matplotlib.color_sequences['tab10']

    norm_xrd_int = rescale_array(xrd, upper=100, arr_min=0)
    [phase.get_hkl_reflections(tth_range=(np.min(tth), np.max(tth)), ignore_less=ignore_less) for phase in phases if phase.reflections is None];

    xrd_plot = ax.plot(tth, norm_xrd_int, label='Composite XRD', c='k', zorder=(len(phases) + 1))
    fig.subplots_adjust(right=0.75)

    def update_max(LineCollection, phase, slider):
        seg = np.asarray(LineCollection.get_segments())
        #seg[:, 1, 1] = slider.val * phase.reflections['int'] / np.min(phase.reflections['int'])
        seg[:, 1, 1] = slider.val * phase.reflections['int'] / np.max([5, np.min(phase.reflections['int'])])
        return seg

    lines = [xrd_plot]
    slider_lst = []
    update_lst = []
    #zero_phases = [phase.name for phase in phases]

    def update_factory(index):
        def update(val):
            lines[index + 1].set_segments(update_max(lines[index + 1], phases[index], slider_lst[index]))
            
            # Testing extracting values from plot
            #if slider_lst[index].val < 0.1:
            #    if phases[index].name not in zero_phases:
            #        zero_phases.append(phases[index].name)
            #elif slider_lst[index].val >= 0.1:
            #    if phases[index].name in zero_phases:
            #        zero_phases.remove(phases[index].name)
                
            fig.canvas.draw_idle()
        return update

    slider_vpos = np.linspace(0.8, 0.1, len(phases))
    for i, phase in enumerate(phases):
        phase_intensities = phase.reflections['int']
        #phase_scale = np.max(norm_xrd_int) / np.max(phase_intensities)
        phase_plot = ax.vlines(phase.reflections['tth'], ymin=0, ymax=0, color=colors[i], lw=2)
        lines.append(phase_plot)

        axphase = fig.add_axes([0.8, slider_vpos[i], 0.1, 0.03])
        axphase.set_title(phase.name, fontsize=10)
        phase_amp = Slider(
                ax=axphase,
                label='',
                valmin=0,
                valmax=100,
                valinit=0,
                initcolor='none',
                color = colors[i],
                handle_style={'edgecolor' : colors[i], 'size' : 8}
                )
        
        slider_lst.append(phase_amp)
        update_lst.append(update_factory(i))
        slider_lst[i].on_changed(update_lst[i])

    ax.set_ylim(0)
    ax.set_xlabel('Scattering Angle, 2θ [°]')
    ax.set_title('Phase Selector')
    ax.legend(fontsize=10)

    phase_vals = {}
    def on_close(event):
        for phase, slider in zip(phases, slider_lst):
            #print(f'{phase.name} has {slider.val} value.')
            phase_vals[phase.name] = slider.val
        #print(zero_phases)
        return phase_vals
    
    fig.canvas.mpl_connect('close_event', on_close)
    plt.show(block=True)
    plt.pause(0.01)
    return phase_vals
    

    





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
    
    
##################
### Deprecated ###   
##################
    
'''def phase_selector(xrd_plot, phases):
    # Re-write to accept list of possible phases

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

    fig.canvas.mpl_connect('pick_event', on_pick)'''