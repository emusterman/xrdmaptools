import xrayutilities as xu
from xrayutilities.materials.spacegrouplattice import sgrp_sym, SGLattice
from xrayutilities.materials.atom import Atom
import matplotlib
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from matplotlib.widgets import Slider
from collections import OrderedDict

# Local imports
from xrdmaptools.utilities.math import (
    tth_2_q,
    vector_angle
)
from xrdmaptools.utilities.utilities import rescale_array



class Phase(xu.materials.Crystal):
    # TODO: Add specific calls useful for this analysis
    # Find way to generate Phase class automatically from XRD card files too
    # Add flag to limit functionality depending on if from cif or XRD card

    def __init__(self, name, lat, energy=None, tth=None):
        super().__init__(name, lat, cij=None, thetaDebye=None)
        self.reflections = None # Place holder to check later
        if energy is not None:
            self.energy = energy
        if tth is not None:
            self.tth = tth
            self.tth_range = (np.min(tth), np.max(tth))


    def __str__(self):
        return f'{self.name} crystal phase'
    

    def __repr__(self):
        ostr = f'{self.name} crystal phase'
        # Gettin the lattice information is just a bit to much
        # TODO: Parse down lattice information to sleeker format
        #if hasattr(self, 'lattice'):
        #    ostr += '\t'.join(self.lattice.__str__().splitlines(True))
        return ostr

    ### Class methods, Static methods, and Properties

    @classmethod
    def from_xrd_card(cls):
        # Generic and not fully functioning phase instance with only individual peak information
        # Should be able to generate from reference standards...
        raise NotImplementedError()
    

    @classmethod
    def from_hdf(cls, group, **kwargs):
        # Load values to reconstruct phase instance from standard hdf group or dataset
        # Will need to implement a save to hdf group function as well...    
        name = group.name.split('/')[-1]
        
        params = OrderedDict()
        key_lst = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
        for key in key_lst:
            params[key] = group.attrs[key]
        
        space_group = group.attrs['space_group']
        space_group_number = group.attrs['space_group_number']

        wbase = group['WyckoffBase']
        atom_lst = []
        for atom_key in wbase.keys():
            atom = Atom(atom_key.split('[')[0], wbase[atom_key].attrs['number'])
            #atom = Atom(atom_key, atom_dict[atom_key])
            pos = wbase[atom_key].attrs['position']
            if np.any(np.isnan(wbase[atom_key][0])):
                positions = None
            else:
                positions = (*wbase[atom_key][:][0],)
            atom_lst.append((atom, (pos, positions)))

        args = cls.get_sym_args(space_group, params)
        lattice = SGLattice(space_group, *args, atoms=[atom[0] for atom in atom_lst],
                                                    pos=[atom[1] for atom in atom_lst])
        
        return cls(name, lattice, **kwargs)
    

    def save_to_hdf(self, parent_group):
        iphase = parent_group.require_group(self.name)
    
        # Mostly saves lattice information...
        for key, value in self.lattice._parameters.items():
            iphase.attrs[key] = value

        iphase.attrs['space_group'] = self.lattice.space_group
        iphase.attrs['space_group_number'] = sgrp_sym[int(self.lattice.space_group.split(':')[0])][0]

        wbase = iphase.require_group('WyckoffBase')
        for i, atom in enumerate(self.lattice._wbase):
            data = np.array(atom[1][1:])
            #data = np.array([np.nan for pos in data if pos is None else pos])
            data = np.array([np.nan if pos is None else pos for pos in data])
            dset = wbase.require_dataset(f'{atom[0].name}[{i}]', data=data, shape=data.shape, dtype=data.dtype)
            dset.attrs['number'] = atom[0].num
            dset.attrs['position'] = atom[1][0]
    

    @property
    def energy(self):
            return self._energy
    
    @energy.setter
    def energy(self, val):
        if val < 1e3:
            val *= 1e3
        self._energy = val

        return self._energy


    @staticmethod
    def get_sym_args(sgrp, params):

        sgrp = str(sgrp).lower()
        sgrp = sgrp.split(':')[0]

        if sgrp in ['triclinic', *[str(num) for num in range(1, 3)]]:
            args = [params[key] for key in ['a', 'b', 'c', 'alpha', 'beta', 'gamma']]

        elif sgrp in ['monoclinic', *[str(num) for num in range(3, 16)]]:
            args = [params[key] for key in ['a', 'b', 'c', 'beta']]

        elif sgrp in ['orthorhombic', *[str(num) for num in range(16, 75)]]:
            args = [params[key] for key in ['a', 'b', 'c']]

        elif sgrp in ['tetragonal', *[str(num) for num in range(75, 143)]]:
            args = [params[key] for key in ['a', 'c']]

        elif sgrp in ['trigonal', *[str(num) for num in range(143, 168)]]:
            args = [params[key] for key in ['a', 'c']]

        elif sgrp in ['hexagonal', *[str(num) for num in range(168, 195)]]:
            args = [params[key] for key in ['a', 'c']]

        elif sgrp in ['cubic', *[str(num) for num in range(195, 231)]]:
            args = [params[key] for key in ['a']]

        return args


    # May need to add a conditional to pass this function if Phase generated from XRD card
    def get_hkl_reflections(self,
                            tth_range=None,
                            energy=None,
                            ignore_less=1,
                            save_reflections=True):
        if energy is None:
            if hasattr(self, 'energy'):
                energy = self.energy
            else:
                raise IOError('Must define energy somewhere.')
            
        if tth_range is None:
            if hasattr(self, 'tth_range'):
                tth_range = self.tth_range
            else:
                tth_range = (0, 90)

        wavelength = energy_2_wavelength(energy)

        all_refl = self.lattice.get_allowed_hkl(qmax=tth_2_q(tth_range[1], wavelength=wavelength))
        all_q = np.linalg.norm(self.Q(*all_refl), axis=1)
        # TODO: Add conditional to remove below a qmin
        all_q = np.round(all_q, 10) # Clean up some errors
        sort_refl = [tuple(x) for _, x in sorted(zip(all_q, all_refl))]
        all_q.sort()
        F_hkl = np.abs(self.StructureFactor(sort_refl))**2
        #F_hkl = rescale_array(F_hkl, lower=0, upper=100)

        hkl_list = []
        q_list = []
        int_list = []
        tth_list = []
        d_list = []

        for index, norm_int in enumerate(F_hkl):
            if all_q[index] not in q_list:
                hkl_list.append(sort_refl[index]) # Only takes first hkl value
                q_list.append(all_q[index])
                int_list.append(norm_int)
                tth_list.append(q_2_tth(all_q[index], wavelength=wavelength))
                d_list.append(convert_qd(all_q[index]))
            else:
                int_list[-1] += norm_int # Handles multiplicity
                if np.sum(sort_refl[index]) > np.sum(hkl_list[-1]):
                    hkl_list[-1] = sort_refl[index] # bias towards positive hkl values

        # Changing back to list is not necessary, but is consistent with other data
        int_list = list(rescale_array(np.array(int_list), arr_min=0, upper=100))
        mask = (np.array(int_list) > ignore_less) & (np.array(tth_list) > tth_range[0])

        data = {
            'hkl' : np.array(hkl_list)[mask],
            'q' : np.array(q_list)[mask],
            'int' : np.array(int_list)[mask],
            'tth' : np.array(tth_list)[mask],
            'd' : np.array(d_list)[mask]
            }
        
        if save_reflections:
            self.reflections = data
        else:
            return data

    # Returns full recirpocal lattice
    # TODO: all qmin trimming
    def generate_reciprocal_lattice(qmax=None,
                                    return_values=True):

        if qmax is None:
            q_max = tth_2_q(self.tth_range[1], wavelength=self.wavelength)

        all_hkls = list(self.lattice.get_allowed_hkl(qmax=qmax))
        all_qs = self.Q(all_hkls)
        all_fs = np.abs(self.StructureFactor(all_qs))**2
        rescale_array(all_fs, arr_min = 0, upper=100)

        self.all_hkls = all_hkls
        self.all_qs = all_qs
        self.all_fs = all_fs

        if return_values:
            return self.all_hkls, self.all_qs, self.all_fs

    
    def planeDistances(self, hkl_lst):
        # Re-write of planeDistance to accomadate multiple plances at once
        return 2 * np.pi / np.linalg.norm(self.Q(hkl_lst), axis=1)
    

    # Horribly optimized...
    # OPTIMIZE ME
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

#######################
### Other Functions ###
#######################


# filedir and filename need conditionals
# Should probably wrap this into the phase class as well...
# Need to better identify and consolidate peaks with convolution as well
# Overall this is very old and needs to be rewritten
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

# WIP...obviously...
def approximate_powder_xrd(xrd_map, poni, energy=None, background=None):
    raise NotImplementedError()


def phase_selector(xrd, phases, tth, ignore_less=1):
    # TODO:
    # Add 2d plot for better understanding of peak character and significance
    # Add background subtraction?
        # I shouldn't have to do that if the 2d background is working correctly yes???

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=200)
    colors = matplotlib.color_sequences['tab10']

    norm_xrd_int = rescale_array(xrd, upper=100, arr_min=0)

    [phase.get_hkl_reflections(tth_range=(np.min(tth), np.max(tth)),
                               ignore_less=ignore_less)
        for phase in phases];

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


 # Unused
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
    