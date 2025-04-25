import xrayutilities as xu
from xrayutilities.materials.spacegrouplattice import (
    RangeDict,
    sgrp_sym,
    sgrp_name,
    sgrp_params,
    SGLattice
)
from xrayutilities.materials.atom import Atom
import matplotlib
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from matplotlib.widgets import Slider
from collections import OrderedDict
from copy import deepcopy

# Local imports
from xrdmaptools.utilities.math import (
    energy_2_wavelength,
    tth_2_q,
    q_2_tth,
    convert_qd,
    vector_angle
)
from xrdmaptools.utilities.utilities import rescale_array


### Useful crystallography tools ###

# Point group from space group
point_grp = RangeDict({# Space group number, (Schoenflies, Hermann-Mauguin, Order)
                       # Triclinic
                       range(1, 2): ('C1', '1', 1),
                       range(2, 3): ('Ci', '-1', 2),
                       # Monoclinic
                       range(3, 6): ('C2', '2', 2),
                       range(6, 10): ('CS', 'm', 2),
                       range(10, 16): ('C2h', '2/m', 4),
                       # Orthorhombic
                       range(16, 25): ('D2', '222', 4),
                       range(25, 47): ('C2v', 'mm2', 4),
                       range(47, 75): ('D2h', 'mmm', 8),
                       # Tetragonal
                       range(75, 81): ('C4', '4', 4),
                       range(81, 83): ('S4', '-4', 4),
                       range(83, 89): ('C4h', '4/m', 8),
                       range(89, 99): ('D4', '422', 8),
                       range(99, 111): ('C4v', '4mm', 8),
                       range(111, 123): ('D2d', '-42m', 8),
                       range(123, 143): ('D4h', '4/mmm', 16),
                       # Trigonal
                       range(143, 147): ('C3', '3', 3),
                       range(147, 149): ('S6', '-3', 6),
                       range(149, 156): ('D3', '32', 6),
                       range(156, 162): ('C3v', '3m', 6),
                       range(162, 168): ('D3d', '-3m', 12),
                       # Hexagonal
                       range(168, 174): ('C6', '6', 6),
                       range(174, 175): ('C3h', '-6', 6),
                       range(175, 177): ('C6h', '6/m', 12),
                       range(177, 183): ('D6', '622', 12),
                       range(183, 187): ('C6v', '6mm', 12),
                       range(187, 191): ('D3h', '-6m2', 12),
                       range(191, 195): ('D6h', '6/mmm', 24),
                       # Cubic
                       range(195, 200): ('T', '23', 12),
                       range(200, 207): ('Th', 'm-3', 24),
                       range(207, 215): ('O', '432', 24),
                       range(215, 221): ('Td', '-43m', 24),
                       range(221, 231): ('Oh', 'm-3m', 48),
                       })

# Laue grp from space group
laue_grp_nr = RangeDict({# Space group number, (Schoenflies, Hermann-Mauguin, Order)
                       # Triclinic
                       range(1, 3): 2,
                       # Monoclinic
                       range(3, 16): 15,
                       # Orthorhombic
                       range(16, 75): 74,
                       # Tetragonal
                       range(75, 89): 88,
                       range(89, 143): 142,
                       # Trigonal
                       range(143, 149): 148,
                       range(149, 168): 167,
                       # Hexagonal
                       range(168, 177): 176,
                       range(177, 195): 194,
                       # Cubic
                       range(195, 207): 206,
                       range(207, 231): 230,
                       })


# Laue grp from space group
laue_grp = RangeDict({# Space group number, (Schoenflies, Hermann-Mauguin, Order)
                       # Triclinic
                       range(1, 3): ('Ci', '-1', 2),
                       # Monoclinic
                       range(3, 16): ('C2h', '2/m', 4),
                       # Orthorhombic
                       range(16, 75): ('D2h', 'mmm', 8),
                       # Tetragonal
                       range(75, 89): ('C4h', '4/m', 8),
                       range(89, 143): ('D4h', '4/mmm', 16),
                       # Trigonal
                       range(143, 149): ('S6', '-3', 6),
                       range(149, 168): ('D3d', '-3m', 12),
                       # Hexagonal
                       range(168, 177): ('C6h', '6/m', 12),
                       range(177, 195): ('D6h', '6/mmm', 24),
                       # Cubic
                       range(195, 207): ('Th', 'm-3', 24),
                       range(207, 231): ('Oh', 'm-3m', 48),
                       })


# Pre-computed Laue group symmetry operations
# For symmetry reduction


class Phase(xu.materials.Crystal):
    # TODO: Add specific calls useful for this analysis
    # Find way to generate Phase class automatically from XRD card files too
    # Add flag to limit functionality depending on if from cif or XRD card

    def __init__(self, name, lat):
        super().__init__(name,
                         lat,
                         cij=None,
                         thetaDebye=None)
        self.reflections = None # Place holder to check later


    def __str__(self):
        return f'{self.name} crystal phase'
    

    def __repr__(self):
        ostr = self.__str__() + '\n'
        ostr += f'\t|{self.chemical_composition()}\t|{self.lattice.name}'
        ostr += f'\n\t|a = {self.a:.4f}\t|b = {self.b:.4f}\t|c = {self.c:.4f}'
        ostr += (f'\n\t|alpha = {self.alpha:.2f}'
                 + f'\t|beta = {self.beta:.2f}'
                 + f'\t|gamma = {self.gamma:.2f}')
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
        name = group.name.split('/')[-1]
        
        params = OrderedDict()
        key_lst = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
        for key in key_lst:
            params[key] = group.attrs[key]
        
        space_group = group.attrs['space_group']
        space_group_number = group.attrs['space_group_number']
        
        # Rebuild lattice from atomic positions and space group
        wbase = group['WyckoffBase']
        atom_lst = []
        for atom_key in wbase.keys():
            atom = Atom(atom_key.split('[')[0], wbase[atom_key].attrs['number'])
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
            dset = wbase.require_dataset(f'{atom[0].name}[{i}]',
                                         data=data,
                                         shape=data.shape,
                                         dtype=data.dtype)
            dset.attrs['number'] = atom[0].num
            dset.attrs['position'] = atom[1][0]

    
    # Simple and to the point
    def copy(self):
        return deepcopy(self)

    
    @property
    def min_q(self):
        return np.linalg.norm(self.Q([[1, 0, 0],
                                      [0, 1, 0],
                                      [0, 0, 1]]), axis=0).min()


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
                            energy,
                            tth_range=None,
                            ignore_less=1,
                            save_reflections=True):

        if tth_range is None:
            tth_range = (0, 90)

        wavelength = energy_2_wavelength(energy)

        all_refl = self.lattice.get_allowed_hkl(qmax=tth_2_q(tth_range[1],
                                                             wavelength=wavelength,
                                                             radians=False))
        all_q = np.linalg.norm(self.Q(*all_refl), axis=1)
        # TODO: Add conditional to remove below a qmin
        all_q = np.round(all_q, 10) # Clean up some errors
        sort_refl = [tuple(x) for _, x in sorted(zip(all_q, all_refl))]
        all_q.sort()
        # Only real values...
        F_hkl = np.abs(self.StructureFactor(sort_refl))**2
        if not isinstance(F_hkl, np.ndarray):
            F_hkl = np.ones((len(sort_refl)))
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
    # TODO: allow qmin trimming
    def generate_reciprocal_lattice(self,
                                    qmax,
                                    return_values=False):
                                    
        all_hkls = list(self.lattice.get_allowed_hkl(qmax=qmax))
        all_qs = self.Q(all_hkls)
        all_fs = np.abs(self.StructureFactor(all_qs))**2
        rescale_array(all_fs, arr_min = 0, upper=100)

        self.all_hkls = np.asarray(all_hkls)
        self.all_qs = np.asarray(all_qs)
        self.all_fs = np.asarray(all_fs) # might be redundant

        if return_values:
            return self.all_hkls, self.all_qs, self.all_fs

    
    def planeDistances(self, hkl_lst):
        # Re-write of planeDistance to accomadate multiple planes at once
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

    
    # Wrapper to relax lattice to P1 version of itself
    def convert_to_P1(self):
        self.lattice = self.lattice.convert_to_P1()

#######################
### Other Functions ###
#######################


# wd and filename need conditionals
# Should probably wrap this into the phase class as well...
# Need to better identify and consolidate peaks with convolution as well
# Overall this is very old and needs to be rewritten
def write_calibration_file(mat, name=None, tt_cutoff=90, ignore_less=1,
                           wd=None, filename=None,
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

    file = open(wd + filename, 'w')
    file.writelines(s + '\n' for s in header + data)
    file.close()

    return header + data

# WIP...obviously...
def approximate_powder_xrd(xrd_map, poni, energy=None, background=None):
    raise NotImplementedError()


def phase_selector(xrd,
                   phases,
                   energy,
                   tth,
                   title=None,
                   ignore_less=1,
                   update_reflections=False,
                   return_plot=False):
    # TODO:
    # Add 2d plot for better understanding of peak character and significance
    # Add background subtraction?
        # I shouldn't have to do that if the 2d background is working correctly yes???
    
    if len(phases) <= 10:
        colors = matplotlib.color_sequences['tab10']
    else:
        colors = matplotlib.color_sequences['tab20']
    norm_xrd_int = rescale_array(xrd, upper=100, arr_min=0)

    for phase in phases:
        phase.get_hkl_reflections(energy,
                                  tth_range=(np.min(tth),
                                  np.max(tth)),
                                  ignore_less=ignore_less)

    # Waiting is odd..
    if update_reflections:
        plt.close('all') 

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=200)
    xrd_plot = ax.plot(tth, norm_xrd_int, label='Composite XRD', c='k', zorder=(len(phases) + 1))
    fig.subplots_adjust(right=0.75)

    def update_max(LineCollection, phase, slider):
        seg = np.asarray(LineCollection.get_segments())
        seg[:, 1, 1] = slider.val * phase.reflections['int'] / np.max([5, np.min(phase.reflections['int'])])
        return seg

    lines = [xrd_plot]
    slider_lst = []
    update_lst = []
    #zero_phases = [phase.name for phase in phases]

    def update_factory(index):
        def update(val):
            lines[index + 1].set_segments(update_max(lines[index + 1], phases[index], slider_lst[index]))                
            fig.canvas.draw_idle()
        return update

    slider_vpos = np.linspace(0.8, 0.1, len(phases))
    for i, phase in enumerate(phases):
        phase_intensities = phase.reflections['int']
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
    ax.legend(fontsize=10)

    if title is None:
        title = 'Phase Selector'
    ax.set_title(title)

    phase_vals = {}
    def on_close(event):
        for phase, slider in zip(phases, slider_lst):
            phase_vals[phase.name] = slider.val
        #print(zero_phases)
        return phase_vals
    
    if update_reflections:
        print('Updating reflections...')
        fig.canvas.mpl_connect('close_event', on_close)
        plt.show(block=True)
        plt.pause(0.01)
        return phase_vals
    else:
        print('Not updating reflections...')
        if return_plot:
            return fig, ax, slider_lst
        else:
            fig.show()
            return slider_lst
        #return phase_vals



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
    