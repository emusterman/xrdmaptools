import numpy as np
import os
import re
from scipy import constants
from xrayutilities.materials.material import Amorphous
from xrayutilities.materials import elements

# Local imports
from xrdmaptools.utilities.math import (
    energy_2_wavelength,
)

# Subclasses xrayutilites Amorphous class to generate a gas glass for
# ion chamber absorption calculations
# Probably overkill for what I am using this for
class Gas(Amorphous):

    def __init__(self,
                 name,
                 density=None,
                 atoms=None,
                 pressure=101325, # Room pressure in Pa
                 temperature=293.15 # Room temperature in K
                 ):
        
        # Define some ideal gas paramters
        self._pressure = pressure
        self._temperature = temperature

        molecular_weight = self.get_molecular_weight(
                                name=name,
                                atoms=atoms
                                )
        self.molecular_weight = molecular_weight

        if density is None:
            density = self.get_ideal_gas_mass_density()

        super().__init__(name, density, atoms=atoms, cij=None)
    

    # Wrapper
    def __repr__(self):
        return self.__str__()
    

    @property
    def pressure(self):
        return self._pressure

    @pressure.setter
    def pressure(self, pressure):
        self._pressure = pressure
        self._density = self.get_ideal_gas_mass_density()
    

    @property
    def temperature(self):
        return self._temperature
    
    @temperature.setter
    def temperature(self, temperature):
        self._temperature = temperature
        self._density = self.get_ideal_gas_mass_density()

    
    def get_ideal_gas_molar_density(self,
                                    pressure=None,
                                    temperature=None):

        # Fill with internal parameters
        if pressure is None:
            pressure = self.pressure
        if temperature is None:
            temperature = self.temperature
        
        return pressure / (constants.R * temperature) # mol / m^3
    
    
    def get_ideal_gas_mass_density(self,
                                   molecular_weight=None,
                                   pressure=None,
                                   temperature=None):

        # Fill with internal parameters
        if molecular_weight is None:
            molecular_weight = self.molecular_weight
        
        # mol / m^3
        rho = self.get_ideal_gas_molar_density(pressure=pressure,
                                          temperature=temperature)
    
        return rho * molecular_weight # kg / m^3


    def get_molecular_weight(self,
                             name=None,
                             atoms=None):
        
        if name is None:
            name = self.name
        if atoms is None and hasattr(self, 'base'):
            atoms = self.base

        # Figure out molar fraction of atoms in gas
        if atoms is None:
            atoms = []
            #print(name)
            comp = Gas.parse_gas_formula(name)
            for (e, c) in comp:
                atoms.append((e, c))
        else:
            parsed_atoms = []
            for at, fr in atoms:
                if not isinstance(at, Atom):
                    a = getattr(elements, at)
                else:
                    a = at
                parsed_atoms.append((a, fr))
            atoms = parsed_atoms
        
        # Get molar weight
        molecular_weight = 0
        for at, frac in atoms:
            molecular_weight += at.weight * constants.N_A * frac
        
        return molecular_weight


    # Overwrite to allow for greater than 1 sums
    @staticmethod
    def parse_gas_formula(cstring):
        if re.findall(r'[\(\)]', cstring):
            raise ValueError(
                f"unsupported chemical formula ({cstring}) given.")
        elems = re.findall('[A-Z][^A-Z]*', cstring)
        r = re.compile(r"([a-zA-Z]+)([0-9\.]+)")
        ret = []
        csum = 0
        for e in elems:
            if r.match(e):
                elstr, cont = r.match(e).groups()
                cont = float(cont)
            else:
                elstr, cont = (e, 1.0)
            ret.append((elstr, cont))
        for i, r in enumerate(ret):
            ret[i] = (getattr(elements, r[0]), r[1])
        return ret

    
    def get_absorption(self,
                       energy,
                       length):

        if energy < 1000:
            energy *= 1000
        
        wavelength = energy_2_wavelength(energy) # in A

        absorption_coefficient = ((4 * np.pi * self.ibeta(en=energy))
                                  / wavelength)
        
        absorption = 1 - np.exp(-absorption_coefficient
                                  * length * 1e8 # cm -> A
                                  )
        return absorption