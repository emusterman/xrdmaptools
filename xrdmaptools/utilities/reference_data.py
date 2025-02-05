import numpy as np
import os


# Local imports
from xrdmaptools.utilities.math import (
    energy_2_wavelength,
    wavelength_2_energy,
    tth_2_d,
    d_2_tth,
    convert_qd,
    q_2_tth,
    tth_2_q
)
from xrdmaptools.utilities.gas_references import Gas

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

gases = {
    'He' : Gas('He'),
    'N2' : Gas('N2'),
    # Doubled mole fraction of N and O and regular mole fraction of Ar
    'air' : Gas('N1.56168O0.41892Ar0.00934'), 
    'Ne' : Gas('Ne'),
    'Ar' : Gas('Ar'),
    'Kr' : Gas('Kr'),
    'Xe' : Gas('Xe')
}


# From Table 4-3 in CXRO Handbook
average_gas_ionization_energies = {
    'He' : 41,
    'N2' : 36,
    'air' : 34.4,
    'Ne' : 36.3,
    'Ar' : 26, 
    'Kr' : 24,
    'Xe' : 22
}