
# Module for working with scanning monochromatic X-ray diffraction data
# =====================================================================
#
# This Python module is intended to automate data
# processing of scanning monochromatic diffraction data acquired from
# sychrotron beamlines. It was designed originally for data acquired at
# the 5-ID (SRX) beamline at at NSLS-II, but is intended to be generally
# applicable to similarly acquired data.
#
# No other documentation is currently avialable

__author__ = ['Evan J. Musterman',  'Andrew M. Kiss']
__contact__ = 'emusterma@bnl.gov'
__license__ = 'N/A'
__copyright__ = 'N/A'
__status__ = 'testing'
__date__ = "03/14/2024" # DD/MM/YYYY


# This version is not currently published!
__version__ = '0.1.0'


submodules = [
    'crystal',
    'geometry',
    'io',
    'reflections',
    'utilities',
    'plot'
]

# This is required for wildcard (*) imports
__all__ = submodules + [
    'XRDData',
    'XRDBaseScan',
    'XRDMap',
    'XRDRockingCurve',
    'XRDMapStack'
]


def __dir__():
    return __all__