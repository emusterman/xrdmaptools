
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
__date__ = "03/14/2024" # MM/DD/YYYY


# This version is not currently published!
__version__ = '0.1.0'


__submodules__ = [
    'crystal',
    'geometry',
    'io',
    'reflections',
    'utilities',
    'plot',
    'gui'
]

__base_classes__ = [
    'XRDData',
    'XRDBaseScan',
    'XRDMap',
    'XRDRockingCurve',
    'XRDMapStack'
]


# This is required for wildcard (*) imports
__all__ = [s for s in dir() if not s.startswith('_')]
__all__ += __submodules__ + __base_classes__


# Import submodules as necessary
def __getattr__(attr):
    import importlib

    if attr in __base_classes__:
        mod = importlib.import_module(f'{__name__}.{attr}')
        # Base classes are brought up one level for convenience
        globals()[attr] = getattr(mod, attr)
        return globals()[attr]   
    elif attr in __submodules__:
        mod = importlib.import_module(f'{__name__}.{attr}')
        return mod
    else:
        err_str = f"module {__name__} has no attribute {attr}"
        raise AttributeError(err_str)
