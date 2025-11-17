
# Submodule for working and analyzing crystal parameters
# =====================================================================
#
# This submodule contains functions and classes for working with
# crystals and crystal parameters related to monochromatic XRD.
# Reference crystal phases are based around the xrayutilities Crystal
# phase. This subpackage also includes methods for working with crystal
# strain and orientation.
#
# No other documentation is currently avialable

__author__ = ['Evan J. Musterman',  'Andrew M. Kiss']
__contact__ = 'emusterma@bnl.gov'
__license__ = 'N/A'
__copyright__ = 'N/A'
__status__ = 'testing'
__date__ = "03/14/2024" # MM/DD/YYYY


__submodules__ = [
    'crystal',
    'map_alignment',
    'orientation',
    'Phase',
    'rsm',
    'strain',
    'zero_point_correction'
]

__base_classes__ = [
    # 'Phase' # Too many other values in module; not a base class
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