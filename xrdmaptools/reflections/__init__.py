
# Submodule for indentifying, characterizing, and indexing reflections
# =====================================================================
#
# This submodule contains functions for working with X-ray diffraction
# reflections. This includes spots and ring (in 2D) and peaks when full
# patterns are integrated to 1D. These functions are currently limited
# to spots in the current version.
#
# The major roles of these functions are to identify significant
# reflections in their respective patterns, to initially characterize
# the reflections (e.g., FWHM, intensity) and then to initially index
# the reflections according to some reference crystal. Indexing may be 
# shifted to the crystal subpackage in future version.
#
# No other documentation is currently avialable

__author__ = ['Evan J. Musterman',  'Andrew M. Kiss']
__contact__ = 'emusterma@bnl.gov'
__license__ = 'N/A'
__copyright__ = 'N/A'
__status__ = 'testing'
__date__ = "03/14/2024" # MM/DD/YYYY


__submodules__ = [
    'peak_indexing',
    'peak_search',
    'ring_indexing',
    'ring_search',
    'spot_blob_indexing_3D',
    'spot_blob_search_3D',
    'spot_blob_indexing',
    'spot_blob_search',
    'SpotModels'
]

__base_classes__ = [

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