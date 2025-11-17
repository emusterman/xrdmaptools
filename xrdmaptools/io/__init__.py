
# Submodule for I/O functions and support functions
# =====================================================================
#
# This submodule contains functions for I/O and functions to assist
# with these processes.
# No other documentation is currently avialable

__author__ = ['Evan J. Musterman',  'Andrew M. Kiss']
__contact__ = 'emusterma@bnl.gov'
__license__ = 'N/A'
__copyright__ = 'N/A'
__status__ = 'testing'
__date__ = "06/10/2024" # MM/DD/YYYY


__submodules__ = [
    'db_io',
    'db_utils',
    'hdf_io',
    'hdf_utils'
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