
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


# This is required for wildcard (*) imports
__all__ = [s for s in dir()]