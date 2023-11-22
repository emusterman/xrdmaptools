import numpy as np
from pyFAI.detectors._common import Detector
from collections import OrderedDict


# TODO:
# Add mask constuctors?
# Add mask healing functions?
# Add dark field functions?

class Dexela2315(Detector):
    """
    Dexela CMOS family detector
    """
    force_pixel = True
    aliases = ["Dexela 2315"]
    #MAX_SHAPE = (3072, 1944)
    MAX_SHAPE = (1944, 3072)

    def __init__(self, pixel1=74.8e-6, pixel2=74.8e-6, max_shape=None):
        super(Dexela2315, self).__init__(pixel1=pixel1, pixel2=pixel2, max_shape=max_shape)

    def __repr__(self):
        return "Detector %s\t PixelSize= %.3e, %.3e m" % \
            (self.name, self._pixel1, self._pixel2)

    def get_config(self):
        """Return the configuration with arguments to the constructor

        :return: dict with param for serialization
        """
        return OrderedDict((("pixel1", self._pixel1),
                            ("pixel2", self._pixel2)))
    

class MerlinX(Detector):
    """
    MerlinX from Quantum Electronics
    """
    force_pixel = True
    aliases = ["MerlinX"]
    MAX_SHAPE = (512, 512)

    def __init__(self, pixel1=55e-6, pixel2=55e-6, max_shape=None):
        super(MerlinX, self).__init__(pixel1=pixel1, pixel2=pixel2, max_shape=max_shape)

    def __repr__(self):
        return "Detector %s\t PixelSize= %.3e, %.3e m" % \
            (self.name, self._pixel1, self._pixel2)

    def get_config(self):
        """Return the configuration with arguments to the constructor

        :return: dict with param for serialization
        """
        return OrderedDict((("pixel1", self._pixel1),
                            ("pixel2", self._pixel2)))