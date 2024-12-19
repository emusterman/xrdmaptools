import numpy as np
import os
from tqdm import tqdm
from skimage import io
import h5py
import time as ttime
from scipy.optimize import curve_fit

from ophyd import (
    EpicsSignal,
    EpicsSignalRO,
    EpicsMotor,
    Device,
    Signal,
    PseudoPositioner,
    PseudoSingle,
)

from ophyd.pseudopos import pseudo_position_argument, real_position_argument
from ophyd.positioner import PositionerBase
from ophyd import Component as Cpt
from ophyd.status import SubscriptionStatus


# class Test():

#     def __init__(self):
#         pass

#     def forward(self, projx, projz, th):
#         th = np.radians(th / 1000) # to radians
#         topx = projx * np.cos(th) - projz * np.sin(th)
#         topz = projx * np.sin(th) + projz * np.sin(th)
#         return topx, topz


#     def inverse(self, topx, topz, th):
#         th = np.radians(th / 1000) # to radians
#         projx = topx * np.cos(th) + topz * np.sin(th)
#         projz = -topx * np.sin(th) + topz * np.sin(th)
#         return projx, projz


# Make relative???
class ProjectedTopStage(PseudoPositioner):

    # Pseudo axes
    projx = Cpt(PsuedoSingle)
    projz = Cpt(PsuedoSingle)

    # Real axes. From XRXNanoStage class definition.
   
    topx = Cpt(EpicsMotor, 'xth}Mtr', read_attrs=['user_readback'])  # XF:05IDD-ES:1{nKB:Smpl-Ax:xth}Mtr.RBV
    topz = Cpt(EpicsMotor, 'zth}Mtr', read_attrs=['user_readback'])  # XF:05IDD-ES:1{nKB:Smpl-Ax:zth}Mtr.RBV

    # Configureation signal
    th = Cpt(EpicsSignalRO, 'th}Mtr', read_attrs=['user_readback'])  # XF:05IDD-ES:1{nKB:Smpl-Ax:th}Mtr.RBV

    # def __init__(self,
    #              *args,
    #              projected_axis=None,
    #              **kwargs):

    #     if projected_aixs is None:
    #         err_str = "Must define projected_axis as 'x' or 'z'"
    #         raise ValueError(err_str)
    #     elif str(projected_axis).lower() not in ['x', 'z']:
    #         err_str = ("ProjectedTopStage axis only supported for 'x' "
    #                    + f"or 'z' projected axis not {projected_axis}." )
    #         raise ValueError(err_str)

    #     self._axis = str(projected_axis).lower()
    #     super().__init__(*arg, **kwargs)


    # # Get current real space coordinates
    # # th returned in radians    
    # def get_current_real_coords(self):

    #     th = self.th.user_readback.get()
    #     th = np.radians(th / 1000) # mdeg to radians
    #     topx = self.topx.user_readback.get()
    #     topz = self.topz.user_readback.get()

    #     return th, topx, topz

    # Convenience function to get rotation matrix between 
    # rotated top stage axes and projected lab axes
    def R(th):
        return np.array([[np.cos(th), np.sin(th)],
                         [-np.sin(th), np.cos(th)]])



    ### UPDATE FROM MACROS ON WS7!!! ###
        

    @pseudo_position_argument
    def forward(self, p_pos):
        projx = p_pos.projx
        projz = p_pos.projz
        th = p_pos.th.get()
        th = np.radians(th / 1000) # to radians
        # R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])

        # |topx|   | cos(th)  sin(th)| |projx|
        # |topz| = |-sin(th)  cos(th)| |projz|

        # topx = projx * np.cos(th) - projz * np.sin(th)
        # topz = projx * np.sin(th) + projz * np.sin(th)
        topx, topz = self.R(th).T @ [projx, projz]

        return self.RealPosition(topx=topx, topz=topz)

    
    @real_position_argument
    def inverse(self, r_pos):
        new_topx = r_pos.topx
        new_topz = r_pos.topz
        th = p_pos.th.get()
        th = np.radians(th / 1000) # to radians
        # R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])

        # |projx|   |cos(th)  -sin(th)| |topx|
        # |projz| = |sin(th)   cos(th)| |topz|

        # projx = topx * np.cos(th) + topz * np.sin(th)
        # projz = -topx * np.sin(th) + topz * np.sin(th)
        projx, projz = self.R(th) @ [topx, topz]

        return self.PseudoPosition(projx=projx, projz=projz)

proj = ProjectedTopStage(name='projected_top_stage')



# WIP: can we create a compucentric rotation?
# May not be able to considering the moveable axis of rotation
# class CompucentricRotation(PseudoPositioner):