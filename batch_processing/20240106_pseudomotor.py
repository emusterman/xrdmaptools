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


class ProjectedTopStage(PseudoPositioner):

    # Pseudo axes
    projx = Cpt(PsuedoSingle)
    projz = Cpt(PsuedoSingle)

    # Real axes. From XRXNanoStage class definition.
    topx = Cpt(EpicsMotor, 'XF:05IDD-ES:1{nKB:Smpl-Ax:xth}Mtr')  # XF:05IDD-ES:1{nKB:Smpl-Ax:xth}Mtr.RBV
    topz = Cpt(EpicsMotor, 'XF:05IDD-ES:1{nKB:Smpl-Ax:zth}Mtr')  # XF:05IDD-ES:1{nKB:Smpl-Ax:zth}Mtr.RBV

    # Configuration signals
    th = Cpt(EpicsSignalRO, 'XF:05IDD-ES:1{nKB:Smpl-Ax:th}Mtr.RBV')  # XF:05IDD-ES:1{nKB:Smpl-Ax:th}Mtr.RBV
    velocity_x = Cpt(EpicsSignal, 'XF:05IDD-ES:1{nKB:Smpl-Ax:xth}Mtr.VELO')
    velocity_z = Cpt(EpicsSignal, 'XF:05IDD-ES:1{nKB:Smpl-Ax:zth}Mtr.VELO')
    acceleration_x = Cpt(EpicsSignalRO, 'XF:05IDD-ES:1{nKB:Smpl-Ax:xth}Mtr.ACCL.RBV')
    acceleration_z = Cpt(EpicsSignalRO, 'XF:05IDD-ES:1{nKB:Smpl-Ax:zth}Mtr.ACCL.RBV')
    motor_egu_x = Cpt(EpicsSignalRO, 'XF:05IDD-ES:1{nKB:Smpl-Ax:xth}Mtr.EGU.RBV')
    motor_egu_z = Cpt(EpicsSignalRO, 'XF:05IDD-ES:1{nKB:Smpl-Ax:zth}Mtr.EGU.RBV')

    # Create projected signals to read
    # Consider replacing with DerivedSignal class
    velocity = Cpt(Signal, '.VELO', add_prefix=(), kind='config', value=0)
    acceleration = Cpt(SignalRO, '.ACCL', add_prefix=(), kind='config', value=0)
    motor_egu = Cpt(SignalRO, '.EGU', add_previs=(), kind='config', value=0)

    # # user_readback = Cpt(DerivedSignal, ".RBV", kind="hinted", auto_monitor=True)
    # projx.user_readback = Cpt(DerivedSignal([topx, topz], write_access=False), '.RBV')
    # projz.user_readback = Cpt(DerivedSignal([topx, topz], write_access=False), '.RBV')
    # def projx_user_readback_inverse(self, value):
    # def projz_user_readback_inverse(self, value):

    # Create modified user_readback functions
    projx.user_readback = Cpt(SignalRO, '.RBV', kind='hinted', auto_monitor=True)
    projz.user_readback = Cpt(SignalRO, '.RBV', kind='hinted', auto_monitor=True)

    # Overwrite .get()
    def user_readback_get(self, axis=0):
        return self._inverse(topx.get(), topz.get())[0]
    projx.user_readback.get = lambda : self.user_readback_get(axis=0)
    projz.user_readback.get = lambda : self.user_readback_get(axis=1)


    def __init__(self,
                 *args
                 projected_axis=None,
                 **kwargs):
        super().__init__(*arg, **kwargs)
        
        # Store projected axis for determining projected velocity
        if projected_axis is None:
            err_str = "Must define projected_axis as 'x' or 'z'."
            raise ValueError(err_str)
        elif str(projected_axis).lower() not in ['x', 'z']:
            err_str = ("ProjectedTopStage axis only supported for 'x' "
                       + f"or 'z' projected axis not {projected_axis}.")
            raise ValueError(err_str)
        self._axis = str(projected_axis).lower()

        # Define defualt projected signals
        velocity = min([self.velocity_x.get(),
                        self.velocity_z.get()])
        acceleration = min([self.acceleration_x.get(),
                            self.acceleration_z.get()])
        if motor_egu_x.get() == motor_egu_z.get():
            motor_egu = motor_egu_x.get()
        else:
            err_str = (f'topx motor_egu of {motor_egu_x.get()} does '
                       + 'not match topz motor_egu of '
                       + f'{motor_egu_z.get()}')
            raise AttributeError(err_str)

        self.velocity.set(velocity)
        self.acceleration.set(acceleration)
        self.motor_egu.set(motor_egu)

        # Set velocity limits
        velocity_limits = (
            max([self.velocity_x.low_limit,
                 self.velocity_z.low_limit])
            min([self.velocity_x.high_limit,
                 self.velocity_z.high_limit])
        )
        self.velocity.limits = property(lambda : velocity_limits)


    # Convenience function to get rotation matrix between 
    # rotated top stage axes and projected lab axes
    def R(self):
        th = self.th.get()
        th = np.radians(th / 1000) # to radians
        return np.array([[np.cos(th), np.sin(th)],
                         [-np.sin(th), np.cos(th)]])
    

    # Function to change component motor velocities
    def set_component_velocities(self,
                                 topx_velocity=None,
                                 topz_velocity=None):
        
        bool_flags = sum([topx_velocity is None,
                          topz_velocity is None])

        if bool_flags == 1:
            err_str = ('Must specify both topx_velocity and '
                       + 'topz_velocity or neither.')
            raise ValueError(err_str)
        elif bool_flags == 2:
            # Determine component velocities from projected
            velocity = self.velocity.get()
            if projected_axis == 'x':
                velocity_vector = [velocity, 0]
            else:
                velocity_vector = [0, velocity]

            (topx_velocity,
             topz_velocity) = self.R() @ velocity_vector
        
        if topx_velocity < 1e-8: # too small
            topx_velocity = np.max([1e-8, self.topx.velocity.low_limit])
        if topz_velocity < 1e-8: # too small
            topz_velocity = np.max([1e-8, self.topz.velocity.low_limit])
        
        self.velocity_x.set(topx_velocity)
        self.velocity_z.set(topz_velocity)

    
    # Wrap move function with stage_sigs-like behavior
    def move(self, *args, **kwargs):
        # Get starting velocities
        start_topx_velocity = self.velocity_x.get()
        start_topz_velocity = self.velocity_z.get()
        
        # Set component velocities based on internal velocity signal
        self.set_component_velocities()

        # Move like normal
        super().move(*args, **kwargs)
        
        # Reset component velocities to original values
        self.set_component_velocity(
                    topx_velocity=start_topx_velocity,
                    topz_velocity=start_topz_velocity)


    def _forward(self, projx, projz):
        #     # |topx|   | cos(th)  sin(th)| |projx|
        #     # |topz| = |-sin(th)  cos(th)| |projz|
        return self.R().T @ [projx, projz]

    
    def _inverse(self, topx, topz):
        #     # |projx|   |cos(th)  -sin(th)| |topx|
        #     # |projz| = |sin(th)   cos(th)| |topz|
        return self.R() @ [topx, topz]


    @pseudo_position_argument
    def forward(self, p_pos):
        projx = p_pos.projx
        projz = p_pos.projz
        topx, topz = self._forward(projx, projz)
        return self.RealPosition(topx=topx, topz=topz)


    @real_position_argument
    def inverse(self, r_pos):
        topx = r_pos.topx
        topz = r_pos.topz
        projx, projz = self._inverse(topx, topz)
        return self.PseudoPosition(projx=projx, projz=projz)


    # @pseudo_position_argument
    # def forward(self, p_pos):
    #     projx = p_pos.projx
    #     projz = p_pos.projz

    #     # |topx|   | cos(th)  sin(th)| |projx|
    #     # |topz| = |-sin(th)  cos(th)| |projz|

    #     # topx = projx * np.cos(th) - projz * np.sin(th)
    #     # topz = projx * np.sin(th) + projz * np.sin(th)
    #     topx, topz = self.R().T @ [projx, projz]

    #     return self.RealPosition(topx=topx, topz=topz)

    
    # @real_position_argument
    # def inverse(self, r_pos):
    #     new_topx = r_pos.topx
    #     new_topz = r_pos.topz
    #     # R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])

    #     # |projx|   |cos(th)  -sin(th)| |topx|
    #     # |projz| = |sin(th)   cos(th)| |topz|

    #     # projx = topx * np.cos(th) + topz * np.sin(th)
    #     # projz = -topx * np.sin(th) + topz * np.sin(th)
    #     projx, projz = self.R() @ [topx, topz]

    #     return self.PseudoPosition(projx=projx, projz=projz)


# proj_stage = ProjectedTopStage(name='projected_top_stage')

projx = ProjectedTopStage(name='projected_top_x', projected_axis='x')
projz = ProjectedTopStage(name='projected_top_z', projected_axis='z')