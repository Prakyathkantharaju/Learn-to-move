# createing a base class for the viewer and the simulation

from __future__ import annotations

# general mujoco imports
import mujoco_py
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py.mjviewer import save_video
from mujoco_py.generated import const

import os





class BaseSim:

    def __init__(self, model_path: str, render: bool, timestep: float = 0.01):

        print("Loading model from: {}".format(model_path))
        print(f"Current path {os.getcwd()}")
        self.model = load_model_from_path(model_path)
        # self.model.timestep = timestep
        self.sim = MjSim(self.model)
        self.torso_id = self.model.body_name2id("torso")
        self.leg_id = self.model.body_name2id("leg")
        print("Model loaded")
        print(f"ID for torso {self.torso_id}, legs {self.leg_id}")
        if render:
            print("****setup viewer****")
            self._setup_render()
        else:
            self.viewer = None


        # setup the parameters
        self._setup_params()

        # start the actuation
        self._setup_servo()


    def _setup_params(self) -> None:
        """
         Setup the parameters for the simulation
        """ 
        # actuator id
        self.HIP_SERVO = 0
        self.HIP_VELOCITY_SERVO = 1
        self.KNEE_SERVO = 2
        self.KNEE_VELOCITY_SERVO = 3

        # state ID 
        self.STATE_AIR1 = 0
        self.STATE_STANCE1 = 1
        self.STATE_STANCE2 = 2
        self.STATE_AIR2 = 3
        self.state = self.STATE_AIR1

        # step count
        self.step_count = 0

        

    def _setup_servo(self) -> None:
        # TODO: dataclass
        # pservo- hip
        self._set_position_gain(self.HIP_SERVO, 100)
        # vservo- hip
        self._set_position_gain(self.HIP_VELOCITY_SERVO, 10)
        # pservo- knee
        self._set_position_gain(self.KNEE_SERVO, 1000)
        # vservo- knee
        self._set_position_gain(self.KNEE_VELOCITY_SERVO, 0)


        
    def _get_fsm(self) -> int:
        """
        Get the state of the robot
        """
        # foot pos in z axis
        z_foot = self.sim.data.body_xpos[self.leg_id][2]

        # torso velocity
        torso_vel = self.sim.data.qvel[self.torso_id]

        if self.state == self.STATE_AIR1 and z_foot < 0.05:
            self.state = self.STATE_STANCE1
        elif self.state == self.STATE_STANCE1 and torso_vel > 0:
            self.state = self.STATE_STANCE2
        elif self.state == self.STATE_STANCE2 and z_foot > 0.05:
            self.state = self.STATE_AIR2
        elif self.state == self.STATE_AIR2 and torso_vel < 0:
            self.state = self.STATE_AIR1
            self.step_count += 1

    def _set_torque_gain(self, act_no: int, flag: int):
        """
        Set control torque
        """
        self.model.actuator_gainprm[act_no,0] = flag

    def _set_position_gain(self, act_no: int, gain: float):
        """
        Set control position
        """
        self.model.actuator_gainprm[act_no,0] = gain
        self.model.actuator_biasprm[act_no,1] = -gain

    def _set_vecloity_servo(self, act_no: int, gain: float):
        """
        Set control position
        """
        self.model.actuator_gainprm[act_no,0] = gain
        self.model.actuator_biasprm[act_no,2] = -gain


    def step(self, action: list[float]) -> None:
        # based on the initial testing, position servo values where better than velocity
        # print(f"Action: {action}, state: {self.sim.data.ctrl}")
        self.sim.data.ctrl[0] = action[0]
        self.sim.data.ctrl[2] = action[1]
        self._get_fsm()
        self.sim.step()


    def _setup_render(self) -> None:
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.trackbodyid = self.torso_id
        self.viewer.cam.distance = 10
        self.viewer.cam.lookat[2] = 1.0
        self.viewer.cam.elevation = -20
        self.viewer._record_video = True


    def render(self) -> None:
        if self.viewer is None:
            # raise ModuleNotFoundError("No viewer found")
            # not implemented
            return 0
        self.viewer.render()
        #print(f"frame count: {self.viewer._video_queue}")
        

    def get_state(self) -> list[float]:
        state = []
        state.append(self.sim.data.body_xpos[self.torso_id].tolist())
        state.append(self.sim.data.body_xvelp[self.torso_id].tolist())
        state.append(self.sim.data.body_xpos[self.leg_id].tolist())
        state.append(self.sim.data.body_xvelp[self.leg_id].tolist())
        return np.array(state)


    def reset(self) -> None:
        self.sim.reset()
        self._setup_params()
        self._setup_servo()

    def save_video_from_frame(self, video_path: str) -> None:
        save_video(self.viewer._video_queue, video_path, fps = 100)
        
