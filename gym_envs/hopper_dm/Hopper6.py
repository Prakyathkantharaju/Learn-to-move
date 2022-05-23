from dm_control import mujoco
from dm_control import viewer
from dm_control.rl import control
from dm_control.suite import base
import numpy as np


class Hopper6(control.Environment):
    def __init__(self, xml_path : str) -> None:
        self.xml_string =  open(xml_path, 'r').read()
        physics = mujoco.Physics.from_xml_string(self.xml_string)
        pixels = physics.render()
        task = HopperParkour(physics, random=None)
        super().__init__(physics, task=task)
        print(pixels.shape)


    # def get_observation(self, physics):
    #     pixels = self.physics.render()
    #     return pixels.flatten()


    # def get_reward(self, physics):
    #     return 1


    # def action_spec(self):
    #     return super().action_spec(self.physics)


    # def reset(self):
    #     self.physics.reset()
    #     return self.physics


    # def step(self, action):
    #     self.physics.step(action)
    #     return self.physics


class HopperParkour(base.Task):
    def __init__(self, physics ,random=None):
        self._physics = physics
        super().__init__(random)



    def get_reward(self, physics):
        return 1


    def get_observation(self, physics):
        return physics.render().flatten()
        