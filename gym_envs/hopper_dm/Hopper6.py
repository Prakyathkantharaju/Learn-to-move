from types import NoneType
from xmlrpc.client import Boolean
import gym
from dm_control import mujoco
from dm_control import viewer
from dm_control.rl import control
from dm_control.suite import base

from gym.spaces import Dict, Box

import numpy as np

import collections


# loading the cv2 to resize the render
import cv2
from torch import DictType



def get_model_and_assets_xml():
	""" Get the xml path of the model and assets and load them into Mujoco. """
	return open('gym_envs/hopper_dm/mujoco_models/hopper_parkour.xml', 'r').read()

def Hopper6(time_limit:int=10, random:NoneType=None, environment_kwargs:NoneType|DictType=None):
	xml_string = get_model_and_assets_xml()
	physics = mujoco.Physics.from_xml_string(xml_string)
	pixels = physics.render()
	environment_kwargs = environment_kwargs or {}
	task = HopperParkour(physics, random=None, environment_kwargs=environment_kwargs)
	return HopperEnv(physics, task=task, time_limit=time_limit)




# Creating a env with
class HopperEnv(control.Environment, gym.Env):
	def __init__(self, physics:mujoco.Physics, task, time_limit:float=10, control_timestep:NoneType=None, 
					n_sub_steps:NoneType=None, flat_observation:bool=False):
		super().__init__(physics, task, time_limit)
		super(gym.Env, self).__init__()
		self._set_observation_space()

	def _set_observation_space(self):
		obs = self.task.get_observation(self.physics)
		if self.task._observation_mode == 'render':
			shape_ = obs['render'].shape[0]
			shape_data = Box( low = 0, high = 255, shape = (shape_,))
			self.observation_space = Dict({"render":shape_data})
		elif self.task._observation_mode == 'state':
			shape_ = obs['state'].shape[0]
			shape_data = Box( low = -np.inf, high = np.inf, shape = (shape_,))
			self.observation_space = Dict({"state":shape_data})
		
		# set action space
		self.action_space = Box(low = -1, high=1, shape = (2,))

			
		




class HopperParkour(base.Task):
	def __init__(self, physics: mujoco.Physics ,random=None, environment_kwargs=None):

		self._physics = physics
		if environment_kwargs.get('alive_bonus') is not None:
			self._alive_bonus = environment_kwargs.get('alive_bonus')
		else:
			self._alive_bonus = 0
		
		if environment_kwargs.get('velocity_cost') is not None:
			self._velocity_cost = environment_kwargs.get('velocity_cost')
		else:
			self._velocity_cost = 0

		if environment_kwargs.get('time_limit') is not None:
			self._time_limit = environment_kwargs.get('time_limit')
		else:
			self._time_limit = float('inf')

		if environment_kwargs.get('position_reward') is not None:
			self._position_reward_cost = environment_kwargs.get('position_reward')
		else:
			self._position_reward_cost = False

		if environment_kwargs.get('observation_mode') is not None:
			self._observation_mode = environment_kwargs.get('observation_mode')
		else:
			self._observation_mode = 'render'

		if environment_kwargs.get('observation_scale') is not None:
			self._observation_scale  = 0.5
		else:
			self._observation_scale = 1
			
		super().__init__(random)



	def get_reward(self, physics: mujoco.Physics) -> float:
		# start the rewards
		reward = 0
		if self._position_reward_cost:
			reward += self._position_reward(physics)
		if self._alive_bonus > 0:
			reward += self._alive_bonus
		if self._velocity_cost > 0:
			reward += self._velocity_cost * physics.data.qvel[['torso'] , 'x']
		return reward

	def _position_reward(self, physics: mujoco.Physics):
		"""
		Reward for traveling forward.
		"""
		reward = 0
		reward += self._physics.named.data.xpos[['torso'] , 'x'].tolist()[0]
		return reward


	def get_observation(self, physics):
		"""
		Returns an observation of the state.
		"""
		obs = collections.OrderedDict()
		if self._observation_mode == 'render':
			obs['render'] = self._get_render(physics)
			return obs
		elif self._observation_mode == 'state':
			obs['state'] = self._get_state(physics)
			return obs
		else:
			raise NotImplementedError

	def _get_render(self, physics):
		"""
		Returns an observation of the state.
		"""
		img = physics.render(height=256, width=256)
		img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		img = cv2.resize(img, (256 * self._observation_scale, 256 * self._observation_scale))
		return img.flatten()


	def _get_state(self, physics):
		"""
		Returns an observation of the state.
		"""
		return physics.named.data.qpos[['torso']]
		


		