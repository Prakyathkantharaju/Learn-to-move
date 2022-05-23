from dm_control import mujoco
from dm_control import viewer
from dm_control.rl import control
from dm_control.suite import base
import numpy as np



# loading the cv2 to resize the render
import cv2



def get_model_and_assets_xml():
	""" Get the xml path of the model and assets and load them into Mujoco. """
	return open('gym_envs/hopper_dm/mujoco_models/hopper.xml', 'r').read()

def Hopper6(time_limit=10, random=None, environment_kwargs=None):
	xml_string = get_model_and_assets_xml()
	physics = mujoco.Physics.from_xml_string(xml_string)
	pixels = physics.render()
	environment_kwargs = environment_kwargs or {}
	task = HopperParkour(physics, random=None, environment_kwargs=environment_kwargs)
	return control.Environment(physics, task=task, time_limit=time_limit)





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
		reward += self._physics.named.data.xpos[['torso'] , 'x']
		return reward


	def get_observation(self, physics):
		"""
		Returns an observation of the state.
		"""
		if self._observation_mode == 'render':
			return self._get_render(physics)
		elif self._observation_mode == 'state':
			return self._get_state(physics)
		else:
			raise NotImplementedError

	def _get_render(self, physics):
		"""
		Returns an observation of the state.
		"""
		img = physics.render(height=256, width=256)
		img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		img = cv2.resize(img, (256 * self._observation_scale, 256 * self._observation_scale))
		return img


	def _get_state(self, physics):
		"""
		Returns an observation of the state.
		"""
		return physics.named.data.qpos[['torso']]
		


		