from types import NoneType
from xmlrpc.client import Boolean
import gym
from dm_control import mujoco
from dm_control import viewer
from dm_control.rl import control
from dm_control.suite import base
from dm_control import mjcf
import dm_env

from gym.spaces import Dict, Box

import numpy as np

import collections


# loading the cv2 to resize the render
import cv2
from torch import DictType



def get_model_and_assets_xml(path:str):
	""" Get the xml path of the model and assets and load them into Mujoco. """
	return open(path, 'r').read()

def Hopper6(time_limit:int=10, random:NoneType=None, environment_kwargs:NoneType|DictType=None):
	xml_string = get_model_and_assets_xml(environment_kwargs['path'])
	print(f"path: {environment_kwargs['path']}")
	model = mjcf.from_path(environment_kwargs['path'])

	# get hip joint
	# Need to add this in the environment_kwargs
	hip_joint = model.find('joint', 'hip')
	hip_joint = add_position_actuator(hip_joint, [-15, 15], [-1, 1])
	knee_joint = model.find('joint', 'knee')
	knee_joint = add_position_actuator(knee_joint, [-0.001, 0.001], [-1, 1], kp =1)


	# physics = mujoco.Physics.from_xml_string(xml_string)
	physics = mjcf.Physics.from_mjcf_model(model)
	environment_kwargs = environment_kwargs or {}
	task = HopperParkour(physics, random=None, environment_kwargs=environment_kwargs)
	return HopperEnvWrapper(physics, task=task, time_limit=time_limit, control_timestep=0.05)


# TODO: change the location when refactor.
# copied from https://github.com/deepmind/dm_control/blob/main/dm_control/locomotion/walkers/scaled_actuators.py
def add_position_actuator(target: mjcf.Element, qposrange:list, ctrlrange:tuple =(-1, 1),
                          kp:int=10.0, **kwargs):
  """Adds a scaled position actuator that is bound to the specified element.
  This is equivalent to MuJoCo's built-in `<position>` actuator where an affine
  transformation is pre-applied to the control signal, such that the minimum
  control value corresponds to the minimum desired position, and the
  maximum control value corresponds to the maximum desired position.
  Args:
    target: A PyMJCF joint, tendon, or site element object that is to be
      controlled.
    qposrange: A sequence of two numbers specifying the allowed range of target
      position.
    ctrlrange: A sequence of two numbers specifying the allowed range of
      this actuator's control signal.
    kp: The gain parameter of this position actuator.
    **kwargs: Additional MJCF attributes for this actuator element.
      The following attributes are disallowed: `['biastype', 'gainprm',
      'biasprm', 'ctrllimited', 'joint', 'tendon', 'site',
      'slidersite', 'cranksite']`.
  Returns:
    A PyMJCF actuator element that has been added to the MJCF model containing
    the specified `target`.
  Raises:
    TypeError: `kwargs` contains an unrecognized or disallowed MJCF attribute,
      or `target` is not an allowed MJCF element type.
  """
#   _check_target_and_kwargs(target, **kwargs)
  kwargs[target.tag] = target

  slope = (qposrange[1] - qposrange[0]) / (ctrlrange[1] - ctrlrange[0])
  g0 = kp * slope
  b0 = kp * (qposrange[0] - slope * ctrlrange[0])
  b1 = -kp
  b2 = 0
  return target.root.actuator.add('general',
                                  biastype='affine',
                                  gainprm=[g0],
                                  biasprm=[b0, b1, b2],
                                  ctrllimited=True,
                                  ctrlrange=ctrlrange,
                                  **kwargs)




# Creating a env with
class HopperEnv(control.Environment):
	def __init__(self, physics:mujoco.Physics, task, time_limit:float=10, control_timestep:NoneType=None,
					n_sub_steps:NoneType=None, flat_observation:bool=False):
		super().__init__(physics, task, time_limit, control_timestep=control_timestep)






# this is bad but I have to use subvecprocess but it does not account for the timestep of dm_control env.
class HopperEnvWrapper(gym.Env):
	metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": int(1/0.05)}
	def __init__(self,  physics:mujoco.Physics, task, time_limit:float=10, control_timestep:NoneType=None,
					n_sub_steps:NoneType=None, flat_observation:bool=False):

		self.env = HopperEnv(physics, task, time_limit, control_timestep, n_sub_steps, flat_observation)
		super(HopperEnvWrapper, self).__init__()
		self._set_observation_space()

	def _set_observation_space(self):
		obs = self.env.task.get_observation(self.env.physics)
		if self.env.task._observation_mode == 'render':
			shape_1 = obs['image'].shape[0]
			shape_2 = obs['image'].shape[1]
			shape_3 = obs['image'].shape[2]
			shape_data = Box( low = 0, high = 255, shape = (shape_1, shape_2,shape_3))
			self.observation_space = Dict({'image':shape_data})
		elif self.env.task._observation_mode == 'state':
			shape_ = obs['state'].shape[0]
			shape_data = Box( low = -np.inf, high = np.inf, shape = (shape_,))
			self.observation_space = Dict({"state":shape_data})
		elif self.env.task._observation_mode == 'range':
			shape_ = obs['range'].shape[0]
			shape_data = Box( low = -np.inf, high = np.inf, shape = (shape_,))
			self.observation_space = Dict({"range":shape_data})

		# set action space
		self.action_space = Box(low = np.array([-0.2,-0.01]), high= np.array([0.2, 0.01]), shape = (2,))

	def step(self, action):
		timestep, reward, discount, obs =  self.env.step(action)
		obs, reward, done, info =  self._convert_output(timestep, reward, discount, obs)

		# since the action is not taken care in the deepmind env, we need to do it here.
		if reward is not None:
			reward -= np.sqrt(np.sum(np.square(action)))
		# print("action: ", action, reward)


		# if done:
			# print(f"done: {self.env.task.model_path}, reward: {reward}")
		return obs, reward, done, info

	def reset(self):
		timestep, reward, discount, obs =  self.env.reset()
		obs, reward, done, info =  self._convert_output(timestep, reward, discount, obs)
		return obs

	def render(self, mode='human'):
		return self.env._physics.render(camera_id = "camera")


	def _convert_output(self, timestep, reward, discount, obs):
		# Convert output from dm_control to gym format
		if timestep == dm_env.StepType.LAST:
			return obs, reward, True, {}
		else:
			return obs, reward, False, {}










class HopperParkour(base.Task):
	metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": int(1/0.05)}
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

		self.model_path = environment_kwargs.get('path')

		super().__init__(random)



	def get_reward(self, physics: mujoco.Physics) -> float:
		# start the rewards
		reward = 0
		if self._position_reward_cost:
			reward += self._position_reward(physics)
		if self._alive_bonus > 0:
			reward += self._alive_bonus
		if self._velocity_cost > 0:
			reward += physics.named.data.qvel[['hip']][0] * self._velocity_cost
		return reward



	def _position_reward(self, physics: mujoco.Physics):
		"""
		Reward for traveling forward.
		"""
		reward = 0
		reward += self._physics.named.data.xpos[['torso'] , 'x'][0]

		return reward


	def get_observation(self, physics):
		"""
		Returns an observation of the state.
		"""
		obs = collections.OrderedDict()
		if self._observation_mode == 'render':
			obs = {'image':  self._get_render(physics)}
			return obs
		elif self._observation_mode == 'state':
			obs['state'] = self._get_state(physics)
			return obs
		elif self._observation_mode == 'range':
			obs['range'] = self._get_range(physics)
			return obs
		else:
			raise NotImplementedError

	def _get_range(self, physics):
		"""
		Returns an observation of the state.
		"""
		# obs = physics.named.data.xpos[['torso']].reshape(-1)

		obs = physics.data.sensordata.__array__()
		return obs.flatten().astype(np.float32)

	def _get_render(self, physics):
		"""
		Returns an observation of the state.
		"""
		image = physics.render(camera_id = "camera", depth = True)

		# Display the contents of the first channel, which contains object
		# IDs. The second channel, seg[:, :, 1], contains object types.
		geom_ids = image
		# Infinity is mapped to -1

		geom_ids = geom_ids.astype(np.float64) + 1
		# Scale to [0, 1]
		# print(geom_ids.shape)
		geom_ids = geom_ids / geom_ids.max()
		pixels = 255*geom_ids
		img = pixels.astype(np.uint8)
		return img[np.newaxis, :, :]

	def _get_state(self, physics):
		"""
		Returns an observation of the state.
		"""
		return physics.named.data.qpos[['torso']]

	def get_termination(self, physics) -> bool|NoneType:
		get_z_distance = physics.named.data.xpos[['torso'], 'z'][0]
		get_x_distance = physics.named.data.xpos[['torso'], 'x'][0]
		get_z_leg = physics.named.data.xpos[['leg'], 'z'][0]
		if get_z_distance < 1.2 or get_z_distance > 4.5 or get_z_leg > 3.5 or get_x_distance < -0.5:
			return 1
		elif physics.time() > self._time_limit:
			return 1
		else:
			# no discount
			return None





