import gym
from gym.envs.registration import register
from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder

from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.utils import set_random_seed
#import sys and set the path
import sys
import numpy as np


from stable_baselines3.common.env_checker import check_env



sys.path.append('gym_envs/hopper_dm')
sys.path.append('./')

# load environment
from Hopper6 import Hopper6








environment_kwargs = 	{'alive_bonus': 0.5,
						'velocity_cost': 0.0,
						'time_limit': 10,
						'position_reward':True,
						'observation_mode':'render',
						'path':'gym_envs/hopper_dm/mujoco_models/hopper_parkour.xml'}

env = Hopper6(environment_kwargs=environment_kwargs)


def make_env(seed=0):
	"""
	Create a wrapped, monitored SubprocVecEnv for Hopper
	"""
	def _init():
		print(env.reset())
		env.seed(seed)
		return env
	
	set_random_seed(seed)
	return _init

check_env(env)
env_list = [make_env(0), make_env(1)]

train_env = SubprocVecEnv(env_list, start_method='fork')
# train_env = DummyVecEnv(env_list)
train_env = VecVideoRecorder(train_env, video_folder=f'./run_logs/videos/', record_video_trigger=lambda x: x % 100 == 0, video_length = 200)


model = PPO("MultiInputPolicy", train_env, n_steps=200, 
			n_epochs=10, normalize_advantage = True,  target_kl = 0.5, clip_range=0.4, vf_coef = 0.6, verbose=1)
# model.load("Models_parkour_large_1")

model.learn(total_timesteps=5000)


# model.save(f"./run_logs/models/model_safe.pkl")
