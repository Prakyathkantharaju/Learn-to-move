from asyncore import write
import warnings
warnings.filterwarnings("ignore")


import time, os
import gym
from gym.envs.registration import register
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control import mujoco
from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv

from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

#import sys and set the path
import sys
import numpy as np

import matplotlib.pyplot as plt

# import imageio
import imageio

sys.path.append('./gym_envs/hopper_openai')
sys.path.append('./')

path_ = os.getcwd()

rel_path = 'gym_envs/hopper_openai/mujoco_models/hopper_obsticle.xml'
path = path_ + '/' + rel_path
# load environment
from hopper_v4 import HopperEnv

env = HopperEnv(xml_file = path)


env.reset()

for i in range(1000):
	obs, reward, done, info = env.step(env.action_space.sample())
	env.render()
	print(done)
	if done:
		env.reset()
	time.sleep(0.01)