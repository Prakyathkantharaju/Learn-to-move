
import gym
from gym.envs.registration import register
from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common import env_checker

from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

#import sys and set the path
import sys
import numpy as np

sys.path.append('gym_envs/hopper')

env_spec ={
    'name': 'Hopper-v5',
    'model_path': 'hopper_wall_hard_dist.xml',
    'path': 'gym_envs/hopper',
    'action_space': gym.spaces.Box(low=-1, high=1, shape=(2,)),
    'observation_space': gym.spaces.Box(low=-100, high=100, shape=(6,)),
    'reward_range': (-float('inf'), float('inf')),
    'timestep': 0.01,
    'max_time': 100,
    'max_steps': 100000,
    'render': True,
    'viewer': None,
    'record': False,
}


register(
    id=env_spec['name'],
    entry_point='hopper_env:HopperMine',
    max_episode_steps=env_spec["max_steps"],
    reward_threshold=None,
    kwargs=env_spec
)

# test the environment
env  = gym.make(env_spec['name'])



env_checker.check_env(env)


obs = env.reset()
dones = False
while not dones:
    action = env.action_space.sample()
    obs, rewards, dones, info = env.step(action)
    env.render()
env.close()


