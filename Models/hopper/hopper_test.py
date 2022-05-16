from atexit import register
import gym
from gym.envs.registration import register
from stable_baselines3 import PPO, A2C, DDPG


from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
#import sys and set the path
import sys
import numpy as np

sys.path.append('/home/prakyathkantharaju/gitfolder/personal/mucojo/gym_envs/hopper')
env_spec ={
    'name': 'Hopper-v5',
    'model_path': 'hopper.xml',
    'path': '/home/prakyathkantharaju/gitfolder/personal/mucojo/gym_envs/hopper',
    'action_space': gym.spaces.Box(low=-1, high=1, shape=(2,)),
    'observation_space': gym.spaces.Box(low=-100, high=100, shape=(6,)),
    'reward_range': (-float('inf'), float('inf')),
    'timestep': 0.01,
    'max_time': 1000,
    'max_steps': 100000,
    'render': True,
    'viewer': True
}


register(
    id=env_spec['name'],
    entry_point='hopper_env:HopperMine',
)

model = PPO.load("gym_envs/hopper/custom_hopper.zip")
# test the environment
env  = gym.make(env_spec['name'])

env.initialize_simulator(env_spec)
obs = env.reset()
dones = False
while not dones:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(rewards)
    env.render()