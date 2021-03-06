from atexit import register
import gym
from gym.envs.registration import register
from stable_baselines3 import PPO, A2C, DDPG


from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
#import sys and set the path
import sys
import numpy as np

sys.path.append('gym_envs/hopper')
env_spec ={
    'name': 'Hopper-v5',
    'model_path': 'gym_envs/hopper/hopper.xml',
    # 'model_path': 'gym_envs/hopper/hopper_wall.xml',
    'path': 'gym_envs/hopper',
    'action_space': gym.spaces.Box(low=-1, high=1, shape=(2,)),
    'observation_space': gym.spaces.Box(low=-100, high=100, shape=(6,)),
    'reward_range': (-float('inf'), float('inf')),
    'timestep': 0.01,
    'max_time': 50,
    'max_steps': 100000,
    'render': True,
    'viewer': True,
    'record': True
}


register(
    id=env_spec['name'],
    entry_point='hopper_env:HopperMine',
    max_episode_steps=env_spec["max_steps"],
    reward_threshold=None,
    kwargs=env_spec
)

model = PPO.load("hopper_multiple_duplicates_2.zip")
# test the environment
env  = gym.make(env_spec['name'])

obs = env.reset()
dones = False
while not dones:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(rewards, dones, env.time, obs)
    env.render()
print("done")
env.close()

# takes long time so be patient
#env.mujoco_env.save_video_from_frame("custom_hopper.mp4")
