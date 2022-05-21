from atexit import register
import gym
from gym.envs.registration import register
from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv

from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

#import sys and set the path
import sys
import numpy as np
from torch import tensor



# load wandb
import wandb
from wandb.integration.sb3 import WandbCallback



sys.path.append('gym_envs/hopper')
sys.path.append('./')

env_spec ={
    'name': 'Hopper-v5',
    'model_path': 'gym_envs/hopper/hopper_wall_hard_dist.xml',
    'path': 'gym_envs/hopper',
    'action_space': gym.spaces.Box(low=-1, high=1, shape=(2,)),
    'observation_space': gym.spaces.Box(low=-100, high=100, shape=(6,)),
    'reward_range': (-float('inf'), float('inf')),
    'timestep': 0.01,
    'max_time': 25,
    'max_steps': 100000,
    'render': False,
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


# wandb config
config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 100,
    "env_name": "Hopper-v5",
}


run = wandb.init(
    project="sb3",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)

env_checker.check_env(env)


# env maker
def make_env():
    env = gym.make(config['env_name'])
    env = Monitor(env)
    return env

n_actions = env.action_space.shape[-1]

env = make_env()
print(env.action_space)

model = PPO("MlpPolicy", env, n_steps=int(env_spec['max_time']/env_spec['timestep']), 
            n_epochs=10, normalize_advantage = True,  target_kl = 0.5, clip_range=0.4, vf_coef = 0.6, verbose=1,
            tensorboard_log=f"Models/hopper/runs/{run.id}")



print("Training...")

model.learn(total_timesteps=100000, log_interval=1, 
callback=WandbCallback(gradient_save_freq=1, 
model_save_path=f"Models/hopper/models/{run.id}",verbose=2))

model.save("hopper_hard_dist")
env = model.get_env()

del model # remove to demonstrate saving and loading

model = PPO.load("hopper_hard_dist")

# # turn on the viewer

obs = env.reset()
dones = False
while not dones:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(rewards)
    env.render()
env.close()

run.finish()

