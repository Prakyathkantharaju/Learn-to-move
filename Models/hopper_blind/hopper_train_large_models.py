
from asyncore import readwrite
import gym
from gym.envs.registration import register
from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.utils import set_random_seed

#import sys and set the path
import sys
import numpy as np
from torch import tensor



# load wandb
import wandb
from wandb.integration.sb3 import WandbCallback



sys.path.append('gym_envs/hopper')
sys.path.append('./')

env_spec_1 ={
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

env_spec_2 ={
    'name': 'Hopper-v6',
    'model_path': 'gym_envs/hopper/hopper_wall.xml',
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


env_spec_3 ={
    'name': 'Hopper-v7',
    'model_path': 'gym_envs/hopper/hopper_no_dist.xml',
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
    id=env_spec_1['name'],
    entry_point='hopper_env:HopperMine',
    max_episode_steps=env_spec_1["max_steps"],
    reward_threshold=None,
    kwargs=env_spec_1
)
register(
    id=env_spec_2['name'],
    entry_point='hopper_env:HopperMine',
    max_episode_steps=env_spec_2["max_steps"],
    reward_threshold=None,
    kwargs=env_spec_2
)
register(
    id=env_spec_3['name'],
    entry_point='hopper_env:HopperMine',
    max_episode_steps=env_spec_3["max_steps"],
    reward_threshold=None,
    kwargs=env_spec_3
)



# wandb config
config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 100,
    "env_name": "custom-Hopper",
}


run = wandb.init(
    project="sb3",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)



# env maker
def make_env(env_id):
    env = gym.make(env_id)
    env = Monitor(env)
    return env


# settting up mulitprocessing env training
def make_env(env_id, n_envs=1, seed=0):
    """
    Create a wrapped, monitored SubprocVecEnv for Hopper
    """
    # env = DummyVecEnv([make_env])
    def _init():
        env = gym.make(env_id)

        env.seed(seed)
        return env
    
    set_random_seed(seed)
    return _init

env_list = [make_env(env_spec_1['name']), make_env(env_spec_2['name']), make_env(env_spec_3['name']), 
            make_env(env_spec_1['name'], seed=2022), make_env(env_spec_2['name'], seed=2022), make_env(env_spec_3['name'], seed=2022),
            make_env(env_spec_1['name'], seed=2021), make_env(env_spec_2['name'], seed=2021), make_env(env_spec_3['name'], seed=2021),
            make_env(env_spec_1['name'], seed=2020), make_env(env_spec_2['name'], seed=2020), make_env(env_spec_3['name'], seed=2020),
            make_env(env_spec_1['name'], seed=2019), make_env(env_spec_2['name'], seed=2019), make_env(env_spec_3['name'], seed=2019)]
train_env = SubprocVecEnv(env_list, start_method='fork')
print("Done setting up env")

model = PPO("MlpPolicy", train_env, n_steps=int(env_spec_1['max_time']/env_spec_1['timestep']), 
            n_epochs=10, normalize_advantage = True,  target_kl = 0.5, clip_range=0.4, vf_coef = 0.6, verbose=1,
            tensorboard_log=f"Models/hopper/runs/{run.id}")

print("Training...")

model.learn(total_timesteps=2500000, log_interval=1, 
            callback=WandbCallback(gradient_save_freq=1, 
            model_save_path=f"Models/hopper/models/{run.id}",verbose=2))

evaluation_env = gym.make("Hopper-v5")
rewards, _ = evaluate_policy(model, evaluation_env, n_eval_episodes=10)

model.save("hopper_multiple_duplicates_2")
train_env.close()

run.finish()

