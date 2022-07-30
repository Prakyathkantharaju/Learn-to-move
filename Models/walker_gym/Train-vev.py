import gym
from gym.envs.registration import register
from stable_baselines3 import PPO, A2C, DDPG, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder

from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.utils import set_random_seed
#import sys and set the path
import sys, os
import numpy as np

from stable_baselines3.common.env_checker import check_env

# # load wandb
import wandb
from wandb.integration.sb3 import WandbCallback

# from mujoco_py import GlfwContext
# GlfwContext(offscreen=True)



sys.path.append('./gym_envs/walker_openai')
sys.path.append('./')

path_ = os.getcwd()

rel_path = 'gym_envs/walker_openai/mujoco_models/walker2d.xml'
path = path_ + '/' + rel_path
rel_path = 'gym_envs/walker_openai/mujoco_models/walker2d-obsticles-large.xml'
path_2 = path_ + '/' + rel_path
rel_path = 'gym_envs/walker_openai/mujoco_models/walker2d-obsticles.xml'
path_3 = path_ + '/' + rel_path
# load environment
from walker2d import Walker2dEnv

env = Walker2dEnv(xml_file = path)
env_1 = Walker2dEnv(xml_file = path_2)
env_2 = Walker2dEnv(xml_file = path_3)

# wandb config
config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 1000,
    "env_name": "Hopper-v4",
}


run = wandb.init(
    project="Walker_obstacles",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)




def make_env(seed=0):
    """
    Create a wrapped, monitored SubprocVecEnv for Hopper
    """
    def _init():
        # env.reset()
        if seed == 0:
            env.seed(seed)
            env.render()
            print(f"env seed: {seed}")
            return env
        elif seed == 1:
            env_1.seed(seed)
            env_1.render()
            print(f"env seed: {seed}")
            return env_1
        elif seed == 2:
            env_2.seed(seed)
            env_2.render()
            print(f"env seed: {seed}")
            return env_2
        else:
            env.seed(seed)
            env.render()
            print(f"env seed: {seed}")
            return env

    set_random_seed(seed)
    return _init



if __name__ == '__main__':

    check_env(env)
    # env_list = [make_env(0), make_env_1(1), ]

    #train_env = SubprocVecEnv(env_list, start_method='fork')
    n_procs = 5
    train_env = SubprocVecEnv([make_env(i) for i in range(n_procs)], start_method='fork')
    train_env = VecVideoRecorder(train_env, f'./run_logs/videos/{run.id}', record_video_trigger=lambda x: x % 100000 == 0, video_length = 2000)

    train_env.reset()

    model = PPO("MlpPolicy", train_env, batch_size=32, n_steps=512, n_epochs=20, gae_lambda= 0.95, learning_rate=5e-5 ,verbose=1, gamma=0.995, clip_range=0.1, ent_coef=4.5e-04,
                create_eval_env= True, 
                max_grad_norm=1,
                vf_coef=0.87,
                tensorboard_log=f"./run_logs/logs/{run.id}")
    # model.load("Models_parkour_large_1")

    model.learn(total_timesteps=5000000, log_interval=1, callback=WandbCallback(gradient_save_freq=1000,  model_save_freq=10000,
                                    model_save_path=f"./run_logs/models/{run.id}", verbose=2))


    model.save(f"./.run_logs/full_save_model/walker/walker_plain.pkl")
    run.finish()
