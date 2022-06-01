from atexit import register
import gym
from gym.envs.registration import register
from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.utils import set_random_seed
#import sys and set the path
import sys
import numpy as np



# # load wandb
import wandb
from wandb.integration.sb3 import WandbCallback





sys.path.append('gym_envs/hopper_dm')
sys.path.append('./')

# load environment
from Hopper6 import Hopper6

# wandb config
config = {
	"policy_type": "MlpPolicy",
	"total_timesteps": 100,
	"env_name": "Hopper-v5",
}


run = wandb.init(
	project="hopper-env",
	config=config,
	sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
	monitor_gym=True,  # auto-upload the videos of agents playing the game
	save_code=True,  # optional
)
PATH = 'gym_env/hopper_dm/mujoco_models/hopper_parkour.xml'


environment_kwargs = 	{'alive_bonus': 0.5,
						'velocity_cost': 0.0,
						'time_limit': 10,
						'position_reward':True,
						'observation_mode':'render'}

env = Hopper6(environment_kwargs=environment_kwargs)


action_spec = env.action_spec()
#env_checker.check_env(env)

# Define a uniform random policy.
def random_policy(time_step):
	del time_step  # Unused.
	return np.random.uniform(low=np.array([-1, -1]),
							high=np.array([1, 1]),
							size=np.array([2]))

# Launch the viewer application.
# viewer.launch(env, policy=random_policy)

def make_env(seed=0):
	"""
	Create a wrapped, monitored SubprocVecEnv for Hopper
	"""
	def _init():
		env.reset()

		env.seed(seed)
		return env
	
	set_random_seed(seed)
	return _init

env_list = [make_env(i) for i in range(1,10)]
train_env = SubprocVecEnv(env_list, start_method='fork')


model = PPO("MultiInputPolicy", env, n_steps=int(environment_kwargs['time_limit']/0.01), 
			n_epochs=10, normalize_advantage = True,  target_kl = 0.5, clip_range=0.4, vf_coef = 0.6, verbose=1,
			tensorboard_log=f"Models/hopper/runs/{run.id}")

model.learn(total_timesteps=100000, log_interval=1, callback=WandbCallback(gradient_save_freq=1, 
								model_save_path=f"Models/hopper/models/{run.id}", verbose=2))


model.save("Models_parkour_large")
run.finish()