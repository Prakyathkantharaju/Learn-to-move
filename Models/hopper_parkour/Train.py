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
						'observation_mode':'render',
						'path':'gym_envs/hopper_dm/mujoco_models/hopper_parkour.xml'}

env = Hopper6(environment_kwargs=environment_kwargs)

environment_kwargs = 	{'alive_bonus': 0.5,
						'velocity_cost': 0.0,
						'time_limit': 10,
						'position_reward':True,
						'observation_mode':'render',
						'path':'gym_envs/hopper_dm/mujoco_models/hopper_parkour_plain.xml'}

env_1 = Hopper6(environment_kwargs=environment_kwargs)


environment_kwargs = 	{'alive_bonus': 0.5,
						'velocity_cost': 0.0,
						'time_limit': 10,
						'position_reward':True,
						'observation_mode':'render',
						'path':'gym_envs/hopper_dm/mujoco_models/hopper_parkour_step.xml'}

env_2 = Hopper6(environment_kwargs=environment_kwargs)


environment_kwargs = 	{'alive_bonus': 0.5,
						'velocity_cost': 0.0,
						'time_limit': 10,
						'position_reward':True,
						'observation_mode':'render',
						'path':'gym_envs/hopper_dm/mujoco_models/hopper_parkour_gaps.xml'}

env_3 = Hopper6(environment_kwargs=environment_kwargs)

environment_kwargs = 	{'alive_bonus': 0.5,
						'velocity_cost': 0.0,
						'time_limit': 10,
						'position_reward':True,
						'observation_mode':'render',
						'path':'gym_envs/hopper_dm/mujoco_models/hopper_parkour_climb.xml'}

env_4 = Hopper6(environment_kwargs=environment_kwargs)



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
		print(f"env seed: {seed}")
		return env
	
	set_random_seed(seed)
	return _init

def make_env_1(seed=0):
	"""
	Create a wrapped, monitored SubprocVecEnv for Hopper
	"""
	def _init():
		env_1.reset()

		env_1.seed(seed)
		print(f"env_1 seed: {seed}")
		return env_1
	
	set_random_seed(seed)
	return _init

def make_env_2(seed=0):
	"""
	Create a wrapped, monitored SubprocVecEnv for Hopper
	"""
	def _init():
		env_2.reset()

		env_2.seed(seed)
		print(f"env_2 seed: {seed}")
		return env_2
	
	set_random_seed(seed)
	return _init

def make_env_3(seed=0):
	"""
	Create a wrapped, monitored SubprocVecEnv for Hopper
	"""
	def _init():
		env_3.reset()

		env_3.seed(seed)
		print(f"env_2 seed: {seed}")
		return env_3
	
	set_random_seed(seed)
	return _init

def make_env_4(seed=0):
	"""
	Create a wrapped, monitored SubprocVecEnv for Hopper
	"""
	def _init():
		env_4.reset()

		env_4.seed(seed)
		print(f"env_2 seed: {seed}")
		return env_4
	
	set_random_seed(seed)
	return _init

if __name__ == '__main__':
	env_list = [make_env(0), make_env_1(1), make_env_2(2), 
	make_env_3(6), make_env_4(2)]
	train_env = DummyVecEnv(env_list)


	model = PPO("MultiInputPolicy", train_env, n_steps=1000, 
				n_epochs=10, normalize_advantage = True,  target_kl = 0.5, clip_range=0.4, vf_coef = 0.6, verbose=1,
				tensorboard_log=f"Models/hopper/runs/{run.id}")
	model.load("Models_parkour_large_1")

	model.learn(total_timesteps=500000, log_interval=1, callback=WandbCallback(gradient_save_freq=1, 
									model_save_path=f"Models/hopper/models/{run.id}", verbose=2))


	# model.save("Models_parkour_large_1")
	run.finish()