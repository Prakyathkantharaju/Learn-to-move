from asyncore import write
import warnings
warnings.filterwarnings("ignore")



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

PATH = 'gym_env/hopper_dm/mujoco_models/hopper_parkour.xml'


environment_kwargs = 	{'alive_bonus': 0.5,
						'velocity_cost': 0.0,
						'time_limit': 10,
						'position_reward':True,
						'observation_mode':'render'}

env = Hopper6(environment_kwargs=environment_kwargs)




image = env._physics.render(camera_id = "camera", segmentation = True)

# print(image.shape)
# Display the contents of the first channel, which contains object
# IDs. The second channel, seg[:, :, 1], contains object types.
geom_ids = image[:, :, 0]
# Infinity is mapped to -1
geom_ids = geom_ids.astype(np.float64) + 1
# Scale to [0, 1]
geom_ids = geom_ids / geom_ids.max()
pixels = 255*geom_ids
img = pixels.astype(np.uint8)
plt.imshow(img)
plt.show()

model = PPO("MultiInputPolicy", env, n_steps=int(environment_kwargs['time_limit']/0.01), 
			n_epochs=10, normalize_advantage = True,  target_kl = 0.5, clip_range=0.4, vf_coef = 0.6, verbose=1)

model.load("Models_parkour_large")

_, _, _, obs = env.reset()
print(obs)


render_store = []
for _ in range(1000):
	actions = model.predict(obs)
	timestep, reward, discount, obs = env.step(actions[0])
	render_store.append(env._physics.render(camera_id = "camera"))
	print(reward)

file_name = 'hopper_parkour.mp4'
writer = imageio.get_writer(file_name, fps = 100)
for im in render_store:
	writer.append_data(im)
writer.close()
