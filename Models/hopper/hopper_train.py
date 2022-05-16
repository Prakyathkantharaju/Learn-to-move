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
    'model_path': 'hopper.xml',
    'path': 'gym_envs/hopper',
    'action_space': gym.spaces.Box(low=-1, high=1, shape=(2,)),
    'observation_space': gym.spaces.Box(low=-100, high=100, shape=(6,)),
    'reward_range': (-float('inf'), float('inf')),
    'timestep': 0.01,
    'max_time': 100,
    'max_steps': 100000,
    'render': False,
    'viewer': True
}


register(
    id=env_spec['name'],
    entry_point='hopper_env:HopperMine',
)

# test the environment
env  = gym.make(env_spec['name'])

env.initialize_simulator(env_spec)
# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = PPO("MlpPolicy", env, n_epochs=100, normalize_advantage = True,  target_kl = 0.08, clip_range=0.3, vf_coef = 0.6, verbose=2)


model.learn(total_timesteps=10000, log_interval=1)
model.save("custom_hopper")
env = model.get_env()

del model # remove to demonstrate saving and loading

model = PPO.load("custom_hopper")

obs = env.reset()
dones = False
while not dones:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(rewards)
    env.render()



