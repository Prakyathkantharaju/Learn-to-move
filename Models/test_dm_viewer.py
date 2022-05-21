from dm_control  import viewer
import mujoco_py 
import gym
import sys
from gym.envs.registration import register





sys.path.append('gym_envs/hopper')
env_spec ={
    'name': 'Hopper-v5',
    'model_path': 'hopper.xml',
    'path': 'gym_envs/hopper',
    'action_space': gym.spaces.Box(low=-1, high=1, shape=(2,)),
    'observation_space': gym.spaces.Box(low=-100, high=100, shape=(6,)),
    'reward_range': (-float('inf'), float('inf')),
    'timestep': 0.01,
    'max_time': 50,
    'max_steps': 100000,
    'render': False,
    'viewer': False,
    'record': False
}


register(
    id=env_spec['name'],
    entry_point='hopper_env:HopperMine',
)


env  = gym.make(env_spec['name'])
env.initialize_simulator(env_spec)
action_space = env.action_space

def random_action(time_step):
    del time_step
    action = action_space.sample()
    return action

viewer.launch(env, policy=random_action)
action_spec = env.action_spec()
print(action_spec)
#print(action_space.shape)
print(type(action_spec))

from dm_control import suite
from dm_control import viewer
import numpy as np

env = suite.load(domain_name="humanoid", task_name="stand")
action_spec = env.action_spec()
print(action_spec.shape)
print(type(action_spec))

# # Define a uniform random policy.
# def random_policy(time_step):
#   del time_step  # Unused.
#   return np.random.uniform(low=action_spec.minimum,
#                            high=action_spec.maximum,
#                            size=action_spec.shape)

# # Launch the viewer application.
# viewer.launch(env, policy=random_policy)