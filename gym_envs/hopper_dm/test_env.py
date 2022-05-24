# File to test the env and setup

import numpy as np
from dm_control import viewer

from Hopper6 import Hopper6

PATH = './mujoco_models/hopper_parkour.xml'


environment_kwargs = 	{'alive_bonus': 0.5,
						'velocity_cost': 0.0,
						'time_limit': 10,
						'position_reward':True,
						'observation_mode':'render'}

env = Hopper6(environment_kwargs=environment_kwargs)


action_spec = env.action_spec()

# Define a uniform random policy.
def random_policy(time_step):
	del time_step  # Unused.
	return np.random.uniform(low=np.array([-1, -1, -1, -1]),
							high=np.array([1, 1, 1, 1]),
							size=np.array([4]))

# Launch the viewer application.
# viewer.launch(env, policy=random_policy)
env.reset()

step, reward, discount, obs = env.step(random_policy(None))
print(obs)