# File to test the env and setup

import numpy as np
from dm_control import viewer

from Hopper6 import Hopper6

PATH = './mujoco_models/hopper.xml'

env = Hopper6(PATH) 


action_spec = env.action_spec()

# Define a uniform random policy.
def random_policy(time_step):
  del time_step  # Unused.
  return np.random.uniform(low=np.array([-1, -1, -1, -1]),
                           high=np.array([1, 1, 1, 1]),
                           size=np.array([4]))

# Launch the viewer application.
viewer.launch(env, policy=random_policy)