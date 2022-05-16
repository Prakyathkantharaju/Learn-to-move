
import numpy as np
import gym.spaces as spaces
from collections import OrderedDict




def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(
            OrderedDict(
                [
                    (key, convert_observation_to_space(value))
                    for key, value in observation.items()
                ]
            )
        )
    elif isinstance(observation, np.ndarray):
        print("here, observation shape: ", observation)
        low = np.full(observation.shape, -float("inf"), dtype=np.float32)
        high = np.full(observation.shape, float("inf"), dtype=np.float32)
        print(f"Observation space: {low} {high}")
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space