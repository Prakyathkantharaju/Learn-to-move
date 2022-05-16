# remove the warning
import warnings
warnings.filterwarnings("ignore")

# import the typing
from typing import List, Tuple, Dict, Any
import numpy as np

import os

# import gym
import gym
import mujoco_py

# local imports
from base_sim import BaseSim
from utils import convert_observation_to_space

class HopperMine(gym.Env):
    metadata: Dict[str, List[str]] = {'render.modes': ['human']}

    def __init___(self, **kwargs):

        super(HopperMine, self).__init__()
        
        

    def initialize_simulator(self, kwargs: Dict[str, Any]):
        # self.action_space = gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), shape=(2,))
        # self.observation_space = gym.spaces.Box(low= np.array([0, 0, 0, 0, 0, 0]), high= np.array([100, 100, 100, 100, 100, 100]), shape=(6,))
        os.chdir(kwargs['path'])
        self.kwargs = kwargs
        self.mujoco_env = BaseSim(kwargs['model_path'], kwargs['render'], kwargs['timestep'])
        self.viewer = kwargs['viewer']
        self.time = 0
        self._set_action_space()

        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)
        assert not done

        self._set_observation_space(observation)
        print(self.observation_space.shape, 'here')

    def _set_action_space(self):
        bounds = self.mujoco_env.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        low = np.array([ -1, -1])
        high = np.array([ 1, 1])
        print(f"Action space: {low} {high}")
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space
        

    def step(self, action):

        # check for nan in action
        if np.isnan(np.sum(action)):
            action = self.action_space.sample() 


        self.mujoco_env.step(action)
        self.time += self.kwargs['timestep']
        #self.viewer.render()
        state = self.mujoco_env.get_state()
        rewards = self._calcualte_reward(state, action)
        done = self._done_state(state)
        self.rewards = rewards
        #print(f"rewards: {rewards}, time {self.time}")
        return state[[0,2]].flatten(), rewards, done, {'full_state':state}

    def _calcualte_reward(self, state: np.ndarray, action: List[float]) -> float:
        """
        Calculate the reward for the given state
        """
        rewards = state[0][0]
        rewards -= np.sqrt(np.sum(np.square(action)))
        rewards += 1 * self.kwargs['timestep']# alive bonus
        rewards += self.mujoco_env.step_count
        return rewards

    def _done_state(self, state: np.ndarray) -> bool:
        """
        Check if the state is done
            - Conditions are check the hip height is less than one. 
            - Check the step count is greater than max_steps
            - Check the time is greater than max_time
        """
        
        # print(self.time)
        if state[0][2] < 0.1 or self.mujoco_env.step_count > self.kwargs['max_steps'] \
            or self.time > self.kwargs['max_time']:

            print(f"rewards: {self.rewards}, time {self.time}")
            return True
        return False

    def reset(self):
        self.mujoco_env.reset()
        self.time = 0
        return self.mujoco_env.get_state()[[0,2]].flatten()
        

    def seed(self, seed=None):
        #self.mujoco_env.seed(seed)
        pass

    def render(self, mode='human'):
        self.mujoco_env.render()


    def close(self):
        #self.mujoco_env.close()
        pass


 