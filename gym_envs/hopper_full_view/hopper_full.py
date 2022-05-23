

import numpy as np



# downloaded from: https://github.com/openai/gym/blob/master/gym/envs/mujoco/hopper_v3.py
from gym import utils
from gym.envs.mujoco import mujoco_env

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 3.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}




class HopperMine(mujoco_env.MujocoEnv):
    def __init__(self, model_path, frame_skip):
        super().__init__(model_path, frame_skip)


    @property
    def healthy_reward(self):
        # return (
        #     float(self.is_healthy or self._terminate_when_unhealthy)
        #     * self._healthy_reward
        # )
        return ( 1 )

    def control_cost(self, action):
        # control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return 1

    @property
    def is_healthy(self):
        # z, angle = self.sim.data.qpos[1:3]
        # state = self.state_vector()[2:]

        # min_state, max_state = self._healthy_state_range
        # min_z, max_z = self._healthy_z_range
        # min_angle, max_angle = self._healthy_angle_range

        # healthy_state = np.all(np.logical_and(min_state < state, state < max_state))
        # healthy_z = min_z < z < max_z
        # healthy_angle = min_angle < angle < max_angle

        # is_healthy = all((healthy_state, healthy_z, healthy_angle))

        # return is_healthy
        return True



            
    @property
    def done(self):
        # done = not self.is_healthy if self._terminate_when_unhealthy else False
        # return done
        return False

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = np.clip(self.sim.data.qvel.flat.copy(), -10, 10)
        observation = self.render(mode = 'rgb_array')


        # if self._exclude_current_positions_from_observation:
            # position = position[1:]

        # observation = np.concatenate((position, velocity)).ravel()
        # print(obs.shape)
        return observation

    def step(self, action):
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        ctrl_cost = self.control_cost(action)

        observation = self._get_obs()

        print(self.done)
        return observation, ctrl_cost, self.done, 1

    def reset_model(self):
        # noise_low = -self._reset_noise_scale
        # noise_high = self._reset_noise_scale

        # qpos = self.init_qpos + self.np_random.uniform(
        #     low=noise_low, high=noise_high, size=self.model.nq
        # )
        # qvel = self.init_qvel + self.np_random.uniform(
        #     low=noise_low, high=noise_high, size=self.model.nv
        # )

        # self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        return 0
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)