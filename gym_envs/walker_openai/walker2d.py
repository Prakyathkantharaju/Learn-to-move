import numpy as np

from gym import utils
from gym.envs.mujoco import mujoco_env

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 6.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}


class Walker2dEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "single_rgb_array",
            "single_depth_array",
        ],
        "render_fps": 125,
    }

    def __init__(
        self,
        xml_file="walker2d.xml",
        forward_reward_weight=2.0,
        ctrl_cost_weight=1e-5,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.9, 1.9),
        healthy_angle_range=(-0.5, 0.5),
        reset_noise_scale=5e-3,
        exclude_current_positions_from_observation=False,
        **kwargs
    ):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        self.x_position = 0.0
        if 'gap_2' in xml_file:
            self.gap = 2
            print("started gap 2")
        elif 'gap' in xml_file:
            self.gap = 1
            print("started gap 1")
        else:
            self.gap = 0

        mujoco_env.MujocoEnv.__init__(
            self, xml_file, 4, **kwargs
        )

    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def is_healthy(self):
        z, angle = self.sim.data.qpos[1:3]

        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle
        is_healthy = healthy_z and healthy_angle

        if self.gap == 1:
            if (15 < self.x_position < 15) or \
                (20 < self.x_position < 21) or \
                    (27 < self.x_position < 28) or \
                        (32 < self.x_position < 34):
                is_healthy = False
        if self.gap == 2:
            if (7 < self.x_position < 9) or \
                (13 < self.x_position < 14) or \
                    (18 < self.x_position < 20) or \
                        (24 < self.x_position < 26):
                is_healthy = False

        return is_healthy

    @property
    def done(self):
        done = not self.is_healthy if self._terminate_when_unhealthy else False
        return done

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = np.clip(self.sim.data.qvel.flat.copy(), -10, 10)
        rangefinder = self.sim.data.sensordata.flat.copy()
        # print(rangefinder.shape, position.shape)

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity, rangefinder)).ravel()
        return observation

    def step(self, action):
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt
        self.x_position = x_position_after

        # self.renderer.render_step()

        ctrl_cost = self.control_cost(action)
        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost

        observation = self._get_obs()
        reward = rewards - costs
        done = self.done
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
        }

        return observation, reward, done, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
