import numpy as np
import gymnasium as gym
from gymnasium import spaces
import genesis as gs

from adrobo_inverted_pendulum_genesis.entity.inverted_pendulum import InvertedPendulum
from adrobo_inverted_pendulum_genesis.reward_function.reward_function import RewardFunction
from adrobo_inverted_pendulum_genesis.tools.calculation_tools import CalculationTool

class Environment(gym.Env):
    def __init__(self, max_steps: int):
        super().__init__()

        gs.init(
            seed = None,
            precision = '64',
            debug = False,
            eps = 1e-12,
            logging_level = "warning",
            backend = gs.cpu,
            theme = 'dark',
            logger_verbose_time = False
        )

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=0.01,
                gravity=(0, 0, -9.81),
            ),
            show_viewer=True,
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.5, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(
                show_world_frame = True,
                world_frame_size = 1.0,
                show_link_frame = False,
                show_cameras = False,
                plane_reflection = True,
                ambient_light = (0.1, 0.1, 0.1),
            ),
            renderer = gs.renderers.Rasterizer(),
        )

        self.plane = self.scene.add_entity(gs.morphs.Plane())

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(low=-1.0,
                                            high=1.0,
                                            shape=(1,),
                                            dtype=np.float32)

        self.max_steps = max_steps
        self.step_count = 0
        self.prev_inverted_degree = 0.0

        self.inverted_pendulum = InvertedPendulum(self.scene)
        self.inverted_pendulum.create()
        self.reward_function = RewardFunction()
        self.calculation_tool = CalculationTool()

        self.reset()

        num = 2
        self.scene.build(n_envs=num, env_spacing=(0.5, 0.5))


    def step(self, action):
        terminated = False
        truncated = False
        info = {}

        self.step_count += 1

        self.inverted_pendulum.action(velocity_right=action[0], velocity_left=action[1])

        for _ in range(10):
            self.scene.step()

        inverted_degree = self.inverted_pendulum.read_inverted_degree()
        self.prev_inverted_degree = inverted_degree
        theta_vel = (inverted_degree - self.prev_inverted_degree) / 0.01

        observation = self.calculation_tool.normalization_inverted_degree(inverted_degree)

        reward = self.reward_function.calculate_reward(theta=inverted_degree, theta_vel=theta_vel, action=action)

        if self.step_count >= self.max_steps:
            terminated = True

        if inverted_degree <= -130 or inverted_degree >= 30:
            terminated = True

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        initial_observation = np.array([0.0], dtype=np.float32)
        info = {}
        return initial_observation, info

    def render(self):
        pass

    def close(self):
        pass