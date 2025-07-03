import random
import math

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import genesis as gs

from adrobo_inverted_pendulum_genesis.entity.inverted_pendulum import InvertedPendulum


class Environment(gym.Env):
    def __init__(self):
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
        self.inverted_pendulum = InvertedPendulum(self.scene)

    def step(self, action):
        observation = None
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        initial_observation = None
        info = {}
        return initial_observation, info

    def render(self):
        pass

    def close(self):
        pass