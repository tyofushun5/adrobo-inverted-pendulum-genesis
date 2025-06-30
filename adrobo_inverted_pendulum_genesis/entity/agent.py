import abc
import os
import math

import numpy as np
import genesis as gs
import torch

from adrobo_inverted_pendulum_genesis.entity.entity import Robot


script_dir = os.path.dirname(os.path.abspath(__file__))

MJCF_dir = os.path.join(script_dir, 'MJCF')

robot = os.path.join(MJCF_dir, 'inverted_pendulum.xml')

gs.init(
    seed = None,
    precision = '64',
    debug = False,
    eps = 1e-12,
    logging_level = None,
    backend = gs.cuda,
    theme = 'dark',
    logger_verbose_time = False
)

scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=0.01,
        gravity=(0, 0, -9.806),
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

plane = scene.add_entity(gs.morphs.Plane())

class Agent(Robot):

    def __init__(self, create_position=None):
        super().__init__()
        if create_position is None:
            create_position = [0.0, 0.0, 0.0]
        self.cp = create_position
        self.start_pos = None
        self.default_ori = [0.0, 0.0, 0.0]
        self.position = self.start_pos
        self.agent = None
        self.surfaces = gs.surfaces.Default(
            color=(0.0, 0.0, 0.0),
            opacity=1.0,
            roughness=0.5,
            metallic=0.0,
            emissive=None
        )

    def create(self, position=None):

        self.position = position

        if position is None:
            self.position = self.start_pos

        self.agent = scene.add_entity(
            morph = gs.morphs.MJCF(
                file=os.path.join(script_dir, robot),
                # scale=1.0,
                # pos=self.cp,
                # euler=None,
                convexify=False,
                visualization=True,
                collision=True,
                requires_jac_and_IK=True,
            ),
            material=None,
            surface=self.surfaces,
            visualize_contact=False,
            vis_mode="collision",
        )

        return self.agent

    def action(self, wheel_angle=0.0):
        pass

agent = Agent()
agent.create()

num = 1
scene.build(n_envs=num, env_spacing=(0.5, 0.5))

for i in range(100000):
    scene.step()