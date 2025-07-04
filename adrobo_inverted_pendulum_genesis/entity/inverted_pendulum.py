import abc
import os
import math
from typing import Union, Tuple, Dict, Any, Optional

import numpy as np
import gymnasium as gym
import copy
import genesis as gs
import torch

from skrl.memories.torch import Memory
from skrl.agents.torch import Agent
from adrobo_inverted_pendulum_genesis.entity.entity import Robot


script_dir = os.path.dirname(os.path.abspath(__file__))
robot = os.path.join(script_dir, 'MJCF', 'inverted_pendulum.xml')

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

scene = gs.Scene(
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

plane = scene.add_entity(gs.morphs.Plane())

class InvertedPendulum(Robot):

    def __init__(self, scene: gs.Scene = None):
        super().__init__()
        self.agent = None
        self.wheel_joints = ["right_wheel", "left_wheel"]
        self.pipe_joints = ["pipe"]
        self.wheel_dofs = None
        self.pipe_dof = None
        self.scene = scene
        self.surfaces = gs.surfaces.Default(
            color=(0.0, 0.0, 0.0),
            opacity=1.0,
            roughness=0.5,
            metallic=0.0,
            emissive=None
        )

    def create(self):

        self.agent = self.scene.add_entity(
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
        self.wheel_dofs = [
            self.agent.get_joint(name).dof_idx_local
            for name in self.wheel_joints
        ]

        pipe_joint = self.agent.get_joint("pipe")
        self.pipe_dof = pipe_joint.dof_idx_local

        return self.agent

    def action(self, velocity_right=0.0, velocity_left=0.0):
        base = np.array([velocity_right, velocity_left], dtype=np.float64)
        vel_cmd = np.tile(base, (self.scene.n_envs, 1))
        self.agent.control_dofs_velocity(vel_cmd, self.wheel_dofs)

    def read_inverted_degree(self):
        """
        Read the inverted degrees of the pendulum.
        """
        angle_rad = self.agent.get_dofs_position([self.pipe_dof])[0]
        angle_deg = angle_rad * 180 / math.pi
        return float(angle_deg)



agent = InvertedPendulum(scene)
agent.create()

num = 2
scene.build(n_envs=num, env_spacing=(0.5, 0.5))

kp = np.zeros(len(agent.wheel_dofs), dtype=np.float64)
kv = np.ones(len(agent.wheel_dofs), dtype=np.float64) * 100.0
agent.agent.set_dofs_kp(kp=kp, dofs_idx_local=agent.wheel_dofs)
agent.agent.set_dofs_kv(kv=kv, dofs_idx_local=agent.wheel_dofs)

# print("wheel_dofs:", agent.wheel_dofs)
# print(agent.pipe_dof)

for i in range(100000):
    agent.action(velocity_right=-5.0, velocity_left=5.0)
    print(agent.read_inverted_rad())
    scene.step()
