import abc
import os
import math
from typing import Union, Tuple, Dict, Any, Optional

import numpy as np
import torch

import genesis as gs

from adrobo_inverted_pendulum_genesis.entity.entity import Robot


script_dir = os.path.dirname(os.path.abspath(__file__))
robot = os.path.join(script_dir, 'MJCF', 'inverted_pendulum.xml')

class InvertedPendulum(Robot):
    def __init__(self, scene: gs.Scene = None, num_envs: int = 1):
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
                # pos=(0.0, 0.0, 0.0),
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

    def action(self, velocity_right, velocity_left, envs_idx=None):
        vel_cmd = np.stack([velocity_right*3.0, velocity_left*-3.0], axis=1).astype(np.float64)

        if envs_idx is not None:
            idx = np.r_[envs_idx].tolist()
            vel_cmd = vel_cmd[idx]

        self.agent.control_dofs_velocity(vel_cmd, self.wheel_dofs, envs_idx)

    def read_inverted_degree(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.scene.n_envs)
        rad = self.agent.get_dofs_position([self.pipe_dof], envs_idx=env_ids)
        deg = rad * 180 / math.pi
        return deg.squeeze(-1)

    def reset(self, env_idx):
        env_idx = np.asarray(env_idx, dtype=np.int32)
        num_envs = len(env_idx)

        zeros = np.zeros((num_envs, 1), dtype=np.float64)
        zeros_wheel = np.zeros((num_envs, len(self.wheel_dofs)), dtype=np.float64)

        self.agent.set_dofs_position(
            zeros_wheel,
            self.wheel_dofs,
            zero_velocity=True,
            envs_idx=env_idx.tolist()
        )

        self.agent.set_dofs_velocity(
            zeros_wheel,
            self.wheel_dofs,
            envs_idx=env_idx.tolist()
)

        self.agent.set_dofs_position(
            zeros,
            [self.pipe_dof],
            zero_velocity=True,
            envs_idx=env_idx.tolist()
        )

        self.agent.set_dofs_velocity(
            zeros,
            [self.pipe_dof],
            envs_idx=env_idx.tolist()
        )

if __name__ == "__main__":

    gs.init(
        seed = None,
        precision = '32',
        debug = False,
        eps = 1e-12,
        backend = gs.gpu,
        theme = 'dark',
        logger_verbose_time = False
    )

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0, 0, -9.81),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_joint_limit=True,
            enable_collision=True,
            constraint_solver=gs.constraint_solver.Newton,
            iterations=150,
            tolerance=1e-6,
            contact_resolve_time=0.01,
            use_contact_island=False,
            use_hibernation=False
        ),
        show_viewer=False,
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
            plane_reflection = False,
            ambient_light = (0.1, 0.1, 0.1),
            n_rendered_envs = 100,
        ),
        renderer = gs.renderers.Rasterizer(),
    )

    cam = scene.add_camera(
        res    = (1280, 960),
        pos    = (3.5, 0.0, 2.5),
        lookat = (0, 0, 0.5),
        fov    = 30,
        GUI    = False
    )

    plane = scene.add_entity(gs.morphs.Plane())
    num = 100

    inverted_pendulum = InvertedPendulum(scene, num_envs=num)
    inverted_pendulum.create()

    scene.build(n_envs=num, env_spacing=(0.5, 0.5))
    cam.start_recording()

    for i in range(10000):
        scene.step()
        cam.set_pose(
            pos    = (3.0 * np.sin(i / 60), 3.0 * np.cos(i / 60), 2.5),
            lookat = (0, 0, 0.5),
        )
        cam.render()
    cam.stop_recording(save_to_filename='video.mp4', fps=60)