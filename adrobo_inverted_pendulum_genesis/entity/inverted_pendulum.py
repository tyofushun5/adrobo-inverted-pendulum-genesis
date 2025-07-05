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

    def action(self, vel_r, vel_l, envs_idx=None):
        vel_cmd = np.stack([vel_r, vel_l], axis=1)
        self.agent.control_dofs_velocity(
            vel_cmd,
            self.wheel_dofs,
            envs_idx=envs_idx
        )

    def read_inverted_degree(self, env_ids=None):
        # env_ids が None なら全環境
        if env_ids is None:
            env_ids = np.arange(self.scene.n_envs)
        rad = self.agent.get_dofs_position([self.pipe_dof], envs_idx=env_ids)  # shape (N, 1)
        deg = rad * 180 / math.pi                                              # ndarray / tensor
        return deg.squeeze(-1)   # shape (N,)  ベクトルで返す


    def reset(self, env_idx):
        env_idx = np.asarray(env_idx, dtype=np.int32)

        # 1) 角度を 0 に戻す ―― 形状 (n_env, 1) にする
        zeros = np.zeros((len(env_idx), 1), dtype=np.float64)
        self.agent.set_dofs_position(
            zeros,                   # position  shape = (n_env, 1)
            [self.pipe_dof],         # dofs_idx  length = 1
            zero_velocity=True,
            envs_idx=env_idx.tolist()
        )


        self.agent.set_dofs_velocity(
            zeros,
            [self.pipe_dof],
            envs_idx=env_idx.tolist()
        )

