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
from skrl.models.torch import Model
from skrl.agents.torch import Agent
from adrobo_inverted_pendulum_genesis.entity.entity import Robot

script_dir = os.path.dirname(os.path.abspath(__file__))
robot = os.path.join(script_dir, 'MJCF', 'inverted_pendulum.xml')


CUSTOM_DEFAULT_CONFIG = {
    "experiment": {
        "directory": "",
        "experiment_name": "inverted_pendulum",
        "write_interval": 250,
        "checkpoint_interval": 1000,
        "store_separately": False,
        "wandb": False,
        "wandb_kwargs": {}
    }
}

class InvertedPendulum(Agent):

    def __init__(self,
                 models: Dict[str, Model],
                 memory: Optional[Union[Memory, Tuple[Memory]]] = None,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space]] = None,
                 action_space: Optional[Union[int, Tuple[int], gym.Space]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 cfg: Optional[dict] = None) -> None:

        _cfg = copy.deepcopy(CUSTOM_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(models=models,
                         memory=memory,
                         observation_space=observation_space,
                         action_space=action_space,
                         device=device,
                         cfg=_cfg)

        self.wheel_joints = ["right_wheel", "left_wheel"]
        self.wheel_dofs = None

    def init(self, trainer_cfg: Optional[Dict[str, Any]] = None) -> None:

        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """
        # ======================================
        # - sample random actions if required or
        #   sample and return agent's actions
        # ======================================

    def record_transition(self,
                          states: torch.Tensor,
                          actions: torch.Tensor,
                          rewards: torch.Tensor,
                          next_states: torch.Tensor,
                          terminated: torch.Tensor,
                          truncated: torch.Tensor,
                          infos: Any,
                          timestep: int,
                          timesteps: int) -> None:
        """Record an environment transition in memory

        :param states: Observations/states of the environment used to make the decision
        :type states: torch.Tensor
        :param actions: Actions taken by the agent
        :type actions: torch.Tensor
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: torch.Tensor
        :param next_states: Next observations/states of the environment
        :type next_states: torch.Tensor
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: torch.Tensor
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: torch.Tensor
        :param infos: Additional information about the environment
        :type infos: Any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        super().record_transition(states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps)
        # ========================================
        # - record agent's specific data in memory
        # ========================================

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called before the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # =====================================
        # - call `self.update(...)` if required
        # =====================================

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # =====================================
        # - call `self.update(...)` if required
        # =====================================
        # call parent's method for checkpointing and TensorBoard writing
        super().post_interaction(timestep, timesteps)

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # ===================================================
        # - implement algorithm's update step
        # - record tracking data using `self.track_data(...)`
        # ===================================================