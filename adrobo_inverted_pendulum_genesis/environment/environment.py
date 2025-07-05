import numpy as np
import gymnasium as gym
from gymnasium.vector import VectorEnv
from gymnasium import spaces
import torch
import genesis as gs

from adrobo_inverted_pendulum_genesis.entity.inverted_pendulum import InvertedPendulum
from adrobo_inverted_pendulum_genesis.reward_function.reward_function import RewardFunction
from adrobo_inverted_pendulum_genesis.tools.calculation_tools import CalculationTool


class Environment(VectorEnv):
    def __init__(self, num_envs=1, max_steps=1000):

        self.max_steps = max_steps
        self.step_count = np.zeros(num_envs, dtype=np.int32)
        self.prev_inverted_degree = np.zeros(num_envs, np.float32)
        self.num_envs = num_envs

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

        self.single_action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.single_observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # self.action_space = spaces.Box(
        #     low=np.array([-1.0, -1.0], dtype=np.float32),
        #     high=np.array([1.0, 1.0], dtype=np.float32),
        #     shape=(self.num_envs, 2),
        #     dtype=np.float32
        # )
        #
        # self.observation_space = spaces.Box(
        #     low=-1.0,
        #     high=1.0,
        #     shape=(self.num_envs, 1),
        #     dtype=np.float32
        # )

        super().__init__()

        self.max_steps = max_steps

        self.inverted_pendulum = InvertedPendulum(self.scene, num_envs=self.num_envs)
        self.inverted_pendulum.create()
        self.reward_function = RewardFunction()
        self.calculation_tool = CalculationTool()

        self.scene.build(n_envs=self.num_envs, env_spacing=(0.5, 0.5))

        self.reset()


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count[:] = 0
        self.prev_inverted_degree[:] = 0.0

        self.scene.reset()

        obs = np.zeros((self.num_envs, 1), dtype=np.float32)
        infos = {}
        return obs, infos

    def reset_idx(self, env_ids):
        self.inverted_pendulum.reset(env_idx=env_ids)


    def step(self, action):
        terminated = np.zeros(self.num_envs, dtype=np.bool_)
        truncated  = np.zeros(self.num_envs, dtype=np.bool_)
        info = {}

        # 1 シミュレーション step
        self.inverted_pendulum.action(action[:, 0], action[:, 1])
        self.scene.step(10)

        # 観測
        inv_deg = self.inverted_pendulum.read_inverted_degree()                # shape=(N,)
        inv_vel = (inv_deg - self.prev_inverted_degree) / 0.01                 # shape=(N,)
        self.prev_inverted_degree[:] = inv_deg
        self.step_count += 1                                                   # shape=(N,)

        observation = self.calculation_tool.normalization_inverted_degree(inv_deg).reshape(self.num_envs, 1)

        # 報酬
        reward = self.reward_function.calculate_reward(inv_deg, inv_vel, action)
        if isinstance(reward, torch.Tensor):
            reward = reward.cpu().numpy()                                      # shape=(N,)

        # 終了判定
        step_timeout_mask = self.step_count >= self.max_steps                  # ndarray(bool)
        angle_fail_mask   = np.logical_or(inv_deg <= -100.0, inv_deg >= 30.0)  # ndarray(bool)
        terminated = np.logical_or(step_timeout_mask, angle_fail_mask)         # ndarray(bool)

        # タイムアウトは truncated 扱い
        truncated[:] = step_timeout_mask

        # リセット
        done_ids = np.where(terminated)[0]
        if done_ids.size > 0:
            self.reset_idx(done_ids)

        return observation, reward, terminated, truncated, info



    def render(self):
        pass

    def close(self):
        pass
