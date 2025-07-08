import numpy as np
from gymnasium.vector import VectorEnv
from gymnasium import spaces
import genesis as gs

from adrobo_inverted_pendulum_genesis.entity.inverted_pendulum import InvertedPendulum
from adrobo_inverted_pendulum_genesis.reward_function.reward_function import RewardFunction
from adrobo_inverted_pendulum_genesis.tools.calculation_tools import CalculationTool


class Environment(VectorEnv):
    def __init__(self, num_envs=1, max_steps=1000, show_viewer=False):
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
            show_viewer=show_viewer,
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
                n_rendered_envs = num_envs,
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

        super().__init__()

        self.max_steps = max_steps
        self.step_count = np.zeros(num_envs, dtype=np.int32)
        self.prev_inverted_degree = np.zeros(num_envs, dtype=np.float32)
        self.num_envs = num_envs
        self.env_ids = np.arange(self.num_envs)
        self.substeps = 10

        self.inverted_pendulum = InvertedPendulum(self.scene, num_envs=self.num_envs)
        self.inverted_pendulum.create()
        self.reward_function = RewardFunction()
        self.calculation_tool = CalculationTool()
        self.to_env_list = self.calculation_tool.to_env_list

        self.scene.build(n_envs=self.num_envs, env_spacing=(1.0, 1.0))

        self.dt_phys  = self.scene.dt * self.substeps

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count[:] = 0
        self.prev_inverted_degree[:] = 0
        self.scene.reset()
        self.inverted_pendulum.reset(env_idx=self.env_ids)

        observation = np.zeros((self.num_envs, 1), dtype=np.float64)
        infos = [{} for _ in range(self.num_envs)]
        return observation, infos

    def reset_idx(self, env_ids):
        self.inverted_pendulum.reset(env_idx=self.to_env_list(env_ids))
        self.prev_inverted_degree[self.to_env_list(env_ids)] = 0
        self.step_count[self.to_env_list(env_ids)] = 0

    def step(self, action):
        terminated = np.zeros(self.num_envs, dtype=np.bool_)
        truncated  = np.zeros(self.num_envs, dtype=np.bool_)
        infos = [{} for _ in range(self.num_envs)]

        self.inverted_pendulum.action(action[:, 0], action[:, 1], envs_idx=self.to_env_list(self.env_ids))
        self.scene.step(self.substeps)

        inv_deg = self.inverted_pendulum.read_inverted_degree()
        inv_vel = (inv_deg - self.prev_inverted_degree) / self.dt_phys
        self.prev_inverted_degree[:] = inv_deg
        self.step_count += 1


        observation = self.calculation_tool.normalization_inverted_degree(inv_deg).reshape(self.num_envs, 1)
        reward = self.reward_function.calculate_reward(inv_deg, inv_vel, action) + self.step_count


        step_timeout = self.step_count >= self.max_steps
        angle_fail   = np.logical_or(inv_deg <= -100.0, inv_deg >= 20.0)

        truncated[:] = step_timeout
        terminated[:] = angle_fail

        done_ids = np.where(np.logical_or(terminated, truncated))[0]
        if done_ids.size:
            self.reset_idx(done_ids)

        return observation, reward, terminated, truncated, infos

    def render(self):
        pass

    def close(self):
        pass
