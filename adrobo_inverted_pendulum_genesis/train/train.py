import os

import torch
import torch.nn as nn
from skrl.utils.model_instantiators.torch import gaussian_model, deterministic_model
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import ParallelTrainer
from skrl.utils import set_seed

from adrobo_inverted_pendulum_genesis.environment.environment import Environment


script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
save_dir = os.path.join(parent_dir, 'model','default_model')
os.makedirs(save_dir, exist_ok=True)

set_seed(0)

class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, self.num_actions))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        # Pendulum-v1 action_space is -2 to 2
        return 2 * torch.tanh(self.net(inputs["states"])), self.log_std_parameter, {}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 1))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


vec_env = Environment(num_envs=20, max_steps=2000, show_viewer=False)
env     = wrap_env(vec_env)
# device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = env.device

memory = RandomMemory(memory_size=1024, num_envs=env.num_envs, device=device)

models = {}
models["policy"] = Policy(env.observation_space, env.action_space, device, clip_actions=True)
models["value"] = Value(env.observation_space, env.action_space, device)

cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 1024
cfg["learning_epochs"] = 8
cfg["mini_batches"] = 1024

cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95

cfg["learning_rate"] = 3e-4
cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}

cfg["grad_norm_clip"] = 0.5
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = False

cfg["entropy_loss_scale"] = 0.0
cfg["value_loss_scale"] = 0.5

cfg["kl_threshold"] = 0

cfg["mixed_precision"] = True
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

cfg["experiment"]["directory"] = "../model"
cfg["experiment"]["experiment_name"] = "inverted_pendulum"
cfg["experiment"]["write_interval"] = 1000
cfg["experiment"]["checkpoint_interval"] = 1000
cfg["experiment"]["store_separately"] = False


agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

cfg_trainer = {"timesteps": 1000, "headless": True}
trainer = ParallelTrainer(cfg=cfg_trainer, env=env, agents=[agent])


trainer.train()

env.close()

from pathlib import Path

scaler = agent._state_preprocessor.to("cpu")   # <= ここを修正
policy = models["policy"].to("cpu").eval()

class PolicyWithScaler(nn.Module):
    def __init__(self, scaler, net):
        super().__init__()
        self.scaler, self.net = scaler, net
    def forward(self, x):
        x = self.scaler(x, train=False, no_grad=True)
        mu, log_std, _ = self.net({"states": x}, role="policy")
        return mu, log_std

wrapper = PolicyWithScaler(scaler, policy)
dummy = torch.zeros(1, *env.single_observation_space.shape)

torch.onnx.export(
    wrapper, dummy, "policy.onnx",
    input_names=["obs"],
    output_names=["action_mu", "action_log_std"],
    dynamic_axes={"obs": {0: "batch"},
                  "action_mu": {0: "batch"},
                  "action_log_std": {0: "batch"}},
    opset_version=18
)

