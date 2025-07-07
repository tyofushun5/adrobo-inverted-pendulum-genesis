import os
import torch
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.utils.model_instantiators.torch import gaussian_model, deterministic_model
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer

from adrobo_inverted_pendulum_genesis.environment.environment import Environment


vec_env = Environment(num_envs=3, max_steps=1000, show_viewer=True)
env     = wrap_env(vec_env)
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")


hid_layers = [{"name": "mlp",
               "input": "OBSERVATIONS",
               "layers": [256, 256, 256],
               "activations": "elu"}]

policy = gaussian_model(env.observation_space, env.action_space,
                        device=device, clip_actions=True,
                        network=hid_layers,
                        output="ACTIONS")

value  = deterministic_model(env.observation_space, None,
                             device=device,
                             network=hid_layers,
                             output="ONE")

models  = {"policy": policy, "value": value}

memory = RandomMemory(memory_size=2048,
                      num_envs=env.num_envs,
                      device=device)

cfg = PPO_DEFAULT_CONFIG.copy()
cfg.update({
    "rollouts":        2048 // env.num_envs,
    "learning_epochs": 10,
    "mini_batches":    4,
    "discount_factor": 0.99,
    "lambda":          0.95,
    "learning_rate":   3e-4,
    "grad_norm_clip":  0.5
})

agent = PPO(models=models, memory=memory,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device, cfg=cfg)

trainer = SequentialTrainer(env=env, agents=agent,
                            cfg={"timesteps": 5000000, "headless": True})
trainer.train()

os.makedirs("model", exist_ok=True)
torch.save(policy.state_dict(), "model/default_model_v10.pt")
env.close()
