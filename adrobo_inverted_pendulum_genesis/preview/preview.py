import os
import sys

import numpy as np
import onnxruntime as ort
import torch

from adrobo_inverted_pendulum_genesis.environment.environment import Environment

root = os.path.dirname(os.path.abspath(__file__))
model_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "..", "train", "policy.onnx")

sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
env = Environment(num_envs=1, max_steps=10_000, device="cuda", show_viewer=True)

obs, _ = env.reset()

while True:
    obs_np = obs.detach().cpu().numpy() if isinstance(obs, torch.Tensor) else obs
    mu, _ = sess.run(None, {"obs": obs_np.astype(np.float32)})

    device = obs.device if isinstance(obs, torch.Tensor) else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    action = torch.from_numpy(mu).to(device)

    obs, _, terminated, truncated, _ = env.step(action)
