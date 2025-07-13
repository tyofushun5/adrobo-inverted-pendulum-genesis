import numpy as np
import torch


class CalculationTool(object):

    @staticmethod
    def normalization_inverted_degree(value, low: float = -20.0, high: float = 20.0):
        if isinstance(value, torch.Tensor):
            value = torch.clamp(value, low, high)
            return 2 * (value - low) / (high - low) - 1

        value = np.asarray(value, dtype=np.float32)
        value = np.clip(value, low, high)
        return 2 * (value - low) / (high - low) - 1

    @staticmethod
    def to_env_list(idx):
        return None if idx is None else np.atleast_1d(idx).astype(np.int32).tolist()