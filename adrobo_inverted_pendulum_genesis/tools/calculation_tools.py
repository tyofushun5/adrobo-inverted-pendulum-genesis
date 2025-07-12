import numpy as np

class CalculationTool(object):
    @staticmethod
    def denormalization_inverted_degree(value):
        value = np.asarray(value, dtype=np.float32)
        return (value + 1) * 180

    @staticmethod
    def normalization_inverted_degree(value, low=-100.0, high=20.0):
        value = np.asarray(value, dtype=np.float32)
        value = np.clip(value, low, high)
        return 2 * (value - low) / (high - low) - 1

    @staticmethod
    def to_env_list(idx):
        return None if idx is None else np.atleast_1d(idx).astype(np.int32).tolist()
