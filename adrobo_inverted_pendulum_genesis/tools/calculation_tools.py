import math

import numpy as np


class CalculationTool(object):
    def __init__(self):
        pass

    @staticmethod
    def denormalization_inverted_degree(value):
        denormalized = (value + 1) * 180
        return denormalized

    @staticmethod
    def normalization_inverted_degree(value: float, low: float = -130.0, high: float = 30.0) -> float:
        angle = max(low, min(high, value))
        normalized = 2 * (angle - low) / (high - low) - 1  # low→-1, high→+1
        return normalized
