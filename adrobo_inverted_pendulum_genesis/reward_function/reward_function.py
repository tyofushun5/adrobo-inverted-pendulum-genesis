import math

import numpy as np


class RewardFunction:
    def __init__(self, w1=1.0, w2=0.1, w3=0.001):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    def calculate_reward(
            self,
            theta: float,
            theta_vel: float,
            action: np.ndarray
    ) -> float:
        """
        θ: 正規化済み角度（例：ラジアン／π、[-1, 1]の範囲）
        θ̇: 角速度（ラジアン/秒）
        action: 制御入力
        """
        action_norm = np.linalg.norm(action)
        reward = - (
                self.w1 * theta ** 2 +
                self.w2 * theta_vel ** 2 +
                self.w3 * action_norm ** 2
        )
        return float(reward)
