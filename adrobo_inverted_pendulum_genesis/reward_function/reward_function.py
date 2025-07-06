import numpy as np

class RewardFunction:
    def __init__(self, w1=1.0, w2=0.1, w3=0.001):
        self.w1, self.w2, self.w3 = w1, w2, w3

    def calculate_reward(self, theta, theta_vel, action):
        theta      = np.asarray(theta,      dtype=np.float32)
        theta_vel  = np.asarray(theta_vel,  dtype=np.float32)
        action     = np.asarray(action,     dtype=np.float32)

        action_norm_sq = np.sum(action ** 2, axis=1)
        reward = -(
                self.w1 * theta ** 2 +
                self.w2 * theta_vel ** 2 +
                self.w3 * action_norm_sq
        )
        return reward
