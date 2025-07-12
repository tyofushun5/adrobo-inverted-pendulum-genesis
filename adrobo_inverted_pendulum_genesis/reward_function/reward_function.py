import numpy as np

class RewardFunction(object):
    def __init__(self,
                 w_posture=1.0,   # 姿勢報酬係数
                 w_vel=0.005,     # 角速度罰則
                 w_act=0.0001,    # トルク罰則
                 theta_min=-100.0,
                 theta_max=+20.0,
                 fail_penalty=-5.0):
        self.w_p, self.w_v, self.w_a = w_posture, w_vel, w_act
        self.th_min = float(theta_min)
        self.th_max = float(theta_max)
        self.fail_penalty = fail_penalty

    # --------------------------------------------------
    def _normalize(self, theta: np.ndarray) -> np.ndarray:
        pos_scale = self.th_max      #
        neg_scale = -self.th_min
        theta_n = np.where(theta >= 0,
                           theta / pos_scale,
                           theta / neg_scale)
        return np.clip(theta_n, -1.0, 1.0)

    # --------------------------------------------------
    def calculate_reward(self,
                         theta,
                         theta_vel,
                         action,
                         ):
        theta      = np.asarray(theta,      np.float32)
        theta_vel  = np.asarray(theta_vel,  np.float32)
        action     = np.asarray(action,     np.float32)

        th_n = self._normalize(theta)
        r_posture = 1.0 - th_n**2

        # r_speed  = - self.w_v * theta_vel**2
        # r_energy = - self.w_a * np.sum(action**2, axis=1)

        reward = self.w_p * r_posture #+ r_speed + r_energy

        return reward
