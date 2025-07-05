import numpy as np
import torch

class RewardFunction:
    def __init__(self, w1=1.0, w2=0.1, w3=0.001):
        self.w1, self.w2, self.w3 = w1, w2, w3

    def _to_tensor(self, x, ref_device):
        """NumPy → Tensor 変換（既に Tensor ならそのまま）"""
        if isinstance(x, torch.Tensor):
            return x.to(ref_device)
        return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(ref_device)

    def calculate_reward(self, theta, theta_vel, action):
        """
        どの引数も ndarray / Tensor の混在を許容し、Tensor で演算 → ndarray で返す
        """
        # すべて Tensor 化（device は theta の device を優先）
        device = theta.device if isinstance(theta, torch.Tensor) else torch.device("cpu")
        theta_t      = self._to_tensor(theta,      device)
        theta_vel_t  = self._to_tensor(theta_vel,  device)
        action_t     = self._to_tensor(action,     device)

        # 報酬計算 (Tensor)
        action_norm_sq = (action_t ** 2).sum(dim=1)
        reward_t = -(
                self.w1 * theta_t ** 2 +
                self.w2 * theta_vel_t ** 2 +
                self.w3 * action_norm_sq
        )

        # NumPy 配列で返して skrl / Gymnasium の期待に合わせる
        return reward_t.cpu().numpy()      # shape = (N,)
