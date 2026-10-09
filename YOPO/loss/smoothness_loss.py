import torch.nn as nn
import torch as th
from config.config import cfg


class SmoothnessLoss(nn.Module):
    MIN_TIME = 1e-2  # lower clamp on T, guards the T^-5 division

    def __init__(self, RJ, RA):
        super(SmoothnessLoss, self).__init__()
        self._RJ = RJ  # R matrix in normalized time (T = 1), for the jerk energy
        self._RA = RA  # same, for the acceleration energy
        self.vel_max = cfg["vel_max_train"]
        self.acc_max = 0.7 * cfg["acc_max_train"]

    def forward(self, D_scaled, traj_time, vel, acc):
        """D_scaled: (batch, 3, 6) scaled boundary conditions from YOPOLoss.solve_trajectory.
        traj_time: (batch) duration T.  vel / acc: (batch, N, 3) at the sampled points.
        Returns (integral of jerk^2, integral of acc^2, vel/acc limit penalty), each (batch,).
        """
        T = traj_time.clamp(min=self.MIN_TIME)
        cost_smooth = th.einsum('bai,ij,baj->b', D_scaled, self._RJ, D_scaled) / T ** 5  # summed over the 3 axes
        cost_accel = th.einsum('bai,ij,baj->b', D_scaled, self._RA, D_scaled) / T ** 3

        # Relative excess of the speed / acceleration norms, taken at the worst sample
        vel_over = th.relu(vel.norm(dim=-1) / self.vel_max - 1.0)  # (batch, N)
        acc_over = th.relu(acc.norm(dim=-1) / self.acc_max - 1.0)  # (batch, N)
        cost_feasible = vel_over.amax(dim=-1) ** 2 + acc_over.amax(dim=-1) ** 2

        return cost_smooth, cost_accel, cost_feasible
