import torch.nn as nn
import torch as th
from config.config import cfg


class GuidanceLoss(nn.Module):
    def __init__(self):
        super(GuidanceLoss, self).__init__()
        self.goal_length = cfg['goal_length']

    def forward(self, Df, Dp):
        """Absolute deviation of the trajectory length from goal_length.
        Df / Dp: (batch, 3, 3) start / end state [px, vx, ax; ...].
        """
        cur_pos = Df[:, :, 0]
        end_pos = Dp[:, :, 0]

        traj_dir = end_pos - cur_pos  # [B, 3]
        traj_length = traj_dir.norm(dim=-1)  # [B]

        guidance_loss = th.abs(self.goal_length - traj_length)  # [B]
        return guidance_loss
