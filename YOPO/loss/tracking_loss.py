import torch as th
import torch.nn as nn
import torch.nn.functional as F
from config.config import cfg


class TrackingLoss(nn.Module):
    def __init__(self):
        super(TrackingLoss, self).__init__()
        self.desired_dist = cfg["track_dist"]

    def forward(self, Df, Dp, target):
        """Smooth-L1 distance, in the horizontal plane, from the trajectory endpoint to the point
        desired_dist short of the target.
        Df / Dp: (batch, 3, 3) start / end state [px, vx, ax; ...].  target: (batch, 3).
        """
        start_xy = Df[:, 0:2, 0]
        end_xy = Dp[:, 0:2, 0]
        target_xy = target[:, 0:2]

        target_dist = th.norm(target_xy - start_xy, dim=1)  # [B]
        target_dir = (target_xy - start_xy) / (target_dist.unsqueeze(1) + 1e-3)  # [B, 2]

        desired_xy = start_xy + target_dir * (target_dist - self.desired_dist).unsqueeze(1)  # [B, 2]
        tracking_loss = F.smooth_l1_loss(end_xy, desired_xy, reduction='none').sum(dim=1)  # [B]

        return tracking_loss


class HeightLoss(nn.Module):
    def forward(self, pos, start_pos, target, has_target):
        """Smooth-L1 deviation of the trajectory's height from the target's, or from the drone's
        own where the frame has no target.
        pos: (B, N, 3) sampled positions.  start_pos / target: (B, 3).  has_target: (B,) bool.
        Returns (cost, mean |dz| in metres).
        """
        z_ref = th.where(has_target, target[:, 2], start_pos[:, 2]) - 0.3  # [B]
        # penalised at every sampled point, not just the endpoint
        dz = pos[:, :, 2] - z_ref.unsqueeze(1)  # [B, N]
        height_loss = F.smooth_l1_loss(dz, th.zeros_like(dz), reduction='none').mean(dim=1)
        return height_loss, dz.detach().abs().mean(dim=1)
