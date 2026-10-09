import torch.nn as nn
from torch.nn import functional as F
from config.config import cfg


class DetectionLoss(nn.Module):
    def __init__(self):
        super(DetectionLoss, self).__init__()
        self.traj_num = cfg["traj_num"]
        self.camera_cx = cfg["camera_cx"]
        self.camera_cy = cfg["camera_cy"]
        self.camera_f = cfg["camera_f"]

    def forward(self, pred_vud, label_xyz, pos_mask, dist_mask=None):
        """Smooth-L1 detection loss in the body frame over the positive samples: the predicted
        (v, u, d) is back-projected to (x, y, z) using the ground-truth depth for y and z.
        dist_mask: per-anchor, gates the distance term alone; None keeps it on everywhere.
        """
        label_xyz = label_xyz.repeat_interleave(self.traj_num, dim=0)
        u, v = pred_vud[:, 1], pred_vud[:, 0]
        d = label_xyz[:, 0]
        y = -d * (u - self.camera_cx) / self.camera_f
        z = -d * (v - self.camera_cy) / self.camera_f
        x = pred_vud[:, 2]
        dist_loss = F.smooth_l1_loss(x, label_xyz[:, 0], reduction='none')
        if dist_mask is not None:
            dist_loss = dist_loss * dist_mask
        detection_loss = (dist_loss +
                          F.smooth_l1_loss(y, label_xyz[:, 1], reduction='none') +
                          F.smooth_l1_loss(z, label_xyz[:, 2], reduction='none'))
        detection_loss = detection_loss * pos_mask
        return detection_loss.mean()
