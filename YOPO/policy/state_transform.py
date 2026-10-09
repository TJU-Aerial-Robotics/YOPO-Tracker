import torch
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.config import cfg
from policy.primitive import LatticePrimitive


class StateTransform:
    def __init__(self):
        self.lattice_primitive = LatticePrimitive.get_instance()
        self.grid = self.generate_grid(cfg["vertical_num"], cfg["horizon_num"])
        self.camera_cx = cfg["camera_cx"]
        self.camera_cy = cfg["camera_cy"]
        self.camera_f = cfg["camera_f"]
        self.downsample = cfg["downsample"]
        self.max_traj_time = 2.0 * self.lattice_primitive.segment_time

        # Lattice constants cached on CPU, used by the *_cpu interfaces at test time
        yaw, pitch = self.lattice_primitive.getAngleLattice()
        self.lattice_yaw_np = yaw.cpu().numpy()
        self.lattice_pitch_np = pitch.cpu().numpy()
        self.lattice_Rbp_np = self.lattice_primitive.getRotation().cpu().numpy()
        self.grid_np = self.grid.cpu().numpy().astype(np.float32)  # float32, avoids promotion when added to the predictions

    def pred_to_endstate(self, endstate_pred: torch.Tensor) -> torch.Tensor:
        """
            Transform the predicted state to the body frame (Original prediction → Primitive frame → Body frame).
            endstate_pred: [batch; px py pz vx vy vz ax ay az; primitive_v; primitive_h]
            :return [batch; px py pz vx vy vz ax ay az; primitive_v; primitive_h] in body frame
        """
        B, N = endstate_pred.shape[0], endstate_pred.shape[2] * endstate_pred.shape[3]

        # [B, 9, 3, 5] -> [B, 3, 5, 9] -> [B, 15, 9]
        endstate_pred = endstate_pred.permute(0, 2, 3, 1).reshape(B, N, 9)

        # Lattice angles and rotations (.flip: lattice and grid are ordered oppositely)
        yaw, pitch = self.lattice_primitive.getAngleLattice()  # [15]
        yaw = yaw.flip(0)[None, :].expand(B, -1)  # [B, 15]
        pitch = pitch.flip(0)[None, :].expand(B, -1)  # [B, 15]
        Rbp = self.lattice_primitive.getRotation().flip(0)  # [15, 3, 3]
        Rbp = Rbp[None, :, :, :].expand(B, -1, -1, -1)  # [B, 15, 3, 3]

        delta_yaw = endstate_pred[:, :, 0] * self.lattice_primitive.yaw_diff  # [B, 15]
        delta_pitch = endstate_pred[:, :, 1] * self.lattice_primitive.pitch_diff
        radio = (endstate_pred[:, :, 2] + 1.0) * self.lattice_primitive.radio_range

        cos_pitch = torch.cos(pitch + delta_pitch)
        endstate_x = cos_pitch * torch.cos(yaw + delta_yaw) * radio
        endstate_y = cos_pitch * torch.sin(yaw + delta_yaw) * radio
        endstate_z = torch.sin(pitch + delta_pitch) * radio
        endstate_p = torch.stack([endstate_x, endstate_y, endstate_z], dim=-1)  # [B, 15, 3]

        # vel / acc
        endstate_vp = endstate_pred[:, :, 3:6] * self.lattice_primitive.vel_max  # [B, 15, 3]
        endstate_ap = endstate_pred[:, :, 6:9] * self.lattice_primitive.acc_max  # [B, 15, 3]

        # v / a to the body frame
        endstate_vb = torch.matmul(Rbp, endstate_vp.unsqueeze(-1)).squeeze(-1)  # [B, 15, 3]
        endstate_ab = torch.matmul(Rbp, endstate_ap.unsqueeze(-1)).squeeze(-1)

        endstate = torch.cat([endstate_p, endstate_vb, endstate_ab], dim=-1)  # [B, 15, 9]

        endstate = endstate.permute(0, 2, 1).reshape(B, 9, 3, 5)  # [B, 9, 3, 5]
        return endstate

    def pred_to_endstate_cpu(self, endstate_pred: np.ndarray, lattice_id) -> np.ndarray:
        """
            Used during test:
            Numpy version of pred_to_endstate() on CPU (used in test, x10 times faster than torch on CUDA)
            lattice_id: int or np.ndarray (not a torch.Tensor)
            :return [B; px py pz vx vy vz ax ay az] in body frame
        """
        delta_yaw = endstate_pred[:, 0] * self.lattice_primitive.yaw_diff
        delta_pitch = endstate_pred[:, 1] * self.lattice_primitive.pitch_diff
        radio = (endstate_pred[:, 2] + 1.0) * self.lattice_primitive.radio_range

        yaw, pitch = self.lattice_yaw_np[lattice_id], self.lattice_pitch_np[lattice_id]
        endstate_x = np.cos(pitch + delta_pitch) * np.cos(yaw + delta_yaw) * radio
        endstate_y = np.cos(pitch + delta_pitch) * np.sin(yaw + delta_yaw) * radio
        endstate_z = np.sin(pitch + delta_pitch) * radio
        endstate_p = np.stack((endstate_x, endstate_y, endstate_z), axis=1)

        endstate_vp = endstate_pred[:, 3:6] * self.lattice_primitive.vel_max
        endstate_ap = endstate_pred[:, 6:9] * self.lattice_primitive.acc_max

        Rpb = self.lattice_Rbp_np[lattice_id]
        endstate_vb = np.matmul(Rpb, endstate_vp[:, :, np.newaxis]).squeeze(-1)
        endstate_ab = np.matmul(Rpb, endstate_ap[:, :, np.newaxis]).squeeze(-1)

        return np.concatenate((endstate_p, endstate_vb, endstate_ab), axis=1)

    def pred_to_traj_time(self, time_pred: torch.Tensor) -> torch.Tensor:
        """Scale the normalized time prediction (0, 1) to seconds. Shape is preserved."""
        return time_pred * self.max_traj_time

    def pred_to_traj_time_cpu(self, time_pred: np.ndarray) -> np.ndarray:
        """Numpy version of pred_to_traj_time(), used during test."""
        return time_pred * self.max_traj_time

    def pred_to_target(self, target_pred: torch.Tensor, transfer_world=False) -> torch.Tensor:
        grid = self.grid.repeat(target_pred.shape[0], 1, 1, 1)
        target_pred[:, 0:2] = (target_pred[:, 0:2] + grid) * self.downsample
        if transfer_world:
            u, v, d = target_pred[:, 1], target_pred[:, 0], target_pred[:, 2]
            y = -d * (u - self.camera_cx) / self.camera_f
            z = -d * (v - self.camera_cy) / self.camera_f
            x = d
            target_pred = torch.stack([x, y, z], dim=1)
        return target_pred

    def pred_to_target_cpu(self, target_pred: np.ndarray, transfer_world=False) -> np.ndarray:
        """Numpy version of pred_to_target(), used during test.
            target_pred: [B, 3(v u d), vertical_num, horizon_num]
        """
        vu = (target_pred[:, 0:2] + self.grid_np) * self.downsample
        d = target_pred[:, 2]
        if not transfer_world:
            return np.concatenate((vu, d[:, np.newaxis]), axis=1)
        v, u = vu[:, 0], vu[:, 1]
        y = -d * (u - self.camera_cx) / self.camera_f
        z = -d * (v - self.camera_cy) / self.camera_f
        return np.stack([d, y, z], axis=1)

    def endstate_to_pred(self, target: torch.Tensor) -> torch.Tensor:
        """
            Convert the endstate (target's position) in the body frame into the expected prediction in the primitive frame.
        """
        yaw_actual = torch.atan2(target[:, 1], target[:, 0])
        xy_projection = torch.sqrt(target[:, 0]**2 + target[:, 1]**2)
        pitch_actual = torch.atan2(target[:, 2], xy_projection)
        yaw, pitch = self.lattice_primitive.getAngleLattice()
        delta_yaw = (yaw_actual.unsqueeze(1) - yaw.flip(0).unsqueeze(0))
        delta_pitch = (pitch_actual.unsqueeze(1) - pitch.flip(0).unsqueeze(0))
        return delta_yaw / self.lattice_primitive.yaw_diff, delta_pitch / self.lattice_primitive.pitch_diff

    def prepare_input(self, obs):
        """
            Transform the observation to the primitive frame (Body frame → Primitive frame → Body frame).
            obs: [batch; vx, vy, yz, ax, ay, az] in body frame
            :return [batch; vx, vy, yz, ax, ay, az; primitive_v; primitive_h] in primitive frame
        """
        B, N = obs.shape[0], self.lattice_primitive.traj_num

        # All Rbp, reversed (lattice and grid are ordered oppositely)
        Rbp_all = self.lattice_primitive.getRotation().flip(0)  # shape: [N, 3, 3]

        obs = obs.view(B, 2, 3)  # [B, 2, 3]

        # expand obs and Rbp to [B, N, 2, 3] and [B, N, 3, 3]
        obs_exp = obs[:, None, :, :].expand(B, N, 2, 3)
        Rbp_exp = Rbp_all[None, :, :, :].expand(B, N, 3, 3)

        # batched frame transform
        transformed = torch.matmul(obs_exp, Rbp_exp)  # [B, N, 2, 3]

        transformed_flat = transformed.view(B, N, 6)  # [B, N, 6]
        out = transformed_flat.permute(0, 2, 1).contiguous()  # [B, 6, N]
        out = out.view(B, 6, self.lattice_primitive.vertical_num, self.lattice_primitive.horizon_num)  # [B, 6, V, H]
        return out

    # Returns a new tensor rather than modifying obs in place
    def unnormalize_obs(self, vel_acc):
        return torch.cat([vel_acc[:, 0:3] * self.lattice_primitive.vel_max,
                          vel_acc[:, 3:6] * self.lattice_primitive.acc_max], dim=1)

    def normalize_obs(self, vel_acc):
        return torch.cat([vel_acc[:, 0:3] / self.lattice_primitive.vel_max,
                          vel_acc[:, 3:6] / self.lattice_primitive.acc_max], dim=1)

    def generate_grid(self, vertical_num, horizon_num):
        row_indices = torch.arange(vertical_num)
        col_indices = torch.arange(horizon_num)
        row_grid, col_grid = torch.meshgrid(row_indices, col_indices, indexing="ij")
        grid_array = torch.stack((row_grid, col_grid), dim=0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return grid_array.to(device)


def rotate_body2world(rot_wb, pos_b):
    """
    Rotate pos_b from body frame to world frame using quaternion q_wb.
    rot_wb: (..., 3, 3)
    pos_b: (..., 3)
    """
    pos_w = torch.matmul(rot_wb, pos_b.unsqueeze(-1)).squeeze(-1)
    return pos_w


def transform_body2world(rot_wb, t_w, pos_b):
    """
    Transform pos_b from body frame to world frame using quaternion q_wb and t_w.
    rot_wb: (..., 3, 3)
    t_w: (..., 3)
    pos_b: (..., 3)
    """
    return rotate_body2world(rot_wb, pos_b) + t_w


def state_body2world(pos_w, rot_wb, pos_b, vel_b, acc_b):
    pos_b = transform_body2world(rot_wb, pos_w, pos_b)
    vel_b = rotate_body2world(rot_wb, vel_b)
    acc_b = rotate_body2world(rot_wb, acc_b)
    return pos_b, vel_b, acc_b


if __name__ == '__main__':
    CoordTransform = StateTransform()
