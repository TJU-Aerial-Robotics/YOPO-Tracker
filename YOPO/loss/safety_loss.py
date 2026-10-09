import os
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
from scipy.ndimage import distance_transform_edt
from config.config import cfg
from policy.scene_index import scene_dirs


class SafetyLoss(nn.Module):
    def __init__(self):
        super(SafetyLoss, self).__init__()
        self.traj_num = cfg['traj_num']
        self.map_expand_min = np.array(cfg['map_expand_min'])
        self.map_expand_max = np.array(cfg['map_expand_max'])
        self.d0 = cfg["d0"]
        self.r = cfg["r"]
        self.desired_dist = cfg["track_dist"]

        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")

        # SDF
        self.voxel_size = 0.2
        self.min_bounds = None  # shape: (N, 3)
        self.sdf_shapes = None  # shape: (N, 3)
        print("Building ESDF map...")
        self.sdf_maps = self.get_sdf_from_ply()
        print("Map built!")

    def forward(self, pos_coe, map_id, target):
        """pos_coe: (batch, N, 3) sampled trajectory positions.  map_id: (batch) ESDF map index.
        Returns the obstacle cost, the target-clearance cost, and the per-sample ESDF (metrics only).
        """
        # 1. obstacle barrier, averaged over the samples
        batch_size = pos_coe.shape[0]
        pos_batch = pos_coe.reshape(-1, self.traj_num * pos_coe.shape[1], 3)  # (B*H*V, N, 3) -> (B, H*V*N, 3)
        cost, dist = self.get_distance_cost(pos_batch, map_id)
        dist_esdf = dist.reshape(batch_size, -1)
        cost_colli = cost.reshape(-1, pos_coe.shape[1]).mean(dim=-1)

        # 2. target clearance, 10x the obstacle barrier
        dist_target = th.norm(pos_coe - target[:, None, :], dim=2)  # (B*H*V, N)
        cost_target = 10 * self.cost_function(dist_target)
        mask = (dist_target < self.desired_dist).float()
        cost_target_colli = (cost_target * mask).mean(dim=-1)

        return cost_colli, cost_target_colli, dist_esdf

    def get_distance_cost(self, pos, map_id):
        """Trilinear-sample the ESDF at pos (B, N, 3) in world frame -> (cost, sdf), each (B, N).
        Loops once per distinct map in the batch.
        """
        shapes = self.sdf_shapes[map_id].unsqueeze(1)                          # (B, 1, 3) map dims (x, y, z)
        grid = (pos - self.min_bounds[map_id].unsqueeze(1)) / self.voxel_size  # (B, N, 3) voxel coords
        grid = 2.0 * grid / (shapes - 1.0) - 1.0                               # -> [-1, 1]
        valid_mask = (grid < 0.99).all(-1) & (grid > -0.99).all(-1)            # (B, N) inside the map

        dist = pos.new_empty(pos.shape[:2])                                    # (B, N)
        for mid in map_id.unique().tolist():
            sel = (map_id == mid).nonzero(as_tuple=True)[0]                    # batch rows on this map
            dist[sel] = F.grid_sample(self.sdf_maps[mid], grid[sel].reshape(1, 1, 1, -1, 3),
                                      mode='bilinear', padding_mode='zeros',
                                      align_corners=True).view(len(sel), -1)
        return self.cost_function(dist).masked_fill(~valid_mask, 0.0), dist

    def cost_function(self, d):
        return th.exp(-(d - self.d0) / self.r)

    def get_sdf_from_ply(self):
        # same order as YOPODataset's map_idx
        sorted_files = [os.path.join(p, "environment.ply") for _, p in scene_dirs()]
        sdf_maps = []
        min_bounds, sdf_shapes = [], []

        # First pass to get all sdf_maps and record shape
        for file in sorted_files:
            pcd = o3d.io.read_point_cloud(file)
            min_bound = np.array(pcd.get_min_bound()) - self.map_expand_min
            max_bound = np.array(pcd.get_max_bound()) + self.map_expand_max
            points = np.asarray(pcd.points)
            print(f"    {os.path.basename(file)}: x=({min_bound[0] + self.map_expand_min[0]:.2f}, {max_bound[0] - self.map_expand_max[0]:.2f}), "
                  f"y=({min_bound[1] + self.map_expand_min[1]:.2f}, {max_bound[1] - self.map_expand_max[1]:.2f}), "
                  f"z=({min_bound[2] + self.map_expand_min[2]:.2f}, {max_bound[2] - self.map_expand_max[2]:.2f})")

            sdf_shape = np.ceil((max_bound - min_bound) / self.voxel_size).astype(int)
            voxel_indices = ((points - min_bound) / self.voxel_size).astype(int)

            valid_mask = np.all((voxel_indices >= 0) & (voxel_indices < sdf_shape), axis=1)
            voxel_indices = voxel_indices[valid_mask]

            occupancy = np.zeros(sdf_shape, dtype=np.uint8)
            occupancy[tuple(voxel_indices.T)] = 1

            obstacle_mask = occupancy == 1
            free_mask = occupancy == 0

            dist_to_obstacle = distance_transform_edt(free_mask) * self.voxel_size
            dist_inside_obstacle = distance_transform_edt(obstacle_mask) * self.voxel_size

            dist_to_obstacle[obstacle_mask] = -dist_inside_obstacle[obstacle_mask]

            sdf_tensor = th.from_numpy(dist_to_obstacle).float().unsqueeze(0).unsqueeze(0).permute(0, 1, 4, 3, 2).to(self.device)  # (1, 1, D, H, W)

            sdf_maps.append(sdf_tensor)
            sdf_shapes.append(sdf_tensor.shape[-3:][::-1])  # D, H, W -> X, Y, Z
            min_bounds.append(min_bound)

        # Map bounds and shapes, used by get_distance_cost to normalize the query grid
        self.min_bounds = th.tensor(np.array(min_bounds), device=self.device).float()  # shape: (N, 3)
        self.sdf_shapes = th.tensor(np.array(sdf_shapes), device=self.device).float()  # shape: (N, 3) order: (X, Y, Z)
        return sdf_maps  # shape: (N, 1, D, H, W)

