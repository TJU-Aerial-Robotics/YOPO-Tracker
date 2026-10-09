"""
YOPO Network
forward, prediction, pre-processing, post-processing
"""

import torch
from torch import nn
from policy.models.backbone import YopoBackbone
from policy.models.head import YopoHead
from policy.state_transform import StateTransform


class YopoNetwork(nn.Module):

    def __init__(
            self,
            observation_dim=6,  # 9: v_xyz, a_xyz
            output_dim=15,  # 15: x_pva, y_pva, z_pva, target_xyz, traj_cost, obj_score, traj_time
            hidden_state=64,
    ):
        super(YopoNetwork, self).__init__()
        self.state_transform = StateTransform()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.image_backbone = YopoBackbone(hidden_state)
        self.state_backbone = nn.Sequential()
        self.yopo_head = YopoHead(hidden_state + observation_dim, output_dim)

    def forward(self, depth: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """
            forward propagation of neural network
        """
        depth_feature = self.image_backbone(depth)
        obs_feature = self.state_backbone(obs)
        input_tensor = torch.cat((obs_feature, depth_feature), 1)
        output = self.yopo_head(input_tensor)

        endstate = torch.tanh(output[:, :9])  # [batch, 9, vertical_num, horizon_num]

        # Offset range [-0.5, 1.5] instead of sigmoid's [0, 1], so a cell can point outside itself (YOLOv5)
        target_vu = torch.sigmoid(output[:, 9:11]) * 2.0 - 0.5
        target_d = torch.nn.functional.softplus(output[:, 11]).unsqueeze(1)
        target = torch.cat((target_vu, target_d), dim=1)  # [batch, 3, vertical_num, horizon_num]

        traj_cost = torch.nn.functional.softplus(output[:, 12])  # [batch, vertical_num, horizon_num]
        obj_score = torch.sigmoid(output[:, 13])     # [batch, vertical_num, horizon_num]

        # Trajectory duration: sigmoid-normalized to (0, 1); like endstate, the de-normalization is left to state_transform
        traj_time = torch.sigmoid(output[:, 14])  # [batch, vertical_num, horizon_num]

        return endstate, target, traj_cost, obj_score, traj_time

    def inference(self, depth: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """
            For network training:
            (1) normalize the input state and transform to primitive frame
            (2) forward propagation
            (3) convert the prediction to endstate in body frame, and the time prediction to seconds.
        """
        obs = self.state_transform.normalize_obs(obs)
        obs = self.state_transform.prepare_input(obs)
        endstate_pred, target_pred, cost_pred, score_pred, time_pred = self.forward(depth, obs)
        endstate = self.state_transform.pred_to_endstate(endstate_pred)
        target = self.state_transform.pred_to_target(target_pred)
        traj_time = self.state_transform.pred_to_traj_time(time_pred)
        return endstate, target, cost_pred, score_pred, traj_time

    def print_grad(self, grad):
        print("grad of hook: ", grad)
