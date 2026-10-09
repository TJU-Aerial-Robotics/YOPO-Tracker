import math
import torch as th
import torch.nn as nn
from config.config import cfg
from loss.safety_loss import SafetyLoss
from loss.smoothness_loss import SmoothnessLoss
from loss.tracking_loss import TrackingLoss, HeightLoss
from loss.guidance_loss import GuidanceLoss


class YOPOLoss(nn.Module):
    """Weighted cost of a batch of single-segment quintic polynomial trajectories.

    Smooth / Accel: time integrals of jerk^2 and acc^2.   Safety: ESDF barrier along the trajectory.
    Track: endpoint toward the target, horizontally.   Height: altitude held at the target's.
    Guide: forward flight when no target is visible.
    Time: the duration T.   Feasible: relative excess over vel_max_train / acc_max_train.
    """

    def __init__(self):
        super(YOPOLoss, self).__init__()
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.eval_points = 30  # trajectory samples, shared by the safety and feasibility terms
        self.eval_s = th.linspace(1.0 / self.eval_points, 1.0, self.eval_points, device=self.device)
        self._L, self._RJ, self._RA = [m.to(self.device) for m in self.qp_generation()]

        # Divide the jerk / acc weights by vel_scale^5 and ^3; the other terms are speed-invariant
        vel_scale = cfg["vel_max_train"]
        self.weights = {
            "Smooth":   cfg["ws"] / vel_scale ** 5,
            "Safety":   cfg["wc"],
            "Track":    cfg["wt"],
            "Guide":    cfg["wg"],
            "Height":   cfg["wh"],
            "Time":     cfg["wtime"],
            "Feasible": cfg["wf"],
            "Accel":    cfg["wa"] / vel_scale ** 3,
        }

        self.smoothness_loss = SmoothnessLoss(self._RJ, self._RA)
        self.safety_loss = SafetyLoss()
        self.tracking_loss = TrackingLoss()
        self.guidance_loss = GuidanceLoss()
        self.height_loss = HeightLoss()

        print("------ Actual Loss ------")
        for name, w in self.weights.items():
            print(f"| {name:<12} = {w:8.6f} |")
        print("-------------------------")

    def qp_generation(self):
        """Build the constant matrices in normalized time s = t/T (i.e. T = 1).

        Returns (L, R_jerk, R_acc): L maps boundary conditions to coefficients, R_* are the
        quadratic forms of the jerk / acceleration energies.
        """
        # Mapping matrix of the paper (T = 1, so every power of T is 1)
        A = th.zeros((6, 6))
        for i in range(3):
            A[2 * i, i] = math.factorial(i)
            for j in range(i, 6):
                A[2 * i + 1, j] = math.factorial(j) / math.factorial(j - i)

        # Hessian of the jerk energy (matrix Q in the paper)
        H = th.zeros((6, 6))
        for i in range(3, 6):
            for j in range(3, 6):
                H[i, j] = i * (i - 1) * (i - 2) * j * (j - 1) * (j - 2) / (i + j - 5)

        # Hessian of the acceleration energy
        Q = th.zeros((6, 6))
        for i in range(2, 6):
            for j in range(2, 6):
                Q[i, j] = (i * (i - 1)) * (j * (j - 1)) / (i + j - 3)

        # Ct reorders [p0, v0, a0, pT, vT, aT] into the fixed/free split used by L
        Ct = th.zeros((6, 6))
        Ct[[0, 2, 4, 1, 3, 5], [0, 1, 2, 3, 4, 5]] = 1
        B = th.inverse(A)
        L = B @ Ct
        R_jerk = Ct.T @ B.T @ H @ B @ Ct
        R_acc = Ct.T @ B.T @ Q @ B @ Ct
        return L, R_jerk, R_acc

    def solve_trajectory(self, Df, Dp, traj_time):
        """Solve the quintic from its boundary conditions.

        Df / Dp:   (batch, 3, 3) start / end state, [px, vx, ax; py, vy, ay; pz, vz, az]
        traj_time: (batch) per-trajectory duration T
        Returns coeff (batch, 3, 6), coefficients in normalized time p(t) = sum_j coeff_j * (t/T)^j,
        and D_scaled (batch, 3, 6) = [p0, T*v0, T^2*a0, pT, T*vT, T^2*aT].
        """
        scale = th.stack([th.ones_like(traj_time), traj_time, traj_time ** 2], dim=-1).unsqueeze(1)  # (batch, 1, 3)
        D_scaled = th.cat([Df * scale, Dp * scale], dim=2)  # (batch, 3, 6)
        coeff = th.einsum('ij,baj->bai', self._L, D_scaled)  # (batch, 3, 6)
        return coeff, D_scaled

    @staticmethod
    def get_position_from_coeff(coeff, s):
        """(batch, N, 3) positions. Independent of T: s^j is dimensionless, T is absorbed in coeff."""
        s_power = th.stack([s ** j for j in range(6)], dim=-1)  # (N, 6)
        return th.einsum('nj,baj->bna', s_power, coeff)

    @staticmethod
    def get_velocity_from_coeff(coeff, s, traj_time):
        """(batch, N, 3) velocities in m/s.  dp/dt = (dq/ds) / T"""
        s_power = th.stack([j * s ** (j - 1) for j in range(1, 6)], dim=-1)  # (N, 5)
        vel = th.einsum('nj,baj->bna', s_power, coeff[:, :, 1:])
        return vel / traj_time.view(-1, 1, 1)

    @staticmethod
    def get_acceleration_from_coeff(coeff, s, traj_time):
        """(batch, N, 3) accelerations in m/s^2.  d2p/dt2 = (d2q/ds2) / T^2"""
        s_power = th.stack([j * (j - 1) * s ** (j - 2) for j in range(2, 6)], dim=-1)  # (N, 4)
        acc = th.einsum('nj,baj->bna', s_power, coeff[:, :, 2:])
        return acc / traj_time.view(-1, 1, 1) ** 2

    def forward(self, state, prediction, traj_time, target, map_id, has_target):
        """state / prediction: (batch, 3, 3) start / end state in world frame, rows [pos; vel; acc].
        traj_time: (batch) predicted duration.  target: (batch, 3).  map_id: (batch) ESDF map index.
        has_target: (batch) bool, false where the frame has no target and `target` is a placeholder.
        Returns the weighted per-component cost dict and the detached tensorboard diagnostics.
        """
        Df = state.permute(0, 2, 1)        # (batch, 3, 3) [px, vx, ax; py, vy, ay; pz, vz, az]
        Dp = prediction.permute(0, 2, 1)

        coeff, D_scaled = self.solve_trajectory(Df, Dp, traj_time)
        pos = self.get_position_from_coeff(coeff, self.eval_s)
        vel = self.get_velocity_from_coeff(coeff, self.eval_s, traj_time)
        acc = self.get_acceleration_from_coeff(coeff, self.eval_s, traj_time)

        smooth_cost, accel_cost, feasible_cost = self.smoothness_loss(D_scaled, traj_time, vel, acc)
        safety_cost, target_safety_cost, dist_esdf = self.safety_loss(pos, map_id, target)
        height_cost, height_err = self.height_loss(pos, Df[:, :, 0], target, has_target)

        raw = {"Smooth": smooth_cost, "Safety": safety_cost,
               "Track": self.tracking_loss(Df, Dp, target), "Guide": self.guidance_loss(Df, Dp),
               "Height": height_cost,
               "Time": traj_time, "Feasible": feasible_cost, "Accel": accel_cost}
        costs = {k: self.weights[k] * v for k, v in raw.items()}
        # target-safety shares the obstacle weight and is reported together with tracking
        costs["Track"] = costs["Track"] + self.weights["Safety"] * target_safety_cost

        traj_info = {'vel_norm': vel.detach().norm(dim=-1), 'dist_esdf': dist_esdf.detach(),
                     'height_err': height_err}
        return costs, traj_info
