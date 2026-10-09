"""
Training Strategy
supervised learning, imitation learning, testing, rollout
"""
import os
import math
import time
import atexit
import numpy as np
import torch
from collections import defaultdict
from torch.nn import functional as F
from rich.progress import Progress
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard.writer import SummaryWriter

from config.config import cfg
from loss.loss_function import YOPOLoss
from loss.detection_loss import DetectionLoss
from policy.yopo_network import YopoNetwork
from policy.yopo_dataset import YOPODataset
from policy.public_dataset import PublicRGBDataset
from policy.modality import HAS_RGB, HAS_DEPTH, HAS_DIST
from policy.model_ema import ModelEMA
from policy.state_transform import StateTransform, state_body2world


class YopoTrainer:
    # Per component: (trained on positive samples?, weight of the other group), applied by weight_loss:
    #   on_positive=True:  main = pos (weight 1), other = ignore + neg
    #   on_positive=False: main = neg (weight 1), other = pos; ignore is in neither group, so it gets 0
    #                      (the anchors next to the target get no guidance)
    # Resulting weight per anchor type (masks from get_detection_mask):
    #   component                                  | pos | ignore | neg
    #   -------------------------------------------|-----|--------|-----
    #   Smooth / Safety / Time / Feasible / Accel  |  1  |  0.2   | 0.2
    #   Track                                      |  1  |  0     | 0
    #   Guide                                      |  0  |  0     | 1
    #   Height                                     |  1  |  1     | 1
    TRAJ_LOSS_MASKS = {
        "Smooth":   (True, 0.2),
        "Safety":   (True, 0.2),
        "Track":    (True, 0.0),
        "Guide":    (False, 0.0),
        "Height":   (True, 1.0),
        "Time":     (True, 0.2),
        "Feasible": (True, 0.2),
        "Accel":    (True, 0.2),
    }

    def __init__(
            self,
            learning_rate=0.001,
            batch_size=32,
            tensorboard_path=None,
            checkpoint_path=None,
            save_on_exit=False,
            lr_final_ratio=0.01,    # cosine annealing end lr = learning_rate * lr_final_ratio
            warmup_epochs=1.0,      # linear lr warmup before the cosine decay; 0 disables it
            ema_decay=0.999,        # upper bound of the weight moving average decay
            ema_tau=1000,           # decay ramp-up speed; lower it when the dataset is small (total steps << tau)
    ):
        self.batch_size = batch_size
        self.max_grad_norm = 0.1
        self.lr_final_ratio = lr_final_ratio
        self.warmup_epochs = warmup_epochs
        public_path = cfg["public_dataset_path"]
        public_dir = os.path.join(os.path.dirname(__file__), "../", public_path) if public_path else ""
        self.public_ratio = float(cfg["public_ratio"]) if os.path.isdir(public_dir) else 0.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.w_traj = float(cfg["w_traj"])
        self.w_detect = float(cfg["w_detect"])
        self.w_score = float(cfg["w_score"])
        if save_on_exit: self._exit_func = atexit.register(self.save_model)
        # logger
        self.progress_log = Progress()
        self.tensorboard_path = self.get_next_log_path(tensorboard_path)
        self.tensorboard_log = SummaryWriter(log_dir=self.tensorboard_path)
        # params
        self.traj_num = cfg['traj_num']
        self.state_transform = StateTransform()

        # network
        print("Loading network...")
        self.policy = YopoNetwork()
        self.policy = self.policy.to(self.device)
        try:
            state_dict = torch.load(checkpoint_path, weights_only=True)
            self.policy.load_state_dict(state_dict)
            print("Checkpoint ", checkpoint_path, " loaded successfully")
        except FileNotFoundError:
            print("Training from scratch")

        # EMA weights, used for validation and checkpoints
        self.ema = ModelEMA(self.policy, decay=ema_decay, tau=ema_tau)

        # loss
        self.yopo_loss = YOPOLoss()
        self.detection_loss = DetectionLoss()

        # optimizer
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=learning_rate, fused=True)
        print("Network Loaded! Loading Dataset...")

        # dataset: the validation set stays purely real so its metrics remain comparable
        real_train = YOPODataset(mode='train')
        self.train_dataloader = self.build_train_loader(real_train)
        val_set = YOPODataset(mode='valid')
        self.val_dataloader = None if len(val_set) == 0 else DataLoader(
            val_set, batch_size=self.batch_size, shuffle=False,
            num_workers=4, pin_memory=True, persistent_workers=True)
        self.public_val_dataloader = None
        if self.public_ratio > 0:
            self.public_val_dataloader = DataLoader(PublicRGBDataset(mode='valid'), batch_size=self.batch_size,
                                                    shuffle=False, num_workers=4, pin_memory=True,
                                                    persistent_workers=True)
        print("Dataset Loaded!")

    def build_train_loader(self, real_train):
        """Real and public samples in one batch, mixed to public_ratio. One epoch keeps the length
        of the real set, so metrics stay comparable across runs."""
        if self.public_ratio <= 0:
            return DataLoader(real_train, batch_size=self.batch_size, shuffle=True, drop_last=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
        public_train = PublicRGBDataset(mode='train')
        weights = ([(1.0 - self.public_ratio) / len(real_train)] * len(real_train) +
                   [self.public_ratio / len(public_train)] * len(public_train))
        sampler = WeightedRandomSampler(weights, num_samples=len(real_train), replacement=True)
        return DataLoader(ConcatDataset([real_train, public_train]), batch_size=self.batch_size,
                          sampler=sampler, drop_last=True, num_workers=4, pin_memory=True,
                          persistent_workers=True)

    def build_scheduler(self, epoch):
        """Linear warmup over warmup_epochs, then cosine annealing from learning_rate to
        learning_rate * lr_final_ratio. Stepped per batch."""
        steps_per_epoch = len(self.train_dataloader)
        warmup_steps = int(self.warmup_epochs * steps_per_epoch)
        total_steps = max(epoch * steps_per_epoch - warmup_steps, 1)

        def lr_lambda(step):
            if warmup_steps and step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = min((step - warmup_steps) / total_steps, 1.0)
            return self.lr_final_ratio + (1 - self.lr_final_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train(self, epoch, save_interval=None):
        self.scheduler = self.build_scheduler(epoch)
        with self.progress_log:
            total_progress = self.progress_log.add_task("Training", total=epoch)
            for self.epoch_i in range(epoch):
                self.policy.train()
                self.train_one_epoch(self.epoch_i, total_progress)
                self.policy.eval()
                self.eval_one_epoch(self.epoch_i)
                if save_interval is not None and (self.epoch_i + 1) % save_interval == 0:
                    self.progress_log.console.log("Saving model...")
                    torch.save(self.ema.ema.state_dict(), f"{self.tensorboard_path}/epoch{self.epoch_i + 1}.pth")
            self.progress_log.console.log("Train YOPO Finish!")
            self.progress_log.remove_task(total_progress)

    def train_one_epoch(self, epoch: int, total_progress):
        one_epoch_progress = self.progress_log.add_task(f"Epoch: {epoch}", total=len(self.train_dataloader))
        inspect_interval = max(1, len(self.train_dataloader) // 16)
        running, start_time = defaultdict(list), time.time()
        for step, (depth, pos, rot, obs_b, target_b, map_id, flags) in enumerate(self.train_dataloader):  # obs: body frame
            if depth.shape[0] != self.batch_size:  continue  # batch size == number of env

            self.optimizer.zero_grad()
            loss, metrics = self.forward_and_compute_loss(depth, pos, rot, obs_b, target_b, map_id, flags)

            # Optimize the policy
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.ema.update(self.policy)

            self.accumulate(running, metrics)

            if step % inspect_interval == inspect_interval - 1:
                batch_fps = inspect_interval / (time.time() - start_time)
                gstep = epoch * len(self.train_dataloader) + step
                means = {k: float(np.mean(v)) for k, v in running.items()}
                self.progress_log.console.log(f"Epoch: {epoch}, Traj Loss: {means['TrajLoss']:.3g}, "
                                              f"Cost Loss: {means['CostLoss']:.3g} "
                                              f"Batch FPS: {batch_fps:.3g}")
                for k, mv in means.items():
                    self.tensorboard_log.add_scalar(k if "/" in k else f"Train/{k}", mv, gstep)
                self.tensorboard_log.add_scalar("Detail/LearningRate", self.optimizer.param_groups[0]['lr'], gstep)
                running, start_time = defaultdict(list), time.time()

            self.progress_log.update(one_epoch_progress, advance=1)
            self.progress_log.update(total_progress, advance=1 / len(self.train_dataloader))

        self.progress_log.remove_task(one_epoch_progress)

    @torch.inference_mode()
    def eval_one_epoch(self, epoch: int):
        if self.val_dataloader is None:
            self.eval_public(epoch)
            return
        one_epoch_progress = self.progress_log.add_task(f"Eval: {epoch}", total=len(self.val_dataloader))
        running = defaultdict(list)
        for step, (depth, pos, rot, obs_b, target_b, map_id, flags) in enumerate(self.val_dataloader):  # obs: body frame
            if depth.shape[0] != self.batch_size:  continue  # batch size == num of env

            # validate with the EMA weights
            _, metrics = self.forward_and_compute_loss(depth, pos, rot, obs_b, target_b, map_id, flags,
                                                       policy=self.ema.ema)
            self.accumulate(running, metrics)
            self.progress_log.update(one_epoch_progress, advance=1)

        def mean(k):  # empty-safe: a val set without a full batch gives nan instead of KeyError
            return float(np.mean(running[k])) if running[k] else float("nan")

        self.progress_log.console.log(f"Eval: {epoch}, Traj Loss: {mean('TrajLoss'):.3g}, Cost Loss: {mean('CostLoss'):.3g} ")
        for k in ("TrajLoss", "DetectLoss", "CostLoss", "ScoreLoss"):
            self.tensorboard_log.add_scalar(f"Eval/{k}", mean(k), epoch)
        self.progress_log.remove_task(one_epoch_progress)
        self.eval_public(epoch)

    @torch.inference_mode()
    def eval_public(self, epoch: int):
        """Detection / objectness on held-out public images: whether the extra data was learnt at all."""
        if self.public_val_dataloader is None:
            return
        running = defaultdict(list)
        for depth, pos, rot, obs_b, target_b, map_id, flags in self.public_val_dataloader:
            if depth.shape[0] != self.batch_size:  continue
            _, metrics = self.forward_and_compute_loss(depth, pos, rot, obs_b, target_b, map_id, flags,
                                                       policy=self.ema.ema)
            self.accumulate(running, metrics)
        for k in ("DetectLoss", "ScoreLoss"):
            if running[k]:
                self.tensorboard_log.add_scalar(f"Eval/Public{k}", float(np.mean(running[k])), epoch)

    @staticmethod
    def accumulate(running, metrics):
        """Append this step's metrics to the running buffers, stacked so the dict costs one device sync."""
        keys = list(metrics)
        values = torch.stack([metrics[k] for k in keys]).detach().cpu().tolist()
        for k, v in zip(keys, values):
            running[k].append(v)

    def forward_and_compute_loss(self, depth, pos, rot, obs_b, target_b, map_id, flags, policy=None):
        """Returns the scalar loss to backprop, and a dict of detached-able tensorboard metrics.
        flags: (B, 3) supervision availability per sample, see policy/modality.py."""
        policy = self.policy if policy is None else policy
        depth, pos, rot, obs_b, target_b, map_id, flags = [
            x.to(self.device) for x in [depth, pos, rot, obs_b, target_b, map_id, flags]]
        has_rgb, has_depth, has_dist = (flags[:, i].bool() for i in (HAS_RGB, HAS_DEPTH, HAS_DIST))

        pos_mask, neg_mask, ignore_mask = self.get_detection_mask(target_b)
        has_target = ~torch.isnan(target_b).any(dim=1)  # recorded before the nan fill below
        target_b = torch.nan_to_num(target_b, nan=0.0)

        # 1. pre-process
        target_w, start_vel_w, start_acc_w = state_body2world(pos, rot, target_b, obs_b[:, 0:3], obs_b[:, 3:6])
        start_state_w = torch.stack([pos, start_vel_w, start_acc_w], dim=1)

        # 2. forward propagation
        endstate, target_pred, cost_pred, score_pred, time_pred = policy.inference(depth, obs_b)

        # 3. post-process [B, V, H, 9] -> [B*V*H, 9]
        BN = self.batch_size * self.traj_num
        endstate_flat = endstate.permute(0, 2, 3, 1).reshape(BN, 9)
        target_flat = target_pred.permute(0, 2, 3, 1).reshape(BN, 3)
        cost_flat = cost_pred.reshape(BN)
        score_flat = score_pred.reshape(BN)
        time_flat = time_pred.reshape(BN)

        # expand per-image quantities to per-trajectory: (B, ...) -> (B*V*H, ...)
        pos_expanded = pos.repeat_interleave(self.traj_num, dim=0)
        rot_expanded = rot.repeat_interleave(self.traj_num, dim=0)
        start_state_w = start_state_w.repeat_interleave(self.traj_num, dim=0)
        target_w = target_w.repeat_interleave(self.traj_num, dim=0)

        end_pos_w, end_vel_w, end_acc_w = state_body2world(
            pos_expanded, rot_expanded, endstate_flat[:, 0:3], endstate_flat[:, 3:6], endstate_flat[:, 6:9])
        end_state_w = torch.stack([end_pos_w, end_vel_w, end_acc_w], dim=1)  # (B*V*H, 3, 3)

        # per-anchor view of the flags
        rgb_anchor = has_rgb.repeat_interleave(self.traj_num)
        dist_anchor = has_dist.repeat_interleave(self.traj_num).float()
        depth_anchor = has_depth.repeat_interleave(self.traj_num)
        target_anchor = has_target.repeat_interleave(self.traj_num)

        # 4. trajectory costs, on the samples that carry depth and a map. Sliced rather than masked:
        # the others have a placeholder map_id, and 0 * NaN would poison the gradient.
        traj_loss = depth.new_zeros(())
        traj_losses, traj_metrics = {}, {}
        cost_loss = depth.new_zeros(())
        if has_depth.any():
            costs, traj_info = self.yopo_loss(start_state_w[depth_anchor], end_state_w[depth_anchor],
                                              time_flat[depth_anchor], target_w[depth_anchor],
                                              map_id[has_depth], target_anchor[depth_anchor])
            pos_traj, neg_traj = pos_mask[depth_anchor], neg_mask[depth_anchor]
            for name, cost in costs.items():
                on_positive, neg_weight = self.TRAJ_LOSS_MASKS[name]
                main, other = (pos_traj, ~pos_traj) if on_positive else (neg_traj, pos_traj)
                traj_losses[name] = self.weight_loss(cost, main, other, neg_weight).mean()
            traj_loss = sum(traj_losses.values())
            # track & guide are excluded: they do not apply to all predictions
            cost_label = (costs["Smooth"] + costs["Accel"] + costs["Safety"]).detach()
            cost_loss = F.smooth_l1_loss(cost_flat[depth_anchor], cost_label)
            traj_metrics = self.best_traj_metrics(pos_traj, cost_flat[depth_anchor], time_flat[depth_anchor],
                                                  end_pos_w[depth_anchor], target_w[depth_anchor], traj_info)

        # 5. detection and objectness heads, on the samples that carry RGB
        det_pos, det_neg = pos_mask & rgb_anchor, neg_mask & rgb_anchor
        detection_loss = self.detection_loss(target_flat, target_b, det_pos, dist_mask=dist_anchor)
        score_loss = F.binary_cross_entropy(score_flat, pos_mask.float(), reduction='none')
        score_loss = self.weight_loss(score_loss, det_pos, det_neg, neg_weight=0.5).mean()

        loss = (self.w_traj * traj_loss +
                self.w_detect * detection_loss +
                self.w_score * (cost_loss + score_loss))

        metrics = {"TrajLoss": self.w_traj * traj_loss,
                   "DetectLoss": self.w_detect * detection_loss,
                   "CostLoss": self.w_score * cost_loss,
                   "ScoreLoss": self.w_score * score_loss}
        metrics.update({f"Detail/{k}Loss": self.w_traj * v for k, v in traj_losses.items()})
        metrics.update(traj_metrics)
        return loss, metrics

    @torch.no_grad()
    def best_traj_metrics(self, pos_mask, cost_flat, time_flat, end_pos_w, target_w, traj_info):
        """Metrics of the lowest-predicted-cost positive anchor per image; {} if the batch has none."""
        N = self.traj_num
        B = cost_flat.shape[0] // N
        pos = pos_mask.view(B, N)
        has_pos = pos.any(dim=1)
        if not has_pos.any():
            return {}

        # lowest predicted cost among the positive anchors (non-positive filled with inf to exclude)
        cost = cost_flat.detach().view(B, N).masked_fill(~pos, float('inf'))
        best = cost.argmin(dim=1)  # [B]
        idx = (torch.arange(B, device=self.device) * N + best)[has_pos]  # flat index, images with a positive only

        vel_norm = traj_info['vel_norm'][idx]    # [K, eval_points]
        dist_esdf = traj_info['dist_esdf'][idx]  # [K, eval_points]
        return {
            'BestTraj/MaxVel': vel_norm.max(dim=1).values.mean(),          # peak speed, averaged over the batch
            'BestTraj/MeanVel': vel_norm.mean(),
            'BestTraj/MinObstacleDist': dist_esdf.min(dim=1).values.mean(),  # minimum ESDF distance per trajectory (m)
            'BestTraj/Time': time_flat.detach()[idx].mean(),
            'BestTraj/TargetDist': (end_pos_w.detach()[idx] - target_w.detach()[idx]).norm(dim=-1).mean(),
        }

    @staticmethod
    def get_detection_mask(target_b):
        """
            Anchor samples by target centre and distance, shared by detection and trajectory (docs/pos_neg_samples.png).
            gap: pixel distance from the projected target to a cell; radius r = f * R / depth, so near targets cover more cells.
            pos: gap <= r_pos and reachable by the anchor; ignore: gap <= r_ign; neg: the rest and frames without a target.
            target_b: (B, 3) body frame, NaN when no target.   :return flat (B*V*H) pos, neg, ignore masks
        """
        V, H, s, f = cfg["vertical_num"], cfg["horizon_num"], cfg["downsample"], cfg["camera_f"]
        x, y, z = target_b.unbind(dim=1)
        valid = ~torch.isnan(target_b).any(dim=1) & (x > 0.1)
        depth = torch.where(valid, x, torch.ones_like(x))  # placeholder keeps NaN and 1/0 out of the invalid frames
        u = cfg["camera_cx"] - f * torch.nan_to_num(y) / depth  # same projection as DetectionLoss
        v = cfg["camera_cy"] - f * torch.nan_to_num(z) / depth
        r_pos = f * cfg["pos_radius"] / depth
        r_ign = torch.clamp(f * cfg["ignore_radius"] / depth, min=cfg["ignore_min_px"])

        # offsets from the cell centres: du [B, 1, H], dv [B, V, 1]
        du = (u[:, None] - (torch.arange(H, device=u.device) + 0.5) * s).abs()[:, None, :]
        dv = (v[:, None] - (torch.arange(V, device=v.device) + 0.5) * s).abs()[:, :, None]
        gap = torch.maximum(du, dv).sub(s / 2).clamp(min=0)  # [B, V, H]
        # half anchor fov in pixels, the coordinate bias between the lattice angles and the pinhole cells is ignored
        reach = ((du <= s * cfg["horizon_anchor_fov"] / (2 * cfg["horizon_camera_fov"] / H)) &
                 (dv <= s * cfg["vertical_anchor_fov"] / (2 * cfg["vertical_camera_fov"] / V)))

        valid = valid[:, None, None]
        pos_mask = valid & reach & (gap <= r_pos[:, None, None])
        ignore_mask = valid & ~pos_mask & (gap <= r_ign[:, None, None])
        neg_mask = ~(pos_mask | ignore_mask)
        return pos_mask.reshape(-1), neg_mask.reshape(-1), ignore_mask.reshape(-1)

    @staticmethod
    def weight_loss(costs, pos_mask, neg_mask, neg_weight=0.0):
        factor = torch.zeros_like(costs, dtype=torch.float32)
        factor[pos_mask] = 1.0
        factor[neg_mask] = neg_weight
        return costs * factor

    def save_model(self):
        if hasattr(self, "epoch_i"):
            self.progress_log.console.log("Saving model...")
            torch.save(self.ema.ema.state_dict(), f"{self.tensorboard_path}/epoch{self.epoch_i + 1}.pth")
            atexit.unregister(self._exit_func)

    def get_next_log_path(self, base_path):
        nums = [int(name.split("_")[1])
                for name in os.listdir(base_path)
                if os.path.isdir(os.path.join(base_path, name)) and name.startswith("YOPO_") and name.split("_")[1].isdigit()]
        next_n = max(nums, default=-1) + 1
        next_path = os.path.join(base_path, f"YOPO_{next_n}")
        os.makedirs(next_path, exist_ok=False)
        print("record tensorboard log to ", next_path)
        return next_path
