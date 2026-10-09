import os, sys
import cv2
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.config import cfg
from policy.modality import blank_depth, blank_rgb, make_flags
from policy.scene_index import scene_dirs_for

# state-sampling constants, shared with PublicRGBDataset through sample_random_state()
VEL_MAX = cfg["vel_max_train"]
ACC_MAX = cfg["acc_max_train"]
V_MEAN = np.array(cfg["v_mean_unit"], dtype=float)
V_STD = np.array(cfg["v_std_unit"], dtype=float)
A_MEAN = np.array(cfg["a_mean_unit"], dtype=float)
A_STD = np.array(cfg["a_std_unit"], dtype=float)
VX_LOGNORM_MEAN = np.log(1 - V_MEAN[0])
VX_LOGNORM_SIGMA = np.log(V_STD[0])


def sample_random_state():
    """Random body-frame (vel, acc). x-direction: log-normal, y/z: normal."""
    while True:
        vel = VEL_MAX * (V_MEAN + V_STD * np.random.randn(3))
        right_skewed_vx = -1
        while right_skewed_vx < 0:
            right_skewed_vx = VEL_MAX * np.random.lognormal(mean=VX_LOGNORM_MEAN, sigma=VX_LOGNORM_SIGMA, size=None)
            right_skewed_vx = -right_skewed_vx + 1.2 * VEL_MAX  # * 1.2 to ensure v_max can be sampled
        vel[0] = right_skewed_vx
        if np.linalg.norm(vel) < 1.2 * VEL_MAX:  # avoid outliers
            break

    while True:
        acc = ACC_MAX * (A_MEAN + A_STD * np.random.randn(3))
        if np.linalg.norm(acc) < 1.2 * ACC_MAX:  # avoid outliers
            break
    return vel, acc


class YOPODataset(Dataset):
    def __init__(self, mode='train'):
        super(YOPODataset, self).__init__()
        # image params
        self.height = int(cfg["image_height"])
        self.width = int(cfg["image_width"])
        self.augment = bool(int(cfg["data_augment"])) and mode == 'train'
        self.max_dis = 20.0  # depth truncation distance (m), must match the test-time value in test_yopo_ros.py
        # ramdom state: x-direction: log-normal distribution, yz-direction: normal distribution
        self.vel_max, self.acc_max = VEL_MAX, ACC_MAX
        self.v_mean, self.v_std = V_MEAN, V_STD
        self.a_mean, self.a_std = A_MEAN, A_STD
        self.vx_lognorm_mean, self.vx_logmorm_sigma = VX_LOGNORM_MEAN, VX_LOGNORM_SIGMA
        # modality dropout, train only; RGB is only dropped on frames with no target
        self.p_drop_rgb = float(cfg["modality_dropout_rgb"]) if mode == 'train' else 0.0
        self.p_drop_depth = float(cfg["modality_dropout_depth"]) if mode == 'train' else 0.0
        if mode == 'train': self.print_data()

        # data augmentation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.1),
            # transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
        ])

        # dataset
        self.rgb_img_list, self.depth_img_list, self.map_idx, = [], [], []
        self.positions, self.quaternions, self.targets = np.empty((0, 3), dtype=np.float32), np.empty((0, 4), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

        # each sub-dataset is a scene-{i} folder holding rgb/ depth/ labels.csv environment.ply;
        # train and valid come from separate roots, so no split here
        datafolders = scene_dirs_for(mode)
        print(f"Datafolders ({mode}):" if datafolders else f"No {mode} scenes found")
        for _, folder in datafolders:
            print("    ", folder)

        print("Loading", mode, "dataset")
        for map_id, datafolder in datafolders:

            rgb_file_names = [os.path.join(datafolder, "rgb", filename)
                              for filename in os.listdir(os.path.join(datafolder, "rgb"))
                              if os.path.splitext(filename)[1] == '.jpg']
            rgb_file_names.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

            depth_file_names = [os.path.join(datafolder, "depth", filename)
                                for filename in os.listdir(os.path.join(datafolder, "depth"))
                                if os.path.splitext(filename)[1] == '.png']
            depth_file_names.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

            states = np.loadtxt(os.path.join(datafolder, "labels.csv"), delimiter=',', skiprows=1).astype(np.float32)
            positions, quaternions, targets_w = states[:, 0:3], states[:, 3:7], states[:, 7:10]
            # targets are stored in the world frame (NWU) while the network and loss use the body frame:
            # p_b = R_wb^T * (p_w - p_body_w). Invalid targets are nan and stay nan through the transform.
            rot_wb = R.from_quat(np.stack([quaternions[:, 1], quaternions[:, 2], quaternions[:, 3], quaternions[:, 0]], axis=1)).as_matrix()
            targets = np.einsum('nji,nj->ni', rot_wb, targets_w - positions).astype(np.float32)

            self.rgb_img_list.extend(rgb_file_names)
            self.depth_img_list.extend(depth_file_names)
            self.positions = np.vstack((self.positions, positions.astype(np.float32)))
            self.quaternions = np.vstack((self.quaternions, quaternions.astype(np.float32)))
            self.targets = np.vstack((self.targets, targets.astype(np.float32)))
            self.map_idx.extend([map_id] * len(rgb_file_names))

        print(f"=============== {mode.capitalize()} Data Summary ===============")
        print(f"{'Images'      :<12} | Count: {len(self.rgb_img_list):<3} |  Shape: {self.width}, {self.height}")
        print(f"{'Targets'     :<12} | Count: {len(self.targets):<3} |  Shape: {self.targets.shape[1]}")
        print(f"{'Positions'   :<12} | Count: {self.positions.shape[0]:<3} |  Shape: {self.positions.shape[1]}")
        print(f"{'Quaternions' :<12} | Count: {self.quaternions.shape[0]:<3} |  Shape: {self.quaternions.shape[1]}")
        print("==================================================")

    def __len__(self):
        return len(self.depth_img_list)

    def __getitem__(self, item):
        # 1. read RGB-D image \\ NOTE: The depth images are stored as uint16 in millimeters, and normalized from 0–20m to 0–1 here.
        rgb_image = cv2.imread(self.rgb_img_list[item]).astype(np.float32)
        rgb_image = cv2.resize(rgb_image, (self.width, self.height), interpolation=cv2.INTER_NEAREST) / 255.0
        depth_image = cv2.imread(self.depth_img_list[item], -1).astype(np.float32) / 1000.0  # mm -> m
        depth_image = cv2.resize(depth_image, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        depth_image = np.expand_dims(np.minimum(depth_image, self.max_dis) / self.max_dis, axis=0)

        rgb_image = self.transform(rgb_image) if self.augment else torch.from_numpy(rgb_image.transpose(2, 0, 1))
        depth_image = torch.from_numpy(depth_image)
        rgbd_image = torch.cat((rgb_image, depth_image), dim=0)

        # 2. get random state
        vel_b, acc_b = self._get_random_state()
        random_obs = np.hstack((vel_b, acc_b)).astype(np.float32)
        if not np.isnan(self.targets[item]).any():
            dist = np.linalg.norm(self.targets[item])
            scale = np.minimum(0.5 * (dist / cfg["track_dist"]), 1.0).astype(np.float32)
            random_obs = random_obs * scale

        q_wxyz = self.quaternions[item, :]  # q: wxyz
        R_WB = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
        rot_wb = R_WB.as_matrix().astype(np.float32)  # transform to rot_matrix in numpy is faster than using quat in pytorch

        flags = self._modality_dropout(rgbd_image, np.isnan(self.targets[item]).any())

        # vel & acc & target are in body frame, NWU, and no-normalization
        return rgbd_image, self.positions[item], rot_wb, random_obs, self.targets[item], self.map_idx[item], flags

    def _modality_dropout(self, rgbd_image, target_is_nan):
        """Blank at most one channel group in place and return the matching supervision flags."""
        if target_is_nan and np.random.rand() < self.p_drop_rgb:
            blank_rgb(rgbd_image)
            return make_flags(rgb=False, depth=True, dist=False)
        if np.random.rand() < self.p_drop_depth:
            blank_depth(rgbd_image)
            return make_flags(rgb=True, depth=False, dist=True)
        return make_flags()

    def _get_random_state(self):
        return sample_random_state()

    def print_data(self):
        import scipy.stats as stats
        # Vx 5% ~ 95% interval
        p5 = self.vel_max * np.exp(stats.norm.ppf(0.05, loc=self.vx_lognorm_mean, scale=self.vx_logmorm_sigma))
        p95 = self.vel_max * np.exp(stats.norm.ppf(0.95, loc=self.vx_lognorm_mean, scale=self.vx_logmorm_sigma))

        v_lower = self.vel_max * (self.v_mean - 2 * self.v_std)
        v_upper = self.vel_max * (self.v_mean + 2 * self.v_std)
        v_lower[0] = max(-p95 + 1.2 * self.vel_max, 0)
        v_upper[0] = -p5 + 1.2 * self.vel_max

        a_lower = self.acc_max * (self.a_mean - 2 * self.a_std)
        a_upper = self.acc_max * (self.a_mean + 2 * self.a_std)

        print("----------------- Sampling State --------------------")
        print("| X-Y-Z | Vel 95% Range(m/s)  | Acc 95% Range(m/s2) |")
        print("|-------|---------------------|---------------------|")
        for i in range(3):
            print(f"|  {i:^4} | {v_lower[i]:^9.1f}~{v_upper[i]:^9.1f} |"
                  f" {a_lower[i]:^9.1f}~{a_upper[i]:^9.1f} |")
        print("-----------------------------------------------------")

    def plot_sample_distribution(self):
        import matplotlib.pyplot as plt
        # ===== sampling =====
        N = 10000
        states = np.array([self._get_random_state() for _ in range(N)])
        vels = np.stack([s[0] for s in states])
        accs = np.stack([s[1] for s in states])

        x, y, z = self.targets[:, 0], self.targets[:, 1], self.targets[:, 2]
        yaw = np.degrees(np.arctan2(y, x))  # azimuth [-180, 180]
        pitch = np.degrees(np.arctan2(z, np.sqrt(x ** 2 + y ** 2)))  # elevation [-90, 90]
        dist = np.linalg.norm(self.targets, axis=1)

        fig, axs = plt.subplots(3, 3, figsize=(15, 10))

        # goal bearing distribution
        axs[0, 0].hist(yaw, bins=120, alpha=0.6, label="Yaw")
        axs[0, 0].hist(pitch, bins=120, alpha=0.6, label="Pitch")
        axs[0, 0].set_title("Goal Yaw & Pitch Distribution")
        axs[0, 0].set_xlabel("Angle (deg)")
        axs[0, 0].set_xlim([-45, 45])
        axs[0, 0].grid(True)
        axs[0, 0].legend()

        axs[0, 1].hist(dist, bins=120)
        axs[0, 1].set_title("Goal Dist Distribution")
        axs[0, 1].set_xlabel("Dist (m)")
        axs[0, 1].set_xlim([0, 15])
        axs[0, 1].grid(True)

        # goal projected onto the image (body rotation not considered)
        axs[0, 2].scatter(yaw, pitch, s=2, alpha=0.3)
        axs[0, 2].set_title("Goal Distribution in Image")
        axs[0, 2].set_xlabel("Yaw (deg)")
        axs[0, 2].set_ylabel("Pitch (deg)")
        axs[0, 2].set_xlim([-45, 45])
        axs[0, 2].set_ylim([-30, 30])
        axs[0, 2].grid(True)

        # velocity distribution
        for i, name in enumerate(['Vx', 'Vy', 'Vz']):
            axs[1, i].hist(vels[:, i], bins=100)
            axs[1, i].set_title(f"Velocity {name}")
            axs[1, i].grid(True)

        # acceleration distribution
        for i, name in enumerate(['Ax', 'Ay', 'Az']):
            axs[2, i].hist(accs[:, i], bins=100)
            axs[2, i].set_title(f"Acceleration {name}")
            axs[2, i].grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # plot the random sample
    dataset = YOPODataset()
    dataset.plot_sample_distribution()

    # select the best num_workers
    max_workers = os.cpu_count()
    print(f"\n✅ cpu_count = {max_workers}")

    results = []
    for nw in range(0, max_workers + 1):
        data_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=nw)
        start = time.time()
        for i, _ in enumerate(data_loader):
            if i > 50:  # only benchmark the first 50 batches
                break
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start
        results.append((nw, elapsed))
        print(f"num_workers={nw}: {elapsed:.3f}s")

    best = min(results, key=lambda x: x[1])
    print(f"\nBest num_workers = {best[0]}, mean time = {best[1]:.3f}s")
