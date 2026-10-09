"""公开检测数据，深度图可选；仅监督检测 uv 和 objectness。"""
import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from config.config import cfg
from policy.modality import BLANK_DEPTH, make_flags
from policy.yopo_dataset import sample_random_state


class PublicRGBDataset(Dataset):
    def __init__(self, mode='train', val_ratio=0.1):
        super(PublicRGBDataset, self).__init__()
        self.height = int(cfg["image_height"])
        self.width = int(cfg["image_width"])
        self.max_dis = 20.0
        self.augment = bool(int(cfg["data_augment"])) and mode == 'train'

        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "../", cfg["public_dataset_path"])
        self.depth_dir = os.path.join(data_dir, "depth")
        self.has_depth = os.path.isdir(self.depth_dir)
        labels = np.loadtxt(os.path.join(data_dir, "labels.csv"), delimiter=',', skiprows=1).astype(np.float32)
        files = [os.path.join(data_dir, "rgb", f"{i:06d}.jpg") for i in range(len(labels))]
        targets = labels[:, 7:10]  # position is 0 and rotation identity, so world == body frame

        # same split as YOPODataset, so a sample never lands in both splits across the two datasets
        files_train, files_val, targets_train, targets_val = train_test_split(
            files, targets, test_size=val_ratio, random_state=0)
        if mode == 'train':
            self.files, self.targets = files_train, targets_train
        elif mode == 'valid':
            self.files, self.targets = files_val, targets_val
        else:
            raise ValueError(f"Invalid mode {mode}. Choose from 'train', 'valid'.")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.1),
        ])
        n_pos = int((~np.isnan(self.targets).any(axis=1)).sum())
        print(f"=============== Public {mode.capitalize()} Data Summary ===============")
        print(f"{'Images':<12} | Count: {len(self.files):<6} |  With target: {n_pos}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        rgb_image = cv2.imread(self.files[item]).astype(np.float32)
        rgb_image = cv2.resize(rgb_image, (self.width, self.height), interpolation=cv2.INTER_NEAREST) / 255.0
        rgb_image = self.transform(rgb_image) if self.augment else torch.from_numpy(rgb_image.transpose(2, 0, 1))
        if self.has_depth:
            depth_path = os.path.join(self.depth_dir, os.path.splitext(os.path.basename(self.files[item]))[0] + ".png")
            depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
            depth_image = cv2.resize(depth_image, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            depth_image = torch.from_numpy((np.minimum(depth_image, self.max_dis) / self.max_dis)[None])
        else:
            depth_image = torch.full((1, self.height, self.width), BLANK_DEPTH, dtype=torch.float32)
        rgbd_image = torch.cat((rgb_image, depth_image), dim=0)

        vel_b, acc_b = sample_random_state()
        random_obs = np.hstack((vel_b, acc_b)).astype(np.float32)

        position = np.zeros(3, dtype=np.float32)
        rot_wb = np.eye(3, dtype=np.float32)
        flags = make_flags(rgb=True, depth=False, dist=False)
        return rgbd_image, position, rot_wb, random_obs, self.targets[item], 0, flags
