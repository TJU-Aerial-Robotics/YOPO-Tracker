"""Per-sample supervision flags, and the fill used where a modality is absent.

Which losses each data stream supervises:

    stream                   detect u,v   detect d   score   trajectory   cost
    real (complete)              y            y        y          y         y
    COCO                         y            n        y          n         n
    real + depth dropout         y            y        y          n         n
    real + RGB dropout           n            n        n          y         y

RGB dropout is applied only to frames whose target is NaN, so the trajectory branch is never
asked to locate a target from depth alone.
"""
import numpy as np

from config.config import cfg

HAS_RGB, HAS_DEPTH, HAS_DIST = 0, 1, 2
N_FLAGS = 3

BLANK_RGB = float(cfg["blank_rgb"])      # RGB is normalized to [0, 1]
BLANK_DEPTH = float(cfg["blank_depth"])  # depth is normalized to [0, 1] by max_dis


def make_flags(rgb=True, depth=True, dist=True) -> np.ndarray:
    """[HAS_RGB, HAS_DEPTH, HAS_DIST] as float32, so the DataLoader batches it to (B, 3)."""
    return np.array([rgb, depth, dist], dtype=np.float32)


def blank_rgb(rgbd):
    """rgbd: (4, H, W), channels 0:3 RGB and 3 depth."""
    rgbd[0:3] = BLANK_RGB
    return rgbd


def blank_depth(rgbd):
    rgbd[3] = BLANK_DEPTH
    return rgbd
