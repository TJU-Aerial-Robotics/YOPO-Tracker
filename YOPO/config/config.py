import os
import numpy as np
from ruamel.yaml import YAML


# Global Configuration Management
class Config:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self._data = YAML().load(open(os.path.join(base_dir, "traj_opt.yaml"), 'r'))
        self._data["train"] = True
        self._data["goal_length"] = 2.0 * self._data['radio_range']
        self._data["sgm_time"] = 2 * self._data["radio_range"] / self._data["vel_max_train"]
        self._data["traj_num"] = self._data['horizon_num'] * self._data['vertical_num'] * self._data["radio_num"]
        self._data["camera_cx"] = self._data["image_width"] / 2.0
        self._data["camera_cy"] = self._data["image_height"] / 2.0
        fov_radians = np.radians(self._data["horizon_camera_fov"])
        self._data["camera_f"] = self._data["image_width"] / (2 * np.tan(fov_radians / 2))
        self._data["downsample"] = int(self._data["image_width"] / self._data["horizon_num"])  # must be consistent with the network

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value


cfg = Config()
