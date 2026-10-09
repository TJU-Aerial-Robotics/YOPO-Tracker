import torch
from scipy.spatial.transform import Rotation as R
from config.config import cfg


class LatticeParam:
    def __init__(self):
        ratio = 1.0 if cfg["train"] else cfg["velocity"] / cfg["vel_max_train"]
        self.vel_max = ratio * cfg["vel_max_train"]
        self.acc_max = ratio * ratio * cfg["acc_max_train"]
        self.segment_time = cfg["sgm_time"] / ratio
        self.horizon_num = cfg["horizon_num"]
        self.vertical_num = cfg["vertical_num"]
        self.radio_num = cfg["radio_num"]
        self.traj_num = cfg["traj_num"]
        self.horizon_fov = cfg["horizon_camera_fov"]
        self.vertical_fov = cfg["vertical_camera_fov"]
        self.horizon_anchor_fov = cfg["horizon_anchor_fov"]
        self.vertical_anchor_fov = cfg["vertical_anchor_fov"]
        self.radio_range = cfg["radio_range"]

        print("---------- Param --------")
        print(f"| {'max speed':<12} = {round(self.vel_max, 1):>6} |")
        print(f"| {'max accel':<12} = {round(self.acc_max, 1):>6} |")
        print(f"| {'traj time':<12} = {round(self.segment_time, 1):>6} |")
        print(f"| {'max radio':<12} = {round(2 * self.radio_range, 1):>6} |")
        print("-------------------------")


class LatticePrimitive(LatticeParam):
    """
    Lattice index layout as seen in the image (row-major, bottom-right origin;
    the default horizon_num=5 x vertical_num=3 = 15 primitives):
                   +----+----+----+----+----+
                   | 14 | 13 | 12 | 11 | 10 |
                   +----+----+----+----+----+
                   |  9 |  8 |  7 |  6 |  5 |
                   +----+----+----+----+----+
                   |  4 |  3 |  2 |  1 |  0 |
                   +----+----+----+----+----+
    """
    _instance = None

    def __init__(self):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        direction_diff = (self.horizon_fov / 180.0 * torch.pi) / self.horizon_num
        altitude_diff = (self.vertical_fov / 180.0 * torch.pi) / self.vertical_num
        radio_diff = self.radio_range / self.radio_num

        lattice_pos_list = []
        lattice_angle_list = []
        lattice_Rbp_list = []

        # Primitives: Bottom to Top, Right to Left, Shot to Long(un-used)
        for h in range(0, self.radio_num):
            for i in range(0, self.vertical_num):
                for j in range(0, self.horizon_num):
                    search_radio = (h + 1) * radio_diff
                    alpha = torch.tensor(-direction_diff * (self.horizon_num - 1) / 2 + j * direction_diff)
                    beta = torch.tensor(-altitude_diff * (self.vertical_num - 1) / 2 + i * altitude_diff)

                    pos_node = torch.tensor([torch.cos(beta) * torch.cos(alpha) * search_radio,
                                             torch.cos(beta) * torch.sin(alpha) * search_radio,
                                             torch.sin(beta) * search_radio])

                    lattice_pos_list.append(pos_node)
                    lattice_angle_list.append(torch.tensor([alpha, beta]))
                    Rotation = R.from_euler('ZYX', [alpha, -beta, 0.0], degrees=False)  # inner rotation: yaw-pitch-roll
                    lattice_Rbp_list.append(torch.tensor(Rotation.as_matrix()))

        self.lattice_pos_node = torch.stack(lattice_pos_list).to(dtype=torch.float32, device=device)  # shape: [N, 3]
        self.lattice_angle_node = torch.stack(lattice_angle_list).to(dtype=torch.float32, device=device)  # shape: [N, 2]
        self.lattice_Rbp_node = torch.stack(lattice_Rbp_list).to(dtype=torch.float32, device=device)  # shape: [N, 3, 3]

        self.yaw_diff = 0.5 * self.horizon_anchor_fov / 180.0 * torch.pi
        self.pitch_diff = 0.5 * self.vertical_anchor_fov / 180.0 * torch.pi

    def getAngleLattice(self, id=None):
        if id is not None:
            return self.lattice_angle_node[id, 0], self.lattice_angle_node[id, 1]  # yaw, pitch
        else:
            return self.lattice_angle_node[:, 0], self.lattice_angle_node[:, 1]  # yaw, pitch

    def getRotation(self, id=None):
        if id is not None:
            return self.lattice_Rbp_node[id]
        else:
            return self.lattice_Rbp_node

    def convert_ImageGrid_LatticeID(self, id):
        return self.traj_num - id - 1

    @classmethod
    def get_instance(cls):
        if cls._instance is None: cls._instance = cls()
        return cls._instance
