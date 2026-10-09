import os

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import std_msgs.msg
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Point
import message_filters
from threading import Lock
from collections import deque
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray

import cv2
import time
import torch
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R

# torch.set_num_threads(1)   # GPU 推理, CPU 侧算子无需并行
# cv2.setNumThreads(1)       # resize/cvtColor 在 480x270 上单线程更快且不抢核

from config.config import cfg
from quadrotor_msgs.msg import PositionCommand
from policy.yopo_network import YopoNetwork
from policy.poly_solver import Poly5Solver, Polys5Solver, wrap_to_pi, calculate_yaw
from policy.primitive import LatticePrimitive
from policy.state_transform import StateTransform
from policy.target_ekf import EKF

try:
    from torch2trt import TRTModule
except ImportError:
    print("tensorrt not found.")


class YopoNet(Node):
    def __init__(self, config, weight):
        super().__init__('yopo_net')
        # load params
        cfg["train"] = False
        self.height = cfg['image_height']
        self.width = cfg['image_width']
        self.min_dis, self.max_dis = 0.04, 20.0  # metres
        self.simulation = config['simulation']
        self.conf_thresh = config.get('conf_thresh')
        self.ekf_thresh = config.get('ekf_thresh')
        self.plan_from_reference = config['plan_from_reference']
        self.use_trt = config['use_tensorrt']
        self.verbose = config['verbose']
        self.visualize = config['visualize']
        self.Rotation_bc = R.from_euler('ZYX', [0, config['pitch_angle_deg'], 0], degrees=True).as_matrix()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # constants
        self.ctrl_dt = 0.02
        self.ekf_dt = 1 / config['image_fps']
        # If the target is still not re-found after flying near self.target, plan one braking trajectory to a
        # stop, then spin the yaw in place at a constant rate to search
        self.search_dist = 3.0      # how close to self.target before switching to search
        self.search_yaw_rate = 3.14  # search yaw rate (rad/s), a full turn takes about 2 s
        self.search_yaw_acc = 3.0   # slope of the yaw-rate ramp (rad/s^2)
        self.brake_acc = 4.5        # peak deceleration of the braking segment (m/s^2)
        # Caught up: hover once the drone has stopped and the target has held still for hover_window;
        # resume tracking when it has walked this far from where it stopped
        self.hover_window = 1.0       # window the target is averaged over (s)
        self.hover_speed = 0.4        # drone speed below which it counts as stopped (m/s)
        self.hover_u_move = 20.0      # target movement across the image, column only (px)
        self.hover_target_move = 0.8  # target movement in the world (m)

        # traj_solve
        self.lock = Lock()
        self.state_transform = StateTransform()
        self.lattice_primitive = LatticePrimitive.get_instance()

        self.reset_state()  # all mutable state is initialized here; /respawn reuses the same routine

        # eval
        self.time_forward = 0.0
        self.time_process = 0.0
        self.time_prepare = 0.0
        self.time_interpolation = 0.0
        self.time_visualize = 0.0
        self.count = 0

        # Load Network
        if self.use_trt:
            self.policy = TRTModule()
            self.policy.load_state_dict(torch.load(weight))
        else:
            state_dict = torch.load(weight, weights_only=True)
            self.policy = YopoNetwork()
            self.policy.load_state_dict(state_dict)
            self.policy = self.policy.to(self.device)
            self.policy.eval()
        self.warm_up()

        # ros publisher
        self.lattice_traj_pub = self.create_publisher(MarkerArray, "/yopo_net/lattice_trajs_visual", 1)
        self.best_traj_pub = self.create_publisher(MarkerArray, "/yopo_net/best_traj_visual", 1)
        self.all_trajs_pub = self.create_publisher(MarkerArray, "/yopo_net/trajs_visual", 1)
        self.target_pub = self.create_publisher(PoseStamped, "/yopo_net/target_pred", 1)
        self.ekf_pub = self.create_publisher(PoseStamped, '/yopo_net/target_ekf', 1)
        self.target_image_pub = self.create_publisher(Image, "/yopo_net/target_image", 1)
        self.ctrl_pub = self.create_publisher(PositionCommand, config["ctrl_topic"], 1)
        # QoS: odometry uses BEST_EFFORT, RGB/depth images use RELIABLE.
        # Both use KEEP_LAST with depth=1 to keep only the latest message.
        odom_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        image_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.odom_sub = self.create_subscription(Odometry, config['odom_topic'], self.callback_odometry, odom_qos)
        self.respawn_sub = self.create_subscription(std_msgs.msg.Bool, '/respawn', self.callback_respawn, 1)
        self.depth_sub = message_filters.Subscriber(self, Image, config['depth_topic'], qos_profile=image_qos)
        self.rgb_sub = message_filters.Subscriber(self, Image, config['rgb_topic'], qos_profile=image_qos)
        sync = message_filters.ApproximateTimeSynchronizer([self.depth_sub, self.rgb_sub], queue_size=1, slop=0.05)
        sync.registerCallback(self.callback_images)
        # ros timer
        self.timer_ctrl = self.create_timer(self.ctrl_dt, self.control_pub)
        print("YOPO Net Node Ready!")

    def callback_odometry(self, data):
        self.odom = data
        if not self.desire_init:
            self.desire_pos = np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
            self.desire_vel = np.array((self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z))
            self.desire_acc = np.array((0.0, 0.0, 0.0))
            ypr = R.from_quat([self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y,
                               self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w]).as_euler('ZYX', degrees=False)
            self.last_yaw = ypr[0]
        self.odom_init = True

    def reset_state(self):
        """ All mutable state, shared by __init__ and /respawn. The caller must hold the lock. """
        # odom_init=False waits for a post-teleport odom; desire_init=False re-latches the reference
        self.odom = Odometry()
        self.odom_init = False
        self.desire_init = False
        self.last_yaw = 0.0
        self.desire_pos = self.desire_vel = self.desire_acc = None
        # ctrl_time=None means "nothing to track": control_pub stays silent and the controller hovers
        self.ctrl_time = None
        self.traj_time = self.lattice_primitive.segment_time
        self.optimal_poly_x = self.optimal_poly_y = self.optimal_poly_z = None
        # start from "never seen a target"
        self.target = np.array((10, 0, 2))
        self.target_seen = False
        self.last_trajectory_tracking = False
        self.searching = False
        self.search_yaw_sign = 1.0
        self.search_yaw_dot = 0.0
        self.hovering = False
        self.hover_buf = deque()   # (time, target pos, target image column u)
        self.hover_anchor = None   # (pos, u) where the target stopped
        self.ekf_target = None
        self.EKF = EKF(self.ekf_dt, self.ekf_thresh)  # rebuild: track, hit queue and reachable set all live inside

    def callback_respawn(self, msg):
        """ /respawn: the drone is teleported home; the planner returns to its start-up state (weights kept). """
        if not msg.data:
            return
        with self.lock:
            self.reset_state()
        self.get_logger().warn("[yopo_net] respawn: planner state reset")

    def process_odom(self):
        # Rwb -> Rwc -> Rcw
        Rotation_wb = R.from_quat([self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y,
                                   self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w]).as_matrix()
        self.Rotation_wc = np.dot(Rotation_wb, self.Rotation_bc)
        Rotation_cw = self.Rotation_wc.T

        # vel and acc
        vel_w = self.desire_vel if self.plan_from_reference else np.array([self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z])
        vel_c = np.dot(Rotation_cw, vel_w)
        acc_w = self.desire_acc
        acc_c = np.dot(Rotation_cw, acc_w)

        obs = np.concatenate((vel_c, acc_c), axis=0).astype(np.float32)
        obs_norm = self.state_transform.normalize_obs(torch.from_numpy(obs[None, :]))
        return obs_norm.to(self.device, non_blocking=True)

    def cur_pos(self):
        p = self.odom.pose.pose.position
        return np.array((p.x, p.y, p.z))

    def cur_vel(self):
        v = self.odom.twist.twist.linear
        return np.array((v.x, v.y, v.z))

    def start_state(self):
        """Start state of the next trajectory: the reference state if plan_from_reference, else the odometry."""
        if self.plan_from_reference:
            return self.desire_pos, self.desire_vel
        return self.cur_pos(), self.cur_vel()

    def set_trajectory(self, start_pos, start_vel, end_pos, end_vel, end_acc, traj_time):
        """Solve one quintic per axis and restart the tracking clock. The caller must hold the lock."""
        self.traj_time = traj_time
        self.optimal_poly_x, self.optimal_poly_y, self.optimal_poly_z = (
            Poly5Solver(start_pos[i], start_vel[i], self.desire_acc[i],
                        end_pos[i], end_vel[i], end_acc[i], traj_time) for i in range(3))
        self.ctrl_time = 0.0

    def parse_rgbd(self, depth_msg, rgb_msg):
        """ROS messages -> (network input [1, 4, H, W] on device, BGR image for the debug overlay)"""
        if depth_msg.encoding == "32FC1":
            depth_image = np.frombuffer(depth_msg.data, dtype=np.float32).reshape(depth_msg.height, depth_msg.width)
            depth_scale = 1.0     # raw depth unit: value per metre
        elif depth_msg.encoding == "16UC1":
            depth_image = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape(depth_msg.height, depth_msg.width)
            depth_scale = 1000.0  # millimetres
        else:
            raise ValueError(f"Unsupported depth_image encoding '{depth_msg.encoding}', expected '32FC1' or '16UC1'")
        if rgb_msg.encoding in ("bgr8", "rgb8"):
            rgb_image = np.frombuffer(rgb_msg.data, dtype=np.uint8).reshape(rgb_msg.height, rgb_msg.width, 3)
        else:
            raise ValueError(f"Unsupported rgb_image encoding '{rgb_msg.encoding}', expected 'bgr8' or 'rgb8'")

        # Resize on the uint8/uint16 data, before the float conversion below
        if depth_image.shape[:2] != (self.height, self.width):
            depth_image = cv2.resize(depth_image, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        if rgb_image.shape[:2] != (self.height, self.width):
            rgb_image = cv2.resize(rgb_image, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        if rgb_msg.encoding == "rgb8":  # to BGR, matching cv2.imread used in training
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        vis_image = rgb_image.copy() if self.target_image_pub.get_subscription_count() > 0 else None

        # Inpaint invalid points; the threshold is in the raw depth unit
        nan_mask = None if self.simulation else (depth_image < self.min_dis * depth_scale)
        if nan_mask is not None and depth_image.dtype != np.uint16:
            nan_mask |= np.isnan(depth_image)
        max_val = self.max_dis * depth_scale
        # cast the clamp threshold to the depth dtype, else np.minimum promotes to float64
        depth_image = np.minimum(depth_image, depth_image.dtype.type(max_val)).astype(np.float32, copy=False) / max_val
        if nan_mask is not None:
            depth_image = cv2.inpaint(np.uint8(depth_image * 255), np.uint8(nan_mask), 1, cv2.INPAINT_NS).astype(np.float32) / 255.0

        # Write straight into the destination array
        rgbd_image = np.empty((1, 4, self.height, self.width), dtype=np.float32)
        rgbd_image[0, 0:3] = rgb_image.transpose(2, 0, 1)
        rgbd_image[0, 0:3] /= 255.0
        rgbd_image[0, 3] = depth_image
        return torch.from_numpy(rgbd_image).to(self.device, non_blocking=True), vis_image

    def update_search(self):
        """Enter the yaw search when the drone is within search_dist of the last known target position and
        the EKF has no confirmed track: plan one braking trajectory, then only spin in place.
        Leaves the search as soon as the EKF confirms a track again.
        """
        if self.ekf_target is not None:
            self.searching = False
            return
        if self.searching or not self.target_seen or self.ekf_thresh <= 0:
            return
        cur_pos = self.cur_pos()
        if np.linalg.norm(self.target - cur_pos) >= self.search_dist:
            return

        # Continue from the measured yaw rate (the commanded yaw_dot chatters); keeping its direction
        # stops the ramp crossing zero and braking at the switch
        self.search_yaw_dot = float(self.odom.twist.twist.angular.z)
        self.search_yaw_sign = 1.0 if self.search_yaw_dot >= 0 else -1.0
        with self.lock:
            start_pos, start_vel = self.start_state()
            # T is set by the speed to shed and the initial acceleration to reverse
            traj_time = max((1.5 * float(np.linalg.norm(start_vel)) +
                             2.5 * float(np.linalg.norm(self.desire_acc))) / self.brake_acc, 0.5)
            end_pos = start_pos + 0.5 * start_vel * traj_time + self.desire_acc * traj_time ** 2 / 12.0
            self.set_trajectory(start_pos, start_vel, end_pos, np.zeros(3), np.zeros(3), traj_time)
        self.searching = True

    def target_uv(self, target):
        """Pixel position of a world target in the current camera frame, and its depth (m).
        uv is None when the target is behind the camera."""
        target_c = self.Rotation_wc.T @ (target - self.cur_pos())
        if target_c[0] <= 1e-3:
            return None, float(target_c[0])
        uv = np.array([cfg["camera_cx"] - target_c[1] * cfg["camera_f"] / target_c[0],
                       cfg["camera_cy"] - target_c[2] * cfg["camera_f"] / target_c[0]])
        return uv, float(target_c[0])

    def update_hover(self, confirmed):
        """Hover once the drone has stopped and the target has held still for hover_window; resume
        tracking when it has walked hover_target_move / hover_u_move from where it stopped."""
        target = self.ekf_target if self.ekf_target is not None else self.target
        uv, _ = self.target_uv(target)
        if not confirmed or uv is None or self.searching:
            if self.hovering:
                self.get_logger().info("[yopo_net] hover -> track: target lost")
            self.hovering, self.hover_anchor = False, None
            self.hover_buf.clear()
            return

        self.hover_buf.append((time.time(), target, uv[0]))  # u only: braking pitches the drone, moves v
        while self.hover_buf[-1][0] - self.hover_buf[0][0] > self.hover_window:
            self.hover_buf.popleft()
        if self.hover_buf[-1][0] - self.hover_buf[0][0] < 0.9 * self.hover_window:
            return  # not enough history yet

        # each half of the window is averaged, which keeps the detection noise out of the comparison
        mid = 0.5 * (self.hover_buf[-1][0] + self.hover_buf[0][0])
        half = lambda rows: (np.mean([r[1] for r in rows], axis=0), np.mean([r[2] for r in rows]))
        pos, u = half([r for r in self.hover_buf if r[0] >= mid])
        # hovering: distance from where the target stopped, so even a slow walk accumulates until it
        # crosses; tracking: distance covered within the window, i.e. whether it is moving right now
        ref = self.hover_anchor if self.hovering else half([r for r in self.hover_buf if r[0] < mid])
        d_pos, d_u = float(np.linalg.norm(pos - ref[0])), abs(float(u - ref[1]))
        moving = d_pos > self.hover_target_move or d_u > self.hover_u_move

        if self.hovering and moving:
            self.hovering, self.hover_anchor = False, None
            self.get_logger().info(f"[yopo_net] hover -> track: target moved ({d_pos:.2f}m {d_u:.1f}px)")
        # the drone's own speed only gates entering: once hovering it is zero by construction
        elif not self.hovering and not moving and np.linalg.norm(self.cur_vel()) < self.hover_speed:
            self.hovering, self.hover_anchor = True, (pos, u)
            self.get_logger().info("[yopo_net] track -> hover: caught up, target still")

    def log_timing(self, stamps):
        """Print the running average of the per-stage latency."""
        self.time_interpolation += stamps[1] - stamps[0]
        self.time_prepare += stamps[2] - stamps[1]
        self.time_forward += stamps[3] - stamps[2]
        self.time_process += stamps[4] - stamps[3]
        self.time_visualize += stamps[5] - stamps[4]
        self.count += 1
        n = self.count
        print(f"Time Consuming: image-process: {1000 * self.time_interpolation / n:.2f}ms; "
              f"state-process: {1000 * self.time_prepare / n:.2f}ms; "
              f"network-inference: {1000 * self.time_forward / n:.2f}ms; "
              f"post-process: {1000 * self.time_process / n:.2f}ms; "
              f"visualize-trajectory: {1000 * self.time_visualize / n:.2f}ms")

    @torch.inference_mode()
    def callback_images(self, depth_msg, rgb_msg):
        if not self.odom_init: return

        # 1. RGB-D image process
        time0 = time.time()
        rgbd_image, vis_image = self.parse_rgbd(depth_msg, rgb_msg)

        # 2. YOPO network inference (TensorRT: ~5x faster)
        time1 = time.time()
        obs_input = self.state_transform.prepare_input(self.process_odom()).to(self.device, non_blocking=True)
        time2 = time.time()
        preds = self.policy(rgbd_image, obs_input)
        # All outputs move back to CPU together, one device sync
        endstate_pred, target_pred, cost_pred, score_pred, time_pred = (p.cpu().numpy() for p in preds)
        time3 = time.time()

        # 3. Post-processing on CPU with numpy instead of CUDA with torch (~10x faster)
        endstate, target, cost, score, traj_time, action_id = self.process_output(
            endstate_pred, target_pred, cost_pred, score_pred, time_pred, return_all_preds=self.visualize)
        # vectorized body -> world rotation of the predicted P V A (attitude only, no translation)
        endstate_c = endstate.reshape(-1, 3, 3).transpose(0, 2, 1)  # [N, 9] -> [px vx ax, py vy ay, pz vz az]
        endstate_w = np.matmul(self.Rotation_wc, endstate_c)

        if score > self.conf_thresh:
            self.target = target
            self.last_trajectory_tracking = True
            self.target_seen = True
            self.publish_target(self.target_pub, target)
        elif self.ctrl_time is not None and self.ctrl_time > 0.3 * self.traj_time:
            self.last_trajectory_tracking = False

        self.update_search()
        # the EKF track already rides out brief occlusions; without it, fall back to the frame
        self.update_hover(self.ekf_target is not None if self.ekf_thresh > 0 else score > self.conf_thresh)

        # target_seen: nothing is planned before the first confirmed target, so the controller hovers
        if not self.hovering and (score > self.conf_thresh or (self.target_seen and not self.last_trajectory_tracking and not self.searching)):
            with self.lock:
                start_pos, start_vel = self.start_state()
                end = endstate_w[action_id]  # [px vx ax, py vy ay, pz vz az]
                self.set_trajectory(start_pos, start_vel, end[:, 0] + start_pos, end[:, 1], end[:, 2],
                                    float(traj_time[action_id]))  # duration is predicted per trajectory

        time4 = time.time()
        self.visualize_trajectory(score_pred.reshape(-1), endstate_w, traj_time, score)
        if vis_image is not None:
            self.publish_target_image(vis_image, target, score, rgb_msg.header.stamp)
        time5 = time.time()

        if self.verbose:
            self.log_timing((time0, time1, time2, time3, time4, time5))

    def desired_yaw(self):
        """Yaw command: while searching, ramp the rate to search_yaw_rate and hold;
        otherwise point at the target (the EKF estimate when confirmed)."""
        if self.searching:
            target_rate = self.search_yaw_sign * self.search_yaw_rate
            step = self.search_yaw_acc * self.ctrl_dt
            self.search_yaw_dot += np.clip(target_rate - self.search_yaw_dot, -step, step)
            return wrap_to_pi(self.last_yaw + self.search_yaw_dot * self.ctrl_dt), self.search_yaw_dot
        towards_target = self.ekf_target if self.ekf_target is not None else self.target
        return calculate_yaw(towards_target - self.desire_pos, self.last_yaw, self.ctrl_dt, max_yaw_rate=1)

    def control_pub(self):
        # The lock guards against respawn clearing ctrl_time and the polynomials mid-call, so the early returns must be inside it too
        with self.lock:
            if self.ctrl_time is None or (self.ctrl_time > self.traj_time and not self.searching and not self.hovering):
                return
            if self.hovering:  # freeze the reference; the EMPTY command below brakes and holds it
                self.desire_vel, self.desire_acc = np.zeros(3), np.zeros(3)
            else:
                self.ctrl_time += self.ctrl_dt
                if self.searching:  # hold the braking endpoint once the segment ends
                    self.ctrl_time = min(self.ctrl_time, self.traj_time)
                polys = (self.optimal_poly_x, self.optimal_poly_y, self.optimal_poly_z)
                t = self.ctrl_time
                self.desire_pos = np.array([poly.get_position(t) for poly in polys])
                self.desire_vel = np.array([poly.get_velocity(t) for poly in polys])
                self.desire_acc = np.array([poly.get_acceleration(t) for poly in polys])
            yaw, yaw_dot = self.desired_yaw()
            self.last_yaw = yaw

            control_msg = PositionCommand()
            control_msg.header.stamp = self.get_clock().now().to_msg()
            # EMPTY switches the controller to position feedback, which brakes to desire_pos and holds it
            control_msg.trajectory_flag = (control_msg.TRAJECTORY_STATUS_EMPTY if self.hovering else control_msg.TRAJECTORY_STATUS_READY)
            control_msg.position.x, control_msg.position.y, control_msg.position.z = self.desire_pos
            control_msg.velocity.x, control_msg.velocity.y, control_msg.velocity.z = self.desire_vel
            control_msg.acceleration.x, control_msg.acceleration.y, control_msg.acceleration.z = self.desire_acc
            control_msg.yaw, control_msg.yaw_dot = yaw, yaw_dot
            self.desire_init = True
            self.ctrl_pub.publish(control_msg)

    def process_output(self, endstate_pred, target_pred, cost_pred, score_pred, time_pred, return_all_preds=False):
        endstate_pred = endstate_pred.reshape(9, self.lattice_primitive.traj_num).T
        cost_pred = cost_pred.reshape(self.lattice_primitive.traj_num)
        score_pred = score_pred.reshape(self.lattice_primitive.traj_num)
        time_pred = self.state_transform.pred_to_traj_time_cpu(time_pred.reshape(self.lattice_primitive.traj_num))

        target_pred = self.state_transform.pred_to_target_cpu(target_pred, transfer_world=True)
        target_pred = target_pred.reshape(3, self.lattice_primitive.traj_num)

        cur_pos = np.array([[self.odom.pose.pose.position.x], [self.odom.pose.pose.position.y], [self.odom.pose.pose.position.z]])
        target_w = np.dot(self.Rotation_wc, target_pred) + cur_pos
        action_id, have_target = self.NMS(cost_pred, score_pred, target_w)
        target = target_w[:, action_id]
        score = have_target

        if not return_all_preds:
            lattice_id = self.lattice_primitive.convert_ImageGrid_LatticeID(action_id)
            endstate = self.state_transform.pred_to_endstate_cpu(endstate_pred[action_id, :][np.newaxis, :], lattice_id)
            cost = cost_pred[action_id]
            traj_time = time_pred[action_id][np.newaxis]  # keep the leading dim consistent with endstate
            action_id = 0
        else:
            endstate = self.state_transform.pred_to_endstate_cpu(endstate_pred, np.arange(self.lattice_primitive.traj_num-1, -1, -1))
            cost = cost_pred
            traj_time = time_pred

        return endstate, target, cost, score, traj_time, action_id

    def NMS(self, cost_pred, score_pred, target_pred):
        """score_pred: [traj_num,]; target_pred: [3, traj_num].
        Return the lowest-cost anchor whose score exceeds conf_thresh, else the best no-target fallback."""
        mask = score_pred > self.conf_thresh
        valid_indices = np.nonzero(mask)[0]

        if self.ekf_thresh > 0:
            valid_targets = target_pred.T[valid_indices]
            idx_of_valid_indices, ekf_predicted, ekf_reliable = self.EKF.predict_and_update(valid_targets)
            valid_indices = valid_indices[idx_of_valid_indices] if idx_of_valid_indices.size > 0 else np.empty(0)

            self.ekf_target = ekf_predicted if ekf_reliable else None
            if ekf_reliable:  # publish confirmed states only
                self.publish_target(self.ekf_pub, ekf_predicted)

        if valid_indices.size == 1:
            optimal_index = valid_indices[0]
            have_target = 1
        elif valid_indices.size > 1:
            valid_costs = cost_pred[valid_indices]
            optimal_index = valid_indices[np.argmin(valid_costs)]
            have_target = 1
        else:
            # Fallback: cheapest anchor among those reporting no target, so an anchor the EKF just rejected cannot be selected again
            fallback = np.nonzero(~mask)[0]
            optimal_index = fallback[np.argmin(cost_pred[fallback])] if fallback.size > 0 else np.argmin(cost_pred)
            have_target = 0

        return optimal_index, have_target

    def publish_lines(self, publisher, points, colors, width):
        """Publish sampled trajectories as LINE_STRIP markers in the world frame.
        points: (N, K, 3); colors: RGBA, (4,) shared or (N, 4) per trajectory."""
        header = std_msgs.msg.Header(frame_id='world', stamp=self.get_clock().now().to_msg())
        colors = np.broadcast_to(colors, (len(points), 4)).tolist()
        markers = MarkerArray()
        for i, (line, (r, g, b, a)) in enumerate(zip(points.tolist(), colors)):
            m = Marker(header=header, id=i, type=Marker.LINE_STRIP, action=Marker.ADD, color=std_msgs.msg.ColorRGBA(r=r, g=g, b=b, a=a))
            m.pose.orientation.w, m.scale.x = 1.0, width
            m.points = [Point(x=x, y=y, z=z) for x, y, z in line]
            markers.markers.append(m)
        publisher.publish(markers)

    def sample_polys(self, solvers, t_values):
        """Positions of a per-axis solver triple, (N, K, 3) for N trajectories at K samples."""
        return np.stack([s.get_position(t_values) for s in solvers], axis=-1).reshape(-1, len(t_values), 3)

    def visualize_trajectory(self, pred_score, pred_endstate, pred_time, target_conf):
        # normalized time, so trajectories of different durations get the same point count
        s_values = np.linspace(0, 1.0, 20)
        start_pos, start_vel = self.start_state()

        # best predicted trajectory: green when tracking a target, orange otherwise
        if self.optimal_poly_x is not None and self.best_traj_pub.get_subscription_count() > 0:
            t_values = np.linspace(0, self.traj_time, 20)
            points = self.sample_polys((self.optimal_poly_x, self.optimal_poly_y, self.optimal_poly_z), t_values)
            color = (0.0, 1.0, 0.2, 1.0) if target_conf > self.conf_thresh else (1.0, 0.6, 0.0, 1.0)
            self.publish_lines(self.best_traj_pub, points, color, width=0.1)

        # lattice primitive
        if self.visualize and self.lattice_traj_pub.get_subscription_count() > 0:
            lattice_endstate = self.lattice_primitive.lattice_pos_node.cpu().numpy()
            lattice_endstate = np.dot(lattice_endstate, self.Rotation_wc.T)
            zero = np.zeros_like(lattice_endstate[:, 0])
            solvers = [Polys5Solver(start_pos[i], start_vel[i], self.desire_acc[i],
                                    lattice_endstate[:, i] + start_pos[i], zero, zero,
                                    self.lattice_primitive.segment_time) for i in range(3)]
            self.publish_lines(self.lattice_traj_pub, self.sample_polys(solvers, s_values), (0.5, 0.5, 0.5, 0.6), width=0.03)

        # all predicted trajectories: blue for positive anchors (target score > conf_thresh), orange for negative
        if self.visualize and self.all_trajs_pub.get_subscription_count() > 0:
            solvers = [Polys5Solver(start_pos[i], start_vel[i], self.desire_acc[i],
                                    pred_endstate[:, i, 0] + start_pos[i], pred_endstate[:, i, 1],
                                    pred_endstate[:, i, 2], pred_time) for i in range(3)]
            colors = np.where((pred_score > self.conf_thresh)[:, None], (0.1, 0.4, 1.0, 0.9), (1.0, 0.6, 0.0, 0.5))
            self.publish_lines(self.all_trajs_pub, self.sample_polys(solvers, s_values), colors, width=0.04)

    def publish_target(self, publisher, target):
        target_pose = PoseStamped()
        target_pose.header.frame_id = 'world'
        target_pose.pose.position.x, target_pose.pose.position.y, target_pose.pose.position.z = target
        publisher.publish(target_pose)

    def publish_target_image(self, vis_image, target, score, stamp):
        uv, dist = self.target_uv(target)
        if uv is not None:
            u, v = int(round(uv[0])), int(round(uv[1]))
            if 0 <= u < self.width and 0 <= v < self.height:
                color = (255, 255, 0) if self.hovering else (0, 255, 0) if score > self.conf_thresh else (0, 165, 255)
                cv2.drawMarker(vis_image, (u, v), color, cv2.MARKER_CROSS, 8, 1)
                cv2.circle(vis_image, (u, v), 5, color, 1)
                text = f"{score:.2f} {dist:.1f}m"
                font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1
                (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
                tx = min(max(u + 7, 0), self.width - tw)
                ty = min(max(v - 5, th), self.height - 1)
                cv2.putText(vis_image, text, (tx, ty), font, scale, color, thick, cv2.LINE_AA)

        img_msg = Image()
        img_msg.header.stamp = stamp
        img_msg.header.frame_id = 'camera'
        img_msg.height = vis_image.shape[0]
        img_msg.width = vis_image.shape[1]
        img_msg.encoding = 'bgr8'
        img_msg.is_bigendian = 0
        img_msg.step = vis_image.shape[1] * 3
        img_msg.data = vis_image.tobytes()
        self.target_image_pub.publish(img_msg)

    def warm_up(self):
        depth = torch.zeros((1, 4, self.height, self.width), dtype=torch.float32, device=self.device)
        obs = torch.zeros((1, 6), dtype=torch.float32, device=self.device)
        obs = self.state_transform.prepare_input(obs)
        _ = self.policy(depth, obs)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_tensorrt", type=int, default=0, help="use tensorrt or not")
    parser.add_argument("--trial", type=int, default=1, help="trial number")
    parser.add_argument("--epoch", type=int, default=50, help="epoch number")
    return parser


if __name__ == "__main__":
    args = parser().parse_args()
    rclpy.init()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weight = "yopo_trt.pth" if args.use_tensorrt else base_dir + "/saved/YOPO_{}/epoch{}.pth".format(args.trial, args.epoch)
    print("load weight from:", weight)

    settings = {'use_tensorrt': args.use_tensorrt,
                'pitch_angle_deg': -0,          # camera pitch angle (negative is upward)
                'odom_topic': '/drone_0/odom',               # odometry topic
                'depth_topic': '/drone_0/depth',             # depth image topic
                'rgb_topic': '/drone_0/rgb',                 # rgb image topic
                'ctrl_topic': '/so3_control/pos_cmd',        # controller topic
                'image_fps': 30,                # image frame rate (used by the EKF)
                'simulation': True,             # simulation? simulated depth has no invalid points, so inpainting is skipped; set False for real flight
                'conf_thresh': 0.9,             # target score threshold
                'ekf_thresh': 2.0,              # target rejection distance [-1 to disable]
                'plan_from_reference': False,   # plan from the reference state? True for a position controller, False for direct network control
                'verbose': False,               # print timing?
                'visualize': True               # visualize all trajectories? (set False for real flight to save computation)
                }
    node = YopoNet(settings, weight)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
