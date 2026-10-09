#!/usr/bin/env python3
"""按 0.1 秒间隔记录并发布最近 100 个位置。"""
from collections import deque
from copy import deepcopy

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker


class OdomPaths(Node):
    def __init__(self):
        super().__init__('odom_paths')
        self.mesh_pub = self.create_publisher(Marker, 'character_0/mesh', 1)
        self.mesh = Marker()
        self.mesh.header.frame_id = 'world'
        self.mesh.ns = 'mesh'
        self.mesh.type = Marker.MESH_RESOURCE
        self.mesh.mesh_resource = 'file://' + get_package_share_directory('so3_quadrotor_simulator') + '/config/uav.dae'
        self.mesh.scale.x = self.mesh.scale.y = self.mesh.scale.z = 2.0
        self.mesh.color.r = self.mesh.color.a = 1.0
        self.history = {name: deque(maxlen=100) for name in ('character_0', 'drone_0')}
        self.last_stamp = {}
        self.publishers_by_name = {}
        for name in self.history:
            self.publishers_by_name[name] = self.create_publisher(Path, f'{name}/path', 1)
            self.create_subscription(
                Odometry, f'{name}/odom',
                lambda msg, name=name: self.on_odom(name, msg),
                qos_profile_sensor_data)
        self.create_subscription(Bool, '/respawn', self.reset, 1)

    def reset(self, msg):
        if msg.data:
            self.last_stamp.clear()
            for name, history in self.history.items():
                history.clear()
                path = Path()
                path.header.frame_id = 'world'
                path.header.stamp = self.get_clock().now().to_msg()
                self.publishers_by_name[name].publish(path)

    def on_odom(self, name, msg):
        if name == 'character_0':
            self.mesh.header.stamp = deepcopy(msg.header.stamp)
            self.mesh.pose = deepcopy(msg.pose.pose)
            self.mesh.pose.position.z += 1.0
            self.mesh_pub.publish(self.mesh)
        stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        previous = self.last_stamp.get(name)
        if previous is not None:
            if stamp < previous:
                self.history[name].clear()
            elif stamp - previous < 100_000_000:
                return
        self.last_stamp[name] = stamp
        pose = PoseStamped()
        pose.header.stamp = deepcopy(msg.header.stamp)
        pose.header.frame_id = 'world'
        pose.pose = deepcopy(msg.pose.pose)
        if name == 'character_0':
            pose.pose.position.z += 1.0
        self.history[name].append(pose)
        path = Path()
        path.header = deepcopy(pose.header)
        path.poses = list(self.history[name])
        self.publishers_by_name[name].publish(path)


def main():
    rclpy.init()
    node = OdomPaths()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
