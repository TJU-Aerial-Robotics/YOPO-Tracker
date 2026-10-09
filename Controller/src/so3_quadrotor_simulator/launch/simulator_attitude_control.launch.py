"""Tracker dynamics + attitude controller + camera TF (single drone, no namespace).

Odom is published straight to /drone_0/odom, the pose the Unity renderer consumes.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    tx, ty, tz = (LaunchConfiguration(k) for k in ('tracker_x', 'tracker_y', 'tracker_z'))
    logger_dir = get_package_share_directory('so3_control') + '/logger/'

    tracker_sim = Node(
        package='so3_quadrotor_simulator',
        executable='quadrotor_simulator_so3',
        name='quadrotor_simulator_so3',
        output='screen',
        parameters=[{
            'is_tracker': True,
            'rate.odom': 50.0,
            'simulator.init_state_x': tx,
            'simulator.init_state_y': ty,
            'simulator.init_state_z': tz,
        }],
        remappings=[
            ('odom', '/drone_0/odom'),
            ('imu', '/sim/imu'),
            ('cmd', '/so3_cmd'),
            ('force_disturbance', '/force_disturbance'),
            ('moment_disturbance', '/moment_disturbance'),
        ],
    )

    controller = Node(
        package='so3_control',
        executable='network_control_node',
        name='network_controller_node',
        output='screen',
        parameters=[{
            'is_simulation': True,
            'use_disturbance_observer': True,
            'disturbance_observer_type': 'HGDO',  # ADO or HGDO
            'hover_thrust': 0.375,
            'record_log': False,
            'logger_file_name': logger_dir,
        }],
        remappings=[
            ('odom', '/drone_0/odom'),
            ('imu', '/sim/imu'),
            ('position_cmd', '/so3_control/pos_cmd'),
            ('so3_cmd', '/so3_cmd'),
        ],
    )

    # Body (odom, FLU) -> camera optical frame (z forward, x right, y down), the frame_id of /drone_0/rgb.
    camera_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_tf_publisher',
        arguments=['--x', '0', '--y', '0', '--z', '0.1',
                   '--roll', '-1.5708', '--pitch', '0', '--yaw', '-1.5708',
                   '--frame-id', 'odom', '--child-frame-id', 'camera_frame'],
    )

    odom_paths = Node(
        package='so3_quadrotor_simulator',
        executable='odom_paths.py',
        output='screen',
        remappings=[
            ('character_0/odom', LaunchConfiguration('target_odom_topic')),
            ('drone_0/odom', LaunchConfiguration('drone_odom_topic')),
            ('character_0/path', LaunchConfiguration('target_path_topic')),
            ('drone_0/path', LaunchConfiguration('drone_path_topic')),
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument('tracker_x', default_value='45.0'),
        DeclareLaunchArgument('tracker_y', default_value='-50.0'),
        DeclareLaunchArgument('tracker_z', default_value='2.0'),
        DeclareLaunchArgument('target_odom_topic', default_value='/character_0/odom'),
        DeclareLaunchArgument('drone_odom_topic', default_value='/drone_0/odom'),
        DeclareLaunchArgument('target_path_topic', default_value='/character_0/path'),
        DeclareLaunchArgument('drone_path_topic', default_value='/drone_0/path'),
        tracker_sim,
        controller,
        camera_tf,
        odom_paths,
    ])
