from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    hover_thrust = LaunchConfiguration('hover_thrust')
    logger_dir = get_package_share_directory('so3_control') + '/logger/'

    return LaunchDescription([
        DeclareLaunchArgument('hover_thrust', default_value='0.38'),
        Node(
            package='so3_control',
            executable='network_control_node',
            name='network_controller_node',
            output='screen',
            parameters=[{
                'is_simulation': False,
                'use_disturbance_observer': True,
                'disturbance_observer_type': 'ADO',  # ADO or HGDO
                'hover_thrust': hover_thrust,
                'kx_xy': 5.7,
                'kx_z': 6.2,
                'kv_xy': 3.4,
                'kv_z': 4.0,
                'record_log': True,
                'logger_file_name': logger_dir,
            }],
            remappings=[
                ('odom', '/vins_estimator/imu_propagate'),
                ('imu', '/mavros/imu/data_raw'),
                ('position_cmd', '/so3_control/pos_cmd'),
                ('so3_cmd', 'so3_cmd'),
            ],
        ),
    ])
