# UAV Simulator

### Build:
```bash
colcon build --symlink-install
```

### MODE 1. PID Position Controller and Simulator
Work with traditional planner
```
source install/setup.bash
ros2 launch so3_quadrotor_simulator simulator_position_control.launch.py
```

### MODE 2. Attitude Controller with Disturbance Observer
Work with our learning-based planner (without position controller)
```
source install/setup.bash
ros2 launch so3_quadrotor_simulator simulator_attitude_control.launch.py
```

### Others
pub disturbance
```
ros2 topic pub /force_disturbance geometry_msgs/msg/Vector3 "{x: 0.0, y: 0.0, z: 1.0}"
```

takeoff and land (used in realworld flight)
```
ros2 service call /takeoff_land quadrotor_msgs/srv/SetTakeoffLand "{takeoff: true, takeoff_altitude: 2.0}"
```

### acknowledgment

This repo is modified from https://github.com/HKUST-Aerial-Robotics/Fast-Planner, thanks for their excellent work!