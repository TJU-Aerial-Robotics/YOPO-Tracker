cmake_minimum_required(VERSION 3.8)
project(quadrotor_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  msg/AuxCommand.msg
  msg/Corrections.msg
  msg/Gains.msg
  msg/OutputData.msg
  msg/PositionCommand.msg
  msg/PPROutputData.msg
  msg/Serial.msg
  msg/SO3Command.msg
  msg/StatusData.msg
  msg/TRPYCommand.msg
  msg/Odometry.msg
  msg/PolynomialTrajectory.msg
  msg/LQRTrajectory.msg
  srv/SetTakeoffLand.srv
  DEPENDENCIES std_msgs geometry_msgs nav_msgs
)

ament_package()
