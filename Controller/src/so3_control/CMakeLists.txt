cmake_minimum_required(VERSION 3.8)
project(so3_control)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(quadrotor_msgs REQUIRED)
find_package(mavros_msgs REQUIRED)
find_package(uav_utils REQUIRED)
find_package(Eigen3 REQUIRED)

set(DEPS rclcpp nav_msgs sensor_msgs std_msgs quadrotor_msgs mavros_msgs uav_utils)

add_library(SO3Control src/SO3Control.cpp)
target_include_directories(SO3Control PUBLIC include)
target_link_libraries(SO3Control Eigen3::Eigen)

# PID position controller (was the ROS1 nodelet)
add_executable(so3_control_node src/so3_control_node.cpp)
target_include_directories(so3_control_node PRIVATE include)
target_link_libraries(so3_control_node SO3Control)
ament_target_dependencies(so3_control_node ${DEPS})

# simple command publisher example
add_executable(control_example src/control_example.cpp)
target_link_libraries(control_example Eigen3::Eigen)
ament_target_dependencies(control_example rclcpp quadrotor_msgs)

# maps acceleration commands to attitude, with a disturbance observer as inner-loop input
add_executable(network_control_node src/NetworkControl.cpp src/network_control_node.cpp)
target_include_directories(network_control_node PRIVATE include)
target_link_libraries(network_control_node SO3Control)
ament_target_dependencies(network_control_node ${DEPS})

install(TARGETS SO3Control
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(TARGETS so3_control_node control_example network_control_node
  DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch config logger DESTINATION share/${PROJECT_NAME})

ament_package()
