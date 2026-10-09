cmake_minimum_required(VERSION 3.8)
project(so3_quadrotor_simulator)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(quadrotor_msgs REQUIRED)
find_package(uav_utils REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(quadrotor_dynamics src/dynamics/Quadrotor.cpp)
target_include_directories(quadrotor_dynamics PUBLIC
  include ${CMAKE_CURRENT_SOURCE_DIR}/include/ode)
target_link_libraries(quadrotor_dynamics Eigen3::Eigen)

add_executable(quadrotor_simulator_so3 src/quadrotor_simulator_so3.cpp)
target_include_directories(quadrotor_simulator_so3 PRIVATE include)
target_link_libraries(quadrotor_simulator_so3 quadrotor_dynamics)
ament_target_dependencies(quadrotor_simulator_so3
  rclcpp nav_msgs sensor_msgs geometry_msgs visualization_msgs std_msgs
  tf2_ros ament_index_cpp quadrotor_msgs uav_utils)

install(TARGETS quadrotor_dynamics
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(TARGETS quadrotor_simulator_so3 DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS scripts/odom_paths.py DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

ament_package()
