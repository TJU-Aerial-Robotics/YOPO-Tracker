// PID position controller (ROS1 nodelet -> ROS2 node).
#include <Eigen/Geometry>
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/bool.hpp>
#include <quadrotor_msgs/msg/corrections.hpp>
#include <quadrotor_msgs/msg/position_command.hpp>
#include <quadrotor_msgs/msg/so3_command.hpp>
#include <so3_control/SO3Control.h>
#include <uav_utils/converters.h>
#include <string>
#include <iostream>
#include <fstream>

class SO3ControlNode : public rclcpp::Node
{
public:
  using PositionCommand = quadrotor_msgs::msg::PositionCommand;
  using SO3Command = quadrotor_msgs::msg::SO3Command;
  using Corrections = quadrotor_msgs::msg::Corrections;

  SO3ControlNode();

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  void publishSO3Command(void);
  void position_cmd_callback(const PositionCommand::ConstSharedPtr cmd);
  void odom_callback(const nav_msgs::msg::Odometry::ConstSharedPtr odom);
  void enable_motors_callback(const std_msgs::msg::Bool::ConstSharedPtr msg);
  void corrections_callback(const Corrections::ConstSharedPtr msg);
  void imu_callback(const sensor_msgs::msg::Imu::ConstSharedPtr imu);

  void initLogRecorder();
  void recordLog();

  SO3Control controller_;
  rclcpp::Publisher<SO3Command>::SharedPtr so3_command_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<PositionCommand>::SharedPtr position_cmd_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr enable_motors_sub_;
  rclcpp::Subscription<Corrections>::SharedPtr corrections_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;

  bool position_cmd_updated_{false}, position_cmd_init_{false};
  std::string frame_id_;

  bool record_log_{false};
  bool cur_acc_init_{false}, cur_odom_init_{false};
  Eigen::Vector3d des_pos_, des_vel_, des_acc_, kx_, kv_;
  Eigen::Vector3d cur_pos_, cur_vel_, cur_acc_;
  double des_yaw_{0}, des_yaw_dot_{0};
  double current_yaw_{0};
  bool enable_motors_{true};
  bool use_external_yaw_{false};
  double kR_[3], kOm_[3], corrections_[3];
  double init_x_, init_y_, init_z_;

  std::ofstream logger;
  std::string logger_file_name;
};

SO3ControlNode::SO3ControlNode() : rclcpp::Node("so3_control")
{
  std::string quadrotor_name = declare_parameter("quadrotor_name", std::string("quadrotor"));
  frame_id_ = "/" + quadrotor_name;

  controller_.setMass(declare_parameter("mass", 0.5));

  record_log_ = declare_parameter("record_log", false);
  logger_file_name = declare_parameter("PID_logger_file_name", std::string("/tmp/"));
  use_external_yaw_ = declare_parameter("use_external_yaw", true);

  // ROS2 parameter names use '.' instead of '/' for nesting
  kR_[0] = declare_parameter("gains.rot.x", 1.5);
  kR_[1] = declare_parameter("gains.rot.y", 1.5);
  kR_[2] = declare_parameter("gains.rot.z", 1.0);
  kOm_[0] = declare_parameter("gains.ang.x", 0.13);
  kOm_[1] = declare_parameter("gains.ang.y", 0.13);
  kOm_[2] = declare_parameter("gains.ang.z", 0.1);
  kx_[0] = declare_parameter("gains.kx.x", 5.7);
  kx_[1] = declare_parameter("gains.kx.y", 5.7);
  kx_[2] = declare_parameter("gains.kx.z", 6.2);
  kv_[0] = declare_parameter("gains.kv.x", 3.4);
  kv_[1] = declare_parameter("gains.kv.y", 3.4);
  kv_[2] = declare_parameter("gains.kv.z", 4.0);

  corrections_[0] = declare_parameter("corrections.z", 0.0);
  corrections_[1] = declare_parameter("corrections.r", 0.0);
  corrections_[2] = declare_parameter("corrections.p", 0.0);

  init_x_ = declare_parameter("so3_control.init_state_x", 0.0);
  init_y_ = declare_parameter("so3_control.init_state_y", 0.0);
  init_z_ = declare_parameter("so3_control.init_state_z", -10000.0);

  // Commands reliable (ROS1 used tcpNoDelay), sensor streams best-effort.
  const auto cmd_qos = rclcpp::QoS(10);
  const auto sensor_qos = rclcpp::SensorDataQoS().keep_last(10);
  so3_command_pub_ = create_publisher<SO3Command>("so3_cmd", 10);
  odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
    "odom", sensor_qos, std::bind(&SO3ControlNode::odom_callback, this, std::placeholders::_1));
  position_cmd_sub_ = create_subscription<PositionCommand>(
    "position_cmd", cmd_qos, std::bind(&SO3ControlNode::position_cmd_callback, this, std::placeholders::_1));
  enable_motors_sub_ = create_subscription<std_msgs::msg::Bool>(
    "motors", cmd_qos, std::bind(&SO3ControlNode::enable_motors_callback, this, std::placeholders::_1));
  corrections_sub_ = create_subscription<Corrections>(
    "corrections", cmd_qos, std::bind(&SO3ControlNode::corrections_callback, this, std::placeholders::_1));
  imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
    "imu", sensor_qos, std::bind(&SO3ControlNode::imu_callback, this, std::placeholders::_1));

  if (record_log_)
    initLogRecorder();
}

void SO3ControlNode::publishSO3Command(void)
{
  controller_.calculateControl(des_pos_, des_vel_, des_acc_, des_yaw_,
                               des_yaw_dot_, kx_, kv_);

  const Eigen::Vector3d& force = controller_.getComputedForce();
  const Eigen::Quaterniond& orientation = controller_.getComputedOrientation();

  SO3Command so3_command;
  so3_command.header.stamp = now();
  so3_command.header.frame_id = frame_id_;
  so3_command.force.x = force(0);
  so3_command.force.y = force(1);
  so3_command.force.z = force(2);
  so3_command.orientation.x = orientation.x();
  so3_command.orientation.y = orientation.y();
  so3_command.orientation.z = orientation.z();
  so3_command.orientation.w = orientation.w();
  for (int i = 0; i < 3; i++)
  {
    so3_command.k_r[i] = kR_[i];
    so3_command.k_om[i] = kOm_[i];
  }
  so3_command.aux.current_yaw = current_yaw_;
  so3_command.aux.kf_correction = corrections_[0];
  so3_command.aux.angle_corrections[0] = corrections_[1];
  so3_command.aux.angle_corrections[1] = corrections_[2];
  so3_command.aux.enable_motors = enable_motors_;
  so3_command.aux.use_external_yaw = use_external_yaw_;
  so3_command_pub_->publish(so3_command);
}

void SO3ControlNode::position_cmd_callback(const PositionCommand::ConstSharedPtr cmd)
{
  des_pos_ = Eigen::Vector3d(cmd->position.x, cmd->position.y, cmd->position.z);
  des_vel_ = Eigen::Vector3d(cmd->velocity.x, cmd->velocity.y, cmd->velocity.z);
  des_acc_ = Eigen::Vector3d(cmd->acceleration.x, cmd->acceleration.y, cmd->acceleration.z);

  if (cmd->kx[0] > 1e-5 || cmd->kx[1] > 1e-5 || cmd->kx[2] > 1e-5)
    kx_ = Eigen::Vector3d(cmd->kx[0], cmd->kx[1], cmd->kx[2]);
  if (cmd->kv[0] > 1e-5 || cmd->kv[1] > 1e-5 || cmd->kv[2] > 1e-5)
    kv_ = Eigen::Vector3d(cmd->kv[0], cmd->kv[1], cmd->kv[2]);

  des_yaw_ = cmd->yaw;
  des_yaw_dot_ = cmd->yaw_dot;
  position_cmd_updated_ = true;
  position_cmd_init_ = true;

  publishSO3Command();

  if (record_log_ && cur_acc_init_ && cur_odom_init_)
    recordLog();
}

void SO3ControlNode::odom_callback(const nav_msgs::msg::Odometry::ConstSharedPtr odom)
{
  const Eigen::Vector3d position(odom->pose.pose.position.x,
                                 odom->pose.pose.position.y,
                                 odom->pose.pose.position.z);
  const Eigen::Vector3d velocity(odom->twist.twist.linear.x,
                                 odom->twist.twist.linear.y,
                                 odom->twist.twist.linear.z);

  cur_odom_init_ = true;
  current_yaw_ = uav_utils::get_yaw(odom->pose.pose.orientation);
  cur_pos_ = position;
  cur_vel_ = velocity;

  controller_.setPosition(position);
  controller_.setVelocity(velocity);

  if (position_cmd_init_)
  {
    // Expect position_cmd_callback right after each odom; if it did not fire,
    // publish the so3 command ourselves.
    if (!position_cmd_updated_)
      publishSO3Command();
    position_cmd_updated_ = false;
  }
  else if (init_z_ > -9999.0)
  {
    des_pos_ = Eigen::Vector3d(init_x_, init_y_, init_z_);
    des_vel_ = Eigen::Vector3d(0, 0, 0);
    des_acc_ = Eigen::Vector3d(0, 0, 0);
    publishSO3Command();
  }
}

void SO3ControlNode::enable_motors_callback(const std_msgs::msg::Bool::ConstSharedPtr msg)
{
  RCLCPP_INFO(get_logger(), msg->data ? "Enabling motors" : "Disabling motors");
  enable_motors_ = msg->data;
}

void SO3ControlNode::corrections_callback(const Corrections::ConstSharedPtr msg)
{
  corrections_[0] = msg->kf_correction;
  corrections_[1] = msg->angle_corrections[0];
  corrections_[2] = msg->angle_corrections[1];
}

void SO3ControlNode::imu_callback(const sensor_msgs::msg::Imu::ConstSharedPtr imu)
{
  cur_acc_init_ = true;
  const Eigen::Vector3d acc(imu->linear_acceleration.x,
                            imu->linear_acceleration.y,
                            imu->linear_acceleration.z);
  cur_acc_ = acc;
  controller_.setAcc(acc);
}

void SO3ControlNode::initLogRecorder()
{
  std::cout << "logger_file_name: " << logger_file_name << std::endl;
  std::string temp_file_name = logger_file_name + "PID_logger_";
  time_t timep = time(0);
  char tmp[64];
  strftime(tmp, sizeof(tmp), "%Y_%m_%d_%H_%M_%S", localtime(&timep));
  temp_file_name += tmp;
  temp_file_name += ".csv";
  if (logger.is_open())
    logger.close();
  logger.open(temp_file_name.c_str(), std::ios::out);
  std::cout << "PID logger: " << temp_file_name << std::endl;
  if (!logger.is_open())
  {
    std::cout << "cannot open the logger." << std::endl;
    return;
  }
  logger << "timestamp" << ',';
  logger << "cur_x" << ',' << "cur_y" << ',' << "cur_z" << ',';
  logger << "cur_vx" << ',' << "cur_vy" << ',' << "cur_vz" << ',';
  logger << "cur_ax" << ',' << "cur_ay" << ',' << "cur_az" << ',';
  logger << "not_use" << ',' << "not_use" << ',';
  logger << "cur_yaw" << ',' << "des_yaw" << ',';
  logger << "des_pos_x" << ',' << "des_pos_y" << ',' << "des_pos_z" << ',';
  logger << "des_vel_x" << ',' << "des_vel_y" << ',' << "des_vel_z" << ',';
  logger << "des_acc_x" << ',' << "des_acc_y" << ',' << "des_acc_z" << std::endl;
}

void SO3ControlNode::recordLog()
{
  if (!logger.is_open())
    return;
  logger << now().nanoseconds() << ',';
  logger << cur_pos_(0) << ',' << cur_pos_(1) << ',' << cur_pos_(2) << ',';
  logger << cur_vel_(0) << ',' << cur_vel_(1) << ',' << cur_vel_(2) << ',';
  logger << cur_acc_(0) << ',' << cur_acc_(1) << ',' << cur_acc_(2) << ',';
  logger << 0.0 << ',' << 0.0 << ',';
  logger << current_yaw_ << ',' << des_yaw_ << ',';
  logger << des_pos_(0) << ',' << des_pos_(1) << ',' << des_pos_(2) << ',';
  logger << des_vel_(0) << ',' << des_vel_(1) << ',' << des_vel_(2) << ',';
  logger << des_acc_(0) << ',' << des_acc_(1) << ',' << des_acc_(2) << std::endl;
}

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SO3ControlNode>());
  rclcpp::shutdown();
  return 0;
}
