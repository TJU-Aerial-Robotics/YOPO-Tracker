#include <Eigen/Geometry>
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <quadrotor_msgs/msg/so3_command.hpp>
#include <quadrotor_simulator/Quadrotor.h>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/bool.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <uav_utils/geometry_utils.h>
#include <ament_index_cpp/get_package_share_directory.hpp>

typedef struct _Control
{
  double rpm[4];
} Control;

typedef struct _Command
{
  float force[3];
  float qx, qy, qz, qw;
  float kR[3];
  float kOm[3];
  float corrections[3];
  float current_yaw;
  bool  use_external_yaw;
} Command;

typedef struct _Disturbance
{
  Eigen::Vector3d f;
  Eigen::Vector3d m;
} Disturbance;

static Command     command;
static Disturbance disturbance;
static bool        respawn_flag = false;  // 主循环里消费, 与 spin_some 同线程, 不需要加锁

void stateToOdomMsg(const QuadrotorSimulator::Quadrotor::State& state,
                    nav_msgs::msg::Odometry&                         odom);
void quadToImuMsg(const QuadrotorSimulator::Quadrotor& quad,
                  sensor_msgs::msg::Imu&                    imu);
void odomToTF(const nav_msgs::msg::Odometry& odom_msg,
              geometry_msgs::msg::TransformStamped& transformStamped);
void odomToMesh(const nav_msgs::msg::Odometry& odom_msg,
                visualization_msgs::msg::Marker& meshROS, bool is_tracker);

static Control
getControl(const QuadrotorSimulator::Quadrotor& quad, const Command& cmd)
{
  const double _kf = quad.getPropellerThrustCoefficient();
  const double _km = quad.getPropellerMomentCoefficient();
  const double kf  = _kf - cmd.corrections[0];
  const double km  = _km / _kf * kf;

  const double          d       = quad.getArmLength();
  const Eigen::Matrix3f J       = quad.getInertia().cast<float>();
  const float           I[3][3] = { { J(0, 0), J(0, 1), J(0, 2) },
                          { J(1, 0), J(1, 1), J(1, 2) },
                          { J(2, 0), J(2, 1), J(2, 2) } };
  const QuadrotorSimulator::Quadrotor::State state = quad.getState();

  // Rotation, may use external yaw
  Eigen::Vector3d _ypr = uav_utils::R_to_ypr(state.R);
  Eigen::Vector3d ypr  = _ypr;
  if (cmd.use_external_yaw)
    ypr[0] = cmd.current_yaw;
  Eigen::Matrix3d R;
  R = Eigen::AngleAxisd(ypr[0], Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(ypr[1], Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(ypr[2], Eigen::Vector3d::UnitX());
  float R11 = R(0, 0);
  float R12 = R(0, 1);
  float R13 = R(0, 2);
  float R21 = R(1, 0);
  float R22 = R(1, 1);
  float R23 = R(1, 2);
  float R31 = R(2, 0);
  float R32 = R(2, 1);
  float R33 = R(2, 2);
  /*
    float R11 = state.R(0,0);
    float R12 = state.R(0,1);
    float R13 = state.R(0,2);
    float R21 = state.R(1,0);
    float R22 = state.R(1,1);
    float R23 = state.R(1,2);
    float R31 = state.R(2,0);
    float R32 = state.R(2,1);
    float R33 = state.R(2,2);
  */
  float Om1 = state.omega(0);
  float Om2 = state.omega(1);
  float Om3 = state.omega(2);

  float Rd11 =
    cmd.qw * cmd.qw + cmd.qx * cmd.qx - cmd.qy * cmd.qy - cmd.qz * cmd.qz;
  float Rd12 = 2 * (cmd.qx * cmd.qy - cmd.qw * cmd.qz);
  float Rd13 = 2 * (cmd.qx * cmd.qz + cmd.qw * cmd.qy);
  float Rd21 = 2 * (cmd.qx * cmd.qy + cmd.qw * cmd.qz);
  float Rd22 =
    cmd.qw * cmd.qw - cmd.qx * cmd.qx + cmd.qy * cmd.qy - cmd.qz * cmd.qz;
  float Rd23 = 2 * (cmd.qy * cmd.qz - cmd.qw * cmd.qx);
  float Rd31 = 2 * (cmd.qx * cmd.qz - cmd.qw * cmd.qy);
  float Rd32 = 2 * (cmd.qy * cmd.qz + cmd.qw * cmd.qx);
  float Rd33 =
    cmd.qw * cmd.qw - cmd.qx * cmd.qx - cmd.qy * cmd.qy + cmd.qz * cmd.qz;

  float Psi = 0.5f * (3.0f - (Rd11 * R11 + Rd21 * R21 + Rd31 * R31 +
                              Rd12 * R12 + Rd22 * R22 + Rd32 * R32 +
                              Rd13 * R13 + Rd23 * R23 + Rd33 * R33));

  float force = 0;
  if (Psi < 1.0f) // Position control stability guaranteed only when Psi < 1
    force = cmd.force[0] * R13 + cmd.force[1] * R23 + cmd.force[2] * R33;

  float eR1 = 0.5f * (R12 * Rd13 - R13 * Rd12 + R22 * Rd23 - R23 * Rd22 +
                      R32 * Rd33 - R33 * Rd32);
  float eR2 = 0.5f * (R13 * Rd11 - R11 * Rd13 - R21 * Rd23 + R23 * Rd21 -
                      R31 * Rd33 + R33 * Rd31);
  float eR3 = 0.5f * (R11 * Rd12 - R12 * Rd11 + R21 * Rd22 - R22 * Rd21 +
                      R31 * Rd32 - R32 * Rd31);

  float eOm1 = Om1;
  float eOm2 = Om2;
  float eOm3 = Om3;

  float in1 = Om2 * (I[2][0] * Om1 + I[2][1] * Om2 + I[2][2] * Om3) -
              Om3 * (I[1][0] * Om1 + I[1][1] * Om2 + I[1][2] * Om3);
  float in2 = Om3 * (I[0][0] * Om1 + I[0][1] * Om2 + I[0][2] * Om3) -
              Om1 * (I[2][0] * Om1 + I[2][1] * Om2 + I[2][2] * Om3);
  float in3 = Om1 * (I[1][0] * Om1 + I[1][1] * Om2 + I[1][2] * Om3) -
              Om2 * (I[0][0] * Om1 + I[0][1] * Om2 + I[0][2] * Om3);
  /*
    // Robust Control --------------------------------------------
    float c2       = 0.6;
    float epsilonR = 0.04;
    float deltaR   = 0.1;
    float eA1 = eOm1 + c2 * 1.0/I[0][0] * eR1;
    float eA2 = eOm2 + c2 * 1.0/I[1][1] * eR2;
    float eA3 = eOm3 + c2 * 1.0/I[2][2] * eR3;
    float neA = sqrt(eA1*eA1 + eA2*eA2 + eA3*eA3);
    float muR1 = -deltaR*deltaR * eA1 / (deltaR * neA + epsilonR);
    float muR2 = -deltaR*deltaR * eA2 / (deltaR * neA + epsilonR);
    float muR3 = -deltaR*deltaR * eA3 / (deltaR * neA + epsilonR);
    // Robust Control --------------------------------------------
  */
  float M1 = -cmd.kR[0] * eR1 - cmd.kOm[0] * eOm1 + in1; // - I[0][0]*muR1;
  float M2 = -cmd.kR[1] * eR2 - cmd.kOm[1] * eOm2 + in2; // - I[1][1]*muR2;
  float M3 = -cmd.kR[2] * eR3 - cmd.kOm[2] * eOm3 + in3; // - I[2][2]*muR3;

  float w_sq[4];
  w_sq[0] = force / (4 * kf) - M2 / (2 * d * kf) + M3 / (4 * km);
  w_sq[1] = force / (4 * kf) + M2 / (2 * d * kf) + M3 / (4 * km);
  w_sq[2] = force / (4 * kf) + M1 / (2 * d * kf) - M3 / (4 * km);
  w_sq[3] = force / (4 * kf) - M1 / (2 * d * kf) - M3 / (4 * km);

  Control control;
  for (int i = 0; i < 4; i++)
  {
    if (w_sq[i] < 0)
      w_sq[i] = 0;

    control.rpm[i] = sqrtf(w_sq[i]);
  }
  return control;
}

static void
cmd_callback(const quadrotor_msgs::msg::SO3Command::ConstSharedPtr cmd)
{
  command.force[0]         = cmd->force.x;
  command.force[1]         = cmd->force.y;
  command.force[2]         = cmd->force.z;
  command.qx               = cmd->orientation.x;
  command.qy               = cmd->orientation.y;
  command.qz               = cmd->orientation.z;
  command.qw               = cmd->orientation.w;
  command.kR[0]            = cmd->k_r[0];
  command.kR[1]            = cmd->k_r[1];
  command.kR[2]            = cmd->k_r[2];
  command.kOm[0]           = cmd->k_om[0];
  command.kOm[1]           = cmd->k_om[1];
  command.kOm[2]           = cmd->k_om[2];
  command.corrections[0]   = cmd->aux.kf_correction;
  command.corrections[1]   = cmd->aux.angle_corrections[0];
  command.corrections[2]   = cmd->aux.angle_corrections[1];
  command.current_yaw      = cmd->aux.current_yaw;
  command.use_external_yaw = cmd->aux.use_external_yaw;
}

static void
force_disturbance_callback(const geometry_msgs::msg::Vector3::ConstSharedPtr f)
{
  disturbance.f(0) = f->x;
  disturbance.f(1) = f->y;
  disturbance.f(2) = f->z;
}

static void
moment_disturbance_callback(const geometry_msgs::msg::Vector3::ConstSharedPtr m)
{
  disturbance.m(0) = m->x;
  disturbance.m(1) = m->y;
  disturbance.m(2) = m->z;
}

static void
respawn_callback(const std_msgs::msg::Bool::ConstSharedPtr msg)
{
  if (msg->data)
    respawn_flag = true;
}

// 整机退回节点启动时的样子: 位置回出生点, 速度/姿态/角速度/转速全部清零
static void
respawn(QuadrotorSimulator::Quadrotor& quad, const Eigen::Vector3d& init_pos,
        const rclcpp::Logger& logger)
{
  QuadrotorSimulator::Quadrotor::State init_state;
  init_state.x         = init_pos;
  init_state.v         = Eigen::Vector3d::Zero();
  init_state.R         = Eigen::Matrix3d::Identity();
  init_state.omega     = Eigen::Vector3d::Zero();
  init_state.motor_rpm = Eigen::Array4d::Zero();
  quad.setState(init_state);

  // 缓存的还是瞬移前那条姿态指令(可能是大倾角), 不清掉会在控制器反应过来之前(一个控制周期)
  // 按旧指令积分, 刚复位就被推歪; 只覆盖力和期望姿态, kR/kOm 保留控制器给的增益
  command.force[0] = 0;
  command.force[1] = 0;
  command.force[2] = quad.getMass() * quad.getGravity();
  command.qx = command.qy = command.qz = 0;
  command.qw = 1;
  command.current_yaw = 0;
  disturbance.f.setZero();
  disturbance.m.setZero();

  RCLCPP_WARN(logger, "[quadrotor_simulator] respawn: state reset to (%.2f, %.2f, %.2f)",
              init_pos(0), init_pos(1), init_pos(2));
}

int
main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto n = rclcpp::Node::make_shared("quadrotor_simulator_so3");

  auto odom_pub = n->create_publisher<nav_msgs::msg::Odometry>("odom", 100);
  auto imu_pub  = n->create_publisher<sensor_msgs::msg::Imu>("imu", 10);
  auto mesh_pub = n->create_publisher<visualization_msgs::msg::Marker>("uav", 1);

  tf2_ros::TransformBroadcaster tf_broadcaster(n);

  // Reliable: these are commands, and ROS1 subscribed to them over TCP (tcpNoDelay).
  const auto qos = rclcpp::QoS(100);
  auto cmd_sub = n->create_subscription<quadrotor_msgs::msg::SO3Command>("cmd", qos, &cmd_callback);
  auto f_sub = n->create_subscription<geometry_msgs::msg::Vector3>(
    "force_disturbance", qos, &force_disturbance_callback);
  auto m_sub = n->create_subscription<geometry_msgs::msg::Vector3>(
    "moment_disturbance", qos, &moment_disturbance_callback);
  // 全局话题: 前导 "/" 必须保留, 否则会被解析成节点私有的 ~respawn
  auto respawn_sub = n->create_subscription<std_msgs::msg::Bool>(
    "/respawn", rclcpp::QoS(1), &respawn_callback);

  QuadrotorSimulator::Quadrotor quad;
  const double _init_x = n->declare_parameter("simulator.init_state_x", 0.0);
  const double _init_y = n->declare_parameter("simulator.init_state_y", 0.0);
  const double _init_z = n->declare_parameter("simulator.init_state_z", 1.0);
  const bool is_tracker = n->declare_parameter("is_tracker", true);

  Eigen::Vector3d position = Eigen::Vector3d(_init_x, _init_y, _init_z);
  quad.setStatePos(position);

  const double simulation_rate = n->declare_parameter("rate.simulation", 1000.0);
  assert(simulation_rate > 0);

  const double odom_rate = n->declare_parameter("rate.odom", 100.0);
  const rclcpp::Duration odom_pub_duration = rclcpp::Duration::from_seconds(1 / odom_rate);

  const std::string quad_name = n->declare_parameter("quadrotor_name", std::string("quadrotor"));

  QuadrotorSimulator::Quadrotor::State state = quad.getState();

  rclcpp::Rate  r(simulation_rate);
  const double dt = 1 / simulation_rate;

  Control control;

  nav_msgs::msg::Odometry odom_msg;
  odom_msg.header.frame_id = "world";
  odom_msg.child_frame_id  = "/" + quad_name;

  sensor_msgs::msg::Imu imu;
  imu.header.frame_id = "world";

  geometry_msgs::msg::TransformStamped transformStamped;

  visualization_msgs::msg::Marker meshROS;

  // One executor for the whole run
  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(n);

  rclcpp::Time next_odom_pub_time = n->now();
  while (rclcpp::ok())
  {
    exec.spin_some();

    if (respawn_flag)
    {
      respawn_flag = false;
      respawn(quad, position, n->get_logger());
    }

    auto last = control;
    control   = getControl(quad, command);
    for (int i = 0; i < 4; ++i)
    {
      //! @bug might have nan when the input is legal
      if (std::isnan(control.rpm[i]))
        control.rpm[i] = last.rpm[i];
    }
    quad.setInput(control.rpm[0], control.rpm[1], control.rpm[2],
                  control.rpm[3]);
    quad.setExternalForce(disturbance.f);
    quad.setExternalMoment(disturbance.m);
    quad.step(dt);

    rclcpp::Time tnow = n->now();

    if (tnow >= next_odom_pub_time)
    {
      next_odom_pub_time = next_odom_pub_time + odom_pub_duration;
      odom_msg.header.stamp = tnow;
      state                 = quad.getState();
      stateToOdomMsg(state, odom_msg);
      quadToImuMsg(quad, imu);
      imu.header.stamp = tnow;
      odomToTF(odom_msg, transformStamped);
      odom_pub->publish(odom_msg);
      imu_pub->publish(imu);
      if (is_tracker) {
        tf_broadcaster.sendTransform(transformStamped);
      }
      if (mesh_pub->get_subscription_count() > 0) {
        odomToMesh(odom_msg, meshROS, is_tracker);
        mesh_pub->publish(meshROS);
      }
    }

    r.sleep();
  }

  rclcpp::shutdown();
  return 0;
}

void
stateToOdomMsg(const QuadrotorSimulator::Quadrotor::State& state,
               nav_msgs::msg::Odometry&                         odom)
{
  odom.pose.pose.position.x = state.x(0);
  odom.pose.pose.position.y = state.x(1);
  odom.pose.pose.position.z = state.x(2);

  Eigen::Quaterniond q(state.R);
  odom.pose.pose.orientation.x = q.x();
  odom.pose.pose.orientation.y = q.y();
  odom.pose.pose.orientation.z = q.z();
  odom.pose.pose.orientation.w = q.w();

  odom.twist.twist.linear.x = state.v(0);
  odom.twist.twist.linear.y = state.v(1);
  odom.twist.twist.linear.z = state.v(2);

  odom.twist.twist.angular.x = state.omega(0);
  odom.twist.twist.angular.y = state.omega(1);
  odom.twist.twist.angular.z = state.omega(2);
}

void
quadToImuMsg(const QuadrotorSimulator::Quadrotor& quad, sensor_msgs::msg::Imu& imu)

{
  QuadrotorSimulator::Quadrotor::State state = quad.getState();
  Eigen::Quaterniond                   q(state.R);
  imu.orientation.x = q.x();
  imu.orientation.y = q.y();
  imu.orientation.z = q.z();
  imu.orientation.w = q.w();

  imu.angular_velocity.x = state.omega(0);
  imu.angular_velocity.y = state.omega(1);
  imu.angular_velocity.z = state.omega(2);

  imu.linear_acceleration.x = quad.getAcc()[0];
  imu.linear_acceleration.y = quad.getAcc()[1];
  imu.linear_acceleration.z = quad.getAcc()[2];
}


void 
odomToTF(const nav_msgs::msg::Odometry& odom_msg, geometry_msgs::msg::TransformStamped& transformStamped) {
  transformStamped.header.stamp = odom_msg.header.stamp;
  transformStamped.header.frame_id = "world";
  transformStamped.child_frame_id = "odom";

  transformStamped.transform.translation.x = odom_msg.pose.pose.position.x;
  transformStamped.transform.translation.y = odom_msg.pose.pose.position.y;
  transformStamped.transform.translation.z = odom_msg.pose.pose.position.z;

  transformStamped.transform.rotation = odom_msg.pose.pose.orientation;
}

void 
odomToMesh(const nav_msgs::msg::Odometry& odom_msg, visualization_msgs::msg::Marker& meshROS,
           bool is_tracker) {
  meshROS.mesh_resource = "file://" + ament_index_cpp::get_package_share_directory("so3_quadrotor_simulator") + "/config/uav.dae";
  meshROS.mesh_use_embedded_materials = true;

  meshROS.header = odom_msg.header;
  meshROS.header.frame_id = "world";

  meshROS.ns     = "mesh";
  meshROS.id     = 0;
  meshROS.type   = visualization_msgs::msg::Marker::MESH_RESOURCE;
  meshROS.action = visualization_msgs::msg::Marker::ADD;

  meshROS.pose = odom_msg.pose.pose;

  meshROS.scale.x = 2.0;
  meshROS.scale.y = 2.0;
  meshROS.scale.z = 2.0;
  meshROS.color.a = 1.0;
  if (is_tracker) {   // tracker 黑色, target 红色
    meshROS.color.r = 0.0;
    meshROS.color.g = 0.0;
    meshROS.color.b = 0.0;
  } else {
    meshROS.color.r = 1.0;
    meshROS.color.g = 0.0;
    meshROS.color.b = 0.0;
  }
}