#ifndef MAVROS_INTERFACE_H_
#define MAVROS_INTERFACE_H_

#include <rclcpp/rclcpp.hpp>
#include <string>

#include <mavros_msgs/msg/attitude_target.hpp>
#include <mavros_msgs/msg/state.hpp>
#include <mavros_msgs/srv/set_mode.hpp>
#include <mavros_msgs/srv/command_bool.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

class Mavros_Interface
{
public:
    // The interface borrows the owning node; all mavros topics/services are absolute names.
    void init(rclcpp::Node *node)
    {
        node_ = node;
        const std::string base_name = "/mavros";

        att_target_pub_ = node_->create_publisher<mavros_msgs::msg::AttitudeTarget>(
            base_name + "/setpoint_raw/attitude", 10);
        state_sub_ = node_->create_subscription<mavros_msgs::msg::State>(
            base_name + "/state", 10,
            std::bind(&Mavros_Interface::state_cb, this, std::placeholders::_1));
        set_mode_client_ = node_->create_client<mavros_msgs::srv::SetMode>(base_name + "/set_mode");
        arm_disarm_client_ = node_->create_client<mavros_msgs::srv::CommandBool>(base_name + "/cmd/arming");
    }

    void state_cb(const mavros_msgs::msg::State::SharedPtr state_data)
    {
        has_armed_ = state_data->armed;
        offboard_enabled_ = (state_data->mode == "OFFBOARD");
    }

    void get_status(bool &arm_state, bool &offboard_enabled)
    {
        arm_state = has_armed_;
        offboard_enabled = offboard_enabled_;
    }

    bool set_arm_and_offboard()
    {
        rclcpp::Rate rate(1.0);
        int try_times = 0;
        while (rclcpp::ok() && (!offboard_enabled_ || !has_armed_))
        {
            if (offboard_enabled_)
            {
                auto req = std::make_shared<mavros_msgs::srv::CommandBool::Request>();
                req->value = true;
                if (call_service(arm_disarm_client_, req))
                    RCLCPP_INFO(node_->get_logger(), "vehicle ARMED");
                if (++try_times >= 3)
                {
                    RCLCPP_ERROR(node_->get_logger(), "try 3 times, cannot arm uav, give up!");
                    return false;
                }
            }
            else
            {
                RCLCPP_INFO(node_->get_logger(), "not in OFFBOARD mode");
                auto req = std::make_shared<mavros_msgs::srv::SetMode::Request>();
                req->base_mode = 0;
                req->custom_mode = "OFFBOARD";
                if (!call_service(set_mode_client_, req))
                    return false;
                RCLCPP_INFO(node_->get_logger(), "switch to OFFBOARD mode");
            }
            rate.sleep();
        }
        return true;
    }

    bool set_disarm()
    {
        rclcpp::Rate rate(1.0);
        while (rclcpp::ok() && has_armed_)
        {
            auto req = std::make_shared<mavros_msgs::srv::CommandBool::Request>();
            req->value = false;
            if (!call_service(arm_disarm_client_, req))
                return false;
            RCLCPP_INFO(node_->get_logger(), "vehicle DISARMED");
            rate.sleep();
        }
        return true;
    }

    void pub_att_thrust_cmd(const Eigen::Quaterniond &q_d, const double &thrust_d)
    {
        /* mavros uses a NED frame; convert here instead of patching mavros. */
        mavros_msgs::msg::AttitudeTarget at_cmd;
        at_cmd.header.stamp = node_->now();
        at_cmd.type_mask = at_cmd.IGNORE_ROLL_RATE | at_cmd.IGNORE_PITCH_RATE | at_cmd.IGNORE_YAW_RATE;
        at_cmd.thrust = static_cast<float>(thrust_d);
        at_cmd.orientation.w = q_d.w();
        at_cmd.orientation.x = q_d.x();
        at_cmd.orientation.y = -q_d.y();
        at_cmd.orientation.z = -q_d.z();
        att_target_pub_->publish(at_cmd);
    }

    // for simulation
    void set_arm_and_offboard_manually()
    {
        has_armed_ = true;
        offboard_enabled_ = true;
    }

    void set_disarm_manually() { has_armed_ = false; }

private:
    // Blocking call from a worker thread; the executor spins the node meanwhile.
    template <typename ClientT, typename RequestT>
    bool call_service(const ClientT &client, const RequestT &req)
    {
        if (!client->wait_for_service(std::chrono::seconds(1)))
            return false;
        auto future = client->async_send_request(req);
        if (future.wait_for(std::chrono::seconds(2)) != std::future_status::ready)
        {
            // Otherwise the request stays in the client's pending map forever.
            client->remove_pending_request(future);
            return false;
        }
        return true;
    }

    rclcpp::Node *node_{nullptr};
    rclcpp::Publisher<mavros_msgs::msg::AttitudeTarget>::SharedPtr att_target_pub_;
    rclcpp::Subscription<mavros_msgs::msg::State>::SharedPtr state_sub_;
    rclcpp::Client<mavros_msgs::srv::SetMode>::SharedPtr set_mode_client_;
    rclcpp::Client<mavros_msgs::srv::CommandBool>::SharedPtr arm_disarm_client_;

    std::atomic<bool> has_armed_{false};
    std::atomic<bool> offboard_enabled_{false};
};

#endif
