#include "so3_control/NetworkControl.h"

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto controller = std::make_shared<NetworkControl>();
    controller->init();
    // Multi-threaded: the takeoff/land worker blocks while the executor keeps serving callbacks.
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(controller);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
