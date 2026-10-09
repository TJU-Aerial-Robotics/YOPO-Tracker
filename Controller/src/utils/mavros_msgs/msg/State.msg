# Current autopilot state (minimal subset of the upstream mavros_msgs/State)
std_msgs/Header header
bool connected
bool armed
bool guided
bool manual_input
string mode
uint8 system_status
