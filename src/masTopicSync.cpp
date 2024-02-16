#include <ros/ros.h>
#include <nodelet/nodelet.h>

#include <deque>

#include <mrs_lib/subscribe_handler.h>
#include <mrs_lib/publisher_handler.h>
#include <mrs_lib/param_loader.h>

#include <std_msgs/Float64.h>
#include <mrs_msgs/Float64ArrayStamped.h>

#include <mavros_msgs/ESCStatus.h>

namespace maslo
{

namespace mas_topic_sync
{

/*//{ class masTopicSync*/
class MasTopicSync : public nodelet::Nodelet {

public:
  MasTopicSync(){};

  ~MasTopicSync(){};

  virtual void onInit();

private:
  const std::string node_name_ = "MasTopicSync";

  const std::string motor_speed_topic_wo_num_ = "motor_speed/";

  bool is_initialized_ = false;

  std::string uav_name_;
  int         num_motors_;
  bool        is_simulation_;
  double      motor_slowdown_constant_;

  std::vector<mrs_lib::SubscribeHandler<std_msgs::Float64>> motor_speed_sub_list_;
  mrs_lib::SubscribeHandler<mavros_msgs::ESCStatus>         sub_esc_status_;

  void callbackMas(const std_msgs::Float64ConstPtr msg, const int motor_num);
  void callbackESCStatus(const mavros_msgs::ESCStatusConstPtr msg);

  std::vector<std::pair<double, double>> motor_speeds_;
  std::mutex                             mtx_motor_speeds_;

  mrs_lib::PublisherHandler<mrs_msgs::Float64ArrayStamped> ph_motor_speeds_sync_;
};
/*//}*/

/*//{ onInit() */
void MasTopicSync::onInit() {

  ros::NodeHandle nh = nodelet::Nodelet::getMTPrivateNodeHandle();

  ros::Time::waitForValid();

  ROS_INFO("[%s]: Initializing node", node_name_.c_str());

  // load parameters
  mrs_lib::ParamLoader param_loader(nh, getName());

  param_loader.loadParam("uav_name", uav_name_);
  param_loader.loadParam("is_simulation", is_simulation_);
  param_loader.loadParam("num_motors", num_motors_);
  param_loader.loadParam("motor_slowdown_constant", motor_slowdown_constant_);

  if (!param_loader.loadedSuccessfully()) {
    ROS_ERROR("[%s]: Could not load all non-optional parameters. Shutting down.", node_name_.c_str());
    ros::shutdown();
  }

  // | --------------- subscribers initialization --------------- |
  mrs_lib::SubscribeHandlerOptions shopts;
  shopts.nh                 = nh;
  shopts.node_name          = node_name_;
  shopts.no_message_timeout = mrs_lib::no_timeout;
  shopts.threadsafe         = true;
  shopts.autostart          = true;
  shopts.queue_size         = 10;
  shopts.transport_hints    = ros::TransportHints().tcpNoDelay();

  for (int i = 0; i < num_motors_; i++) {
    const std::string                                      topic_abs = "/" + uav_name_ + "/" + motor_speed_topic_wo_num_ + std::to_string(i);
    const std::function<void(std_msgs::Float64::ConstPtr)> cbk       = std::bind(&MasTopicSync::callbackMas, this, std::placeholders::_1, i);
    motor_speed_sub_list_.push_back(mrs_lib::SubscribeHandler<std_msgs::Float64>(shopts, topic_abs, cbk));
  }

  sub_esc_status_ = mrs_lib::SubscribeHandler<mavros_msgs::ESCStatus>(shopts, "esc_status_in", &MasTopicSync::callbackESCStatus, this);

  ph_motor_speeds_sync_ = mrs_lib::PublisherHandler<mrs_msgs::Float64ArrayStamped>(nh, "motor_speeds_sync_out", 1);

  for (int i = 0; i < num_motors_; i++) {
    motor_speeds_.push_back(std::make_pair(ros::Time::now().toSec(), 0.0));
  }

  is_initialized_ = true;
  ROS_INFO("[%s]: initialized", node_name_.c_str());
}
/*//}*/

/*//{ callbackMas() */
void MasTopicSync::callbackMas(const std_msgs::Float64ConstPtr msg_in, const int motor_num) {

  if (!is_initialized_) {
    return;
  }

  std::scoped_lock lock(mtx_motor_speeds_);
  // store motor speed in correct place of vector
  motor_speeds_[motor_num] = std::make_pair(ros::Time::now().toSec(), is_simulation_ ? msg_in->data * motor_slowdown_constant_ : msg_in->data);

  mrs_msgs::Float64ArrayStamped msg_out;
  double                        t = 0.0;
  for (int i = 0; i < num_motors_; i++) {
    t = motor_speeds_.at(i).first > t ? motor_speeds_.at(i).first : t;
    msg_out.values.push_back(motor_speeds_.at(i).second);
  }
  msg_out.header.stamp = ros::Time(t);

  ph_motor_speeds_sync_.publish(msg_out);
  ROS_INFO_ONCE("[%s]: publishing synchronized motor speeds", node_name_.c_str());
}
/*//}*/

/*//{ callbackESCStatus() */
void MasTopicSync::callbackESCStatus(const mavros_msgs::ESCStatusConstPtr msg_in) {

  if (!is_initialized_) {
    return;
  }

  if (msg_in->esc_status.size() < num_motors_) {
    return;
  }

  mrs_msgs::Float64ArrayStamped msg_out;
  double                        t = 0;
  for (size_t i = 0; i < msg_in->esc_status.size(); i++) {
    t += msg_in->esc_status[i].header.stamp.toSec();
    double rpm = msg_in->esc_status[i].rpm;

    // sometimes the RPM value is not measured and reported as 0, which would produce biased acceleration
    if (rpm == 0) {
      return;
    }
    msg_out.values.push_back(rpm);
  }
  msg_out.header.stamp = ros::Time(t / num_motors_);

  ph_motor_speeds_sync_.publish(msg_out);
  ROS_INFO_ONCE("[%s]: publishing synchronized motor speeds", node_name_.c_str());
}
/*//}*/

}  // namespace mas_topic_sync
}  // namespace maslo
#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(maslo::mas_topic_sync::MasTopicSync, nodelet::Nodelet)
