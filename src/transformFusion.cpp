#include "utility.h"

namespace maslo
{
namespace transform_fusion
{

/*//{ class TransformFusion() */
class TransformFusion : public nodelet::Nodelet {
public:
  /*//{ parameters*/
  std::string uavName;

  // Frames
  std::string lidarFrame;
  std::string baselinkFrame;
  std::string odometryFrame;
  std::string mapFrame;

  /*//}*/

  tf::StampedTransform tfLidar2Baselink;

  std::mutex mtx;

  ros::Subscriber subPreOdometry;
  ros::Subscriber subLaserOdometry;

  ros::Publisher pubFusedOdometry;
  ros::Publisher pubFusedPath;

  Eigen::Affine3f lidarOdomAffine;
  Eigen::Affine3f preOdomAffineFront;
  Eigen::Affine3f preOdomAffineBack;


  double                    lidarOdomTime = -1;
  deque<nav_msgs::Odometry> preOdomQueue;

  bool is_initialized_ = false;

public:
  /*//{ onInit() */
  virtual void onInit() {

    ros::NodeHandle nh = nodelet::Nodelet::getMTPrivateNodeHandle();

    /*//{ load parameters */
    mrs_lib::ParamLoader pl(nh, "TransformFusion");

    pl.loadParam("uavName", uavName);

    pl.loadParam("lidarFrame", lidarFrame);
    pl.loadParam("baselinkFrame", baselinkFrame);
    pl.loadParam("odometryFrame", odometryFrame);
    pl.loadParam("mapFrame", mapFrame);

    if (!pl.loadedSuccessfully()) {
      ROS_ERROR("[TransformFusion]: Could not load all parameters!");
      ros::shutdown();
    }
    /*//}*/

    subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("odom_mapping_in", 5, &TransformFusion::lidarOdometryHandler, this, ros::TransportHints().tcpNoDelay());
    subPreOdometry =
        nh.subscribe<nav_msgs::Odometry>("odom_pre_incremental_in", 2000, &TransformFusion::preOdometryHandler, this, ros::TransportHints().tcpNoDelay());

    pubFusedOdometry = nh.advertise<nav_msgs::Odometry>("fused_odometry_out", 2000);
    pubFusedPath     = nh.advertise<nav_msgs::Path>("fused_path_out", 1);

    // get static transform from lidar to baselink
    if (lidarFrame != baselinkFrame) {

      tf::TransformListener tfListener;

      bool tfFound = false;
      while (!tfFound) {
        try {
          ROS_WARN_THROTTLE(3.0, "[TransformFusion]: Waiting for transform from: %s, to: %s.", lidarFrame.c_str(), baselinkFrame.c_str());
          tfListener.waitForTransform(lidarFrame, baselinkFrame, ros::Time(0), ros::Duration(3.0));
          tfListener.lookupTransform(lidarFrame, baselinkFrame, ros::Time(0), tfLidar2Baselink);
          tfFound = true;
        }
        catch (tf::TransformException ex) {
          ROS_WARN_THROTTLE(3.0, "[TransformFusion]: could not find transform from: %s, to: %s.", lidarFrame.c_str(), baselinkFrame.c_str());
        }
      }

      ROS_INFO("[TransformFusion]: Found transform from: %s, to: %s.", lidarFrame.c_str(), baselinkFrame.c_str());

    } else {
      tfLidar2Baselink.setOrigin(tf::Vector3(0.0, 0.0, 0.0));
      tfLidar2Baselink.setRotation(tf::createQuaternionFromRPY(0.0, 0.0, 0.0));
    }

    ROS_INFO("\033[1;32m----> [TransformFusion]: initialized.\033[0m");
    is_initialized_ = true;
  }
  /*//}*/

  /*//{ odom2affine() */
  Eigen::Affine3f odom2affine(nav_msgs::Odometry odom) {
    double x, y, z, roll, pitch, yaw;
    x = odom.pose.pose.position.x;
    y = odom.pose.pose.position.y;
    z = odom.pose.pose.position.z;
    tf::Quaternion orientation;
    tf::quaternionMsgToTF(odom.pose.pose.orientation, orientation);
    tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
    return pcl::getTransformation(x, y, z, roll, pitch, yaw);
  }
  /*//}*/

  /*//{ lidarOdometryHandler() */
  void lidarOdometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg) {

    if (!is_initialized_) {
      return;
    }

    ROS_INFO_ONCE("[TransformFusion]: lidarOdometryCallback first callback");

    std::lock_guard<std::mutex> lock(mtx);

    lidarOdomAffine = odom2affine(*odomMsg);

    lidarOdomTime = odomMsg->header.stamp.toSec();
  }
  /*//}*/

  /*//{ preOdometryHandler() */
  void preOdometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg) {

    if (!is_initialized_) {
      return;
    }

    ROS_INFO_ONCE("[TransformFusion]: preOdometryCallback first callback");

    // TODO: publish TFs correctly as `map -> odom -> fcu` (control has to handle this by continuously republishing the reference in map frame)

    // publish tf map->odom (inverted tf-tree)
    static tf::TransformBroadcaster tfMap2Odom;
    static tf::Transform            map_to_odom = tf::Transform(tf::createQuaternionFromRPY(0, 0, 0), tf::Vector3(0, 0, 0));
    tfMap2Odom.sendTransform(tf::StampedTransform(map_to_odom.inverse(), odomMsg->header.stamp, odometryFrame, mapFrame));

    std::lock_guard<std::mutex> lock(mtx);

    preOdomQueue.push_back(*odomMsg);

    // get latest odometry (at current MAS stamp)
    if (lidarOdomTime == -1) {
      return;
    }

    while (!preOdomQueue.empty()) {
      if (preOdomQueue.front().header.stamp.toSec() <= lidarOdomTime) {
        preOdomQueue.pop_front();
      } else {
        break;
      }
    }

    if (preOdomQueue.empty()) {
      return;
    }

    const Eigen::Affine3f preOdomAffineFront = odom2affine(preOdomQueue.front());
    const Eigen::Affine3f preOdomAffineBack  = odom2affine(preOdomQueue.back());
    const Eigen::Affine3f preOdomAffineIncre = preOdomAffineFront.inverse() * preOdomAffineBack;
    const Eigen::Affine3f preOdomAffineLast  = lidarOdomAffine * preOdomAffineIncre;
    /* const Eigen::Affine3f preOdomAffineLast  = lidarOdomAffine; */
    float x, y, z, roll, pitch, yaw;
    pcl::getTranslationAndEulerAngles(preOdomAffineLast, x, y, z, roll, pitch, yaw);

    // publish latest odometry
    tf::Transform tCur;
    tCur.setOrigin(tf::Vector3(x, y, z));
    tCur.setRotation(tf::createQuaternionFromRPY(roll, pitch, yaw));
    if (lidarFrame != baselinkFrame) {
      tCur = tCur * tfLidar2Baselink;
    }
    tCur.setRotation(tCur.getRotation().normalized());

    nav_msgs::Odometry::Ptr laserOdometry = boost::make_shared<nav_msgs::Odometry>(preOdomQueue.back());
    laserOdometry->header.frame_id        = odometryFrame;
    laserOdometry->child_frame_id         = baselinkFrame;
    laserOdometry->pose.pose.position.x   = tCur.getOrigin().getX();
    laserOdometry->pose.pose.position.y   = tCur.getOrigin().getY();
    laserOdometry->pose.pose.position.z   = tCur.getOrigin().getZ();
    tf::quaternionTFToMsg(tCur.getRotation(), laserOdometry->pose.pose.orientation);

    if (std::isfinite(laserOdometry->pose.pose.orientation.x) && std::isfinite(laserOdometry->pose.pose.orientation.y) &&
        std::isfinite(laserOdometry->pose.pose.orientation.z) && std::isfinite(laserOdometry->pose.pose.orientation.w)) {
      pubFusedOdometry.publish(laserOdometry);

      // publish tf odom->fcu (inverted tf-tree)
      static tf::TransformBroadcaster tfOdom2BaseLink;
      tf::StampedTransform            odom_2_baselink = tf::StampedTransform(tCur.inverse(), odomMsg->header.stamp, baselinkFrame, odometryFrame);
      tfOdom2BaseLink.sendTransform(odom_2_baselink);

      // publish fused path
      static nav_msgs::Path::Ptr fusedPath      = boost::make_shared<nav_msgs::Path>();
      static double              last_path_time = -1;
      const double               fusedTime      = preOdomQueue.back().header.stamp.toSec();
      if (fusedTime - last_path_time > 0.1) {
        last_path_time = fusedTime;
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp    = preOdomQueue.back().header.stamp;
        pose_stamped.header.frame_id = odometryFrame;
        pose_stamped.pose            = laserOdometry->pose.pose;
        fusedPath->poses.push_back(pose_stamped);
        while (!fusedPath->poses.empty() && fusedPath->poses.front().header.stamp.toSec() < lidarOdomTime - 1.0) {
          fusedPath->poses.erase(fusedPath->poses.begin());
        }
        if (pubFusedPath.getNumSubscribers() > 0) {
          fusedPath->header.stamp    = preOdomQueue.back().header.stamp;
          fusedPath->header.frame_id = odometryFrame;
          pubFusedPath.publish(fusedPath);
        }
      }
    }
  }
  /*//}*/
};
/*//}*/

}  // namespace transform_fusion
}  // namespace maslo

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(maslo::transform_fusion::TransformFusion, nodelet::Nodelet)
