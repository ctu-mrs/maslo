#ifndef UTILITY_H
#define UTILITY_H

#include <ros/ros.h>
#include <nodelet/nodelet.h>

#include <std_msgs/Header.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Float64MultiArray.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl_conversions/pcl_conversions.h>

#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>

#include <mrs_lib/param_loader.h>

#include <mrs_msgs/Float64ArrayStamped.h>

#include "maslo/cloud_info.h"

using namespace std;

typedef pcl::PointXYZI PointType;

/*//{ publishCloud() */
sensor_msgs::PointCloud2 publishCloud(ros::Publisher *thisPub, pcl::PointCloud<PointType>::Ptr thisCloud, ros::Time thisStamp, std::string thisFrame) {

  sensor_msgs::PointCloud2::Ptr tempCloud = boost::make_shared<sensor_msgs::PointCloud2>();
  pcl::toROSMsg(*thisCloud, *tempCloud);
  tempCloud->header.stamp    = thisStamp;
  tempCloud->header.frame_id = thisFrame;

  if (thisPub->getNumSubscribers() > 0) {
    try {
      thisPub->publish(tempCloud);
    }
    catch (...) {
      ROS_ERROR("[MAS-LO]: Exception caught during publishing topic %s.", thisPub->getTopic().c_str());
    }
  }
  return *tempCloud;
}

template <typename T>
double ROS_TIME(T msg) {
  return msg->header.stamp.toSec();
}
/*//}*/

/*//{ pointDistance() */
float pointDistance(PointType p) {
  return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

float pointDistance(PointType p1, PointType p2) {
  return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z));
}
/*//}*/

/*//{ findLidar2BaselinkTf() */
void findLidar2BaselinkTf(const string &lidarFrame, const string &baselinkFrame, Eigen::Matrix3d &extRot, Eigen::Quaterniond &extQRPY,
                          tf::StampedTransform &tfLidar2Baselink) {

  tf::TransformListener tfListener;
  /*//{ find lidar -> base_link transform */
  if (lidarFrame != baselinkFrame) {


    bool tf_found = false;
    while (!tf_found) {
      try {
        tfListener.waitForTransform(lidarFrame, baselinkFrame, ros::Time(0), ros::Duration(3.0));
        tfListener.lookupTransform(lidarFrame, baselinkFrame, ros::Time(0), tfLidar2Baselink);
        tf_found = true;
      }
      catch (tf::TransformException ex) {
        ROS_WARN_THROTTLE(3.0, "Waiting for transform from: %s, to: %s.", lidarFrame.c_str(), baselinkFrame.c_str());
      }
    }

    ROS_INFO("Found transform from: %s, to: %s.", lidarFrame.c_str(), baselinkFrame.c_str());

  } else {

    tfLidar2Baselink.setOrigin(tf::Vector3(0.0, 0.0, 0.0));
    tfLidar2Baselink.setRotation(tf::createQuaternionFromRPY(0.0, 0.0, 0.0));
  }
  /*//}*/

  /*//}*/

  ROS_INFO("Transform from: %s, to: %s.", lidarFrame.c_str(), baselinkFrame.c_str());
  ROS_INFO("   xyz: (%0.1f, %0.1f, %0.1f); xyzq: (%0.1f, %0.1f, %0.1f, %0.1f)", tfLidar2Baselink.getOrigin().x(), tfLidar2Baselink.getOrigin().y(),
           tfLidar2Baselink.getOrigin().z(), tfLidar2Baselink.getRotation().x(), tfLidar2Baselink.getRotation().y(), tfLidar2Baselink.getRotation().z(),
           tfLidar2Baselink.getRotation().w());
}
/*//}*/

#endif  // UTILITY_H
