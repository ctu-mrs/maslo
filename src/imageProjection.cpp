#include "utility.h"

#include <opencv2/opencv.hpp>

struct PointXYZIRT
{
  PCL_ADD_POINT4D
  PCL_ADD_INTENSITY;
  uint32_t t;
  uint8_t  ring;
  uint32_t range;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRT,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(std::uint32_t, t, t)(std::uint8_t, ring,
                                                                                                                            ring)(std::uint32_t, range, range))

namespace maslo
{
namespace image_projection
{

const int queueLength = 2000;

/*//{ class ImageProjection() */
class ImageProjection : public nodelet::Nodelet {
private:
  std::mutex odoLock;

  ros::Subscriber subLaserCloud;
  ros::Publisher  pubLaserCloud;

  ros::Publisher pubExtractedCloud;
  ros::Publisher pubLaserCloudInfo;

  ros::Subscriber                subOdom;
  std::deque<nav_msgs::Odometry> odomQueue;

  std::deque<sensor_msgs::PointCloud2> cloudQueue;
  sensor_msgs::PointCloud2             currentCloudMsg;

  bool            firstPointFlag;
  Eigen::Affine3f transStartInverse;

  pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;
  pcl::PointCloud<PointType>::Ptr   fullCloud;
  pcl::PointCloud<PointType>::Ptr   extractedCloud;

  int     deskewFlag;
  cv::Mat rangeMat;

  bool  odomDeskewFlag;
  float odomIncreX;
  float odomIncreY;
  float odomIncreZ;

  maslo::cloud_info::Ptr cloudInfo = boost::make_shared<maslo::cloud_info>();
  double                 timeScanCur;
  double                 timeScanEnd;
  std_msgs::Header       cloudHeader;

  /*//{ parameters */

  std::string uavName;

  // Frames
  std::string lidarFrame;
  std::string baselinkFrame;

  // LIDAR
  int    numberOfRings;
  int    samplesPerRing;
  string timeField;
  int    downsampleRate;
  float  lidarMinRange;
  float  lidarMaxRange;

  /*//}*/

public:
  /*//{ onInit() */
  virtual void onInit() {

    ROS_INFO("[ImageProjection]: initializing");

    deskewFlag = 0;

    ros::NodeHandle nh = nodelet::Nodelet::getMTPrivateNodeHandle();

    /*//{ load parameters */
    mrs_lib::ParamLoader pl(nh, "ImageProjection");

    pl.loadParam("uavName", uavName);

    pl.loadParam("lidarFrame", lidarFrame);
    pl.loadParam("baselinkFrame", baselinkFrame);

    pl.loadParam("numberOfRings", numberOfRings);
    pl.loadParam("samplesPerRing", samplesPerRing);
    pl.loadParam("timeField", timeField, std::string("t"));
    pl.loadParam("downsampleRate", downsampleRate, 1);
    pl.loadParam("lidarMinRange", lidarMinRange, 0.1f);
    pl.loadParam("lidarMaxRange", lidarMaxRange, 1000.0f);

    if (!pl.loadedSuccessfully()) {
      ROS_ERROR("[ImageProjection]: Could not load all parameters!");
      ros::shutdown();
    }

    /*//}*/

    subOdom       = nh.subscribe<nav_msgs::Odometry>("odom_incremental_in", 2000, &ImageProjection::odometryHandler, this, ros::TransportHints().tcpNoDelay());
    subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("cloud_in", 5, &ImageProjection::cloudHandler, this, ros::TransportHints().tcpNoDelay());

    pubExtractedCloud = nh.advertise<sensor_msgs::PointCloud2>("maslo/deskew/deskewed_cloud_out", 1);
    pubLaserCloudInfo = nh.advertise<maslo::cloud_info>("maslo/deskew/deskewed_cloud_info_out", 1);

    allocateMemory();
    resetParameters();

    /* pcl::console::setVerbosityLevel(pcl::console::L_ERROR); */

    ROS_INFO("\033[1;32m----> [Image Projection]: initialized.\033[0m");
  }
  /*//}*/

  /*//{ allocateMemory() */
  void allocateMemory() {
    laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
    fullCloud.reset(new pcl::PointCloud<PointType>());
    extractedCloud.reset(new pcl::PointCloud<PointType>());

    fullCloud->points.resize(numberOfRings * samplesPerRing);

    cloudInfo->startRingIndex.assign(numberOfRings, 0);
    cloudInfo->endRingIndex.assign(numberOfRings, 0);

    cloudInfo->pointColInd.assign(numberOfRings * samplesPerRing, 0);
    cloudInfo->pointRange.assign(numberOfRings * samplesPerRing, 0);

    resetParameters();
  }
  /*//}*/

  /*//{ resetParameters() */
  void resetParameters() {
    laserCloudIn->clear();
    extractedCloud->clear();
    // reset range matrix for range image projection
    rangeMat = cv::Mat(numberOfRings, samplesPerRing, CV_32F, cv::Scalar::all(FLT_MAX));

    firstPointFlag = true;
    odomDeskewFlag = false;
  }
  /*//}*/

  /*//{ odometryHandler() */
  void odometryHandler(const nav_msgs::Odometry::ConstPtr &odometryMsg) {

    ROS_INFO_ONCE("[ImageProjection]: odometryHandler first callback");

    std::lock_guard<std::mutex> lock2(odoLock);
    odomQueue.push_back(*odometryMsg);
  }
  /*//}*/

  /*//{ cloudHandler() */
  void cloudHandler(const sensor_msgs::PointCloud2::ConstPtr &laserCloudMsg) {

    ROS_INFO_ONCE("[ImageProjection]: cloudHandler first callback");
    if (!cachePointCloud(laserCloudMsg)) {
      return;
    }

    if (!deskewInfo()) {
      return;
    }

    projectPointCloud();

    cloudExtraction();

    publishClouds();

    resetParameters();
  }
  /*//}*/

  /*//{ cachePointCloud() */
  bool cachePointCloud(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg) {
    // cache point cloud
    cloudQueue.push_back(*laserCloudMsg);
    if (cloudQueue.size() <= 2) {
      return false;
    }

    // convert cloud
    // why front of queue? this causes always the oldest point cloud to be processed i.e. delay of 200 ms?
    /* currentCloudMsg = std::move(cloudQueue.front()); */
    /* cloudQueue.pop_front(); */
    currentCloudMsg = std::move(cloudQueue.back());
    cloudQueue.pop_back();
    pcl::fromROSMsg(currentCloudMsg, *laserCloudIn);

    // get timestamp
    cloudHeader = currentCloudMsg.header;
    timeScanCur = cloudHeader.stamp.toSec();
    /* timeScanEnd = timeScanCur + laserCloudIn->points.back().time; // Velodyne */
    /* timeScanEnd = timeScanCur + (float)laserCloudIn->points.back().t / 1.0e9;  // Ouster */
    timeScanEnd = timeScanCur;  // sim

    // check dense flag
    if (!laserCloudIn->is_dense) {
      removeNaNFromPointCloud(laserCloudIn, laserCloudIn);
      /* ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!"); */
      /* ros::shutdown(); */
    }

    // check ring channel
    static int ringFlag = 0;
    if (ringFlag == 0) {
      ringFlag = -1;
      for (auto &field : currentCloudMsg.fields) {
        if (field.name == "ring") {
          ringFlag = 1;
          break;
        }
      }
      if (ringFlag == -1) {
        ROS_ERROR("Point cloud ring channel not available, please configure your point cloud data!");
        ros::shutdown();
      }
    }

    // check point time
    if (deskewFlag == 0) {
      deskewFlag = -1;
      for (auto &field : currentCloudMsg.fields) {
        if (field.name == timeField) {
          deskewFlag = 1;
          break;
        }
      }
      if (deskewFlag == -1)
        ROS_WARN("Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
    }

    return true;
  }
  /*//}*/

  /*//{ deskewInfo() */
  bool deskewInfo() {

    odomDeskewInfo();

    return true;
  }
  /*//}*/

  /*//{ odomDeskewInfo() */
  void odomDeskewInfo() {
    std::lock_guard<std::mutex> lock2(odoLock);
    cloudInfo->odomAvailable = false;

    while (!odomQueue.empty()) {
      if (odomQueue.front().header.stamp.toSec() < timeScanCur - 0.01) {
        odomQueue.pop_front();
      } else {
        break;
      }
    }

    if (odomQueue.empty()) {
      return;
    }

    if (odomQueue.front().header.stamp.toSec() > timeScanCur) {
      return;
    }

    // get start odometry at the beinning of the scan
    nav_msgs::Odometry startOdomMsg;

    for (int i = 0; i < (int)odomQueue.size(); ++i) {
      startOdomMsg = odomQueue[i];

      if (ROS_TIME(&startOdomMsg) < timeScanCur) {
        continue;
      } else {
        break;
      }
    }

    tf::Quaternion orientation;
    tf::quaternionMsgToTF(startOdomMsg.pose.pose.orientation, orientation);

    double roll, pitch, yaw;
    tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

    // Initial guess used in mapOptimization
    cloudInfo->initialGuessX     = startOdomMsg.pose.pose.position.x;
    cloudInfo->initialGuessY     = startOdomMsg.pose.pose.position.y;
    cloudInfo->initialGuessZ     = startOdomMsg.pose.pose.position.z;
    cloudInfo->initialGuessRoll  = roll;
    cloudInfo->initialGuessPitch = pitch;
    cloudInfo->initialGuessYaw   = yaw;

    cloudInfo->odomAvailable = true;
  }
  /*//}*/

  /*//{ findPosition() */
  void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur) {
    *posXCur = 0;
    *posYCur = 0;
    *posZCur = 0;

    // If the sensor moves relatively slow, like walking speed, positional deskew seems to have little benefits. Thus code below is commented.

    // if (cloudInfo->odomAvailable == false || odomDeskewFlag == false)
    //     return;

    // float ratio = relTime / (timeScanEnd - timeScanCur);

    // *posXCur = ratio * odomIncreX;
    // *posYCur = ratio * odomIncreY;
    // *posZCur = ratio * odomIncreZ;
  }
  /*//}*/

  /*//{ deskewPoint() */
  PointType deskewPoint(PointType *point, double relTime) {
    return *point;
  }
  /*//}*/

  /*//{ projectPointCloud() */
  // petrlmat: this functions only copies points from one point cloud to another (of another type) when deskewing is disabled (default so far) - potential
  // performance gain if we get rid of the copy
  void projectPointCloud() {
    // range image projection
    const unsigned int cloudSize = laserCloudIn->points.size();
    for (unsigned int i = 0; i < cloudSize; i++) {
      /* ROS_WARN("(%d, %d) - xyz: (%0.2f, %0.2f, %0.2f), ring: %d", j, rowIdn, laserCloudIn->at(j, rowIdn).x, laserCloudIn->at(j, rowIdn).y, */
      /*          laserCloudIn->at(j, rowIdn).z, laserCloudIn->at(j, rowIdn).ring); */

      /* const float range = pointDistance(thisPoint); */
      const float range = laserCloudIn->points.at(i).range / 1000.0f;
      if (range < lidarMinRange || range > lidarMaxRange) {
        continue;
      }

      const int rowIdn = laserCloudIn->points.at(i).ring;
      if (rowIdn < 0 || rowIdn >= numberOfRings) {
        /* ROS_ERROR("Invalid ring: %d", rowIdn); */
        continue;
      }

      if (rowIdn % downsampleRate != 0) {
        /* ROS_ERROR("Downsampling. Throwing away row: %d", rowIdn); */
        continue;
      }

      PointType thisPoint;
      thisPoint.x         = laserCloudIn->points.at(i).x;
      thisPoint.y         = laserCloudIn->points.at(i).y;
      thisPoint.z         = laserCloudIn->points.at(i).z;
      thisPoint.intensity = laserCloudIn->points.at(i).intensity;

      // TODO: polish this monstrosity
      const float  horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;
      static float ang_res_x    = 360.0 / float(samplesPerRing);
      int          columnIdn    = -round((horizonAngle - 90.0) / ang_res_x) + samplesPerRing / 2;
      if (columnIdn >= samplesPerRing) {
        columnIdn -= samplesPerRing;
      }

      if (columnIdn < 0 || columnIdn >= samplesPerRing) {
        continue;
      }

      if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX) {
        continue;
      }

      // petrlmat: so far, we were using maslo without deskewing
      /* thisPoint = deskewPoint(&thisPoint, laserCloudIn->at(j, rowIdn).time); // Velodyne */
      /* thisPoint = deskewPoint(&thisPoint, (float)laserCloudIn->at(j, rowIdn).t / 1000000000.0);  // Ouster */

      rangeMat.at<float>(rowIdn, columnIdn) = range;

      const int index          = columnIdn + rowIdn * samplesPerRing;
      fullCloud->points[index] = thisPoint;
    }
  }
  /*//}*/

  /*//{ cloudExtraction() */
  void cloudExtraction() {
    int count = 0;
    // extract segmented cloud for lidar odometry
    for (int i = 0; i < numberOfRings; ++i) {
      cloudInfo->startRingIndex[i] = count - 1 + 5;

      for (int j = 0; j < samplesPerRing; ++j) {
        /* ROS_WARN("i: %d, j: %d, rangeMat: %0.10f, isFltMax: %d", i, j, rangeMat.at<float>(i,j), rangeMat.at<float>(i,j) == FLT_MAX); */
        if (rangeMat.at<float>(i, j) != FLT_MAX) {
          /* ROS_WARN("i: %d, j: %d, rangeMat: %0.10f, isFltMax: %d", i, j, rangeMat.at<float>(i,j), rangeMat.at<float>(i,j) == FLT_MAX); */
          // mark the points' column index for marking occlusion later
          cloudInfo->pointColInd[count] = j;
          // save range info
          cloudInfo->pointRange[count] = rangeMat.at<float>(i, j);
          // save extracted cloud
          extractedCloud->push_back(fullCloud->points[j + i * samplesPerRing]);
          // size of extracted cloud
          ++count;
        }
      }
      cloudInfo->endRingIndex[i] = count - 1 - 5;
    }
  }
  /*//}*/

  /*//{ publishClouds() */
  void publishClouds() {
    cloudInfo->header         = cloudHeader;
    cloudInfo->cloud_deskewed = publishCloud(&pubExtractedCloud, extractedCloud, cloudHeader.stamp, lidarFrame);
    try {
      pubLaserCloudInfo.publish(cloudInfo);
    }
    catch (...) {
      ROS_ERROR("[ImageProjection]: Exception caught during publishing topic %s.", pubLaserCloudInfo.getTopic().c_str());
    }
  }
  /*//}*/

  /*//{ removeNaNFromPointCloud() */
  void removeNaNFromPointCloud(const pcl::PointCloud<PointXYZIRT>::Ptr &cloud_in, pcl::PointCloud<PointXYZIRT>::Ptr &cloud_out) {

    if (cloud_in->is_dense) {
      cloud_out = cloud_in;
      return;
    }

    unsigned int k = 0;

    cloud_out->resize(cloud_in->size());

    for (unsigned int i = 0; i < cloud_in->size(); i++) {

      if (std::isfinite(cloud_in->at(i).x) && std::isfinite(cloud_in->at(i).y) && std::isfinite(cloud_in->at(i).z)) {
        cloud_out->at(k++) = cloud_in->at(i);
      }
    }

    cloud_out->header   = cloud_in->header;
    cloud_out->is_dense = true;

    if (cloud_in->size() != k) {
      cloud_out->resize(k);
    }
  }
  /*//}*/
};
/*//}*/

}  // namespace image_projection
}  // namespace maslo

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(maslo::image_projection::ImageProjection, nodelet::Nodelet)
