#include "utility.h"

#include <opencv2/opencv.hpp>

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>

using namespace gtsam;

using symbol_shorthand::B;  // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G;  // GPS pose
using symbol_shorthand::V;  // Vel   (xdot,ydot,zdot)
using symbol_shorthand::X;  // Pose3 (x,y,z,r,p,y)

/*
 * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
 */
struct PointXYZIRPYT
{
  PCL_ADD_POINT4D
  PCL_ADD_INTENSITY;  // preferred way of adding a XYZ+padding
  float  roll;
  float  pitch;
  float  yaw;
  double time;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // make sure our new allocators are aligned
} EIGEN_ALIGN16;                   // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRPYT, (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(float, roll, roll)(float, pitch, pitch)(
                                                     float, yaw, yaw)(double, time, time))

typedef PointXYZIRPYT PointTypePose;

namespace maslo
{
namespace map_optimization
{

/*//{ class MapOptimization() */
class MapOptimization : public nodelet::Nodelet {

public:
  /*//{ parameters */

  string uavName;

  // Frames
  std::string lidarFrame;
  std::string baselinkFrame;
  std::string odometryFrame;

  // MAS
  float preintegratedRPYWeight;

  // GPS Settings
  bool  useGpsElevation;
  float gpsCovThreshold;
  float poseCovThreshold;

  // LIDAR
  int numberOfRings;
  int samplesPerRing;

  // LOAM
  int edgeFeatureMinValidNum;
  int surfFeatureMinValidNum;

  // Voxel filter
  float mappingCornerLeafSize;
  float mappingSurfLeafSize;

  // Transformation constraints
  float z_tollerance;
  float rotation_tollerance;

  // Loop closure
  bool  loopClosureEnableFlag;
  float loopClosureFrequency;
  int   surroundingKeyframeSize;
  float historyKeyframeSearchRadius;
  float historyKeyframeSearchTimeDiff;
  int   historyKeyframeSearchNum;
  float historyKeyframeFitnessScore;

  // Global map visualization radius
  float globalMapVisualizationSearchRadius;
  float globalMapVisualizationPoseDensity;
  float globalMapVisualizationLeafSize;

  // CPU Params
  int numberOfCores;

  // Surrounding map
  float surroundingKeyframeDensity;
  float surroundingKeyframeSearchRadius;
  float surroundingkeyframeAddingDistThreshold;
  float surroundingkeyframeAddingAngleThreshold;

  // Save pcd
  bool   savePCD;
  string savePCDDirectory;
  /*//}*/

  // TF
  tf::StampedTransform tfLidar2Baselink;
  Eigen::Matrix3d      extRot;
  Eigen::Quaterniond   extQRPY;

  // gtsam
  NonlinearFactorGraph gtSAMgraph;
  Values               initialEstimate;
  Values               optimizedEstimate;
  ISAM2*               isam;
  Values               isamCurrentEstimate;
  Eigen::MatrixXd      poseCovariance;

  ros::Publisher pubLaserCloudSurround;
  ros::Publisher pubLaserOdometryGlobal;
  ros::Publisher pubLaserOdometryIncremental;
  ros::Publisher pubKeyPoses;
  ros::Publisher pubPath;

  ros::Publisher pubHistoryKeyFrames;
  ros::Publisher pubIcpKeyFrames;
  ros::Publisher pubRecentKeyFrames;
  ros::Publisher pubRecentKeyFrame;
  ros::Publisher pubCloudRegisteredRaw;
  ros::Publisher pubLoopConstraintEdge;

  ros::Subscriber subCloud;
  ros::Subscriber subGPS;
  ros::Subscriber subLoop;
  ros::Subscriber subOrientation;

  std::deque<nav_msgs::Odometry> gpsQueue;
  maslo::cloud_info              cloudInfo;

  geometry_msgs::QuaternionStamped orientationMsg;
  bool                             gotOrientation = false;

  vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
  vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;

  pcl::PointCloud<PointType>::Ptr     cloudKeyPoses3D;
  pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
  pcl::PointCloud<PointType>::Ptr     copy_cloudKeyPoses3D;
  pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;

  pcl::PointCloud<PointType>::Ptr laserCloudCornerLast;    // corner feature set from odoOptimization
  pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;      // surf feature set from odoOptimization
  pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS;  // downsampled corner featuer set from odoOptimization
  pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS;    // downsampled surf featuer set from odoOptimization

  pcl::PointCloud<PointType>::Ptr laserCloudOri;
  pcl::PointCloud<PointType>::Ptr coeffSel;

  std::vector<PointType> laserCloudOriCornerVec;  // corner point holder for parallel computation
  std::vector<PointType> coeffSelCornerVec;
  std::vector<bool>      laserCloudOriCornerFlag;
  std::vector<PointType> laserCloudOriSurfVec;  // surf point holder for parallel computation
  std::vector<PointType> coeffSelSurfVec;
  std::vector<bool>      laserCloudOriSurfFlag;

  map<int, pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>> laserCloudMapContainer;
  pcl::PointCloud<PointType>::Ptr                                        laserCloudCornerFromMap;
  pcl::PointCloud<PointType>::Ptr                                        laserCloudSurfFromMap;
  pcl::PointCloud<PointType>::Ptr                                        laserCloudCornerFromMapDS;
  pcl::PointCloud<PointType>::Ptr                                        laserCloudSurfFromMapDS;

  pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
  pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

  pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
  pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

  pcl::VoxelGrid<PointType> downSizeFilterCorner;
  pcl::VoxelGrid<PointType> downSizeFilterSurf;
  pcl::VoxelGrid<PointType> downSizeFilterICP;
  pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses;  // for surrounding key poses of scan-to-map optimization

  ros::Time timeLaserInfoStamp;
  double    timeLaserInfoCur;

  float transformTobeMapped[6];

  bool isFirstMapOptimizationSuccessful = false;

  std::mutex mtx;
  std::mutex mtxLoopInfo;

  bool    isDegenerate = false;
  cv::Mat matP;

  int laserCloudCornerFromMapDSNum = 0;
  int laserCloudSurfFromMapDSNum   = 0;
  int laserCloudCornerLastDSNum    = 0;
  int laserCloudSurfLastDSNum      = 0;

  bool                                            aLoopIsClosed = false;
  map<int, int>                                   loopIndexContainer;  // from new to old
  vector<pair<int, int>>                          loopIndexQueue;
  vector<gtsam::Pose3>                            loopPoseQueue;
  vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;
  deque<std_msgs::Float64MultiArray>              loopInfoVec;

  nav_msgs::Path::Ptr globalPath = boost::make_shared<nav_msgs::Path>();

  Eigen::Affine3f transPointAssociateToMap;
  Eigen::Affine3f incrementalOdometryAffineFront;
  Eigen::Affine3f incrementalOdometryAffineBack;

  bool isInitialized = false;

  ros::Timer timerVisualizeGlobalMap;
  ros::Timer timerLoopClosure;

public:
  /*//{ onInit() */
  virtual void onInit() {

    ros::NodeHandle nh = nodelet::Nodelet::getMTPrivateNodeHandle();

    /*//{ load parameters */
    mrs_lib::ParamLoader pl(nh, "MapOptimization");

    pl.loadParam("uavName", uavName);

    pl.loadParam("lidarFrame", lidarFrame);
    pl.loadParam("baselinkFrame", baselinkFrame);
    pl.loadParam("odometryFrame", odometryFrame);

    pl.loadParam("useGpsElevation", useGpsElevation, false);
    pl.loadParam("gpsCovThreshold", gpsCovThreshold, 2.0f);
    pl.loadParam("poseCovThreshold", poseCovThreshold, 25.0f);

    pl.loadParam("savePCD", savePCD, false);
    pl.loadParam("savePCDDirectory", savePCDDirectory, std::string("/Downloads/LOAM/"));

    pl.loadParam("numberOfRings", numberOfRings);
    pl.loadParam("samplesPerRing", samplesPerRing);

    pl.loadParam("motor_speeds/preintegratedRPYWeight", preintegratedRPYWeight, 0.01f);

    pl.loadParam("edgeFeatureMinValidNum", edgeFeatureMinValidNum, 10);
    pl.loadParam("surfFeatureMinValidNum", surfFeatureMinValidNum, 100);

    pl.loadParam("mappingCornerLeafSize", mappingCornerLeafSize, 0.2f);
    pl.loadParam("mappingSurfLeafSize", mappingSurfLeafSize, 0.2f);

    pl.loadParam("z_tollerance", z_tollerance, FLT_MAX);
    pl.loadParam("rotation_tollerance", rotation_tollerance, FLT_MAX);

    pl.loadParam("numberOfCores", numberOfCores, 4);

    pl.loadParam("surroundingkeyframeAddingDistThreshold", surroundingkeyframeAddingDistThreshold, 1.0f);
    pl.loadParam("surroundingkeyframeAddingAngleThreshold", surroundingkeyframeAddingAngleThreshold, 0.2f);
    pl.loadParam("surroundingKeyframeDensity", surroundingKeyframeDensity, 1.0f);
    pl.loadParam("surroundingKeyframeSearchRadius", surroundingKeyframeSearchRadius, 50.0f);

    pl.loadParam("loopClosureEnableFlag", loopClosureEnableFlag, false);
    pl.loadParam("loopClosureFrequency", loopClosureFrequency, 1.0f);
    pl.loadParam("surroundingKeyframeSize", surroundingKeyframeSize, 50);
    pl.loadParam("historyKeyframeSearchRadius", historyKeyframeSearchRadius, 10.0f);
    pl.loadParam("historyKeyframeSearchTimeDiff", historyKeyframeSearchTimeDiff, 30.0f);
    pl.loadParam("historyKeyframeSearchNum", historyKeyframeSearchNum, 25);
    pl.loadParam("historyKeyframeFitnessScore", historyKeyframeFitnessScore, 0.3f);

    pl.loadParam("globalMapVisualizationSearchRadius", globalMapVisualizationSearchRadius, 1e3f);
    pl.loadParam("globalMapVisualizationPoseDensity", globalMapVisualizationPoseDensity, 10.0f);
    pl.loadParam("globalMapVisualizationLeafSize", globalMapVisualizationLeafSize, 1.0f);

    if (!pl.loadedSuccessfully()) {
      ROS_ERROR("[MapOptimization]: Could not load all parameters!");
      ros::shutdown();
    }

    /*//}*/


    ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.1;
    parameters.relinearizeSkip      = 1;
    isam                            = new ISAM2(parameters);

    timerLoopClosure        = nh.createTimer(ros::Rate(loopClosureFrequency), &MapOptimization::callbackLoopClosureTimer, this);
    timerVisualizeGlobalMap = nh.createTimer(ros::Rate(0.2), &MapOptimization::callbackVisualizeGlobalMapTimer, this);

    subCloud =
        nh.subscribe<maslo::cloud_info>("maslo/mapping/cloud_info_in", 1, &MapOptimization::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
    subGPS         = nh.subscribe<nav_msgs::Odometry>("maslo/mapping/gps_in", 200, &MapOptimization::gpsHandler, this, ros::TransportHints().tcpNoDelay());
    subLoop        = nh.subscribe<std_msgs::Float64MultiArray>("maslo/loop_closure_detection_in", 1, &MapOptimization::loopInfoHandler, this,
                                                        ros::TransportHints().tcpNoDelay());
    subOrientation = nh.subscribe<geometry_msgs::QuaternionStamped>("maslo/mapping/orientation_in", 1, &MapOptimization::orientationHandler, this,
                                                                    ros::TransportHints().tcpNoDelay());

    pubKeyPoses                 = nh.advertise<sensor_msgs::PointCloud2>("maslo/mapping/trajectory_out", 1);
    pubLaserCloudSurround       = nh.advertise<sensor_msgs::PointCloud2>("maslo/mapping/map_global_out", 1);
    pubLaserOdometryGlobal      = nh.advertise<nav_msgs::Odometry>("maslo/mapping/odometry_out", 1);
    pubLaserOdometryIncremental = nh.advertise<nav_msgs::Odometry>("maslo/mapping/odometry_incremental_out", 1);
    pubPath                     = nh.advertise<nav_msgs::Path>("maslo/mapping/path_out", 1);


    pubHistoryKeyFrames   = nh.advertise<sensor_msgs::PointCloud2>("maslo/mapping/icp_loop_closure_history_cloud_out", 1);
    pubIcpKeyFrames       = nh.advertise<sensor_msgs::PointCloud2>("maslo/mapping/icp_loop_closure_corrected_cloud_out", 1);
    pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("maslo/mapping/loop_closure_constraints_out", 1);

    pubRecentKeyFrames    = nh.advertise<sensor_msgs::PointCloud2>("maslo/mapping/map_local_out", 1);
    pubRecentKeyFrame     = nh.advertise<sensor_msgs::PointCloud2>("maslo/mapping/cloud_registered_out", 1);
    pubCloudRegisteredRaw = nh.advertise<sensor_msgs::PointCloud2>("maslo/mapping/cloud_registered_raw_out", 1);

    downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
    downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
    downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
    downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity,
                                                  surroundingKeyframeDensity);  // for surrounding key poses of scan-to-map optimization

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

    allocateMemory();

    ROS_INFO("\033[1;32m----> [MapOptimization]: initialized.\033[0m");
    isInitialized = true;
  }
  /*//}*/

  /*//{ allocateMemory() */
  void allocateMemory() {
    cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
    cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
    copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
    copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

    kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
    kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

    laserCloudCornerLast.reset(new pcl::PointCloud<PointType>());    // corner feature set from odoOptimization
    laserCloudSurfLast.reset(new pcl::PointCloud<PointType>());      // surf feature set from odoOptimization
    laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>());  // downsampled corner featuer set from odoOptimization
    laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>());    // downsampled surf featuer set from odoOptimization

    laserCloudOri.reset(new pcl::PointCloud<PointType>());
    coeffSel.reset(new pcl::PointCloud<PointType>());

    laserCloudOriCornerVec.resize(numberOfRings * samplesPerRing);
    coeffSelCornerVec.resize(numberOfRings * samplesPerRing);
    laserCloudOriCornerFlag.resize(numberOfRings * samplesPerRing);
    laserCloudOriSurfVec.resize(numberOfRings * samplesPerRing);
    coeffSelSurfVec.resize(numberOfRings * samplesPerRing);
    laserCloudOriSurfFlag.resize(numberOfRings * samplesPerRing);

    std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
    std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);

    laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
    laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
    laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
    laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

    kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
    kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

    for (int i = 0; i < 6; ++i) {
      transformTobeMapped[i] = 0;
    }

    matP = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0));
  }
  /*//}*/

  /*//{ orientationHandler() */
  void orientationHandler(const geometry_msgs::QuaternionStamped::ConstPtr& msgIn) {

    if (!isInitialized) {
      return;
    }

    ROS_INFO_ONCE("[MapOptimization]: orientationHandler first callback");

    orientationMsg = *msgIn;

    gotOrientation = true;
  }
  /*//}*/

  /*//{ laserCloudInfoHandler() */
  void laserCloudInfoHandler(const maslo::cloud_info::ConstPtr& msgIn) {

    if (!isInitialized) {
      return;
    }

    if (!gotOrientation) {
      ROS_INFO_THROTTLE(1.0, "[MapOptimization]: waiting for orientation");
      return;
    }

    ROS_INFO_ONCE("[MapOptimization]: laserCloudInfoHandler first callback");
    // extract time stamp
    timeLaserInfoStamp = msgIn->header.stamp;
    timeLaserInfoCur   = msgIn->header.stamp.toSec();

    // extract info and feature cloud
    cloudInfo = *msgIn;
    pcl::fromROSMsg(msgIn->cloud_corner, *laserCloudCornerLast);
    pcl::fromROSMsg(msgIn->cloud_surface, *laserCloudSurfLast);

    std::lock_guard<std::mutex> lock(mtx);

    updateInitialGuess();

    extractSurroundingKeyFrames();

    downsampleCurrentScan();

    scan2MapOptimization();

    saveKeyFramesAndFactor();

    correctPoses();

    if (!isFirstMapOptimizationSuccessful) {
      ROS_WARN("[MapOptimization]: optimization was not successful");
      return;
    }

    publishOdometry();

    publishFrames();
  }
  /*//}*/

  /*//{ gpsHandler() */
  void gpsHandler(const nav_msgs::Odometry::ConstPtr& gpsMsg) {

    if (!isInitialized) {
      return;
    }

    gpsQueue.push_back(*gpsMsg);
  }
  /*//}*/

  /*//{ pointAssociateToMap() */
  void pointAssociateToMap(PointType const* const pi, PointType* const po) {
    po->x = transPointAssociateToMap(0, 0) * pi->x + transPointAssociateToMap(0, 1) * pi->y + transPointAssociateToMap(0, 2) * pi->z +
            transPointAssociateToMap(0, 3);
    po->y = transPointAssociateToMap(1, 0) * pi->x + transPointAssociateToMap(1, 1) * pi->y + transPointAssociateToMap(1, 2) * pi->z +
            transPointAssociateToMap(1, 3);
    po->z = transPointAssociateToMap(2, 0) * pi->x + transPointAssociateToMap(2, 1) * pi->y + transPointAssociateToMap(2, 2) * pi->z +
            transPointAssociateToMap(2, 3);
    po->intensity = pi->intensity;
  }
  /*//}*/

  /*//{ transformPointCloud() */
  pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn) {
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    const int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    const Eigen::Affine3f transCur =
        pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);

#pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i) {
      const auto& pointFrom         = cloudIn->points[i];
      cloudOut->points[i].x         = transCur(0, 0) * pointFrom.x + transCur(0, 1) * pointFrom.y + transCur(0, 2) * pointFrom.z + transCur(0, 3);
      cloudOut->points[i].y         = transCur(1, 0) * pointFrom.x + transCur(1, 1) * pointFrom.y + transCur(1, 2) * pointFrom.z + transCur(1, 3);
      cloudOut->points[i].z         = transCur(2, 0) * pointFrom.x + transCur(2, 1) * pointFrom.y + transCur(2, 2) * pointFrom.z + transCur(2, 3);
      cloudOut->points[i].intensity = pointFrom.intensity;
    }
    return cloudOut;
  }
  /*//}*/

  /*//{ pclPointTogtsamPose3() */
  gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint) {
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                        gtsam::Point3(double(thisPoint.x), double(thisPoint.y), double(thisPoint.z)));
  }
  /*//}*/

  /*//{ trans2gtsamPose() */
  gtsam::Pose3 trans2gtsamPose(float transformIn[]) {
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]), gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
  }
  /*//}*/

  /*//{ pclPointToAffine3f() */
  Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint) {
    return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
  }
  /*//}*/

  /*//{ trans2Affine3f() */
  Eigen::Affine3f trans2Affine3f(float transformIn[]) {
    return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
  }
  /*//}*/

  /*//{ trans2PointTypePose() */
  PointTypePose trans2PointTypePose(float transformIn[]) {
    PointTypePose thisPose6D;
    thisPose6D.x     = transformIn[3];
    thisPose6D.y     = transformIn[4];
    thisPose6D.z     = transformIn[5];
    thisPose6D.roll  = transformIn[0];
    thisPose6D.pitch = transformIn[1];
    thisPose6D.yaw   = transformIn[2];
    return thisPose6D;
  }
  /*//}*/

  /*//{ callbackVisualizeGlobalMapTimer() */
  void callbackVisualizeGlobalMapTimer([[maybe_unused]] const ros::TimerEvent& event) {

    if (!isInitialized) {
      return;
    }

    publishGlobalMap();

    if (!savePCD) {
      return;
    }

    cout << "****************************************************" << endl;
    cout << "Saving map to pcd files ..." << endl;
    // create directory and remove old files;
    savePCDDirectory = std::getenv("HOME") + savePCDDirectory;
    int unused       = system((std::string("exec rm -r ") + savePCDDirectory).c_str());
    unused           = system((std::string("mkdir ") + savePCDDirectory).c_str());
    // save key frame transformations
    pcl::io::savePCDFileASCII(savePCDDirectory + "trajectory.pcd", *cloudKeyPoses3D);
    pcl::io::savePCDFileASCII(savePCDDirectory + "transformations.pcd", *cloudKeyPoses6D);
    // extract global point cloud map
    pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalCornerCloudDS(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());
    for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++) {
      *globalCornerCloud += *transformPointCloud(cornerCloudKeyFrames[i], &cloudKeyPoses6D->points[i]);
      *globalSurfCloud += *transformPointCloud(surfCloudKeyFrames[i], &cloudKeyPoses6D->points[i]);
      cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...";
    }
    // down-sample and save corner cloud
    downSizeFilterCorner.setInputCloud(globalCornerCloud);
    downSizeFilterCorner.filter(*globalCornerCloudDS);
    pcl::io::savePCDFileASCII(savePCDDirectory + "cloudCorner.pcd", *globalCornerCloudDS);
    // down-sample and save surf cloud
    downSizeFilterSurf.setInputCloud(globalSurfCloud);
    downSizeFilterSurf.filter(*globalSurfCloudDS);
    pcl::io::savePCDFileASCII(savePCDDirectory + "cloudSurf.pcd", *globalSurfCloudDS);
    // down-sample and save global point cloud map
    *globalMapCloud += *globalCornerCloud;
    *globalMapCloud += *globalSurfCloud;
    pcl::io::savePCDFileASCII(savePCDDirectory + "cloudGlobal.pcd", *globalMapCloud);
    cout << "****************************************************" << endl;
    cout << "Saving map to pcd files completed" << endl;
  }
  /*//}*/

  /*//{ publishGlobalMap() */
  void publishGlobalMap() {
    if (pubLaserCloudSurround.getNumSubscribers() == 0) {
      return;
    }

    if (cloudKeyPoses3D->points.empty() == true) {
      return;
    }

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());

    pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

    // kd-tree to find near key frames to visualize
    std::vector<int>   pointSearchIndGlobalMap;
    std::vector<float> pointSearchSqDisGlobalMap;
    // search near key frames to visualize
    mtx.lock();
    kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
    kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
    mtx.unlock();

    for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
      globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
    // downsample near selected key frames
    pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses;  // for global map visualization
    downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity,
                                                globalMapVisualizationPoseDensity);  // for global map visualization
    downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
    downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);
    for (auto& pt : globalMapKeyPosesDS->points) {
      kdtreeGlobalMap->nearestKSearch(pt, 1, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap);
      pt.intensity = cloudKeyPoses3D->points[pointSearchIndGlobalMap[0]].intensity;
    }

    // extract visualized and downsampled key frames
    for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i) {
      if (pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius) {
        continue;
      }
      int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
      *globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
      *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
    }
    // downsample visualized points
    pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames;  // for global map visualization
    downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize,
                                                 globalMapVisualizationLeafSize);  // for global map visualization
    downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
    downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
    publishCloud(&pubLaserCloudSurround, globalMapKeyFramesDS, timeLaserInfoStamp, odometryFrame);
  }
  /*//}*/

  /*//{ callbackLoopClosureTimer() */
  void callbackLoopClosureTimer([[maybe_unused]] const ros::TimerEvent& event) {

    if (!isInitialized) {
      return;
    }

    if (!loopClosureEnableFlag) {
      return;
    }

    performLoopClosure();
    visualizeLoopClosure();
  }
  /*//}*/

  /*//{ loopInfoHandler() */
  void loopInfoHandler(const std_msgs::Float64MultiArray::ConstPtr& loopMsg) {

    if (!isInitialized) {
      return;
    }

    std::lock_guard<std::mutex> lock(mtxLoopInfo);
    if (loopMsg->data.size() != 2) {
      return;
    }

    loopInfoVec.push_back(*loopMsg);

    while (loopInfoVec.size() > 5) {
      loopInfoVec.pop_front();
    }
  }
  /*//}*/

  /*//{ performLoopClosure() */
  void performLoopClosure() {
    if (cloudKeyPoses3D->points.empty()) {
      return;
    }

    mtx.lock();
    *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
    *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
    mtx.unlock();

    // find keys
    int loopKeyCur;
    int loopKeyPre;
    if (!detectLoopClosureExternal(&loopKeyCur, &loopKeyPre)) {
      if (!detectLoopClosureDistance(&loopKeyCur, &loopKeyPre)) {
        return;
      }
    }

    // extract cloud
    pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
    {
      loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0);
      loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);
      if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000) {
        return;
      }
      if (pubHistoryKeyFrames.getNumSubscribers() != 0) {
        publishCloud(&pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);
      }
    }

    // ICP Settings
    static pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius * 2);
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);

    // Align clouds
    icp.setInputSource(cureKeyframeCloud);
    icp.setInputTarget(prevKeyframeCloud);
    pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
    icp.align(*unused_result);

    if (!icp.hasConverged() || icp.getFitnessScore() > historyKeyframeFitnessScore) {
      return;
    }

    // publish corrected cloud
    if (pubIcpKeyFrames.getNumSubscribers() != 0) {
      pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
      pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
      publishCloud(&pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
    }

    // Get pose transformation
    float           x, y, z, roll, pitch, yaw;
    Eigen::Affine3f correctionLidarFrame;
    correctionLidarFrame = icp.getFinalTransformation();
    // transform from world origin to wrong pose
    const Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
    // transform from world origin to corrected pose
    const Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;  // pre-multiplying -> successive rotation about a fixed frame
    pcl::getTranslationAndEulerAngles(tCorrect, x, y, z, roll, pitch, yaw);
    const gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
    const gtsam::Pose3 poseTo   = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
    gtsam::Vector      Vector6(6);
    const double       noiseScore = icp.getFitnessScore();
    Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
    const noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

    // Add pose constraint
    mtx.lock();
    loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
    loopPoseQueue.push_back(poseFrom.between(poseTo));
    loopNoiseQueue.push_back(constraintNoise);
    mtx.unlock();

    // add loop constriant
    loopIndexContainer[loopKeyCur] = loopKeyPre;
  }
  /*//}*/

  /*//{ detectLoopClosureDistance() */
  bool detectLoopClosureDistance(int* latestID, int* closestID) {
    const int loopKeyCur = int(copy_cloudKeyPoses3D->size()) - 1;
    int       loopKeyPre = -1;

    // check loop constraint added before
    const auto it = loopIndexContainer.find(loopKeyCur);
    if (it != loopIndexContainer.end()) {
      return false;
    }

    // find the closest history key frame
    std::vector<int>   pointSearchIndLoop;
    std::vector<float> pointSearchSqDisLoop;
    kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D);
    kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);

    for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i) {
      int id = pointSearchIndLoop[i];
      if (abs(copy_cloudKeyPoses6D->points[id].time - timeLaserInfoCur) > historyKeyframeSearchTimeDiff) {
        loopKeyPre = id;
        break;
      }
    }

    if (loopKeyPre == -1 || loopKeyCur == loopKeyPre) {
      return false;
    }

    *latestID  = loopKeyCur;
    *closestID = loopKeyPre;

    return true;
  }
  /*//}*/

  /*//{ detectLoopClosureExternal() */
  bool detectLoopClosureExternal(int* latestID, int* closestID) {
    // this function is not used yet, please ignore it
    int loopKeyCur = -1;
    int loopKeyPre = -1;

    std::lock_guard<std::mutex> lock(mtxLoopInfo);
    if (loopInfoVec.empty()) {
      return false;
    }

    const double loopTimeCur = loopInfoVec.front().data[0];
    const double loopTimePre = loopInfoVec.front().data[1];
    loopInfoVec.pop_front();

    if (abs(loopTimeCur - loopTimePre) < historyKeyframeSearchTimeDiff) {
      return false;
    }

    const int cloudSize = copy_cloudKeyPoses6D->size();
    if (cloudSize < 2) {
      return false;
    }

    // latest key
    loopKeyCur = cloudSize - 1;
    for (int i = cloudSize - 1; i >= 0; --i) {
      if (copy_cloudKeyPoses6D->points[i].time >= loopTimeCur) {
        loopKeyCur = int(round(copy_cloudKeyPoses6D->points[i].intensity));
      } else {
        break;
      }
    }

    // previous key
    loopKeyPre = 0;
    for (int i = 0; i < cloudSize; ++i) {
      if (copy_cloudKeyPoses6D->points[i].time <= loopTimePre) {
        loopKeyPre = int(round(copy_cloudKeyPoses6D->points[i].intensity));
      } else {
        break;
      }
    }

    if (loopKeyCur == loopKeyPre) {
      return false;
    }

    const auto it = loopIndexContainer.find(loopKeyCur);
    if (it != loopIndexContainer.end()) {
      return false;
    }

    *latestID  = loopKeyCur;
    *closestID = loopKeyPre;

    return true;
  }
  /*//}*/

  /*//{ loopFindNearKeyframes() */
  void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& searchNum) {
    // extract near keyframes
    nearKeyframes->clear();
    const int cloudSize = copy_cloudKeyPoses6D->size();
    for (int i = -searchNum; i <= searchNum; ++i) {
      const int keyNear = key + i;
      if (keyNear < 0 || keyNear >= cloudSize) {
        continue;
      }
      *nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
      *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
    }

    if (nearKeyframes->empty()) {
      return;
    }

    // downsample near keyframes
    pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
    downSizeFilterICP.setInputCloud(nearKeyframes);
    downSizeFilterICP.filter(*cloud_temp);
    *nearKeyframes = *cloud_temp;
  }
  /*//}*/

  /*//{ visualizeLoopClosure() */
  void visualizeLoopClosure() {
    if (loopIndexContainer.empty() || pubLoopConstraintEdge.getNumSubscribers() == 0) {
      return;
    }

    visualization_msgs::MarkerArray::Ptr markerArray = boost::make_shared<visualization_msgs::MarkerArray>();
    // loop nodes
    visualization_msgs::Marker markerNode;
    markerNode.header.frame_id    = odometryFrame;
    markerNode.header.stamp       = timeLaserInfoStamp;
    markerNode.action             = visualization_msgs::Marker::ADD;
    markerNode.type               = visualization_msgs::Marker::SPHERE_LIST;
    markerNode.ns                 = "loop_nodes";
    markerNode.id                 = 0;
    markerNode.pose.orientation.w = 1;
    markerNode.scale.x            = 0.3;
    markerNode.scale.y            = 0.3;
    markerNode.scale.z            = 0.3;
    markerNode.color.r            = 0;
    markerNode.color.g            = 0.8;
    markerNode.color.b            = 1;
    markerNode.color.a            = 1;
    // loop edges
    visualization_msgs::Marker markerEdge;
    markerEdge.header.frame_id    = odometryFrame;
    markerEdge.header.stamp       = timeLaserInfoStamp;
    markerEdge.action             = visualization_msgs::Marker::ADD;
    markerEdge.type               = visualization_msgs::Marker::LINE_LIST;
    markerEdge.ns                 = "loop_edges";
    markerEdge.id                 = 1;
    markerEdge.pose.orientation.w = 1;
    markerEdge.scale.x            = 0.1;
    markerEdge.color.r            = 0.9;
    markerEdge.color.g            = 0.9;
    markerEdge.color.b            = 0;
    markerEdge.color.a            = 1;

    for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it) {
      const int            key_cur = it->first;
      const int            key_pre = it->second;
      geometry_msgs::Point p;
      p.x = copy_cloudKeyPoses6D->points[key_cur].x;
      p.y = copy_cloudKeyPoses6D->points[key_cur].y;
      p.z = copy_cloudKeyPoses6D->points[key_cur].z;
      markerNode.points.push_back(p);
      markerEdge.points.push_back(p);
      p.x = copy_cloudKeyPoses6D->points[key_pre].x;
      p.y = copy_cloudKeyPoses6D->points[key_pre].y;
      p.z = copy_cloudKeyPoses6D->points[key_pre].z;
      markerNode.points.push_back(p);
      markerEdge.points.push_back(p);
    }

    markerArray->markers.push_back(markerNode);
    markerArray->markers.push_back(markerEdge);
    try {
      pubLoopConstraintEdge.publish(markerArray);
    }
    catch (...) {
      ROS_ERROR("[MAS-LO|MO]: Exception caught during publishing topic %s.", pubLoopConstraintEdge.getTopic().c_str());
    }
  }
  /*//}*/

  /*//{ updateInitialGuess() */
  void updateInitialGuess() {
    // save current transformation before any processing
    incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped);

    static Eigen::Affine3f lastPreTransformation;

    // initialization
    // orientation is needed here to initialize the orientation of the map origin
    // we can set it to orientation obtained from e.g., orientation from HW API
    if (cloudKeyPoses3D->points.empty()) {
      transformTobeMapped[0] = cloudInfo.rollInit;
      transformTobeMapped[1] = cloudInfo.pitchInit;
      transformTobeMapped[2] = cloudInfo.yawInit;

      lastPreTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.rollInit, cloudInfo.pitchInit, cloudInfo.yawInit);
    }

    // use pre-integration estimation for pose guess
    static bool            lastPreTransAvailable = false;
    static Eigen::Affine3f lastMasPreTransformation;
    if (cloudInfo.odomAvailable) {
      const Eigen::Affine3f transBack = pcl::getTransformation(cloudInfo.initialGuessX, cloudInfo.initialGuessY, cloudInfo.initialGuessZ,
                                                               cloudInfo.initialGuessRoll, cloudInfo.initialGuessPitch, cloudInfo.initialGuessYaw);
      if (!lastPreTransAvailable) {
        lastMasPreTransformation = transBack;
        lastPreTransAvailable    = true;
      } else {
        const Eigen::Affine3f transIncre = lastMasPreTransformation.inverse() * transBack;
        const Eigen::Affine3f transTobe  = trans2Affine3f(transformTobeMapped);
        const Eigen::Affine3f transFinal = transTobe * transIncre;

        // transformedTobeMapped will be used as initial guess in mapOptimization
        pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], transformTobeMapped[0],
                                          transformTobeMapped[1], transformTobeMapped[2]);

        lastMasPreTransformation = transBack;

        lastPreTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.rollInit, cloudInfo.pitchInit, cloudInfo.yawInit);
        return;
      }
    }

    // this code section won't be reached if pre-integrated rotation is available
    // if available, use preintegrated incremental estimation for pose guess (only rotation)
    // if not available, the pre-integrated rotation from above will be used
    // therefore, orientation is not necessary here
    if (cloudInfo.orientationAvailable) {
      const Eigen::Affine3f transBack  = pcl::getTransformation(0, 0, 0, cloudInfo.rollInit, cloudInfo.pitchInit, cloudInfo.yawInit);
      const Eigen::Affine3f transIncre = lastPreTransformation.inverse() * transBack;  // only place where lastTransformation is used

      const Eigen::Affine3f transTobe  = trans2Affine3f(transformTobeMapped);
      const Eigen::Affine3f transFinal = transTobe * transIncre;
      pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], transformTobeMapped[0],
                                        transformTobeMapped[1], transformTobeMapped[2]);

      lastPreTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.rollInit, cloudInfo.pitchInit, cloudInfo.yawInit);
      return;
    }
  }
  /*//}*/

  /*//{ extractForLoopClosure() */
  void extractForLoopClosure() {
    pcl::PointCloud<PointType>::Ptr cloudToExtract(new pcl::PointCloud<PointType>());
    const int                       numPoses = cloudKeyPoses3D->size();
    for (int i = numPoses - 1; i >= 0; --i) {
      if ((int)cloudToExtract->size() <= surroundingKeyframeSize) {
        cloudToExtract->push_back(cloudKeyPoses3D->points[i]);
      } else {
        break;
      }
    }

    extractCloud(cloudToExtract);
  }
  /*//}*/

  /*//{ extractNearby() */
  void extractNearby() {
    pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
    std::vector<int>                pointSearchInd;
    std::vector<float>              pointSearchSqDis;

    // extract all the nearby key poses and downsample them
    kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D);  // create kd-tree
    kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(), (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis);
    for (int i = 0; i < (int)pointSearchInd.size(); ++i) {
      int id = pointSearchInd[i];
      surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
    }

    downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
    downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);
    for (auto& pt : surroundingKeyPosesDS->points) {
      kdtreeSurroundingKeyPoses->nearestKSearch(pt, 1, pointSearchInd, pointSearchSqDis);
      pt.intensity = cloudKeyPoses3D->points[pointSearchInd[0]].intensity;
    }

    // also extract some latest key frames in case the robot rotates in one position
    const int numPoses = cloudKeyPoses3D->size();
    for (int i = numPoses - 1; i >= 0; --i) {
      if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 10.0) {
        surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
      } else {
        break;
      }
    }

    extractCloud(surroundingKeyPosesDS);
  }
  /*//}*/

  /*//{ extractCloud() */
  void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract) {
    // fuse the map
    laserCloudCornerFromMap->clear();
    laserCloudSurfFromMap->clear();
    for (int i = 0; i < (int)cloudToExtract->size(); ++i) {
      if (pointDistance(cloudToExtract->points[i], cloudKeyPoses3D->back()) > surroundingKeyframeSearchRadius) {
        continue;
      }

      const int thisKeyInd = (int)cloudToExtract->points[i].intensity;
      if (laserCloudMapContainer.find(thisKeyInd) != laserCloudMapContainer.end()) {
        // transformed cloud available
        *laserCloudCornerFromMap += laserCloudMapContainer[thisKeyInd].first;
        *laserCloudSurfFromMap += laserCloudMapContainer[thisKeyInd].second;
      } else {
        // transformed cloud not available
        pcl::PointCloud<PointType> laserCloudCornerTemp = *transformPointCloud(cornerCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
        pcl::PointCloud<PointType> laserCloudSurfTemp   = *transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
        *laserCloudCornerFromMap += laserCloudCornerTemp;
        *laserCloudSurfFromMap += laserCloudSurfTemp;
        laserCloudMapContainer[thisKeyInd] = make_pair(laserCloudCornerTemp, laserCloudSurfTemp);
      }
    }

    // Downsample the surrounding corner key frames (or map)
    downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
    downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
    laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->size();
    // Downsample the surrounding surf key frames (or map)
    downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
    downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
    laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();

    // clear map cache if too large
    if (laserCloudMapContainer.size() > 1000) {
      laserCloudMapContainer.clear();
    }
  }
  /*//}*/

  /*//{ extractSurroundingKeyFrames() */
  void extractSurroundingKeyFrames() {
    if (cloudKeyPoses3D->points.empty()) {
      return;
    }

    // if (loopClosureEnableFlag == true)
    // {
    //     extractForLoopClosure();
    // } else {
    //     extractNearby();
    // }

    extractNearby();
  }
  /*//}*/

  /*//{ downsampleCurrentScan() */
  void downsampleCurrentScan() {
    // Downsample cloud from current scan
    laserCloudCornerLastDS->clear();
    downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
    downSizeFilterCorner.filter(*laserCloudCornerLastDS);
    laserCloudCornerLastDSNum = laserCloudCornerLastDS->size();

    laserCloudSurfLastDS->clear();
    downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
    downSizeFilterSurf.filter(*laserCloudSurfLastDS);
    laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();
  }
  /*//}*/

  /*//{ updatePointAssociateToMap() */
  void updatePointAssociateToMap() {
    transPointAssociateToMap = trans2Affine3f(transformTobeMapped);
  }
  /*//}*/

  /*//{ cornerOptimization() */
  void cornerOptimization() {
    updatePointAssociateToMap();

#pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < laserCloudCornerLastDSNum; i++) {
      PointType          pointOri, pointSel, coeff;
      std::vector<int>   pointSearchInd;
      std::vector<float> pointSearchSqDis;

      pointOri = laserCloudCornerLastDS->points[i];
      pointAssociateToMap(&pointOri, &pointSel);
      kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

      cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
      cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
      cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));

      if (pointSearchSqDis[4] < 1.0) {
        float cx = 0, cy = 0, cz = 0;
        for (int j = 0; j < 5; j++) {
          cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
          cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
          cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
        }
        cx /= 5.0f;
        cy /= 5.0f;
        cz /= 5.0f;

        float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
        for (int j = 0; j < 5; j++) {
          const float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
          const float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
          const float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

          a11 += ax * ax;
          a12 += ax * ay;
          a13 += ax * az;
          a22 += ay * ay;
          a23 += ay * az;
          a33 += az * az;
        }
        a11 /= 5;
        a12 /= 5;
        a13 /= 5;
        a22 /= 5;
        a23 /= 5;
        a33 /= 5;

        matA1.at<float>(0, 0) = a11;
        matA1.at<float>(0, 1) = a12;
        matA1.at<float>(0, 2) = a13;
        matA1.at<float>(1, 0) = a12;
        matA1.at<float>(1, 1) = a22;
        matA1.at<float>(1, 2) = a23;
        matA1.at<float>(2, 0) = a13;
        matA1.at<float>(2, 1) = a23;
        matA1.at<float>(2, 2) = a33;

        cv::eigen(matA1, matD1, matV1);

        if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) {

          const float x0 = pointSel.x;
          const float y0 = pointSel.y;
          const float z0 = pointSel.z;
          const float x1 = cx + 0.1f * matV1.at<float>(0, 0);
          const float y1 = cy + 0.1f * matV1.at<float>(0, 1);
          const float z1 = cz + 0.1f * matV1.at<float>(0, 2);
          const float x2 = cx - 0.1f * matV1.at<float>(0, 0);
          const float y2 = cy - 0.1f * matV1.at<float>(0, 1);
          const float z2 = cz - 0.1f * matV1.at<float>(0, 2);

          const float a012 = sqrt(((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) +
                                  ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) +
                                  ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)));

          const float l12 = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));

          const float la =
              ((y1 - y2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) + (z1 - z2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1))) / a012 / l12;

          const float lb =
              -((x1 - x2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) - (z1 - z2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) / a012 / l12;

          const float lc =
              -((x1 - x2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) + (y1 - y2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) / a012 / l12;

          const float ld2 = a012 / l12;

          const float s = 1.0f - 0.9f * fabs(ld2);

          coeff.x         = s * la;
          coeff.y         = s * lb;
          coeff.z         = s * lc;
          coeff.intensity = s * ld2;

          if (s > 0.1) {
            laserCloudOriCornerVec[i]  = pointOri;
            coeffSelCornerVec[i]       = coeff;
            laserCloudOriCornerFlag[i] = true;
          }
        }
      }
    }
  }
  /*//}*/

  /*//{ surfOptimization() */
  void surfOptimization() {
    updatePointAssociateToMap();

#pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < laserCloudSurfLastDSNum; i++) {
      PointType          pointOri, pointSel, coeff;
      std::vector<int>   pointSearchInd;
      std::vector<float> pointSearchSqDis;

      pointOri = laserCloudSurfLastDS->points[i];
      pointAssociateToMap(&pointOri, &pointSel);
      kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

      Eigen::Matrix<float, 5, 3> matA0;
      Eigen::Matrix<float, 5, 1> matB0;
      Eigen::Vector3f            matX0;

      matA0.setZero();
      matB0.fill(-1);
      matX0.setZero();

      if (pointSearchSqDis[4] < 1.0) {
        for (int j = 0; j < 5; j++) {
          matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
          matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
          matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
        }

        matX0 = matA0.colPivHouseholderQr().solve(matB0);

        float pa = matX0(0, 0);
        float pb = matX0(1, 0);
        float pc = matX0(2, 0);
        float pd = 1;

        const float ps = sqrt(pa * pa + pb * pb + pc * pc);
        pa /= ps;
        pb /= ps;
        pc /= ps;
        pd /= ps;

        bool planeValid = true;
        for (int j = 0; j < 5; j++) {
          if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x + pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                   pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2) {
            planeValid = false;
            break;
          }
        }

        if (planeValid) {
          const float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

          const float s = 1.0f - 0.9f * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x + pointSel.y * pointSel.y + pointSel.z * pointSel.z));

          coeff.x         = s * pa;
          coeff.y         = s * pb;
          coeff.z         = s * pc;
          coeff.intensity = s * pd2;

          if (s > 0.1) {
            laserCloudOriSurfVec[i]  = pointOri;
            coeffSelSurfVec[i]       = coeff;
            laserCloudOriSurfFlag[i] = true;
          }
        }
      }
    }
  }
  /*//}*/

  /*//{ combineOptimizationCoeffs() */
  void combineOptimizationCoeffs() {
    // combine corner coeffs
    for (int i = 0; i < laserCloudCornerLastDSNum; ++i) {
      if (laserCloudOriCornerFlag[i] == true) {
        laserCloudOri->push_back(laserCloudOriCornerVec[i]);
        coeffSel->push_back(coeffSelCornerVec[i]);
      }
    }
    // combine surf coeffs
    for (int i = 0; i < laserCloudSurfLastDSNum; ++i) {
      if (laserCloudOriSurfFlag[i] == true) {
        laserCloudOri->push_back(laserCloudOriSurfVec[i]);
        coeffSel->push_back(coeffSelSurfVec[i]);
      }
    }
    // reset flag for next iteration
    std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
    std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);
  }
  /*//}*/

  /*//{ LMOptimization() */
  bool LMOptimization(int iterCount) {
    // This optimization is from the original loam_velodyne by Ji Zhang, need to cope with coordinate transformation
    // lidar <- camera      ---     camera <- lidar
    // x = z                ---     x = y
    // y = x                ---     y = z
    // z = y                ---     z = x
    // roll = yaw           ---     roll = pitch
    // pitch = roll         ---     pitch = yaw
    // yaw = pitch          ---     yaw = roll

    // lidar -> camera
    const float srx = sin(transformTobeMapped[1]);
    const float crx = cos(transformTobeMapped[1]);
    const float sry = sin(transformTobeMapped[2]);
    const float cry = cos(transformTobeMapped[2]);
    const float srz = sin(transformTobeMapped[0]);
    const float crz = cos(transformTobeMapped[0]);

    const int laserCloudSelNum = laserCloudOri->size();
    if (laserCloudSelNum < 50) {
      return false;
    }

    cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
    cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
    cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
    cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
    cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
    cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));

    PointType pointOri, coeff;

    for (int i = 0; i < laserCloudSelNum; i++) {
      // lidar -> camera
      pointOri.x = laserCloudOri->points[i].y;
      pointOri.y = laserCloudOri->points[i].z;
      pointOri.z = laserCloudOri->points[i].x;
      // lidar -> camera
      coeff.x         = coeffSel->points[i].y;
      coeff.y         = coeffSel->points[i].z;
      coeff.z         = coeffSel->points[i].x;
      coeff.intensity = coeffSel->points[i].intensity;
      // in camera
      const float arx = (crx * sry * srz * pointOri.x + crx * crz * sry * pointOri.y - srx * sry * pointOri.z) * coeff.x +
                        (-srx * srz * pointOri.x - crz * srx * pointOri.y - crx * pointOri.z) * coeff.y +
                        (crx * cry * srz * pointOri.x + crx * cry * crz * pointOri.y - cry * srx * pointOri.z) * coeff.z;

      const float ary = ((cry * srx * srz - crz * sry) * pointOri.x + (sry * srz + cry * crz * srx) * pointOri.y + crx * cry * pointOri.z) * coeff.x +
                        ((-cry * crz - srx * sry * srz) * pointOri.x + (cry * srz - crz * srx * sry) * pointOri.y - crx * sry * pointOri.z) * coeff.z;

      const float arz = ((crz * srx * sry - cry * srz) * pointOri.x + (-cry * crz - srx * sry * srz) * pointOri.y) * coeff.x +
                        (crx * crz * pointOri.x - crx * srz * pointOri.y) * coeff.y +
                        ((sry * srz + cry * crz * srx) * pointOri.x + (crz * sry - cry * srx * srz) * pointOri.y) * coeff.z;
      // lidar -> camera
      matA.at<float>(i, 0) = arz;
      matA.at<float>(i, 1) = arx;
      matA.at<float>(i, 2) = ary;
      matA.at<float>(i, 3) = coeff.z;
      matA.at<float>(i, 4) = coeff.x;
      matA.at<float>(i, 5) = coeff.y;
      matB.at<float>(i, 0) = -coeff.intensity;
    }

    cv::transpose(matA, matAt);
    matAtA = matAt * matA;
    matAtB = matAt * matB;
    cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

    if (iterCount == 0) {

      cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
      cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
      cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

      cv::eigen(matAtA, matE, matV);
      matV.copyTo(matV2);

      isDegenerate            = false;
      const float eignThre[6] = {100, 100, 100, 100, 100, 100};
      for (int i = 5; i >= 0; i--) {
        if (matE.at<float>(0, i) < eignThre[i]) {
          for (int j = 0; j < 6; j++) {
            matV2.at<float>(i, j) = 0;
          }
          isDegenerate = true;
        } else {
          break;
        }
      }
      matP = matV.inv() * matV2;
    }

    if (isDegenerate) {
      cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
      matX.copyTo(matX2);
      matX = matP * matX2;
    }

    transformTobeMapped[0] += matX.at<float>(0, 0);
    transformTobeMapped[1] += matX.at<float>(1, 0);
    transformTobeMapped[2] += matX.at<float>(2, 0);
    transformTobeMapped[3] += matX.at<float>(3, 0);
    transformTobeMapped[4] += matX.at<float>(4, 0);
    transformTobeMapped[5] += matX.at<float>(5, 0);

    const float deltaR =
        sqrt(pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) + pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) + pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
    const float deltaT = sqrt(pow(matX.at<float>(3, 0) * 100, 2) + pow(matX.at<float>(4, 0) * 100, 2) + pow(matX.at<float>(5, 0) * 100, 2));

    if (deltaR < 0.05 && deltaT < 0.05) {
      return true;  // converged
    }
    return false;  // keep optimizing
  }
  /*//}*/

  /*//{ scan2MapOptimization() */
  void scan2MapOptimization() {
    if (cloudKeyPoses3D->points.empty()) {
      return;
    }

    if (laserCloudCornerLastDSNum > edgeFeatureMinValidNum && laserCloudSurfLastDSNum > surfFeatureMinValidNum) {
      kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
      kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);

      for (int iterCount = 0; iterCount < 30; iterCount++) {
        laserCloudOri->clear();
        coeffSel->clear();

        cornerOptimization();
        surfOptimization();

        combineOptimizationCoeffs();

        if (LMOptimization(iterCount)) {
          break;
        }
      }

      incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped);

      isFirstMapOptimizationSuccessful = true;

    } else {
      ROS_WARN("Not enough features! Only %d edge and %d planar features available.", laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
    }
  }
  /*//}*/

  /*//{ saveFrame() */
  // this checks whether a keyframe should be added when large enough motion detected
  bool saveFrame() {

    if (cloudKeyPoses3D->points.empty()) {
      ROS_INFO("[MapOptimization]: cloudKeyPoses3D empty");
      return true;
    }

    const Eigen::Affine3f transStart   = pclPointToAffine3f(cloudKeyPoses6D->back());
    const Eigen::Affine3f transFinal   = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], transformTobeMapped[0],
                                                              transformTobeMapped[1], transformTobeMapped[2]);
    const Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
    float                 x, y, z, roll, pitch, yaw;
    pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

    if (abs(roll) < surroundingkeyframeAddingAngleThreshold && abs(pitch) < surroundingkeyframeAddingAngleThreshold &&
        abs(yaw) < surroundingkeyframeAddingAngleThreshold && sqrt(x * x + y * y + z * z) < surroundingkeyframeAddingDistThreshold) {
      /* ROS_INFO("[MapOptimization]: not enough motion, not adding keyframe"); */
      return false;
    }

    return true;
  }
  /*//}*/

  /*//{ addOdomFactor() */
  void addOdomFactor() {

    // First pose prior
    if (cloudKeyPoses3D->points.empty()) {
      const noiseModel::Diagonal::shared_ptr priorNoise =
          noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI * M_PI, 1e8, 1e8, 1e8).finished());  // rad*rad, meter*meter
      gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
      initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));

    } else {
      const noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
      const gtsam::Pose3                     poseFrom      = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
      const gtsam::Pose3                     poseTo        = trans2gtsamPose(transformTobeMapped);
      gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size() - 1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
      initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
    }
  }
  /*//}*/

  /*//{ addGPSFactor() */
  void addGPSFactor() {
    if (gpsQueue.empty()) {
      return;
    }

    // wait for system initialized and settles down
    if (cloudKeyPoses3D->points.empty()) {
      return;
    } else if (pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < 5.0) {
      return;
    }

    // pose covariance small, no need to correct
    if (poseCovariance(3, 3) < poseCovThreshold && poseCovariance(4, 4) < poseCovThreshold) {
      return;
    }

    // last gps position
    static PointType lastGPSPoint;

    while (!gpsQueue.empty()) {
      if (gpsQueue.front().header.stamp.toSec() < timeLaserInfoCur - 0.2) {
        // message too old
        gpsQueue.pop_front();
      } else if (gpsQueue.front().header.stamp.toSec() > timeLaserInfoCur + 0.2) {
        // message too new
        break;
      } else {
        nav_msgs::Odometry thisGPS = gpsQueue.front();
        gpsQueue.pop_front();

        // GPS too noisy, skip
        const float noise_x = thisGPS.pose.covariance[0];
        const float noise_y = thisGPS.pose.covariance[7];
        float       noise_z = thisGPS.pose.covariance[14];
        if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold) {
          continue;
        }

        const float gps_x = thisGPS.pose.pose.position.x;
        const float gps_y = thisGPS.pose.pose.position.y;
        float       gps_z = thisGPS.pose.pose.position.z;
        if (!useGpsElevation) {
          gps_z   = transformTobeMapped[5];
          noise_z = 0.01;
        }

        // GPS not properly initialized (0,0,0)
        if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6) {
          continue;
        }

        // Add GPS every a few meters
        PointType curGPSPoint;
        curGPSPoint.x = gps_x;
        curGPSPoint.y = gps_y;
        curGPSPoint.z = gps_z;
        if (pointDistance(curGPSPoint, lastGPSPoint) < 5.0) {
          continue;
        } else {
          lastGPSPoint = curGPSPoint;
        }

        gtsam::Vector Vector3(3);
        Vector3 << max(noise_x, 1.0f), max(noise_y, 1.0f), max(noise_z, 1.0f);
        const noiseModel::Diagonal::shared_ptr gps_noise = noiseModel::Diagonal::Variances(Vector3);
        gtsam::GPSFactor                       gps_factor(cloudKeyPoses3D->size(), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
        gtSAMgraph.add(gps_factor);

        aLoopIsClosed = true;
        break;
      }
    }
  }
  /*//}*/

  /*//{ addLoopFactor() */
  void addLoopFactor() {
    if (loopIndexQueue.empty()) {
      return;
    }

    for (int i = 0; i < (int)loopIndexQueue.size(); ++i) {
      const int                                     indexFrom    = loopIndexQueue[i].first;
      const int                                     indexTo      = loopIndexQueue[i].second;
      const gtsam::Pose3                            poseBetween  = loopPoseQueue[i];
      const gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
      gtSAMgraph.add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
    }

    loopIndexQueue.clear();
    loopPoseQueue.clear();
    loopNoiseQueue.clear();
    aLoopIsClosed = true;
  }
  /*//}*/

  /*//{ saveKeyFramesAndFactor() */
  void saveKeyFramesAndFactor() {
    if (saveFrame() == false) {
      return;
    }

    // odom factor
    addOdomFactor();

    // gps factor
    addGPSFactor();

    // loop factor
    addLoopFactor();

    /* cout << "****************************************************" << endl; */
    /* gtSAMgraph.print("[MapOptimization]: graph\n"); */

    // update iSAM
    isam->update(gtSAMgraph, initialEstimate);
    isam->update();

    if (aLoopIsClosed == true) {
      isam->update();
      isam->update();
      isam->update();
      isam->update();
      isam->update();
    }

    gtSAMgraph.resize(0);
    initialEstimate.clear();

    // save key poses
    PointType     thisPose3D;
    PointTypePose thisPose6D;
    Pose3         latestEstimate;

    isamCurrentEstimate = isam->calculateEstimate();
    latestEstimate      = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size() - 1);
    // cout << "****************************************************" << endl;
    // isamCurrentEstimate.print("Current estimate: ");

    thisPose3D.x         = latestEstimate.translation().x();
    thisPose3D.y         = latestEstimate.translation().y();
    thisPose3D.z         = latestEstimate.translation().z();
    thisPose3D.intensity = cloudKeyPoses3D->size();  // this can be used as index
    cloudKeyPoses3D->push_back(thisPose3D);

    thisPose6D.x         = thisPose3D.x;
    thisPose6D.y         = thisPose3D.y;
    thisPose6D.z         = thisPose3D.z;
    thisPose6D.intensity = thisPose3D.intensity;  // this can be used as index
    thisPose6D.roll      = latestEstimate.rotation().roll();
    thisPose6D.pitch     = latestEstimate.rotation().pitch();
    thisPose6D.yaw       = latestEstimate.rotation().yaw();
    thisPose6D.time      = timeLaserInfoCur;
    cloudKeyPoses6D->push_back(thisPose6D);

    // cout << "****************************************************" << endl;
    // cout << "Pose covariance:" << endl;
    // cout << isam->marginalCovariance(isamCurrentEstimate.size()-1) << endl << endl;
    poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size() - 1);

    // save updated transform
    transformTobeMapped[0] = latestEstimate.rotation().roll();
    transformTobeMapped[1] = latestEstimate.rotation().pitch();
    transformTobeMapped[2] = latestEstimate.rotation().yaw();
    transformTobeMapped[3] = latestEstimate.translation().x();
    transformTobeMapped[4] = latestEstimate.translation().y();
    transformTobeMapped[5] = latestEstimate.translation().z();

    // save all the received edge and surf points
    pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
    pcl::copyPointCloud(*laserCloudCornerLastDS, *thisCornerKeyFrame);
    pcl::copyPointCloud(*laserCloudSurfLastDS, *thisSurfKeyFrame);

    // save key frame cloud
    cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
    surfCloudKeyFrames.push_back(thisSurfKeyFrame);

    // save path for visualization
    updatePath(thisPose6D);
  }
  /*//}*/

  /*//{ correctPoses() */
  void correctPoses() {
    if (cloudKeyPoses3D->points.empty()) {
      return;
    }

    if (aLoopIsClosed == true) {
      // clear map cache
      laserCloudMapContainer.clear();
      // clear path
      globalPath->poses.clear();
      // update key poses
      const int numPoses = isamCurrentEstimate.size();
      for (int i = 0; i < numPoses; ++i) {
        cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().x();
        cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().y();
        cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().z();

        cloudKeyPoses6D->points[i].x     = cloudKeyPoses3D->points[i].x;
        cloudKeyPoses6D->points[i].y     = cloudKeyPoses3D->points[i].y;
        cloudKeyPoses6D->points[i].z     = cloudKeyPoses3D->points[i].z;
        cloudKeyPoses6D->points[i].roll  = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
        cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
        cloudKeyPoses6D->points[i].yaw   = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();

        updatePath(cloudKeyPoses6D->points[i]);
      }

      aLoopIsClosed = false;
    }
  }
  /*//}*/

  /*//{ updatePath() */
  void updatePath(const PointTypePose& pose_in) {
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp       = ros::Time().fromSec(pose_in.time);
    pose_stamped.header.frame_id    = odometryFrame;
    pose_stamped.pose.position.x    = pose_in.x;
    pose_stamped.pose.position.y    = pose_in.y;
    pose_stamped.pose.position.z    = pose_in.z;
    tf::Quaternion q                = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
    pose_stamped.pose.orientation.x = q.x();
    pose_stamped.pose.orientation.y = q.y();
    pose_stamped.pose.orientation.z = q.z();
    pose_stamped.pose.orientation.w = q.w();

    globalPath->poses.push_back(pose_stamped);
  }
  /*//}*/

  /*//{ publishOdometry() */
  void publishOdometry() {
    // transform from lidar to fcu frame
    /* Eigen::Matrix4d T = Eigen::Matrix4d::Identity(); */
    /* Eigen::Matrix4d T_lidar = Eigen::Matrix4d::Identity(); */

    tf::Transform T;
    tf::Transform T_lidar;

    // TODO: Load this as static TF
    T.setOrigin(tf::Vector3(tfLidar2Baselink.getOrigin().x(), tfLidar2Baselink.getOrigin().y(), tfLidar2Baselink.getOrigin().z()));
    /* T.setRotation(tf::createQuaternionFromRPY(0, 0, M_PI)); // os_lidar -> os_sensor */
    T.setRotation(tf::createQuaternionFromRPY(0, 0, 0));  // os_sensor -> fcu

    // transform from map to lidar
    T_lidar.setOrigin(tf::Vector3(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5]));
    T_lidar.setRotation(tf::createQuaternionFromRPY(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]));

    const tf::Transform T_odom = T * T_lidar;

    // Publish in fcu frame
    nav_msgs::Odometry::Ptr laserOdometryROS = boost::make_shared<nav_msgs::Odometry>();
    laserOdometryROS->header.stamp           = timeLaserInfoStamp;
    laserOdometryROS->header.frame_id        = odometryFrame;
    laserOdometryROS->child_frame_id         = baselinkFrame;
    laserOdometryROS->pose.pose.position.x   = T_odom.getOrigin().getX();
    laserOdometryROS->pose.pose.position.y   = T_odom.getOrigin().getY();
    laserOdometryROS->pose.pose.position.z   = T_odom.getOrigin().getZ();
    tf::quaternionTFToMsg(T_odom.getRotation(), laserOdometryROS->pose.pose.orientation);
    /* laserOdometryROS->pose.pose.orientation   = orientationMsg.quaternion; */
    // ODOM: M -> FCU
    try {
      pubLaserOdometryGlobal.publish(laserOdometryROS);
    }
    catch (...) {
      ROS_ERROR("[MAS-LO|MO]: Exception caught during publishing topic %s.", pubLaserOdometryGlobal.getTopic().c_str());
    }

    // Publish odometry for ROS (global) in lidar frame
    /* nav_msgs::Odometry laserOdometryROS; */
    /* laserOdometryROS.header.stamp          = timeLaserInfoStamp; */
    /* laserOdometryROS.header.frame_id       = odometryFrame; */
    /* laserOdometryROS.child_frame_id        = baselinkFrame; */
    /* laserOdometryROS.pose.pose.position.x  = transformTobeMapped[3]; */
    /* laserOdometryROS.pose.pose.position.y  = transformTobeMapped[4]; */
    /* laserOdometryROS.pose.pose.position.z  = transformTobeMapped[5]; */
    /* laserOdometryROS.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
     */
    /* pubLaserOdometryGlobal.publish(laserOdometryROS); */

    // Publish TF
    /* static tf::TransformBroadcaster br; */
    /* tf::Transform        t_odom_to_lidar = tf::Transform(tf::createQuaternionFromRPY(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]),
     */
    /*                                               tf::Vector3(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5])); */
    /* tf::StampedTransform trans_odom_to_lidar = tf::StampedTransform(t_odom_to_lidar, timeLaserInfoStamp, odometryFrame, "lidar_link"); */
    /* br.sendTransform(trans_odom_to_lidar); */

    // Publish odometry for ROS (incremental)
    static bool                    lastIncreOdomPubFlag = false;
    static nav_msgs::Odometry::Ptr laserOdomIncremental = boost::make_shared<nav_msgs::Odometry>();  // incremental odometry msg
    static Eigen::Affine3f         increOdomAffine;                                                  // incremental odometry in affine
    if (!lastIncreOdomPubFlag) {
      lastIncreOdomPubFlag = true;
      laserOdomIncremental = laserOdometryROS;
      increOdomAffine      = trans2Affine3f(transformTobeMapped);
    } else {
      Eigen::Affine3f affineIncre = incrementalOdometryAffineFront.inverse() * incrementalOdometryAffineBack;
      increOdomAffine             = increOdomAffine * affineIncre;
      float x, y, z, roll, pitch, yaw;
      pcl::getTranslationAndEulerAngles(increOdomAffine, x, y, z, roll, pitch, yaw);

      laserOdomIncremental->header.stamp          = timeLaserInfoStamp;
      laserOdomIncremental->header.frame_id       = odometryFrame;
      laserOdomIncremental->child_frame_id        = baselinkFrame;
      laserOdomIncremental->pose.pose.position.x  = x;
      laserOdomIncremental->pose.pose.position.y  = y;
      laserOdomIncremental->pose.pose.position.z  = z;
      laserOdomIncremental->pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
      /* laserOdomIncremental->pose.pose.orientation   = orientationMsg.quaternion; */
      if (isDegenerate) {
        laserOdomIncremental->pose.covariance[0] = 1;
      } else {
        laserOdomIncremental->pose.covariance[0] = 0;
      }
    }
    try {
      pubLaserOdometryIncremental.publish(laserOdomIncremental);
    }
    catch (...) {
      ROS_ERROR("[MAS-LO|MO]: Exception caught during publishing topic %s.", pubLaserOdometryIncremental.getTopic().c_str());
    }
  }
  /*//}*/

  /*//{ publishFrames() */
  void publishFrames() {
    if (cloudKeyPoses3D->points.empty()) {
      return;
    }
    // publish key poses
    publishCloud(&pubKeyPoses, cloudKeyPoses3D, timeLaserInfoStamp, odometryFrame);
    // Publish surrounding key frames
    publishCloud(&pubRecentKeyFrames, laserCloudSurfFromMapDS, timeLaserInfoStamp, odometryFrame);
    // publish registered key frame
    if (pubRecentKeyFrame.getNumSubscribers() != 0) {
      pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
      PointTypePose                   thisPose6D = trans2PointTypePose(transformTobeMapped);
      *cloudOut += *transformPointCloud(laserCloudCornerLastDS, &thisPose6D);
      *cloudOut += *transformPointCloud(laserCloudSurfLastDS, &thisPose6D);
      publishCloud(&pubRecentKeyFrame, cloudOut, timeLaserInfoStamp, odometryFrame);
    }
    // publish registered high-res raw cloud
    if (pubCloudRegisteredRaw.getNumSubscribers() != 0) {
      pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
      pcl::fromROSMsg(cloudInfo.cloud_deskewed, *cloudOut);
      PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
      *cloudOut                = *transformPointCloud(cloudOut, &thisPose6D);
      publishCloud(&pubCloudRegisteredRaw, cloudOut, timeLaserInfoStamp, odometryFrame);
    }

    // publish path
    if (pubPath.getNumSubscribers() > 0) {
      globalPath->header.stamp    = timeLaserInfoStamp;
      globalPath->header.frame_id = odometryFrame;

      try {
        pubPath.publish(globalPath);
      }
      catch (...) {
        ROS_ERROR("[MAS-LO|MO]: Exception caught during publishing topic %s.", pubPath.getTopic().c_str());
      }
    }
  }
  /*//}*/
};
/*//}*/

}  // namespace map_optimization
}  // namespace maslo

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(maslo::map_optimization::MapOptimization, nodelet::Nodelet)
