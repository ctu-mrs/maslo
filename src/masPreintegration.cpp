#include "utility.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

#include <mas_factor/mas_factor.h>
#include <mas_factor/mas_bias.h>

using gtsam::symbol_shorthand::B;  // Bias    (alx,aly,alz,aax,aay,aaz)
using gtsam::symbol_shorthand::V;  // Lin Vel (xdot,ydot,zdot)
using gtsam::symbol_shorthand::W;  // Ang Vel (rdot,pdot,ydot)
using gtsam::symbol_shorthand::X;  // Pose3   (x,y,z,r,p,y)

namespace maslo
{
namespace mas_preintegration
{

/*//{ class MasPreintegration() */
class MasPreintegration : public nodelet::Nodelet {

public:
  /*//{ parameters */
  std::string uavName;

  // Frames
  std::string lidarFrame;
  std::string baselinkFrame;
  std::string odometryFrame;

  // Motor Speeds
  int   numMotors;
  float mass;
  float gravity;
  float propMass;
  float motorConstant;
  float momentConstant;
  float linAccNoise;
  float angAccNoise;
  float linAccBiasNoise;
  float angAccBiasNoise;

  /*//}*/

  std::mutex mtx;

  ros::Subscriber subMas;
  ros::Subscriber subOdometry;
  ros::Publisher  pubPreOdometry;
  ros::Publisher  pubLinAcc;
  ros::Publisher  pubLinAccBias;
  ros::Publisher  pubAngAcc;
  ros::Publisher  pubAngAccBias;

  bool systemInitialized = false;

  gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;
  gtsam::noiseModel::Diagonal::shared_ptr priorLinVelNoise;
  gtsam::noiseModel::Diagonal::shared_ptr priorAngVelNoise;
  gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;
  gtsam::noiseModel::Diagonal::shared_ptr correctionNoise;
  gtsam::noiseModel::Diagonal::shared_ptr correctionNoise2;
  gtsam::Vector                           noiseModelBetweenBias;


  gtsam::PreintegratedMasMeasurements* masIntegratorOpt_;
  gtsam::PreintegratedMasMeasurements* masIntegratorPre_;

  std::deque<mrs_msgs::Float64ArrayStamped> masQueOpt;
  std::deque<mrs_msgs::Float64ArrayStamped> masQuePre;

  gtsam::Pose3                  prevPose_;
  gtsam::Vector3                prevLinVel_;
  gtsam::Vector3                prevAngVel_;
  gtsam::FullState              prevState_;
  gtsam::mas_bias::ConstantBias prevBias_;

  gtsam::mas_bias::ConstantBias initBias_;

  gtsam::FullState              prevStateOdom;
  gtsam::mas_bias::ConstantBias prevBiasOdom;

  bool   doneFirstOpt     = false;
  double lastMasT_predict = -1;
  double lastMasT_opt     = -1;

  gtsam::ISAM2                optimizer;
  gtsam::NonlinearFactorGraph graphFactors;
  gtsam::Values               graphValues;

  const double delta_t = 0;

  int key = 1;

  gtsam::Pose3 baselink2Lidar;

  bool is_initialized_ = false;

public:
  /*//{ onInit() */
  virtual void onInit() {
    ros::NodeHandle nh = nodelet::Nodelet::getMTPrivateNodeHandle();

    /*//{ load parameters */
    mrs_lib::ParamLoader pl(nh, "masPreintegration");

    pl.loadParam("uavName", uavName);

    pl.loadParam("lidarFrame", lidarFrame);
    pl.loadParam("baselinkFrame", baselinkFrame);
    pl.loadParam("odometryFrame", odometryFrame);

    pl.loadParam("mas/mass", mass);
    pl.loadParam("mas/gravity", gravity);
    pl.loadParam("mas/numMotors", numMotors);
    pl.loadParam("mas/propMass", propMass);
    pl.loadParam("mas/motorConstant", motorConstant);
    pl.loadParam("mas/momentConstant", momentConstant);

    pl.loadParam("mas/linAccBiasNoise", linAccBiasNoise);
    pl.loadParam("mas/angAccBiasNoise", angAccBiasNoise);

    pl.loadParam("mas/linAccNoise", linAccNoise);
    pl.loadParam("mas/angAccNoise", angAccNoise);

    double init_bias_acc_lin_x, init_bias_acc_lin_y, init_bias_acc_lin_z;
    pl.loadParam("mas/initBias/linear/x", init_bias_acc_lin_x);
    pl.loadParam("mas/initBias/linear/y", init_bias_acc_lin_y);
    pl.loadParam("mas/initBias/linear/z", init_bias_acc_lin_z);

    double init_bias_acc_ang_x, init_bias_acc_ang_y, init_bias_acc_ang_z;
    pl.loadParam("mas/initBias/angular/x", init_bias_acc_ang_x);
    pl.loadParam("mas/initBias/angular/y", init_bias_acc_ang_y);
    pl.loadParam("mas/initBias/angular/z", init_bias_acc_ang_z);

    if (!pl.loadedSuccessfully()) {
      ROS_ERROR("[MasPreintegration]: Could not load all parameters!");
      ros::shutdown();
    }
    /*//}*/

    initBias_ = gtsam::mas_bias::ConstantBias((gtsam::Vector3() << init_bias_acc_lin_x, init_bias_acc_lin_y, init_bias_acc_lin_z).finished(),
                                              (gtsam::Vector3() << init_bias_acc_ang_x, init_bias_acc_ang_y, init_bias_acc_ang_z).finished());

    subMas      = nh.subscribe<mrs_msgs::Float64ArrayStamped>("maslo/preintegration/mas_in", 10, &MasPreintegration::masHandler, this,
                                                         ros::TransportHints().tcpNoDelay());
    subOdometry = nh.subscribe<nav_msgs::Odometry>("maslo/preintegration/odom_mapping_incremental_in", 5, &MasPreintegration::odometryHandler, this,
                                                   ros::TransportHints().tcpNoDelay());

    pubPreOdometry = nh.advertise<nav_msgs::Odometry>("maslo/preintegration/odom_preintegrated_out", 10);
    pubLinAcc      = nh.advertise<geometry_msgs::Vector3Stamped>("maslo/preintegration/lin_acc_out", 10);
    pubAngAcc      = nh.advertise<geometry_msgs::Vector3Stamped>("maslo/preintegration/ang_acc_out", 10);
    pubLinAccBias  = nh.advertise<geometry_msgs::Vector3Stamped>("maslo/preintegration/lin_acc_bias_out", 10);
    pubAngAccBias  = nh.advertise<geometry_msgs::Vector3Stamped>("maslo/preintegration/ang_acc_bias_out", 10);

    boost::shared_ptr<gtsam::MasParams> p = gtsam::MasParams::MakeSharedU(gravity);
    p->setMass(mass + (float)numMotors * propMass);
    p->setMotorConstant(motorConstant);
    p->setMomentConstant(momentConstant);
    p->setNumRotors(numMotors);
    p->setRotorDirs(std::vector<int>{-1, -1, 1, 1});
    p->setAccelerationCovariance(gtsam::Matrix33::Identity(3, 3) * pow(linAccNoise, 2));              // acc white noise in continuous
    p->setAlphaCovariance(gtsam::Matrix33::Identity(3, 3) * pow(angAccNoise, 2));                     // acc white noise in continuous
    p->integrationCovariance = gtsam::Matrix33::Identity(3, 3) * pow(1e-4, 2);                        // error committed in integrating position from velocities
    gtsam::mas_bias::ConstantBias prior_mas_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());  // assume zero initial bias

    priorPoseNoise   = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished());  // rad,rad,rad,m, m, m
    priorLinVelNoise = gtsam::noiseModel::Isotropic::Sigma(3, 1e4);                                                               // m/s
    priorAngVelNoise = gtsam::noiseModel::Isotropic::Sigma(3, 1e4);                                                               // m/s
    priorBiasNoise   = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3);                                                           // 1e-2 ~ 1e-3 seems to be good
    correctionNoise  = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished());  // rad,rad,rad,m, m, m
    correctionNoise2 = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1, 1, 1, 1, 1, 1).finished());                 // rad,rad,rad,m, m, m
    noiseModelBetweenBias =
        (gtsam::Vector(6) << linAccBiasNoise, linAccBiasNoise, linAccBiasNoise, angAccBiasNoise, angAccBiasNoise, angAccBiasNoise).finished();

    masIntegratorPre_ = new gtsam::PreintegratedMasMeasurements(p);
    masIntegratorOpt_ = new gtsam::PreintegratedMasMeasurements(p);

    // get static transform from lidar to baselink
    tf::StampedTransform tfBaselink2Lidar;
    if (lidarFrame != baselinkFrame) {

      tf::TransformListener tfListener;

      bool tfFound = false;
      while (!tfFound) {
        try {
          ROS_WARN_THROTTLE(3.0, "[MasPreintegration]: Waiting for transform from: %s, to: %s.", baselinkFrame.c_str(), lidarFrame.c_str());
          tfListener.waitForTransform(baselinkFrame, lidarFrame, ros::Time(0), ros::Duration(3.0));
          tfListener.lookupTransform(baselinkFrame, lidarFrame, ros::Time(0), tfBaselink2Lidar);
          tfFound = true;
        }
        catch (tf::TransformException ex) {
          ROS_WARN_THROTTLE(3.0, "[MasPreintegration]: could not find transform from: %s, to: %s.", baselinkFrame.c_str(), lidarFrame.c_str());
        }
      }

      ROS_INFO("[MasPreintegration]: Found transform from: %s, to: %s.", baselinkFrame.c_str(), lidarFrame.c_str());

    } else {
      tfBaselink2Lidar.setOrigin(tf::Vector3(0.0, 0.0, 0.0));
      tfBaselink2Lidar.setRotation(tf::createQuaternionFromRPY(0.0, 0.0, 0.0));
    }
    tf::Quaternion rot = tfBaselink2Lidar.getRotation();
    tf::Vector3    pos = tfBaselink2Lidar.getOrigin();
    baselink2Lidar = gtsam::Pose3(gtsam::Rot3::Quaternion(rot.getW(), rot.getX(), rot.getY(), rot.getZ()), gtsam::Point3(pos.getX(), pos.getY(), pos.getZ()));


    ROS_INFO("\033[1;32m----> [MasPreintegration]: initialized.\033[0m");
    is_initialized_ = true;
  }
  /*//}*/

  /*//{ resetOptimization() */
  void resetOptimization() {
    gtsam::ISAM2Params optParameters;
    optParameters.relinearizeThreshold = 0.1;
    optParameters.relinearizeSkip      = 1;
    optimizer                          = gtsam::ISAM2(optParameters);

    gtsam::NonlinearFactorGraph newGraphFactors;
    graphFactors = newGraphFactors;

    gtsam::Values NewGraphValues;
    graphValues = NewGraphValues;
  }
  /*//}*/

  /*//{ resetParams() */
  void resetParams() {
    lastMasT_predict  = -1;
    doneFirstOpt      = false;
    systemInitialized = false;
  }
  /*//}*/

  /*//{ odometryHandler() */
  // callback of incremental odometry from mapping
  void odometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg) {

    if (!is_initialized_) {
      return;
    }

    ros::Time t_start = ros::Time::now();

    ROS_INFO_ONCE("[MasPreintegration]: odometryHandler first callback");
    std::lock_guard<std::mutex> lock(mtx);

    const double currentCorrectionTime = ROS_TIME(odomMsg);

    // make sure we have MAS data to integrate
    if (masQueOpt.empty()) {
      ROS_INFO_THROTTLE(1.0, "[MasPreintegration]: no data to integrate");
      return;
    }

    const float        p_x        = odomMsg->pose.pose.position.x;
    const float        p_y        = odomMsg->pose.pose.position.y;
    const float        p_z        = odomMsg->pose.pose.position.z;
    const float        r_x        = odomMsg->pose.pose.orientation.x;
    const float        r_y        = odomMsg->pose.pose.orientation.y;
    const float        r_z        = odomMsg->pose.pose.orientation.z;
    const float        r_w        = odomMsg->pose.pose.orientation.w;
    const bool         degenerate = (int)odomMsg->pose.covariance[0] == 1;
    const gtsam::Pose3 lidarPose  = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));


    // 0. initialize system
    if (!systemInitialized) {
      resetOptimization();

      // pop old MAS message
      while (!masQueOpt.empty()) {
        if (ROS_TIME(&masQueOpt.front()) < currentCorrectionTime - delta_t) {
          lastMasT_opt = ROS_TIME(&masQueOpt.front());
          masQueOpt.pop_front();
        } else {
          break;
        }
      }

      // initial pose
      prevPose_ = lidarPose;
      const gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, priorPoseNoise);
      graphFactors.add(priorPose);

      // initial linear velocity
      prevLinVel_ = gtsam::Vector3(0, 0, 0);
      const gtsam::PriorFactor<gtsam::Vector3> priorLinVel(V(0), prevLinVel_, priorLinVelNoise);
      graphFactors.add(priorLinVel);

      // initial angular velocity
      prevAngVel_ = gtsam::Vector3(0, 0, 0);
      const gtsam::PriorFactor<gtsam::Vector3> priorAngVel(V(0), prevAngVel_, priorAngVelNoise);
      graphFactors.add(priorAngVel);

      // initial bias
      const gtsam::PriorFactor<gtsam::mas_bias::ConstantBias> priorBias(B(0), initBias_, priorBiasNoise);
      graphFactors.add(priorBias);

      // add values
      graphValues.insert(X(0), prevPose_);
      graphValues.insert(V(0), prevLinVel_);
      graphValues.insert(W(0), prevAngVel_);
      graphValues.insert(B(0), initBias_);

      // optimize once
      optimizer.update(graphFactors, graphValues);
      graphFactors.resize(0);
      graphValues.clear();

      masIntegratorPre_->resetIntegration();
      masIntegratorOpt_->resetIntegration();

      key               = 1;
      systemInitialized = true;
      ROS_INFO_THROTTLE(1.0, "[MasPreintegration]: initialized graph");
      return;
    }

    // reset graph for speed
    if (key == 100) {
      ROS_INFO("[MasPreintegration]: resetting graph");
      // get updated noise before reset
      const gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise   = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(X(key - 1)));
      const gtsam::noiseModel::Gaussian::shared_ptr updatedLinVelNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(V(key - 1)));
      const gtsam::noiseModel::Gaussian::shared_ptr updatedAngVelNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(W(key - 1)));
      const gtsam::noiseModel::Gaussian::shared_ptr updatedBiasNoise   = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(B(key - 1)));
      // reset graph
      resetOptimization();
      // add pose
      const gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, updatedPoseNoise);
      graphFactors.add(priorPose);
      // add linear velocity
      const gtsam::PriorFactor<gtsam::Vector3> priorLinVel(V(0), prevLinVel_, updatedLinVelNoise);
      graphFactors.add(priorLinVel);
      // add angular velocity
      const gtsam::PriorFactor<gtsam::Vector3> priorAngVel(W(0), prevAngVel_, updatedAngVelNoise);
      graphFactors.add(priorAngVel);
      // add bias
      const gtsam::PriorFactor<gtsam::mas_bias::ConstantBias> priorBias(B(0), prevBias_, updatedBiasNoise);
      graphFactors.add(priorBias);
      // add values
      graphValues.insert(X(0), prevPose_);
      graphValues.insert(V(0), prevLinVel_);
      graphValues.insert(W(0), prevAngVel_);
      graphValues.insert(B(0), prevBias_);
      // optimize once
      optimizer.update(graphFactors, graphValues);
      graphFactors.resize(0);
      graphValues.clear();

      key = 1;
    }


    ros::Time t_reset           = ros::Time::now();
    bool      is_mas_integrated = false;

    // 1. integrate MAS data and optimize
    while (!masQueOpt.empty()) {
      // pop and integrate MAS data that is between two optimizations
      const mrs_msgs::Float64ArrayStamped* thisMas = &masQueOpt.front();
      const double                         masTime  = ROS_TIME(thisMas);
      if (masTime < currentCorrectionTime - delta_t) {
        const double dt = (lastMasT_opt < 0) ? (1.0 / 500.0) : (masTime - lastMasT_opt);

        if (dt <= 0) {
          ROS_WARN_COND(dt < 0, "invalid dt (opt): (%0.2f - %0.2f) = %0.2f", masTime, lastMasT_opt, dt);
          masQueOpt.pop_front();
          continue;
        }

        masIntegratorOpt_->integrateMeasurement(gtsam::Vector4(thisMas->values[0], thisMas->values[1], thisMas->values[2], thisMas->values[3]), dt);
        ROS_INFO_THROTTLE(1.0, "[MasPreintegration]: motor speeds: %.2f %.2f %.2f %.2f dt: %.2f", thisMas->values[0], thisMas->values[1], thisMas->values[2],
                          thisMas->values[3], dt);

        lastMasT_opt = masTime;
        masQueOpt.pop_front();
        is_mas_integrated = true;

      } else {
        break;
      }
    }

    if (!is_mas_integrated) {
      ROS_INFO("[MasPreintegration]: No motor speeds were integrated. Skipping optimization.");
      return;
    }

    ros::Time t_integrate = ros::Time::now();

    // add motor speed factor to graph
    const gtsam::PreintegratedMasMeasurements& preint_mas = dynamic_cast<const gtsam::PreintegratedMasMeasurements&>(*masIntegratorOpt_);
    const gtsam::MasFactor                     mas_factor(X(key - 1), V(key - 1), W(key - 1), X(key), V(key), W(key), B(key), preint_mas);
    const gtsam::Vector3                       lin_acc_b = preint_mas.lastAcc();
    const gtsam::Vector3                       ang_acc_b = preint_mas.lastAlpha();

    graphFactors.add(mas_factor);
    // add MAS bias between factor
    graphFactors.add(gtsam::BetweenFactor<gtsam::mas_bias::ConstantBias>(
        B(key - 1), B(key), gtsam::mas_bias::ConstantBias(), gtsam::noiseModel::Diagonal::Sigmas(sqrt(masIntegratorOpt_->deltaTij()) * noiseModelBetweenBias)));
    // add pose factor
    const gtsam::Pose3                     curPose = lidarPose;
    const gtsam::PriorFactor<gtsam::Pose3> pose_factor(X(key), curPose, degenerate ? correctionNoise2 : correctionNoise);
    graphFactors.add(pose_factor);
    // insert predicted values
    const gtsam::FullState propState_ = masIntegratorOpt_->predict(prevState_, prevBias_);
    graphValues.insert(X(key), propState_.pose());
    graphValues.insert(V(key), propState_.v());
    graphValues.insert(W(key), propState_.w());
    graphValues.insert(B(key), prevBias_);
    ROS_INFO_THROTTLE(1.0, "[MasPreintegration]: preintegrated: pos: %.2f %.2f %2.f rot: %.2f %.2f %.2f lin_vel: %.2f %.2f %.2f ang_vel: %.2f %.2f %.2f",
                      propState_.pose().translation().x(), propState_.pose().translation().y(), propState_.pose().translation().z(),
                      propState_.pose().rotation().roll(), propState_.pose().rotation().pitch(), propState_.pose().rotation().yaw(),
                      propState_.linVelocity()[0], propState_.linVelocity()[1], propState_.linVelocity()[2], propState_.angVelocity()[0],
                      propState_.angVelocity()[1], propState_.angVelocity()[2]);
    ROS_INFO_THROTTLE(1.0, "[MasPreintegration]: lin_acc_bias: %.2f %.2f %.2f ang_acc_bias: %.2f %.2f %.2f", prevBias_.linAcc()[0], prevBias_.linAcc()[1],
                      prevBias_.linAcc()[2], prevBias_.angAcc()[0], prevBias_.angAcc()[1], prevBias_.angAcc()[2]);

    // optimize
    /* cout << "****************************************************" << endl; */
    /* graphFactors.print("[MasPreintegration]: graph\n"); */
    optimizer.update(graphFactors, graphValues);
    optimizer.update();
    graphFactors.resize(0);
    graphValues.clear();
    // Overwrite the beginning of the preintegration for the next step.
    const gtsam::Values result = optimizer.calculateEstimate();
    prevPose_                  = result.at<gtsam::Pose3>(X(key));
    prevLinVel_                = result.at<gtsam::Vector3>(V(key));
    prevAngVel_                = result.at<gtsam::Vector3>(W(key));
    prevState_                 = gtsam::FullState(prevPose_, prevLinVel_, prevAngVel_);
    prevBias_                  = result.at<gtsam::mas_bias::ConstantBias>(B(key));
    // Reset the optimization preintegration object.
    masIntegratorOpt_->resetIntegration();
    // check optimization
    if (failureDetection(prevLinVel_, prevBias_)) {
      resetParams();
      return;
    }

    ros::Time t_optimization = ros::Time::now();

    // 2. after optimization, re-propagate MAS odometry preintegration
    prevStateOdom = prevState_;
    prevBiasOdom  = prevBias_;

    // first pop MAS message older than current correction data
    double lastMasQT = -1;
    while (!masQuePre.empty() && ROS_TIME(&masQuePre.front()) < currentCorrectionTime - delta_t) {
      lastMasQT = ROS_TIME(&masQuePre.front());
      masQuePre.pop_front();
    }
    // repropogate
    if (!masQuePre.empty()) {
      // reset bias use the newly optimized bias
      masIntegratorPre_->resetIntegration(); // TODO why not set bias here?
      // integrate MAS message from the beginning of this optimization
      ros::Time ms_stamp;
      for (int i = 0; i < (int)masQuePre.size(); ++i) {
        mrs_msgs::Float64ArrayStamped* thisMas = &masQuePre[i];
        ms_stamp                                = thisMas->header.stamp;
        const double masTime                    = ROS_TIME(thisMas);
        const double dt                         = (lastMasQT < 0) ? (1.0 / 500.0) : (masTime - lastMasQT);

        if (dt <= 0) {
          ROS_WARN_COND(dt < 0, "[MasPreintegration]: invalid dt (QT): (%0.2f - %0.2f) = %0.2f", masTime, lastMasQT, dt);
          continue;
        }

        masIntegratorPre_->integrateMeasurement(gtsam::Vector4(thisMas->values[0], thisMas->values[1], thisMas->values[2], thisMas->values[3]), dt);
        lastMasQT = masTime;
      }

      /* ROS_INFO("[MasPreintegration]: motor speed integration delay: %.4f", (ros::Time::now() - ms_stamp).toSec()); */

      const gtsam::Vector3          lin_acc_b = masIntegratorPre_->lastAcc();
      const gtsam::Vector3          ang_acc_b = masIntegratorPre_->lastAlpha();
      geometry_msgs::Vector3Stamped lin_acc_msg;
      gtsam::Vector3                lin_acc_w = prevPose_.rotation() * lin_acc_b;
      lin_acc_msg.header.stamp                = ros::Time::now();
      lin_acc_msg.header.frame_id             = "fcu";
      lin_acc_msg.vector.x                    = lin_acc_w[0];
      lin_acc_msg.vector.y                    = lin_acc_w[1];
      lin_acc_msg.vector.z                    = lin_acc_w[2];
      pubLinAcc.publish(lin_acc_msg);

      geometry_msgs::Vector3Stamped ang_acc_msg;
      ang_acc_msg.header.stamp    = ros::Time::now();
      ang_acc_msg.header.frame_id = "fcu";
      ang_acc_msg.vector.x        = ang_acc_b[0];
      ang_acc_msg.vector.y        = ang_acc_b[1];
      ang_acc_msg.vector.z        = ang_acc_b[2];
      pubAngAcc.publish(ang_acc_msg);
    }

    geometry_msgs::Vector3Stamped lin_acc_bias_msg;
    lin_acc_bias_msg.header.stamp    = ros::Time::now();
    lin_acc_bias_msg.header.frame_id = "fcu";
    lin_acc_bias_msg.vector.x        = prevBias_.linAcc()[0];
    lin_acc_bias_msg.vector.y        = prevBias_.linAcc()[1];
    lin_acc_bias_msg.vector.z        = prevBias_.linAcc()[2];
    pubLinAccBias.publish(lin_acc_bias_msg);

    geometry_msgs::Vector3Stamped ang_acc_bias_msg;
    ang_acc_bias_msg.header.stamp    = ros::Time::now();
    ang_acc_bias_msg.header.frame_id = "fcu";
    ang_acc_bias_msg.vector.x        = prevBias_.angAcc()[0];
    ang_acc_bias_msg.vector.y        = prevBias_.angAcc()[1];
    ang_acc_bias_msg.vector.z        = prevBias_.angAcc()[2];
    pubAngAccBias.publish(ang_acc_bias_msg);

    ++key;
    doneFirstOpt = true;
  }
  /*//}*/

  /*//{ failureDetection() */
  bool failureDetection(const gtsam::Vector3& velCur, const gtsam::mas_bias::ConstantBias& biasCur) {
    const Eigen::Vector3f vel(velCur.x(), velCur.y(), velCur.z());
    if (vel.norm() > 30) {
      ROS_WARN("[MasPreintegration]: Large velocity (%0.1f, %0.1f, %0.1f), reset MAS-preintegration!", vel.x(), vel.y(), vel.z());
      return true;
    }

    const Eigen::Vector3f bla(biasCur.linAcc().x(), biasCur.linAcc().y(), biasCur.linAcc().z());
    const Eigen::Vector3f baa(biasCur.angAcc().x(), biasCur.angAcc().y(), biasCur.angAcc().z());
    /* if (bla.norm() > 1.0 || baa.norm() > 1.0) { */
    /*   ROS_WARN("[MasPreintegration]: Large bias, reset MAS-preintegration!"); */
    /*   return true; */
    /* } */

    return false;
  }
  /*//}*/

  /*//{ masHandler() */
  void masHandler(const mrs_msgs::Float64ArrayStamped::ConstPtr& msg_in) {

    if (!is_initialized_) {
      return;
    }

    ROS_INFO_ONCE("[MasPreintegration]: masHandler first callback");
    std::lock_guard<std::mutex> lock(mtx);

    masQueOpt.push_back(*msg_in);
    masQuePre.push_back(*msg_in);

    if (doneFirstOpt == false) {
      ROS_INFO_THROTTLE(1.0, "[MasPreintegration]: waiting for first optimalization");
      return;
    }

    const double masTime = ROS_TIME(msg_in);
    const double dt      = (lastMasT_predict < 0) ? (1.0 / 500.0) : (masTime - lastMasT_predict);
    if (dt <= 0) {
      ROS_WARN_COND(dt < 0, "[MasPreintegration]: invalid dt (MAS): (%0.2f - %0.2f) = %0.2f", masTime, lastMasT_predict, dt);
      return;
    }
    lastMasT_predict = masTime;

    // integrate this single MAS message
    masIntegratorPre_->integrateMeasurement(gtsam::Vector4(msg_in->values[0], msg_in->values[1], msg_in->values[2], msg_in->values[3]), dt);

    // predict odometry
    const gtsam::FullState currentState = masIntegratorPre_->predict(prevStateOdom, prevBias_);

    // publish odometry
    nav_msgs::Odometry::Ptr odometry = boost::make_shared<nav_msgs::Odometry>();
    odometry->header.stamp           = msg_in->header.stamp;
    odometry->header.frame_id        = odometryFrame;
    odometry->child_frame_id         = baselinkFrame;

    // transform mas pose to lidar
    const gtsam::Pose3 masPose   = gtsam::Pose3(currentState.quaternion(), currentState.position());
    const gtsam::Pose3 lidarPose = masPose.compose(baselink2Lidar);

    odometry->pose.pose.position.x    = lidarPose.translation().x();
    odometry->pose.pose.position.y    = lidarPose.translation().y();
    odometry->pose.pose.position.z    = lidarPose.translation().z();
    odometry->pose.pose.orientation.x = lidarPose.rotation().toQuaternion().x();
    odometry->pose.pose.orientation.y = lidarPose.rotation().toQuaternion().y();
    odometry->pose.pose.orientation.z = lidarPose.rotation().toQuaternion().z();
    odometry->pose.pose.orientation.w = lidarPose.rotation().toQuaternion().w();

    odometry->twist.twist.linear.x  = currentState.linVelocity().x();
    odometry->twist.twist.linear.y  = currentState.linVelocity().y();
    odometry->twist.twist.linear.z  = currentState.linVelocity().z();
    odometry->twist.twist.angular.x = currentState.angVelocity().x();
    odometry->twist.twist.angular.y = currentState.angVelocity().y();
    odometry->twist.twist.angular.z = currentState.angVelocity().z();
    pubPreOdometry.publish(odometry);
  }
  /*//}*/
};
/*//}*/

}  // namespace mas_preintegration
}  // namespace maslo

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(maslo::mas_preintegration::MasPreintegration, nodelet::Nodelet)
