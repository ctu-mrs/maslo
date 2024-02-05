cmake_minimum_required(VERSION 3.5)
project(maslo)


set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -pthread")
set(CMAKE_CXX_FLAGS_RELDEB "${CMAKE_CXX_FLAGS_RELDEB} -pthread")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(catkin REQUIRED COMPONENTS
  tf
  roscpp
  rospy
  cv_bridge
  nodelet
  pcl_conversions
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  message_generation
  visualization_msgs
  mrs_lib
  mrs_msgs
  mas_factor
)

find_package(PCL REQUIRED)
find_package(GTSAM REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread timer chrono)

add_message_files(
  DIRECTORY msg
  FILES
  cloud_info.msg
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
  nav_msgs
  sensor_msgs
  mrs_msgs
)

catkin_package(
  INCLUDE_DIRS include
  DEPENDS PCL GTSAM Boost

  CATKIN_DEPENDS 
  std_msgs
  nav_msgs
  geometry_msgs
  sensor_msgs
  message_runtime 
  message_generation
  visualization_msgs
  cv_bridge
  mrs_lib
  mrs_msgs
)

# add definitions
add_definitions(
    ${PCL_DEFINITIONS}
    ${GTSAM_DEFINITIONS}
  )

# include directories
include_directories(
	include
	${catkin_INCLUDE_DIRS}
	${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
	${GTSAM_INCLUDE_DIRS}
)

# link directories
link_directories(
	include
	${PCL_LIBRARY_DIRS}
  ${OpenCV_LIBRARY_DIRS}
  ${GTSAM_LIBRARY_DIRS}
)

###########
## Build ##
###########

# Range Image Projection
add_library(ImageProjection src/imageProjection.cpp)
add_dependencies(ImageProjection
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
  ${PROJECT_NAME}_generate_messages_cpp
  )
target_link_libraries(ImageProjection
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
  )

# Feature Association
add_library(FeatureExtraction src/featureExtraction.cpp)
add_dependencies(FeatureExtraction
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
  ${PROJECT_NAME}_generate_messages_cpp
  )
target_link_libraries(FeatureExtraction
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  )

# Mapping Optimization
add_library(MapOptimization src/mapOptimization.cpp)
add_dependencies(MapOptimization
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
  ${PROJECT_NAME}_generate_messages_cpp
  )
target_compile_options(MapOptimization
  PRIVATE
  ${OpenMP_CXX_FLAGS}
  )
target_link_libraries(MapOptimization
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${Boost_LIBRARIES}
  ${GTSAM_LIBRARIES}
  gtsam
  )

# Mas Preintegration
add_library(MasPreintegration src/masPreintegration.cpp)
add_dependencies(MasPreintegration
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
  ${PROJECT_NAME}_generate_messages_cpp
  )
target_link_libraries(MasPreintegration
  ${catkin_LIBRARIES}
  ${Boost_LIBRARIES}
  ${GTSAM_LIBRARIES}
  gtsam
  )

# Transform Fusion
add_library(TransformFusion src/transformFusion.cpp)
add_dependencies(TransformFusion
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
  )
target_link_libraries(TransformFusion
  ${catkin_LIBRARIES}
  )
