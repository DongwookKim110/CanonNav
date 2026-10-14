cmake_minimum_required(VERSION 3.0.2)
project(canonnav_ros)

find_package(catkin REQUIRED COMPONENTS rospy canonnav_msgs)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS rospy canonnav_msgs
)

catkin_install_python(PROGRAMS
  scripts/canonnav_node.py
  scripts/tracker_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
