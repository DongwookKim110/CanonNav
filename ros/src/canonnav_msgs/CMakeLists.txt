cmake_minimum_required(VERSION 3.0.2)
project(canonnav_msgs)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  message_generation
  std_msgs
)

add_message_files(
  FILES
  CandidatePath2D.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  nav_msgs
)

catkin_package(
  CATKIN_DEPENDS std_msgs geometry_msgs message_runtime nav_msgs
)
