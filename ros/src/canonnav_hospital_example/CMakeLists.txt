cmake_minimum_required(VERSION 3.0.2)
project(canonnav_hospital_example)

find_package(catkin REQUIRED)
catkin_package()

catkin_install_python(PROGRAMS
  scripts/spawn_robot.py
  scripts/recovery_node.py
  scripts/set_sim_speed.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch urdf worlds models waypoints
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
