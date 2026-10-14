#!/usr/bin/env python3
"""Spawn the robot (~robot_description) in Gazebo at the start of a waypoint file.

Pose: the first waypoint, heading toward the first waypoint at least 1 m away.
~x/~y/~yaw override the waypoint-derived values when set.
"""
import math

import numpy as np
import rospy
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose

from canonnav_ros.common import load_goals_xy, path_param


def start_pose(waypoint_file, lookahead=1.0):
    pts = load_goals_xy(waypoint_file)
    if len(pts) == 0:
        raise RuntimeError(f"no waypoints in {waypoint_file}")
    x0, y0 = pts[0, 0], pts[0, 1]
    far = np.hypot(pts[:, 0] - x0, pts[:, 1] - y0) >= lookahead
    target = pts[np.argmax(far)] if far.any() else pts[-1]
    return x0, y0, math.atan2(target[1] - y0, target[0] - x0)


def main():
    rospy.init_node("spawn_robot")
    x, y, yaw = 0.0, 0.0, 0.0
    waypoint_file = path_param("~waypoint_file", "")
    if waypoint_file:
        x, y, yaw = start_pose(waypoint_file)
    x = float(rospy.get_param("~x", x))
    y = float(rospy.get_param("~y", y))
    yaw = float(rospy.get_param("~yaw", yaw))

    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = x, y, float(rospy.get_param("~z", 0.1))
    pose.orientation.z, pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)

    rospy.wait_for_service("/gazebo/spawn_urdf_model")
    spawn = rospy.ServiceProxy("/gazebo/spawn_urdf_model", SpawnModel)
    resp = spawn(rospy.get_param("~model_name", "jackal"), rospy.get_param("robot_description"),
                 "", pose, "world")
    if not resp.success:
        raise RuntimeError(resp.status_message)
    rospy.loginfo(f"[spawn] jackal at x={x:.2f} y={y:.2f} yaw={math.degrees(yaw):.1f} deg")


if __name__ == "__main__":
    main()
