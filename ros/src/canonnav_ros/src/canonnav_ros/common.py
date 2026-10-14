"""Helpers shared by the CanonNav ROS nodes.

Frames: nodes work in the *camodom* frame, i.e. the odometry pose re-expressed at the
camera ground point with x forward / y left (T' = C^-1 T C, C = camodom_to_odom).
World points in this frame are converted back to the odom frame only for publishing.
"""
import os

import numpy as np
import yaml


def path_param(name, default=None):
    """File-path ROS parameter with `~` expanded (roslaunch passes `arg:=~/x` through literally)."""
    import rospy
    value = rospy.get_param(name) if default is None else rospy.get_param(name, default)
    return os.path.expanduser(value) if value else value


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def quat_to_rot(qx, qy, qz, qw):
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def odom_to_T(msg):
    """nav_msgs/Odometry -> 4x4 body-to-world transform."""
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_to_rot(q.x, q.y, q.z, q.w)
    T[:3, 3] = [p.x, p.y, p.z]
    return T


def transform_points(xyz, T):
    """(N, 3) points by a 4x4 transform."""
    return xyz @ T[:3, :3].T + T[:3, 3]


def load_goals_xy(path, interval=0.0, include_last=False, last_merge_dist=0.0):
    """Waypoint file (one `x y z` per line, world frame) -> goals (N, 3).

    interval > 0: resample so that consecutive goals are `interval` meters apart along
    the waypoint polyline (the first waypoint is skipped).
    include_last: always end with the last waypoint; if it is closer than
    `last_merge_dist` to the previous goal, it replaces that goal instead.
    """
    if not os.path.exists(path):
        return np.zeros((0, 3), dtype=np.float64)

    all_goals = []
    with open(path, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3 or parts[0].startswith("#"):
                continue
            all_goals.append([float(parts[0]), float(parts[1]), float(parts[2])])
    if not all_goals:
        return np.zeros((0, 3), dtype=np.float64)
    all_goals = np.array(all_goals, dtype=np.float64)
    if interval <= 0:
        return all_goals

    sampled = []
    cum_dist = 0.0
    next_threshold = interval
    for i in range(1, len(all_goals)):
        cum_dist += np.linalg.norm(all_goals[i] - all_goals[i - 1])
        if cum_dist >= next_threshold:
            sampled.append(all_goals[i])
            next_threshold += interval

    if include_last:
        last_pt = all_goals[-1]
        if not sampled:
            sampled.append(last_pt)
        else:
            d = float(np.linalg.norm(last_pt - sampled[-1]))
            if d <= 1e-6:
                pass                    # last waypoint already sampled
            elif d < last_merge_dist:
                sampled[-1] = last_pt   # merge into the previous goal
            else:
                sampled.append(last_pt)

    if not sampled:
        return np.zeros((0, 3), dtype=np.float64)
    return np.array(sampled, dtype=np.float64)
