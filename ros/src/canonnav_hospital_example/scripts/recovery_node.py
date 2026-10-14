#!/usr/bin/env python3
"""Move the robot onto the active goal when it cannot reach it, so a demo run continues to the end.

The active goal is read from canonnav_node (/canonnav/current_goal). A goal counts as failed when
  - it has been active for longer than ~goal_timeout,
  - the robot moved less than ~stuck_min_dist within the last ~stuck_window, or
  - the robot is tilted by more than ~max_tilt_deg.
The robot is then placed on the goal (facing the direction it was heading to the goal); canonnav_node
sees the goal as reached and switches to the next one. Nothing happens while the robot is within
~reach_radius of the goal (e.g. after the final goal).
"""
import math
from collections import deque

import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry


class RecoveryNode:
    def __init__(self):
        self.model_name = rospy.get_param("~model_name", "jackal")
        self.goal_timeout = float(rospy.get_param("~goal_timeout", 48.0))     # [s]
        self.stuck_window = float(rospy.get_param("~stuck_window", 15.0))     # [s]
        self.stuck_min_dist = float(rospy.get_param("~stuck_min_dist", 1.0))  # [m]
        self.max_tilt_deg = float(rospy.get_param("~max_tilt_deg", 50.0))
        self.reach_radius = float(rospy.get_param("~reach_radius", 1.0))      # [m]
        self.grace = float(rospy.get_param("~grace", 5.0))                    # [s] no checks after a move
        self.spawn_z = float(rospy.get_param("~z", 0.1))

        self.pose = None            # (x, y, tilt_deg)
        self.history = deque()      # (t, x, y)
        self.goal = None            # (x, y)
        self.goal_since = None
        self.last_move = rospy.Time.now()
        self.failures = 0

        rospy.wait_for_service("/gazebo/set_model_state")
        self.set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        rospy.Subscriber("odom", Odometry, self.odom_cb, queue_size=10)
        rospy.Subscriber("/canonnav/current_goal", PointStamped, self.goal_cb, queue_size=1)
        rospy.Timer(rospy.Duration(0.5), self.check)

    def odom_cb(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        up_z = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)          # z component of the body z axis
        self.pose = (p.x, p.y, math.degrees(math.acos(max(-1.0, min(1.0, up_z)))))
        t = msg.header.stamp.to_sec()
        self.history.append((t, p.x, p.y))
        while self.history and t - self.history[0][0] > self.stuck_window:
            self.history.popleft()

    def goal_cb(self, msg):
        g = (msg.point.x, msg.point.y)
        if self.goal is None or math.hypot(g[0] - self.goal[0], g[1] - self.goal[1]) > 1e-3:
            self.goal, self.goal_since = g, rospy.Time.now()

    def failure_reason(self, now):
        x, y, tilt = self.pose
        if math.hypot(self.goal[0] - x, self.goal[1] - y) <= self.reach_radius:
            return None
        if tilt > self.max_tilt_deg:
            return f"tilted {tilt:.0f} deg"
        active = (now - self.goal_since).to_sec()
        if active > self.goal_timeout:
            return f"not reached in {self.goal_timeout:.0f} s"
        h = list(self.history)
        # only once the goal has been active for a full window (the robot waits before the first goal)
        if active >= self.stuck_window and len(h) >= 2 and h[-1][0] - h[0][0] >= self.stuck_window - 1.0:
            moved = sum(math.hypot(h[i][1] - h[i - 1][1], h[i][2] - h[i - 1][2]) for i in range(1, len(h)))
            if moved < self.stuck_min_dist:
                return f"moved {moved:.2f} m in {self.stuck_window:.0f} s"
        return None

    def check(self, _event):
        if self.pose is None or self.goal is None:
            return
        now = rospy.Time.now()
        if (now - self.last_move).to_sec() < self.grace:
            return
        reason = self.failure_reason(now)
        if reason is None:
            return
        self.failures += 1
        rospy.logwarn(f"[recovery] goal ({self.goal[0]:.2f}, {self.goal[1]:.2f}) failed: {reason} "
                      f"-> robot moved onto the goal (failures: {self.failures})")
        self.move_to_goal()

    def move_to_goal(self):
        x, y, _ = self.pose
        gx, gy = self.goal
        yaw = math.atan2(gy - y, gx - x)
        state = ModelState()
        state.model_name = self.model_name
        state.reference_frame = "world"
        state.pose.position.x, state.pose.position.y, state.pose.position.z = gx, gy, self.spawn_z
        state.pose.orientation.z, state.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        resp = self.set_state(state)
        if not resp.success:
            rospy.logerr(f"[recovery] set_model_state failed: {resp.status_message}")
        self.last_move = rospy.Time.now()
        self.goal_since = self.last_move
        self.history.clear()


if __name__ == "__main__":
    rospy.init_node("recovery")
    RecoveryNode()
    rospy.spin()
