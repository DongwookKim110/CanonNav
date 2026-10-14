#!/usr/bin/env python3
"""CanonNav path tracker: best path + odometry -> cmd_vel.

Each new path (robot frame at its image stamp) is anchored once to the world frame with the
odometry at that stamp, then re-expressed in the current robot frame at every control step.
The tracking target is taken up to the *expected reach distance* of the path (per-waypoint
traversability read as a per-meter survival probability), capped by lookahead_dist.

Subscribes
  canonnav/path            canonnav_msgs/CandidatePath2D
  odom                     nav_msgs/Odometry
Publishes
  cmd_vel                  geometry_msgs/Twist
  canonnav/path_world      nav_msgs/Path                (odom frame, for RViz)
  canonnav/expected_point  geometry_msgs/PointStamped   (odom frame)
  canonnav/tracking_target geometry_msgs/PointStamped   (odom frame)

Control: proportional to the target in polar form (v = k_v r, w = k_w theta), followed by
saturation, acceleration limits and optional EMA smoothing. Any stale input -> zero command.
"""
import bisect
import math
import threading
from collections import deque

import numpy as np
import rospy
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from nav_msgs.msg import Odometry, Path

from canonnav_msgs.msg import CandidatePath2D
from canonnav_ros.common import load_yaml, odom_to_T, path_param, transform_points


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class TrackerNode:
    def __init__(self):
        rospy.init_node("canonnav_tracker")
        cfg = load_yaml(path_param("~config"))
        t = cfg["tracker"]

        self.C = np.array(cfg["odom"]["camodom_to_odom"], dtype=np.float64)
        self.C_inv = np.linalg.inv(self.C)

        self.max_v, self.max_w = float(t["max_v"]), float(t["max_w"])
        self.k_v, self.k_w = float(t["k_v"]), float(t["k_w"])
        self.turn_slowdown = bool(t["turn_slowdown"])
        self.theta_slowdown_gain = float(t["theta_slowdown_gain"])
        self.max_lin_accel = float(t["max_lin_accel"])   # <= 0 disables
        self.max_ang_accel = float(t["max_ang_accel"])   # <= 0 disables
        self.cmd_ema_alpha = float(t["cmd_ema_alpha"])   # 1.0 = no smoothing
        self.lookahead_dist = float(t["lookahead_dist"])
        self.stop_dist = float(t["stop_dist"])
        self.min_path_traversability = float(t["min_path_traversability"])
        self.max_path_age_sec = float(t["max_path_age_sec"])
        self.max_odom_age_sec = float(t["max_odom_age_sec"])
        self.odom_buffer_sec = float(t["odom_buffer_sec"])
        self.max_odom_match_dt = float(t["max_odom_match_dt"])
        self.use_only_past_odom = bool(t["use_only_past_odom"])

        self.prev_v = 0.0
        self.prev_w = 0.0
        self.prev_cmd_time = None

        self.lock = threading.Lock()
        self.have_odom = False
        self.odom_frame = "odom"
        self.latest_odom_stamp = rospy.Time(0)
        self.latest_T_wb = np.eye(4)
        self.odom_buf = deque()  # (stamp_sec, T_wb)

        self.latest_path_msg = None
        self.latest_path_trav = None
        self.latest_path_stamp = rospy.Time(0)
        self.latest_path_seq = 0

        self.anchored_path_seq = -1
        self.path_world_xyz = None
        self.path_trav = None
        self.path_world_msg_cache = None
        self.expected_point_world = None
        self.target_point_world = None

        self.pub_cmd = rospy.Publisher("cmd_vel", Twist, queue_size=1)
        self.pub_path_world = rospy.Publisher("canonnav/path_world", Path, queue_size=1)
        self.pub_expected = rospy.Publisher("canonnav/expected_point", PointStamped, queue_size=1)
        self.pub_target = rospy.Publisher("canonnav/tracking_target", PointStamped, queue_size=1)
        rospy.Subscriber("odom", Odometry, self.odom_cb, queue_size=50)
        rospy.Subscriber("canonnav/path", CandidatePath2D, self.path_cb, queue_size=1)
        rospy.Timer(rospy.Duration(1.0 / float(t["control_hz"])), self.control_cb)
        rospy.Timer(rospy.Duration(1.0 / float(t["viz_hz"])), self.viz_cb)
        rospy.loginfo("[Tracker] ready")

    # ------------------------------------------------------------------ #
    # odometry buffer
    # ------------------------------------------------------------------ #
    def _push_odom(self, stamp, T_wb):
        t = stamp.to_sec()
        self.odom_buf.append((t, T_wb.copy()))
        while self.odom_buf and self.odom_buf[0][0] < t - self.odom_buffer_sec:
            self.odom_buf.popleft()

    def _match_odom(self, stamp):
        """Odometry at `stamp`: latest one not after it (use_only_past_odom) or the nearest one."""
        if not self.odom_buf:
            return None, float("inf")
        t = stamp.to_sec()
        times = [item[0] for item in self.odom_buf]
        if self.use_only_past_odom:
            i = bisect.bisect_right(times, t) - 1
            if i < 0:
                return None, float("inf")
            return self.odom_buf[i][1].copy(), t - self.odom_buf[i][0]
        i = bisect.bisect_left(times, t)
        best = min((j for j in (i - 1, i) if 0 <= j < len(times)), key=lambda j: abs(times[j] - t))
        return self.odom_buf[best][1].copy(), abs(times[best] - t)

    # ------------------------------------------------------------------ #
    # path geometry
    # ------------------------------------------------------------------ #
    @staticmethod
    def _cumdist(path_xyz):
        with_origin = np.concatenate([np.zeros((1, 3)), path_xyz], axis=0)
        seglen = np.linalg.norm(with_origin[1:] - with_origin[:-1], axis=1)
        return with_origin, seglen, np.concatenate([[0.0], np.cumsum(seglen)])

    @staticmethod
    def _interpolate(with_origin, seglen, cumdist, target_dist):
        total_len = cumdist[-1]
        if total_len <= 1e-9:
            return with_origin[-1].copy()
        d = float(np.clip(target_dist, 0.0, total_len))
        idx = int(np.clip(np.searchsorted(cumdist, d, side="right") - 1, 0, len(seglen) - 1))
        d0, d1 = cumdist[idx], cumdist[idx + 1]
        if d1 - d0 < 1e-9:
            return with_origin[idx + 1].copy()
        alpha = (d - d0) / (d1 - d0)
        return (1.0 - alpha) * with_origin[idx] + alpha * with_origin[idx + 1]

    @staticmethod
    def _expected_reach(seglen, trav):
        """Expected travel distance and full-path survival, trav = per-meter survival per segment."""
        m = min(len(seglen), len(trav))
        if m <= 0:
            return 0.0, 0.0
        seglen = np.asarray(seglen[:m], dtype=np.float64)
        q = np.clip(np.asarray(trav[:m], dtype=np.float64), 1e-6, 1.0)
        seg_survival = q ** seglen
        reach_prefix = np.ones(m)
        if m > 1:
            reach_prefix[1:] = np.cumprod(seg_survival[:-1])
        lam = -np.log(q)
        g = np.empty(m)
        small = lam < 1e-12
        g[small] = seglen[small]
        g[~small] = (1.0 - np.exp(-lam[~small] * seglen[~small])) / lam[~small]
        return float(np.sum(reach_prefix * g)), float(np.prod(seg_survival))

    def _select_target(self, path_base_xyz, trav):
        """-> (target point or None to stop, expected reach point) in the robot frame."""
        with_origin, seglen, cumdist = self._cumdist(path_base_xyz)
        d_exp, total_survival = self._expected_reach(seglen, trav)
        expected_point = self._interpolate(with_origin, seglen, cumdist, d_exp)
        if total_survival < self.min_path_traversability or d_exp < self.stop_dist:
            return None, expected_point

        # target = mean of 12 points sampled on the path up to min(d_exp, lookahead)
        max_dist = float(np.clip(min(d_exp, self.lookahead_dist), 0.0, cumdist[-1]))
        if max_dist < 1e-6:
            return with_origin[0].copy(), expected_point
        dists = np.linspace(0.0, max_dist, 13)[1:]
        pts = [self._interpolate(with_origin, seglen, cumdist, d) for d in dists]
        return np.mean(pts, axis=0), expected_point

    def _anchor_latest_path(self):
        """Anchor a newly received path to the world frame using odometry at its stamp."""
        with self.lock:
            msg, seq, trav = self.latest_path_msg, self.latest_path_seq, self.latest_path_trav
            if msg is None:
                self.path_world_xyz = None
                self.anchored_path_seq = -1
                return
            if seq == self.anchored_path_seq:
                return
            stamp = msg.path.header.stamp if msg.path.header.stamp != rospy.Time(0) else rospy.Time.now()
            T_wb, dt = self._match_odom(stamp)

        if T_wb is None or dt > self.max_odom_match_dt:
            rospy.logwarn_throttle(1.0, f"[Tracker] cannot anchor path: odom dt={dt:.3f}s")
            with self.lock:
                self.path_world_xyz = None
                self.path_world_msg_cache = None
            return

        n = min(len(msg.path.poses), len(trav))
        base = np.array([[ps.pose.position.x, ps.pose.position.y, 0.0] for ps in msg.path.poses[:n]])
        world = transform_points(base, T_wb)
        path_msg = Path()
        path_msg.header.stamp = stamp
        path_msg.header.frame_id = self.odom_frame
        for x, y, z in transform_points(world, self.C):
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = x, y, z
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)

        with self.lock:
            self.path_world_xyz = world
            self.path_trav = trav[:n]
            self.path_world_msg_cache = path_msg
            self.anchored_path_seq = seq

    # ------------------------------------------------------------------ #
    # control
    # ------------------------------------------------------------------ #
    def _vw_p(self, tx, ty):
        r, theta = math.hypot(tx, ty), math.atan2(ty, tx)
        v, w = self.k_v * r, self.k_w * theta
        if self.turn_slowdown:
            v *= math.exp(-self.theta_slowdown_gain * abs(theta))
        return v, w

    def _command(self, target):
        tx, ty = float(target[0]), float(target[1])
        v, w = self._vw_p(tx, ty)
        v = clamp(v, -self.max_v, self.max_v)
        w = clamp(w, -self.max_w, self.max_w)

        now = rospy.Time.now().to_sec()
        if self.prev_cmd_time is not None:
            dt = now - self.prev_cmd_time
            if dt > 1e-3:
                if self.max_lin_accel > 0.0:
                    v = clamp(v, self.prev_v - self.max_lin_accel * dt, self.prev_v + self.max_lin_accel * dt)
                if self.max_ang_accel > 0.0:
                    w = clamp(w, self.prev_w - self.max_ang_accel * dt, self.prev_w + self.max_ang_accel * dt)
        if 0.0 < self.cmd_ema_alpha < 1.0:
            v = self.cmd_ema_alpha * v + (1.0 - self.cmd_ema_alpha) * self.prev_v
            w = self.cmd_ema_alpha * w + (1.0 - self.cmd_ema_alpha) * self.prev_w
        self.prev_v, self.prev_w, self.prev_cmd_time = v, w, now

        cmd = Twist()
        cmd.linear.x, cmd.angular.z = v, w
        return cmd

    def publish_stop(self):
        self.pub_cmd.publish(Twist())
        # restart from zero so acceleration limits ramp up smoothly
        self.prev_v, self.prev_w, self.prev_cmd_time = 0.0, 0.0, rospy.Time.now().to_sec()

    # ------------------------------------------------------------------ #
    # callbacks
    # ------------------------------------------------------------------ #
    def odom_cb(self, msg):
        stamp = msg.header.stamp if msg.header.stamp != rospy.Time(0) else rospy.Time.now()
        T_wb = self.C_inv @ odom_to_T(msg) @ self.C
        with self.lock:
            self.have_odom = True
            self.odom_frame = msg.header.frame_id or self.odom_frame
            self.latest_odom_stamp = stamp
            self.latest_T_wb = T_wb
            self._push_odom(stamp, T_wb)

    def path_cb(self, msg):
        n = min(len(msg.path.poses), len(msg.traversability))
        with self.lock:
            self.latest_path_seq += 1
            if n == 0:
                self.latest_path_msg, self.latest_path_trav = None, None
                self.latest_path_stamp = rospy.Time(0)
                return
            self.latest_path_msg = msg
            self.latest_path_trav = np.clip(np.asarray(msg.traversability[:n], dtype=np.float64), 0.0, 1.0)
            self.latest_path_stamp = msg.path.header.stamp if msg.path.header.stamp != rospy.Time(0) \
                else rospy.Time.now()

    def control_cb(self, _event):
        now = rospy.Time.now()
        with self.lock:
            have_odom, odom_stamp = self.have_odom, self.latest_odom_stamp
            T_wb = self.latest_T_wb.copy()
            path_stamp = self.latest_path_stamp

        if not have_odom:
            self.publish_stop()
            return
        odom_age = (now - odom_stamp).to_sec()
        if odom_age > self.max_odom_age_sec:
            rospy.logwarn_throttle(1.0, f"[Tracker] odom stale: {odom_age:.3f}s")
            self.publish_stop()
            return
        path_age = (now - path_stamp).to_sec() if path_stamp != rospy.Time(0) else float("inf")
        if path_age > self.max_path_age_sec:
            rospy.logwarn_throttle(1.0, f"[Tracker] path stale: {path_age:.3f}s")
            self.publish_stop()
            return

        self._anchor_latest_path()
        with self.lock:
            world = None if self.path_world_xyz is None else self.path_world_xyz.copy()
            trav = None if self.path_trav is None else self.path_trav.copy()
        if world is None or len(world) == 0:
            self.publish_stop()
            return

        # world path -> current robot frame
        R, p = T_wb[:3, :3], T_wb[:3, 3]
        path_base = (world - p) @ R
        target, expected = self._select_target(path_base, trav)

        to_world = lambda x: transform_points((x @ R.T + p)[None], self.C)[0]
        with self.lock:
            self.expected_point_world = to_world(expected)
            self.target_point_world = None if target is None else to_world(target)

        if target is None:
            self.publish_stop()
            return
        self.pub_cmd.publish(self._command(target))

    def viz_cb(self, _event):
        with self.lock:
            path_msg = self.path_world_msg_cache
            expected, target = self.expected_point_world, self.target_point_world
            stamp = self.latest_odom_stamp if self.latest_odom_stamp != rospy.Time(0) else rospy.Time.now()
            frame = self.odom_frame
        if path_msg is not None:
            self.pub_path_world.publish(path_msg)
        for pub, pt in ((self.pub_expected, expected), (self.pub_target, target)):
            if pt is not None:
                msg = PointStamped()
                msg.header.stamp, msg.header.frame_id = stamp, frame
                msg.point.x, msg.point.y, msg.point.z = (float(v) for v in pt)
                pub.publish(msg)


if __name__ == "__main__":
    TrackerNode()
    rospy.spin()
