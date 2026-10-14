#!/usr/bin/env python3
"""CanonNav inference node: camera image + odometry + waypoint goals -> SoR and best path.

Subscribes
  image                  sensor_msgs/CompressedImage
  odom                   nav_msgs/Odometry
Publishes
  canonnav/path          canonnav_msgs/CandidatePath2D  best path in the robot frame + per-waypoint traversability
  canonnav/vis/compressed  sensor_msgs/CompressedImage    prediction drawn on the camera image
  canonnav/current_goal  geometry_msgs/PointStamped     active waypoint goal (odom frame)

Goals come from a waypoint file (world = odom frame). The active goal advances when the robot
comes within goal.reach_radius of it. With inference.goal_directed=false the node explores.
"""
import math
import threading
from collections import deque

import cv2
import numpy as np
import rospy
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage

from canonnav.config import CameraCfg, load_config
from canonnav.inference import CanonNavPredictor
from canonnav.utils.visualize import visualize_prediction
from canonnav_msgs.msg import CandidatePath2D
from canonnav_ros.common import load_goals_xy, load_yaml, odom_to_T, path_param, transform_points


def bgr_to_compressed_msg(bgr, header, quality=90):
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    msg = CompressedImage()
    msg.header = header
    msg.format = "jpeg"
    msg.data = np.array(buf).tobytes()
    return msg


class CanonNavNode:
    def __init__(self):
        rospy.init_node("canonnav")

        cfg = load_yaml(path_param("~config"))
        net = load_config(path_param("~network_config")).network
        camera = load_config(path_param("~camera_config"), schema=CameraCfg)
        self.predictor = CanonNavPredictor(net, path_param("~checkpoint"), camera,
                                           device=rospy.get_param("~device", None) or None)
        self.robot_frame = rospy.get_param("~robot_frame", "base_link")

        inf = cfg["inference"]
        self.goal_directed = bool(inf["goal_directed"])
        self.sample_num = int(inf["sample_num"])
        vis = cfg["visualize"]
        self.max_candidates = int(vis["max_candidates"])

        # ----- odometry (camodom frame, see canonnav_ros.common) -----
        self.C = np.array(cfg["odom"]["camodom_to_odom"], dtype=np.float64)
        self.C_inv = np.linalg.inv(self.C)
        self.max_odom_sync_dt = float(cfg["odom"]["max_sync_dt"])
        self.odom_lock = threading.Lock()
        self.odom_buffer = deque(maxlen=200)
        self.have_odom = False
        self.p_w = np.zeros(3)
        self.R_wr = np.eye(3)
        self.odom_frame = "odom"

        # ----- waypoint goals -----
        goal_cfg = cfg["goal"]
        self.goal_reach_radius = float(goal_cfg["reach_radius"])
        self.goal_cooldown = float(goal_cfg["cooldown"])
        self.goal_lock = threading.Lock()
        self.goal_idx = 0
        self._last_goal_switch = None
        if self.goal_directed:
            # ~waypoint_file, else /waypoint_file (set by canonnav_hospital_example/hospital_world.launch)
            waypoint_file = path_param("~waypoint_file", "") or path_param("/waypoint_file", "")
            if not waypoint_file:
                raise RuntimeError("no waypoint file: pass waypoint_file:= (or set /waypoint_file)")
            self.goals_raw = load_goals_xy(
                waypoint_file, interval=float(goal_cfg["interval"]),
                include_last=bool(goal_cfg["include_last"]),
                last_merge_dist=float(goal_cfg["last_merge_dist"]))
            if self.goals_raw.shape[0] == 0:
                raise RuntimeError(f"no goals loaded from {waypoint_file}")
            self.goals_world = transform_points(self.goals_raw, self.C_inv)
            rospy.loginfo(f"[CanonNav] {len(self.goals_world)} goals from {waypoint_file}")

        # ----- previous best path (heading-consistency term of the path score) -----
        self.prev_lock = threading.Lock()
        self.prev_path = None   # (T, 2) in the robot frame at the time it was predicted
        self.prev_pose = None   # (p_w, R_wr) at that time

        # ----- ROS I/O; inference runs on a timer on the latest image only -----
        self.latest_img = None
        self.img_lock = threading.Lock()
        self.is_running = False
        self.pub_path = rospy.Publisher("canonnav/path", CandidatePath2D, queue_size=1)
        self.pub_vis = rospy.Publisher("canonnav/vis/compressed", CompressedImage, queue_size=1)
        self.pub_goal = rospy.Publisher("canonnav/current_goal", PointStamped, queue_size=1)
        rospy.Subscriber("image", CompressedImage, self.image_cb, queue_size=1, buff_size=2**24)
        rospy.Subscriber("odom", Odometry, self.odom_cb, queue_size=50)
        rospy.Timer(rospy.Duration(1.0 / float(inf["hz"])), self.timer_cb)
        rospy.loginfo(f"[CanonNav] ready ({'goal-directed' if self.goal_directed else 'exploration'})")

    # ------------------------------------------------------------------ #
    # callbacks
    # ------------------------------------------------------------------ #
    def image_cb(self, msg):
        with self.img_lock:
            self.latest_img = msg

    def odom_cb(self, msg):
        T = self.C_inv @ odom_to_T(msg) @ self.C
        stamp = msg.header.stamp if msg.header.stamp != rospy.Time(0) else rospy.Time.now()
        with self.odom_lock:
            self.odom_buffer.append((stamp, T[:3, 3].copy(), T[:3, :3].copy()))
            self.p_w, self.R_wr = T[:3, 3].copy(), T[:3, :3].copy()
            self.have_odom = True
            self.odom_frame = msg.header.frame_id or self.odom_frame
        if self.goal_directed:
            self.advance_goal(stamp)

    def timer_cb(self, _event):
        if self.is_running:
            return
        with self.img_lock:
            msg, self.latest_img = self.latest_img, None
        if msg is None:
            return
        self.is_running = True
        try:
            self.run_inference(msg)
        except Exception as e:
            rospy.logerr(f"[CanonNav] inference failed: {e}")
        finally:
            self.is_running = False

    # ------------------------------------------------------------------ #
    # goals
    # ------------------------------------------------------------------ #
    def advance_goal(self, stamp):
        """Switch to the next goal once within reach_radius (with a cooldown)."""
        if self._last_goal_switch is not None and \
                (stamp - self._last_goal_switch).to_sec() < self.goal_cooldown:
            return
        with self.odom_lock:
            p_w = self.p_w.copy()
        with self.goal_lock:
            g = self.goals_world[self.goal_idx]
            dist = math.hypot(g[0] - p_w[0], g[1] - p_w[1])
            if dist > self.goal_reach_radius:
                return
            self._last_goal_switch = stamp
            n = len(self.goals_world)
            if self.goal_idx < n - 1:
                self.goal_idx += 1
                rospy.loginfo(f"[CanonNav] goal {self.goal_idx - 1} reached -> goal {self.goal_idx}/{n - 1}")
            else:
                rospy.loginfo_throttle(5.0, f"[CanonNav] final goal reached ({dist:.2f} m)")

    def goal_rtheta(self, p_w, R_wr):
        """Active goal as (r, theta) in the robot frame at pose (p_w, R_wr)."""
        with self.goal_lock:
            g = self.goals_world[self.goal_idx].copy()
        g_rel = R_wr.T @ (g - p_w)
        return math.hypot(g_rel[0], g_rel[1]), math.atan2(g_rel[1], g_rel[0])

    def publish_goal(self, stamp):
        with self.goal_lock:
            g = self.goals_raw[self.goal_idx]
        msg = PointStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.odom_frame
        msg.point.x, msg.point.y, msg.point.z = float(g[0]), float(g[1]), float(g[2])
        self.pub_goal.publish(msg)

    # ------------------------------------------------------------------ #
    # odometry helpers
    # ------------------------------------------------------------------ #
    def synced_pose(self, stamp):
        """Odometry pose closest in time to `stamp`, or None if farther than max_sync_dt."""
        with self.odom_lock:
            if not self.odom_buffer:
                return None
            t, p_w, R_wr = min(self.odom_buffer, key=lambda x: abs((x[0] - stamp).to_sec()))
        if abs((t - stamp).to_sec()) > self.max_odom_sync_dt:
            return None
        return p_w, R_wr

    def prev_path_in_current_frame(self):
        with self.prev_lock:
            if self.prev_path is None:
                return None
            path, (p0, R0) = self.prev_path.copy(), self.prev_pose
        with self.odom_lock:
            p1, R1 = self.p_w.copy(), self.R_wr.copy()
        path3 = np.concatenate([path, np.zeros((len(path), 1))], axis=1)
        world = path3 @ R0.T + p0
        return ((world - p1) @ R1)[:, :2]

    def save_prev_path(self, path):
        with self.odom_lock:
            pose = (self.p_w.copy(), self.R_wr.copy())
        with self.prev_lock:
            self.prev_path, self.prev_pose = path.copy(), pose

    # ------------------------------------------------------------------ #
    # inference
    # ------------------------------------------------------------------ #
    def run_inference(self, msg):
        bgr = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            rospy.logwarn("[CanonNav] failed to decode image")
            return
        stamp = msg.header.stamp if msg.header.stamp != rospy.Time(0) else rospy.Time.now()
        pose = self.synced_pose(stamp)
        if pose is None:
            rospy.logwarn_throttle(1.0, "[CanonNav] no odometry close to the image stamp")
            return

        goal_rt = self.goal_rtheta(*pose) if self.goal_directed else (0.0, 0.0)
        res = self.predictor(bgr, goal_rt=goal_rt, goal_directed=self.goal_directed,
                             sample_num=self.sample_num, prev_path=self.prev_path_in_current_frame())
        best_path = res.paths[res.best_index]
        best_trav = res.traversability[res.best_index]
        self.save_prev_path(best_path)

        vis = visualize_prediction(
            rgb=res.warped_rgb, K=self.predictor.K, camera_height=self.predictor.camera_height,
            sor_mu=res.sor_mu, sor_std=res.sor_std, paths=res.paths,
            best_index=res.best_index, goal_rt=goal_rt if self.goal_directed else None,
            max_candidates=self.max_candidates,
        )
        vis_msg = bgr_to_compressed_msg(vis, msg.header)
        if vis_msg is not None:
            self.pub_vis.publish(vis_msg)

        path_msg = CandidatePath2D()
        path_msg.path.header.stamp = msg.header.stamp
        path_msg.path.header.frame_id = self.robot_frame
        for x, y in best_path:
            ps = PoseStamped()
            ps.header = path_msg.path.header
            ps.pose.position.x, ps.pose.position.y = float(x), float(y)
            ps.pose.orientation.w = 1.0
            path_msg.path.poses.append(ps)
        path_msg.traversability = best_trav.tolist()
        self.pub_path.publish(path_msg)

        if self.goal_directed:
            self.publish_goal(msg.header.stamp)


if __name__ == "__main__":
    CanonNavNode()
    rospy.spin()
