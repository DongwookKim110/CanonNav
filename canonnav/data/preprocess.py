"""Training-label generation from recorded sequences.

For a sampled frame, the robot's recorded future trajectory (from odometry) is expressed in the
ground-aligned camera frame, resampled every `traj.interp_step_m` meters, and checked step by step
against the BEV of the frame's traversability mask. Each step yields a goal (r, theta) and the SoR
target for that goal: the last trajectory point that is still visible and safe (the robot footprint
is mostly free on the BEV). After the first unsafe/unseen point, the SoR stays at the last safe one.

Per frame, <seq>/<label_folder>/<stamp>.npz holds
  goal_sor : (N, 4) [r, theta, sor_x, sor_y]   x forward, y left, meters (ground-aligned camera frame)
  traj     : (N, 4, 4) future camera poses in the ground-aligned camera frame of this frame
  bev      : (nF, nH) uint8 BEV map (canonnav.data.bev values)
  bev_sdf  : (nF, nH) signed distance to non-traversable cells [m], + outside, - inside
  meta     : dict(K, stamp, cam_height, T_C2alignedC, T_alignedC2odom, pitch [rad],
                  bev_cell_size, bev_forward_range, bev_horizontal_range)

Frames are taken at most every 1/velocity seconds, where velocity comes from calib.yaml.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation, Slerp

from ..config import PreprocessCfg
from .bev import BEV_BELOW_IMAGE, BEV_FREE, binary_image_to_bev


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
@dataclass
class Calib:
    W: int
    H: int
    K: np.ndarray
    T_C2alignedC: np.ndarray      # camera -> ground-aligned camera (pitch removed)
    T_alignedC2odom: np.ndarray   # ground-aligned camera -> odometry body
    camera_height: float
    body_edge: np.ndarray         # (4, 4) robot footprint corners (homogeneous) on the ground, aligned camera frame
    pitch: float                  # [rad], + looks up
    velocity: float               # [m/s], frames are sampled every 1/velocity s
    sample_d_gap: float           # min spacing of raw future poses before interpolation [m]
    sample_d_max: float           # future trajectory length limit [m]


def load_calib(path) -> Calib:
    with open(path, "r") as f:
        calib = yaml.safe_load(f)
    K = np.array(calib["intrinsic"]["K"])
    assert "C2Odom" in calib["extrinsic"]
    pitch = calib["extrinsic"]["pitch"] * np.pi / 180.0
    T_C2odom = np.array(calib["extrinsic"]["C2Odom"])
    T_C2alignedC = np.array([[1, 0, 0, 0],
                             [0, np.cos(pitch), -np.sin(pitch), 0],
                             [0, np.sin(pitch), np.cos(pitch), 0],
                             [0, 0, 0, 1]])
    T_alignedC2odom = T_C2odom @ np.linalg.inv(T_C2alignedC)
    camera_height = calib["extrinsic"]["camera_height"]

    [[left, right, backward, forward]] = calib["extrinsic"]["body_edge"]
    body_edge = np.array([[left, camera_height, forward, 1.0],
                          [right, camera_height, forward, 1.0],
                          [right, camera_height, backward, 1.0],
                          [left, camera_height, backward, 1.0]])

    return Calib(W=calib["intrinsic"]["W"], H=calib["intrinsic"]["H"], K=K,
                 T_C2alignedC=T_C2alignedC, T_alignedC2odom=T_alignedC2odom,
                 camera_height=camera_height, body_edge=body_edge, pitch=pitch,
                 velocity=calib["velocity"], sample_d_gap=calib["sample_d_gap"],
                 sample_d_max=calib["sample_d_max"])


def _sorted_by_stamp(folder):
    return sorted(os.listdir(folder), key=lambda f: float(os.path.splitext(f)[0]))


def load_sequence(seq: Path, cfg: PreprocessCfg):
    """-> (rgb file names, trav-map file names, (N, 4, 4) odometry poses), paired by sorted order."""
    ds = cfg.dataset
    rgb_files = _sorted_by_stamp(seq / ds.rgb_folder)
    trav_files = _sorted_by_stamp(seq / ds.trav_map_folder)
    pose_csv = pd.read_csv(seq / ds.odometry_file)

    xyz = pose_csv[["position_x", "position_y", "position_z"]].to_numpy(np.float64)
    q_xyzw = pose_csv[["orientation_x", "orientation_y", "orientation_z", "orientation_w"]].to_numpy(np.float64)
    q_xyzw /= np.linalg.norm(q_xyzw, axis=1, keepdims=True)

    pose = np.tile(np.eye(4), (len(pose_csv), 1, 1))
    pose[:, :3, :3] = Rotation.from_quat(q_xyzw).as_matrix()
    pose[:, :3, 3] = xyz
    return rgb_files, trav_files, pose[:len(rgb_files)]


def find_sequences(root, cfg: PreprocessCfg) -> List[Path]:
    """Every directory under `root` that contains an rgb folder."""
    seqs = [Path(p) for p, dirnames, _ in os.walk(root) if cfg.dataset.rgb_folder in dirnames]
    return sorted(seqs)


# --------------------------------------------------------------------------- #
# trajectory helpers
# --------------------------------------------------------------------------- #
def interpolate_poses_by_distance(Ts: np.ndarray, step_m: float, include_last: bool = True):
    """(N, 4, 4) poses -> poses every `step_m` meters along the path (slerp rotation, linear translation)."""
    assert Ts.ndim == 3 and Ts.shape[1:] == (4, 4)
    assert len(Ts) >= 2
    assert step_m > 0
    trans = Ts[:, :3, 3]
    rots = Rotation.from_matrix(Ts[:, :3, :3])
    seg_len = np.linalg.norm(trans[1:] - trans[:-1], axis=1)
    cumdist = np.concatenate([[0.0], np.cumsum(seg_len)])
    total_len = cumdist[-1]
    if total_len == 0:
        return Ts[:1].copy()

    targets = np.arange(0.0, total_len, step_m)
    if include_last and (len(targets) == 0 or not np.isclose(targets[-1], total_len)):
        targets = np.concatenate([targets, [total_len]])

    interp_rots = Slerp(cumdist, rots)(targets).as_matrix()
    interp_trans = np.zeros((len(targets), 3), dtype=np.float64)
    for k, u in enumerate(targets):
        i = np.searchsorted(cumdist, u, side="right") - 1
        i = min(max(i, 0), len(seg_len) - 1)
        denom = cumdist[i + 1] - cumdist[i]
        alpha = 0.0 if denom < 1e-12 else (u - cumdist[i]) / denom
        interp_trans[k] = (1.0 - alpha) * trans[i] + alpha * trans[i + 1]

    out = np.tile(np.eye(4), (len(targets), 1, 1))
    out[:, :3, :3] = interp_rots
    out[:, :3, 3] = interp_trans
    return out


def indices_min_gap(cumdist: np.ndarray, d: float, max_d: float) -> np.ndarray:
    """Greedy indices with consecutive cumulative distances >= d, stopping after max_d."""
    ts = np.asarray(cumdist).reshape(-1)
    if ts.size == 0:
        return np.array([], dtype=np.int64)
    idxs = [0]
    last = ts[0]
    for i in range(1, ts.size):
        if ts[i] - last >= d:
            idxs.append(i)
            last = ts[i]
        if last > max_d:
            break
    return np.asarray(idxs, dtype=np.int64)


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #
@dataclass
class StepVis:
    """Per-step drawing info (only used for visualization)."""
    color: Optional[tuple]      # BGR, None = not drawn
    bev_px: tuple               # (col, row)
    img_px: tuple               # (u, v)
    img_visible: bool           # ground point in front of the camera
    in_bev: bool
    prev_bev_px: Optional[tuple] = None   # also mark the previous step (first unsafe step)
    prev_img_px: Optional[tuple] = None


_GREEN, _RED, _BLUE = (0, 255, 0), (0, 0, 255), (255, 0, 0)


def make_frame_labels(bev: np.ndarray, pose_i: np.ndarray, future_odom: np.ndarray,
                      calib: Calib, cfg: PreprocessCfg):
    """Labels for one frame. -> (goal_sor (N, 4), traj (N, 4, 4), steps) or None if too few future poses."""
    b, t = cfg.bev, cfg.traj
    T_a2o = calib.T_alignedC2odom
    future = np.linalg.inv(T_a2o) @ np.linalg.inv(pose_i) @ future_odom @ T_a2o

    with_origin = np.vstack([np.eye(4)[None], future])
    dist = np.cumsum(np.linalg.norm(with_origin[1:, :3, 3] - with_origin[:-1, :3, 3], axis=1))
    future = future[indices_min_gap(dist, d=calib.sample_d_gap, max_d=calib.sample_d_max)]
    if future.shape[0] < t.min_traj_len:
        return None
    future = interpolate_poses_by_distance(future, step_m=t.interp_step_m)

    goal_sor, traj, steps = [], [], []
    first_unsafe = False
    prev_bev_px = prev_img_px = None
    body_point = np.array([0, calib.camera_height, 0, 1])
    for pose in future:
        x, y = pose[2, 3], -pose[0, 3]   # x forward, y left
        goal_r = math.sqrt(x * x + y * y)
        goal_th = math.atan2(y, x)

        # robot ground point and footprint on the BEV
        ground = pose @ body_point
        bev_px = np.zeros(2, dtype=np.int32)
        bev_px[0] = np.round((ground[2] - b.forward_range[0]) / b.cell_size).astype(np.int32)
        bev_px[1] = np.round((-ground[0] - b.horizontal_range[0]) / b.cell_size).astype(np.int32)
        in_bev = 0 <= bev_px[0] < bev.shape[0] and 0 <= bev_px[1] < bev.shape[1]

        edge = (pose @ calib.body_edge.T).T
        p_forward = np.round((edge[:, 2] - b.forward_range[0]) / b.cell_size).astype(np.int32)
        p_left = np.round((-edge[:, 0] - b.horizontal_range[0]) / b.cell_size).astype(np.int32)
        body_mask = np.zeros_like(bev, dtype=np.uint8)
        cv2.fillPoly(body_mask, np.stack([p_left, p_forward], axis=1).reshape(1, -1, 2), 1)

        # ground point in the original (pitched) camera frame, for visualization
        ground_cam = np.linalg.inv(calib.T_C2alignedC) @ ground
        img_px = calib.K @ ground_cam[:3]
        img_px = (img_px / (img_px[2] + 1e-6))[:2].astype(np.int32)
        step = StepVis(None, (int(bev_px[1]), int(bev_px[0])), (int(img_px[0]), int(img_px[1])),
                       bool(ground_cam[2] > 0), bool(in_bev))

        def keep_last_sor():
            if goal_sor:
                return np.array([goal_r, goal_th, goal_sor[-1][2], goal_sor[-1][3]])
            return np.array([goal_r, goal_th, x, y])

        if first_unsafe:
            assert len(goal_sor) > 0
            goal_sor.append(keep_last_sor())
            step.color = _RED
        elif in_bev and np.sum(body_mask) > 1:   # footprint appears ahead
            center_value = bev[bev_px[0], bev_px[1]]
            footprint = bev[body_mask == 1]
            safety_ratio = (np.sum(footprint == BEV_BELOW_IMAGE) + np.sum(footprint == BEV_FREE)) / footprint.shape[0]
            if safety_ratio > t.safety_ratio_threshold and center_value in [BEV_BELOW_IMAGE, BEV_FREE]:
                goal_sor.append(np.array([goal_r, goal_th, x, y]))
                step.color = _GREEN
            else:   # first unsafe step
                goal_sor.append(keep_last_sor())
                first_unsafe = True
                step.color = _RED
                step.prev_bev_px, step.prev_img_px = prev_bev_px, prev_img_px
        else:
            goal_sor.append(keep_last_sor())
            first_unsafe = True

        traj.append(pose)
        steps.append(step)
        prev_bev_px = step.bev_px
        prev_img_px = step.img_px

    if not first_unsafe and steps:   # whole trajectory safe: mark its end
        steps[-1] = StepVis(_BLUE, steps[-1].bev_px, steps[-1].img_px, steps[-1].img_visible, True)
    return np.array(goal_sor), np.array(traj), steps


def bev_sdf(bev: np.ndarray, cell_size: float) -> np.ndarray:
    """Signed distance to blocked cells [m]: + outside, - inside."""
    blocked = np.zeros_like(bev)
    blocked[bev == 0] = 1
    return distance_transform_edt(1 - blocked) * cell_size - distance_transform_edt(blocked) * cell_size


def draw_frame(bgr: np.ndarray, trav: np.ndarray, bev: np.ndarray, steps: List[StepVis]) -> np.ndarray:
    """[image | BEV] with the trajectory: green = safe (SoR candidates), red = unsafe, blue = end / last safe."""
    bgr = bgr.copy()
    bgr[trav == 0] = bgr[trav == 0] * 0.5 + np.array([0, 0, 255]) * 0.5
    bev_vis = np.expand_dims(bev, -1).repeat(3, axis=-1)
    r = bgr.shape[1] // 80
    for s in steps:
        if s.color is None:
            continue
        if s.in_bev:
            cv2.circle(bev_vis, s.bev_px, color=s.color, radius=3, thickness=-1)
        if s.img_visible:
            cv2.circle(bgr, center=s.img_px, radius=r, color=s.color, thickness=-1)
        if s.prev_bev_px is not None:
            cv2.circle(bev_vis, s.prev_bev_px, color=_BLUE, radius=3, thickness=-1)
            if s.img_visible:
                cv2.circle(bgr, center=s.prev_img_px, radius=r, color=_BLUE, thickness=-1)

    bev_vis = bev_vis[::-1, ::-1, :]
    w = int(bgr.shape[0] / bev_vis.shape[0] * bev_vis.shape[1])
    bev_vis = cv2.resize(bev_vis, (w, bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.concatenate([bgr, bev_vis], axis=1)


def process_sequence(seq: Path, calib: Calib, cfg: PreprocessCfg, vis: bool = False, progress=None):
    """Write labels (and optionally visualizations) for one sequence. -> (frames saved, frames total)."""
    ds, b, out = cfg.dataset, cfg.bev, cfg.output
    rgb_files, trav_files, odom = load_sequence(seq, cfg)
    label_dir = seq / out.label_folder
    label_dir.mkdir(exist_ok=True)
    if vis:
        (seq / out.vis_folder).mkdir(exist_ok=True)

    saved = 0
    previous_stamp = -99999
    frames = range(len(rgb_files)) if progress is None else progress(range(len(rgb_files)))
    for i in frames:
        stamp_str = os.path.splitext(rgb_files[i])[0]
        stamp = float(stamp_str)
        if stamp - previous_stamp < 1 / calib.velocity:
            continue

        trav = cv2.imread(str(seq / ds.trav_map_folder / trav_files[i]), cv2.IMREAD_GRAYSCALE)
        bev = binary_image_to_bev(trav, K=calib.K, camera_height=calib.camera_height, pitch_rad=calib.pitch,
                                  res=b.cell_size, forward_range=b.forward_range,
                                  horizontal_range=b.horizontal_range, inclusive=True)
        labels = make_frame_labels(bev, odom[i], odom[i + 1:], calib, cfg)
        if labels is None:
            continue
        goal_sor, traj, steps = labels
        if goal_sor.shape[0] == 0:
            continue

        meta = {"K": calib.K, "stamp": stamp_str, "cam_height": calib.camera_height,
                "T_C2alignedC": calib.T_C2alignedC, "T_alignedC2odom": calib.T_alignedC2odom,
                "pitch": calib.pitch, "bev_cell_size": b.cell_size,
                "bev_forward_range": b.forward_range, "bev_horizontal_range": b.horizontal_range}
        np.savez(label_dir / f"{stamp_str}.npz", goal_sor=goal_sor, traj=traj, bev=bev,
                 bev_sdf=bev_sdf(bev, b.cell_size), meta=meta)
        if vis:
            bgr = cv2.imread(str(seq / ds.rgb_folder / rgb_files[i]))
            cv2.imwrite(str(seq / out.vis_folder / f"{stamp_str}.jpg"), draw_frame(bgr, trav, bev, steps))
        previous_stamp = stamp
        saved += 1
    return saved, len(rgb_files)
