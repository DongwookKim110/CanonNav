"""Framework-free single-image inference: raw camera image + goal -> SoR, candidate paths, best path.

Shared by the image demo (scripts/demo_image.py) and the ROS node.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from .config import CameraCfg, NetworkCfg
from .models.build import build_model, load_checkpoint
from .utils.image import normalize_image


@dataclass
class InferenceResult:
    sor_mu: np.ndarray        # (2,) [m], (forward, left)
    sor_std: np.ndarray       # (2,) [m]
    paths: np.ndarray         # (S, T, 2) [m]
    traversability: np.ndarray  # (S, T) per-waypoint traversability in [0, 1]
    best_index: int
    warped_rgb: np.ndarray    # (H, W, 3) float32 RGB [0, 255], image in the model intrinsic K


def expected_reach_point(paths: np.ndarray, traversability: np.ndarray) -> np.ndarray:
    """Expected stopping point along each path, treating per-waypoint traversability as a
    per-meter survival probability. paths (S, T, 2), traversability (S, T) -> (S, 2)."""
    S, T, _ = paths.shape
    pts = np.concatenate([np.zeros((S, 1, 2)), paths], axis=1)   # (S, T+1, 2)
    seg_len = np.linalg.norm(pts[:, 1:] - pts[:, :-1], axis=-1)  # (S, T)

    seg_survival = traversability ** seg_len
    reach_prefix = np.ones((S, T))
    if T > 1:
        reach_prefix[:, 1:] = np.cumprod(seg_survival[:, :-1], axis=1)

    lam = -np.log(traversability)
    g = np.empty((S, T))
    small = lam < 1e-12
    g[small] = seg_len[small]
    g[~small] = (1.0 - np.exp(-lam[~small] * seg_len[~small])) / lam[~small]
    expected_length = np.sum(reach_prefix * g, axis=1)

    cum_len = np.cumsum(seg_len, axis=1)
    idx = np.clip(np.sum(cum_len < expected_length[:, None], axis=1), 0, T - 1)
    rows = np.arange(S)
    prev_cum = np.where(idx > 0, cum_len[rows, np.maximum(idx - 1, 0)], 0.0)

    alpha = np.clip((expected_length - prev_cum) / np.maximum(seg_len[rows, idx], 1e-6), 0.0, 1.0)
    return (1.0 - alpha[:, None]) * pts[rows, idx] + alpha[:, None] * pts[rows, idx + 1]


def select_best_path(paths: np.ndarray, traversability: np.ndarray, goal_rt: Tuple[float, float],
                     goal_directed: bool, prev_path: Optional[np.ndarray] = None) -> int:
    """Score candidates by goal progress + travel distance of the expected reach point,
    plus heading consistency with the previous best path (in the current robot frame)."""
    reach = expected_reach_point(paths, traversability)
    r, theta = goal_rt
    goal_xy = np.array([r * np.cos(theta), r * np.sin(theta)])

    progress = r - np.linalg.norm(goal_xy - reach, axis=1)
    progress_score = np.clip(progress / 5, a_min=-9999, a_max=1)
    travel_score = np.clip(np.linalg.norm(reach, axis=1) / 5, a_min=0, a_max=1)
    score = 0.6 * progress_score + 0.4 * travel_score if goal_directed else travel_score

    if prev_path is not None:
        ref_theta = np.arctan2(prev_path[None, :, 1], prev_path[None, :, 0])
        path_theta = np.arctan2(paths[:, :, 1], paths[:, :, 0])
        diff = (path_theta - ref_theta + np.pi) % (2 * np.pi) - np.pi
        mean_diff = np.mean(np.abs(diff), axis=1) / (np.pi / 4)
        score = score + np.clip(1 - mean_diff, 0, 1.0) * 0.2

    return int(np.argmax(score))


class CanonNavPredictor:
    """Holds the model and the fixed camera -> model-intrinsic warp."""

    def __init__(self, net: NetworkCfg, checkpoint: str, camera: CameraCfg,
                 device: Optional[str] = None):
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.model = build_model(net).to(self.device)
        load_checkpoint(self.model, checkpoint)
        self.model.eval()

        self.W, self.H = net.W, net.H
        self.K = np.array(net.K, dtype=np.float64)
        self.camera_height = float(camera.height)

        # warp raw image to the model intrinsic, removing the camera pitch
        # (same convention and rotation as the training labels / calib.yaml: + looks up, - looks down)
        pitch_rad = math.radians(camera.pitch)
        R_pitch = np.array([
            [1, 0, 0],
            [0, math.cos(pitch_rad), -math.sin(pitch_rad)],
            [0, math.sin(pitch_rad), math.cos(pitch_rad)],
        ], dtype=np.float64)
        self.homography = self.K @ R_pitch @ np.linalg.inv(np.array(camera.K, dtype=np.float64))

        self.K_t = torch.from_numpy(self.K).to(dtype=torch.float64, device=self.device).unsqueeze(0)
        self.camera_height_t = torch.tensor([self.camera_height], device=self.device)

    def preprocess(self, bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        warped = cv2.warpPerspective(rgb, self.homography, (self.W, self.H),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        tensor = torch.from_numpy(normalize_image(warped / 255.0))
        return warped, tensor.unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def __call__(self, bgr: np.ndarray, goal_rt: Tuple[float, float] = (0.0, 0.0),
                 goal_directed: bool = True, sample_num: int = 30,
                 prev_path: Optional[np.ndarray] = None) -> InferenceResult:
        """bgr: raw camera image. goal_rt: (r [m], theta [rad], left +) in the robot frame.
        goal_directed=False runs exploration (goal ignored)."""
        warped, x = self.preprocess(bgr)
        goal = torch.tensor([[list(goal_rt)]], dtype=torch.float32, device=self.device)
        goal[..., 0] /= self.camera_height  # network works in camera-height units
        nav_type = torch.tensor([[1.0 if goal_directed else 0.0]], device=self.device)

        pred = self.model.predict(rgb=x, goal=goal, nav_type=nav_type,
                                  camera_height=self.camera_height_t, K=self.K_t,
                                  sample_num=sample_num)

        paths = pred.path.cpu().numpy()
        traversability = np.clip(pred.traversability.cpu().numpy(), 1e-6, 1.0)
        best = select_best_path(paths, traversability, goal_rt, goal_directed, prev_path)
        return InferenceResult(
            sor_mu=pred.sor_mu.reshape(2).cpu().numpy(),
            sor_std=pred.sor_std.reshape(2).cpu().numpy(),
            paths=paths, traversability=traversability, best_index=best, warped_rgb=warped,
        )
