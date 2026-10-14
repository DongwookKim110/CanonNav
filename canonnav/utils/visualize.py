"""Inference visualization on the camera image: SoR ellipse, candidate paths, best path and goal.

Ground coordinates are (forward, left) in meters.
"""
from __future__ import annotations

import cv2
import numpy as np

SOR_COLOR = (0, 255, 255)        # BGR yellow
BEST_PATH_COLOR = (0, 255, 0)    # green
CAND_PATH_COLOR = (0, 125, 255)  # orange
GOAL_COLOR = (0, 0, 255)         # red


def project_ground_path_to_image(path_fl: np.ndarray, K: np.ndarray, camera_height: float,
                                 min_depth: float = 1e-4):
    """(T, 2) (forward, left) ground points -> (T, 2) pixel uv and (T,) validity (in front of camera)."""
    forward = path_fl[:, 0]
    lateral = path_fl[:, 1]

    X_cam = -lateral
    Y_cam = np.full_like(forward, camera_height)
    Z_cam = forward

    valid = Z_cam > min_depth
    pts_cam = np.stack([X_cam, Y_cam, Z_cam], axis=-1)

    uv = np.full((len(path_fl), 2), np.nan, dtype=np.float32)
    if np.any(valid):
        proj = (K @ pts_cam[valid].T).T
        uv[valid] = (proj[:, :2] / proj[:, 2:3]).astype(np.float32)
    return uv, valid


def sample_sor_ellipse_points(sor_mu: np.ndarray, sor_std: np.ndarray,
                              sigma_scale: float = 2.0, num_points: int = 100):
    """Ground-plane ellipse (N, 2) in (forward, left)."""
    mu_x, mu_y = np.asarray(sor_mu, dtype=np.float64).reshape(2)
    std_x, std_y = np.asarray(sor_std, dtype=np.float64).reshape(2)
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    forward = mu_x + sigma_scale * std_x * np.cos(theta)
    lateral = mu_y + sigma_scale * std_y * np.sin(theta)
    return np.stack([forward, lateral], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------- #
# Front image
# ---------------------------------------------------------------------------- #
def draw_sor_ellipse_on_front(rgb_bgr, sor_mu, sor_std, K, camera_height, sigma_scale=2.0,
                              color=SOR_COLOR, thickness=2, fill_alpha=0.3, num_points=60):
    """Ground-plane SoR ellipse projected onto the image."""
    h, w = rgb_bgr.shape[:2]
    pts_fl = sample_sor_ellipse_points(sor_mu, sor_std, sigma_scale, num_points)
    uv, valid = project_ground_path_to_image(pts_fl, K, camera_height)

    pts_img = []
    for (u, v), ok in zip(uv, valid):
        if not ok:
            continue
        ui, vi = int(round(u)), int(round(v))
        # drop far off-screen points for OpenCV drawing stability
        if -2 * w <= ui <= 3 * w and -2 * h <= vi <= 3 * h:
            pts_img.append([ui, vi])

    if len(pts_img) >= 3:
        pts_img = np.array(pts_img, dtype=np.int32).reshape(-1, 1, 2)
        overlay = rgb_bgr.copy()
        cv2.fillPoly(overlay, [pts_img], color=color, lineType=cv2.LINE_AA)
        rgb_bgr = cv2.addWeighted(overlay, fill_alpha, rgb_bgr, 1 - fill_alpha, 0)
        cv2.polylines(rgb_bgr, [pts_img], isClosed=True, color=color, thickness=thickness,
                      lineType=cv2.LINE_AA)
    return rgb_bgr


def draw_path_on_front(image, path_fl, K, camera_height, color, thickness=2):
    uv, valid = project_ground_path_to_image(path_fl, K, camera_height)
    for i in range(len(uv) - 1):
        if valid[i] and valid[i + 1]:
            p1 = (int(round(uv[i, 0])), int(round(uv[i, 1])))
            p2 = (int(round(uv[i + 1, 0])), int(round(uv[i + 1, 1])))
            cv2.line(image, p1, p2, color, thickness, lineType=cv2.LINE_AA)
    return image


def draw_goal_on_front(rgb_bgr, goal_rt, K, camera_height, color=GOAL_COLOR, radius=6,
                       text_scale=0.6, text_thickness=1):
    """Goal point (clamped into the image) and "(r, th)" text in the top-left corner."""
    h, w = rgb_bgr.shape[:2]
    r, theta = np.asarray(goal_rt, dtype=np.float64).reshape(2)
    goal_fl = np.array([[r * np.cos(theta), r * np.sin(theta)]], dtype=np.float32)
    uv, valid = project_ground_path_to_image(goal_fl, K, camera_height)
    if valid[0]:
        margin = 10
        u, v = int(round(uv[0, 0])), int(round(uv[0, 1]))
        cv2.circle(rgb_bgr, (min(max(u, margin), w - margin), min(max(v, margin), h - margin)),
                   radius, color, -1, lineType=cv2.LINE_AA)

    text = f"(r, th) = ({r:.2f}m, {np.degrees(theta):.1f}deg)"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(rgb_bgr, text, (15, 30), font, text_scale, (0, 0, 0), text_thickness + 2, cv2.LINE_AA)  # outline
    cv2.putText(rgb_bgr, text, (15, 30), font, text_scale, color, text_thickness, cv2.LINE_AA)
    return rgb_bgr


# ---------------------------------------------------------------------------- #
# Canvas
# ---------------------------------------------------------------------------- #
def _prepend_origin(path_fl: np.ndarray):
    """Paths start at the robot: prepend (0.01, 0)."""
    return np.concatenate([np.array([[0.01, 0.0]], dtype=path_fl.dtype), path_fl], axis=0)


def visualize_prediction(
    rgb: np.ndarray,              # (H, W, 3) RGB, uint8 or float in [0, 255]; image the model saw
    K: np.ndarray,                # (3, 3) intrinsic of `rgb`
    camera_height: float,
    sor_mu: np.ndarray,           # (2,) [m]
    sor_std: np.ndarray,          # (2,) [m]
    paths: np.ndarray,            # (S, T, 2) [m]
    best_index: int,
    goal_rt=None,                 # (r, theta); drawn when given (goal-directed mode)
    max_candidates: int = 8,
) -> np.ndarray:
    """Return the BGR camera image with the prediction drawn on it."""
    rgb_img = rgb if rgb.dtype == np.uint8 else rgb.clip(0, 255).astype(np.uint8)
    front = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    front = draw_sor_ellipse_on_front(front, sor_mu, sor_std, K, camera_height)
    for i in [i for i in range(len(paths)) if i != best_index][:max_candidates]:
        front = draw_path_on_front(front, _prepend_origin(paths[i]), K, camera_height, CAND_PATH_COLOR, thickness=2)
    if goal_rt is not None:
        front = draw_goal_on_front(front, goal_rt, K, camera_height)
    front = draw_path_on_front(front, _prepend_origin(paths[best_index]), K, camera_height, BEST_PATH_COLOR,
                               thickness=5)
    return front
