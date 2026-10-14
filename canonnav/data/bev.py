"""Traversability image -> bird's-eye-view (BEV) map by ground-plane back-projection."""
import cv2
import numpy as np

# BEV cell values
BEV_BLOCKED = 0          # visible, not traversable
BEV_OUT_OF_FOV = 100     # behind the camera, outside the image sides, or above the image
BEV_BELOW_IMAGE = 200    # in front of the camera but below the image bottom (not visible)
BEV_FREE = 255           # visible and traversable


def make_axis(min_val: float, max_val: float, res: float, inclusive: bool = True):
    """[min, min+res, ..., max] (inclusive) or [min, ..., < max]."""
    if inclusive:
        n = int(round((max_val - min_val) / res)) + 1
    else:
        n = int(np.floor((max_val - min_val) / res))
    return min_val + np.arange(n, dtype=np.float32) * res


def rx_pitch(pitch_rad: float) -> np.ndarray:
    """Rotation about the camera x axis (optical frame: x right, y down, z forward)."""
    c = np.cos(pitch_rad)
    s = np.sin(pitch_rad)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ], dtype=np.float32)


def project_camera_points(points_cam: np.ndarray, K: np.ndarray):
    """(N, 3) optical-frame points -> (N, 2) pixels and (N,) in-front mask."""
    X, Y, Z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
    valid_z = Z >= 0
    u = K[0, 0] * (X / (Z + 1e-6)) + K[0, 2]
    v = K[1, 1] * (Y / (Z + 1e-6)) + K[1, 2]
    return np.stack([u, v], axis=1), valid_z


def binary_image_to_bev(
    binary_img: np.ndarray,
    K: np.ndarray,
    camera_height: float,
    pitch_rad: float,
    res: float,
    forward_range,
    horizontal_range,
    inclusive: bool = True,
) -> np.ndarray:
    """Traversability image (0 / 255) -> BEV map (uint8, values BEV_*).

    Ground plane: y = camera_height in the level camera frame (x right, y down, z forward).
    bev[i, j] <-> forward = forward_range[0] + i*res, left = horizontal_range[0] + j*res.
    pitch_rad: camera pitch, + looks up.
    """
    if binary_img.ndim != 2:
        raise ValueError("binary_img must be a single-channel image")
    if K.shape != (3, 3):
        raise ValueError("K must be (3,3)")

    H_img, W_img = binary_img.shape[:2]
    z_vals = make_axis(forward_range[0], forward_range[1], res, inclusive=inclusive)
    x_vals = make_axis(horizontal_range[0], horizontal_range[1], res, inclusive=inclusive)

    # rows: forward, cols: left (= -x in the camera frame)
    Zg, Xg = np.meshgrid(z_vals, -x_vals, indexing="ij")
    Yg = np.full_like(Zg, camera_height, dtype=np.float32)
    nF, nH = Zg.shape

    pts_level = np.stack([Xg.reshape(-1), Yg.reshape(-1), Zg.reshape(-1)], axis=1).astype(np.float32)
    pts_cam = (rx_pitch(-pitch_rad) @ pts_level.T).T   # level -> pitched camera

    uv, valid_z = project_camera_points(pts_cam, K)
    u = uv[:, 0].reshape(nF, nH).astype(np.float32)
    v = uv[:, 1].reshape(nF, nH).astype(np.float32)
    valid_z = valid_z.reshape(nF, nH)

    bev = np.full((nF, nH), BEV_OUT_OF_FOV, dtype=np.uint8)
    inside_img = valid_z & (u >= 0) & (u <= W_img - 1) & (v >= 0) & (v <= H_img - 1)
    below_bottom = valid_z & (v > H_img - 1)

    sampled = cv2.remap(binary_img, u, v, interpolation=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    bev[inside_img] = sampled[inside_img]
    bev[below_bottom] = BEV_BELOW_IMAGE
    bev[0] = BEV_BELOW_IMAGE
    return bev
