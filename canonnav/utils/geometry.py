"""Geometry ops used by the model.

- project_path_fl_to_uv: (forward, left) ground path -> image pixel (u, v)
- sample_feat_map_at_uv: bilinear feature sampling at pixel locations
- bilinear_sample_with_grid_sample: bilinear sampling of a BEV map (training)
- curvature_reg_varstep: path curvature penalty (training)
"""

import torch
import torch.nn.functional as F


def project_path_fl_to_uv(
    path_fl: torch.Tensor,   # (B,T,2) (forward, left) in meters
    K: torch.Tensor,         # (B,3,3)
    cam_height: torch.Tensor # (B,)  camera height above ground (meters)
):
    """
    (forward, left) ground-plane path -> image pixel (u, v), per-batch K / height.
    Assumes a level camera (pitch = yaw = roll = 0).

    Frames:
      - Camera (OpenCV): x right, y down, z forward
      - A ground point is cam_height below the camera: y_cam = cam_height
      - (forward, left) -> x_cam = -left, z_cam = forward

    returns:
      uv: (B,T,2) pixel coords
    """
    assert path_fl.dim() == 3 and path_fl.size(-1) == 2
    assert K.shape[-2:] == (3, 3)
    assert cam_height.dim() == 1 and cam_height.size(0) == path_fl.size(0)

    B, T, _ = path_fl.shape
    device = path_fl.device
    dtype = path_fl.dtype

    K = K.to(device=device, dtype=dtype)
    cam_height = cam_height.to(device=device, dtype=dtype)

    xc = -path_fl[..., 1]                     # right+
    zc = path_fl[..., 0]                      # forward+
    yc = cam_height.view(B, 1).expand(B, T)   # down+

    fx = K[:, 0, 0].view(B, 1)
    fy = K[:, 1, 1].view(B, 1)
    cx = K[:, 0, 2].view(B, 1)
    cy = K[:, 1, 2].view(B, 1)

    z_safe = zc + 1e-6
    u = fx * (xc / z_safe) + cx
    v = fy * (yc / z_safe) + cy

    return torch.stack([u, v], dim=-1)  # (B,T,2)


def sample_feat_map_at_uv(
    feat_map: torch.Tensor,  # (B,C,Hf,Wf)
    uv: torch.Tensor,        # (B,T,2) in pixels of the input image
    img_hw: tuple,           # (H_img, W_img)
):
    B, C, Hf, Wf = feat_map.shape
    H_img, W_img = img_hw

    u = uv[..., 0]
    v = uv[..., 1]
    x_norm = 2.0 * (u / max(W_img - 1, 1)) - 1.0
    y_norm = 2.0 * (v / max(H_img - 1, 1)) - 1.0

    grid = torch.stack([x_norm, y_norm], dim=-1)  # (B,T,2)
    grid = grid.view(B, -1, 1, 2)                 # (B,T,1,2)

    sampled = F.grid_sample(
        feat_map, grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True
    )  # (B,C,T,1)

    return sampled.squeeze(-1).transpose(1, 2).contiguous()  # (B,T,C)


def bilinear_sample_with_grid_sample(
    image: torch.Tensor,   # (B,H,W)
    coords: torch.Tensor,  # (B,N,2) row, col in pixel coords
    padding_mode: str = "border",
) -> torch.Tensor:
    B, H, W = image.shape
    x = image[:, None, :, :]   # (B,1,H,W)

    r = coords[..., 0]
    c = coords[..., 1]
    # grid_sample expects (x, y) = (col, row) normalized to [-1, 1]
    x_norm = 2.0 * (c / (W - 1)) - 1.0
    y_norm = 2.0 * (r / (H - 1)) - 1.0

    grid = torch.stack([x_norm, y_norm], dim=-1)[:, None, :, :]   # (B,1,N,2)
    out = F.grid_sample(x, grid, mode="bilinear", padding_mode=padding_mode, align_corners=True)
    return out[:, 0, 0, :]   # (B,N)


def curvature_reg_varstep(p, delta=0.3, eps=1e-6):
    """Huber penalty on the discrete curvature of a (B,T,2) path that starts at the origin."""
    origin = torch.zeros((p.size(0), 1, p.size(2))).to(p.device)
    p = torch.cat([origin, p], dim=1)
    a = p[:, 1:-1] - p[:, :-2]          # (B,T-1,2)
    b = p[:, 2:] - p[:, 1:-1]           # (B,T-1,2)

    ds1 = torch.linalg.norm(a, dim=-1, keepdim=True).clamp_min(eps)
    ds2 = torch.linalg.norm(b, dim=-1, keepdim=True).clamp_min(eps)
    v1 = a / ds1
    v2 = b / ds2

    d2 = (2.0 / (ds1 + ds2)) * (v2 - v1)   # second derivative w.r.t. arc length
    r = torch.linalg.norm(d2, dim=-1)
    return F.huber_loss(r, torch.zeros_like(r), delta=delta, reduction='none')
