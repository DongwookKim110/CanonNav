"""FullModel: DINOv3(+LoRA) -> side branch -> goal-conditioned decoder -> {SoR head, diffusion policy}.

Heights: the network works in camera-height-normalized units (meters / camera height).
`forward()` is the training path (returns the terms the losses need); `predict()` is inference:
it takes a normalized goal and returns SoR / paths in meters.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.geometry import (
    bilinear_sample_with_grid_sample,
    curvature_reg_varstep,
    project_path_fl_to_uv,
    sample_feat_map_at_uv,
)


@dataclass
class TrainOutput:
    sor_mu: torch.Tensor               # (B*P, 2) normalized SoR mean
    sor_std: torch.Tensor              # (B*P, 2)
    gt_noise: torch.Tensor             # (B*P, T, 2) diffusion target
    noise_pred: torch.Tensor           # (B*P, T, 2)
    gt_traversability: torch.Tensor    # (B*P, T) 1 safe, 0 unsafe, 0.5 unknown (outside BEV / image)
    traversability_logit: torch.Tensor # (B*P, T)
    collision_cost: torch.Tensor       # (B*P,) soft-max collision cost of the denoised path
    curv_reg: torch.Tensor             # (valid steps,) curvature penalty
    dist_soft: torch.Tensor            # (B*P,) soft-min SoR distance of the denoised path


@dataclass
class Prediction:
    sor_mu: torch.Tensor      # (B, 1, 2)  [m], (forward, left)
    sor_std: torch.Tensor     # (B, 1, 2)  [m]
    path: torch.Tensor        # (S, T, 2)  [m], S = sample_num candidate paths
    traversability: torch.Tensor  # (S, T) per-waypoint traversability in [0, 1]


class FullModel(nn.Module):
    def __init__(self,
                 encoder,                       # DinoBackbone
                 side_branch,
                 goal_conditioned_attn_layer,
                 sor_head,
                 noise_scheduler,
                 diffuser_noise_pred_net,
                 path_traversability_net,
                 pred_action_step=16,
                 # training-only geometry (BEV grid of the labels and supervision margins)
                 bev_cell_length=0.1,
                 bev_forward_start=0.0,
                 bev_horizontal_start=-32.0,
                 robot_size=0.25,
                 safety_margin=0.5,
                 path_consistency_tau=0.5,
                 ):
        super().__init__()
        self.encoder = encoder
        self.side_branch = side_branch
        self.goal_conditioned_attn_layer = goal_conditioned_attn_layer
        self.sor_head = sor_head
        self.diffuser_noise_pred_net = diffuser_noise_pred_net
        self.path_traversability_net = path_traversability_net

        self.noise_scheduler = noise_scheduler
        self.scheduler_cache = {}
        self.pred_action_step = pred_action_step
        self.bev_cell_length = bev_cell_length
        self.bev_forward_start = bev_forward_start
        self.bev_horizontal_start = bev_horizontal_start
        self.robot_size = robot_size
        self.safety_margin = safety_margin
        self.path_consistency_tau = path_consistency_tau

    def get_scheduler_for_device(self, device: torch.device):
        if device not in self.scheduler_cache:
            sched = copy.deepcopy(self.noise_scheduler)
            for name in ["alphas_cumprod", "alphas", "betas"]:
                if hasattr(sched, name):
                    t = getattr(sched, name)
                    if torch.is_tensor(t):
                        setattr(sched, name, t.to(device))
            if hasattr(sched, "timesteps") and torch.is_tensor(sched.timesteps):
                sched.timesteps = sched.timesteps.to(device)
            self.scheduler_cache[device] = sched
        return self.scheduler_cache[device]

    # ------------------------------------------------------------------ #
    # encoder -> side branch -> goal conditioning -> SoR head
    # ------------------------------------------------------------------ #
    def _condition(self, rgb, goal, nav_type):
        encoder_feat = self.encoder.encoder.get_intermediate_layers(
            rgb, n=[2, 5, 8, 11], reshape=True, norm=True, return_class_token=False,
        )
        side_tuned_feat = self.side_branch(encoder_feat)  # (B, C, h, w)

        img_context, goal_conditioned_feat, goal_embed = self.goal_conditioned_attn_layer(
            side_tuned_feat, goal=goal
        )
        # exploration (nav_type = 0) => goal features are masked out
        goal_conditioned_feat = goal_conditioned_feat * nav_type.unsqueeze(-1)
        goal_embed = goal_embed * nav_type.unsqueeze(-1)

        path_per_image = goal_conditioned_feat.size(1)
        img_context = img_context.unsqueeze(1).expand(-1, path_per_image, -1)

        cond_feat = torch.cat(
            [img_context, goal_embed, goal_conditioned_feat, nav_type.unsqueeze(-1)], dim=-1)
        sor_mu, sor_std = self.sor_head(cond_feat)  # each (B, N, 2)
        return side_tuned_feat, cond_feat, sor_mu, sor_std

    def forward(self, rgb, goal, bev, bev_sdf, action, nav_type, camera_height, K) -> TrainOutput:
        """Training step outputs.

        rgb:           (B, 3, H, W) normalized canonical image
        goal:          (B, P, 2) (r / camera_height, theta), P goals per image
        bev:           (B, nF, nH) BEV map of the labels
        bev_sdf:       (B, nF, nH) signed distance to blocked cells / camera_height
        action:        (B, P, T, 2) normalized path increments (expert)
        nav_type:      (B, P) 1 goal-directed, 0 exploration
        camera_height: (B,) [m]
        K:             (B, 3, 3) model intrinsic
        """
        side_tuned_feat, cond_feat, sor_mu, sor_std = self._condition(rgb, goal, nav_type)
        H, W = rgb.shape[-2:]

        # diffusion: noise the expert increments, predict the noise
        sched = self.get_scheduler_for_device(rgb.device)
        B = rgb.size(0)
        _, P, T, _ = action.size()
        timesteps = torch.randint(0, sched.config.num_train_timesteps, (B, P), device=rgb.device).long()
        gt_noise = torch.randn(action.size(), device=rgb.device)

        timesteps_flat = timesteps.view(-1)
        action_flat = action.view(-1, T, 2)
        gt_noise_flat = gt_noise.view(-1, T, 2)
        noisy_action = sched.add_noise(action_flat, gt_noise_flat, timesteps_flat)
        noise_pred = self.diffuser_noise_pred_net(
            sample=noisy_action, timestep=timesteps_flat, global_cond=cond_feat.view(-1, cond_feat.size(-1)))

        # one-step denoised path (x0 estimate) for the path-level losses
        abar = sched.alphas_cumprod[timesteps].to(noisy_action.dtype)
        sqrt_abar = torch.sqrt(abar).view(-1, 1, 1)
        sqrt_one_minus = torch.sqrt(1.0 - abar).view(-1, 1, 1)
        action_pred = (noisy_action - sqrt_one_minus * noise_pred) / (sqrt_abar + 1e-12)
        normalized_path = torch.cumsum(action_pred, dim=1)                       # (B*P,T,2)
        cam_h = camera_height.repeat_interleave(P)[:, None, None]
        path = normalized_path * cam_h                                           # [m]

        # traversability of the denoised path, from image features at its projection
        uv = project_path_fl_to_uv(
            path_fl=normalized_path.detach(),
            K=K.repeat_interleave(P, dim=0).to(rgb.device),
            cam_height=torch.ones((normalized_path.size(0),), device=rgb.device))
        feat_t = sample_feat_map_at_uv(
            feat_map=side_tuned_feat.repeat_interleave(P, dim=0), uv=uv, img_hw=(H, W))
        traversability_logit = self.path_traversability_net(normalized_path.detach(), feat_t.detach())

        # clearance of the denoised path on the BEV signed distance field
        sdf = (bev_sdf * camera_height[:, None, None]).repeat_interleave(P, dim=0)   # [m]
        row = torch.clamp((path[:, :, 0] - self.bev_forward_start) / self.bev_cell_length, min=0, max=bev.size(1) - 1)
        col = torch.clamp((path[:, :, 1] - self.bev_horizontal_start) / self.bev_cell_length, min=0, max=bev.size(2) - 1)
        clearance = bilinear_sample_with_grid_sample(sdf, torch.stack([row, col], dim=-1))   # + free, - inside

        gt_traversability = (clearance > self.robot_size).to(torch.float32)
        b_idx = torch.arange(row.size(0)).unsqueeze(1)
        out_of_fov = bev.repeat_interleave(P, dim=0)[b_idx, row.to(torch.int32), col.to(torch.int32)] == 100
        out_of_image = (uv[..., 0] < 0) | (uv[..., 0] >= W) | (uv[..., 1] <= 0)
        gt_traversability[out_of_fov | out_of_image] = 0.5

        collision = F.softplus(-(clearance - (self.robot_size + self.safety_margin)) / self.bev_cell_length)
        collision_cost = torch.logsumexp(collision * 2, dim=1) / 2

        action_valid = (action_flat != 0).any(dim=-1)
        curv_reg = curvature_reg_varstep(path, delta=1.0)[action_valid[:, :-1]]

        # SoR-path consistency: at least one waypoint should pass near the predicted SoR
        mu = sor_mu.view(-1, 2).unsqueeze(1).expand(-1, T, -1)
        var = (sor_std.view(-1, 2) ** 2).unsqueeze(1).expand(-1, T, -1)
        mahalanobis = ((normalized_path - mu) ** 2 / (var + 1e-6)).sum(dim=-1)
        dist_soft = -self.path_consistency_tau * torch.logsumexp(-mahalanobis / self.path_consistency_tau, dim=1)

        return TrainOutput(
            sor_mu=sor_mu.view(-1, 2), sor_std=sor_std.view(-1, 2),
            gt_noise=gt_noise_flat, noise_pred=noise_pred,
            gt_traversability=gt_traversability, traversability_logit=traversability_logit,
            collision_cost=collision_cost, curv_reg=curv_reg, dist_soft=dist_soft,
        )

    @torch.no_grad()
    def predict(self, rgb, goal, nav_type, camera_height, K, sample_num=30) -> Prediction:
        """Single-image inference (B = 1).

        rgb:           (1, 3, H, W) normalized image already warped to the model intrinsic K
        goal:          (1, 1, 2) (r / camera_height, theta)
        nav_type:      (1, 1) 1 = goal-directed, 0 = exploration
        camera_height: (1,) [m]
        K:             (1, 3, 3) model intrinsic
        """
        side_tuned_feat, cond_feat, sor_mu, sor_std = self._condition(rgb, goal, nav_type)
        H, W = rgb.shape[-2:]

        sched = self.get_scheduler_for_device(rgb.device)
        cond_feat = cond_feat.repeat(1, sample_num, 1).squeeze(0)
        sched.set_timesteps(num_inference_steps=sched.config.num_train_timesteps, device=rgb.device)
        action = torch.randn((sample_num, self.pred_action_step, 2), device=rgb.device)
        for t in sched.timesteps:
            noise_pred = self.diffuser_noise_pred_net(sample=action, timestep=t, global_cond=cond_feat)
            action = sched.step(model_output=noise_pred, timestep=t, sample=action).prev_sample

        normalized_path = torch.cumsum(action, dim=1)  # (S,T,2)
        path = normalized_path * camera_height.repeat_interleave(sample_num, dim=0)[:, None, None]

        uv = project_path_fl_to_uv(
            path_fl=normalized_path,
            K=K.repeat_interleave(sample_num, dim=0).to(rgb.device),
            cam_height=torch.ones((sample_num,), device=rgb.device))
        feat_t = sample_feat_map_at_uv(
            feat_map=side_tuned_feat.repeat_interleave(sample_num, dim=0),
            uv=uv, img_hw=(H, W),
        )  # (S,T,C)
        traversability = torch.sigmoid(self.path_traversability_net(normalized_path, feat_t))

        h = camera_height[:, None, None]
        return Prediction(sor_mu=sor_mu * h, sor_std=sor_std * h, path=path, traversability=traversability)
