"""Training losses.

  diffusion        imitation: MSE between predicted and true diffusion noise
  sor_nll          local progress: Gaussian NLL + smooth L1 of the SoR against its target (goal samples only)
  sor_neg          local progress: SoR probability mass on blocked BEV cells (goal samples only)
  sor_consistency  local progress: soft-min distance of the denoised path to the SoR (goal samples only)
  traversability   safety: BCE of per-waypoint traversability against BEV clearance labels
  collision        safety: soft-max collision cost of the denoised path
  curvature        regularizer: curvature of the denoised path
All terms are returned already weighted; "total" is their sum.
"""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BevCfg, LossWeightCfg
from .models.model import TrainOutput


def gaussian_2d_pdf(sor_mu, sor_std, bev, horizontal_range, forward_range, cell_length):
    """Per-cell probability mass of the axis-aligned SoR Gaussian on the BEV grid. -> (B, nF, nH)"""
    nF, nH = bev.size(1), bev.size(2)
    horizontal_centers = torch.linspace(horizontal_range[0] + cell_length / 2, horizontal_range[1] - cell_length / 2,
                                        nH, device=sor_mu.device)
    forward_centers = torch.linspace(forward_range[0] + cell_length / 2, forward_range[1] - cell_length / 2,
                                     nF, device=sor_mu.device)
    forward, horizontal = torch.meshgrid(forward_centers, horizontal_centers, indexing='ij')
    forward, horizontal = forward.unsqueeze(0), horizontal.unsqueeze(0)   # (1, nF, nH)

    mu_f, mu_h = sor_mu[:, 0].view(-1, 1, 1), sor_mu[:, 1].view(-1, 1, 1)
    sigma_f, sigma_h = sor_std[:, 0].view(-1, 1, 1), sor_std[:, 1].view(-1, 1, 1)
    df2 = ((forward - mu_f) ** 2) / (sigma_f ** 2 + 1e-6)
    dh2 = ((horizontal - mu_h) ** 2) / (sigma_h ** 2 + 1e-6)
    norm_const = 1.0 / (2.0 * math.pi * sigma_f * sigma_h + 1e-6)
    return norm_const * torch.exp(-0.5 * (df2 + dh2)) * (cell_length ** 2)


class CanonNavCriterion(nn.Module):
    def __init__(self, weights: LossWeightCfg, bev: BevCfg):
        super().__init__()
        self.w = weights
        self.bev = bev

    def _local_progress(self, out, goal_mask, gt_sor, cam_height, bev, P):
        """SoR losses on goal-directed samples: NLL + smooth L1, path consistency, mass on blocked cells."""
        sor_var = out.sor_std ** 2 + 1e-6
        nll = 0.5 * ((out.sor_mu - gt_sor) ** 2 / sor_var).sum(dim=-1) + torch.log(out.sor_std + 1e-6).sum(dim=-1)
        sor_nll = nll[goal_mask].mean() + F.smooth_l1_loss(out.sor_mu[goal_mask], gt_sor[goal_mask])

        sor_consistency = out.dist_soft[goal_mask].mean()

        cam_h = cam_height.repeat_interleave(P)[goal_mask, None]
        bev_goal = bev.repeat_interleave(P, dim=0)[goal_mask]
        sor_mass = gaussian_2d_pdf(out.sor_mu[goal_mask] * cam_h, out.sor_std[goal_mask] * cam_h, bev_goal,
                                   horizontal_range=self.bev.horizontal_range, forward_range=self.bev.forward_range,
                                   cell_length=self.bev.cell_length)
        sor_neg = sor_mass[bev_goal == 0].sum() / bev_goal.size(0)
        return sor_nll, sor_consistency, sor_neg

    def forward(self, out: TrainOutput, *, nav_type, gt_sor, cam_height, bev) -> Dict[str, torch.Tensor]:
        """nav_type (B, P), gt_sor (B, P, 2) normalized, cam_height (B,), bev (B, nF, nH)."""
        w = self.w
        P = nav_type.size(1)
        goal_mask = nav_type.reshape(-1) == 1

        diffusion = F.mse_loss(out.noise_pred, out.gt_noise, reduction="mean")
        if goal_mask.any():
            sor_nll, sor_consistency, sor_neg = self._local_progress(
                out, goal_mask, gt_sor.reshape(-1, 2), cam_height, bev, P)
        else:   # exploration-only batch: no local-progress terms
            sor_nll = sor_consistency = sor_neg = out.sor_mu.sum() * 0.0
        traversability = F.binary_cross_entropy_with_logits(out.traversability_logit, out.gt_traversability)
        collision = out.collision_cost.mean(dim=-1)
        curvature = out.curv_reg.mean()

        losses = {
            "sor_nll": w.sor_nll * sor_nll,
            "sor_neg": w.sor_neg * sor_neg,
            "sor_consistency": w.sor_path_consistency * sor_consistency,
            "diffusion": w.diffusion * diffusion,
            "traversability": w.traversability * traversability,
            "curvature": w.curvature * curvature,
            "collision": w.collision * collision,
        }
        losses["total"] = sum(losses.values())
        return losses
