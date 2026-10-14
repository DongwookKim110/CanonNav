"""build_model(NetworkCfg) -> FullModel, and checkpoint loading / export.

Checkpoint formats (both readable by load_checkpoint):
  release   {attr: state}                                    scripts/export_checkpoint.py output
  training  {"format": "canonnav-train", "model": {attr: state}, "ema": {attr: state}, ...}
            written by canonnav.engine.trainer; inference uses the EMA weights
The encoder entry holds only the LoRA adapters (the frozen DINOv3 base comes from network.dino.pth).
"""
from __future__ import annotations

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from ..config import BevCfg, NetworkCfg, SupervisionCfg
from .backbone.dino import DinoBackbone, load_dino_encoder
from .heads import (
    DinoSideHead,
    GoalCondTransformerDecoder,
    PathTraversabilityNet,
    SoR_Gaussian_Head,
)
from .model import FullModel
from .policy.conditional_unet1d import ConditionalUnet1D

TRAIN_CKPT_FORMAT = "canonnav-train"

# FullModel attributes stored in checkpoints
MODULES = ("encoder", "side_branch", "goal_conditioned_attn_layer", "sor_head", "diffuser_noise_pred_net",
           "path_traversability_net")


def build_model(net: NetworkCfg, bev: BevCfg = None, supervision: SupervisionCfg = None) -> FullModel:
    """bev / supervision are only needed for training (FullModel.forward)."""
    bev = bev or BevCfg()
    supervision = supervision or SupervisionCfg()
    raw_encoder, dinov3_C = load_dino_encoder(net.dino)
    encoder = DinoBackbone(raw_encoder, r=net.lora_rank)

    side_branch = DinoSideHead(in_channels=dinov3_C, out_channels=dinov3_C, mid_channels=dinov3_C // 4)

    dec = net.decoder
    goal_conditioned_attn_layer = GoalCondTransformerDecoder(
        C_in=dinov3_C,
        D=dec.out_channel,
        nhead=dec.nhead,
        num_cross_layers=dec.cross_attn_layers,
        mlp_ratio=dec.mlp_ratio,
        r_max=dec.goal_r_max,
    )

    # conditioning vector: [img_context, goal_embed, goal_cond, nav_type]
    cond_dim = 3 * dec.out_channel + 1
    sor_head = SoR_Gaussian_Head(in_dim=cond_dim, hidden_dim=net.sor_head.hidden_dim)

    diff = net.diffusion_policy
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=diff.diffusion_step,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        clip_sample_range=5.0,
        prediction_type='epsilon',
    )
    diffuser_noise_pred_net = ConditionalUnet1D(
        input_dim=2,
        global_cond_dim=cond_dim,
        down_dims=[dec.out_channel // 2, dec.out_channel, 2 * dec.out_channel],
    )

    path_traversability_net = PathTraversabilityNet(
        feat_c=dinov3_C,
        hidden=net.path_traversability_net.hidden,
        ch=net.path_traversability_net.ch,
    )

    return FullModel(
        encoder, side_branch, goal_conditioned_attn_layer,
        sor_head, noise_scheduler, diffuser_noise_pred_net, path_traversability_net,
        pred_action_step=diff.pred_action_step,
        bev_cell_length=bev.cell_length,
        bev_forward_start=bev.forward_range[0],
        bev_horizontal_start=bev.horizontal_range[0],
        robot_size=supervision.robot_size,
        safety_margin=supervision.safety_margin,
        path_consistency_tau=supervision.path_consistency_tau,
    )


def module_states(ckpt: dict) -> dict:
    """{attr: state} of the weights to run with (EMA weights for training checkpoints)."""
    if ckpt.get("format") == TRAIN_CKPT_FORMAT:
        return ckpt["ema"]
    return {attr: ckpt[attr] for attr in MODULES}


def load_states(model: FullModel, states: dict) -> None:
    """Load {attr: state} into the model; the encoder takes only its LoRA adapters."""
    for attr in MODULES:
        if attr == "encoder":
            model.encoder.load_lora(states[attr])
        else:
            getattr(model, attr).load_state_dict(states[attr], strict=True)


def save_states(model: FullModel) -> dict:
    """{attr: state} to store; the encoder contributes only its LoRA adapters."""
    return {attr: model.encoder.lora_state_dict() if attr == "encoder" else getattr(model, attr).state_dict()
            for attr in MODULES}


def load_checkpoint(model: FullModel, path: str) -> None:
    """Load a release or training checkpoint for inference (training ones: EMA weights)."""
    load_states(model, module_states(torch.load(path, map_location="cpu", weights_only=True)))


def export_checkpoint(train_ckpt_path: str, out_path: str) -> None:
    """Training checkpoint -> release checkpoint: EMA weights, LoRA adapters only."""
    torch.save(module_states(torch.load(train_ckpt_path, map_location="cpu", weights_only=True)), out_path)
