"""DINOv3 backbone with LoRA adapters on every block's qkv.

The base DINOv3 weights are loaded from the official pretrained file and kept
frozen; only the LoRA adapters (w_a / w_b) are learned and stored in CanonNav
checkpoints.

The model calls `backbone.encoder.get_intermediate_layers(...)` directly
(`encoder` is the raw DINOv3 module with LoRA injected).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ...config import DinoCfg
from .lora import LoRA

DINOV3_CHANNEL = {
    "dinov3_vitb16": 768,
    "dinov3_vits16plus": 384,
}


def load_dino_encoder(cfg: DinoCfg):
    """Return the pretrained raw DINOv3 encoder and its channel count."""
    encoder = torch.hub.load(cfg.repo_dir, cfg.model, source="local", weights=cfg.pth)
    return encoder, DINOV3_CHANNEL[cfg.model]


class DinoBackbone(nn.Module):
    """Frozen DINOv3 + LoRA adapters on Q and V of every block."""

    def __init__(self, encoder: nn.Module, r: int = 3):
        super().__init__()
        assert r > 0
        self.encoder = encoder
        self.w_a = nn.ModuleList()
        self.w_b = nn.ModuleList()

        for param in self.encoder.parameters():
            param.requires_grad = False

        for block in self.encoder.blocks:
            w_qkv_linear = block.attn.qkv
            dim = w_qkv_linear.in_features
            w_a_linear_q, w_b_linear_q = self._create_lora_layer(dim, r)
            w_a_linear_v, w_b_linear_v = self._create_lora_layer(dim, r)
            self.w_a.extend([w_a_linear_q, w_a_linear_v])
            self.w_b.extend([w_b_linear_q, w_b_linear_v])
            block.attn.qkv = LoRA(
                w_qkv_linear, w_a_linear_q, w_b_linear_q, w_a_linear_v, w_b_linear_v,
            )
        self._reset_lora_parameters()

    def _create_lora_layer(self, dim: int, r: int):
        w_a = nn.Linear(dim, r, bias=False)
        w_b = nn.Linear(r, dim, bias=False)
        return w_a, w_b

    def _reset_lora_parameters(self) -> None:
        for w_a in self.w_a:
            nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
        for w_b in self.w_b:
            nn.init.zeros_(w_b.weight)

    def lora_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if k.startswith(("w_a.", "w_b."))}

    def load_lora(self, state):
        """Load LoRA adapters. Base encoder keys in `state`, if any, are ignored."""
        lora = {k: v for k, v in state.items() if k.startswith(("w_a.", "w_b."))}
        expected = self.lora_state_dict().keys()
        missing = set(expected) - set(lora)
        if missing:
            raise KeyError(f"missing LoRA keys in checkpoint: {sorted(missing)[:4]} ...")
        self.load_state_dict(lora, strict=False)
