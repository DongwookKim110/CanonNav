"""Network modules: goal-conditioned decoder, DINO side branch, SoR head, path traversability net."""
from typing import Optional

import torch
import torch.nn as nn


# ---------- 2D Sin-Cos positional encoding ----------
def build_2d_sincos_pos_embed(H, W, D, device, dtype=torch.float32):
    assert D % 4 == 0, "D must be divisible by 4 for 2D sincos"
    d = D // 4
    y = torch.arange(H, device=device, dtype=dtype)
    x = torch.arange(W, device=device, dtype=dtype)
    wy = 1.0 / (10000 ** (torch.arange(d, device=device, dtype=dtype) / d))
    wx = 1.0 / (10000 ** (torch.arange(d, device=device, dtype=dtype) / d))
    Y = torch.einsum('h,d->hd', y, wy)  # (H,d)
    X = torch.einsum('w,d->wd', x, wx)  # (W,d)
    pos_y = torch.cat([Y.sin(), Y.cos()], dim=-1)            # (H,2d)
    pos_x = torch.cat([X.sin(), X.cos()], dim=-1)            # (W,2d)
    pos = torch.cat([pos_y[:, None, :].expand(H, W, -1), pos_x[None, :, :].expand(H, W, -1)], dim=-1)
    return pos  # (H,W,D)


class SoR_Gaussian_Head(nn.Module):
    """Predicts the SoR as an axis-aligned 2D Gaussian (mu, std) in (forward, left)."""

    def __init__(self, in_dim, hidden_dim=256, std_min=0.02, std_max=0.7):
        super().__init__()
        self.std_min = std_min
        self.std_max = std_max

        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mu_head = nn.Linear(hidden_dim, 2)
        self.std_head = nn.Linear(hidden_dim, 2)

        nn.init.constant_(self.std_head.bias, 0.0)

    def forward(self, x):
        h = self.backbone(x)

        mu = self.mu_head(h)
        raw_std = self.std_head(h)

        std = self.std_min + (self.std_max - self.std_min) * torch.sigmoid(raw_std)

        return mu, std


# ---------- Transformer blocks ----------
class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class CrossAttnBlock(nn.Module):
    """ Query: goal tokens; Key/Value: image tokens """
    def __init__(self, dim, nhead, mlp_ratio=4.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, nhead, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)

    def forward(self, q_tokens, kv_tokens, kv_padding_mask: Optional[torch.Tensor] = None):
        # q_tokens: (B,S,D), kv_tokens: (B,N,D)
        y, _ = self.attn(self.norm_q(q_tokens),
                         self.norm_kv(kv_tokens), self.norm_kv(kv_tokens),
                         key_padding_mask=kv_padding_mask, need_weights=False)
        q = q_tokens + y
        q = q + self.mlp(self.norm2(q))
        return q  # (B,S,D)


# ---------- Goal embedding ----------
class GoalEmbed(nn.Module):
    """(r, theta) goal -> D-dim embedding via [r / r_max, sin(theta), cos(theta)]."""

    def __init__(self, r_max=20.0, out_dim=256):
        super().__init__()
        self.r_max = r_max
        self.mlp = nn.Sequential(
            nn.Linear(3, out_dim), nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, goal):  # (B,N,2) or (B,2)
        if goal.dim() == 2:
            goal = goal.unsqueeze(1)   # (B,1,2)

        B, N, _ = goal.shape
        goal = goal.view(B * N, 2)
        r, theta = goal[:, 0], goal[:, 1]
        r_n = (r / self.r_max).clamp(0, 1)

        g = torch.stack([r_n, torch.sin(theta), torch.cos(theta)], dim=-1)  # (B*N,3)
        g = self.mlp(g)  # (B*N, D)
        return g.view(B, N, -1)  # (B, N, D)


# ---------- Goal-Conditioned Transformer Decoder ----------
class GoalCondTransformerDecoder(nn.Module):
    """
    Image tokens (feat + 2D pos) attended by the goal token through cross-attention.

    inputs : feat (B, C_in, H, W), goal (B, N, 2) as (r, theta)
    outputs: img_context (B, D)     mean-pooled image tokens
             goal_cond   (B, N, D)  goal token after cross-attention
             goal_embed  (B, N, D)  goal token before cross-attention
    """
    def __init__(self,
                 C_in: int,
                 D: int = 256,
                 nhead: int = 8,
                 num_cross_layers: int = 2,
                 mlp_ratio: float = 4.0,
                 r_max: float = 20.0):
        super().__init__()
        self.D = D

        self.proj = nn.Conv2d(C_in, D, kernel_size=1, bias=False)
        self.norm_tok = nn.LayerNorm(D)

        self.goal_embed = GoalEmbed(r_max=r_max, out_dim=D)
        self.cross_blocks = nn.ModuleList([
            CrossAttnBlock(D, nhead, mlp_ratio)
            for _ in range(num_cross_layers)
        ])

    def forward(self, feat: torch.Tensor, goal: torch.Tensor):
        B, C, H, W = feat.shape
        device, dtype = feat.device, feat.dtype

        # tokenize + 2D pos
        x = self.proj(feat)                             # (B, D, H, W)
        pos = build_2d_sincos_pos_embed(H, W, self.D, device, dtype)  # (H, W, D)
        pos = pos.unsqueeze(0).expand(B, -1, -1, -1)    # (B, H, W, D)

        x = x.permute(0, 2, 3, 1).contiguous()          # (B, H, W, D)
        x = x.view(B, H * W, self.D)                    # (B, HW, D)
        x = x + pos.view(B, H * W, self.D)
        x = self.norm_tok(x)

        # cross-attn (Q=goal, KV=image)
        g_embed = self.goal_embed(goal)                 # (B, N, D)
        g = g_embed
        for blk in self.cross_blocks:
            g = blk(g, x)                               # (B, N, D)

        img_context = x.mean(dim=1)
        return img_context, g, g_embed


class DinoSideHead(nn.Module):
    """Fuses 4 intermediate DINO layers into one feature map (same resolution)."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = 256):
        super().__init__()
        self.proj0 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.proj1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.proj2 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.proj3 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)

        fused_channels = mid_channels * 4

        self.conv_block = nn.Sequential(
            nn.Conv2d(fused_channels, fused_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=fused_channels),
            nn.GELU(),
            nn.Conv2d(fused_channels, fused_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=fused_channels),
            nn.GELU(),
        )

        self.head = nn.Conv2d(fused_channels, out_channels, kernel_size=1)

    def forward(self, feats):
        # feats: length-4 tuple/list, each [B,C,H,W]
        f0, f1, f2, f3 = feats
        x = torch.cat([self.proj0(f0), self.proj1(f1), self.proj2(f2), self.proj3(f3)], dim=1)
        x = self.conv_block(x)                 # [B, 4*mid, H, W]
        return self.head(x)                    # [B, out_channels, H, W]


class PathTraversabilityNet(nn.Module):
    """
    Inputs:
      - path_fl: (B,T,2) (forward,left)
      - feat_t : (B,T,C) image features sampled at the projected path points
    Output:
      - logits: (B,T), per-waypoint traversability (sigmoid -> [0, 1])
    """
    def __init__(self, feat_c=768, hidden=256, ch=256, blocks=3, dropout=0.1):
        super().__init__()
        self.step_mlp = nn.Sequential(
            nn.Linear(2 + feat_c, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, ch),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(ch, ch, 3, padding=1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(ch, ch, 3, padding=1),
            )
            for _ in range(blocks)
        ])
        self.out = nn.Conv1d(ch, 1, 1)

    def forward(self, path_fl, feat_t):
        x = torch.cat([path_fl, feat_t], dim=-1)            # (B,T,2+C)
        h = self.step_mlp(x)                                # (B,T,ch)
        h = h.transpose(1, 2).contiguous()                  # (B,ch,T)
        for blk in self.blocks:
            h = h + blk(h)
        return self.out(h).squeeze(1)                       # (B,T)
