"""Training dataset: label files from scripts/preprocess.py + camera images.

Layout: <env>/<seq>/labels/<stamp>.npz and <env>/<seq>/rgb/<stamp>.jpg (see canonnav/data/preprocess.py
for the label format). Every sample is
  1. a goal (and its SoR target and the expert path up to it) drawn from the label's goal list; with
     probability exploration_prob it becomes an exploration sample (goal masked, path to the last label),
  2. augmented (train only): horizontal flip, yaw rotation, color jitter,
  3. canonicalized: image warped to the model intrinsic with pitch (and yaw augmentation) removed,
     lengths divided by the camera height.
Validation samples are deterministic (goal drawn from a per-sample seed, no augmentation).
"""
from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import List

import albumentations as A
import cv2
import numpy as np
from torch.utils.data import ConcatDataset, Dataset

from ..config import BevCfg, TrainConfig
from ..utils.image import normalize_image

BEV_OUT_OF_FOV = 100


def find_label_sequences(root, rgb_folder: str, label_folder: str) -> List[Path]:
    """Sequence folders (containing both rgb/ and labels/) at or below `root`."""
    root = Path(root)
    seqs = [Path(p) for p, dirnames, _ in os.walk(root) if rgb_folder in dirnames and label_folder in dirnames]
    if not seqs:
        raise FileNotFoundError(f"no sequence with '{rgb_folder}/' and '{label_folder}/' under {root}")
    return sorted(seqs)


def check_bev(meta: dict, bev: BevCfg, where) -> None:
    """The labels' BEV grid must be the one the model is configured with."""
    got = (float(meta["bev_cell_size"]), [float(v) for v in meta["bev_forward_range"]],
           [float(v) for v in meta["bev_horizontal_range"]])
    want = (float(bev.cell_length), [float(v) for v in bev.forward_range], [float(v) for v in bev.horizontal_range])
    if got != want:
        raise ValueError(f"BEV of the labels in {where} (cell, forward, horizontal) = {got} "
                         f"differs from config bev = {want}")


class SequenceDataset(Dataset):
    def __init__(self, seq: Path, cfg: TrainConfig, train: bool):
        self.seq = Path(seq)
        self.train = train
        self.files = sorted((self.seq / cfg.data.label_folder).glob("*.npz"))
        self.rgb_dir = self.seq / cfg.data.rgb_folder
        if self.files:
            check_bev(np.load(self.files[0], allow_pickle=True)["meta"].item(), cfg.bev, self.seq)

        net, tc = cfg.network, cfg.train
        self.K = np.array(net.K, dtype=np.float64)
        self.H, self.W = net.H, net.W
        self.T = net.diffusion_policy.pred_action_step
        self.path_per_image = tc.path_per_image if train else 1
        self.exploration_prob = tc.exploration_prob
        self.rotation_aug = tc.rotation_aug if train else 0.0
        cj = tc.color_jitter
        self.color_jitter = A.ColorJitter(brightness=cj.brightness, contrast=cj.contrast,
                                          saturation=cj.saturation, hue=cj.hue, p=cj.p) if train else None

    def __len__(self):
        return len(self.files)

    def _rng(self, idx):
        """Train: the global (per-worker seeded) generators. Validation: a fixed per-sample seed."""
        if self.train:
            return np.random, random
        return np.random.RandomState(idx), random.Random(idx)

    def __getitem__(self, idx):
        np_rng, py_rng = self._rng(idx)
        d = np.load(self.files[idx], allow_pickle=True)
        goal_sor = d["goal_sor"]                    # (N, 4) r, theta, sor_x, sor_y
        traj = d["traj"][:, :3, 3]                  # ground-aligned camera frame (x right, z forward)
        traj = np.stack([traj[:, 2], -traj[:, 0]], axis=-1)   # (forward, left)
        bev = d["bev"]
        bev_sdf = d["bev_sdf"]
        meta = d["meta"].item()

        # ----- goal / exploration -----
        P, T = self.path_per_image, self.T
        exploration = (goal_sor.shape[0] >= T) & (np_rng.rand(P) < self.exploration_prob)
        goal_index = np.where(exploration, goal_sor.shape[0] - 1, np_rng.randint(0, goal_sor.shape[0], size=P))
        goal_r_th = goal_sor[goal_index, :2]

        paths, sors = [], []
        for i in range(P):
            n = min(traj.shape[0], goal_index[i] + 1, T)    # up to the goal, at most T waypoints
            path = traj[:n]
            sors.append(goal_sor[:n, 2:4][-1])
            if n < T:                                        # pad by repeating the last waypoint
                path = np.concatenate([path, np.repeat(path[-1:], T - n, axis=0)], axis=0)
            paths.append(np.vstack([np.zeros(2), path]))
        paths = np.stack(paths)                             # (P, T+1, 2)
        sors = np.stack(sors)                               # (P, 2)

        # ----- flip -----
        flip = py_rng.random() > 0.5
        if self.train and flip:
            goal_r_th[:, 1] = -goal_r_th[:, 1]
            sors[:, 1] = -sors[:, 1]
            paths[:, :, 1] = -paths[:, :, 1]

        bgr = cv2.imread(str(self.rgb_dir / (meta["stamp"] + ".jpg")), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if self.train and flip:
            rgb = np.ascontiguousarray(rgb[:, ::-1, :])
            bev = np.ascontiguousarray(bev[:, ::-1])
            bev_sdf = np.ascontiguousarray(bev_sdf[:, ::-1])

        # ----- yaw rotation (labels and BEV about the robot) -----
        yaw = self.rotation_aug * (py_rng.random() * 2 - 1) / 180.0 * math.pi   # + counter-clockwise
        rot2 = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        paths = paths @ rot2.T
        sors = sors @ rot2.T
        goal_r_th[:, 1] += yaw
        M = cv2.getRotationMatrix2D((bev.shape[1] // 2, 0), yaw * 180 / math.pi, 1.0)
        bev = cv2.warpAffine(bev, M, (bev.shape[1], bev.shape[0]), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_REPLICATE)
        bev_sdf = cv2.warpAffine(bev_sdf, M, (bev.shape[1], bev.shape[0]), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)

        # ----- canonical image: model intrinsic, no pitch, yaw augmentation applied -----
        pitch = meta["pitch"]
        R_pitch = np.array([[1., 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
        R_yaw = np.array([[math.cos(-yaw), 0, math.sin(-yaw)], [0, 1, 0], [-math.sin(-yaw), 0, math.cos(-yaw)]])
        homography = self.K @ (R_yaw @ R_pitch) @ np.linalg.inv(meta["K"])
        canon = cv2.warpPerspective(rgb, homography, (self.W, self.H), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT)

        image = canon if self.color_jitter is None else self.color_jitter(image=canon)["image"]
        image = normalize_image(image)

        # ----- height normalization -----
        h = float(meta["cam_height"])
        goal_r_th = goal_r_th.copy()
        goal_r_th[:, 0] /= h
        return {
            "rgb": image,                                        # (3, H, W)
            "goal": goal_r_th.astype(np.float32),                # (P, 2) r / h, theta
            "sor": (sors / h).astype(np.float32),                # (P, 2)
            "action": (np.diff(paths, axis=1) / h).astype(np.float32),   # (P, T, 2)
            "nav_type": (1 - exploration.astype(np.int64)).astype(np.float32),   # (P,) 1 goal, 0 exploration
            "bev": bev,                                          # (nF, nH) uint8
            "bev_sdf": (bev_sdf / h).astype(np.float32),         # (nF, nH)
            "cam_height": np.float32(h),
            "vis_rgb": (canon * 255).astype(np.uint8),           # (H, W, 3) canonical image before jitter
        }


def build_dataset(root, cfg: TrainConfig, train: bool) -> ConcatDataset:
    """All sequences at or below `root` (an environment folder, a sequence, or a dataset root)."""
    seqs = find_label_sequences(root, cfg.data.rgb_folder, cfg.data.label_folder)
    return ConcatDataset([SequenceDataset(s, cfg, train) for s in seqs])
