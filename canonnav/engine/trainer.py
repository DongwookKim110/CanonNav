"""Training loop: DDP, EMA, checkpoints (LoRA adapters only for the encoder), validation."""
from __future__ import annotations

import os
import random

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from ..config import TrainConfig
from ..criterion import CanonNavCriterion
from ..data.loader import build_train_loader, build_val_loader
from ..inference import select_best_path
from ..models.build import TRAIN_CKPT_FORMAT, build_model, load_states, save_states
from ..models.ema import EMAModel
from ..utils.visualize import visualize_prediction
from .distributed import barrier, setup_distributed

LOSS_KEYS = ["total", "sor_nll", "sor_neg", "sor_consistency", "diffusion", "traversability", "curvature",
             "collision"]


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        tc = cfg.train
        self.dist = setup_distributed(tc.gpu_ids)
        self.device = self.dist.device
        self.is_main = self.dist.is_main
        self._seed(tc.seed + self.dist.rank)

        self.out_dir = tc.output_dir
        self.ckpt_dir = os.path.join(self.out_dir, "checkpoints")
        self.val_dir = os.path.join(self.out_dir, "val")
        if self.is_main:
            os.makedirs(self.ckpt_dir, exist_ok=True)
            OmegaConf.save(OmegaConf.structured(cfg), os.path.join(self.out_dir, "config.yaml"))
            self.log_path = os.path.join(self.out_dir, "train_log.txt")

        # ----- model, EMA, optimizer -----
        model = build_model(cfg.network, cfg.bev, cfg.supervision).to(self.device)
        self.model = DDP(model, device_ids=[self.device.index], output_device=self.device.index) \
            if self.dist.distributed else model
        self.net = model                                   # unwrapped
        self.ema = EMAModel(model, device=self.device)
        lr = tc.lr
        self.optimizer = torch.optim.AdamW([
            {"params": list(model.encoder.w_a.parameters()) + list(model.encoder.w_b.parameters()), "lr": lr.lora},
            {"params": model.side_branch.parameters(), "lr": lr.side_branch},
            {"params": model.goal_conditioned_attn_layer.parameters(), "lr": lr.goal_conditioned_attn_layer},
            {"params": model.diffuser_noise_pred_net.parameters(), "lr": lr.diffuser_noise_pred_net},
            {"params": model.sor_head.parameters(), "lr": lr.sor_head},
            {"params": model.path_traversability_net.parameters(), "lr": lr.path_traversability_net},
        ])
        self.criterion = CanonNavCriterion(tc.loss_weight, cfg.bev)
        self.K = torch.tensor(cfg.network.K, dtype=torch.float64)

        self.start_epoch = 0
        if tc.resume:
            self._load(tc.resume)

        # dataloaders after DDP init so the sampler sees the process group
        self.train_loader = build_train_loader(cfg, log=self.log)
        self.val_loader = build_val_loader(cfg)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def log(self, msg):
        if not self.is_main:
            return
        print(msg)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def _save(self, epoch):
        torch.save({
            "format": TRAIN_CKPT_FORMAT,
            "epoch": epoch,
            "model": save_states(self.net),
            "ema": save_states(self.ema.averaged_model),
            "ema_step": self.ema.optimization_step,
            "optimizer": self.optimizer.state_dict(),
        }, os.path.join(self.ckpt_dir, f"{epoch:03d}.pt"))

    def _load(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if ckpt.get("format") != TRAIN_CKPT_FORMAT:
            raise ValueError(f"{path} is not a training checkpoint written by this trainer")
        load_states(self.net, ckpt["model"])
        load_states(self.ema.averaged_model, ckpt["ema"])
        self.ema.optimization_step = ckpt["ema_step"]
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.start_epoch = ckpt["epoch"] + 1
        self.log(f"resumed from {path} (next epoch {self.start_epoch})")

    def _to_device(self, batch):
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items() if k != "vis_rgb"}

    def _losses(self, model, b):
        out = model(rgb=b["rgb"], goal=b["goal"], bev=b["bev"], bev_sdf=b["bev_sdf"], action=b["action"],
                    nav_type=b["nav_type"], camera_height=b["cam_height"],
                    K=self.K.expand(b["rgb"].size(0), 3, 3))
        return self.criterion(out, nav_type=b["nav_type"], gt_sor=b["sor"], cam_height=b["cam_height"], bev=b["bev"])

    # ------------------------------------------------------------------ #
    def train(self):
        tc = self.cfg.train
        for ep in range(self.start_epoch, tc.epoch):
            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(ep)
            self.model.train()
            sums = {k: 0.0 for k in LOSS_KEYS}
            for batch in tqdm(self.train_loader, disable=not self.is_main, desc=f"epoch {ep}"):
                losses = self._losses(self.model, self._to_device(batch))
                self.optimizer.zero_grad()
                losses["total"].backward()
                self.optimizer.step()
                self.ema.step(self.net)
                for k in LOSS_KEYS:
                    sums[k] += losses[k].item()

            n = max(len(self.train_loader), 1)
            self.log(f"epoch {ep} train " + " ".join(f"{k} {sums[k] / n:.4f}" for k in LOSS_KEYS)
                     + f" ema_decay {self.ema.decay:.4f}")
            if self.is_main:
                self._save(ep)
                if self.val_loader is not None and (ep % tc.val_every == 0 or ep == tc.epoch - 1):
                    self.validate(ep)
            barrier(self.dist)

    @torch.no_grad()
    def validate(self, ep):
        """EMA model: mean losses on the validation set, and images of the first val_vis_frames samples."""
        net = self.ema.averaged_model
        net.eval()
        rng_state = torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(self.cfg.train.seed)          # same diffusion noise every validation

        sums, n = {k: 0.0 for k in LOSS_KEYS}, 0
        for batch in self.val_loader:
            losses = self._losses(net, self._to_device(batch))
            bs = batch["rgb"].size(0)
            for k in LOSS_KEYS:
                sums[k] += losses[k].item() * bs
            n += bs
        self.log(f"epoch {ep} val   " + " ".join(f"{k} {sums[k] / n:.4f}" for k in LOSS_KEYS))

        out_dir = os.path.join(self.val_dir, f"{ep:03d}")
        os.makedirs(out_dir, exist_ok=True)
        dataset = self.val_loader.dataset
        K = self.K[None].to(self.device)
        for i in range(min(self.cfg.train.val_vis_frames, len(dataset))):
            s = dataset[i]
            h = float(s["cam_height"])
            nav = torch.tensor(s["nav_type"][:1], device=self.device)[None]
            pred = net.predict(rgb=torch.from_numpy(s["rgb"])[None].to(self.device),
                               goal=torch.from_numpy(s["goal"][:1])[None].to(self.device), nav_type=nav,
                               camera_height=torch.tensor([h], device=self.device), K=K, sample_num=10)
            goal_directed = nav.item() == 1
            goal_rt = (float(s["goal"][0, 0]) * h, float(s["goal"][0, 1]))
            paths = pred.path.cpu().numpy()
            trav = np.clip(pred.traversability.cpu().numpy(), 1e-6, 1.0)
            vis = visualize_prediction(rgb=s["vis_rgb"], K=self.K.numpy(), camera_height=h,
                                       sor_mu=pred.sor_mu.reshape(2).cpu().numpy(),
                                       sor_std=pred.sor_std.reshape(2).cpu().numpy(), paths=paths,
                                       best_index=select_best_path(paths, trav, goal_rt, goal_directed),
                                       goal_rt=goal_rt if goal_directed else None)
            cv2.imwrite(os.path.join(out_dir, f"{i:04d}.jpg"), vis)

        torch.set_rng_state(rng_state[0])
        if rng_state[1] is not None:
            torch.cuda.set_rng_state_all(rng_state[1])
        net.eval()
