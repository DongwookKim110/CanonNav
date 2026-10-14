"""DataLoaders for training (weighted over environment folders) and validation."""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from ..config import TrainConfig
from .dataset import build_dataset
from .sampler import DistributedWeightedSampler


def _seed_worker(worker_id):
    """Seed numpy / random per worker (PyTorch seeds only torch), so goals and augmentation differ."""
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def _loader_kwargs(cfg: TrainConfig):
    tc = cfg.train
    workers = tc.num_workers
    return dict(num_workers=workers, pin_memory=tc.pin_memory, persistent_workers=workers > 0,
                prefetch_factor=tc.prefetch_factor if workers > 0 else None, worker_init_fn=_seed_worker)


def build_train_loader(cfg: TrainConfig, log=print) -> DataLoader:
    """One epoch draws sum(len(source) * weight) samples, each source with probability ~ its weight."""
    is_dist = dist.is_available() and dist.is_initialized()
    datasets, weights = [], []
    for src in cfg.data.train:
        ds = build_dataset(src.path, cfg, train=True)
        datasets.append(ds)
        weights += [src.weight] * len(ds)
    samples_per_epoch = int(sum(len(ds) * src.weight for ds, src in zip(datasets, cfg.data.train)))

    log(f"{'train source':60s} {'seqs':>5s} {'labels':>7s} {'weight':>7s} {'share':>7s}")
    for ds, src in zip(datasets, cfg.data.train):
        share = len(ds) * src.weight / max(samples_per_epoch, 1)
        log(f"{src.path:60s} {len(ds.datasets):5d} {len(ds):7d} {src.weight:7.2f} {share:7.1%}")
    log(f"samples per epoch: {samples_per_epoch}")

    if is_dist:
        sampler = DistributedWeightedSampler(weights, num_samples=samples_per_epoch,
                                             num_replicas=dist.get_world_size(), rank=dist.get_rank(),
                                             base_seed=cfg.train.seed)
    else:
        sampler = WeightedRandomSampler(weights, num_samples=samples_per_epoch, replacement=True,
                                        generator=torch.Generator().manual_seed(cfg.train.seed))
    return DataLoader(ConcatDataset(datasets), batch_size=cfg.train.batch_size, sampler=sampler,
                      drop_last=True, **_loader_kwargs(cfg))


def build_val_loader(cfg: TrainConfig) -> DataLoader:
    """Validation sequences in a fixed order (deterministic samples, no augmentation)."""
    if not cfg.data.val:
        return None
    dataset = ConcatDataset([build_dataset(p, cfg, train=False) for p in cfg.data.val])
    return DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False, **_loader_kwargs(cfg))
