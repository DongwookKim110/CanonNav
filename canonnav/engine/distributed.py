"""DDP setup: torchrun environment -> process group and device."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.distributed as dist


@dataclass
class DistInfo:
    distributed: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    is_main: bool


def setup_distributed(gpu_ids: Optional[List[int]]) -> DistInfo:
    """Initialize DDP under torchrun, otherwise describe a single process.

    gpu_ids: train.gpu_ids. DDP maps local_rank -> gpu_ids[local_rank]; a single process uses gpu_ids[0].
    """
    is_dist = ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ)
    if is_dist:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        dev_id = gpu_ids[local_rank] if gpu_ids else local_rank
        device = torch.device(f"cuda:{dev_id}")
        torch.cuda.set_device(device)
        return DistInfo(True, rank, world_size, local_rank, device, rank == 0)

    # single process
    if torch.cuda.is_available():
        dev_id = gpu_ids[0] if gpu_ids else 0
        device = torch.device(f"cuda:{dev_id}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    return DistInfo(False, 0, 1, 0, device, True)


def cleanup_distributed(info: DistInfo) -> None:
    if info.distributed and dist.is_initialized():
        dist.destroy_process_group()


def barrier(info: DistInfo) -> None:
    if info.distributed:
        dist.barrier()
