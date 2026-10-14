"""Weighted sampler for DDP.

Every rank draws the same global weighted multinomial sample (same seed) and keeps
idx[rank::world_size], so ranks do not overlap and the per-dataset weights are preserved.
The trainer must call set_epoch(epoch) every epoch.
"""
import torch
from torch.utils.data import Sampler


class DistributedWeightedSampler(Sampler):
    def __init__(self, weights, num_samples, num_replicas, rank, replacement=True, base_seed=0):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.total_num_samples = int(num_samples)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.replacement = replacement
        self.base_seed = int(base_seed)
        self.epoch = 0
        self.num_samples = self.total_num_samples // self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.base_seed + self.epoch)
        all_indices = torch.multinomial(self.weights, self.total_num_samples, self.replacement, generator=g)
        upper = self.num_replicas * self.num_samples
        return iter(all_indices[self.rank:upper:self.num_replicas].tolist())

    def __len__(self):
        return self.num_samples
