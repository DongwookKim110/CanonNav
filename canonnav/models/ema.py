"""Exponential moving average of a whole model (shadow copy), as in NoMaD / diffusion_policy.

Decay warmup (@crowsonkb): decay = 1 - (1 + step / inv_gamma) ** -power, clipped to [min_value, max_value].
Frozen parameters and BatchNorm parameters/buffers are copied instead of averaged.
"""
import copy

import torch
from torch.nn.modules.batchnorm import _BatchNorm


class EMAModel:
    def __init__(self, model, device=None, update_after_step: int = 0, inv_gamma: float = 1.0,
                 power: float = 2 / 3, min_value: float = 0.0, max_value: float = 0.9999):
        self.averaged_model = copy.deepcopy(model)
        if device is not None:
            self.averaged_model.to(device)
        self.averaged_model.eval()
        self.averaged_model.requires_grad_(False)

        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value

        self.decay = 0.0
        self.optimization_step = 0

    def get_decay(self, optimization_step: int) -> float:
        step = max(0, optimization_step - self.update_after_step - 1)
        if step <= 0:
            return 0.0
        value = 1 - (1 + step / self.inv_gamma) ** -self.power
        return max(self.min_value, min(value, self.max_value))

    @staticmethod
    @torch.no_grad()
    def _sync_batchnorm_buffers(src: _BatchNorm, dst: _BatchNorm):
        if hasattr(src, "running_mean") and src.running_mean is not None:
            dst.running_mean.copy_(src.running_mean)
        if hasattr(src, "running_var") and src.running_var is not None:
            dst.running_var.copy_(src.running_var)
        if hasattr(src, "num_batches_tracked") and src.num_batches_tracked is not None:
            dst.num_batches_tracked.copy_(src.num_batches_tracked)

    @torch.no_grad()
    def step(self, new_model):
        """Update the EMA weights from `new_model`. Call after optimizer.step()."""
        self.decay = self.get_decay(self.optimization_step)

        # new_model and averaged_model have identical module order
        for module, ema_module in zip(new_model.modules(), self.averaged_model.modules()):
            if isinstance(module, _BatchNorm) and isinstance(ema_module, _BatchNorm):
                self._sync_batchnorm_buffers(module, ema_module)

            for param, ema_param in zip(module.parameters(recurse=False), ema_module.parameters(recurse=False)):
                src = param.data.to(dtype=ema_param.dtype)
                if isinstance(module, _BatchNorm) or not param.requires_grad:
                    ema_param.copy_(src)
                else:
                    ema_param.mul_(self.decay).add_(src, alpha=1.0 - self.decay)

        self.optimization_step += 1
