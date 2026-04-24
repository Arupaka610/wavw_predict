"""
Data augmentation for wave field and structural datasets.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class AugmentationConfig:
    flip_prob: float = 0.3
    amplitude_scale_prob: float = 0.5
    amplitude_scale_range: tuple[float, float] = (0.7, 1.3)
    time_jitter_steps: int = 5
    bathy_perturb_prob: float = 0.2
    bathy_perturb_sigma: float = 0.5     # [m] std of bathymetry noise
    gauge_dropout_prob: float = 0.3      # fraction of gauges dropped (fine-tune)
    gauge_dropout_enabled: bool = False


class WaveAugmentation:
    """
    Applies stochastic augmentations to a sample dict containing wave fields.

    Expected sample keys:
        input_fields : Tensor [C_in, H, W]   — channels 0-3 are eta history,
                                                channel 4 is bathymetry,
                                                channel 5 is coast mask
        target_fields: Tensor [C_out, H, W]  — [eta_next, u_next, v_next]
    """

    def __init__(self, cfg: AugmentationConfig):
        self.cfg = cfg

    def __call__(self, sample: dict) -> dict:
        sample = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in sample.items()}

        if random.random() < self.cfg.flip_prob:
            sample = self._horizontal_flip(sample)

        if random.random() < self.cfg.amplitude_scale_prob:
            sample = self._amplitude_scale(sample)

        sample = self._time_jitter(sample)

        if random.random() < self.cfg.bathy_perturb_prob:
            sample = self._bathy_perturb(sample)

        if self.cfg.gauge_dropout_enabled:
            sample = self._gauge_dropout(sample)

        return sample

    def _horizontal_flip(self, sample: dict) -> dict:
        sample["input_fields"] = sample["input_fields"].flip(-1)
        sample["target_fields"] = sample["target_fields"].flip(-1)
        # Flip u-velocity sign (x-component)
        if sample["target_fields"].shape[0] >= 2:
            sample["target_fields"][1] = -sample["target_fields"][1]
        return sample

    def _amplitude_scale(self, sample: dict) -> dict:
        lo, hi = self.cfg.amplitude_scale_range
        alpha = random.uniform(lo, hi)
        # Scale eta channels (indices 0-3) and target eta
        sample["input_fields"][:4] *= alpha
        sample["target_fields"][0] *= alpha   # eta
        sample["target_fields"][1] *= alpha   # u (linear in amplitude for shallow water)
        sample["target_fields"][2] *= alpha   # v
        # If force targets present, scale by alpha^2 (Morison ~ u^2)
        if "force_targets" in sample:
            sample["force_targets"] *= alpha ** 2
        return sample

    def _time_jitter(self, sample: dict) -> dict:
        # Metadata shift only — actual window selection is in Dataset.__getitem__
        jitter = random.randint(-self.cfg.time_jitter_steps, self.cfg.time_jitter_steps)
        sample["_time_jitter"] = jitter
        return sample

    def _bathy_perturb(self, sample: dict) -> dict:
        bathy = sample["input_fields"][4]   # channel 4 = bathymetry
        H, W = bathy.shape
        noise = self._spatially_correlated_noise(H, W, sigma=self.cfg.bathy_perturb_sigma)
        sample["input_fields"][4] = bathy + noise
        return sample

    @staticmethod
    def _spatially_correlated_noise(H: int, W: int, sigma: float, length_scale: int = 5) -> torch.Tensor:
        """2D noise smoothed with a Gaussian kernel."""
        import torch.nn.functional as F  # noqa: PLC0415
        raw = torch.randn(1, 1, H, W) * sigma
        kernel_size = length_scale * 2 + 1
        x = torch.arange(kernel_size, dtype=torch.float32) - length_scale
        g1d = torch.exp(-0.5 * (x / length_scale) ** 2)
        g1d /= g1d.sum()
        kernel = g1d.unsqueeze(0) * g1d.unsqueeze(1)
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        noise = F.conv2d(raw, kernel, padding=length_scale)
        return noise.squeeze()

    def _gauge_dropout(self, sample: dict) -> dict:
        if "gauge_mask" not in sample:
            return sample
        mask = sample["gauge_mask"].clone()
        n = mask.sum().int().item()
        n_drop = int(n * self.cfg.gauge_dropout_prob)
        indices = mask.nonzero(as_tuple=False)
        if len(indices) > 0 and n_drop > 0:
            drop_idx = torch.randperm(len(indices))[:n_drop]
            for i in drop_idx:
                r, c = indices[i]
                mask[r, c] = False
        sample["gauge_mask"] = mask
        return sample
