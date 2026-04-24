"""
Preprocessing utilities: normalization, spatial interpolation, observation alignment.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class FieldNormalizer:
    """
    Per-channel z-score or min-max normalizer for wave field tensors.

    Fits statistics from training data and serializes to .npz for inference.
    """

    def __init__(self, strategy: str = "z-score", eps: float = 1e-8):
        assert strategy in {"z-score", "min-max"}
        self.strategy = strategy
        self.eps = eps
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._min: np.ndarray | None = None
        self._max: np.ndarray | None = None

    def fit(self, samples: list[torch.Tensor]) -> "FieldNormalizer":
        """
        samples: list of tensors [C, H, W] or [C].
        Computes per-channel statistics across all samples.
        """
        stacked = torch.stack(samples, dim=0)   # [N, C, ...]
        flat = stacked.reshape(stacked.shape[0], stacked.shape[1], -1)  # [N, C, H*W]
        vals = flat.permute(1, 0, 2).reshape(stacked.shape[1], -1).numpy()  # [C, N*H*W]
        self._mean = vals.mean(axis=1)
        self._std = vals.std(axis=1)
        self._min = vals.min(axis=1)
        self._max = vals.max(axis=1)
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """x: [C, ...] or [B, C, ...]."""
        ndim = x.ndim
        mean, std = self._get_params(x.shape[-len(x.shape) + (1 if ndim > 2 else 0)])
        m = torch.tensor(mean, dtype=x.dtype, device=x.device)
        if self.strategy == "z-score":
            s = torch.tensor(std + self.eps, dtype=x.dtype, device=x.device)
        else:
            lo = torch.tensor(self._min, dtype=x.dtype, device=x.device)
            hi = torch.tensor(self._max + self.eps, dtype=x.dtype, device=x.device)
            return self._reshape_for_broadcast(x, (hi - lo), ndim).clamp(0, 1)

        return (x - self._reshape_for_broadcast(x, m, ndim)) / self._reshape_for_broadcast(x, s, ndim)

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        m = torch.tensor(self._mean, dtype=x.dtype, device=x.device)
        if self.strategy == "z-score":
            s = torch.tensor(self._std + self.eps, dtype=x.dtype, device=x.device)
            return x * self._reshape_for_broadcast(x, s, x.ndim) + self._reshape_for_broadcast(x, m, x.ndim)
        lo = torch.tensor(self._min, dtype=x.dtype, device=x.device)
        hi = torch.tensor(self._max + self.eps, dtype=x.dtype, device=x.device)
        return x * self._reshape_for_broadcast(x, hi - lo, x.ndim) + self._reshape_for_broadcast(x, lo, x.ndim)

    def _get_params(self, _):
        return self._mean, self._std

    @staticmethod
    def _reshape_for_broadcast(x: torch.Tensor, param: torch.Tensor, ndim: int) -> torch.Tensor:
        C = param.shape[0]
        shape = [1] * ndim
        # Find channel dimension (assumed dim=0 for [C,...] or dim=1 for [B,C,...])
        c_dim = 0 if x.shape[0] == C else 1
        shape[c_dim] = C
        return param.view(shape)

    def save(self, path: str | Path) -> None:
        np.savez(
            path,
            mean=self._mean,
            std=self._std,
            min=self._min,
            max=self._max,
            strategy=np.array([self.strategy]),
        )

    def load(self, path: str | Path) -> "FieldNormalizer":
        data = np.load(path, allow_pickle=True)
        self._mean = data["mean"]
        self._std = data["std"]
        self._min = data["min"]
        self._max = data["max"]
        self.strategy = str(data["strategy"][0])
        return self


def bilinear_regrid(
    field: np.ndarray,       # [H_src, W_src]
    src_x: np.ndarray,       # [W_src]
    src_y: np.ndarray,       # [H_src]
    tgt_x: np.ndarray,       # [W_tgt]
    tgt_y: np.ndarray,       # [H_tgt]
) -> np.ndarray:
    """Regrid 2D field from source grid to target grid via bilinear interpolation."""
    from scipy.interpolate import RegularGridInterpolator  # noqa: PLC0415
    interp = RegularGridInterpolator(
        (src_y, src_x), field, method="linear", bounds_error=False, fill_value=0.0
    )
    yy, xx = np.meshgrid(tgt_y, tgt_x, indexing="ij")
    pts = np.stack([yy.ravel(), xx.ravel()], axis=-1)
    return interp(pts).reshape(len(tgt_y), len(tgt_x))


def bathymetry_to_grid(
    depth: np.ndarray,
    src_x: np.ndarray,
    src_y: np.ndarray,
    target_shape: tuple[int, int],
) -> torch.Tensor:
    """Resample bathymetry to (ny, nx) grid."""
    ny, nx = target_shape
    tgt_y = np.linspace(src_y.min(), src_y.max(), ny)
    tgt_x = np.linspace(src_x.min(), src_x.max(), nx)
    regridded = bilinear_regrid(depth, src_x, src_y, tgt_x, tgt_y)
    return torch.tensor(regridded, dtype=torch.float32)


def resample_timeseries(
    series: np.ndarray,
    src_times: np.ndarray,
    tgt_dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a 1D time series to uniform dt via linear interpolation."""
    t_start, t_end = src_times[0], src_times[-1]
    tgt_times = np.arange(t_start, t_end + tgt_dt * 0.5, tgt_dt)
    resampled = np.interp(tgt_times, src_times, series)
    return resampled, tgt_times


def align_observations_to_grid(
    eta_obs: np.ndarray,      # [T, N_gauges]
    gauge_locs: np.ndarray,   # [N_gauges, 2]  (row, col) in grid coords
    grid_shape: tuple[int, int],
) -> torch.Tensor:
    """
    Scatter point observations onto grid as a sparse observation field.
    Returns [T, H, W] with NaN where no gauge exists.
    """
    T = eta_obs.shape[0]
    H, W = grid_shape
    field = torch.full((T, H, W), float("nan"))
    for g, (r, c) in enumerate(gauge_locs.astype(int)):
        if 0 <= r < H and 0 <= c < W:
            field[:, r, c] = torch.tensor(eta_obs[:, g], dtype=torch.float32)
    return field
