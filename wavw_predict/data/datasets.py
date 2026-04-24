"""
PyTorch Dataset classes for simulation and observation data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentation import AugmentationConfig, WaveAugmentation
from .preprocessing import FieldNormalizer


# ---------------------------------------------------------------------------
# SimulationDataset
# ---------------------------------------------------------------------------

class SimulationDataset(Dataset):
    """
    Loads wave field time series from preprocessed .npz files (one per scenario).

    Each .npz contains:
        eta  : [T, H, W]  float32
        u    : [T, H, W]
        v    : [T, H, W]
        bathy: [H, W]    (static)
        mask : [H, W]    coast mask (static)

    Each sample yields:
        input_fields : [6, H, W]   — eta×4 history + bathy + coast_mask
        target_fields: [3, H, W]   — [eta(t+1), u(t+1), v(t+1)]
        metadata     : dict
    """

    def __init__(
        self,
        data_dir: str | Path,
        window_size: int = 4,
        stride: int = 1,
        normalizer: FieldNormalizer | None = None,
        augment: bool = False,
        aug_config: AugmentationConfig | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.stride = stride
        self.normalizer = normalizer
        self.augment = augment
        self.aug = WaveAugmentation(aug_config or AugmentationConfig()) if augment else None

        self._scenarios: list[dict] = []
        self._index: list[tuple[int, int]] = []   # (scenario_idx, t_start)
        self._load_index()

    def _load_index(self) -> None:
        files = sorted(self.data_dir.glob("*.npz"))
        for sc_idx, f in enumerate(files):
            data = np.load(f, mmap_mode="r")
            T = data["eta"].shape[0]
            # Cache static fields in memory
            self._scenarios.append({
                "path": f,
                "T": T,
                "bathy": torch.tensor(data["bathy"], dtype=torch.float32),
                "mask": torch.tensor(data["mask"], dtype=torch.float32),
            })
            for t in range(0, T - self.window_size - 1, self.stride):
                self._index.append((sc_idx, t))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        sc_idx, t_start = self._index[idx]
        sc = self._scenarios[sc_idx]
        data = np.load(sc["path"], mmap_mode="r")

        t_end = t_start + self.window_size
        eta_hist = torch.tensor(data["eta"][t_start:t_end], dtype=torch.float32)   # [W, H, W]
        eta_next = torch.tensor(data["eta"][t_end], dtype=torch.float32)
        u_next = torch.tensor(data["u"][t_end], dtype=torch.float32)
        v_next = torch.tensor(data["v"][t_end], dtype=torch.float32)

        bathy = sc["bathy"]
        mask = sc["mask"]

        input_fields = torch.cat([eta_hist, bathy.unsqueeze(0), mask.unsqueeze(0)], dim=0)
        target_fields = torch.stack([eta_next, u_next, v_next], dim=0)

        if self.normalizer is not None:
            input_fields = self.normalizer.transform(input_fields)
            target_fields = self.normalizer.transform(target_fields)

        sample = {
            "input_fields": input_fields,
            "target_fields": target_fields,
            "metadata": {
                "scenario": sc["path"].stem,
                "t_start": t_start,
            },
        }

        if self.augment and self.aug is not None:
            sample = self.aug(sample)

        return sample


# ---------------------------------------------------------------------------
# ObservationDataset
# ---------------------------------------------------------------------------

class ObservationDataset(Dataset):
    """
    Loads tide gauge + GNSS observations for data assimilation fine-tuning.

    Each .npz contains:
        gauge_eta  : [T, N_gauges]   tide gauge water surface [m]
        gauge_locs : [N_gauges, 2]   (lat, lon)
        times      : [T]             seconds from event origin
        gnss_disp  : [T, N_gnss, 3] (optional) E/N/U displacement [m]
        gnss_locs  : [N_gnss, 2]    (optional)

    Each sample: sliding window of length window_steps.
    """

    def __init__(
        self,
        data_dir: str | Path,
        window_steps: int = 360,    # e.g. 6 h × 60 s/step = 360 steps
        stride: int = 60,
        dt_s: float = 60.0,
    ):
        self.data_dir = Path(data_dir)
        self.window_steps = window_steps
        self.stride = stride
        self.dt_s = dt_s
        self._events: list[dict] = []
        self._index: list[tuple[int, int]] = []
        self._load_index()

    def _load_index(self) -> None:
        for f in sorted(self.data_dir.glob("*.npz")):
            data = np.load(f, allow_pickle=True)
            T = data["gauge_eta"].shape[0]
            self._events.append({
                "path": f,
                "T": T,
                "gauge_locs": torch.tensor(data["gauge_locs"], dtype=torch.float32),
                "gnss_locs": torch.tensor(data["gnss_locs"], dtype=torch.float32)
                    if "gnss_locs" in data else None,
            })
            for t in range(0, T - self.window_steps, self.stride):
                self._index.append((len(self._events) - 1, t))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        ev_idx, t0 = self._index[idx]
        ev = self._events[ev_idx]
        data = np.load(ev["path"], mmap_mode="r", allow_pickle=True)
        t1 = t0 + self.window_steps

        gauge_series = torch.tensor(data["gauge_eta"][t0:t1], dtype=torch.float32)   # [T, G]
        times = torch.tensor(data["times"][t0:t1], dtype=torch.float32)

        result = {
            "gauge_series": gauge_series,
            "gauge_locs": ev["gauge_locs"],
            "times": times,
        }

        if "gnss_disp" in data and ev["gnss_locs"] is not None:
            result["gnss_series"] = torch.tensor(data["gnss_disp"][t0:t1], dtype=torch.float32)
            result["gnss_locs"] = ev["gnss_locs"]

        return result


# ---------------------------------------------------------------------------
# WaveForceDataset
# ---------------------------------------------------------------------------

class WaveForceDataset(Dataset):
    """
    Structured dataset for wave force training (laboratory / CFD data).

    Each .npz contains:
        features : [N, 14]   — flow/geometry feature vectors
        targets  : [N, 4]    — [F_drag, F_inertia, F_total, F_behind]  [N/m]
    """

    def __init__(self, data_dir: str | Path, normalize: bool = True):
        self.data_dir = Path(data_dir)
        features_list, targets_list = [], []
        for f in sorted(self.data_dir.glob("*.npz")):
            data = np.load(f)
            features_list.append(data["features"])
            targets_list.append(data["targets"])

        self.features = torch.tensor(np.concatenate(features_list, axis=0), dtype=torch.float32)
        self.targets = torch.tensor(np.concatenate(targets_list, axis=0), dtype=torch.float32)

        if normalize:
            self._feat_mean = self.features.mean(0, keepdim=True)
            self._feat_std = self.features.std(0, keepdim=True) + 1e-8
            self.features = (self.features - self._feat_mean) / self._feat_std

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict:
        return {"features": self.features[idx], "targets": self.targets[idx]}


# ---------------------------------------------------------------------------
# EmbankmentDataset
# ---------------------------------------------------------------------------

class EmbankmentDataset(Dataset):
    """
    Time-series dataset for embankment failure training.

    Each .npz contains:
        sequences  : [N, T, 8]   — overflow feature time series
        outcomes   : [N, 3]      — [failed(0/1), final_width, final_depth]
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        seqs_list, out_list = [], []
        for f in sorted(self.data_dir.glob("*.npz")):
            data = np.load(f)
            seqs_list.append(data["sequences"])
            out_list.append(data["outcomes"])

        self.sequences = torch.tensor(np.concatenate(seqs_list, axis=0), dtype=torch.float32)
        self.outcomes = torch.tensor(np.concatenate(out_list, axis=0), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        return {"sequence": self.sequences[idx], "outcome": self.outcomes[idx]}
