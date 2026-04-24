#!/usr/bin/env python3
"""
Preprocess tide gauge and GNSS observations.

Usage:
    python scripts/preprocess_observations.py \
        --gauge-dir data/raw/observations \
        --output-dir data/processed/observations \
        --dt 60
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wavw_predict.utils.io import read_csv_timeseries
from wavw_predict.data.preprocessing import resample_timeseries


def process_event(
    gauge_dir: Path,
    output_dir: Path,
    dt_s: float,
) -> None:
    csv_files = sorted(gauge_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files in {gauge_dir}")
        return

    series_list, loc_list = [], []
    ref_times = None

    for f in csv_files:
        df = read_csv_timeseries(f, time_col="time")
        t_sec = (df.index - df.index[0]).total_seconds().values
        eta_col = [c for c in df.columns if "eta" in c.lower() or "wl" in c.lower()]
        if not eta_col:
            continue
        eta_raw = df[eta_col[0]].values

        resampled, t_res = resample_timeseries(eta_raw, t_sec, dt_s)
        series_list.append(resampled)
        if ref_times is None:
            ref_times = t_res
        # Coordinates from filename convention: gauge_<lat>_<lon>.csv
        parts = f.stem.split("_")
        try:
            lat, lon = float(parts[-2]), float(parts[-1])
        except (IndexError, ValueError):
            lat, lon = 0.0, 0.0
        loc_list.append([lat, lon])

    if not series_list:
        return

    T = min(len(s) for s in series_list)
    gauge_eta = np.stack([s[:T] for s in series_list], axis=1)  # [T, N]
    gauge_locs = np.array(loc_list, dtype=np.float32)
    times = ref_times[:T].astype(np.float32)

    out_path = output_dir / (gauge_dir.name + ".npz")
    np.savez_compressed(out_path, gauge_eta=gauge_eta.astype(np.float32),
                        gauge_locs=gauge_locs, times=times)
    print(f"Saved: {out_path}  shape={gauge_eta.shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gauge-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dt", type=float, default=60.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    process_event(Path(args.gauge_dir), output_dir, args.dt)


if __name__ == "__main__":
    main()
