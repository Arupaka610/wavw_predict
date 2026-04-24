#!/usr/bin/env python3
"""
Convert raw COMCOT/GeoClaw NetCDF output to processed .npz files.

Usage:
    python scripts/preprocess_simulations.py \
        --input-dir data/raw/simulations \
        --bathy-dir data/raw/bathymetry \
        --output-dir data/processed/train \
        --grid-shape 256 256
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wavw_predict.utils.io import read_netcdf, read_bathymetry
from wavw_predict.data.preprocessing import bathymetry_to_grid


def process_scenario(
    nc_path: Path,
    bathy_path: Path,
    output_dir: Path,
    grid_shape: tuple[int, int],
) -> None:
    data = read_netcdf(nc_path)
    depth, lon, lat = read_bathymetry(bathy_path)
    bathy_grid = bathymetry_to_grid(depth, lon, lat, grid_shape).numpy()

    ny, nx = grid_shape
    coast_mask = (bathy_grid <= 0).astype(np.float32)

    # Expect variables: eta [T, ny, nx], u, v
    eta = data.get("eta", data.get("zeta", np.zeros((1, ny, nx), dtype=np.float32)))
    u = data.get("u", np.zeros_like(eta))
    v = data.get("v", np.zeros_like(eta))

    out_path = output_dir / (nc_path.stem + ".npz")
    np.savez_compressed(
        out_path,
        eta=eta.astype(np.float32),
        u=u.astype(np.float32),
        v=v.astype(np.float32),
        bathy=bathy_grid.astype(np.float32),
        mask=coast_mask,
    )
    print(f"Saved: {out_path}  shape={eta.shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--bathy-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--grid-shape", nargs=2, type=int, default=[256, 256])
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    bathy_dir = Path(args.bathy_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_shape = tuple(args.grid_shape)

    nc_files = sorted(input_dir.glob("*.nc"))
    if not nc_files:
        print("No .nc files found.")
        return

    bathy_files = sorted(bathy_dir.glob("*.nc"))
    bathy_path = bathy_files[0] if bathy_files else bathy_dir / "bathy.nc"

    for nc_path in nc_files:
        process_scenario(nc_path, bathy_path, output_dir, grid_shape)

    print(f"Done. Processed {len(nc_files)} scenarios → {output_dir}")


if __name__ == "__main__":
    main()
