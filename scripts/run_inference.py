#!/usr/bin/env python3
"""
Full hybrid pipeline inference entry point.

Usage:
    python scripts/run_inference.py \
        --config configs/base.yaml \
        --ic-file data/processed/ic_example.npz \
        --fno-ckpt checkpoints/fno/best.pt \
        --wf-ckpt checkpoints/wave_force/best.pt \
        --emb-ckpt checkpoints/embankment/best.pt \
        --n-steps 1800 \
        --output results/run_001.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from wavw_predict.hybrid.pipeline import (
    HybridPipeline, PipelineConfig, InitialConditions,
    StructureSpec, EmbankmentSpec,
)
from wavw_predict.physics.wave_force import StructureGeometry
from wavw_predict.physics.embankment import EmbankmentProperties
from wavw_predict.utils.config import load_base_with_override
from wavw_predict.utils.logging import get_logger

logger = get_logger("run_inference")


def load_initial_conditions(ic_file: Path) -> InitialConditions:
    data = np.load(ic_file)
    return InitialConditions(
        eta=torch.tensor(data["eta"], dtype=torch.float32),
        u=torch.tensor(data.get("u", np.zeros_like(data["eta"])), dtype=torch.float32),
        v=torch.tensor(data.get("v", np.zeros_like(data["eta"])), dtype=torch.float32),
        bathymetry=torch.tensor(data["bathy"], dtype=torch.float32),
        coast_mask=torch.tensor(data["mask"], dtype=torch.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--ic-file", required=True)
    parser.add_argument("--fno-ckpt", default=None)
    parser.add_argument("--wf-ckpt", default=None)
    parser.add_argument("--emb-ckpt", default=None)
    parser.add_argument("--n-steps", type=int, default=1800)
    parser.add_argument("--n-forecast", type=int, default=0)
    parser.add_argument("--output", default="results/output.npz")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg_raw = load_base_with_override(args.config)
    pipe_cfg = PipelineConfig(
        dx=cfg_raw.grid.dx,
        dy=cfg_raw.grid.dy,
        dt=cfg_raw.grid.dt,
        g=cfg_raw.physics.g,
        sponge_cells=cfg_raw.physics.sponge_cells,
        fno_enabled=args.fno_ckpt is not None,
        device=args.device,
    )

    pipeline = HybridPipeline.from_checkpoints(
        pipe_cfg,
        fno_ckpt=args.fno_ckpt,
        wf_ckpt=args.wf_ckpt,
        emb_ckpt=args.emb_ckpt,
    )

    ic = load_initial_conditions(Path(args.ic_file))
    logger.info(f"Grid: {ic.eta.shape}  Steps: {args.n_steps}")

    # Example: add a default structure for demonstration if no external config
    structures = [
        StructureSpec(
            name="building_A",
            grid_row=ic.eta.shape[0] // 2,
            grid_col=ic.eta.shape[1] // 2,
            geometry=StructureGeometry(width=10.0, height=8.0, length=15.0),
        )
    ]

    result = pipeline.run(
        ic=ic,
        n_steps=args.n_steps,
        n_forecast=args.n_forecast,
        structures=structures,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        "eta": result.eta.numpy(),
        "u": result.u.numpy(),
        "v": result.v.numpy(),
    }
    for name, F in result.building_forces.items():
        save_dict[f"force_{name}"] = F.numpy()
    for name, F in result.behind_forces.items():
        save_dict[f"force_behind_{name}"] = F.numpy()

    np.savez_compressed(args.output, **save_dict)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
