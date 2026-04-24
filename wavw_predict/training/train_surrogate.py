"""
Entry point: train the FNO surrogate model.

Usage:
    python -m wavw_predict.training.train_surrogate --config configs/train_surrogate.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..data.datasets import SimulationDataset
from ..data.augmentation import AugmentationConfig
from ..data.loaders import make_train_val_loaders
from ..models.fno import FNO2D
from ..training.losses import fno_total_loss
from ..training.trainer import Trainer
from ..utils.config import load_base_with_override
from ..utils.logging import get_logger

logger = get_logger("train_surrogate")


def _make_loss_fn(cfg, dx: float, dy: float, dt: float):
    """Returns a closure that computes the full FNO loss from a batch."""
    def loss_fn(model, batch):
        inp = batch["input_fields"]
        tgt = batch["target_fields"]
        pred = model(inp)
        lp = cfg.training.fno.lambda_physics
        lg = cfg.training.fno.lambda_grad
        eta_pred = pred[:, 0]
        u_pred = pred[:, 1]
        v_pred = pred[:, 2]
        eta_prev = inp[:, 3]   # most recent eta in history window
        u_prev = torch.zeros_like(eta_prev)
        v_prev = torch.zeros_like(eta_prev)
        bathy = inp[:, 4]
        return fno_total_loss(
            pred, tgt,
            eta_pred=eta_pred, u_pred=u_pred, v_pred=v_pred,
            eta_prev=eta_prev, u_prev=u_prev, v_prev=v_prev,
            bathy=bathy, dx=dx, dy=dy, dt=dt,
            lambda_physics=lp, lambda_grad=lg,
        )
    return loss_fn


def main(config_path: str) -> None:
    cfg = load_base_with_override(config_path)
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    data_root = Path(cfg.paths.data_root)
    aug_cfg = AugmentationConfig()
    train_ds = SimulationDataset(data_root / "processed" / "train", augment=True, aug_config=aug_cfg)
    val_ds = SimulationDataset(data_root / "processed" / "val")

    train_loader, val_loader = make_train_val_loaders(
        train_ds, val_ds, batch_size=cfg.training.fno.batch_size
    )

    model_cfg = cfg.model.fno
    model = FNO2D(
        in_channels=model_cfg.in_channels,
        out_channels=model_cfg.out_channels,
        width=model_cfg.width,
        modes1=model_cfg.modes1,
        modes2=model_cfg.modes2,
        n_layers=model_cfg.n_layers,
    )
    logger.info(f"FNO parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = AdamW(model.parameters(),
                      lr=cfg.training.fno.lr,
                      weight_decay=cfg.training.fno.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.training.fno.epochs)

    loss_fn = _make_loss_fn(cfg, cfg.grid.dx, cfg.grid.dy, cfg.grid.dt)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=Path(cfg.paths.checkpoint_dir) / "fno",
        loss_fn=loss_fn,
        mixed_precision=cfg.training.mixed_precision,
        grad_clip=cfg.training.fno.grad_clip,
    )

    trainer.fit(train_loader, val_loader, epochs=cfg.training.fno.epochs)
    logger.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_surrogate.yaml")
    args = parser.parse_args()
    main(args.config)
