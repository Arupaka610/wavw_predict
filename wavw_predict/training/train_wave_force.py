"""
Entry point: train the WaveForceNet model.

Usage:
    python -m wavw_predict.training.train_wave_force --config configs/train_wave_force.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..data.datasets import WaveForceDataset
from ..data.loaders import make_train_val_loaders
from ..models.wave_force_net import WaveForceNet
from ..training.losses import force_prediction_loss
from ..training.trainer import Trainer
from ..utils.config import load_base_with_override
from ..utils.logging import get_logger

logger = get_logger("train_wave_force")


def _make_loss_fn(lambda_reg: float):
    def loss_fn(model, batch):
        feats = batch["features"]
        tgt = batch["targets"]            # [B, 4]: F_drag, F_inertia, F_total, F_behind
        coeffs = model(feats)
        # Use F_total (index 2) as primary supervision target for now
        F_pred = coeffs.Cd.unsqueeze(-1) * tgt[:, 0:1] + coeffs.Cm.unsqueeze(-1) * tgt[:, 1:2]
        F_total_pred = F_pred.sum(-1)
        return force_prediction_loss(F_total_pred, tgt[:, 2], coeffs.Cd, coeffs.Cm, lambda_reg)
    return loss_fn


def main(config_path: str) -> None:
    cfg = load_base_with_override(config_path)
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")

    data_root = Path(cfg.paths.data_root)
    all_ds = WaveForceDataset(data_root / "processed" / "train")
    n = len(all_ds)
    n_val = max(1, int(n * 0.1))
    train_ds, val_ds = torch.utils.data.random_split(all_ds, [n - n_val, n_val])

    cfg_wf = cfg.training.wave_force_net
    train_loader, val_loader = make_train_val_loaders(train_ds, val_ds, batch_size=cfg_wf.batch_size)

    model = WaveForceNet(in_features=14, hidden=cfg.model.wave_force_net.hidden)
    optimizer = Adam(model.parameters(), lr=cfg_wf.lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    loss_fn = _make_loss_fn(cfg_wf.lambda_physics)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=Path(cfg.paths.checkpoint_dir) / "wave_force",
        loss_fn=loss_fn,
        mixed_precision=False,
        grad_clip=None,
    )
    trainer.fit(train_loader, val_loader, epochs=cfg_wf.epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_wave_force.yaml")
    args = parser.parse_args()
    main(args.config)
