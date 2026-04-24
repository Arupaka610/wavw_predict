"""
Entry point: train the EmbankmentNet model.

Usage:
    python -m wavw_predict.training.train_embankment --config configs/train_embankment.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from ..data.datasets import EmbankmentDataset
from ..data.loaders import make_train_val_loaders
from ..models.embankment_net import EmbankmentNet
from ..training.losses import embankment_failure_loss
from ..training.trainer import Trainer
from ..utils.config import load_base_with_override
from ..utils.logging import get_logger

logger = get_logger("train_embankment")


def _make_loss_fn(focal_gamma: float, lambda_breach: float):
    def loss_fn(model, batch):
        seq = batch["sequence"]     # [B, T, 8]
        outcome = batch["outcome"]  # [B, 3]: failed, final_width, final_depth
        pred = model(seq)
        return embankment_failure_loss(
            p_pred=pred.p_failure,
            failed=outcome[:, 0],
            width_mean=pred.width_mean,
            width_std=pred.width_std,
            depth_mean=pred.depth_mean,
            depth_std=pred.depth_std,
            final_width=outcome[:, 1],
            final_depth=outcome[:, 2],
            lambda_breach=lambda_breach,
            focal_gamma=focal_gamma,
        )
    return loss_fn


def main(config_path: str) -> None:
    cfg = load_base_with_override(config_path)
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")

    data_root = Path(cfg.paths.data_root)
    all_ds = EmbankmentDataset(data_root / "processed" / "train")
    n = len(all_ds)
    n_val = max(1, int(n * 0.15))
    train_ds, val_ds = torch.utils.data.random_split(all_ds, [n - n_val, n_val])

    cfg_em = cfg.training.embankment_net
    train_loader, val_loader = make_train_val_loaders(train_ds, val_ds, batch_size=cfg_em.batch_size)

    em_cfg = cfg.model.embankment_net
    model = EmbankmentNet(
        in_features=em_cfg.in_features,
        tcn_channels=list(em_cfg.tcn_channels),
        kernel_size=em_cfg.kernel_size,
        dilations=list(em_cfg.dilations),
        dropout=em_cfg.dropout,
    )

    optimizer = AdamW(model.parameters(), lr=cfg_em.lr, weight_decay=cfg_em.weight_decay)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg_em.lr,
        steps_per_epoch=len(train_loader),
        epochs=cfg_em.epochs,
        pct_start=cfg_em.pct_start,
    )
    loss_fn = _make_loss_fn(cfg_em.focal_gamma, cfg_em.lambda_breach)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=Path(cfg.paths.checkpoint_dir) / "embankment",
        loss_fn=loss_fn,
        mixed_precision=False,
        grad_clip=1.0,
    )
    trainer.fit(train_loader, val_loader, epochs=cfg_em.epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_embankment.yaml")
    args = parser.parse_args()
    main(args.config)
