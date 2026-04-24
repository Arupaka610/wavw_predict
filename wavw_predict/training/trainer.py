"""
Generic Trainer class supporting mixed-precision, grad clipping, and checkpointing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from ..utils.logging import get_logger


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: _LRScheduler | None,
        device: torch.device,
        checkpoint_dir: str | Path,
        loss_fn: Callable,
        mixed_precision: bool = True,
        grad_clip: float | None = 1.0,
        log_interval: int = 20,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.ckpt_dir = Path(checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.loss_fn = loss_fn
        self.mixed_precision = mixed_precision and device.type == "cuda"
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.scaler = GradScaler() if self.mixed_precision else None
        self.logger = get_logger(self.__class__.__name__)
        self.global_step = 0
        self.best_val_loss = float("inf")

    def train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        for step, batch in enumerate(loader):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            self.optimizer.zero_grad()

            if self.mixed_precision:
                with autocast():
                    loss = self.loss_fn(self.model, batch)
                self.scaler.scale(loss).backward()
                if self.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss = self.loss_fn(self.model, batch)
                loss.backward()
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            total_loss += loss.item()
            self.global_step += 1

            if step % self.log_interval == 0:
                self.logger.info(f"Epoch {epoch} step {step}: loss={loss.item():.6f}")

        return total_loss / len(loader)

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        for batch in loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            loss = self.loss_fn(self.model, batch)
            total_loss += loss.item()
        return total_loss / len(loader)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
    ) -> dict:
        history = {"train_loss": [], "val_loss": []}
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader)
            if self.scheduler is not None:
                if hasattr(self.scheduler, "step") and "metrics" in str(
                    self.scheduler.__class__.__name__).lower():
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            self.logger.info(
                f"Epoch {epoch}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, name="best.pt")

            if epoch % 10 == 0:
                self.save_checkpoint(epoch, val_loss, name=f"epoch_{epoch:04d}.pt")

        return history

    def save_checkpoint(self, epoch: int, val_loss: float, name: str = "checkpoint.pt") -> None:
        path = self.ckpt_dir / name
        torch.save({
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "global_step": self.global_step,
        }, path)
        self.logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str | Path) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.global_step = ckpt.get("global_step", 0)
        return ckpt["epoch"]
