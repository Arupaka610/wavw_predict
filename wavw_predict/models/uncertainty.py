"""
Uncertainty quantification wrappers.

  MCDropoutWrapper  — enables Dropout at inference for N forward passes.
  DeepEnsemble      — manages N independently trained models.

Both return UncertaintyResult with mean, std, epistemic_std, aleatoric_std.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn


@dataclass
class UncertaintyResult:
    mean: torch.Tensor
    std: torch.Tensor
    epistemic_std: torch.Tensor   # model uncertainty (from MC/ensemble spread)
    aleatoric_std: torch.Tensor   # data noise (from predicted variance heads)


def _enable_dropout(model: nn.Module) -> None:
    """Set all Dropout layers to training mode during inference."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


class MCDropoutWrapper(nn.Module):
    """
    Wraps any model that has Dropout layers.
    At inference: enables dropout and runs n_samples forward passes.
    """

    def __init__(self, model: nn.Module, n_samples: int = 50):
        super().__init__()
        self.model = model
        self.n_samples = n_samples

    def forward(self, x: torch.Tensor) -> Any:
        return self.model(x)

    @torch.no_grad()
    def predict_with_uncertainty(self, x: torch.Tensor) -> UncertaintyResult:
        self.model.eval()
        _enable_dropout(self.model)

        samples = []
        for _ in range(self.n_samples):
            out = self.model(x)
            if isinstance(out, torch.Tensor):
                samples.append(out)
            else:
                # Assume dataclass with tensor fields; stack first tensor field
                samples.append(torch.stack([getattr(out, f) for f in out.__dataclass_fields__], dim=-1))

        stacked = torch.stack(samples, dim=0)   # [N, B, ...]
        mean = stacked.mean(0)
        epistemic_std = stacked.std(0)

        return UncertaintyResult(
            mean=mean,
            std=epistemic_std,
            epistemic_std=epistemic_std,
            aleatoric_std=torch.zeros_like(epistemic_std),
        )


class DeepEnsemble:
    """
    Manages an ensemble of N independently trained models.
    Requires models to already be trained and loaded.
    """

    def __init__(self, models: list[nn.Module]):
        self.models = models

    def predict(self, x: torch.Tensor, extract_fn: Callable | None = None) -> UncertaintyResult:
        """
        extract_fn: optional fn(model_output) -> Tensor, for non-Tensor outputs.
        """
        results = []
        with torch.no_grad():
            for m in self.models:
                m.eval()
                out = m(x)
                if extract_fn is not None:
                    out = extract_fn(out)
                if not isinstance(out, torch.Tensor):
                    out = torch.stack([getattr(out, f) for f in out.__dataclass_fields__], dim=-1)
                results.append(out)

        stacked = torch.stack(results, dim=0)   # [N, B, ...]
        mean = stacked.mean(0)
        epistemic_std = stacked.std(0)

        return UncertaintyResult(
            mean=mean,
            std=epistemic_std,
            epistemic_std=epistemic_std,
            aleatoric_std=torch.zeros_like(epistemic_std),
        )


def combine_uncertainties(epistemic: torch.Tensor, aleatoric: torch.Tensor) -> torch.Tensor:
    """Total uncertainty = sqrt(epistemic² + aleatoric²)."""
    return (epistemic ** 2 + aleatoric ** 2).sqrt()
