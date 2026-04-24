"""
Evaluation metrics for all model components.
"""
from __future__ import annotations

import math

import torch


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float((pred - target).pow(2).mean().sqrt())


def max_eta_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Maximum surface elevation absolute error [m]."""
    return float((pred - target).abs().max())


def nash_sutcliffe_efficiency(pred: torch.Tensor, target: torch.Tensor) -> float:
    """NSE = 1 - SS_res / SS_tot. Perfect = 1.0."""
    ss_res = (pred - target).pow(2).sum()
    ss_tot = (target - target.mean()).pow(2).sum()
    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else float("-inf")
    return float(1.0 - ss_res / ss_tot)


def wave_arrival_time_error(
    pred: torch.Tensor,   # [T] or [T, H, W] eta time series
    target: torch.Tensor,
    dt: float,
    threshold: float = 0.1,  # [m] arrival threshold
) -> float:
    """Difference in wave arrival time [s]."""
    def _first_exceed(series: torch.Tensor) -> int:
        exceed = (series.abs() > threshold).nonzero(as_tuple=False)
        return exceed[0].item() if len(exceed) > 0 else -1

    if pred.ndim == 1:
        t_pred = _first_exceed(pred)
        t_tgt = _first_exceed(target)
    else:
        t_pred = _first_exceed(pred.abs().max(dim=-1)[0].max(dim=-1)[0])
        t_tgt = _first_exceed(target.abs().max(dim=-1)[0].max(dim=-1)[0])

    if t_pred < 0 or t_tgt < 0:
        return float("nan")
    return float(abs(t_pred - t_tgt) * dt)


def force_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return rmse(pred, target)


def force_bias(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float((pred - target).mean())


def normalized_force_error(pred: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> float:
    """NRMSE as percentage of mean absolute target."""
    denom = target.abs().mean().item()
    return 100.0 * rmse(pred, target) / max(denom, eps)


def breach_detection_auroc(p_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """AUROC for binary breach detection."""
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415
    try:
        return float(roc_auc_score(y_true.numpy(), p_pred.numpy()))
    except Exception:
        return float("nan")


def breach_width_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return rmse(pred, target)


def breach_timing_error(pred: torch.Tensor, target: torch.Tensor, dt: float) -> float:
    """Mean absolute error in breach initiation time [s]."""
    return float((pred - target).abs().mean()) * dt


def calibration_error(p_pred: torch.Tensor, y_true: torch.Tensor, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE) for binary classification.
    Lower is better; 0 = perfectly calibrated.
    """
    ece = 0.0
    n = len(p_pred)
    edges = torch.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        lo, hi = edges[i].item(), edges[i + 1].item()
        mask = (p_pred >= lo) & (p_pred < hi)
        if mask.sum() == 0:
            continue
        conf = p_pred[mask].mean().item()
        acc = y_true[mask].float().mean().item()
        ece += (mask.sum().item() / n) * abs(conf - acc)
    return ece


def summarize_metrics(
    eta_pred: torch.Tensor,
    eta_target: torch.Tensor,
    dt: float,
) -> dict[str, float]:
    """Compute all wave-field metrics and return as dict."""
    return {
        "rmse_eta_m": rmse(eta_pred, eta_target),
        "max_eta_error_m": max_eta_error(eta_pred, eta_target),
        "nse": nash_sutcliffe_efficiency(eta_pred, eta_target),
        "arrival_time_error_s": wave_arrival_time_error(eta_pred, eta_target, dt),
    }
