"""
Loss functions for all three models.

  fno_total_loss          — MSE + SWE physics residual + spatial gradient penalty + rollout
  force_prediction_loss   — wave force MSE + coefficient regularization
  embankment_failure_loss — Focal Loss (classification) + Gaussian NLL (regression)
  data_assimilation_loss  — tide gauge observation matching (fine-tuning)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# FNO surrogate losses
# ---------------------------------------------------------------------------

def data_loss_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def gradient_penalty(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Penalizes mismatch in spatial gradients to sharpen wave fronts."""
    def _grad(x: torch.Tensor):
        gx = x[:, :, :, 1:] - x[:, :, :, :-1]
        gy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return gx, gy

    pg_x, pg_y = _grad(pred)
    tg_x, tg_y = _grad(target)
    return F.mse_loss(pg_x, tg_x) + F.mse_loss(pg_y, tg_y)


def swe_residual_loss(
    eta_pred: torch.Tensor,  # [B, H, W]
    u_pred: torch.Tensor,
    v_pred: torch.Tensor,
    eta_prev: torch.Tensor,  # state at previous step
    u_prev: torch.Tensor,
    v_prev: torch.Tensor,
    bathy: torch.Tensor,     # [B, H, W] or [H, W]
    dx: float,
    dy: float,
    dt: float,
    g: float = 9.81,
    h_min: float = 1e-3,
) -> torch.Tensor:
    """
    Computes SWE residual for (eta_pred, u_pred, v_pred) given previous state.
    Mass residual: |∂η/∂t + ∂(hu)/∂x + ∂(hv)/∂y|
    """
    if bathy.ndim == 2:
        bathy = bathy.unsqueeze(0)

    h_prev = (eta_prev + bathy).clamp(min=h_min)
    h_pred = (eta_pred + bathy).clamp(min=h_min)

    # Finite differences for divergence of flux
    qu = h_prev * u_prev
    qv = h_prev * v_prev

    div_u = (qu[:, :, 2:] - qu[:, :, :-2]) / (2.0 * dx)
    div_v = (qv[:, 2:, :] - qv[:-2, :, :]) / (2.0 * dy) if qv.ndim == 3 else (qv[:, 2:, :] - qv[:, :-2, :]) / (2.0 * dy)

    deta_dt = (eta_pred[:, 1:-1, 1:-1] - eta_prev[:, 1:-1, 1:-1]) / dt

    min_hw = min(div_u.shape[-2], div_v.shape[-2], deta_dt.shape[-2])
    min_ww = min(div_u.shape[-1], div_v.shape[-1], deta_dt.shape[-1])
    R_mass = deta_dt[:, :min_hw, :min_ww] + div_u[:, :min_hw, :min_ww] + div_v[:, :min_hw, :min_ww]

    return R_mass.pow(2).mean()


def rollout_loss(
    model: nn.Module,
    x0: torch.Tensor,
    targets: torch.Tensor,     # [B, n_steps, 3, H, W]
    bathymetry: torch.Tensor,
    coast_mask: torch.Tensor,
    gamma: float = 0.9,
    window_size: int = 4,
) -> torch.Tensor:
    """
    Multi-step rollout loss with exponentially decaying weights.
    w_t = gamma^t  to prevent error accumulation.
    """
    n_steps = targets.shape[1]
    preds = model.rollout(x0, n_steps, bathymetry, coast_mask, window_size)   # [B, T, 3, H, W]

    total = torch.tensor(0.0, device=x0.device)
    weight_sum = 0.0
    for t in range(n_steps):
        w = gamma ** t
        total = total + w * F.mse_loss(preds[:, t], targets[:, t])
        weight_sum += w

    return total / weight_sum


def fno_total_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eta_pred: torch.Tensor | None = None,
    u_pred: torch.Tensor | None = None,
    v_pred: torch.Tensor | None = None,
    eta_prev: torch.Tensor | None = None,
    u_prev: torch.Tensor | None = None,
    v_prev: torch.Tensor | None = None,
    bathy: torch.Tensor | None = None,
    dx: float = 100.0,
    dy: float = 100.0,
    dt: float = 2.0,
    lambda_physics: float = 0.1,
    lambda_grad: float = 0.01,
) -> torch.Tensor:
    loss = data_loss_mse(pred, target) + lambda_grad * gradient_penalty(pred, target)
    if all(v is not None for v in [eta_pred, u_pred, v_pred, eta_prev, u_prev, v_prev, bathy]):
        loss = loss + lambda_physics * swe_residual_loss(
            eta_pred, u_pred, v_pred, eta_prev, u_prev, v_prev, bathy, dx, dy, dt
        )
    return loss


# ---------------------------------------------------------------------------
# Wave force losses
# ---------------------------------------------------------------------------

def coefficient_regularization(Cd: torch.Tensor, Cm: torch.Tensor) -> torch.Tensor:
    """Smooth hinge loss to keep Cd/Cm in physical range [0.5,5] / [1,4]."""
    def hinge(x, lo, hi):
        return F.relu(lo - x).pow(2).mean() + F.relu(x - hi).pow(2).mean()
    return hinge(Cd, 0.5, 5.0) + hinge(Cm, 1.0, 4.0)


def force_prediction_loss(
    F_pred: torch.Tensor,
    F_target: torch.Tensor,
    Cd: torch.Tensor,
    Cm: torch.Tensor,
    lambda_reg: float = 0.5,
) -> torch.Tensor:
    mse = F.mse_loss(F_pred, F_target)
    reg = coefficient_regularization(Cd, Cm)
    return mse + lambda_reg * reg


# ---------------------------------------------------------------------------
# Embankment losses
# ---------------------------------------------------------------------------

def focal_loss(
    pred: torch.Tensor,    # [B] probabilities
    target: torch.Tensor,  # [B] binary labels
    gamma: float = 2.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    p = pred.clamp(eps, 1 - eps)
    fl_pos = -target * (1 - p) ** gamma * torch.log(p)
    fl_neg = -(1 - target) * p ** gamma * torch.log(1 - p)
    return (fl_pos + fl_neg).mean()


def gaussian_nll(
    mu: torch.Tensor,
    log_sigma: torch.Tensor,  # or std directly
    target: torch.Tensor,
    use_log_sigma: bool = False,
) -> torch.Tensor:
    if use_log_sigma:
        sigma = log_sigma.exp() + 1e-6
    else:
        sigma = log_sigma + 1e-6
    return (0.5 * ((target - mu) / sigma) ** 2 + sigma.log()).mean()


def embankment_failure_loss(
    p_pred: torch.Tensor,         # [B] failure probability
    failed: torch.Tensor,         # [B] binary ground truth
    width_mean: torch.Tensor,     # [B]
    width_std: torch.Tensor,
    depth_mean: torch.Tensor,
    depth_std: torch.Tensor,
    final_width: torch.Tensor,    # [B] ground truth
    final_depth: torch.Tensor,
    lambda_breach: float = 1.0,
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    cls_loss = focal_loss(p_pred, failed, gamma=focal_gamma)

    # Only compute regression loss where breach actually occurred
    breach_mask = failed > 0.5
    reg_loss = torch.tensor(0.0, device=p_pred.device)
    if breach_mask.any():
        reg_loss = gaussian_nll(width_mean[breach_mask], width_std[breach_mask],
                                final_width[breach_mask]) + \
                   gaussian_nll(depth_mean[breach_mask], depth_std[breach_mask],
                                final_depth[breach_mask])

    return cls_loss + lambda_breach * reg_loss


# ---------------------------------------------------------------------------
# Data assimilation loss (fine-tuning)
# ---------------------------------------------------------------------------

def data_assimilation_loss(
    eta_pred: torch.Tensor,      # [B, T, H, W]
    eta_obs: torch.Tensor,       # [B, T, H, W] — NaN where no gauge
    weight_by_proximity: bool = False,
) -> torch.Tensor:
    """
    MSE at gauge locations only (where eta_obs is not NaN).
    """
    valid = ~torch.isnan(eta_obs)
    if not valid.any():
        return torch.tensor(0.0, device=eta_pred.device)
    diff = (eta_pred[valid] - eta_obs[valid]) ** 2
    return diff.mean()
