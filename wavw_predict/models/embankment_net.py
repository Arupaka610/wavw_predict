"""
Embankment failure prediction network.

Architecture: Temporal Convolutional Network (TCN) with dilated causal convolutions.
Output: failure probability + breach geometry as Gaussian distributions.

Input sequence [B, T, 8]:
  0  Q_overflow           [m³/s/m]
  1  H_upstream           [m]
  2  overtopping_head      = H_upstream - H_crest  [m]
  3  dQ_dt                [m³/s²/m]
  4  cumulative_overflow  [m³/m]
  5  tau_bed              [Pa]
  6  clay_fraction        [0-1]
  7  D50_norm             = D50 / 0.01  (normalized to 10mm reference)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EmbankmentPrediction:
    p_failure: torch.Tensor           # [B] breach probability ∈ [0,1]
    width_mean: torch.Tensor          # [B] predicted final breach width [m]
    width_std: torch.Tensor           # [B] uncertainty (aleatoric)
    depth_mean: torch.Tensor          # [B] predicted final breach depth [m]
    depth_std: torch.Tensor           # [B]


class CausalConv1dBlock(nn.Module):
    """
    Dilated causal convolution block with weight normalization.
    Uses left-padding to maintain causality without looking at future steps.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int,
                 dropout: float = 0.1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=0)
        )
        self.norm = nn.LayerNorm(out_ch)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T]."""
        # Causal padding: pad only on the left
        x_pad = F.pad(x, (self.pad, 0))
        out = self.conv(x_pad)
        out = self.drop(F.gelu(self.norm(out.permute(0, 2, 1)).permute(0, 2, 1)))
        return out + self.skip(x)


class TCNEncoder(nn.Module):
    """Stack of dilated causal conv blocks."""

    def __init__(self, in_ch: int, channels: list[int], kernel_size: int,
                 dilations: list[int], dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = in_ch
        for ch, dil in zip(channels, dilations):
            layers.append(CausalConv1dBlock(prev, ch, kernel_size, dil, dropout))
            prev = ch
        self.net = nn.Sequential(*layers)
        self.out_channels = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, C_in] → [B, C_out, T]."""
        return self.net(x.permute(0, 2, 1))


class EmbankmentNet(nn.Module):
    """
    Full embankment failure model.

    in_features : 8
    tcn_channels: [64, 128, 128]
    kernel_size : 3
    dilations   : [1, 2, 4]  → receptive field = (3-1)*(1+2+4) = 14 steps
    """

    def __init__(
        self,
        in_features: int = 8,
        tcn_channels: list[int] | None = None,
        kernel_size: int = 3,
        dilations: list[int] | None = None,
        dropout: float = 0.1,
        hidden: int = 64,
    ):
        super().__init__()
        tcn_channels = tcn_channels or [64, 128, 128]
        dilations = dilations or [1, 2, 4]

        self.encoder = TCNEncoder(in_features, tcn_channels, kernel_size, dilations, dropout)
        enc_dim = self.encoder.out_channels

        # Global context
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.context_proj = nn.Sequential(
            nn.Linear(enc_dim, hidden),
            nn.GELU(),
        )

        # Failure probability head
        self.cls_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

        # Breach geometry head (outputs mean+std for width and depth)
        self.reg_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 4),   # [width_mu, width_log_sigma, depth_mu, depth_log_sigma]
        )

    def forward(self, x: torch.Tensor) -> EmbankmentPrediction:
        """x: [B, T, 8]."""
        enc = self.encoder(x)              # [B, C, T]
        ctx = self.pool(enc).squeeze(-1)   # [B, C]
        ctx = self.context_proj(ctx)       # [B, hidden]

        p_failure = torch.sigmoid(self.cls_head(ctx)).squeeze(-1)   # [B]

        reg_out = self.reg_head(ctx)       # [B, 4]
        width_mean = F.softplus(reg_out[:, 0])
        width_std = F.softplus(reg_out[:, 1]) + 1e-4
        depth_mean = F.softplus(reg_out[:, 2])
        depth_std = F.softplus(reg_out[:, 3]) + 1e-4

        return EmbankmentPrediction(
            p_failure=p_failure,
            width_mean=width_mean,
            width_std=width_std,
            depth_mean=depth_mean,
            depth_std=depth_std,
        )
