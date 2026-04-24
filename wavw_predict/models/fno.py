"""
Fourier Neural Operator (FNO-2D) for wave propagation surrogate.

Architecture:
  Input  [B, 6, H, W] = eta×4 history + bathymetry + coast_mask
  4× FNO blocks (SpectralConv2d + bypass linear + GeLU)
  Output [B, 3, H, W] = [eta(t+1), u(t+1), v(t+1)]

Reference: Li et al. (2021) "Fourier Neural Operator for Parametric PDEs"
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """
    2D spectral convolution: FFT → truncated complex weight multiply → IFFT.

    Weights shape: [in_ch, out_ch, modes1, modes2]  (complex).
    Only the low-frequency modes (top-left and top-right corners) are retained.
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    @staticmethod
    def _compl_mul2d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Batched complex matrix multiply: [B, Cin, H, W] × [Cin, Cout, H, W]."""
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)   # [B, C, H, W//2+1] complex

        m1, m2 = self.modes1, min(self.modes2, x_ft.shape[-1])
        out_ft = torch.zeros(B, self.out_channels, H, x_ft.shape[-1],
                             dtype=torch.cfloat, device=x.device)

        # Low-low corner
        out_ft[:, :, :m1, :m2] = self._compl_mul2d(x_ft[:, :, :m1, :m2], self.weights1)
        # Low-high corner (negative frequencies in H dimension)
        out_ft[:, :, -m1:, :m2] = self._compl_mul2d(x_ft[:, :, -m1:, :m2], self.weights2)

        return torch.fft.irfft2(out_ft, s=(H, W))


class FNOBlock(nn.Module):
    """Single FNO residual block: spectral conv + bypass linear + GeLU."""

    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.spec_conv = SpectralConv2d(width, width, modes1, modes2)
        self.bypass = nn.Conv2d(width, width, kernel_size=1)
        self.norm = nn.InstanceNorm2d(width, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.spec_conv(x) + self.bypass(x)))


class FNO2D(nn.Module):
    """
    Full FNO-2D model for one-step wave field prediction.

    in_channels : 6  (eta×4 + bathy + coast_mask)
    out_channels: 3  (eta, u, v at next timestep)
    width       : hidden feature dimension (default 64)
    modes1/2    : retained Fourier modes (default 32)
    n_layers    : number of FNO blocks (default 4)
    """

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 3,
        width: int = 64,
        modes1: int = 32,
        modes2: int = 32,
        n_layers: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lift = nn.Conv2d(in_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList([FNOBlock(width, modes1, modes2) for _ in range(n_layers)])
        self.proj = nn.Sequential(
            nn.Conv2d(width, width * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width * 2, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_channels, H, W] → [B, out_channels, H, W]."""
        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
        return self.proj(x)

    def rollout(
        self,
        x0: torch.Tensor,          # [B, in_channels, H, W]
        n_steps: int,
        bathymetry: torch.Tensor,  # [B, 1, H, W] or [1, 1, H, W]
        coast_mask: torch.Tensor,  # [B, 1, H, W]
        window_size: int = 4,
    ) -> torch.Tensor:
        """
        Auto-regressive multi-step rollout.
        Returns [B, n_steps, out_channels, H, W].
        """
        B, C, H, W = x0.shape
        # eta history buffer: keep last window_size eta fields
        eta_buf = x0[:, :window_size]   # [B, 4, H, W]
        outputs = []

        for _ in range(n_steps):
            inp = torch.cat([eta_buf, bathymetry, coast_mask], dim=1)
            pred = self(inp)                  # [B, 3, H, W]
            outputs.append(pred)
            # Update eta buffer
            eta_buf = torch.cat([eta_buf[:, 1:], pred[:, :1]], dim=1)

        return torch.stack(outputs, dim=1)   # [B, n_steps, 3, H, W]
