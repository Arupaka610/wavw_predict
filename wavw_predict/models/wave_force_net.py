"""
Physics-augmented network for wave force coefficient prediction.

Learns Cd, Cm (Morison coefficients) and K_t (transmission factor)
as functions of dimensionless flow parameters and structure geometry.
Final force is computed by the physics module using the learned coefficients.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ForceCoefficients:
    Cd: torch.Tensor         # drag coefficient  [B] or scalar
    Cm: torch.Tensor         # inertia coefficient
    Kt: torch.Tensor         # transmission coefficient [0, 1]
    dF_behind: torch.Tensor  # additive force correction behind structure [N/m]


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.net(x))


class WaveForceNet(nn.Module):
    """
    Input features [B, 14]:
      0  Fr          = u / sqrt(g*h)              Froude number
      1  log_Re      = log10(u * L / nu)          log-Reynolds number
      2  KC          = u * T / D                  Keulegan-Carpenter
      3  h_over_H    = h / H_bldg                 relative submergence
      4  eta_over_H  = eta / H_bldg               freeboard ratio
      5  du_dt       = ∂u/∂t                      flow acceleration
      6  T_wave      = wave period                 [s]
      7  H_incident  = incident wave height        [m]
      8-11 structure one-hot (wall/rect/cylinder/breakwater)
      12 Rc_over_Hi  = R_c / H_i                  relative freeboard
      13 B_over_Hi   = B / H_i                    relative crest width

    Output: Cd, Cm, Kt, dF_behind
    """

    # Physical bounds for output clamping
    Cd_min, Cd_max = 0.5, 5.0
    Cm_min, Cm_max = 1.0, 4.0

    def __init__(self, in_features: int = 14, hidden: int = 128, n_res_blocks: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden, dropout) for _ in range(n_res_blocks)])
        # Separate output heads
        self.head_Cd = nn.Linear(hidden, 1)
        self.head_Cm = nn.Linear(hidden, 1)
        self.head_Kt = nn.Linear(hidden, 1)
        self.head_dF = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> ForceCoefficients:
        """x: [B, 14] → ForceCoefficients."""
        h = self.input_proj(x)
        for block in self.res_blocks:
            h = block(h)

        # Soft clamping via Softplus then linear rescale
        Cd_raw = F.softplus(self.head_Cd(h)).squeeze(-1)
        Cm_raw = F.softplus(self.head_Cm(h)).squeeze(-1)
        # Map Softplus output to physical range with a smooth hinge
        Cd = self.Cd_min + (self.Cd_max - self.Cd_min) * torch.sigmoid(Cd_raw - 1.0)
        Cm = self.Cm_min + (self.Cm_max - self.Cm_min) * torch.sigmoid(Cm_raw - 1.0)
        Kt = torch.sigmoid(self.head_Kt(h)).squeeze(-1)
        dF = F.softplus(self.head_dF(h)).squeeze(-1)

        return ForceCoefficients(Cd=Cd, Cm=Cm, Kt=Kt, dF_behind=dF)


def build_feature_vector(
    Fr: float | torch.Tensor,
    log_Re: float | torch.Tensor,
    KC: float | torch.Tensor,
    h_over_H: float | torch.Tensor,
    eta_over_H: float | torch.Tensor,
    du_dt: float | torch.Tensor,
    T_wave: float | torch.Tensor,
    H_incident: float | torch.Tensor,
    structure_type: str,
    Rc_over_Hi: float | torch.Tensor,
    B_over_Hi: float | torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Assemble the 14-dimensional feature vector expected by WaveForceNet.
    """
    type_map = {"wall": 0, "rectangular": 1, "cylindrical": 2, "breakwater": 3}
    one_hot = [0.0] * 4
    one_hot[type_map.get(structure_type, 1)] = 1.0

    def _t(v):
        return torch.tensor(v, dtype=torch.float32, device=device) if not isinstance(v, torch.Tensor) else v.float()

    feats = torch.stack([
        _t(Fr), _t(log_Re), _t(KC),
        _t(h_over_H), _t(eta_over_H), _t(du_dt),
        _t(T_wave), _t(H_incident),
        *[torch.tensor(v, dtype=torch.float32, device=device) for v in one_hot],
        _t(Rc_over_Hi), _t(B_over_Hi),
    ])
    return feats   # [14]
