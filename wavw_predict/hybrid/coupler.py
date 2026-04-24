"""
Physics-ML Coupler: converts raw physics state tensors into ML feature vectors.

All dimensionless numbers are computed consistently and clipped to valid ranges
before being passed to neural network inference.
"""
from __future__ import annotations

import math

import torch

from ..physics.wave_force import StructureGeometry
from ..physics.embankment import EmbankmentProperties


_NU_WATER = 1.0e-6    # kinematic viscosity [m²/s]
_G = 9.81


class PhysicsMLCoupler:
    """Extracts ML-ready feature vectors from SWE solver outputs."""

    def __init__(self, g: float = 9.81, nu: float = _NU_WATER):
        self.g = g
        self.nu = nu

    # ------------------------------------------------------------------
    # FNO input
    # ------------------------------------------------------------------

    def extract_fno_input(
        self,
        eta_history: torch.Tensor,   # [4, H, W]  last 4 eta timesteps
        bathymetry: torch.Tensor,    # [H, W]
        coast_mask: torch.Tensor,    # [H, W]  1=land, 0=ocean
    ) -> torch.Tensor:               # [6, H, W]
        return torch.cat(
            [eta_history, bathymetry.unsqueeze(0), coast_mask.unsqueeze(0)], dim=0
        )

    # ------------------------------------------------------------------
    # Wave force features
    # ------------------------------------------------------------------

    def extract_wave_force_features(
        self,
        h: torch.Tensor,              # total water depth at structure [scalar or T]
        u: torch.Tensor,              # depth-averaged velocity
        du_dt: torch.Tensor,          # flow acceleration
        geom: StructureGeometry,
        T_wave: float,
        H_incident: float,
        freeboard: float,             # R_c = crest_elev - water_level
    ) -> torch.Tensor:               # [14]
        """
        Compute dimensionless feature vector for WaveForceNet.
        All scalar inputs; h, u, du_dt can be single-step values.
        """
        h_val = float(h.mean()) if isinstance(h, torch.Tensor) else float(h)
        u_val = float(u.mean()) if isinstance(u, torch.Tensor) else float(u)
        du_val = float(du_dt.mean()) if isinstance(du_dt, torch.Tensor) else float(du_dt)

        h_val = max(h_val, 1e-4)
        u_abs = abs(u_val)

        Fr = u_abs / math.sqrt(self.g * h_val)
        D = geom.width
        Re = u_abs * D / self.nu if u_abs > 0 else 0.0
        log_Re = math.log10(max(Re, 1.0))
        KC = u_abs * T_wave / max(D, 1e-3)
        h_over_H = h_val / max(geom.height, 1e-3)
        eta_net = h_val - geom.height
        eta_over_H = eta_net / max(geom.height, 1e-3)
        Rc_over_Hi = freeboard / max(H_incident, 1e-3)
        B_over_Hi = geom.width / max(H_incident, 1e-3)

        type_map = {"wall": 0, "rectangular": 1, "cylindrical": 2, "breakwater": 3}
        one_hot = [0.0, 0.0, 0.0, 0.0]
        one_hot[type_map.get(geom.shape, 1)] = 1.0

        vals = [
            Fr, log_Re, KC,
            h_over_H, eta_over_H, du_val,
            T_wave, H_incident,
            *one_hot,
            Rc_over_Hi, B_over_Hi,
        ]
        return torch.tensor(vals, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Embankment features
    # ------------------------------------------------------------------

    def extract_embankment_features(
        self,
        Q_overflow_seq: list[float],     # [T]
        H_upstream_seq: list[float],     # [T]
        embankment: EmbankmentProperties,
        dt: float,
        cumulative_overflow: list[float] | None = None,
        tau_bed_seq: list[float] | None = None,
    ) -> torch.Tensor:                   # [T, 8]
        T = len(Q_overflow_seq)
        feats = torch.zeros(T, 8, dtype=torch.float32)

        H_crest = embankment.crest_elevation
        cum_vol = 0.0

        for t in range(T):
            Q = Q_overflow_seq[t]
            H_up = H_upstream_seq[t]
            cum_vol += Q * dt

            dQ_dt = ((Q_overflow_seq[t] - Q_overflow_seq[t - 1]) / dt) if t > 0 else 0.0
            tau = tau_bed_seq[t] if tau_bed_seq else 0.0

            feats[t, 0] = Q
            feats[t, 1] = H_up
            feats[t, 2] = H_up - H_crest                       # overtopping head
            feats[t, 3] = dQ_dt
            feats[t, 4] = cum_vol
            feats[t, 5] = tau
            feats[t, 6] = 0.3   # clay_fraction (default; override from soil data)
            feats[t, 7] = embankment.D50 / 0.01               # normalized D50

        return feats
