"""
Wave force calculations on structures.

Morison equation: F = F_drag + F_inertia
Hydrostatic pressure: F = 0.5 * rho * g * eta^2
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class StructureGeometry:
    width: float     # B [m]  — dimension perpendicular to flow
    height: float    # H [m]  — vertical extent
    length: float    # L [m]  — dimension along flow
    shape: str = "rectangular"   # 'rectangular' | 'cylindrical'


@dataclass
class ForceResult:
    F_drag: torch.Tensor       # [N/m] or [N]
    F_inertia: torch.Tensor
    F_total: torch.Tensor
    F_hydrostatic: torch.Tensor
    pressure_profile: torch.Tensor   # [N/m²] at n_levels depth levels


class MorisonForceCalculator:
    """
    Computes wave force on a structure using the Morison equation.

    For rectangular structures the projected area is A = B * min(h, H).
    Cd and Cm may be overridden by the ML model (wave_force_net).
    """

    def __init__(self, rho: float = 1025.0, g: float = 9.81, Cd: float = 2.0, Cm: float = 2.0):
        self.rho = rho
        self.g = g
        self.Cd = Cd
        self.Cm = Cm

    def compute(
        self,
        h: torch.Tensor,          # total water depth [m]  scalar or [T]
        u: torch.Tensor,          # depth-averaged velocity [m/s]
        du_dt: torch.Tensor,      # flow acceleration [m/s²]
        geom: StructureGeometry,
        Cd: float | None = None,
        Cm: float | None = None,
    ) -> ForceResult:
        Cd = Cd if Cd is not None else self.Cd
        Cm = Cm if Cm is not None else self.Cm
        rho, g = self.rho, self.g

        if geom.shape == "cylindrical":
            D = geom.width   # diameter
            A_proj = D * h.clamp(max=geom.height)   # [m²/m] (per unit length implicitly)
            V_sub = 0.25 * torch.pi * D ** 2 * h.clamp(max=geom.height)
        else:  # rectangular
            A_proj = geom.width * h.clamp(max=geom.height)
            V_sub = geom.width * geom.length * h.clamp(max=geom.height)

        F_drag = 0.5 * rho * Cd * A_proj * u * u.abs()
        F_inertia = rho * Cm * V_sub * du_dt

        # Hydrostatic net force on a vertical wall (per unit width)
        eta_net = h - (h - geom.height).clamp(min=0.0)  # submerged height
        F_hydrostatic = 0.5 * rho * g * eta_net ** 2 * geom.width

        F_total = F_drag + F_inertia

        # Simplified pressure profile at 10 depth levels
        n_levels = 10
        z_levels = torch.linspace(0, 1, n_levels, dtype=h.dtype, device=h.device)
        z = z_levels.unsqueeze(-1) * h.unsqueeze(0)   # [10, T] or [10]
        submerged = (z < h.unsqueeze(0)).float()
        pressure_profile = rho * g * (h.unsqueeze(0) - z) * submerged

        return ForceResult(
            F_drag=F_drag,
            F_inertia=F_inertia,
            F_total=F_total,
            F_hydrostatic=F_hydrostatic,
            pressure_profile=pressure_profile,
        )


class HydrostaticForce:
    """Net hydrostatic force on a vertical wall."""

    def __init__(self, rho: float = 1025.0, g: float = 9.81):
        self.rho = rho
        self.g = g

    def compute(self, eta: torch.Tensor, geom: StructureGeometry) -> torch.Tensor:
        """Returns net horizontal force [N/m] for unit-width wall."""
        h = eta.clamp(min=0.0)
        return 0.5 * self.rho * self.g * h ** 2


def wave_transmission_coefficient(
    H_incident: float,
    T_wave: float,
    freeboard: float,   # R_c = crest_elevation - water_level (negative = overtopping)
    crest_width: float,
    slope_angle: float,  # degrees, seaward slope
    structure_type: str = "levee",
) -> float:
    """
    Analytical K_t from van der Meer / EurOtop (2000).

    K_t = a * (R_c / H_i)^b + c   (clamped to [0, 1])
    """
    if H_incident < 1e-6:
        return 0.0

    Rc_Hi = freeboard / H_incident

    if structure_type == "levee":
        # d'Angremond et al. (1996)
        Bw = crest_width
        Kt = -0.4 * Rc_Hi + 0.64 * (Bw / H_incident) ** (-0.31) * (
            1.0 - torch.e.item() ** (-0.5 * 1.0)
        )
    else:
        # Generic
        Kt = -0.4 * Rc_Hi + 0.64

    return float(max(0.0, min(1.0, Kt)))
