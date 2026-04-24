"""
Analytical wave transmission behind coastal structures.

Base formula: van der Meer / EurOtop (2000) K_t formulation.
ML model in wave_force_net.py learns a correction δK_t on top.
"""
from __future__ import annotations

import math


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def transmission_coefficient_levee(
    H_incident: float,
    T_wave: float,
    freeboard: float,        # R_c [m], positive = crest above water
    crest_width: float,
    downstream_slope: float = 2.0,  # H:V
) -> float:
    """
    d'Angremond et al. (1996) formula for rubble-mound/levee structures.

        K_t = -0.4 * (R_c/H_i) + 0.64 * (B/H_i)^-0.31 * (1 - exp(-0.5*Ir))

    Ir = Iribarren number  (tan(α) / sqrt(H_i / L_0))
    L_0 = g * T^2 / (2π)   deep-water wavelength
    """
    if H_incident < 1e-6:
        return 0.0
    g = 9.81
    L0 = g * T_wave ** 2 / (2.0 * math.pi)
    alpha = math.atan(1.0 / max(downstream_slope, 0.1))  # slope angle [rad]
    Ir = math.tan(alpha) / math.sqrt(max(H_incident / L0, 1e-9))
    B = crest_width
    Rc_Hi = freeboard / H_incident
    Kt = -0.4 * Rc_Hi + 0.64 * (B / H_incident) ** (-0.31) * (1.0 - math.exp(-0.5 * Ir))
    return _clamp01(Kt)


def transmission_coefficient_vertical_wall(
    H_incident: float,
    freeboard: float,
) -> float:
    """
    Simple empirical formula for vertical walls.
    K_t ≈ 0  for non-overtopping, increases with overtopping head.
    """
    if H_incident < 1e-6:
        return 0.0
    Rc_Hi = freeboard / H_incident
    if Rc_Hi >= 1.0:
        return 0.0
    Kt = 0.5 * (1.0 - Rc_Hi)
    return _clamp01(Kt)


def behind_force_fraction(
    Kt: float,
    F_incident: float,
) -> float:
    """Transmitted force behind structure [N/m] ~ Kt^2 * F_incident."""
    return Kt ** 2 * F_incident
