"""
Embankment overflow, breach initiation, and breach growth model.

Physical basis:
  Stage 1 — overflow discharge  (broad-crested weir formula)
  Stage 2 — breach initiation   (critical erosion energy threshold)
  Stage 3 — breach growth       (empirical width/depth erosion ODEs)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import torch


class EmbankmentStage(Enum):
    INTACT = auto()
    OVERFLOWING = auto()
    BREACHING = auto()
    FAILED = auto()


@dataclass
class EmbankmentProperties:
    crest_elevation: float     # [m] above datum
    crest_width: float         # [m]
    upstream_slope: float      # H:V (e.g., 2.0 means 2 horizontal to 1 vertical)
    downstream_slope: float
    D50: float                 # median grain size [m]
    tau_critical: float        # critical shear stress for erosion [Pa]
    K_breach: float = 1.5e-4   # lateral breach growth coefficient
    K_depth: float = 2.0e-5    # breach depth growth coefficient


@dataclass
class EmbankmentState:
    Q_out: float = 0.0
    breach_width: float = 0.0
    breach_depth: float = 0.0
    stage: EmbankmentStage = EmbankmentStage.INTACT
    cumulative_overflow_volume: float = 0.0
    cumulative_erosion_energy: float = 0.0
    is_failed: bool = False


class EmbankmentModel:
    """
    Single-embankment breach model integrating breach physics.

    Call `step(H_upstream, dt)` each timestep.
    """

    def __init__(
        self,
        props: EmbankmentProperties,
        rho: float = 1000.0,
        g: float = 9.81,
        Cd_weir: float = 1.7,
        E_critical: float = 5000.0,    # [J/m²] erosion energy threshold for breach initiation
    ):
        self.props = props
        self.rho = rho
        self.g = g
        self.Cd_weir = Cd_weir
        self.E_critical = E_critical
        self.state = EmbankmentState()

    def compute_overflow_discharge(self, H_upstream: float) -> float:
        """Broad-crested weir formula [m³/s per unit width]."""
        h_over = H_upstream - self.props.crest_elevation
        if h_over <= 0.0:
            return 0.0
        return self.Cd_weir * self.props.crest_width * h_over ** 1.5

    def _bed_shear_stress(self, Q: float, h: float) -> float:
        """Approximate bed shear stress from Manning's equation."""
        if h < 1e-4 or Q < 1e-10:
            return 0.0
        v = Q / max(h, 1e-4)
        n = 0.025
        R_h = h   # approximate hydraulic radius
        tau = self.rho * self.g * n ** 2 * v ** 2 / (R_h ** (1.0 / 3.0))
        return tau

    def _check_breach_initiation(self, Q_overflow: float, dt: float) -> bool:
        if Q_overflow <= 0.0:
            return False
        h_over = max(0.0, self.state.Q_out / max(self.props.crest_width, 1e-3))
        tau = self._bed_shear_stress(Q_overflow, max(h_over, 0.05))
        excess = max(0.0, tau - self.props.tau_critical)
        self.state.cumulative_erosion_energy += excess * dt * self.props.crest_width
        return self.state.cumulative_erosion_energy >= self.E_critical

    def _grow_breach(self, H_upstream: float, dt: float) -> None:
        """Update breach width and depth using empirical ODE."""
        p = self.props
        B = self.state.breach_width
        Z_b = self.state.breach_depth   # depth below crest

        h_over = max(0.0, H_upstream - (p.crest_elevation - Z_b))
        Q_breach = self.Cd_weir * max(B, 0.1) * h_over ** 1.5 if h_over > 0 else 0.0
        q_unit = Q_breach / max(B, 0.1)   # unit discharge

        # Lateral widening
        dB_dt = p.K_breach * q_unit ** 1.5 if q_unit > 0 else 0.0
        # Bed lowering
        tau = self._bed_shear_stress(Q_breach, max(h_over * 0.5, 0.05))
        tau_excess = max(0.0, tau - p.tau_critical)
        dZ_dt = p.K_depth * tau_excess / max(p.D50, 1e-6)

        self.state.breach_width = B + dB_dt * dt
        self.state.breach_depth = min(Z_b + dZ_dt * dt, p.crest_elevation)
        self.state.Q_out = Q_breach

    def step(self, H_upstream: float, dt: float) -> EmbankmentState:
        """Advance breach model by one timestep."""
        p = self.props
        s = self.state

        Q_overflow = self.compute_overflow_discharge(H_upstream)
        s.cumulative_overflow_volume += Q_overflow * dt

        if s.stage == EmbankmentStage.INTACT:
            if Q_overflow > 0.0:
                s.stage = EmbankmentStage.OVERFLOWING
                s.Q_out = Q_overflow
            else:
                s.Q_out = 0.0

        elif s.stage == EmbankmentStage.OVERFLOWING:
            s.Q_out = Q_overflow
            if self._check_breach_initiation(Q_overflow, dt):
                s.stage = EmbankmentStage.BREACHING
                s.breach_width = max(1.0, 0.1 * p.crest_width)
                s.breach_depth = 0.1

        elif s.stage == EmbankmentStage.BREACHING:
            self._grow_breach(H_upstream, dt)
            # Failure when breach bottom reaches foundation or width > crest_width
            if s.breach_depth >= p.crest_elevation or s.breach_width >= p.crest_width * 5:
                s.stage = EmbankmentStage.FAILED
                s.is_failed = True

        elif s.stage == EmbankmentStage.FAILED:
            # Fully breached: treat as wide rectangular opening
            h_diff = max(0.0, H_upstream - (p.crest_elevation - s.breach_depth))
            s.Q_out = self.Cd_weir * s.breach_width * h_diff ** 1.5 if h_diff > 0 else 0.0

        return EmbankmentState(
            Q_out=s.Q_out,
            breach_width=s.breach_width,
            breach_depth=s.breach_depth,
            stage=s.stage,
            cumulative_overflow_volume=s.cumulative_overflow_volume,
            cumulative_erosion_energy=s.cumulative_erosion_energy,
            is_failed=s.is_failed,
        )

    def reset(self) -> None:
        self.state = EmbankmentState()
