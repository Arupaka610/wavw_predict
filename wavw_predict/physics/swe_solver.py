"""
Shallow Water Equations (SWE) finite-difference solver.

1D: Lax-Wendroff scheme in conservative flux form.
2D: Operator-splitting ADI — x-sweep then y-sweep using the 1D kernel.

All tensors use torch.float64 on CPU for numerical stability.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


_DTYPE = torch.float64
_DEVICE = torch.device("cpu")


def _to(x: torch.Tensor) -> torch.Tensor:
    return x.to(dtype=_DTYPE, device=_DEVICE)


# ---------------------------------------------------------------------------
# CFL helper
# ---------------------------------------------------------------------------

def compute_cfl_dt(
    dx: float,
    dy: float,
    u: torch.Tensor,
    v: torch.Tensor,
    eta: torch.Tensor,
    d: torch.Tensor,
    g: float = 9.81,
    cfl: float = 0.5,
) -> float:
    h = eta + d
    h = h.clamp(min=0.0)
    c = (g * h).sqrt()
    max_speed = (u.abs() + c).max().item()
    max_speed_y = (v.abs() + c).max().item()
    max_speed = max(max_speed, max_speed_y, 1e-6)
    return cfl * min(dx, dy) / max_speed


def _apply_sponge(field: torch.Tensor, n: int, alpha_max: float = 0.1) -> torch.Tensor:
    """Linear sponge damping on all four edges."""
    w = field.clone()
    for i in range(n):
        alpha = alpha_max * (n - i) / n
        w[i, :] *= 1.0 - alpha
        w[-(i + 1), :] *= 1.0 - alpha
        w[:, i] *= 1.0 - alpha
        w[:, -(i + 1)] *= 1.0 - alpha
    return w


# ---------------------------------------------------------------------------
# 1D solver
# ---------------------------------------------------------------------------

class SWESolver1D:
    """
    1D nonlinear SWE via Lax-Wendroff in conservative variables (eta, q=h*u).

    Governing equations:
        ∂η/∂t + ∂q/∂x = 0
        ∂q/∂t + ∂(q²/h + g·h²/2)/∂x = -g·h·∂d/∂x - C_f·q|q|/h²

    Boundary conditions:
        left  — time-series injection (eta prescribed, u from continuity)
        right — absorbing (zero-gradient + sponge)
    """

    def __init__(
        self,
        dx: float,
        dt: float,
        nx: int,
        bathymetry: torch.Tensor,   # [nx] undisturbed depth d (positive down)
        friction_n: float = 0.025,
        h_min: float = 1e-3,
        g: float = 9.81,
    ):
        self.dx = dx
        self.dt = dt
        self.nx = nx
        self.d = _to(bathymetry)       # [nx]
        self.friction_n = friction_n
        self.h_min = h_min
        self.g = g

    def _friction(self, q: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Manning bottom friction term."""
        cf = self.g * self.friction_n ** 2 / (h.clamp(min=self.h_min) ** (1.0 / 3.0))
        return cf * q * q.abs() / h.clamp(min=self.h_min)

    def step(
        self,
        eta: torch.Tensor,   # [nx]
        u: torch.Tensor,     # [nx]
        bc_eta: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dx, dt, g = self.dx, self.dt, self.g
        h = (eta + self.d).clamp(min=self.h_min)
        q = h * u

        # --- fluxes at cell interfaces (Lax-Wendroff) ---
        h_l, h_r = h[:-1], h[1:]
        q_l, q_r = q[:-1], q[1:]
        eta_l, eta_r = eta[:-1], eta[1:]

        # Lax-Friedrichs predictor at half-step
        h_half = 0.5 * (h_l + h_r) - 0.5 * (dt / dx) * (q_r - q_l)
        q_half = 0.5 * (q_l + q_r) - 0.5 * (dt / dx) * (
            q_r ** 2 / h_r + 0.5 * g * (eta_r + self.d[1:]) ** 2
            - q_l ** 2 / h_l - 0.5 * g * (eta_l + self.d[:-1]) ** 2
        )
        h_half = h_half.clamp(min=self.h_min)

        # Corrector fluxes
        F_eta = q_half                                      # mass flux
        F_q = q_half ** 2 / h_half + 0.5 * g * h_half ** 2  # momentum flux

        # Update interior cells
        eta_new = eta.clone()
        u_new = u.clone()

        eta_new[1:-1] = eta[1:-1] - (dt / dx) * (F_eta[1:] - F_eta[:-1])
        q_new_int = q[1:-1] - (dt / dx) * (F_q[1:] - F_q[:-1])

        # Bed slope source
        bed_slope = (self.d[2:] - self.d[:-2]) / (2.0 * dx)
        h_int = (eta_new[1:-1] + self.d[1:-1]).clamp(min=self.h_min)
        q_new_int = q_new_int - dt * g * h_int * bed_slope

        # Bottom friction (implicit-explicit split)
        q_new_int = q_new_int - dt * self._friction(q_new_int, h_int)

        u_new[1:-1] = q_new_int / h_int

        # Wetting/drying
        dry = (eta_new + self.d) < self.h_min
        u_new[dry] = 0.0

        # Boundary conditions
        if bc_eta is not None:
            eta_new[0] = bc_eta
        u_new[0] = u_new[1]    # zero-gradient velocity at left
        eta_new[-1] = eta_new[-2]
        u_new[-1] = u_new[-2]

        return eta_new, u_new

    def run(
        self,
        eta0: torch.Tensor,
        u0: torch.Tensor,
        n_steps: int,
        bc_series: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run solver for n_steps.
        Returns (eta_history [n_steps+1, nx], u_history [n_steps+1, nx]).
        """
        eta = _to(eta0.clone())
        u = _to(u0.clone())
        eta_hist = [eta.clone()]
        u_hist = [u.clone()]
        for t in range(n_steps):
            bc = float(bc_series[t]) if bc_series is not None else None
            eta, u = self.step(eta, u, bc_eta=bc)
            eta_hist.append(eta.clone())
            u_hist.append(u.clone())
        return torch.stack(eta_hist), torch.stack(u_hist)


# ---------------------------------------------------------------------------
# 2D solver (operator splitting ADI)
# ---------------------------------------------------------------------------

class SWESolver2D:
    """
    2D nonlinear SWE using operator-splitting (Strang splitting):
        Full step = x-sweep (dt/2) + y-sweep (dt) + x-sweep (dt/2)

    State variables: eta [ny, nx], u [ny, nx], v [ny, nx].
    """

    def __init__(
        self,
        dx: float,
        dy: float,
        dt: float,
        nx: int,
        ny: int,
        bathymetry: torch.Tensor,    # [ny, nx] undisturbed depth
        friction_n: float = 0.025,
        h_min: float = 1e-3,
        g: float = 9.81,
        sponge_cells: int = 10,
    ):
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.nx = nx
        self.ny = ny
        self.d = _to(bathymetry)          # [ny, nx]
        self.friction_n = friction_n
        self.h_min = h_min
        self.g = g
        self.sponge_cells = sponge_cells

    def _friction_coeff(self, h: torch.Tensor) -> torch.Tensor:
        return self.g * self.friction_n ** 2 / (h.clamp(min=self.h_min) ** (1.0 / 3.0))

    def _sweep_x(self, eta: torch.Tensor, u: torch.Tensor, v: torch.Tensor, dt_sub: float):
        """One x-direction LW sweep."""
        dx, g = self.dx, self.g
        h = (eta + self.d).clamp(min=self.h_min)
        q = h * u

        h_l, h_r = h[:, :-1], h[:, 1:]
        q_l, q_r = q[:, :-1], q[:, 1:]
        eta_l, eta_r = eta[:, :-1], eta[:, 1:]
        d_l, d_r = self.d[:, :-1], self.d[:, 1:]

        h_half = 0.5 * (h_l + h_r) - 0.5 * (dt_sub / dx) * (q_r - q_l)
        q_half = 0.5 * (q_l + q_r) - 0.5 * (dt_sub / dx) * (
            q_r ** 2 / h_r.clamp(min=self.h_min) + 0.5 * g * (eta_r + d_r).clamp(min=0) ** 2
            - q_l ** 2 / h_l.clamp(min=self.h_min) - 0.5 * g * (eta_l + d_l).clamp(min=0) ** 2
        )
        h_half = h_half.clamp(min=self.h_min)
        F_eta = q_half
        F_q = q_half ** 2 / h_half + 0.5 * g * h_half ** 2

        eta_new = eta.clone()
        u_new = u.clone()

        eta_new[:, 1:-1] = eta[:, 1:-1] - (dt_sub / dx) * (F_eta[:, 1:] - F_eta[:, :-1])
        q_new = q[:, 1:-1] - (dt_sub / dx) * (F_q[:, 1:] - F_q[:, :-1])

        bed_x = (self.d[:, 2:] - self.d[:, :-2]) / (2.0 * dx)
        h_int = (eta_new[:, 1:-1] + self.d[:, 1:-1]).clamp(min=self.h_min)
        q_new = q_new - dt_sub * g * h_int * bed_x
        cf = self._friction_coeff(h_int)
        q_new = q_new - dt_sub * cf * q_new * q_new.abs() / h_int.clamp(min=self.h_min)
        u_new[:, 1:-1] = q_new / h_int

        # zero-gradient BC
        eta_new[:, 0] = eta_new[:, 1]
        eta_new[:, -1] = eta_new[:, -2]
        u_new[:, 0] = u_new[:, 1]
        u_new[:, -1] = u_new[:, -2]

        dry = (eta_new + self.d) < self.h_min
        u_new[dry] = 0.0
        return eta_new, u_new, v

    def _sweep_y(self, eta: torch.Tensor, u: torch.Tensor, v: torch.Tensor, dt_sub: float):
        """One y-direction LW sweep."""
        dy, g = self.dy, self.g
        h = (eta + self.d).clamp(min=self.h_min)
        p = h * v   # y-momentum

        h_b, h_t = h[:-1, :], h[1:, :]
        p_b, p_t = p[:-1, :], p[1:, :]
        eta_b, eta_t = eta[:-1, :], eta[1:, :]
        d_b, d_t = self.d[:-1, :], self.d[1:, :]

        h_half = 0.5 * (h_b + h_t) - 0.5 * (dt_sub / dy) * (p_t - p_b)
        p_half = 0.5 * (p_b + p_t) - 0.5 * (dt_sub / dy) * (
            p_t ** 2 / h_t.clamp(min=self.h_min) + 0.5 * g * (eta_t + d_t).clamp(min=0) ** 2
            - p_b ** 2 / h_b.clamp(min=self.h_min) - 0.5 * g * (eta_b + d_b).clamp(min=0) ** 2
        )
        h_half = h_half.clamp(min=self.h_min)
        G_eta = p_half
        G_p = p_half ** 2 / h_half + 0.5 * g * h_half ** 2

        eta_new = eta.clone()
        v_new = v.clone()

        eta_new[1:-1, :] = eta[1:-1, :] - (dt_sub / dy) * (G_eta[1:, :] - G_eta[:-1, :])
        p_new = p[1:-1, :] - (dt_sub / dy) * (G_p[1:, :] - G_p[:-1, :])

        bed_y = (self.d[2:, :] - self.d[:-2, :]) / (2.0 * dy)
        h_int = (eta_new[1:-1, :] + self.d[1:-1, :]).clamp(min=self.h_min)
        p_new = p_new - dt_sub * g * h_int * bed_y
        cf = self._friction_coeff(h_int)
        p_new = p_new - dt_sub * cf * p_new * p_new.abs() / h_int.clamp(min=self.h_min)
        v_new[1:-1, :] = p_new / h_int

        # zero-gradient BC
        eta_new[0, :] = eta_new[1, :]
        eta_new[-1, :] = eta_new[-2, :]
        v_new[0, :] = v_new[1, :]
        v_new[-1, :] = v_new[-2, :]

        dry = (eta_new + self.d) < self.h_min
        v_new[dry] = 0.0
        return eta_new, u, v_new

    def step(
        self,
        eta: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Strang-splitting full timestep."""
        dt = self.dt
        eta, u, v = self._sweep_x(eta, u, v, dt / 2.0)
        eta, u, v = self._sweep_y(eta, u, v, dt)
        eta, u, v = self._sweep_x(eta, u, v, dt / 2.0)
        if self.sponge_cells > 0:
            eta = _apply_sponge(eta, self.sponge_cells)
        return eta, u, v

    def run(
        self,
        eta0: torch.Tensor,
        u0: torch.Tensor,
        v0: torch.Tensor,
        n_steps: int,
        bc_series: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run solver for n_steps.
        Returns (eta [n_steps+1, ny, nx], u [...], v [...]).
        """
        eta = _to(eta0.clone())
        u = _to(u0.clone())
        v = _to(v0.clone())
        eta_hist = [eta.clone()]
        u_hist = [u.clone()]
        v_hist = [v.clone()]
        for t in range(n_steps):
            if bc_series is not None:
                # bc_series: [n_steps, ny] — south boundary eta
                eta[0, :] = _to(bc_series[t])
            eta, u, v = self.step(eta, u, v)
            eta_hist.append(eta.clone())
            u_hist.append(u.clone())
            v_hist.append(v.clone())
        return torch.stack(eta_hist), torch.stack(u_hist), torch.stack(v_hist)
