"""Tests for the SWE solver."""
import math
import torch
import pytest
from wavw_predict.physics.swe_solver import SWESolver1D, SWESolver2D, compute_cfl_dt


def test_swe1d_still_water():
    """Still water (no perturbation) should remain still."""
    nx = 50
    d = torch.full((nx,), 10.0)
    solver = SWESolver1D(dx=100.0, dt=1.0, nx=nx, bathymetry=d)
    eta = torch.zeros(nx)
    u = torch.zeros(nx)
    eta_new, u_new = solver.step(eta, u)
    assert torch.allclose(eta_new, torch.zeros(nx), atol=1e-10)


def test_swe1d_mass_conservation():
    """Total volume (eta + d integrated) should be conserved (no in/out flow)."""
    nx = 100
    d = torch.linspace(20, 5, nx)
    solver = SWESolver1D(dx=50.0, dt=0.5, nx=nx, bathymetry=d)
    # Gaussian initial perturbation
    x = torch.linspace(0, 1, nx)
    eta = 0.5 * torch.exp(-((x - 0.5) ** 2) / 0.01)
    u = torch.zeros(nx)
    eta_run, _ = solver.run(eta, u, n_steps=20)
    # Volume (sum of eta) should be approximately conserved (small BC leakage is ok)
    vol_init = eta_run[0].sum()
    vol_final = eta_run[-1].sum()
    assert abs((vol_final - vol_init) / max(vol_init.abs(), 1e-6)) < 0.1


def test_swe1d_wave_speed():
    """
    Verify linear wave speed by tracking the global peak of a right-going wave.
    Initialize with u = (c/d)*eta to ensure rightward propagation.
    """
    g = 9.81
    d_val = 10.0
    c_lin = math.sqrt(g * d_val)
    nx = 200
    dx = 50.0
    dt = 0.5
    d = torch.full((nx,), d_val)
    solver = SWESolver1D(dx=dx, dt=dt, nx=nx, bathymetry=d)

    amp = 0.3
    x = torch.linspace(0, nx * dx, nx)
    x0 = nx * dx * 0.3
    sigma = 5 * dx
    eta = amp * torch.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
    u = eta * (c_lin / d_val)

    # Run enough steps so the wave visibly moves, but not so many it exits domain
    n_steps = 50
    eta_run, _ = solver.run(eta, u, n_steps=n_steps)

    # Track global peak (rightward wave dominates because of initial u)
    t_start, t_end = 2, n_steps
    peak_start = int(eta_run[t_start].argmax())
    peak_end = int(eta_run[t_end].argmax())
    dist = (peak_end - peak_start) * dx
    elapsed = (t_end - t_start) * dt

    # Check wave moved rightward by a distance consistent with c_lin (25% tolerance)
    expected_dist = c_lin * elapsed
    assert dist > 0, "Wave peak should move to the right"
    assert abs(dist - expected_dist) / expected_dist < 0.25, (
        f"Wave traveled {dist:.1f} m, expected ~{expected_dist:.1f} m"
    )


def test_swe2d_still_water():
    nx, ny = 20, 20
    d = torch.full((ny, nx), 5.0)
    solver = SWESolver2D(dx=100.0, dy=100.0, dt=1.0, nx=nx, ny=ny,
                         bathymetry=d, sponge_cells=0)
    eta = torch.zeros(ny, nx)
    u = torch.zeros(ny, nx)
    v = torch.zeros(ny, nx)
    eta_new, u_new, v_new = solver.step(eta, u, v)
    assert torch.allclose(eta_new, torch.zeros(ny, nx), atol=1e-9)


def test_compute_cfl_dt():
    h = torch.full((10, 10), 5.0)
    u = torch.ones(10, 10) * 2.0
    v = torch.zeros(10, 10)
    eta = torch.zeros(10, 10)
    dt = compute_cfl_dt(100.0, 100.0, u, v, eta, h, g=9.81, cfl=0.5)
    # dt should be positive and physically reasonable
    assert dt > 0
    c = math.sqrt(9.81 * 5.0)
    expected = 0.5 * 100.0 / (2.0 + c)
    assert abs(dt - expected) / expected < 0.01
