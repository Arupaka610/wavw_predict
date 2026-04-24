"""Tests for wave force calculations."""
import torch
import pytest
from wavw_predict.physics.wave_force import (
    MorisonForceCalculator, HydrostaticForce, StructureGeometry,
)


def _geom(w=5.0, h=8.0, l=10.0, shape="rectangular"):
    return StructureGeometry(width=w, height=h, length=l, shape=shape)


def test_morison_zero_velocity():
    calc = MorisonForceCalculator()
    geom = _geom()
    h = torch.tensor([3.0])
    u = torch.tensor([0.0])
    du_dt = torch.tensor([0.0])
    result = calc.compute(h, u, du_dt, geom)
    assert float(result.F_drag) == pytest.approx(0.0, abs=1e-9)
    assert float(result.F_inertia) == pytest.approx(0.0, abs=1e-9)


def test_morison_drag_scaling():
    """Drag force should scale as u^2."""
    calc = MorisonForceCalculator(Cd=2.0, Cm=0.0)
    geom = _geom()
    h = torch.tensor([5.0])
    du_dt = torch.tensor([0.0])

    u1 = torch.tensor([1.0])
    u2 = torch.tensor([2.0])
    F1 = calc.compute(h, u1, du_dt, geom).F_drag
    F2 = calc.compute(h, u2, du_dt, geom).F_drag
    assert float(F2 / F1) == pytest.approx(4.0, rel=1e-5)


def test_hydrostatic_force():
    hf = HydrostaticForce()
    geom = _geom(w=1.0)
    eta = torch.tensor([2.0])
    F = hf.compute(eta, geom)
    expected = 0.5 * 1025.0 * 9.81 * 4.0
    assert float(F) == pytest.approx(expected, rel=1e-4)


def test_transmission_zero_incident():
    from wavw_predict.physics.transmission import transmission_coefficient_levee
    Kt = transmission_coefficient_levee(0.0, 10.0, 0.5, 5.0)
    assert Kt == 0.0


def test_transmission_large_positive_freeboard():
    """Large positive freeboard → minimal transmission."""
    from wavw_predict.physics.transmission import transmission_coefficient_levee
    Kt = transmission_coefficient_levee(1.0, 10.0, 5.0, 5.0)
    assert 0.0 <= Kt <= 1.0


def test_submerged_height_clamping():
    """Submerged area should not exceed structure height."""
    calc = MorisonForceCalculator()
    geom = _geom(h=3.0)
    h = torch.tensor([10.0])  # water much deeper than structure
    u = torch.tensor([1.0])
    du_dt = torch.tensor([0.0])
    result = calc.compute(h, u, du_dt, geom)
    # Projected area should use min(h, H) = 3.0
    A_proj_expected = geom.width * geom.height
    A_proj_used = float(result.F_drag) / (0.5 * 1025.0 * 2.0 * 1.0)  # Cd=2, u=1, rho=1025
    assert A_proj_used == pytest.approx(A_proj_expected, rel=1e-4)
