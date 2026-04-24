"""Tests for embankment breach model."""
import pytest
from wavw_predict.physics.embankment import (
    EmbankmentModel, EmbankmentProperties, EmbankmentStage,
)


def _props(crest_elev=3.0, crest_width=5.0, D50=0.001, tau_crit=1.0, K_b=2e-4):
    return EmbankmentProperties(
        crest_elevation=crest_elev,
        crest_width=crest_width,
        upstream_slope=2.0,
        downstream_slope=2.0,
        D50=D50,
        tau_critical=tau_crit,
        K_breach=K_b,
    )


def test_no_overflow_below_crest():
    model = EmbankmentModel(_props())
    state = model.step(H_upstream=2.5, dt=1.0)
    assert state.Q_out == pytest.approx(0.0, abs=1e-9)
    assert state.stage == EmbankmentStage.INTACT


def test_overflow_above_crest():
    model = EmbankmentModel(_props())
    state = model.step(H_upstream=4.0, dt=1.0)
    assert state.Q_out > 0.0
    assert state.stage == EmbankmentStage.OVERFLOWING


def test_breach_eventually_triggered():
    """With sustained high overflow, breach should eventually initiate."""
    model = EmbankmentModel(_props(tau_crit=0.01), E_critical=1.0)
    dt = 1.0
    breached = False
    for _ in range(500):
        state = model.step(H_upstream=5.0, dt=dt)
        if state.stage in {EmbankmentStage.BREACHING, EmbankmentStage.FAILED}:
            breached = True
            break
    assert breached, "Breach should initiate with sustained overtopping"


def test_no_breach_without_overflow():
    """With water level below crest, breach must never occur."""
    model = EmbankmentModel(_props())
    for _ in range(1000):
        state = model.step(H_upstream=1.0, dt=1.0)
    assert not state.is_failed


def test_reset():
    model = EmbankmentModel(_props(tau_crit=0.01), E_critical=1.0)
    for _ in range(200):
        model.step(H_upstream=5.0, dt=1.0)
    model.reset()
    state = model.step(H_upstream=1.0, dt=1.0)
    assert state.stage == EmbankmentStage.INTACT


def test_overflow_discharge_formula():
    """Test broad-crested weir formula: Q = 1.7 * L * h^1.5."""
    model = EmbankmentModel(_props(crest_elev=2.0, crest_width=10.0))
    h_over = 1.0
    Q = model.compute_overflow_discharge(H_upstream=3.0)
    expected = 1.7 * 10.0 * h_over ** 1.5
    assert Q == pytest.approx(expected, rel=1e-5)
