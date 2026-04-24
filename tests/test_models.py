"""Tests for ML model forward passes and output shapes."""
import torch
import pytest
from wavw_predict.models.fno import FNO2D
from wavw_predict.models.wave_force_net import WaveForceNet, build_feature_vector
from wavw_predict.models.embankment_net import EmbankmentNet
from wavw_predict.models.uncertainty import MCDropoutWrapper


@pytest.fixture
def fno_small():
    return FNO2D(in_channels=6, out_channels=3, width=8, modes1=4, modes2=4, n_layers=2)


def test_fno_forward(fno_small):
    B, H, W = 2, 32, 32
    x = torch.randn(B, 6, H, W)
    out = fno_small(x)
    assert out.shape == (B, 3, H, W)


def test_fno_rollout(fno_small):
    B, H, W = 2, 32, 32
    x0 = torch.randn(B, 6, H, W)
    bathy = torch.randn(B, 1, H, W)
    mask = torch.zeros(B, 1, H, W)
    out = fno_small.rollout(x0, n_steps=5, bathymetry=bathy, coast_mask=mask)
    assert out.shape == (B, 5, 3, H, W)


def test_wave_force_net_forward():
    model = WaveForceNet(in_features=14, hidden=32, n_res_blocks=2)
    x = torch.randn(8, 14)
    coeffs = model(x)
    assert coeffs.Cd.shape == (8,)
    assert coeffs.Cm.shape == (8,)
    assert coeffs.Kt.shape == (8,)
    # Physical bounds
    assert (coeffs.Cd >= 0.5).all() and (coeffs.Cd <= 5.0).all()
    assert (coeffs.Cm >= 1.0).all() and (coeffs.Cm <= 4.0).all()
    assert (coeffs.Kt >= 0.0).all() and (coeffs.Kt <= 1.0).all()


def test_wave_force_net_feature_vector():
    feats = build_feature_vector(
        Fr=0.3, log_Re=5.0, KC=8.0,
        h_over_H=0.8, eta_over_H=0.2, du_dt=0.1,
        T_wave=120.0, H_incident=2.0,
        structure_type="rectangular",
        Rc_over_Hi=0.5, B_over_Hi=2.5,
    )
    assert feats.shape == (14,)
    assert torch.isfinite(feats).all()


def test_embankment_net_forward():
    model = EmbankmentNet(in_features=8, tcn_channels=[16, 32], kernel_size=3,
                          dilations=[1, 2], dropout=0.0)
    B, T = 4, 60
    x = torch.randn(B, T, 8)
    pred = model(x)
    assert pred.p_failure.shape == (B,)
    assert pred.width_mean.shape == (B,)
    assert (pred.p_failure >= 0).all() and (pred.p_failure <= 1).all()
    assert (pred.width_mean >= 0).all()


def test_mc_dropout_uncertainty():
    base_model = WaveForceNet(in_features=14, hidden=32, n_res_blocks=2)
    wrapper = MCDropoutWrapper(base_model, n_samples=10)
    x = torch.randn(3, 14)
    result = wrapper.predict_with_uncertainty(x)
    assert result.mean is not None
    assert result.epistemic_std is not None
