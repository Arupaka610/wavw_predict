"""End-to-end pipeline tests (no ML models — physics only)."""
import torch
import pytest
from wavw_predict.hybrid.pipeline import (
    HybridPipeline, PipelineConfig, InitialConditions, StructureSpec,
)
from wavw_predict.physics.wave_force import StructureGeometry


def _make_ic(ny=20, nx=20):
    return InitialConditions(
        eta=torch.zeros(ny, nx),
        u=torch.zeros(ny, nx),
        v=torch.zeros(ny, nx),
        bathymetry=torch.full((ny, nx), 5.0),
        coast_mask=torch.zeros(ny, nx),
    )


def test_pipeline_runs_without_models():
    cfg = PipelineConfig(dx=100.0, dy=100.0, dt=2.0, sponge_cells=2, fno_enabled=False)
    pipeline = HybridPipeline(cfg, fno_model=None, wave_force_model=None, embankment_model=None)
    ic = _make_ic()
    result = pipeline.run(ic, n_steps=5)
    assert result.eta.shape[0] == 6   # n_steps+1
    assert result.eta.shape[1:] == (20, 20)


def test_pipeline_with_structure():
    cfg = PipelineConfig(dx=100.0, dy=100.0, dt=2.0, sponge_cells=0, fno_enabled=False)
    pipeline = HybridPipeline(cfg)
    ic = _make_ic()

    structures = [
        StructureSpec(
            name="bldg_1",
            grid_row=10,
            grid_col=10,
            geometry=StructureGeometry(width=5.0, height=8.0, length=10.0),
        )
    ]
    result = pipeline.run(ic, n_steps=5, structures=structures)
    assert "bldg_1" in result.building_forces
    assert result.building_forces["bldg_1"].shape[0] == 6


def test_pipeline_output_finite():
    cfg = PipelineConfig(dx=100.0, dy=100.0, dt=1.0, sponge_cells=2, fno_enabled=False)
    pipeline = HybridPipeline(cfg)
    ny, nx = 16, 16
    ic = InitialConditions(
        eta=0.3 * torch.randn(ny, nx),
        u=torch.zeros(ny, nx),
        v=torch.zeros(ny, nx),
        bathymetry=torch.full((ny, nx), 10.0),
        coast_mask=torch.zeros(ny, nx),
    )
    result = pipeline.run(ic, n_steps=10)
    assert torch.isfinite(result.eta).all(), "NaN/Inf in eta output"
