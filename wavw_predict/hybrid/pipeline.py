"""
HybridPipeline: orchestrates the full physics + ML inference workflow.

Execution order:
  1. SWE solver (low-res or reduced domain) → wave fields
  2. FNO surrogate → high-res refinement / forecast extension
  3. WaveForceNet → Cd/Cm/Kt → building wave forces
  4. EmbankmentNet → failure probability / breach geometry
  5. Breach feedback → SWE boundary update → local re-run
  6. Uncertainty aggregation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..physics.swe_solver import SWESolver2D
from ..physics.wave_force import MorisonForceCalculator, StructureGeometry, HydrostaticForce
from ..physics.embankment import EmbankmentModel, EmbankmentProperties
from ..physics.transmission import transmission_coefficient_levee
from ..models.fno import FNO2D
from ..models.wave_force_net import WaveForceNet
from ..models.embankment_net import EmbankmentNet
from ..models.uncertainty import MCDropoutWrapper, UncertaintyResult
from ..hybrid.coupler import PhysicsMLCoupler
from ..utils.logging import get_logger


@dataclass
class InitialConditions:
    eta: torch.Tensor    # [H, W]
    u: torch.Tensor      # [H, W]
    v: torch.Tensor      # [H, W]
    bathymetry: torch.Tensor  # [H, W]
    coast_mask: torch.Tensor  # [H, W]


@dataclass
class PipelineConfig:
    dx: float = 100.0
    dy: float = 100.0
    dt: float = 2.0
    g: float = 9.81
    sponge_cells: int = 10
    fno_enabled: bool = True
    uncertainty_samples: int = 50
    device: str = "cpu"


@dataclass
class StructureSpec:
    name: str
    grid_row: int
    grid_col: int
    geometry: StructureGeometry
    T_wave: float = 300.0         # [s] representative wave period
    H_incident: float = 1.0      # [m] design incident wave height
    freeboard: float = 0.5       # [m]


@dataclass
class EmbankmentSpec:
    name: str
    grid_row: int
    grid_col: int
    props: EmbankmentProperties


@dataclass
class PipelineResult:
    # Wave fields [T, H, W]
    eta: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor
    # Building forces [n_structures, T]
    building_forces: dict[str, torch.Tensor]
    behind_forces: dict[str, torch.Tensor]
    # Embankment state [n_embankments, T]
    embankment_p_failure: dict[str, torch.Tensor]
    embankment_breach_width: dict[str, torch.Tensor]
    embankment_breach_depth: dict[str, torch.Tensor]


class HybridPipeline:
    """
    End-to-end tsunami analysis pipeline combining SWE physics and ML surrogates.
    """

    def __init__(
        self,
        cfg: PipelineConfig,
        fno_model: FNO2D | None = None,
        wave_force_model: WaveForceNet | None = None,
        embankment_model: EmbankmentNet | None = None,
    ):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.coupler = PhysicsMLCoupler(g=cfg.g)
        self.morison = MorisonForceCalculator(g=cfg.g)
        self.logger = get_logger("HybridPipeline")

        self.fno = fno_model
        self.wave_force_net = wave_force_model
        self.emb_net = embankment_model

        if self.fno:
            self.fno.to(self.device).eval()
        if self.wave_force_net:
            self.wave_force_net.to(self.device).eval()
        if self.emb_net:
            self.emb_net.to(self.device).eval()

    @classmethod
    def from_checkpoints(
        cls,
        cfg: PipelineConfig,
        fno_ckpt: str | Path | None = None,
        wf_ckpt: str | Path | None = None,
        emb_ckpt: str | Path | None = None,
    ) -> "HybridPipeline":
        dev = torch.device(cfg.device)

        fno = None
        if fno_ckpt:
            fno = FNO2D()
            ckpt = torch.load(fno_ckpt, map_location=dev)
            fno.load_state_dict(ckpt["model_state"])

        wf_net = None
        if wf_ckpt:
            wf_net = WaveForceNet()
            ckpt = torch.load(wf_ckpt, map_location=dev)
            wf_net.load_state_dict(ckpt["model_state"])

        emb_net = None
        if emb_ckpt:
            emb_net = EmbankmentNet()
            ckpt = torch.load(emb_ckpt, map_location=dev)
            emb_net.load_state_dict(ckpt["model_state"])

        return cls(cfg, fno_model=fno, wave_force_model=wf_net, embankment_model=emb_net)

    def _run_swe(
        self,
        ic: InitialConditions,
        n_steps: int,
        bc_series: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ny, nx = ic.eta.shape
        solver = SWESolver2D(
            dx=self.cfg.dx, dy=self.cfg.dy, dt=self.cfg.dt,
            nx=nx, ny=ny,
            bathymetry=ic.bathymetry,
            sponge_cells=self.cfg.sponge_cells,
        )
        return solver.run(ic.eta, ic.u, ic.v, n_steps, bc_series)

    def _fno_refine(
        self,
        eta_hist: torch.Tensor,   # [T, H, W]
        ic: InitialConditions,
        n_forecast: int,
    ) -> torch.Tensor:
        if self.fno is None:
            return eta_hist

        T, H, W = eta_hist.shape
        window = 4
        if T < window + 1:
            return eta_hist

        # Build initial FNO input from last 4 SWE steps
        eta_window = eta_hist[-(window):].unsqueeze(0).to(self.device)   # [1, 4, H, W]
        bathy = ic.bathymetry.unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, H, W]
        mask = ic.coast_mask.unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            preds = self.fno.rollout(
                torch.cat([eta_window, bathy, mask], dim=1),
                n_steps=n_forecast,
                bathymetry=bathy,
                coast_mask=mask,
            )   # [1, n_forecast, 3, H, W]

        refined_eta = preds[0, :, 0]   # [n_forecast, H, W]
        return torch.cat([eta_hist, refined_eta.cpu()], dim=0)

    @torch.no_grad()
    def _compute_building_forces(
        self,
        eta_hist: torch.Tensor,    # [T, H, W]
        u_hist: torch.Tensor,
        structures: list[StructureSpec],
        dt: float,
    ) -> tuple[dict, dict]:
        F_dict: dict[str, torch.Tensor] = {}
        Fb_dict: dict[str, torch.Tensor] = {}

        for spec in structures:
            r, c = spec.grid_row, spec.grid_col
            h_seq = (eta_hist[:, r, c] + 0.0).clamp(min=0.0)  # [T]
            u_seq = u_hist[:, r, c]
            du_dt = torch.gradient(u_seq, spacing=(dt,))[0]

            forces = []
            behind_forces = []
            for t in range(len(h_seq)):
                h_t = h_seq[t].unsqueeze(0)
                u_t = u_seq[t].unsqueeze(0)
                du_t = du_dt[t].unsqueeze(0)

                Cd, Cm, Kt = 2.0, 2.0, 0.3
                if self.wave_force_net is not None:
                    feats = self.coupler.extract_wave_force_features(
                        h_t, u_t, du_t, spec.geometry,
                        T_wave=spec.T_wave,
                        H_incident=spec.H_incident,
                        freeboard=spec.freeboard,
                    ).unsqueeze(0).to(self.device)
                    coeffs = self.wave_force_net(feats)
                    Cd = coeffs.Cd.item()
                    Cm = coeffs.Cm.item()
                    Kt = coeffs.Kt.item()
                    dF = coeffs.dF_behind.item()
                else:
                    dF = 0.0

                result = self.morison.compute(h_t, u_t, du_t, spec.geometry, Cd=Cd, Cm=Cm)
                F_total = result.F_total.item()
                F_behind = Kt ** 2 * F_total + dF

                forces.append(F_total)
                behind_forces.append(F_behind)

            F_dict[spec.name] = torch.tensor(forces)
            Fb_dict[spec.name] = torch.tensor(behind_forces)

        return F_dict, Fb_dict

    @torch.no_grad()
    def _compute_embankment_states(
        self,
        eta_hist: torch.Tensor,   # [T, H, W]
        embankments: list[EmbankmentSpec],
        dt: float,
    ) -> tuple[dict, dict, dict]:
        p_dict: dict[str, torch.Tensor] = {}
        w_dict: dict[str, torch.Tensor] = {}
        d_dict: dict[str, torch.Tensor] = {}

        for spec in embankments:
            r, c = spec.grid_row, spec.grid_col
            H_up_seq = eta_hist[:, r, c].tolist()

            phys_model = EmbankmentModel(spec.props)
            Q_seq = [phys_model.compute_overflow_discharge(h) for h in H_up_seq]

            if self.emb_net is not None:
                feats = self.coupler.extract_embankment_features(
                    Q_seq, H_up_seq, spec.props, dt
                ).unsqueeze(0).to(self.device)   # [1, T, 8]
                pred = self.emb_net(feats)
                p_fail = pred.p_failure[0].item()
                w_mean = pred.width_mean[0].item()
                d_mean = pred.depth_mean[0].item()
            else:
                # Fall back to pure physics
                p_fail = 0.0
                w_mean, d_mean = 0.0, 0.0
                for h in H_up_seq:
                    state = phys_model.step(h, dt)
                    if state.is_failed:
                        p_fail = 1.0
                        w_mean = state.breach_width
                        d_mean = state.breach_depth
                        break

            T = len(H_up_seq)
            p_dict[spec.name] = torch.full((T,), p_fail)
            w_dict[spec.name] = torch.full((T,), w_mean)
            d_dict[spec.name] = torch.full((T,), d_mean)

        return p_dict, w_dict, d_dict

    def run(
        self,
        ic: InitialConditions,
        n_steps: int,
        bc_series: torch.Tensor | None = None,
        structures: list[StructureSpec] | None = None,
        embankments: list[EmbankmentSpec] | None = None,
        n_forecast: int = 0,
    ) -> PipelineResult:
        """
        Full hybrid pipeline inference.

        ic          : initial wave conditions and bathymetry
        n_steps     : number of SWE timesteps to run
        bc_series   : [n_steps, ny] open boundary eta time series
        structures  : list of building/structure specifications
        embankments : list of embankment specifications
        n_forecast  : additional FNO rollout steps beyond SWE
        """
        self.logger.info(f"Running SWE solver for {n_steps} steps...")
        eta_h, u_h, v_h = self._run_swe(ic, n_steps, bc_series)
        # eta_h: [n_steps+1, H, W]

        if self.cfg.fno_enabled and n_forecast > 0 and self.fno is not None:
            self.logger.info(f"FNO refinement/forecast for {n_forecast} steps...")
            eta_h = self._fno_refine(eta_h, ic, n_forecast)
            u_ext = torch.zeros(n_forecast, *u_h.shape[1:])
            v_ext = torch.zeros(n_forecast, *v_h.shape[1:])
            u_h = torch.cat([u_h, u_ext], dim=0)
            v_h = torch.cat([v_h, v_ext], dim=0)

        dt = self.cfg.dt
        F_dict, Fb_dict = {}, {}
        if structures:
            self.logger.info(f"Computing wave forces for {len(structures)} structures...")
            F_dict, Fb_dict = self._compute_building_forces(eta_h, u_h, structures, dt)

        p_dict, w_dict, d_dict = {}, {}, {}
        if embankments:
            self.logger.info(f"Analyzing {len(embankments)} embankments...")
            p_dict, w_dict, d_dict = self._compute_embankment_states(eta_h, embankments, dt)

        return PipelineResult(
            eta=eta_h,
            u=u_h,
            v=v_h,
            building_forces=F_dict,
            behind_forces=Fb_dict,
            embankment_p_failure=p_dict,
            embankment_breach_width=w_dict,
            embankment_breach_depth=d_dict,
        )
