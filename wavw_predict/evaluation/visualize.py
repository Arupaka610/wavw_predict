"""
Visualization utilities for wave fields, forces, and embankment states.
"""
from __future__ import annotations

from pathlib import Path

import torch


def plot_wave_field(
    eta: torch.Tensor,         # [H, W] single timestep
    bathymetry: torch.Tensor | None = None,
    title: str = "Surface elevation η [m]",
    save_path: str | Path | None = None,
    vmax: float | None = None,
) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415
    fig, ax = plt.subplots(figsize=(8, 6))
    vmax = vmax or float(eta.abs().max().item())
    im = ax.imshow(eta.numpy(), origin="lower", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, aspect="auto")
    if bathymetry is not None:
        ax.contour(bathymetry.numpy(), levels=[0], colors="k", linewidths=0.8)
    plt.colorbar(im, ax=ax, label="η [m]")
    ax.set_title(title)
    ax.set_xlabel("x [cells]")
    ax.set_ylabel("y [cells]")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_time_series(
    times: torch.Tensor,
    series_dict: dict[str, torch.Tensor],
    ylabel: str = "Value",
    title: str = "Time series",
    save_path: str | Path | None = None,
) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, series in series_dict.items():
        ax.plot(times.numpy(), series.numpy(), label=label)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_force_comparison(
    F_pred: torch.Tensor,
    F_target: torch.Tensor,
    times: torch.Tensor | None = None,
    save_path: str | Path | None = None,
) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415
    t = times.numpy() if times is not None else range(len(F_pred))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].plot(t, F_target.numpy(), label="Target", color="k")
    axes[0].plot(t, F_pred.numpy(), label="Predicted", color="r", linestyle="--")
    axes[0].set_ylabel("Force [N/m]")
    axes[0].legend()
    axes[1].plot(t, (F_pred - F_target).numpy(), color="b")
    axes[1].axhline(0, color="k", linewidth=0.5)
    axes[1].set_ylabel("Residual [N/m]")
    axes[1].set_xlabel("Time [s]")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_embankment_state(
    p_failure: torch.Tensor,
    breach_width: torch.Tensor,
    breach_depth: torch.Tensor,
    times: torch.Tensor | None = None,
    save_path: str | Path | None = None,
) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415
    t = times.numpy() if times is not None else range(len(p_failure))
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, p_failure.numpy())
    axes[0].set_ylabel("P(failure)")
    axes[0].set_ylim(0, 1)
    axes[1].plot(t, breach_width.numpy())
    axes[1].set_ylabel("Breach width [m]")
    axes[2].plot(t, breach_depth.numpy())
    axes[2].set_ylabel("Breach depth [m]")
    axes[2].set_xlabel("Time [s]")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def animate_wave_field(
    eta_hist: torch.Tensor,   # [T, H, W]
    dt: float,
    save_path: str | Path = "wave_animation.gif",
    fps: int = 10,
    vmax: float | None = None,
) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: PLC0415

    vmax = vmax or float(eta_hist.abs().max().item())
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(eta_hist[0].numpy(), origin="lower", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, animated=True)
    plt.colorbar(im, ax=ax, label="η [m]")
    title = ax.set_title("t = 0.0 s")

    def update(frame):
        im.set_data(eta_hist[frame].numpy())
        title.set_text(f"t = {frame * dt:.1f} s")
        return im, title

    anim = FuncAnimation(fig, update, frames=len(eta_hist), interval=1000 // fps, blit=True)
    anim.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
