"""Visualisation helpers for the TG-state pipeline.

Conventions:
  * FPP heatmap: phase (degrees, 0..360) on x, frequency on y, log power
    (z-scored amplitude) colour-coded with ``viridis``.
  * PPC heatmap: phase on x, frequency on y, ``hot`` colormap (per user
    preference).
  * Marker for gravity center: small black/white triangle.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .clustering import STATE_NAMES, TGClusterResult


PPC_CMAP = "hot"
FPP_CMAP = "viridis"
STATE_COLORS = {
    0: "#1f77b4",   # S-gamma     (blue)
    1: "#9467bd",   # M-gamma     (purple)
    2: "#2ca02c",   # EF-gamma    (green)
    3: "#ff7f0e",   # LF-gamma    (orange)
}


def _phase_axis_deg(phase_centers_deg: np.ndarray) -> np.ndarray:
    return np.asarray(phase_centers_deg, dtype=float)


# ---------------------------------------------------------------------------
# FPP / m-FPP plots
# ---------------------------------------------------------------------------

def plot_fpp(fpp: np.ndarray,
             frequencies: np.ndarray,
             phase_centers_deg: np.ndarray,
             *,
             ax=None,
             title: str | None = None,
             cmap: str = FPP_CMAP,
             vmin: float | None = None,
             vmax: float | None = None,
             gravity_freq: float | None = None,
             gravity_phase_deg: float | None = None,
             mask: np.ndarray | None = None) -> "plt.Axes":
    """Plot one FPP (or m-FPP) as a heatmap.

    Optionally overlays the gamma-field mask (white contour) and the
    gravity centre (triangle marker).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(3.2, 3.0))
    phases = _phase_axis_deg(phase_centers_deg)
    im = ax.pcolormesh(phases, frequencies, fpp, shading="auto",
                       cmap=cmap, vmin=vmin, vmax=vmax)
    if mask is not None:
        ax.contour(phases, frequencies, mask.astype(float),
                   levels=[0.5], colors="white", linewidths=0.7)
    if gravity_freq is not None and gravity_phase_deg is not None:
        ax.plot(gravity_phase_deg, gravity_freq, marker="^",
                markersize=8, markerfacecolor="white",
                markeredgecolor="black", linestyle="")
    ax.set_xlabel("theta phase (deg)")
    ax.set_ylabel("frequency (Hz)")
    if title:
        ax.set_title(title)
    ax.set_xlim(phases.min(), phases.max())
    return ax


def plot_state_panel(result: TGClusterResult,
                     *,
                     fig=None,
                     cmap: str = FPP_CMAP,
                     vmin: float | None = None,
                     vmax: float | None = None) -> "plt.Figure":
    """One row of four m-FPP heatmaps, labelled S/M/EF/LF."""
    if fig is None:
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), constrained_layout=True)
    else:
        axes = fig.subplots(1, 4)
    phase_deg = np.degrees(result.phase_centers_rad)
    for s in range(4):
        m_fpp = result.m_fpps[s]
        gp_rad = result.gravity_phases[s]
        gp_deg = float(np.degrees(gp_rad) % 360.0)
        plot_fpp(m_fpp, result.frequencies, phase_deg,
                 ax=axes[s], cmap=cmap, vmin=vmin, vmax=vmax,
                 title=f"{STATE_NAMES[s]}\n{result.gravity_freqs[s]:.1f} Hz, "
                       f"{np.degrees(gp_rad):+.0f}°",
                 gravity_freq=result.gravity_freqs[s],
                 gravity_phase_deg=gp_deg,
                 mask=result.masks[s])
    return fig


# ---------------------------------------------------------------------------
# State summary plots
# ---------------------------------------------------------------------------

def plot_polar_state_density(gravity_freqs: np.ndarray,
                             gravity_phases: np.ndarray,
                             *,
                             ax=None,
                             title: str | None = None) -> "plt.Axes":
    """Polar scatter where radius = gravity frequency, angle = phase."""
    if ax is None:
        fig = plt.figure(figsize=(3.5, 3.5))
        ax = fig.add_subplot(111, projection="polar")
    elif ax.name != "polar":
        raise ValueError("ax must be a polar axes")
    for s in range(len(gravity_freqs)):
        ax.scatter(gravity_phases[s], gravity_freqs[s],
                   color=STATE_COLORS[s], label=STATE_NAMES[s],
                   s=60, alpha=0.85, edgecolor="black")
    ax.set_rlim(0, 180)
    ax.set_rticks([60, 120, 180])
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.05), fontsize=8)
    return ax


def plot_state_occurrence(occurrences: dict[str, np.ndarray],
                          *, ax=None,
                          title: str | None = None) -> "plt.Axes":
    """Bar chart of state occurrence per behavioural condition."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4.0, 3.0))
    n_states = 4
    conds = list(occurrences.keys())
    x = np.arange(n_states)
    width = 0.8 / len(conds)
    for ci, cname in enumerate(conds):
        ax.bar(x + (ci - (len(conds) - 1) / 2) * width,
               occurrences[cname],
               width=width, label=cname,
               edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(STATE_NAMES, rotation=20)
    ax.set_ylabel("occurrence probability")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title)
    return ax


# ---------------------------------------------------------------------------
# Transition matrices
# ---------------------------------------------------------------------------

def plot_transition_matrix(T: np.ndarray,
                           *, ax=None,
                           title: str | None = None,
                           cmap: str = "magma",
                           vmin: float = 0.0, vmax: float = 0.6) -> "plt.Axes":
    if ax is None:
        _, ax = plt.subplots(figsize=(3.2, 3.0))
    im = ax.imshow(T, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(STATE_NAMES, rotation=30, fontsize=8)
    ax.set_yticklabels(STATE_NAMES, fontsize=8)
    ax.set_xlabel("next state")
    ax.set_ylabel("current state")
    for i in range(4):
        for j in range(4):
            val = T[i, j]
            color = "white" if val > 0.5 * vmax else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)
    if title:
        ax.set_title(title)
    return ax


def plot_transition_diff(T_phasic: np.ndarray, T_tonic: np.ndarray,
                         *, ax=None,
                         title: str | None = None,
                         vlim: float = 0.2) -> "plt.Axes":
    if ax is None:
        _, ax = plt.subplots(figsize=(3.2, 3.0))
    diff = T_phasic - T_tonic
    im = ax.imshow(diff, cmap="RdBu_r", vmin=-vlim, vmax=vlim, origin="upper")
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(STATE_NAMES, rotation=30, fontsize=8)
    ax.set_yticklabels(STATE_NAMES, fontsize=8)
    ax.set_xlabel("next state"); ax.set_ylabel("current state")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{diff[i, j]:+.2f}", ha="center", va="center",
                    fontsize=7,
                    color="white" if abs(diff[i, j]) > 0.6 * vlim else "black")
    if title:
        ax.set_title(title)
    return ax


# ---------------------------------------------------------------------------
# PPC plots — hot colormap by user request
# ---------------------------------------------------------------------------

def plot_ppc_heatmap(ppc: np.ndarray,
                     frequencies: np.ndarray,
                     phase_centers_deg: np.ndarray,
                     *, ax=None,
                     title: str | None = None,
                     cmap: str = PPC_CMAP,
                     vmin: float | None = 0.0,
                     vmax: float | None = None,
                     n_levels: int = 14,
                     overlay_lines: bool = True,
                     line_levels: int = 5) -> "plt.Axes":
    """Filled-contour PPC plot (replacement for pcolormesh).

    ``n_levels`` is the number of colour bands in the filled contour.
    Set ``overlay_lines=False`` if you don't want thin black contour
    outlines drawn on top.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(3.2, 3.0))
    arr = np.asarray(ppc, dtype=float)
    if vmin is None:
        vmin = float(np.nanmin(arr))
    if vmax is None:
        vmax = float(np.nanmax(arr))
    if vmin == vmax:
        vmax = vmin + 1e-9
    levels = np.linspace(vmin, vmax, n_levels + 1)
    cs = ax.contourf(phase_centers_deg, frequencies, arr,
                     levels=levels, cmap=cmap, vmin=vmin, vmax=vmax,
                     extend="both")
    if overlay_lines:
        line_lev = np.linspace(vmin, vmax, line_levels + 2)[1:-1]
        ax.contour(phase_centers_deg, frequencies, arr,
                   levels=line_lev, colors="black", linewidths=0.4,
                   alpha=0.5)
    ax.set_xlabel("theta phase (deg)")
    ax.set_ylabel("frequency (Hz)")
    if title:
        ax.set_title(title)
    return ax


def plot_ppc_state_grid(ppc_per_state: np.ndarray,
                        frequencies: np.ndarray,
                        phase_centers_deg: np.ndarray,
                        *, suptitle: str | None = None,
                        vmin: float = 0.0,
                        vmax: float | None = None,
                        n_cycles_per_state: np.ndarray | None = None
                        ) -> "plt.Figure":
    """Grid: 1 x 4 PPC heatmaps for the four TG states."""
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.0), constrained_layout=True,
                              sharey=True)
    for s in range(4):
        extra = ""
        if n_cycles_per_state is not None:
            extra = f"\n(n_cycles = {int(n_cycles_per_state[s])})"
        plot_ppc_heatmap(ppc_per_state[s], frequencies, phase_centers_deg,
                         ax=axes[s], title=f"{STATE_NAMES[s]}{extra}",
                         vmin=vmin, vmax=vmax)
        if s > 0:
            axes[s].set_ylabel("")
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    return fig


def plot_ppc_per_freq(ppc_per_state_freq: np.ndarray,
                      frequencies: np.ndarray,
                      *, ax=None,
                      title: str | None = None,
                      ppc_per_state_freq_std: np.ndarray | None = None,
                      band_alpha: float = 0.22,
                      band_label: str = "±1 SD") -> "plt.Axes":
    """Plot PPC(f) averaged in the gravity phase window, four curves.

    If ``ppc_per_state_freq_std`` is provided (same shape as
    ``ppc_per_state_freq``), a shaded band of ±1 standard deviation is
    drawn around each state's mean curve.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 3.0))
    has_band = ppc_per_state_freq_std is not None
    for s in range(ppc_per_state_freq.shape[0]):
        mean = np.asarray(ppc_per_state_freq[s], dtype=float)
        ax.plot(frequencies, mean,
                color=STATE_COLORS[s], label=STATE_NAMES[s], lw=1.5)
        if has_band:
            sd = np.asarray(ppc_per_state_freq_std[s], dtype=float)
            lo = mean - sd
            hi = mean + sd
            ax.fill_between(frequencies, lo, hi,
                            color=STATE_COLORS[s], alpha=band_alpha,
                            linewidth=0)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PPC")
    if has_band:
        # one ghost entry so the band semantics are documented in the legend
        from matplotlib.patches import Patch
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Patch(facecolor="gray", alpha=band_alpha,
                              edgecolor="none", label=band_label))
        labels.append(band_label)
        ax.legend(handles, labels, fontsize=8, frameon=False)
    else:
        ax.legend(fontsize=8, frameon=False)
    if title:
        ax.set_title(title)
    return ax


def plot_phasic_vs_tonic_ppc(ppc_phasic: np.ndarray,
                             ppc_tonic: np.ndarray,
                             frequencies: np.ndarray,
                             phase_centers_deg: np.ndarray,
                             *,
                             vmin: float = 0.0,
                             vmax: float | None = None,
                             title_prefix: str = "PFC-HPC PPC") -> "plt.Figure":
    """4 cols × 3 rows: top = phasic, middle = tonic, bottom = difference."""
    fig, axes = plt.subplots(3, 4, figsize=(14, 8.4), constrained_layout=True,
                              sharey=True, sharex=True)
    vlim = vmax
    if vlim is None:
        vlim = float(np.nanpercentile(np.concatenate([ppc_phasic.ravel(),
                                                       ppc_tonic.ravel()]), 99))
    diff_lim = float(np.nanmax(np.abs(ppc_phasic - ppc_tonic)))
    for s in range(4):
        plot_ppc_heatmap(ppc_phasic[s], frequencies, phase_centers_deg,
                         ax=axes[0, s], title=f"{STATE_NAMES[s]} (phasic)",
                         vmin=vmin, vmax=vlim)
        plot_ppc_heatmap(ppc_tonic[s], frequencies, phase_centers_deg,
                         ax=axes[1, s], title=f"{STATE_NAMES[s]} (tonic)",
                         vmin=vmin, vmax=vlim)
        diff = ppc_phasic[s] - ppc_tonic[s]
        diff_levels = np.linspace(-diff_lim, diff_lim, 15)
        axes[2, s].contourf(phase_centers_deg, frequencies, diff,
                            levels=diff_levels, cmap="RdBu_r",
                            vmin=-diff_lim, vmax=diff_lim, extend="both")
        # zero contour line so the sign boundary is visible
        axes[2, s].contour(phase_centers_deg, frequencies, diff,
                           levels=[0.0], colors="black", linewidths=0.6,
                           alpha=0.7)
        axes[2, s].set_title(f"{STATE_NAMES[s]} (phasic − tonic)")
        axes[2, s].set_xlabel("theta phase (deg)")
        if s == 0:
            axes[2, s].set_ylabel("frequency (Hz)")
    fig.suptitle(title_prefix, fontsize=13)
    return fig
