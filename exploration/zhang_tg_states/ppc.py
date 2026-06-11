"""LFP-LFP pair-wise phase consistency (PPC) for PFC-HPC, conditioned on
TG state and on REM substate (phasic vs tonic).

This is the wavelet-based PPC from Rohenkohl et al. 2018 / Zhang 2019:

  For each theta cycle k:
      W_k(f, theta)  = angle( CWT_x(f, t) * conj( CWT_y(f, t) ) )
                       averaged within each theta-phase bin of cycle k
  PPC(f, theta)      = ( |sum_k exp(i W_k(f, theta))|^2 - N ) / ( N (N-1) )

This is the unbiased V-statistic estimator of phase consistency (Vinck
et al. 2010 / 2012); it ranges from -1 (no locking) to +1 (perfectly
locked phase lag).

The full PPC matrix is 81 frequencies x 20 theta phases. Following
Zhang's Methods, we report PPC(f) by averaging across phases in a
window around the *gravity phase* of the CA1 (here HPC) gamma field:

      [gravity_phase - 7 * phase_std,   gravity_phase + 1 * phase_std]

i.e. shifted to "earlier" theta phases to capture leading input from
the source (PFC) before the HPC gamma peak. The window is wrapped on
the circle.

This module exposes the same Morlet wavelet from ``fpp.py`` so the FPP
clustering and PPC computation are exactly consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .fpp import (
    ANALYSIS_FS,
    FREQUENCIES,
    N_PHASE_BINS,
    PHASE_CENTERS_DEG,
    PHASE_CENTERS_RAD,
    THETA_BAND,
    THETA_CYCLE_FREQ_RANGE,
    extract_theta_cycles,
    morlet_cwt,
    resample_to_analysis_fs,
    theta_unwrapped_phase,
)


@dataclass
class CrossSpectrumResult:
    """Per-cycle binned cross-spectrum angles for one LFP segment."""
    angles: np.ndarray            # (n_cycles, n_freq, n_phase_bins)
    cycle_bounds: np.ndarray      # (n_cycles, 2) sample indices in analysis_fs
    cycle_starts_sec: np.ndarray
    fs: float

    def __len__(self) -> int:
        return self.angles.shape[0]


def _bin_cross_one_cycle(cross: np.ndarray,
                         theta_phase: np.ndarray,
                         start: int, end: int,
                         phase_edges_deg: np.ndarray,
                         phase_is_wrapped: bool = False) -> np.ndarray | None:
    """Average complex cross-spectrum within each phase bin, then take
    the argument. Returns angles or ``None`` if any bin is empty.

    If ``phase_is_wrapped`` is True the cycle's segment is unwrapped
    locally (matches the EMD-driven FPP path).
    """
    seg_phase = theta_phase[start:end]
    if phase_is_wrapped:
        seg_phase = np.unwrap(seg_phase)
    rel = seg_phase - seg_phase[0]
    rel_deg = np.clip(np.degrees(rel), 0.0, np.nextafter(360.0, 0.0))
    n_freq = cross.shape[0]
    n_bins = len(phase_edges_deg) - 1
    binned = np.empty((n_freq, n_bins), dtype=np.complex128)
    segment = cross[:, start:end]
    for bi, (lo, hi) in enumerate(zip(phase_edges_deg[:-1], phase_edges_deg[1:])):
        mask = (rel_deg >= lo) & (rel_deg < hi)
        if not np.any(mask):
            return None
        binned[:, bi] = segment[:, mask].mean(axis=1)
    return np.angle(binned)


def cross_spectrum_for_segment(source_lfp: np.ndarray,
                               target_lfp: np.ndarray,
                               fs: float,
                               analysis_fs: float = ANALYSIS_FS,
                               frequencies: np.ndarray = FREQUENCIES,
                               n_phase_bins: int = N_PHASE_BINS,
                               theta_band: tuple[float, float] = THETA_BAND,
                               theta_cycle_freq_range: tuple[float, float] = THETA_CYCLE_FREQ_RANGE,
                               segment_offset_sec: float = 0.0) -> CrossSpectrumResult:
    """End-to-end: resample, theta-cycle, CWT both, bin cross.

    Theta phase is taken from the *target* LFP (matches the FPP pipeline
    where HPC is the target). source and target need to be aligned in
    time at the input sampling rate ``fs``; they are resampled together.
    """
    # Resample paired
    L = min(len(source_lfp), len(target_lfp))
    s = np.asarray(source_lfp[:L], dtype=float)
    t = np.asarray(target_lfp[:L], dtype=float)
    s_rs, fs_a = resample_to_analysis_fs(s, fs, analysis_fs=analysis_fs)
    t_rs, _ = resample_to_analysis_fs(t, fs, analysis_fs=analysis_fs)
    L = min(len(s_rs), len(t_rs))
    s_rs = s_rs[:L]; t_rs = t_rs[:L]

    if L < int(2 * fs_a):
        return CrossSpectrumResult(
            angles=np.empty((0, len(frequencies), n_phase_bins), dtype=float),
            cycle_bounds=np.empty((0, 2), dtype=int),
            cycle_starts_sec=np.empty((0,), dtype=float),
            fs=fs_a,
        )

    phase = theta_unwrapped_phase(t_rs, fs_a, theta_band=theta_band)
    cycles = extract_theta_cycles(phase, fs_a, freq_range=theta_cycle_freq_range)
    if not cycles:
        return CrossSpectrumResult(
            angles=np.empty((0, len(frequencies), n_phase_bins), dtype=float),
            cycle_bounds=np.empty((0, 2), dtype=int),
            cycle_starts_sec=np.empty((0,), dtype=float),
            fs=fs_a,
        )

    Ws = morlet_cwt(s_rs, fs_a, frequencies=frequencies)
    Wt = morlet_cwt(t_rs, fs_a, frequencies=frequencies)
    cross = Ws * np.conj(Wt)

    edges = np.linspace(0.0, 360.0, n_phase_bins + 1)
    angles: list[np.ndarray] = []
    keep: list[tuple[int, int]] = []
    for s_idx, e_idx in cycles:
        a = _bin_cross_one_cycle(cross, phase, s_idx, e_idx, edges)
        if a is None or not np.all(np.isfinite(a)):
            continue
        angles.append(a)
        keep.append((s_idx, e_idx))

    if not angles:
        return CrossSpectrumResult(
            angles=np.empty((0, len(frequencies), n_phase_bins), dtype=float),
            cycle_bounds=np.empty((0, 2), dtype=int),
            cycle_starts_sec=np.empty((0,), dtype=float),
            fs=fs_a,
        )

    arr = np.stack(angles, axis=0)
    bounds = np.asarray(keep, dtype=int)
    starts = segment_offset_sec + bounds[:, 0] / fs_a
    return CrossSpectrumResult(angles=arr, cycle_bounds=bounds,
                               cycle_starts_sec=starts, fs=fs_a)


def cross_spectrum_with_cycles(source_lfp: np.ndarray,
                               target_lfp: np.ndarray,
                               fs: float,
                               cycle_bounds: list[tuple[int, int]],
                               theta_phase_wrapped: np.ndarray,
                               *,
                               frequencies: np.ndarray = FREQUENCIES,
                               n_phase_bins: int = N_PHASE_BINS,
                               segment_offset_sec: float = 0.0
                               ) -> CrossSpectrumResult:
    """EMD-driven cross-spectrum: take pre-computed cycle bounds and a
    wrapped per-sample phase reference (typically the theta IMF's
    ``angle(hilbert(...))``).

    No resampling is performed; both LFPs must already be at ``fs``.
    """
    L = min(len(source_lfp), len(target_lfp), len(theta_phase_wrapped))
    s = np.asarray(source_lfp[:L], dtype=float)
    t = np.asarray(target_lfp[:L], dtype=float)
    phase = np.asarray(theta_phase_wrapped[:L], dtype=float)
    min_samples = max(n_phase_bins + 1, int(fs / 5))
    if L < min_samples or not cycle_bounds:
        return CrossSpectrumResult(
            angles=np.empty((0, len(frequencies), n_phase_bins), dtype=float),
            cycle_bounds=np.empty((0, 2), dtype=int),
            cycle_starts_sec=np.empty((0,), dtype=float),
            fs=float(fs),
        )

    Ws = morlet_cwt(s, fs, frequencies=frequencies)
    Wt = morlet_cwt(t, fs, frequencies=frequencies)
    cross = Ws * np.conj(Wt)

    edges = np.linspace(0.0, 360.0, n_phase_bins + 1)
    angles: list[np.ndarray] = []
    keep: list[tuple[int, int]] = []
    for s_idx, e_idx in cycle_bounds:
        s_idx = int(s_idx); e_idx = int(e_idx)
        if e_idx <= s_idx + n_phase_bins or e_idx > L:
            continue
        a = _bin_cross_one_cycle(cross, phase, s_idx, e_idx, edges,
                                  phase_is_wrapped=True)
        if a is None or not np.all(np.isfinite(a)):
            continue
        angles.append(a)
        keep.append((s_idx, e_idx))

    if not angles:
        return CrossSpectrumResult(
            angles=np.empty((0, len(frequencies), n_phase_bins), dtype=float),
            cycle_bounds=np.empty((0, 2), dtype=int),
            cycle_starts_sec=np.empty((0,), dtype=float),
            fs=float(fs),
        )

    arr = np.stack(angles, axis=0)
    bounds = np.asarray(keep, dtype=int)
    starts = segment_offset_sec + bounds[:, 0] / float(fs)
    return CrossSpectrumResult(angles=arr, cycle_bounds=bounds,
                               cycle_starts_sec=starts, fs=float(fs))


# ---------------------------------------------------------------------------
# PPC across cycles
# ---------------------------------------------------------------------------

def ppc_from_angles(angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unbiased V-statistic PPC across cycles.

    ``angles`` is ``(n_cycles, n_freq, n_phase_bins)``.
    Returns ``(ppc, n_valid)`` with shape ``(n_freq, n_phase_bins)``.
    """
    angles = np.asarray(angles, dtype=float)
    if angles.ndim != 3:
        raise ValueError("angles must have shape (n_cycles, n_freq, n_phase)")
    valid = np.isfinite(angles)
    z = np.zeros_like(angles, dtype=np.complex128)
    z[valid] = np.exp(1j * angles[valid])
    n = valid.sum(axis=0)
    sum_z = z.sum(axis=0)
    ppc = np.full(n.shape, np.nan, dtype=float)
    ok = n > 1
    ppc[ok] = (np.abs(sum_z[ok]) ** 2 - n[ok]) / (n[ok] * (n[ok] - 1))
    return ppc, n


# ---------------------------------------------------------------------------
# Phase-window pooling around the gravity phase
# ---------------------------------------------------------------------------

def _wrap_phase_window(gravity_phase_rad: float,
                       phase_std_rad: float,
                       phase_centers_rad: np.ndarray,
                       pre_sd: float = 7.0,
                       post_sd: float = 1.0) -> np.ndarray:
    """Indices of phase bins inside
    ``[gravity - pre_sd*sd, gravity + post_sd*sd]`` on the circle.
    """
    if not np.isfinite(gravity_phase_rad) or not np.isfinite(phase_std_rad):
        return np.arange(len(phase_centers_rad))  # fall back: average all bins
    lo = gravity_phase_rad - pre_sd * phase_std_rad
    hi = gravity_phase_rad + post_sd * phase_std_rad
    # If the window already spans >= 2 pi, just use everything
    if (hi - lo) >= 2.0 * np.pi:
        return np.arange(len(phase_centers_rad))

    # Wrap by checking unsigned circular distance from gravity and the
    # signed distance "after gravity".
    # signed distance (target - gravity) wrapped to (-pi, pi]
    d = (phase_centers_rad - gravity_phase_rad + np.pi) % (2.0 * np.pi) - np.pi
    mask = (d >= -pre_sd * phase_std_rad) & (d <= post_sd * phase_std_rad)
    return np.where(mask)[0]


def ppc_phase_pooled(ppc: np.ndarray,
                     gravity_phase_rad: float,
                     phase_std_rad: float,
                     phase_centers_rad: np.ndarray = PHASE_CENTERS_RAD,
                     pre_sd: float = 7.0,
                     post_sd: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Average PPC across the gravity-centred phase window.

    Returns ``(ppc_per_freq, used_bin_indices)``.
    """
    idx = _wrap_phase_window(gravity_phase_rad, phase_std_rad,
                             phase_centers_rad, pre_sd=pre_sd, post_sd=post_sd)
    if len(idx) == 0:
        return np.full(ppc.shape[0], np.nan), idx
    return np.nanmean(ppc[:, idx], axis=1), idx


# ---------------------------------------------------------------------------
# State-conditioned PPC for one segment (already has labels per cycle)
# ---------------------------------------------------------------------------

@dataclass
class StatePPCResult:
    ppc_per_state: np.ndarray            # (n_states, n_freq, n_phase)
    n_cycles_per_state: np.ndarray       # (n_states,)
    n_per_freq_phase: np.ndarray         # (n_states, n_freq, n_phase) valid counts
    frequencies: np.ndarray
    phase_centers_deg: np.ndarray


def state_conditioned_ppc(angles: np.ndarray,
                          labels: np.ndarray,
                          n_states: int = 4,
                          frequencies: np.ndarray = FREQUENCIES,
                          phase_centers_deg: np.ndarray = PHASE_CENTERS_DEG
                          ) -> StatePPCResult:
    """Compute one PPC matrix per TG state."""
    n_states = int(n_states)
    n_freq = angles.shape[1]
    n_phase = angles.shape[2]
    out = np.full((n_states, n_freq, n_phase), np.nan, dtype=float)
    n_valid = np.zeros((n_states, n_freq, n_phase), dtype=int)
    n_cyc = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        sel = labels == s
        n_cyc[s] = int(sel.sum())
        if not np.any(sel):
            continue
        ppc, n = ppc_from_angles(angles[sel])
        out[s] = ppc
        n_valid[s] = n
    return StatePPCResult(
        ppc_per_state=out,
        n_cycles_per_state=n_cyc,
        n_per_freq_phase=n_valid,
        frequencies=np.asarray(frequencies),
        phase_centers_deg=np.asarray(phase_centers_deg),
    )


def split_by_membership(angles: np.ndarray,
                        labels: np.ndarray,
                        cycle_starts_sec: np.ndarray,
                        cycle_ends_sec: np.ndarray,
                        interval_starts: np.ndarray,
                        interval_ends: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(angles_inside, labels_inside)`` for cycles fully contained
    in any of the given intervals.
    """
    n = len(cycle_starts_sec)
    keep = np.zeros(n, dtype=bool)
    for i in range(n):
        cs = cycle_starts_sec[i]
        ce = cycle_ends_sec[i]
        if np.any((interval_starts <= cs) & (interval_ends >= ce)):
            keep[i] = True
    return angles[keep], labels[keep]
