"""Frequency-phase power (FPP) construction per single theta cycle.

Faithful re-implementation of the wavelet+theta-phase pipeline from
Zhang et al. 2019, eLife, "Sub-second dynamics of theta-gamma coupling in
hippocampal CA1" (Materials and Methods: "Wavelet spectrum normalized by
theta phase").

Pipeline per LFP segment:
    1. Downsample to ANALYSIS_FS = 625 Hz.
    2. Compute complex Morlet CWT across FREQUENCIES (20:2:180, 81 freqs).
    3. Take magnitude squared, then sequential time-then-frequency boxcar
       smoothing: 11 samples in time (= +/- 8 ms at 625 Hz) and 3 samples
       in frequency (= +/- 2 Hz at 2-Hz spacing). Take sqrt to recover
       a smoothed amplitude-like quantity (matches Zhang's MATLAB code).
    4. z-score each frequency row across time.
    5. Bandpass-filter the LFP in theta band (5-10 Hz), unwrap the Hilbert
       phase, and detect theta cycles as 2-pi crossings (with monotonicity
       guaranteed by unwrapping) and instantaneous-frequency filter
       (5-12 Hz).
    6. For each cycle, average the smoothed/zscored power within each of
       20 equal theta-phase bins to obtain the cycle's FPP, an 81 x 20
       matrix.

The resulting FPP per cycle is the feature that is clustered downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

import numpy as np
from scipy.signal import butter, fftconvolve, hilbert, resample_poly, sosfiltfilt


ANALYSIS_FS = 625.0
FREQUENCIES = np.arange(25.0, 182.0, 2.0)            # 81 freqs, 2 Hz spacing
N_PHASE_BINS = 20
PHASE_EDGES_DEG = np.linspace(0.0, 360.0, N_PHASE_BINS + 1)
PHASE_CENTERS_DEG = 0.5 * (PHASE_EDGES_DEG[:-1] + PHASE_EDGES_DEG[1:])
PHASE_CENTERS_RAD = np.deg2rad(PHASE_CENTERS_DEG)

THETA_BAND = (5.0, 12.0)
THETA_FILTER_ORDER = 4
THETA_CYCLE_FREQ_RANGE = (5.0, 12.0)

MORLET_OMEGA0 = 5.0
MORLET_N_SIGMA = 4.0

TIME_SMOOTH_MS = 8.0        # +/- 8 ms boxcar (Zhang's ntw = 11 pts at 625 Hz)
# +/- 2 Hz boxcar (Zhang's nsw = 3 pts at 2 Hz spacing)
FREQ_SMOOTH_HZ = 2.0


# ---------------------------------------------------------------------------
# Resampling and theta-phase utilities
# ---------------------------------------------------------------------------

def resample_to_analysis_fs(lfp: np.ndarray, fs: float,
                            analysis_fs: float = ANALYSIS_FS) -> tuple[np.ndarray, float]:
    """Polyphase-resample a 1-D LFP to ``analysis_fs`` (625 Hz by default)."""
    lfp = np.asarray(lfp, dtype=float)
    if analysis_fs is None or int(round(fs)) == int(round(analysis_fs)):
        return lfp, float(fs)
    frac = Fraction(float(analysis_fs) / float(fs)).limit_denominator(1000)
    out = resample_poly(lfp, frac.numerator, frac.denominator)
    return out, float(analysis_fs)


def theta_unwrapped_phase(lfp: np.ndarray, fs: float,
                          theta_band: tuple[float, float] = THETA_BAND,
                          order: int = THETA_FILTER_ORDER) -> np.ndarray:
    """Bandpass-filter to theta then return the *unwrapped* Hilbert phase.

    Unwrapping makes monotonicity automatic (Zhang requires strictly
    increasing phase on each cycle; ``np.unwrap`` enforces this).
    """
    lo, hi = theta_band
    if len(lfp) < int(fs):
        raise ValueError("LFP segment too short for theta-phase estimation")
    sos = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    theta = sosfiltfilt(sos, np.asarray(lfp, dtype=float))
    phase = np.unwrap(np.angle(hilbert(theta)))
    if np.nanmedian(np.diff(phase)) < 0:
        phase = -phase
    return phase


def extract_theta_cycles(theta_phase: np.ndarray, fs: float,
                         freq_range: tuple[float,
                                           float] = THETA_CYCLE_FREQ_RANGE
                         ) -> list[tuple[int, int]]:
    """Return list of ``(start, end)`` sample indices for theta cycles.

    A cycle is one full sweep between two successive 2-pi crossings of the
    unwrapped theta phase. Cycles with instantaneous frequency outside
    ``freq_range`` are rejected (matches the project convention).
    """
    two_pi = 2.0 * np.pi
    first_level = np.ceil(theta_phase[0] / two_pi) * two_pi
    last_level = np.floor(theta_phase[-1] / two_pi) * two_pi
    if last_level <= first_level:
        return []

    levels = np.arange(first_level, last_level + 0.5 * two_pi, two_pi)
    bounds = []
    for level in levels:
        hits = np.where((theta_phase[:-1] < level)
                        & (theta_phase[1:] >= level))[0]
        if len(hits):
            bounds.append(int(hits[0] + 1))
    bounds = np.asarray(bounds, dtype=int)
    bounds = np.unique(bounds)
    if len(bounds) < 2:
        return []

    cycles: list[tuple[int, int]] = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        if e <= s + 2:
            continue
        duration = (e - s) / float(fs)
        if duration <= 0:
            continue
        inst_freq = 1.0 / duration
        if freq_range is not None:
            lo, hi = freq_range
            if not (lo <= inst_freq <= hi):
                continue
        cycles.append((int(s), int(e)))
    return cycles


# ---------------------------------------------------------------------------
# Morlet CWT + smoothing
# ---------------------------------------------------------------------------

def complex_morlet_kernel(freq: float, fs: float,
                          omega0: float = MORLET_OMEGA0,
                          n_sigma: float = MORLET_N_SIGMA) -> np.ndarray:
    """Unit-energy complex Morlet kernel for a single frequency."""
    sigma_t = omega0 / (2.0 * np.pi * freq)
    half_width = int(np.ceil(n_sigma * sigma_t * fs))
    half_width = max(half_width, 2)
    t = np.arange(-half_width, half_width + 1) / fs
    wavelet = np.exp(2j * np.pi * freq * t) * \
        np.exp(-(t ** 2) / (2.0 * sigma_t ** 2))
    wavelet = wavelet - wavelet.mean()
    wavelet = wavelet / np.sqrt(np.sum(np.abs(wavelet) ** 2))
    return wavelet


def morlet_cwt(signal: np.ndarray, fs: float,
               frequencies: np.ndarray = FREQUENCIES,
               omega0: float = MORLET_OMEGA0,
               n_sigma: float = MORLET_N_SIGMA) -> np.ndarray:
    """Complex Morlet CWT, shape ``(n_freq, n_time)``."""
    x = np.asarray(signal, dtype=float)
    x = x - np.nanmean(x)
    out = np.empty((len(frequencies), len(x)), dtype=np.complex128)
    for fi, freq in enumerate(frequencies):
        kernel = complex_morlet_kernel(
            freq, fs, omega0=omega0, n_sigma=n_sigma)
        out[fi] = fftconvolve(x, kernel.conj()[::-1], mode="same")
    return out


def _boxcar_smooth_1d(arr: np.ndarray, window: int, axis: int) -> np.ndarray:
    """Same-length boxcar smoothing along one axis (uniform kernel)."""
    if window <= 1:
        return arr
    kernel_shape = [1] * arr.ndim
    kernel_shape[axis] = window
    kernel = np.ones(window, dtype=float) / window
    kernel = kernel.reshape(kernel_shape)
    return fftconvolve(arr, kernel, mode="same", axes=axis)


def smoothed_zscored_power(lfp: np.ndarray, fs: float,
                           frequencies: np.ndarray = FREQUENCIES,
                           omega0: float = MORLET_OMEGA0,
                           n_sigma: float = MORLET_N_SIGMA,
                           time_smooth_ms: float = TIME_SMOOTH_MS,
                           freq_smooth_hz: float = FREQ_SMOOTH_HZ) -> np.ndarray:
    """Compute the smoothed, z-scored wavelet amplitude used as FPP input.

    Returns array of shape ``(n_freq, n_time)`` at the sampling rate ``fs``
    (which is the ANALYSIS_FS in this project).

    Steps (mirrors Zhang's SpecThetaExtract.m):
      1. Magnitude-squared of complex Morlet CWT.
      2. Boxcar smoothing in time (window = round(2 * time_smooth_ms / 1000 * fs) + 1).
      3. Boxcar smoothing in frequency (window = round(2 * freq_smooth_hz / df) + 1)
         where ``df`` is the frequency spacing of ``frequencies``.
      4. sqrt to recover amplitude-like quantity.
      5. z-score across time per frequency row.
    """
    W = morlet_cwt(lfp, fs, frequencies=frequencies,
                   omega0=omega0, n_sigma=n_sigma)
    power = np.abs(W) ** 2

    half_t = int(round(time_smooth_ms / 1000.0 * fs))
    time_window = 2 * half_t + 1
    df = float(np.median(np.diff(frequencies))) if len(
        frequencies) > 1 else 1.0
    half_f = int(round(freq_smooth_hz / df))
    freq_window = 2 * half_f + 1

    power = _boxcar_smooth_1d(power, time_window, axis=1)
    power = _boxcar_smooth_1d(power, freq_window, axis=0)
    power = np.sqrt(np.clip(power, 0.0, None))

    mu = power.mean(axis=1, keepdims=True)
    sd = power.std(axis=1, ddof=0, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (power - mu) / sd


# ---------------------------------------------------------------------------
# FPP per cycle (81 x 20)
# ---------------------------------------------------------------------------

def cycle_fpp(power_tf: np.ndarray, theta_phase: np.ndarray,
              start: int, end: int,
              n_phase_bins: int = N_PHASE_BINS,
              phase_is_wrapped: bool = False) -> np.ndarray | None:
    """Average wavelet power across 20 equal theta-phase bins for one cycle.

    ``power_tf`` is the (n_freq, n_time) smoothed/zscored matrix. The
    cycle is the half-open range ``[start, end)``.

    If ``phase_is_wrapped`` is False (default), the input ``theta_phase``
    is assumed to be **monotonically increasing** across the cycle and we
    simply compute ``theta_phase[start:end] - theta_phase[start]`` to get
    elapsed phase in radians, then bin into 20 equal bins in [0, 2π].
    This is the behaviour expected by the Butterworth+Hilbert pipeline,
    where ``theta_phase`` is the unwrapped phase.

    If ``phase_is_wrapped`` is True, the input is the wrapped phase
    (e.g. ``np.angle(hilbert(theta_imf))`` from the EMD pipeline). The
    cycle's segment is locally unwrapped here. This is robust to the
    cycle boundaries not falling exactly at the ±π wrap point.

    Returns the ``(n_freq, n_phase_bins)`` matrix or ``None`` if any bin
    is empty (cycle too short for ``n_phase_bins``).
    """
    seg_phase = theta_phase[start:end]
    if phase_is_wrapped:
        seg_phase = np.unwrap(seg_phase)
    rel = seg_phase - seg_phase[0]
    rel_deg = np.degrees(rel)
    # numerical headroom so that 360.0 falls into the last bin
    rel_deg = np.clip(rel_deg, 0.0, np.nextafter(360.0, 0.0))
    edges = np.linspace(0.0, 360.0, n_phase_bins + 1)

    n_freq = power_tf.shape[0]
    fpp = np.empty((n_freq, n_phase_bins), dtype=float)
    segment = power_tf[:, start:end]
    for bi, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (rel_deg >= lo) & (rel_deg < hi)
        if not np.any(mask):
            return None
        fpp[:, bi] = segment[:, mask].mean(axis=1)
    return fpp


@dataclass
class FPPCollection:
    """Container for FPPs and the cycle metadata behind them."""
    fpps: np.ndarray            # (n_cycles, n_freq, n_phase_bins)
    cycle_bounds: np.ndarray    # (n_cycles, 2) sample indices in analysis_fs
    # (n_cycles,) absolute seconds from segment start
    cycle_starts_sec: np.ndarray
    frequencies: np.ndarray
    phase_centers_deg: np.ndarray
    fs: float                   # analysis fs
    segment_meta: dict | None = None  # arbitrary metadata, e.g. rat/session/condition

    def __len__(self) -> int:
        return self.fpps.shape[0]

    def flat(self) -> np.ndarray:
        """Flatten each FPP to a vector (n_cycles, n_freq * n_phase_bins)."""
        n = self.fpps.shape[0]
        return self.fpps.reshape(n, -1)


def fpps_from_lfp_segment(lfp: np.ndarray, fs: float,
                          analysis_fs: float = ANALYSIS_FS,
                          frequencies: np.ndarray = FREQUENCIES,
                          n_phase_bins: int = N_PHASE_BINS,
                          theta_band: tuple[float, float] = THETA_BAND,
                          theta_cycle_freq_range: tuple[float,
                                                        float] = THETA_CYCLE_FREQ_RANGE,
                          time_smooth_ms: float = TIME_SMOOTH_MS,
                          freq_smooth_hz: float = FREQ_SMOOTH_HZ,
                          segment_offset_sec: float = 0.0,
                          segment_meta: dict | None = None) -> FPPCollection:
    """Full FPP pipeline for one contiguous LFP segment.

    The segment must be long enough to estimate at least one theta cycle.
    Returns an FPPCollection. The ``cycle_starts_sec`` field is in absolute
    seconds (offset + cycle start / analysis_fs) so cycles from many
    segments can be concatenated and still be aligned to behavioral state
    timestamps.
    """
    lfp_rs, fs_a = resample_to_analysis_fs(lfp, fs, analysis_fs=analysis_fs)
    if len(lfp_rs) < int(2 * fs_a):
        return FPPCollection(
            fpps=np.empty((0, len(frequencies), n_phase_bins), dtype=float),
            cycle_bounds=np.empty((0, 2), dtype=int),
            cycle_starts_sec=np.empty((0,), dtype=float),
            frequencies=np.asarray(frequencies),
            phase_centers_deg=PHASE_CENTERS_DEG.copy(),
            fs=fs_a,
            segment_meta=segment_meta,
        )

    phase = theta_unwrapped_phase(lfp_rs, fs_a, theta_band=theta_band)
    cycles = extract_theta_cycles(
        phase, fs_a, freq_range=theta_cycle_freq_range)
    if not cycles:
        return FPPCollection(
            fpps=np.empty((0, len(frequencies), n_phase_bins), dtype=float),
            cycle_bounds=np.empty((0, 2), dtype=int),
            cycle_starts_sec=np.empty((0,), dtype=float),
            frequencies=np.asarray(frequencies),
            phase_centers_deg=PHASE_CENTERS_DEG.copy(),
            fs=fs_a,
            segment_meta=segment_meta,
        )

    power_tf = smoothed_zscored_power(
        lfp_rs, fs_a, frequencies=frequencies,
        time_smooth_ms=time_smooth_ms, freq_smooth_hz=freq_smooth_hz,
    )

    fpps: list[np.ndarray] = []
    keep_bounds: list[tuple[int, int]] = []
    for s, e in cycles:
        fpp = cycle_fpp(power_tf, phase, s, e, n_phase_bins=n_phase_bins)
        if fpp is None or not np.all(np.isfinite(fpp)):
            continue
        fpps.append(fpp)
        keep_bounds.append((s, e))

    if not fpps:
        return FPPCollection(
            fpps=np.empty((0, len(frequencies), n_phase_bins), dtype=float),
            cycle_bounds=np.empty((0, 2), dtype=int),
            cycle_starts_sec=np.empty((0,), dtype=float),
            frequencies=np.asarray(frequencies),
            phase_centers_deg=PHASE_CENTERS_DEG.copy(),
            fs=fs_a,
            segment_meta=segment_meta,
        )

    fpps_arr = np.stack(fpps, axis=0)
    bounds_arr = np.asarray(keep_bounds, dtype=int)
    starts_sec = segment_offset_sec + bounds_arr[:, 0] / fs_a

    return FPPCollection(
        fpps=fpps_arr,
        cycle_bounds=bounds_arr,
        cycle_starts_sec=starts_sec,
        frequencies=np.asarray(frequencies),
        phase_centers_deg=PHASE_CENTERS_DEG.copy(),
        fs=fs_a,
        segment_meta=segment_meta,
    )


def fpps_from_segment_with_cycles(lfp: np.ndarray, fs: float,
                                  cycle_bounds: Sequence[tuple[int, int]],
                                  theta_phase_wrapped: np.ndarray,
                                  *,
                                  frequencies: np.ndarray = FREQUENCIES,
                                  n_phase_bins: int = N_PHASE_BINS,
                                  time_smooth_ms: float = TIME_SMOOTH_MS,
                                  freq_smooth_hz: float = FREQ_SMOOTH_HZ,
                                  segment_offset_sec: float = 0.0,
                                  segment_meta: dict | None = None
                                  ) -> FPPCollection:
    """EMD-driven FPP construction.

    Unlike :func:`fpps_from_lfp_segment` this skips the Butterworth +
    Hilbert + 2π-crossing path. The caller supplies:

      * ``lfp``: the LFP segment at native sampling rate ``fs`` (no
        resampling is done here; pass the same fs you used for EMD).
      * ``cycle_bounds``: half-open ``(start, end)`` sample indices into
        ``lfp`` for every kept theta cycle (e.g. from
        :mod:`emd_cycles`).
      * ``theta_phase_wrapped``: per-sample phase reference, in radians
        in ``[-π, π]``, the same length as ``lfp``. Typically
        ``np.angle(hilbert(theta_imf))``.

    The wavelet pipeline (Morlet CWT → smoothing → z-score per
    frequency) is identical to the original path, only the cycle
    boundaries and phase reference change.
    """
    lfp = np.asarray(lfp, dtype=float)
    cycle_bounds = list(cycle_bounds)
    # Minimum: one theta cycle worth of samples (fs/5 at 5 Hz) for the wavelet
    # to not be dominated by edge effects. EMD-supplied cycles make the 2-sec
    # guard unnecessary.
    min_samples = max(n_phase_bins + 1, int(fs / 5))
    if len(lfp) < min_samples or not cycle_bounds:
        return FPPCollection(
            fpps=np.empty((0, len(frequencies), n_phase_bins), dtype=float),
            cycle_bounds=np.empty((0, 2), dtype=int),
            cycle_starts_sec=np.empty((0,), dtype=float),
            frequencies=np.asarray(frequencies),
            phase_centers_deg=PHASE_CENTERS_DEG.copy(),
            fs=float(fs),
            segment_meta=segment_meta,
        )

    power_tf = smoothed_zscored_power(
        lfp, fs, frequencies=frequencies,
        time_smooth_ms=time_smooth_ms, freq_smooth_hz=freq_smooth_hz,
    )

    fpps: list[np.ndarray] = []
    keep_bounds: list[tuple[int, int]] = []
    for s, e in cycle_bounds:
        s = int(s)
        e = int(e)
        if e <= s + n_phase_bins:
            continue
        if e > power_tf.shape[1]:
            continue
        fpp = cycle_fpp(power_tf, theta_phase_wrapped, s, e,
                        n_phase_bins=n_phase_bins,
                        phase_is_wrapped=True)
        if fpp is None or not np.all(np.isfinite(fpp)):
            continue
        fpps.append(fpp)
        keep_bounds.append((s, e))

    if not fpps:
        return FPPCollection(
            fpps=np.empty((0, len(frequencies), n_phase_bins), dtype=float),
            cycle_bounds=np.empty((0, 2), dtype=int),
            cycle_starts_sec=np.empty((0,), dtype=float),
            frequencies=np.asarray(frequencies),
            phase_centers_deg=PHASE_CENTERS_DEG.copy(),
            fs=float(fs),
            segment_meta=segment_meta,
        )

    fpps_arr = np.stack(fpps, axis=0)
    bounds_arr = np.asarray(keep_bounds, dtype=int)
    starts_sec = segment_offset_sec + bounds_arr[:, 0] / float(fs)

    return FPPCollection(
        fpps=fpps_arr,
        cycle_bounds=bounds_arr,
        cycle_starts_sec=starts_sec,
        frequencies=np.asarray(frequencies),
        phase_centers_deg=PHASE_CENTERS_DEG.copy(),
        fs=float(fs),
        segment_meta=segment_meta,
    )


def concat_collections(collections: Sequence[FPPCollection]) -> FPPCollection:
    """Concatenate FPP collections (e.g. from many REM segments)."""
    collections = [c for c in collections if len(c) > 0]
    if not collections:
        raise ValueError("no non-empty collections to concatenate")
    fpps = np.concatenate([c.fpps for c in collections], axis=0)
    bounds = np.concatenate([c.cycle_bounds for c in collections], axis=0)
    starts = np.concatenate([c.cycle_starts_sec for c in collections], axis=0)
    return FPPCollection(
        fpps=fpps,
        cycle_bounds=bounds,
        cycle_starts_sec=starts,
        frequencies=collections[0].frequencies,
        phase_centers_deg=collections[0].phase_centers_deg,
        fs=collections[0].fs,
        segment_meta=None,
    )
