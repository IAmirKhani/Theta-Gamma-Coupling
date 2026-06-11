"""Cross-region phase-amplitude coupling (PAC), EMD-based.

Bidirectional Tort modulation index between HPC theta phase and PFC
gamma amplitude (and vice versa), computed per phasic / tonic REM
interval and aggregated per rat.

Pipeline per interval
---------------------
1.  Mask-sift EMD on HPC LFP  ->  HPC IMFs + mean IFs.
2.  Mask-sift EMD on PFC LFP  ->  PFC IMFs + mean IFs.
3.  HPC theta IMF        ->  HPC theta phase  = angle(hilbert(theta_imf))
4.  PFC theta IMF        ->  PFC theta phase
5.  For each gamma band (slow 30-50, medium 60-120, fast 120-180):
        sum IMFs whose mean instantaneous frequency falls in the band,
        take |hilbert(sum)| = gamma amplitude envelope.
6.  Tort MI for each (phase source, amplitude target, gamma band) cell:
        MI = (log N_bins  -  H(P)) / log N_bins
        with P = normalised mean amplitude per phase bin, 18 bins.
7.  Phase-shift permutation null: random circular shift of phase,
        recompute MI; do this N_SHUFFLES times. z = (MI - mu_null) / std_null.

Per-substate aggregation
------------------------
Phase and amplitude time-series are *concatenated* across all intervals
of the same substate before MI is computed once on the concatenated data.
This avoids noisy single-interval estimates and aligns with the standard
Tort protocol of computing MI on a long continuous time series.

This module reuses ``src/utils.py::extract_imfs_by_pt_intervals`` to be
consistent with the rest of the project's EMD work.
"""

from __future__ import annotations
from .emd_cycles import (
    THETA_BAND_HZ,
    THETA_IMF_PREFER,
    choose_theta_imf_index,
    load_default_emd_config,
)
from utils import extract_imfs_by_pt_intervals  # type: ignore

import os
import sys
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

# Project src on path
_PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GAMMA_BANDS: dict[str, tuple[float, float]] = {
    "slow_gamma":   (25.0,  40.0),
    "medium_gamma": (40.0, 80.0),
    "fast_gamma":  (80.0, 140.0),
}
N_PHASE_BINS_PAC = 18
DEFAULT_N_SHUFFLES = 100
MIN_INTERVAL_SAMPLES = 500      # at 1 kHz this is 0.5 s. Since intervals are
# CONCATENATED before MI is computed, short
# intervals still contribute and we just skip
# pathologically tiny ones.

DIRECTION_NAMES = ("hpc_theta_to_pfc_gamma", "pfc_theta_to_hpc_gamma")
DIRECTION_NAMES_HPC_ONLY = ("hpc_theta_to_pfc_gamma", "hpc_theta_to_hpc_gamma")


def direction_names_for(phase_source: str) -> tuple[str, str]:
    """Return the two direction-key names used in ``RatPACResult.mi/null/z/pad``
    for a given ``phase_source``.

    When ``phase_source == 'hpc'`` HPC theta phase is the modulator for
    BOTH gamma targets (cross-region + within-HPC control). When
    ``phase_source == 'each_region'`` each region's own theta phase is
    used (the original Fujisawa & Buzsáki 2011 directional protocol).
    """
    if phase_source == "hpc":
        return DIRECTION_NAMES_HPC_ONLY
    if phase_source == "each_region":
        return DIRECTION_NAMES
    raise ValueError(
        f"phase_source must be 'hpc' or 'each_region', got {phase_source!r}"
    )


# ---------------------------------------------------------------------------
# Extracting phase and amplitude from EMD IMFs
# ---------------------------------------------------------------------------

def theta_phase_from_imfs(imfs: np.ndarray,
                          imf_freqs: np.ndarray,
                          theta_band: tuple[float, float] = THETA_BAND_HZ,
                          prefer_index: int = THETA_IMF_PREFER) -> np.ndarray:
    """Pick the theta IMF, return wrapped Hilbert phase in [-π, π]."""
    theta_idx = choose_theta_imf_index(imfs, imf_freqs,
                                       theta_band=theta_band,
                                       prefer_index=prefer_index)
    return np.angle(hilbert(imfs[:, theta_idx]))


def gamma_amplitude_from_imfs(imfs: np.ndarray,
                              imf_freqs: np.ndarray,
                              band: tuple[float, float]) -> np.ndarray | None:
    """Sum IMFs whose mean instantaneous frequency falls in ``band``, then
    return the Hilbert magnitude envelope of that sum.

    Returns ``None`` if no IMF mean frequency lies in the band. EMD only
    produces a handful of IMFs per segment, so for fixed Hz bands this
    is often the case; use :func:`gamma_amplitude_from_lfp` as the
    robust default and reserve this for pure-EMD experiments.
    """
    centers = np.asarray(imf_freqs, dtype=float).ravel()
    if centers.size != imfs.shape[1]:
        return None
    lo, hi = band
    in_band = (centers >= lo) & (centers <= hi)
    if not np.any(in_band):
        return None
    summed = imfs[:, in_band].sum(axis=1)
    return np.abs(hilbert(summed))


def gamma_amplitude_from_lfp(lfp: np.ndarray, fs: float,
                             band: tuple[float, float],
                             order: int = 4) -> np.ndarray:
    """Bandpass + Hilbert envelope (the standard Tort PAC recipe).

    ``order`` is the Butterworth order; we apply ``sosfiltfilt`` so the
    filter is zero-phase. Works for any Hz band and any LFP length
    > a few periods of the lowest cutoff.
    """
    lo, hi = band
    nyq = 0.5 * fs
    lo_n = max(lo / nyq, 1e-6)
    hi_n = min(hi / nyq, 1.0 - 1e-6)
    sos = butter(order, [lo_n, hi_n], btype="bandpass", output="sos")
    filt = sosfiltfilt(sos, np.asarray(lfp, dtype=float))
    return np.abs(hilbert(filt))


# ---------------------------------------------------------------------------
# Tort modulation index + phase-shift null
# ---------------------------------------------------------------------------

def tort_modulation_index(phase: np.ndarray,
                          amplitude: np.ndarray,
                          n_bins: int = N_PHASE_BINS_PAC
                          ) -> tuple[float, np.ndarray]:
    """Tort et al. 2010 modulation index.

    Bin samples by ``phase`` into ``n_bins`` equal bins in [-π, π], take
    the mean ``amplitude`` per bin, normalise to a probability vector
    ``P`` summing to 1, then
        MI = (log N_bins  -  H(P)) / log N_bins
    where H is Shannon entropy in nats.

    Returns ``(MI, P)``. ``P`` has length ``n_bins``.
    """
    phase = np.asarray(phase, dtype=float)
    amplitude = np.asarray(amplitude, dtype=float)
    L = min(len(phase), len(amplitude))
    phase = phase[:L]
    amplitude = amplitude[:L]
    if L < n_bins * 4:
        return float("nan"), np.full(n_bins, np.nan)

    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_idx = np.digitize(phase, edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    bin_amps = np.zeros(n_bins, dtype=float)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        sel = bin_idx == b
        if not np.any(sel):
            return float("nan"), np.full(n_bins, np.nan)
        bin_amps[b] = amplitude[sel].mean()
        counts[b] = int(sel.sum())
    total = bin_amps.sum()
    if total <= 0:
        return float("nan"), bin_amps
    P = bin_amps / total
    P_safe = np.where(P > 0, P, 1e-300)
    H = -float(np.sum(P * np.log(P_safe)))
    log_n = float(np.log(n_bins))
    MI = (log_n - H) / log_n
    return float(MI), P


def phase_shift_null_mi(phase: np.ndarray,
                        amplitude: np.ndarray,
                        n_shuffles: int = DEFAULT_N_SHUFFLES,
                        n_bins: int = N_PHASE_BINS_PAC,
                        min_shift_frac: float = 0.05,
                        rng: np.random.Generator | None = None) -> np.ndarray:
    """Return ``n_shuffles`` null MI values from random circular shifts of phase.

    Each shuffle: roll the phase array by a random offset in
    ``[min_shift_frac · N, (1 − min_shift_frac) · N]`` samples and
    recompute MI. The amplitude time-series is untouched.

    Returns array of shape ``(n_shuffles,)``.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    phase = np.asarray(phase, dtype=float)
    amplitude = np.asarray(amplitude, dtype=float)
    n = min(len(phase), len(amplitude))
    if n < n_bins * 4:
        return np.full(n_shuffles, np.nan)
    min_shift = max(1, int(round(min_shift_frac * n)))
    max_shift = max(min_shift + 1, n - min_shift)
    nulls = np.full(n_shuffles, np.nan, dtype=float)
    for i in range(n_shuffles):
        shift = int(rng.integers(min_shift, max_shift))
        rolled = np.roll(phase[:n], shift)
        mi, _ = tort_modulation_index(rolled, amplitude[:n], n_bins=n_bins)
        nulls[i] = mi
    return nulls


# ---------------------------------------------------------------------------
# Per-substate PAC for one rat (segments aggregated)
# ---------------------------------------------------------------------------

@dataclass
class RatPACResult:
    """Per-rat PAC summary.

    All MI / null / pad / z dicts are nested as
    ``[substate]['hpc_theta_to_pfc_gamma' | 'pfc_theta_to_hpc_gamma'][band_name]``.
    """
    rat_id: int
    group: str
    n_phase_bins: int
    n_shuffles: int
    n_samples_per_substate: dict[str, int]
    n_intervals_per_substate: dict[str, int]
    mi: dict
    null: dict
    z: dict
    # phase-amplitude distributions (length n_phase_bins)
    pad: dict
    gamma_bands: dict[str, tuple[float, float]]


def _aggregate_imfs_into_phase_and_amplitudes(
    imfs_list: Sequence[np.ndarray],
    freqs_list: Sequence[np.ndarray],
    theta_band: tuple[float, float] = THETA_BAND_HZ,
    prefer_index: int = THETA_IMF_PREFER,
    gamma_bands: dict[str, tuple[float, float]] = GAMMA_BANDS,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[int]] | tuple[None, None, None]:
    """Concatenate theta-phase + per-band gamma envelopes across intervals.

    Returns ``(phase_concat, {band_name: amp_concat}, segment_lengths)``
    or ``(None, None, None)`` if no usable data.
    """
    phase_chunks: list[np.ndarray] = []
    amp_chunks: dict[str, list[np.ndarray]] = {b: [] for b in gamma_bands}
    seg_lengths: list[int] = []
    for imfs, freqs in zip(imfs_list, freqs_list):
        if imfs is None or imfs.size == 0:
            continue
        if imfs.shape[0] < MIN_INTERVAL_SAMPLES:
            continue
        phase = theta_phase_from_imfs(imfs, freqs,
                                      theta_band=theta_band,
                                      prefer_index=prefer_index)
        per_band = {b: gamma_amplitude_from_imfs(imfs, freqs, band)
                    for b, band in gamma_bands.items()}
        if any(v is None for v in per_band.values()):
            continue
        phase_chunks.append(phase)
        for b in gamma_bands:
            amp_chunks[b].append(per_band[b])
        seg_lengths.append(imfs.shape[0])
    if not phase_chunks:
        return None, None, None
    phase_all = np.concatenate(phase_chunks)
    amp_all = {b: np.concatenate(amp_chunks[b]) for b in gamma_bands}
    return phase_all, amp_all, seg_lengths


def _compute_one_direction(phase: np.ndarray,
                           amp_by_band: dict[str, np.ndarray],
                           n_bins: int,
                           n_shuffles: int,
                           rng: np.random.Generator
                           ) -> tuple[dict, dict, dict, dict]:
    """Internal helper. Compute MI / null / z / pad per band for one direction."""
    mi_per_band, null_per_band = {}, {}
    z_per_band, pad_per_band = {}, {}
    for band_name, amp in amp_by_band.items():
        mi, pad = tort_modulation_index(phase, amp, n_bins=n_bins)
        null = phase_shift_null_mi(phase, amp, n_shuffles=n_shuffles,
                                   n_bins=n_bins, rng=rng)
        mu = float(np.nanmean(null)) if np.any(np.isfinite(null)) else np.nan
        sd = float(np.nanstd(null, ddof=1)) if np.any(
            np.isfinite(null)) else np.nan
        z = float("nan") if (sd is None or not np.isfinite(
            sd) or sd == 0) else (mi - mu) / sd
        mi_per_band[band_name] = mi
        null_per_band[band_name] = null
        z_per_band[band_name] = z
        pad_per_band[band_name] = pad
    return mi_per_band, null_per_band, z_per_band, pad_per_band


def pac_for_rat(rat_id: int,
                group: str,
                cfg=None,
                base_path: str | None = None,
                conditions: Sequence[str] | None = None,
                gamma_bands: dict[str, tuple[float, float]] = GAMMA_BANDS,
                n_phase_bins: int = N_PHASE_BINS_PAC,
                n_shuffles: int = DEFAULT_N_SHUFFLES,
                theta_band: tuple[float, float] = THETA_BAND_HZ,
                prefer_theta_imf_index: int = THETA_IMF_PREFER,
                random_state: int = 0,
                phase_source: str = "hpc",
                verbose: bool = True) -> RatPACResult | None:
    """End-to-end PAC computation for one rat.

    Loops over the rat's sessions; for each session runs the substate
    extraction and per-interval EMD on both HPC and PFC; concatenates
    theta-phase and gamma-amplitude across all phasic (or tonic)
    intervals; computes Tort MI + phase-shift null per gamma band per
    direction; returns one ``RatPACResult``.

    Parameters
    ----------
    phase_source : {'hpc', 'each_region'}, default 'hpc'
        Which theta phase to use as the modulator.
          * 'hpc' (recommended): HPC theta IMF phase is used for BOTH
            gamma targets. The two output directions are then
            ``'hpc_theta_to_pfc_gamma'`` (cross-region) and
            ``'hpc_theta_to_hpc_gamma'`` (within-HPC control). This is
            the safer default when PFC theta is weak or noisy.
          * 'each_region': each region's own theta IMF phase is used.
            The two output directions are
            ``'hpc_theta_to_pfc_gamma'`` and
            ``'pfc_theta_to_hpc_gamma'`` (original Fujisawa-style
            directional protocol).
    """
    dir_a, dir_b = direction_names_for(phase_source)
    from .data_pipeline import (
        BASE_PATH, list_rat_sessions,
    )
    from utils import (  # type: ignore
        get_data, extract_pt_intervals,
    )
    if cfg is None:
        cfg = load_default_emd_config()
    if base_path is None:
        base_path = BASE_PATH
    rng = np.random.default_rng(random_state)

    sessions = list_rat_sessions(rat_id, base_path=base_path,
                                 conditions=conditions)
    if verbose:
        print(f"[rat {rat_id}] {len(sessions)} sessions")

    # buffers: per-substate concatenations across all sessions
    buf = {
        sub: dict(
            hpc_phase=[], pfc_phase=[],
            hpc_amp={b: [] for b in gamma_bands},
            pfc_amp={b: [] for b in gamma_bands},
            n_intervals=0,
        )
        for sub in ("phasic", "tonic")
    }

    for sid, sess in enumerate(sessions):
        try:
            hpc_lfp, hypno, fs_hpc = get_data(sess["hpc_path"],
                                              sess["state_path"], type="hpc")
            pfc_lfp, _, _ = get_data(sess["pfc_path"],
                                     sess["state_path"], type="pfc")
        except Exception as ex:
            if verbose:
                print(f"  [rat {rat_id}] cannot load "
                      f"{sess['folder']}/{sess['sub']}: {ex}")
            continue
        fs = int(fs_hpc)
        L = min(len(hpc_lfp), len(pfc_lfp))
        hpc_lfp = np.asarray(hpc_lfp[:L], dtype=float)
        pfc_lfp = np.asarray(pfc_lfp[:L], dtype=float)
        try:
            phasic_iv, tonic_iv, _ = extract_pt_intervals(
                hpc_lfp, hypno, fs=fs)
        except Exception:
            continue

        for iv, name in [(phasic_iv, "phasic"), (tonic_iv, "tonic")]:
            if iv is None or len(iv) == 0:
                continue
            try:
                hpc_imfs_list, hpc_freqs_list, hpc_seg_list = extract_imfs_by_pt_intervals(
                    hpc_lfp, fs, iv, cfg, return_imfs_freqs=True,
                )
                pfc_imfs_list, pfc_freqs_list, pfc_seg_list = extract_imfs_by_pt_intervals(
                    pfc_lfp, fs, iv, cfg, return_imfs_freqs=True,
                )
            except Exception as ex:
                if verbose:
                    print(f"  [rat {rat_id}] EMD failed ({name}): {ex}")
                continue

            n_used = 0
            for hi, hf, hpc_seg, pi, pf, pfc_seg in zip(
                hpc_imfs_list, hpc_freqs_list, hpc_seg_list,
                pfc_imfs_list, pfc_freqs_list, pfc_seg_list,
            ):
                if hi is None or pi is None:
                    continue
                if hi.shape[0] < MIN_INTERVAL_SAMPLES:
                    continue
                # align lengths across HPC IMFs, PFC IMFs, HPC raw, PFC raw
                L_iv = min(hi.shape[0], pi.shape[0],
                           len(hpc_seg), len(pfc_seg))
                if L_iv < MIN_INTERVAL_SAMPLES:
                    continue
                hi = hi[:L_iv]
                pi = pi[:L_iv]
                hpc_seg = np.asarray(hpc_seg[:L_iv], dtype=float)
                pfc_seg = np.asarray(pfc_seg[:L_iv], dtype=float)
                # phases (still EMD-derived)
                hpc_phase = theta_phase_from_imfs(hi, hf, theta_band=theta_band,
                                                  prefer_index=prefer_theta_imf_index)
                pfc_phase = theta_phase_from_imfs(pi, pf, theta_band=theta_band,
                                                  prefer_index=prefer_theta_imf_index)
                # amplitudes via Butterworth bandpass on the raw segment.
                # EMD IMFs at fixed Hz bands are unreliable (only ~5–6 IMFs per
                # segment, frequencies vary), so the canonical Tort recipe
                # (bandpass -> |Hilbert|) is used for the amplitude side.
                hpc_amp_b = {b: gamma_amplitude_from_lfp(hpc_seg, fs, gamma_bands[b])
                             for b in gamma_bands}
                pfc_amp_b = {b: gamma_amplitude_from_lfp(pfc_seg, fs, gamma_bands[b])
                             for b in gamma_bands}
                buf[name]["hpc_phase"].append(hpc_phase)
                buf[name]["pfc_phase"].append(pfc_phase)
                for b in gamma_bands:
                    buf[name]["hpc_amp"][b].append(hpc_amp_b[b])
                    buf[name]["pfc_amp"][b].append(pfc_amp_b[b])
                n_used += 1
            buf[name]["n_intervals"] += n_used

    # Aggregate and compute MI per substate
    mi_all, null_all, z_all, pad_all = {}, {}, {}, {}
    n_samples_per_sub: dict[str, int] = {}
    n_intervals_per_sub: dict[str, int] = {}

    for name in ("phasic", "tonic"):
        b = buf[name]
        n_intervals_per_sub[name] = b["n_intervals"]
        if not b["hpc_phase"]:
            mi_all[name] = None
            null_all[name] = None
            z_all[name] = None
            pad_all[name] = None
            n_samples_per_sub[name] = 0
            continue
        hpc_phase = np.concatenate(b["hpc_phase"])
        pfc_phase = np.concatenate(b["pfc_phase"])
        hpc_amp = {bb: np.concatenate(b["hpc_amp"][bb]) for bb in gamma_bands}
        pfc_amp = {bb: np.concatenate(b["pfc_amp"][bb]) for bb in gamma_bands}
        n_samples_per_sub[name] = int(len(hpc_phase))

        # Direction A: HPC theta → PFC gamma (cross-region; always
        # computed identically regardless of phase_source).
        mi_a, null_a, z_a, pad_a = _compute_one_direction(
            hpc_phase, pfc_amp, n_phase_bins, n_shuffles, rng,
        )
        # Direction B: depends on phase_source.
        #   'hpc'         -> HPC theta → HPC gamma (within-HPC control)
        #   'each_region' -> PFC theta → HPC gamma (Fujisawa-style reverse)
        if phase_source == "hpc":
            mi_b, null_b, z_b, pad_b = _compute_one_direction(
                hpc_phase, hpc_amp, n_phase_bins, n_shuffles, rng,
            )
        else:
            mi_b, null_b, z_b, pad_b = _compute_one_direction(
                pfc_phase, hpc_amp, n_phase_bins, n_shuffles, rng,
            )

        mi_all[name] = {dir_a: mi_a,  dir_b: mi_b}
        null_all[name] = {dir_a: null_a, dir_b: null_b}
        z_all[name] = {dir_a: z_a,  dir_b: z_b}
        pad_all[name] = {dir_a: pad_a, dir_b: pad_b}
        if verbose:
            mi_str = ", ".join(
                f"{bb}={mi_a[bb]:.4f}" for bb in gamma_bands
            )
            print(f"  [rat {rat_id} | {name}] {dir_a} MI: {mi_str}")

    return RatPACResult(
        rat_id=rat_id, group=group,
        n_phase_bins=n_phase_bins, n_shuffles=n_shuffles,
        n_samples_per_substate=n_samples_per_sub,
        n_intervals_per_substate=n_intervals_per_sub,
        mi=mi_all, null=null_all, z=z_all, pad=pad_all,
        gamma_bands=dict(gamma_bands),
    )
