"""EMD-based theta cycle extraction.

Wraps the existing project pipeline used in `exploration/RGS_pt_counter.py`:

    extract_imfs_by_pt_intervals(lfp, fs, IntervalSet, SiftConfig, return_imfs_freqs=True)
        -> (list_of_IMFs_per_segment,
            list_of_mean_freqs_per_segment,
            list_of_HPC_segments_per_segment)
    choose_theta_imf_index(IMF, imf_freqs, theta_band=(5,12), prefer_index=5)
    get_cycle_data(theta_imf, fs)
        -> dict with 'IP', 'IF', 'IA', 'cycles' (emd.cycles.Cycles)
    get_cycles_with_conditions(cycles_obj, ['is_good==1', 'duration_samples<200',
                                            'duration_samples>83.3', 'max_amp>amp_thresh'])
        -> filtered Cycles object

Cycle (start, end) sample indices come from
    cycles.get_inds_of_cycle(i, mode='standard')
which returns a 1-D array of all sample indices that belong to cycle i.
Indices are in the **native fs** that was passed to ``get_cycle_data``
(1000 Hz in our data).

The cycle phase reference per the user's request is
    np.angle(hilbert(theta_imf))
i.e. the wrapped instantaneous phase of the theta IMF itself. Within a
good cycle (``is_good == 1``) this phase is monotonic; we unwrap it
locally and renormalise to [0, 2π] before binning the wavelet power
into 20 equal phase bins.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.signal import hilbert

# Project src/ on path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import emd  # type: ignore
from utils import (  # type: ignore
    extract_imfs_by_pt_intervals,
    get_cycle_data,
    get_cycles_with_conditions,
)


DEFAULT_EMD_CONFIG = "/Users/amir/Desktop/for Abdel/emd_masksift_CA1_config.yml"
THETA_BAND_HZ = (5.0, 12.0)
THETA_IMF_PREFER = 5            # 0-indexed; same as RGS_pt_counter.THETA_IMF_PREFER
AMP_PERCENTILE = 25             # cycles must have max_amp above this percentile of IA
DEFAULT_FS_NATIVE = 1000


def load_default_emd_config(path: str = DEFAULT_EMD_CONFIG):
    """Load the project's mask-sift YAML config (same one used by
    RGS_pt_counter.py).
    """
    return emd.sift.SiftConfig.from_yaml_file(path)


# ---------------------------------------------------------------------------
# Theta IMF selection (same logic as RGS_pt_counter.choose_theta_idx)
# ---------------------------------------------------------------------------

def choose_theta_imf_index(imfs: np.ndarray,
                           imf_freqs: np.ndarray,
                           theta_band: tuple[float, float] = THETA_BAND_HZ,
                           prefer_index: int = THETA_IMF_PREFER) -> int:
    """Pick the IMF that carries theta.

    Rules (mirroring RGS_pt_counter.choose_theta_idx):
      1. If ``imfs`` has more than ``prefer_index`` columns, return
         ``prefer_index``.
      2. Otherwise return the IMF whose mean instantaneous frequency is
         closest to the centre of the theta band.
      3. Fallback: last IMF.
    """
    if imfs.shape[1] > prefer_index:
        return prefer_index
    centers = np.asarray(imf_freqs, dtype=float).ravel()
    if centers.size == imfs.shape[1]:
        center = 0.5 * (theta_band[0] + theta_band[1])
        return int(np.argmin(np.abs(centers - center)))
    return imfs.shape[1] - 1


# ---------------------------------------------------------------------------
# Per-cycle bounds from an emd.cycles.Cycles object
# ---------------------------------------------------------------------------

def _cycle_bounds_from_filtered_cycles(cycles_obj) -> list[tuple[int, int]]:
    """Get ``(start, end)`` sample indices for every cycle in the subset.

    ``cycles_obj`` is the object returned by ``get_cycles_with_conditions``;
    its ``.subset_vect`` attribute marks which cycles passed the filter.

    Uses ``cycles_obj.get_inds_of_cycle(i, mode='standard')`` which
    returns the array of sample indices belonging to cycle ``i``.
    """
    if cycles_obj is None:
        return []
    # Try the metric DataFrame to get the subset's original cycle ids
    try:
        df = cycles_obj.get_metric_dataframe(subset=True)
    except Exception:
        df = None
    if df is None or len(df) == 0:
        return []

    # cycle id column candidates differ across emd versions; try a few
    if "index" in df.columns:
        cycle_ids = df["index"].to_numpy(dtype=int)
    elif "cycle_ind" in df.columns:
        cycle_ids = df["cycle_ind"].to_numpy(dtype=int)
    else:
        # fall back to using the dataframe index itself
        cycle_ids = np.asarray(df.index, dtype=int)

    bounds: list[tuple[int, int]] = []
    for ci in cycle_ids:
        inds = None
        # Try 'standard' first; if it returns None, try without mode kwarg
        # (which defaults to 'augmented' in newer emd, 'standard' in older).
        for fn_kwargs in ({"mode": "standard"}, {}):
            try:
                cand = cycles_obj.get_inds_of_cycle(int(ci), **fn_kwargs)
            except Exception:
                continue
            if cand is not None and len(cand) > 0:
                inds = cand
                break
        if inds is None:
            continue
        s = int(inds[0])
        e = int(inds[-1]) + 1   # half-open [s, e)
        if e > s:
            bounds.append((s, e))
    return bounds


# ---------------------------------------------------------------------------
# Public per-segment API
# ---------------------------------------------------------------------------

@dataclass
class SegmentCycles:
    """All cycle info for one phasic or tonic interval."""
    hpc: np.ndarray              # raw HPC LFP segment at native fs
    pfc: np.ndarray              # raw PFC LFP segment at native fs
    theta_imf: np.ndarray        # theta IMF for the segment
    phase_wrapped: np.ndarray    # angle(hilbert(theta_imf)) in [-pi, pi]
    cycle_bounds: list[tuple[int, int]]   # half-open (s, e) in samples
    substate: str                # 'phasic' or 'tonic'
    interval_idx: int            # index within the IntervalSet
    t0_sec: float                # absolute interval start (sec) in session
    fs: float                    # native fs (Hz)
    theta_imf_idx: int           # which IMF was chosen
    n_imfs: int


def cycles_for_segment(hpc_seg: np.ndarray,
                       pfc_seg: np.ndarray,
                       imfs: np.ndarray,
                       imf_freqs: np.ndarray,
                       fs: float,
                       substate: str,
                       interval_idx: int,
                       t0_sec: float,
                       theta_band: tuple[float, float] = THETA_BAND_HZ,
                       prefer_index: int = THETA_IMF_PREFER,
                       amp_percentile: float = AMP_PERCENTILE
                       ) -> SegmentCycles | None:
    """Compute filtered EMD cycles for one already-extracted segment.

    Replicates the filter from ``RGS_pt_counter.count_cycles``:
      is_good == 1
      duration_samples < fs / theta_band[0]   (slower than 5 Hz cap)
      duration_samples > fs / theta_band[1]   (faster than 12 Hz cap)
      max_amp > percentile(IA, amp_percentile)
    """
    if imfs is None or imfs.size == 0 or imfs.shape[1] == 0:
        return None
    theta_idx = choose_theta_imf_index(imfs, imf_freqs,
                                       theta_band=theta_band,
                                       prefer_index=prefer_index)
    theta_imf = imfs[:, theta_idx]
    cycle_data = get_cycle_data(theta_imf, fs=int(fs))
    if cycle_data is None or "IA" not in cycle_data:
        return None
    amp_thresh = float(np.percentile(cycle_data["IA"], amp_percentile))
    lo_dur = fs / theta_band[0]       # 200 at 1 kHz with 5 Hz
    hi_dur = fs / theta_band[1]       # 83.3 at 1 kHz with 12 Hz
    conditions = [
        "is_good==1",
        f"duration_samples<{lo_dur}",
        f"duration_samples>{hi_dur}",
        f"max_amp>{amp_thresh}",
    ]
    all_cycles = get_cycles_with_conditions(cycle_data["cycles"], conditions)
    bounds = _cycle_bounds_from_filtered_cycles(all_cycles)
    if not bounds:
        return None

    phase_wrapped = np.angle(hilbert(theta_imf))
    return SegmentCycles(
        hpc=np.asarray(hpc_seg, dtype=float),
        pfc=np.asarray(pfc_seg, dtype=float),
        theta_imf=theta_imf,
        phase_wrapped=phase_wrapped,
        cycle_bounds=bounds,
        substate=substate,
        interval_idx=int(interval_idx),
        t0_sec=float(t0_sec),
        fs=float(fs),
        theta_imf_idx=int(theta_idx),
        n_imfs=int(imfs.shape[1]),
    )


def cycles_for_session(hpc_lfp: np.ndarray,
                       pfc_lfp: np.ndarray,
                       fs: float,
                       phasic_iv,
                       tonic_iv,
                       cfg,
                       theta_band: tuple[float, float] = THETA_BAND_HZ,
                       prefer_index: int = THETA_IMF_PREFER,
                       amp_percentile: float = AMP_PERCENTILE,
                       verbose: bool = False) -> list[SegmentCycles]:
    """Full per-session EMD cycle extraction.

    ``hpc_lfp`` and ``pfc_lfp`` are time-aligned 1-D LFPs at sampling
    rate ``fs``. ``phasic_iv`` and ``tonic_iv`` are ``pynapple``
    IntervalSets in absolute seconds.

    The HPC LFP is decomposed with mask-sift EMD per interval (this is
    exactly what ``extract_imfs_by_pt_intervals`` does). The PFC LFP is
    sliced at the same interval bounds so the PPC pipeline can use
    matched (HPC, PFC) segments.
    """
    out: list[SegmentCycles] = []
    for iv, name in [(phasic_iv, "phasic"), (tonic_iv, "tonic")]:
        if iv is None or len(iv) == 0:
            continue
        try:
            imfs_list, freqs_list, hpc_seg_list = extract_imfs_by_pt_intervals(
                hpc_lfp, fs, iv, cfg, return_imfs_freqs=True,
            )
        except Exception as ex:
            if verbose:
                print(f"    [emd] failed on {name}: {ex}")
            continue
        for i, (imfs, freqs, hpc_seg_check) in enumerate(zip(imfs_list,
                                                              freqs_list,
                                                              hpc_seg_list)):
            try:
                start_sec = float(iv.loc[i, "start"])
                end_sec = float(iv.loc[i, "end"])
            except Exception:
                try:
                    start_sec = float(iv.start[i])
                    end_sec = float(iv.end[i])
                except Exception:
                    continue
            s_idx = int(round(start_sec * fs))
            e_idx = int(round(end_sec * fs))
            s_idx = max(0, s_idx)
            e_idx = min(len(hpc_lfp), e_idx)
            if e_idx - s_idx < 2:
                continue
            hpc_seg = np.asarray(hpc_lfp[s_idx:e_idx], dtype=float)
            pfc_seg = np.asarray(pfc_lfp[s_idx:e_idx], dtype=float)
            # Length safety: align with the IMF length (extract_imfs_by_pt_intervals
            # uses int() floor so it can be 1 sample shorter).
            L = min(len(hpc_seg), len(pfc_seg), imfs.shape[0])
            hpc_seg = hpc_seg[:L]; pfc_seg = pfc_seg[:L]
            imfs_clipped = imfs[:L]
            seg = cycles_for_segment(
                hpc_seg=hpc_seg, pfc_seg=pfc_seg,
                imfs=imfs_clipped, imf_freqs=freqs,
                fs=fs, substate=name, interval_idx=i,
                t0_sec=start_sec,
                theta_band=theta_band,
                prefer_index=prefer_index,
                amp_percentile=amp_percentile,
            )
            if seg is None:
                continue
            out.append(seg)
        if verbose:
            n_cycles = sum(len(s.cycle_bounds) for s in out if s.substate == name)
            print(f"    [emd] {name}: {n_cycles} cycles "
                  f"from {len(imfs_list)} intervals")
    return out
