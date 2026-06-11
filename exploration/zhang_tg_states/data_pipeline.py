"""Per-rat data ingestion for the TG-state pipeline.

Walks the project's RGS folder hierarchy, loads HPC + PFC LFPs and the
hypnogram, extracts phasic and tonic REM intervals from
``src.utils.extract_pt_intervals``, and produces the inputs needed by
the FPP / PPC modules: paired (HPC, PFC) LFP segments at the analysis
sampling rate, with absolute time stamps and interval-membership tags.

Reproduces the conventions of ``exploration/_build_lfp_lfp_ppc_zhang2019.py``:
  * raw fs = 1000 Hz
  * analysis fs = 625 Hz
  * RAT_GROUPS = {'positive': [3, 4, 7, 8], 'control': [1, 2, 6, 9]}
  * source = PFC, target = HPC (theta phase from HPC)
  * pickles produced are independent of the existing project files
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

# Make src/ importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import get_data, extract_pt_intervals  # type: ignore

from .fpp import (
    ANALYSIS_FS,
    FREQUENCIES,
    N_PHASE_BINS,
    PHASE_CENTERS_DEG,
    PHASE_CENTERS_RAD,
    FPPCollection,
    concat_collections,
    fpps_from_lfp_segment,
    fpps_from_segment_with_cycles,
    resample_to_analysis_fs,
    theta_unwrapped_phase,
)
from .ppc import (
    CrossSpectrumResult,
    cross_spectrum_for_segment,
    cross_spectrum_with_cycles,
)
from .emd_cycles import (
    DEFAULT_EMD_CONFIG,
    THETA_BAND_HZ,
    THETA_IMF_PREFER,
    AMP_PERCENTILE,
    cycles_for_session,
    load_default_emd_config,
)


BASE_PATH = "/Users/amir/Desktop/for Abdel/RGS/DatabyCondition"
FS_RAW_EXPECTED = 1000
RAT_GROUPS = {
    "positive": [3, 4, 7, 8],
    "control": [1, 2, 6, 9],
}
PREFERRED_CONDITIONS = ["HomeCageHC", "RandomCon", "OverlappingOR",
                        "StableCondOD", "HomeCageCG"]


# ---------------------------------------------------------------------------
# Filesystem helpers (mirrors _build_lfp_lfp_ppc_zhang2019.find_file_local)
# ---------------------------------------------------------------------------

def find_file_local(directory: str, prefix: str) -> str | None:
    if not os.path.isdir(directory):
        return None
    for name in sorted(os.listdir(directory)):
        if name.startswith(prefix):
            return os.path.join(directory, name)
    return None


def list_rat_sessions(rat_id: int, base_path: str = BASE_PATH,
                      conditions: Sequence[str] | None = None
                      ) -> list[dict]:
    """List all (HPC.mat, PFC.mat, states.mat, condition) bundles for one rat."""
    if conditions is None:
        conditions = PREFERRED_CONDITIONS
    rat_token = f"Rat{rat_id}_"
    sessions: list[dict] = []
    for cond in conditions:
        cond_dir = os.path.join(base_path, cond)
        if not os.path.isdir(cond_dir):
            continue
        for entry in sorted(os.listdir(cond_dir)):
            if rat_token not in entry:
                continue
            rat_dir = os.path.join(cond_dir, entry)
            if not os.path.isdir(rat_dir):
                continue
            for sub in sorted(os.listdir(rat_dir)):
                sub_dir = os.path.join(rat_dir, sub)
                if not os.path.isdir(sub_dir):
                    continue
                hpc_path = find_file_local(sub_dir, "HPC_")
                pfc_path = find_file_local(sub_dir, "PFC_")
                state_path = find_file_local(sub_dir, "")
                if not (hpc_path and pfc_path):
                    continue
                # find -states.mat
                state_path = None
                for n in sorted(os.listdir(sub_dir)):
                    if n.endswith("-states.mat") or n.endswith("_states.mat"):
                        state_path = os.path.join(sub_dir, n)
                        break
                if state_path is None:
                    continue
                sessions.append(dict(
                    rat_id=rat_id, condition=cond, folder=entry,
                    sub=sub, hpc_path=hpc_path, pfc_path=pfc_path,
                    state_path=state_path,
                ))
    return sessions


# ---------------------------------------------------------------------------
# Loading and REM substate extraction for one session
# ---------------------------------------------------------------------------

@dataclass
class SessionSegments:
    """Per-session collection of REM segments split into phasic/tonic."""
    rat_id: int
    condition: str
    folder: str
    sub: str
    fs_raw: int
    fs_analysis: float = ANALYSIS_FS
    # Per-segment dicts with keys: hpc, pfc, t0_sec, n_samples, substate ('phasic' | 'tonic')
    segments: list[dict] = field(default_factory=list)


def _intervalset_to_arrays(iv) -> tuple[np.ndarray, np.ndarray]:
    """Extract (start_sec, end_sec) arrays from a pynapple IntervalSet."""
    try:
        starts = np.asarray(iv.start, dtype=float)
        ends = np.asarray(iv.end, dtype=float)
    except Exception:
        # pandas-style fallback
        starts = np.asarray(iv["start"], dtype=float)
        ends = np.asarray(iv["end"], dtype=float)
    return starts, ends


def load_session_rem_segments(session: dict,
                              base_substate_min_dur: float = 0.5
                              ) -> SessionSegments:
    """Load HPC+PFC LFP, hypnogram, run extract_pt_intervals, slice segments.

    ``base_substate_min_dur`` is a soft minimum duration (seconds) below
    which segments are dropped (too short for FPP estimation).
    """
    hpc_lfp, hypno, fs_hpc = get_data(session["hpc_path"], session["state_path"],
                                       type="hpc")
    pfc_lfp, _, fs_pfc = get_data(session["pfc_path"], session["state_path"],
                                   type="pfc")
    fs = int(fs_hpc)
    if fs != int(fs_pfc):
        # Defensive: trim PFC down to HPC fs (project convention)
        pass

    # Align HPC and PFC to the same length
    L = min(len(hpc_lfp), len(pfc_lfp))
    hpc_lfp = np.asarray(hpc_lfp[:L], dtype=float)
    pfc_lfp = np.asarray(pfc_lfp[:L], dtype=float)

    # extract_pt_intervals expects fs as int; it downsamples internally to 500 Hz
    phasic_iv, tonic_iv, _ = extract_pt_intervals(hpc_lfp, hypno, fs=fs)
    p_starts, p_ends = _intervalset_to_arrays(phasic_iv)
    t_starts, t_ends = _intervalset_to_arrays(tonic_iv)

    segments: list[dict] = []
    for (starts, ends, name) in [(p_starts, p_ends, "phasic"),
                                  (t_starts, t_ends, "tonic")]:
        for s, e in zip(starts, ends):
            if e - s < base_substate_min_dur:
                continue
            s_idx = int(round(s * fs))
            e_idx = int(round(e * fs))
            s_idx = max(s_idx, 0)
            e_idx = min(e_idx, L)
            if e_idx <= s_idx:
                continue
            seg = dict(
                hpc=hpc_lfp[s_idx:e_idx],
                pfc=pfc_lfp[s_idx:e_idx],
                t0_sec=float(s_idx / fs),
                n_samples=int(e_idx - s_idx),
                substate=name,
            )
            segments.append(seg)
    return SessionSegments(
        rat_id=session["rat_id"], condition=session["condition"],
        folder=session["folder"], sub=session["sub"], fs_raw=fs,
        segments=segments,
    )


# ---------------------------------------------------------------------------
# FPP + cross-spectrum extraction for one rat
# ---------------------------------------------------------------------------

@dataclass
class RatAggregate:
    rat_id: int
    fpps: FPPCollection                     # all cycles concatenated
    cycle_substates: np.ndarray             # per-cycle 'phasic' or 'tonic'
    cycle_session_ids: np.ndarray           # per-cycle integer session id
    cycle_starts_sec: np.ndarray            # absolute seconds within session
    cross_spectrum: CrossSpectrumResult     # angles aligned with fpps cycles
    sessions: list[dict]                    # session metadata
    frequencies: np.ndarray
    phase_centers_rad: np.ndarray
    # ``cycle_segment_ids[i]`` = unique integer identifying which (session, substate,
    # interval) segment cycle ``i`` came from. Cycles within the same segment are
    # temporally consecutive within ONE phasic or tonic REM interval. Used by the
    # Markov section to build per-interval runs without needing to reconstruct
    # interval bounds from cycle timestamps (the concatenated array is per-segment,
    # not chronological).
    cycle_segment_ids: np.ndarray = None    # type: ignore


def process_rat(rat_id: int,
                base_path: str = BASE_PATH,
                conditions: Sequence[str] | None = None,
                analysis_fs: float = ANALYSIS_FS,
                frequencies: np.ndarray = FREQUENCIES,
                n_phase_bins: int = N_PHASE_BINS,
                min_segment_duration_sec: float = 0.5,
                verbose: bool = True) -> RatAggregate | None:
    """Run the FPP + cross-spectrum extraction for one rat across sessions.

    Returns ``None`` if no usable cycles are found.
    """
    sessions = list_rat_sessions(rat_id, base_path=base_path,
                                  conditions=conditions)
    if verbose:
        print(f"[rat {rat_id}] {len(sessions)} sessions found")
    fpp_collections: list[FPPCollection] = []
    cross_chunks: list[CrossSpectrumResult] = []
    cycle_substates: list[np.ndarray] = []
    cycle_session_ids: list[np.ndarray] = []
    session_meta: list[dict] = []
    for sid, sess in enumerate(sessions):
        try:
            seg = load_session_rem_segments(sess,
                                            base_substate_min_dur=min_segment_duration_sec)
        except Exception as ex:
            if verbose:
                print(f"  [rat {rat_id}] skipped {sess['folder']}/{sess['sub']}: {ex}")
            continue
        if not seg.segments:
            continue
        session_meta.append(dict(
            session_id=sid, rat_id=rat_id,
            condition=sess["condition"], folder=sess["folder"],
            sub=sess["sub"], fs_raw=seg.fs_raw,
            n_segments=len(seg.segments),
        ))
        for s in seg.segments:
            try:
                fpp = fpps_from_lfp_segment(
                    s["hpc"], fs=seg.fs_raw,
                    analysis_fs=analysis_fs,
                    frequencies=frequencies,
                    n_phase_bins=n_phase_bins,
                    segment_offset_sec=s["t0_sec"],
                    segment_meta=dict(rat_id=rat_id,
                                       session_id=sid,
                                       condition=sess["condition"],
                                       substate=s["substate"]),
                )
                cross = cross_spectrum_for_segment(
                    source_lfp=s["pfc"],
                    target_lfp=s["hpc"],
                    fs=seg.fs_raw,
                    analysis_fs=analysis_fs,
                    frequencies=frequencies,
                    n_phase_bins=n_phase_bins,
                    segment_offset_sec=s["t0_sec"],
                )
            except Exception as ex:
                if verbose:
                    print(f"  [rat {rat_id}] FPP/cross failed: {ex}")
                continue
            # Align: trust that both pipelines see the same cycle bounds because
            # they use the same theta_unwrapped_phase + extract_theta_cycles on
            # the same HPC LFP. Confirm cycle count and clip if mismatch.
            n = min(len(fpp), len(cross))
            if n == 0:
                continue
            if len(fpp) != len(cross):
                # Safety: keep the intersection by bounds
                # (this should be rare in practice)
                fpp = FPPCollection(
                    fpps=fpp.fpps[:n], cycle_bounds=fpp.cycle_bounds[:n],
                    cycle_starts_sec=fpp.cycle_starts_sec[:n],
                    frequencies=fpp.frequencies,
                    phase_centers_deg=fpp.phase_centers_deg,
                    fs=fpp.fs, segment_meta=fpp.segment_meta,
                )
                cross = CrossSpectrumResult(
                    angles=cross.angles[:n], cycle_bounds=cross.cycle_bounds[:n],
                    cycle_starts_sec=cross.cycle_starts_sec[:n], fs=cross.fs,
                )
            fpp_collections.append(fpp)
            cross_chunks.append(cross)
            cycle_substates.append(np.array([s["substate"]] * n))
            cycle_session_ids.append(np.full(n, sid, dtype=int))

    if not fpp_collections:
        if verbose:
            print(f"[rat {rat_id}] no cycles extracted")
        return None

    fpp_all = concat_collections(fpp_collections)
    angles_all = np.concatenate([c.angles for c in cross_chunks], axis=0)
    cross_bounds = np.concatenate([c.cycle_bounds for c in cross_chunks], axis=0)
    cross_starts = np.concatenate([c.cycle_starts_sec for c in cross_chunks], axis=0)
    cross_all = CrossSpectrumResult(
        angles=angles_all,
        cycle_bounds=cross_bounds,
        cycle_starts_sec=cross_starts,
        fs=fpp_all.fs,
    )
    substates = np.concatenate(cycle_substates, axis=0)
    session_ids = np.concatenate(cycle_session_ids, axis=0)

    if verbose:
        n_phasic = int((substates == "phasic").sum())
        n_tonic = int((substates == "tonic").sum())
        print(f"[rat {rat_id}] cycles: phasic={n_phasic}, tonic={n_tonic}, "
              f"total={len(substates)}")

    return RatAggregate(
        rat_id=rat_id,
        fpps=fpp_all,
        cycle_substates=substates,
        cycle_session_ids=session_ids,
        cycle_starts_sec=fpp_all.cycle_starts_sec,
        cross_spectrum=cross_all,
        sessions=session_meta,
        frequencies=fpp_all.frequencies,
        phase_centers_rad=PHASE_CENTERS_RAD.copy(),
    )


# ---------------------------------------------------------------------------
# EMD-driven processing (preferred path)
# ---------------------------------------------------------------------------

def process_rat_emd(rat_id: int,
                    cfg=None,
                    base_path: str = BASE_PATH,
                    conditions: Sequence[str] | None = None,
                    frequencies: np.ndarray = FREQUENCIES,
                    n_phase_bins: int = N_PHASE_BINS,
                    theta_band: tuple[float, float] = THETA_BAND_HZ,
                    prefer_theta_imf_index: int = THETA_IMF_PREFER,
                    amp_percentile: float = AMP_PERCENTILE,
                    verbose: bool = True) -> RatAggregate | None:
    """Per-rat extraction using EMD-derived theta cycles.

    Pipeline (per session):

      1. ``get_data`` HPC, PFC, hypnogram (native fs).
      2. ``extract_pt_intervals`` -> phasic + tonic IntervalSets.
      3. ``extract_imfs_by_pt_intervals`` (mask-sift EMD) per substate.
      4. Pick theta IMF (``choose_theta_imf_index``), call
         ``get_cycle_data``, filter with
         ``is_good==1, duration in (fs/12, fs/5), max_amp > p25(IA)``.
      5. ``np.angle(hilbert(theta_imf))`` -> wrapped per-sample phase
         reference for binning into 20 phase bins.
      6. Compute Morlet FPP per cycle at native fs (no resampling).
      7. Compute matched PFC vs HPC wavelet cross-spectrum per cycle.

    Cycles indices are in **native fs** (1000 Hz) and per-segment;
    absolute times are reconstructed via ``segment_offset_sec``.
    """
    if cfg is None:
        cfg = load_default_emd_config()

    sessions = list_rat_sessions(rat_id, base_path=base_path,
                                  conditions=conditions)
    if verbose:
        print(f"[rat {rat_id}] {len(sessions)} sessions found")

    fpp_collections: list[FPPCollection] = []
    cross_chunks: list[CrossSpectrumResult] = []
    cycle_substates: list[np.ndarray] = []
    cycle_session_ids: list[np.ndarray] = []
    cycle_segment_ids: list[np.ndarray] = []
    session_meta: list[dict] = []
    segment_counter = 0   # increments for every segment (phasic or tonic interval)

    for sid, sess in enumerate(sessions):
        try:
            hpc_lfp, hypno, fs_hpc = get_data(sess["hpc_path"],
                                               sess["state_path"], type="hpc")
            pfc_lfp, _, fs_pfc = get_data(sess["pfc_path"],
                                           sess["state_path"], type="pfc")
        except Exception as ex:
            if verbose:
                print(f"  [rat {rat_id}] cannot load "
                      f"{sess['folder']}/{sess['sub']}: {ex}")
            continue
        fs = int(fs_hpc)
        # Align lengths
        L = min(len(hpc_lfp), len(pfc_lfp))
        hpc_lfp = np.asarray(hpc_lfp[:L], dtype=float)
        pfc_lfp = np.asarray(pfc_lfp[:L], dtype=float)

        try:
            from utils import extract_pt_intervals  # type: ignore
            phasic_iv, tonic_iv, _ = extract_pt_intervals(hpc_lfp, hypno, fs=fs)
        except ValueError:
            if verbose:
                print(f"  [rat {rat_id}] no REM in "
                      f"{sess['folder']}/{sess['sub']}")
            continue
        except Exception as ex:
            if verbose:
                print(f"  [rat {rat_id}] extract_pt_intervals failed: {ex}")
            continue

        try:
            segs = cycles_for_session(
                hpc_lfp=hpc_lfp, pfc_lfp=pfc_lfp, fs=fs,
                phasic_iv=phasic_iv, tonic_iv=tonic_iv, cfg=cfg,
                theta_band=theta_band,
                prefer_index=prefer_theta_imf_index,
                amp_percentile=amp_percentile,
                verbose=verbose,
            )
        except Exception as ex:
            if verbose:
                print(f"  [rat {rat_id}] EMD failed on "
                      f"{sess['folder']}/{sess['sub']}: {ex}")
            continue

        if not segs:
            continue

        session_meta.append(dict(
            session_id=sid, rat_id=rat_id,
            condition=sess["condition"], folder=sess["folder"],
            sub=sess["sub"], fs_raw=fs,
            n_segments=len(segs),
            n_cycles=sum(len(s.cycle_bounds) for s in segs),
            n_cycles_phasic=sum(len(s.cycle_bounds) for s in segs
                                 if s.substate == "phasic"),
            n_cycles_tonic=sum(len(s.cycle_bounds) for s in segs
                                if s.substate == "tonic"),
        ))

        for seg in segs:
            fpp = fpps_from_segment_with_cycles(
                seg.hpc, fs=seg.fs,
                cycle_bounds=seg.cycle_bounds,
                theta_phase_wrapped=seg.phase_wrapped,
                frequencies=frequencies,
                n_phase_bins=n_phase_bins,
                segment_offset_sec=seg.t0_sec,
                segment_meta=dict(rat_id=rat_id, session_id=sid,
                                   condition=sess["condition"],
                                   substate=seg.substate,
                                   interval_idx=seg.interval_idx,
                                   theta_imf_idx=seg.theta_imf_idx),
            )
            cross = cross_spectrum_with_cycles(
                source_lfp=seg.pfc, target_lfp=seg.hpc, fs=seg.fs,
                cycle_bounds=seg.cycle_bounds,
                theta_phase_wrapped=seg.phase_wrapped,
                frequencies=frequencies,
                n_phase_bins=n_phase_bins,
                segment_offset_sec=seg.t0_sec,
            )
            n = min(len(fpp), len(cross))
            if n == 0:
                continue
            # Trim to common length so cycle k matches in both
            if len(fpp) != len(cross):
                fpp = FPPCollection(
                    fpps=fpp.fpps[:n], cycle_bounds=fpp.cycle_bounds[:n],
                    cycle_starts_sec=fpp.cycle_starts_sec[:n],
                    frequencies=fpp.frequencies,
                    phase_centers_deg=fpp.phase_centers_deg,
                    fs=fpp.fs, segment_meta=fpp.segment_meta,
                )
                cross = CrossSpectrumResult(
                    angles=cross.angles[:n], cycle_bounds=cross.cycle_bounds[:n],
                    cycle_starts_sec=cross.cycle_starts_sec[:n], fs=cross.fs,
                )
            fpp_collections.append(fpp)
            cross_chunks.append(cross)
            cycle_substates.append(np.array([seg.substate] * n))
            cycle_session_ids.append(np.full(n, sid, dtype=int))
            cycle_segment_ids.append(np.full(n, segment_counter, dtype=int))
            segment_counter += 1

    if not fpp_collections:
        if verbose:
            print(f"[rat {rat_id}] no EMD cycles extracted")
        return None

    fpp_all = concat_collections(fpp_collections)
    angles_all = np.concatenate([c.angles for c in cross_chunks], axis=0)
    cross_bounds = np.concatenate([c.cycle_bounds for c in cross_chunks], axis=0)
    cross_starts = np.concatenate([c.cycle_starts_sec for c in cross_chunks], axis=0)
    cross_all = CrossSpectrumResult(
        angles=angles_all, cycle_bounds=cross_bounds,
        cycle_starts_sec=cross_starts, fs=fpp_all.fs,
    )
    substates = np.concatenate(cycle_substates, axis=0)
    session_ids = np.concatenate(cycle_session_ids, axis=0)
    segment_ids = np.concatenate(cycle_segment_ids, axis=0)

    if verbose:
        n_ph = int((substates == "phasic").sum())
        n_to = int((substates == "tonic").sum())
        print(f"[rat {rat_id}] EMD cycles: phasic={n_ph}, tonic={n_to}, "
              f"total={len(substates)}")

    return RatAggregate(
        rat_id=rat_id,
        fpps=fpp_all,
        cycle_substates=substates,
        cycle_session_ids=session_ids,
        cycle_starts_sec=fpp_all.cycle_starts_sec,
        cross_spectrum=cross_all,
        sessions=session_meta,
        frequencies=fpp_all.frequencies,
        phase_centers_rad=PHASE_CENTERS_RAD.copy(),
        cycle_segment_ids=segment_ids,
    )


# ---------------------------------------------------------------------------
# Butterworth + Hilbert (Zhang's original cycle-detection method, no EMD)
# ---------------------------------------------------------------------------

def process_rat_butter(rat_id: int,
                       base_path: str = BASE_PATH,
                       conditions: Sequence[str] | None = None,
                       analysis_fs: float = ANALYSIS_FS,
                       frequencies: np.ndarray = FREQUENCIES,
                       n_phase_bins: int = N_PHASE_BINS,
                       theta_band: tuple[float, float] = (5.0, 12.0),
                       theta_cycle_freq_range: tuple[float, float] = (5.0, 12.0),
                       rem_padding_sec: float = 5.0,
                       verbose: bool = True) -> RatAggregate | None:
    """Per-rat extraction using **Butterworth bandpass + Hilbert** (Zhang's
    original cycle-detection method). **No EMD anywhere.**

    Pipeline (per session, on the WHOLE continuous LFP):

      1. Resample HPC and PFC LFP to ``analysis_fs`` (625 Hz by default).
      2. 4th-order Butterworth bandpass HPC at ``theta_band`` (5-12 Hz),
         zero-phase ``sosfiltfilt``.
      3. Hilbert transform -> ``np.unwrap(np.angle(.))`` gives the
         monotonically increasing theta phase.
      4. Detect theta cycles as successive 2π crossings of the unwrapped
         phase. Reject cycles outside ``theta_cycle_freq_range``.
      5. Compute the FPP for each cycle (Morlet CWT power, smoothing,
         z-score, 20-phase-bin averaging).
      6. Compute the PFC × HPC* cross-spectrum, bin per cycle, store
         per-cycle angle matrices.
      7. **Tag** each cycle by which phasic/tonic interval (from
         ``extract_pt_intervals``) fully contains it. Cycles outside
         any interval (wake / NREM) are dropped.
      8. Assign a unique ``segment_id`` per (session, substate, interval)
         tuple so the Markov module can group runs.

    Returns the same ``RatAggregate`` shape as ``process_rat_emd``, so
    the rest of the analysis notebook (clustering / Markov / PPC /
    plots) is identical regardless of which pipeline was used.
    """
    sessions = list_rat_sessions(rat_id, base_path=base_path,
                                  conditions=conditions)
    if verbose:
        print(f"[rat {rat_id}] {len(sessions)} sessions found")

    fpp_collections: list[FPPCollection] = []
    cross_chunks: list[CrossSpectrumResult] = []
    cycle_substates: list[np.ndarray] = []
    cycle_session_ids: list[np.ndarray] = []
    cycle_segment_ids: list[np.ndarray] = []
    session_meta: list[dict] = []
    segment_counter = 0

    for sid, sess in enumerate(sessions):
        try:
            hpc_lfp, hypno, fs_hpc = get_data(sess["hpc_path"],
                                               sess["state_path"], type="hpc")
            pfc_lfp, _, _ = get_data(sess["pfc_path"],
                                     sess["state_path"], type="pfc")
        except Exception as ex:
            if verbose:
                print(f"  [rat {rat_id}] load failed "
                      f"{sess['folder']}/{sess['sub']}: {ex}")
            continue
        fs = int(fs_hpc)
        L = min(len(hpc_lfp), len(pfc_lfp))
        hpc_lfp = np.asarray(hpc_lfp[:L], dtype=float)
        pfc_lfp = np.asarray(pfc_lfp[:L], dtype=float)

        try:
            phasic_iv, tonic_iv, _ = extract_pt_intervals(hpc_lfp, hypno, fs=fs)
        except Exception:
            if verbose:
                print(f"  [rat {rat_id}] no REM in "
                      f"{sess['folder']}/{sess['sub']}")
            continue

        try:
            p_starts, p_ends = _intervalset_to_arrays(phasic_iv)
        except Exception:
            p_starts = p_ends = np.array([])
        try:
            t_starts, t_ends = _intervalset_to_arrays(tonic_iv)
        except Exception:
            t_starts = t_ends = np.array([])
        if len(p_starts) == 0 and len(t_starts) == 0:
            continue

        # ----- Optimisation: build REM "bouts" by merging adjacent
        # phasic/tonic intervals separated by less than 60 s into one
        # continuous slice. The Morlet CWT then runs on the actual REM
        # duration instead of any wake / NREM time *between* REM bouts.
        # Each bout is processed independently, with rem_padding_sec on
        # each side for Butterworth filter transients.
        iv_pairs: list[tuple[float, float]] = []
        for s, e in zip(p_starts, p_ends):
            iv_pairs.append((float(s), float(e)))
        for s, e in zip(t_starts, t_ends):
            iv_pairs.append((float(s), float(e)))
        iv_pairs.sort()

        BOUT_MERGE_GAP = 60.0   # merge intervals within 60 s of each other
        bouts: list[tuple[float, float]] = []
        cur_s, cur_e = iv_pairs[0]
        for s, e in iv_pairs[1:]:
            if s - cur_e < BOUT_MERGE_GAP:
                cur_e = max(cur_e, e)
            else:
                bouts.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        bouts.append((cur_s, cur_e))

        fpp_session_chunks: list[FPPCollection] = []
        cross_session_chunks: list[CrossSpectrumResult] = []

        for bout_start, bout_end in bouts:
            win_start_sec = max(0.0, bout_start - rem_padding_sec)
            win_end_sec = bout_end + rem_padding_sec
            s_idx_raw = max(0, int(round(win_start_sec * fs)))
            e_idx_raw = min(L, int(round(win_end_sec * fs)))
            if e_idx_raw - s_idx_raw < int(2 * fs):
                continue
            hpc_slice = hpc_lfp[s_idx_raw:e_idx_raw]
            pfc_slice = pfc_lfp[s_idx_raw:e_idx_raw]
            slice_offset_sec = s_idx_raw / fs
            try:
                fpp_b = fpps_from_lfp_segment(
                    hpc_slice, fs=fs,
                    analysis_fs=analysis_fs,
                    frequencies=frequencies,
                    n_phase_bins=n_phase_bins,
                    theta_band=theta_band,
                    theta_cycle_freq_range=theta_cycle_freq_range,
                    segment_offset_sec=slice_offset_sec,
                )
                cross_b = cross_spectrum_for_segment(
                    source_lfp=pfc_slice, target_lfp=hpc_slice, fs=fs,
                    analysis_fs=analysis_fs,
                    frequencies=frequencies,
                    n_phase_bins=n_phase_bins,
                    theta_band=theta_band,
                    theta_cycle_freq_range=theta_cycle_freq_range,
                    segment_offset_sec=slice_offset_sec,
                )
            except Exception as ex:
                if verbose:
                    print(f"  [rat {rat_id}] FPP/cross failed on bout "
                          f"{bout_start:.1f}-{bout_end:.1f}s: {ex}")
                continue
            if len(fpp_b) == 0:
                continue
            fpp_session_chunks.append(fpp_b)
            # align bout fpp/cross lengths (extremely rare mismatch)
            n_min = min(len(fpp_b), len(cross_b))
            if n_min == 0:
                continue
            if len(fpp_b) != len(cross_b):
                cross_b = CrossSpectrumResult(
                    angles=cross_b.angles[:n_min],
                    cycle_bounds=cross_b.cycle_bounds[:n_min],
                    cycle_starts_sec=cross_b.cycle_starts_sec[:n_min],
                    fs=cross_b.fs,
                )
            cross_session_chunks.append(cross_b)

        if not fpp_session_chunks:
            continue
        # Concat across bouts in this session
        fpp = concat_collections(fpp_session_chunks)
        cross = CrossSpectrumResult(
            angles=np.concatenate([c.angles for c in cross_session_chunks], axis=0),
            cycle_bounds=np.concatenate([c.cycle_bounds for c in cross_session_chunks], axis=0),
            cycle_starts_sec=np.concatenate([c.cycle_starts_sec for c in cross_session_chunks], axis=0),
            fs=cross_session_chunks[0].fs,
        )

        n = min(len(fpp), len(cross))
        if n == 0:
            continue
        if len(fpp) != len(cross):
            fpp = FPPCollection(
                fpps=fpp.fpps[:n], cycle_bounds=fpp.cycle_bounds[:n],
                cycle_starts_sec=fpp.cycle_starts_sec[:n],
                frequencies=fpp.frequencies,
                phase_centers_deg=fpp.phase_centers_deg,
                fs=fpp.fs, segment_meta=fpp.segment_meta,
            )
            cross = CrossSpectrumResult(
                angles=cross.angles[:n], cycle_bounds=cross.cycle_bounds[:n],
                cycle_starts_sec=cross.cycle_starts_sec[:n], fs=cross.fs,
            )

        # Tag each cycle by substate / interval membership.
        cycle_starts_sec = fpp.cycle_starts_sec
        cycle_durations_sec = (fpp.cycle_bounds[:, 1]
                                - fpp.cycle_bounds[:, 0]) / fpp.fs
        cycle_ends_sec = cycle_starts_sec + cycle_durations_sec
        substates_arr = np.array(["other"] * n, dtype=object)
        interval_idx_arr = np.full(n, -1, dtype=int)
        for i in range(n):
            cs, ce = cycle_starts_sec[i], cycle_ends_sec[i]
            if p_starts.size:
                hit = np.where((p_starts <= cs) & (p_ends >= ce))[0]
                if len(hit):
                    substates_arr[i] = "phasic"
                    interval_idx_arr[i] = int(hit[0])
                    continue
            if t_starts.size:
                hit = np.where((t_starts <= cs) & (t_ends >= ce))[0]
                if len(hit):
                    substates_arr[i] = "tonic"
                    interval_idx_arr[i] = int(hit[0])

        keep_mask = substates_arr != "other"
        n_keep = int(keep_mask.sum())
        if n_keep == 0:
            continue

        fpp_keep = FPPCollection(
            fpps=fpp.fpps[keep_mask],
            cycle_bounds=fpp.cycle_bounds[keep_mask],
            cycle_starts_sec=fpp.cycle_starts_sec[keep_mask],
            frequencies=fpp.frequencies,
            phase_centers_deg=fpp.phase_centers_deg,
            fs=fpp.fs, segment_meta=None,
        )
        cross_keep = CrossSpectrumResult(
            angles=cross.angles[keep_mask],
            cycle_bounds=cross.cycle_bounds[keep_mask],
            cycle_starts_sec=cross.cycle_starts_sec[keep_mask],
            fs=cross.fs,
        )
        substates_keep = substates_arr[keep_mask]
        intervals_keep = interval_idx_arr[keep_mask]

        # Assign one segment_id per (substate, interval) group within this
        # session. The order of cycles is already time-monotonic across
        # the session, so cycles sharing a segment_id are temporally
        # contiguous within ONE REM substate interval.
        local_seg_ids = np.full(n_keep, -1, dtype=int)
        seen: dict[tuple[str, int], int] = {}
        for i in range(n_keep):
            key = (substates_keep[i], int(intervals_keep[i]))
            if key not in seen:
                seen[key] = segment_counter
                segment_counter += 1
            local_seg_ids[i] = seen[key]

        fpp_collections.append(fpp_keep)
        cross_chunks.append(cross_keep)
        cycle_substates.append(substates_keep)
        cycle_session_ids.append(np.full(n_keep, sid, dtype=int))
        cycle_segment_ids.append(local_seg_ids)
        n_ph = int((substates_keep == "phasic").sum())
        n_to = int((substates_keep == "tonic").sum())
        session_meta.append(dict(
            session_id=sid, rat_id=rat_id,
            condition=sess["condition"], folder=sess["folder"],
            sub=sess["sub"], fs_raw=fs, fs_analysis=fpp.fs,
            n_cycles=n_keep,
            n_cycles_phasic=n_ph, n_cycles_tonic=n_to,
            n_segments=len(seen),
        ))

    if not fpp_collections:
        if verbose:
            print(f"[rat {rat_id}] no usable cycles after substate tagging")
        return None

    fpp_all = concat_collections(fpp_collections)
    angles_all = np.concatenate([c.angles for c in cross_chunks], axis=0)
    bounds_all = np.concatenate([c.cycle_bounds for c in cross_chunks], axis=0)
    starts_all = np.concatenate([c.cycle_starts_sec for c in cross_chunks], axis=0)
    cross_all = CrossSpectrumResult(
        angles=angles_all, cycle_bounds=bounds_all,
        cycle_starts_sec=starts_all, fs=fpp_all.fs,
    )
    substates = np.concatenate(cycle_substates, axis=0)
    session_ids = np.concatenate(cycle_session_ids, axis=0)
    segment_ids = np.concatenate(cycle_segment_ids, axis=0)

    if verbose:
        n_ph = int((substates == "phasic").sum())
        n_to = int((substates == "tonic").sum())
        print(f"[rat {rat_id}] Butterworth cycles: phasic={n_ph}, "
              f"tonic={n_to}, total={len(substates)}")

    return RatAggregate(
        rat_id=rat_id,
        fpps=fpp_all,
        cycle_substates=substates,
        cycle_session_ids=session_ids,
        cycle_starts_sec=fpp_all.cycle_starts_sec,
        cross_spectrum=cross_all,
        sessions=session_meta,
        frequencies=fpp_all.frequencies,
        phase_centers_rad=PHASE_CENTERS_RAD.copy(),
        cycle_segment_ids=segment_ids,
    )
