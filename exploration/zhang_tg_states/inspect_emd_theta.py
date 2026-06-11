#!/usr/bin/env python
"""Visualise EMD theta extraction on HPC and PFC for one REM interval.

Picks one rat / one session / the longest phasic-or-tonic interval, runs
mask-sift EMD on both HPC and PFC for that interval, identifies the
theta IMF in each, and plots:

  Panel 1: HPC raw LFP (grey) + HPC theta IMF (red) + HPC cycle starts
           (green circles) + PFC cycle starts (orange triangles, faded)
  Panel 2: PFC raw LFP (grey) + PFC theta IMF (red) + HPC cycle starts
           (green circles, faded) + PFC cycle starts (orange triangles)

Cycle starts are the half-open `[start, end)` bounds returned by the
project's existing filter (`is_good==1`, 5–12 Hz duration window,
amplitude > 25th-percentile-of-IA).

Run from the project root:

    python exploration/zhang_tg_states/inspect_emd_theta.py \
        --rat 3 --substate tonic --n_seconds 5 \
        --out /tmp/emd_theta_inspect.png
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXPL = os.path.dirname(_HERE)
_PROJECT_ROOT = os.path.dirname(_EXPL)
for p in (_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "src"), _EXPL):
    if p not in sys.path:
        sys.path.insert(0, p)

from utils import (  # type: ignore  # noqa: E402
    extract_imfs_by_pt_intervals,
    extract_pt_intervals,
    get_cycle_data,
    get_cycles_with_conditions,
    get_data,
)

from zhang_tg_states.data_pipeline import list_rat_sessions  # noqa: E402
from zhang_tg_states.emd_cycles import (  # noqa: E402
    AMP_PERCENTILE,
    THETA_BAND_HZ,
    THETA_IMF_PREFER,
    _cycle_bounds_from_filtered_cycles,
    choose_theta_imf_index,
    load_default_emd_config,
)


# ---------------------------------------------------------------------------

def filter_cycles_from_theta_imf(theta_imf: np.ndarray, fs: float,
                                 theta_band: tuple[float, float] = THETA_BAND_HZ,
                                 amp_percentile: float = AMP_PERCENTILE,
                                 ) -> list[tuple[int, int]]:
    """Return (start, end) sample indices for cycles passing the project's
    is_good/duration/amplitude filter. Same logic as ``cycles_for_segment``.
    """
    cd = get_cycle_data(theta_imf, fs=int(fs))
    if cd is None or "IA" not in cd:
        return []
    amp_thresh = float(np.percentile(cd["IA"], amp_percentile))
    conds = [
        "is_good==1",
        f"duration_samples<{fs / theta_band[0]}",
        f"duration_samples>{fs / theta_band[1]}",
        f"max_amp>{amp_thresh}",
    ]
    filt = get_cycles_with_conditions(cd["cycles"], conds)
    return _cycle_bounds_from_filtered_cycles(filt)


def find_session_with_substate(rat_id: int, substate: str,
                               min_seconds: float):
    """Walk this rat's sessions; return the first session whose chosen
    substate has at least one interval of ``min_seconds``.
    """
    cfg = load_default_emd_config()
    sessions = list_rat_sessions(rat_id)
    for sess in sessions:
        try:
            hpc, hypno, fs = get_data(sess["hpc_path"], sess["state_path"],
                                      type="hpc")
            pfc, _, _ = get_data(sess["pfc_path"], sess["state_path"],
                                 type="pfc")
        except Exception as ex:
            continue
        L = min(len(hpc), len(pfc))
        hpc = np.asarray(hpc[:L], dtype=float)
        pfc = np.asarray(pfc[:L], dtype=float)
        try:
            phasic_iv, tonic_iv, _ = extract_pt_intervals(hpc, hypno, fs=fs)
        except Exception:
            continue
        iv = phasic_iv if substate == "phasic" else tonic_iv
        if iv is None or len(iv) == 0:
            continue
        try:
            starts = np.asarray(iv.start, dtype=float)
            ends = np.asarray(iv.end, dtype=float)
        except Exception:
            try:
                starts = np.asarray(iv["start"], dtype=float)
                ends = np.asarray(iv["end"], dtype=float)
            except Exception:
                continue
        durs = ends - starts
        if not np.any(durs >= min_seconds):
            continue
        return dict(sess=sess, hpc=hpc, pfc=pfc, fs=int(fs),
                    phasic_iv=phasic_iv, tonic_iv=tonic_iv,
                    iv=iv, cfg=cfg)
    return None


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rat", type=int, default=3,
                        help="rat id (default: 3)")
    parser.add_argument("--substate", choices=["phasic", "tonic"],
                        default="tonic",
                        help="which REM substate to inspect (default: tonic)")
    parser.add_argument("--n_seconds", type=float, default=5.0,
                        help="how many seconds of the interval to plot "
                             "(default: 5 s)")
    parser.add_argument("--out", default="/tmp/emd_theta_inspect.png",
                        help="output PNG path")
    parser.add_argument("--interval_idx", type=int, default=None,
                        help="optional: choose a specific interval index "
                             "into the substate IntervalSet (default: "
                             "longest)")
    args = parser.parse_args()

    bundle = find_session_with_substate(args.rat, args.substate,
                                        min_seconds=min(args.n_seconds, 1.0))
    if bundle is None:
        print(f"no usable session for rat {args.rat} / {args.substate}")
        return 1
    sess = bundle["sess"]; hpc = bundle["hpc"]; pfc = bundle["pfc"]
    fs = bundle["fs"]; iv = bundle["iv"]; cfg = bundle["cfg"]

    print(f"session: {sess['condition']}/{sess['folder']}/{sess['sub']}")
    print(f"fs = {fs} Hz")

    # Run EMD on the WHOLE IntervalSet for this substate (matches the
    # main PAC pipeline). extract_imfs_by_pt_intervals returns one IMF
    # matrix per interval, in IntervalSet order.
    hpc_imfs_list, hpc_freqs_list, _ = extract_imfs_by_pt_intervals(
        hpc, fs, iv, cfg, return_imfs_freqs=True,
    )
    pfc_imfs_list, pfc_freqs_list, _ = extract_imfs_by_pt_intervals(
        pfc, fs, iv, cfg, return_imfs_freqs=True,
    )

    if args.interval_idx is None:
        # pick the longest interval (most context for plotting)
        durs = np.array([m.shape[0] for m in hpc_imfs_list])
        sel = int(np.argmax(durs))
    else:
        sel = int(args.interval_idx)
    hpc_imfs = hpc_imfs_list[sel]
    hpc_freqs = hpc_freqs_list[sel]
    pfc_imfs = pfc_imfs_list[sel]
    pfc_freqs = pfc_freqs_list[sel]
    try:
        start_sec = float(iv.loc[sel, "start"])
        end_sec = float(iv.loc[sel, "end"])
    except Exception:
        start_sec = float(np.asarray(iv.start)[sel])
        end_sec = float(np.asarray(iv.end)[sel])

    print(f"interval {sel}: {start_sec:.2f}s -> {end_sec:.2f}s "
          f"({end_sec - start_sec:.2f} s long)")

    # Trim to args.n_seconds for display
    s_idx_abs = int(round(start_sec * fs))
    e_idx_abs = s_idx_abs + int(round(args.n_seconds * fs))
    e_idx_abs = min(e_idx_abs,
                    s_idx_abs + hpc_imfs.shape[0],
                    s_idx_abs + pfc_imfs.shape[0])
    L = e_idx_abs - s_idx_abs
    if L < 2 * fs:
        print(f"warning: only {L/fs:.2f}s of data available, less than "
              f"requested {args.n_seconds}s")

    hpc_seg = hpc[s_idx_abs:e_idx_abs]
    pfc_seg = pfc[s_idx_abs:e_idx_abs]
    hpc_imfs = hpc_imfs[:L]
    pfc_imfs = pfc_imfs[:L]
    t = np.arange(L) / fs

    # Pick theta IMF in each region
    hpc_theta_idx = choose_theta_imf_index(hpc_imfs, hpc_freqs,
                                            theta_band=THETA_BAND_HZ,
                                            prefer_index=THETA_IMF_PREFER)
    pfc_theta_idx = choose_theta_imf_index(pfc_imfs, pfc_freqs,
                                            theta_band=THETA_BAND_HZ,
                                            prefer_index=THETA_IMF_PREFER)
    hpc_theta_imf = hpc_imfs[:, hpc_theta_idx]
    pfc_theta_imf = pfc_imfs[:, pfc_theta_idx]

    print()
    print("HPC IMF mean freqs (Hz):", np.round(hpc_freqs, 2))
    print(f"  -> theta IMF index = {hpc_theta_idx}  "
          f"({hpc_freqs[hpc_theta_idx]:.2f} Hz)")
    print("PFC IMF mean freqs (Hz):", np.round(pfc_freqs, 2))
    print(f"  -> theta IMF index = {pfc_theta_idx}  "
          f"({pfc_freqs[pfc_theta_idx]:.2f} Hz)")

    # Get cycle bounds (filtered) for each region. Note: these are run on
    # the WHOLE-INTERVAL IMF, so they extend beyond our display window.
    hpc_bounds_full = filter_cycles_from_theta_imf(hpc_imfs[:, hpc_theta_idx]
                                                    if hpc_imfs.shape[0] >= L
                                                    else hpc_theta_imf, fs)
    pfc_bounds_full = filter_cycles_from_theta_imf(pfc_imfs[:, pfc_theta_idx]
                                                    if pfc_imfs.shape[0] >= L
                                                    else pfc_theta_imf, fs)
    # restrict to the display window
    hpc_bounds = [(s, e) for s, e in hpc_bounds_full
                  if 0 <= s < L]
    pfc_bounds = [(s, e) for s, e in pfc_bounds_full
                  if 0 <= s < L]

    print()
    print(f"good cycles in display window ({args.n_seconds}s):")
    print(f"  HPC: {len(hpc_bounds)} cycles")
    print(f"  PFC: {len(pfc_bounds)} cycles")

    # Compute coherence/correlation between the two theta IMFs as a
    # sanity check that they represent the "same" oscillation.
    corr = float(np.corrcoef(hpc_theta_imf, pfc_theta_imf)[0, 1])
    # Phase lag estimate from cross-correlation peak
    cc = np.correlate(hpc_theta_imf - hpc_theta_imf.mean(),
                       pfc_theta_imf - pfc_theta_imf.mean(),
                       mode="full")
    lags = np.arange(-(L - 1), L) / fs
    peak_lag_sec = float(lags[int(np.argmax(cc))])
    print()
    print(f"correlation(HPC theta IMF, PFC theta IMF) = {corr:+.3f}")
    print(f"peak cross-correlation lag (HPC vs PFC)   = {peak_lag_sec*1000:+.1f} ms"
          f"  (positive: HPC leads PFC)")

    # ----------------------------------------------------------------------
    # Butterworth 5-12 Hz bandpass on raw PFC LFP (4th order, zero-phase).
    # Used in the third subplot, with HPC EMD cycle starts/ends overlaid.
    # ----------------------------------------------------------------------
    nyq = 0.5 * fs
    sos_theta = butter(4, [5.0 / nyq, 12.0 / nyq],
                       btype="bandpass", output="sos")
    pfc_bp = sosfiltfilt(sos_theta, pfc_seg)

    # ----------------------------------------------------------------------
    # Plot
    # ----------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(14, 9.5), sharex=True,
                              constrained_layout=True)

    def _scale_for_overlay(raw: np.ndarray, imf: np.ndarray, k: float = 0.8):
        """Scale ``imf`` so its peak-to-peak ≈ k × raw's peak-to-peak,
        for nicer visual overlay. Returns the scaled IMF copy.
        """
        raw_pp = np.percentile(raw, 99) - np.percentile(raw, 1)
        imf_pp = np.percentile(imf, 99) - np.percentile(imf, 1)
        if imf_pp <= 0:
            return imf
        return imf * (k * raw_pp / imf_pp)

    # ---------------- PANEL 1: HPC ----------------
    ax_hpc = axes[0]
    hpc_theta_disp = _scale_for_overlay(hpc_seg, hpc_theta_imf, k=0.8)
    ax_hpc.plot(t, hpc_seg, color="lightgray", lw=0.7,
                 label="HPC raw LFP", zorder=1)
    ax_hpc.plot(t, hpc_theta_disp, color="tab:red", lw=1.3, alpha=0.9,
                 label="HPC theta IMF (scaled)", zorder=2)
    for s, e in hpc_bounds:
        ax_hpc.axvline(t[s], color="tab:green", lw=0.4, alpha=0.45)
        ax_hpc.plot(t[s], hpc_theta_disp[s], "o", color="tab:green",
                     markersize=8, markeredgecolor="black",
                     markeredgewidth=0.6, zorder=5)
    ax_hpc.plot([], [], "o", color="tab:green", markeredgecolor="black",
                 markersize=8, label="HPC EMD cycle start")
    ax_hpc.set_title(
        f"HPC: theta IMF idx = {hpc_theta_idx}, "
        f"mean freq = {hpc_freqs[hpc_theta_idx]:.2f} Hz  |  "
        f"{len(hpc_bounds)} cycles"
    )
    ax_hpc.set_ylabel("HPC amplitude (a.u.)")
    ax_hpc.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax_hpc.grid(alpha=0.2)

    # ---------------- PANEL 2: PFC with HPC theta IMF overlaid ----------------
    ax_pfc = axes[1]
    pfc_theta_disp = _scale_for_overlay(pfc_seg, pfc_theta_imf, k=0.8)
    # Scale HPC theta IMF to the SAME visual range as the PFC theta IMF
    # so they can be compared on the same axes.
    hpc_on_pfc_disp = _scale_for_overlay(pfc_seg, hpc_theta_imf, k=0.8)
    ax_pfc.plot(t, pfc_seg, color="lightgray", lw=0.7,
                 label="PFC raw LFP", zorder=1)
    ax_pfc.plot(t, pfc_theta_disp, color="tab:red", lw=1.3, alpha=0.9,
                 label="PFC theta IMF (scaled)", zorder=2)
    ax_pfc.plot(t, hpc_on_pfc_disp, color="tab:purple", lw=1.3, alpha=0.85,
                 label="HPC theta IMF (overlaid, scaled)", zorder=2)
    # PFC cycle starts (green ● on PFC theta IMF)
    for s, e in pfc_bounds:
        ax_pfc.axvline(t[s], color="tab:green", lw=0.4, alpha=0.45)
        ax_pfc.plot(t[s], pfc_theta_disp[s], "o", color="tab:green",
                     markersize=8, markeredgecolor="black",
                     markeredgewidth=0.6, zorder=5)
    # HPC cycle starts (blue ● on HPC theta IMF curve)
    for s, e in hpc_bounds:
        ax_pfc.axvline(t[s], color="tab:blue", lw=0.3,
                        alpha=0.35, linestyle="--")
        ax_pfc.plot(t[s], hpc_on_pfc_disp[s], "o", color="tab:blue",
                     markersize=8, markeredgecolor="black",
                     markeredgewidth=0.6, zorder=5)
    ax_pfc.plot([], [], "o", color="tab:green", markeredgecolor="black",
                 markersize=8, label="PFC EMD cycle start")
    ax_pfc.plot([], [], "o", color="tab:blue", markeredgecolor="black",
                 markersize=8, label="HPC EMD cycle start")
    ax_pfc.set_title(
        f"PFC: theta IMF idx = {pfc_theta_idx}, "
        f"mean freq = {pfc_freqs[pfc_theta_idx]:.2f} Hz  |  "
        f"{len(pfc_bounds)} PFC cycles, {len(hpc_bounds)} HPC cycles"
    )
    ax_pfc.set_ylabel("PFC amplitude (a.u.)")
    ax_pfc.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax_pfc.grid(alpha=0.2)

    # ----------------------------------------------------------------------
    # Third panel: PFC bandpass 5-12 Hz, with HPC EMD cycle starts/ends.
    # Diagnostic: do HPC's cycle boundaries delimit PFC theta cycles?
    # ----------------------------------------------------------------------
    ax3 = axes[2]
    ax3.plot(t, pfc_seg, color="lightgray", lw=0.7,
              label="PFC raw LFP", zorder=1)
    ax3.plot(t, pfc_bp, color="tab:blue", lw=1.2, alpha=0.9,
              label="PFC bandpass 5-12 Hz (Butterworth, order 4)", zorder=2)
    for s, e in hpc_bounds:
        e_clamp = min(e, len(t) - 1)
        ax3.axvline(t[s], color="tab:green", lw=0.3, alpha=0.4)
        ax3.axvline(t[e_clamp], color="tab:red", lw=0.3, alpha=0.4)
        ax3.plot(t[s], pfc_bp[s], "o", color="tab:green",
                  markersize=8, markeredgecolor="black",
                  markeredgewidth=0.6, zorder=6)
        ax3.plot(t[e_clamp], pfc_bp[e_clamp], "s", color="tab:red",
                  markersize=7, markeredgecolor="black",
                  markeredgewidth=0.6, zorder=6)
    ax3.plot([], [], "o", color="tab:green", markeredgecolor="black",
              markersize=8, label="HPC EMD cycle START")
    ax3.plot([], [], "s", color="tab:red", markeredgecolor="black",
              markersize=7, label="HPC EMD cycle END")
    ax3.set_title("PFC bandpass 5-12 Hz with HPC EMD cycle starts/ends  |  "
                  f"{len(hpc_bounds)} HPC cycles in window")
    ax3.set_ylabel("PFC amplitude (a.u.)")
    ax3.set_xlabel("time within interval (s)")
    ax3.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax3.grid(alpha=0.2)

    fig.suptitle(
        f"Rat {args.rat} | {args.substate} REM interval #{sel} "
        f"({start_sec:.1f}-{end_sec:.1f}s, displaying first {args.n_seconds:.1f}s)\n"
        f"corr(HPC, PFC theta IMF) = {corr:+.3f}, "
        f"peak cross-corr lag = {peak_lag_sec*1000:+.1f} ms (positive: HPC leads)",
        fontsize=11,
    )
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"\nsaved figure to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
