"""Markov-chain analysis of TG state sequences.

For each behavioural condition (here: phasic vs tonic REM) we compute:
  * state occurrence probabilities (4-vector)
  * 4 x 4 transition matrix, where T[i, j] = P(next state = j | current
    state = i). Rows sum to 1 (excluding self-loops? No, including them,
    which matches Zhang's "probability of remaining in the same state").

Cycle sequences are formed *within contiguous REM episodes*. Transitions
that would cross from one REM episode to another are not counted.

Cycles must also be tagged with the behavioural condition (phasic vs
tonic) -- see ``utils_intervals.assign_cycles_to_intervals``. A
transition is counted in the "phasic" matrix iff both the current and
next cycle are inside the same phasic interval (i.e. no boundary
crossings between phasic <-> tonic / phasic <-> nothing).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TransitionResult:
    counts: np.ndarray              # (n_states, n_states)
    transition_matrix: np.ndarray   # (n_states, n_states), row-normalised
    occurrence: np.ndarray          # (n_states,) marginal probabilities
    n_cycles: int                   # number of cycles contributing
    n_transitions: int              # number of valid transitions counted


def transition_from_labelled_runs(runs: list[np.ndarray],
                                  n_states: int = 4) -> TransitionResult:
    """Build transitions from a list of per-episode label sequences.

    Each entry of ``runs`` is a 1-D array of state labels for cycles in
    one contiguous REM-substate interval (phasic or tonic). Transitions
    are counted only within each run.
    """
    counts = np.zeros((n_states, n_states), dtype=float)
    occurrence_counts = np.zeros(n_states, dtype=float)
    n_cycles = 0
    n_trans = 0
    for seq in runs:
        seq = np.asarray(seq, dtype=int)
        seq = seq[(seq >= 0) & (seq < n_states)]
        if len(seq) == 0:
            continue
        for s in seq:
            occurrence_counts[s] += 1
        n_cycles += len(seq)
        if len(seq) < 2:
            continue
        for a, b in zip(seq[:-1], seq[1:]):
            counts[a, b] += 1
            n_trans += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    T = counts / row_sums
    occ_total = occurrence_counts.sum()
    occ = occurrence_counts / occ_total if occ_total else np.full(n_states, np.nan)

    return TransitionResult(counts=counts, transition_matrix=T,
                            occurrence=occ, n_cycles=int(n_cycles),
                            n_transitions=int(n_trans))


# ---------------------------------------------------------------------------
# Interval-aware cycle tagging
# ---------------------------------------------------------------------------

def cycles_in_interval(cycle_starts_sec: np.ndarray,
                       cycle_ends_sec: np.ndarray,
                       interval_starts_sec: np.ndarray,
                       interval_ends_sec: np.ndarray) -> np.ndarray:
    """Return ``(n_cycles,)`` int array with the index of the containing
    interval, or -1 if no interval fully contains the cycle.
    """
    n = len(cycle_starts_sec)
    out = np.full(n, -1, dtype=int)
    for i in range(n):
        cs = cycle_starts_sec[i]
        ce = cycle_ends_sec[i]
        mask = (interval_starts_sec <= cs) & (interval_ends_sec >= ce)
        idx = np.where(mask)[0]
        if len(idx):
            out[i] = int(idx[0])
    return out


def runs_within_intervals(labels: np.ndarray,
                          interval_membership: np.ndarray
                          ) -> list[np.ndarray]:
    """Split a label sequence into contiguous runs by interval membership.

    Cycles tagged with interval index ``-1`` (outside any interval) are
    dropped entirely so they cannot bridge transitions.
    """
    runs: list[np.ndarray] = []
    current_idx = -2
    current: list[int] = []
    for lab, iv in zip(labels, interval_membership):
        if iv == -1:
            if current:
                runs.append(np.asarray(current, dtype=int))
                current = []
                current_idx = -2
            continue
        if iv != current_idx:
            if current:
                runs.append(np.asarray(current, dtype=int))
            current = [int(lab)]
            current_idx = int(iv)
        else:
            current.append(int(lab))
    if current:
        runs.append(np.asarray(current, dtype=int))
    return runs


# ---------------------------------------------------------------------------
# Convenience: phasic vs tonic comparison from raw inputs
# ---------------------------------------------------------------------------

def transitions_by_substate(labels: np.ndarray,
                            substates: np.ndarray,
                            segment_ids: np.ndarray,
                            n_states: int = 4) -> dict[str, TransitionResult]:
    """Build per-substate transition matrices using segment IDs as the run key.

    This is the preferred Markov entry point when the cycle array is
    organised per-segment rather than chronologically.

    Each unique ``segment_id`` corresponds to one phasic OR tonic REM
    interval; all of its cycles are temporally consecutive within that
    interval. A run = all cycles sharing one segment_id. Transitions are
    counted only within a run, so transitions never cross a phasic/tonic
    or session boundary.

    Returns a dict ``{'phasic': TransitionResult, 'tonic': TransitionResult}``.
    """
    labels = np.asarray(labels)
    substates = np.asarray(substates)
    segment_ids = np.asarray(segment_ids)

    out: dict[str, TransitionResult] = {}
    for name in ("phasic", "tonic"):
        mask = substates == name
        if not np.any(mask):
            out[name] = transition_from_labelled_runs([], n_states=n_states)
            continue
        sub_labels = labels[mask]
        sub_seg = segment_ids[mask]
        # Group cycles by segment_id, preserving order
        unique_segs, first_idx = np.unique(sub_seg, return_index=True)
        # iterate in the order segments first appear in the array
        seg_order = unique_segs[np.argsort(first_idx)]
        runs: list[np.ndarray] = []
        for seg in seg_order:
            sel = sub_seg == seg
            runs.append(sub_labels[sel])
        out[name] = transition_from_labelled_runs(runs, n_states=n_states)
    return out


def phasic_vs_tonic_transitions(labels: np.ndarray,
                                cycle_starts_sec: np.ndarray,
                                cycle_ends_sec: np.ndarray,
                                phasic_starts: np.ndarray,
                                phasic_ends: np.ndarray,
                                tonic_starts: np.ndarray,
                                tonic_ends: np.ndarray,
                                n_states: int = 4) -> dict[str, TransitionResult]:
    """Build separate transition results for phasic and tonic REM."""
    phasic_iv = cycles_in_interval(cycle_starts_sec, cycle_ends_sec,
                                   phasic_starts, phasic_ends)
    tonic_iv = cycles_in_interval(cycle_starts_sec, cycle_ends_sec,
                                  tonic_starts, tonic_ends)

    runs_phasic = runs_within_intervals(labels, phasic_iv)
    runs_tonic = runs_within_intervals(labels, tonic_iv)

    return dict(
        phasic=transition_from_labelled_runs(runs_phasic, n_states=n_states),
        tonic=transition_from_labelled_runs(runs_tonic, n_states=n_states),
    )
