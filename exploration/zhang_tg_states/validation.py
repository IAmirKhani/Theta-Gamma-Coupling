"""Validation of single-cycle TG state assignments.

Two procedures from Zhang et al. 2019:

1. **Intra- vs. inter-cluster correlation** (Methods, "Intra-cluster
   correlation versus inter-cluster correlation"). For each test cycle:
       rho_intra   = corr(FPP_i, m_FPP[k])           # k = assigned state
       rho_max_int = max_{j != k} corr(FPP_i, m_FPP[j])
   The bigger the gap, the more uniquely the cycle belongs to its
   assigned state. Most cycles should have a large positive gap; cycles
   close to zero are ambiguous (Zhang reports ~20% of cycles fit
   multiple states with gap < 0.05).

   We compute this under 5-fold cross-validation: m-FPPs are built from
   the training fold and the held-out cycles are scored.

2. **Cross-channel / cross-rat assignment cross-validation** (Methods,
   "Cross-validation for individual theta cycle assignment"). The test
   cycle is re-assigned to the state whose *reference* m-FPP (from a
   different rat/channel/session) gives the highest correlation. The
   new label is compared with the within-rat label and accuracy is the
   fraction of cycles whose label is preserved.

In our PFC-HPC RGS data the equivalent of "cross-channel" is
"cross-session" or "cross-rat". The helpers below are written to take
*any* reference set of m-FPPs from the user.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.model_selection import KFold

from .clustering import (
    TGClusterResult,
    cluster_fpps_into_tg_states,
    mean_fpp_per_cluster,
)


def _flat_corr(flat_a: np.ndarray, flat_b: np.ndarray) -> np.ndarray:
    """Pearson correlation between rows of ``flat_a`` and rows of ``flat_b``.

    ``flat_a`` is ``(n_a, d)`` and ``flat_b`` is ``(n_b, d)``. Returns
    ``(n_a, n_b)``.
    """
    A = flat_a - flat_a.mean(axis=1, keepdims=True)
    B = flat_b - flat_b.mean(axis=1, keepdims=True)
    A /= np.where(np.linalg.norm(A, axis=1, keepdims=True) == 0, 1.0,
                  np.linalg.norm(A, axis=1, keepdims=True))
    B /= np.where(np.linalg.norm(B, axis=1, keepdims=True) == 0, 1.0,
                  np.linalg.norm(B, axis=1, keepdims=True))
    return A @ B.T


# ---------------------------------------------------------------------------
# Intra- vs. inter-cluster correlation
# ---------------------------------------------------------------------------

@dataclass
class IntraInterResult:
    intra: np.ndarray           # (n_cycles,) intra-cluster correlation
    inter_max: np.ndarray       # (n_cycles,) max inter-cluster correlation
    labels: np.ndarray          # (n_cycles,) the label used for `intra`
    gap: np.ndarray             # intra - inter_max

    def fraction_above(self, gap_threshold: float) -> float:
        return float(np.mean(self.gap > gap_threshold))


def intra_inter_correlation(fpps: np.ndarray, labels: np.ndarray,
                            m_fpps: np.ndarray) -> IntraInterResult:
    """Intra and max-inter correlation for each cycle given m-FPPs."""
    n_cycles, n_freq, n_phase = fpps.shape
    n_states = m_fpps.shape[0]
    flat = fpps.reshape(n_cycles, -1)
    flat_m = m_fpps.reshape(n_states, -1)

    C = _flat_corr(flat, flat_m)  # (n_cycles, n_states)
    intra = C[np.arange(n_cycles), labels]
    masked = C.copy()
    masked[np.arange(n_cycles), labels] = -np.inf
    inter_max = masked.max(axis=1)
    return IntraInterResult(intra=intra, inter_max=inter_max,
                            labels=labels, gap=intra - inter_max)


def kfold_intra_inter(fpps: np.ndarray,
                      frequencies: np.ndarray,
                      phase_centers_rad: np.ndarray,
                      n_folds: int = 5,
                      k: int = 4,
                      random_state: int = 0) -> IntraInterResult:
    """5-fold CV version: train on 4 folds, score the held-out fold.

    The training-fold cycles are clustered and labelled; the test cycles
    are then assigned to the *closest* reference m-FPP and scored.
    """
    n_cycles = fpps.shape[0]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    intra_all = np.full(n_cycles, np.nan, dtype=float)
    inter_all = np.full(n_cycles, np.nan, dtype=float)
    labels_all = np.full(n_cycles, -1, dtype=int)

    flat_all = fpps.reshape(n_cycles, -1)
    for fold_id, (tr_idx, te_idx) in enumerate(kf.split(np.arange(n_cycles))):
        train_result = cluster_fpps_into_tg_states(
            fpps[tr_idx], frequencies, phase_centers_rad, k=k,
            random_state=random_state + fold_id,
        )
        m_flat = train_result.m_fpps.reshape(k, -1)
        # assign every test cycle to the reference m-FPP of highest corr
        C = _flat_corr(flat_all[te_idx], m_flat)
        te_labels = np.argmax(C, axis=1)
        rows = np.arange(len(te_idx))
        intra_all[te_idx] = C[rows, te_labels]
        masked = C.copy()
        masked[rows, te_labels] = -np.inf
        inter_all[te_idx] = masked.max(axis=1)
        labels_all[te_idx] = te_labels

    return IntraInterResult(intra=intra_all, inter_max=inter_all,
                            labels=labels_all, gap=intra_all - inter_all)


# ---------------------------------------------------------------------------
# Cross-channel / cross-rat assignment accuracy
# ---------------------------------------------------------------------------

def assign_from_reference(fpps: np.ndarray,
                          reference_m_fpps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Assign each cycle to the state of the most correlated reference m-FPP.

    Returns ``(labels, corr_matrix)`` where ``labels`` is ``(n_cycles,)``
    and ``corr_matrix`` is ``(n_cycles, n_states)``.
    """
    n_cycles, n_freq, n_phase = fpps.shape
    n_states = reference_m_fpps.shape[0]
    flat = fpps.reshape(n_cycles, -1)
    flat_ref = reference_m_fpps.reshape(n_states, -1)
    C = _flat_corr(flat, flat_ref)
    return np.argmax(C, axis=1), C


def cross_dataset_accuracy(test_fpps: np.ndarray,
                           test_native_labels: np.ndarray,
                           reference_m_fpps: np.ndarray) -> dict:
    """Compare reference-based labels to a 'native' label set.

    Both label conventions must already be in the same S/M/EF/LF order.
    Returns a dict with ``accuracy`` and a 4x4 ``confusion`` matrix.
    """
    ref_labels, _ = assign_from_reference(test_fpps, reference_m_fpps)
    n_states = reference_m_fpps.shape[0]
    confusion = np.zeros((n_states, n_states), dtype=int)
    for native, ref in zip(test_native_labels, ref_labels):
        if 0 <= native < n_states and 0 <= ref < n_states:
            confusion[native, ref] += 1
    total = confusion.sum()
    accuracy = float(np.trace(confusion) / total) if total > 0 else float("nan")
    return dict(accuracy=accuracy, confusion=confusion, ref_labels=ref_labels)


def pairwise_cross_dataset(per_dataset_fpps: Sequence[np.ndarray],
                           per_dataset_labels: Sequence[np.ndarray],
                           per_dataset_m_fpps: Sequence[np.ndarray]) -> np.ndarray:
    """Build an n x n accuracy matrix across datasets (rats/sessions).

    Entry [i, j] = accuracy of using dataset ``j``'s m-FPPs as reference
    to classify dataset ``i``'s cycles, vs. dataset ``i``'s own labels.
    Diagonal is 1.0 by construction.
    """
    n = len(per_dataset_fpps)
    out = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                out[i, j] = 1.0
                continue
            res = cross_dataset_accuracy(per_dataset_fpps[i],
                                         per_dataset_labels[i],
                                         per_dataset_m_fpps[j])
            out[i, j] = res["accuracy"]
    return out
