"""Clustering of single-theta-cycle FPPs into theta-gamma (TG) states.

Faithful to Zhang et al. 2019:

  * **k-means** with Pearson correlation distance (D = 1 - r) and k-means++
    initialisation (sklearn). k = 4 by default. Matches Demo's
    ``kmeans(..., 'distance', 'correlation', 'Maxiter', 10000)``.

  * **Cluster labelling** by the "gravity" features of each cluster's
    mean FPP (m-FPP):
        - gamma field = pixels >= 95% of the m-FPP peak
        - gravity frequency = power-weighted mean of frequencies in the
          gamma field
        - gravity phase = power-weighted *circular* mean of theta phases
          in the gamma field
    The four clusters are then sorted into S, M, EF, LF gamma using the
    rule in Zhang's PhaseFreSort.m:
        - sort all four by gravity frequency ascending
        - the two lowest frequencies become S and M
        - the two highest frequencies are split into early (EF) and late
          (LF) by their gravity phase relative to the M-gamma phase

  * Optional **community clustering** (Louvain modularity, sklearn
    network-x-free implementation via python-louvain) for an unsupervised
    estimate of k.  k-means is used for the final labelling because it is
    more robust to channel position and faster.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


STATE_NAMES = ["S-gamma", "M-gamma", "EF-gamma", "LF-gamma"]


# ---------------------------------------------------------------------------
# k-means with Pearson correlation distance
# ---------------------------------------------------------------------------

def _row_zscore(X: np.ndarray) -> np.ndarray:
    """Per-row z-score; rows with zero std are returned as zeros."""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, ddof=0, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    Z = (X - mu) / sd
    return Z


def _correlation_distance_kmeans(X: np.ndarray, k: int,
                                 n_init: int = 20,
                                 max_iter: int = 10_000,
                                 random_state: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """k-means with Pearson correlation distance.

    Trick: 1 - corr(a, b) = 0.5 * || z(a) - z(b) ||^2 / n, where z is the
    row-wise standardisation. So Euclidean k-means on z-scored, L2-
    normalised rows is equivalent to correlation-distance k-means.
    """
    Xz = _row_zscore(X)
    Xz = normalize(Xz, norm="l2", axis=1)
    km = KMeans(n_clusters=k, init="k-means++", n_init=n_init,
                max_iter=max_iter, random_state=random_state, algorithm="lloyd")
    labels = km.fit_predict(Xz)
    centroids = km.cluster_centers_       # in z-normalised space
    return labels, centroids


# ---------------------------------------------------------------------------
# Mean FPP per cluster and gravity features
# ---------------------------------------------------------------------------

def mean_fpp_per_cluster(fpps: np.ndarray, labels: np.ndarray,
                        n_clusters: int) -> np.ndarray:
    """Return ``(n_clusters, n_freq, n_phase_bins)`` array of m-FPPs."""
    n_freq = fpps.shape[1]
    n_phase = fpps.shape[2]
    out = np.empty((n_clusters, n_freq, n_phase), dtype=float)
    for c in range(n_clusters):
        sel = labels == c
        if not np.any(sel):
            out[c] = np.nan
        else:
            out[c] = fpps[sel].mean(axis=0)
    return out


def mean_fpp_per_cluster_substate(fpps: np.ndarray,
                                  labels: np.ndarray,
                                  substates: np.ndarray,
                                  n_clusters: int = 4
                                  ) -> dict[str, np.ndarray]:
    """Per-substate m-FPPs.

    Returns a dict with keys ``'phasic'`` and ``'tonic'`` mapping to
    ``(n_clusters, n_freq, n_phase_bins)`` arrays of mean FPPs computed
    over only the cycles whose substate matches. Empty cells get NaN.
    """
    substates = np.asarray(substates)
    out: dict[str, np.ndarray] = {}
    for name in ("phasic", "tonic"):
        mask = substates == name
        if not np.any(mask):
            out[name] = np.full((n_clusters, fpps.shape[1], fpps.shape[2]),
                                 np.nan, dtype=float)
            continue
        out[name] = mean_fpp_per_cluster(fpps[mask], labels[mask], n_clusters)
    return out


def _gravity_field_mask(m_fpp: np.ndarray, threshold_ratio: float) -> np.ndarray:
    """Boolean mask of pixels at/above ``threshold_ratio`` times the peak."""
    peak = np.nanmax(m_fpp)
    return m_fpp >= threshold_ratio * peak


def gravity_features(m_fpp: np.ndarray,
                     frequencies: np.ndarray,
                     phase_centers_rad: np.ndarray,
                     threshold_ratio: float = 0.95) -> dict:
    """Compute gravity frequency / phase / their dispersion for one m-FPP.

    Returns dict with keys:
        ``mask``           : boolean array, same shape as m_fpp
        ``gravity_freq``   : float, power-weighted mean frequency (Hz)
        ``gravity_phase``  : float, power-weighted circular mean phase (rad)
        ``freq_std``       : float, power-weighted frequency std (Hz)
        ``phase_std``      : float, circular std of phase (rad)
        ``peak_freq``      : float, frequency at the global peak
        ``peak_phase``     : float, theta phase at the global peak
        ``n_pixels``       : int, # pixels in the gamma field
    """
    mask = _gravity_field_mask(m_fpp, threshold_ratio)
    weights = m_fpp[mask].astype(float)
    if weights.size == 0 or np.all(weights <= 0):
        return dict(mask=mask, gravity_freq=np.nan, gravity_phase=np.nan,
                    freq_std=np.nan, phase_std=np.nan,
                    peak_freq=np.nan, peak_phase=np.nan, n_pixels=0)

    weights = weights - weights.min()
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    w_norm = weights / weights.sum()

    freq_grid, phase_grid = np.meshgrid(frequencies, phase_centers_rad,
                                        indexing="ij")
    f_in = freq_grid[mask]
    p_in = phase_grid[mask]

    gravity_freq = float(np.sum(w_norm * f_in))
    freq_var = float(np.sum(w_norm * (f_in - gravity_freq) ** 2))
    freq_std = float(np.sqrt(max(freq_var, 0.0)))

    # circular weighted mean and std (Berens CircStat conventions)
    c = float(np.sum(w_norm * np.cos(p_in)))
    s = float(np.sum(w_norm * np.sin(p_in)))
    R = float(np.hypot(c, s))
    gravity_phase = float(np.arctan2(s, c))
    phase_std = float(np.sqrt(max(-2.0 * np.log(R), 0.0))) if R > 0 else np.nan

    peak_idx = np.unravel_index(np.nanargmax(m_fpp), m_fpp.shape)
    peak_freq = float(frequencies[peak_idx[0]])
    peak_phase = float(phase_centers_rad[peak_idx[1]])

    return dict(mask=mask, gravity_freq=gravity_freq, gravity_phase=gravity_phase,
                freq_std=freq_std, phase_std=phase_std,
                peak_freq=peak_freq, peak_phase=peak_phase,
                n_pixels=int(mask.sum()))


def _circ_dist_signed(target: float, ref: float) -> float:
    """Signed circular distance, target - ref, wrapped to (-pi, pi]."""
    d = (target - ref + np.pi) % (2.0 * np.pi) - np.pi
    return float(d)


def sort_clusters_into_states(gravity_freqs: Sequence[float],
                              gravity_phases: Sequence[float]) -> np.ndarray:
    """Return permutation that maps the input order to [S, M, EF, LF].

    Implements Zhang's PhaseFreSort.m:
      1. Sort by gravity frequency ascending. The two lowest become S, M.
      2. Among the two high-frequency clusters, the one whose phase is at
         an *earlier* point in the theta cycle than the M-gamma phase
         becomes EF; the other becomes LF. "Earlier" = larger signed
         circular distance ``M_phase - candidate_phase`` mod 2 pi
         (i.e. the candidate that comes first when sweeping the theta
         cycle forward from the EF position to the M position).
    """
    gravity_freqs = np.asarray(gravity_freqs, dtype=float)
    gravity_phases = np.asarray(gravity_phases, dtype=float)
    order_by_freq = np.argsort(gravity_freqs)  # ascending
    s_idx, m_idx = order_by_freq[0], order_by_freq[1]
    fast_indices = order_by_freq[2:]

    m_phase = gravity_phases[m_idx]
    # Signed circular distance: how far AFTER the candidate phase does the
    # M-gamma phase come?  Larger (positive, closer to 2 pi) = candidate is
    # earlier in the cycle than M-gamma.
    distances_after_m = np.array([
        (m_phase - gravity_phases[i]) % (2.0 * np.pi) for i in fast_indices
    ])
    # EF = the one further ahead of M (largest distance_after_m)
    ef_idx = fast_indices[int(np.argmax(distances_after_m))]
    lf_idx = fast_indices[int(np.argmin(distances_after_m))]

    return np.array([s_idx, m_idx, ef_idx, lf_idx], dtype=int)


# ---------------------------------------------------------------------------
# Public clustering result
# ---------------------------------------------------------------------------

@dataclass
class TGClusterResult:
    """Outcome of TG clustering for one set of cycles."""
    labels: np.ndarray              # (n_cycles,) integer state label in [0..3] = [S, M, EF, LF]
    m_fpps: np.ndarray              # (4, n_freq, n_phase_bins) sorted as S, M, EF, LF
    gravity_freqs: np.ndarray       # (4,) Hz, ordered S, M, EF, LF
    gravity_phases: np.ndarray      # (4,) rad
    freq_stds: np.ndarray
    phase_stds: np.ndarray
    masks: np.ndarray               # (4, n_freq, n_phase_bins) boolean gamma-field masks
    raw_labels: np.ndarray          # original k-means labels before sorting
    permutation: np.ndarray         # raw_label -> sorted_label permutation
    frequencies: np.ndarray
    phase_centers_rad: np.ndarray

    def state_index_of(self, raw_label: int) -> int:
        """Map a raw kmeans label to the sorted index in {0,1,2,3}."""
        return int(np.where(self.permutation == raw_label)[0][0])


def cluster_fpps_into_tg_states(fpps: np.ndarray,
                                frequencies: np.ndarray,
                                phase_centers_rad: np.ndarray,
                                k: int = 4,
                                threshold_ratio: float = 0.95,
                                random_state: int = 0,
                                n_init: int = 20,
                                max_iter: int = 10_000) -> TGClusterResult:
    """Cluster (n_cycles, n_freq, n_phase) FPPs and label as S/M/EF/LF.

    The full Zhang pipeline:
      1. Flatten each FPP, run k-means with correlation distance.
      2. Build m-FPP per cluster.
      3. Compute gravity features.
      4. Sort clusters into [S, M, EF, LF] order.
    """
    n_cycles, n_freq, n_phase = fpps.shape
    flat = fpps.reshape(n_cycles, -1)

    raw_labels, _ = _correlation_distance_kmeans(
        flat, k, n_init=n_init, max_iter=max_iter, random_state=random_state,
    )

    m_fpps = mean_fpp_per_cluster(fpps, raw_labels, k)
    feats = [gravity_features(m_fpps[c], frequencies, phase_centers_rad,
                              threshold_ratio=threshold_ratio)
             for c in range(k)]
    gravity_freqs = np.array([f["gravity_freq"] for f in feats])
    gravity_phases = np.array([f["gravity_phase"] for f in feats])

    perm = sort_clusters_into_states(gravity_freqs, gravity_phases)

    # Remap raw labels to sorted [0, 1, 2, 3]
    sorted_labels = np.empty_like(raw_labels)
    for sorted_idx, raw_idx in enumerate(perm):
        sorted_labels[raw_labels == raw_idx] = sorted_idx

    sorted_m_fpps = m_fpps[perm]
    sorted_feats = [feats[i] for i in perm]
    sorted_freqs = np.array([f["gravity_freq"] for f in sorted_feats])
    sorted_phases = np.array([f["gravity_phase"] for f in sorted_feats])
    sorted_freq_stds = np.array([f["freq_std"] for f in sorted_feats])
    sorted_phase_stds = np.array([f["phase_std"] for f in sorted_feats])
    sorted_masks = np.stack([f["mask"] for f in sorted_feats], axis=0)

    return TGClusterResult(
        labels=sorted_labels,
        m_fpps=sorted_m_fpps,
        gravity_freqs=sorted_freqs,
        gravity_phases=sorted_phases,
        freq_stds=sorted_freq_stds,
        phase_stds=sorted_phase_stds,
        masks=sorted_masks,
        raw_labels=raw_labels,
        permutation=perm,
        frequencies=np.asarray(frequencies),
        phase_centers_rad=np.asarray(phase_centers_rad),
    )


# ---------------------------------------------------------------------------
# Optional: community clustering (k estimation)
# ---------------------------------------------------------------------------

def _build_correlation_adjacency(X: np.ndarray) -> np.ndarray:
    """Adjacency = Pearson correlation + 1 (avoids negative weights).

    Following Zhang: ``B = C + 1``; diagonal set to 0.
    """
    Xz = _row_zscore(X)
    Xz = normalize(Xz, norm="l2", axis=1)
    # correlation = dot product of row-normalised z-scored rows
    C = Xz @ Xz.T
    np.clip(C, -1.0, 1.0, out=C)
    B = C + 1.0
    np.fill_diagonal(B, 0.0)
    return B


def community_clustering(X: np.ndarray,
                         subsample: int | None = 5000,
                         random_state: int = 0,
                         resolution: float = 1.0) -> tuple[int, np.ndarray] | None:
    """Estimate number of communities via Louvain on a correlation graph.

    Uses ``python-louvain`` (``community`` package) if available;
    if not installed, returns ``None`` so the caller can fall back to a
    fixed k.

    Parameters
    ----------
    X : (n_samples, n_features) array, flattened FPPs.
    subsample : sample at most this many points for tractability (Zhang
        uses 5000 in his MATLAB demo).
    """
    try:
        import community as community_louvain  # python-louvain
        import networkx as nx
    except Exception:
        return None

    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    if subsample is not None and n > subsample:
        idx = rng.choice(n, size=subsample, replace=False)
    else:
        idx = np.arange(n)
    Xs = X[idx]
    B = _build_correlation_adjacency(Xs)

    G = nx.from_numpy_array(B)
    partition = community_louvain.best_partition(G, resolution=resolution,
                                                 random_state=random_state)
    labels = np.array([partition[i] for i in range(len(idx))], dtype=int)
    n_communities = int(labels.max() + 1)
    full = -np.ones(n, dtype=int)
    full[idx] = labels
    return n_communities, full
