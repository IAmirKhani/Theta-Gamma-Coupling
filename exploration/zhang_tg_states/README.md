# Zhang et al. 2019 TG-state pipeline — PFC ↔ HPC, phasic vs tonic REM

Re-implementation of the analysis pipeline from

> Zhang L, Lee J, Rozell CJ, Singer AC (2019). *Sub-second dynamics of
> theta-gamma coupling in hippocampal CA1.* eLife 8:e44320.

adapted to this project's RGS dataset. Two substantive substitutions vs
the paper:

| Paper                      | Here                                |
|----------------------------|-------------------------------------|
| CA1–CA3 / CA1–EC LFP-LFP   | **PFC ↔ HPC** LFP-LFP                |
| Pre / track / post awake   | **Phasic vs tonic REM** substates    |

No spike-based analyses are reproduced (no spikes in this dataset).

## Modules

```
fpp.py           wavelet + theta-phase FPP construction (81 freqs × 20 phases)
clustering.py    k-means (correlation distance) + gravity-based S/M/EF/LF labels
validation.py    intra/inter correlation, 5-fold CV, cross-rat accuracy
markov.py        4×4 transition matrices and state occurrences
ppc.py           wavelet cross-spectrum, V-stat PPC, conditioned on TG state
plotting.py      heatmaps and summary plots (hot colormap on PPC by request)
data_pipeline.py per-rat ingestion using src.utils.extract_pt_intervals
```

## Faithful reproductions of the paper's methods

* **Wavelet FPP**: complex Morlet CWT → magnitude squared → sequential
  boxcar smoothing (`±8 ms × ±2 Hz`, matching Zhang's `ntw=11`,
  `nsw=3`) → √ → z-score per frequency. Theta phase from the *unwrapped*
  Hilbert phase of 5–10 Hz bandpassed HPC LFP. 20 equal theta-phase
  bins per cycle.
* **K-means** with Pearson correlation distance (D = 1 − r), k-means++,
  10 000 max iterations, k = 4.
* **Gravity labelling**: pixels at ≥ 95 % of the m-FPP peak define the
  gamma field; gravity frequency = power-weighted mean; gravity phase
  = power-weighted *circular* mean. Cluster sort by gravity frequency
  ascending → S, M, and the two fast clusters → split EF (earlier
  than M-gamma phase) vs LF.
* **Intra-vs inter-cluster correlation** under 5-fold cross-validation.
* **Cross-rat accuracy**: classify rat *i*'s cycles using rat *j*'s
  reference m-FPPs; compare to native labels; build a pairwise matrix.
* **Markov transitions** with cycle sequences broken at substate
  boundaries (no transition crosses a phasic / tonic boundary).
* **LFP-LFP PPC**: per-cycle 81 × 20 wavelet cross-spectrum (PFC × HPC*),
  angle, V-statistic across cycles (unbiased PPC). Phase-window pooling
  uses the gravity-centered window `[gravity − 7σ, gravity + σ]` for
  PPC(f) curves.

## Running

The driving notebook is `exploration/zhang_tg_states_pfc_hpc.ipynb` (built
by `exploration/_build_zhang_tg_states_pfc_hpc.py`). Use the
`thetaGamma` conda env.

```python
from zhang_tg_states.data_pipeline import process_rat
agg = process_rat(rat_id=3)
# -> agg.fpps  : (n_cycles, 81, 20)
# -> agg.cross_spectrum.angles  : (n_cycles, 81, 20)
# -> agg.cycle_substates : 'phasic' or 'tonic' per cycle

from zhang_tg_states import cluster_fpps_into_tg_states, state_conditioned_ppc
cl = cluster_fpps_into_tg_states(agg.fpps.fpps, agg.frequencies,
                                  agg.phase_centers_rad)
ppc_res = state_conditioned_ppc(agg.cross_spectrum.angles, cl.labels)
```

## Outputs

Pickle files saved to this folder:

* `tg_states_positive_pfc_hpc.pkl` — RGS14 rats (3, 4, 7, 8)
* `tg_states_control_pfc_hpc.pkl` — WT rats (1, 2, 6, 9)

Each contains per-rat clusters, intra/inter validation, transition
matrices for phasic and tonic, and state-conditioned PPC matrices.
