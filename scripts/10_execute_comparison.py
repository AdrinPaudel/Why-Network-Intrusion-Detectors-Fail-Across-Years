"""
10_execute_comparison.py — Cross-dataset comparison executor for NIDS features (Branch B).

PURPOSE:
  Execute script 9's per-feature plan across the two years and emit the 2-D verdict:
  marginal shift (Axis 1) crossed with separation-change (Axis 2) -> one of
  {stable, shifted, flipped, collapsed, weak, restructured}. Produces rich per-feature detail
  (Layer A), bounded cross-feature-comparable scalars (Layer B), overlap/Q-Q plots, and the
  E1 cross-metric agreement check that step 11 consumes. Pure consumer of script 7's profiles
  + script 9's plan.

AXIS-1 DECISION RULE (one rule, used everywhere):
  Calibrated C2ST-AUC decides stable-vs-shifted, at the verdict layer here AND in step 11's
  H1/H1.5/H2. C2ST is chosen because it is the ONE metric computed identically for every
  feature type (nominal, discrete-count, continuous, multimodal) — Wasserstein/MMD/JS are only
  well-defined for some routes, so none of them can rank ALL features on one scale.
  The routed distances (Wasserstein-qn / MMD / Jensen-Shannon) plus KS / Anderson-Darling /
  energy distance are CORROBORATION ONLY: each is calibrated against its
  own permutation null and checked for AGREEMENT with the C2ST verdict (the E1 layer), never
  overriding it. Corroboration metrics are computed POOLED ONLY; the
  benign/attack/per-attack-family slices carry C2ST alone (slice_axis1()).

SUB-STEPS:
  Sub-step 10.1 (Load plan & profiles):
    - Read script 9's plan and both datasets' profiles; intersect to the executable feature set.
    Input: output/9_plan_comparison/comparison_plans_<ds1>_<ds2>.json,
           output/7_profile/<ds>/profiles.json
    Output: plan dict + two profile dicts in memory

  Sub-step 10.2 (Execute comparisons):
    - One process per feature: pooled C2ST-AUC (CV CI + permutation-null calibration — the
      Axis-1 decision value), pooled corroboration distances with their own null calibrations
      + E1 agreement flags, C2ST-only slices (benign/attack/per-family), per-mode + zero-mass
      descriptive handling, Axis-2 separation verdict, Q-Q diagnostics, overlap/Q-Q plots.
    Input: the plan + profiles from 10.1, native-unit columns (read per feature)
    Output: Layer A / Layer B records per feature + overlap/Q-Q PNGs

  Sub-step 10.3 (Save verdicts & tables):
    - Write Layer A / Layer B JSON, the flat per-feature CSV.
    Input: results from 10.2
    Output: output/10_execute_comparison/verdicts_layer{A,B}_<ds1>_<ds2>.json + .csv

  Sub-step 10.4 (Render plots):
    - Render the per-feature verdict scatter (overlap/Q-Q plots are written by the workers in 10.2).
    Input: Layer B table from 10.2
    Output: results/10_execute_comparison/verdict_scatter.png

  Sub-step 10.5 (Write report):
    - Write the human-readable report: verdict distribution, C2ST-threshold sensitivity sweep,
      E1 cross-metric agreement, flip corroboration, C2ST stability, quantile-shift breakdown.
    Input: results from 10.2-10.3
    Output: results/10_execute_comparison/10_execute_comparison_report.txt

GUARANTEES:
  - No source data is modified; only script-1 cleaned parquet (read-only) + script 7/9 outputs are read.
  - Verdict file names are stable (verdicts_layer{A,B}_<ds1>_<ds2>.json) — step 11 reads them.
  - Discrete-count features (script 9 plan, metric_family='nominal') run the Jensen-Shannon PMF
    corroboration path here, so integer counts are never treated as continuous — their VERDICT,
    like every feature's, comes from calibrated C2ST-AUC.
  - MATCHED-N null floors: every metric's permutation null uses the same
    per-side sample cap as its actual statistic (see Config10) — a smaller-n null is biased
    high and under-detects shift.

NOTES:
  - Step 10 is CROSS-DATASET: one combined results/output folder, no per-dataset subfolders.
  - All tuning (sample caps, thresholds, CV folds, quantile breakpoints) lives in Config10 — the
    script body hardcodes no numeric tuning values.
  - C2ST-AUC (Lopez-Paz & Oquab 2017) is the universal commensurable shift scalar; its CI from
    the CV folds shows metric stability.
"""

import sys
import gc
import zlib
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import (wasserstein_distance, ks_2samp, anderson_ksamp,
                         t as student_t)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unified_config import (
    Config10, Logger, DATASETS, plan_workers,
    BENIGN_LABEL, COMPARE_OUTPUT_DIR, profile_json_path,
    read_feature_only, read_labels_encoded, safe_filename,
)

_VERDICT_COLORS = {
    'stable': '#2ecc71', 'shifted': '#f39c12', 'flipped': '#e74c3c',
    'collapsed': '#9b59b6', 'weak': '#95a5a6', 'restructured': '#16a085',
}
# Probability grid for the Q-Q plot (clip the 1%/99% tails), built from Config10.
_QQ_QUANTILES = np.linspace(Config10.QQ_Q_LOW, Config10.QQ_Q_HIGH, Config10.QQ_Q_COUNT)
_RNG = np.random.default_rng(Config10.SEED)


# ── Parallel execution (one process per feature; labels + profiles loaded ONCE per worker) ──────
_ES: dict = {}        # per-worker state, populated by the pool initializer


class _NoOpLog:
    """Workers can't share the parent Logger; per-feature warnings are returned instead."""
    def warn(self, *a):     pass
    def info(self, *a):     pass
    def ok(self, *a):       pass
    def step(self, *a):     pass
    def step_end(self, *a): pass
    def section(self, *a):  pass
    def close(self, *a):    pass


def _exec_init(ds1, ds2, overlap_dir, qq_dir):
    with open(profile_json_path(ds1), encoding='utf-8') as f:
        p1 = json.load(f)
    with open(profile_json_path(ds2), encoding='utf-8') as f:
        p2 = json.load(f)
    with open(COMPARE_OUTPUT_DIR / f'comparison_plans_{ds1}_{ds2}.json', encoding='utf-8') as f:
        plans = json.load(f)
    _ES.update(p1=p1, p2=p2, plans=plans, ds1=ds1, ds2=ds2,
               lab17=read_labels_encoded(ds1), lab18=read_labels_encoded(ds2),
               overlap_dir=Path(overlap_dir), qq_dir=Path(qq_dir), log=_NoOpLog())


def _exec_one(feature):
    """Worker task: run one feature's comparison using the per-worker shared state."""
    p1, p2, plans = _ES['p1'], _ES['p2'], _ES['plans']
    if feature not in p1 or feature not in p2:
        return feature, None, None, None
    try:
        r = execute_one(feature, p1[feature], p2[feature], plans[feature], _ES['log'],
                        _ES['lab17'], _ES['lab18'], _ES['ds1'], _ES['ds2'],
                        overlap_dir=_ES['overlap_dir'], qq_dir=_ES['qq_dir'])
        return feature, r['layer_a'], {'verdict': r['verdict'], **r['layer_b']}, None
    except Exception as e:
        return feature, None, None, f'{type(e).__name__}: {e}'


def _subsample(x: np.ndarray, n: int = Config10.MAX_DIST_SAMPLE, *, rng=None) -> np.ndarray:
    x = x[np.isfinite(x)]
    if x.size > n:
        return (rng if rng is not None else _RNG).choice(x, n, replace=False)
    return x


def _scale_transform(x: np.ndarray, scale: str) -> np.ndarray:
    """Strictly-monotonic signed-log compression for log/symlog features: sign(x)*log1p(|x|).

    Applied only to MMD's inputs: its RBF median-heuristic bandwidth is computed from raw pairwise
    distances and degrades on heavy-tailed data (Gretton et al. 2012), so a feature spanning ~6
    orders of magnitude must be compared in a compressed space. ALL the other Axis-1 metrics —
    Wasserstein-qn, energy distance, KS and Anderson-Darling — are rank/CDF-based and hence
    invariant under any monotone transform, so they need no scaling and the headline shift scalar
    and step-11 H1 are provably unaffected by this."""
    if scale in ('log', 'symlog'):
        return np.sign(x) * np.log1p(np.abs(x))
    return x


def _rank_normalize(x1: np.ndarray, x2: np.ndarray):
    """Pooled rank-normalize two samples to [0,1] -> unit-free, monotone-invariant. Returns (r1, r2)."""
    pooled = np.concatenate([x1, x2])
    order = np.argsort(pooled, kind='mergesort')
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(pooled.size) / max(pooled.size - 1, 1)
    return ranks[:x1.size], ranks[x1.size:]


def quantile_norm_wasserstein(x1: np.ndarray, x2: np.ndarray, *, rng=None) -> float:
    """Wasserstein-1 on pooled-rank-normalized values -> unit-free, heavy-tail-robust (on up to
    MAX_DIST_SAMPLE rows = effectively the full column for a 1-D distance)."""
    x1, x2 = _subsample(x1, rng=rng), _subsample(x2, rng=rng)
    if x1.size == 0 or x2.size == 0:
        return 0.0
    r1, r2 = _rank_normalize(x1, x2)
    return float(wasserstein_distance(r1, r2))


def energy_distance_1d(x1: np.ndarray, x2: np.ndarray, *, rng=None) -> float:
    """
    Székely-Rizzo energy distance on pooled-rank-normalized values (unit-free, no bandwidth):
    D^2 = 2 E|X-Y| - E|X-X'| - E|Y-Y'|, estimated on a capped sample (O(n^2) -> justified cap).
    """
    _r = rng if rng is not None else _RNG
    x1, x2 = _subsample(x1, rng=_r), _subsample(x2, rng=_r)
    if x1.size < 2 or x2.size < 2:
        return 0.0
    r1, r2 = _rank_normalize(x1, x2)
    cap = Config10.ENERGY_SAMPLE
    a = r1 if r1.size <= cap else _r.choice(r1, cap, replace=False)
    b = r2 if r2.size <= cap else _r.choice(r2, cap, replace=False)
    A = np.abs(a[:, None] - b[None, :]).mean()
    B = np.abs(a[:, None] - a[None, :]).mean()
    C = np.abs(b[:, None] - b[None, :]).mean()
    return float(max(2.0 * A - B - C, 0.0))


def null_shift_threshold(x1: np.ndarray, x2: np.ndarray,
                         n_resplits: int = Config10.NULL_RESPLITS, *, rng=None) -> float:
    """
    Per-feature NULL shift level for qn-Wasserstein. Pool both years and randomly RE-SPLIT into
    two same-size halves `n_resplits` times, recomputing qn-Wasserstein each time -> the
    distribution of shift under 'no real difference'. Returns the NULL_PERCENTILE percentile. The
    verdict then calls a feature 'shifted' only if its real shift exceeds BOTH the global floor
    AND this data-driven null.

    Thin wrapper around the shared `_permutation_null` (below) — kept as its own named function
    so its existing output field name doesn't change, but the resplit-loop logic itself lives in
    one place.
    """
    return _permutation_null(x1, x2, quantile_norm_wasserstein, n_resplits,
                             Config10.NULL_SAMPLE, Config10.NULL_PERCENTILE, rng=rng)


def _direct_auc(v: np.ndarray, y: np.ndarray) -> float:
    """Single-feature folded AUC: how well v alone ranks binary y, direction-agnostic
    (max(auc, 1-auc), same folding convention as separation_stability())."""
    if np.unique(y).size < 2:
        return 0.5
    auc = roc_auc_score(y, v)
    return float(max(auc, 1.0 - auc))


def null_separation_threshold(v: np.ndarray, yb: np.ndarray,
                              n_resplits: int = Config10.NULL_RESPLITS, *, rng=None) -> float:
    """
    Per-feature null floor for Axis-2 separation strength: SEPARATION_STRONG/
    MI_STRONG were flat constants with no per-feature calibration, unlike Axis 1's
    null_shift_threshold() above. Mirrors that treatment: pool this feature's values regardless of
    true benign/attack label, randomly reassign to two pseudo-groups of the SAME sizes as the real
    split, recompute a direct rank-AUC each time -> the distribution of separation you'd see under
    "benign/attack carries no information". Returns the NULL_PERCENTILE percentile; a feature only
    counts as separated if it clears max(SEPARATION_STRONG, this) as well, same logic as Axis 1's
    max(shift_low_global, null_shift).
    NOTE: this direct single-feature rank-AUC is a proxy for, not a re-implementation of, step 7's
    own folded-AUC/MI algorithm (which may bin/smooth differently) — it calibrates the same family
    of statistic, but sep17/sep18 themselves are read from step 7's profiles, not recomputed here.
    """
    mask = np.isfinite(v)
    v, yb = v[mask], yb[mask].astype(np.int64)
    local_rng = rng if rng is not None else np.random.default_rng(Config10.SEED)
    if v.size > Config10.NULL_SAMPLE:
        idx = local_rng.choice(v.size, Config10.NULL_SAMPLE, replace=False)
        v, yb = v[idx], yb[idx]
    if v.size < 50 or np.unique(yb).size < 2:
        return 0.5
    # Fix: reuse local_rng (the caller's per-feature rng) for the permutation loop
    # too, instead of a fresh fixed-seed generator — see null_shift_threshold() above for why.
    vals = [_direct_auc(v, local_rng.permutation(yb)) for _ in range(n_resplits)]
    return float(np.percentile(vals, Config10.NULL_PERCENTILE)) if vals else 0.5


def _direct_mi_norm(v: np.ndarray, y: np.ndarray, *, seed: int) -> float:
    """Single-feature normalized-MI proxy via a fast k-NN estimator (sklearn's
    mutual_info_classif), analogous to _direct_auc's proxy role for null_separation_threshold:
    same statistic FAMILY as step 7's own KSG mutual-information algorithm, not a literal re-run
    of it (step 7's estimator is too expensive to repeat inside a permutation loop). Normalized
    by the label's own entropy, same convention as step 7's mutual_info_normalized."""
    if np.unique(y).size < 2:
        return 0.0
    mi_nats = mutual_info_classif(v.reshape(-1, 1), y, discrete_features=False,
                                  n_neighbors=3, random_state=seed)[0]
    p = float(np.mean(y))
    if p <= 0.0 or p >= 1.0:
        return 0.0
    h = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    return float(mi_nats / h) if h > 0 else 0.0


def null_mi_threshold(v: np.ndarray, yb: np.ndarray,
                      n_resplits: int = Config10.MI_NULL_RESPLITS, *, rng=None) -> float:
    """
    Per-feature null floor for Axis-2 normalized MI, mirroring null_separation_threshold's
    treatment of the folded-AUC bar: MI_STRONG was previously a flat constant with no per-feature
    calibration, unlike SEPARATION_STRONG (calibrated above). Same permute-relabel-resplit design:
    pool this feature's values regardless of true benign/attack label, randomly reassign to two
    pseudo-groups of the SAME sizes as the real split, recompute a direct MI proxy each time ->
    the distribution of MI you'd see under "benign/attack carries no information". Returns the
    NULL_PERCENTILE percentile; a feature only counts as separated via the MI route if it clears
    max(MI_STRONG, this) as well, same logic as the AUC route's max(SEPARATION_STRONG, null).
    Uses MI_NULL_RESPLITS/MI_NULL_SAMPLE (smaller than the AUC null's budget) since the k-NN MI
    estimator is more expensive per call than a rank-based AUC.
    """
    mask = np.isfinite(v)
    v, yb = v[mask], yb[mask].astype(np.int64)
    local_rng = rng if rng is not None else np.random.default_rng(Config10.SEED)
    if v.size > Config10.MI_NULL_SAMPLE:
        idx = local_rng.choice(v.size, Config10.MI_NULL_SAMPLE, replace=False)
        v, yb = v[idx], yb[idx]
    if v.size < 50 or np.unique(yb).size < 2:
        return 0.0
    vals = [_direct_mi_norm(v, local_rng.permutation(yb),
                            seed=int(local_rng.integers(0, 2**31 - 1)))
            for _ in range(n_resplits)]
    return float(np.percentile(vals, Config10.NULL_PERCENTILE)) if vals else 0.0


def _mode_assign(s: np.ndarray, centers: np.ndarray, scale: str) -> np.ndarray:
    """Nearest-mode assignment in the space the GMM was FIT in. Step 7 fits log/symlog features
    on log10(positive values), so assigning in native units would let the largest-magnitude mode
    dominate the distance and misassign points near small-magnitude modes."""
    if scale in ('log', 'symlog'):
        eps = 1e-12
        st = np.log10(np.clip(np.abs(s), eps, None))
        ct = np.log10(np.clip(np.abs(centers), eps, None))
        return np.abs(st[:, None] - ct[None, :]).argmin(axis=1)
    return np.abs(s[:, None] - centers[None, :]).argmin(axis=1)


def per_mode_comparison(v17: np.ndarray, v18: np.ndarray, p17: dict, p18: dict) -> dict:
    """
    Blob-to-blob comparison for multimodal features. Align the GMM modes (from step 7) by
    SORTED CENTER — the proven 1-D optimal coupling (monotone) — match k-th<->k-th, assign sampled
    points to their nearest mode, and compute qn-Wasserstein per matched mode plus the per-mode mass
    shift. This is the H3 comparison the old code routed to but never ran.
    (Resolved, not a bug): the concern raised was "same mode COUNT doesn't guarantee
    the modes are the same underlying clusters." For a 1-D SCALAR feature this doesn't apply the
    way it would for multi-dimensional clustering — sorted-by-center matching (k-th-smallest <->
    k-th-smallest) is the mathematically OPTIMAL (Monge) coupling between two 1-D point sets under
    any convex transport cost, not a heuristic guess at cluster correspondence. A mode's "identity"
    for a scalar feature IS its position on the number line, so there is no separate "cluster
    identity" to get wrong here the way there could be in >=2-D space.
    """
    modes17 = sorted(p17.get('modes', []) or [], key=lambda m: m.get('center', 0.0))
    modes18 = sorted(p18.get('modes', []) or [], key=lambda m: m.get('center', 0.0))
    k = min(len(modes17), len(modes18))
    base = {'per_mode': [], 'n_modes_2017': len(modes17), 'n_modes_2018': len(modes18),
            'max_mode_shift': 0.0, 'max_mode_mass_shift': 0.0}
    if k == 0:
        return base
    s17, s18 = _subsample(v17), _subsample(v18)
    if s17.size < 10 or s18.size < 10:
        return base
    c17 = np.array([m.get('center', 0.0) for m in modes17], dtype=np.float64)
    c18 = np.array([m.get('center', 0.0) for m in modes18], dtype=np.float64)
    scale = p17.get('recommended_scale', 'linear')
    a17 = _mode_assign(s17, c17, scale)   # nearest mode per point, in the GMM's fit space
    a18 = _mode_assign(s18, c18, scale)
    results = []
    for j in range(k):
        x, y = s17[a17 == j], s18[a18 == j]
        w = quantile_norm_wasserstein(x, y) if (x.size > 10 and y.size > 10) else 0.0
        m17, m18 = float(modes17[j].get('mass', 0.0)), float(modes18[j].get('mass', 0.0))
        results.append({'mode': j,
                        'center_2017': float(c17[j]), 'center_2018': float(c18[j]),
                        'wasserstein_qn': w,
                        'mass_2017': m17, 'mass_2018': m18, 'mass_shift': abs(m17 - m18)})
    base['per_mode'] = results
    base['max_mode_shift'] = max((r['wasserstein_qn'] for r in results), default=0.0)
    base['max_mode_mass_shift'] = max((r['mass_shift'] for r in results), default=0.0)
    return base


def jensen_shannon_pmf(x1: np.ndarray, x2: np.ndarray, top_k: int = 50) -> float:
    """JS divergence between PMFs over shared categories; high-card -> top-K + 'other'."""
    x1 = x1[np.isfinite(x1)]
    x2 = x2[np.isfinite(x2)]
    if x1.size == 0 or x2.size == 0:
        return 0.0
    vals, counts = np.unique(np.concatenate([x1, x2]), return_counts=True)
    if vals.size > top_k:
        keep = set(vals[np.argsort(counts)[::-1][:top_k]].tolist())
    else:
        keep = set(vals.tolist())

    def pmf(x):
        u, c = np.unique(x, return_counts=True)
        d = {float(k): int(v) for k, v in zip(u, c)}
        vec = [d.get(float(k), 0) for k in keep]
        other = x.size - sum(vec)
        vec.append(other)
        arr = np.asarray(vec, dtype=np.float64)
        return arr / arr.sum() if arr.sum() else arr

    return float(jensenshannon(pmf(x1), pmf(x2)))


def anderson_darling_2samp(x1: np.ndarray, x2: np.ndarray, n: int = Config10.KS_SAMPLE,
                           *, rng=None) -> float:
    """Two-sample Anderson-Darling statistic (Engmann & Cousineau) — far more tail-sensitive than
    KS, which matters for heavy-tailed NIDS features. CDF/rank-based, so it needs no scale transform.
    Computed on a capped sample; returns the raw statistic (larger = more divergent). Failure -> 0.0."""
    import warnings
    a = _subsample(x1, n, rng=rng)
    b = _subsample(x2, n, rng=rng)
    if a.size < 2 or b.size < 2:
        return 0.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')          # AD warns when the stat is capped at the table edge
            return float(anderson_ksamp([a, b]).statistic)
    except Exception:
        return 0.0


def mmd_rbf(x1: np.ndarray, x2: np.ndarray, n: int = Config10.MMD_SAMPLE,
            *, scale: str = 'linear', rng=None) -> float:
    """Unbiased MMD^2 with RBF kernel, median-heuristic bandwidth (Gretton et al. 2012). For
    log/symlog features the inputs are signed-log compressed first so the median-heuristic
    bandwidth is not dominated by the heavy tail."""
    _r = rng if rng is not None else _RNG
    a = _scale_transform(_subsample(x1, n, rng=_r).astype(np.float64), scale)
    b = _scale_transform(_subsample(x2, n, rng=_r).astype(np.float64), scale)
    if a.size < 2 or b.size < 2:
        return 0.0
    pooled = np.concatenate([a, b])
    med_cap = Config10.MMD_MEDIAN_SAMPLE
    sub = pooled if pooled.size <= med_cap else _r.choice(pooled, med_cap, replace=False)
    d = np.abs(sub[:, None] - sub[None, :])
    med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
    gamma = 1.0 / (2.0 * med * med + 1e-12)

    def k(u, v):
        return np.exp(-gamma * (u[:, None] - v[None, :]) ** 2)

    kaa = k(a, a); kbb = k(b, b); kab = k(a, b)
    na, nb = a.size, b.size
    term_a = (kaa.sum() - np.trace(kaa)) / (na * (na - 1))
    term_b = (kbb.sum() - np.trace(kbb)) / (nb * (nb - 1))
    term_ab = kab.mean()
    return float(max(term_a + term_b - 2 * term_ab, 0.0))


def _ci_from_scores(scores: list) -> dict:
    """Student-t CI (Config10.C2ST_CI_CONFIDENCE) from the per-fold AUCs. With few folds the
    t-distribution is the honest choice. The CI WIDTH is the metric-stability signal: a wide CI
    means the C2ST-AUC is fold-sensitive (treat that feature's shift scalar with caution)."""
    arr = np.asarray(scores, dtype=np.float64)
    mean = float(arr.mean()) if arr.size else 0.5
    if arr.size < 2:
        return {'auc': mean, 'folds': [float(s) for s in arr],
                'ci_low': mean, 'ci_high': mean, 'ci_width': 0.0}
    se = float(arr.std(ddof=1)) / np.sqrt(arr.size)
    tcrit = float(student_t.ppf(1 - (1 - Config10.C2ST_CI_CONFIDENCE) / 2, df=arr.size - 1))
    half = tcrit * se
    lo, hi = max(0.0, mean - half), min(1.0, mean + half)
    return {'auc': mean, 'folds': [float(s) for s in arr],
            'ci_low': lo, 'ci_high': hi, 'ci_width': float(hi - lo)}


def c2st_auc(x1: np.ndarray, x2: np.ndarray, n: int = Config10.C2ST_SAMPLE, *, rng=None) -> dict:
    """
    Classifier Two-Sample Test AUC (Lopez-Paz & Oquab 2017).
    Trains a shallow decision tree to distinguish 2017 vs 2018 values; returns the stratified
    CV mean AUC on [0.5, 1.0] PLUS a confidence interval computed from the per-fold AUCs (the CI
    width is a stability indicator for step 11). Universal commensurable shift scalar: same scale
    for any feature type. 0.5 = indistinguishable years; 1.0 = perfectly separable.
    """
    a = _subsample(x1, n, rng=rng)
    b = _subsample(x2, n, rng=rng)
    if a.size < 20 or b.size < 20:
        return _ci_from_scores([0.5])
    X = np.concatenate([a, b]).reshape(-1, 1)
    y = np.concatenate([np.zeros(a.size), np.ones(b.size)])
    try:
        clf = DecisionTreeClassifier(max_depth=Config10.C2ST_TREE_DEPTH, random_state=Config10.SEED)
        skf = StratifiedKFold(n_splits=Config10.C2ST_CV_FOLDS, shuffle=True, random_state=Config10.SEED)
        scores = [
            roc_auc_score(y[ti], clf.fit(X[tr], y[tr]).predict_proba(X[ti])[:, 1])
            for tr, ti in skf.split(X, y)
        ]
        return _ci_from_scores(scores)
    except Exception:
        return _ci_from_scores([0.5])


def _permutation_null(x1: np.ndarray, x2: np.ndarray, metric_fn, n_resplits: int,
                      sample_cap: int, percentile: float = Config10.NULL_PERCENTILE,
                      *, rng=None) -> float:
    """Generic per-feature null floor, shared by every metric EXCEPT Wasserstein-qn (which keeps
    its own null_shift_threshold() above — same idea, kept separate so its existing output field
    names don't change). Pool both years, randomly re-split into two same-size halves n_resplits
    times, recompute metric_fn each time -> the distribution of the metric under 'no real year
    difference'. Returns the given percentile of that null distribution."""
    a = _subsample(x1, sample_cap, rng=rng)
    b = _subsample(x2, sample_cap, rng=rng)
    if a.size < 50 or b.size < 50:
        return 0.0
    pooled = np.concatenate([a, b])
    na = a.size
    # Fix: see null_shift_threshold() above — reuse the caller's per-feature rng
    # instead of a fresh fixed-seed generator, or every feature/slice gets an identical resplit
    # pattern for MMD/energy/KS/AD, understating Monte-Carlo variance of these null floors too.
    perm_rng = rng if rng is not None else np.random.default_rng(Config10.SEED)
    vals = []
    for _ in range(n_resplits):
        perm = perm_rng.permutation(pooled.size)
        h1, h2 = pooled[perm[:na]], pooled[perm[na:]]
        vals.append(metric_fn(h1, h2))
    return float(np.percentile(vals, percentile)) if vals else 0.0


def _c2st_null_auc(a: np.ndarray, b: np.ndarray, *, rng=None) -> float:
    """Single train/test split AUC (NOT the full 5-fold CV used by c2st_auc()) — used only inside
    the null permutation loop, where averaging over many cheap resplits matters more than a tight
    per-resplit estimate. Keeps null calibration affordable at C2ST_NULL_RESPLITS resplits/feature."""
    if a.size < 20 or b.size < 20:
        return 0.5
    X = np.concatenate([a, b]).reshape(-1, 1)
    y = np.concatenate([np.zeros(a.size), np.ones(b.size)])
    idx = np.arange(X.shape[0])
    (rng if rng is not None else _RNG).shuffle(idx)
    split = int(0.7 * idx.size)
    tr, te = idx[:split], idx[split:]
    if np.unique(y[tr]).size < 2 or np.unique(y[te]).size < 2:
        return 0.5
    try:
        clf = DecisionTreeClassifier(max_depth=Config10.C2ST_TREE_DEPTH, random_state=Config10.SEED)
        clf.fit(X[tr], y[tr])
        return float(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    except Exception:
        return 0.5


def null_c2st_threshold(x1: np.ndarray, x2: np.ndarray,
                        n_resplits: int = Config10.C2ST_NULL_RESPLITS, *, rng=None) -> float:
    """Per-feature null floor for C2ST-AUC (C2ST had no calibration, unlike Wasserstein's
    null_shift_threshold above). Called once pooled and once per slice (benign/attack), so the
    verdict can be checked against a NULL that matches that slice's own sample size / noise level."""
    return _permutation_null(x1, x2, lambda a, b: _c2st_null_auc(a, b, rng=rng),
                             n_resplits, Config10.C2ST_NULL_SAMPLE, rng=rng)


def calibrate_c2st(raw_auc: float, null_floor: float) -> float:
    """Normalize a raw C2ST-AUC against its own per-feature/per-slice null floor.
    Scale: negative = feature more stable than null-floor noise, 0 = at null floor,
    1 = maximal separation (AUC=1.0). Allows negative values (permutation-test standard)
    to preserve information about super-stable features. This is the CALIBRATED C2ST-AUC
    that drives every H1/H1.5/H2 decision downstream (step 11) — raw C2ST-AUC is kept
    alongside it for reference only."""
    if not np.isfinite(raw_auc) or not np.isfinite(null_floor):
        return float('nan')
    denom = max(1.0 - null_floor, 1e-6)
    return float((raw_auc - null_floor) / denom)


def null_mmd_threshold(x1: np.ndarray, x2: np.ndarray, *, scale: str = 'linear',
                       n_resplits: int = Config10.HEAVY_NULL_RESPLITS, rng=None) -> float:
    """Per-feature (pooled-only) null floor for MMD. MMD is O(n^2), so its null re-split sample
    is capped tighter (HEAVY_NULL_SAMPLE) than the cheaper metrics."""
    return _permutation_null(x1, x2, lambda a, b: mmd_rbf(a, b, scale=scale, rng=rng),
                             n_resplits, Config10.HEAVY_NULL_SAMPLE, rng=rng)


def null_energy_threshold(x1: np.ndarray, x2: np.ndarray,
                          n_resplits: int = Config10.HEAVY_NULL_RESPLITS, *, rng=None) -> float:
    """Per-feature (pooled-only) null floor for energy distance (also O(n^2) -> tight sample cap)."""
    return _permutation_null(x1, x2, lambda a, b: energy_distance_1d(a, b, rng=rng),
                             n_resplits, Config10.HEAVY_NULL_SAMPLE, rng=rng)


def null_ks_threshold(x1: np.ndarray, x2: np.ndarray,
                      n_resplits: int = Config10.LIGHT_NULL_RESPLITS, *, rng=None) -> float:
    """Per-feature (pooled-only) null floor for the KS statistic."""
    def _ks(a, b):
        return float(ks_2samp(a, b).statistic) if a.size > 1 and b.size > 1 else 0.0
    return _permutation_null(x1, x2, _ks, n_resplits, Config10.LIGHT_NULL_SAMPLE, rng=rng)


def null_ad_threshold(x1: np.ndarray, x2: np.ndarray,
                      n_resplits: int = Config10.LIGHT_NULL_RESPLITS, *, rng=None) -> float:
    """Per-feature (pooled-only) null floor for the two-sample Anderson-Darling statistic."""
    return _permutation_null(x1, x2, lambda a, b: anderson_darling_2samp(a, b, rng=rng),
                             n_resplits, Config10.LIGHT_NULL_SAMPLE, rng=rng)


def null_js_threshold(x1: np.ndarray, x2: np.ndarray,
                      n_resplits: int = Config10.LIGHT_NULL_RESPLITS, *, rng=None) -> float:
    """Per-feature (pooled-only) null floor for Jensen-Shannon on the PMF route. Previously the
    nominal/discrete-count route had NO null calibration at all — every other
    corroboration metric had one — so its E1 agreement check would have compared a raw JS value
    against calibrated metrics. JS on a PMF is cheap, so the standard resplit-null applies."""
    return _permutation_null(x1, x2, jensen_shannon_pmf,
                             n_resplits, Config10.KS_SAMPLE, rng=rng)


def calibrate_excess(raw: float, null_floor: float) -> float:
    """Calibration for the four distance-style metrics (MMD, energy distance, KS, Anderson-Darling):
    how far the raw value sits ABOVE its own null floor, in null-floor units. Unlike C2ST-AUC (which
    is bounded [0.5, 1] so a ratio-to-[0,1] normalization applies), these metrics are unbounded above
    and can sit arbitrarily close to 0 at their null floor, so a subtractive/ratio hybrid is used:
    0 = at or below the null (not distinguishable from a same-year re-split); grows with real excess
    shift, scaled by the floor itself so a noisy (high-floor) feature needs proportionally more raw
    shift to register the same calibrated value as a quiet (low-floor) one.
    Clipped at Config10.EXCESS_RATIO_CAP — without an upper bound, a near-zero
    null floor (e.g. 1e-6) blows an otherwise-small raw value up to an arbitrarily large ratio
    (raw=0.001, floor=1e-6 -> ~999), which would make these diagnostic-only (E1) values
    incomparable across features. Values at or beyond the cap are already unambiguously
    "far above null" — the exact magnitude past that point carries no extra decision weight."""
    if not np.isfinite(raw) or not np.isfinite(null_floor):
        return float('nan')
    floor = max(null_floor, 1e-6)
    excess = (raw - null_floor) / floor
    return float(np.clip(excess, 0.0, Config10.EXCESS_RATIO_CAP))


def marginal_shift(x1: np.ndarray, x2: np.ndarray, family: str,
                   *, scale: str = 'linear', rng=None) -> dict:
    """
    Pooled CORROBORATION distances (the E1 inputs — they never decide the verdict, calibrated
    C2ST-AUC does). family=='nominal' is the PMF (Jensen-Shannon) path —
    discrete-count features route here too (script 9 marks them metric_family='nominal');
    the continuous families get Wasserstein-qn + MMD + energy + Anderson-Darling + KS.
    Computed POOLED ONLY — slices carry C2ST alone (slice_axis1()).

    `scale` (from the plan's cross-year recommended_scale) only affects MMD (its bandwidth is a
    raw-distance heuristic); the rank/CDF-based metrics are monotone-invariant.
    """
    out: dict = {}
    if family == 'nominal':
        out['jensen_shannon'] = jensen_shannon_pmf(x1, x2)
        return out
    out['wasserstein_qn'] = quantile_norm_wasserstein(x1, x2, rng=rng)
    out['mmd'] = mmd_rbf(x1, x2, scale=scale, rng=rng)
    out['energy_distance'] = energy_distance_1d(x1, x2, rng=rng)
    out['anderson_darling'] = anderson_darling_2samp(x1, x2, rng=rng)
    xs1 = _subsample(x1, Config10.KS_SAMPLE, rng=rng)
    xs2 = _subsample(x2, Config10.KS_SAMPLE, rng=rng)
    out['ks_statistic'] = (float(ks_2samp(xs1, xs2).statistic)
                           if xs1.size > 1 and xs2.size > 1 else 0.0)
    return out


def slice_axis1(x1: np.ndarray, x2: np.ndarray, rng) -> dict:
    """Axis-1 shift for a label-restricted SLICE (benign-only / attack-only / one attack family):
    the universal C2ST-AUC with its own per-slice null floor — nothing else. Slices exist so step
    11 can ATTRIBUTE pooled instability to the class that drove it ("pooled says X is unstable —
    which slice is the culprit?"), and C2ST is the one metric comparable across every feature and
    slice. The corroboration distances (Wasserstein/MMD/KS/AD/energy/JS) are pooled-only by design —
    recomputing them per slice would add permutation-null cost without
    feeding any downstream decision. Degenerate slices return NaN scalars (+ counts)."""
    n1 = int(np.isfinite(x1).sum()) if x1.size else 0
    n2 = int(np.isfinite(x2).sum()) if x2.size else 0
    if n1 < 20 or n2 < 20:
        return {'c2st_auc': float('nan'), 'c2st_null': float('nan'),
                'c2st_calibrated': float('nan'), 'n_2017': n1, 'n_2018': n2}
    raw_auc = c2st_auc(x1, x2, rng=rng)['auc']
    null_auc = null_c2st_threshold(x1, x2, rng=rng)
    return {'c2st_auc': raw_auc,
            'c2st_null': null_auc,                           # per-slice null floor (H1/H1.5/H2 input)
            'c2st_calibrated': calibrate_c2st(raw_auc, null_auc),
            'n_2017': n1, 'n_2018': n2}


def separation_stability(sep17: float, dir17: float, sep18: float, dir18: float,
                         strong: float = Config10.SEPARATION_STRONG) -> float:
    """
    Layer B scalar 2: separation-stability in [-1, 1].
    ~1 = separation preserved (strong both years, same sign).
    ~0 = collapsed (was strong, now gone).
    <0 = flipped (strong both years, sign reversed).
    """
    s17 = (sep17 - 0.5) * 2
    s18 = (sep18 - 0.5) * 2
    base = min(max(s17, 0), 1) * min(max(s18, 0), 1)
    # A flip requires a TRUSTED direction in BOTH years (direction 0 = AUC inside the
    # deadband = no trusted sign). dir17 != 0 alone let "strong vs untrusted" count as a flip.
    if (sep17 >= strong and sep18 >= strong
            and dir17 != 0 and dir18 != 0 and np.sign(dir17) != np.sign(dir18)):
        return -base
    return base


def execute_one(feature: str, p17: dict, p18: dict, plan: dict, log,
                lab17_enc, lab18_enc, ds1: str, ds2: str,
                overlap_dir=None, qq_dir=None) -> dict:
    # Deterministic per-feature RNG seeded from a stable hash of the feature name (zlib.crc32 —
    # Python's built-in hash() is salted per process). Every feature draws the same sample on every
    # run regardless of scheduling or worker count.
    rng = np.random.default_rng(zlib.crc32(feature.encode('utf-8')))

    det = p17.get('detected_type', 'continuous')
    family = plan.get('metric_family', 'continuous')
    route = plan.get('route', family)   # script 9 'route' tag (e.g. discrete_count vs nominal)
    scale = plan.get('recommended_scale', 'linear')   # drives the scale-sensitive corroboration (MMD)
    # plan['comparison_mode'] is consumed below — multimodal features run per_mode_comparison()
    # (GMM modes aligned by sorted center) and fold the worst per-mode shift into the verdict.

    # Labels were read ONCE in main() (memory-safe encoded form); only the feature columns are
    # read here. Avoids the large per-feature to_pylist that would OOM on the 2018 dataset.
    codes17, cats17, yb17 = lab17_enc
    codes18, cats18, yb18 = lab18_enc
    idx17 = {c: i for i, c in enumerate(cats17)}
    idx18 = {c: i for i, c in enumerate(cats18)}

    # Read the full column (parquet is columnar — one column from a 34 GB file is ~480 MB,
    # not the full file). Workers are capped so at most MAX_WORKERS × ~480 MB are concurrent.
    v17 = np.asarray(read_feature_only(ds1, feature), np.float64)
    v18 = np.asarray(read_feature_only(ds2, feature), np.float64)

    # Confounders for step 11 §7.1b partial-correlation control
    fin17 = v17[np.isfinite(v17)]
    fin18 = v18[np.isfinite(v18)]
    cardinality = int(np.unique(np.concatenate([
        _subsample(fin17, 50_000, rng=rng), _subsample(fin18, 50_000, rng=rng)
    ])).size) if fin17.size and fin18.size else 0
    variance_2017 = float(np.var(fin17)) if fin17.size else 0.0
    variance_2018 = float(np.var(fin18)) if fin18.size else 0.0

    # Axis 1 — POOLED corroboration distances (E1 inputs; the verdict comes from calibrated C2ST)
    pooled = marginal_shift(v17, v18, family, scale=scale, rng=rng)

    # C2ST-AUC — universal commensurable shift scalar (+ CI from the CV folds = stability)
    c2st = c2st_auc(v17, v18, rng=rng)

    # Null calibration (previously only Wasserstein had a per-feature null floor). C2ST-AUC gets its
    # own floor here (pooled) and again per-slice below via slice_axis1(); this POOLED calibrated value
    # is what step 11 uses for every H1/H1.5/H2 decision (see calibrate_c2st() docstring above).
    c2st_null_pooled = null_c2st_threshold(v17, v18, rng=rng)
    c2st_calibrated_pooled = calibrate_c2st(c2st['auc'], c2st_null_pooled)
    c2st['null_pooled'] = c2st_null_pooled
    c2st['calibrated_pooled'] = c2st_calibrated_pooled

    # Corroboration metrics are calibrated POOLED-ONLY: they feed E1 (the
    # diagnostic-only cross-metric agreement check against calibrated C2ST-AUC, FIX-2), which only
    # ever reads the pooled values — a per-slice null here would add cost (each is a permutation
    # loop) without feeding any downstream decision. Every null uses the SAME per-side sample cap
    # as its actual statistic (matched-n rule, FIX-5). The nominal/PMF route gets its own JS null
    # (previously it had none).
    if family != 'nominal':
        # Same null_fn(v17, v18, rng=rng, **extra) call for every metric (mmd alone also needs
        # scale=); looping preserves the exact original call order (wass, mmd, energy, ks, ad),
        # so the shared per-feature rng is consumed identically to the unrolled version.
        for mkey, null_fn, extra in (
            ('wasserstein_qn', null_shift_threshold, {}),
            ('mmd', null_mmd_threshold, {'scale': scale}),
            ('energy_distance', null_energy_threshold, {}),
            ('ks_statistic', null_ks_threshold, {}),
            ('anderson_darling', null_ad_threshold, {}),
        ):
            null_val = null_fn(v17, v18, rng=rng, **extra)
            pooled[f'{mkey}_null'] = null_val
            pooled[f'{mkey}_calibrated'] = calibrate_excess(pooled.get(mkey, 0.0), null_val)
    else:
        null_js = null_js_threshold(v17, v18, rng=rng)
        pooled['jensen_shannon_null'] = null_js
        pooled['jensen_shannon_calibrated'] = calibrate_excess(
            pooled.get('jensen_shannon', 0.0), null_js)

    # Axis 1 — per-SLICE attribution (benign-only / attack-only / per attack family): C2ST only,
    # each slice against its OWN null floor, so step 11 can attribute pooled instability to the
    # class that drove it on the one scale every feature and slice shares.
    benign = slice_axis1(v17[yb17 == 0], v18[yb18 == 0], rng)
    attack = slice_axis1(v17[yb17 == 1], v18[yb18 == 1], rng)
    shared_fams = sorted((set(cats17) - {BENIGN_LABEL}) & (set(cats18) - {BENIGN_LABEL}))
    per_attack_axis1 = {
        cls: slice_axis1(v17[codes17 == idx17[cls]], v18[codes18 == idx18[cls]], rng)
        for cls in shared_fams
    }

    # Axis 2 — separation from profiles, using BOTH folded-AUC (monotonic) AND normalized mutual
    # information (non-monotonic-capable, step 7) so a multimodal-separated feature is NOT mislabeled
    # "weak/collapsed" just because its blobs aren't monotonically ordered.
    sep17 = p17.get('separation_magnitude', 0.5)
    dir17 = p17.get('separation_direction', 0.0)
    sep18 = p18.get('separation_magnitude', 0.5)
    dir18 = p18.get('separation_direction', 0.0)
    mi17  = float(p17.get('mutual_info_normalized', 0.0))
    mi18  = float(p18.get('mutual_info_normalized', 0.0))
    strong    = Config10.SEPARATION_STRONG   # folded-AUC "strong separation" bar (global floor)
    mi_strong = Config10.MI_STRONG           # normalized-MI "strong separation" bar (non-monotonic)

    # Null-calibrate the AUC floor per feature (B7), same treatment as Axis 1's null_shift above:
    # a feature counts as separated only if it clears BOTH the global bar AND its own null floor.
    null_sep17 = null_separation_threshold(v17, yb17, rng=rng)
    null_sep18 = null_separation_threshold(v18, yb18, rng=rng)
    eff_strong17 = max(strong, null_sep17)
    eff_strong18 = max(strong, null_sep18)

    # Null-calibrate the MI floor per feature too — MI_STRONG previously had no per-feature
    # calibration at all, unlike SEPARATION_STRONG just above (see null_mi_threshold() docstring).
    null_mi17 = null_mi_threshold(v17, yb17, rng=rng)
    null_mi18 = null_mi_threshold(v18, yb18, rng=rng)
    eff_mi_strong17 = max(mi_strong, null_mi17)
    eff_mi_strong18 = max(mi_strong, null_mi18)

    def _sep_strength(auc, mi):          # combined separation strength in [0, 1]
        return float(max((auc - 0.5) * 2.0, mi))
    sepstr17, sepstr18 = _sep_strength(sep17, mi17), _sep_strength(sep18, mi18)
    is_sep17 = (sep17 >= eff_strong17) or (mi17 >= eff_mi_strong17)
    is_sep18 = (sep18 >= eff_strong18) or (mi18 >= eff_mi_strong18)

    # Axis 2 — per-attack-class separation stability (cross p17/p18 per_class_separation)
    pcs17 = p17.get('per_class_separation') or {}
    pcs18 = p18.get('per_class_separation') or {}
    per_class_stability: dict = {}
    for cls in sorted(set(pcs17) & set(pcs18)):
        s = pcs17[cls]
        t = pcs18[cls]
        per_class_stability[cls] = float(separation_stability(
            s.get('magnitude', 0.5), s.get('direction', 0.0),
            t.get('magnitude', 0.5), t.get('direction', 0.0), strong))

    # Per-family flip corroboration. The POOLED flip flag (below) is computed
    # on the binary benign-vs-attack direction, which the changing attack MIXTURE between years can
    # reverse without any single attack's relationship to benign actually flipping. separation_stability
    # returns a NEGATIVE value exactly when a SHARED family's own separation reversed sign with a
    # trusted direction in BOTH years — a genuine per-class flip. So a pooled 'flipped' verdict is
    # "corroborated" iff at least one shared family really flips here; an UNcorroborated pooled flip
    # is most likely a mixture artifact. These fields let step 11 count corroborated vs artifact flips.
    family_flips = sorted([c for c, v in per_class_stability.items() if v < 0])
    n_family_flips = len(family_flips)
    flip_corroborated = n_family_flips > 0

    # Per-mode (blob-to-blob) comparison when the plan routed this feature to per_mode.
    comparison_mode = plan.get('comparison_mode', 'whole_distribution')
    per_mode_results: dict = {}
    if comparison_mode == 'per_mode' and family != 'nominal':
        try:
            per_mode_results = per_mode_comparison(v17, v18, p17, p18)
        except Exception as e:
            log.warn(f'  per-mode comparison failed for {feature}: {e}')

    # Zero-mass-separate: for zero-inflated features compare the
    # zero FRACTION as its own scalar and the qn-Wasserstein on the NON-ZERO tails only.
    # Pooling zeros into the Wasserstein lets a large shared zero mass dilute a real tail
    # shift (and conflates zero-fraction change with tail-shape change).
    zero_mass: dict = {}
    if plan.get('zero_mass_separate') and family != 'nominal':
        if fin17.size and fin18.size:
            zf17 = float(np.mean(fin17 == 0))
            zf18 = float(np.mean(fin18 == 0))
            zero_mass = {
                'zero_frac_2017': zf17,
                'zero_frac_2018': zf18,
                'zero_frac_delta': abs(zf17 - zf18),
                'tail_wasserstein_qn': quantile_norm_wasserstein(fin17[fin17 != 0], fin18[fin18 != 0]),
            }

    # Descriptive marginal-shift magnitude (Layer B scalar / CSV / scatter context). NOT the
    # verdict driver — it is the route-family distance a reader can eyeball
    # next to the C2ST decision. For multimodal features it escalates with the worst per-mode
    # shift; for zero-inflated features it takes the max of (non-zero tail shift, zero-fraction
    # delta) — both in [0, 1], so the max flags whichever component actually moved.
    if family == 'nominal':
        shift_mag = pooled.get('jensen_shannon', 0.0)
    else:
        shift_mag = pooled.get('wasserstein_qn', 0.0)
        if zero_mass:
            shift_mag = max(zero_mass['tail_wasserstein_qn'], zero_mass['zero_frac_delta'])
        if comparison_mode == 'per_mode':
            shift_mag = max(shift_mag, per_mode_results.get('max_mode_shift', 0.0))

    # Verdict. ONE Axis-1 decision rule everywhere: calibrated C2ST-AUC above
    # its threshold -> the two years are more classifier-distinguishable than this feature's own
    # permutation-null noise -> 'shifted'. Same quantity step 11 uses for H1/H1.5/H2, so the
    # descriptive labels here and the hypothesis tests can never disagree about Axis 1 again.
    # Flip/collapse/weak (Axis-2 verdicts — how the label relationship changed) take precedence;
    # a flip needs a TRUSTED direction in BOTH years (0 = AUC inside the deadband, see step 7).
    # 'restructured' = structural_change route (modality/type mismatch between years) WHOSE C2ST
    # also confirms real distinguishability — a structure mismatch the classifier cannot detect
    # is treated as GMM/type-detection noise and stays 'stable' (C2ST is the arbiter, for real).
    flip = (is_sep17 and is_sep18
            and dir17 != 0 and dir18 != 0 and np.sign(dir17) != np.sign(dir18))
    collapse = (is_sep17 and not is_sep18)
    weak = (not is_sep17 and not is_sep18)
    c2st_shifted = bool(np.isfinite(c2st_calibrated_pooled)
                        and c2st_calibrated_pooled > Config10.C2ST_SHIFT_THRESHOLD)
    if flip:
        verdict = 'flipped'
    elif collapse:
        verdict = 'collapsed'
    elif weak:
        verdict = 'weak'
    elif route == 'structural_change' and c2st_shifted:
        verdict = 'restructured'
    elif c2st_shifted:
        verdict = 'shifted'
    else:
        verdict = 'stable'

    # E1 cross-metric agreement (reinstated): does each corroboration metric's
    # own null-calibrated verdict AGREE with the C2ST decision? Pure corroboration — a
    # disagreement flags a feature whose shift evidence is metric-dependent (read its Q-Q/overlap
    # plots before leaning on it); it never changes the verdict.
    e1_agreement: dict = {}
    for mkey in ('wasserstein_qn', 'mmd', 'energy_distance', 'ks_statistic',
                 'anderson_darling', 'jensen_shannon'):
        cal = pooled.get(f'{mkey}_calibrated')
        if cal is None or not np.isfinite(cal):
            continue
        e1_agreement[mkey] = bool((cal > 0.0) == c2st_shifted)
    e1_agreement_rate = (float(np.mean([v for v in e1_agreement.values()]))
                         if e1_agreement else float('nan'))

    # MI-aware combined separation stability (the H1 dependent variable in step 11): ~1 preserved,
    # ~0 collapsed, <0 flipped — now using the combined strength so non-monotonic separation counts.
    base_stab = min(max(sepstr17, 0.0), 1.0) * min(max(sepstr18, 0.0), 1.0)
    if flip:                                   # same trusted-both-directions rule as the verdict
        sep_stability = -base_stab
    else:
        sep_stability = base_stab

    layer_b = {
        'c2st_auc': c2st['auc'],
        'c2st_auc_ci_low': c2st['ci_low'],           # 95% CI from the CV folds (stability)
        'c2st_auc_ci_high': c2st['ci_high'],
        'c2st_auc_ci_width': c2st['ci_width'],
        'c2st_auc_folds': c2st['folds'],
        # Calibrated C2ST-AUC (this is what step 11 uses for H1/H1.5/H2 — raw is reference only).
        'c2st_null_pooled': c2st_null_pooled,
        'c2st_calibrated_pooled': c2st_calibrated_pooled,
        'c2st_null_benign': benign.get('c2st_null'),
        'c2st_calibrated_benign': benign.get('c2st_calibrated'),
        'c2st_null_attack': attack.get('c2st_null'),
        'c2st_calibrated_attack': attack.get('c2st_calibrated'),
        # The C2ST verdict flag + E1 agreement (corroboration metrics vs the C2ST decision).
        'c2st_shifted': c2st_shifted,
        'e1_agreement': e1_agreement,
        'e1_agreement_rate': e1_agreement_rate,
        # Calibrated corroboration metrics (pooled-only; E1 inputs, never the verdict).
        'wasserstein_qn_null': pooled.get('wasserstein_qn_null'),
        'wasserstein_qn_calibrated': pooled.get('wasserstein_qn_calibrated'),
        'mmd_null': pooled.get('mmd_null'),
        'mmd_calibrated': pooled.get('mmd_calibrated'),
        'energy_distance_null': pooled.get('energy_distance_null'),
        'energy_distance_calibrated': pooled.get('energy_distance_calibrated'),
        'ks_statistic_null': pooled.get('ks_statistic_null'),
        'ks_statistic_calibrated': pooled.get('ks_statistic_calibrated'),
        'anderson_darling_null': pooled.get('anderson_darling_null'),
        'anderson_darling_calibrated': pooled.get('anderson_darling_calibrated'),
        'jensen_shannon_null': pooled.get('jensen_shannon_null'),
        'jensen_shannon_calibrated': pooled.get('jensen_shannon_calibrated'),
        'marginal_shift_magnitude': shift_mag,   # descriptive route-family distance (not the verdict)
        'separation_stability': float(sep_stability),                 # MI-aware (used by step 11)
        'separation_stability_auc': float(                            # legacy folded-AUC-only
            separation_stability(sep17, dir17, sep18, dir18, strong)),
        'mutual_info_norm_2017': mi17, 'mutual_info_norm_2018': mi18,
        'null_separation_threshold_2017': float(null_sep17),
        'null_separation_threshold_2018': float(null_sep18),
        'separation_strong_effective_2017': float(eff_strong17),
        'separation_strong_effective_2018': float(eff_strong18),
        'null_mi_threshold_2017': float(null_mi17),
        'null_mi_threshold_2018': float(null_mi18),
        'mi_strong_effective_2017': float(eff_mi_strong17),
        'mi_strong_effective_2018': float(eff_mi_strong18),
        'per_class_stability': per_class_stability,
        'flip_corroborated': bool(flip_corroborated),   # pooled flip backed by a real family flip?
        'n_family_flips': int(n_family_flips),
        'cardinality': cardinality,
        'variance_2017': variance_2017,
        'variance_2018': variance_2018,
        'route': route,
        # Joint-shape routing context from script 9 (drives the 'restructured' verdict).
        'n_modes_2017': int(plan.get('n_modes_2017', p17.get('n_modes', 1))),
        'n_modes_2018': int(plan.get('n_modes_2018', p18.get('n_modes', 1))),
        'modality_mismatch': bool(plan.get('modality_mismatch', False)),
        'recommended_scale': scale,
    }
    if per_mode_results:
        # Same mode counts as the plan; per-mode adds the blob-to-blob shift magnitudes.
        layer_b['n_modes_2017'] = per_mode_results['n_modes_2017']
        layer_b['n_modes_2018'] = per_mode_results['n_modes_2018']
        layer_b['max_mode_shift'] = per_mode_results['max_mode_shift']
        layer_b['max_mode_mass_shift'] = per_mode_results['max_mode_mass_shift']
    if zero_mass:
        layer_b['zero_frac_2017'] = zero_mass['zero_frac_2017']
        layer_b['zero_frac_2018'] = zero_mass['zero_frac_2018']
        layer_b['zero_frac_delta'] = zero_mass['zero_frac_delta']
        layer_b['tail_wasserstein_qn'] = zero_mass['tail_wasserstein_qn']

    # Q-Q diagnostics (numbers behind the Q-Q plot): slope/intercept/R^2 + shape class +
    # quantile-shift breakdown (P25/P50/P75 — where in the distribution the shift concentrates).
    qq = qq_diagnostics(v17, v18)

    layer_a = {
        'feature': feature, 'detected_type': det, 'verdict': verdict,
        'route': route, 'comparison_mode': comparison_mode,
        'c2st': {'auc': c2st['auc'], 'ci_low': c2st['ci_low'],
                 'ci_high': c2st['ci_high'], 'ci_width': c2st['ci_width'],
                 'folds': c2st['folds'], 'null_pooled': c2st_null_pooled,
                 'calibrated_pooled': c2st_calibrated_pooled},
        'axis1_pooled': pooled,
        'axis1_benign': benign,                 # benign-only shift (environment change); feeds E5 prior-shift analysis
        'axis1_attack': attack,                 # attack-only shift (adversary change); feeds E5 prior-shift analysis
        'axis1_per_attack': per_attack_axis1,   # per-family shift (instability attribution)
        'axis2_separation_2017': sep17, 'axis2_separation_2018': sep18,
        'axis2_direction_2017': dir17, 'axis2_direction_2018': dir18,
        'axis2_mi_norm_2017': mi17, 'axis2_mi_norm_2018': mi18,
        'axis2_per_class_stability': per_class_stability,
        'axis2_family_flips': family_flips,                 # shared families that genuinely reversed
        'flip_corroborated': bool(flip_corroborated),       # pooled flip backed by a real family flip
        'axis1_zero_mass': zero_mass,
        'per_mode_results': per_mode_results.get('per_mode', []),
        'flip_detected': bool(flip), 'collapse_detected': bool(collapse),
        'e1_agreement': e1_agreement,
        'e1_agreement_rate': e1_agreement_rate,
        'qq': qq,
        'thresholds': {'separation_strong': strong, 'mi_strong': mi_strong,
                       'c2st_shift_threshold': Config10.C2ST_SHIFT_THRESHOLD,
                       'c2st_null_pooled': float(c2st_null_pooled)},
    }
    # Surface the compact Q-Q scalars in Layer B too (flat, cross-feature comparable).
    if qq is not None:
        layer_b['qq_slope'] = qq['slope']
        layer_b['qq_intercept'] = qq['intercept']
        layer_b['qq_r2'] = qq['r2']
        layer_b['qq_shape_class'] = qq['shape_class']
        qshift = qq.get('qshift', {})
        for key, val in qshift.items():                     # qshift_p25 / qshift_p50 / qshift_p75
            layer_b[f'qshift_{key}'] = val
        layer_b['qshift_dominant'] = qq.get('qshift_dominant')

    if overlap_dir is not None:
        try:
            plot_overlap(feature, v17, v18, p17, p18, verdict, overlap_dir)
        except Exception as e:
            log.warn(f'  overlap plot failed for {feature}: {e}')

    # v17/v18 are not needed beyond this point — explicitly release the ~480 MB arrays
    # so the worker process doesn't accumulate them across the features it handles.
    del v17, v18
    gc.collect()

    if qq_dir is not None and qq is not None:
        try:
            plot_qq(feature, qq, verdict, qq_dir)
        except Exception as e:
            log.warn(f'  qq plot failed for {feature}: {e}')

    return {'feature': feature, 'verdict': verdict, 'layer_a': layer_a, 'layer_b': layer_b}


def plot_overlap(feature, v17, v18, p17, p18, verdict, out_dir):
    """
    Cross-year overlap for one feature: 2017 and 2018 on the same axis.
    Uses the profile's scale and robust view range.
    """
    scale = p17.get('recommended_scale', 'linear')
    sp = p17.get('scale_param')
    lo = min(p17.get('view_low', np.nanmin(v17)), p18.get('view_low', np.nanmin(v18)))
    hi = max(p17.get('view_high', np.nanmax(v17)), p18.get('view_high', np.nanmax(v18)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, 1.0

    rng = np.random.default_rng(Config10.SEED)
    fig, ax = plt.subplots(figsize=(12, 4))
    for vals, color, name, y in ((v17, '#2980b9', '2017', 0), (v18, '#c0392b', '2018', 1)):
        x = vals[np.isfinite(vals)]
        x = x[(x >= lo) & (x <= hi)]
        if x.size > Config10.OVERLAP_SAMPLE:
            x = rng.choice(x, Config10.OVERLAP_SAMPLE, replace=False)
        jitter = y + rng.normal(0, 0.06, size=x.size)
        ax.scatter(x, jitter, s=8, alpha=0.25, color=color, edgecolors='none', label=name)
    if scale == 'log':
        ax.set_xscale('log')
    elif scale == 'symlog':
        ax.set_xscale('symlog', linthresh=sp or 1e-6)
    ax.set_xlim(lo, hi)
    ax.set_yticks([0, 1]); ax.set_yticklabels(['2017', '2018'])
    ax.set_xlabel(f'{feature} (native units, scale={scale})', fontsize=9)
    ax.set_title(f'{feature}  ->  VERDICT: {verdict.upper()}'
                 f'  (sep17={p17.get("separation_magnitude", 0):.2f}, '
                 f'sep18={p18.get("separation_magnitude", 0):.2f})', fontsize=9)
    ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    out_path = out_dir / f'{safe_filename(feature)}.png'
    fig.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    return out_path.name


def qq_diagnostics(v17: np.ndarray, v18: np.ndarray) -> dict | None:
    """
    Numbers behind the Q-Q plot. Compares the two years quantile-for-quantile and fits a line
    q18 ~ slope*q17 + intercept, then classifies how the distribution moved:
      identical       slope~1, intercept~0, high R^2   -> on y=x
      location_shift  slope~1, intercept!=0            -> same shape, moved (gap preserved)
      scale_change    slope!=1                         -> spread grew/shrank
      shape_change    low R^2 (curved)                 -> skew/tails differ = real drift
    Also extracts the quantile-shift breakdown at Config10.QUANTILE_BREAKPOINTS (P25/P50/P75) so
    step 11 can see WHERE in the distribution the shift concentrates.
    """
    s17 = _subsample(v17, Config10.QQ_SAMPLE)
    s18 = _subsample(v18, Config10.QQ_SAMPLE)
    if s17.size < 10 or s18.size < 10:
        return None
    q17 = np.quantile(s17, _QQ_QUANTILES)
    q18 = np.quantile(s18, _QQ_QUANTILES)
    base = {'quantiles': _QQ_QUANTILES.tolist(),
            'q_2017': q17.tolist(), 'q_2018': q18.tolist()}

    # Quantile-shift breakdown: signed (q18 - q17) at each breakpoint; the dominant region is the
    # breakpoint with the largest |shift| (native-unit deltas within ONE feature are comparable).
    bps = Config10.QUANTILE_BREAKPOINTS
    qv17 = np.interp(bps, _QQ_QUANTILES, q17)
    qv18 = np.interp(bps, _QQ_QUANTILES, q18)
    deltas = np.abs(qv18 - qv17)
    base['qshift'] = {f'p{int(round(bp * 100))}': float(b - a)
                      for bp, a, b in zip(bps, qv17, qv18)}
    base['qshift_dominant'] = (f'p{int(round(bps[int(np.argmax(deltas))] * 100))}'
                               if np.any(np.isfinite(deltas)) and float(np.max(deltas)) > 0
                               else 'none')

    # Degenerate guard: near-constant feature in either year (e.g. all-zero flag counts like
    # URG/CWR/ECE Flag Count). A constant quantile vector makes np.polyfit singular ("SVD did
    # not converge" + LAPACK DLASCLS spam), so classify WITHOUT fitting and keep the feature.
    span17 = float(np.ptp(q17))
    span18 = float(np.ptp(q18))
    if span17 < 1e-12 or span18 < 1e-12:
        same = abs(float(np.median(q17)) - float(np.median(q18))) < 1e-9
        return {**base, 'slope': float('nan'), 'intercept': float('nan'),
                'r2': float('nan'), 'degenerate': True,
                'shape_class': 'identical' if same else 'location_shift'}

    try:
        slope, intercept = np.polyfit(q17, q18, 1)
    except Exception:
        return {**base, 'slope': float('nan'), 'intercept': float('nan'),
                'r2': float('nan'), 'degenerate': True, 'shape_class': 'undetermined'}

    pred = intercept + slope * q17
    ss_res = float(np.sum((q18 - pred) ** 2))
    ss_tot = float(np.sum((q18 - np.mean(q18)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    span = span17 or 1e-12
    if r2 < Config10.QQ_R2_SHAPE:
        shape = 'shape_change'
    elif abs(slope - 1) > Config10.QQ_SLOPE_TOL:
        shape = 'scale_change'
    elif abs(intercept) > Config10.QQ_INTERCEPT_TOL * span:
        shape = 'location_shift'
    else:
        shape = 'identical'
    return {**base, 'slope': float(slope), 'intercept': float(intercept),
            'r2': float(r2), 'degenerate': False, 'shape_class': shape}


def plot_qq(feature, qq: dict, verdict: str, out_dir):
    """
    Q-Q plot: 2017 quantiles (x) vs 2018 quantiles (y) against the y=x line.
      on y=x            -> same distribution
      parallel, offset  -> location shift (same shape, moved)
      different slope   -> scale change
      curved/S-shaped   -> shape change (real drift)
    Points are colored by quantile so you can see WHERE in the distribution the divergence is.
    """
    q17 = np.asarray(qq['q_2017'], dtype=np.float64)
    q18 = np.asarray(qq['q_2018'], dtype=np.float64)
    slope, intercept, r2 = qq['slope'], qq['intercept'], qq['r2']
    lo = float(min(q17.min(), q18.min()))
    hi = float(max(q17.max(), q18.max()))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, 1.0

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([lo, hi], [lo, hi], color='#333', ls='--', lw=1.2, label='y = x (identical)', zorder=2)
    if np.isfinite(slope) and np.isfinite(intercept):
        ax.plot(q17, intercept + slope * q17, color='#e74c3c', lw=1.0, alpha=0.85, zorder=2,
                label=f'fit y={slope:.2f}x{intercept:+.3g} (R^2={r2:.3f})')
    sc = ax.scatter(q17, q18, c=_QQ_QUANTILES, cmap='viridis', s=22, zorder=3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(f'{feature} — 2017 quantiles', fontsize=9)
    ax.set_ylabel(f'{feature} — 2018 quantiles', fontsize=9)
    ax.set_title(f'Q-Q: {feature}  ->  {verdict.upper()}\n[{qq["shape_class"]}]', fontsize=9)
    fig.colorbar(sc, ax=ax, label='quantile', fraction=0.046, pad=0.04)
    ax.legend(fontsize=7, loc='upper left')
    fig.tight_layout()
    out_path = out_dir / f'{safe_filename(feature)}.png'
    fig.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    return out_path.name


def plot_verdict_scatter(layer_b_table, out_path):
    """
    Hypothesis overview: every feature on calibrated C2ST-AUC (x — the Axis-1 decision value)
    vs separation-stability (y — Axis 2), colored by verdict. Step 11 overlays importance on this.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    for verdict, color in _VERDICT_COLORS.items():
        pts = [(d.get('c2st_calibrated_pooled'), d.get('separation_stability'))
               for d in layer_b_table.values() if d.get('verdict') == verdict]
        pts = [(x, y) for x, y in pts
               if isinstance(x, (int, float)) and np.isfinite(x)
               and isinstance(y, (int, float)) and np.isfinite(y)]
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, s=60, alpha=0.75, color=color, label=f'{verdict} ({len(xs)})',
                       edgecolors='white', linewidths=0.5)
    ax.axhline(0, color='#888', lw=0.8, ls='--')
    ax.axvline(Config10.C2ST_SHIFT_THRESHOLD, color='#888', lw=0.8, ls='--')
    ax.set_xlabel('Calibrated C2ST-AUC (Axis 1 decision value; > threshold = shifted)', fontsize=10)
    ax.set_ylabel('Separation stability (Axis 2): 1=preserved, 0=collapsed, <0=flipped', fontsize=10)
    ax.set_title('Per-feature cross-year verdict: Axis-1 (calibrated C2ST) vs Axis-2', fontsize=11)
    ax.legend(fontsize=9, loc='best')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def write_flat_table(results: list, layer_b_table: dict, out_path):
    """
    One row per feature, all the verdict numbers flattened to CSV -- for scanning a lot of data
    at once (Excel / pandas) instead of reading the nested JSON. Complements, not replaces,
    verdicts_layerA.json (full detail incl. Q-Q quantile arrays) and verdicts_layerB.json.
    """
    rows = []
    for a in results:
        f = a['feature']
        b = layer_b_table.get(f, {})
        pooled = a.get('axis1_pooled', {})
        qq = a.get('qq') or {}
        qshift = qq.get('qshift', {})
        rows.append({
            'feature': f,
            'verdict': a.get('verdict'),
            'detected_type': a.get('detected_type'),
            'route': a.get('route'),
            'c2st_auc': b.get('c2st_auc'),
            'c2st_auc_ci_low': b.get('c2st_auc_ci_low'),
            'c2st_auc_ci_high': b.get('c2st_auc_ci_high'),
            'c2st_auc_ci_width': b.get('c2st_auc_ci_width'),
            'c2st_null_pooled': b.get('c2st_null_pooled'),
            'c2st_calibrated_pooled': b.get('c2st_calibrated_pooled'),
            'c2st_null_benign': b.get('c2st_null_benign'),
            'c2st_calibrated_benign': b.get('c2st_calibrated_benign'),
            'c2st_null_attack': b.get('c2st_null_attack'),
            'c2st_calibrated_attack': b.get('c2st_calibrated_attack'),
            'c2st_shifted': b.get('c2st_shifted'),
            'e1_agreement_rate': b.get('e1_agreement_rate'),
            'wasserstein_qn_null': b.get('wasserstein_qn_null'),
            'wasserstein_qn_calibrated': b.get('wasserstein_qn_calibrated'),
            'jensen_shannon_null': b.get('jensen_shannon_null'),
            'jensen_shannon_calibrated': b.get('jensen_shannon_calibrated'),
            'mmd_null': b.get('mmd_null'),
            'mmd_calibrated': b.get('mmd_calibrated'),
            'energy_distance_null': b.get('energy_distance_null'),
            'energy_distance_calibrated': b.get('energy_distance_calibrated'),
            'ks_statistic_null': b.get('ks_statistic_null'),
            'ks_statistic_calibrated': b.get('ks_statistic_calibrated'),
            'anderson_darling_null': b.get('anderson_darling_null'),
            'anderson_darling_calibrated': b.get('anderson_darling_calibrated'),
            'marginal_shift_magnitude': b.get('marginal_shift_magnitude'),
            'separation_stability': b.get('separation_stability'),
            'sep_2017': a.get('axis2_separation_2017'),
            'sep_2018': a.get('axis2_separation_2018'),
            'dir_2017': a.get('axis2_direction_2017'),
            'dir_2018': a.get('axis2_direction_2018'),
            'wasserstein_qn': pooled.get('wasserstein_qn'),
            'mmd': pooled.get('mmd'),
            'energy_distance': pooled.get('energy_distance'),
            'anderson_darling': pooled.get('anderson_darling'),
            'ks_statistic': pooled.get('ks_statistic'),
            'n_modes_2017': b.get('n_modes_2017'),
            'n_modes_2018': b.get('n_modes_2018'),
            'modality_mismatch': b.get('modality_mismatch'),
            'recommended_scale': b.get('recommended_scale'),
            'qq_slope': qq.get('slope'),
            'qq_intercept': qq.get('intercept'),
            'qq_r2': qq.get('r2'),
            'qq_shape_class': qq.get('shape_class'),
            'qshift_p25': qshift.get('p25'),
            'qshift_p50': qshift.get('p50'),
            'qshift_p75': qshift.get('p75'),
            'qshift_dominant': qq.get('qshift_dominant'),
            'cardinality': b.get('cardinality'),
            'variance_2017': b.get('variance_2017'),
            'variance_2018': b.get('variance_2018'),
            'flip_detected': a.get('flip_detected'),
            'collapse_detected': a.get('collapse_detected'),
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)


def _sensitivity_sweep(layer_b_table: dict, thresholds: tuple = None) -> list:
    """Re-derive verdict counts at alternate CALIBRATED-C2ST verdict thresholds from the
    already-computed per-feature values (the verdict rule is calibrated C2ST > threshold).
    Only stable/shifted/restructured move with the threshold;
    flip/collapse/weak are separation-based and threshold-independent."""
    if thresholds is None:
        thresholds = (Config10.C2ST_SHIFT_THRESHOLD,) + tuple(Config10.SENSITIVITY_THRESHOLDS)
    rows = []
    for thr in thresholds:
        cts: dict = {'stable': 0, 'weak': 0, 'shifted': 0, 'flipped': 0,
                     'collapsed': 0, 'restructured': 0}
        for b in layer_b_table.values():
            v = b.get('verdict', 'shifted')
            if v in ('flipped', 'collapsed', 'weak'):
                cts[v] = cts.get(v, 0) + 1
                continue
            cal = b.get('c2st_calibrated_pooled')
            shifted = isinstance(cal, (int, float)) and np.isfinite(cal) and cal > thr
            if not shifted:
                cts['stable'] += 1
            elif b.get('route') == 'structural_change':
                cts['restructured'] += 1
            else:
                cts['shifted'] += 1
        rows.append({'threshold': thr, **cts})
    return rows


def _write_verdict_report(ds1: str, ds2: str, results: list, counts: dict,
                          layer_b_table: dict,
                          a_out: 'Path', b_out: 'Path', table_out: 'Path',
                          overlap_dir: 'Path', qq_dir: 'Path',
                          out_path: 'Path', log) -> None:
    lines: list[str] = []

    def h(t: str) -> None:
        lines.extend(['', '=' * 70, t, '=' * 70])

    h(f'EXECUTE COMPARISON REPORT — {ds1} <-> {ds2}')
    lines.append(f'Generated : {datetime.now():%Y-%m-%d %H:%M:%S}')
    lines.append(f'Features  : {len(results)} evaluated')
    lines.append('')
    lines.append('  This report is diagnostic output for the Axis 1/Axis 2 shift tests computed in')
    lines.append('  this step. It feeds step 11 (cross-analysis), which correlates these shift/')
    lines.append('  stability values against feature importance — see results.md for the canonical')
    lines.append('  C1-C9/H1/H1.5/H2 analysis IDs and what each one means.')
    lines.append('')

    h('VERDICT DISTRIBUTION (Flip vs Collapse: HOW did discrimination fail?)')
    total = sum(counts.values())
    for verdict, n in sorted(counts.items()):
        pct = 100.0 * n / total if total else 0.0
        lines.append(f'  {verdict:<14} : {n:3d}  ({pct:.1f}%)')

    # Sensitivity sweep — how verdict counts move as the calibrated-C2ST verdict threshold varies.
    # If the stable count barely moves across the sweep, the stable/shifted split is robust to the
    # exact threshold choice; if it swings, caveat the split.
    sweep = _sensitivity_sweep(layer_b_table)
    h('SENSITIVITY SWEEP (robustness — verdict counts at alternate calibrated-C2ST thresholds)')
    lines.append('  Verdict rule: calibrated C2ST-AUC (excess over the per-feature permutation null)')
    lines.append(f'  > threshold -> shifted. Current threshold: {Config10.C2ST_SHIFT_THRESHOLD}.')
    lines.append('')
    lines.append(f'  {"threshold":>9} | {"stable":>6} | {"weak":>4} | {"shifted":>8} | '
                 f'{"flipped":>7} | {"collapsed":>9} | {"restruct":>8}')
    lines.append(f'  {"-"*9}-+-{"-"*6}-+-{"-"*4}-+-{"-"*8}-+-{"-"*7}-+-{"-"*9}-+-{"-"*8}')
    for row in sweep:
        marker = '  <- current' if abs(row['threshold'] - Config10.C2ST_SHIFT_THRESHOLD) < 1e-9 else ''
        lines.append(
            f'  {row["threshold"]:>9.2f} | {row.get("stable", 0):>6} | {row.get("weak", 0):>4} | '
            f'{row.get("shifted", 0):>8} | {row.get("flipped", 0):>7} | {row.get("collapsed", 0):>9} | '
            f'{row.get("restructured", 0):>8}{marker}')
    lines.append('')
    lines.append('  Interpretation: a stable count that barely moves across the sweep means the')
    lines.append('  stable/shifted split is robust to the exact threshold; a big swing needs a caveat.')

    # E1 cross-metric agreement (reinstated): per-metric agreement rate with the
    # calibrated-C2ST verdict + the features whose corroboration disagrees. Diagnostic only.
    metric_names = ('wasserstein_qn', 'mmd', 'energy_distance', 'ks_statistic',
                    'anderson_darling', 'jensen_shannon')
    per_metric: dict = {m: [] for m in metric_names}
    disagreeing: list = []
    for fname, b in layer_b_table.items():
        agr = b.get('e1_agreement') or {}
        for m, ok in agr.items():
            per_metric.setdefault(m, []).append(bool(ok))
        rate = b.get('e1_agreement_rate')
        if isinstance(rate, (int, float)) and np.isfinite(rate) and rate < 1.0:
            dis = sorted(m for m, ok in agr.items() if not ok)
            disagreeing.append((fname, float(rate), dis))
    if any(per_metric.values()):
        h('E1 CROSS-METRIC AGREEMENT (corroboration vs the calibrated-C2ST verdict — diagnostic only)')
        lines.append('  Each corroboration metric, calibrated against its OWN permutation null, votes')
        lines.append('  shifted/stable; agreement = same vote as the C2ST verdict. Disagreement never')
        lines.append('  changes a verdict — it flags features whose shift evidence is metric-dependent.')
        lines.append('')
        for m in metric_names:
            votes = per_metric.get(m) or []
            if votes:
                lines.append(f'    {m:<18} : {100.0 * np.mean(votes):5.1f}% agreement '
                             f'({sum(votes)}/{len(votes)} features)')
        if disagreeing:
            lines.append('')
            lines.append(f'  features with at least one disagreeing metric ({len(disagreeing)}):')
            for fname, rate, dis in sorted(disagreeing, key=lambda t: t[1])[:15]:
                lines.append(f'    {fname[:40]:<40}  agreement={rate:.2f}  disagrees: {", ".join(dis)}')
            if len(disagreeing) > 15:
                lines.append(f'    ... and {len(disagreeing) - 15} more (see e1_agreement in layer B)')

    # Flip-corroboration audit: of the pooled 'flipped' verdicts, how many are backed by a real
    # shared-family direction reversal vs how many are likely attack-mixture artifacts.
    flips = [r for r in results if r.get('verdict') == 'flipped']
    if flips:
        corro = [r for r in flips if r.get('flip_corroborated')]
        h('FLIP CORROBORATION AUDIT (pooled flip vs real per-family reversal)')
        lines.append(f'  pooled flipped                : {len(flips)}')
        lines.append(f'  corroborated by a family flip : {len(corro)}  '
                     f'({100.0 * len(corro) / len(flips):.1f}%)')
        lines.append(f'  UNcorroborated (likely mixture artifact): {len(flips) - len(corro)}')
        if corro:
            lines.append('  corroborated features (family that flipped):')
            for r in corro:
                fams = ', '.join(r.get('axis2_family_flips', []))
                lines.append(f'    {r["feature"][:40]:<40}  [{fams}]')

    # C2ST-AUC distribution + STABILITY (CI width from the CV folds).
    c2st_vals  = [b.get('c2st_auc') for b in layer_b_table.values()
                  if isinstance(b.get('c2st_auc'), (int, float))]
    ci_widths  = [(f, b.get('c2st_auc_ci_width')) for f, b in layer_b_table.items()
                  if isinstance(b.get('c2st_auc_ci_width'), (int, float))]
    if c2st_vals:
        h('C2ST-AUC DISTRIBUTION + STABILITY (raw data — note: per-feature CIs are wide)')
        lines.append(f'  mean   : {np.mean(c2st_vals):.4f}')
        lines.append(f'  median : {np.median(c2st_vals):.4f}')
        lines.append(f'  min    : {min(c2st_vals):.4f}')
        lines.append(f'  max    : {max(c2st_vals):.4f}')
        if ci_widths:
            widths = [w for _, w in ci_widths]
            lines.append(f'  mean 95% CI width : {np.mean(widths):.4f}  '
                         f'(wider = less fold-stable)')
            top = sorted(ci_widths, key=lambda t: t[1], reverse=True)[:5]
            lines.append('  least stable (widest CI):')
            for f, w in top:
                lines.append(f'    {f[:40]:<40}  CI width={w:.4f}')

    # Quantile-shift breakdown — where in the distribution the shift concentrates.
    dom_counts = Counter(
        (r.get('qq') or {}).get('qshift_dominant')
        for r in results
        if (r.get('qq') or {}).get('qshift_dominant') not in (None, 'none'))
    if dom_counts:
        h('QUANTILE-SHIFT BREAKDOWN (WHERE in the distribution did the shift occur?)')
        lines.append('  Where the largest |q18 - q17| sits (P25=lower, P50=median, P75=upper):')
        for region, n in sorted(dom_counts.items()):
            lines.append(f'    {region:<6} : {n}')

    h('OUTPUT')
    lines.append(f'  Layer A JSON      : {a_out}')
    lines.append(f'  Layer B JSON      : {b_out}')
    lines.append(f'  flat CSV table    : {table_out}')
    lines.append(f'  overlap plots     : {overlap_dir}')
    lines.append(f'  Q-Q plots         : {qq_dir}')
    lines.append(f'  verdict scatter   : {out_path.parent / "verdict_scatter.png"}')
    lines.append(f'  steps log         : {out_path.parent / Config10.STEPS_FILE}')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(lines), encoding='utf-8')
    log.ok(f'Saved {out_path.name}')


def main():
    ap = argparse.ArgumentParser(description='Script 10: execute comparisons, produce verdicts.')
    ap.add_argument('--datasets', nargs='+', default=list(DATASETS))
    args = ap.parse_args()

    # Cross-dataset comparison needs exactly two datasets; fall back to the full default pair.
    dsets = args.datasets if len(args.datasets) >= 2 else list(DATASETS)
    ds1, ds2 = dsets[0], dsets[1]

    rdir = Config10.RESULTS_DIR
    odir = Config10.OUTPUT_DIR
    rdir.mkdir(parents=True, exist_ok=True)
    odir.mkdir(parents=True, exist_ok=True)
    overlap_dir = rdir / 'overlap_plots'; overlap_dir.mkdir(parents=True, exist_ok=True)
    qq_dir = rdir / 'qq_plots'; qq_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(rdir / Config10.STEPS_FILE, step_prefix=10,
                 title=f'SCRIPT 10 EXECUTE COMPARISON — {ds1} <-> {ds2}')

    # 10.1 — load script 9's plan and both profiles
    log.step('Load plan & profiles')
    plans_file = COMPARE_OUTPUT_DIR / f'comparison_plans_{ds1}_{ds2}.json'
    if not plans_file.exists():
        log.warn(f'plans missing: {plans_file} -- run 9_plan_comparison.py first')
        log.step_end(); log.close(); return
    with open(plans_file, encoding='utf-8') as f:
        plans = json.load(f)
    with open(profile_json_path(ds1), encoding='utf-8') as f:
        p1 = json.load(f)
    with open(profile_json_path(ds2), encoding='utf-8') as f:
        p2 = json.load(f)
    feats = [f for f in plans if f in p1 and f in p2]
    log.ok(f'{len(plans)} plans, {len(p1)}/{len(p2)} profiles -> {len(feats)} features to execute')
    log.step_end()

    # 10.2 — execute every feature's comparison in PARALLEL. Each worker loads both encoded labels +
    # both profiles + plans ONCE (initializer); per feature it reads the two columns, computes the
    # verdict (incl. per-mode + null threshold + C2ST CI + Q-Q breakdown), writes its overlap/qq
    # plots, and returns Layer A/B.
    log.step('Execute comparisons')
    n_workers = plan_workers(len(feats), Config10.MAX_WORKERS)
    log.info(f'{len(feats)} features, {n_workers} workers')
    results: list = []
    layer_b_table: dict = {}
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_exec_init,
                             initargs=(ds1, ds2, str(overlap_dir), str(qq_dir))) as ex:
        futs = {ex.submit(_exec_one, f): f for f in feats}
        for i, fut in enumerate(as_completed(futs), 1):
            feature, la, lb, err = fut.result()
            if err:
                log.warn(f'  {feature}: {err}')
            elif la is not None:
                results.append(la)
                layer_b_table[feature] = lb
            if i % 10 == 0:
                log.info(f'  {i}/{len(feats)} compared (last: {feature})')
    log.ok(f'{len(results)} features executed')
    log.step_end()

    # 10.3 — E1 cross-metric agreement is computed per-feature inside execute_one (10.2) and
    # summarized in the report (10.5); no separate pass needed here.

    # 10.4 — save verdicts (Layer A/B JSON), the flat CSV
    log.step('Save verdicts & tables')
    a_out = odir / f'verdicts_layerA_{ds1}_{ds2}.json'
    b_out = odir / f'verdicts_layerB_{ds1}_{ds2}.json'
    with open(a_out, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    with open(b_out, 'w', encoding='utf-8') as f:
        json.dump(layer_b_table, f, indent=2, default=str)
    table_out = odir / f'verdicts_table_{ds1}_{ds2}.csv'
    try:
        write_flat_table(results, layer_b_table, table_out)
        log.ok(f'flat table -> {table_out.name}')
    except Exception as e:
        log.warn(f'flat table failed: {e}')
    log.step_end()

    # 10.5 — render the verdict scatter (overlap/Q-Q plots were written by the workers in 10.2)
    log.step('Render plots')
    scatter_path = rdir / 'verdict_scatter.png'
    try:
        plot_verdict_scatter(layer_b_table, scatter_path)
        log.ok(f'verdict scatter -> {scatter_path.name}')
    except Exception as e:
        log.warn(f'verdict scatter failed: {e}')
    log.step_end()

    # 10.6 — write the human-readable report
    log.step('Write report')
    counts: dict = {}
    for r in results:
        counts[r['verdict']] = counts.get(r['verdict'], 0) + 1
    log.info('  ' + ', '.join(f'{k}={c}' for k, c in sorted(counts.items())))
    _write_verdict_report(ds1, ds2, results, counts, layer_b_table,
                          a_out, b_out, table_out, overlap_dir, qq_dir,
                          rdir / Config10.RESULTS_FILE, log)
    log.step_end()

    log.close()


if __name__ == '__main__':
    main()
