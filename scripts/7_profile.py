"""
7_profile.py — Per-feature distribution profiler / characterizer for NIDS datasets.

PURPOSE:
  For each feature, in each dataset independently, discover its shape and compute every
  descriptive fact downstream steps need. This script is the single source of truth for
  distribution facts; the visualizer (8) and the comparison (9-10) consume profiles.json and
  never re-derive them. Reads the CLEANED, RAW-UNIT parquet written by script 1 (never the
  z-scored preprocessing output in output/4_preprocessing).

SUB-STEPS:
  Sub-step 7.1 (Discover feature columns):
    - List analyzable columns from the cleaned parquet (excludes Label + identifiers).
    Input: data/cc_data/<ds>_cleaned.parquet
    Output: ordered list of feature names

  Sub-step 7.2 (Profile features):
    - Profile every feature in parallel (one process per feature; label read once per worker).
    - Per feature: type/quality detection, exact statistics, full percentiles, tail metrics
      (skewness/kurtosis), variability, entropy, outliers, scale, modality, and label
      separation (folded AUC + KSG mutual information, per attack family).
    Input: feature columns from 7.1
    Output: in-memory dict of per-feature profiles

  Sub-step 7.3 (Write profiles.json):
    - Serialize all profiles to JSON for downstream steps.
    Input: profiles from 7.2
    Output: output/7_profile/<ds>/profiles.json

  Sub-step 7.4 (Write report):
    - Write a brief human-readable summary (type/characteristic counts, output paths).
    Input: profiles from 7.2
    Output: results/7_profile/<ds>/7_profile_report.txt

GUARANTEES:
  - No source data is modified (read-only on the cleaned parquet).
  - Exact statistics and the display range use the FULL column; only the expensive AUC/MI/
    modality estimators use a deterministic stratified sample (minorities kept in full).
  - profiles.json contains one entry per analyzed feature.

NOTES:
  - Heavy estimators (KSG MI, GMM) are super-linear, so they run on capped samples
    (Config7.MI_SAMPLE_MAX / GMM_SAMPLE_MAX) — a documented speed trade-off.
  - Multimodality is gated by the Hartigan dip test when `diptest` is installed.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from scipy.stats import skew, kurtosis
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif   # KSG mutual information
from sklearn.mixture import GaussianMixture                 # GMM+BIC modes

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unified_config import (
    Config7, Logger, ensure_results_dir, DATASETS, plan_workers,
    BENIGN_LABEL, feature_columns, is_identifier, read_feature_only, read_labels_encoded,
)

try:
    from diptest import diptest as _diptest
    _HAVE_DIPTEST = True
except Exception:
    _HAVE_DIPTEST = False


def detect_type(feature: str, finite: np.ndarray, n_total: int, n_zero: int) -> dict:
    """Auto-detect type + flags from name, cardinality, integer-ness, value semantics.
    `n_zero` (count of finite==0) is precomputed once by profile_feature_core and passed in —
    same value as int(np.sum(finite == 0)), just not recomputed here."""
    name = feature.lower()
    n_unique = int(np.unique(finite).size) if finite.size else 0
    is_intlike = finite.size > 0 and np.all(finite == np.round(finite))

    if is_identifier(feature):
        return {'detected_type': 'identifier', 'zero_inflated': False, 'has_sentinel': False,
                'sentinel_value': None, 'has_impossible_values': False, 'degenerate': n_unique <= 1}

    if name in ('dst port', 'protocol'):
        return {'detected_type': 'nominal', 'zero_inflated': False, 'has_sentinel': False,
                'sentinel_value': None, 'has_impossible_values': False, 'degenerate': n_unique <= 1}

    degenerate = n_unique <= 1

    # Sentinel: -1 used as "N/A" (ICMP), or a single frequent negative value in a count feature.
    has_sentinel, sentinel_value = False, None
    if name in Config7.SENTINEL_COLS and finite.size and finite.min() < 0:
        has_sentinel, sentinel_value = True, float(finite.min())
    elif is_intlike and finite.size and finite.min() < 0:
        c = Counter(finite.tolist())
        mn = finite.min()
        if c[mn] > n_total * Config7.SENTINEL_MASS_FRAC:
            has_sentinel, sentinel_value = True, float(mn)

    # Impossible: a negative value in a column that should be non-negative (extraction bug).
    has_impossible = False
    if finite.size and finite.min() < 0 and not has_sentinel:
        if any(h in name for h in Config7.NONNEG_HINTS):
            has_impossible = True

    zero_frac = n_zero / n_total if n_total else 0.0
    zero_inflated = zero_frac >= Config7.ZERO_INFLATION_THRESH

    if n_unique <= 2:
        det = 'binary'
    elif n_unique <= Config7.TYPE_N_UNIQUE_NOMINAL:
        det = 'low-cardinality-discrete'
    elif is_intlike and n_unique <= Config7.TYPE_N_UNIQUE_DISCRETE:
        det = 'discrete-count'
    else:
        det = 'continuous'

    return {'detected_type': det, 'zero_inflated': zero_inflated, 'has_sentinel': has_sentinel,
            'sentinel_value': sentinel_value, 'has_impossible_values': has_impossible,
            'degenerate': degenerate}


def basic_stats(finite: np.ndarray, n_total: int, n_zero: int, p25: float, p75: float,
                median: float) -> dict:
    """Exact descriptive statistics on the full finite column (central tendency, spread, tails).
    `n_zero`/`p25`/`p75`/`median` are precomputed once by profile_feature_core (same values as
    int(np.sum(finite == 0)), np.quantile(finite, 0.25), np.quantile(finite, 0.75), and
    np.median(finite) respectively) and passed in instead of being recomputed here."""
    if finite.size == 0:
        return {'n_total': n_total, 'n_missing': n_total, 'n_zero': 0, 'zero_fraction': 0.0,
                'min': 0.0, 'max': 0.0, 'mean': 0.0, 'median': 0.0, 'std': 0.0, 'range': 0.0,
                'iqr': 0.0, 'mad': 0.0, 'skewness': 0.0, 'kurtosis': 0.0, 'sentinel_mass': 0.0}
    med = median
    return {
        'n_total': n_total,
        'n_missing': int(n_total - finite.size),
        'n_zero': n_zero,
        'zero_fraction': float(n_zero / n_total) if n_total else 0.0,
        'min': float(finite.min()),
        'max': float(finite.max()),
        'mean': float(np.mean(finite)),
        'median': med,
        'std': float(np.std(finite)),
        'range': float(finite.max() - finite.min()),
        'iqr': float(p75 - p25),
        'mad': float(np.median(np.abs(finite - med))),
        # Tail behaviour: skew = asymmetry, excess kurtosis = tail heaviness (need >2 points).
        'skewness': float(skew(finite)) if finite.size > 2 else 0.0,
        'kurtosis': float(kurtosis(finite, fisher=True)) if finite.size > 2 else 0.0,
    }


def percentile_stats(finite: np.ndarray, pcts: tuple, p25: float, median: float,
                     p75: float) -> dict:
    """Full percentile distribution (p01..p99) for downstream scaling/clipping decisions.
    p25/p50(median)/p75 are precomputed once by profile_feature_core and reused here instead of
    being recomputed — np.percentile(finite, 25/50/75) and np.quantile(finite, 0.25/0.75) /
    np.median(finite) are the same linear-interpolation order statistic on the same array, so the
    precomputed values are numerically identical to what this function used to compute itself.
    The remaining percentiles (everything in `pcts` other than 25/50/75) are still computed here
    with the original batched np.percentile call."""
    if finite.size == 0:
        return {f'p{p:02d}': 0.0 for p in pcts}
    precomputed = {25: float(p25), 50: float(median), 75: float(p75)}
    remaining = [p for p in pcts if p not in precomputed]
    out = dict(precomputed)
    if remaining:
        qs = np.percentile(finite, remaining)
        for p, v in zip(remaining, qs):
            out[p] = float(v)
    return {f'p{p:02d}': out[p] for p in pcts if p in out}


def variability_stats(finite: np.ndarray) -> dict:
    """Relative spread: coefficient of variation = std/mean (unit-free, comparable across scales)."""
    if finite.size == 0:
        return {'coefficient_of_variation': 0.0}
    mean = float(np.mean(finite))
    cv = float(np.std(finite) / mean) if mean != 0 else 0.0
    return {'coefficient_of_variation': cv}


def entropy_stat(finite: np.ndarray, bins: int) -> dict:
    """Shannon entropy (bits) of a binned histogram — higher = more spread/random, lower = ordered."""
    if finite.size == 0:
        return {'entropy': 0.0}
    hist, _ = np.histogram(finite, bins=bins)
    total = hist.sum()
    if total == 0:
        return {'entropy': 0.0}
    prob = hist / total
    nz = prob[prob > 0]
    return {'entropy': float(-np.sum(nz * np.log2(nz)))}


def outlier_stats(finite: np.ndarray, n_total: int, mult: float, p25: float, p75: float) -> dict:
    """Tukey-fence outlier counts/fraction (below Q1-k*IQR or above Q3+k*IQR). Fraction over n_total.
    `p25`/`p75` are precomputed once by profile_feature_core (same values as
    np.quantile(finite, 0.25)/0.75) and passed in instead of being recomputed here."""
    if finite.size == 0:
        return {'outlier_count_low': 0, 'outlier_count_high': 0, 'outlier_fraction': 0.0}
    q1 = p25
    q3 = p75
    iqr = q3 - q1
    lo, hi = q1 - mult * iqr, q3 + mult * iqr
    n_lo = int(np.sum(finite < lo))
    n_hi = int(np.sum(finite > hi))
    frac = (n_lo + n_hi) / n_total if n_total else 0.0
    return {'outlier_count_low': n_lo, 'outlier_count_high': n_hi, 'outlier_fraction': float(frac)}


def robust_view_range(finite: np.ndarray, has_impossible: bool, lo_q=Config7.VIEW_LO_Q, hi_q=Config7.VIEW_HI_Q):
    """
    Display range for the visualizer (Sec 4.2.1): clip the VIEW to robust quantiles so a single
    extraction-error outlier cannot fog the whole plot. Impossible (negative) values are excluded
    from the range entirely. Returns (low, high, n_clipped_low, n_clipped_high). Statistics are
    computed on full data elsewhere; this only affects what the eye sees.
    """
    if has_impossible:
        finite = finite[finite >= 0]
    if finite.size == 0:
        return 0.0, 1.0, 0, 0
    lo = float(np.quantile(finite, lo_q))
    hi = float(np.quantile(finite, hi_q))
    if hi <= lo:
        hi = lo + 1.0
    n_lo = int(np.sum(finite < lo))
    n_hi = int(np.sum(finite > hi))
    return lo, hi, n_lo, n_hi


def select_scale(det_type: str, finite: np.ndarray, zero_inflated: bool):
    """Pick representation scale from profiled shape (Sec 3.3). Returns (scale, param)."""
    if det_type in ('identifier', 'nominal', 'binary', 'low-cardinality-discrete', 'discrete-count'):
        return ('categorical', None)
    if finite.size == 0:
        return ('linear', None)
    pos = finite[finite > 0]
    has_zero_or_neg = np.any(finite <= 0)
    if zero_inflated or has_zero_or_neg:
        # linear near zero, log in the tails; threshold at a small positive quantile of |x|.
        nz = np.abs(finite[finite != 0])
        thr = float(np.quantile(nz, 0.10)) if nz.size else 1.0
        return ('symlog', max(thr, 1e-9))
    if pos.size and pos.max() / pos.min() > Config7.LOG_RATIO_THRESH:
        return ('log', None)
    return ('linear', None)


def _fit_gmm_modes(t: np.ndarray, base: np.ndarray, max_k: int) -> list:
    """
    Fit GMMs with 1..max_k components on the (scale-transformed) values `t`, pick the best by BIC,
    and return per-component {center, spread, mass} in NATIVE units. `base` are the native values
    aligned with `t`. Each component's spread is computed from the points ASSIGNED to it (argmax
    responsibility).
    """
    if t.size < Config7.MIN_SAMPLES_GMM:
        return [{'center': float(np.median(base)), 'spread': float(np.std(base)), 'mass': 1.0}]
    tt = t.reshape(-1, 1)
    best = None
    for k in range(1, max_k + 1):
        try:
            gm = GaussianMixture(n_components=k, covariance_type='full',
                                 random_state=Config7.SEED, n_init=1, max_iter=100, reg_covar=1e-6)
            gm.fit(tt)
            bic = gm.bic(tt)
        except Exception:
            continue
        if best is None or bic < best[0]:
            best = (bic, gm)
    if best is None:
        return [{'center': float(np.median(base)), 'spread': float(np.std(base)), 'mass': 1.0}]
    gm = best[1]
    labels = gm.predict(tt)
    modes = []
    for c in range(gm.n_components):
        m = labels == c
        if not np.any(m):
            continue
        b = base[m]
        modes.append({'center': float(np.median(b)), 'spread': float(np.std(b)),
                      'mass': float(m.mean())})
    modes = [md for md in modes if md['mass'] >= Config7.MIN_GMM_MODE_MASS] or modes
    modes.sort(key=lambda d: d['center'])
    return modes


def detect_modality(finite: np.ndarray, scale: str, sentinel_value):
    """
    Strip zero/sentinel masses, assess modality on the chosen scale (Sec 3.4). Gate with the dip
    test (if available); if multimodal, fit GMM+BIC for real per-mode center/spread/weight. Runs on
    a marginal subsample (Config7.GMM_SAMPLE_MAX) for speed. Centers in NATIVE units.
    """
    if sentinel_value is not None:
        finite = finite[finite != sentinel_value]
    cont = finite[finite != 0]
    if cont.size < Config7.MIN_SAMPLES_CONT:
        if cont.size == 0:
            return [{'center': 0.0, 'spread': 0.0, 'mass': 1.0}]
        return [{'center': float(np.median(cont)), 'spread': float(np.std(cont)), 'mass': 1.0}]

    # marginal subsample for shape analysis (modality is a property of the feature, not the classes)
    if cont.size > Config7.GMM_SAMPLE_MAX:
        cont = np.random.default_rng(Config7.SEED).choice(cont, Config7.GMM_SAMPLE_MAX, replace=False)

    # transform to the analysis scale so log-normal-ish modes become separable
    if scale in ('log', 'symlog'):
        pos = cont[cont > 0]
        if pos.size < Config7.MIN_SAMPLES_CONT:
            return [{'center': float(np.median(cont)), 'spread': float(np.std(cont)), 'mass': 1.0}]
        t, base = np.log10(pos), pos
    else:
        t, base = cont, cont

    # dip-test gate: clearly unimodal -> one mode (skip GMM)
    if _HAVE_DIPTEST and t.size >= Config7.MIN_SAMPLES_CONT:
        try:
            _, pval = _diptest(t)
        except Exception:
            pval = 1.0
        if pval > Config7.DIP_TEST_PVALUE:
            return [{'center': float(np.median(base)), 'spread': float(np.std(base)), 'mass': 1.0}]

    return _fit_gmm_modes(t, base, Config7.GMM_MAX_K)


def separation_one(values: np.ndarray, y_bin: np.ndarray) -> dict:
    """
    Axis-2 separation, MONOTONIC view: folded AUC magnitude + signed direction (Sec 3.5). Kept as a
    SECONDARY measure — paired with mutual_information() which catches the non-monotonic case (attack
    blob between two benign blobs) that folded AUC reports as ~0.5.

    Direction is derived from the RAW AUC sign (rank-based, prevalence-invariant), with a deadband
    (Config7.DIRECTION_AUC_BAND) that zeroes the direction when the AUC is too close to 0.5 to carry
    a trustworthy sign.
    """
    mask = np.isfinite(values)
    x, y = values[mask], y_bin[mask]
    if np.unique(y).size < 2 or x.size == 0:
        return {'magnitude': 0.5, 'direction': 0.0, 'auc_raw': 0.5}
    try:
        auc = roc_auc_score(y, x)
    except Exception:
        return {'magnitude': 0.5, 'direction': 0.0, 'auc_raw': 0.5}
    folded = max(auc, 1 - auc)
    band = Config7.DIRECTION_AUC_BAND
    direction = float(np.sign(auc - 0.5)) if abs(auc - 0.5) >= band else 0.0
    return {'magnitude': float(folded), 'direction': direction, 'auc_raw': float(auc)}


def mutual_information(x: np.ndarray, y_bin: np.ndarray) -> dict:
    """
    KSG mutual information between a feature and the binary attack label. Unlike folded AUC it
    catches NON-monotonic separation. Returns MI in nats and MI normalized by the label entropy ->
    a [0,1] fraction. Computed on a capped sample (Config7.MI_SAMPLE_MAX) because the KSG k-NN
    estimator is super-linear.
    `mi_normalized` is fraction of the STRATIFIED SAMPLE's label uncertainty
    explained, not the true population's — both the MI numerator AND the entropy denominator (`h`
    below) are computed on the SAME post-subsample `y`, since the class-balance-preserving cap
    happens before either is computed. This is internally consistent (MI/entropy always agree on
    which distribution they describe) but means 2017 and 2018 are each normalized against their
    OWN stratified sample's class balance, which can differ slightly between years — a per-year-
    varying normalization basis, not a fixed one, if this value is compared cross-year.
    """
    mask = np.isfinite(x)
    x, y = x[mask], y_bin[mask]
    if x.size == 0 or np.unique(y).size < 2:
        return {'mi_nats': 0.0, 'mi_normalized': 0.0}
    if x.size > Config7.MI_SAMPLE_MAX:
        idx = np.random.default_rng(Config7.SEED).choice(x.size, Config7.MI_SAMPLE_MAX, replace=False)
        x, y = x[idx], y[idx]
    try:
        mi = float(mutual_info_classif(x.reshape(-1, 1), y,
                                       discrete_features=False, random_state=Config7.SEED)[0])
    except Exception:
        return {'mi_nats': 0.0, 'mi_normalized': 0.0}
    p = float(np.mean(y))  # SAMPLE class balance (see docstring) — not the true population's
    h = 0.0 if p <= 0.0 or p >= 1.0 else -(p * np.log(p) + (1 - p) * np.log(1 - p))  # entropy, nats
    mi_norm = float(min(1.0, mi / h)) if h > 1e-12 else 0.0
    return {'mi_nats': max(mi, 0.0), 'mi_normalized': max(mi_norm, 0.0)}


def per_class_separation(x_s: np.ndarray, codes_s: np.ndarray, categories: list) -> dict:
    """
    Per attack-family separation vs Benign on the (already stratified-sampled) data, one-vs-benign.
    Uses integer label codes (fast) and reports folded AUC + direction + normalized MI per family.
    """
    out = {}
    benign_code = categories.index(BENIGN_LABEL) if BENIGN_LABEL in categories else -1
    benign = x_s[codes_s == benign_code]
    if benign.size == 0:
        return out
    for ci, cls in enumerate(categories):
        if ci == benign_code:
            continue
        attack = x_s[codes_s == ci]
        if attack.size == 0:
            continue
        x = np.concatenate([benign, attack])
        y = np.concatenate([np.zeros(benign.size, np.int8), np.ones(attack.size, np.int8)])
        d  = separation_one(x, y)
        mi = mutual_information(x, y)
        out[cls] = {'magnitude': d['magnitude'], 'direction': d['direction'],
                    'mi_normalized': mi['mi_normalized'], 'n_attack': int(attack.size)}
    return out


def _build_sample_index(codes: np.ndarray, per_class_cap: int, seed: int = Config7.SEED) -> np.ndarray:
    """
    STRATIFIED row indices — keep every row of any class with <= cap rows, else a uniform `cap` of
    it. Minority attacks (e.g. WebAttack ~200 rows) are preserved in full so per-class separation
    stays meaningful, while Benign is capped. Deterministic (fixed seed) -> reproducible profile.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for c in np.unique(codes):
        idx_c = np.flatnonzero(codes == c)
        if per_class_cap > 0 and idx_c.size > per_class_cap:
            idx_c = rng.choice(idx_c, per_class_cap, replace=False)
        parts.append(idx_c)
    out = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
    out.sort()
    return out


def profile_feature_core(ds: str, feature: str, codes: np.ndarray, categories: list,
                         y_bin: np.ndarray, sample_idx: np.ndarray) -> dict:
    """
    Profile one feature. Reads ONLY the feature column (label was read once upstream). Exact stats,
    percentiles, variability, entropy, outliers and the view range use the FULL column; the
    expensive AUC/MI/modality run on samples. `finite` is computed once and shared.
    """
    values = np.asarray(read_feature_only(ds, feature), dtype=np.float64)
    finite = values[np.isfinite(values)]
    n_total = len(values)

    # Computed ONCE here and threaded into every sub-function below instead of each one
    # independently recomputing the same np.sum(finite==0) / np.quantile(finite, .25/.75) /
    # np.median(finite) from the shared `finite` array (detect_type, basic_stats, outlier_stats,
    # and percentile_stats all used to redo this work separately).
    if finite.size:
        n_zero = int(np.sum(finite == 0))
        p25, p75 = (float(v) for v in np.quantile(finite, [0.25, 0.75]))
        median = float(np.median(finite))
    else:
        n_zero, p25, p75, median = 0, 0.0, 0.0, 0.0

    type_info = detect_type(feature, finite, n_total, n_zero)

    stats = basic_stats(finite, n_total, n_zero, p25, p75, median)
    stats.update(percentile_stats(finite, Config7.PERCENTILES, p25, median, p75))
    stats.update(variability_stats(finite))
    stats.update(entropy_stat(finite, Config7.ENTROPY_BINS))
    stats.update(outlier_stats(finite, n_total, Config7.OUTLIER_IQR_MULT, p25, p75))
    stats['sentinel_mass'] = (float(np.sum(finite == type_info['sentinel_value']) / n_total)
                              if type_info['sentinel_value'] is not None and n_total else 0.0)

    scale, scale_param = select_scale(type_info['detected_type'], finite, type_info['zero_inflated'])

    if type_info['detected_type'] in ('identifier', 'nominal', 'binary', 'low-cardinality-discrete'):
        modes = [{'center': stats['median'], 'spread': stats['iqr'], 'mass': 1.0}]
    else:
        modes = detect_modality(finite, scale, type_info['sentinel_value'])

    # Stratified sample for separation/MI (aligned with codes/y_bin) — fast on 63M-row columns.
    xs = values[sample_idx]
    ys = y_bin[sample_idx]
    cs = codes[sample_idx]
    sep     = separation_one(xs, ys)
    mi      = mutual_information(xs, ys)
    per_cls = per_class_separation(xs, cs, categories)
    v_lo, v_hi, n_clo, n_chi = robust_view_range(finite, type_info['has_impossible_values'])

    return {
        'feature': feature, 'dataset': ds,
        **type_info, **stats,
        'n_modes': len(modes), 'modes': modes,
        'recommended_scale': scale, 'scale_param': scale_param,
        'roc_auc_benign_vs_attack': sep['magnitude'],
        'separation_magnitude': sep['magnitude'],       # folded AUC (monotonic, secondary)
        'separation_direction': sep['direction'],       # sign(auc_raw - 0.5) with deadband; 0 = untrusted
        'separation_auc_raw': sep.get('auc_raw', 0.5),  # raw (unfolded) AUC, for auditing direction
        'mutual_info': mi['mi_nats'],                   # non-monotonic-capable (nats)
        'mutual_info_normalized': mi['mi_normalized'],  # [0,1], comparable to (auc-0.5)*2
        'per_class_separation': per_cls,
        'view_low': v_lo, 'view_high': v_hi,
        'n_clipped_low': n_clo, 'n_clipped_high': n_chi,
        'is_identifier': is_identifier(feature),
    }


# ── Parallel feature profiling (one process per feature; label read ONCE per worker) ────────────
_WS: dict = {}   # per-worker state, populated by the pool initializer


def _worker_init(ds: str, per_class_cap: int, seed: int) -> None:
    """Each worker loads the dictionary-encoded label ONCE (not per feature) and builds the
    stratified sample index once, then reuses both across every feature it handles."""
    codes, cats, y_bin = read_labels_encoded(ds)
    _WS['ds']         = ds
    _WS['codes']      = codes
    _WS['categories'] = cats
    _WS['y_bin']      = y_bin
    _WS['sample_idx'] = _build_sample_index(codes, per_class_cap, seed)


def _profile_one(feature: str):
    """Worker task: profile one feature using the per-worker shared label/sample state."""
    try:
        prof = profile_feature_core(_WS['ds'], feature, _WS['codes'], _WS['categories'],
                                    _WS['y_bin'], _WS['sample_idx'])
        return feature, prof, None
    except Exception as e:
        return feature, None, f'{type(e).__name__}: {e}'


def profile_features(ds: str, feats: list, n_workers: int, log: Logger) -> dict:
    """Profile every feature in parallel; collect results as they complete."""
    profiles: dict = {}
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init,
                             initargs=(ds, Config7.SEP_SAMPLE_CAP, Config7.SEED)) as ex:
        futs = {ex.submit(_profile_one, f): f for f in feats}
        for i, fut in enumerate(as_completed(futs), 1):
            feature, prof, err = fut.result()
            if err:
                log.warn(f'failed {feature}: {err}')
            else:
                profiles[feature] = prof
            if i % Config7.LOG_INTERVAL == 0:
                log.info(f'{i}/{len(feats)} profiled (last: {feature})')
    return profiles


def _write_profile_report(ds: str, profiles: dict, json_out: Path,
                          out_path: Path, log: Logger) -> None:
    lines: list[str] = []

    def h(t: str) -> None:
        lines.extend(['', '=' * 70, t, '=' * 70])

    h(f'PROFILE REPORT  --  {ds}')
    lines.append(f'Generated : {datetime.now():%Y-%m-%d %H:%M:%S}')
    lines.append(f'Features  : {len(profiles)} profiled')

    type_counts = Counter(p.get('detected_type', 'unknown') for p in profiles.values())
    h('TYPE DISTRIBUTION')
    for t, n in sorted(type_counts.items()):
        lines.append(f'  {t:<35} : {n}')

    zero_inf = sum(1 for p in profiles.values() if p.get('zero_inflated'))
    multimod = sum(1 for p in profiles.values() if p.get('n_modes', 1) > 1)
    skewed   = sum(1 for p in profiles.values() if abs(p.get('skewness', 0.0)) > 1.0)
    heavy    = sum(1 for p in profiles.values() if p.get('kurtosis', 0.0) > 1.0)
    outlying = sum(1 for p in profiles.values() if p.get('outlier_fraction', 0.0) > 0.01)
    h('CHARACTERISTICS')
    lines.append(f'  zero-inflated     : {zero_inf}')
    lines.append(f'  multimodal        : {multimod}')
    lines.append(f'  highly skewed     : {skewed}   (|skewness| > 1)')
    lines.append(f'  heavy-tailed      : {heavy}   (excess kurtosis > 1)')
    lines.append(f'  >1% outliers      : {outlying}')

    h('OUTPUT')
    lines.append(f'  profiles.json : {json_out}')
    lines.append(f'  report.txt    : {out_path}')
    lines.append(f'  steps.log     : {out_path.parent / Config7.STEPS_FILE}')

    text = '\n'.join(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding='utf-8')
    log.ok(f'Saved {out_path.name}')


def main():
    ap = argparse.ArgumentParser(description='Script 7: profile features from the cleaned parquet.')
    ap.add_argument('--datasets', nargs='+', default=list(DATASETS))
    args = ap.parse_args()

    Config7.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Config7.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    n_workers = plan_workers(os.cpu_count() or 4, Config7.MAX_WORKERS)

    for ds in args.datasets:
        rdir = ensure_results_dir(7, ds)
        log = Logger(rdir / Config7.STEPS_FILE, step_prefix=7)
        log.info(f'workers={n_workers}, sep-sample={Config7.SEP_SAMPLE_CAP:,}/class, '
                 f'include_identifiers={Config7.INCLUDE_IDENTIFIERS}')

        # 7.1 — discover feature columns from the cleaned parquet
        log.step('Discover feature columns')
        feats = feature_columns(ds, include_identifiers=Config7.INCLUDE_IDENTIFIERS)
        log.ok(f'{len(feats)} features to profile')
        log.step_end()

        # 7.2 — profile every feature in parallel
        log.step('Profile features')
        profiles = profile_features(ds, feats, n_workers, log)
        log.ok(f'{len(profiles)} features profiled')
        log.step_end()

        # 7.3 — write profiles.json (single source of truth for steps 8-10)
        log.step('Write profiles.json')
        out = Config7.OUTPUT_DIR / ds / 'profiles.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(profiles, f, indent=2, default=str)
        log.ok(f'Saved {out} ({len(profiles)} features)')
        log.step_end()

        # 7.4 — write the brief human-readable report
        log.step('Write report')
        _write_profile_report(ds, profiles, out, rdir / Config7.RESULTS_FILE, log)
        log.step_end()

        log.close()


if __name__ == '__main__':
    main()
