"""
11_cross_analysis.py — Importance × distribution-shift cross analysis for NIDS features (Branch A × B).

PURPOSE:
  Joins the model's per-feature reliance (Branch A: LightGBM gain + permutation importance) with
  the data's per-feature cross-dataset behaviour (Branch B: the two-axis distribution verdict,
  Axis 1 = calibrated C2ST), then tests the project hypotheses: H1 — important features show
  greater cross-dataset shift (the model leans on features whose P(Y|X) does not transport
  2017 -> 2018); H2 — a set of low-importance but stable features are transportable signal the
  model under-uses. Reads only finished step-5/step-10 artifacts (no raw-data reads); the
  ablation retrains on the preprocessed, model-ready parquet. This is the joint step that
  closes the pipeline.

SUB-STEPS:
  Sub-step 11.1 (Join importance with distribution verdicts):
    - Join importance (step 5) + Layer B verdicts (step 10) + profiles into one cross table
    - Assign each feature an importance×stability quadrant
    Input:  output/5_training/<ds>_<algorithm>/feature_importance_*.json,
            output/10_execute_comparison/verdicts_layer{A,B}_<ds1>_<ds2>.json
    Output: join_report.json, cross_table.csv / cross_table.json

  Sub-step 11.2 (Headline rank statistics):
    - Spearman/Kendall + bootstrap CI for importance vs shift/stability
    - Partial correlation controlling for cardinality+variance; collinearity-cluster bootstrap CI
    Input:  the cross table from 11.1
    Output: rank_correlation.json

  Sub-step 11.3 (Drift exposure vs permutation null):
    - Importance-weighted mean shift vs a label-permutation null
    Input:  the cross table from 11.1
    Output: drift_exposure.json

  Sub-step 11.4 (Bidirectional K rank test):
    - Sort by importance then by shift; confirm the relation holds both directions
    Input:  the cross table from 11.1
    Output: bidirectional_k.json

  Sub-step 11.5 (Per-attack-family breakdown):
    - Per attack family, each feature's per-class separation stability (Layer A)
    Input:  output/10_execute_comparison/verdicts_layerA_<ds1>_<ds2>.json
    Output: per_attack/<family>.csv

  Sub-step 11.6 (Visuals):
    - Quadrant scatters, C2ST bar, rank-rank scatter, shift-metric agreement heatmap
    Input:  the cross table from 11.1
    Output: quadrant_axis{1,2}.png, c2st_bar.png, rankrank_scatter.png, method_agreement_heatmap.png

  Sub-step 11.7 (Cross-domain ablation):
    - Retrain top-importance vs stable-only feature sets, score in- and cross-domain (decisive)
    Input:  output/4_preprocessing/<ds>/{train,test}.parquet, scaler.json, feature_names.json
    Output: ablation_results.csv, ablation_*.png

  Sub-step 11.8 (Generate results doc):
    - Cache all analysis outputs, then run 11_result_gen.py to write results.md
    Input:  all artifacts above
    Output: results/11_cross_analysis/<algorithm>/results.md

  Sub-step 11.9 (Rank stability):
    - Spearman correlation between 2017 and 2018 importance rankings + top-K overlap
    Input:  cross table from 11.1
    Output: rank_stability.json

  Sub-step 11.10 (Benign vs attack shift):
    - Per feature: compare benign-only shift vs attack-family shift (which class drove more change?)
    Input:  verdicts_layerA from step 10 (axis1_benign + axis1_per_attack)
    Output: benign_vs_attack_shift.json, benign_vs_attack_shift_per_feature.csv

  Sub-step 11.11 (MI preservation):
    - Check if mutual information (signal strength) survived the distribution shift
    Input:  cross table from 11.1 (mutual_info_norm_2017 / _2018 columns)
    Output: mi_preservation.json

  Sub-step 11.12 (Univariate transfer):
    - Train tiny decision tree per feature; evaluate on 2017 test + 2018 cross-domain test
    - Accuracy drop = which individual features survive the shift
    Input:  output/4_preprocessing/<ds>/{train,test}.parquet
    Output: univariate_transfer.csv

GUARANTEES:
  - Reads only finished upstream artifacts; no source data is modified.
  - An explicit precheck lists every missing upstream input before any work begins.
  - Outputs are algorithm-versioned (results/output/11_cross_analysis/<algorithm>/).

NOTES:
  - Cross-dataset step: SINGLE combined folder; the only sub-split is the algorithm suffix
    (lightgbm — the pipeline's only engine). All tuning lives in unified_config.Config11.
  - Run modes are Config11 toggles, not CLI flags: RUN_ABLATION (skip the expensive ablation) and
    ABLATION_ONLY (ablation only, skip stats/visuals). The script takes no CLI arguments.
  - Layer B verdicts are read from the step-10 dir 10_execute_comparison (renamed from 10_verdict).
"""

import sys
import json
import pickle
import argparse
import subprocess
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau, rankdata, pearsonr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unified_config import (  # noqa: E402
    PROJECT_ROOT, Config11, Logger, DATASETS, ALGORITHM,
    cross_output_dir, cross_results_dir, dataset_dir, load_capped_subsample,
    load_feature_names, train_path, test_path, training_output_dir,
    LABEL_BINARY, CLEANED_DATA_ROOT, BENIGN_LABEL, read_labels_encoded,
)

# ── Paths — algorithm-versioned at runtime via unified_config helpers ────────────
# OUTPUT_DIR / RESULTS_DIR are set in main() (algorithm-suffixed via unified_config.ALGORITHM).
# Use the cross_output_dir / cross_results_dir / training_output_dir helpers.
VERDICT_DIR = PROJECT_ROOT / 'output' / '10_execute_comparison'  # step 10 dir

# Resolved in main() — set as module-level vars so all stage functions can see them.
OUTPUT_DIR: Path
RESULTS_DIR: Path

DS1, DS2 = list(DATASETS)[:2]            # the two cross-compared years (DATASETS order)


# ════════════════════════════════════════════════════════════════════════════════
# Stage 0 — input loading
# ════════════════════════════════════════════════════════════════════════════════
def _require(path: Path, what: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f'{what} missing: {path}')
    return path


def _train_dir_for_ds(ds: str) -> Path:
    """Algorithm-versioned training output dir for one dataset
    (output/5_training/<ds>_<algorithm>/). The algorithm is taken from the resolved
    OUTPUT_DIR name set in main(), so load_importance stays algorithm-aware."""
    algorithm = OUTPUT_DIR.name   # 'lightgbm' (set in main; the only engine)
    return training_output_dir(PROJECT_ROOT, ds, algorithm)


def load_importance(ds: str, task: str, kind: str = 'native') -> dict:
    """
    feature -> importance for one dataset/task.
      kind='native' : feature_importance_<task>.json       (LightGBM GAIN importance,
                      normalized to sum 1 — the primary native importance; step 5 also writes
                      split-count as feature_importance_split_<task>.json, diagnostic only)
      kind='perm'   : feature_importance_perm_<task>.json  (permutation, UNBIASED,
                      overlap-free held-out sample — step 5)
    'perm' falls back to 'native' if the permutation file is absent (older step-5 runs).
    Reads from the algorithm-suffixed folder (output/5_training/cicids2017_lightgbm/).
    """
    ds_dir = _train_dir_for_ds(ds)
    if kind == 'perm':
        pp = ds_dir / f'feature_importance_perm_{task}.json'
        if pp.exists():
            data = json.loads(pp.read_text(encoding='utf-8'))
            return {feat: float(val) for feat, val in data['importances']}
        kind = 'native'   # fall through
    p = _require(ds_dir / f'feature_importance_{task}.json',
                 f'{ds} {task} importance [{OUTPUT_DIR.name}] (run: python main.py --steps 5)')
    data = json.loads(p.read_text(encoding='utf-8'))
    return {feat: float(val) for feat, val in data['importances']}


def has_perm_importance(ds: str, task: str = 'binary') -> bool:
    return (_train_dir_for_ds(ds) / f'feature_importance_perm_{task}.json').exists()


def load_perm_importance_full(ds: str, task: str = 'binary') -> dict:
    """feature -> {'mean': float, 'std': float} from the importances_full block in
    the permutation importance JSON (written by step 5 with PERM_REPEATS >= 2)."""
    ds_dir = _train_dir_for_ds(ds)
    pp = ds_dir / f'feature_importance_perm_{task}.json'
    if not pp.exists():
        return {}
    data = json.loads(pp.read_text(encoding='utf-8'))
    full = data.get('importances_full', [])
    return {rec['feature']: {'mean': float(rec['mean']), 'std': float(rec['std'])}
            for rec in full if isinstance(rec, dict) and 'feature' in rec}


def load_layer_b() -> dict:
    # main() runs an explicit precheck (lists all missing upstream files clearly) BEFORE
    # reaching here, so step 11 fails fast with guidance instead of a mid-run surprise.
    p = _require(VERDICT_DIR / f'verdicts_layerB_{DS1}_{DS2}.json',
                 'Branch B Layer B verdicts (run: python main.py --steps 10)')
    return json.loads(p.read_text(encoding='utf-8'))


def load_profiles(ds: str) -> dict:
    """feature -> step-7 profile dict (output/7_profile/<ds>/profiles.json). Best-effort,
    returns {} if the file is missing (older runs without step 7, or step 7 skipped).
    NOTE: duplicated in 11_result_gen.py, which needs the same loader independently for the
    doc-generation sections; both copies must stay in sync if this logic ever changes."""
    p = PROJECT_ROOT / 'output' / '7_profile' / ds / 'profiles.json'
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return {}


def load_layer_a() -> dict:
    """feature -> layer A record (for the multi-metric Wasserstein/MMD robustness check)."""
    p = VERDICT_DIR / f'verdicts_layerA_{DS1}_{DS2}.json'
    if not p.exists():
        return {}
    records = json.loads(p.read_text(encoding='utf-8'))
    return {r['feature']: r for r in records if 'feature' in r}


def _slice_shift(d: dict) -> float:
    """Extract a slice's Axis-1 shift value: the CALIBRATED per-slice C2ST-AUC. Slices carry
    C2ST only (the corroboration distances are pooled-only by design), so
    this is both the only and the RIGHT quantity: it is the one metric comparable across every
    feature and slice, each calibrated against its own null floor. Shared by every caller that
    reads an axis1_benign/axis1_attack/axis1_per_attack slice dict."""
    if not isinstance(d, dict):
        return float('nan')
    return float(d.get('c2st_calibrated', float('nan')))


# ════════════════════════════════════════════════════════════════════════════════
# Stage 1 — join + unified cross table
# ════════════════════════════════════════════════════════════════════════════════
def build_cross_table(log) -> tuple[pd.DataFrame, dict]:
    imp17b = load_importance(DS1, 'binary')
    imp18b = load_importance(DS2, 'binary')
    imp17m = load_importance(DS1, 'multiclass')
    imp18m = load_importance(DS2, 'multiclass')
    # Permutation importance (unbiased) — the version H1 is PRIMARILY tested under.
    perm_ok = has_perm_importance(DS1, 'binary') and has_perm_importance(DS2, 'binary')
    imp17p = load_importance(DS1, 'binary', kind='perm')
    imp18p = load_importance(DS2, 'binary', kind='perm')
    # Permutation importance std (M1: report noise alongside ranking)
    perm_full_17 = load_perm_importance_full(DS1)
    perm_full_18 = load_perm_importance_full(DS2)
    if perm_ok:
        log.info('  permutation importance found -> H1 tested under BOTH native and permutation')
    else:
        log.warn('  permutation importance MISSING (old step-5 run) -> permutation cols mirror '
                 'native; re-run step 5 for the unbiased H1.')
    layer_b = load_layer_b()
    layer_a = load_layer_a()
    profiles_17 = load_profiles(DS1)
    profiles_18 = load_profiles(DS2)

    model_feats = set(imp17b) & set(imp18b)          # features the model used in both years
    data_feats = set(layer_b)                         # features Branch B compared
    joined = sorted(model_feats & data_feats)
    only_model = sorted(model_feats - data_feats)
    only_data = sorted(data_feats - model_feats)

    # dense ranks (1 = most important)
    def ranks(d: dict) -> dict:
        feats = list(d)
        order = np.argsort([-d[f] for f in feats])   # descending importance
        r = {feats[order[i]]: i + 1 for i in range(len(feats))}
        return r
    rk17, rk18 = ranks(imp17b), ranks(imp18b)
    rkp17, rkp18 = ranks(imp17p), ranks(imp18p)       # permutation ranks

    def _bf(d: dict, key: str) -> float:
        """None-safe float read from a layer-B record. Step 10 writes some keys as literal
        None for routes where the metric doesn't exist (e.g. wasserstein_qn_calibrated on
        nominal/PMF features) — the key EXISTS, so dict.get's default never applies and
        float(None) would crash. Coerce None (and anything non-numeric) to NaN."""
        v = d.get(key, np.nan)
        try:
            return float(v) if v is not None else float('nan')
        except (TypeError, ValueError):
            return float('nan')

    rows = []
    for f in joined:
        b = layer_b[f]
        a = layer_a.get(f, {})
        pooled = a.get('axis1_pooled', {}) if isinstance(a, dict) else {}
        sep_stab = _bf(b, 'separation_stability')
        # Benign-only, all-attacks-combined, and per-attack marginal shift (for the benign-vs-attack shift breakdown)
        benign_d = a.get('axis1_benign', {}) if isinstance(a, dict) else {}
        attack_d = a.get('axis1_attack', {}) if isinstance(a, dict) else {}
        per_atk_d = a.get('axis1_per_attack', {}) if isinstance(a, dict) else {}
        # Slice shift = calibrated per-slice C2ST-AUC (slices carry C2ST only; FIX-3). The raw
        # per-slice C2ST is kept alongside for reference, same convention as the pooled columns.
        benign_shift = _slice_shift(benign_d)
        attack_shift = _slice_shift(attack_d)
        benign_shift_raw = float(benign_d.get('c2st_auc', np.nan))
        attack_shift_raw = float(attack_d.get('c2st_auc', np.nan))
        atk_shifts = [_slice_shift(v) for v in per_atk_d.values() if isinstance(v, dict)]
        atk_shifts = [v for v in atk_shifts if np.isfinite(v)]
        max_attack_shift = float(max(atk_shifts)) if atk_shifts else float('nan')
        zero_frac_17 = profiles_17.get(f, {}).get('zero_fraction', float('nan'))
        zero_frac_18 = profiles_18.get(f, {}).get('zero_fraction', float('nan'))
        rows.append({
            'feature': f,
            'imp_2017_bin': imp17b[f], 'imp_2018_bin': imp18b[f],
            'imp_2017_multi': imp17m.get(f, np.nan), 'imp_2018_multi': imp18m.get(f, np.nan),
            'imp_rank_2017': rk17[f], 'imp_rank_2018': rk18[f],
            'imp_mean': 0.5 * (imp17b[f] + imp18b[f]),
            'imp_rank_mean': 0.5 * (rk17[f] + rk18[f]),
            # permutation importance (unbiased) — primary for H1
            'imp_perm_2017_bin': imp17p.get(f, np.nan), 'imp_perm_2018_bin': imp18p.get(f, np.nan),
            'imp_perm_rank_2017': rkp17.get(f, np.nan), 'imp_perm_rank_2018': rkp18.get(f, np.nan),
            'imp_perm_mean': 0.5 * (imp17p.get(f, np.nan) + imp18p.get(f, np.nan)),
            'imp_perm_rank_mean': 0.5 * (rkp17.get(f, np.nan) + rkp18.get(f, np.nan)),
            # M1: permutation importance std alongside ranking (noise gauge for tail ranks)
            'imp_perm_std_2017_bin': perm_full_17.get(f, {}).get('std', float('nan')),
            'imp_perm_std_2018_bin': perm_full_18.get(f, {}).get('std', float('nan')),
            'c2st_auc': _bf(b, 'c2st_auc'),                              # raw (reference)
            # Calibrated pooled C2ST-AUC — THIS is what every H1/H1.5/H2 test below reads (Axis 1
            # is now the calibrated value, not raw; see calibrate_c2st() in 10_execute_comparison.py).
            'c2st_auc_calibrated': _bf(b, 'c2st_calibrated_pooled'),
            'c2st_auc_null': _bf(b, 'c2st_null_pooled'),
            'marginal_shift': _bf(b, 'marginal_shift_magnitude'),
            'separation_stability': sep_stab,
            'instability': 1.0 - sep_stab,
            'verdict': b.get('verdict', 'unknown'),
            'flip_corroborated': bool(b.get('flip_corroborated', False)),
            'n_family_flips': int(b.get('n_family_flips', 0)),
            'cardinality': _bf(b, 'cardinality'),
            'variance_2017': _bf(b, 'variance_2017'),
            'variance_2018': _bf(b, 'variance_2018'),
            'wasserstein_qn': float(pooled.get('wasserstein_qn', np.nan)),
            'mmd': float(pooled.get('mmd', np.nan)),
            'ks_statistic': float(pooled.get('ks_statistic', np.nan)),
            'zero_fraction_mean': float(np.nanmean([zero_frac_17, zero_frac_18])),
            'zero_fraction_2017': float(zero_frac_17),
            'zero_fraction_2018': float(zero_frac_18),
            'zero_fraction_delta': float(abs(zero_frac_18 - zero_frac_17))
                                   if np.isfinite(zero_frac_17) and np.isfinite(zero_frac_18) else float('nan'),
            # MI preservation — already computed by step 7 and stored in layer_b
            'mutual_info_norm_2017': _bf(b, 'mutual_info_norm_2017'),
            'mutual_info_norm_2018': _bf(b, 'mutual_info_norm_2018'),
            # Benign vs attack shift breakdown (calibrated per-slice C2ST; *_raw = uncalibrated)
            'benign_shift': benign_shift,
            'attack_shift': attack_shift,
            'benign_shift_raw': benign_shift_raw,
            'attack_shift_raw': attack_shift_raw,
            'max_attack_shift': max_attack_shift,
            # E1 corroboration agreement (step 10): fraction of corroboration metrics whose own
            # null-calibrated verdict agrees with the C2ST decision for this feature.
            'e1_agreement_rate': _bf(b, 'e1_agreement_rate'),
            'wasserstein_qn_calibrated': _bf(b, 'wasserstein_qn_calibrated'),
            # Q-Q diagnostics: shape of the shift, not just its magnitude
            'qq_slope': _bf(b, 'qq_slope'),
            'qq_intercept': _bf(b, 'qq_intercept'),
            'qq_r2': _bf(b, 'qq_r2'),
            'qq_shape_class': b.get('qq_shape_class', np.nan),
            'qshift_p25': _bf(b, 'qshift_p25'),
            'qshift_p50': _bf(b, 'qshift_p50'),
            'qshift_p75': _bf(b, 'qshift_p75'),
            'qshift_dominant': b.get('qshift_dominant', np.nan),
            # Per-mode ("blob") comparison — only populated for multimodal-routed features
            'n_modes_2017': _bf(b, 'n_modes_2017'),
            'n_modes_2018': _bf(b, 'n_modes_2018'),
            'modality_mismatch': bool(b.get('modality_mismatch', False)),
            'max_mode_shift': _bf(b, 'max_mode_shift'),
            'max_mode_mass_shift': _bf(b, 'max_mode_mass_shift'),
            # Zero-mass-separate comparison — only populated for zero-inflated features
            'zero_frac_2017': _bf(b, 'zero_frac_2017'),
            'zero_frac_2018': _bf(b, 'zero_frac_2018'),
            'zero_frac_delta': _bf(b, 'zero_frac_delta'),
            'tail_wasserstein_qn': _bf(b, 'tail_wasserstein_qn'),
        })
    df = pd.DataFrame(rows).set_index('feature', drop=False)

    report = {
        'n_joined': len(joined),
        'n_model_only': len(only_model),
        'n_data_only': len(only_data),
        'permutation_importance_available': bool(perm_ok),
        'joined': joined,
        'model_only_no_branchB': only_model,
        'data_only_no_importance': only_data,   # collinear twins + Dst Port/Protocol (expected)
    }
    log.info(f'  joined {len(joined)} features | model-only {len(only_model)} | '
             f'data-only {len(only_data)} (collinear twins / nominal — expected)')
    if len(joined) < Config11.JOIN_MIN_FEATURES:
        log.warn(f'  only {len(joined)} features joined — expected ~71. Check feature naming!')
    return df, report


# ════════════════════════════════════════════════════════════════════════════════
# Stage 3 — headline rank statistics (with bootstrap CI + partial correlation)
# ════════════════════════════════════════════════════════════════════════════════
def _finite_pairs(x, y) -> tuple[np.ndarray, np.ndarray]:
    """Mask both arrays to rows where BOTH are finite. Nominal features carry NaN in the
    Wasserstein/MMD columns; without this mask spearmanr propagates NaN to the headline
    statistic while the bootstrap silently keeps only the (biased) resamples that happened
    to exclude every NaN row."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def _spearman(x, y) -> float:
    x, y = _finite_pairs(x, y)
    if len(x) < 3:
        return float('nan')
    return float(spearmanr(x, y).correlation)


def bootstrap_ci(x: np.ndarray, y: np.ndarray,
                 n: int = Config11.BOOTSTRAP_N, seed: int = Config11.SEED) -> dict:
    x, y = _finite_pairs(x, y)   # resample only complete pairs (same universe as the point estimate)
    rng = np.random.default_rng(seed)
    m = len(x)
    if m < 3:
        return {'lo': float('nan'), 'med': float('nan'), 'hi': float('nan')}
    vals = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        v = _spearman(x[idx], y[idx])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return {'lo': float('nan'), 'med': float('nan'), 'hi': float('nan'), 'p_boot': float('nan')}
    lo, med, hi = np.percentile(vals, [2.5, 50, 97.5])
    varr = np.asarray(vals)
    # Empirical two-sided bootstrap p-value: 2x the smaller tail fraction crossing zero. Standard
    # bootstrap-test-inversion, consistent with everything else here being resample-based rather
    # than relying on a parametric null (see benjamini_hochberg() below for what consumes this).
    p_boot = float(min(1.0, 2.0 * min(np.mean(varr <= 0), np.mean(varr >= 0))))
    return {'lo': float(lo), 'med': float(med), 'hi': float(hi), 'p_boot': p_boot}


def _correlation_block(a, b, min_pairs: int = 0) -> dict | None:
    """Shared core of the Spearman/Kendall/bootstrap-CI "block" closures duplicated (with minor
    per-site variations) in headline_stats(), rank_change_vs_shift(), and
    delta_importance_vs_stability(): mask to finite pairs, compute Spearman, Kendall (only once
    there are >2 finite pairs — matches every site's own guard/guarantee), and the bootstrap CI.

    `min_pairs`: if >0 and the number of finite pairs is below it, returns None so the caller can
    build its own site-specific "insufficient data" payload (the three sites differ slightly in
    what they return/log for that case) instead of this helper guessing one for all of them.
    headline_stats() never hit this case (min_pairs=0 there — same as its original, guard-less
    `block()`), so it always gets a real dict back.
    """
    af, bf = _finite_pairs(a, b)
    if min_pairs and len(af) < min_pairs:
        return None
    sp = _spearman(a, b)
    kt = float(kendalltau(af, bf).correlation) if len(af) > 2 else float('nan')
    ci = bootstrap_ci(np.asarray(a, float), np.asarray(b, float))
    return {'spearman': sp, 'kendall': kt, 'bootstrap_ci95': ci, 'n_pairs': int(len(af))}


def partial_spearman(x: np.ndarray, y: np.ndarray, Z: np.ndarray) -> float:
    """Spearman partial correlation of x,y controlling for confounders Z (rank-residualized)."""
    mask = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    if mask.sum() < 5:
        return float('nan')
    xr = rankdata(x[mask]); yr = rankdata(y[mask])
    Zr = np.column_stack([rankdata(Z[mask, j]) for j in range(Z.shape[1])])
    A = np.column_stack([np.ones(mask.sum()), Zr])
    bx, *_ = np.linalg.lstsq(A, xr, rcond=None)
    by, *_ = np.linalg.lstsq(A, yr, rcond=None)
    rx, ry = xr - A @ bx, yr - A @ by
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return float('nan')
    return float(pearsonr(rx, ry)[0])


def feature_clusters(features: list, threshold: float = Config11.CLUSTER_CORR_THRESHOLD) -> np.ndarray:
    """Map each feature to a collinearity-cluster id so the cluster bootstrap treats correlated
    feature twins as ONE observation (effective n = cluster count, not the full feature set).

    Clusters = connected components of the graph on `features` with an edge wherever
    |avg Pearson| >= threshold, read from step-3 comparison_matrix.json (which covers the full
    common feature set). NOTE: the step-3 0.95 DROP groups are NOT usable here — their redundant
    members were already removed in step 4, so the surviving features would look independent;
    the residual 0.7-0.95 correlation AMONG the kept features is exactly the collinearity we must
    account for. Falls back to all-singletons if the matrix is missing/unreadable."""
    feats = list(features)
    n = len(feats)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    if Config11.COMPARISON_MATRIX.exists():
        try:
            d = json.loads(Config11.COMPARISON_MATRIX.read_text(encoding='utf-8'))
            ap = np.abs(np.asarray(d['avg_pearson'], dtype=float))
            pos = {f: i for i, f in enumerate(d['features'])}
            for i in range(n):
                if feats[i] not in pos:
                    continue
                for j in range(i + 1, n):
                    if feats[j] not in pos:
                        continue
                    if ap[pos[feats[i]], pos[feats[j]]] >= threshold:
                        union(i, j)
        except Exception:
            pass

    roots: dict = {}
    out = []
    for i in range(n):
        r = find(i)
        if r not in roots:
            roots[r] = len(roots)
        out.append(roots[r])
    return np.asarray(out, dtype=np.int64)


def cluster_bootstrap_ci(x: np.ndarray, y: np.ndarray, clusters: np.ndarray,
                         n: int = Config11.BOOTSTRAP_N, seed: int = Config11.SEED) -> dict:
    """Bootstrap the Spearman 95% CI by resampling CLUSTERS (collinear feature groups) with
    replacement instead of individual features, so collinear twins don't inflate the effective n
    and shrink the CI artificially. Returns lo/med/hi plus the cluster count."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    clusters = np.asarray(clusters)
    m = np.isfinite(x) & np.isfinite(y)
    x, y, clusters = x[m], y[m], clusters[m]
    uniq = np.unique(clusters)
    nan = float('nan')
    if len(x) < 3 or len(uniq) < 3:
        return {'lo': nan, 'med': nan, 'hi': nan, 'n_clusters': int(len(uniq))}
    idx_by = {int(c): np.flatnonzero(clusters == c) for c in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx_by[int(c)] for c in chosen])
        if sel.size < 3:
            continue
        v = _spearman(x[sel], y[sel])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return {'lo': nan, 'med': nan, 'hi': nan, 'n_clusters': int(len(uniq)), 'p_boot': nan}
    lo, med, hi = np.percentile(vals, [2.5, 50, 97.5])
    varr = np.asarray(vals)
    p_boot = float(min(1.0, 2.0 * min(np.mean(varr <= 0), np.mean(varr >= 0))))
    return {'lo': float(lo), 'med': float(med), 'hi': float(hi), 'n_clusters': int(len(uniq)),
            'p_boot': p_boot}


def benjamini_hochberg(pvals: list, alpha: float = 0.05) -> tuple:
    """Standard BH step-up FDR correction. Returns (reject, q_value) arrays aligned to `pvals`'
    input order; non-finite p-values pass through as not-rejected / NaN q-value.

    Added so the headline_stats() correlation blocks carry a multiple-comparisons-aware significance
    flag instead of only an unadjusted per-test 95% CI (running this many correlated
    sub-analyses means at least one nominally-significant result is expected by chance alone)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    reject = np.zeros(n, dtype=bool)
    q = np.full(n, np.nan)
    finite = np.isfinite(p)
    if not finite.any():
        return reject, q
    idx = np.flatnonzero(finite)
    order = idx[np.argsort(p[idx])]
    m = len(order)
    ranked_p = p[order]
    thresh = (np.arange(1, m + 1) / m) * alpha
    below = ranked_p <= thresh
    k_max = (np.flatnonzero(below).max() + 1) if below.any() else 0
    reject[order[:k_max]] = True
    q_sorted = np.clip(ranked_p * m / np.arange(1, m + 1), 0, 1)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]   # monotone step-up adjustment
    q[order] = q_sorted
    return reject, q


# The 8 H1 cells: {native, permutation} x {2017, 2018} x {Axis 1, Axis 2}. This tuple IS the
# H1 family — the FDR correction below runs over exactly these, matching
# H1; supporting analyses (partial correlation, group medians, H1.5,
# rank-change) are corrected within their own families or reported as exploratory.
H1_CELLS = (
    'importance_vs_c2st',                        # native x 2017 x axis1
    'importance_2018_vs_c2st',                   # native x 2018 x axis1
    'importance_perm_2017_vs_c2st',              # permutation x 2017 x axis1
    'importance_perm_2018_vs_c2st',              # permutation x 2018 x axis1
    'importance_vs_concept_stability',           # native x 2017 x axis2
    'importance_2018_vs_concept_stability',      # native x 2018 x axis2
    'importance_perm_vs_concept_stability',      # permutation x 2017 x axis2
    'importance_perm_2018_vs_concept_stability', # permutation x 2018 x axis2
)


def apply_fdr_to_headline_blocks(out: dict, alpha: float = 0.05) -> dict:
    """BH-correct across EXACTLY the 8 H1 cells (H1_CELLS), one p-value per cell: the
    collinearity-honest cluster-bootstrap p when available, else the plain bootstrap p.
    Writes 'fdr_significant' / 'q_value_bh' / 'fdr_p_source' onto each cell block in place.
    Returns a summary dict (also stored as out['fdr_correction_summary']) so 11_result_gen.py
    can report how many H1 claims survive correction without recomputing anything."""
    cells, pvals, sources = [], [], []
    for key in H1_CELLS:
        blk = out.get(key)
        if not isinstance(blk, dict):
            continue
        p, src = float('nan'), None
        clu = blk.get('cluster_bootstrap_ci95')
        if isinstance(clu, dict) and np.isfinite(clu.get('p_boot', float('nan'))):
            p, src = clu['p_boot'], 'cluster_bootstrap'
        else:
            ci = blk.get('bootstrap_ci95')
            if isinstance(ci, dict) and np.isfinite(ci.get('p_boot', float('nan'))):
                p, src = ci['p_boot'], 'plain_bootstrap'
        if src is not None:
            cells.append(key)
            pvals.append(p)
            sources.append(src)
    reject, qvals = benjamini_hochberg(pvals, alpha=alpha)
    for key, rej, q, src in zip(cells, reject, qvals, sources):
        out[key]['fdr_significant'] = bool(rej)
        out[key]['q_value_bh'] = float(q) if np.isfinite(q) else float('nan')
        out[key]['fdr_p_source'] = src
    return {
        'alpha': alpha,
        'family': 'the 8 H1 cells ({native,permutation} x {2017,2018} x {axis1,axis2})',
        'p_source': 'cluster bootstrap where available, else plain bootstrap',
        'n_tests': len(pvals),
        'n_significant_uncorrected': int(sum(1 for p in pvals if p < alpha)),
        'n_significant_bh_corrected': int(reject.sum()),
    }


def _preferred_importance_rank(df: pd.DataFrame, year: int = 2017) -> str:
    """Return the importance-rank column name to anchor on for `year`: the permutation-importance
    rank column (`imp_perm_rank_<year>`) when it exists and has at least one finite value, else the
    native rank column (`imp_rank_<year>`). This is the identical fallback guard that used to be
    copy-pasted at 4 call sites (headline_stats, quadrant_assignment, plot_quadrant,
    bidirectional_k_analysis) — returning just the column NAME (not the values) lets each call site
    derive whatever shape it actually needs (ndarray vs Series, negated or not, plus a label string
    at one site) without forcing a one-size-fits-all return type here."""
    perm_col = f'imp_perm_rank_{year}'
    if perm_col in df.columns and np.isfinite(df[perm_col].to_numpy(dtype=float)).any():
        return perm_col
    return f'imp_rank_{year}'


def headline_stats(df: pd.DataFrame, log) -> dict:
    # H1 fix: anchor to 2017 ONLY, matching _policies()/quadrant_assignment()/
    # bidirectional_k_analysis(), which were already fixed for the same reason — selection/labeling
    # elsewhere in this file is blind to 2018, so the headline correlation computed here must be
    # too, not silently averaged in. imp_rank_mean remains a valid stored column for other uses
    # (e.g. H1.5 rank-delta), just not as the H1 headline anchor.
    imp = df['imp_rank_2017'].to_numpy()        # lower rank = more important
    # Use NEGATIVE rank so "higher importance" increases with the variable -> intuitive sign.
    imp_hi = -imp
    stab = df['separation_stability'].to_numpy()
    # H1/H1.5/H2 decision variable: CALIBRATED C2ST-AUC (user decision — Axis 1 throughout is the
    # calibrated value, not raw; raw stays in df['c2st_auc'] for the C2a reference display only).
    c2st = df['c2st_auc_calibrated'].to_numpy()

    out = {'n_features': int(len(df))}

    def block(name, a, b):
        res = _correlation_block(a, b)                      # NaN-safe (nominal features)
        log.info(f'  {name:<40} spearman={res["spearman"]:+.3f}  kendall={res["kendall"]:+.3f}  '
                 f'CI95=[{res["bootstrap_ci95"]["lo"]:+.3f},{res["bootstrap_ci95"]["hi"]:+.3f}]  '
                 f'n={res["n_pairs"]}')
        return res

    _nan_block = {'spearman': float('nan'), 'kendall': float('nan'),
                  'bootstrap_ci95': {'lo': float('nan'), 'med': float('nan'), 'hi': float('nan')}}

    # THE H1 FAMILY IS EXACTLY THESE 8 CELLS: {native, permutation} x
    # {2017, 2018} x {Axis 1 = calibrated C2ST, Axis 2 = separation_stability}. The old extra
    # Wasserstein/MMD-vs-importance blocks were removed from this function: after FIX-1/FIX-2,
    # Wasserstein/MMD are corroboration-only metrics whose place is step 10's E1 agreement check,
    # not the hypothesis-test family — keeping them here inflated the test family past
    # H1 and muddied the FDR correction below.
    # Axis 1 (calibrated C2ST). Expected sign under H1: POSITIVE (important features shift more).
    out['importance_vs_c2st'] = block('imp_2017 vs calib C2ST (H1 axis1 native)', imp_hi, c2st)
    # Axis 2 (separation_stability = univariate attack-vs-benign gap survival; a UNIVARIATE PROXY
    # for concept stability — the H2 ablation (C9, decisive) is the definitive test).
    out['importance_vs_concept_stability'] = block('imp_2017 vs sep_stability', imp_hi, stab)

    _rank_arrays: dict = {}   # name -> already-computed -rank array, reused below (H1 cell inputs)
    if 'imp_rank_2018' in df.columns and np.isfinite(df['imp_rank_2018'].to_numpy(dtype=float)).any():
        imp18_hi = -df['imp_rank_2018'].to_numpy(dtype=float)
        _rank_arrays['imp_rank_2018'] = imp18_hi
        out['importance_2018_vs_c2st'] = block('imp_2018 vs calib C2ST', imp18_hi, c2st)
        out['importance_2018_vs_concept_stability'] = block('imp_2018 vs sep_stability', imp18_hi, stab)

    if _preferred_importance_rank(df, 2017) == 'imp_perm_rank_2017':
        imp_perm17_hi = -df['imp_perm_rank_2017'].to_numpy(dtype=float)
        _rank_arrays['imp_perm_rank_2017'] = imp_perm17_hi
        out['importance_perm_2017_vs_c2st'] = block('imp_perm_2017 vs calib C2ST', imp_perm17_hi, c2st)
        out['importance_perm_vs_concept_stability'] = block(
            'imp_perm_2017 vs sep_stability', imp_perm17_hi, stab)
    else:
        out['importance_perm_vs_concept_stability'] = dict(_nan_block)

    if 'imp_perm_rank_2018' in df.columns and np.isfinite(df['imp_perm_rank_2018'].to_numpy(dtype=float)).any():
        imp_perm18_hi = -df['imp_perm_rank_2018'].to_numpy(dtype=float)
        _rank_arrays['imp_perm_rank_2018'] = imp_perm18_hi
        out['importance_perm_2018_vs_c2st'] = block('imp_perm_2018 vs calib C2ST', imp_perm18_hi, c2st)
        out['importance_perm_2018_vs_concept_stability'] = block(
            'imp_perm_2018 vs sep_stability', imp_perm18_hi, stab)

    # Partial correlation: importance vs concept stability controlling for cardinality + variance
    # (supporting analysis, not an H1 cell).
    Z = np.column_stack([
        df['cardinality'].to_numpy(),
        np.nanmax(np.column_stack([df['variance_2017'], df['variance_2018']]), axis=1),
    ])
    pcorr = partial_spearman(imp_hi, stab, Z)
    out['importance_vs_concept_stability_partial_cardinality_variance'] = pcorr
    log.info(f'  partial (control cardinality,variance)  spearman_partial={pcorr:+.3f}')

    # Group medians — top-K vs bottom-K median C2ST shift (effect size, not just sign).
    group_medians = {}
    for K in (5, 10, 15):
        if K > len(df):
            continue
        imp_rank_arr = df['imp_rank_2017'].to_numpy(dtype=float)  # same 2017-only anchor as the headline blocks above
        top_k_idx = np.argsort(imp_rank_arr)[:K]    # lowest rank number = most important
        bot_k_idx = np.argsort(imp_rank_arr)[-K:]   # highest rank number = least important
        c2st_finite = c2st.copy(); c2st_finite[~np.isfinite(c2st_finite)] = float('nan')
        group_medians[str(K)] = {
            'top_k_median_c2st':    float(np.nanmedian(c2st_finite[top_k_idx])),
            'bottom_k_median_c2st': float(np.nanmedian(c2st_finite[bot_k_idx])),
            'effect_diff':          float(np.nanmedian(c2st_finite[top_k_idx])
                                          - np.nanmedian(c2st_finite[bot_k_idx])),
        }
        log.info(f'  group medians K={K}: top_C2ST={group_medians[str(K)]["top_k_median_c2st"]:.4f} '
                 f'bot_C2ST={group_medians[str(K)]["bottom_k_median_c2st"]:.4f} '
                 f'diff={group_medians[str(K)]["effect_diff"]:+.4f}')
    out['h3_group_medians_c2st'] = group_medians

    # Cluster-aware bootstrap CI (collinearity-honest): resample step-3 redundancy groups as units
    # rather than the full feature set as if independent. Effective n is the cluster count, so this
    # CI is the one to quote for the H1 significance statement. Computed for ALL 8 H1 cells
    # (previously only 3 of them had it, while the results.md caption claimed "cluster bootstrap
    # where available" for the whole table).
    clusters = feature_clusters(list(df.index))
    out['n_feature_clusters'] = int(len(np.unique(clusters)))
    _h1_cell_inputs = {
        'importance_vs_c2st':                       (imp_hi, c2st),
        'importance_vs_concept_stability':          (imp_hi, stab),
        'importance_2018_vs_c2st':                  ('imp_rank_2018', c2st),
        'importance_2018_vs_concept_stability':     ('imp_rank_2018', stab),
        'importance_perm_2017_vs_c2st':             ('imp_perm_rank_2017', c2st),
        'importance_perm_vs_concept_stability':     ('imp_perm_rank_2017', stab),
        'importance_perm_2018_vs_c2st':             ('imp_perm_rank_2018', c2st),
        'importance_perm_2018_vs_concept_stability': ('imp_perm_rank_2018', stab),
    }
    for cell, (xa, ya) in _h1_cell_inputs.items():
        blk = out.get(cell)
        if not isinstance(blk, dict) or 'bootstrap_ci95' not in blk:
            continue
        if isinstance(xa, np.ndarray):
            xv = xa
        else:
            # Reuse the -rank array already computed above (imp18_hi/imp_perm17_hi/imp_perm18_hi,
            # cached in _rank_arrays) instead of re-deriving it from the column name a second time;
            # the df-lookup fallback only fires if that cache is ever missing an entry (defensive —
            # should not happen, since a cell only reaches this branch when its guard above ran).
            xv = _rank_arrays.get(xa)
            if xv is None:
                xv = -df[xa].to_numpy(dtype=float)
        blk['cluster_bootstrap_ci95'] = cluster_bootstrap_ci(xv, ya, clusters)
    cc = out['importance_vs_concept_stability'].get('cluster_bootstrap_ci95', {})
    log.info(f'  cluster bootstrap (native vs concept_stability)   '
             f'CI95=[{cc.get("lo", float("nan")):+.3f},{cc.get("hi", float("nan")):+.3f}]  '
             f'clusters={cc.get("n_clusters", "?")} (vs {len(df)} feats)')

    # B3: quantify how noisy the permutation-importance ESTIMATE itself is, using the per-feature
    # std already saved from PERM_REPEATS resamples — turns the qualitative "noisy in the
    # tail" claim into a concrete count, flagging features whose own resampling noise is large
    # enough that the sign of their permutation importance is not reliably determined.
    if 'imp_perm_2017_bin' in df.columns and 'imp_perm_std_2017_bin' in df.columns:
        pmean = df['imp_perm_2017_bin'].to_numpy(dtype=float)
        pstd = df['imp_perm_std_2017_bin'].to_numpy(dtype=float)
        valid = np.isfinite(pmean) & np.isfinite(pstd)
        low_conf = valid & (np.abs(pmean) < pstd)
        rank17 = df['imp_rank_2017'].to_numpy(dtype=float)   # tail = bottom half by native rank
        tail_mask = valid & (rank17 > np.nanmedian(rank17))
        out['perm_importance_noise_2017'] = {
            'n_features_valid': int(valid.sum()),
            'n_low_confidence': int(low_conf.sum()),
            'frac_low_confidence': float(low_conf.sum() / valid.sum()) if valid.sum() else float('nan'),
            'n_low_confidence_in_tail_half': int((low_conf & tail_mask).sum()),
            'n_tail_half': int(tail_mask.sum()),
        }
        pin = out['perm_importance_noise_2017']
        log.info(f"  perm-importance noise (2017): {pin['n_low_confidence']}/{pin['n_features_valid']} "
                 f"features have |mean|<std ({pin['n_low_confidence_in_tail_half']}/{pin['n_tail_half']} "
                 f"in the bottom-importance half)")

    # H1 axis grouping: the 8 importance-variant x axis cells are all
    # already computed above under their original flat keys (kept untouched so every existing
    # reader of those keys keeps working); these two dicts just GROUP references to the same blocks
    # by axis, plus an explicit marker of which single metric each axis's verdict is based on.
    # MMD and Wasserstein-qn are NOT listed as axis1 verdict inputs: not every feature has both
    # (e.g. discrete features skip MMD), so only C2ST-AUC, which every feature has, decides the
    # Axis-1 verdict; MMD/Wasserstein remain observational context, reported but not aggregated in.
    out['axis1_primary_metric'] = 'c2st_auc'
    out['axis1_secondary_metrics'] = ['mmd', 'wasserstein_qn']
    out['axis1_secondary_metrics_caveat'] = (
        'MMD and Wasserstein-qn are observational sub-metrics of Axis 1 (supporting context only); '
        'not every feature has both (discrete features skip MMD), so only C2ST-AUC, present for '
        'every feature, decides the Axis-1 verdict.')
    out['axis1_tests'] = {
        'native_2017':      out['importance_vs_c2st'],
        'native_2018':      out.get('importance_2018_vs_c2st'),
        'permutation_2017': out.get('importance_perm_2017_vs_c2st'),
        'permutation_2018': out.get('importance_perm_2018_vs_c2st'),
    }
    # Axis 2's separation_stability is already MI-aware at the source (step 10 combines folded-AUC
    # and normalized mutual information into one combined strength before this script ever sees it
    # — see 10_execute_comparison.py's `_sep_strength`/`base_stab`), so "Separation Stability + MI-AUC"
    # is this single column, not two columns to be combined here.
    out['axis2_primary_metric'] = 'separation_stability'
    out['axis2_secondary_metrics'] = []
    out['axis2_tests'] = {
        'native_2017':      out['importance_vs_concept_stability'],
        'native_2018':      out.get('importance_2018_vs_concept_stability'),
        'permutation_2017': out.get('importance_perm_vs_concept_stability'),
        'permutation_2018': out.get('importance_perm_2018_vs_concept_stability'),
    }

    out['fdr_correction_summary'] = apply_fdr_to_headline_blocks(out)
    fdr = out['fdr_correction_summary']
    log.info(f'  FDR (Benjamini-Hochberg, alpha={fdr["alpha"]}): '
             f'{fdr["n_significant_bh_corrected"]}/{fdr["n_tests"]} tests survive correction '
             f'(uncorrected at p<{fdr["alpha"]}: {fdr["n_significant_uncorrected"]})')
    return out


# ════════════════════════════════════════════════════════════════════════════════
# Stage 4 — quadrant + per-attack
# ════════════════════════════════════════════════════════════════════════════════
def quadrant_assignment(df: pd.DataFrame) -> pd.Series:
    """Q1 good (hi-imp, stable) / Q2 fragile-shortcut (hi-imp, unstable) /
       Q3 noise (lo-imp, unstable) / Q4 underused-stable (lo-imp, stable).
    Importance anchored to 2017 only (H1 fix: top/low sets are well-defined, not fuzzy)."""
    imp_hi = -df[_preferred_importance_rank(df, 2017)]
    instab = df['instability']
    imp_med = float(np.median(imp_hi))
    ins_med = float(np.median(instab))
    out = {}
    for f in df.index:
        hi_imp = imp_hi[f] >= imp_med
        unstable = instab[f] >= ins_med
        if hi_imp and not unstable:
            out[f] = 'Q1_good'
        elif hi_imp and unstable:
            out[f] = 'Q2_fragile_shortcut'
        elif not hi_imp and unstable:
            out[f] = 'Q3_noise'
        else:
            out[f] = 'Q4_underused_stable'
    return pd.Series(out)


def _compute_single_dataset_attacks(log) -> tuple:
    """Return attack families present in exactly one dataset by reading cleaned parquets.

    Falls back to Config11.ATTACK_2017_ONLY if a parquet is missing or unreadable.
    """
    family_sets: dict[str, set] = {}
    for ds in DATASETS:
        path = CLEANED_DATA_ROOT / f'{ds}_cleaned.parquet'
        if not path.exists():
            log.warn(f'  cleaned parquet missing for {ds} — falling back to static ATTACK list')
            return Config11.ATTACK_2017_ONLY
        try:
            _, categories, _ = read_labels_encoded(ds)
            family_sets[ds] = {c for c in categories if c != BENIGN_LABEL}
        except Exception as exc:
            log.warn(f'  could not read labels for {ds} ({exc}) — falling back to static ATTACK list')
            return Config11.ATTACK_2017_ONLY
    if len(family_sets) < 2:
        return Config11.ATTACK_2017_ONLY
    all_fams: set = set.union(*family_sets.values())
    single_ds = tuple(sorted(
        fam for fam in all_fams
        if sum(1 for s in family_sets.values() if fam in s) == 1
    ))
    log.info(f'  single-dataset attack families (derived from parquets): {single_ds}')
    return single_ds


def per_attack_tables(log, attack_single_ds: tuple) -> dict:
    """Per attack-family: each feature's Axis-2 separation stability AND Axis-1 marginal shift.

    Axis 2 (separation_stability): does the attack-vs-benign gap survive 2017->2018?
      e.g. DoS 2017: packet_size attacked=[250,260] vs benign=[10,20], gap=240
           DoS 2018: packet_size attacked=[240,255] vs benign=[12,22], gap=228 -> stable
    Axis 1 (per-family slice C2ST, calibrated): does the raw distribution of THIS feature change
    for THIS attack family?
      e.g. DDoS 2017: flow_duration=[300ms..800ms]; DDoS 2018: flow_duration=[50ms..200ms] -> shifted
    Both together answer: "Which attack families will still be detectable next year?"
    """
    layer_a = load_layer_a()
    if not layer_a:
        log.warn('  Layer A missing -- per-attack breakdown skipped')
        return {}
    families: dict[str, list] = {}
    for feat, rec in layer_a.items():
        if not isinstance(rec, dict):
            continue
        pcs = rec.get('axis2_per_class_stability', {})       # Axis 2: gap stability per family
        per_atk_shift = rec.get('axis1_per_attack', {})      # Axis 1: per-family slice C2ST
        all_fams = set(pcs.keys()) | set(per_atk_shift.keys())
        for fam in all_fams:
            if fam in attack_single_ds:
                continue
            stab = float(pcs.get(fam, float('nan')))
            shift = _slice_shift(per_atk_shift.get(fam, {}))
            families.setdefault(fam, []).append({
                'feature': feat,
                'separation_stability': stab,          # ~1=gap preserved, 0=collapsed, <0=flipped
                'axis1_shift_calibrated': shift,        # calibrated per-family slice C2ST-AUC
            })
    return families


# ════════════════════════════════════════════════════════════════════════════════
# Stage 4b — new tests: rank stability (C), benign vs attack shift (E),
#             MI preservation (F), univariate transfer (A)
# ════════════════════════════════════════════════════════════════════════════════

def rank_stability_analysis(df: pd.DataFrame) -> dict:
    """Do feature importance rankings stay the same between 2017 and 2018?

    Example: 2017 top-5 = [packet_size, flow_duration, byte_rate, iat_mean, flag_count]
             2018 top-5 = [packet_size, byte_rate, flow_iat_std, iat_mean, protocol]
             Overlap = 3/5 = 60% -> rankings changed somewhat

    Spearman correlation between imp_rank_2017 and imp_rank_2018 across all features:
      ~1.0 = same features stay at the same position year-to-year (very stable)
      ~0.0 = rankings are random year-to-year (completely unstable)
      <0   = rankings flip (unusual, would mean bottom features become top)
    """
    r17 = df['imp_rank_2017'].to_numpy(dtype=float)
    r18 = df['imp_rank_2018'].to_numpy(dtype=float)
    mask = np.isfinite(r17) & np.isfinite(r18)
    sp = (float(spearmanr(r17[mask], r18[mask]).correlation)
          if mask.sum() >= 3 else float('nan'))
    interpretation = ('stable (rankings mostly preserved)' if sp > 0.7
                      else 'moderate (some rank movement)' if sp > 0.4
                      else 'unstable (rankings changed substantially)')

    overlaps = {}
    for K in Config11.RANK_STABILITY_K_VALUES:
        if K > len(df):
            continue
        top17 = set(df.nsmallest(K, 'imp_rank_2017').index)
        top18 = set(df.nsmallest(K, 'imp_rank_2018').index)
        overlap = len(top17 & top18)
        overlaps[str(K)] = {
            'top_2017': sorted(top17),
            'top_2018': sorted(top18),
            'overlap_count': overlap,
            'overlap_fraction': round(overlap / K, 3),
        }

    return {
        'spearman_rank_2017_vs_2018': float(sp),
        'interpretation': interpretation,
        'top_k_overlaps': overlaps,
    }


def benign_vs_attack_shift(log, df: pd.DataFrame = None) -> dict:
    """For each feature, compare benign-only distribution shift vs
    attack-only distribution shift between 2017 and 2018.

    If `df` (the main cross table) is given, also compares benign-only and attack-only
    C2ST-AUC against the POOLED (all-rows, benign+attack mixed) C2ST-AUC for the same
    features — i.e. is each slice individually more or less shifted than the whole-dataset
    figure everyone quotes as "the" Axis 1 number?

    Example for feature 'flow_duration':
      benign 2017: mean=150ms  ->  benign 2018: mean=160ms  (shift=0.05, small)
      DoS    2017: mean=500ms  ->  DoS    2018: mean=180ms  (shift=0.42, large!)
      => Attack distributions changed more -> attack tactics evolved

    Example for feature 'packet_size':
      benign 2017: mean=800B   ->  benign 2018: mean=1100B  (shift=0.31, environment changed!)
      DDoS   2017: mean=64B    ->  DDoS   2018: mean=70B    (shift=0.04, stable pattern)
      => Benign traffic shifted more -> environment/tooling change, not adversary change

    Answers: "Did attacks change their patterns, or did benign traffic change?
    If attack shift >> benign shift, adversary evolved. If benign shift dominates, the
    environment changed (network upgrade, different tool version, different user behaviour)."
    """
    layer_a = load_layer_a()
    if not layer_a:
        log.warn('  Layer A missing -- benign vs attack shift skipped')
        return {}

    rows = []
    for feat, rec in layer_a.items():
        if not isinstance(rec, dict):
            continue
        benign_d = rec.get('axis1_benign', {})
        attack_d = rec.get('axis1_attack', {})
        per_atk = rec.get('axis1_per_attack', {})

        benign_shift = _slice_shift(benign_d)
        attack_shift = _slice_shift(attack_d)   # direct: all attack rows pooled together
        atk_by_family = {cls: _slice_shift(ad) for cls, ad in per_atk.items()}
        valid_atk = [v for v in atk_by_family.values() if np.isfinite(v)]
        max_atk = float(max(valid_atk)) if valid_atk else float('nan')
        mean_atk = float(np.mean(valid_atk)) if valid_atk else float('nan')

        # prefer the direct all-attacks measure; fall back to max-per-family proxy
        compare_atk = attack_shift if np.isfinite(attack_shift) else max_atk
        if np.isfinite(benign_shift) and np.isfinite(compare_atk):
            driver = 'benign' if benign_shift > compare_atk else 'attack'
        else:
            driver = 'unknown'

        benign_c2st = float(benign_d.get('c2st_auc', float('nan')))
        attack_c2st = float(attack_d.get('c2st_auc', float('nan')))
        pooled_c2st = (float(df.loc[feat, 'c2st_auc'])
                       if df is not None and feat in df.index else float('nan'))
        # Calibrated counterparts (C2a shows BEFORE/AFTER calibration side by side).
        benign_c2st_calibrated = float(benign_d.get('c2st_calibrated', float('nan')))
        attack_c2st_calibrated = float(attack_d.get('c2st_calibrated', float('nan')))
        pooled_c2st_calibrated = (float(df.loc[feat, 'c2st_auc_calibrated'])
                                  if df is not None and feat in df.index
                                  and 'c2st_auc_calibrated' in df.columns else float('nan'))
        benign_c2st_null = float(benign_d.get('c2st_null', float('nan')))
        attack_c2st_null = float(attack_d.get('c2st_null', float('nan')))
        pooled_c2st_null = (float(df.loc[feat, 'c2st_auc_null'])
                            if df is not None and feat in df.index
                            and 'c2st_auc_null' in df.columns else float('nan'))

        rows.append({
            'feature': feat,
            'benign_shift': benign_shift,
            'attack_shift': attack_shift,
            'max_attack_shift': max_atk,
            'mean_attack_shift': mean_atk,
            'shift_by_family': atk_by_family,
            'dominant_driver': driver,
            'benign_c2st': benign_c2st,
            'attack_c2st': attack_c2st,
            'pooled_c2st': pooled_c2st,
            'benign_c2st_calibrated': benign_c2st_calibrated,
            'attack_c2st_calibrated': attack_c2st_calibrated,
            'pooled_c2st_calibrated': pooled_c2st_calibrated,
            'benign_c2st_null': benign_c2st_null,
            'attack_c2st_null': attack_c2st_null,
            'pooled_c2st_null': pooled_c2st_null,
        })

    benign_driven = sum(1 for r in rows if r['dominant_driver'] == 'benign')
    attack_driven = sum(1 for r in rows if r['dominant_driver'] == 'attack')
    all_b = [r['benign_shift'] for r in rows if np.isfinite(r['benign_shift'])]
    # prefer direct all-attacks measure; fill gaps with per-family mean as proxy
    all_a_direct = [r['attack_shift'] for r in rows if np.isfinite(r['attack_shift'])]
    all_a_proxy  = [r['mean_attack_shift'] for r in rows if np.isfinite(r['mean_attack_shift'])]
    all_a = all_a_direct if all_a_direct else all_a_proxy
    mean_b = float(np.mean(all_b)) if all_b else float('nan')
    mean_a = float(np.mean(all_a)) if all_a else float('nan')

    if np.isfinite(mean_b) and np.isfinite(mean_a):
        if mean_a > mean_b * 1.2:
            interpretation = ('Attack evolution dominates: attack distributions changed '
                              'more than benign traffic -> adversary changed tactics')
        elif mean_b > mean_a * 1.2:
            interpretation = ('Environment/benign evolution dominates: benign traffic '
                              'distributions shifted more than attack patterns -> network '
                              'environment changed, not the adversary')
        else:
            interpretation = 'Benign and attack shifts are comparable — both evolved similarly'
    else:
        interpretation = 'Insufficient data to determine dominant driver'

    # Pooled (all-rows, benign+attack mixed) RAW C2ST-AUC comparison. mean_b/mean_a above use
    # the CALIBRATED per-slice C2ST (each slice against its own null floor); this block keeps
    # the raw-AUC view alongside for reference. Only meaningful if `df` was supplied.
    all_bc = [r['benign_c2st'] for r in rows if np.isfinite(r['benign_c2st'])]
    all_ac = [r['attack_c2st'] for r in rows if np.isfinite(r['attack_c2st'])]
    all_pc = [r['pooled_c2st'] for r in rows if np.isfinite(r['pooled_c2st'])]
    mean_benign_c2st = float(np.mean(all_bc)) if all_bc else float('nan')
    mean_attack_c2st = float(np.mean(all_ac)) if all_ac else float('nan')
    mean_pooled_c2st = float(np.mean(all_pc)) if all_pc else float('nan')
    pooled_interpretation = 'n/a (no cross table supplied)'
    if np.isfinite(mean_pooled_c2st) and np.isfinite(mean_benign_c2st) and np.isfinite(mean_attack_c2st):
        b_vs_p = 'above' if mean_benign_c2st > mean_pooled_c2st else 'at/below'
        a_vs_p = 'above' if mean_attack_c2st > mean_pooled_c2st else 'at/below'
        pooled_interpretation = (
            f'Mean benign-only C2ST-AUC ({mean_benign_c2st:.4f}) is {b_vs_p} the pooled '
            f'baseline ({mean_pooled_c2st:.4f}); mean attack-only C2ST-AUC '
            f'({mean_attack_c2st:.4f}) is {a_vs_p} the pooled baseline.')

    return {
        'n_features': len(rows),
        'benign_driven_count': benign_driven,
        'attack_driven_count': attack_driven,
        'mean_benign_shift_across_features': mean_b,
        'mean_attack_shift_across_features': mean_a,
        'interpretation': interpretation,
        'mean_benign_c2st': mean_benign_c2st,
        'mean_attack_c2st': mean_attack_c2st,
        'mean_pooled_c2st': mean_pooled_c2st,
        'pooled_interpretation': pooled_interpretation,
        'per_feature': rows,
    }


def mi_preservation(df: pd.DataFrame) -> dict:
    """Did mutual information (feature-to-label signal) survive the shift?

    mutual_info_norm_2017 and mutual_info_norm_2018 are already computed per feature by
    step 7 (profiler) and stored in layer_b. They measure how much information a feature's
    distribution carries about the attack/benign label.

    Example for feature 'packet_size':
      MI 2017 = 0.65 (high: packet_size separates attacks from benign very well in 2017)
      MI 2018 = 0.58 (drop of 0.07 = 11%; signal mostly survived, still useful)

    Example for feature 'flow_duration':
      MI 2017 = 0.71 (high separation in 2017)
      MI 2018 = 0.22 (drop of 0.49 = 69%; signal largely destroyed, feature no longer useful!)

    'Signal survived' = MI 2018 >= 50% of MI 2017 (feature still informative)
    'Signal lost'     = MI 2018 <  50% of MI 2017 (shift destroyed the discriminative pattern)

    Answers: "Is the signal still there after the distribution shift, or did the shift
    scramble the decision boundary so the model can't distinguish attacks from benign anymore?"
    """
    if 'mutual_info_norm_2017' not in df.columns or 'mutual_info_norm_2018' not in df.columns:
        return {'error': 'MI columns not in cross table (re-run step 10 to add them)'}

    mi17 = df['mutual_info_norm_2017'].to_numpy(dtype=float)
    mi18 = df['mutual_info_norm_2018'].to_numpy(dtype=float)
    valid = np.isfinite(mi17) & np.isfinite(mi18)
    mi17v, mi18v = mi17[valid], mi18[valid]

    has_signal_17 = mi17v > 0.01   # features that actually had measurable signal in 2017
    survived = has_signal_17 & (mi18v >= 0.5 * mi17v)
    lost = has_signal_17 & (mi18v < 0.5 * mi17v)

    per_quadrant: dict = {}
    if 'quadrant' in df.columns:
        for q in ('Q1_good', 'Q2_fragile_shortcut', 'Q3_noise', 'Q4_underused_stable'):
            qm = (df['quadrant'] == q).to_numpy()
            q17 = mi17[qm]; q18 = mi18[qm]
            vm = np.isfinite(q17) & np.isfinite(q18)
            per_quadrant[q] = {
                'mean_mi_2017': float(np.nanmean(q17)),
                'mean_mi_2018': float(np.nanmean(q18)),
                'mean_mi_drop': float(np.mean(q17[vm] - q18[vm])) if vm.any() else float('nan'),
            }

    return {
        'n_features': int(valid.sum()),
        'mean_mi_2017': float(np.nanmean(mi17v)),
        'mean_mi_2018': float(np.nanmean(mi18v)),
        'mean_mi_drop': float(np.mean(mi17v - mi18v)),
        'signal_survived_count': int(survived.sum()),
        'signal_lost_count': int(lost.sum()),
        'per_quadrant': per_quadrant,
        'interpretation': (
            f'{int(survived.sum())} of {int(has_signal_17.sum())} informative features retain '
            f'>50% of their MI signal 2017->2018; {int(lost.sum())} features lost >50% '
            f'of their discriminative signal due to distribution shift.'
        ),
    }


def per_class_c2st_attribution(log) -> list:
    """ATTRIBUTION: for each feature, attribute its POOLED covariate shift (C2ST-AUC) to the label
    SLICE that drove it, using the per-slice C2ST-AUC step 10 now emits (axis1_benign / axis1_attack /
    axis1_per_attack). Answers "pooled says feature X is unstable — which slice is the culprit?"
    Lightweight by design: stability numbers only (no correlations / ablation / quadrants per class).
    Both RAW and CALIBRATED C2ST-AUC are included per slice/family (calibrated = each slice's own
    null floor, computed by slice_axis1() in 10_execute_comparison.py); dominant-slice attribution
    itself still uses RAW (whichever slice has the highest c2st_auc), consistent with pooled_c2st
    above it also being raw — the calibrated columns are for direct comparison against results.md's
    calibrated tables, not a second attribution ranking.
    Output: per_class_c2st_attribution.csv. Empty if step 10 predates the per-slice change."""
    layer_a = load_layer_a()
    rows = []
    for feat, a in layer_a.items():
        if not isinstance(a, dict):
            continue
        c2st_blk = a.get('c2st') or {}
        pooled_c2st = float(c2st_blk.get('auc', float('nan')))
        pooled_c2st_calibrated = float(c2st_blk.get('calibrated_pooled', float('nan')))
        benign_blk = a.get('axis1_benign') or {}
        attack_blk = a.get('axis1_attack') or {}
        benign_c2st = float(benign_blk.get('c2st_auc', float('nan')))
        attack_c2st = float(attack_blk.get('c2st_auc', float('nan')))
        benign_c2st_calibrated = float(benign_blk.get('c2st_calibrated', float('nan')))
        attack_c2st_calibrated = float(attack_blk.get('c2st_calibrated', float('nan')))
        per_atk = a.get('axis1_per_attack') or {}
        fam_c2st = {fam: float((d or {}).get('c2st_auc', float('nan')))
                    for fam, d in per_atk.items() if isinstance(d, dict)}
        fam_c2st_calibrated = {fam: float((d or {}).get('c2st_calibrated', float('nan')))
                               for fam, d in per_atk.items() if isinstance(d, dict)}
        # dominant culprit slice = the slice with the HIGHEST C2ST (most shifted) among finite ones
        candidates = {'benign': benign_c2st, 'attack': attack_c2st, **fam_c2st}
        finite = {k: v for k, v in candidates.items() if np.isfinite(v)}
        dominant = max(finite, key=finite.get) if finite else 'none'
        rows.append({
            'feature': feat,
            'pooled_c2st': pooled_c2st,
            'pooled_c2st_calibrated': pooled_c2st_calibrated,
            'benign_c2st': benign_c2st,
            'attack_c2st': attack_c2st,
            'benign_c2st_calibrated': benign_c2st_calibrated,
            'attack_c2st_calibrated': attack_c2st_calibrated,
            'dominant_slice': dominant,
            'dominant_c2st': float(finite.get(dominant, float('nan'))),
            **{f'c2st_{fam}': v for fam, v in fam_c2st.items()},
            **{f'c2st_calibrated_{fam}': v for fam, v in fam_c2st_calibrated.items()},
        })
    return rows


def load_cross_year_baseline(algorithm: str, log) -> dict:
    """Load step-6's full-data cross-year transfer baseline (cross_year_baseline_<algo>.json) so the
    row-limited ablation can be read against the REAL full-data train-A->test-B numbers. Returns {}
    (with a warning) if step 6's cross-test has not been run yet — the ablation still runs without it."""
    p = PROJECT_ROOT / 'output' / '6_testing' / f'cross_year_baseline_{algorithm}.json'
    if not p.exists():
        log.warn(f'  step-6 cross-year baseline not found ({p.name}); '
                 f'run step 6 with both datasets for the ablation reference')
        return {}
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception as e:
        log.warn(f'  could not read {p.name}: {type(e).__name__}: {e}')
        return {}


def prior_shift_analysis(log) -> dict:
    """E5: Measure class-prior (P(Y)) shift between DS1 and DS2.

    Benign dominates in both years but at different rates. This P(Y) shift alone can crater
    deployed classifier precision and is the single largest component of dataset shift between
    these years. Exact fractions are computed from the actual train.parquet files here.
    The shift is NOT captured by either the covariate or covariate-shape axes.
    """
    results = {}
    for ds in (DS1, DS2):
        tp = train_path(ds)
        if not tp.exists():
            log.warn(f'  prior shift: {ds} train.parquet not found, skipping')
            results[ds] = {}
            continue
        pf = pq.ParquetFile(tp)
        total, n_benign = 0, 0
        for batch in pf.iter_batches(batch_size=500_000, columns=[LABEL_BINARY]):
            y = batch.column(LABEL_BINARY).to_numpy(zero_copy_only=False)
            total += len(y)
            n_benign += int((y == 0).sum())
        n_attack = total - n_benign
        results[ds] = {
            'total_rows': total,
            'n_benign': n_benign,
            'n_attack': n_attack,
            'p_benign': round(n_benign / total, 4) if total else float('nan'),
            'p_attack': round(n_attack / total, 4) if total else float('nan'),
        }
        log.info(f'  {ds}: P(benign)={results[ds]["p_benign"]:.3f}  '
                 f'P(attack)={results[ds]["p_attack"]:.3f}  n={total:,}')

    p17 = results.get(DS1, {}).get('p_benign', float('nan'))
    p18 = results.get(DS2, {}).get('p_benign', float('nan'))
    shift = abs(p17 - p18) if (np.isfinite(p17) and np.isfinite(p18)) else float('nan')
    if np.isfinite(shift):
        severity = 'LARGE' if shift > 0.10 else ('MODERATE' if shift > 0.05 else 'SMALL')
        interpretation = (
            f'P(benign) shifted from {p17:.1%} ({DS1}) to {p18:.1%} ({DS2}) '
            f'(|delta|={shift:.3f}, {severity}). '
            f'This prior shift acts independently of feature-distribution shift and '
            f'is a likely contributor to cross-year precision degradation.'
        )
    else:
        interpretation = 'Prior shift could not be computed (missing training parquet).'
    return {
        'per_dataset': results,
        'prior_shift_abs': float(shift) if np.isfinite(shift) else float('nan'),
        'interpretation': interpretation,
        'note': ('Prior shift is NOT captured by the covariate or covariate-shape axes. '
                 'It requires a prior-corrected evaluation (natural priors, uncapped test set).'),
    }


def univariate_transfer(df: pd.DataFrame, log) -> pd.DataFrame:
    """Train a tiny decision tree on ONE feature at a time (complements the rank-stability check).
    Evaluate on 2017 in-domain test AND 2018 cross-domain test.
    The accuracy DROP (2017 acc - 2018 acc) shows which individual features survive
    the distribution shift and which ones fail to transfer.

    Example:
      Feature 'packet_size':   acc_2017=0.82, acc_2018=0.79, drop=0.03 -> transfers well
      Feature 'flow_duration': acc_2017=0.74, acc_2018=0.51, drop=0.23 -> fails to transfer
      Feature 'byte_rate':     acc_2017=0.91, acc_2018=0.88, drop=0.03 -> transfers well

    Compare with importance rank: if high-importance features also have large drops,
    the model is relying on features that won't generalize (confirms H1 from a different angle).
    """
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import balanced_accuracy_score

    feat17 = load_feature_names(DS1)
    feat18 = load_feature_names(DS2)
    canonical = [f for f in feat17 if f in set(feat18) and f in df.index]
    col17 = {f: feat17.index(f) for f in feat17}
    col18 = {f: feat18.index(f) for f in feat18}

    log.info(f'  Loading matrices for univariate transfer ({Config11.UNIVARIATE_CAP} per-class cap)')
    d17 = _load_domain(DS1, feat17, Config11.UNIVARIATE_CAP, log)
    d18 = _load_domain(DS2, feat18, Config11.UNIVARIATE_CAP, log)

    results = []
    for feat in canonical:
        i17 = col17[feat]
        i18 = col18[feat]
        clf = DecisionTreeClassifier(
            max_depth=Config11.UNIVARIATE_DEPTH,
            class_weight='balanced',
            random_state=Config11.SEED,
        )
        clf.fit(d17['Xtr'][:, [i17]], d17['ytr'])
        acc17 = float(balanced_accuracy_score(d17['yte'], clf.predict(d17['Xte'][:, [i17]])))
        acc18 = float(balanced_accuracy_score(d18['yte'], clf.predict(d18['Xte'][:, [i18]])))
        drop = acc17 - acc18
        imp_rank = int(df.at[feat, 'imp_rank_2017']) if feat in df.index else -1
        quadrant = str(df.at[feat, 'quadrant']) if ('quadrant' in df.columns and feat in df.index) else 'unknown'
        results.append({
            'feature': feat,
            'imp_rank_2017': imp_rank,
            'quadrant': quadrant,
            'acc_2017': round(acc17, 4),
            'acc_2018': round(acc18, 4),
            'accuracy_drop': round(drop, 4),
            'transfers_well': bool(drop < 0.05 and acc17 > 0.55),
        })

    log.info(f'  Univariate transfer: {len(results)} features evaluated')
    return pd.DataFrame(results).sort_values('imp_rank_2017')


# ════════════════════════════════════════════════════════════════════════════════
# Stage 5 — drift exposure vs permutation null
# ════════════════════════════════════════════════════════════════════════════════
def drift_exposure(df: pd.DataFrame, n_perm: int = Config11.DRIFT_N_PERM,
                   seed: int = Config11.SEED) -> dict:
    # Normalize importances to sum 1 so the reported exposure is interpretable
    # ("importance-weighted mean shift") instead of an arbitrary-scale gain sum.
    # The permutation-null percentile is scale-invariant either way.
    imp_raw = df['imp_mean'].to_numpy(dtype=float)
    tot = np.nansum(imp_raw[np.isfinite(imp_raw)])
    imp = imp_raw / tot if np.isfinite(tot) and tot > 0 else imp_raw
    # Axis-1 signal = calibrated C2ST (the one Axis-1 decision variable everywhere in this file).
    shift = df['c2st_auc_calibrated'].to_numpy()
    concept = df['instability'].to_numpy()

    def exposure(weights, signal):
        m = np.isfinite(weights) & np.isfinite(signal)
        return float(np.sum(weights[m] * signal[m]))

    def null_percentile(signal):
        m = np.isfinite(imp) & np.isfinite(signal)
        w, s = imp[m], signal[m]
        real = float(np.sum(w * s))
        rng = np.random.default_rng(seed)
        null = np.array([np.sum(rng.permutation(w) * s) for _ in range(n_perm)])
        pct = float((null < real).mean())
        return real, pct

    drift_real, drift_pct = null_percentile(shift)
    concept_real, concept_pct = null_percentile(concept)
    return {
        'drift_exposure': exposure(imp, shift),
        'drift_exposure_null_percentile': drift_pct,
        'concept_exposure': exposure(imp, concept),
        'concept_exposure_null_percentile': concept_pct,
        'importance_weights': 'normalized to sum 1 (importance-weighted mean of the signal)',
        'interpretation': ('percentile >0.95 => the model preferentially weights '
                           'shifting/concept-unstable features more than chance'),
    }


# ════════════════════════════════════════════════════════════════════════════════
# Stage 6 — the cross-domain ablation (decisive experiment)
# ════════════════════════════════════════════════════════════════════════════════
def _load_domain(ds: str, feats: list, cap: int, log) -> dict:
    """Load capped train + in-domain test (test.parquet if present, else 80/20 split of train)."""
    from sklearn.model_selection import train_test_split
    Xtr, ytr, _, _ = load_capped_subsample(
        train_path(ds), feats, 8, cap, log)
    tp = test_path(ds)
    if tp.exists():
        test_cap = max(cap // Config11.TEST_CAP_DIVISOR, Config11.TEST_CAP_FLOOR)
        Xte, yte, _, _ = load_capped_subsample(tp, feats, 8, test_cap, log)
    else:
        Xtr, Xte, ytr, yte = train_test_split(
            Xtr, ytr, test_size=Config11.IN_DOMAIN_TEST_SIZE, random_state=Config11.SEED,
            stratify=ytr)
        log.info(f'  {ds}: no test.parquet -> in-domain split of train for in-domain test')
    return {'Xtr': Xtr, 'ytr': ytr, 'Xte': Xte, 'yte': yte}


def _policies(df: pd.DataFrame, canonical: list) -> dict:
    """Ordered feature-name lists per selection policy (best-ranked first).

    Naming convention: axis1_* and axis2_* denote which drift axis drives the ranking.
      top_importance  — ranked by native LightGBM importance (Gini/gain), most important first
      axis1_stable    — ranked by Axis 1 (C2ST-AUC), least value-drifted first
      axis2_stable    — ranked by Axis 2 (separation stability), most concept-stable first
    random and all_features are injected by run_ablation directly.

    top_importance is anchored to 2017 importance ONLY (imp_rank_2017), even when the ablation
    trains on 2018 — selection must not peek at the cross-test year (using imp_rank_mean, the average of BOTH years, made "selected without peeking at the cross-test
    year" not strictly true). axis1_stable/axis2_stable need no such anchor: c2st_auc and
    separation_stability are themselves already cross-year comparison metrics, not importance scores
    tied to one year.
    """
    sub  = df.loc[[f for f in canonical if f in df.index]].copy()
    imp  = sub['imp_rank_2017']
    stab = sub['separation_stability']
    # H2 decision variable: CALIBRATED C2ST-AUC (same Axis-1 quantity headline_stats() correlates
    # against — a feature ranks as "least drifted" relative to its OWN null floor, not raw AUC).
    c2st = (sub['c2st_auc_calibrated'] if 'c2st_auc_calibrated' in sub.columns
           else pd.Series(0.0, index=sub.index))
    return {
        'top_importance': list(imp.sort_values(ascending=True).index),
        'axis1_stable':   list(c2st.sort_values(ascending=True).index),   # low calibrated C2ST = least drifted
        'axis2_stable':   list(stab.sort_values(ascending=False).index),  # high sep-stab = most stable
    }


def _load_scaler(ds: str) -> dict:
    """feature -> (mean, scale) from step-4 scaler.json (the per-year Z-score parameters)."""
    p = dataset_dir(ds) / 'scaler.json'
    d = json.loads(p.read_text(encoding='utf-8'))
    return {f: (float(m), float(s)) for f, m, s in zip(d['features'], d['mean'], d['scale'])}


def _build_ablation_clf(seed: int, use_gpu: bool = False):
    """Construct one ablation retrain model: LightGBM RF-mode (boosting_type='rf', mirrors
    5_train.build_lgbm_rf, including its GPU params when use_gpu=True) — the pipeline's only
    engine, so the importance definition and the retrained model always agree. `seed` varies
    per replicate (every H2 arm is retrained ABLATION_SEEDS times so the
    verdict rests on per-seed means + a paired test, not a single-seed point estimate)."""
    import lightgbm as lgb
    params = dict(
        boosting_type='rf',
        n_estimators=Config11.ABLATION_TREES,
        num_leaves=Config11.ABLATION_LGBM_NUM_LEAVES,
        max_depth=(Config11.ABLATION_DEPTH if Config11.ABLATION_DEPTH else -1),
        min_child_samples=Config11.ABLATION_LGBM_MIN_CHILD,
        bagging_fraction=Config11.ABLATION_LGBM_BAGGING_FRACTION,
        bagging_freq=1,
        feature_fraction=Config11.ABLATION_LGBM_FEATURE_FRACTION,
        class_weight='balanced',
        n_jobs=Config11.N_JOBS,
        random_state=seed,
        verbose=-1,
    )
    if use_gpu:
        # Mirrors 5_train.build_lgbm_rf's GPU-safe params (same OpenCL empty-bin crash
        # avoidance for RF-mode bagging; see that function's docstring for details).
        params.update(
            device_type='gpu',
            gpu_use_dp=True,
            max_bin=63,
            min_data_in_bin=1,
            min_split_gain=1e-8,
        )
    return lgb.LGBMClassifier(**params)


def find_peak_k(abl: pd.DataFrame, metric: str = 'macro_f1_cross_domain') -> pd.DataFrame:
    """After a coarse K sweep, identify the peak K for each (policy, direction).
    Flags whether the peak is an interior point (implying the true optimum lies between
    two consecutive K values and a finer search around that window is worthwhile).
    Default metric migrated from f1_cross_domain (attack-only F1) to
    macro_f1_cross_domain, matching the H2 decision metric (_compute_h2_verdict already uses
    macro F1 — this secondary/exploratory "peak K" artifact previously hadn't been migrated).
    Seed replicates (FIX-7) are averaged per K before the peak is located."""
    rows = []
    for (pol, direction), grp in abl.groupby(['policy', 'direction']):
        grp = (grp.groupby('K', as_index=False)[metric].mean()   # mean over seed replicates
                  .sort_values('K').reset_index(drop=True))
        vals = grp[metric].to_numpy()
        Ks   = grp['K'].to_numpy()
        best_i = int(np.argmax(vals))
        is_interior = 0 < best_i < len(vals) - 1
        # Suggest fine-search window: one step below and above the peak K
        fine_lo = int(Ks[best_i - 1]) if best_i > 0 else int(Ks[0])
        fine_hi = int(Ks[best_i + 1]) if best_i < len(Ks) - 1 else int(Ks[-1])
        rows.append({
            'policy':       pol,
            'direction':    direction,
            'peak_K':       int(Ks[best_i]),
            f'peak_{metric}': float(vals[best_i]),
            'is_interior':  is_interior,
            'fine_search_window': f'{fine_lo}-{fine_hi}' if is_interior else 'at boundary',
        })
    return pd.DataFrame(rows)


_H2_SCENARIOS = ('in_domain', 'cross_domain', 'cross_covariate', 'indomain_other_scaler')
# metric -> column prefix. 'f1' (attack F1) and 'acc' (plain accuracy) match the pre-existing
# column names so old readers (gen_gap_f1 etc.) keep working; the rest are new H2 columns.
_H2_METRIC_PREFIX = {
    'f1_attack': 'f1', 'f1_benign': 'benign_f1', 'f1_macro': 'macro_f1',
    'accuracy': 'acc', 'sensitivity': 'sensitivity', 'fpr': 'fpr',
    'precision': 'precision', 'specificity': 'specificity',
    'balanced_acc': 'balanced_acc', 'mcc': 'mcc',
}


def _full_metrics(y_true, y_pred) -> dict:
    """All H2 decision + supporting metrics for one scenario's predictions. f1_macro (mean of
    f1_attack and f1_benign) is the PRIMARY H2 decision metric (user decision: a policy that wins on
    attack F1 alone but tanks benign F1 is not actually a better feature-selection policy). The rest
    (sensitivity/fpr/precision/specificity/balanced_acc/mcc) are supporting context, not verdict
    drivers."""
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, matthews_corrcoef
    f1_kw = {'zero_division': 0}
    f1_attack = float(f1_score(y_true, y_pred, pos_label=1, **f1_kw))
    f1_benign = float(f1_score(y_true, y_pred, pos_label=0, **f1_kw))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else 0.0     # attack recall
    fpr         = float(fp / (tn + fp)) if (tn + fp) else 0.0     # false-alarm rate
    precision   = float(tp / (tp + fp)) if (tp + fp) else 0.0     # attack precision
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    return {
        'f1_attack': f1_attack, 'f1_benign': f1_benign,
        'f1_macro': (f1_attack + f1_benign) / 2.0,
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'sensitivity': sensitivity, 'fpr': fpr, 'precision': precision, 'specificity': specificity,
        'balanced_acc': (sensitivity + specificity) / 2.0,
        'mcc': float(matthews_corrcoef(y_true, y_pred)),
    }


def _flatten_h2_scenarios(scen: dict) -> dict:
    """scen: {'in_domain': {metric: val, ...}, 'cross_domain': {...}, ...} -> flat {'f1_in_domain':
    val, 'macro_f1_cross_domain': val, ...} row, using the established column-naming convention."""
    return {f'{prefix}_{scenario}': scen[scenario][metric]
           for scenario in _H2_SCENARIOS
           for metric, prefix in _H2_METRIC_PREFIX.items()}


def _rescale(X: np.ndarray, sel_names: list, sc_from: dict, sc_to: dict) -> np.ndarray:
    """Per-column affine rescale of `X` (whose columns are `sel_names`, in that order) from one
    dataset's z-score parameters into another's: for each column, x_to = (x_from * s_from +
    m_from - m_to) / s_to_safe, where s_to_safe clamps an exactly-zero `s_to` to 1.0 (matching the
    original per-column loop's `s_x if s_x else 1.0` div-by-zero guard — NaN or nonzero values pass
    through unchanged, only exact 0.0/-0.0 is replaced).

    This factors out the two near-identical loops previously duplicated (with `sc_from`/`sc_to`
    swapped) in `run_ablation.fit_eval()`: the cross_covariate rescale used sc_from=sc_te,
    sc_to=sc_tr; the indomain_other_scaler rescale used sc_from=sc_tr, sc_to=sc_te. Vectorized:
    gathers each feature's (mean, std) pair once into arrays instead of a per-column Python loop,
    then performs the identical per-element arithmetic as one broadcasted array operation."""
    from_ms = np.array([sc_from.get(f, (0.0, 1.0)) for f in sel_names], dtype=np.float64)
    to_ms = np.array([sc_to.get(f, (0.0, 1.0)) for f in sel_names], dtype=np.float64)
    m_from, s_from = from_ms[:, 0], from_ms[:, 1]
    m_to, s_to = to_ms[:, 0], to_ms[:, 1]
    s_to_safe = np.where(s_to == 0, 1.0, s_to)   # same guard as the original "s_x if s_x else 1.0"
    return (X.astype(np.float64) * s_from + m_from - m_to) / s_to_safe


def run_ablation(df: pd.DataFrame, log) -> pd.DataFrame:
    """Run the cross-domain binary ablation across all feature-selection policies and K values.

    SEED REPLICATES: every (policy, K, direction) cell is retrained
    Config11.ABLATION_SEEDS times (seed_i = SEED + i) and each replicate is one output ROW
    (column 'seed'), so downstream readers can aggregate mean ± CI and run a PAIRED test of
    each policy against top_importance across matched (K, direction, seed) cells instead of
    comparing single-seed point estimates. The 'random' policy draws a FRESH random feature
    subset per seed (subset variability + training variability together)."""
    feat17 = load_feature_names(DS1)
    feat18 = load_feature_names(DS2)
    canonical = [f for f in feat17 if f in set(feat18) and f in df.index]
    col17 = {f: feat17.index(f) for f in feat17}
    col18 = {f: feat18.index(f) for f in feat18}
    log.info(f'  ablation feature space: {len(canonical)} features (name-aligned across domains)')
    gpu_state = {'use_gpu': True}   # GPU by default, mirrors step 5; one failure downgrades all later fits
    log.info(f'  ablation engine: LGBMClassifier(rf) — the pipeline\'s only engine '
             f'({Config11.ABLATION_SEEDS} seeds per (policy, K, direction) cell)')

    log.info(f'Loading ablation matrices (capped, cap={Config11.TRAIN_CAP:,})')
    d17 = _load_domain(DS1, feat17, Config11.TRAIN_CAP, log)
    d18 = _load_domain(DS2, feat18, Config11.TRAIN_CAP, log)
    sc  = {DS1: _load_scaler(DS1), DS2: _load_scaler(DS2)}

    policies = _policies(df, canonical)
    k_values = [k for k in Config11.ABLATION_K_VALUES if k < len(canonical)] + [len(canonical)]

    def fit_eval(sel_names, train, test_in, test_cross, names_tr_dom, names_te_dom,
                 sc_tr, sc_te, seed):
        """Train on `train` (with `seed`); score in-domain + cross-domain under two framings,
        plus a rescaling-only control, completing the 4 (data-year x scaler-year) combos per
        training_year:
          - IN-DOMAIN        : train data + train scaler (native)            = own-year, own-scaler
          - CONCEPT  (cross)  : cross-test in its own per-year scaler          = other-year, other-scaler
          - COVARIATE (cross) : cross-test re-expressed in train-year scaler   = other-year, train-scaler
          - SWAP (in-domain)  : train-domain test rows re-expressed in the OTHER year's scaler
                                = own-year, other-scaler. No real distribution shift (same rows as
                                IN-DOMAIN, just de-scaled then re-scaled with mismatched params) --
                                isolates how much of the COVARIATE drop is pure scaler-mismatch
                                sensitivity vs a genuine cross-year distribution difference."""
        idx_tr = [names_tr_dom[f] for f in sel_names]
        idx_in = [names_tr_dom[f] for f in sel_names]
        idx_cr = [names_te_dom[f] for f in sel_names]
        clf = _build_ablation_clf(seed, gpu_state['use_gpu'])
        try:
            clf.fit(train['Xtr'][:, idx_tr], train['ytr'])
        except Exception as e:
            if gpu_state['use_gpu']:
                log.warn(f'  LightGBM GPU fit failed ({type(e).__name__}: {e}); '
                         'continuing on CPU for the rest of the ablation')
                gpu_state['use_gpu'] = False
                clf = _build_ablation_clf(seed, use_gpu=False)
                clf.fit(train['Xtr'][:, idx_tr], train['ytr'])
            else:
                raise
        Xin = test_in['Xte'][:, idx_in]
        pin = clf.predict(Xin)
        Xcr = test_cross['Xte'][:, idx_cr]
        pcr = clf.predict(Xcr)
        Xcov = _rescale(Xcr, sel_names, sc_from=sc_te, sc_to=sc_tr)
        pcov = clf.predict(Xcov)
        Xswap = _rescale(Xin, sel_names, sc_from=sc_tr, sc_to=sc_te)
        pswap = clf.predict(Xswap)
        return {
            'in_domain':             _full_metrics(test_in['yte'], pin),
            'cross_domain':          _full_metrics(test_cross['yte'], pcr),
            'cross_covariate':       _full_metrics(test_cross['yte'], pcov),
            'indomain_other_scaler': _full_metrics(test_in['yte'], pswap),
        }

    def _row(pol_name, K, direction, seed, flat):
        return {
            'policy': pol_name, 'K': K, 'direction': direction, 'seed': seed,
            **flat,
            'gen_gap_f1': flat['f1_in_domain'] - flat['f1_cross_domain'],
            'gen_gap_f1_covariate': flat['f1_in_domain'] - flat['f1_cross_covariate'],
            # Macro-F1-based gap alongside the pre-existing attack-F1-only
            # one, matching the H2 decision metric (macro F1).
            'gen_gap_macro_f1': flat['macro_f1_in_domain'] - flat['macro_f1_cross_domain'],
            'gen_gap_macro_f1_covariate':
                flat['macro_f1_in_domain'] - flat['macro_f1_cross_covariate']}

    seeds = [Config11.SEED + i for i in range(Config11.ABLATION_SEEDS)]
    results = []
    for K in k_values:
        is_all = (K == len(canonical))
        # random draws a fresh subset per seed inside the seed loop below (None sentinel).
        pol_items = ({'all_features': canonical} if is_all
                     else {**policies, 'random': None})
        for pol_name, ordered in pol_items.items():
            for direction, tr, te in (('2017->2018', d17, d18), ('2018->2017', d18, d17)):
                names_tr = col17 if tr is d17 else col18
                names_te = col18 if te is d18 else col17
                ds_tr = DS1 if tr is d17 else DS2
                ds_te = DS2 if te is d18 else DS1
                macro_seeds = []
                for seed in seeds:
                    if pol_name == 'random':
                        sel_rng = np.random.default_rng(seed)
                        sel_names = list(sel_rng.choice(canonical, size=K, replace=False))
                    else:
                        sel_names = ordered[:K]
                    scen = fit_eval(sel_names, tr, tr, te, names_tr, names_te,
                                    sc[ds_tr], sc[ds_te], seed)
                    flat = _flatten_h2_scenarios(scen)
                    results.append(_row(pol_name, K, direction, seed, flat))
                    macro_seeds.append(flat['macro_f1_cross_domain'])
                log.info(f'    K={K:<3} {pol_name:<20} {direction}  '
                         f'macroF1 cross={np.mean(macro_seeds):.3f}±{np.std(macro_seeds):.3f} '
                         f'({len(macro_seeds)} seeds)')
            if is_all:
                break
    return pd.DataFrame(results)


# ════════════════════════════════════════════════════════════════════════════════
# Stage 7 — visuals
# ════════════════════════════════════════════════════════════════════════════════
_QUAD_COLORS = {'Q1_good': '#2ecc71', 'Q2_fragile_shortcut': '#e74c3c',
                'Q3_noise': '#95a5a6', 'Q4_underused_stable': '#3498db'}


def plot_quadrant(df: pd.DataFrame, yname: str, ylabel: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 8))
    # X-axis must be the SAME 2017-anchored rank column quadrant_assignment() split on
    # (permutation rank when available, else native) — otherwise the quadrant colors are
    # assigned by one ranking while the points and the vertical median line are drawn on
    # another, and points can land on the wrong side of the line for their color.
    _imp_col = _preferred_importance_rank(df, 2017)
    imp_hi = -df[_imp_col]
    imp_label = ('importance (higher = more important; -perm_rank_2017)' if _imp_col == 'imp_perm_rank_2017'
                 else 'importance (higher = more important; -rank_2017)')
    y = df[yname]
    for q, color in _QUAD_COLORS.items():
        m = df['quadrant'] == q
        if m.any():
            ax.scatter(imp_hi[m], y[m], s=55, alpha=0.8, color=color,
                       edgecolors='white', linewidths=0.5, label=f'{q} ({int(m.sum())})')
    ax.axvline(float(np.median(imp_hi)), color='#888', ls='--', lw=0.8)
    ax.axhline(float(np.median(y)), color='#888', ls='--', lw=0.8)
    ax.set_xlabel(imp_label, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title('Importance vs cross-dataset behaviour (per feature)', fontsize=11)
    ax.legend(fontsize=8, loc='best')
    fig.tight_layout(); fig.savefig(out_path, dpi=120, bbox_inches='tight'); plt.close(fig)


def plot_c2st_bar(df: pd.DataFrame, out_path: Path):
    s = df['c2st_auc'].sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, len(s) * 0.18)))
    ax.barh(range(len(s)), s.to_numpy(), color='#9b59b6')
    ax.axvline(0.5, color='#333', ls='--', lw=1, label='0.5 = no shift')
    ax.set_yticks(range(len(s))); ax.set_yticklabels(s.index, fontsize=6)
    ax.set_xlabel('C2ST-AUC (0.5 = years indistinguishable, 1.0 = fully separable)')
    ax.set_title('Universal cross-year shift per feature (C2ST)', fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=120, bbox_inches='tight'); plt.close(fig)


def plot_rankrank(df: pd.DataFrame, out_path: Path):
    # H1 fix: 2017-only anchor, consistent with headline_stats()/_policies().
    x = df['imp_rank_2017'].rank().to_numpy()
    y = (1 - df['separation_stability']).rank().to_numpy()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x, y, s=45, alpha=0.75, color='#2980b9', edgecolors='white', linewidths=0.4)
    if len(x) > 2:
        b = np.polyfit(x, y, 1)
        xs = np.array([x.min(), x.max()])
        ax.plot(xs, b[0] * xs + b[1], color='#e74c3c', lw=1.5,
                label=f'slope={b[0]:+.2f}')
        ax.legend(fontsize=9)
    ax.set_xlabel('importance rank (1 = most important)')
    ax.set_ylabel('instability rank (1 = most stable)')
    ax.set_title('Importance rank vs instability rank', fontsize=11)
    fig.tight_layout(); fig.savefig(out_path, dpi=120, bbox_inches='tight'); plt.close(fig)


def plot_method_agreement(df: pd.DataFrame, out_path: Path):
    cols = ['c2st_auc', 'wasserstein_qn', 'mmd', 'ks_statistic']
    present = [c for c in cols if df[c].notna().any()]
    if not present:
        return
    ranks = np.column_stack([rankdata(df[c].fillna(df[c].median()).to_numpy()) for c in present])
    order = np.argsort(ranks[:, 0])
    fig, ax = plt.subplots(figsize=(6, max(4, len(df) * 0.18)))
    im = ax.imshow(ranks[order], aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(present))); ax.set_xticklabels(present, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(df))); ax.set_yticklabels(df.index[order], fontsize=5)
    ax.set_title('Shift-metric agreement (rank per feature)', fontsize=10)
    fig.colorbar(im, ax=ax, label='rank (low=less shift)')
    fig.tight_layout(); fig.savefig(out_path, dpi=120, bbox_inches='tight'); plt.close(fig)


def plot_ablation(abl: pd.DataFrame, out_dir: Path):
    if abl.empty:
        return
    # Plotted metric migrated from attack-only F1 to macro F1, matching the H2
    # decision metric (_compute_h2_verdict already uses macro F1 for the actual verdict; this
    # plot previously still showed attack-F1-only "cross-domain F1"/"gap").
    agg = abl.groupby(['policy', 'K']).agg(
        f1_cross=('macro_f1_cross_domain', 'mean'), gap=('gen_gap_macro_f1', 'mean')).reset_index()
    for metric, ylab, fname in (('f1_cross', 'cross-domain macro F1 (mean of both directions)',
                                 'ablation_crossdomain_f1.png'),
                                ('gap', 'generalization gap (in - cross macro F1)', 'ablation_gap.png')):
        fig, ax = plt.subplots(figsize=(9, 6))
        for pol in sorted(agg['policy'].unique()):
            s = agg[agg['policy'] == pol].sort_values('K')
            ax.plot(s['K'], s[metric], marker='o', label=pol)
        ax.set_xlabel('K (number of features)'); ax.set_ylabel(ylab)
        ax.set_title('Cross-domain ablation', fontsize=11); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out_dir / fname, dpi=120, bbox_inches='tight'); plt.close(fig)


def plot_importance_vs_shift_scatter(
        df: pd.DataFrame, metric_col: str, metric_label: str,
        imp_col: str, imp_label: str, sp_value: float, out_path: Path):
    """Scatter: importance rank vs one E sub-metric, with regression line and Spearman annotation."""
    if metric_col not in df.columns or imp_col not in df.columns:
        return
    valid = df[[imp_col, metric_col]].dropna()
    if len(valid) < 3:
        return
    x = valid[imp_col].to_numpy(float)
    y = valid[metric_col].to_numpy(float)
    v_colors = {'stable': '#2ecc71', 'shifted': '#e74c3c', 'flipped': '#9b59b6',
                'collapsed': '#e67e22', 'weak': '#95a5a6', 'restructured': '#16a085'}
    fig, ax = plt.subplots(figsize=(9, 7))
    if 'verdict' in df.columns:
        verdicts = df.loc[valid.index, 'verdict']
        for v, color in v_colors.items():
            m = (verdicts == v).to_numpy()
            if m.any():
                ax.scatter(x[m], y[m], s=50, alpha=0.75, color=color,
                           edgecolors='white', linewidths=0.4, label=v)
        ax.legend(fontsize=8, loc='upper left')
    else:
        ax.scatter(x, y, s=50, alpha=0.75, color='#3498db', edgecolors='white', linewidths=0.4)
    if len(x) > 2:
        b = np.polyfit(x, y, 1)
        xs = np.array([x.min(), x.max()])
        ax.plot(xs, b[0] * xs + b[1], color='#333', lw=1.5, ls='--')
    lbl = f'Spearman ρ={sp_value:+.3f}' if np.isfinite(sp_value) else ''
    ax.text(0.97, 0.03, lbl, transform=ax.transAxes, ha='right', va='bottom',
            fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
    ax.set_xlabel(imp_label, fontsize=10)
    ax.set_ylabel(metric_label, fontsize=10)
    ax.set_title(f'{imp_label}  vs  {metric_label}', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_verdict_distribution(df: pd.DataFrame, out_path: Path):
    """Bar chart of per-feature verdict counts (stable/shifted/flipped/collapsed/weak)."""
    if 'verdict' not in df.columns:
        return
    vc = df['verdict'].value_counts()
    c_map = {'stable': '#2ecc71', 'shifted': '#e74c3c', 'flipped': '#9b59b6',
             'collapsed': '#e67e22', 'weak': '#95a5a6', 'restructured': '#16a085'}
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(vc.index, vc.values, color=[c_map.get(v, '#888') for v in vc.index])
    for bar, val in zip(bars, vc.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2, str(val),
                ha='center', va='bottom', fontsize=9)
    ax.set_xlabel('Verdict', fontsize=10)
    ax.set_ylabel('Number of features', fontsize=10)
    ax.set_title('Distribution of per-feature stability verdicts', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_effect_size_box(df: pd.DataFrame, imp_col: str, shift_col: str,
                         out_path: Path, K_list: list | None = None):
    """Box plot: top-K vs bottom-K shift distribution — effect-size visual."""
    if imp_col not in df.columns or shift_col not in df.columns:
        return
    if K_list is None:
        K_list = [5, 10, 15]
    valid = df[[imp_col, shift_col]].dropna()
    if len(valid) < max(K_list) * 2:
        return
    sorted_df = valid.sort_values(imp_col)   # ascending rank = most important first
    groups, labels = [], []
    for K in K_list:
        groups.append(sorted_df.head(K)[shift_col].to_numpy())
        groups.append(sorted_df.tail(K)[shift_col].to_numpy())
        labels += [f'Top-{K} imp', f'Bot-{K} imp']
    fig, ax = plt.subplots(figsize=(max(6, len(groups) * 0.9), 5))
    bp = ax.boxplot(groups, labels=labels, patch_artist=True, notch=False)
    colors = ['#e74c3c', '#2ecc71'] * len(K_list)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xlabel('Feature group (top vs bottom by importance rank)', fontsize=10)
    ax.set_ylabel(shift_col, fontsize=10)
    ax.set_title(f'Effect size: top-K vs bottom-K  —  {shift_col}', fontsize=11)
    ax.tick_params(axis='x', labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_rank_change_scatter(df: pd.DataFrame, out_path: Path):
    """Scatter: L (importance rank change) vs C2ST-AUC, annotated with Spearman."""
    if 'imp_rank_delta_L' not in df.columns or 'c2st_auc' not in df.columns:
        return
    valid = df[['imp_rank_delta_L', 'c2st_auc']].dropna()
    if len(valid) < 3:
        return
    x = valid['imp_rank_delta_L'].to_numpy(float)
    y = valid['c2st_auc'].to_numpy(float)
    sp, _ = spearmanr(x, y)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, s=50, alpha=0.7, color='#3498db', edgecolors='white', linewidths=0.4)
    if len(x) > 2:
        b = np.polyfit(x, y, 1)
        xs = np.array([x.min(), x.max()])
        ax.plot(xs, b[0] * xs + b[1], color='#e74c3c', lw=1.5, ls='--')
    ax.text(0.97, 0.03, f'Spearman ρ={sp:+.3f}', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
    ax.axvline(0, color='#888', ls=':', lw=0.8)
    ax.set_xlabel('L = rank_2017 − rank_2018  (positive = feature rose in 2018)', fontsize=10)
    ax.set_ylabel('C2ST-AUC  (distributional shift 2017→2018)', fontsize=10)
    ax.set_title('Importance rank change (L) vs covariate shift', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_shape_class_bar(df: pd.DataFrame, out_path: Path):
    """Bar chart: count of features per Q-Q shape class."""
    if 'qq_shape_class' not in df.columns:
        return
    vc = df['qq_shape_class'].dropna().value_counts()
    if vc.empty:
        return
    c_map = {'identical': '#2ecc71', 'location_shift': '#3498db',
              'scale_change': '#e67e22', 'shape_change': '#e74c3c'}
    order = [c for c in ('identical', 'location_shift', 'scale_change', 'shape_change')
             if c in vc.index] + [c for c in vc.index if c not in c_map]
    vc = vc.reindex(order).dropna()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(vc.index, vc.values, color=[c_map.get(v, '#888') for v in vc.index])
    for bar, val in zip(bars, vc.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, str(val),
                ha='center', va='bottom', fontsize=9)
    ax.set_xlabel('Q-Q shape class', fontsize=10)
    ax.set_ylabel('Number of features', fontsize=10)
    ax.set_title("HOW did each feature's distribution move? (Q-Q shape class)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_family_instability_bar(family_df: pd.DataFrame, out_path: Path):
    """Horizontal bar: attack families ranked by median separation stability."""
    if family_df is None or not hasattr(family_df, 'empty') or family_df.empty:
        return
    df_plot = family_df.sort_values('median_separation_stability', ascending=True)
    vals = df_plot['median_separation_stability'].to_numpy(float)
    colors = ['#e74c3c' if v < 0.3 else ('#e67e22' if v < 0.6 else '#2ecc71') for v in vals]
    fig, ax = plt.subplots(figsize=(9, max(4, len(df_plot) * 0.38)))
    ax.barh(range(len(df_plot)), vals, color=colors)
    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels(df_plot['family'].to_numpy(), fontsize=8)
    ax.axvline(0.5, color='#333', ls='--', lw=1, label='0.5 threshold')
    ax.set_xlabel('Median separation stability  (1=preserved, 0=collapsed, <0=flipped)', fontsize=10)
    ax.set_title('Attack families ranked by instability impact', fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_prior_shift_bar(prior_shift_data: dict, out_path: Path):
    """Grouped bar chart: P(benign) and P(attack) for 2017 vs 2018."""
    if not prior_shift_data:
        return
    per_ds = prior_shift_data.get('per_dataset', {})
    if not per_ds:
        return
    datasets = list(per_ds.keys())
    p_benign = [per_ds[d].get('p_benign', float('nan')) for d in datasets]
    p_attack = [per_ds[d].get('p_attack', float('nan')) for d in datasets]
    x = np.arange(len(datasets))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 5))
    bars1 = ax.bar(x - width / 2, p_benign, width, label='P(benign)', color='#2ecc71', alpha=0.85)
    bars2 = ax.bar(x + width / 2, p_attack, width, label='P(attack)', color='#e74c3c', alpha=0.85)
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Proportion of traffic', fontsize=10)
    ax.set_title('Prior shift P(Y) — class balance 2017 vs 2018', fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_feature_family_heatmap(matrix_path: Path, out_path: Path):
    """Heatmap: feature × attack-family stability matrix."""
    if not matrix_path.exists():
        return
    try:
        mat = pd.read_csv(matrix_path, index_col=0)
    except Exception:
        return
    if mat.empty:
        return
    fig, ax = plt.subplots(figsize=(max(8, len(mat.columns) * 0.55),
                                    max(6, len(mat) * 0.22)))
    im = ax.imshow(mat.to_numpy(float), aspect='auto', cmap='RdYlGn', vmin=-1, vmax=1)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels(mat.index, fontsize=6)
    fig.colorbar(im, ax=ax, label='separation_stability (1=stable, 0=collapsed, <0=flipped)')
    ax.set_title('Feature × attack-family stability matrix', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════════
# Stage 7 — findings
# ════════════════════════════════════════════════════════════════════════════════


def bidirectional_k_analysis(df: pd.DataFrame,
                             k_values: list = None) -> dict:
    """Bidirectional K rank test for the importance-vs-shift relationship.

    Direction A (keep-existing logic, now explicit):
      Sort by importance rank -> top/mid/bottom K -> measure mean C2ST shift per group.
      Hypothesis: top-importance group has higher mean shift than bottom group.

    Direction B (new):
      Sort by shift (C2ST-AUC) -> top-K most unstable + top-K most stable features.
      For each group find their median position in the importance ranking.
      Hypothesis: most-unstable features sit higher (lower rank number) in the importance list
      than most-stable features.

    Both directions should agree: the same important-equals-unstable relationship holds
    regardless of which axis you sort by first. Bidirectional confirmation is stronger
    than one direction alone.
    """
    if k_values is None:
        k_values = list(Config11.BIDIR_K_VALUES)

    # H1 fix: anchor importance to 2017 only (not average of both years) so the "important set"
    # is well-defined and the measured shift is strictly 2017->2018.
    # (No both-year-mean fallback here: imp_perm_rank_2017 and imp_perm_rank_mean derive from the
    # same rank dicts, which always cover the full joined feature set via the native-importance
    # fallback, so imp_perm_rank_2017 is never all-NaN while imp_perm_rank_mean isn't — falling
    # straight through to imp_rank_2017 when permutation importance is unavailable at all.)
    imp_col = _preferred_importance_rank(df, 2017)

    imp_rank = df[imp_col].to_numpy(dtype=float)   # low number = high importance
    # Calibrated C2ST-AUC — same Axis-1 decision variable every other decision point in this file
    # uses (headline_stats, _policies, quadrant_assignment); see calibrate_c2st() in
    # 10_execute_comparison.py for why raw c2st_auc is not used directly.
    shift    = df['c2st_auc_calibrated'].to_numpy(dtype=float)
    n = len(df)

    dir_a = []
    dir_b = []
    for K in k_values:
        if K > n:
            continue

        # Direction A: sort by importance, measure shift of each group
        imp_order = np.argsort(imp_rank)           # ascending = most important first
        top_imp   = imp_order[:K]
        bot_imp   = imp_order[-K:]
        mid_start = (n - K) // 2
        mid_imp   = imp_order[mid_start:mid_start + K]
        mean_top  = float(np.nanmean(shift[top_imp]))
        mean_mid  = float(np.nanmean(shift[mid_imp]))
        mean_bot  = float(np.nanmean(shift[bot_imp]))
        dir_a.append({
            'K': K,
            'mean_shift_top_importance': mean_top,
            'mean_shift_mid_importance': mean_mid,
            'mean_shift_bottom_importance': mean_bot,
            'direction_confirms_H1': bool(mean_top > mean_bot),
        })

        # Direction B: sort by shift, measure importance rank of each group
        valid_shift = np.isfinite(shift)
        shift_order = np.argsort(-shift)            # descending = most unstable first
        valid_order = [i for i in shift_order if valid_shift[i]]
        most_unstable = np.array(valid_order[:K])
        most_stable   = np.array(valid_order[-K:])
        med_rank_unstable = float(np.median(imp_rank[most_unstable]))
        med_rank_stable   = float(np.median(imp_rank[most_stable]))
        # lower rank number = higher importance; unstable features should have lower rank
        dir_b.append({
            'K': K,
            'median_imp_rank_most_unstable': med_rank_unstable,
            'median_imp_rank_most_stable': med_rank_stable,
            'direction_confirms_H1': bool(med_rank_unstable < med_rank_stable),
        })

    return {
        'importance_rank_column': imp_col,
        'direction_a': dir_a,
        'direction_b': dir_b,
    }


def emit_ranked_listings(df: pd.DataFrame, out_dir: Path, log) -> None:
    """Standalone ranked CSVs, one per listing axis:

    - ranked by c2st_auc_calibrated descending (most unstable first — the Axis-1 decision value)
    - ranked by imp_rank_2017 ascending (most important first)
    - ranked by imp_rank_2018 ascending
    - ranked by benign_shift descending (if column present)
    - ranked by attack_shift descending
    - ranked by |imp_rank_delta_L| descending
    """
    def _save(frame, fname, sort_col, ascending=False):
        if sort_col not in frame.columns:
            log.warn(f'  emit_ranked_listings: column "{sort_col}" not in cross_table — skipping {fname}')
            return
        out = frame.sort_values(sort_col, ascending=ascending).reset_index().rename(columns={'index': 'feature'})
        out.to_csv(out_dir / fname, index=False)
        log.info(f'  {fname}: {len(out)} rows, sorted by {sort_col}')

    _save(df, 'ranked_by_instability.csv',          'c2st_auc_calibrated', ascending=False)
    _save(df, 'ranked_by_importance_2017.csv',      'imp_rank_2017',    ascending=True)
    _save(df, 'ranked_by_importance_2018.csv',      'imp_rank_2018',    ascending=True)

    if 'benign_shift' in df.columns:
        _save(df, 'ranked_by_benign_shift.csv',     'benign_shift',     ascending=False)
    if 'attack_shift' in df.columns:
        _save(df, 'ranked_by_attack_shift.csv',     'attack_shift',     ascending=False)

    if 'imp_rank_delta_L' in df.columns:
        df2 = df.copy()
        df2['abs_L'] = df2['imp_rank_delta_L'].abs()
        _save(df2, 'ranked_by_rank_change.csv',     'abs_L',            ascending=False)


def family_instability_ranking(log, df: pd.DataFrame = None) -> pd.DataFrame:
    """Rank attack families by median instability/shift across all features.

    Reads the per_attack/*.csv files written in sub-step 11.5 (per_attack_tables),
    aggregates per family: median separation_stability, median axis1_shift_calibrated
    (per-family slice C2ST), pct_features_stable, and returns a DataFrame sorted by stability ascending
    (most unstable family first). If `df` (the main cross table, benign+all-attacks
    pooled) is given, an 'ALL/POOLED' row is added so each family's stability can be
    read against the pooled baseline rather than only against other families.
    """
    atk_dir = OUTPUT_DIR / 'per_attack'
    if not atk_dir.exists():
        log.warn('  family_instability_ranking: per_attack/ dir not found — skipping')
        return pd.DataFrame()
    rows = []
    for csv in sorted(atk_dir.glob('*.csv')):
        family = csv.stem
        try:
            fdf = pd.read_csv(csv)
        except Exception as e:
            log.warn(f'  family_instability_ranking: cannot read {csv.name}: {e}')
            continue
        if 'separation_stability' not in fdf.columns:
            continue
        stab_vals = fdf['separation_stability'].dropna()
        shift_vals = (fdf['axis1_shift_calibrated'].dropna()
                      if 'axis1_shift_calibrated' in fdf.columns else pd.Series(dtype=float))
        n_stable = int((stab_vals > 0.5).sum())
        rows.append({
            'family': family,
            'n_features': int(len(stab_vals)),
            'median_separation_stability': float(stab_vals.median()) if len(stab_vals) else float('nan'),
            'median_axis1_shift_calibrated': float(shift_vals.median()) if len(shift_vals) else float('nan'),
            'pct_features_stable':         float(n_stable / len(stab_vals)) if len(stab_vals) else float('nan'),
        })
    if not rows:
        return pd.DataFrame()
    if df is not None and 'separation_stability' in df.columns:
        rows.append({
            'family': 'ALL/POOLED',
            'n_features': int(len(df)),
            'median_separation_stability': float(df['separation_stability'].median()),
            'median_axis1_shift_calibrated': float(df['c2st_auc_calibrated'].median())
                                            if 'c2st_auc_calibrated' in df.columns else float('nan'),
            'pct_features_stable':         float((df['separation_stability'] > 0.5).mean()),
        })
    result = pd.DataFrame(rows).sort_values('median_separation_stability', ascending=True)
    out_path = OUTPUT_DIR / 'family_instability_ranking.csv'
    result.to_csv(out_path, index=False)
    log.info(f'  family ranking: {len(result)} families -> {out_path.name}')
    return result


def pivot_feature_family_matrix(out_dir: Path, log) -> None:
    """Pivot the per_attack/*.csv files into a feature × family stability matrix.

    Rows = features, columns = attack families, values = separation_stability.
    Values near 1 = gap preserved; near 0 = collapsed; negative = flipped.
    """
    atk_dir = OUTPUT_DIR / 'per_attack'
    if not atk_dir.exists():
        log.warn('  pivot_feature_family_matrix: per_attack/ dir not found — skipping')
        return
    dfs = []
    for csv in sorted(atk_dir.glob('*.csv')):
        family = csv.stem
        try:
            fdf = pd.read_csv(csv)
        except Exception:
            continue
        if 'feature' not in fdf.columns or 'separation_stability' not in fdf.columns:
            continue
        fdf = fdf[['feature', 'separation_stability']].rename(columns={'separation_stability': family})
        dfs.append(fdf.set_index('feature'))
    if not dfs:
        log.warn('  pivot_feature_family_matrix: no valid per-attack files found')
        return
    matrix = pd.concat(dfs, axis=1)
    out_path = out_dir / 'feature_family_stability_matrix.csv'
    matrix.to_csv(out_path)
    log.info(f'  feature×family matrix: {matrix.shape} -> {out_path.name}')


def rank_change_vs_shift(df: pd.DataFrame, log) -> dict:
    """Spearman(L, E): does importance rank change correlate with covariate shift?

    L = imp_rank_delta_L = imp_rank_2017 - imp_rank_2018
    E = c2st_auc (total distributional shift)

    Also tests Wasserstein-qn and MMD. Adds quadrant counts for |L| × E.
    """
    if 'imp_rank_delta_L' not in df.columns:
        log.warn('  rank_change_vs_shift: L column missing — add it to main() before calling')
        return {}

    L = df['imp_rank_delta_L'].to_numpy(dtype=float)
    c2st = df['c2st_auc'].to_numpy(dtype=float) if 'c2st_auc' in df.columns else np.full(len(L), float('nan'))
    wass = df['wasserstein_qn'].to_numpy(dtype=float) if 'wasserstein_qn' in df.columns else np.full(len(L), float('nan'))
    mmd  = df['mmd'].to_numpy(dtype=float) if 'mmd' in df.columns else np.full(len(L), float('nan'))

    def _corr_block(name, a, b):
        res = _correlation_block(a, b, min_pairs=3)
        if res is None:
            return {'spearman': float('nan'), 'kendall': float('nan'),
                    'bootstrap_ci95': {'lo': float('nan'), 'med': float('nan'), 'hi': float('nan')},
                    'n_pairs': 0}
        log.info(f'  {name}: Spearman={res["spearman"]:+.3f}  Kendall={res["kendall"]:+.3f}  '
                 f'CI95=[{res["bootstrap_ci95"]["lo"]:+.3f},{res["bootstrap_ci95"]["hi"]:+.3f}]  '
                 f'n={res["n_pairs"]}')
        return res

    out = {
        'rank_delta_L_vs_c2st':        _corr_block('L vs C2ST',        L, c2st),
        'rank_delta_L_vs_wasserstein':  _corr_block('L vs Wasserstein', L, wass),
        'rank_delta_L_vs_mmd':          _corr_block('L vs MMD',         L, mmd),
    }

    # Partial correlation controlling for cardinality + variance
    Z_cols = []
    if 'cardinality' in df.columns:
        Z_cols.append(df['cardinality'].to_numpy(dtype=float))
    if 'variance_2017' in df.columns and 'variance_2018' in df.columns:
        Z_cols.append(np.nanmax(np.column_stack([df['variance_2017'], df['variance_2018']]), axis=1))
    if Z_cols:
        Z = np.column_stack(Z_cols)
        pcorr_lc = partial_spearman(L, c2st, Z)
        out['rank_delta_L_vs_c2st_partial_cardinality_variance'] = float(pcorr_lc)
        log.info(f'  partial_spearman(L, C2ST | cardinality, variance) = {pcorr_lc:+.3f}')

    # Quadrant counts: high/low |L| × high/low C2ST
    abs_L = np.abs(L)
    finite_mask = np.isfinite(abs_L) & np.isfinite(c2st)
    if finite_mask.sum() > 0:
        med_absL = float(np.median(abs_L[finite_mask]))
        med_c2st = float(np.median(c2st[finite_mask]))
        quads = Counter()
        for al, ce in zip(abs_L[finite_mask], c2st[finite_mask]):
            q_l = 'hi_L' if al >= med_absL else 'lo_L'
            q_e = 'hi_E' if ce >= med_c2st else 'lo_E'
            quads[f'{q_l}_{q_e}'] += 1
        out['quadrants_L_vs_E'] = dict(quads)

    # Interpretation
    sp_lc = out['rank_delta_L_vs_c2st'].get('spearman', float('nan'))
    if np.isfinite(sp_lc):
        direction = 'POSITIVE' if sp_lc > 0 else 'NEGATIVE'
        out['interpretation'] = (
            f'L vs C2ST Spearman = {sp_lc:+.3f} ({direction}): '
            + ('features whose importance rank rose 2017→2018 also shifted more in raw value.'
               if sp_lc > 0 else
               'features whose importance rank rose 2017→2018 shifted LESS in raw value.'))
    else:
        out['interpretation'] = 'Insufficient data for rank-change-vs-shift interpretation.'

    out_path = OUTPUT_DIR / 'rank_change_vs_shift.json'
    out_path.write_text(json.dumps(out, indent=2), encoding='utf-8')
    log.info(f'  rank_change_vs_shift -> {out_path.name}')
    return out


# ════════════════════════════════════════════════════════════════════════════════
# Sub-step 11.18 — H1.5: delta importance vs stability (4 tests)
# ════════════════════════════════════════════════════════════════════════════════
def delta_importance_vs_stability(df: pd.DataFrame, log) -> dict:
    """H1.5 (a-d): does the CHANGE in importance 2017->2018 correlate with each drift axis?

    Distinct from rank_change_vs_shift() (rank delta, imp_rank_2017 - imp_rank_2018, vs Axis 1 only): this uses the
    VALUE delta (imp_2018 - imp_2017) for BOTH native and permutation importance, against BOTH
    axes — 4 independent correlation tests:
      H1.5a: delta native importance      vs Axis 1 (C2ST-AUC)
      H1.5b: delta permutation importance vs Axis 1 (C2ST-AUC)
      H1.5c: delta native importance      vs Axis 2 (separation_stability)
      H1.5d: delta permutation importance vs Axis 2 (separation_stability)
    Answers: do features whose importance changed most also show the most stability change?
    Closer to H1 than to H2 — correlation only, no retraining/feature-selection, so there is no
    ablation companion for this sub-step (no H2 ablation applies here).
    """
    n = len(df)
    # Calibrated C2ST-AUC (Axis 1 decision variable — see headline_stats()).
    c2st = (df['c2st_auc_calibrated'].to_numpy(dtype=float)
            if 'c2st_auc_calibrated' in df.columns else np.full(n, float('nan')))
    stab = (df['separation_stability'].to_numpy(dtype=float)
            if 'separation_stability' in df.columns else np.full(n, float('nan')))
    has_native = {'imp_2017_bin', 'imp_2018_bin'} <= set(df.columns)
    delta_native = ((df['imp_2018_bin'] - df['imp_2017_bin']).to_numpy(dtype=float)
                    if has_native else np.full(n, float('nan')))
    has_perm = {'imp_perm_2017_bin', 'imp_perm_2018_bin'} <= set(df.columns)
    delta_perm = ((df['imp_perm_2018_bin'] - df['imp_perm_2017_bin']).to_numpy(dtype=float)
                  if has_perm else np.full(n, float('nan')))

    def _block(name, a, b):
        # NOTE: this used to also persist a 'verdict' field (POSITIVE/NEGATIVE/NEUTRAL), but that
        # field was never read back — 11_result_gen.py's _plain_delta_verdict() independently
        # re-derives the same classification (plus an extra strength tier) straight from
        # spearman/bootstrap_ci95 at render time, which is the one and only place a verdict for
        # this block is now computed. Kept in the log line below for console visibility only.
        res = _correlation_block(a, b, min_pairs=3)
        if res is None:
            af, _ = _finite_pairs(a, b)
            log.info(f'  H1.5 {name:<32} insufficient finite pairs (n={len(af)})')
            return {'spearman': float('nan'), 'kendall': float('nan'),
                    'bootstrap_ci95': {'lo': float('nan'), 'med': float('nan'), 'hi': float('nan')},
                    'n_pairs': int(len(af))}
        sp, kt, ci = res['spearman'], res['kendall'], res['bootstrap_ci95']
        excludes_zero = np.isfinite(ci['lo']) and np.isfinite(ci['hi']) and (ci['lo'] > 0 or ci['hi'] < 0)
        if not excludes_zero:
            verdict = 'NEUTRAL: CI includes 0'
        else:
            verdict = ('POSITIVE: CI excludes 0, rho>0' if sp > 0 else 'NEGATIVE: CI excludes 0, rho<0')
        log.info(f'  H1.5 {name:<32} Spearman={sp:+.3f}  Kendall={kt:+.3f}  '
                 f'CI95=[{ci["lo"]:+.3f},{ci["hi"]:+.3f}]  n={res["n_pairs"]}  verdict={verdict}')
        return res

    out = {
        'permutation_delta_available': bool(has_perm),
        'h1_5a_delta_native_vs_axis1': _block('H1.5a delta_native vs C2ST',         delta_native, c2st),
        'h1_5b_delta_perm_vs_axis1':   _block('H1.5b delta_perm vs C2ST',           delta_perm,   c2st),
        'h1_5c_delta_native_vs_axis2': _block('H1.5c delta_native vs sep_stability', delta_native, stab),
        'h1_5d_delta_perm_vs_axis2':   _block('H1.5d delta_perm vs sep_stability',   delta_perm,   stab),
    }
    # BH-FDR within the H1.5 family (4 tests) — its own family, separate from the 8 H1 cells,
    # so each family's correction matches that hypothesis family.
    h15_keys = ('h1_5a_delta_native_vs_axis1', 'h1_5b_delta_perm_vs_axis1',
                'h1_5c_delta_native_vs_axis2', 'h1_5d_delta_perm_vs_axis2')
    pvals = [out[k].get('bootstrap_ci95', {}).get('p_boot', float('nan')) for k in h15_keys]
    reject, qvals = benjamini_hochberg(pvals, alpha=0.05)
    for k, rej, q in zip(h15_keys, reject, qvals):
        out[k]['fdr_significant'] = bool(rej)
        out[k]['q_value_bh'] = float(q) if np.isfinite(q) else float('nan')
    out['fdr_correction_summary'] = {
        'alpha': 0.05, 'family': 'the 4 H1.5 delta tests',
        'n_tests': int(np.isfinite(np.asarray(pvals, dtype=float)).sum()),
        'n_significant_bh_corrected': int(reject.sum()),
    }
    out_path = OUTPUT_DIR / 'delta_importance_vs_stability.json'
    out_path.write_text(json.dumps(out, indent=2), encoding='utf-8')
    log.info(f'  H1.5 delta_importance_vs_stability -> {out_path.name}')
    return out


# ════════════════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════════════════
def main():
    # Cross-dataset step: no CLI arguments — the engine is LightGBM only (unified_config.ALGORITHM)
    # and all tuning + run modes live in unified_config.Config11.
    ap = argparse.ArgumentParser(description='Step 11: importance x distribution-shift cross analysis.')
    ap.parse_args()

    # Resolve algorithm-suffixed path globals BEFORE any I/O.
    global OUTPUT_DIR, RESULTS_DIR
    OUTPUT_DIR  = cross_output_dir(PROJECT_ROOT, ALGORITHM)
    RESULTS_DIR = cross_results_dir(PROJECT_ROOT, ALGORITHM)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / 'per_attack').mkdir(parents=True, exist_ok=True)
    log = Logger(RESULTS_DIR / Config11.STEPS_FILE, step_prefix=11,
                 title=f'SCRIPT 11 — CROSS ANALYSIS (importance × shift)  [{ALGORITHM}]')
    log.info(f'Engine    : {ALGORITHM} (the pipeline\'s only engine)')
    log.info(f'Output    : {OUTPUT_DIR}')

    # Explicit precheck of all upstream inputs BEFORE any work, with a clear listing of what
    # is missing (rather than failing partway). Step 11 must run AFTER step 5 (training) and
    # step 10 (execute comparison) have both fully completed.
    required = {
        f'{DS1} binary importance ({ALGORITHM}, step 5)':
            training_output_dir(PROJECT_ROOT, DS1, ALGORITHM) / 'feature_importance_binary.json',
        f'{DS2} binary importance ({ALGORITHM}, step 5)':
            training_output_dir(PROJECT_ROOT, DS2, ALGORITHM) / 'feature_importance_binary.json',
        'Branch B Layer B verdicts (step 10)':
            VERDICT_DIR / f'verdicts_layerB_{DS1}_{DS2}.json',
    }
    missing = {what: p for what, p in required.items() if not p.exists()}
    if missing:
        log.warn('STEP 11 BLOCKED — required upstream outputs are not ready:')
        for what, p in missing.items():
            log.warn(f'    missing: {what}  ->  {p}')
        log.warn('Run: python main.py --steps 5 10  (to completion), then re-run step 11.')
        log.close()
        sys.exit(1)

    # Derive which attack families appear in only one dataset from cleaned parquets.
    attack_single_ds = _compute_single_dataset_attacks(log)

    # Sub-step 11.1 — join
    log.step('Join importance with distribution verdicts')
    df, report = build_cross_table(log)
    df['quadrant'] = quadrant_assignment(df)
    # L column (rank_change_vs_shift's ranked listing + correlation test): importance rank change 2017→2018 (positive = rose in 2018)
    if 'imp_rank_2017' in df.columns and 'imp_rank_2018' in df.columns:
        df['imp_rank_delta_L'] = df['imp_rank_2017'] - df['imp_rank_2018']
        log.info('  Added imp_rank_delta_L (L = rank_2017 - rank_2018) to cross_table')
    (OUTPUT_DIR / 'join_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    df.to_csv(OUTPUT_DIR / 'cross_table.csv', index=False)
    df.to_json(OUTPUT_DIR / 'cross_table.json', orient='records', indent=2)
    log.step_end()

    stats, drift, abl, bidir_k = {}, {}, None, {}
    rank_stab, benign_atk, mi_pres, uni_xfer, prior_shift = None, None, None, None, None
    rank_change_shift, family_ranking, baseline = None, None, None
    delta_imp_stab = None
    if not Config11.ABLATION_ONLY:
        # Sub-step 11.2 — headline rank statistics
        log.step('Headline rank statistics')
        stats = headline_stats(df, log)
        (OUTPUT_DIR / 'rank_correlation.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
        log.step_end()

        # Sub-step 11.3 — drift exposure
        log.step('Drift exposure vs permutation null')
        drift = drift_exposure(df)
        (OUTPUT_DIR / 'drift_exposure.json').write_text(json.dumps(drift, indent=2), encoding='utf-8')
        log.step_end()

        # Sub-step 11.4 — bidirectional K rank test
        log.step('Bidirectional K rank test')
        bidir_k = bidirectional_k_analysis(df)
        (OUTPUT_DIR / 'bidirectional_k.json').write_text(
            json.dumps(bidir_k, indent=2), encoding='utf-8')
        log.info(f'  direction_a: {len(bidir_k["direction_a"])} K values; '
                 f'direction_b: {len(bidir_k["direction_b"])} K values')
        log.step_end()

        # Sub-step 11.5 — per-attack-family breakdown (Axis-2 stability + Axis-1 shift per family)
        log.step('Per-attack-family breakdown')
        fams = per_attack_tables(log, attack_single_ds)
        for fam, rows in fams.items():
            pd.DataFrame(rows).to_csv(OUTPUT_DIR / 'per_attack' / f'{fam}.csv', index=False)
        log.info(f'  wrote {len(fams)} per-attack tables (separation_stability + axis1_shift_calibrated)')
        log.step_end()

        # Sub-step 11.6 — visuals
        log.step('Visuals')
        plot_quadrant(df, 'instability', 'instability (1 - separation_stability; higher = worse)',
                      RESULTS_DIR / 'quadrant_axis2.png')
        plot_quadrant(df, 'c2st_auc_calibrated', 'Axis 1 (calibrated C2ST-AUC)',
                      RESULTS_DIR / 'quadrant_axis1.png')
        plot_c2st_bar(df, RESULTS_DIR / 'c2st_bar.png')
        plot_rankrank(df, RESULTS_DIR / 'rankrank_scatter.png')
        plot_method_agreement(df, RESULTS_DIR / 'method_agreement_heatmap.png')

        # ── new visuals ──────────────────────────────────────────────────────────────
        # Verdict distribution (overall E verdict breakdown)
        plot_verdict_distribution(df, RESULTS_DIR / 'verdict_distribution.png')

        # Importance vs shift scatters (diagnostic-only; not embedded in results.md — see the
        # numbered H1 cells C4a-C7b for the tested claims). C2ST (the Axis-1 decision metric)
        # reads its Spearman from the H1 cell; the MMD/Wasserstein scatters are
        # CORROBORATION-ONLY context (their correlation blocks were removed from the H1 family),
        # so their Spearman is computed inline for the annotation.
        _sp = lambda blk: blk.get('spearman', float('nan')) if isinstance(blk, dict) else float('nan')
        _imp_hi_plot = -df['imp_rank_2017'].to_numpy(dtype=float)
        plot_importance_vs_shift_scatter(
            df, 'c2st_auc', 'C2ST-AUC (total shift)', 'imp_rank_2017',
            'importance rank 2017 (1=most important)',
            _sp(stats.get('importance_vs_c2st', {})) if stats else float('nan'),
            RESULTS_DIR / 'importance_vs_c2st_scatter.png')
        plot_importance_vs_shift_scatter(
            df, 'mmd', 'MMD (corroboration only)', 'imp_rank_2017',
            'importance rank 2017 (1=most important)',
            _spearman(_imp_hi_plot, df['mmd'].to_numpy(dtype=float)),
            RESULTS_DIR / 'importance_vs_mmd_scatter.png')
        plot_importance_vs_shift_scatter(
            df, 'wasserstein_qn', 'Wasserstein-qn (corroboration only)', 'imp_rank_2017',
            'importance rank 2017 (1=most important)',
            _spearman(_imp_hi_plot, df['wasserstein_qn'].to_numpy(dtype=float)),
            RESULTS_DIR / 'importance_vs_wasserstein_scatter.png')

        # importance vs separation stability (diagnostic-only; see C4b-C7b for the tested claims)
        plot_importance_vs_shift_scatter(
            df, 'separation_stability', 'separation stability (concept axis; 1=preserved)',
            'imp_rank_2017', 'importance rank 2017 (1=most important)',
            _sp(stats.get('importance_vs_concept_stability', {})) if stats else float('nan'),
            RESULTS_DIR / 'importance_vs_sep_stability_scatter.png')

        # rank change (2017->2018) vs C2ST
        plot_rank_change_scatter(df, RESULTS_DIR / 'rank_change_vs_c2st_scatter.png')

        # effect-size box plots
        plot_effect_size_box(df, 'imp_rank_2017', 'c2st_auc',
                             RESULTS_DIR / 'effect_size_box.png')

        # Q-Q shape class summary bar
        plot_shape_class_bar(df, RESULTS_DIR / 'shape_class_bar.png')

        # family instability bar
        if family_ranking is not None and not family_ranking.empty:
            plot_family_instability_bar(family_ranking,
                                        RESULTS_DIR / 'family_instability_bar.png')

        # feature × family heatmap
        plot_feature_family_heatmap(
            OUTPUT_DIR / 'feature_family_stability_matrix.csv',
            RESULTS_DIR / 'feature_family_heatmap.png')

        # prior shift bar is generated after prior_shift_analysis() runs (later sub-step)

        log.ok('static visuals written (baseline + new per-comparison plots)')
        log.step_end()

        # Sub-step 11.9 — rank stability analysis
        log.step('Rank stability analysis')
        try:
            rank_stab = rank_stability_analysis(df)
            (OUTPUT_DIR / 'rank_stability.json').write_text(
                json.dumps(rank_stab, indent=2), encoding='utf-8')
            sp_rs = rank_stab.get('spearman_rank_2017_vs_2018', float('nan'))
            log.ok(f'rank stability Spearman={sp_rs:+.3f} '
                   f'({rank_stab.get("interpretation", "")}) -> rank_stability.json')
        except Exception as e:
            log.warn(f'rank stability failed ({type(e).__name__}: {e})')
        log.step_end()

        # Sub-step 11.10 — benign vs attack shift breakdown
        log.step('Benign vs attack shift breakdown')
        try:
            benign_atk = benign_vs_attack_shift(log, df)
            if benign_atk:
                # save per-feature rows as CSV; aggregate as JSON
                per_feat = benign_atk.pop('per_feature', [])
                (OUTPUT_DIR / 'benign_vs_attack_shift.json').write_text(
                    json.dumps(benign_atk, indent=2), encoding='utf-8')
                pd.DataFrame(per_feat).to_csv(
                    OUTPUT_DIR / 'benign_vs_attack_shift_per_feature.csv', index=False)
                benign_atk['per_feature'] = per_feat   # restore for write_findings
                log.ok(f'benign_driven={benign_atk.get("benign_driven_count")} '
                       f'attack_driven={benign_atk.get("attack_driven_count")} '
                       f'-> benign_vs_attack_shift.json')
        except Exception as e:
            log.warn(f'benign vs attack shift failed ({type(e).__name__}: {e})')
        log.step_end()

        # Sub-step 11.10b — per-class C2ST attribution (which slice drives each feature's shift?)
        log.step('Per-class C2ST attribution')
        try:
            attrib = per_class_c2st_attribution(log)
            if attrib:
                pd.DataFrame(attrib).to_csv(
                    OUTPUT_DIR / 'per_class_c2st_attribution.csv', index=False)
                dom = Counter(r['dominant_slice'] for r in attrib)
                log.ok(f'attributed {len(attrib)} features; dominant-slice counts={dict(dom)} '
                       f'-> per_class_c2st_attribution.csv')
            else:
                log.warn('  no per-slice C2ST found in layer A '
                         '(re-run step 10 after the per-slice change for attribution)')
        except Exception as e:
            log.warn(f'per-class C2ST attribution failed ({type(e).__name__}: {e})')
        log.step_end()

        # Sub-step 11.11 — MI signal preservation
        log.step('MI signal preservation')
        try:
            mi_pres = mi_preservation(df)
            (OUTPUT_DIR / 'mi_preservation.json').write_text(
                json.dumps(mi_pres, indent=2), encoding='utf-8')
            log.ok(f'survived={mi_pres.get("signal_survived_count")} '
                   f'lost={mi_pres.get("signal_lost_count")} -> mi_preservation.json')
        except Exception as e:
            log.warn(f'MI preservation failed ({type(e).__name__}: {e})')
        log.step_end()

        # Sub-step 11.13 — Prior-probability shift (feeds the E5 supplementary check)
        log.step('Prior-probability shift')
        try:
            prior_shift = prior_shift_analysis(log)
            (OUTPUT_DIR / 'prior_shift.json').write_text(
                json.dumps(prior_shift, indent=2), encoding='utf-8')
            log.ok(f'|ΔP(benign)|={prior_shift.get("prior_shift_abs", float("nan")):.3f} '
                   f'-> prior_shift.json')
            plot_prior_shift_bar(prior_shift, RESULTS_DIR / 'prior_shift_bar.png')
        except Exception as e:
            log.warn(f'Prior shift analysis failed ({type(e).__name__}: {e})')
        log.step_end()

        # Sub-step 11.14 — ranked listing CSVs
        log.step('Ranked listing CSVs')
        try:
            emit_ranked_listings(df, OUTPUT_DIR, log)
            log.ok('ranked listing CSVs written')
        except Exception as e:
            log.warn(f'emit_ranked_listings failed ({type(e).__name__}: {e})')
        log.step_end()

        # Sub-step 11.15 — family instability ranking
        log.step('Family instability ranking')
        try:
            family_ranking = family_instability_ranking(log, df)
            if family_ranking is not None and not family_ranking.empty:
                log.ok(f'{len(family_ranking)} families ranked -> family_instability_ranking.csv')
        except Exception as e:
            log.warn(f'family_instability_ranking failed ({type(e).__name__}: {e})')
        log.step_end()

        # Sub-step 11.16 — feature × family stability matrix
        log.step('Feature × family stability matrix')
        try:
            pivot_feature_family_matrix(OUTPUT_DIR, log)
        except Exception as e:
            log.warn(f'pivot_feature_family_matrix failed ({type(e).__name__}: {e})')
        log.step_end()

        # Sub-step 11.17 — rank change vs covariate shift
        log.step('Rank change vs shift')
        try:
            rank_change_shift = rank_change_vs_shift(df, log)
        except Exception as e:
            log.warn(f'rank_change_vs_shift failed ({type(e).__name__}: {e})')
        log.step_end()

        # Sub-step 11.18 — H1.5: delta importance vs stability (4 tests)
        log.step('Delta importance vs stability (H1.5)')
        try:
            delta_imp_stab = delta_importance_vs_stability(df, log)
        except Exception as e:
            log.warn(f'delta_importance_vs_stability failed ({type(e).__name__}: {e})')
        log.step_end()

    if Config11.RUN_ABLATION:
        # Sub-step 11.7 — cross-domain ablation (decisive experiment)
        log.step('Cross-domain ablation')
        # Reference: step-6's full-data cross-year transfer (the ablation is row-limited for speed,
        # so its all_features number is read against the real full-data train-A->test-B baseline).
        baseline = load_cross_year_baseline(ALGORITHM, log)
        if baseline:
            (OUTPUT_DIR / 'cross_year_baseline_ref.json').write_text(
                json.dumps(baseline, indent=2), encoding='utf-8')
            for direction, tasks_d in (baseline.get('directions') or {}).items():
                cov = ((tasks_d.get('binary') or {}).get('covariate') or {})
                con = ((tasks_d.get('binary') or {}).get('concept') or {})
                log.info(f'  step-6 baseline {direction}: binary attack_f1 '
                         f'concept={con.get("attack_f1")} covariate={cov.get("attack_f1")}')
        abl = run_ablation(df, log)
        abl.to_csv(OUTPUT_DIR / 'ablation_results.csv', index=False)
        peak = find_peak_k(abl)
        peak.to_csv(OUTPUT_DIR / 'ablation_peak_k.csv', index=False)
        plot_ablation(abl, RESULTS_DIR)
        log.ok(f'ablation: {len(abl)} runs ({Config11.ABLATION_SEEDS} seeds/cell) '
               f'-> ablation_results.csv  |  peak_k -> ablation_peak_k.csv')
        log.step_end()

    if not Config11.ABLATION_ONLY:
        # Sub-step 11.x — prior/threshold recalibration summary (recalibration fix, previously
        # untested). Reads step-6's per-direction binary recalibration (baseline-0.5 vs
        # prior-ratio vs SLD-EM vs oracle) and flattens it into one artifact for results.md.
        # Independent of RUN_ABLATION so it is produced on a stats-only re-run too.
        log.step('Prior/threshold recalibration summary')
        try:
            _bl = load_cross_year_baseline(ALGORITHM, log)
            recal = (_bl or {}).get('recalibration') or {}
            if recal:
                rows = []
                for direction, fr_map in recal.items():
                    for framing, r in (fr_map or {}).items():
                        for strat in ('baseline_0.5', 'prior_ratio_known', 'sld_em',
                                      'oracle_best_f1'):
                            s = (r or {}).get(strat) or {}
                            if not s:
                                continue
                            rows.append({
                                'direction': direction, 'framing': framing, 'strategy': strat,
                                'attack_f1': s.get('attack_f1'), 'macro_f1': s.get('macro_f1'),
                                'recall': s.get('recall'), 'precision': s.get('precision'),
                                'p_src_train_attack': (r or {}).get('p_src_train_attack'),
                                'p_tgt_true_attack': (r or {}).get('p_tgt_true_attack'),
                                'p_tgt_estimated': s.get('p_tgt_estimated'),
                                'threshold': s.get('threshold'),
                            })
                (OUTPUT_DIR / 'recalibration_summary.json').write_text(
                    json.dumps({'source': 'step-6 cross_year_baseline', 'rows': rows,
                                'per_direction': recal}, indent=2), encoding='utf-8')
                pd.DataFrame(rows).to_csv(OUTPUT_DIR / 'recalibration_summary.csv', index=False)
                log.ok(f'recalibration: {len(rows)} (direction x framing x strategy) rows '
                       f'-> recalibration_summary.json/.csv')
            else:
                log.warn('no recalibration block in cross_year_baseline — re-run step 6 with the '
                         'recalibration patch first; skipping recalibration summary')
        except Exception as e:
            log.warn(f'recalibration summary failed ({type(e).__name__}: {e})')
        log.step_end()

    if Config11.RUN_UNIVARIATE and not Config11.ABLATION_ONLY:
        # Sub-step 11.12 — univariate transfer test
        log.step('Univariate transfer test')
        try:
            uni_xfer = univariate_transfer(df, log)
            uni_xfer.to_csv(OUTPUT_DIR / 'univariate_transfer.csv', index=False)
            good = int(uni_xfer['transfers_well'].sum())
            log.ok(f'{good}/{len(uni_xfer)} features transfer well -> univariate_transfer.csv')
        except Exception as e:
            log.warn(f'univariate transfer failed ({type(e).__name__}: {e})')
        log.step_end()

    # Sub-step 11.8 — cache analysis outputs, then call 11_result_gen.py to write results.md.
    # Split into a separate script so the doc can be regenerated on its own
    # (python scripts/11_result_gen.py) without re-running this whole analysis.
    log.step('Cache analysis outputs for doc generation')
    doc_cache = {
        'df': df, 'stats': stats, 'drift': drift, 'abl': abl, 'report': report,
        'bidir_k': bidir_k, 'rank_stab': rank_stab, 'benign_atk': benign_atk,
        'mi_pres': mi_pres, 'uni_xfer': uni_xfer,
        'prior_shift': prior_shift if not Config11.ABLATION_ONLY else None,
        'rank_change_shift': rank_change_shift, 'family_ranking': family_ranking,
        'baseline': baseline, 'delta_imp_stab': delta_imp_stab,
    }
    cache_path = OUTPUT_DIR / '_doc_cache.pkl'
    with open(cache_path, 'wb') as f:
        pickle.dump(doc_cache, f)
    log.ok(f'analysis cache -> {cache_path.name}')
    log.step_end()

    log.step('Generate results.md (scripts/11_result_gen.py)')
    gen_script = Path(__file__).resolve().parent / '11_result_gen.py'
    proc = subprocess.run([sys.executable, str(gen_script)])
    if proc.returncode != 0:
        log.warn(f'  11_result_gen.py exited with code {proc.returncode}; '
                 f'results.md may be missing or stale. Run it directly to see the full error: '
                 f'python {gen_script.name}')
    else:
        log.ok('results.md generated')
    log.step_end()

    log.info(f'cross_table: {len(df)} features | outputs in {OUTPUT_DIR}')
    log.close()


if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as e:
        print(f'\n[STEP 11 BLOCKED] {e}')
        print('Required upstream outputs are not ready yet. Run the missing step, then re-run.')
        sys.exit(1)
