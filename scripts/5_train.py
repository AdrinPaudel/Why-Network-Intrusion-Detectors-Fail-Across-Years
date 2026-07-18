"""
5_train.py — Train LightGBM random-forest classifiers on preprocessed CIC-IDS splits.

PURPOSE:
  Fit LightGBM RF-mode models (boosting_type='rf') on training splits from 4_preprocessing.py
  for BOTH detection tasks:
  - Binary: Benign (0) vs Attack (1) — "is this flow malicious?"
  - Multiclass: Benign + 7 attack families — "which attack is it?"
  Training is separate from testing (6_test.py loads models and evaluates on held-out data).
  Uses fixed, individually justified hyperparameters (no hyperparameter search).

ENGINE:
  LightGBM in RANDOM-FOREST mode is the pipeline's ONLY engine. The earlier sklearn
  RandomForestClassifier path was removed (2026-07-02): it could not train on the full
  ~50M-row 2018 split without row-capping Benign, which distorted class_weight calibration,
  and its results were never part of the reported pipeline.

SUB-STEPS:
  Sub-step 5.1 (Prepare directories and load metadata):
    - Create output and results directories (algorithm-suffixed: <ds>_lightgbm)
    - Load feature names from preprocessing step
    Input: output/4_preprocessing/<dataset>/feature_names.json
    Output: Directories ready, metadata in memory

  Sub-step 5.2 (Load training data — full, uncapped):
    - Read train.parquet from preprocessing (LightGBM's histogram binning handles full data)
    Input: output/4_preprocessing/<dataset>/train.parquet
    Output: Training data (X matrix + binary labels + multiclass labels + class counts)

  Sub-step 5.3 (Load held-out test sample for permutation importance, overlap-filtered):
    - If test.parquet exists, load Config5.PERM_SAMPLE held-out rows (per-class capped)
    - OVERLAP FIX: rows whose feature values exactly duplicate a train row
      are removed from the permutation sample first — a measured 23.16% of cicids2017 test rows
      are exact feature-duplicates of train rows (see 6_test.check_train_test_overlap), and
      permutation importance on duplicated rows partly measures memorization.
    - If test.parquet is missing, permutation importance is SKIPPED entirely (never computed
      on training data).
    Input: output/4_preprocessing/<dataset>/{train,test}.parquet
    Output: Overlap-free held-out sample OR None

  Sub-step 5.4 (Train binary classifier):
    - Fit LightGBM RF (200 trees, fixed config) with class weighting
    - Save model + THREE importance rankings:
        native GAIN  (feature_importance_<task>.json      — the primary native importance;
                      total split-gain per feature, normalized to sum 1)
        native SPLIT (feature_importance_split_<task>.json — split counts, kept as a secondary
                      diagnostic; split-count is the more cardinality-biased of the two,
                      Strobl 2007)
        PERMUTATION  (feature_importance_perm_<task>.json  — held-out, overlap-free, unbiased)
    Input: Training data, feature names, held-out sample
    Output: rf_binary.joblib, importance JSONs, PNG charts

  Sub-step 5.5 (Train multiclass classifier):
    - Repeat 5.4 for the multiclass task (8 classes)
    Input: Training data, feature names, held-out sample
    Output: rf_multiclass.joblib, importance JSONs, PNG charts

  Sub-step 5.6 (Save training metadata):
    - Record all parameters: hyperparameters, class counts, fit time, overlap-filter counts
    Output: training_meta.json per dataset

  Sub-step 5.7 (Generate training report):
    - Summarize the run: datasets processed, models trained, importance files written
    Output: 5_training_report.txt (human-readable), printed to console

GUARANTEES:
  - Models are reproducible (random_state = PIPELINE_SEED across everything)
  - Full training data is used (no row capping)
  - Importance is recorded three ways: gain (primary native), split-count (secondary native),
    permutation (held-out, overlap-free, unbiased) — step 11 tests H1 under native AND permutation
  - No test data is used during training; permutation importance never sees train-duplicated rows
"""

import sys
import time
import json
import copy
import argparse
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime

import numpy as np
import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance   # unbiased importance (sklearn = metrics/inspection only, not a model engine)

try:
    import lightgbm as lgb
except ImportError as e:
    raise ImportError(
        'lightgbm is required: it is the pipeline\'s only training engine '
        '(the sklearn RandomForest path was removed). pip install lightgbm') from e

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unified_config import (
    Config5, Logger, PROJECT_ROOT, DATASETS, ALGORITHM,
    load_capped_subsample, load_feature_names, train_path, test_path,
    training_output_dir, training_results_dir, CANONICAL_MULTICLASS,
    feature_row_hashes, hash_feature_matrix, save_fig,
)

TASKS = ('binary', 'multiclass')


def build_lgbm_rf(use_gpu: bool):
    """
    LightGBM in RANDOM-FOREST mode (boosting_type='rf') — a histogram-based RandomForest that
    trains on the FULL data in a fraction of a naive RF's memory, and runs on GPU when
    available. RF mode needs bagging (bagging_freq>0, bagging_fraction<1); feature_fraction ~ the
    'sqrt' column subsampling; class_weight='balanced' provides inverse-frequency reweighting.
    When use_gpu is True the GPU device + GPU-safe params are added; a CPU fallback is handled
    at fit time.
    """
    params = dict(
        boosting_type='rf',
        n_estimators=Config5.N_ESTIMATORS,
        num_leaves=Config5.LGBM_NUM_LEAVES,     # allow deep-ish trees (RF de-correlates anyway)
        max_depth=(Config5.MAX_DEPTH if Config5.MAX_DEPTH else -1),
        min_child_samples=Config5.LGBM_MIN_CHILD,
        bagging_fraction=Config5.LGBM_BAGGING_FRACTION,  # bootstrap-like row subsample (RF mode <1)
        bagging_freq=Config5.LGBM_BAGGING_FREQ,          # bag every iteration (required for rf mode)
        feature_fraction=Config5.LGBM_FEATURE_FRACTION,  # column subsample per split (~sqrt de-corr)
        class_weight='balanced',
        n_jobs=Config5.N_JOBS,
        random_state=Config5.SEED,
        verbose=-1,
    )
    if use_gpu:
        # GPU path. The OpenCL histogram learner crashes with
        #   "Check failed: (best_split_info.left_count) > (0) ... serial_tree_learner.cpp"
        # when a single-precision histogram rounds a bin to zero count and the learner
        # still tries to split on it (a known LightGBM GPU bug, worse in 'rf' mode where
        # bagging reshuffles the data every iteration). The settings below make the GPU
        # path numerically match the CPU path so the split is never empty:
        #   gpu_use_dp=True   -> double-precision histograms (kills the rounding that
        #                        produces a zero left_count; the actual root-cause fix).
        #   max_bin=63        -> GPU-friendly bin count (LightGBM recommends <=63 on GPU);
        #                        fewer, denser bins also avoid empty-bin splits.
        #   min_data_in_bin   -> never let a bin hold <1 row (no degenerate split point).
        #   min_split_gain    -> require a positive gain to split, so a numerically
        #                        marginal/empty candidate split is rejected, not crashed on.
        # If no OpenCL GPU is actually present, fit() still raises and train_one()'s
        # existing CPU failsafe retries with device_type unset — that path is untouched.
        params.update(
            device_type='gpu',
            gpu_use_dp=True,
            max_bin=63,
            min_data_in_bin=1,
            min_split_gain=1e-8,
        )
    return lgb.LGBMClassifier(**params)


def native_importances(model, features: list[str]) -> dict[str, list]:
    """Both LightGBM native importance types, each normalized to sum 1 so cross-year
    VALUE deltas (H1.5) are on a common basis (raw gain totals scale with data size).

    Returns {'gain': [(feature, value), ...desc], 'split': [(feature, value), ...desc]}.
    GAIN (total split gain) is the primary native importance — closer in spirit to Gini/MDI
    and less cardinality-biased than SPLIT (split counts), which is kept as a secondary
    diagnostic (Strobl 2007 bias caveat applies to both, more strongly to split).
    """
    out: dict[str, list] = {}
    for kind in ('gain', 'split'):
        imp = np.asarray(model.booster_.feature_importance(importance_type=kind), dtype=np.float64)
        total = imp.sum()
        if total > 0:
            imp = imp / total
        order = np.argsort(imp)[::-1]
        out[kind] = [(features[i], float(imp[i])) for i in order]
    return out


def compute_permutation_importance(model, X_val, y_val, features, log) -> 'list | None':
    """
    Permutation importance on HELD-OUT, overlap-free data — shuffles each feature and measures
    the drop in balanced accuracy. Unbiased w.r.t. cardinality (unlike gain/split), so it is the
    importance H1 must be (re-)tested under in step 11. Returns [(feature, mean_drop, std), ...]
    sorted desc, or None.
    """
    if X_val is None or len(X_val) == 0:
        log.warn('  permutation importance skipped (no held-out sample)')
        return None
    try:
        model_cpu = copy.deepcopy(model)
        model_cpu.n_jobs = 1
        r = permutation_importance(
            model_cpu, X_val, y_val,
            n_repeats=Config5.PERM_REPEATS, scoring=Config5.PERM_SCORING,
            random_state=Config5.SEED, n_jobs=-1)
    except Exception as e:
        log.warn(f'  permutation importance failed: {type(e).__name__}: {e}')
        return None
    order = np.argsort(r.importances_mean)[::-1]
    return [(features[i], float(r.importances_mean[i]), float(r.importances_std[i]))
            for i in order]


def class_counts_for_task(task: str, kept_counts: dict[int, int]) -> dict[str, int]:
    """Human-readable per-class training counts for the report."""
    if task == 'binary':
        benign = kept_counts.get(0, 0)
        attack = sum(n for c, n in kept_counts.items() if c != 0)
        return {'Benign': benign, 'Attack': attack}
    inv = {v: k for k, v in CANONICAL_MULTICLASS.items()}
    return {inv[c]: n for c, n in sorted(kept_counts.items())}


def save_native_importance(ranked: list, task: str, importance_kind: str,
                           json_path: Path, png_path: Path, ds: str, log: Logger):
    """
    Persist one native importance ranking (gain or split, already normalized to sum 1) + a
    top-K bar chart. NOTE: native importance is BIASED toward high-cardinality features
    (Strobl 2007; split-count more than gain) — so H1 is ALSO re-tested in step 11 under the
    permutation importance saved by save_permutation_importance().
    """
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'task': task, 'importance_kind': importance_kind, 'normalized': 'sum_to_1',
                   'importances': [[n, v] for n, v in ranked]},
                  f, indent=2)
    log.ok(f'Saved {json_path.name} ({importance_kind})')

    top = ranked[:Config5.TOP_K_IMPORTANCES][::-1]   # reversed for horizontal bar (largest on top)
    names = [n for n, _ in top]
    vals  = [v for _, v in top]
    fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.32)))
    ax.barh(range(len(top)), vals, color='#2980b9')
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel(f'{importance_kind} importance (normalized)')
    ax.set_title(f'{ds} — {task}: top {len(top)} feature importances ({importance_kind})',
                 fontsize=10)
    save_fig(fig, png_path, log)


def save_permutation_importance(ranked, task: str, json_path: Path, png_path: Path,
                                ds: str, log: Logger):
    """
    Persist permutation importance (the unbiased one) + a top-K bar chart with error bars.
    `ranked` is [(feature, mean_drop, std), ...] from compute_permutation_importance, or None.
    The JSON keeps an 'importances' [[feat, mean], ...] block so step 11 can load it the same way
    it loads the native importance.
    """
    if ranked is None:
        return
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'task': task, 'importance_kind': 'permutation', 'scoring': Config5.PERM_SCORING,
            'importances': [[n, m] for n, m, _ in ranked],
            'importances_full': [{'feature': n, 'mean': m, 'std': s} for n, m, s in ranked],
        }, f, indent=2)
    log.ok(f'Saved {json_path.name} (permutation)')

    top   = ranked[:Config5.TOP_K_IMPORTANCES][::-1]
    names = [n for n, _, _ in top]
    vals  = [m for _, m, _ in top]
    errs  = [s for _, _, s in top]
    fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.32)))
    ax.barh(range(len(top)), vals, xerr=errs, color='#16a085', ecolor='#7f8c8d')
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel(f'permutation importance (drop in {Config5.PERM_SCORING})')
    ax.set_title(f'{ds} — {task}: top {len(top)} permutation importances', fontsize=10)
    save_fig(fig, png_path, log)


def load_overlap_free_holdout(ds: str, features: list[str], log: Logger):
    """Load the held-out permutation-importance sample and REMOVE rows whose feature values
    exactly duplicate a train row (23.16% of cicids2017 test rows were exact
    feature-duplicates of train rows, so permutation importance on the raw test sample partly
    scored memorized rows).

    Hashing uses the same float32 feature representation on both sides (feature_row_hashes /
    hash_feature_matrix stream through the identical iter_feature_batches path), so hash
    equality == exact feature-value duplication.

    Returns (X_val, yb_val, ym_val, overlap_meta dict) or (None, None, None, meta) if no test split.
    """
    te_path = test_path(ds)
    if not te_path.exists():
        log.warn(f'  no test.parquet for {ds} -> permutation importance will be skipped')
        return None, None, None, {'available': False}

    log.step(f'Load held-out sample for permutation importance (cap {Config5.PERM_SAMPLE:,}/class)')
    X_val, yb_val, ym_val, _ = load_capped_subsample(
        te_path, features, Config5.N_CLASSES_MULTI, Config5.PERM_SAMPLE, log)

    log.info('  hashing train rows to remove exact train-duplicates from the held-out sample ...')
    t0 = time.time()
    train_distinct = np.unique(feature_row_hashes(train_path(ds), features))
    val_hashes = hash_feature_matrix(X_val, features)
    dup_mask = np.isin(val_hashes, train_distinct)
    n_before, n_dup = int(len(X_val)), int(dup_mask.sum())
    keep = ~dup_mask
    X_val, yb_val, ym_val = X_val[keep], yb_val[keep], ym_val[keep]
    log.ok(f'overlap filter: removed {n_dup:,}/{n_before:,} rows '
           f'({100.0 * n_dup / max(n_before, 1):.2f}%) that exactly duplicate a train row '
           f'({time.time() - t0:.1f}s); {len(X_val):,} overlap-free rows remain')
    log.step_end()
    meta = {'available': True, 'n_sampled': n_before, 'n_train_duplicates_removed': n_dup,
            'n_overlap_free': int(len(X_val)),
            'frac_train_duplicates': n_dup / max(n_before, 1)}
    return X_val, yb_val, ym_val, meta


def train_one(task: str, X: np.ndarray, y: np.ndarray,
              X_val, y_val, features: list[str], ds: str,
              use_gpu: bool, out_dir: Path, results_dir: Path, log: Logger,
              kept_counts: dict) -> dict:
    """
    Fit one LightGBM RF-mode model (binary or multiclass), persist it, save native (gain +
    split) and permutation importance, and return its training-meta dict.
    """
    n_classes = int(np.unique(y).size)
    model_path = out_dir / f'rf_{task}.joblib'   # filename kept so 6_test.py finds it unchanged

    # reuse path — reload the existing model and recompute importance only, so a
    # permutation-repeat change updates the H1 perm cells WITHOUT retraining (no model drift into
    # Table 2 / H1-native / H2). Falls through to a normal fresh train if no saved model exists.
    reuse = getattr(Config5, 'REUSE_EXISTING_MODEL', False) and model_path.exists()
    if reuse:
        log.step(f'Reuse existing {task} model (REUSE_EXISTING_MODEL) — recompute importance only')
        model = joblib.load(model_path)
        # LGBMClassifier records the device it was fit on; reflect it in this run's metadata.
        use_gpu = bool(getattr(model, 'get_params', dict)().get('device', 'cpu') == 'gpu')
        elapsed = 0.0
        size_mb = model_path.stat().st_size / 1e6
        log.ok(f'loaded {model_path.name} ({size_mb:.1f} MB); skipping fit')
        log.step_end()
    else:
        model = build_lgbm_rf(use_gpu)

        log.step(f'Train {task} classifier [LGBMClassifier(rf)]')
        log.info(f'  X={X.shape[0]:,}x{X.shape[1]}  classes={n_classes}  '
                 f'n_estimators={Config5.N_ESTIMATORS}  max_depth={Config5.MAX_DEPTH}  '
                 f'device={"gpu" if use_gpu else "cpu"}')

        t0 = time.time()
        try:
            model.fit(X, y)
        except Exception as e:
            # Failsafe: GPU build/runtime may be unavailable or crash -> retry once on CPU.
            if use_gpu:
                log.warn(f'  LightGBM GPU fit failed ({type(e).__name__}: {e}); retrying on CPU')
                use_gpu = False
                model = build_lgbm_rf(use_gpu)
                model.fit(X, y)
            else:
                raise
        elapsed = time.time() - t0
        log.ok(f'fit in {elapsed:.1f}s')
        log.step_end()

        joblib.dump(model, model_path, compress=3)
        size_mb = model_path.stat().st_size / 1e6
        log.ok(f'Saved {model_path.name} ({size_mb:.1f} MB)')

    # Native importance, both types. GAIN is the primary (feature_importance_<task>.json — the
    # file step 11 reads as "native"); SPLIT is the secondary diagnostic.
    natives = native_importances(model, features)
    save_native_importance(
        natives['gain'], task, 'lgbm_gain',
        out_dir / f'feature_importance_{task}.json',
        results_dir / f'feature_importance_{task}.png',
        ds, log)
    save_native_importance(
        natives['split'], task, 'lgbm_split_count',
        out_dir / f'feature_importance_split_{task}.json',
        results_dir / f'feature_importance_split_{task}.png',
        ds, log)

    # Permutation importance on the overlap-free held-out sample (unbiased; H1 re-tested on it)
    perm_ranked = None
    if Config5.PERM_IMPORTANCE and X_val is not None and len(X_val):
        log.info(f'  permutation importance on {len(X_val):,} overlap-free held-out rows '
                 f'({Config5.PERM_REPEATS} repeats, scoring={Config5.PERM_SCORING})...')
        perm_ranked = compute_permutation_importance(model, X_val, y_val, features, log)
        save_permutation_importance(
            perm_ranked, task,
            out_dir / f'feature_importance_perm_{task}.json',
            results_dir / f'feature_importance_perm_{task}.png',
            ds, log)

    return {
        'task': task,
        'model_file': str(model_path),
        'model_size_mb': round(size_mb, 1),
        'n_train_rows': int(X.shape[0]),
        'n_features': int(X.shape[1]),
        'n_classes': n_classes,
        'fit_seconds': round(elapsed, 1),
        'has_permutation_importance': perm_ranked is not None,
        'class_counts_train': class_counts_for_task(task, kept_counts),
        'config': {
            'engine': 'lightgbm',
            'model': 'LGBMClassifier(rf)',
            'native_importance_primary': 'lgbm_gain',
            'native_importance_secondary': 'lgbm_split_count',
            'n_estimators': Config5.N_ESTIMATORS,
            'max_depth': Config5.MAX_DEPTH,
            'num_leaves': Config5.LGBM_NUM_LEAVES,
            'min_child_samples': Config5.LGBM_MIN_CHILD,
            'bagging_fraction': Config5.LGBM_BAGGING_FRACTION,
            'feature_fraction': Config5.LGBM_FEATURE_FRACTION,
            'class_weight': 'balanced',
            'gpu': use_gpu,
            'seed': Config5.SEED,
        },
    }


def process_dataset(ds: str, use_gpu: bool, log: Logger) -> dict:
    tr_path = train_path(ds)
    if not tr_path.exists():
        log.warn(f'Missing {tr_path} — run 4_preprocessing.py for {ds} first')
        sys.exit(1)

    # Algorithm-suffixed output folders: output/5_training/cicids2017_lightgbm/ etc.
    out_dir     = training_output_dir(PROJECT_ROOT, ds, ALGORITHM)
    results_dir = training_results_dir(PROJECT_ROOT, ds, ALGORITHM)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    log.section(f'DATASET: {ds}  [engine: lightgbm rf-mode]')
    features = load_feature_names(ds)

    log.step('Load training data (full, uncapped)')
    X, y_bin, y_multi, kept_counts = load_capped_subsample(
        tr_path, features, Config5.N_CLASSES_MULTI, 0, log)   # 0 = no cap
    log.ok(f'Loaded {X.shape[0]:,} rows')
    log.step_end()

    # Held-out, overlap-free sample for permutation importance (the model never trains on it).
    X_val = yb_val = ym_val = None
    overlap_meta = {'available': False}
    if Config5.PERM_IMPORTANCE:
        X_val, yb_val, ym_val, overlap_meta = load_overlap_free_holdout(ds, features, log)

    task_metas = []
    for task in TASKS:
        y     = y_bin if task == 'binary' else y_multi
        y_val = None if X_val is None else (yb_val if task == 'binary' else ym_val)
        task_metas.append(
            train_one(task, X, y, X_val, y_val, features, ds, use_gpu,
                      out_dir, results_dir, log, kept_counts))

    # free the big matrices before the next dataset
    del X, y_bin, y_multi
    if X_val is not None:
        del X_val, yb_val, ym_val

    meta = {
        'dataset': ds,
        'generated': datetime.now().isoformat(timespec='seconds'),
        'source_train_parquet': str(tr_path),
        'train_class_counts': class_counts_for_task('multiclass', kept_counts),
        'train_total_rows': int(sum(kept_counts.values())),
        'perm_holdout_overlap_filter': overlap_meta,
        'tasks': task_metas,
    }
    log.step('Save training metadata')
    meta_path = out_dir / 'training_meta.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    log.ok(f'Saved {meta_path.name}')
    log.step_end()
    return meta


def write_report(metas: list[dict], out_path: Path, log: Logger):
    lines: list[str] = []

    def h(t):
        lines.extend(['', '=' * 70, t, '=' * 70])

    h('TRAINING REPORT  --  LightGBM RF-mode on CIC-IDS 2017 / 2018')
    lines.append(f'Generated : {datetime.now():%Y-%m-%d %H:%M:%S}')

    h('MODEL CONFIGURATION (fixed, no hyperparameter search)')
    cfg = metas[0]['tasks'][0]['config']
    for k in ('engine', 'model', 'native_importance_primary', 'native_importance_secondary',
              'n_estimators', 'max_depth', 'num_leaves', 'min_child_samples',
              'bagging_fraction', 'feature_fraction', 'class_weight', 'gpu', 'seed'):
        lines.append(f'  {k:<28}: {cfg.get(k)}')
    lines.append('  imbalance                   : class_weight rebalances priors; full data, no capping.')
    lines.append('  importance                  : NATIVE gain (primary, feature_importance_<task>.json)')
    lines.append('                                + NATIVE split-count (secondary, feature_importance_split_<task>.json)')
    lines.append('                                + PERMUTATION (held-out, overlap-free,')
    lines.append('                                  feature_importance_perm_<task>.json)')
    lines.append('                                — step 11 tests H1 under native AND permutation.')

    for m in metas:
        h(f'DATASET: {m["dataset"]}')
        lines.append(f'  training rows       : {m["train_total_rows"]:,}')
        lines.append(f'  per-class counts    : {m["train_class_counts"]}')
        ov = m.get('perm_holdout_overlap_filter', {})
        if ov.get('available'):
            lines.append(f'  perm holdout        : {ov["n_overlap_free"]:,} overlap-free rows '
                         f'({ov["n_train_duplicates_removed"]:,} exact train-duplicates removed, '
                         f'{100.0 * ov["frac_train_duplicates"]:.2f}% of the sampled holdout)')
        lines.append('')
        for t in m['tasks']:
            lines.append(f'  [{t["task"]}]  rows={t["n_train_rows"]:,}  classes={t["n_classes"]}  '
                         f'fit={t["fit_seconds"]}s  '
                         f'perm_imp={"yes" if t.get("has_permutation_importance") else "no"}  '
                         f'model={t["model_size_mb"]}MB')

    h('NEXT STEP')
    lines.append('  Test (evaluate) on the held-out test split:')
    lines.append('    python main.py --steps 6 --datasets ' +
                 ' '.join(m['dataset'] for m in metas))

    text = '\n'.join(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text)
    log.ok(f'Saved {out_path.name}')
    print(text)


def main():
    parser = argparse.ArgumentParser(
        description='Train LightGBM RF-mode classifiers (binary + multiclass, both always) on the '
                    'preprocessed CIC-IDS splits. Hyperparameters are fixed constants in '
                    'unified_config.Config5 — there are no tuning flags.')
    parser.add_argument('--datasets', nargs='+', metavar='NAME', default=list(DATASETS),
                        help=f'Datasets to train on (default: {" ".join(DATASETS)})')
    device = parser.add_mutually_exclusive_group()
    device.add_argument('--cpu', dest='device', action='store_const', const='cpu',
                        help='Train on CPU.')
    device.add_argument('--gpu', dest='device', action='store_const', const='gpu',
                        help='Train on GPU, with automatic CPU fallback if it fails (default).')
    parser.set_defaults(device='gpu')
    args = parser.parse_args()
    use_gpu = (args.device == 'gpu')

    metas = []
    for ds in args.datasets:
        # Per-dataset results folder (same folder that holds the feature-importance PNGs).
        ds_results_dir = training_results_dir(PROJECT_ROOT, ds, ALGORITHM)
        ds_results_dir.mkdir(parents=True, exist_ok=True)

        ds_log = Logger(ds_results_dir / Config5.STEPS_FILE,
                        step_prefix=5,
                        title=f'5_TRAINING STEPS LOG  [{ds} / lightgbm]')
        ds_log.info('Engine    : lightgbm (RF mode) — the pipeline\'s only engine')
        ds_log.info(f'Device    : {"gpu (CPU fallback on failure)" if use_gpu else "cpu"}')
        ds_log.info('Tasks     : binary + multiclass (both)')
        ds_log.info(f'Perm imp  : {"on" if Config5.PERM_IMPORTANCE else "off"} '
                    f'(held-out cap {Config5.PERM_SAMPLE:,}/class, {Config5.PERM_REPEATS} repeats, '
                    'exact train-duplicates removed)')

        meta = process_dataset(ds, use_gpu, ds_log)
        metas.append(meta)

        ds_log.step('Write per-dataset training report')
        write_report([meta], ds_results_dir / Config5.RESULTS_FILE, ds_log)
        ds_log.step_end()

        ds_log.section('COMPLETE')
        for t in meta['tasks']:
            ds_log.info(f'[{t["task"]}]: fit={t["fit_seconds"]}s  -> {t["model_file"]}')
        ds_log.close()


if __name__ == '__main__':
    main()
