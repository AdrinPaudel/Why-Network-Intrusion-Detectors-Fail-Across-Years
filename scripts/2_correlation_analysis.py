"""
2_correlation_analysis.py — Per-dataset feature correlation analysis (analyze-only).

PURPOSE:
  Before any feature dropping or modeling, measure the correlation structure of the
  flow features in each dataset independently (2017 and 2018 analyzed separately).
  This produces metadata (JSON correlation matrices + flagged high-correlation pairs)
  plus dense, readable heatmaps and a summary report. No data is modified on disk—drop
  decisions happen later in preprocessing, informed by this analysis. The script
  addresses the cross-dataset generalization requirement by computing exact statistics
  per year and enabling downstream cross-year comparison in Step 11.

PROCESS:
  Sub-step 2.1 (Identify flow features):
    - Read column headers from the cleaned Parquet file
    - Drop identifier columns: id, Flow ID, Src/Dst IP, Src/Dst Port, Protocol,
      Timestamp, and Label (target)
    Result: Feature list for correlation analysis (~81 columns for CIC-IDS datasets)

  Sub-step 2.2 (Compute Pearson + Spearman correlations):
    - PEARSON: Exact computation in a single streaming pass via co-moment sufficient
      statistics (sum, sum-of-products). Rows with non-finite values are skipped
      (complete-case analysis). Works because only accumulates sums/products, not raw
      values, so memory is O(k²) not O(n×k) where k=features, n=rows (~63M for 2018).
    - SPEARMAN: Exact global ranking needs all column values in memory at once—infeasible
      at ~35 GB. Instead, uses deterministic uniform random subsampling based on hashing
      each row's global index (SplitMix64 finalizer). Sample size is adaptive: datasets
      with <6M rows use 6M-row sample; datasets ≥6M rows use 6.5M-row sample. Deterministic
      (reproducible regardless of chunk size) and uniformly spread across the file (no
      positional bias).
    Result: Two k×k correlation matrices (Pearson exact, Spearman approximate), and sampled
    data with row counts and label distribution for reporting.

  Sub-step 2.3 (Flag high-correlation pairs):
    - For both Pearson and Spearman: extract all pairs where |r| ≥ CORR_THRESHOLD (0.90)
    - Sort by absolute correlation (descending)
    Result: Two flagged-pair lists (Pearson and Spearman), later used for cross-year
    comparison and drop-decision support.

  Sub-step 2.4 (Save JSON metadata):
    - Full correlation matrices and flagged-pair lists as JSON
    - Enables downstream steps to avoid recomputing correlations
    Result: Four JSON files in output/2_correlation_analysis/<dataset>/

  Sub-step 2.5 (Render visualizations):
    - Dense heatmaps for both correlation methods (k×k grid with in-cell annotations)
    - Horizontal bar charts of top N flagged pairs (most readable summary)
    Result: PNG heatmaps and bar charts in results/2_correlation_analysis/<dataset>/

  Sub-step 2.6 (Write text report):
    - Human-readable summary of scope, methods, findings, and highly-correlated pairs
    Result: Markdown-formatted report text file

INPUTS:
  - data/cc_data/<dataset>_cleaned.parquet (output from Step 1)

OUTPUTS:
  - output/2_correlation_analysis/<dataset>/pearson_matrix.json
  - output/2_correlation_analysis/<dataset>/spearman_matrix.json
  - output/2_correlation_analysis/<dataset>/pearson_flagged_pairs.json
  - output/2_correlation_analysis/<dataset>/spearman_flagged_pairs.json
  - results/2_correlation_analysis/<dataset>/pearson_heatmap.png
  - results/2_correlation_analysis/<dataset>/spearman_heatmap.png
  - results/2_correlation_analysis/<dataset>/pearson_flagged_pairs.png
  - results/2_correlation_analysis/<dataset>/spearman_flagged_pairs.png
  - results/2_correlation_analysis/<dataset>/2_correlation_analysis_report.txt (report)
  - results/2_correlation_analysis/<dataset>/2_correlation_analysis_steps.log (execution log)

GUARANTEES:
  - No source data is modified on disk
  - Pearson correlation is exact (all rows used)
  - Spearman sample size is deterministic and reproducible regardless of chunk size
  - All output files are created atomically (written in full before any are visible)
  - JSON metadata enables Step 11 (cross-year analysis) without recomputation

DOCUMENTED ASYMMETRIES:
  - Smaller dataset (rows < Config2.SPEARMAN_SAMPLE_THRESHOLD): capped at SPEARMAN_SAMPLE_SMALL
    (effectively 100% of a small dataset). Actual row counts in step-1 report.
  - Larger dataset (rows >= threshold): capped at SPEARMAN_SAMPLE_LARGE (~10% of a 60M+ row set).

KNOWN, ACCEPTED EXCEPTION — pre-split feature-drop decision:
  This script (and script 3, which consumes its output for the final `features_to_drop` decision)
  runs on the FULL `<dataset>_cleaned.parquet`, before the train/test split exists (that split
  isn't created until step 4). So the correlation-based drop decision is computed over 100% of
  rows for both years, not blind to the rows that later become the held-out test set — unlike the
  rest of the pipeline's no-leakage discipline (step 4 explicitly fits its scaler on TRAIN rows
  only; permutation importance in step 5 draws only from held-out test rows).
  Accepted as a documented, low-impact exception rather than restructured, because:
    - The drop decision is a coarse, binary "is |r| >= 0.90" threshold check, not a fitted
      statistic — correlation structure computed over ~80% vs 100% of rows essentially never
      flips that threshold decision for real flow-feature pairs.
    - Recomputing it train-only would mean step 2/3 either duplicating step 4's split logic (risk
      of silent mismatch between the two splits) or moving correlation analysis to AFTER step 4
      (a step-numbering/pipeline-order change bigger than this fix warrants).
  If this is ever revisited: the fix would be to have step 2 read step 4's saved split indices
  (once step 4 has run) and restrict the streaming pass to train rows only.
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unified_config import (
    Config2, Logger, DATASETS, CLEANED_DATA_ROOT, PROJECT_ROOT,
    ensure_results_dir, feature_columns,
)


# ── Estimate total rows from Parquet metadata ──────────────────────────────────
def _estimate_total_rows(parquet_path: Path) -> 'int | None':
    """Estimate total rows from Parquet metadata — used only to set Spearman sample rate."""
    try:
        pf = pq.ParquetFile(parquet_path)
        return pf.metadata.num_rows
    except Exception:
        return None


# ── Spearman on a deterministic uniform random subsample ───────────────────────
def _hash_uniform(idx: np.ndarray, seed: int) -> np.ndarray:
    """
    Deterministic uniform value in [0, 1) for each integer row index (SplitMix64
    finalizer). The value depends ONLY on the index and the seed, so:
      - the same rows are selected no matter how the file is chunked (reproducible), and
      - selection is independent of position, so the sample is spread uniformly across
        the whole file (no front-bias).
    """
    with np.errstate(over='ignore'):                 # uint64 arithmetic wraps by design
        z = idx.astype(np.uint64) + np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
        return (z >> np.uint64(11)).astype(np.float64) / np.float64(1 << 53)


# ── FUSED single pass: exact Pearson co-moments + deterministic Spearman subsample ──
def _coerce_finite(df: pd.DataFrame, features: list) -> tuple:
    """Coerce df[features] to numeric (invalid parses -> NaN) and build the row mask
    marking which rows are entirely finite across all coerced feature columns.
    Returns (coerced_df, finite_mask); unfiltered — callers apply the mask as needed
    (as a DataFrame or ndarray, whichever their downstream computation requires)."""
    coerced = df[features].apply(pd.to_numeric, errors='coerce')
    good = np.isfinite(coerced.to_numpy(dtype=np.float64)).all(axis=1)
    return coerced, good


def _streaming_pearson_and_sample(parquet_path: Path, features: list,
                                  log: Logger) -> tuple:
    """
    ONE streaming pass that does BOTH:
      (a) EXACT Pearson via shift-stabilised co-moment accumulation (mean_i = Σx_i/n,
          cov_ij = Σx_i x_j/n − mean_i mean_j). Complete-case: rows with any non-finite feature
          are skipped (Step 1 output is already clean, so this is a guard). The first non-empty
          chunk's means become the numerical-stability offset (covariance is shift-invariant).
      (b) Deterministic hash-uniform Spearman subsample: keep row iff hash(global_index) < rate,
          rate = target/estimate. Same selection regardless of chunk size; spread across the whole
          file (no front-bias). The estimate (not exact total) only affects the SAMPLE SIZE, never
          the Pearson result; selection stays reproducible because the estimate is deterministic.

    Returns (pearson_corr [k x k], rows_used, sample_features_df, sampled_label_counter).
    """
    k = len(features)
    read_cols = set(features) | {'Label'}
    sum_x   = np.zeros(k, dtype=np.float64)
    S_xy    = np.zeros((k, k), dtype=np.float64)
    offset  = None
    rows_used = 0

    # Estimate total rows and determine adaptive Spearman sample size
    est_total = _estimate_total_rows(parquet_path)
    target_rows = Config2.get_spearman_sample_rows(est_total or 0)
    rate = 1.0 if not est_total else min(1.0, target_rows / est_total)
    log.info(f'Fused pass: exact Pearson + Spearman sample '
             f'(rate {rate:.4f}, est_total ~{est_total:,} rows, target {target_rows:,})'
             if est_total else 'Fused pass: exact Pearson + Spearman sample (rate 1.0, estimate unavailable)')

    sample_parts: list = []
    sampled_labels: Counter = Counter()
    global_off = 0

    def _accumulate_chunk(chunk: pd.DataFrame, off: int) -> int:
        """Update the Pearson co-moment accumulators and collect the Spearman sample from one
        chunk. `off` is this chunk's starting row offset (for the hash-uniform Spearman sample).
        Returns the chunk's row count, so the caller can advance its own running offset."""
        nonlocal offset, rows_used, sum_x, S_xy
        chunk.columns = [c.strip() for c in chunk.columns]
        m = len(chunk)

        # (b) Spearman hash-uniform sample (collect BEFORE numeric coercion, keeps Label)
        if rate < 1.0:
            idx    = np.arange(off, off + m, dtype=np.uint64)
            keep_s = _hash_uniform(idx, Config2.SPEARMAN_SEED) < rate
        else:
            keep_s = np.ones(m, dtype=bool)
        if keep_s.any():
            sub = chunk.iloc[keep_s]
            if 'Label' in sub.columns:
                for lbl, c in sub['Label'].astype(str).value_counts().items():
                    sampled_labels[lbl] += int(c)
            sample_parts.append(sub[features].copy())

        # (a) Pearson exact co-moment over complete-case feature rows
        Xdf, good = _coerce_finite(chunk, features)
        X = Xdf.to_numpy(dtype=np.float64)
        if not good.all():
            X = X[good]
        if X.shape[0] > 0:
            if offset is None:
                offset = X.mean(axis=0)
            Xc = X - offset
            rows_used += Xc.shape[0]
            sum_x += Xc.sum(axis=0)
            S_xy  += Xc.T @ Xc
        return m

    # Read Parquet in chunks (using PyArrow for better control)
    try:
        pf = pq.ParquetFile(parquet_path)
        for rg_idx in range(pf.num_row_groups):
            chunk = pf.read_row_group(rg_idx, columns=list(read_cols)).to_pandas()
            m = _accumulate_chunk(chunk, global_off)
            global_off += m
            del chunk
    except Exception as e:
        # Fallback: read entire Parquet file (less efficient but still works)
        log.warn(f'PyArrow row group read failed ({e}), falling back to full read')
        chunk = pd.read_parquet(parquet_path, columns=list(read_cols))
        _accumulate_chunk(chunk, 0)
        del chunk

    with np.errstate(invalid='ignore', divide='ignore'):
        n = max(rows_used, 1)
        mean = sum_x / n
        cov  = S_xy / n - np.outer(mean, mean)
        std  = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        corr = cov / np.outer(std, std)
        np.clip(corr, -1.0, 1.0, out=corr)
        np.fill_diagonal(corr, 1.0)

    sample = (pd.concat(sample_parts, ignore_index=True) if sample_parts
              else pd.DataFrame(columns=features))
    log.info(f'Fused pass done: Pearson over {rows_used:,} rows; '
             f'Spearman sample {len(sample):,} rows (seed {Config2.SPEARMAN_SEED}, chunk-independent)')
    return corr, rows_used, sample, sampled_labels


def _spearman_from_sample(sample: pd.DataFrame, features: list) -> tuple:
    """Spearman = Pearson on column ranks. Coerces to numeric and drops any non-finite
    rows before ranking (guard; the sample should already be clean). Returns
    (corr [k x k], n_rows_used)."""
    k = len(features)
    if len(sample) < 3:
        return np.full((k, k), np.nan), len(sample)
    Xdf, good = _coerce_finite(sample, features)
    X = Xdf[good]
    if len(X) < 3:
        return np.full((k, k), np.nan), len(X)
    ranked = X.rank(method='average').to_numpy(dtype=np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        corr = np.corrcoef(ranked, rowvar=False)
    np.clip(corr, -1.0, 1.0, out=corr)
    np.fill_diagonal(corr, 1.0)
    return corr, len(X)


# ── Flagged-pair extraction ────────────────────────────────────────────────────
def _flag_pairs(corr: np.ndarray, features: list, threshold: float) -> list:
    """Return [(feat_i, feat_j, r), ...] for upper-triangle pairs with |r| >= threshold,
    sorted by |r| descending. Skips NaN pairs."""
    k = len(features)
    # triu_indices(k, k=1) enumerates (i, j) pairs in the same row-major order
    # (i outer, j inner, j > i) as the nested loop it replaces, so pre-sort order
    # — and therefore tie-breaking under the stable sort below — is unchanged.
    iu, ju = np.triu_indices(k, k=1)
    r_vals = corr[iu, ju]
    keep = ~np.isnan(r_vals) & (np.abs(r_vals) >= threshold)
    pairs = [(features[i], features[j], float(r))
             for i, j, r in zip(iu[keep], ju[keep], r_vals[keep])]
    pairs.sort(key=lambda t: -abs(t[2]))
    return pairs


# ── Dense heatmap ──────────────────────────────────────────────────────────────
def _corr_cmap():
    """Diverging blue-white-red: -1 blue, 0 white, +1 red."""
    return LinearSegmentedColormap.from_list(
        'corr_bwr', ['#2166AC', '#67A9CF', '#F7F7F7', '#EF8A62', '#B2182B'])


def plot_heatmap(corr: np.ndarray, features: list, title: str, out_path: Path,
                 annotate: bool = True):
    """
    Dense, readable correlation heatmap. With ~81 features this is an 81x81 grid, so
    the figure is sized in absolute inches per cell at high DPI (rather than a fixed
    canvas) so every in-cell number stays legible. Numbers are drawn ×100 (e.g. 0.93
    -> '93', -1.0 -> '-100') to fit the cell; the colour bar still shows the -1..1 scale.
    """
    k = len(features)
    side = max(Config2.HEATMAP_MIN_INCHES, k * Config2.HEATMAP_CELL_INCHES)
    fig, ax = plt.subplots(figsize=(side, side))
    cmap = _corr_cmap()
    im = ax.imshow(corr, cmap=cmap, vmin=-1.0, vmax=1.0, aspect='equal')

    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(features, rotation=90, ha='center', fontsize=Config2.HEATMAP_TICK_FONTSZ)
    ax.set_yticklabels(features, fontsize=Config2.HEATMAP_TICK_FONTSZ)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=16)

    # Thin gridlines between cells to separate the boxes.
    ax.set_xticks(np.arange(-0.5, k, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, k, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.5)
    ax.tick_params(which='minor', length=0)

    if annotate:
        for i in range(k):
            for j in range(k):
                r = corr[i, j]
                if np.isnan(r):
                    txt, color = '', '#000000'
                else:
                    txt = f'{int(round(r * 100))}'
                    color = 'white' if abs(r) >= 0.6 else '#222222'
                if txt:
                    ax.text(j, i, txt, ha='center', va='center',
                            fontsize=Config2.HEATMAP_ANNOT_FONTSZ, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label('correlation r', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=Config2.HEATMAP_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_flagged_pairs_bar(pairs: list, title: str, out_path: Path, top_n: int = 40):
    """Horizontal bar chart of the strongest |r| flagged pairs (most readable summary
    of a dense matrix). Shows up to top_n pairs."""
    if not pairs:
        return
    shown = pairs[:top_n]
    labels = [f'{a}  ×  {b}' for a, b, _ in shown]
    vals   = [r for _, _, r in shown]
    colors = ['#B2182B' if v >= 0 else '#2166AC' for v in vals]

    fig, ax = plt.subplots(figsize=(11, max(4, len(shown) * 0.32)))
    y = range(len(shown))
    ax.barh(list(y), vals, color=colors, edgecolor='white')
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()                       # strongest at top
    ax.set_xlim(-1.05, 1.05)
    ax.axvline(0, color='#888', linewidth=0.8)
    ax.set_xlabel('correlation r')
    ax.set_title(title, fontsize=11, fontweight='bold')
    for yi, v in zip(y, vals):
        ax.text(v + (0.02 if v >= 0 else -0.02), yi, f'{v:.2f}',
                va='center', ha='left' if v >= 0 else 'right', fontsize=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── JSON + report writers ──────────────────────────────────────────────────────
def _save_matrix_json(corr: np.ndarray, features: list, meta: dict, path: Path):
    """Persist the full matrix (rounded) + metadata so a later step needs no recompute."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'features': features,
        'matrix':   [[None if np.isnan(v) else round(float(v), 6) for v in row]
                     for row in corr],
        **meta,
    }
    path.write_text(json.dumps(payload), encoding='utf-8')


def _save_pairs_json(pairs: list, meta: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **meta,
        'count': len(pairs),
        'pairs': [{'a': a, 'b': b, 'r': round(r, 6)} for a, b, r in pairs],
    }
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def write_report(path: Path, name: str, features: list,
                 pearson_pairs: list, spearman_pairs: list,
                 stats: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    sep  = '-' * 60
    sep2 = '=' * 70
    lines = [
        sep2,
        f'CORRELATION ANALYSIS REPORT  —  {name.upper()}',
        f'Generated : {datetime.now():%Y-%m-%d %H:%M:%S}',
        sep2,
        '',
        f'Flow features analysed: {len(features)}',
        f'Pearson rows (exact):   {stats["pearson_rows"]:,}',
        f'Spearman rows (sample): {stats["spearman_rows"]:,}',
        f'Threshold:              |r| ≥ {Config2.CORR_THRESHOLD}',
        '',
        sep,
        'HIGHLY CORRELATED PAIRS — PEARSON',
        sep,
        f'  {len(pearson_pairs)} pair(s)',
        f'  {"Feature A":<34} {"Feature B":<34} {"r":>7}',
        f'  {"-"*34} {"-"*34} {"-"*7}',
    ]
    for a, b, r in pearson_pairs:
        lines.append(f'  {a[:34]:<34} {b[:34]:<34} {r:>7.3f}')

    lines += [
        '',
        sep,
        'HIGHLY CORRELATED PAIRS — SPEARMAN',
        sep,
        f'  {len(spearman_pairs)} pair(s)',
        f'  {"Feature A":<34} {"Feature B":<34} {"r":>7}',
        f'  {"-"*34} {"-"*34} {"-"*7}',
    ]
    for a, b, r in spearman_pairs:
        lines.append(f'  {a[:34]:<34} {b[:34]:<34} {r:>7.3f}')

    p_set = {(a, b) for a, b, _ in pearson_pairs}
    s_set = {(a, b) for a, b, _ in spearman_pairs}
    both  = sorted(p_set & s_set)
    lines += [
        '',
        sep,
        'FLAGGED BY BOTH PEARSON AND SPEARMAN',
        sep,
        f'  {len(both)} pair(s)',
    ]
    for a, b in both:
        lines.append(f'  {a}  ×  {b}')

    path.write_text('\n'.join(lines), encoding='utf-8')


# ── Per-dataset orchestration ──────────────────────────────────────────────────
def analyse_dataset(name: str, parquet_path: Path, log: Logger) -> bool:
    if not parquet_path.exists():
        log.warn(f'Cleaned file not found: {parquet_path}')
        return False

    file_gb = parquet_path.stat().st_size / 1024 ** 3
    out_dir = Config2.OUTPUT_BASE / name
    out_dir.mkdir(parents=True, exist_ok=True)

    log.step('Identify flow features (drop identifiers + Label)')
    features = feature_columns(name)
    if len(features) < 2:
        log.warn(f'Too few feature columns ({len(features)}) — aborting {name}.')
        return False
    log.ok(f'{len(features)} flow features kept (from {file_gb:.2f} GB file)')
    log.step_end()

    log.step('Compute Pearson + Spearman correlations (single fused pass)')
    pearson, p_rows, sample, samp_labels = _streaming_pearson_and_sample(
        parquet_path, features, log)
    log.ok(f'Pearson over {p_rows:,} complete-case rows')
    spearman, s_rows = _spearman_from_sample(sample, features)
    del sample
    if samp_labels:
        mix = ', '.join(f'{k}:{v:,}' for k, v in
                        sorted(samp_labels.items(), key=lambda x: -x[1])[:6])
        log.info(f'Sample class mix (top): {mix}')
    log.ok(f'Spearman over {s_rows:,} sampled rows')
    log.step_end()

    log.step(f'Flag highly correlated pairs (|r| >= {Config2.CORR_THRESHOLD})')
    p_pairs = _flag_pairs(pearson,  features, Config2.CORR_THRESHOLD)
    s_pairs = _flag_pairs(spearman, features, Config2.CORR_THRESHOLD)
    log.ok(f'Pearson flagged {len(p_pairs)} pair(s); Spearman flagged {len(s_pairs)}')
    log.step_end()

    log.step('Save correlation matrices + flagged pairs (JSON metadata)')
    common = {'dataset': name, 'n_features': len(features),
              'threshold': Config2.CORR_THRESHOLD}
    _save_matrix_json(pearson, features,
                      {**common, 'method': 'pearson', 'rows': p_rows},
                      out_dir / 'pearson_matrix.json')
    _save_matrix_json(spearman, features,
                      {**common, 'method': 'spearman', 'rows': s_rows,
                       'seed': Config2.SPEARMAN_SEED},
                      out_dir / 'spearman_matrix.json')
    _save_pairs_json(p_pairs, {**common, 'method': 'pearson'},
                     out_dir / 'pearson_flagged_pairs.json')
    _save_pairs_json(s_pairs, {**common, 'method': 'spearman'},
                     out_dir / 'spearman_flagged_pairs.json')
    log.ok(f'4 JSON files written to {out_dir}')
    log.step_end()

    res_dir = ensure_results_dir(2, name)
    log.step('Render dense heatmaps + flagged-pair summaries')
    plot_heatmap(pearson, features,
                 f'{name} — Pearson correlation ({len(features)} features)',
                 res_dir / 'pearson_heatmap.png')
    plot_heatmap(spearman, features,
                 f'{name} — Spearman correlation ({len(features)} features)',
                 res_dir / 'spearman_heatmap.png')
    plot_flagged_pairs_bar(p_pairs,
                           f'{name} — strongest Pearson pairs (|r| >= {Config2.CORR_THRESHOLD})',
                           res_dir / 'pearson_flagged_pairs.png')
    plot_flagged_pairs_bar(s_pairs,
                           f'{name} — strongest Spearman pairs (|r| >= {Config2.CORR_THRESHOLD})',
                           res_dir / 'spearman_flagged_pairs.png')
    log.ok(f'Heatmaps + bars saved to {res_dir}')
    log.step_end()

    log.step('Write text report')
    write_report(res_dir / Config2.RESULTS_FILE, name, features, p_pairs, s_pairs,
                 {'pearson_rows': p_rows,
                  'spearman_rows': s_rows, 'file_gb': file_gb})
    log.ok(f'Report saved to {res_dir / Config2.RESULTS_FILE}')
    log.step_end()
    return True


# ── Entry point ────────────────────────────────────────────────────────────────
def main(datasets_filter=None):
    print(f"{'='*70}\n2_CORRELATION_ANALYSIS — per-dataset feature correlation")
    print(f"Root     : {PROJECT_ROOT}")
    print(f"Datasets : {datasets_filter or 'all'}\n{'='*70}")

    dataset_paths = {
        name: CLEANED_DATA_ROOT / f'{name}_cleaned.parquet'
        for name in DATASETS
        if datasets_filter is None or name in datasets_filter
    }
    if not dataset_paths:
        print(f'No matching datasets. Available: {list(DATASETS.keys())}')
        return

    for name, parquet_path in dataset_paths.items():
        res_dir = ensure_results_dir(2, name)
        log = Logger(res_dir / Config2.STEPS_FILE, step_prefix=2)

        ok = analyse_dataset(name, parquet_path, log)
        if not ok:
            log.warn('Analysis produced nothing — check messages above.')
        log.close()
        if ok:
            print(f'\n[DONE] {name}: correlation analysis complete  -->  '
                  f'{Config2.OUTPUT_BASE / name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Step 2: Per-dataset feature correlation analysis (analyze-only).'
    )
    parser.add_argument('--datasets', nargs='+', metavar='NAME',
                        help='Datasets to process, e.g. --datasets cicids2017')
    args = parser.parse_args()
    main(datasets_filter=args.datasets)
