"""
0_dataexplore.py — Raw data exploration for NIDS datasets.

PURPOSE:
  Single-pass exploration of raw CSV data. Counts rows, tracks label distributions,
  measures data quality (NaN/Inf), and generates reports and visualizations. No changes
  to the data only reads and makes reports.

PROCESS:
  Sub-step 0.1 (Validation):
    - Validate that ALL CSV files across both datasets have identical headers
    - Fails fast if any mismatch detected

  Sub-step 0.2 (Column inventory):
    - Read headers from all CSV files (header-only, no data loading)
    - Build complete column list and data types

  Sub-step 0.3 (Single-pass analysis):
    - Parallel processing (one worker per file)
    - For each file: count raw rows, track label distributions,
      count NaN/Inf rows, measure data quality
    - Mental notes: (1) quality flags (clean/nan_only/inf_only/both),
                     (2) labels (before/after attempted consolidation)

  Sub-step 0.4 (Visualizations):
    - Generate 4 PNG charts per dataset (dataset-specific formatting)
    - Results stored in results/0_dataexplore/<dataset>/

INPUTS:
  - Raw CSV files from data/raw_data/cicids2017/ and data/raw_data/cicids2018/

OUTPUTS:
  - results/0_dataexplore/<dataset>/0_dataexplore_report.txt (statistics report)
  - results/0_dataexplore/<dataset>/0_dataexplore_steps.log (execution log with sub-steps)
  - results/0_dataexplore/<dataset>/ (4 PNG visualization charts)

GUARANTEES:
  - No source data is modified
  - All CSV files across both datasets must have identical headers (validation fails if not)
  - Each file is read exactly once (single-pass design)
"""

import sys
import argparse
import csv
import numpy as np
from pathlib import Path
from collections import Counter
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── Config Import ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from unified_config import (
    Config0, DATASETS, PROJECT_ROOT, ensure_results_dir, Logger, plan_workers,
    LABEL_STANDARDIZATION, DATA_QUALITY_TYPES, DATA_QUALITY_COLORS, is_label_column,
)


# ─── Helpers ────────────────────────────────────────────────────────────────
def _csv_files(folder: Path) -> list:
    """Return sorted list of CSV files (largest first)."""
    return sorted(folder.glob('*.csv'), key=lambda p: p.stat().st_size, reverse=True)

def _human_size(path: Path) -> str:
    """Format file size as human-readable."""
    mb = path.stat().st_size / (1024 ** 2)
    return f"{mb / 1024:.2f} GB" if mb >= 1024 else f"{mb:.0f} MB"

def _normalise(lbl: str) -> str:
    """Normalize label using the shared standardization map."""
    s = str(lbl).strip()
    return LABEL_STANDARDIZATION.get(s.lower(), s)

def _is_attempted(lbl: str) -> bool:
    """Check if label ends with '- attempted' (case-insensitive)."""
    return lbl.lower().endswith('- attempted')

def _get_csv_header(filepath: Path) -> list:
    """Read and return the first row (header) of a CSV file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)
            return headers
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


# ─── Sub-step 0.1: Validate CSV headers across all datasets ─────────────────
def validate_headers(log: Logger) -> bool:
    """
    Check that ALL CSV files (both 2017 and 2018) have identical headers.
    If any mismatch: log details and return False. Otherwise return True.
    """
    log.step('Validate CSV headers across all datasets')

    all_files = {}
    for ds_name, ds_path in DATASETS.items():
        if not ds_path.exists():
            log.info(f"{ds_name}: folder not found, skipping header check")
            continue

        csv_files = _csv_files(ds_path)
        if not csv_files:
            log.info(f"{ds_name}: no CSV files found")
            continue

        for fp in csv_files:
            headers = _get_csv_header(fp)
            if headers:
                all_files[f"{ds_name}/{fp.name}"] = headers

    if not all_files:
        log.warn("No CSV files found across any dataset")
        return False

    log.info(f"Found {len(all_files)} CSV files total")

    # Group by header signature
    headers_groups = {}
    for filepath, headers in all_files.items():
        headers_key = tuple(headers)
        if headers_key not in headers_groups:
            headers_groups[headers_key] = []
        headers_groups[headers_key].append(filepath)

    if len(headers_groups) == 1:
        log.ok(f"All {len(all_files)} CSV files have identical headers ({len(list(all_files.values())[0])} columns)")
        return True
    else:
        log.warn(f"MISMATCH: Found {len(headers_groups)} different header configurations")
        for idx, (headers_tuple, filepaths) in enumerate(headers_groups.items(), 1):
            headers_list = list(headers_tuple)
            log.warn(f"  Config {idx}: {len(headers_list)} columns, {len(filepaths)} file(s)")
            for fp in filepaths[:3]:  # Show first 3
                log.warn(f"    - {fp}")
            if len(filepaths) > 3:
                log.warn(f"    ... and {len(filepaths) - 3} more")
        log.warn("Please fix header mismatches and try again.")
        return False


# ─── Sub-step 0.2: Column inventory (header-only reads) ──────────────────────
def inventory_columns(files: list, log: Logger) -> dict:
    """Read headers from all files and check alignment."""
    log.step('Column inventory (header-only reads, 0 data rows loaded)')

    file_cols = {}
    for fp in files:
        try:
            hdr = pd.read_csv(fp, nrows=0)
            file_cols[fp.name] = list(hdr.columns)
            log.info(f"{fp.name}: {len(hdr.columns)} columns")
        except Exception as e:
            log.warn(f"{fp.name}: failed to read header ({e})")
            return {'columns': None, 'file_cols': {}, 'aligned': False}

    reference = file_cols[files[0].name]
    mismatches = {fn: cols for fn, cols in file_cols.items() if cols != reference}

    if not mismatches:
        log.ok(f"All {len(files)} files aligned: {len(reference)} columns, same order")
    else:
        log.warn(f"{len(mismatches)} file(s) differ from {files[0].name}")
        for fn, cols in mismatches.items():
            missing = sorted(set(reference) - set(cols))
            extra = sorted(set(cols) - set(reference))
            if missing:
                log.warn(f"  {fn} missing: {missing}")
            if extra:
                log.warn(f"  {fn} extra: {extra}")

    return {'columns': reference, 'file_cols': file_cols, 'aligned': not mismatches}


# ─── Sub-step 0.3: Single-pass analysis (parallel, one process per file) ────
def _explore_one_file(fp_str: str) -> dict:
    """
    Worker process: single-pass analysis of one CSV.
    Returns dict with two mental notes:
      1. Quality flags (clean, nan_only, inf_only, both)
      2. Labels (before and after attempted→Benign consolidation)
    """
    fp = Path(fp_str)

    # Mental Note 1: Data quality flags
    row_quality = {
        'clean': 0,
        'nan_only': 0,
        'inf_only': 0,
        'both': 0,
    }

    # Mental Note 2: Labels
    label_before = Counter()
    label_after = Counter()
    attempted_map = Counter()

    total_rows = 0

    for chunk in pd.read_csv(fp, chunksize=Config0.CHUNK_ROWS, low_memory=False):
        total_rows += len(chunk)

        # Find Label column (may have leading space)
        lc = next((c for c in chunk.columns if is_label_column(c)), None)
        if lc:
            for raw_lbl, cnt in chunk[lc].fillna('__NaN__').astype(str).value_counts().items():
                norm = _normalise(raw_lbl)
                label_before[norm] += cnt
                if _is_attempted(norm):
                    label_after['Benign'] += cnt
                    attempted_map[norm] += cnt
                else:
                    label_after[norm] += cnt

        # Quality flags per row
        row_has_nan = chunk.isna().any(axis=1)

        # num_cols is restricted to numeric dtypes by select_dtypes(include='number'),
        # so a single float64 cast + vectorized isinf over all columns at once is
        # equivalent to the per-column loop it replaces (no per-column try/except needed).
        num_cols = list(chunk.select_dtypes(include='number').columns)
        row_has_inf = np.isinf(chunk[num_cols].to_numpy(dtype='float64')).any(axis=1)

        row_quality['clean'] += int((~row_has_nan & ~row_has_inf).sum())
        row_quality['nan_only'] += int((row_has_nan & ~row_has_inf).sum())
        row_quality['inf_only'] += int((~row_has_nan & row_has_inf).sum())
        row_quality['both'] += int((row_has_nan & row_has_inf).sum())

    return {
        'file': fp.name,
        'total': total_rows,
        'quality': row_quality,
        'label_before': label_before,
        'label_after': label_after,
        'attempted_map': attempted_map,
    }


def analyze_files_parallel(files: list, log: Logger) -> dict:
    """Parallel single-pass analysis: one worker per file."""
    n_workers = plan_workers(len(files), Config0.MAX_WORKERS)
    log.step(f'Single-pass analysis (parallel, {n_workers} process(es))')

    results = {}
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        fut_to_fp = {ex.submit(_explore_one_file, str(fp)): fp for fp in files}
        for done, fut in enumerate(as_completed(fut_to_fp), 1):
            fp = fut_to_fp[fut]
            r = fut.result()
            dirty = r['quality']['nan_only'] + r['quality']['inf_only'] + r['quality']['both']
            log.info(
                f"[{done}/{len(files)}] {r['file']} ({_human_size(fp)}): "
                f"{r['total']:,} rows | clean {r['quality']['clean']:,} | NaN/Inf {dirty:,}"
            )
            if r['attempted_map']:
                log.info(
                    f"    Attempted→Benign: {len(r['attempted_map'])} label type(s), "
                    f"{sum(r['attempted_map'].values()):,} rows total"
                )
            results[r['file']] = {
                'total': r['total'],
                'quality': r['quality'],
                'label_before': r['label_before'],
                'label_after': r['label_after'],
                'attempted_map': r['attempted_map'],
            }

    log.ok()
    return results


# ─── Output: Results Report ──────────────────────────────────────────────────
def write_report(out_path: Path, dataset: str, columns: list, results: dict):
    """Write results.txt with dataset overview and per-file statistics."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sep = '-' * 70
    sep2 = '=' * 80

    # Aggregate all results
    all_label_before = Counter()
    all_label_after = Counter()
    all_attempted_map = Counter()
    total_all_rows = 0

    for r in results.values():
        all_label_before.update(r['label_before'])
        all_label_after.update(r['label_after'])
        all_attempted_map.update(r['attempted_map'])
        total_all_rows += r['total']

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"{sep2}\nDATA EXPLORATION REPORT — {dataset.upper()}\n"
                f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n{sep2}\n\n")

        # Section 1: Column inventory
        f.write(f"{sep}\nSECTION 1 — COLUMN INVENTORY\n{sep}\n")
        f.write(f"Total columns: {len(columns)}\n\n")
        f.write("Columns (in order):\n")
        for idx, col in enumerate(columns, 1):
            f.write(f"  {idx:3}. {col}\n")
        f.write("\n")

        # Section 2: Per-file statistics
        f.write(f"{sep}\nSECTION 2 — PER-FILE STATISTICS\n{sep}\n\n")
        for fn, r in results.items():
            t = r['total']
            clean = r['quality']['clean']
            nan_rows = r['quality']['nan_only'] + r['quality']['both']
            inf_rows = r['quality']['inf_only'] + r['quality']['both']

            f.write(f"File: {fn}\n")
            f.write(f"  Total rows: {t:,}\n\n")
            f.write("  Data quality:\n")
            f.write(f"    Clean: {clean:,}\n")
            f.write(f"    NaN (any column): {nan_rows:,}\n")
            f.write(f"    Inf (any column): {inf_rows:,}\n")
            f.write(f"    Both NaN + Inf: {r['quality']['both']:,}\n\n")

            # Before consolidation
            if r['label_before']:
                f.write("  Labels (before consolidation):\n")
                for lbl, cnt in sorted(r['label_before'].items(), key=lambda x: -x[1]):
                    f.write(f"    {lbl}: {cnt:,}\n")
                f.write("\n")

            # After consolidation
            if r['label_after']:
                f.write("  Labels (after consolidation):\n")
                for lbl, cnt in sorted(r['label_after'].items(), key=lambda x: -x[1]):
                    f.write(f"    {lbl}: {cnt:,}\n")
                f.write("\n")

            f.write("\n")

        # Section 3: Dataset-wide totals
        f.write(f"{sep}\nSECTION 3 — DATASET TOTALS\n{sep}\n")
        f.write(f"Total rows across all files: {total_all_rows:,}\n\n")

        if all_label_after:
            f.write("Total rows per label (after consolidation):\n")
            for lbl, cnt in sorted(all_label_after.items(), key=lambda x: -x[1]):
                pct = 100 * cnt / total_all_rows if total_all_rows else 0
                f.write(f"  {lbl}: {cnt:,}  ({pct:.2f}%)\n")



# ─── Output: Visualizations ─────────────────────────────────────────────────
def _axis_formatter(scale: str):
    """Return a matplotlib FuncFormatter for the given scale string ('K', 'M', or 'comma')."""
    if scale == 'K':
        return plt.FuncFormatter(lambda v, _: f'{int(v/1000)}K')
    if scale == 'M':
        return plt.FuncFormatter(lambda v, _: f'{v/1_000_000:.2f}M')
    return plt.FuncFormatter(lambda v, _: f'{int(v):,}')


def write_visuals(visuals_dir: Path, dataset: str, results: dict):
    """Generate 4 PNG visualizations per dataset."""
    ds_key = dataset.lower()
    c1_ymin       = Config0.CHART1_YMIN.get(ds_key,       Config0.CHART1_YMIN_DEFAULT)
    c1_ytick_step = Config0.CHART1_YTICK_STEP.get(ds_key,  Config0.CHART1_YTICK_STEP_DEFAULT)
    c1_yscale     = Config0.CHART1_YSCALE.get(ds_key,      Config0.CHART1_YSCALE_DEFAULT)
    c3_ytick_step = Config0.CHART3_YTICK_STEP.get(ds_key,  Config0.CHART3_YTICK_STEP_DEFAULT)
    c3_yscale     = Config0.CHART3_YSCALE.get(ds_key,      Config0.CHART3_YSCALE_DEFAULT)
    visuals_dir.mkdir(parents=True, exist_ok=True)

    all_label_after = Counter()
    for r in results.values():
        all_label_after.update(r['label_after'])

    if not all_label_after:
        return

    fnames = sorted(results.keys())
    labels_sorted = sorted(all_label_after.keys())

    # CHART 1: Files on x-axis, labels stacked
    fig, ax = plt.subplots(figsize=(max(14, len(fnames) * 1.5), 7))
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels_sorted)))
    bottom = np.zeros(len(fnames))
    for lbl_idx, lbl in enumerate(labels_sorted):
        counts = [results[fn]['label_after'].get(lbl, 0) for fn in fnames]
        ax.bar(range(len(fnames)), counts, label=lbl, bottom=bottom, color=colors[lbl_idx])
        bottom += np.array(counts)
    ax.set_xticks(range(len(fnames)))
    ax.set_xticklabels([fn.replace('.csv', '') for fn in fnames], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Row Count')
    ax.set_title(f'Label Distribution by File — {dataset}', fontsize=12, fontweight='bold')
    ax.set_ylim(c1_ymin, max(bottom) * 1.05)
    ax.set_yticks(np.arange(c1_ymin, int(max(bottom) * 1.05), c1_ytick_step))
    ax.yaxis.set_major_formatter(_axis_formatter(c1_yscale))
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(visuals_dir / '1_labels_per_file.png', dpi=150, bbox_inches='tight')
    plt.close()

    # CHART 2: Labels on x-axis, files stacked (log scale, blue, NO legend, NO lines)
    fig, ax = plt.subplots(figsize=(max(14, len(labels_sorted) * 1.5), 7))
    bottom = np.zeros(len(labels_sorted))
    for fn in fnames:
        counts = [results[fn]['label_after'].get(lbl, 0) for lbl in labels_sorted]
        ax.bar(range(len(labels_sorted)), counts, bottom=bottom, color='#4C9BE8', linewidth=0)
        bottom += np.array(counts)
    ax.set_xticks(range(len(labels_sorted)))
    ax.set_xticklabels(labels_sorted, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Row Count (log scale)')
    ax.set_title(f'Label Composition by Source File — {dataset}', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.set_yticks([10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(visuals_dir / '2_files_per_label.png', dpi=150, bbox_inches='tight')
    plt.close()

    # CHART 3: Files on x-axis, data quality stacked
    fig, ax = plt.subplots(figsize=(max(14, len(fnames) * 1.5), 7))
    bottom = np.zeros(len(fnames))
    for q_idx, q_type in enumerate(DATA_QUALITY_TYPES):
        counts = [results[fn]['quality'].get(q_type, 0) for fn in fnames]
        ax.bar(range(len(fnames)), counts, label=q_type.replace('_', ' ').title(),
               bottom=bottom, color=DATA_QUALITY_COLORS[q_idx])
        bottom += np.array(counts)
    ax.set_xticks(range(len(fnames)))
    ax.set_xticklabels([fn.replace('.csv', '') for fn in fnames], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Row Count')
    ax.set_title(f'Data Quality by File (NaN/Inf) — {dataset}', fontsize=12, fontweight='bold')
    if c3_ytick_step is not None:
        ax.set_yticks(np.arange(0, int(max(bottom) * 1.05), c3_ytick_step))
    ax.yaxis.set_major_formatter(_axis_formatter(c3_yscale))
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(visuals_dir / '3_quality_per_file.png', dpi=150, bbox_inches='tight')
    plt.close()

    # CHART 4: Quality on x-axis, files stacked (log scale, blue, NO legend, NO lines)
    fig, ax = plt.subplots(figsize=(max(12, len(DATA_QUALITY_TYPES) * 2), 7))
    bottom = np.zeros(len(DATA_QUALITY_TYPES))
    for fn in fnames:
        counts = [results[fn]['quality'].get(q_type, 0) for q_type in DATA_QUALITY_TYPES]
        ax.bar(range(len(DATA_QUALITY_TYPES)), counts, bottom=bottom, color='#4C9BE8', linewidth=0)
        bottom += np.array(counts)
    ax.set_xticks(range(len(DATA_QUALITY_TYPES)))
    ax.set_xticklabels([q.replace('_', ' ').title() for q in DATA_QUALITY_TYPES],
                       rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Row Count (log scale)')
    ax.set_title(f'Quality Type Composition — {dataset}', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.set_yticks([10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(visuals_dir / '4_files_per_quality.png', dpi=150, bbox_inches='tight')
    plt.close()


# ─── Per-dataset orchestration ──────────────────────────────────────────────
def explore_dataset(name: str, path: Path):
    """Run analysis for one dataset."""
    print(f"\n{'#'*70}\n# EXPLORING: {name.upper()}\n# Path: {path}\n{'#'*70}")

    results_dir = ensure_results_dir(0, name)
    log = Logger(results_dir / Config0.STEPS_FILE, step_prefix=0, title='0_DATAEXPLORE STEPS LOG')

    files = _csv_files(path)
    if not files:
        log.warn(f"No CSV files found in {path}")
        log.close()
        return None

    log.info(f"Found {len(files)} CSV file(s) (largest first):")
    for fp in files:
        log.info(f"  {fp.name}  ({_human_size(fp)})")
    log.info("")

    col_info = inventory_columns(files, log)
    if not col_info['aligned']:
        log.warn("Column alignment check failed. Aborting.")
        log.close()
        return None

    results = analyze_files_parallel(files, log)

    log.step('Writing results report')
    write_report(results_dir / Config0.RESULTS_FILE, name, col_info['columns'], results)

    log.step('Writing visualization charts')
    write_visuals(results_dir, name, results)

    log.ok(f'All outputs written to {results_dir}')
    log.close()
    print(f"\n  Results: {results_dir}")

    return results


# ─── Entry point ────────────────────────────────────────────────────────────
def main(datasets_filter: list = None):
    """Main orchestration."""
    print(
        f"{'='*70}\n"
        f"0_DATAEXPLORE — NIDS Raw Data Exploration\n"
        f"Root: {PROJECT_ROOT}\n"
        f"Datasets: {datasets_filter or 'all'}\n"
        f"{'='*70}"
    )

    # Create a temporary logger for header validation
    temp_log_path = PROJECT_ROOT / '.temp_validation.log'
    log = Logger(temp_log_path, step_prefix=0, title='0_DATAEXPLORE VALIDATION LOG')

    # Sub-step 0.1: Validate all headers before processing any dataset
    if not validate_headers(log):
        log.warn("Header validation failed. Cannot proceed.")
        log.close()
        temp_log_path.unlink(missing_ok=True)
        sys.exit(1)

    log.ok("Header validation passed. Proceeding with data exploration.")
    log.close()
    temp_log_path.unlink(missing_ok=True)

    # Determine which datasets to process
    active = {}
    for name, path in DATASETS.items():
        if datasets_filter and name not in datasets_filter:
            print(f"[SKIP] {name}: not in --datasets filter")
            continue
        if not path.exists():
            print(f"[SKIP] {name}: folder not found at {path}")
            continue
        active[name] = path

    if not active:
        print("\nNo datasets to process.")
        sys.exit(1)

    # Process each dataset
    for name, path in active.items():
        dataset_files = _csv_files(path)
        if dataset_files:
            explore_dataset(name, path)

    print(f"\n{'='*70}\nExploration complete.\nResults: {Config0.RESULTS_DIR}\n{'='*70}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Explore raw NIDS CSV data — columns, rows, labels, data quality.'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        metavar='NAME',
        help='Datasets to process (e.g. --datasets cicids2017 cicids2018)'
    )
    args = parser.parse_args()
    main(datasets_filter=args.datasets)
