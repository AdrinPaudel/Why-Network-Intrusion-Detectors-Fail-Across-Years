"""
1_load_clean_combine.py — Load, clean, combine, and label-consolidate CIC-IDS datasets.

PURPOSE:
  Clean raw CSV data, deduplicate globally, and consolidate labels into canonical families.
  Optimized pipeline: clean once, deduplicate by marking, combine once.

PROCESS:
  Sub-step 1.1 (Parallel cleaning + vectorized hashing):
    For each raw CSV (largest first, processed in parallel processes):
      a. Stream in chunks (chunk size auto-sized to available RAM)
      b. Strip column names; drop 'Attempted Category' (redundant after relabel)
      c. Relabel any Label ending in '- Attempted' -> Benign
      d. Drop rows with NaN
      e. Drop rows with Inf (numeric columns only)
      f. Hash each cleaned chunk while in RAM (vectorized, excluding 'id') and keep one
         uint64 hash array per file — chunk order == temp-CSV row order
      g. Write cleaned rows to per-file temp CSV (id and all data columns kept)
    Result: 10 temp CSVs + 10 per-file hash arrays (in memory)

  Sub-step 1.2 (Global deduplication decision):
    - Concatenate all per-file hash arrays (no I/O)
    - One vectorized pandas duplicated(keep='first') pass over the whole dataset
    - Split the result back into a per-file positional keep mask
    Result: One boolean keep mask per file (applied while writing shards in 1.3)

  Sub-step 1.3 (Filter + label consolidation + floor filter):
    - Parallel workers re-read each temp CSV once, apply the positional keep mask
      (so dedup and shard-writing share a single read — no separate delete pass),
      map raw labels to canonical families (DoS Hulk -> DoS, etc.), drop unmapped rows,
      and STREAM survivors to a per-file Parquet shard (one row group per chunk via
      pyarrow.ParquetWriter — constant memory, never materializes the whole file)
    - Combine shards with pl.scan_parquet(...).filter(floor).sink_parquet(out): a
      streaming lazy pass that drops families below the 100-row floor (MIN_CLASS_ROWS)
      without ever holding the full dataset in RAM
    Result: Final cleaned dataset (data/cc_data/<dataset>_cleaned.parquet)

  Sub-step 1.4 (Write report):
    - Generate statistics report (row counts, per-file stats, label inventory)
    Result: results/1_load_clean_combine/<dataset>/1_load_clean_combine_report.txt

INPUTS:
  - Raw CSV files from data/raw_data/<dataset>/ (cicids2017, cicids2018)

OUTPUTS:
  - data/cc_data/<dataset>_cleaned.parquet (final cleaned dataset, ready for step 2)
  - results/1_load_clean_combine/<dataset>/1_load_clean_combine_report.txt (statistics report)
  - results/1_load_clean_combine/<dataset>/1_load_clean_combine_steps.log (execution log)

GUARANTEES:
  - No source data is modified
  - Each file is read exactly once during cleaning (sub-step 1.1)
  - Each temp file is read exactly once during deduplication (sub-step 1.3)
  - Identifiers (id, IPs, ports, timestamp, FlowID) preserved through cleaning

DOCUMENTED ASYMMETRIES:
  - PortScan: present in 2017, absent in 2018 (kept in 2017, no cross-year counterpart)
  - Infiltration: SUBSTANTIAL in BOTH years (exact counts in step-1 report)
  - FTP-BruteForce (2018): all rows were Attempted → relabeled Benign here
  - Shared families for cross-year work (Step 11): DoS, DDoS, BruteForce, WebAttack, Botnet, Infiltration
"""

import sys
import gc
import os
import json
import argparse
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Import config from unified_config ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from unified_config import (
    Config1, DATASETS, CLEANED_DATA_ROOT, map_label,
    ensure_results_dir, Logger, plan_workers,
)


# ── Workload planner ──────────────────────────────────────────────────────────
def _plan_workload(n_files: int, label: str, log: 'Logger') -> tuple:
    """Fixed worker cap scaled by file count + a fixed per-chunk row count. No RAM probing."""
    workers = plan_workers(n_files, Config1.MAX_WORKERS)
    rows    = Config1.CHUNK_ROWS
    log.ok(f'{label}: {workers} processes x {rows:,} rows/chunk '
           f'({os.cpu_count() or 4} cores, {n_files} files)')
    return workers, rows


# ── Per-chunk cleaning ────────────────────────────────────────────────────────
def clean_chunk(chunk: pd.DataFrame) -> tuple:
    """
    Clean chunk: drop columns, relabel Attempted->Benign, drop NaN/Inf.
    Returns (cleaned_chunk, nan_removed, inf_removed, inf_diagnostics, nan_by_label).
    """
    chunk.columns = [c.strip() for c in chunk.columns]

    drop = [c for c in chunk.columns if c.lower() in Config1.DROP_COL_SET]
    if drop:
        chunk = chunk.drop(columns=drop)

    label_col = next((c for c in chunk.columns if c.lower() == 'label'), None)
    if label_col:
        labels = chunk[label_col].astype(str).str.strip()
        attempted = labels.str.lower().str.endswith('- attempted', na=False)
        labels = labels.mask(attempted, 'Benign')
        chunk = chunk.copy()
        chunk[label_col] = labels

    before_nan = len(chunk)
    # Capture per-label NaN-drop counts the same way inf_diag['by_label'] already
    # does below, BEFORE the rows are dropped — lets a later cross-year comparison check whether
    # the NaN-drop rate disproportionately removes certain attack families in one year vs the
    # other (previously only the Inf-drop side of this had a by-label breakdown).
    nan_by_label = {}
    has_nan = chunk.isna().any(axis=1)
    if has_nan.any() and label_col is not None:
        nan_by_label = {
            str(k): int(v)
            for k, v in chunk.loc[has_nan, label_col].astype(str).value_counts().items()
        }
    chunk = chunk.dropna()
    nan_rm = before_nan - len(chunk)

    before_inf = len(chunk)
    num_cols = chunk.select_dtypes(include='number').columns
    inf_diag = {'n': 0, 'n_zero_duration': 0, 'by_label': {}}
    if len(num_cols) > 0:
        # num_cols is restricted to numeric dtypes by select_dtypes(include='number'),
        # so a single float64 cast + vectorized isinf over all columns at once is
        # equivalent to the per-column loop it replaces (no per-column try/except needed).
        has_inf = np.isinf(chunk[list(num_cols)].to_numpy(dtype='float64')).any(axis=1)
        if has_inf.any():
            inf_rows = chunk.loc[has_inf]
            inf_diag['n'] = int(has_inf.sum())
            dur_col = next((c for c in chunk.columns if c.lower() == 'flow duration'), None)
            if dur_col is not None:
                try:
                    inf_diag['n_zero_duration'] = int((inf_rows[dur_col].astype(float) == 0).sum())
                except (ValueError, TypeError):
                    pass
            lc = next((c for c in chunk.columns if c.lower() == 'label'), None)
            if lc is not None:
                inf_diag['by_label'] = {
                    str(k): int(v)
                    for k, v in inf_rows[lc].astype(str).value_counts().items()
                }
            chunk = chunk.loc[~has_inf]
    inf_rm = before_inf - len(chunk)

    return chunk, nan_rm, inf_rm, inf_diag, nan_by_label

# ── Shared per-label tally accumulation (used by both the Inf-drop and NaN-drop audits) ──
def _accumulate_by_label(counter: dict, chunk_dict: dict) -> None:
    """Add per-label counts from chunk_dict into counter, in place."""
    for lbl, cnt in chunk_dict.items():
        counter[lbl] = counter.get(lbl, 0) + cnt

# ── Vectorized row hashing ─────────────────────────────────────────────────────
def _hash_chunk(df: pd.DataFrame, exclude_cols: frozenset) -> np.ndarray:
    """
    Vectorized row hash (uint64) over every column except `exclude_cols` ('id').

    Uses pandas' C-level hash_pandas_object — one vectorized pass over the chunk, no
    per-row Python loop and no GPU round-trip (the previous blake3+cupy version hashed
    one row at a time in Python, which dominated runtime on tens of millions of rows).
    """
    hcols = [c for c in df.columns if c.lower() not in exclude_cols]
    if not hcols or len(df) == 0:
        return np.empty(0, dtype='uint64')
    return pd.util.hash_pandas_object(df[hcols], index=False).to_numpy()

# ── Sub-step 1.1: Parallel clean + hash ───────────────────────────────────────
def _clean_and_hash_file(fp: Path, temp_path: Path, chunk_rows: int) -> dict:
    """
    Read, clean, hash, and write one raw CSV to temp file.
    Returns hash table [hash, id] and statistics.
    """
    f_raw = f_nan = f_inf = f_clean = 0
    label_counts = {}
    pre_clean_counts = {}
    inf_zero_dur = 0
    inf_by_label = {}
    nan_by_label = {}
    hash_parts = []  # one uint64 array per cleaned chunk, in row (write) order
    header_written = False

    for chunk in pd.read_csv(fp, chunksize=chunk_rows, low_memory=False):
        f_raw += len(chunk)

        # Count labels before cleaning
        lc_pre = next((c for c in chunk.columns if c.strip().lower() == 'label'), None)
        if lc_pre:
            for lbl, cnt in chunk[lc_pre].astype(str).str.strip().value_counts().items():
                pre_clean_counts[lbl] = pre_clean_counts.get(lbl, 0) + int(cnt)

        chunk, nan_rm, inf_rm, inf_diag, nan_diag = clean_chunk(chunk)
        f_nan += nan_rm
        f_inf += inf_rm
        inf_zero_dur += inf_diag['n_zero_duration']
        _accumulate_by_label(inf_by_label, inf_diag['by_label'])
        _accumulate_by_label(nan_by_label, nan_diag)

        if len(chunk) > 0:
            f_clean += len(chunk)

            # Hash this chunk while it is already in RAM (vectorized — no extra read).
            # Order of the array == order rows are written to the temp CSV below.
            hash_parts.append(_hash_chunk(chunk, Config1.HASH_EXCLUDE))

            # Count labels after cleaning
            lc = next((c for c in chunk.columns if c.lower() == 'label'), None)
            if lc:
                for lbl, cnt in chunk[lc].value_counts().items():
                    label_counts[lbl] = label_counts.get(lbl, 0) + int(cnt)

            # Write to temp CSV
            chunk.to_csv(
                temp_path,
                mode='w' if not header_written else 'a',
                header=not header_written,
                index=False,
            )
            header_written = True

        del chunk
        gc.collect()

    file_hashes = np.concatenate(hash_parts) if hash_parts else np.empty(0, dtype='uint64')
    return {
        'file': fp.name,
        'raw': f_raw,
        'nan': f_nan,
        'inf': f_inf,
        'clean': f_clean,
        'label_counts': label_counts,
        'pre_clean_counts': pre_clean_counts,
        'inf_zero_dur': inf_zero_dur,
        'inf_by_label': inf_by_label,
        'nan_by_label': nan_by_label,
        'temp_path': temp_path,
        'hashes': file_hashes,
    }

# ── Sub-step 1.3 worker: dedup-filter, map labels, stream shard ───────────────
def _filter_and_map_shard(temp_path: Path, keep: np.ndarray, label_mapping: dict,
                          label_col: str, shard_path: Path, chunk_rows: int) -> dict:
    """
    Worker: re-read one temp Parquet file in chunks, apply the positional dedup keep
    mask, map raw labels to canonical families, drop unmapped rows, and STREAM the
    survivors to a per-file Parquet shard — one row group per chunk via
    pyarrow.ParquetWriter.

    Constant memory: never accumulates the file (the previous version appended every
    chunk to a list, pd.concat'd, then pl.from_pandas'd — materializing the entire file
    per worker AND, in the parent, the entire dataset — which thrashed swap). Dedup and
    shard-writing share this single read, so there is no separate delete pass.
    Process-safe: module-level so ProcessPoolExecutor can pickle it.
    """
    post_raw = {}
    final = {}
    unmapped = set()
    writer = None
    schema = None
    off = 0

    for chunk in pd.read_csv(temp_path, chunksize=chunk_rows, low_memory=False):
        n = len(chunk)
        # Apply the global dedup decision positionally
        chunk = chunk[keep[off:off + n]]
        off += n
        if len(chunk) == 0:
            continue

        # Count raw labels post-dedup (before mapping)
        for lbl, cnt in chunk[label_col].value_counts().items():
            post_raw[str(lbl)] = post_raw.get(str(lbl), 0) + int(cnt)

        # Map labels, drop unmapped rows
        mapped = chunk[label_col].astype(str).map(label_mapping)
        unmapped_mask = mapped.isna()
        if unmapped_mask.any():
            for u in chunk.loc[unmapped_mask, label_col].unique():
                unmapped.add(str(u))
            chunk = chunk.loc[~unmapped_mask].copy()
            mapped = mapped.loc[~unmapped_mask]
            if len(chunk) == 0:
                continue
        chunk[label_col] = mapped.values

        # Count canonical families
        for lbl, cnt in chunk[label_col].value_counts().items():
            final[str(lbl)] = final.get(str(lbl), 0) + int(cnt)

        # Stream chunk to Parquet shard
        table = (pa.Table.from_pandas(chunk, preserve_index=False) if schema is None
                 else pa.Table.from_pandas(chunk, schema=schema, preserve_index=False))
        if writer is None:
            schema = table.schema
            writer = pq.ParquetWriter(str(shard_path), schema, compression='snappy')
        writer.write_table(table)

    if writer is not None:
        writer.close()
    Path(temp_path).unlink(missing_ok=True)

    return {'post_raw': post_raw, 'final': final, 'unmapped': unmapped,
            'shard': shard_path if writer is not None else None}

# ── Core processing ───────────────────────────────────────────────────────────
def process_dataset(name: str, folder: Path, log: Logger) -> dict:
    """Full pipeline for one dataset."""

    files = sorted(folder.glob('*.csv'), key=lambda p: p.stat().st_size, reverse=True)
    if not files:
        log.warn(f'No CSV files found in {folder}')
        return {}

    sizes_mb = [f'{f.name} ({f.stat().st_size / 1e6:.0f} MB)' for f in files]
    log.ok(f'Found {len(files)} file(s): {sizes_mb}')

    CLEANED_DATA_ROOT.mkdir(parents=True, exist_ok=True)

    # Remove stale temp/shard files from prior crashed runs
    stale = (list(CLEANED_DATA_ROOT.glob(f'{name}_temp_*.csv'))
             + list(CLEANED_DATA_ROOT.glob(f'{name}_shard_*.parquet')))
    for sp in stale:
        try:
            sp.unlink()
        except OSError:
            pass
    if stale:
        log.warn(f'Removed {len(stale)} stale temp/shard file(s) from prior run.')

    n_workers, chunk_rows = _plan_workload(len(files), 'Workload plan', log)

    # ── Sub-step 1.1: Parallel clean + hash ────────────────────────────────
    log.step(f'Parallel clean + hash in memory ({n_workers} workers, {chunk_rows:,} rows/chunk)')
    temp_paths = [CLEANED_DATA_ROOT / f'{name}_temp_{i}.csv' for i in range(len(files))]
    file_results = [None] * len(files)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {
            executor.submit(_clean_and_hash_file, fp, temp_paths[i], chunk_rows): i
            for i, fp in enumerate(files)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            result = future.result()
            file_results[idx] = result
            log.ok(
                f'{result["file"]}: {result["raw"]:,} -> {result["clean"]:,} '
                f'(NaN:{result["nan"]:,}  Inf:{result["inf"]:,})'
            )

    log.step_end()

    total_raw = sum(r['raw'] for r in file_results)
    total_nan = sum(r['nan'] for r in file_results)
    total_inf = sum(r['inf'] for r in file_results)
    file_stats = [
        {'file': r['file'], 'raw': r['raw'], 'nan': r['nan'], 'inf': r['inf'], 'clean': r['clean']}
        for r in file_results
    ]

    raw_label_counts = {}
    pre_clean_label_counts = {}
    inf_zero_dur_total = 0
    inf_by_label_total = {}
    nan_by_label_total = {}
    for r in file_results:
        for lbl, cnt in r['label_counts'].items():
            raw_label_counts[lbl] = raw_label_counts.get(lbl, 0) + cnt
        for lbl, cnt in r.get('pre_clean_counts', {}).items():
            pre_clean_label_counts[lbl] = pre_clean_label_counts.get(lbl, 0) + cnt
        inf_zero_dur_total += r.get('inf_zero_dur', 0)
        _accumulate_by_label(inf_by_label_total, r.get('inf_by_label', {}))
        _accumulate_by_label(nan_by_label_total, r.get('nan_by_label', {}))

    if total_inf:
        zpct = 100.0 * inf_zero_dur_total / total_inf if total_inf else 0.0
        log.info(f'Inf-drop audit: {total_inf:,} rows removed; {inf_zero_dur_total:,} '
                 f'({zpct:.1f}%) were zero-duration short flows.')
        attack_inf = {k: v for k, v in inf_by_label_total.items()
                      if str(k).strip().lower() not in ('benign',)}
        if attack_inf:
            top = ', '.join(f'{k}:{v:,}' for k, v in
                            sorted(attack_inf.items(), key=lambda x: -x[1])[:6])
            log.warn(f'Inf-drop removed NON-benign rows (class-balance risk): {top}')

    # Same by-label audit for NaN-drop (previously only Inf-drop had one), so a
    # later cross-year comparison of dtype_reconciliation-style artifacts can check whether the
    # NaN-drop rate disproportionately removes certain attack families in one year vs the other.
    if total_nan:
        attack_nan = {k: v for k, v in nan_by_label_total.items()
                     if str(k).strip().lower() not in ('benign',)}
        log.info(f'NaN-drop audit: {total_nan:,} rows removed '
                 f'({100.0 * total_nan / total_raw:.2f}% of raw rows).')
        if attack_nan:
            top = ', '.join(f'{k}:{v:,}' for k, v in
                            sorted(attack_nan.items(), key=lambda x: -x[1])[:6])
            log.warn(f'NaN-drop removed NON-benign rows (class-balance risk): {top}')

    gc.collect()

    # ── Sub-step 1.2: Global dedup decision (one vectorized duplicated() pass) ──
    # Concatenate the per-file uint64 hash arrays and call duplicated() once in C.
    # keep=first is deterministic given the fixed file order. The result is split back
    # into a per-file positional keep mask, applied while shards are written in 1.3 —
    # no DuckDB, no row-by-row record building, no separate delete read pass.
    log.step('Global dedup decision (vectorized duplicated() over all row hashes)')
    per_file_hashes = [fr['hashes'] for fr in file_results]
    lengths = [len(h) for h in per_file_hashes]
    all_hash = (np.concatenate(per_file_hashes) if per_file_hashes
                else np.empty(0, dtype='uint64'))
    dup_mask = pd.Series(all_hash).duplicated(keep='first').to_numpy()
    dup_count = int(dup_mask.sum())
    keep_full = ~dup_mask

    keep_masks, off = [], 0
    for L in lengths:
        keep_masks.append(keep_full[off:off + L])
        off += L

    # Free hash memory — no longer needed after the decision.
    for fr in file_results:
        fr.pop('hashes', None)
    del per_file_hashes, all_hash, dup_mask, keep_full
    gc.collect()

    log.ok(f'Found {dup_count:,} duplicate rows to remove (keep=first) '
           f'from {sum(lengths):,} rows')
    log.step_end()

    # ── Sub-step 1.3: Parallel dedup-filter + consolidate labels + floor filter ──
    log.step('Parallel filter + consolidate labels + floor filter')

    # Validate columns across temp files
    first_cols = list(pd.read_csv(file_results[0]['temp_path'], nrows=0).columns)
    label_col = next((c for c in first_cols if c.lower() == 'label'), None)
    if label_col is None:
        log.warn('No Label column in cleaned data — aborting.')
        return {}

    for fr in file_results[1:]:
        cols = list(pd.read_csv(fr['temp_path'], nrows=0).columns)
        if cols != first_cols:
            log.warn(f'Column mismatch in {fr["file"]} vs {file_results[0]["file"]} — aborting.')
            return {}

    log.ok(f'{len(raw_label_counts)} unique raw labels before consolidation:')
    for lbl, cnt in sorted(raw_label_counts.items(), key=lambda x: -x[1]):
        log.ok(f'    {lbl:<50}  {cnt:>10,}')

    # Build label mapping lookup (once for all files)
    label_mapping = {str(lbl): map_label(str(lbl)) for lbl in raw_label_counts.keys()}

    shard_paths = [CLEANED_DATA_ROOT / f'{name}_shard_{i}.parquet' for i in range(len(file_results))]
    results_13 = [None] * len(file_results)

    n_shard_workers = plan_workers(len(file_results), Config1.MAX_SHARD_WORKERS)
    with ProcessPoolExecutor(max_workers=n_shard_workers) as executor:
        futs = {
            executor.submit(_filter_and_map_shard, fr['temp_path'], keep_masks[i],
                           label_mapping, label_col, shard_paths[i], chunk_rows): i
            for i, fr in enumerate(file_results)
        }
        for fut in as_completed(futs):
            idx = futs[fut]
            results_13[idx] = fut.result()

    del keep_masks
    gc.collect()

    # Consolidate results across all shards
    post_dedup_raw = {}
    final_counts = {}
    unmapped_set = set()

    for r in results_13:
        for k, v in r['post_raw'].items():
            post_dedup_raw[k] = post_dedup_raw.get(k, 0) + v
        for k, v in r['final'].items():
            final_counts[k] = final_counts.get(k, 0) + v
        unmapped_set |= r['unmapped']

    if unmapped_set:
        log.warn(f'Unmapped labels (dropped): {sorted(unmapped_set)}')

    log.ok(f'{len(final_counts)} canonical families, {sum(final_counts.values()):,} rows:')
    for lbl, cnt in sorted(final_counts.items(), key=lambda x: -x[1]):
        log.ok(f'    {lbl:<20}  {cnt:>10,}')

    # Apply floor filter + combine shards via a single STREAMING lazy pass.
    # scan_parquet -> filter -> sink_parquet never holds the full dataset in RAM
    # (the previous read_parquet-all + pl.concat materialized every row in the parent).
    small = {k: v for k, v in final_counts.items() if v < Config1.MIN_CLASS_ROWS}
    out_parquet = CLEANED_DATA_ROOT / f'{name}_cleaned.parquet'
    existing_shards = [str(r['shard']) for r in results_13
                       if r['shard'] and Path(r['shard']).exists()]

    if existing_shards:
        # vertical_relaxed upcasts columns whose dtype was inferred differently across
        # day-files (e.g. 'Fwd RST Flags' = Int64 in some files, Float64 in others) to a
        # common supertype, instead of raising SchemaError when the shards are stacked.
        lf = pl.concat([pl.scan_parquet(s) for s in existing_shards],
                       how='vertical_relaxed')
        if small:
            log.info(f'Filtering families below {Config1.MIN_CLASS_ROWS}-row floor: {small}')
            keep_families = [k for k in final_counts if k not in small]
            lf = lf.filter(pl.col(label_col).is_in(keep_families))
            for lbl in small:
                final_counts.pop(lbl, None)

        lf.sink_parquet(out_parquet, compression='snappy')

        for sp in existing_shards:
            Path(sp).unlink(missing_ok=True)

        if small:
            log.ok(f'{sum(final_counts.values()):,} rows after floor filter')
        log.ok(f'Saved: {out_parquet}  ({out_parquet.stat().st_size / 1e9:.2f} GB)')
    else:
        log.warn('No rows remain after filtering — wrote empty file.')
        pl.DataFrame().write_parquet(out_parquet)

    gc.collect()
    log.step_end()

    return {
        'total_raw': total_raw,
        'nan_removed': total_nan,
        'inf_removed': total_inf,
        'dup_removed': dup_count,
        'final_rows': sum(final_counts.values()),
        'final_counts': final_counts,
        'pre_clean_counts': pre_clean_label_counts,
        'raw_counts': raw_label_counts,
        'post_dedup_raw': post_dedup_raw,
        'dropped_labels': small,
        'file_stats': file_stats,
        'inf_zero_dur': inf_zero_dur_total,
        'inf_by_label': inf_by_label_total,
        'nan_by_label': nan_by_label_total,
    }

# ── Cross-year dtype reconciliation ──────────────────────────────────────
def check_cross_year_dtypes(names: list, log: Logger) -> dict:
    """Structural check: `process_dataset()` runs once per year
    with no shared state, so pandas/polars infers each year's column dtypes independently — if the
    SAME nominal column (e.g. Protocol) ends up a different dtype in the two years' cleaned
    parquets, downstream C2ST-AUC/MMD/etc. (steps 7-10, which read these parquets directly as the
    "raw-unit" basis for every shift metric) would see that as covariate shift when it is actually
    a cleaning-stage artifact, not a real distributional difference. Cheap (schema-only, no data
    read) since it only needs pyarrow's parquet metadata. Returns and also writes a JSON report;
    only ever WARNS (never hard-fails) since a dtype difference in an EXCLUDED column, or a
    numeric-widening difference (int32 vs int64) is harmless — the report exists to let a human
    rule this in or out, not to auto-decide."""
    paths = {n: CLEANED_DATA_ROOT / f'{n}_cleaned.parquet' for n in names}
    present = {n: p for n, p in paths.items() if p.exists()}
    if len(present) < 2:
        log.warn(f'  dtype reconciliation skipped: need both years\' cleaned parquets on disk, '
                 f'found {list(present.keys())}')
        return {}
    schemas = {n: pq.read_schema(p) for n, p in present.items()}
    (n1, s1), (n2, s2) = list(schemas.items())[:2]
    dtypes1 = {f.name: str(f.type) for f in s1}
    dtypes2 = {f.name: str(f.type) for f in s2}
    shared = sorted(set(dtypes1) & set(dtypes2))
    mismatches = {c: {n1: dtypes1[c], n2: dtypes2[c]} for c in shared if dtypes1[c] != dtypes2[c]}
    report = {
        'columns_compared': len(shared),
        'columns_mismatched': len(mismatches),
        'mismatches': mismatches,
    }
    out_path = CLEANED_DATA_ROOT / 'dtype_reconciliation.json'
    out_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    if mismatches:
        log.warn(f'  dtype reconciliation: {len(mismatches)}/{len(shared)} shared columns have a '
                 f'DIFFERENT dtype between {n1} and {n2} — see {out_path.name}. A mismatched '
                 f'NOMINAL column would look like covariate shift downstream but is actually a '
                 f'cleaning-stage artifact; review before trusting that feature\'s C2ST-AUC.')
        for c, d in mismatches.items():
            log.warn(f'    {c}: {n1}={d[n1]}  {n2}={d[n2]}')
    else:
        log.ok(f'  dtype reconciliation: all {len(shared)} shared columns match dtype between '
               f'{n1} and {n2}')
    return report


# ── Text report ───────────────────────────────────────────────────────────────
def write_report(stats: dict, name: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    after_clean = stats['total_raw'] - stats['nan_removed'] - stats['inf_removed']
    after_dedup = after_clean - stats['dup_removed']
    lines = [
        f'STEP 1 REPORT  --  {name.upper()}',
        f'Generated : {datetime.now():%Y-%m-%d %H:%M:%S}',
        '=' * 60,
        '',
        '## Row Counts',
        f'  Total raw rows       : {stats["total_raw"]:>12,}',
        f'  NaN rows removed     : {stats["nan_removed"]:>12,}',
        f'  Inf rows removed     : {stats["inf_removed"]:>12,}',
        f'  After NaN/Inf clean  : {after_clean:>12,}',
        f'  Duplicates removed   : {stats["dup_removed"]:>12,}',
        f'  After global dedup   : {after_dedup:>12,}',
        f'  Final (post-labels)  : {stats["final_rows"]:>12,}',
        '',
        '## Per-File Stats',
        f'  {"File":<42} {"Raw":>12}  {"NaN":>9}  {"Inf":>9}  {"Clean":>12}',
        f'  {"-"*42} {"-"*12}  {"-"*9}  {"-"*9}  {"-"*12}',
    ]
    for fs in stats['file_stats']:
        lines.append(
            f'  {fs["file"]:<42} {fs["raw"]:>12,}  {fs["nan"]:>9,}  '
            f'{fs["inf"]:>9,}  {fs["clean"]:>12,}'
        )
    lines += [
        '',
        '## Raw Label Inventory (before consolidation)',
        f'  {"Label":<52} {"Count":>10}',
        f'  {"-"*52} {"-"*10}',
    ]
    for lbl, cnt in sorted(stats['raw_counts'].items(), key=lambda x: -x[1]):
        lines.append(f'  {lbl:<52} {cnt:>10,}')

    lines += [
        '',
        '## Canonical Families (after consolidation)',
        f'  {"Family":<22} {"Count":>10}',
        f'  {"-"*22} {"-"*10}',
    ]
    for lbl, cnt in sorted(stats['final_counts'].items(), key=lambda x: -x[1]):
        lines.append(f'  {lbl:<22} {cnt:>10,}')

    if stats.get('dropped_labels'):
        lines += ['', f'## Dropped (below {Config1.MIN_CLASS_ROWS}-row floor)']
        for lbl, cnt in stats['dropped_labels'].items():
            lines.append(f'  {lbl:<22} {cnt:>10,}  [DROPPED]')

    lines += [
        '',
        '## Output',
        f'  data/cc_data/{name}_cleaned.parquet',
    ]
    path.write_text('\n'.join(lines), encoding='utf-8')

# ── Visualizations ────────────────────────────────────────────────────────────
def _save(fig: plt.Figure, path: Path):
    """Save figure and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def _to_canonical(raw_counts: dict) -> dict:
    """Map raw label strings to canonical families, summing counts."""
    out = {}
    for lbl, cnt in raw_counts.items():
        fam = map_label(str(lbl))
        if fam:
            out[fam] = out.get(fam, 0) + cnt
    return out

def plot_label_stages_all(stats: dict, name: str, out_dir: Path):
    """Four bar charts — canonical family distribution at each pipeline stage."""
    stages = [
        (_to_canonical(stats['pre_clean_counts']), 'Stage 1 — Raw (before cleaning)', 'labels_stage1_raw.png'),
        (_to_canonical(stats['raw_counts']), 'Stage 2 — After NaN / Inf removal', 'labels_stage2_after_naninf.png'),
        (_to_canonical(stats['post_dedup_raw']), 'Stage 3 — After deduplication', 'labels_stage3_after_dedup.png'),
        (stats['final_counts'], 'Stage 4 — After label consolidation', 'labels_stage4_final.png'),
    ]

    for canonical, subtitle, fname in stages:
        if not canonical:
            continue
        items = sorted(canonical.items(), key=lambda x: -x[1])
        lbls = [it[0] for it in items]
        counts = [it[1] for it in items]
        total = sum(counts) or 1
        mx = max(counts)

        fig, ax = plt.subplots(figsize=(max(7, len(lbls) * 0.9), 5))
        ax.bar(range(len(lbls)), counts, color='#4C9BE8', edgecolor='white')
        ax.set_xticks(range(len(lbls)))
        ax.set_xticklabels(lbls, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Row count')
        ax.set_title(f'{name} — {subtitle}', fontsize=11, fontweight='bold')

        # Set scale, limits, and ticks BEFORE placing text (ensures correct coordinates)
        ax.set_yscale('log')
        ax.set_ylim(1, mx * 1.28)
        ax.set_yticks([1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))

        # Now place labels after scale is set
        for bar, cnt in zip(ax.patches, counts):
            pct = 100 * cnt / total
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + mx * 0.01,
                    f'{cnt:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=7)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        _save(fig, out_dir / fname)

def plot_row_pipeline(stats: dict, name: str, out_dir: Path):
    """Waterfall: rows at each pipeline stage with annotations (linear scale)."""
    raw = stats['total_raw']
    after_clean = raw - stats['nan_removed'] - stats['inf_removed']
    after_dedup = after_clean - stats['dup_removed']
    final = stats['final_rows']
    label_dropped = after_dedup - final

    stages = ['Raw', 'After\nNaN/Inf', 'After\nDedup', 'Final\n(post-labels)']
    counts = [raw, after_clean, after_dedup, final]
    drops = [stats['nan_removed'] + stats['inf_removed'], stats['dup_removed'], label_dropped]
    reasons = ['NaN / Inf', 'Duplicates', 'Label\ndrops']
    bar_colors = ['#90CAF9', '#4CAF50', '#FF8C00', '#4C9BE8']

    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = list(range(len(stages)))
    bars = ax.bar(x_pos, counts, color=bar_colors, width=0.55, edgecolor='white', linewidth=0)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + raw * 0.012,
                f'{count:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    for i, (rm, reason) in enumerate(zip(drops, reasons)):
        if rm == 0:
            continue
        mid_y = counts[i + 1] + (counts[i] - counts[i + 1]) * 0.5
        ax.annotate(f'−{rm:,}\n({reason})', xy=(i + 0.72, mid_y),
                    ha='left', va='center', fontsize=8, color='#333',
                    bbox=dict(boxstyle='round,pad=0.35', facecolor='#FFF9C4', alpha=0.92))

    ax.set_xticks(x_pos)
    ax.set_xticklabels(stages, fontsize=10)
    ax.set_ylabel('Row count')
    ax.set_title(f'{name} — rows remaining at each pipeline stage')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
    # Linear scale for row pipeline
    ax.set_ylim(0, raw * 1.20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    _save(fig, out_dir / 'row_pipeline.png')

def plot_labels_stages(raw_counts: dict, post_dedup_raw: dict, final_counts: dict,
                       dropped_labels: dict, name: str, out_dir: Path):
    """Two stacked bar charts: before and after floor filter."""
    records = []
    for lbl in sorted(raw_counts.keys(), key=lambda k: -raw_counts[k]):
        raw = raw_counts[lbl]
        post = post_dedup_raw.get(lbl, 0)
        dup_rm = raw - post
        canonical = map_label(lbl)

        if canonical is None:
            kept, floor_drop, unmapped = 0, 0, post
        elif canonical in dropped_labels:
            kept, floor_drop, unmapped = 0, post, 0
        else:
            kept, floor_drop, unmapped = post, 0, 0

        records.append({
            'lbl': lbl, 'kept': kept, 'dup_rm': dup_rm,
            'floor_drop': floor_drop, 'unmapped': unmapped,
            'canonical': canonical,
        })

    def _draw_stages(recs: list, title: str, filepath: Path) -> None:
        if not recs:
            return
        lbls = [r['lbl'] for r in recs]
        kept_v = [r['kept'] for r in recs]
        dup_v = [r['dup_rm'] for r in recs]
        floor_v = [r['floor_drop'] for r in recs]
        unmapped_v = [r['unmapped'] for r in recs]

        x = list(range(len(lbls)))
        bot1 = kept_v
        bot2 = [a + b for a, b in zip(kept_v, dup_v)]
        bot3 = [a + b for a, b in zip(bot2, floor_v)]

        fig, ax = plt.subplots(figsize=(max(10, len(lbls) * 0.85), 6))
        ax.bar(x, kept_v, color='#4C9BE8', edgecolor='white', label='Kept (final)')
        ax.bar(x, dup_v, color='#FF8C00', edgecolor='white', label='Dedup removed', bottom=bot1)
        ax.bar(x, floor_v, color='#E84C4C', edgecolor='white', label='Floor-dropped (<100 rows)', bottom=bot2)
        ax.bar(x, unmapped_v, color='#BDBDBD', edgecolor='white', label='Unmapped label', bottom=bot3)

        ax.set_xticks(x)
        ax.set_xticklabels(lbls, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Row count')
        ax.set_title(title, fontsize=11, fontweight='bold')

        # Logarithmic scale: 1, 10, 100, 1K, 10K, 100K, 1M, 10M
        ax.set_yscale('log')
        ax.set_yticks([1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))

        # Add total labels on top of each stacked bar
        totals = [k + d + f + u for k, d, f, u in zip(kept_v, dup_v, floor_v, unmapped_v)]
        for xi, total in zip(x, totals):
            if total > 0:
                ax.text(xi, total * 1.05, f'{int(total):,}', ha='center', va='bottom', fontsize=7)

        ax.legend(fontsize=8, loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        _save(fig, filepath)

    _draw_stages(records, f'{name} — label stages before floor filter (red = will be dropped)',
                 out_dir / 'labels_stages_prefilt.png')
    final_recs = [r for r in records
                  if r['canonical'] is not None and r['canonical'] not in dropped_labels]
    _draw_stages(final_recs, f'{name} — label stages after floor filter',
                 out_dir / 'labels_stages_final.png')

def plot_labels_after(final_counts: dict, name: str, out_dir: Path):
    """Final canonical family distribution with % annotations."""
    sorted_items = sorted(final_counts.items(), key=lambda x: -x[1])
    lbls = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]
    total = sum(counts) or 1

    fig, ax = plt.subplots(figsize=(max(7, len(lbls) * 0.9), 5))
    x = list(range(len(lbls)))
    bars = ax.bar(x, counts, color='#4C9BE8', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(lbls, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Row count')
    ax.set_title(f'{name} — canonical family distribution (after consolidation)')
    mx = max(counts) if counts else 1

    # Set scale, limits, and ticks BEFORE placing text (ensures correct coordinates)
    ax.set_yscale('log')
    ax.set_ylim(1, mx * 1.22)
    ax.set_yticks([1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))

    # Now place labels after scale is set
    for bar, count in zip(bars, counts):
        pct = 100 * count / total
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + mx * 0.01,
                f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=7)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    _save(fig, out_dir / 'labels_after.png')

def plot_quality_per_file(file_stats: list, dup_removed: int, name: str, out_dir: Path):
    """Stacked bar per file: clean/NaN/Inf/dedup rows (linear scale)."""
    labels = [s['file'] for s in file_stats] + ['[global\ndedup]']
    cleans = [s['clean'] for s in file_stats] + [0]
    nans = [s['nan'] for s in file_stats] + [0]
    infs = [s['inf'] for s in file_stats] + [0]
    dups = [0] * len(file_stats) + [dup_removed]

    bot2 = [c + n for c, n in zip(cleans, nans)]
    bot3 = [b + i for b, i in zip(bot2, infs)]

    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.4), 5))
    x = list(range(len(labels)))
    ax.bar(x, cleans, label='Clean rows', color='#4C9BE8')
    ax.bar(x, nans, label='NaN removed', color='#FF8C00', bottom=cleans)
    ax.bar(x, infs, label='Inf removed', color='#E84C4C', bottom=bot2)
    ax.bar(x, dups, label='Dup removed', color='#9C27B0', bottom=bot3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('Rows')
    ax.set_title(f'{name} — data quality per file (+ global dedup)')
    ax.legend(fontsize=8)

    # Add total labels on top of each stacked bar
    totals = [c + n + i + d for c, n, i, d in zip(cleans, nans, infs, dups)]
    mx = max(totals) if totals else 1
    for xi, total in zip(x, totals):
        if total > 0:
            ax.text(xi, total + mx * 0.01, f'{int(total):,}', ha='center', va='bottom', fontsize=7)

    # Linear scale for quality chart
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    _save(fig, out_dir / 'quality_per_file.png')

# ── Entry point ───────────────────────────────────────────────────────────────
def main(datasets_filter=None):
    selected = {
        k: v for k, v in DATASETS.items()
        if datasets_filter is None or k in datasets_filter
    }
    if not selected:
        print(f'No matching datasets. Available: {list(DATASETS.keys())}')
        return

    for name, folder in selected.items():
        if not folder.exists():
            print(f'[SKIP] {name}: folder not found at {folder}')
            continue

        res_dir = ensure_results_dir(1, name)
        log = Logger(res_dir / Config1.STEPS_FILE, step_prefix=1)
        log.section(f'STEP 1  --  LOAD / CLEAN / COMBINE  --  {name.upper()}')
        log.ok(f'Started: {datetime.now():%Y-%m-%d %H:%M:%S}')

        stats = process_dataset(name, folder, log)

        if not stats:
            log.warn('Processing returned nothing. Check errors above.')
            log.close()
            continue

        log.step('Generate visualizations (9 PNG charts)')
        plot_row_pipeline(stats, name, res_dir)
        plot_label_stages_all(stats, name, res_dir)
        plot_labels_stages(stats['raw_counts'], stats['post_dedup_raw'],
                           stats['final_counts'], stats['dropped_labels'], name, res_dir)
        plot_labels_after(stats['final_counts'], name, res_dir)
        plot_quality_per_file(stats['file_stats'], stats['dup_removed'], name, res_dir)
        log.ok('9 charts saved to {}'.format(res_dir))
        log.step_end()

        log.step('Write text report')
        write_report(stats, name, res_dir / Config1.RESULTS_FILE)
        log.ok(f'Report saved to {res_dir / Config1.RESULTS_FILE}')
        log.step_end()

        log.ok(f'Finished: {datetime.now():%Y-%m-%d %H:%M:%S}')
        log.close()
        print(f'\n[DONE] {name}: {stats["final_rows"]:,} rows  -->  '
              f'{CLEANED_DATA_ROOT / f"{name}_cleaned.parquet"}')

    # Cross-year dtype reconciliation: runs against whatever cleaned parquets
    # are CURRENTLY on disk for all known dataset names, not just the ones filtered this run —
    # so it still checks both years even if this invocation only (re)processed one of them.
    recon_log = Logger(CLEANED_DATA_ROOT / 'dtype_reconciliation_log.txt', step_prefix=1)
    check_cross_year_dtypes(list(DATASETS.keys()), recon_log)
    recon_log.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Step 1: Load, clean, combine, and label-consolidate CIC-IDS datasets.'
    )
    parser.add_argument('--datasets', nargs='+', metavar='NAME',
                        help='Datasets to process, e.g. --datasets cicids2017')
    args = parser.parse_args()
    main(datasets_filter=args.datasets)
