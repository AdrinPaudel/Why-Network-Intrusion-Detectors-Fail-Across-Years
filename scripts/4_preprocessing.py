"""
4_preprocessing.py — ML preprocessing: drop redundant features, scale, stratified split.

Purpose:
  Turn the CORRECTED cleaned CIC-IDS 2017/2018 CSVs into ML-ready, scaled, train/test
  Parquet splits. This is the bridge between the correlation analysis (scripts 2-3) and
  model training (script 5). It performs, per dataset, INDEPENDENTLY:
    1. Drop the redundant features identified in script 3 (output/3_.../drop_decisions.json).
    2. Encode the Label column into BOTH a binary (Benign=0 / Attack=1) and a multiclass
       target, using one CANONICAL label map shared across both years.
    3. Fit a StandardScaler (Z-score) on TRAIN ROWS ONLY (no test leakage), then apply it.
    4. Write train.parquet / test.parquet (+ scaler, label map, feature list, meta).

Why these choices (research grounding):
  - Separate per-year scaling, not combined:
      2017 and 2018 were captured on different networks/years with different traffic
      distributions. Fitting one scaler across both would force them onto a shared scale and
      erase year-specific structure. Each dataset is therefore standardised against its OWN
      training statistics. (Standard practice in the CIC-IDS ML literature.)
  - StandardScaler (Z-score), not Min-Max:
      Flow features are heavy-tailed with extreme outliers (e.g. Flow Duration, *Bytes/s).
      Min-Max squashes almost all mass into a tiny sub-range because a single huge value
      sets the max. Z-score is the robust default for tree/SVM/NN models on flow data.
  - Fit on TRAIN only:
      Fitting the scaler on all rows (train+test) leaks test-set distribution into training
      and inflates reported performance. The scaler sees training rows only; test is merely
      transformed with those frozen statistics.
  - Class imbalance handled later, not here:
      Benign heavily dominates (exact per-year P(benign) in the step-4 report). We do NOT
      oversample/undersample during preprocessing — that distorts priors and must live inside
      the CV fold during training.
      The split is stratified so every class keeps its proportion in train and test; the
      model script handles imbalance via class weights.

Scale of the data (drives the streaming design):
  cicids2018_cleaned.parquet (actual size and row count reported at run time) — it cannot be
  loaded into RAM. Everything
  here streams in two passes over each file:
    PASS 1 (fit) : stream, assign each row to train/test deterministically, and update the
                   streaming scaler on TRAIN rows only. Also tally per-class/per-split counts.
    PASS 2 (write): stream again (replaying the identical split), scale, and append to the
                    train/test Parquet writers.
  Two passes are required because leakage-free global scaling needs the full train mean/std
  before any row can be transformed.

Streaming stratified split (deterministic, replayable, order-independent):
  For each row we compute a per-class running index `gidx` (its position among rows of the
  same multiclass label, continued across chunks). A row goes to TEST iff
      ((gidx + phase[class]) % M) < round(test_size * M)        (M = 1000)
  This selects exactly ~test_size of EVERY class (exact stratification, +/-1 row per class),
  is fully deterministic, spreads the test set across the whole capture window (mitigates
  temporal bias vs a contiguous tail split), and -- crucially -- yields the IDENTICAL split
  when replayed in pass 2 because the file is read in the same order both times.

Compute backend (all CPU cores, numpy):
  - CSV parsing uses PyArrow's multithreaded reader (all cores).
  - The streaming scaler math runs on numpy.

Reads:
  data/cc_data/cicids2017_cleaned.parquet
  data/cc_data/cicids2018_cleaned.parquet
  output/3_correlation_comparison/drop_decisions.json   (features_to_drop)

Writes (per dataset <ds> in cicids2017, cicids2018):
  output/4_preprocessing/<ds>/train.parquet
  output/4_preprocessing/<ds>/test.parquet
  output/4_preprocessing/<ds>/scaler.json
  output/4_preprocessing/<ds>/label_mapping.json
  output/4_preprocessing/<ds>/feature_names.json
  output/4_preprocessing/<ds>/preprocessing_meta.json
  results/4_preprocessing/<ds>/steps.log
  results/4_preprocessing/<ds>/report.txt
  results/4_preprocessing/<ds>/class_distribution.png
  results/4_preprocessing/<ds>/scaling_check.png

Does NOT modify raw or cleaned CSVs. Does NOT touch script 0-3 outputs.
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unified_config import (
    PROJECT_ROOT, CLEANED_DATA_ROOT, Config4, Logger, DATASETS,
    ensure_results_dir, CANONICAL_MULTICLASS, BENIGN_LABEL
)

# ── Paths ──────────────────────────────────────────────────────────────────────
DROP_JSON = PROJECT_ROOT / 'output' / '3_correlation_comparison' / 'drop_decisions.json'
OUTPUT_DIR = Config4.OUTPUT_DIR
RESULTS_DIR = Config4.RESULTS_DIR


# ── Compute backend ──────────────────────────────────────────────────────────
BACKEND_LABEL = 'cpu(numpy)'


# ── Streaming Z-score scaler (numerically stable) ───────────────────────────────
class StreamingScaler:
    """Chan/Welford parallel-combine mean & variance accumulated over streamed chunks.

    Equivalent to sklearn.StandardScaler with with_mean=with_std=True (population variance,
    ddof=0), but computed incrementally so a 35 GB file never lands fully in memory. Zero-
    variance features get scale=1 (sklearn's handle_zeros behaviour) so transform never
    divides by zero.
    """

    def __init__(self, n_features: int):
        self._n  = 0
        self._mean = np.zeros(n_features, dtype=np.float64)
        self._M2   = np.zeros(n_features, dtype=np.float64)

    def partial_fit(self, X) -> None:
        """Update running statistics with a chunk X (rows x n_features)."""
        nb = X.shape[0]
        if nb == 0:
            return
        mb  = X.mean(axis=0)
        M2b = ((X - mb) ** 2).sum(axis=0)
        if self._n == 0:
            self._mean = mb
            self._M2   = M2b
            self._n    = nb
            return
        delta = mb - self._mean
        tot   = self._n + nb
        self._mean = self._mean + delta * (nb / tot)
        self._M2   = self._M2 + M2b + (delta ** 2) * (self._n * nb / tot)
        self._n    = tot

    @property
    def n_samples(self) -> int:
        return self._n

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, scale) numpy arrays; scale = std with zeros mapped to 1.0."""
        var = self._M2 / max(self._n, 1)
        scale = np.sqrt(var)
        scale = np.where(scale == 0, 1.0, scale)
        return self._mean.astype(np.float64), scale.astype(np.float64)


# ── Deterministic streaming stratified split ───────────────────────────────────
class StratifiedSplitter:
    """Assigns each row to train/test, stratified per multiclass label, deterministically.

    A row is TEST iff ((gidx + phase[class]) % M) < thresh, where gidx is the row's running
    position among rows of its class. Exactly ~test_size of every class lands in test, the
    selection is reproducible across passes (same file order), and the test set is spread
    across the whole file rather than concentrated in a tail.
    """

    def __init__(self, test_size: float, seed: int, modulus: int = Config4.SPLIT_MODULUS):
        self._M      = modulus
        self._thresh = max(1, min(modulus - 1, round(test_size * modulus)))
        # Per-class phase scatters each class's test residues so classes don't align.
        self._phase  = {c: (c * Config4.PHASE_HASH_MULTIPLIER + seed) % modulus for c in CANONICAL_MULTICLASS.values()}
        self.realized_test_fraction = self._thresh / modulus
        self._offsets: dict[int, int] = defaultdict(int)

    def reset(self) -> None:
        """Clear running per-class offsets so pass 2 replays pass 1 exactly."""
        self._offsets = defaultdict(int)

    def assign(self, y_multi: np.ndarray) -> np.ndarray:
        """Return a boolean is_test mask for a chunk of multiclass label ints."""
        s = pd.Series(y_multi)
        within = s.groupby(s, sort=False).cumcount().to_numpy()         # 0-based within-chunk
        base   = np.fromiter((self._offsets[c] for c in y_multi), dtype=np.int64, count=len(y_multi))
        gidx   = base + within
        phase  = np.fromiter((self._phase[c] for c in y_multi), dtype=np.int64, count=len(y_multi))
        is_test = ((gidx + phase) % self._M) < self._thresh
        for c, cnt in s.value_counts().items():                          # advance offsets
            self._offsets[int(c)] += int(cnt)
        return is_test


# ── Header / feature selection ─────────────────────────────────────────────────
def resolve_features(parquet_path: Path, drop_set: set[str], log: Logger) -> list[str]:
    """Feature columns = all Parquet columns minus identifiers, Label, and script-3 drops."""
    pf = pq.ParquetFile(parquet_path)
    header = [name.strip() for name in pf.schema.names]
    feats = [
        c for c in header
        if c.lower() not in Config4.IDENTIFIER_COLS and c not in drop_set
    ]
    missing_drops = sorted(drop_set - set(header))
    if missing_drops:
        log.warn(f'drop_decisions features not in {parquet_path.name}: {missing_drops}')
    log.info(f'{parquet_path.name}: {len(header)} cols → {len(feats)} features '
             f'(excluded {len(header) - len(feats)}: identifiers + Label + {len(drop_set)} drops)')
    return feats


# ── Parquet streaming ──────────────────────────────────────────────────────────
def open_stream(parquet_path: Path, features: list[str], chunk_rows: int):
    """Open a PyArrow Parquet reader yielding RecordBatches.

    Reads Parquet file with only feature columns + Label. Yields batches via row groups
    for streaming processing without loading full file into memory.
    """
    parquet_file = pq.ParquetFile(parquet_path)
    cols_to_read = features + ['Label']

    def batch_generator():
        """Yield batches from row groups."""
        for i in range(parquet_file.num_row_groups):
            batch = parquet_file.read_row_group(i, columns=cols_to_read)
            yield batch

    return batch_generator()


def batch_to_arrays(batch, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """RecordBatch → (X float64 [rows x n_features], raw label string array)."""
    cols = [batch.column(f).to_numpy(zero_copy_only=False) for f in features]
    X = np.column_stack(cols).astype(np.float64, copy=False)
    labels = batch.column('Label').to_numpy(zero_copy_only=False)
    return X, labels


def encode_labels(
    labels: np.ndarray, csv_name: str, need_binary: bool = True
) -> tuple[np.ndarray | None, np.ndarray]:
    """Map raw label strings → (binary int8 or None, multiclass int8). Fail fast on unknown labels."""
    s = pd.Series(labels, dtype='object')
    unknown = sorted(set(s.unique()) - set(CANONICAL_MULTICLASS))
    if unknown:
        raise ValueError(
            f'{csv_name}: labels not in canonical map: {unknown}. '
            f'Known: {sorted(CANONICAL_MULTICLASS)}'
        )
    y_multi = s.map(CANONICAL_MULTICLASS).to_numpy(dtype=np.int8)
    y_bin   = (s.to_numpy() != BENIGN_LABEL).astype(np.int8) if need_binary else None
    return y_bin, y_multi


def assert_finite(X: np.ndarray, features: list[str], csv_name: str, batch_no: int) -> None:
    """Fail fast on NaN/Inf — script 1 guarantees clean data, so this should never trigger."""
    if np.isfinite(X).all():
        return
    bad_cols = [features[j] for j in range(X.shape[1]) if not np.isfinite(X[:, j]).all()]
    n_bad = int((~np.isfinite(X)).sum())
    raise ValueError(
        f'{csv_name}: {n_bad} non-finite value(s) in batch {batch_no}, columns {bad_cols}. '
        f'Re-run 1_load_clean_combine.py — preprocessing expects finite cleaned data.'
    )


# ── Pass 1: fit scaler on train rows + tally counts ────────────────────────────
def fit_pass(
    parquet_path: Path,
    features: list[str],
    splitter: StratifiedSplitter,
    chunk_rows: int,
    log: Logger,
) -> tuple[StreamingScaler, dict, dict, int]:
    """Stream once. Update scaler on TRAIN rows only; count classes per split."""
    scaler = StreamingScaler(len(features))
    # counts[split][multiclass_int] = n ; also track binary via class 0 vs rest
    counts = {'train': defaultdict(int), 'test': defaultdict(int)}
    splitter.reset()
    reader = open_stream(parquet_path, features, chunk_rows)
    total_rows = 0
    batch_no = 0
    t0 = time.time()

    for batch in reader:
        batch_no += 1
        X, labels = batch_to_arrays(batch, features)
        assert_finite(X, features, parquet_path.name, batch_no)
        _, y_multi = encode_labels(labels, parquet_path.name, need_binary=False)
        is_test = splitter.assign(y_multi)
        total_rows += X.shape[0]

        train_mask = ~is_test
        if train_mask.any():
            scaler.partial_fit(X[train_mask])

        for c, cnt in pd.Series(y_multi[is_test]).value_counts().items():
            counts['test'][int(c)] += int(cnt)
        for c, cnt in pd.Series(y_multi[train_mask]).value_counts().items():
            counts['train'][int(c)] += int(cnt)

        if batch_no % Config4.LOG_INTERVAL == 0:
            rate = total_rows / max(time.time() - t0, 1e-9)
            log.info(f'batch {batch_no}: {total_rows:,} rows ({rate:,.0f} rows/s)')

    log.info(f'  [fit] done: {batch_no} batches, {total_rows:,} rows, '
             f'{time.time() - t0:.1f}s, scaler n_train={scaler.n_samples:,}')
    return scaler, dict(counts['train']), dict(counts['test']), total_rows


# ── Pass 2: scale + write train/test Parquet ───────────────────────────────────
def write_pass(
    parquet_path: Path,
    features: list[str],
    splitter: StratifiedSplitter,
    mean: np.ndarray,
    scale: np.ndarray,
    out_dir: Path,
    compression: str,
    chunk_rows: int,
    log: Logger,
) -> dict:
    """Stream again, replay the split, Z-score with frozen (mean, scale), append to Parquet.

    Also accumulates post-scaling mean/std on TRAIN rows so the report can verify the scaler
    actually produced ~0 mean / ~1 std (a cheap correctness check).
    """
    schema = pa.schema(
        [pa.field(f, pa.float32()) for f in features]
        + [pa.field('label_binary', pa.int8()), pa.field('label_multiclass', pa.int8())]
    )
    train_path = out_dir / 'train.parquet'
    test_path  = out_dir / 'test.parquet'
    out_dir.mkdir(parents=True, exist_ok=True)

    writers = {
        'train': pq.ParquetWriter(train_path, schema, compression=compression),
        'test':  pq.ParquetWriter(test_path,  schema, compression=compression),
    }

    # streaming verification of scaled TRAIN distribution
    verify = StreamingScaler(len(features))
    mean_f  = mean.astype(np.float64)
    scale_f = scale.astype(np.float64)
    splitter.reset()
    reader = open_stream(parquet_path, features, chunk_rows)
    written = {'train': 0, 'test': 0}
    batch_no = 0
    t0 = time.time()
    out_dtype = np.float32 if Config4.OUTPUT_FLOAT_DTYPE == 'float32' else np.float64

    try:
        for batch in reader:
            batch_no += 1
            X, labels = batch_to_arrays(batch, features)
            y_bin, y_multi = encode_labels(labels, parquet_path.name)
            is_test = splitter.assign(y_multi)

            Xs = ((X - mean_f) / scale_f).astype(out_dtype)

            for split, mask in (('train', ~is_test), ('test', is_test)):
                if not mask.any():
                    continue
                Xm = Xs[mask]
                arrays = [pa.array(Xm[:, j]) for j in range(len(features))]
                arrays.append(pa.array(y_bin[mask]))
                arrays.append(pa.array(y_multi[mask]))
                writers[split].write_table(pa.Table.from_arrays(arrays, schema=schema))
                written[split] += int(mask.sum())

            if (~is_test).any():
                verify.partial_fit(Xs[~is_test].astype(np.float64))

            if batch_no % Config4.LOG_INTERVAL == 0:
                rate = (written['train'] + written['test']) / max(time.time() - t0, 1e-9)
                log.info(f'batch {batch_no}: train={written["train"]:,} test={written["test"]:,} ({rate:,.0f} rows/s)')
    finally:
        writers['train'].close()
        writers['test'].close()

    v_mean, v_std = verify.finalize()
    return {
        'written': written,
        'train_path': str(train_path),
        'test_path': str(test_path),
        'verify_mean_abs_max': float(np.abs(v_mean).max()),
        'verify_std_min': float(v_std.min()),
        'verify_std_max': float(v_std.max()),
    }


# ── Persisted artifacts ────────────────────────────────────────────────────────
def save_scaler(features, mean, scale, n_train, out_path: Path, log: Logger):
    doc = {
        'method': 'standard_zscore',
        'fitted_on': 'train_rows_only',
        'with_mean': True, 'with_std': True, 'ddof': 0,
        'n_train_samples': int(n_train),
        'features': features,
        'mean':  [round(float(x), 8) for x in mean],
        'scale': [round(float(x), 8) for x in scale],
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(doc, f, indent=2)
    log.ok(f'Saved {out_path.name}')


def save_label_mapping(out_path: Path, log: Logger,
                       train_counts: dict | None = None,
                       test_counts: dict | None = None):
    # Build an absent-classes note from actual label counts so the JSON reflects
    # what is truly in this dataset rather than a hardcoded assumption.
    absent_note = ''
    if train_counts is not None and test_counts is not None:
        combined = {
            lbl: train_counts.get(lbl, 0) + test_counts.get(lbl, 0)
            for lbl in set(train_counts) | set(test_counts)
        }
        absent = sorted(
            lbl for lbl in CANONICAL_MULTICLASS
            if lbl != BENIGN_LABEL and combined.get(lbl, 0) == 0
        )
        if absent:
            absent_note = f' Absent from this dataset (0 rows): {", ".join(absent)}.'
    doc = {
        'binary': {'Benign': 0, 'Attack': 1, 'rule': 'Benign=0, every other label=1'},
        'multiclass': CANONICAL_MULTICLASS,
        'multiclass_inverse': {v: k for k, v in CANONICAL_MULTICLASS.items()},
        'note': ('Canonical map is the union of classes across BOTH years so encodings match '
                 f'between datasets.{absent_note}'),
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(doc, f, indent=2)
    log.ok(f'Saved {out_path.name}')


def save_feature_names(features, drop_set, out_path: Path, log: Logger):
    doc = {
        'n_features': len(features),
        'features': features,
        'dropped_redundant': sorted(drop_set),
        'excluded_identifiers': sorted(Config4.IDENTIFIER_COLS),
        'target_columns': ['label_binary', 'label_multiclass'],
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(doc, f, indent=2)
    log.ok(f'Saved {out_path.name}')


def save_meta(ds, features, train_counts, test_counts, total_rows, split_info,
              test_size, realized_frac, seed, backend, out_path: Path, log: Logger):
    inv = {v: k for k, v in CANONICAL_MULTICLASS.items()}
    def named(counts):
        return {inv[c]: int(n) for c, n in sorted(counts.items())}
    n_train = sum(train_counts.values())
    n_test  = sum(test_counts.values())
    doc = {
        'dataset': ds,
        'generated': datetime.now().isoformat(timespec='seconds'),
        'backend': backend,
        'n_features': len(features),
        'rows_total': int(total_rows),
        'rows_train': int(n_train),
        'rows_test':  int(n_test),
        'requested_test_size': test_size,
        'realized_test_fraction': round(realized_frac, 6),
        'seed': seed,
        'split': 'deterministic per-class systematic (stratified, replayable)',
        'scaling': 'StandardScaler (z-score) fit on train rows only, applied to train+test',
        'class_counts_train': named(train_counts),
        'class_counts_test':  named(test_counts),
        'verify': {k: split_info[k] for k in
                   ('verify_mean_abs_max', 'verify_std_min', 'verify_std_max')},
        'outputs': {'train': split_info['train_path'], 'test': split_info['test_path']},
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(doc, f, indent=2)
    log.ok(f'Saved {out_path.name}  (train={n_train:,}  test={n_test:,})')
    return doc


# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_class_distribution(ds, train_counts, test_counts, out_path: Path, log: Logger):
    inv = {v: k for k, v in CANONICAL_MULTICLASS.items()}
    classes = sorted(set(train_counts) | set(test_counts))
    names = [inv[c] for c in classes]
    tr = [train_counts.get(c, 0) for c in classes]
    te = [test_counts.get(c, 0) for c in classes]
    x = np.arange(len(classes))
    w = 0.4
    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 1.1), 5))
    ax.bar(x - w / 2, tr, w, label='train', color='#3498db')
    ax.bar(x + w / 2, te, w, label='test',  color='#e67e22')
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('rows (log scale)')
    ax.set_title(f'{ds} — class distribution per split (stratified)', fontsize=10)
    ax.legend()
    for xi, (a, b) in enumerate(zip(tr, te)):
        ax.text(xi - w / 2, a, f'{a:,}', ha='center', va='bottom', fontsize=6, rotation=90)
        ax.text(xi + w / 2, b, f'{b:,}', ha='center', va='bottom', fontsize=6, rotation=90)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.ok(f'Saved {out_path.name}')


def plot_scaling_check(ds, split_info, out_path: Path, log: Logger):
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    mam = split_info['verify_mean_abs_max']
    smin, smax = split_info['verify_std_min'], split_info['verify_std_max']
    ax.axhline(0.0, color='#2ecc71', lw=1, ls='--', label='ideal mean = 0')
    ax.axhline(1.0, color='#9b59b6', lw=1, ls='--', label='ideal std = 1')
    ax.scatter([0], [mam], color='#2ecc71', s=60, zorder=3,
               label=f'max |mean| = {mam:.2e}')
    ax.errorbar([1], [(smin + smax) / 2], yerr=[[(smin + smax) / 2 - smin], [smax - (smin + smax) / 2]],
                fmt='o', color='#9b59b6', capsize=5, zorder=3,
                label=f'std range [{smin:.3f}, {smax:.3f}]')
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['scaled-train mean', 'scaled-train std'])
    ax.set_title(f'{ds} — scaling verification (train rows after Z-score)', fontsize=10)
    ax.legend(fontsize=7, loc='center right')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.ok(f'Saved {out_path.name}')


# ── Report ─────────────────────────────────────────────────────────────────────
def write_report(metas: list[dict], features, drop_set, backend, out_path: Path, log: Logger):
    """Write brief summary report (key numbers only, no methodology)."""
    lines: list[str] = []

    lines.append('4_PREPROCESSING REPORT')
    lines.append(f'Generated: {datetime.now():%Y-%m-%d %H:%M:%S}')
    lines.append(f'Backend: {backend}')
    lines.append('')
    lines.append(f'Features: {len(features)} kept, {len(drop_set)} dropped')
    lines.append('')

    for m in metas:
        lines.append(f'{m["dataset"]}:')
        lines.append(f'  rows total: {m["rows_total"]:,}')
        lines.append(f'  train: {m["rows_train"]:,}  test: {m["rows_test"]:,}')
        cc = m.get('class_counts_train', {})
        if cc:
            total_cc = sum(cc.values()) or 1
            n_ben = cc.get('Benign', 0)
            lines.append(f'  P(benign) in train: {n_ben / total_cc:.1%}  ({n_ben:,} benign / {total_cc:,} total)')
        v = m['verify']
        lines.append(f'  scaling: mean={v["verify_mean_abs_max"]:.2e}, std=[{v["verify_std_min"]:.4f}, {v["verify_std_max"]:.4f}]')
        lines.append('')

    text = '\n'.join(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text)
    log.ok(f'Saved {out_path.name}')


# ── Per-dataset driver ─────────────────────────────────────────────────────────
def process_dataset(ds, drop_set, backend, test_size, seed, chunk_rows,
                    compression, log: Logger) -> tuple[dict, list[str]]:
    parquet_path = CLEANED_DATA_ROOT / f'{ds}_cleaned.parquet'
    if not parquet_path.exists():
        log.warn(f'Missing {parquet_path}')
        sys.exit(1)

    out_dir = OUTPUT_DIR / ds
    out_dir.mkdir(parents=True, exist_ok=True)

    features = resolve_features(parquet_path, drop_set, log)
    splitter = StratifiedSplitter(test_size, seed)

    log.step('PASS 1 — fit scaler (train rows only)')
    scaler, train_counts, test_counts, total_rows = fit_pass(
        parquet_path, features, splitter, chunk_rows, log)
    mean, scale = scaler.finalize()
    log.ok('scaler fitted')
    log.step_end()

    log.step('PASS 2 — scale + write train/test Parquet')
    split_info = write_pass(parquet_path, features, splitter, mean, scale,
                            out_dir, compression, chunk_rows, log)
    log.ok(f'train={split_info["written"]["train"]:,} test={split_info["written"]["test"]:,}')
    log.step_end()

    log.step('Save metadata + visualizations')
    meta = save_meta(ds, features, train_counts, test_counts, total_rows, split_info,
                     test_size, splitter.realized_test_fraction, seed, backend,
                     out_dir / 'preprocessing_meta.json', log)
    save_feature_names(features, drop_set, out_dir / 'feature_names.json', log)
    save_label_mapping(out_dir / 'label_mapping.json', log, train_counts, test_counts)
    save_scaler(features, mean, scale, scaler.n_samples, out_dir / 'scaler.json', log)
    results_ds_dir = RESULTS_DIR / ds
    results_ds_dir.mkdir(parents=True, exist_ok=True)
    plot_class_distribution(ds, train_counts, test_counts,
                            results_ds_dir / 'class_distribution.png', log)
    plot_scaling_check(ds, split_info, results_ds_dir / 'scaling_check.png', log)
    log.ok('complete')
    log.step_end()
    return meta, features


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # main.py's DATASET_STEPS includes step 4 and passes --datasets to it like
    # every other per-dataset step, but this file previously had no argparse at all, so that
    # filter was silently ignored (both datasets always ran regardless). Config4 itself still has
    # no CLI-exposed TUNING parameters — only --datasets is new.
    parser = argparse.ArgumentParser(
        description='Step 4: ML preprocessing (drop redundant features, scale, stratified split).')
    parser.add_argument('--datasets', nargs='+', metavar='NAME', default=list(DATASETS),
                        help=f'Datasets to process (default: {" ".join(DATASETS)})')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    backend = BACKEND_LABEL
    compression = None if Config4.COMPRESSION == 'none' else Config4.COMPRESSION

    # Load drop decisions once
    if not DROP_JSON.exists():
        print(f'Missing {DROP_JSON} — run script 3 first')
        sys.exit(1)
    with open(DROP_JSON, encoding='utf-8') as f:
        drop_set = set(json.load(f).get('features_to_drop', []))

    # Process each dataset
    for ds in args.datasets:
        ds_results_dir = ensure_results_dir(4, ds)
        ds_log = Logger(ds_results_dir / Config4.STEPS_FILE, step_prefix=4)
        ds_log.info(f'backend={backend}, compression={Config4.COMPRESSION}')

        meta, features = process_dataset(ds, drop_set, backend, Config4.TEST_SIZE,
                                        Config4.SEED, Config4.CHUNK_ROWS, compression, ds_log)

        write_report([meta], features, drop_set, backend,
                    ds_results_dir / Config4.RESULTS_FILE, ds_log)

        ds_log.ok(f'{ds} complete: train={meta["rows_train"]:,}, test={meta["rows_test"]:,}')
        ds_log.close()


if __name__ == '__main__':
    main()
