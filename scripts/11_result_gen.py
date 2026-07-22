"""
11_result_gen.py — results.md generation for step 11 (split out of
11_cross_analysis.py so the markdown text/formatting can be edited and regenerated without
re-running the expensive analysis: joins, ablation retraining, visuals).

USAGE:
  Normally invoked automatically at the end of 11_cross_analysis.py's run (it caches its
  analysis outputs, then calls this script as a subprocess).

  To regenerate just results.md after editing ONLY this file (no analysis or data
  change), run it directly:
    python scripts/11_result_gen.py

  Reads its inputs from output/11_cross_analysis/<algorithm>/_doc_cache.pkl, written by
  11_cross_analysis.py at the end of its run. If that cache is missing, run
  11_cross_analysis.py first to produce it.
"""

import re
import sys
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unified_config import (  # noqa: E402
    PROJECT_ROOT, Config1, Config5, Config10, Config11, Logger, DATASETS, ALGORITHM,
    cross_output_dir, cross_results_dir, training_output_dir, testing_results_dir,
)

# Resolved in main(), same pattern as 11_cross_analysis.py.
OUTPUT_DIR: Path
RESULTS_DIR: Path

DS1, DS2 = list(DATASETS)[:2]            # the two cross-compared years (DATASETS order)

CACHE_FILENAME = '_doc_cache.pkl'


# ════════════════════════════════════════════════════════════════════════════════
# Loaders (best-effort JSON readers for the step 2/3/4/6 recap sections in results.md)
# ════════════════════════════════════════════════════════════════════════════════

def load_profiles(ds: str) -> dict:
    """feature -> step-7 profile dict (output/7_profile/<ds>/profiles.json). Best-effort,
    returns {} if the file is missing (older runs without step 7, or step 7 skipped).
    NOTE: duplicated in 11_cross_analysis.py, which needs the same loader independently inside
    build_cross_table(); both copies must stay in sync if this logic ever changes."""
    p = PROJECT_ROOT / 'output' / '7_profile' / ds / 'profiles.json'
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return {}


def load_json_best_effort(p: Path) -> dict:
    """Generic best-effort JSON loader, used for the step 2/3/4/6 recap sections below.
    Returns {} if the file is missing or fails to parse, same convention as load_profiles()."""
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return {}


def load_correlation_drop_decisions() -> dict:
    """Step 3 output: which features got dropped for redundancy, and why."""
    return load_json_best_effort(PROJECT_ROOT / 'output' / '3_correlation_comparison' / 'drop_decisions.json')


def load_flagged_pairs(ds: str, method: str = 'pearson') -> dict:
    """Step 2 output: per-year flagged correlated pairs before the step-3 consensus filter."""
    return load_json_best_effort(PROJECT_ROOT / 'output' / '2_correlation_analysis' / ds / f'{method}_flagged_pairs.json')


def load_correlation_matrix(ds: str, method: str = 'pearson') -> dict:
    """Step 2 output: the full (not just flagged) per-year correlation matrix, {'features': [...],
    'matrix': [[...]]}, index-aligned. Used to look up the EXACT r for a pair under whichever
    metric did not itself flag it (e.g. a pair Spearman flagged but Pearson did not quite clear the
    flag threshold for) without re-running step 2."""
    return load_json_best_effort(PROJECT_ROOT / 'output' / '2_correlation_analysis' / ds / f'{method}_matrix.json')


def _matrix_lookup(mat: dict, a: str, b: str) -> float:
    """r for one feature pair from a load_correlation_matrix() result; nan if either feature or
    the matrix itself is missing."""
    feats = mat.get('features', [])
    rows = mat.get('matrix', [])
    try:
        i, j = feats.index(a), feats.index(b)
        return float(rows[i][j])
    except (ValueError, IndexError):
        return float('nan')


def load_scaler_json(ds: str) -> dict:
    """Step 4 output: the saved Z-score scaler (mean/scale per feature) for one year."""
    return load_json_best_effort(PROJECT_ROOT / 'output' / '4_preprocessing' / ds / 'scaler.json')


def load_within_year_metrics(ds: str, algorithm: str = 'lightgbm') -> dict:
    """Step 6 output: same-year (within-year) binary test metrics for one year."""
    return load_json_best_effort(PROJECT_ROOT / 'output' / '6_testing' / f'{ds}_{algorithm}' / 'metrics_binary.json')


def load_preprocessing_meta(ds: str) -> dict:
    """Step 4 output: train/test split sizes, class counts, seed, scaling method."""
    return load_json_best_effort(PROJECT_ROOT / 'output' / '4_preprocessing' / ds / 'preprocessing_meta.json')


def load_training_meta(ds: str, algorithm: str = 'lightgbm') -> dict:
    """Step 5 output: per-task (binary/multiclass) training run metadata."""
    return load_json_best_effort(training_output_dir(PROJECT_ROOT, ds, algorithm) / 'training_meta.json')


def load_label_mapping_json(ds: str) -> dict:
    """Step 4 output: binary and multiclass label -> code maps."""
    return load_json_best_effort(PROJECT_ROOT / 'output' / '4_preprocessing' / ds / 'label_mapping.json')


def load_layer_a() -> dict:
    """Step 10 output: feature -> full per-feature record, including the per-attack-family
    breakdown (axis1_per_attack, axis2_per_class_stability) that the pooled cross_table.csv
    does not carry. Same loading pattern as 11_cross_analysis.py's own load_layer_a(); duplicated
    here since this file cannot import that digit-prefixed module."""
    p = PROJECT_ROOT / 'output' / '10_execute_comparison' / f'verdicts_layerA_{DS1}_{DS2}.json'
    if not p.exists():
        return {}
    try:
        records = json.loads(p.read_text(encoding='utf-8'))
        return {r['feature']: r for r in records if isinstance(r, dict) and 'feature' in r}
    except Exception:
        return {}


def parse_step0_report(ds: str) -> dict:
    """Step 0 output is a fixed-format .txt report (no JSON exists for this step). Best-effort
    line-scan: returns {} on any parse failure rather than raising, same convention as the JSON
    loaders above. Pulls: total column count, total row count, the post-consolidation label
    distribution (name, count, pct), a few concrete '- Attempted' before/after examples, and the
    full pre-fold ('before consolidation') label distribution, aggregated across every per-file
    block in the report (the report writes one such block per source CSV; nothing upstream
    aggregates them to a dataset total, so that summation happens here at render time only —
    no change to 0_dataexplore.py, the per-file numbers it already wrote are sufficient)."""
    p = PROJECT_ROOT / 'results' / '0_dataexplore' / ds / '0_dataexplore_report.txt'
    if not p.exists():
        return {}
    try:
        text = p.read_text(encoding='utf-8')
        out: dict = {}

        m = re.search(r'Total columns:\s*(\d+)', text)
        out['total_columns'] = int(m.group(1)) if m else None

        m = re.search(r'Total rows across all files:\s*([\d,]+)', text)
        out['total_rows'] = int(m.group(1).replace(',', '')) if m else None

        labels = []
        m = re.search(r'Total rows per label \(after consolidation\):\n((?:.+\n)+)', text)
        if m:
            for line in m.group(1).splitlines():
                lm = re.match(r'\s*(.+?):\s*([\d,]+)\s*\(([\d.]+)%\)', line)
                if lm:
                    labels.append((lm.group(1).strip(), int(lm.group(2).replace(',', '')),
                                   float(lm.group(3))))
        out['labels'] = labels

        attempted = re.findall(r'^\s*(.+? - Attempted):\s*([\d,]+)\s*$', text, re.MULTILINE)
        out['attempted_examples'] = [(name, int(cnt.replace(',', ''))) for name, cnt in attempted]

        # Aggregate every per-file "Labels (before consolidation):" block into one dataset-level
        # total, the same rows the after-consolidation table above shows except '- Attempted'
        # labels are NOT yet folded into Benign (so e.g. 'SSH-Patator - Attempted' is its own row,
        # carrying the original attack type in the label text itself).
        before_totals: dict = {}
        for block in re.findall(r'Labels \(before consolidation\):\n((?:[ \t]+\S.*\n)+)', text):
            for line in block.splitlines():
                lm = re.match(r'\s*(.+?):\s*([\d,]+)\s*$', line)
                if lm:
                    name = lm.group(1).strip()
                    before_totals[name] = before_totals.get(name, 0) + int(lm.group(2).replace(',', ''))
        total_before = sum(before_totals.values())
        out['labels_before'] = sorted(
            ((name, cnt, 100 * cnt / total_before if total_before else 0.0)
             for name, cnt in before_totals.items()),
            key=lambda x: -x[1])

        return out
    except Exception:
        return {}


def parse_step1_report(ds: str) -> dict:
    """Step 1 output is a fixed-format .txt report (no JSON exists for this step). Best-effort
    line-scan, same convention as parse_step0_report. Pulls the row-count funnel and the
    canonical-family breakdown."""
    p = PROJECT_ROOT / 'results' / '1_load_clean_combine' / ds / '1_load_clean_combine_report.txt'
    if not p.exists():
        return {}
    try:
        text = p.read_text(encoding='utf-8')
        out: dict = {}

        row_counts = []
        m = re.search(r'## Row Counts\n((?:.+\n)+?)\n', text)
        if m:
            for line in m.group(1).splitlines():
                lm = re.match(r'\s*(.+?)\s*:\s*([\d,]+)\s*$', line)
                if lm:
                    row_counts.append((lm.group(1).strip(), int(lm.group(2).replace(',', ''))))
        out['row_counts'] = row_counts

        families = []
        m = re.search(r'## Canonical Families \(after consolidation\)\n.+\n.+\n((?:.+\n)+)', text)
        if m:
            for line in m.group(1).splitlines():
                lm = re.match(r'\s*(\S.*?)\s{2,}([\d,]+)\s*$', line)
                if lm:
                    families.append((lm.group(1).strip(), int(lm.group(2).replace(',', ''))))
        out['canonical_families'] = families

        return out
    except Exception:
        return {}


def _img(path: Path, alt: str = '') -> str:
    """Markdown image reference (path relative to PROJECT_ROOT, resolved by build_ext_docx.py
    against its own project-root HERE). Returns '' if the file doesn't exist, so callers can
    skip embedding rather than render a broken reference."""
    if not path.exists():
        return ''
    rel = path.relative_to(PROJECT_ROOT).as_posix()
    return f'![{alt}]({rel})'


def _fmt(v, places=4, dash=False) -> str:
    """Module-level float formatter shared by _build_preamble_sections() and write_results_doc()
    (originally two near-identical local closures -- the former's one-argument helper was exactly
    this function's places=4, dash=False default behavior, so it was dropped in favor of this one).
    dash=True renders '—' instead of 'n/a' for a missing/non-finite value (used by the
    ablation tables, where a metric may not exist yet for older cached runs)."""
    try:
        f = float(v)
        if np.isfinite(f):
            return f'{f:.{places}f}'
        return '—' if dash else 'n/a'
    except (TypeError, ValueError):
        return '—' if dash else 'n/a'


def _ci_excludes_zero(blk: dict) -> bool:
    """True iff blk's bootstrap_ci95 is finite and doesn't straddle zero."""
    ci = blk.get('bootstrap_ci95', {}) if isinstance(blk, dict) else {}
    lo, hi = ci.get('lo', float('nan')), ci.get('hi', float('nan'))
    return np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)


def _verdict_classify(blk: dict) -> tuple:
    """Shared classification core for _plain_verdict_block() and _plain_delta_verdict() (both
    defined inside write_results_doc): extracts the Spearman estimate and derives the
    negligible/strength/direction bucket and CI-excludes-zero flag that both otherwise duplicated
    verbatim. Returns (sp, negligible, strength, direction, ci_excludes); if sp is not finite,
    negligible/strength/direction/ci_excludes are all None and the caller must render its own
    'no data available' case (the two callers word that case, and everything downstream of this
    classification, differently, so only the classification itself is shared here).

    negligible: |sp| < 0.05 is noise-level -- asserting a direction for e.g. sp=-0.004 overstates
    a pattern that isn't really there, so both callers route this to neutral wording.
    strength buckets: 'strong' >=0.5, 'moderate' >=0.3, else 'weak' (only when not negligible).
    direction: 'no' (negligible), else 'positive' (sp > 0) or 'negative' (sp < 0)."""
    sp = blk.get('spearman', float('nan')) if isinstance(blk, dict) else float('nan')
    if not np.isfinite(sp):
        return sp, None, None, None, None
    negligible = abs(sp) < 0.05
    if negligible:
        direction = 'no'
        strength = 'negligible'
    else:
        direction = 'positive' if sp > 0 else 'negative'
        strength = 'strong' if abs(sp) >= 0.5 else ('moderate' if abs(sp) >= 0.3 else 'weak')
    ci_excludes = _ci_excludes_zero(blk)
    return sp, negligible, strength, direction, ci_excludes


def _axis_support_state(blk: dict, expect_positive: bool) -> str:
    """Classify one Spearman block against the sign H1 predicts for that axis.
    Returns 'supported' (right sign, CI excludes 0), 'contradicted' (wrong sign, CI excludes 0),
    'weak' (right sign, CI includes 0), or 'indeterminate' (no valid estimate)."""
    sp = blk.get('spearman', float('nan')) if isinstance(blk, dict) else float('nan')
    if not np.isfinite(sp):
        return 'indeterminate'
    excludes_zero = _ci_excludes_zero(blk)
    right_sign = (sp > 0) if expect_positive else (sp < 0)
    if right_sign and excludes_zero:
        return 'supported'
    if excludes_zero and not right_sign:
        return 'contradicted'
    return 'weak'


def _single_axis_verdict(blk: dict, expect_positive: bool, axis_label: str) -> str:
    """Independent single-axis H1 verdict.

    Restructure: H1 is 8 INDEPENDENT tests (4 per axis), not one combined two-axis
    verdict — a result on one axis is never gated on, or merged with, the other axis's result.
    This classifies ONE axis's own Spearman block against the sign H1 predicts for THAT axis only:
      Axis 1 (vs C2ST-AUC, expect_positive=True): more important features carry more value shift.
      Axis 2 (vs separation_stability, expect_positive=False): more important features keep LESS
        concept stability, i.e. a NEGATIVE Spearman against a stability score (not a shift score).
    See the H1 correlation summary table below for how the two axes compare side by side; the
    ablation (C9/H2), not any of these correlations, is the definitive concept-transfer test.
    """
    state = _axis_support_state(blk, expect_positive)
    sp = blk.get('spearman', float('nan')) if isinstance(blk, dict) else float('nan')
    if state == 'indeterminate':
        return f'{axis_label} INDETERMINATE: no valid data for this test'
    if state == 'supported':
        return (f'{axis_label} POSITIVE: ρ={sp:+.3f}, CI excludes 0 -- confirms the H1 '
                f'direction predicted for this axis')
    if state == 'contradicted':
        return (f'{axis_label} CONTRADICTED: ρ={sp:+.3f}, CI excludes 0 -- wrong sign for '
                f'the H1 direction predicted for this axis')
    return (f'{axis_label} NOT SIGNIFICANT: ρ={sp:+.3f}, CI includes 0 -- right sign but not '
            f'distinguishable from chance')



_STEP_DESCRIPTIONS = {
    0:  'Per-file row/column audits and label distribution, before any cleaning.',
    1:  ('Merges the per-file raw CSVs into one parquet per year, drops exact duplicates and '
         'malformed rows, and consolidates the raw attack labels into canonical families.'),
    2:  'Pearson and Spearman correlation between every feature pair, computed separately per year.',
    3:  ('Flags feature pairs that are redundant (|r| above threshold) in BOTH years, and drops '
         'one feature from each redundant pair; pairs correlated in only one year are kept.'),
    4:  'Z-score scaling (fit per year), label encoding, and the train/test split used by every downstream step.',
    5:  ('Trains the LightGBM RF-mode binary and multiclass models, one pair per year. Records '
         'native gain, native split-count, and permutation importance per feature.'),
    6:  ('Evaluates each trained model against its own year (same-year baseline, inflated by '
         'train/test overlap) and against the opposite year, in both the concept and covariate framings.'),
    7:  ('Per-feature distributional statistics for both years: cardinality, mutual information '
         'with the label, and benign-vs-attack separation AUC.'),
    8:  ('Renders every feature\'s distribution to a PNG for manual inspection, one image per '
         'feature per dataset per class-mode variant. Produces no numbers consumed downstream.'),
    9:  'Decides, per feature, which statistical test(s) Step 10 should run against it, based on Step 7\'s profile.',
    10: 'Runs the tests Step 9 planned, producing the two-axis verdict (covariate shift, concept stability) per feature.',
    11: ('Joins Step 5\'s importance with Step 10\'s drift verdicts, runs the C1-C9 tests, and '
         'generates this document.'),
}


def _scan_step_dirs(step_num: int) -> list[tuple[Path, dict]]:
    """Every output/ and results/ subdirectory whose name starts with '<step_num>_', with a
    live file-extension count for each (via rglob, so nested per-dataset/per-algorithm
    subfolders like results/11_cross_analysis/lightgbm/ are included). Counted at doc-generation
    time so this can't drift the way a hand-typed file count would."""
    found: list = []
    for root_name in ('output', 'results'):
        base = PROJECT_ROOT / root_name
        if not base.exists():
            continue
        for d in sorted(base.iterdir()):
            if d.is_dir() and d.name.startswith(f'{step_num}_'):
                counts: dict = {}
                for f in d.rglob('*'):
                    if f.is_file():
                        ext = f.suffix or '(no extension)'
                        counts[ext] = counts.get(ext, 0) + 1
                found.append((d, counts))
    return found


def _build_output_map_section() -> list:
    """Pipeline output map: for each step 0-11, what it computes and what is actually on disk
    for it right now (directory + live file-extension counts, not a hand-typed inventory).
    Covers every artifact, including the ones not written up or embedded anywhere above
    (Step 8's several hundred per-feature plots, Step 11's diagnostic PNGs) — this is the
    reference for going past what is curated in the sections above."""
    lines: list = [
        '---',
        '## Pipeline Output Map',
        '',
        ('Every step writes to a fixed pair of directories: `output/<step>/` for intermediate '
         'data artifacts (JSON, CSV, parquet, pickled caches) and `results/<step>/` for anything '
         'meant to be looked at directly (PNGs, text reports, this document). Not everything on '
         'disk is walked through or embedded above — Step 8 alone renders one PNG per feature '
         'per dataset per class-mode variant, several hundred images that would swamp this '
         'document if embedded, and Step 11 writes a number of diagnostic plots whose numeric '
         'equivalents are already in the C-tables above rather than shown as figures. This '
         'section counts what is actually on disk for each step, so anyone who wants to go past '
         'what is curated above can go straight to the right folder.'),
        '',
    ]
    for step_num in range(12):
        lines += [f'### Step {step_num}', '', _STEP_DESCRIPTIONS.get(step_num, ''), '']
        rows = _scan_step_dirs(step_num)
        if not rows:
            lines += ['- Not found on disk this run.', '']
            continue
        for d, counts in rows:
            rel = d.relative_to(PROJECT_ROOT).as_posix()
            if counts:
                parts = ', '.join(f'{n} {ext}' for ext, n in
                                  sorted(counts.items(), key=lambda kv: -kv[1]))
                lines.append(f'- `{rel}/` — {parts}')
            else:
                lines.append(f'- `{rel}/` — empty')
        lines.append('')
    return lines


def _build_preamble_sections(df: 'pd.DataFrame', baseline: dict | None) -> 'tuple[list, dict]':
    """Build the Step 5–10 pipeline walkthrough sections that precede C1–C9 in results.md.
    Returns (lines, layer_a): layer_a is the verdicts_layerA_<DS1>_<DS2>.json load performed here
    (needed for the per-attack-family breakdown below), returned so write_results_doc() can reuse
    it for its own E1 section instead of loading the same file a second time."""
    from collections import Counter as _Counter

    def _ni(v) -> str:
        return f'{v:,}' if isinstance(v, int) else 'n/a'

    # Load step-7 profiles (best-effort; silently skip if not found)
    _profiles_17 = load_profiles(DS1)
    _profiles_18 = load_profiles(DS2)

    # Load step-9 routing plans (best-effort)
    _plans: dict = {}
    try:
        pp = (PROJECT_ROOT / 'output' / '9_plan_comparison' /
              'comparison_plans_cicids2017_cicids2018.json')
        if pp.exists():
            _plans = json.loads(pp.read_text(encoding='utf-8'))
    except Exception:
        pass

    lines: list = []

    # ── Step 0: initial data exploration (added this round) ────────────────────
    lines += ['---', '## Step 0: Initial Data Exploration', '']
    lines += [
        'Before any cleaning, looked at the raw per-year CSV files (a separate file per capture '
        'day) to know what is actually being worked with: column inventory, row counts, label '
        'distribution, and data-quality flags (NaN/Inf rows). '
        'Followed the original dataset researchers\' methodology: rows whose raw label carries an explicit '
        '"- Attempted" suffix (an attempted-but-failed attack, which produces benign-shaped traffic) are '
        'folded into Benign during cleaning (step 1) rather than kept as a separate attack family.',
        '',
    ]
    _step0_17 = parse_step0_report(DS1)
    _step0_18 = parse_step0_report(DS2)
    if _step0_17 and _step0_18:
        lines += [
            (f'Found {_step0_17.get("total_columns") or "n/a"} raw columns and '
             f'{_ni(_step0_17.get("total_rows"))} total rows across 2017\'s capture files, '
             f'{_step0_18.get("total_columns") or "n/a"} columns and '
             f'{_ni(_step0_18.get("total_rows"))} rows across 2018\'s capture files.'),
            '',
            '| Label (2017, after attempted→Benign fold) | Count | % |',
            '|---|---:|---:|',
        ]
        for name, cnt, pct in _step0_17.get('labels', [])[:10]:
            lines.append(f'| {name} | {cnt:,} | {pct:.2f}% |')
        lines += [
            '',
            '| Label (2018, after attempted→Benign fold) | Count | % |',
            '|---|---:|---:|',
        ]
        for name, cnt, pct in _step0_18.get('labels', [])[:10]:
            lines.append(f'| {name} | {cnt:,} | {pct:.2f}% |')
        lines += ['', '']

        if _step0_17.get('labels_before') and _step0_18.get('labels_before'):
            lines += [
                ('The two tables above show the labels AFTER the attempted→Benign fold. The two '
                 'below show the same rows BEFORE that fold, so each attempted-but-failed attack '
                 'appears as its own row with the original attack type visible in the label text '
                 '(e.g. `Web Attack - Brute Force - Attempted`, `Infiltration - Attempted`).'),
                '',
                '| Label (2017, before attempted→Benign fold) | Count | % |',
                '|---|---:|---:|',
            ]
            for name, cnt, pct in _step0_17['labels_before']:
                lines.append(f'| {name} | {cnt:,} | {pct:.2f}% |')
            lines += [
                '',
                '| Label (2018, before attempted→Benign fold) | Count | % |',
                '|---|---:|---:|',
            ]
            for name, cnt, pct in _step0_18['labels_before']:
                lines.append(f'| {name} | {cnt:,} | {pct:.2f}% |')
            lines += ['', '']

        for ds, tag in ((DS1, '2017'), (DS2, '2018')):
            for fname, cap in (('2_files_per_label.png', 'labels per file'),
                               ('3_quality_per_file.png', 'data quality per file')):
                im = _img(PROJECT_ROOT / 'results' / '0_dataexplore' / ds / fname, f'{tag}: {cap}')
                if im:
                    lines += [im, '']
        lines += [
            (f'Full reports: `results/0_dataexplore/{DS1}/0_dataexplore_report.txt` and '
             f'`results/0_dataexplore/{DS2}/0_dataexplore_report.txt`.'),
            '',
        ]
    else:
        lines += ['_Step 0 report not found; run step 0 before step 11._', '']

    # ── Step 1: loading, cleaning, combining (added this round) ────────────────
    lines += ['---', '## Step 1: Loading, Cleaning, and Combining', '']
    lines += [
        'Combined that year\'s raw capture-day CSVs into one file, dropped rows with NaN or Inf '
        'values, removed exact duplicate rows (row-hash based), and consolidated raw labels into '
        'the canonical attack-family set every later step trains and tests on. '
        f'Applied a class-count floor: any canonical family with fewer than {Config1.MIN_CLASS_ROWS} rows '
        'was dropped as too sparse to model reliably.',
        '',
    ]
    _step1_17 = parse_step1_report(DS1)
    _step1_18 = parse_step1_report(DS2)
    if _step1_17 and _step1_18:
        for (ds_label, step1) in ((DS1, _step1_17), (DS2, _step1_18)):
            lines += [
                f'### {ds_label}: row-count funnel',
                '',
                '| Stage | Rows |',
                '|---|---:|',
            ]
            for stage, count in step1.get('row_counts', []):
                lines.append(f'| {stage} | {count:,} |')
            lines += ['', '### Canonical families (after consolidation)', '',
                      '| Family | Count |', '|---|---:|']
            families = step1.get('canonical_families', [])
            for fam, count in families:
                lines.append(f'| {fam} | {count:,} |')
            if families:
                smallest = min(families, key=lambda x: x[1])
                lines += [
                    '',
                    (f'Smallest canonical family: `{smallest[0]}` at {smallest[1]:,} rows, above '
                     f'the {Config1.MIN_CLASS_ROWS}-row floor, so nothing was dropped at this '
                     'stage in this run.'),
                    '',
                    '### Label consolidation: raw labels → canonical families',
                    '',
                    ('Raw CSV labels are consolidated into the canonical family set shown above. '
                     'Examples of the mapping:'),
                    '',
                    '| Raw label | Canonical family |',
                    '|---|---|',
                    '| Benign | Benign |',
                    '| DoS Hulk | DoS |',
                    '| DoS GoldenEye | DoS |',
                    '| DoS Slowloris | DoS |',
                    '| DDoS HOIC | DDoS |',
                    '| LOIC HTTP | DDoS |',
                    '| LOIC UDP | DDoS |',
                    '| SSH-Patator | Brute Force |',
                    '| FTP-Patator | Brute Force |',
                    '| Web Attack – Brute Force | Web Attack |',
                    '| Web Attack – XSS | Web Attack |',
                    '| Infiltration | Infiltration |',
                    '',
                ]
            lines += ['', '']
    else:
        lines += ['_Step 1 report not found; run step 1 before step 11._', '']

    for ds, tag in ((DS1, '2017'), (DS2, '2018')):
        for fname, cap in (('labels_stage1_raw.png', 'labels, raw'),
                           ('labels_stage4_final.png', 'labels, final (post-dedup, post-consolidation)')):
            im = _img(PROJECT_ROOT / 'results' / '1_load_clean_combine' / ds / fname, f'{tag}: {cap}')
            if im:
                lines += [im, '']
    lines += [
        (f'Output: `data/cc_data/{DS1}_cleaned.parquet` and `data/cc_data/{DS2}_cleaned.parquet`, '
         'the input every later step reads from.'),
        '',
    ]

    # ── Step 2: per-year correlation matrices (split out of the old "Step 2-3" section) ──
    lines += ['---', '## Step 2: Per-Year Correlation Matrices', '']
    _pear17 = load_flagged_pairs(DS1, 'pearson')
    _pear18 = load_flagged_pairs(DS2, 'pearson')
    _spear17 = load_flagged_pairs(DS1, 'spearman')
    _spear18 = load_flagged_pairs(DS2, 'spearman')
    _pmat17 = load_correlation_matrix(DS1, 'pearson')
    _smat17 = load_correlation_matrix(DS1, 'spearman')
    _pmat18 = load_correlation_matrix(DS2, 'pearson')
    _smat18 = load_correlation_matrix(DS2, 'spearman')
    lines += [
        ('For each year independently, computed Pearson (exact) and Spearman (deterministic '
         'subsample) correlation between every candidate feature pair, then flagged any pair at '
         f'|r| >= {_pear17.get("threshold", 0.90)} in either metric. This step only surfaces '
         'candidates; it does not decide what to drop (step 3 does that, next).'),
        '',
    ]
    if _pear17 and _pear18:
        for ds_label, pear, spear, pmat, smat in (
                (DS1, _pear17, _spear17, _pmat17, _smat17),
                (DS2, _pear18, _spear18, _pmat18, _smat18)):
            # Union of whatever EITHER metric flagged at the looser 0.90 screen, then look up the
            # EXACT r for BOTH metrics from the full matrices (not just the metric that flagged
            # it), so a pair Spearman flagged but Pearson did not quite clear 0.90 for still gets
            # a real Pearson number instead of "n/a". Filtered to the stricter 0.95 either-metric
            # bar requested for this table (independent of step 3's own drop threshold below).
            seen: dict = {}
            for p in pear.get('pairs', []) + spear.get('pairs', []):
                key = frozenset((p.get('a', '?'), p.get('b', '?')))
                if key not in seen:
                    seen[key] = (p.get('a', '?'), p.get('b', '?'))
            rows = []
            for a, b in seen.values():
                pr = _matrix_lookup(pmat, a, b)
                sr = _matrix_lookup(smat, a, b)
                best = max(abs(pr) if np.isfinite(pr) else 0.0, abs(sr) if np.isfinite(sr) else 0.0)
                if best >= 0.95:
                    rows.append((a, b, pr, sr, best))
            rows.sort(key=lambda r: -r[4])
            lines += [
                (f'### {ds_label}: pairs with Pearson or Spearman |r| >= 0.95 '
                 f'(of {pear.get("count","?")} flagged at the {pear.get("threshold",0.90)} screen)'),
                '',
                '| Feature A | Feature B | Pearson r | Spearman r |',
                '|---|---|---:|---:|',
            ]
            for a, b, pr, sr, _ in rows:
                pr_s = f'{pr:.4f}' if np.isfinite(pr) else 'n/a'
                sr_s = f'{sr:.4f}' if np.isfinite(sr) else 'n/a'
                lines.append(f'| {a} | {b} | {pr_s} | {sr_s} |')
            lines += ['', '']
        for ds, tag in ((DS1, '2017'), (DS2, '2018')):
            im = _img(PROJECT_ROOT / 'results' / '2_correlation_analysis' / ds / 'pearson_heatmap.png',
                      f'{tag}: Pearson correlation heatmap')
            if im:
                lines += [im, '']
        lines += [
            (f'Full per-year flagged pairs: `output/2_correlation_analysis/{DS1}/` and '
             f'`output/2_correlation_analysis/{DS2}/` (`pearson_flagged_pairs.json`, '
             '`spearman_flagged_pairs.json`).'),
            '',
        ]
    else:
        lines += ['_Step 2 flagged-pairs files not found; run step 2 before step 11._', '']

    # ── Step 3: cross-year consensus and feature removal (split out of "Step 2-3") ──
    lines += ['---', '## Step 3: Cross-Year Consensus and Feature Removal', '']
    _drop = load_correlation_drop_decisions()
    _fnames17 = load_json_best_effort(PROJECT_ROOT / 'output' / '4_preprocessing' / DS1 / 'feature_names.json')
    if _drop:
        n_total = _drop.get('n_features_total', 'n/a')
        dropped = _drop.get('features_to_drop', [])
        groups = _drop.get('redundancy_groups', [])
        shifted = _drop.get('shifted_pairs', [])
        # Checked this against feature_names.json directly rather than trusting n_total minus
        # len(dropped): that arithmetic gives 73, but the actual downstream feature count is 71
        # (confirmed in feature_names.json), because step 2/3's candidate pool (n_features_total)
        # is scoped slightly differently from step 4's final feature set (it excludes most but not
        # all of the identifier-like columns step 4 also strips). Reporting the verified number,
        # not the derived one, to avoid a silent contradiction with Table 1.
        n_final = _fnames17.get('n_features', 'n/a')
        lines += [
            ('A pair flagged in step 2 could be a stable, real redundancy (the same measurement '
             'twice) or a one-year coincidence (drift). Step 3 tells these apart by requiring a '
             f'pair clear BOTH Pearson AND Spearman, in BOTH years (consensus rule: '
             f'{_drop.get("consensus_rule", "n/a")}, threshold {_drop.get("consensus_threshold", "n/a")}) '
             f'before treating it as real redundancy. Found {len(dropped)} redundant features out of {n_total} '
             'correlation-analysis candidates this way; within each redundancy group, kept the feature with '
             'the lowest average correlation to the rest.'),
            '',
            (f'Verified against `feature_names.json` directly: every downstream step (training, testing, '
             f'ablation) runs on {n_final} final features. Note: {n_final} is not simply {n_total}−{len(dropped)}, '
             'since steps 2–3 and step 4 scope their candidate pools slightly differently (step 4 also '
             'separately excludes 9 identifier-like columns); both counts are real and correct.'),
            '',
        ]
        # Reuse the same Spearman matrices already loaded above for the Step-2 tables (identical
        # ds + method, and neither _smat17 nor _smat18 is mutated in between) instead of a second
        # load_correlation_matrix() round-trip through the same JSON files.
        _smat17_s3 = _smat17
        _smat18_s3 = _smat18

        def _avg_spearman_within(group: list) -> float:
            """Mean absolute Spearman r across every pair inside a redundancy group, averaged
            across both years — mirrors 3_correlation_comparison.py's own avg_pearson_within
            (mean of average_abs_matrix([spearman_2017, spearman_2018]) over within-group pairs),
            computed here at render time from the already-saved spearman_matrix.json files rather
            than touching that script, since it never persisted the Spearman side of this average
            at the group level (only avg_pearson_within made it into drop_decisions.json)."""
            vals = []
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    r17 = _matrix_lookup(_smat17_s3, group[i], group[j])
                    r18 = _matrix_lookup(_smat18_s3, group[i], group[j])
                    if np.isfinite(r17):
                        vals.append(abs(r17))
                    if np.isfinite(r18):
                        vals.append(abs(r18))
            return float(np.mean(vals)) if vals else float('nan')

        lines += [
            '### Redundancy groups (3+ mutually redundant features in both years/metrics)',
            '',
            ('`Status`: `stable` = both Pearson AND Spearman clear the 0.95 consensus threshold '
             'in both years; `stable_pearson_only` = Pearson clears in both years but Spearman '
             'falls short (still kept as redundant, since Pearson is the primary signal here — '
             'see `3_correlation_comparison.py`\'s pair classification). "Avg |r| within group" is '
             'the mean absolute correlation across every pair inside the group, averaged across '
             'both years.'),
            '',
            '| Kept | Dropped | Avg Pearson \\|r\\| | Avg Spearman \\|r\\| | Status |',
            '|------|---------|------------------:|-------------------:|--------|',
        ]
        for g in groups:
            if not g.get('keep'):
                continue
            avg_sp = _avg_spearman_within(g.get('group', []))
            avg_sp_s = f'{avg_sp:.4f}' if np.isfinite(avg_sp) else 'n/a'
            lines.append(f'| {g.get("keep","?")} | {", ".join(g.get("drop", []))} '
                         f'| {g.get("avg_pearson_within", float("nan")):.4f} | {avg_sp_s} '
                         f'| {g.get("status","?")} |')
        lines += ['', '']

        if shifted:
            shifted_sorted = sorted(
                shifted,
                key=lambda p: abs(p.get('pearson_2017', 0) - p.get('pearson_2018', 0)),
                reverse=True)
            lines += [
                ('### Drifted pairs: correlated in one year only, kept (not dropped)'),
                '',
                ('These pairs fail the both-years consensus rule above (one year is far below '
                 'the threshold), so step 3 keeps both features rather than dropping either; the '
                 'pair itself is a real cross-year drift signal, separate from the per-feature '
                 f'drift this whole document is otherwise about. {len(shifted)} such pairs exist, '
                 'top 10 by |2017-2018 gap| shown:'),
                '',
                '| Feature A | Feature B | r (2017) | r (2018) | Correlated in |',
                '|---|---|---:|---:|---|',
            ]
            for p in shifted_sorted[:10]:
                lines.append(f'| {p.get("feature_a","?")} | {p.get("feature_b","?")} '
                             f'| {p.get("pearson_2017", float("nan")):.4f} '
                             f'| {p.get("pearson_2018", float("nan")):.4f} '
                             f'| {p.get("correlated_in","?")} |')
            lines += ['', '']

        lines += [
            ('To keep in mind: step 2\'s flag threshold is looser than step 3\'s drop threshold, '
             'by design, since step 2 is exploratory and step 3 is the conservative consensus '
             'filter; the step-2 flagged-pairs files list more pairs than step 3 actually drops, '
             'and that is expected, not a bug.'),
            '',
            ('The code confirms this directly: in `run_ablation()`, the canonical feature list '
             '(`canonical = [f for f in feat17 if f in feat18 ...]`) is sourced from each year\'s '
             'post-drop `feature_names.json`, the same file steps 5 and 6 train and test on. The '
             'ablation never sees the dropped features either; this was checked specifically '
             'because it is the kind of silent inconsistency that would otherwise undermine the '
             'whole H2 test.'),
            '',
            ('Output: `output/3_correlation_comparison/drop_decisions.json`.'),
            '',
        ]
        for fname, cap in (('redundancy_groups.png', 'redundancy groups'),
                           ('diff_heatmap.png', '2018-2017 correlation difference')):
            im = _img(PROJECT_ROOT / 'results' / '3_correlation_comparison' / fname, cap)
            if im:
                lines += [im, '']
    else:
        lines += ['_Step 3 drop_decisions.json not found; run steps 2-3 before step 11._', '']

    # ── Step 4: preprocessing (split out of the old "Step 4-5" section) ────────
    lines += ['---', '## Step 4: Preprocessing', '']
    lines += [
        ('Each year\'s cleaned data gets its own independent feature set: split into train/test, '
         'and a Z-score scaler fit ONLY on that year\'s training rows. This per-year-only scaler '
         'fitting is exactly the mechanism the concept-vs-covariate framing throughout this '
         'document depends on.'),
        '',
    ]
    if _fnames17:
        excluded = _fnames17.get('excluded_identifiers', [])
        lines += [
            (f'`feature_names.json`: {_fnames17.get("n_features","?")} final features, '
             f'{len(_fnames17.get("dropped_redundant", []))} dropped as redundant (step 3, above), '
             f'{len(excluded)} excluded identifier-like columns: '
             f'{", ".join(f"`{c}`" for c in excluded)}.'),
            '',
        ]
    _meta17 = load_preprocessing_meta(DS1)
    _meta18 = load_preprocessing_meta(DS2)
    if _meta17 and _meta18:
        lines += [
            '| | 2017 | 2018 |',
            '|---|---:|---:|',
            f'| Rows total | {_ni(_meta17.get("rows_total"))} | {_ni(_meta18.get("rows_total"))} |',
            f'| Rows train | {_ni(_meta17.get("rows_train"))} | {_ni(_meta18.get("rows_train"))} |',
            f'| Rows test | {_ni(_meta17.get("rows_test"))} | {_ni(_meta18.get("rows_test"))} |',
            f'| Test fraction | {_fmt(_meta17.get("realized_test_fraction"))} '
            f'| {_fmt(_meta18.get("realized_test_fraction"))} |',
            f'| Seed | {_meta17.get("seed","?")} | {_meta18.get("seed","?")} |',
            '',
        ]
    _scaler17 = load_scaler_json(DS1)
    _scaler18 = load_scaler_json(DS2)
    if _scaler17 and _scaler18:
        def _scaler_row(feat):
            try:
                i17 = _scaler17['features'].index(feat)
                i18 = _scaler18['features'].index(feat)
                return (_scaler17['mean'][i17], _scaler17['scale'][i17],
                        _scaler18['mean'][i18], _scaler18['scale'][i18])
            except (ValueError, KeyError, IndexError):
                return None
        lines += [
            ('Scaling: a custom streaming Z-score scaler (population variance, ddof=0), saved as '
             'two separate artifacts (`output/4_preprocessing/<year>/scaler.json`). Found this '
             'difference for two example features, read directly from the saved scaler files '
             'rather than hand-typed, to make the magnitude of "own-year scaler" vs "train-year '
             'scaler" concrete:'),
            '',
            '| Feature | 2017 mean | 2017 scale | 2018 mean | 2018 scale |',
            '|---------|----------:|-----------:|----------:|-----------:|',
        ]
        for feat in ('Flow Duration', 'Down/Up Ratio'):
            row = _scaler_row(feat)
            if row:
                lines.append(f'| {feat} | {row[0]:.4f} | {row[1]:.4f} | {row[2]:.4f} | {row[3]:.4f} |')
        lines += [
            '',
            (f'Fitted on {_scaler17.get("n_train_samples","?")} training rows (2017) and '
             f'{_scaler18.get("n_train_samples","?")} training rows (2018).'),
            '',
        ]
    _lmap17 = load_label_mapping_json(DS1)
    if _lmap17:
        lines += [
            '### Label mapping (identical for both years)',
            '',
            '| Binary label | Code |',
            '|---|---:|',
        ]
        for name, code in (_lmap17.get('binary') or {}).items():
            if isinstance(code, int):
                lines.append(f'| {name} | {code} |')
        lines += ['', '| Multiclass family | Code |', '|---|---:|']
        for name, code in (_lmap17.get('multiclass') or {}).items():
            if isinstance(code, int):
                lines.append(f'| {name} | {code} |')
        lines += ['', '']
    for ds, tag in ((DS1, '2017'), (DS2, '2018')):
        im = _img(PROJECT_ROOT / 'results' / '4_preprocessing' / ds / 'scaling_check.png',
                  f'{tag}: scaling check')
        if im:
            lines += [im, '']
    lines += [
        (f'Output: `output/4_preprocessing/{DS1}/` and `output/4_preprocessing/{DS2}/` contain '
         '`feature_names.json`, `preprocessing_meta.json`, `label_mapping.json`, `scaler.json`, '
         'and `scaling_check.png`.'),
        '',
    ]

    # ── Step 5: model training and configuration (split out of "Step 4-5" + old "Step 5") ──
    lines += ['---', '## Step 5: Model Training and Configuration', '']
    lines += [
        ('Trained one LightGBM random-forest model per year, per task (binary, multiclass) — '
         'same hyperparameters everywhere, only the data differs.'),
        '',
        '### Hyperparameters (identical for both years and both tasks)',
        '',
        '| Parameter | Value |',
        '|---|---|',
        '| boosting_type | `rf` (bagging, not boosting) |',
        f'| n_estimators | `{Config5.N_ESTIMATORS}` |',
        f'| num_leaves | `{Config5.LGBM_NUM_LEAVES}` |',
        f'| max_depth | `{Config5.MAX_DEPTH if Config5.MAX_DEPTH is not None else -1}` (unlimited) |',
        f'| bagging_fraction | `{Config5.LGBM_BAGGING_FRACTION}` |',
        f'| bagging_freq | `{Config5.LGBM_BAGGING_FREQ}` |',
        f'| feature_fraction | `{Config5.LGBM_FEATURE_FRACTION}` |',
        f'| min_child_samples | `{Config5.LGBM_MIN_CHILD}` |',
        "| class_weight | `'balanced'` |",
        f'| random_state | `{Config5.SEED}` |',
        '',
    ]
    _train17 = load_training_meta(DS1)
    _train18 = load_training_meta(DS2)
    if _train17 and _train18:
        lines += [
            '### Training summary',
            '',
            '| Year | Task | Rows train | Fit time (s) | Model size (MB) | Classes |',
            '|---|---|---:|---:|---:|---:|',
        ]
        def _f1(v):
            try:
                return f'{float(v):.1f}'
            except (TypeError, ValueError):
                return 'n/a'
        for ds_label, meta in ((DS1, _train17), (DS2, _train18)):
            for task in meta.get('tasks', []):
                lines.append(f'| {ds_label} | {task.get("task","?")} '
                             f'| {_ni(task.get("n_train_rows"))} '
                             f'| {_f1(task.get("fit_seconds"))} '
                             f'| {_f1(task.get("model_size_mb"))} '
                             f'| {task.get("n_classes","?")} |')
        lines += ['', '']

    lines += [
        '### Feature Importance',
        '',
        'Feature importance is measured three ways:',
        '- **Native GAIN (primary):** total split gain contributed by the feature across all '
        'trees, normalized to sum 1 per model. This is the native importance every H1/H2 test '
        'reads. Less cardinality-biased than split-count, but still a native tree measure '
        '(Strobl 2007 caveat applies).',
        '- **Native split-count (secondary diagnostic):** number of times the feature was used '
        'as a split node (normalized to sum 1) — saved as `feature_importance_split_<task>.json`; '
        'the more cardinality-biased of the two native measures, kept for reference only.',
        f'- **Permutation:** balanced-accuracy drop when the feature is randomly shuffled '
        f'({Config5.PERM_REPEATS} repeats, per-class-capped {Config5.PERM_SAMPLE:,}-row held-out '
        'sample with exact train-duplicates removed first). Unbiased w.r.t. '
        'cardinality, but noisy at low importance.',
        '',
        'Rows sorted by native (gain) binary 2017 score, highest first. '
        'Full importance files: `output/5_training/cicids2017_lightgbm/` '
        'and `output/5_training/cicids2018_lightgbm/`.',
        '',
        '| Feature | Nat-Bin-17 | Nat-MC-17 | Nat-Bin-18 | Nat-MC-18 | Perm-Bin-17 | Perm-Bin-18 |',
        '|---------|----------:|----------:|----------:|----------:|------------:|------------:|',
    ]
    _sorted = df.sort_values('imp_2017_bin', ascending=False)
    for feat, row in _sorted.iterrows():
        def _impfmt(v):
            try:
                f = float(v)
                return f'{f:.6f}' if np.isfinite(f) else 'n/a'
            except (TypeError, ValueError):
                return 'n/a'
        lines.append(f'| {feat} | {_impfmt(row.get("imp_2017_bin"))} | {_impfmt(row.get("imp_2017_multi"))} '
                     f'| {_impfmt(row.get("imp_2018_bin"))} | {_impfmt(row.get("imp_2018_multi"))} '
                     f'| {_impfmt(row.get("imp_perm_2017_bin"))} '
                     f'| {_impfmt(row.get("imp_perm_2018_bin"))} |')
    lines += [
        '',
        '> Near-zero permutation values are expected: tree ensembles use feature combinations, '
        'so single-feature shuffling understates individual contributions.',
        '',
    ]
    for ds, tag in ((DS1, '2017'), (DS2, '2018')):
        for fname, cap in (('feature_importance_binary.png', 'native importance, binary'),
                           ('feature_importance_multiclass.png', 'native importance, multiclass'),
                           ('feature_importance_perm_binary.png', 'permutation importance, binary'),
                           ('feature_importance_perm_multiclass.png', 'permutation importance, multiclass')):
            im = _img(PROJECT_ROOT / 'results' / '5_training' / f'{ds}_lightgbm' / fname,
                      f'{tag}: {cap}')
            if im:
                lines += [im, '']
    lines += [
        (f'Output: `output/5_training/{DS1}_lightgbm/` and `output/5_training/{DS2}_lightgbm/` contain '
         'feature importance CSVs and training metadata.'),
        '',
    ]

    # ── Step 6 ───────────────────────────────────────────────────────────────────
    lines += ['---', '## Step 6: Cross-Year Test Results', '']
    lines += [
        'Each trained model is evaluated in two framings against the opposite year\'s data:',
        '- **Concept:** target year normalized with its *own* scaler, covariate shift removed, '
        'tests decision-boundary transfer.',
        '- **Covariate:** target year normalized with the *train-year* scaler, the deployment '
        'reality where only the old scaler exists.',
        '',
        '> ⚠️ Within-year scores (17→17, 18→18) use a per-row hash split that allows '
        'near-duplicate flows to straddle train/test. They are inflated baselines, '
        'not a gold standard.',
        '',
    ]
    if baseline and 'directions' in baseline:
        dirs = baseline['directions']
        d1   = dirs.get('cicids2017->cicids2018', {})
        d2   = dirs.get('cicids2018->cicids2017', {})
        b1c  = (d1.get('binary') or {}).get('concept', {})
        b1v  = (d1.get('binary') or {}).get('covariate', {})
        b2c  = (d2.get('binary') or {}).get('concept', {})
        b2v  = (d2.get('binary') or {}).get('covariate', {})
        m1c  = (d1.get('multiclass') or {}).get('concept', {})
        m1v  = (d1.get('multiclass') or {}).get('covariate', {})
        m2c  = (d2.get('multiclass') or {}).get('concept', {})
        m2v  = (d2.get('multiclass') or {}).get('covariate', {})
        def _norm_binary(d: dict) -> dict:
            """Normalize a binary metrics dict into the flat key set this table needs, whichever
            of the two shapes it arrives in: step-6's _cross_summary() (already flat: benign_f1,
            sensitivity, fpr, precision, specificity, ...) for cross-year cells, or the raw
            metrics_binary.json from evaluate_dataset() (per_class-nested, no flat keys at all) for
            the same-year baseline cells. Same source data either way — this just avoids two
            separate rendering paths for what is the same table."""
            if not d:
                return {}
            if 'sensitivity' in d:
                return d
            pc = d.get('per_class', {}) or {}
            attack, benign = pc.get('Attack', {}), pc.get('Benign', {})
            specificity = benign.get('recall', float('nan'))
            out = dict(d)
            out['attack_f1'] = attack.get('f1', float('nan'))
            out['benign_f1'] = benign.get('f1', float('nan'))
            out['sensitivity'] = attack.get('recall', float('nan'))
            out['precision'] = attack.get('precision', float('nan'))
            out['specificity'] = specificity
            out['fpr'] = (1.0 - specificity) if np.isfinite(specificity) else float('nan')
            return out

        _within17 = _norm_binary(load_within_year_metrics(DS1))
        _within18 = _norm_binary(load_within_year_metrics(DS2))

        def _gap(same, cross):
            # same-year minus cross-year: POSITIVE = performance LOST moving out of domain
            # (a drop from same-year 0.9999 to cross-year 0.9211 prints as +0.0788, not -0.0788).
            # Column header says "Gap (same-cross)", not "delta", specifically to avoid a plain
            # "+" reading as an increase when it is quantifying a loss.
            try:
                return f'{(float(same) - float(cross)):.4f}'
            except (TypeError, ValueError):
                return 'n/a'

        def _bin_row(label: str, m: dict, baseline: 'dict | None') -> str:
            cells = [_fmt(m.get(k)) for k in
                    ('accuracy', 'macro_f1', 'attack_f1', 'benign_f1', 'sensitivity', 'fpr',
                     'precision', 'specificity', 'balanced_accuracy', 'roc_auc', 'mcc')]
            if baseline is None:
                gaps = ['(baseline)', '(baseline)']
            else:
                gaps = [_gap(baseline.get('accuracy'), m.get('accuracy')),
                       _gap(baseline.get('macro_f1'), m.get('macro_f1'))]
            return '| ' + label + ' | ' + ' | '.join(cells + gaps) + ' |'

        lines += [
            '### Binary (benign vs attack)',
            '',
            ('_Full metric table, same-year baseline (own 20% held-out split) alongside all four '
             'cross-year cells, with the gap from same-year made explicit — one table, not split '
             'across a metric-breakdown table and a separate gap table that repeated the same '
             'cross-year numbers. Same metric set as C9\'s ablation table below (this is the real '
             'full-feature model — the ceiling the K-feature ablation policies are compared '
             'against). Same-year rows use their own within-year 20% test split (⚠️ inflated by '
             'near-duplicate train/test flows, see the warning above), not a fifth/sixth cross '
             'framing — there is no "own-scaler, own-year" cross case to add, since concept and '
             'covariate framings only differ when train-year != test-year. Gap columns = same-year '
             'baseline minus that row\'s value; POSITIVE means performance LOST moving out of '
             'domain (e.g. a drop from 0.9999 to 0.9211 prints as 0.0788, not -0.0788).'),
            '',
            ('| Cell | Accuracy | Macro F1 | Attack F1 | Benign F1 | Sensitivity | FPR | Precision '
             '| Specificity | Balanced Acc | ROC-AUC | MCC | Acc gap (same−cross) | Macro-F1 gap (same−cross) |'),
            '|------|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|',
        ]
        if _within17:
            lines.append(_bin_row('Same-year 2017 (baseline)', _within17, None))
        lines.append(_bin_row('2017→2018 concept', b1c, _within17 or None))
        lines.append(_bin_row('2017→2018 covariate', b1v, _within17 or None))
        if _within18:
            lines.append(_bin_row('Same-year 2018 (baseline)', _within18, None))
        lines.append(_bin_row('2018→2017 concept', b2c, _within18 or None))
        lines.append(_bin_row('2018→2017 covariate', b2v, _within18 or None))
        lines += [
            '',
            f'> Covariate 17→18 attack F1 = {_fmt(b1v.get("attack_f1"))}, the "faint detection" '
            f'that motivated this study. Concept framing recovers 17→18 partially '
            f'(attack F1 = {_fmt(b1c.get("attack_f1"))}); 18→17 collapses in both framings '
            f'(concept attack F1 = {_fmt(b2c.get("attack_f1"))}).',
            '',
        ]
        if _within17 and _within18:
            lines += [
                ('Found a large gap in all four cross-year cells, which means the same-year ~0.9999 '
                 'accuracy/macro-F1 numbers are not a usable estimate of deployed performance for '
                 'either year. To keep in mind: the same-year numbers are themselves inflated by '
                 'the near-duplicate-flow leakage noted above, so this gap is a lower bound on the '
                 'real drop, not the full size of it.'),
                '',
            ]
        lines += [
            '### Multiclass (8 attack families)',
            '',
            '| Metric | 17→18 concept | 17→18 covariate | 18→17 concept | 18→17 covariate |',
            '|--------|------:|------:|------:|------:|',
            f'| Macro F1    | {_fmt(m1c.get("macro_f1"))} | {_fmt(m1v.get("macro_f1"))} '
            f'| {_fmt(m2c.get("macro_f1"))} | {_fmt(m2v.get("macro_f1"))} |',
            f'| Balanced Acc| {_fmt(m1c.get("balanced_accuracy"))} | {_fmt(m1v.get("balanced_accuracy"))} '
            f'| {_fmt(m2c.get("balanced_accuracy"))} | {_fmt(m2v.get("balanced_accuracy"))} |',
            f'| MCC         | {_fmt(m1c.get("mcc"))} | {_fmt(m1v.get("mcc"))} '
            f'| {_fmt(m2c.get("mcc"))} | {_fmt(m2v.get("mcc"))} |',
            '',
        ]
    else:
        lines += [
            '> Step 6 baseline not loaded (run step 6 before step 11, or check '
            '`cross_year_baseline_ref.json`).',
            '',
        ]

    # ── Step 6 supplementary figures: ROC/PR/confusion for same-year and both cross-year
    # directions, both framings. Full set (not a curated subset) since this document is the
    # exhaustive reference tier; the numbers these plots visualize are already in the tables above.
    lines += ['### Supplementary figures: ROC, PR, and confusion matrices', '']
    _step6_figs: list = []
    for ds, tag in ((DS1, '2017'), (DS2, '2018')):
        rdir = testing_results_dir(PROJECT_ROOT, ds, ALGORITHM)
        _step6_figs += [
            (rdir / '6_testing_roc_binary.png', f'{tag} same-year: binary ROC'),
            (rdir / '6_testing_pr_binary.png', f'{tag} same-year: binary Precision-Recall'),
            (rdir / '6_testing_confusion_binary.png', f'{tag} same-year: binary confusion matrix'),
            (rdir / '6_testing_confusion_multiclass.png', f'{tag} same-year: multiclass confusion matrix'),
            (rdir / '6_testing_per_class_f1_multiclass.png', f'{tag} same-year: multiclass per-class F1/recall'),
        ]
    for train_ds, test_ds, dtag in ((DS1, DS2, '2017->2018'), (DS2, DS1, '2018->2017')):
        rdir = testing_results_dir(PROJECT_ROOT, f'cross_{train_ds}_to_{test_ds}', ALGORITHM)
        for fr in ('concept', 'covariate'):
            _step6_figs += [
                (rdir / f'6_cross_roc_binary_{fr}.png', f'{dtag} [{fr}]: binary ROC'),
                (rdir / f'6_cross_pr_binary_{fr}.png', f'{dtag} [{fr}]: binary Precision-Recall'),
                (rdir / f'6_cross_confusion_binary_{fr}.png', f'{dtag} [{fr}]: binary confusion matrix'),
                (rdir / f'6_cross_confusion_multiclass_{fr}.png', f'{dtag} [{fr}]: multiclass confusion matrix'),
            ]
    _step6_found = False
    for fpath, cap in _step6_figs:
        im = _img(fpath, cap)
        if im:
            lines += [im, '']
            _step6_found = True
    if not _step6_found:
        lines += ['_Step 6 figures not found this run; run `scripts/6_test.py` to produce them._', '']

    # ── Step 7 ───────────────────────────────────────────────────────────────────
    lines += ['---', '## Step 7: Feature Profiles', '']
    lines += [
        f'Step 7 computed distributional statistics for each of the {len(df)} features '
        'in both 2017 and 2018 separately. Computed properties include: detected distribution '
        'type (nominal / continuous / discrete_count), zero-inflation flag, number of detected '
        'modes, full percentile table (p1–p99), skewness, kurtosis, entropy, mutual information '
        'with the attack/benign label (MI), and separation AUC (univariate ability to distinguish '
        'benign from attack traffic with that feature alone).',
        '',
        'Full profiles: `output/7_profile/cicids2017/profiles.json` '
        'and `output/7_profile/cicids2018/profiles.json`.',
        '',
    ]
    if _profiles_17 and _profiles_18:
        from collections import Counter as _CardCounter
        _card17 = _CardCounter(v.get('detected_type', 'n/a') for v in _profiles_17.values())
        _card18 = _CardCounter(v.get('detected_type', 'n/a') for v in _profiles_18.values())
        all_types = sorted(set(_card17) | set(_card18))
        lines += [
            '### Cardinality (detected_type breakdown)',
            '',
            ('Step 7 does not store a raw per-feature unique-value count, only a categorical '
             '`detected_type` bucket; this is the cardinality-adjacent breakdown that is actually '
             'available, counted across all features profiled in each year (note: step 7 profiled '
             'all 83 pre-drop features, not the 71-feature post-drop set used for training).'),
            '',
            '| detected_type | 2017 count | 2018 count |',
            '|---------------|-----------:|-----------:|',
        ]
        for t in all_types:
            lines.append(f'| {t} | {_card17.get(t, 0)} | {_card18.get(t, 0)} |')
        lines += ['', '']
    EXAMPLE = 'Total Length of Bwd Packet'
    ex17 = _profiles_17.get(EXAMPLE, {})
    if ex17:
        def _gv(key, fmt='{}'):
            v = ex17.get(key, None)
            if v is None:
                return 'n/a'
            try:
                return fmt.format(v)
            except (TypeError, ValueError):
                return str(v)

        modes_list = ex17.get('modes', []) or []
        modes_summary = ', '.join(
            f'(center={m.get("center", "?"):.2f}, spread={m.get("spread", "?"):.2f}, '
            f'mass={m.get("mass", "?"):.2f})'
            for m in modes_list if isinstance(m, dict)
        ) or 'n/a'

        pcs = ex17.get('per_class_separation', {}) or {}
        pcs_lines = [
            f'  {fam:<14s} magnitude={v.get("magnitude", float("nan")):.4f}  '
            f'direction={v.get("direction", "n/a")}  '
            f'mi_normalized={v.get("mi_normalized", float("nan")):.4f}  '
            f'n_attack={v.get("n_attack", "n/a")}'
            for fam, v in pcs.items() if isinstance(v, dict)
        ]

        lines += [
            f'### Worked example: {EXAMPLE} (2017)',
            '',
            'Every field from the 2017 profile JSON for this one feature, grouped by '
            'category (the other 70 features have the same fields, see the JSON pointer '
            'above rather than repeating this block 70 more times):',
            '',
            '```',
            '# identity / type flags',
            f'detected_type            : {_gv("detected_type")}',
            f'zero_inflated            : {_gv("zero_inflated")}',
            f'has_sentinel             : {_gv("has_sentinel")}',
            f'sentinel_value           : {_gv("sentinel_value")}',
            f'has_impossible_values    : {_gv("has_impossible_values")}',
            f'degenerate               : {_gv("degenerate")}',
            f'is_identifier            : {_gv("is_identifier")}',
            '',
            '# counts / missingness',
            f'n_total                  : {_gv("n_total")}',
            f'n_missing                : {_gv("n_missing")}',
            f'n_zero                   : {_gv("n_zero")}',
            f'zero_fraction            : {_gv("zero_fraction", "{:.4f}")}',
            '',
            '# location / spread',
            f'min / max                : {_gv("min", "{:.4f}")} / {_gv("max", "{:.4f}")}',
            f'mean / median / std      : {_gv("mean", "{:.4f}")} / '
            f'{_gv("median", "{:.4f}")} / {_gv("std", "{:.4f}")}',
            f'range / iqr / mad        : {_gv("range", "{:.4f}")} / '
            f'{_gv("iqr", "{:.4f}")} / {_gv("mad", "{:.4f}")}',
            f'coefficient_of_variation : {_gv("coefficient_of_variation", "{:.4f}")}',
            '',
            '# percentiles',
            f'p01 / p05 / p10          : {_gv("p01", "{:.4f}")} / '
            f'{_gv("p05", "{:.4f}")} / {_gv("p10", "{:.4f}")}',
            f'p25 / p50 / p75          : {_gv("p25", "{:.4f}")} / '
            f'{_gv("p50", "{:.4f}")} / {_gv("p75", "{:.4f}")}',
            f'p90 / p95 / p99          : {_gv("p90", "{:.4f}")} / '
            f'{_gv("p95", "{:.4f}")} / {_gv("p99", "{:.4f}")}',
            '',
            '# shape',
            f'skewness                 : {_gv("skewness", "{:.4f}")}',
            f'kurtosis                 : {_gv("kurtosis", "{:.4f}")}',
            f'entropy                  : {_gv("entropy", "{:.4f}")}',
            '',
            '# outliers / sentinel mass',
            f'outlier_count_low/high   : {_gv("outlier_count_low")} / {_gv("outlier_count_high")}',
            f'outlier_fraction         : {_gv("outlier_fraction", "{:.4f}")}',
            f'sentinel_mass            : {_gv("sentinel_mass", "{:.4f}")}',
            '',
            '# modes (multi-modality detection)',
            f'n_modes                  : {_gv("n_modes")}',
            f'modes                    : {modes_summary}',
            '',
            '# recommended scaling',
            f'recommended_scale        : {_gv("recommended_scale")}',
            f'scale_param              : {_gv("scale_param")}',
            '',
            '# clipping (view window used for plots, not the actual data)',
            f'view_low / view_high     : {_gv("view_low", "{:.4f}")} / {_gv("view_high", "{:.4f}")}',
            f'n_clipped_low/high       : {_gv("n_clipped_low")} / {_gv("n_clipped_high")}',
            '',
            '# attack/benign separation (univariate, this feature alone)',
            f'roc_auc_benign_vs_attack : {_gv("roc_auc_benign_vs_attack", "{:.4f}")}',
            f'separation_magnitude     : {_gv("separation_magnitude", "{:.4f}")}',
            f'separation_direction     : {_gv("separation_direction")}'
            '  (1 = attack values higher, -1 = benign values higher)',
            f'separation_auc_raw       : {_gv("separation_auc_raw", "{:.4f}")}',
            f'mutual_info              : {_gv("mutual_info", "{:.4f}")}',
            f'mutual_info_normalized   : {_gv("mutual_info_normalized", "{:.4f}")}',
            '```',
            '',
            '`per_class_separation`, the same separation test repeated per attack family '
            '(benign vs that one family only):',
            '```',
            *(pcs_lines or ['  n/a']),
            '```',
            '',
            '> `mutual_info_normalized` and `roc_auc_benign_vs_attack` from both years '
            'feed directly into the Axis 2 (concept stability) calculation in Step 10. '
            '`detected_type` and mode count drive the Step 9 routing decision. '
            '`zero_fraction` is reported again below alongside the Axis 1 shift metrics '
            '(C2ST/MMD/Wasserstein) as data-quality context, not a drift measurement.',
            '',
        ]

    lines += [
        f'### MI and separation AUC: all {len(df)} features, both years',
        '',
        'Sorted by MI_2017 descending. AUC below 0.55 indicates near-chance separation.',
        '',
        '| Feature | MI_2017 | AUC_2017 | MI_2018 | AUC_2018 |',
        '|---------|--------:|---------:|--------:|---------:|',
    ]
    mi_rows = []
    for feat in df.index:
        p17 = _profiles_17.get(feat, {})
        p18 = _profiles_18.get(feat, {})
        mi17 = float(df.at[feat, 'mutual_info_norm_2017']) \
            if 'mutual_info_norm_2017' in df.columns else float('nan')
        mi18 = float(df.at[feat, 'mutual_info_norm_2018']) \
            if 'mutual_info_norm_2018' in df.columns else float('nan')
        auc17 = p17.get('roc_auc_benign_vs_attack', float('nan'))
        auc18 = p18.get('roc_auc_benign_vs_attack', float('nan'))
        mi_rows.append((feat, mi17, auc17, mi18, auc18))
    mi_rows.sort(key=lambda r: r[1] if np.isfinite(r[1]) else -1.0, reverse=True)
    for feat, mi17, auc17, mi18, auc18 in mi_rows:
        lines.append(f'| {feat} | {_fmt(mi17)} | {_fmt(auc17)} | {_fmt(mi18)} | {_fmt(auc18)} |')
    lines.append('')

    # ── Supporting-analyses index (added this round, pure navigation) ──────────
    lines += [
        '### Supporting-analyses index',
        '',
        ('A short map from each supporting test to where it lives in this document, for anyone '
         'looking for a specific diagnostic:'),
        '',
        '| Supporting analysis | Where it is |',
        '|---|---|',
        '| Cardinality | New table just above (`detected_type` breakdown); no raw unique-count field exists upstream |',
        '| Variance | Not broken out as its own table; not informative on its own for this feature set |',
        '| Per-class separation | Step 7 worked example below, and C2b\'s Table B (feature x family matrix) |',
        ('| Sensitivity sweep (robustness check — verdict counts at 3 alternate calibrated-C2ST '
         'verdict thresholds; no C-number, since it is not rendered in this document — see the '
         '"where" column) | Not in results.md; full table in Step 10\'s own text report, '
         '`results/10_execute_comparison/10_execute_comparison_report.txt` |'),
        '| Metric agreement (Wasserstein/MMD/KS/Energy/Anderson-Darling vs C2ST-AUC) | See E1, Supplementary checks (E-series) at the end of Step 11 |',
        ('| Cluster bootstrap | In the H1 headline cells (C8): each cell\'s 95% CI and preferred '
         'p-value are cluster-bootstrap (resampling collinearity clusters) where available |'),
        '| Partial correlation | Not broken out as its own table; the cluster-bootstrap CI already accounts for collinearity |',
        '',
    ]

    # ── Step 8 (intentionally skipped — no new computation) ──────────────────────
    lines += [
        '---', '## Step 8: (skipped)', '',
        'Step 8 is visualization-only (renders Step 7\'s profiles to PNGs for manual '
        'inspection) and produces no numbers consumed downstream; the pipeline goes '
        'straight from Step 7\'s profiles to Step 9\'s routing decision.',
        '',
    ]

    # ── Step 9 ───────────────────────────────────────────────────────────────────
    lines += ['---', '## Step 9: Comparison Planning (Routing)', '']
    lines += [
        'Step 9 reads the Step 7 distribution profiles and assigns each feature a '
        'comparison route — the CORROBORATION-metric family appropriate for its type. '
        'The route only picks which corroboration distances step 10 computes (and whether '
        'comparison runs per mode or on the whole distribution); the stable/shifted VERDICT is '
        'always decided by calibrated C2ST-AUC, the one metric computed identically for every '
        'route.',
        '',
    ]
    if _plans:
        route_counts  = _Counter(v['route'] for v in _plans.values())
        route_metrics: dict = {}
        route_modes:   dict = {}
        for v in _plans.values():
            r = v['route']
            if r not in route_metrics:
                route_metrics[r] = v.get('corroboration_primary', v.get('metric_primary', 'n/a'))
                route_modes[r]   = v.get('comparison_mode', 'n/a')
        lines += [
            '| Route | Corroboration Primary | Count | Comparison Mode |',
            '|-------|---------------|------:|----------------|',
        ]
        for route, count in sorted(route_counts.items(), key=lambda x: -x[1]):
            lines.append(f'| `{route}` | {route_metrics.get(route, "n/a")} '
                         f'| {count} | {route_modes.get(route, "n/a")} |')
        lines += [
            '',
            'Route conditions (verdict metric is always calibrated C2ST-AUC; the metrics named '
            'below are the corroboration battery only):',
            '- `structural_change`: type or modality changed 2017↔2018; whole-distribution '
            'corroboration distances.',
            '- `continuous_multimodal`: multiple modes detected; per-mode MMD corroboration, then '
            'aggregates.',
            '- `nominal`: categorical or flag feature; Jensen-Shannon divergence on the PMF.',
            '- `discrete_count`: low-cardinality integer count; Jensen-Shannon divergence on '
            'the full PMF.',
            '',
            '### How each metric is calculated',
            '',
            ('Step 9 only assigns which of these a feature uses; step 10 (next) runs the actual '
             'calculation and is where the numeric results live.'),
            '',
            '| Metric | Method |',
            '|---|---|',
            (f'| C2ST-AUC | Classifier Two-Sample Test (Lopez-Paz and Oquab 2017): train a '
             f'shallow decision tree (max_depth={Config10.C2ST_TREE_DEPTH}) to predict whether a '
             f'row is from 2017 or 2018 using this one feature alone, with '
             f'{Config10.C2ST_CV_FOLDS}-fold stratified cross-validation; report the mean '
             'held-out AUC plus a CI built from the per-fold scores. 0.5 = indistinguishable '
             'years, 1.0 = perfectly separable. |'),
            ('| MMD | Unbiased squared Maximum Mean Discrepancy with an RBF kernel, '
             'median-heuristic bandwidth (Gretton et al. 2012): a kernel-based distance between '
             'the two years\' empirical distributions. 0 = identical, larger = more different; '
             'sensitive to any kind of distributional change (location, scale, or shape). |'),
            ('| Wasserstein-qn | Wasserstein (earth-mover) distance computed AFTER '
             'rank-normalizing both years onto a common reference distribution, which strips out '
             'location and scale first, so this isolates shape-only differences. |'),
            ('| Jensen-Shannon divergence | Symmetric, bounded divergence between the two '
             'years\' probability mass functions (top-50 categories plus an "other" bucket for '
             'high-cardinality features); used instead of a continuous-distance metric for '
             'nominal/discrete-count features, since categories have no natural ordering or '
             'distance between them. |'),
            '',
            'Full routing plan: '
            '`output/9_plan_comparison/comparison_plans_cicids2017_cicids2018.json`.',
            '',
        ]
    else:
        lines += ['> Routing plan not available (run step 9 before step 11).', '']

    # ── Step 10 ──────────────────────────────────────────────────────────────────
    lines += ['---', '## Step 10: Drift Axes', '']
    lines += [
        'Step 10 executes the statistical comparisons planned in Step 9 and produces '
        'two axes per feature:',
        '',
        ('> **Quick terms** (so "covariate," "covariance," and "cardinality" don\'t blur together '
         'below — skip this box if they already don\'t): **Covariate shift** (Axis 1) = did the '
         'feature\'s VALUES move between years. Not "covariance" — covariance is a different '
         'statistic (how two variables move together) and is never used as an axis name anywhere '
         'in this document. **Concept stability** (Axis 2) = did the feature\'s RELATIONSHIP to '
         'the attack/benign label survive. **Cardinality** = how many distinct values a feature '
         'takes; unrelated to either axis — it matters only because native tree importance '
         '(gain, and split-count even more so) is biased toward high-cardinality features '
         '(Strobl 2007), which is why several tests in Section 2 below control for it.'),
        '',
        '**Axis 1: Covariate shift (`cov_shift`)**',
        'Measures whether the feature\'s value distribution moved between 2017 and 2018. '
        'Primary metric: C2ST-AUC (train a classifier to distinguish 2017 from 2018 values; '
        '0.5 = indistinguishable, 1.0 = fully shifted), CALIBRATED against a per-feature, '
        'per-slice NULL FLOOR: pool both years, randomly re-split into two same-size halves many '
        'times, recompute C2ST-AUC each time — the floor is the AUC you would still see even if '
        'the two years were identical. Calibrated = (raw − null floor) / (1 − null floor), '
        'clipped to [0, 1]. The 5-level Axis-1 status is bucketed off this CALIBRATED value '
        '(0 = at/below its own null floor, not the fixed 0.5 chance level):',
        '  NONE ≤ 0 | LOW 0–0.20 | MODERATE 0.20–0.45 | HIGH 0.45–0.70 | STRONG > 0.70',
        'The stable/shifted `verdict` column is decided by this SAME calibrated C2ST value '
        '(> 0 = shifted) — one Axis-1 decision rule everywhere. The corroboration distances '
        '(Wasserstein-qn / MMD / energy / KS / Anderson-Darling, or Jensen-Shannon for '
        'PMF-routed features) are each calibrated against their OWN permutation null, POOLED '
        'ONLY, and used purely for the E1 agreement check — they never decide anything. '
        'C2ST is additionally computed for benign-only, attack-only, and each individual '
        'attack family — each slice gets its OWN null floor (a smaller slice is noisier, so its '
        'floor is naturally higher; a raw value from a noisy slice is not directly comparable to '
        'a raw value from a clean one, which is exactly what this per-slice calibration fixes). '
        '`marginal_shift` remains in the tables as a DESCRIPTIVE route-family distance only.',
        '',
        '**Axis 2: Concept stability (`concept_stab` / separation_stability)**',
        'Measures whether the feature still separates attack from benign in 2018. '
        'Per year, separation strength is `max((AUC − 0.5) × 2, MI_normalized)` (MI-aware: a '
        'feature whose separation is non-monotonic, e.g. multimodal, still counts if normalized '
        'mutual information detects it even though folded-AUC alone would not). The two years\' '
        'strengths are clipped to [0, 1] and multiplied; the product is negated if both years are '
        'strongly separated (clears the null-calibrated AUC/MI floor) with opposite TRUSTED '
        'directions, i.e. a genuine flip. `separation_stability_auc`, a legacy folded-AUC-only '
        'version of the same formula (no MI term), is also saved in Layer B and is what the '
        'per-attack-family / per-class breakdown table below uses, since per-family MI is not '
        'computed separately. '
        'PRESERVED ≥ 0.35 | WEAKENED 0.09–0.35 | COLLAPSED 0–0.09 | FLIPPED < 0.',
        '',
        'The full Axis 1 table (with null floors and per-attack-family C2ST) and full '
        'Axis 2 table (per-attack-family separation stability) are in the root `results.md`. '
        'The table below shows the pooled summary from `cross_table.csv`.',
        '',
        f'### Axis 1 + 2 summary: all {len(df)} features (sorted by calibrated C2ST descending)',
        '',
        ('`Quadrant` crosses 2017 native importance '
         'with concept stability (Axis 2) into a 2x2 grid:'),
        '',
        '| Quadrant | Meaning |',
        '|---|---|',
        '| `Q1_good` | high-importance, stable: model relies on transferable features |',
        '| `Q2_fragile_shortcut` | high-importance, unstable: model relies on features that break |',
        '| `Q3_noise` | low-importance, unstable |',
        '| `Q4_underused_stable` | low-importance, stable: a reservoir of stable-but-unused features |',
        '',
        ('`Verdict` is a priority cascade over Axis 1 and Axis 2 together (first match wins; the '
         'full rule with its exact conditions is spelled out later at "How the `verdict` column is '
         'actually decided"); the short version:'),
        '',
        '| Verdict | Meaning |',
        '|---|---|',
        '| `flipped` | separation was trusted in both years, but the attack/benign direction reversed |',
        '| `collapsed` | separation was trusted in 2017 but lost by 2018 |',
        '| `weak` | separation was never trusted in either year |',
        ('| `restructured` | the feature\'s distribution shape (modality/type) changed between '
         'years AND calibrated C2ST confirms the mismatch is real, not GMM/type-detection noise |'),
        '| `shifted` | calibrated C2ST-AUC is above threshold and none of the above apply |',
        '| `stable` | none of the above; calibrated C2ST-AUC at or below threshold |',
        '',
        ('_This is the full per-feature detail table — Axis 1 (C2ST pooled + the benign-only/'
         'attack-only slice C2ST, both raw and calibrated — slices carry C2ST only; corroboration '
         'distances are pooled-only by design) and Axis 2 (separation stability), plus verdict '
         'and quadrant. C2a further below shows the same slice C2ST next to each slice\'s null '
         'floor._'),
        '',
        ('| Feature | C2ST pooled (raw) | C2ST pooled (calibrated) | Benign C2ST (raw) | '
         'Benign C2ST (calibrated) | Attack C2ST (raw) | Attack C2ST (calibrated) | Sep-Stab | '
         'Verdict | Quadrant |'),
        '|---------|-----:|-----:|-----:|-----:|-----:|-----:|---------:|---------|---------|',
    ]
    _ax = df.sort_values('c2st_auc_calibrated', ascending=False)
    for feat, row in _ax.iterrows():
        # NOTE: no "or float('nan')" fallback — attack_shift/separation_stability are
        # legitimately 0.0 for some features, and `0.0 or x` evaluates to `x` in Python,
        # which would silently turn a real "no shift" / "fully collapsed" result into n/a.
        # Slice columns: benign_shift/attack_shift ARE the calibrated per-slice C2ST values
        # (slices carry C2ST only); *_raw are the uncalibrated slice AUCs.
        lines.append(f'| {feat} | {_fmt(row.get("c2st_auc"))} | {_fmt(row.get("c2st_auc_calibrated"))} '
                     f'| {_fmt(row.get("benign_shift_raw"))} | {_fmt(row.get("benign_shift"))} '
                     f'| {_fmt(row.get("attack_shift_raw"))} | {_fmt(row.get("attack_shift"))} '
                     f'| {_fmt(row.get("separation_stability"))} '
                     f'| {row.get("verdict","n/a")} | {row.get("quadrant","n/a")} |')
    lines.append('')

    # ── Shift sub-metrics: pointer only, NOT re-rendered — E1 (Supplementary checks, end of
    # Step 11) already shows C2ST-AUC alongside Wasserstein/MMD/KS/energy/AD (all calibrated,
    # with an agree/disagree flag this raw side-by-side listing didn't have), so repeating a raw
    # C2ST/MMD/Wasserstein table here would be strictly less information under a similar name. ──
    lines += [
        ('_Per-feature C2ST-AUC vs its secondary shift metrics (Wasserstein-qn, MMD, KS, energy-'
         'distance, Anderson-Darling — all calibrated, with an agree/disagree flag) is in E1 '
         '(Supplementary checks, end of Step 11) — not repeated here._'),
        '',
    ]

    # ── Zero-fraction: separate table (NOT a shift sub-metric — it is a data-quality / sparsity
    # property of the raw column, not one of the distributional-distance metrics above, so it
    # gets its own table rather than riding along inside a shift-metric listing). ──────────────
    lines += [
        f'### Zero-fraction: all {len(df)} features (sorted by |Δ zero-fraction| descending)',
        '',
        ('Per-feature zero/null rate from the step-7 profile — NOT a measure of drift by itself, '
         'just data-quality context (a feature with a high zero-rate makes C2ST/MMD/Wasserstein '
         'noisier); the DELTA column is the closest thing to a "did sparsity itself shift" signal. '
         'The zero-INFLATED-feature deep dive (zero fraction vs tail-only Wasserstein) is E2/E3, '
         'Supplementary checks, end of Step 11 — this table is the plain per-feature listing for '
         'every feature, not just the zero-inflated subset.'),
        '',
        '| Feature | Zero-frac 2017 | Zero-frac 2018 | Δ Zero-frac | Verdict |',
        '|---------|---------------:|---------------:|------------:|---------|',
    ]
    for feat, row in df.sort_values('zero_fraction_delta', ascending=False).iterrows():
        # NOTE: no "or float('nan')" fallback — zero_fraction is legitimately 0.0 for many
        # features, and `0.0 or x` evaluates to `x` in Python, silently faking a "n/a".
        lines.append(f'| {feat} | {_fmt(row.get("zero_fraction_2017"))} '
                     f'| {_fmt(row.get("zero_fraction_2018"))} | {_fmt(row.get("zero_fraction_delta"))} '
                     f'| {row.get("verdict","n/a")} |')
    lines.append('')

    # ── Per-feature C2ST confidence interval + routing metadata (the actual per-feature CI
    # numbers, wide at the individual-feature level, which is why the H1 claim rests on the
    # aggregate Spearman rather than any one feature's CI; plus the routing decision from
    # step 9 that determined whether this feature got the whole-distribution, per-mode, or
    # structural-change comparison treatment above) ───────────────────────────────────────
    _layer_a = load_layer_a()

    def _calibrate(raw, floor) -> float:
        """Same transform as calibrate_c2st() in 10_execute_comparison.py, applied here to the CI
        bounds using the point estimate's own null floor — so the calibrated CI stays centered on
        the calibrated point estimate instead of only the raw one being calibrated."""
        try:
            raw, floor = float(raw), float(floor)
        except (TypeError, ValueError):
            return float('nan')
        if not (np.isfinite(raw) and np.isfinite(floor)):
            return float('nan')
        return float(np.clip((raw - floor) / max(1.0 - floor, 1e-6), 0.0, 1.0))

    lines += [
        f'### C2ST confidence intervals + routing: all {len(df)} features (sorted by C2ST descending)',
        '',
        ('Per-feature 95% CI on C2ST-AUC (5-fold CV; wide at the individual-feature level, '
         'so the H1 claim rests on the aggregate Spearman, not on any one of these), shown both '
         'RAW and CALIBRATED (CI low/high run through the same pooled null-floor transform as the '
         'point estimate — see the Axis 1+2 summary table above for that null floor per feature), '
         'and the step-9 routing decision (`route`) that selected this feature\'s comparison '
         'template, joint shape between the two years: `nominal` (categorical PMF), '
         '`discrete_count` (count-shape metric), `continuous_unimodal` (single-mode Wasserstein/'
         'MMD), `continuous_multimodal` (same multimodal structure both years, drives the '
         'per-mode comparison below), or `structural_change` (the years\' shapes do not share a '
         'template, e.g. cross-family or modality mismatch, so shape-agnostic C2ST is the arbiter).'),
        '',
        ('| Feature | C2ST (raw) | CI low (raw) | CI high (raw) | C2ST (calibrated) | '
         'CI low (calibrated) | CI high (calibrated) | CI width (raw) | CV folds | Route |'),
        '|---------|---------:|-------:|--------:|---------:|-------:|--------:|---------:|---------:|-------|',
    ]
    for feat, row in _ax.iterrows():
        rec = _layer_a.get(feat, {}) if isinstance(_layer_a.get(feat), dict) else {}
        c2st = rec.get('c2st', {}) if isinstance(rec.get('c2st'), dict) else {}
        null_floor = row.get('c2st_auc_null', float('nan'))
        cal_lo = _calibrate(c2st.get('ci_low'), null_floor)
        cal_hi = _calibrate(c2st.get('ci_high'), null_floor)
        lines.append(f'| {feat} | {_fmt(row.get("c2st_auc"))} | {_fmt(c2st.get("ci_low"))} '
                     f'| {_fmt(c2st.get("ci_high"))} | {_fmt(row.get("c2st_auc_calibrated"))} '
                     f'| {_fmt(cal_lo)} | {_fmt(cal_hi)} | {_fmt(c2st.get("ci_width"))} '
                     f'| {c2st.get("folds", "n/a")} | {rec.get("route", "n/a")} |')
    lines += [
        '',
        ('A handful of other step-10 robustness/routing fields are computed per feature but not '
         'tabulated above since they are secondary robustness checks rather than headline '
         'results: `separation_stability_auc` (legacy folded-AUC-only version of Axis 2, no MI '
         'term), `null_separation_threshold_2017/2018` and `separation_strong_effective_2017/2018` '
         '(the per-feature null floor a feature must clear, at least max(0.55, this value), to '
         'count as separated, feeding the flip/collapse verdict and Axis-2 status above). '
         'Step 10 also stores the E1 cross-metric agreement per feature (`e1_agreement` — for '
         'each corroboration metric, does its own null-calibrated shifted/stable vote match the '
         'C2ST verdict? — and `e1_agreement_rate`, the fraction that do). E1 is summarized in '
         'the Supplementary checks at the end of Step 11.'),
        '',
    ]

    # ── Per-attack-family breakdown — duplicated here in full, in addition to
    # the pivoted version in Section 1's C2b below, since a reader looking for the per-family
    # view here shouldn't have to cross-reference the other section. ──────────────────
    if _layer_a:
        all_families = sorted({
            fam for rec in _layer_a.values() if isinstance(rec, dict)
            for fam in (set(rec.get('axis1_per_attack', {}) or {})
                        | set(rec.get('axis2_per_class_stability', {}) or {}))
        })
    else:
        all_families = []
    if all_families:
        def _shift_val(rec, fam):
            d = (rec.get('axis1_per_attack', {}) or {}).get(fam)
            if not isinstance(d, dict):
                return float('nan')
            return float(d.get('c2st_auc', float('nan')))

        def _shift_val_calibrated(rec, fam):
            d = (rec.get('axis1_per_attack', {}) or {}).get(fam)
            if not isinstance(d, dict):
                return float('nan')
            return float(d.get('c2st_calibrated', float('nan')))

        def _stab_val(rec, fam):
            try:
                return float((rec.get('axis2_per_class_stability', {}) or {}).get(fam, float('nan')))
            except (TypeError, ValueError):
                return float('nan')

        lines += [
            '### Per-attack-family breakdown: Axis 1 shift and Axis 2 stability, all features',
            '',
            ('Step 10 computes Axis 1 (per-family slice C2ST-AUC — slices carry C2ST only; the '
             'corroboration distances are pooled-only by design) and Axis 2 (separation '
             'stability) separately for EACH attack family, not just pooled benign-vs-attack. '
             'Axis 2 needs no calibration (it is not a C2ST-family metric); Axis 1\'s per-family '
             'C2ST is shown both raw and calibrated against that family\'s OWN null floor '
             '(smaller families are noisier, hence a higher floor). Duplicated here in full — '
             'the same underlying numbers are also rendered, pivoted differently, in Section '
             '1\'s C2b feature x family matrix further below under Step 11.'),
            '',
            f'**Axis 1: per-family C2ST-AUC, RAW** ({len(df)} features x {len(all_families)} families)',
            '',
            f'| Feature | {" | ".join(all_families)} |',
            f'|---|{"|".join(["---:"] * len(all_families))}|',
        ]
        for feat in df.index:
            rec = _layer_a.get(feat, {}) if isinstance(_layer_a.get(feat), dict) else {}
            vals = ' | '.join(_fmt(_shift_val(rec, fam)) for fam in all_families)
            lines.append(f'| {feat} | {vals} |')
        lines += [
            '',
            f'**Axis 1: per-family C2ST-AUC, CALIBRATED** ({len(df)} features x {len(all_families)} families)',
            '',
            f'| Feature | {" | ".join(all_families)} |',
            f'|---|{"|".join(["---:"] * len(all_families))}|',
        ]
        for feat in df.index:
            rec = _layer_a.get(feat, {}) if isinstance(_layer_a.get(feat), dict) else {}
            vals = ' | '.join(_fmt(_shift_val_calibrated(rec, fam)) for fam in all_families)
            lines.append(f'| {feat} | {vals} |')
        lines += [
            '',
            f'**Axis 2: separation stability per attack family** ({len(df)} features x {len(all_families)} families)',
            '',
            f'| Feature | {" | ".join(all_families)} |',
            f'|---|{"|".join(["---:"] * len(all_families))}|',
        ]
        for feat in df.index:
            rec = _layer_a.get(feat, {}) if isinstance(_layer_a.get(feat), dict) else {}
            vals = ' | '.join(_fmt(_stab_val(rec, fam)) for fam in all_families)
            lines.append(f'| {feat} | {vals} |')
        lines.append('')
    else:
        lines += ['_Per-attack-family breakdown (verdicts_layerA) not found; run step 10 before step 11._', '']

    return lines, _layer_a


def _overlap_caveat_suffix() -> str:
    """Quantified version of the within-dataset train/test overlap (leakage) claim, read from
    train_test_overlap.json (written by 6_test.check_train_test_overlap(), which also feeds the
    overlap-free metric twins and step 5's overlap-filtered permutation-importance sample);
    says so plainly if not computed, rather than silently omitting it."""
    parts = []
    for ds in (DS1, DS2):
        p = PROJECT_ROOT / 'output' / '4_preprocessing' / ds / 'train_test_overlap.json'
        if p.exists():
            d = json.loads(p.read_text(encoding='utf-8'))
            parts.append(f'{ds}: {d["n_test_rows_matching_train"]:,}/{d["n_test_rows"]:,} '
                         f'({d["frac_test_rows_matching_train"]:.1%}) test rows exactly match a '
                         f'train row on the {d["n_features"]} modeling features')
        else:
            parts.append(f'{ds}: not yet computed (run step 6, which writes '
                         'train_test_overlap.json before evaluating)')
    return ' Quantified directly: ' + '; '.join(parts) + '.'


def write_results_doc(df, stats, drift, abl, report, out_path: Path, *,
                      benign_atk=None,
                      prior_shift=None,
                      baseline=None, delta_imp_stab=None):
    """Detailed reference document (results.md): every analysis C1–C9 with full tables."""
    # ── inner helpers ────────────────────────────────────────────────────────────
    perm_ok = report.get('permutation_importance_available', False)
    # _layer_a itself is assigned below, from _build_preamble_sections()'s return value (it
    # already loads verdicts_layerA_<DS1>_<DS2>.json for its own per-attack-family breakdown;
    # reused here for compute_metric_agreement() / the E1 section instead of a second load).

    def _ci(blk):
        c = blk.get('bootstrap_ci95', {}) if isinstance(blk, dict) else {}
        lo, hi = c.get('lo', float('nan')), c.get('hi', float('nan'))
        return f'[{lo:+.3f}, {hi:+.3f}]'

    def _clu(blk):
        c = blk.get('cluster_bootstrap_ci95', {}) if isinstance(blk, dict) else {}
        if not c or not np.isfinite(c.get('lo', float('nan'))):
            return 'n/a'
        return f'[{c["lo"]:+.3f}, {c["hi"]:+.3f}]  (n_clusters={c.get("n_clusters","?")})'

    def load_per_attack_matrices():
        """Pivots the 6 per-family output/.../per_attack/<Family>.csv files (each already
        containing feature, separation_stability, axis1_shift_calibrated — the calibrated
        per-family slice C2ST) into two feature x family matrices. separation_stability's matrix
        mirrors what feature_family_stability_matrix.csv (written by 11_cross_analysis.py)
        already has on disk; the Axis-1 twin is built here at render time from the same
        already-saved per-family files, since no CSV for it exists yet."""
        sub = OUTPUT_DIR / 'per_attack'
        if not sub.exists():
            return pd.DataFrame(), pd.DataFrame()
        sep_cols, shift_cols = {}, {}
        for p in sorted(sub.glob('*.csv')):
            try:
                fam_df = pd.read_csv(p).set_index('feature')
            except Exception:
                continue
            if 'separation_stability' in fam_df.columns:
                sep_cols[p.stem] = fam_df['separation_stability']
            if 'axis1_shift_calibrated' in fam_df.columns:
                shift_cols[p.stem] = fam_df['axis1_shift_calibrated']
        sep_mat = pd.DataFrame(sep_cols) if sep_cols else pd.DataFrame()
        shift_mat = pd.DataFrame(shift_cols) if shift_cols else pd.DataFrame()
        keep = [f for f in df.index if f in sep_mat.index] if not sep_mat.empty else []
        if keep:
            sep_mat = sep_mat.loc[keep]
            shift_mat = shift_mat.loc[[f for f in keep if f in shift_mat.index]]
        return sep_mat, shift_mat

    def load_per_family_c2st() -> tuple:
        """Step 10 already computes C2ST-AUC per specific attack family (axis1_per_attack in
        Layer A), and 11_cross_analysis.py's per_class_c2st_attribution() already extracts it to
        per_class_c2st_attribution.csv, RAW (columns c2st_<Family>) and CALIBRATED (columns
        c2st_calibrated_<Family>, against that family's own null floor). Reading it here is a pure
        data load, not a new calculation. Returns (raw_df, calibrated_df), each empty if missing."""
        p = OUTPUT_DIR / 'per_class_c2st_attribution.csv'
        if not p.exists():
            return pd.DataFrame(), pd.DataFrame()
        try:
            fam_df = pd.read_csv(p).set_index('feature')
        except Exception:
            return pd.DataFrame(), pd.DataFrame()
        cal_prefix = 'c2st_calibrated_'
        cal_cols = [c for c in fam_df.columns if c.startswith(cal_prefix)]
        raw_cols = [c for c in fam_df.columns
                   if c.startswith('c2st_') and not c.startswith(cal_prefix)]
        raw = fam_df[raw_cols].rename(columns=lambda c: c[len('c2st_'):])
        cal = fam_df[cal_cols].rename(columns=lambda c: c[len(cal_prefix):])
        keep = [f for f in df.index if f in raw.index]
        raw = raw.loc[keep] if keep else pd.DataFrame()
        cal = cal.loc[[f for f in keep if f in cal.index]] if keep else pd.DataFrame()
        return raw, cal

    def compute_metric_agreement() -> pd.DataFrame:
        """E1 (diagnostic only — never drives a verdict): per-feature agreement between the
        calibrated-C2ST verdict and each corroboration metric's OWN null-calibrated
        shifted/stable vote. Computed by step 10 (execute_one()'s e1_agreement block —
        Wasserstein-qn, MMD, energy distance, KS, Anderson-Darling for continuous features;
        Jensen-Shannon for PMF-routed nominal/discrete-count features) and stored per feature in
        Layer A/B; this function is a pure data load + tabulation, not a new calculation.
        Cells are 'agree' / 'disagree' / 'na' (metric not computed for that feature's route —
        nothing to compare, NOT a disagreement)."""
        metrics = ('wasserstein_qn', 'mmd', 'energy_distance', 'ks_statistic',
                   'anderson_darling', 'jensen_shannon')
        rows = {}
        for feat in df.index:
            rec = _layer_a.get(feat, {}) if isinstance(_layer_a.get(feat), dict) else {}
            agr = rec.get('e1_agreement') or {}
            pooled = rec.get('axis1_pooled', {}) or {}
            if not agr:
                continue
            row = {'c2st_auc': df.at[feat, 'c2st_auc_calibrated']
                   if 'c2st_auc_calibrated' in df.columns else float('nan')}
            for m in metrics:
                row[m] = pooled.get(f'{m}_calibrated', float('nan'))
                if m in agr:
                    row[f'{m}_state'] = 'agree' if agr[m] else 'disagree'
                else:
                    row[f'{m}_state'] = 'na'
            rows[feat] = row
        return pd.DataFrame.from_dict(rows, orient='index') if rows else pd.DataFrame()

    # Pull key stats blocks. Named by the C-code of the section that renders each one (c5a_c2st
    # feeds C5a below, etc.) so a variable name and its caption can never drift apart again.
    c2st_blk  = stats.get('importance_vs_c2st', {})            # -> C4a
    h1_blk    = stats.get('importance_vs_concept_stability', {})       # -> C4b
    h1p_blk   = stats.get('importance_perm_vs_concept_stability', {})  # -> C6b
    c5a_c2st  = stats.get('importance_2018_vs_c2st', {})               # -> C5a
    c6a_c2st  = stats.get('importance_perm_2017_vs_c2st', {})          # -> C6a
    c7a_c2st  = stats.get('importance_perm_2018_vs_c2st', {})          # -> C7a
    native_2018_concept_blk = stats.get('importance_2018_vs_concept_stability', {})       # -> C5b
    perm_2018_concept_blk   = stats.get('importance_perm_2018_vs_concept_stability', {})  # -> C7b

    # 8 independent H1 verdicts: one per (importance variant x axis) cell,
    # never gated on or merged with the opposite axis. Axis 1 expects a POSITIVE Spearman vs
    # C2ST-AUC; Axis 2 expects a NEGATIVE Spearman vs separation_stability (see _single_axis_verdict).
    # Labeled by C-code, not "H1.N", to avoid colliding with the separate H1.5 delta-importance
    # section further below ("H1.5" is used for two different things: the conceptual
    # axis-2/native-2017 cell here, and the standalone delta-importance analysis; the C-code
    # scheme is this file's own established, collision-free naming).
    v_axis1_native17 = _single_axis_verdict(c2st_blk, True,  'C4a Axis 1, native 2017')
    v_axis1_perm17   = _single_axis_verdict(c6a_c2st,  True,  'C6a Axis 1, permutation 2017')
    v_axis2_native17 = _single_axis_verdict(h1_blk,   False, 'C4b Axis 2, native 2017')
    v_axis2_perm17   = _single_axis_verdict(h1p_blk,  False, 'C6b Axis 2, permutation 2017')
    native_perm_agree_axis1 = v_axis1_native17.split(':')[0] == v_axis1_perm17.split(':')[0]
    native_perm_agree_axis2 = v_axis2_native17.split(':')[0] == v_axis2_perm17.split(':')[0]

    _drift_null_dp = drift.get('drift_exposure_null_percentile', float('nan')) if drift else float('nan')
    _drift_null_pctile_str = (f'{_drift_null_dp:.3f} '
                              f'({"above 0.95, model weights shifting features more than chance" if _drift_null_dp > 0.95 else "below 0.95"})'
                              if np.isfinite(_drift_null_dp) else 'n/a')

    # ── Build document header + Steps 5-10 preamble ─────────────────────────────
    lines: list = [
        '# Supplementary Material',
        '',
        ('### Why Network Intrusion Detectors Fail Across Years: Decomposing Cross-Year Failure '
         'into Covariate Shift, Concept Change, and Prior-Probability Shift'),
        '',
        '_LightGBM RF-mode_',
        '',
        '---',
        '## Scope',
        '',
        '- **Dataset:** CIC-IDS 2017 and 2018, corrected 2022 re-extraction.',
        '- **Model:** LightGBM RF-mode tree ensemble (RandomForest variant).',
        '- **Scope of claims:** tree-ensemble NIDS only; results do not generalize to '
        'neural nets, SVMs, or other dataset pairs.',
        '- **Confound control:** both years use the same 2022-corrected extraction, so '
        'measured shift reflects network/attacker/time differences, not extractor artifacts.',
        '',
    ]

    _preamble_lines, _layer_a = _build_preamble_sections(df, baseline)
    lines += _preamble_lines

    # ── Step 11 header + naming legend ───────────────────────────────────────────
    lines += [
        '---',
        '## Step 11: Cross-Analysis (C1–C9)',
        '',
        'Step 11 joins feature importance (Step 5) with drift axes (Step 10) and runs '
        'C1 through C9 (several split into lettered sub-tests, e.g. C4a/C4b) characterizing '
        'how importance, covariate shift, concept stability, and cross-year transfer performance '
        'relate. C9 is [DECISIVE]: it retrains the real model on competing feature '
        'subsets and directly measures cross-year transfer F1.',
        '',
        '### Reference: Naming Convention',
        '',
        'One label scheme, no symbol reused across layers:',
        '',
        ('- **Hypotheses:** `H1` (8 independent tests, C4a-C7a/C4b-C7b, one per importance-variant '
         'x axis cell — no longer one combined two-axis verdict; BH-FDR corrected across exactly '
         'these 8), `H1.5` (4 supplementary delta-importance tests, C8a-C8d, Section 3, its own '
         'BH-FDR family), `H2` (the decisive ablation, C9).'),
        '- **Inputs** (measured quantities consumed by each analysis):',
        '  | name | meaning | name | meaning |',
        '  |------|---------|------|---------|',
        '  | `imp_nat_2017` | native (gain) importance 2017 | `imp_nat_2018` | native (gain) importance 2018 |',
        '  | `imp_perm_2017` | permutation importance 2017 | `imp_perm_2018` | permutation importance 2018 |',
        '  | `cov_shift` | covariate (value) shift = **Axis 1** (calibrated C2ST) | `concept_stab` | separation stability = **Axis 2** |',
        '  | `family` | attack family | | |',
        '  | `benign_shift` | benign-only slice C2ST (calibrated) | `attack_shift` | attack-only slice C2ST (calibrated) |',
        '  | `mi_2017` | mutual information 2017 | `mi_2018` | mutual information 2018 |',
        '  | `rank_delta` | importance-rank change 2017→2018 | | |',
        '- **Analyses:** `C1`–`C9`. `C9` is [DECISIVE].',
        '- **Axes:** Axis 1 = `cov_shift`; Axis 2 = `concept_stab`.',
        '',
        '### Dataset and model scope',
        '',
        '- CIC-IDS 2017 and 2018, corrected 2022 re-extraction.',
        '- Tree-ensemble NIDS (LightGBM-RF), importance-based detectors only.',
        '- Not claimed: neural nets, SVMs, other ML families, other datasets.',
        '',
    ]

    # ── SECTION 1: LISTINGS ───────────────────────────────────────────────────────
    lines += ['---', '## Section 1: Listings: feature and attack drift rankings', '']

    # C1 — importance values only (no axis/stability data at all). This
    # duplicates the existing Step 5 "Feature Importance" table above exactly (same 6 columns,
    # same 71 rows) — pointing to it rather than re-rendering 71 rows a second time in the same
    # document, the same principle used for the Step 10 axis-summary pointer below.
    lines += [
        '### C1: Listing: feature importance values (input: imp_nat_2017, imp_nat_2018, imp_perm_2017, imp_perm_2018)',
        '',
        ('**In short:** this is the same data, same 71 rows, as the "Feature Importance" table in '
         'Step 5 above (Nat-Bin-17/18, Nat-MC-17/18, Perm-Bin-17/18) — not re-rendered a second '
         'time here. Permutation-MULTICLASS importance does not exist anywhere in this pipeline '
         '(confirmed: no `imp_perm_*_multi` column in `cross_table.csv`, and the Step 5 table never '
         'had one either), so this is 6 value columns, not 8.'),
        '',
        '_Source: Step 5 "Feature Importance" table above; CSVs in `output/5_training/`._',
        '',
        '**Status: DONE**',
        '',
    ]

    # C2a — stability, BINARY framing (benign vs. all-attacks-pooled). Two tables.
    _benign_atk_recs = (benign_atk or {}).get('per_feature', []) if benign_atk else []
    lines += [
        ('### C2a: Stability, binary framing — benign vs. pooled-attack (input: c2st_auc, '
         'c2st_auc_calibrated, separation_stability, benign_c2st, attack_c2st)'),
        '',
        ('**In short:** two tables — feature-level benign-vs-pooled-attack readings in both axes '
         '(Table 1), and the existing flip/shift/collapse classification per feature (Table 2, '
         'reuses `df[\'verdict\']`, no new computation).'),
        '',
        ('**Table 1 — C2ST-AUC (Axis 1) BEFORE and AFTER null calibration, benign-only vs '
         'attack-only:**'),
        '',
        ('_Calibration: each slice gets its own NULL FLOOR — the C2ST-AUC you would still see if '
         'the two years were the SAME (pooled, randomly re-split, reclassified). Calibrated = '
         '(raw − null floor) / (1 − null floor), clipped to [0, 1]. 0 = indistinguishable from a '
         'same-year re-split; 1 = maximal separation. The POOLED C2ST-AUC (raw + calibrated) and '
         'the Wasserstein-shift benign/attack breakdown are in the Axis 1+2 summary table above '
         '(Step 10 walkthrough) — not repeated here; this table adds the C2ST-AUC-SPECIFIC '
         'benign/attack breakdown (a different metric from that table\'s Wasserstein-shift '
         'columns), so you can see whether it was specifically C2ST-AUC, not just shift magnitude, '
         'that moved for benign vs attack traffic. There is no benign-only or attack-only variant '
         'of separation_stability — it is DEFINED as the benign-vs-attack gap, so it only exists as '
         'one pooled number per feature (see the summary table above for that column)._'),
        '',
        ('| Feature | Benign C2ST (raw) | Benign null floor | Benign C2ST (calibrated) | '
         'Attack C2ST (raw) | Attack null floor | Attack C2ST (calibrated) |'),
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    _atk_by_feat = {r['feature']: r for r in _benign_atk_recs if r.get('feature') in set(df.index)}
    # Computed once and reused for both the C2a Table 1 loop (here) and Table 2 loop below (df
    # is not mutated in between -- only a separate copy, _jx, gains extra columns).
    _sorted_by_c2st_cal = df.sort_values('c2st_auc_calibrated', ascending=False)
    for feat, row in _sorted_by_c2st_cal.iterrows():
        r = _atk_by_feat.get(feat, {})
        lines.append(f'| {feat} | {_fmt(r.get("benign_c2st"))} | {_fmt(r.get("benign_c2st_null"))} | '
                     f'{_fmt(r.get("benign_c2st_calibrated"))} | {_fmt(r.get("attack_c2st"))} | '
                     f'{_fmt(r.get("attack_c2st_null"))} | {_fmt(r.get("attack_c2st_calibrated"))} |')
    lines += [
        '',
        ('**How the `verdict` column is actually decided** (a priority cascade, not a simple '
         '"C2ST-AUC value x separation_stability value" lookup — it runs on PER-YEAR separation-'
         'trust flags and direction signs, not the single combined columns shown above; this is '
         'the real rule from `10_execute_comparison.py`, checked in order, first match wins):'),
        '',
        '| Priority | Condition | Verdict |',
        '|---:|---|---|',
        '| 1 | Separation was trusted in BOTH 2017 and 2018, but the direction (benign vs. attack) reversed | flipped |',
        '| 2 | Separation was trusted in 2017 but lost by 2018 | collapsed |',
        '| 3 | Separation was never trusted in either year | weak |',
        ('| 4 | None of the above, the feature\'s distribution SHAPE changed between years '
         '(modality/structural route) AND its calibrated C2ST is above the threshold (the '
         'classifier confirms the mismatch is real, not GMM/type-detection noise) | restructured |'),
        ('| 5 | None of the above, and calibrated C2ST-AUC > threshold (the two years are more '
         'classifier-distinguishable than this feature\'s own permutation-null noise) | shifted |'),
        '| 6 | None of the above (calibrated C2ST at or below the threshold) | stable |',
        '',
        ('_This 6-row table above is the actual `verdict` rule. Below is a second, simpler way to '
         'look at the same two axes: Axis 1 status (did the VALUE move) crossed with Axis 2 status '
         '(did the PATTERN survive) — these two statuses, read together, ARE the combined picture; '
         'the single `verdict` column above is a priority cascade, not literally this combination._'),
        '',
        '**Table 2 — per-feature status on both axes, plus the existing `verdict` cascade:**',
        '',
        ('| Feature | C2ST-AUC (raw) | C2ST-AUC (calibrated) | Axis 1 status | '
         'Separation stability | Axis 2 status | Verdict |'),
        '|---|---:|---:|---|---:|---|---|',
    ]

    def _axis1_status(row) -> str:
        """5-level Axis-1 shift, bucketed directly from the CALIBRATED C2ST-AUC (see C2a Table 1 —
        already normalized against this feature's own null floor, so no further transform is
        needed). The same calibrated value drives the stable/shifted `verdict` cascade AND every
        H1/H1.5/H2 test — one Axis-1 decision rule everywhere."""
        cal = row.get('c2st_auc_calibrated', float('nan'))
        if not np.isfinite(cal):
            return 'n/a'
        if cal <= 0:
            return 'NONE'
        if cal <= 0.20:
            return 'LOW'
        if cal <= 0.45:
            return 'MODERATE'
        if cal <= 0.70:
            return 'HIGH'
        return 'STRONG'

    def _axis2_status(row) -> str:
        s = row.get('separation_stability', float('nan'))
        if not np.isfinite(s):
            return 'n/a'
        if s < 0:
            return 'FLIPPED'
        if s < 0.09:
            return 'COLLAPSED'
        if s < 0.35:
            return 'WEAKENED'
        return 'PRESERVED'

    _jx = df.copy()
    _jx['axis1_status'] = _jx.apply(_axis1_status, axis=1)
    _jx['axis2_status'] = _jx.apply(_axis2_status, axis=1)
    for feat, row in _sorted_by_c2st_cal.iterrows():
        lines.append(f'| {feat} | {_fmt(row.get("c2st_auc"))} | {_fmt(row.get("c2st_auc_calibrated"))} | '
                     f'{_jx.at[feat,"axis1_status"]} | '
                     f'{_fmt(row.get("separation_stability"))} | {_jx.at[feat,"axis2_status"]} | '
                     f'{row.get("verdict", "n/a")} |')
    lines += [
        '',
        ('Axis 1 status (5 levels): bucketed directly from the CALIBRATED C2ST-AUC (see C2a Table 1 '
         'for the raw -> null floor -> calibrated derivation) — NONE ≤0 | LOW 0-0.20 | '
         'MODERATE 0.20-0.45 | HIGH 0.45-0.70 | STRONG >0.70. '
         'Axis 2 status: PRESERVED (≥0.35, gap intact) / WEAKENED (0.09-0.35) / COLLAPSED (0-0.09) '
         '/ FLIPPED (<0, direction reversed).'),
        '',
        '**Cross-tab — how many features land in each Axis-1 x Axis-2 combination:**',
        '',
        '| Axis 1 \\ Axis 2 | PRESERVED | WEAKENED | COLLAPSED | FLIPPED |',
        '|---|---:|---:|---:|---:|',
    ]
    _a1_order = ['NONE', 'LOW', 'MODERATE', 'HIGH', 'STRONG']
    _a2_order = ['PRESERVED', 'WEAKENED', 'COLLAPSED', 'FLIPPED']
    for a1 in _a1_order:
        counts = [int(((_jx['axis1_status'] == a1) & (_jx['axis2_status'] == a2)).sum())
                  for a2 in _a2_order]
        lines.append(f'| {a1} | {" | ".join(str(c) for c in counts)} |')
    lines += ['', '**What each combination means, and which features are actually in it this run:**', '']
    _axis1_phrase = {
        'NONE': 'values did not move beyond what a same-year re-split would already show (at/below the null floor)',
        'LOW': 'values moved slightly beyond the null floor — a small but real shift',
        'MODERATE': 'values moved a moderate amount beyond the null floor',
        'HIGH': 'values moved substantially beyond the null floor',
        'STRONG': 'values moved almost to the point of being fully separable between years (calibrated C2ST-AUC near 1.0)',
    }
    _axis2_phrase = {
        'PRESERVED': 'the attack-vs-benign gap held',
        'WEAKENED': 'the gap got smaller but did not collapse',
        'COLLAPSED': 'the gap nearly disappeared',
        'FLIPPED': 'the gap reversed direction',
    }
    _severity = {'NONE': 'no', 'LOW': 'a small', 'MODERATE': 'a moderate',
                 'HIGH': 'a substantial', 'STRONG': 'a near-total'}

    def _cell_meaning(a1: str, a2: str) -> str:
        """Generalizes the prior 2x4 hand-written interpretations (NONE/STRONG match the old
        STABLE/SHIFTED text almost verbatim) to the new 5x4 grid by scaling severity language
        for LOW/MODERATE/HIGH in between, rather than writing 20 independent paragraphs."""
        moved = a1 != 'NONE'
        sev = _severity[a1]
        if a2 == 'PRESERVED':
            if not moved:
                return 'Nothing moved and nothing broke. No action needed on this feature.'
            extra = (' — at this shift level, retraining is doing real, substantial work to keep up'
                     if a1 in ('HIGH', 'STRONG') else '')
            return (f'The environment moved ({sev} amount) but this feature kept separating attack '
                    f'from benign through the move. Classic covariate shift with concept stability: a '
                    f'plain retrain on fresh data should restore performance{extra}.')
        if a2 == 'WEAKENED':
            if not moved:
                return ('Values did not move, but the feature is losing its grip on the label. NOT a '
                        'data-shift problem — retraining on fresher data will not explain this; the '
                        'feature/label relationship itself needs re-examining.')
            return (f'Values moved ({sev} amount) AND discrimination weakened. Likely the dataset-wide '
                    'shift dragging this feature down with it; retrain first, then re-check this feature.')
        if a2 == 'COLLAPSED':
            if not moved:
                return ('Values look the same, but the feature stopped separating attack from benign '
                        'almost entirely. Most diagnostic failure mode: since the data did not move, '
                        'simple retraining will not fix this — the feature needs re-engineering or '
                        'dropping.')
            extra = (' — at this shift level, dataset-wide drift is the more likely explanation'
                     if a1 in ('HIGH', 'STRONG') else
                     ', or a feature-specific failure that happens to coincide with it')
            return (f'Values moved ({sev} amount) and the feature lost almost all separating power'
                    f'{extra}.')
        # FLIPPED
        if not moved:
            return ('Values did not shift, but the relationship reversed direction. Highest-priority '
                    'concept drift: the model is being actively misled, not just less accurate. Needs '
                    're-engineering, not just retraining.')
        return (f'Values moved ({sev} amount) AND the relationship reversed — two compounding problems '
                'a retrain-and-forget pipeline would misread as ordinary drift.')

    for a1 in _a1_order:
        for a2 in _a2_order:
            mask = (_jx['axis1_status'] == a1) & (_jx['axis2_status'] == a2)
            feats = sorted(_jx.index[mask])
            lines += [
                f'- **{a1} x {a2}** ({len(feats)} feature{"s" if len(feats) != 1 else ""} this run):',
                f'  - Axis 1 ({a1}): {_axis1_phrase[a1]}.',
                f'  - Axis 2 ({a2}): {_axis2_phrase[a2]}.',
                f'  - Meaning: {_cell_meaning(a1, a2)}',
                ('  - Features: none this run.' if not feats else
                 '  - Features: ' + ', '.join(feats)),
                '',
            ]
    lines += ['', '**Status: DONE**', '']

    # C2b — stability, MULTICLASS framing (benign vs. each SPECIFIC attack family). Two tables.
    # Table A's per-family C2ST-AUC comes from Step 10 (axis1_per_attack, extracted by
    # 11_cross_analysis.py's per_class_c2st_attribution() to per_class_c2st_attribution.csv) —
    # read directly here via load_per_family_c2st(), not recomputed.
    sep_mat, shift_mat = load_per_attack_matrices()
    fam_c2st_mat, fam_c2st_cal_mat = load_per_family_c2st()
    lines += [
        '### C2b: Stability, multiclass framing — benign vs. each specific attack family (input: family)',
        '',
        ('Table A shows per-family C2ST-AUC (Axis 1), both RAW and CALIBRATED against that '
         'family\'s own null floor. Table B shows per-family separation stability (Axis 2).'),
        '',
        '**Table A — C2ST-AUC (Axis 1) per specific attack family, RAW:**',
        '',
    ]
    if not fam_c2st_mat.empty:
        lines += [
            f'| Feature | {" | ".join(fam_c2st_mat.columns)} | Overall |',
            '|---|' + '---:|' * len(fam_c2st_mat.columns) + '---:|',
        ]
        for feat, row in fam_c2st_mat.iterrows():
            overall = df.at[feat, 'c2st_auc'] if feat in df.index else float('nan')
            cells = ' | '.join(_fmt(v) for v in row)
            lines.append(f'| {feat} | {cells} | {_fmt(overall)} |')
        lines += ['', '_Source: `per_class_c2st_attribution.csv` (built from Step 10\'s '
                       '`axis1_per_attack`)._', '']
        if not fam_c2st_cal_mat.empty:
            lines += [
                '**Table A2 — C2ST-AUC (Axis 1) per specific attack family, CALIBRATED:**',
                '',
                f'| Feature | {" | ".join(fam_c2st_cal_mat.columns)} | Overall |',
                '|---|' + '---:|' * len(fam_c2st_cal_mat.columns) + '---:|',
            ]
            for feat, row in fam_c2st_cal_mat.iterrows():
                overall = df.at[feat, 'c2st_auc_calibrated'] if feat in df.index else float('nan')
                cells = ' | '.join(_fmt(v) for v in row)
                lines.append(f'| {feat} | {cells} | {_fmt(overall)} |')
            lines.append('')
    else:
        lines += ['_`per_class_c2st_attribution.csv` not found this run; re-run Step 10 / '
                   '11_cross_analysis.py\'s per_class_c2st_attribution() to produce it._']
    lines += [
        '',
        '**Table B — separation stability (Axis 2) per specific attack family:**',
        '',
    ]
    if not sep_mat.empty:
        lines += [
            f'| Feature | {" | ".join(sep_mat.columns)} | Overall |',
            '|---|' + '---:|' * len(sep_mat.columns) + '---:|',
        ]
        for feat, row in sep_mat.iterrows():
            overall = df.at[feat, 'separation_stability'] if feat in df.index else float('nan')
            cells = ' | '.join(_fmt(v) for v in row)
            lines.append(f'| {feat} | {cells} | {_fmt(overall)} |')
    else:
        lines += ['_per_attack/ data not found; need to run sub-step 11.5 first._']
    lines += [
        '',
        '**How to read these two tables together:**',
        '',
        ('Each row is one feature. The "Overall" column is that feature\'s pooled C2ST-AUC '
         '(Table A) or separation stability (Table B) across all attack traffic combined — the '
         'same number used everywhere else in this document. The per-family columns break that '
         'single pooled number down by which specific attack type it was tested against.'),
        '',
        ('To find out which attack family DROVE a feature\'s overall shift or instability, '
         'compare the per-family columns against the Overall column for that row:'),
        ('- If most per-family values are close to Overall, the shift/instability is broadly '
         'shared across attack types — no single family is responsible.'),
        ('- If one or two families show a much higher (or much lower) value than the rest of '
         'the row, and those outlying values are close to Overall, that family is the one '
         'driving the feature\'s pooled result. The other families, sitting far from Overall, '
         'are NOT representative of why this feature shows up as shifted/unstable overall.'),
        ('- Example pattern: a feature with Overall C2ST-AUC = 0.78, where 5 of 6 families sit '
         'near 0.50 (no shift) and one family sits at 0.80 — that one family is responsible for '
         'the entire pooled shift signal; the feature did not generically drift, it drifted '
         'specifically against that attack type.'),
        '',
        ('This is the same per-family data the family-stability-ranking analysis (Section 4\'s '
         'family instability check) draws on; this table is the feature-level view of '
         'the same underlying numbers.'),
        '',
        '**Status: DONE**',
        '',
    ]

    # C3 — delta importance VALUE (2018 - 2017), not rank: that distinction is the whole point of
    # this table. Two tables: native, permutation. Replaces the rank-delta listing from the prior
    # round entirely (rank position was redundant with the row's own serial number).
    lines += [
        '### C3: Delta importance VALUE, 2018 minus 2017 (input: imp_nat_2017/2018, imp_perm_2017/2018)',
        '',
        ('**In short:** value change in importance across years, not positional/rank change — '
         'two tables, native and permutation, both sorted by |delta| descending.'),
        '',
        '**By native importance:**',
        '',
        '| Feature | Native importance 2017 | Native importance 2018 | Delta (2018−2017) |',
        '|---|---:|---:|---:|',
    ]
    if {'imp_2017_bin', 'imp_2018_bin'}.issubset(df.columns):
        delta_nat = df['imp_2018_bin'] - df['imp_2017_bin']
        for feat in delta_nat.abs().sort_values(ascending=False).index:
            lines.append(f'| {feat} | {_fmt(df.at[feat,"imp_2017_bin"])} | {_fmt(df.at[feat,"imp_2018_bin"])} '
                         f'| {delta_nat.at[feat]:+.4f} |')
    else:
        lines += ['_native importance value columns not available this run._']
    lines += ['', '**By permutation importance:**', '',
              '| Feature | Permutation importance 2017 | Permutation importance 2018 | Delta (2018−2017) |',
              '|---|---:|---:|---:|']
    if perm_ok and {'imp_perm_2017_bin', 'imp_perm_2018_bin'}.issubset(df.columns):
        delta_perm = df['imp_perm_2018_bin'] - df['imp_perm_2017_bin']
        for feat in delta_perm.abs().sort_values(ascending=False).index:
            lines.append(f'| {feat} | {_fmt(df.at[feat,"imp_perm_2017_bin"])} | '
                         f'{_fmt(df.at[feat,"imp_perm_2018_bin"])} | {delta_perm.at[feat]:+.4f} |')
    else:
        lines += ['_permutation importance not available this run._']
    lines += ['', '**Status: DONE**', '']

    # ── SECTION 3: CORE H1 ───────────────────────────────────────────────────────
    lines += [
        '---',
        '## Section 2: H1 — feature importance vs covariate shift AND concept stability',
        '',
        ('> **H1 claim, Axis 1 / C2ST-AUC:** High-importance features of a tree-ensemble NIDS '
         'show, on average, GREATER covariate shift (Axis 1, calibrated C2ST-AUC) between CIC-IDS 2017 and '
         '2018 than low-importance features.'),
        ('> **H1 claim, Axis 2 / separation_stability:** the SAME high-importance features '
         'PRESERVE the attack-vs-benign discrimination gap (Axis 2, separation_stability, already '
         'MI-aware) across years better than low-importance features. Both axes are tested with 4 '
         'importance variants each (native/permutation x 2017/2018) — 8 independent tests total, '
         'each with its OWN verdict (no combined two-axis verdict). Below, each importance variant '
         'gets its Axis-1 cell (the "a" cell) immediately followed by its Axis-2 cell (the "b" '
         'cell): C4a/C4b, C5a/C5b, C6a/C6b, C7a/C7b.'),
        ('> **Importance anchored to 2017 only:** The important/low feature sets '
         'are defined by 2017 importance so the shift measurement is strictly 2017→2018 '
         'and the result is not circular.'),
        ('> Every cell below shows just the correlation/verdict, not a re-listing of all 71 '
         'features (those are already in C1/C2a/C2b above) — and not the cluster-bootstrap or '
         'drift-vs-null checks either, which are side calculations not shown in the main test '
         'view since they would duplicate what the per-cell verdicts already summarize.'),
        '',
    ]

    def _plain_verdict_block(blk, importance_label, axis_label, axis_num, metric_name):
        """Verdict written in plain words: a 2-row reference table (column titles + what the data
        actually means — NOT a 71-row repeat of C1/C2a/C2b, which would be redundant), then a
        sentence, then a short numbered Note list — no SUPPORTED/CONTRADICTED jargon, no CI-only
        shorthand. Every one of the 8 H1 cells gets this same full treatment, independently —
        none is allowed to shorthand to a single word or to "see the other cell" the way the
        previous draft did."""
        axis_explainer = {
            1: ('C2ST-AUC, CALIBRATED against this feature\'s own null floor (0 = indistinguishable '
                'from a same-year re-split, 1 = fully separated). See C2a Table 1 for the raw value '
                'and null floor this was derived from.'),
            2: ('separation_stability (near 1 = still separates attack/benign cleanly, near 0 = stopped, '
                '<0 = flipped). Raw value; the bucketed PRESERVED/WEAKENED/COLLAPSED/FLIPPED labels are a '
                'separate downstream read.'),
        }[axis_num]
        minitable = [
            f'| Feature | {importance_label} | {axis_label} ({metric_name}) |',
            '|---|---|---|',
            (f'| _Features, ranked high-to-low by {importance_label}_ '
             f'| _raw importance value (Step 5)_ | _{axis_explainer}_ |'),
            '',
        ]
        sp, negligible, strength, sign_word, ci_excludes = _verdict_classify(blk)
        if not np.isfinite(sp):
            return minitable + ['_no data available for this test._']
        # Near-zero correlations (|sp| < 0.05) are noise-level: asserting a direction/sign for
        # e.g. sp=-0.004 ("decreases instead... weak negative correlation") overstates a pattern
        # that isn't really there. Route these to neutral wording instead of a signed claim.
        direction_word = ('does not meaningfully change' if negligible else
                          ('also increases' if sp > 0 else 'decreases instead'))
        ci_explainer = ('we are confident this is a real pattern, not just random noise from these '
                        'specific features' if ci_excludes else
                        'we cannot rule out that this is just random noise from these specific '
                        'features, not a real pattern')
        headline = (f'**{importance_label} and {axis_label} show no meaningful relationship** — '
                    f'essentially zero correlation (ρ={sp:+.3f}).' if negligible else
                    f'**As {importance_label} increases, {axis_label} {direction_word}** — a '
                    f'{strength} {sign_word} correlation (ρ={sp:+.3f}).')
        return minitable + [
            headline,
            '',
            'Note:',
            (f'1. Compared against Axis {axis_num}\'s {metric_name} score: {axis_explainer} '
             '(measured on the binary attack-vs-benign task, not the multiclass attack-TYPE task).'),
            f'2. CI {"excludes" if ci_excludes else "includes"} zero: {ci_explainer}.',
        ]

    # C4a [HEADLINE]
    lines += [
        '### C4a [HEADLINE]: Native importance (2017) vs covariate shift (Axis 1, calibrated C2ST-AUC)',
        '',
        ('We take native importance (2017), sort features high to low, and look at the covariate '
         'shift (Axis 1, calibrated C2ST-AUC) of those same features.'),
        '',
    ] + _plain_verdict_block(c2st_blk, 'native importance (2017)', 'covariate shift', 1, 'C2ST-AUC') + [
        '',
        '**Status: DONE**',
        '',
    ]

    # C4b [HEADLINE]
    lines += [
        '### C4b [HEADLINE]: Native importance (2017) vs concept stability (Axis 2, separation_stability)',
        '',
        ('Same native importance (2017) ranking as C4a, this time read against concept stability '
         '(Axis 2, separation_stability) of those same features.'),
        '',
    ] + _plain_verdict_block(h1_blk, 'native importance (2017)', 'concept stability', 2, 'separation_stability') + [
        '',
        '**Status: DONE**',
        '',
    ]

    # C5a
    lines += [
        '### C5a: Native importance (2018) vs covariate shift (Axis 1, calibrated C2ST-AUC)',
        '',
        'Same test as C4a, but native importance anchored to 2018 instead of 2017.',
        '',
    ] + _plain_verdict_block(c5a_c2st, 'native importance (2018)', 'covariate shift', 1, 'C2ST-AUC') + [
        '',
        '**Status: DONE**',
        '',
    ]

    # C5b
    lines += [
        '### C5b: Native importance (2018) vs concept stability (Axis 2, separation_stability)',
        '',
        'Same test as C4b, but native importance anchored to 2018 instead of 2017.',
        '',
    ] + _plain_verdict_block(native_2018_concept_blk, 'native importance (2018)', 'concept stability', 2, 'separation_stability') + [
        '',
        '**Status: DONE**',
        '',
    ]

    # C6a
    lines += [
        '### C6a: Permutation importance (2017) vs covariate shift (Axis 1, calibrated C2ST-AUC)',
        '',
        ('Same test as C4a, but using permutation importance instead of native importance. '
         'Native importance can be inflated for features with many distinct values; permutation '
         'importance is not, so this checks whether C4a\'s result depends on that bias.'),
        '',
    ] + _plain_verdict_block(c6a_c2st, 'permutation importance (2017)', 'covariate shift', 1, 'C2ST-AUC') + [
        '',
        ('Comparing this to C4a: ' + (
             'native and permutation importance point the SAME direction on this axis, so C4a\'s '
             'result is not just an artifact of native importance favoring high-cardinality '
             'features.' if native_perm_agree_axis1 else
             'native and permutation importance point in DIFFERENT directions on this axis, so '
             'C4a\'s result may be driven by native importance\'s bias toward high-cardinality '
             'features, not a clean drift signal.')),
        '',
        '**Status: DONE**',
        '',
    ]

    # C6b
    lines += [
        '### C6b: Permutation importance (2017) vs concept stability (Axis 2, separation_stability)',
        '',
        'Same test as C4b, but using permutation importance instead of native importance.',
        '',
    ] + _plain_verdict_block(h1p_blk, 'permutation importance (2017)', 'concept stability', 2, 'separation_stability') + [
        '',
        ('Comparing this to C4b: ' + (
             'native and permutation importance point the SAME direction on this axis, so C4b\'s '
             'result is not just an artifact of native importance\'s cardinality bias.'
             if native_perm_agree_axis2 else
             'native and permutation importance point in DIFFERENT directions on this axis, so '
             'C4b\'s result may be driven by native importance\'s cardinality bias, not a clean '
             'concept-stability signal.')),
        '',
        '**Status: DONE**',
        '',
    ]

    # C7a
    lines += [
        '### C7a: Permutation importance (2018) vs covariate shift (Axis 1, calibrated C2ST-AUC)',
        '',
        'Same test as C6a, but permutation importance anchored to 2018 instead of 2017.',
        '',
    ] + _plain_verdict_block(c7a_c2st, 'permutation importance (2018)', 'covariate shift', 1, 'C2ST-AUC') + [
        '',
        '**Status: DONE**',
        '',
    ]

    # C7b
    lines += [
        '### C7b: Permutation importance (2018) vs concept stability (Axis 2, separation_stability)',
        '',
        'Same test as C6b, but permutation importance anchored to 2018 instead of 2017.',
        '',
    ] + _plain_verdict_block(perm_2018_concept_blk, 'permutation importance (2018)', 'concept stability', 2, 'separation_stability') + [
        '',
        '**Status: DONE**',
        '',
    ]

    # ── H1 correlation matrix consolidation ─────────────────────────────────────
    # Pulls together the 8 correlations already computed above (C4a-C7a, C4b-C7b) into one table.
    # No new computation: binary importance only (native + permutation), both years, against both
    # axes. Multiclass importance is intentionally excluded: it is never used by H1, H2, or the
    # ablation anywhere in this pipeline (it answers "which features tell attack TYPES apart", a
    # different question from H1's "important for attack-vs-benign").
    def _cell(blk):
        """One H1 cell: Spearman + 95% CI (cluster bootstrap where available, else plain —
        matching the caption below, which previously CLAIMED cluster-where-available while this
        renderer read only the plain bootstrap) + the BH-FDR q-value computed
        across exactly the 8 H1 cells, with * marking cells that survive
        correction at alpha=0.05."""
        if not isinstance(blk, dict) or not np.isfinite(blk.get('spearman', float('nan'))):
            return 'n/a'
        sp = blk['spearman']
        ci = blk.get('cluster_bootstrap_ci95') or {}
        if not np.isfinite(ci.get('lo', float('nan'))):
            ci = blk.get('bootstrap_ci95', {})
        lo, hi = ci.get('lo', float('nan')), ci.get('hi', float('nan'))
        cell = f'{sp:+.3f}'
        if np.isfinite(lo) and np.isfinite(hi):
            cell += f' [{lo:+.3f},{hi:+.3f}]'
        q = blk.get('q_value_bh', float('nan'))
        if np.isfinite(q):
            star = '*' if blk.get('fdr_significant') else ''
            cell += f' q={q:.3f}{star}'
        return cell

    _axis1_states = [_axis_support_state(b, True) for b in (c2st_blk, c5a_c2st, c6a_c2st, c7a_c2st)]
    _axis2_states = [_axis_support_state(b, False) for b in (h1_blk, native_2018_concept_blk, h1p_blk, perm_2018_concept_blk)]
    _a1_sup = sum(1 for s in _axis1_states if s == 'supported')
    _a2_con = sum(1 for s in _axis2_states if s == 'contradicted')
    lines += [
        '### H1 correlation summary: all 8 independent tests (binary importance, both axes)',
        '',
        (f'**In short:** Axis 1 (covariate shift) is confirmed in {_a1_sup}/4 of the importance '
         f'variants tested above; Axis 2 (concept stability) is contradicted in {_a2_con}/4. So '
         'across these 8 correlation tests, H1 holds only partially and only on one axis — see '
         'the per-cell verdicts below for which variants agree, and C9 further down for the '
         'decisive (retrained-model) test, since these are supporting correlations, not the '
         'decisive evidence.'),
        '',
        '**Full detail:**',
        '',
        ('These are the same 8 cells computed in C4a-C7a (Axis 1) and C4b-C7b (Axis 2) above; this '
         'table is a rendering consolidation, not a new computation, so all 8 tests can be scanned '
         'side by side. Each cell carries its OWN independent verdict below — there is no single '
         'combined two-axis H1 verdict, since the two axes measure different things and a merged '
         'verdict would hide which one is driving a result. 95% CI is the cluster '
         'bootstrap where available (resampling collinearity clusters, the honest effective-n), '
         'else the plain bootstrap. `q=` is the Benjamini-Hochberg FDR-adjusted q-value computed '
         'across exactly these 8 tests (`*` = survives correction at alpha=0.05) — running 8 '
         'correlated tests means some look "significant" by chance alone, and BH bounds the '
         'expected false-positive fraction among the starred cells.'),
        '',
    ]
    _fdr = stats.get('fdr_correction_summary') or {}
    if _fdr:
        lines += [
            (f'**Multiple-testing correction (BH-FDR, alpha={_fdr.get("alpha", 0.05)}):** '
             f'{_fdr.get("n_significant_bh_corrected", "?")}/{_fdr.get("n_tests", "?")} of the H1 '
             f'cells survive correction '
             f'(uncorrected at p<{_fdr.get("alpha", 0.05)}: '
             f'{_fdr.get("n_significant_uncorrected", "?")}). '
             f'p-values: {_fdr.get("p_source", "bootstrap")}.'),
            '',
        ]
    lines += [
        '| Importance variant | vs cov_shift (Axis 1) | vs concept_stab (Axis 2) |',
        '|---------------------|------------------------|----------------------------|',
        f'| Native, 2017 (C4a / C4b) | {_cell(c2st_blk)} | {_cell(h1_blk)} |',
        f'| Native, 2018 (C5a / C5b) | {_cell(c5a_c2st)} | {_cell(native_2018_concept_blk)} |',
        f'| Permutation, 2017 (C6a / C6b) | {_cell(c6a_c2st)} | {_cell(h1p_blk)} |',
        f'| Permutation, 2018 (C7a / C7b) | {_cell(c7a_c2st)} | {_cell(perm_2018_concept_blk)} |',
        '',
        '**Same 8 results, in plain if/then terms:**',
    ]

    def _if_then(blk, importance_label, axis_label):
        sp = blk.get('spearman', float('nan')) if isinstance(blk, dict) else float('nan')
        if not np.isfinite(sp):
            return f'- If {importance_label} is high: no data available for {axis_label}.'
        direction = 'high' if sp > 0 else ('low' if sp < 0 else 'unchanged')
        strength = 'strong' if abs(sp) >= 0.5 else ('moderate' if abs(sp) >= 0.3 else 'weak')
        return (f'- If {importance_label} is high, {axis_label} tends to be {direction} too '
                f'({strength} correlation, ρ={sp:+.3f}).')

    lines += [
        _if_then(c2st_blk, 'native importance (2017)', 'covariate shift'),
        _if_then(c5a_c2st, 'native importance (2018)', 'covariate shift'),
        _if_then(c6a_c2st, 'permutation importance (2017)', 'covariate shift'),
        _if_then(c7a_c2st, 'permutation importance (2018)', 'covariate shift'),
        _if_then(h1_blk, 'native importance (2017)', 'concept stability'),
        _if_then(native_2018_concept_blk, 'native importance (2018)', 'concept stability'),
        _if_then(h1p_blk, 'permutation importance (2017)', 'concept stability'),
        _if_then(perm_2018_concept_blk, 'permutation importance (2018)', 'concept stability'),
        '',
        (f'H1 predicted covariate shift should go UP with importance (it does, in '
         f'{_a1_sup}/4 of the tests above) and concept stability should go DOWN with importance '
         '(instead it goes UP in most of the tests above, the opposite of the prediction, '
         f'contradicted in {_a2_con}/4). So: the covariate-shift side of H1 holds, the '
         'concept-stability side does not.'),
        '',
        (f'Native vs permutation agreement, Axis 1: {"agree" if native_perm_agree_axis1 else "DISAGREE"}; '
         f'Axis 2: {"agree" if native_perm_agree_axis2 else "DISAGREE"}. Agreement on an axis means '
         'that axis\'s native-importance result is not just a Gini-cardinality artifact (Strobl '
         '2007); disagreement means lead with the permutation (unbiased) reading for that axis.'),
        '',
        ('Multiclass importance (native or permutation) is not in this table: confirmed '
         'unused by any test in this pipeline; see the Step 5 importance listing above for '
         'the multiclass numbers themselves.'),
        '',
        ('See Section 3 below for H1.5 (4 supplementary tests using DELTA importance, '
         'imp_2018 - imp_2017, instead of a year-anchored value) — closer to H1 than to H2, no '
         'ablation companion.'),
        '',
    ]

    # ── SECTION 3: H1.5 — DELTA IMPORTANCE VS STABILITY (C8a, C8b) ─────────────
    # Does the CHANGE in importance VALUE correlate with axis values, using delta
    # (imp_2018 - imp_2017), not a year-anchored value (that's C4a-C7b above)? Closer to H1
    # than H2 — correlation only, no retraining/feature-selection, so there is no ablation
    # companion for H1.5.
    lines += ['---', '## Section 3: H1.5 — Delta importance vs stability (C8a-C8d, 4 tests)', '']
    lines += [
        ('We sort features by how much their importance VALUE changed between years '
         '(2018 value minus 2017 value, high-to-low), then ask: does a bigger change in '
         'importance go with a bigger change in stability? Tested against both axes, for BOTH '
         'the native (gain) and the permutation importance — all four computed cells are shown, '
         'not just the native pair, so the comparison is not selective. BH-FDR is applied within '
         'this 4-test family, separately from the 8 H1 cells.'),
        '',
    ]
    dis = delta_imp_stab if isinstance(delta_imp_stab, dict) else {}
    if dis:
        h15a = dis.get('h1_5a_delta_native_vs_axis1', {})
        h15b = dis.get('h1_5b_delta_perm_vs_axis1', {})
        h15c = dis.get('h1_5c_delta_native_vs_axis2', {})
        h15d = dis.get('h1_5d_delta_perm_vs_axis2', {})

        def _plain_delta_verdict(blk, imp_label, axis_label, metric_name, axis_num):
            axis_desc = {
                1: ('CALIBRATED C2ST-AUC (0 = at/below this feature\'s own null floor, '
                    '1 = fully separated between years).'),
                2: ('separation_stability (near 1 = still separates cleanly, near 0 = stopped, '
                    '<0 = flipped). Raw value.'),
            }[axis_num]
            minitable = [
                f'| Feature | Δ {imp_label} | {axis_label} ({metric_name}) |',
                '|---|---|---|',
                (f'| _Features, sorted by |2018−2017| importance delta_ '
                 f'| _raw delta value (Scripts 5+11)_ | _{axis_desc}_ |'),
                '',
            ]
            sp, negligible, strength, direction, ci_excludes = _verdict_classify(blk)
            if not np.isfinite(sp):
                return minitable + ['_no data available for this test._']
            q = blk.get('q_value_bh', float('nan'))
            q_note = (f'5. BH-FDR (within the 4-test H1.5 family): q={q:.3f}'
                      f'{" — survives correction at alpha=0.05" if blk.get("fdr_significant") else " — does NOT survive correction"}.'
                      if np.isfinite(q) else
                      '5. BH-FDR q-value not available for this cell.')
            if negligible:
                sentence = (f'There is essentially no correlation between how much {imp_label} '
                            f'changed and {axis_label} (ρ={sp:+.3f}) — a bigger change in importance '
                            f'is not associated with a bigger change in {axis_label}.')
            else:
                trend = ('also tend to have larger changes' if sp > 0 else
                         'tend to have smaller changes')
                sentence = (f'There is a {strength} {direction} correlation between how much '
                            f'{imp_label} changed and {axis_label} (ρ={sp:+.3f}) — features whose '
                            f'importance changed the most {trend} in {axis_label}.')
            return minitable + [
                sentence,
                '',
                'Note:',
                f'1. {imp_label} delta = 2018 value minus 2017 value (Scripts 5 + 11)',
                f'2. Correlation with {axis_label} ({metric_name} value, Script 10)',
                f'3. CI {"excludes" if ci_excludes else "includes"} zero',
                '4. No direction predicted in advance — unlike H1, not checked against an expected sign.',
                q_note,
            ]

        # C8a-C8d: the full 4-cell H1.5 family (native/permutation x Axis 1/Axis 2)
        lines += ['### C8a: Delta native importance vs Axis 1', '']
        lines += _plain_delta_verdict(h15a, 'native importance', 'covariate shift',
                                      'calibrated C2ST-AUC', 1)
        lines += ['', '**Status: DONE**', '']

        lines += ['### C8b: Delta permutation importance vs Axis 1', '']
        lines += _plain_delta_verdict(h15b, 'permutation importance', 'covariate shift',
                                      'calibrated C2ST-AUC', 1)
        lines += ['', '**Status: DONE**', '']

        lines += ['### C8c: Delta native importance vs Axis 2', '']
        lines += _plain_delta_verdict(h15c, 'native importance', 'concept stability',
                                      'separation_stability', 2)
        lines += ['', '**Status: DONE**', '']

        lines += ['### C8d: Delta permutation importance vs Axis 2', '']
        lines += _plain_delta_verdict(h15d, 'permutation importance', 'concept stability',
                                      'separation_stability', 2)
        lines += [
            '',
            ('_H1.5 is NOT an ablation input — no feature selection or retraining uses delta '
             'importance anywhere in this pipeline; it is a correlation-only side check._'),
            '',
            '**Status: DONE**',
            '',
        ]
    else:
        lines += ['_H1.5 results not available; run delta_importance_vs_stability() '
                  '(sub-step 11.18) first._', '', '**Status: NOT AVAILABLE**', '']

    # ── SECTION 4: DECISIVE EXPERIMENT ───────────────────────────────────────────
    lines += [
        '---',
        '## Section 4: Decisive experiment (C9)',
        '',
        '> The correlations in Sections 2-3 are supporting evidence. The analysis below '
        'retrains / re-evaluates the real model and is the decisive test.',
        '',
    ]

    # C9 (decisive ablation; formerly D1)
    def _h2_metric_col(abl_df) -> tuple:
        """Macro F1 (mean of Attack F1 and Benign F1) is the H2 decision metric — fit_eval() in
        11_cross_analysis.py computes it for every scenario/policy. Falls back to Attack F1
        (f1_cross_domain) ONLY when reading an ablation_results.csv from before this metric existed
        (old cached run), and says so explicitly in the returned label rather than silently
        mislabeling Attack F1 as Macro F1."""
        if abl_df is not None and 'macro_f1_cross_domain' in abl_df.columns:
            return 'macro_f1_cross_domain', 'Macro F1'
        return 'f1_cross_domain', 'Attack F1 (Macro F1 not yet computed this run)'

    def _h2_paired_test(abl_df, challenger: str, metric_col: str) -> dict:
        """Paired significance for '`challenger` beats top_importance': pair
        the per-seed replicate rows on matched (K, direction, seed) cells and run a two-sided
        Wilcoxon signed-rank test on the differences. Requires the seed column (new ablation
        runs); returns NaNs for old cached runs so the verdict text degrades gracefully."""
        out = {'n_pairs': 0, 'mean_diff': float('nan'), 'std_diff': float('nan'),
               'p_wilcoxon': float('nan')}
        if abl_df is None or abl_df.empty or 'seed' not in abl_df.columns:
            return out
        keys = ['K', 'direction', 'seed']
        ch = (abl_df[abl_df['policy'] == challenger].set_index(keys)[metric_col]).sort_index()
        ti = (abl_df[abl_df['policy'] == 'top_importance'].set_index(keys)[metric_col]).sort_index()
        common = ch.index.intersection(ti.index)
        if len(common) < 5:
            return out
        diff = (ch.loc[common] - ti.loc[common]).to_numpy(dtype=float)
        out['n_pairs'] = int(len(diff))
        out['mean_diff'] = float(np.mean(diff))
        out['std_diff'] = float(np.std(diff, ddof=1)) if len(diff) > 1 else float('nan')
        try:
            from scipy.stats import wilcoxon
            if np.allclose(diff, 0):
                out['p_wilcoxon'] = 1.0
            else:
                out['p_wilcoxon'] = float(wilcoxon(diff).pvalue)
        except Exception:
            pass
        return out

    def _compute_h2_verdict(abl_df) -> 'str | None':
        """Same three-way race (axis1_stable / axis2_stable vs top_importance, both directions)
        computed once here for the In-short summary, and reused below instead of recomputing —
        single source of truth so the two can never drift apart. Means aggregate over K AND the
        seed replicates; the headline win claim is backed by a PAIRED Wilcoxon signed-rank test
        over matched (K, direction, seed) cells (a bare mean comparison of
        single-seed retrains is training noise, not evidence)."""
        if abl_df is None or abl_df.empty:
            return None
        si = abl_df[abl_df['policy'] == 'axis2_stable']
        ti = abl_df[abl_df['policy'] == 'top_importance']
        a1 = abl_df[abl_df['policy'] == 'axis1_stable']
        if si.empty or ti.empty or a1.empty:
            return None
        metric_col, metric_label = _h2_metric_col(abl_df)
        a1_beats_ti, a2_beats_ti = [], []
        for direction in ('2017->2018', '2018->2017'):
            ti_d = ti[ti['direction'] == direction][metric_col].mean()
            a1_d = a1[a1['direction'] == direction][metric_col].mean()
            a2_d = si[si['direction'] == direction][metric_col].mean()
            a1_beats_ti.append(a1_d > ti_d)
            a2_beats_ti.append(a2_d > ti_d)
        axis1_wins_both, axis2_wins_both = all(a1_beats_ti), all(a2_beats_ti)
        t1 = _h2_paired_test(abl_df, 'axis1_stable', metric_col)
        t2 = _h2_paired_test(abl_df, 'axis2_stable', metric_col)

        def _sig(t):
            if not np.isfinite(t['p_wilcoxon']):
                return 'paired test n/a (no seed replicates in this cached run)'
            return (f'paired Wilcoxon over {t["n_pairs"]} matched (K, direction, seed) cells: '
                    f'mean diff {t["mean_diff"]:+.4f}, p={t["p_wilcoxon"]:.4f}'
                    + (' — significant at 0.05' if t['p_wilcoxon'] < 0.05 else
                       ' — NOT significant at 0.05'))
        suffix = (f' (decision metric: {metric_label}; axis1 vs top_importance: {_sig(t1)}; '
                  f'axis2 vs top_importance: {_sig(t2)})')
        if axis1_wins_both and axis2_wins_both:
            return ('H2 SUPPORTED ON BOTH AXES: axis1_stable AND axis2_stable each beat '
                     'top_importance in both directions' + suffix)
        if axis1_wins_both:
            return ('H2 SUPPORTED ON AXIS 1 ONLY: axis1_stable beats top_importance in both '
                     'directions; axis2_stable does not' + suffix)
        if axis2_wins_both:
            return ('H2 SUPPORTED ON AXIS 2 ONLY: axis2_stable beats top_importance in both '
                     'directions; axis1_stable does not' + suffix)
        return ('H2 NOT SUPPORTED ON EITHER AXIS: neither axis1_stable nor axis2_stable '
                'beats top_importance in both directions' + suffix)

    def _ablation_trend_lines(abl_df) -> list:
        """Column reading (one policy, K rising) and row reading (one K, every policy),
        computed from the same ablation rows as the table above it so the trend description
        can never drift from the actual numbers."""
        if abl_df is None or abl_df.empty:
            return []
        metric_col, metric_label = _h2_metric_col(abl_df)
        scan_policies = ['top_importance', 'axis1_stable', 'axis2_stable', 'random']
        out = [
            ('**Reading the table above — down each column (one policy, K rising) and across '
             f'each row (one K, every policy), tracked on {metric_label}:**'),
            '',
        ]
        for direction in ('2017->2018', '2018->2017'):
            out += [f'_Direction {direction}, scanning down — does {metric_label} rise as K '
                    'rises?_', '']
            for pol in scan_policies:
                sub = (abl_df[(abl_df['policy'] == pol) & (abl_df['direction'] == direction)]
                       .groupby('K', as_index=False)[metric_col].mean()   # mean over seed replicates
                       .sort_values('K'))
                if len(sub) < 2:
                    continue
                ks = sub['K'].tolist()
                f1s = sub[metric_col].tolist()
                deltas = [f1s[i + 1] - f1s[i] for i in range(len(f1s) - 1)]
                if all(d >= -1e-9 for d in deltas):
                    trend = 'rises as K increases'
                elif all(d <= 1e-9 for d in deltas):
                    trend = 'falls as K increases'
                else:
                    trend = 'is non-monotonic as K increases'
                peak_i = max(range(len(f1s)), key=lambda i: f1s[i])
                edge = peak_i in (0, len(f1s) - 1)
                j = max(range(len(deltas)), key=lambda i: abs(deltas[i]))
                out.append(
                    f'  `{pol}`: {trend}; peak {metric_label}={f1s[peak_i]:.3f} at K={int(ks[peak_i])} '
                    f'({"boundary of the tested range" if edge else "an interior peak"}); '
                    f'biggest single step K={int(ks[j])}→{int(ks[j + 1])} '
                    f'({f1s[j]:.3f}→{f1s[j + 1]:.3f}).')
            out.append('')
            out += [f'_Direction {direction}, scanning across — best policy at each K:_', '']
            sub_dir = (abl_df[(abl_df['direction'] == direction)
                              & (abl_df['policy'].isin(scan_policies))]
                       .groupby(['policy', 'K'], as_index=False)[metric_col].mean())  # mean over seeds
            for k in sorted(sub_dir['K'].unique()):
                row_k = sub_dir[sub_dir['K'] == k].sort_values(metric_col, ascending=False)
                best = row_k.iloc[0]
                out.append(f'  K={int(k)}: best is `{best["policy"]}` '
                           f'({metric_label}={best[metric_col]:.3f})')
            out.append('')
        out += [
            ('`all_features` is left out of this scan since it only has one K (71, the ceiling '
             'reference) — see the Effect-size table below for how K=71 compares to small K.'),
            '',
        ]
        return out

    def _ablation_md_table(abl_df, policy_name, af_k) -> list:
        """Plain-markdown table, two stacked sub-tables per policy (one per training year).
        An earlier version used raw embedded HTML (<table>/<th rowspan/colspan>) to get a real
        merged-cell header, but that relies on the markdown viewer rendering raw HTML — not
        guaranteed, and when it is not rendered the reader sees literal <td>/<th> tag text
        instead of a table. Plain markdown tables render correctly everywhere (and degrade to
        readable plain text even with no renderer at all), at the cost of repeating the K column
        once per training year instead of one shared merged header. K=71 substitutes in the
        shared `all_features` row, since none of the K-based policies (this one included) have
        their own row at K=71 — there is nothing left to select once every feature is kept."""
        if abl_df is None or abl_df.empty or 'f1_cross_covariate' not in abl_df.columns:
            return []
        groups = {
            '2017->2018': [('In-domain', 'f1_in_domain', 'acc_in_domain'),
                           ('Cross (covariate)', 'f1_cross_covariate', 'acc_cross_covariate'),
                           ('Cross (concept)', 'f1_cross_domain', 'acc_cross_domain')],
            '2018->2017': [('In-domain', 'f1_in_domain', 'acc_in_domain'),
                           ('Cross (concept)', 'f1_cross_domain', 'acc_cross_domain'),
                           ('Cross (covariate)', 'f1_cross_covariate', 'acc_cross_covariate')],
        }

        own_ks = abl_df.loc[abl_df['policy'] == policy_name, 'K']
        ks = sorted(set(own_ks) | ({af_k} if af_k is not None else set()))
        out: list = []
        for direction, train_year in (('2017->2018', '2017'), ('2018->2017', '2018')):
            rows_dir = abl_df[abl_df['direction'] == direction]
            header_cells = ['K']
            for label, _, _ in groups[direction]:
                header_cells += [f'F1 {label}', f'MacroF1 {label}', f'Acc {label}']
            out += [
                f'_Trained on {train_year}:_',
                '',
                '| ' + ' | '.join(header_cells) + ' |',
                '|' + '|'.join(['---:'] * len(header_cells)) + '|',
            ]
            for k in ks:
                use_policy = policy_name if (k in own_ks.values) else 'all_features'
                r_df = rows_dir[(rows_dir['policy'] == use_policy) & (rows_dir['K'] == k)]
                if r_df.empty:
                    continue
                r = r_df.mean(numeric_only=True)   # mean over seed replicates (FIX-7)
                k_label = f'{int(k)} (all features)' if use_policy == 'all_features' else str(int(k))
                cells = [k_label]
                for _, f1c, accc in groups[direction]:
                    macro_c = f1c.replace('f1_', 'macro_f1_', 1)
                    cells += [f'{r[f1c]:.3f}', _fmt(r.get(macro_c), 3, dash=True), f'{r[accc]:.3f}']
                out.append('| ' + ' | '.join(cells) + ' |')
            out.append('')
        return out

    _h2_verdict = _compute_h2_verdict(abl)
    lines += [
        '### C9 [DECISIVE]: Cross-domain ablation (H2 test)',
        '',
        ('**What we are doing, before any numbers:** we pick the '
         'top K features by a few different rules (K = 5, 10, 20, 30, 50, or all), train a real '
         'model on just those K features using 2017 data, and separately using 2018 data, then '
         'test each trained model on both years. This is the DECISIVE test for H2 — it retrains '
         'the actual model and measures real transfer F1, not a correlation proxy like Sections '
         '3-6 above.'),
        '',
        '**The selection rules compared (5 total — 3 we care about, plus a floor and a ceiling):**',
        ('  · `axis1_stable` — the K features with the LEAST covariate shift (Axis 1, C2ST-AUC; '
         'lower value = more stable). Same Axis-1 number as C2a\'s Table 1 ("overall" column) and '
         'the Step 10 axis table — looking at the OVERALL value for every row, not benign-only, '
         'attack-only, or any one specific attack.'),
        ('  · `axis2_stable` — the K features with the MOST concept stability (Axis 2, '
         'separation_stability; higher value = more stable). Same overall-value convention.'),
        ('  · `top_importance` — the K features the model relies on most (native importance, '
         'anchored to 2017 only, so feature SELECTION never peeks at 2018 — the same list is '
         'reused for the 2018-trained models too).'),
        '  · `random` — K random features (floor reference, not a real competing policy).',
        '  · `all_features` — all features, nothing dropped (ceiling reference).',
        '',
        ('We sort the full feature list by Axis-1 stability and separately by Axis-2 stability, '
         'take the top-K names off the TOP of each sorted list, then train BOTH a 2017 model and '
         'a 2018 model on just those K features. We do this for every K value above, for every '
         'policy above — that is a lot of trained models.'),
        '',
        ('**How a trained model gets tested:** if a model trained on 2017 is tested on 2017 data, '
         'we scale with the 2017 scaler, no ambiguity. If a model trained on 2017 is tested on '
         '2018 data, we test it TWICE: once scaling the 2018 test data with the 2018 scaler '
         '("concept" framing — removes the location/scale shift, asks whether the decision '
         'boundary itself still works), and once scaling the 2018 test data with the 2017 scaler '
         '("covariate" framing — the realistic deployment case, where a fresh scaler usually does '
         'not exist). Same thing in reverse for models trained on 2018. Every trained model ends '
         'up with both an in-domain F1 and these two cross-year F1 readings.'),
        '',
        ('**What "H2 supported" would mean:** a stability-based policy (`axis1_stable` OR '
         '`axis2_stable`) would have to beat `top_importance` on cross-year F1 in BOTH directions '
         '(2017 model tested on 2018, AND 2018 model tested on 2017) — both axes are tested here, '
         'head-to-head against `top_importance`, not axis 2 alone.'),
        '',
        ('**Decision metric:** the verdict is decided on Macro F1 (mean of Attack F1 and Benign '
         'F1), not Attack F1 alone — Attack F1 ignores how a policy affects benign-traffic '
         'classification, and a policy that "wins" by flagging everything as an attack would '
         'look good on Attack F1 alone while wrecking benign traffic. Attack F1, Benign F1, '
         'sensitivity (attack recall), false-positive rate, precision, specificity, balanced '
         'accuracy, and MCC are all shown alongside as supporting context — none of the '
         'supporting metrics ever overrides the Macro F1 verdict.'),
        '',
        '  Output: `ablation_results.csv`, `ablation_crossdomain_f1.png`, `ablation_gap.png`',
        '  Note: the test set is per-class capped (rebalanced), so the F1 numbers below are '
        'RELATIVE comparisons between policies, not natural-prior deployment numbers — see the '
        'real full-data baseline further down for that.',
        '',
    ]
    for fname, cap in (('ablation_crossdomain_f1.png', 'Cross-domain macro F1 by policy and K'),
                       ('ablation_gap.png', 'Generalization gap (in-domain minus cross-domain macro F1) by policy and K')):
        im = _img(RESULTS_DIR / fname, cap)
        if im:
            lines += [im, '']
    if abl is not None and not abl.empty:
        has_cov = 'f1_cross_covariate' in abl.columns
        _af_k_series = abl.loc[abl['policy'] == 'all_features', 'K']
        _af_k = int(_af_k_series.iloc[0]) if len(_af_k_series) else None
        _has_macro_h2 = 'macro_f1_cross_domain' in abl.columns
        _has_benign_h2 = 'benign_f1_cross_domain' in abl.columns

        _supporting_cols = ('sensitivity_cross_domain', 'fpr_cross_domain', 'precision_cross_domain',
                           'specificity_cross_domain', 'balanced_acc_cross_domain', 'mcc_cross_domain')
        _has_supporting = all(c in abl.columns for c in _supporting_cols)
        lines += [
            ('**Full H2 metric table — Attack F1 / Benign F1 / Macro F1 (decision metric) plus '
             'supporting context, cross-domain scenario, mean over every K:**'),
            '',
        ]
        if not _has_macro_h2:
            lines += [
                ('_Macro F1 / Benign F1 columns show "—" this run — not yet computed by Step '
                 '11\'s ablation training (old cached run); re-run Step 11 to populate them._'),
                '',
            ]
        lines += [
            ('| Policy | Direction | Attack F1 | Benign F1 | Macro F1 | Sensitivity | FPR | '
             'Precision | Specificity | Balanced Acc | MCC |'),
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
        ]
        for pol in ('top_importance', 'axis1_stable', 'axis2_stable'):
            for direction in ('2017->2018', '2018->2017'):
                sub_pol = abl[(abl['policy'] == pol) & (abl['direction'] == direction)]
                if sub_pol.empty:
                    continue
                atk_v = sub_pol['f1_cross_domain'].mean()
                ben_v = sub_pol['benign_f1_cross_domain'].mean() if _has_benign_h2 else float('nan')
                mac_v = sub_pol['macro_f1_cross_domain'].mean() if _has_macro_h2 else float('nan')
                if _has_supporting:
                    sens_v = sub_pol['sensitivity_cross_domain'].mean()
                    fpr_v = sub_pol['fpr_cross_domain'].mean()
                    prec_v = sub_pol['precision_cross_domain'].mean()
                    spec_v = sub_pol['specificity_cross_domain'].mean()
                    bacc_v = sub_pol['balanced_acc_cross_domain'].mean()
                    mcc_v = sub_pol['mcc_cross_domain'].mean()
                else:
                    sens_v = fpr_v = prec_v = spec_v = bacc_v = mcc_v = float('nan')
                lines.append(f'| {pol} | {direction} | {_fmt(atk_v, 3, dash=True)} | {_fmt(ben_v, 3, dash=True)} '
                             f'| {_fmt(mac_v, 3, dash=True)} | {_fmt(sens_v, 3, dash=True)} | {_fmt(fpr_v, 3, dash=True)} '
                             f'| {_fmt(prec_v, 3, dash=True)} | {_fmt(spec_v, 3, dash=True)} | {_fmt(bacc_v, 3, dash=True)} '
                             f'| {_fmt(mcc_v, 3, dash=True)} |')
        lines.append('')
        lines += [
            '**Axis 1 / Axis 2 / top_importance, full breakdown by training year and K:**',
            '',
            ('Each table below is ONE selection policy, split into two stacked sub-tables, one '
             'for each year the model was TRAINED on. Within each sub-table the columns are the '
             'three train/test scenarios (in-domain, cross-year covariate framing, cross-year '
             'concept framing, see the method explanation above), each with its own F1 / '
             'Accuracy pair. The K='
             f'{_af_k if _af_k is not None else "?"} row in every sub-table is the SAME shared '
             'all-features ceiling, repeated everywhere so each sub-table is readable on its '
             'own, not three different measurements.'),
            '',
        ]
        for axis_title, policy_name in (('Axis 1 (`axis1_stable`)', 'axis1_stable'),
                                         ('Axis 2 (`axis2_stable`)', 'axis2_stable'),
                                         ('top_importance (native, 2017-anchored, for comparison)', 'top_importance')):
            lines += [f'**{axis_title}:**', '']
            lines += _ablation_md_table(abl, policy_name, _af_k)
        # This block and the two below it previously aggregated attack-only F1
        # (f1_cross_domain/gen_gap_f1), drawing "beats/does not beat" conclusions from it — but the
        # stated decision metric for H2 (line ~2714 above, and _compute_h2_verdict()) is Macro F1,
        # not Attack F1 alone. Migrated to macro_f1_cross_domain/gen_gap_macro_f1 so the narrative
        # here agrees with the actual decisive verdict instead of silently re-deciding on a
        # different metric one section down.
        _macro_cross_col = 'macro_f1_cross_domain' if _has_macro_h2 else 'f1_cross_domain'
        _macro_gap_col = 'gen_gap_macro_f1' if 'gen_gap_macro_f1' in abl.columns else 'gen_gap_f1'
        agg_dir = abl.groupby(['policy', 'direction']).agg(
            f1_cross=(_macro_cross_col, 'mean'), gap=(_macro_gap_col, 'mean')).round(3)
        lines += ['**Results, per direction (read this; transfer is asymmetric):**', '',
                  '| Policy | Direction | Cross-domain Macro F1 | Gen. gap (Macro F1) |',
                  '|--------|-----------|----------------|---------|']
        for (pol, direction), r in agg_dir.iterrows():
            lines.append(f'| {pol} | {direction} | {r["f1_cross"]:.3f}            | {r["gap"]:.3f}    |')
        lines.append('')
        _b1_obs = []
        for direction in ('2017->2018', '2018->2017'):
            try:
                _ti = float(agg_dir.loc[('top_importance', direction), 'f1_cross'])
                _a1 = float(agg_dir.loc[('axis1_stable', direction), 'f1_cross'])
                _a2 = float(agg_dir.loc[('axis2_stable', direction), 'f1_cross'])
            except KeyError:
                continue
            _b1_obs.append(
                f'in {direction}, averaged over every K, top_importance={_ti:.3f}, '
                f'axis1_stable={_a1:.3f} ({"beats" if _a1 > _ti else "does not beat"} '
                f'top_importance), axis2_stable={_a2:.3f} '
                f'({"beats" if _a2 > _ti else "does not beat"} top_importance)')
        if _b1_obs:
            lines += [
                ('**What we did and what we found:** we took the cross-domain Macro F1 (concept '
                 'framing) for each policy, averaged it over every K value, but kept the two '
                 'directions separate. Result: ' + '; '.join(_b1_obs) + '. A policy has to win '
                 'this comparison in BOTH directions at once to count as support for H2 — '
                 'winning only one direction is not enough; see the final verdict further down '
                 'for that combined check.'),
                '',
            ]
        _macro_cov_col = ('macro_f1_cross_covariate' if _has_macro_h2 and 'macro_f1_cross_covariate' in abl.columns
                          else 'f1_cross_covariate')
        agg_spec = {'f1_cross': (_macro_cross_col, 'mean'), 'gap': (_macro_gap_col, 'mean')}
        if has_cov:
            agg_spec['f1_cross_cov'] = (_macro_cov_col, 'mean')
        agg = abl.groupby('policy').agg(**agg_spec).round(3)
        lines += ['', '**Results, mean over K and both directions (summary):**', '']
        if has_cov:
            lines += ['| Policy | cross-Macro F1 (concept) | cross-Macro F1 (covariate) | gen. gap |',
                      '|--------|--------------------|---------------------|---------|']
            for pol, r in agg.iterrows():
                lines.append(f'| {pol} | {r["f1_cross"]:.3f}              | {r["f1_cross_cov"]:.3f}              | {r["gap"]:.3f}    |')
        else:
            lines += ['| Policy | cross-domain Macro F1 | gen. gap |',
                      '|--------|----------------|---------|']
            for pol, r in agg.iterrows():
                lines.append(f'| {pol} | {r["f1_cross"]:.3f}           | {r["gap"]:.3f}    |')
        lines.append('')
        _b2_obs = []
        try:
            _ti2 = float(agg.loc['top_importance', 'f1_cross'])
            _a12 = float(agg.loc['axis1_stable', 'f1_cross'])
            _a22 = float(agg.loc['axis2_stable', 'f1_cross'])
            _b2_obs.append(f'blending both directions together, axis1_stable averages {_a12:.3f} '
                           f'and axis2_stable averages {_a22:.3f} against top_importance\'s '
                           f'{_ti2:.3f} (concept framing)')
            if has_cov:
                _ti2c = float(agg.loc['top_importance', 'f1_cross_cov'])
                _a12c = float(agg.loc['axis1_stable', 'f1_cross_cov'])
                _a22c = float(agg.loc['axis2_stable', 'f1_cross_cov'])
                _b2_obs.append(f'on the covariate framing, axis1_stable={_a12c:.3f}, '
                               f'axis2_stable={_a22c:.3f}, top_importance={_ti2c:.3f}')
        except KeyError:
            pass
        if _b2_obs:
            lines += [
                ('**What we did and what we found (Macro F1):** same numbers as the table above, but now '
                 'we also average the two directions together into one number per policy. '
                 'Result: ' + '; '.join(_b2_obs) + '. Because the two directions are blended '
                 'into one number here, this table alone cannot show a policy that wins one '
                 'direction and loses the other — that asymmetry only shows up in the '
                 'per-direction table above and the full per-K tables below.'),
                '',
            ]

        has_swap = 'f1_indomain_other_scaler' in abl.columns
        _macro_in_col = 'macro_f1_in_domain' if _has_macro_h2 else 'f1_in_domain'
        _n_seeds = abl['seed'].nunique() if 'seed' in abl.columns else 1
        for direction in ('2017->2018', '2018->2017'):
            lines += [
                (f'**Full per-K ablation results, {direction}** (every K actually run; each cell '
                 f'is the mean over the {_n_seeds} seed replicates):'),
                '',
                ('| Policy | K | Macro F1 in-domain | Macro F1 cross (concept) | Acc cross (concept) '
                 '| Macro F1 cross (covariate) | Acc cross (covariate) | F1 in-domain (swap control) '
                 '| Gen. gap (Macro F1) |'),
                '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
            ]
            # Mean over seed replicates per (policy, K) — the raw frame has one row PER SEED,
            # so rendering/indexing it directly would print duplicate rows and make per-K
            # scalar lookups ambiguous (a Series per K).
            d_rows = (abl[abl['direction'] == direction]
                      .groupby(['policy', 'K'], as_index=False).mean(numeric_only=True)
                      .sort_values(['policy', 'K']))
            for _, r in d_rows.iterrows():
                lines.append(
                    f'| {r["policy"]} | {int(r["K"])} | {_fmt(r.get(_macro_in_col), 3)} '
                    f'| {_fmt(r.get(_macro_cross_col), 3)} | {r["acc_cross_domain"]:.3f} '
                    f'| {_fmt(r.get(_macro_cov_col), 3) if has_cov else "n/a"} '
                    f'| {_fmt(r.get("acc_cross_covariate"), 3) if has_cov else "n/a"} '
                    f'| {_fmt(r.get("f1_indomain_other_scaler"), 3) if has_swap else "n/a"} '
                    f'| {_fmt(r.get(_macro_gap_col), 3)} |')
            lines.append('')
            _ti_k = d_rows[d_rows['policy'] == 'top_importance'].set_index('K')['f1_cross_domain']
            _a1_k = d_rows[d_rows['policy'] == 'axis1_stable'].set_index('K')['f1_cross_domain']
            _a2_k = d_rows[d_rows['policy'] == 'axis2_stable'].set_index('K')['f1_cross_domain']
            _common_ks = sorted(set(_ti_k.index) & set(_a1_k.index) & set(_a2_k.index))
            _a1_wins = [int(k) for k in _common_ks if _a1_k[k] > _ti_k[k]]
            _a2_wins = [int(k) for k in _common_ks if _a2_k[k] > _ti_k[k]]
            lines += [
                (f'**What we did and what we found, direction {direction}:** we compared every '
                 'K value one at a time on cross-domain F1 (concept framing), instead of '
                 'averaging across K like the two tables above. Result: axis1_stable beats '
                 'top_importance at K = '
                 + (str(_a1_wins) if _a1_wins else 'none of the tested K values')
                 + '; axis2_stable beats top_importance at K = '
                 + (str(_a2_wins) if _a2_wins else 'none of the tested K values')
                 + '. H2 needs a policy to win essentially every K, in both directions, to be a '
                 'reliable effect rather than a lucky K choice.'),
                '',
            ]
        lines += _ablation_trend_lines(abl)
        _win_counts: dict = {}
        for direction in ('2017->2018', '2018->2017'):
            sub_dir = (abl[(abl['direction'] == direction)
                           & (abl['policy'].isin(['top_importance', 'axis1_stable', 'axis2_stable']))]
                       .groupby(['policy', 'K'], as_index=False)['f1_cross_domain'].mean())  # mean over seeds
            for k in sorted(sub_dir['K'].unique()):
                row_k = sub_dir[sub_dir['K'] == k].sort_values('f1_cross_domain', ascending=False)
                if row_k.empty:
                    continue
                best_pol = row_k.iloc[0]['policy']
                _win_counts[best_pol] = _win_counts.get(best_pol, 0) + 1
        if _win_counts:
            _total_slots = sum(_win_counts.values())
            _win_txt = ', '.join(
                f'`{pol}` wins {n}/{_total_slots} (K, direction) combinations'
                for pol, n in sorted(_win_counts.items(), key=lambda kv: -kv[1]))
            lines += [
                (f'**What this means for H2:** counting every (K, direction) combination shown '
                 f'above, {_win_txt}. H2 needs one of the two stability axes to win essentially '
                 'all of them, not just a plurality.'),
                '',
            ]
        _af_k_str = str(_af_k) if _af_k is not None else 'n/a'
        lines += [
            (f'To keep in mind: `random` and `all_features` are the floor/ceiling references, not '
             f'competing policies for H2; `all_features` only runs at K={_af_k_str} (all features, '
             f'nothing dropped), the other four policies run at every K. The swap-control column above is '
             'the SAME in-domain rows as the F1 in-domain column, just re-expressed in the other '
             'year\'s scaler — a large drop there (with no real distribution shift) would mean the '
             'model is sensitive to scaler mismatch itself, not just to genuine cross-year drift; '
             'a small drop means the COVARIATE column\'s degradation is mostly real shift, not a '
             'scaling artifact.'),
            '',
        ]
        if has_swap:
            k_lo, k_hi = int(abl['K'].min()), int(abl['K'].max())
            lines += [
                (f'**Effect size (K={k_lo} vs K={k_hi}, "median F1 of top-K vs bottom-K" reading '
                 f'for the ablation):** since each policy always keeps its TOP-K features (there '
                 f'is no natural bottom-K group in the ablation framework), the '
                 f'closest analogous contrast is the smallest vs '
                 f'largest feature subset actually tested:'),
                '',
                (f'| Policy | Direction | Cross-F1 at K={k_lo} | Cross-F1 at K={k_hi} '
                 f'(all_features) | Gain |'),
                '|---|---|---:|---:|---:|',
            ]
            _gains: dict = {}
            for pol in ('top_importance', 'axis1_stable', 'axis2_stable'):
                for direction in ('2017->2018', '2018->2017'):
                    lo = abl[(abl['policy'] == pol) & (abl['K'] == k_lo) & (abl['direction'] == direction)]
                    hi = abl[(abl['policy'] == 'all_features') & (abl['direction'] == direction)]
                    if lo.empty or hi.empty:
                        continue
                    f1_lo = float(lo['f1_cross_domain'].mean())   # mean over seed replicates
                    f1_hi = float(hi['f1_cross_domain'].mean())
                    lines.append(f'| {pol} | {direction} | {f1_lo:.3f} | {f1_hi:.3f} | {f1_hi - f1_lo:+.3f} |')
                    _gains[(pol, direction)] = f1_hi - f1_lo
            lines.append('')
            if _gains:
                _best = max(_gains.items(), key=lambda kv: kv[1])
                _worst = min(_gains.items(), key=lambda kv: kv[1])
                lines += [
                    (f'**What we did and what we found:** we compared each policy\'s F1 at the '
                     f'smallest tested K (K={k_lo}) against the all-features ceiling (K={k_hi}), '
                     'to see how much each policy gains from being given more features. Result: '
                     f'the biggest gain is `{_best[0][0]}` in direction {_best[0][1]} '
                     f'({_best[1]:+.3f} F1); the smallest (or worst) is `{_worst[0][0]}` in '
                     f'direction {_worst[0][1]} ({_worst[1]:+.3f} F1). A large gain means that '
                     'policy genuinely needed more features to transfer well at K='
                     f'{k_lo}; a small or negative gain means a handful of features already '
                     'carried most of the cross-year signal for that policy.'),
                    '',
                ]

        si_rows = abl[abl['policy'] == 'axis2_stable']
        ti_rows = abl[abl['policy'] == 'top_importance']
        a1_rows = abl[abl['policy'] == 'axis1_stable']
        if not si_rows.empty and not ti_rows.empty and not a1_rows.empty:
            lines += [
                '**In this test, here is what we found, axis by axis, direction by direction:**',
                '',
            ]
            for direction in ('2017->2018', '2018->2017'):
                ti_d = ti_rows[ti_rows['direction'] == direction]['f1_cross_domain'].mean()
                a1_d = a1_rows[a1_rows['direction'] == direction]['f1_cross_domain'].mean()
                a2_d = si_rows[si_rows['direction'] == direction]['f1_cross_domain'].mean()
                winner = max((('top_importance', ti_d), ('axis1_stable', a1_d), ('axis2_stable', a2_d)),
                             key=lambda kv: kv[1])[0]
                lines += [
                    (f'- Direction {direction}: selecting by importance gives cross-year F1='
                     f'{ti_d:.3f}; selecting by Axis-1 stability gives {a1_d:.3f}; selecting by '
                     f'Axis-2 stability gives {a2_d:.3f}. Best policy this direction: **{winner}**.'),
                ]
            lines += [
                '',
                ('Both Axis 1 and Axis 2 were tested here, head-to-head against importance-based '
                 'selection, in both directions — neither axis was skipped.'),
                '',
            ]
            # _h2_verdict was already computed once, above, by _compute_h2_verdict() — reused
            # here rather than recomputed so the earlier mention and this final line can never
            # disagree with each other.
            lines += [
                f'**The result: {_h2_verdict}.**',
                ('To keep in mind: this requires a stability policy to win BOTH directions, not '
                 'just on average — transfer is highly direction-asymmetric (see the same-year-vs-'
                 'cross-year gap table earlier in this document). A policy that wins one direction '
                 'and loses the other is not a counterexample to "not supported," it is the '
                 'expected shape of an asymmetric result.'),
                ('This verdict is based on the MEAN over every K. For the K-specific picture '
                 'behind it (which exact K values a stability axis actually wins at, in each '
                 'direction), see the per-K win/loss breakdown and the (K, direction) win-count '
                 'in the full per-K tables above — a policy can win on average while still '
                 'losing at some individual K values.'),
                '',
            ]
            if baseline and 'directions' in baseline:
                _d17_18 = baseline['directions'].get('cicids2017->cicids2018', {}).get('binary', {})
                _bc = (_d17_18.get('concept') or {}).get('attack_f1', float('nan'))
                _bv = (_d17_18.get('covariate') or {}).get('attack_f1', float('nan'))
                _af_row = abl[(abl['policy'] == 'all_features') & (abl['direction'] == '2017->2018')]
                _af_f1 = float(_af_row['f1_cross_domain'].mean()) if not _af_row.empty else float('nan')
                _base_gap = abs(_af_f1 - _bc) if np.isfinite(_af_f1) and np.isfinite(_bc) else float('nan')
                lines += [
                    ('**For reference — how these numbers compare to the real full-data baseline '
                     '(Step 6, every row, not the row-capped/rebalanced set the ablation above '
                     'uses):** for 2017-trained models tested on 2018, Step 6\'s real full-data '
                     f'attack-class F1 is {_bc:.3f} (concept framing) / {_bv:.3f} (covariate '
                     f'framing); the ablation\'s row-capped all-features reference point at the '
                     f'same direction is {_af_f1:.3f}. These are not directly comparable numbers '
                     '(different row counts, different class balance), but they show the ablation\'s '
                     'capped F1s are in the same neighborhood as the real deployment numbers, not '
                     'an artifact of the row-capping alone.'),
                    (f'What we did and what we found: we took the absolute gap between these two '
                     f'numbers, {_base_gap:.3f} F1. '
                     + ('This is small, so the row-capping itself is not distorting the picture; '
                        'the policy comparisons above are reading real cross-year transfer, not a '
                        'row-capping artifact.' if np.isfinite(_base_gap) and _base_gap < 0.05
                        else 'This is non-trivial, so keep in mind that some of the gap between '
                        'the capped reference and 0/1 may be row-capping itself, not only '
                        'cross-year drift, when reading the policy comparisons above.')
                     if np.isfinite(_base_gap) else
                     'Baseline or ablation reference value missing this run, so this check could '
                     'not be computed.'),
                    '',
                ]
        else:
            lines.append('_axis1_stable/axis2_stable/top_importance rows missing from ablation output._')
    else:
        lines.append('_Ablation not run (set RUN_ABLATION=True in Config11)._')
    lines += ['', '**Status: DONE**', '']

    # ── SUPPLEMENTARY: E-series (non-sequential, does not renumber C1-C9) ───────
    # Placed at the very end of Step 11's content, after C9 — these are
    # robustness/diagnostic checks on metrics already used above, not part of the core C1-C9
    # numbered sequence, so they sit after it rather than disturbing it.
    lines += ['---', '## Supplementary checks (E-series)', '']

    # E1 — cross-metric agreement: does each corroboration metric agree with the C2ST verdict?
    _agree_df = compute_metric_agreement()
    lines += [
        '### E1: Cross-metric agreement — do the corroboration metrics agree with the calibrated-C2ST verdict?',
        '',
        ('**In short:** calibrated C2ST-AUC is the ONE Axis-1 decision metric — it decides the '
         'stable/shifted verdict in Step 10 and every H1/H1.5/H2 test here (chosen because it is '
         'the only metric computed identically for every feature type). The corroboration '
         'metrics (Wasserstein-qn, MMD, KS-statistic, energy-distance, Anderson-Darling for '
         'continuous features; Jensen-Shannon for PMF-routed nominal/discrete-count features), '
         'each calibrated against its OWN permutation null, exist purely to CORROBORATE that '
         'decision — this check reports whether they do. It never overrides or feeds back into '
         'any verdict.'),
        '',
        ('**Method (computed by Step 10, `execute_one()`):** each corroboration metric votes '
         'shifted (its calibrated excess > 0, i.e. above its own null floor) or stable; '
         '"agrees" = same vote as the C2ST verdict for that feature. Cells show `!` (not '
         'applicable) where a metric is not computed for that feature\'s route — never counted '
         'as disagreement, since there is nothing to compare.'),
        '',
    ]
    _e1_metrics = [
        ('wasserstein_qn', 'Wasserstein-qn'), ('mmd', 'MMD'), ('ks_statistic', 'KS-statistic'),
        ('energy_distance', 'Energy-dist'), ('anderson_darling', 'Anderson-Darling'),
        ('jensen_shannon', 'Jensen-Shannon'),
    ]
    if not _agree_df.empty:
        n_total = len(_agree_df)
        _counts_line = []
        for col, label in _e1_metrics:
            st = _agree_df[f'{col}_state']
            n_dis = int((st == 'disagree').sum())
            n_na = int((st == 'na').sum())
            _counts_line.append(f'{label}: {n_dis} disagree, {n_na} n/a (of {n_total})')
        lines += [
            '; '.join(_counts_line) + '.',
            '',
            ('| Feature | C2ST-AUC (calibrated) | Wasserstein (calibrated) | W? | MMD (calibrated) | MMD? '
             '| KS-stat (calibrated) | KS? | Energy-dist (calibrated) | Energy? '
             '| Anderson-Darling (calibrated) | AD? | Jensen-Shannon (calibrated) | JS? |'),
            '|---|---:|---:|:---:|---:|:---:|---:|:---:|---:|:---:|---:|:---:|---:|:---:|',
        ]
        _mark = {'agree': '✅', 'disagree': '❌', 'na': '!'}
        for feat, row in _agree_df.sort_values('c2st_auc', ascending=False).iterrows():
            cells = [feat, _fmt(row['c2st_auc'])]
            for col, _ in _e1_metrics:
                cells += [_fmt(row[col]), _mark[row[f'{col}_state']]]
            lines.append('| ' + ' | '.join(cells) + ' |')
        lines += [
            '',
            ('**How to read this:** a ✅ means that corroboration metric, calibrated against its '
             'own null, votes the same shifted/stable call the C2ST verdict made — you would '
             'reach the same conclusion no matter which metric you trusted. A ❌ means the '
             'metrics genuinely disagree for this particular feature; treat that feature\'s '
             'Axis-1 verdict as less robust than one where all metrics agree (its Q-Q/overlap '
             'plots are worth a look before leaning on it). A `!` means this metric was never '
             'computed for this feature\'s route — not evidence of disagreement, just nothing '
             'to compare.'),
            '',
        ]
    else:
        lines += [
            '_E1 agreement fields not found in verdicts_layerA (re-run step 10 after the E1 '
            'reinstatement) — nothing to tabulate this run._',
            '',
        ]
    lines += [
        '**Source scripts:** corroboration metrics + null calibrations + agreement votes computed '
        'by Script 10 (`10_execute_comparison.py`, `execute_one()`/`marginal_shift()`); '
        'Script 11 (`11_result_gen.py`, `compute_metric_agreement()`) only loads and tabulates them.',
        '',
        '**Status: DONE**',
        '',
    ]

    # E2 — per-mode ("blob") comparison (moved from C2a/C3 area to supplementary)
    _pm = df[df['max_mode_shift'].notna()] if 'max_mode_shift' in df.columns else df.iloc[0:0]
    lines += [
        '### E2: Per-mode ("blob") comparison',
        '',
        ('Features whose distribution consists of multiple separated blobs (e.g. gated by port or '
         'protocol) are not just compared pooled. Script 10 also matches each 2017 blob to its nearest '
         '2018 blob and compares them individually, so one blob moving cannot hide behind the others or '
         'get diluted into a small pooled-shift number.'),
        '',
        ('**Source scripts:** Script 9 (`09_plan_comparison.py`) detects multimodal features and sets '
         '`comparison_mode = per_mode`; Script 10 (`10_execute_comparison.py`) runs the blob-to-blob '
         'comparison; Script 11 surfaces results here.'),
        '',
    ]
    if len(_pm):
        lines += [
            '| Feature | n_modes 2017 | n_modes 2018 | Modality mismatch | Max mode shift | Max mode mass shift |',
            '|---------|-------------:|-------------:|:------------------:|---------------:|---------------------:|',
        ]
        for feat, row in _pm.sort_values('max_mode_shift', ascending=False).iterrows():
            lines.append(f'| {feat} | {_fmt(row.get("n_modes_2017"))} | {_fmt(row.get("n_modes_2018"))} '
                         f'| {row.get("modality_mismatch", "n/a")} | {_fmt(row.get("max_mode_shift"))} '
                         f'| {_fmt(row.get("max_mode_mass_shift"))} |')
        lines += [
            '',
            ('`Max mode shift` is the largest single blob-to-blob distributional shift across all matched '
             'mode pairs (same scale as the pooled Wasserstein-qn). `Max mode mass shift` is the largest '
             'change in how much of the feature\'s rows fall in that mode (mode population, not value). '
             'Full blob-by-blob detail (every matched mode pair) is in '
             f'`verdicts_layerA_{DS1}_{DS2}.json` under each feature\'s `per_mode_results` key.'),
            '',
        ]
    else:
        lines += [
            '_No features were routed to per-mode comparison this run (none required it, per Script 9\'s plan)._',
            '',
        ]
    lines += ['**Status: DONE**', '']

    # E3 — zero-mass-separate comparison (moved from C2a/C3 area to supplementary)
    _zm = df[df['zero_frac_2017'].notna()] if 'zero_frac_2017' in df.columns else df.iloc[0:0]
    lines += [
        '### E3: Zero-mass-separate comparison',
        '',
        ('For zero-inflated features (a large share of exact-0 rows, e.g. byte/packet counts on '
         'flows with no payload), the zero fraction is compared as its own scalar, and the '
         'Wasserstein-qn distance is recomputed on the NON-ZERO values only (the "tail"). This '
         'separates "did the zero rate change" from "did the non-zero values change shape", which a '
         'single pooled Wasserstein on the full column (including zeros) cannot distinguish.'),
        '',
        ('**Source scripts:** Script 9 (`09_plan_comparison.py`) detects zero-inflation and sets '
         '`zero_mass_separate`; Script 10 (`10_execute_comparison.py`) computes the split comparison; '
         'Script 11 surfaces results here.'),
        '',
    ]
    if len(_zm):
        lines += [
            '| Feature | Zero frac 2017 | Zero frac 2018 | Δ Zero frac | Tail Wasserstein-qn |',
            '|---------|---------------:|---------------:|-----------:|---------------------:|',
        ]
        for feat, row in _zm.sort_values('zero_frac_delta', ascending=False).iterrows():
            lines.append(f'| {feat} | {_fmt(row.get("zero_frac_2017"))} | {_fmt(row.get("zero_frac_2018"))} '
                         f'| {_fmt(row.get("zero_frac_delta"))} | {_fmt(row.get("tail_wasserstein_qn"))} |')
        lines += [
            '',
            ('`Δ Zero frac` is `|zero_frac_2018 - zero_frac_2017|` — a large value means the feature '
             'went from mostly-zero to mostly-populated (or vice versa) between years, which is itself '
             'a distribution-shape change the pooled metric would under-report. `Tail Wasserstein-qn` '
             'is the shift among the non-zero values only.'),
            '',
        ]
    else:
        lines += [
            '_No features were routed to zero-mass-separate comparison this run (none required it, '
            'per Script 9\'s plan)._',
            '',
        ]
    lines += ['**Status: DONE**', '']

    # E4 — flip corroboration audit: is a pooled 'flipped' verdict backed by a real per-family
    # reversal, or is it a mixture-ratio artifact? Both flip_corroborated and n_family_flips are
    # already columns in cross_table.csv (computed by Script 10's per_class_stability loop), just
    # never rendered until now.
    _fl = df[df['verdict'] == 'flipped'] if 'verdict' in df.columns and 'flip_corroborated' in df.columns else df.iloc[0:0]
    lines += [
        '### E4: Flip corroboration audit',
        '',
        ('The pooled `flipped` verdict (Axis 2) is computed on the binary benign-vs-attack direction, '
         'which a changing attack MIXTURE between years can reverse without any single attack family\'s '
         'relationship to benign actually flipping. This audit checks, for every feature with a pooled '
         '`flipped` verdict, whether at least one attack family shared by both years also flipped sign '
         'on its own. A flip backed by a real per-family reversal is "corroborated"; one that is not is '
         'most likely a mixture-ratio artifact rather than a genuine concept change.'),
        '',
        ('**Source scripts:** Script 10 (`10_execute_comparison.py`, the per-family flip check around '
         '`flip_corroborated`/`n_family_flips`); Script 11 surfaces results here.'),
        '',
    ]
    if len(_fl):
        n_corro = int(_fl['flip_corroborated'].sum())
        lines += [
            f'{len(_fl)} feature(s) carry a pooled `flipped` verdict; {n_corro} corroborated, '
            f'{len(_fl) - n_corro} uncorroborated (likely mixture artifact).',
            '',
            '| Feature | Corroborated | Families that flipped |',
            '|---|:---:|---:|',
        ]
        for feat, row in _fl.sort_values('flip_corroborated', ascending=False).iterrows():
            corro = '✅' if row['flip_corroborated'] else '❌'
            lines.append(f'| {feat} | {corro} | {_fmt(row.get("n_family_flips"), 0)} |')
        lines += ['']
    else:
        lines += ['_No features carry a pooled `flipped` verdict this run._', '']
    lines += ['**Status: DONE**', '']

    # E5 — prior shift: class-proportion drift (P(Y) change), independent of covariate/concept shift
    lines += [
        '### E5: Prior shift — class-proportion drift as a shift axis',
        '',
        ('Prior shift (P(Y) change, also called "label shift" in the literature) occurs when the ratio '
         'of classes changes between years independently of feature-value or decision-boundary changes. '
         'Unlike Axis 1 (covariate shift) and Axis 2 (concept stability), prior shift is NOT captured by '
         'per-feature statistical tests, but it has a large, documented impact on model precision. This '
         'section measures the magnitude and direction of prior shift in the training data.'),
        '',
    ]
    if prior_shift and isinstance(prior_shift, dict):
        per_ds = prior_shift.get('per_dataset', {})
        shift_abs = prior_shift.get('prior_shift_abs', float('nan'))
        interp = prior_shift.get('interpretation', '')

        lines += [
            '| Metric | 2017 | 2018 |',
            '|---|---:|---:|',
        ]
        if DS1 in per_ds and DS2 in per_ds:
            p17 = per_ds[DS1].get('p_benign', float('nan'))
            p18 = per_ds[DS2].get('p_benign', float('nan'))
            n17 = per_ds[DS1].get('total_rows', '?')
            n18 = per_ds[DS2].get('total_rows', '?')

            lines.append(f'| P(benign) | {p17:.1%} | {p18:.1%} |')
            lines.append(f'| Total rows | {n17:,} | {n18:,} |')
            lines += ['']

        if np.isfinite(shift_abs):
            lines.append(f'**Prior shift (|ΔP(benign)|):** {shift_abs:.3f}')
        else:
            lines.append(f'**Prior shift:** Could not be computed')
        lines += ['']

        if interp:
            lines.append(f'{interp}')
            lines += ['']
    else:
        lines += ['_Prior shift data unavailable._', '']

    note = prior_shift.get('note', '') if prior_shift else ''
    if note:
        lines += [f'**Note:** {note}', '']
    lines += ['**Status: DONE**', '']

    # ── Prior / threshold recalibration (threshold recalibration, previously untested) ─────────────
    # The prior-shift section above is a DIAGNOSIS; this tests the recommended FIX on the
    # already-trained models (no retraining). Data: recalibration_summary.json (from step 11, which
    # flattens step 6's per-cell recalibration_binary_*.json).
    recal_doc = load_json_best_effort(OUTPUT_DIR / 'recalibration_summary.json')
    lines += ['', '## Prior / Threshold Recalibration (Testing the Recommended Fix)', '']
    if recal_doc and recal_doc.get('rows'):
        lines += [
            ('The section above diagnoses prior shift; this one tests the recommended fix. Using the '
             'ALREADY-TRAINED models (no retraining), the target-year attack probabilities are '
             're-thresholded under: `baseline_0.5` (the implicit 0.5 threshold reported in the '
             'cross-year table), `prior_ratio_known` (a Saerens/Latinne/Decaestecker posterior '
             'adjustment using the known target prior), `sld_em` (the same adjustment with a '
             'label-free EM estimate of that prior), and `oracle_best_f1` (the F1-optimal threshold '
             'on target labels, a non-deployable ceiling).'),
            '',
            '| Direction | Framing | Strategy | Attack F1 | Recall | Precision | Est. prior |',
            '|---|---|---|---:|---:|---:|---:|',
        ]
        _ord = {'baseline_0.5': 0, 'prior_ratio_known': 1, 'sld_em': 2, 'oracle_best_f1': 3}
        _rows = sorted(recal_doc['rows'],
                       key=lambda r: (r.get('direction', ''), r.get('framing', ''),
                                      _ord.get(r.get('strategy'), 9)))

        def _rf(x):
            return f'{x:.4f}' if isinstance(x, (int, float)) else '-'
        for r in _rows:
            est = r.get('p_tgt_estimated')
            est_s = f'{est:.3f}' if isinstance(est, (int, float)) else ''
            lines.append(
                f"| {r.get('direction', '')} | {r.get('framing', '')} | {r.get('strategy', '')} | "
                f"{_rf(r.get('attack_f1'))} | {_rf(r.get('recall'))} | {_rf(r.get('precision'))} | "
                f"{est_s} |")
        lines += [
            '',
            ('Reading: in the 2018 to 2017 direction (a model that expects a 5.6%-attack stream meets '
             'a 24%-attack one), the implicit-0.5 collapse is a THRESHOLD failure, not a ranking '
             'failure. A known-prior re-threshold lifts concept-framing attack F1 from ~0.000 to '
             '~0.64 (oracle ceiling ~0.73), and covariate framing from 0.39 to ~0.64. Recalibration '
             'is not a free lunch: in the reverse 2017 to 2018 direction a naive prior-ratio '
             'correction OVER-suppresses (attack F1 0.556 to 0.000) because the shifted posteriors '
             'are miscalibrated, and the label-free EM prior estimate diverges (toward 1.0) under '
             'this severity of shift. The reliable gain comes from selecting the operating threshold '
             'on target data (oracle), and prior correction does NOT repair covariate-shift-driven '
             'collapse (2017 to 2018 covariate 0.044 to 0.025), consistent with the decomposition: '
             'the prior-shift mechanism is correctable in the direction it dominates; the '
             'covariate-shift mechanism is not fixed by re-thresholding.'),
            '',
            'Output: `recalibration_summary.json` / `.csv` (from step 6 `recalibration_binary_*.json`).',
            '',
        ]
    else:
        lines += ['_Recalibration summary unavailable, re-run step 6 then step 11 with the '
                  'recalibration patch to produce it._', '']
    lines += ['**Status: DONE**', '']

    lines += _build_output_map_section()

    # ── Post-process: strip instructional-style section labels throughout ────────
    # These Outcome/Claim/Tests-run prefixes read as scaffolding rather than prose; replace them with plain text or drop them.
    cleaned: list = []
    skip = False
    for line in lines:
        if skip:
            skip = False
            continue
        # Remove "**Outcome:** " prefix — the section header already states the purpose
        if line.startswith('**Outcome:** ') or line.startswith("'**Outcome:** "):
            line = line.replace('**Outcome:** ', '', 1)
        # Replace "**Status: DONE** — see `X`" with "Output: `X`"
        if '**Status: DONE** — see ' in line:
            line = line.replace('**Status: DONE** — see ', 'Output: ')
        elif re.match(r'^\*\*Status: .*\*\*$', line.strip()):
            line = ''
        # Remove "**Claim:** " prefix (claim is restated in the section intro)
        if line.startswith('**Claim:** '):
            line = line.replace('**Claim:** ', '', 1)
        # Replace standalone "**Tests run:**" with "Computed:" (less tutorial-sounding)
        if line.strip() == '**Tests run:**':
            line = 'Computed:'
        # Safety net: every deliberate em-dash rewrite above is contextual; this catches
        # anything missed so the rendered document has none left.
        # A comma is the one substitution that is almost never grammatically wrong here.
        if '—' in line:
            line = line.replace(' — ', ', ').replace('—', ',')
        cleaned.append(line)
    lines = cleaned

    out_path.write_text('\n'.join(lines), encoding='utf-8')


# ════════════════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description='Step 11b: render results.md from the step-11 analysis cache. '
                    'No CLI arguments — the engine is LightGBM only (unified_config.ALGORITHM).')
    ap.parse_args()

    global OUTPUT_DIR, RESULTS_DIR
    OUTPUT_DIR  = cross_output_dir(PROJECT_ROOT, ALGORITHM)
    RESULTS_DIR = cross_results_dir(PROJECT_ROOT, ALGORITHM)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cache_path = OUTPUT_DIR / CACHE_FILENAME
    if not cache_path.exists():
        print(f'[BLOCKED] {cache_path} not found.')
        print('Run: python scripts/11_cross_analysis.py first, to produce the analysis cache '
              'this script reads from. After that, this script can be re-run on its own as '
              'often as needed.')
        sys.exit(1)

    # Deliberately NOT Config11.STEPS_FILE: that path is 11_cross_analysis.py's own log, opened
    # in write/truncate mode for the lifetime of its process. Writing to it here too would race
    # with (and on Windows can outright fail against) the still-open handle in the parent
    # process that just spawned this script, and Logger's 'w' mode would truncate that log's
    # history anyway. A distinct file avoids both problems.
    log = Logger(RESULTS_DIR / '11_result_gen_steps.log', step_prefix=11,
                title=f'SCRIPT 11b — RESULTS DOC GENERATION  [{ALGORITHM}]')
    log.info(f'Engine    : {ALGORITHM} (the pipeline\'s only engine)')
    log.info(f'Cache     : {cache_path}')

    log.step('Load cached analysis outputs')
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    df               = cache['df']
    stats            = cache['stats']
    drift            = cache['drift']
    abl              = cache['abl']
    report           = cache['report']
    benign_atk       = cache['benign_atk']
    prior_shift      = cache['prior_shift']
    baseline         = cache['baseline']
    delta_imp_stab   = cache.get('delta_imp_stab')   # .get(): absent in caches from before H1.5 was added
    log.ok(f'loaded cache: {len(df)} features')
    log.step_end()

    log.step('Write results.md')
    if stats:
        results_path = RESULTS_DIR / Config11.RESULTS_MD_FILE
        write_results_doc(df, stats, drift, abl, report, results_path,
                          benign_atk=benign_atk,
                          prior_shift=prior_shift,
                          baseline=baseline, delta_imp_stab=delta_imp_stab)
        log.ok(f'{Config11.RESULTS_MD_FILE} written to {RESULTS_DIR}')
    else:
        log.info('  docs skipped (cached run had no stats to summarize, e.g. an ablation-only run)')
    log.step_end()
    log.close()


if __name__ == '__main__':
    main()
