"""
3_correlation_comparison.py — Cross-dataset correlation comparison and feature drop decisions.

PURPOSE:
  Compare correlation structure of 2017 and 2018 datasets to identify consistently redundant
  features across both years. Produces drop list for Script 4 (preprocessing) and visualizations
  for manual review. Uses intersection rule (MIN across years, BOTH metrics) to ensure safe drops.

PROCESS:
  Sub-step 3.1 (Load matrices):
    - Read 4 JSON files from Script 2: Pearson 2017/2018, Spearman 2017/2018
    Result: 4 correlation matrices in memory

  Sub-step 3.2 (Align features):
    - Find common feature set across both years
    - Reindex all 4 matrices to common features only
    Result: 4 aligned matrices (81 features each)

  Sub-step 3.3 (MIN Pearson matrix):
    - For each pair, compute MIN(|Pearson 2017|, |Pearson 2018|)
    - Render heatmap
    Result: MIN Pearson matrix, saved as JSON + PNG

  Sub-step 3.4 (MIN Spearman matrix):
    - For each pair, compute MIN(|Spearman 2017|, |Spearman 2018|)
    - Render heatmap
    Result: MIN Spearman matrix, saved as JSON + PNG

  Sub-step 3.5 (Build consensus graph):
    - Connect pairs where MIN_Pearson >= 0.95 AND MIN_Spearman >= 0.95
    - Find connected components (redundancy groups)
    Result: List of redundancy groups (features to consolidate)

  Sub-step 3.6 (Select representatives):
    - Per group: keep feature with lowest avg correlation to all others
    - Classify remaining pairs (stable/shifted/weak)
    Result: Drop list and classification map

  Sub-step 3.7 (Detect drift):
    - Find pairs correlated in one year only (correlation drift)
    Result: Drift pair list

  Sub-step 3.8 (Generate visualizations):
    - Diff heatmap (2018 - 2017 correlation change)
    - Redundancy groups bar chart
    Result: 2 PNG files

  Sub-step 3.9 (Save outputs):
    - Write JSON files for downstream scripts
    Result: drop_decisions.json, comparison_matrix.json

  Sub-step 3.10 (Write report):
    - Human-readable summary of findings and decisions
    Result: results.txt file

INPUTS:
  - output/2_correlation_analysis/<dataset>/pearson_matrix.json
  - output/2_correlation_analysis/<dataset>/spearman_matrix.json

OUTPUTS:
  - output/3_correlation_comparison/drop_decisions.json (feature drop list)
  - output/3_correlation_comparison/comparison_matrix.json (all correlation matrices)
  - output/3_correlation_comparison/min_pearson_matrix.json (MIN Pearson)
  - output/3_correlation_comparison/min_spearman_matrix.json (MIN Spearman)
  - results/3_correlation_comparison/3_correlation_comparison_report.txt (human report)
  - results/3_correlation_comparison/3_correlation_comparison_steps.log (execution log)
  - results/3_correlation_comparison/pearson_min_heatmap.png
  - results/3_correlation_comparison/spearman_min_heatmap.png
  - results/3_correlation_comparison/diff_heatmap.png
  - results/3_correlation_comparison/redundancy_groups.png

GUARANTEES:
  - No source data is modified on disk
  - All drop decisions based on MIN rule (stable across both years and both metrics)
  - Features marked "shifted" are NOT dropped (appear in only one year)
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ── Import config from unified_config ─────────────────────────────────────────
# (unified_config reconfigures stdout to UTF-8 at import time, so no need to repeat it here)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from unified_config import Config3, Logger, DATASETS

# ── Paths ──────────────────────────────────────────────────────────────────────
OUTPUT_DIR   = Config3.OUTPUT_DIR
RESULTS_DIR  = Config3.RESULTS_DIR


# ── Matrix I/O ─────────────────────────────────────────────────────────────────
def load_matrix(path: Path, log: Logger) -> dict:
    """Load a correlation matrix JSON produced by script 2.

    Returns {'features': list[str], 'matrix': np.ndarray (n x n), 'meta': dict}.
    """
    log.info(f'Loading {path.name} ...')
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)

    features = raw['features']
    mat = np.array(raw['matrix'], dtype=np.float64)
    meta = {k: v for k, v in raw.items() if k not in ('features', 'matrix')}
    log.info(f'  {len(features)} features, matrix {mat.shape}')
    return {'features': features, 'matrix': mat, 'meta': meta}


# ── Feature alignment ──────────────────────────────────────────────────────────
def align_features(matrices: dict[str, dict], log: Logger) -> list[str]:
    """Return features present in ALL loaded matrices, in the order of the first."""
    sets = {key: set(m['features']) for key, m in matrices.items()}
    common = set.intersection(*sets.values())
    # preserve order from the first matrix key
    first_features = next(iter(matrices.values()))['features']
    ordered = [f for f in first_features if f in common]
    for key, m in matrices.items():
        missing = set(m['features']) - common
        if missing:
            log.warn(f'{key}: {len(missing)} features not in common set → excluded: {sorted(missing)}')
    log.info(f'Common feature set: {len(ordered)} features')
    return ordered


def reindex_matrix(m: dict, features: list[str]) -> np.ndarray:
    """Return a sub-matrix of m['matrix'] restricted to `features` in that order."""
    idx = {f: i for i, f in enumerate(m['features'])}
    sel = [idx[f] for f in features]
    return m['matrix'][np.ix_(sel, sel)]


# ── Aggregated absolute correlation ────────────────────────────────────────────
def average_abs_matrix(mats: list[np.ndarray]) -> np.ndarray:
    """Element-wise average of absolute values.  Self-correlation diagonal is set to 1.0.

    Used for REPORTING and for the mRMR representative score — NOT for the drop decision.
    """
    avg = np.mean([np.abs(m) for m in mats], axis=0)
    np.fill_diagonal(avg, 1.0)
    return avg


def min_abs_matrix(mats: list[np.ndarray]) -> np.ndarray:
    """Element-wise MINIMUM of absolute values across years.  Diagonal set to 1.0.

    This is the intersection-rule matrix: min_abs[A,B] >= t  <=>  |r| >= t in EVERY year.
    Used for the consensus graph (the actual drop decision).
    """
    mn = np.min([np.abs(m) for m in mats], axis=0)
    np.fill_diagonal(mn, 1.0)
    return mn


# ── Union-Find for connected components ────────────────────────────────────────
class UnionFind:
    def __init__(self, n: int):
        self._parent = list(range(n))
        self._rank   = [0] * n

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, x: int, y: int):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1

    def components(self) -> dict[int, list[int]]:
        groups: dict[int, list[int]] = defaultdict(list)
        for i in range(len(self._parent)):
            groups[self.find(i)].append(i)
        return dict(groups)


# ── Consensus graph + connected components ─────────────────────────────────────
def build_redundancy_groups(
    min_pearson: np.ndarray,
    min_spearman: np.ndarray,
    features: list[str],
    threshold: float,
    log: Logger,
) -> list[list[str]]:
    """Return connected components of the consensus correlation graph.

    An edge exists iff |r| >= threshold in BOTH years for BOTH metrics — encoded as the
    per-year MINIMUM absolute correlation meeting the threshold (intersection rule).
    """
    n = len(features)
    uf = UnionFind(n)
    # Union-find's final connected components don't depend on union() call order, so
    # vectorizing the edge scan (rather than a nested Python loop) is safe here.
    iu, ju = np.triu_indices(n, k=1)
    edge_mask = (min_pearson[iu, ju] >= threshold) & (min_spearman[iu, ju] >= threshold)
    for i, j in zip(iu[edge_mask], ju[edge_mask]):
        uf.union(int(i), int(j))
    edge_count = int(edge_mask.sum())
    log.info(f'Consensus graph edges (intersection rule) at threshold {threshold}: {edge_count}')
    groups = [
        [features[i] for i in sorted(members)]
        for members in uf.components().values()
        if len(members) > 1
    ]
    log.info(f'Redundancy groups (size > 1): {len(groups)}')
    return groups


# ── mRMR minimum-redundancy representative ────────────────────────────────────
def pick_representative(group: list[str], features: list[str], avg_pearson: np.ndarray) -> str:
    """Keep the group member with the lowest average |r| to ALL features.

    This is the minimum-redundancy criterion from the mRMR framework
    (Peng et al., TPAMI 2005): the feature least correlated with everything else
    carries the most independent information and should survive the drop.
    """
    feat_idx = {f: i for i, f in enumerate(features)}
    scores: dict[str, float] = {}
    for f in group:
        col = avg_pearson[feat_idx[f], :]
        # exclude self-correlation (diagonal = 1.0)
        others = np.concatenate([col[:feat_idx[f]], col[feat_idx[f]+1:]])
        scores[f] = float(np.mean(others))
    return min(scores, key=scores.__getitem__)


# ── Pair classification across years ──────────────────────────────────────────
def classify_pairs(
    groups: list[list[str]],
    pearson_17: np.ndarray,
    pearson_18: np.ndarray,
    spearman_17: np.ndarray,
    spearman_18: np.ndarray,
    features: list[str],
    avg_pearson: np.ndarray,
    avg_spearman: np.ndarray,
    threshold: float,
) -> list[dict]:
    """For each (keep, drop) pair, record per-year correlations and an intersection-rule status.

    status semantics (threshold applied to EACH year, the intersection rule):
      'stable'              -- Pearson AND Spearman >= threshold in BOTH years (safe drop)
      'stable_pearson_only' -- Pearson >= threshold both years, Spearman falls short (NOT dropped)
      'shifted_2017'        -- Pearson >= threshold in 2017 only (transitively grouped)
      'shifted_2018'        -- Pearson >= threshold in 2018 only (transitively grouped)
      'weak_transitive'     -- neither year clears it directly (linked through a chain)
    Only 'stable' pairs feed the final drop list — a drop requires BOTH metrics to agree in
    BOTH years, the same intersection rule the consensus graph uses. 'stable_pearson_only'
    pairs are monitored, not dropped (rank correlation disagreeing with linear correlation
    means the pair's relationship is not robustly redundant).
    """
    feat_idx = {f: i for i, f in enumerate(features)}
    classified: list[dict] = []

    for group in groups:
        keep = pick_representative(group, features, avg_pearson)
        to_drop = [f for f in group if f != keep]
        for d in to_drop:
            i, j = feat_idx[keep], feat_idx[d]
            p17 = float(abs(pearson_17[i, j]))
            p18 = float(abs(pearson_18[i, j]))
            s17 = float(abs(spearman_17[i, j]))
            s18 = float(abs(spearman_18[i, j]))

            pearson_both  = p17 >= threshold and p18 >= threshold
            spearman_both = s17 >= threshold and s18 >= threshold

            if pearson_both and spearman_both:
                status = 'stable'
            elif pearson_both:
                status = 'stable_pearson_only'
            elif p17 >= threshold:
                status = 'shifted_2017'
            elif p18 >= threshold:
                status = 'shifted_2018'
            else:
                status = 'weak_transitive'

            classified.append({
                'keep': keep,
                'drop': d,
                'group': group,
                'pearson_2017':  round(p17, 6),
                'pearson_2018':  round(p18, 6),
                'spearman_2017': round(s17, 6),
                'spearman_2018': round(s18, 6),
                'avg_pearson':   round(float(avg_pearson[i, j]), 6),
                'avg_spearman':  round(float(avg_spearman[i, j]), 6),
                'min_pearson':   round(min(p17, p18), 6),
                'status': status,
            })

    return classified


# ── Final drop list ────────────────────────────────────────────────────────────
def derive_drop_list(classified: list[dict]) -> tuple[list[str], list[str]]:
    """Separate confirmed drops (status 'stable' ONLY: Pearson AND Spearman, both years)
    from monitored pairs (everything else, including 'stable_pearson_only')."""
    to_drop: set[str]   = set()
    to_monitor: set[str] = set()
    for rec in classified:
        if rec['status'] == 'stable':
            to_drop.add(rec['drop'])
        else:
            to_monitor.add(rec['drop'])
    # a feature can't be both (stable wins)
    to_monitor -= to_drop
    return sorted(to_drop), sorted(to_monitor)


# ── Shifted pairs (for monitoring report) ─────────────────────────────────────
def shifted_pairs(
    pearson_17: np.ndarray,
    pearson_18: np.ndarray,
    features: list[str],
    per_year_thr: float,
    consensus_thr: float,
) -> list[dict]:
    """Pairs correlated in one year but NOT in the consensus graph — correlation drift."""
    n = len(features)
    # triu_indices(n, k=1) walks (i, j) pairs in the same row-major order (i outer,
    # j inner, j > i) as the nested loop it replaces. The final `sorted(...)` re-orders
    # by key, but ties are broken by the stable sort's input order, so preserving this
    # pre-sort order keeps tie-breaking identical to the original loop.
    iu, ju = np.triu_indices(n, k=1)
    p17_all = np.abs(pearson_17[iu, ju]).astype(float)
    p18_all = np.abs(pearson_18[iu, ju]).astype(float)
    avg_all = (p17_all + p18_all) / 2
    in_17_all = p17_all >= per_year_thr
    in_18_all = p18_all >= per_year_thr
    # Only record pairs that appear in one year but fell short of consensus
    keep = (in_17_all != in_18_all) & (avg_all < consensus_thr)
    shifts: list[dict] = [
        {
            'feature_a': features[i],
            'feature_b': features[j],
            'pearson_2017': round(p17, 6),
            'pearson_2018': round(p18, 6),
            'correlated_in': '2017_only' if in_17 else '2018_only',
        }
        for i, j, p17, p18, in_17 in zip(
            iu[keep], ju[keep], p17_all[keep], p18_all[keep], in_17_all[keep]
        )
    ]
    return sorted(shifts, key=lambda x: -max(x['pearson_2017'], x['pearson_2018']))


# ── Visualisation ──────────────────────────────────────────────────────────────
def plot_diff_heatmap(
    pearson_17: np.ndarray,
    pearson_18: np.ndarray,
    features: list[str],
    out_path: Path,
    log: Logger,
):
    """Heatmap of (|r_2018| - |r_2017|): positive = grew more correlated, negative = less."""
    log.info('Rendering diff heatmap ...')
    diff = np.abs(pearson_18) - np.abs(pearson_17)
    np.fill_diagonal(diff, 0.0)

    n = len(features)
    cell = max(0.20, min(0.35, 14.0 / n))
    fig_size = max(14.0, n * cell)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    vabs = max(0.01, np.nanpercentile(np.abs(diff[diff != 0]), 95))
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)
    im = ax.imshow(diff, cmap='RdBu_r', norm=norm, aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label='|r|₂₀₁₈ − |r|₂₀₁₇  (blue = less correlated in 2018)')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(features, rotation=90, fontsize=4.5)
    ax.set_yticklabels(features, fontsize=4.5)
    ax.set_title('Correlation Drift: |Pearson 2018| − |Pearson 2017|', fontsize=9, pad=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    log.ok(f'Saved {out_path.name}')


def plot_min_matrix_heatmap(
    matrix: np.ndarray,
    features: list[str],
    metric_name: str,
    out_path: Path,
    log: Logger,
):
    """Heatmap of MIN correlation matrix (across years)."""
    log.info(f'Rendering {metric_name} MIN heatmap ...')
    n = len(features)
    cell = max(0.20, min(0.35, 14.0 / n))
    fig_size = max(14.0, n * cell)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(matrix, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='MIN |r| (across years)')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(features, rotation=90, fontsize=4.5)
    ax.set_yticklabels(features, fontsize=4.5)
    ax.set_title(f'MIN {metric_name} Correlation (2017 vs 2018)', fontsize=9, pad=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    log.ok(f'Saved {out_path.name}')


def plot_redundancy_groups(
    groups: list[list[str]],
    classified: list[dict],
    out_path: Path,
    log: Logger,
):
    """Bar chart of redundancy group sizes, coloured by keep (green) / drop (red)."""
    if not groups:
        log.info('No redundancy groups to plot.')
        return

    log.info('Rendering redundancy groups chart ...')
    # Collect keep/drop per group
    keeps  = {rec['keep'] for rec in classified}
    labels = []
    kept   = []
    dropped = []
    for g in sorted(groups, key=len, reverse=True):
        keep_feat = next(f for f in g if f in keeps)
        labels.append(keep_feat)
        kept.append(1)
        dropped.append(len(g) - 1)

    x = np.arange(len(labels))
    width = 0.55
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 5))
    ax.bar(x, kept,    width, label='Kept (representative)',  color='#2ecc71')
    ax.bar(x, dropped, width, bottom=kept, label='Dropped (redundant)', color='#e74c3c')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Features in group')
    ax.set_title('Redundancy Groups — Feature Kept vs Dropped per Group', fontsize=9)
    ax.legend(fontsize=7)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.ok(f'Saved {out_path.name}')


# ── Threshold sensitivity sweep ─────────────────────────────────────────────────
def threshold_sensitivity_sweep(
    min_pearson: dict, min_spearman: dict,
    pearson_17: dict, pearson_18: dict, spearman_17: dict, spearman_18: dict,
    features: list[str], avg_pearson: dict, avg_spearman: dict,
    official_threshold: float, official_drop: 'set[str]', log: Logger,
) -> list[dict]:
    """Re-run the SAME redundancy-group + drop-list logic at alternate consensus thresholds, using
    matrices already computed for the official run (cheap — no retraining, no change to the
    official 71-feature set). Reports how much the dropped-feature set would change at a stricter
    or looser cutoff than the official 0.95, since that threshold was previously never swept even
    though it determines the feature set every downstream step trains and tests on."""
    results = []
    for thr in Config3.SENSITIVITY_THRESHOLDS:
        groups = build_redundancy_groups(min_pearson, min_spearman, features, thr, log)
        classified = classify_pairs(groups, pearson_17, pearson_18, spearman_17, spearman_18,
                                    features, avg_pearson, avg_spearman, thr)
        dropped, _ = derive_drop_list(classified)
        dropped = set(dropped)
        diff = dropped.symmetric_difference(official_drop)
        results.append({
            'threshold': thr,
            'n_dropped': len(dropped),
            'n_features_remaining': len(features) - len(dropped),
            'n_diff_from_official': len(diff),
            'dropped_not_in_official': sorted(dropped - official_drop),
            'official_not_in_dropped': sorted(official_drop - dropped),
        })
        log.info(f'  threshold={thr}: {len(dropped)} dropped ({len(features) - len(dropped)} '
                 f'remain), {len(diff)} differ from the official {official_threshold} drop set')
    return results


# ── JSON output ────────────────────────────────────────────────────────────────
def save_drop_decisions(
    features_to_drop: list[str],
    features_to_monitor: list[str],
    groups: list[list[str]],
    classified: list[dict],
    drift_pairs: list[dict],
    threshold: float,
    per_year_thr: float,
    n_features_total: int,
    out_path: Path,
    log: Logger,
):
    keeps_map: dict[str, str] = {}
    for rec in classified:
        keeps_map[rec['drop']] = rec['keep']

    # Group `classified` by frozenset(group) once, instead of re-scanning the full
    # list for every group encountered below.
    by_group: dict[frozenset, list[dict]] = defaultdict(list)
    for rec in classified:
        by_group[frozenset(rec['group'])].append(rec)

    redundancy_groups_out = []
    seen_groups: list[frozenset] = []
    for rec in classified:
        fs = frozenset(rec['group'])
        if fs in seen_groups:
            continue
        seen_groups.append(fs)
        g = rec['group']
        # All recs in a group share the representative -> read it directly.
        keep_feat = rec['keep']
        drop_feats = [f for f in g if f != keep_feat]
        group_recs = by_group[fs]
        statuses = [r['status'] for r in group_recs]
        redundancy_groups_out.append({
            'group':  g,
            'keep':   keep_feat,
            'drop':   drop_feats,
            'status': statuses[0] if statuses else 'unknown',
            'avg_pearson_within': round(
                float(np.mean([r['avg_pearson'] for r in group_recs])), 6
            ),
        })

    doc = {
        'generated':         datetime.now().isoformat(timespec='seconds'),
        'note':              (
            'Derived from CORRECTED CIC-IDS 2017/2018 datasets. '
            'Do not substitute published feature lists from original (uncorrected) releases.'
        ),
        'strategy':          'consensus_intersection_both_metrics_both_years',
        'consensus_rule':    'min(|r_2017|, |r_2018|) >= threshold for BOTH Pearson and Spearman',
        'consensus_threshold':  threshold,
        'per_year_threshold':   per_year_thr,
        'criterion':         'mRMR minimum-redundancy (lowest avg |r| to all features)',
        # Real common-feature count passed in from main (was a hardcoded 81 that
        # double-counted group members -> reported 88).
        'n_features_total':  n_features_total,
        'features_to_drop':  features_to_drop,
        'features_to_monitor': features_to_monitor,
        'redundancy_groups': redundancy_groups_out,
        'shifted_pairs':     drift_pairs[:50],   # top 50 by correlation magnitude
        'usage': {
            'drop':    'Remove features_to_drop before any ML training/preprocessing.',
            'monitor': (
                'features_to_monitor are correlated in only one year — review manually. '
                'If training on combined 2017+2018, they are safe to keep.'
            ),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(doc, f, indent=2)
    log.ok(f'Saved {out_path.name}  ({len(features_to_drop)} drop, {len(features_to_monitor)} monitor)')


def save_min_matrix(
    matrix: np.ndarray,
    features: list[str],
    out_path: Path,
    log: Logger,
):
    """Save MIN correlation matrix as JSON."""
    doc = {
        'features': features,
        'matrix': matrix.tolist(),
        'description': 'Minimum absolute correlation across 2017 and 2018',
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(doc, f, indent=2)
    log.ok(f'Saved {out_path.name}')


def save_comparison_matrix(
    avg_pearson: np.ndarray,
    avg_spearman: np.ndarray,
    pearson_17: np.ndarray,
    pearson_18: np.ndarray,
    features: list[str],
    out_path: Path,
    log: Logger,
):
    doc = {
        'features':       features,
        'avg_pearson':    avg_pearson.tolist(),
        'avg_spearman':   avg_spearman.tolist(),
        'pearson_2017':   pearson_17.tolist(),
        'pearson_2018':   pearson_18.tolist(),
        'diff_pearson':   (np.abs(pearson_18) - np.abs(pearson_17)).tolist(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(doc, f, indent=2)
    log.ok(f'Saved {out_path.name}')


# ── Report ────────────────────────────────────────────────────────────────────
def write_report(
    features: list[str],
    groups: list[list[str]],
    classified: list[dict],
    features_to_drop: list[str],
    features_to_monitor: list[str],
    drift_pairs: list[dict],
    threshold: float,
    per_year_thr: float,
    out_path: Path,
    log: Logger,
):
    lines: list[str] = []

    # `classified['drop']` values are unique across the whole list (redundancy groups
    # are disjoint connected components, so each dropped feature belongs to exactly one
    # group/record) — build the lookup once instead of linear-scanning `classified` for
    # every feature below.
    by_drop: dict[str, dict] = {r['drop']: r for r in classified}

    def _fmt(v): return f'{v:.4f}' if isinstance(v, float) else str(v)

    def h(title: str):
        lines.append('')
        lines.append('=' * 70)
        lines.append(title)
        lines.append('=' * 70)

    def s(title: str):
        lines.append('')
        lines.append('-' * 60)
        lines.append(title)
        lines.append('-' * 60)

    h('CORRELATION COMPARISON REPORT  --  CIC-IDS 2017 vs 2018 (corrected)')
    lines.append(f'Generated : {datetime.now():%Y-%m-%d %H:%M:%S}')

    h('SETTINGS')
    lines.append(f'  Features compared     : {len(features)}')
    lines.append(f'  Consensus threshold   : {threshold}')
    lines.append(f'  Per-year threshold    : {per_year_thr}')
    lines.append('')

    h('SUMMARY')
    lines.append(f'  Redundancy groups (pairs to consolidate) : {len(groups)}')
    lines.append(f'  Features recommended to DROP              : {len(features_to_drop)}')
    lines.append(f'  Features recommended to MONITOR (1 year) : {len(features_to_monitor)}')
    lines.append(f'  Correlation drift pairs (shifted only)   : {len(drift_pairs)}')

    if features_to_drop:
        h('FEATURES TO DROP  (stable redundancy — both years, both metrics)')
        for i, f in enumerate(features_to_drop, 1):
            kept_as = by_drop[f]['keep'] if f in by_drop else '?'
            lines.append(f'  {i:3d}.  {f:<42}  → kept: {kept_as}')

    if groups:
        h('REDUNDANCY GROUPS (detail)')
        seen: set[frozenset] = set()
        for rec in classified:
            fs = frozenset(rec['group'])
            if fs in seen:
                continue
            seen.add(fs)
            g = rec['group']
            # All classified recs in a group share the same representative -> just read it.
            keep_feat = rec['keep']
            drops_in_group = [f for f in g if f != keep_feat]
            status = rec['status']
            lines.append('')
            lines.append(f'  Group ({len(g)} features)  status={status}')
            lines.append(f'    KEEP: {keep_feat}')
            for d in drops_in_group:
                # `drop` values are unique across `classified` (see by_drop above), so the
                # frozenset(group)==fs filter in the original scan was always redundant with
                # matching on `drop` alone; by_drop.get(d, {}) is exactly equivalent.
                row = by_drop.get(d, {})
                lines.append(
                    f'    DROP: {d:<42} '
                    f'  P17={_fmt(row.get("pearson_2017", "?"))}  '
                    f'P18={_fmt(row.get("pearson_2018", "?"))}  '
                    f'avg_P={_fmt(row.get("avg_pearson", "?"))}  '
                    f'avg_S={_fmt(row.get("avg_spearman", "?"))}'
                )

    if features_to_monitor:
        h('FEATURES TO MONITOR  (correlated in ONE year only — do not drop blindly)')
        lines.append(
            '  These features may appear redundant in one year but carry different\n'
            '  information in the other.  Safe to keep when training on both datasets.'
        )
        for f in features_to_monitor:
            row = by_drop.get(f, {})
            lines.append(
                f'  {f:<42}  status={row.get("status","?")}'
                f'  P17={_fmt(row.get("pearson_2017","?"))}  P18={_fmt(row.get("pearson_2018","?"))}'
            )

    if drift_pairs:
        h('CORRELATION DRIFT PAIRS  (new/lost correlation between years)')
        lines.append(
            '  Pairs where |Pearson| >= per_year_threshold in one year but NOT the other.\n'
            '  These represent structural changes in network traffic between 2017 and 2018.\n'
            '  Not candidates for dropping — they are a research finding.\n'
        )
        lines.append(f'  {"Feature A":<40}  {"Feature B":<40}  {"In":<10}  P17      P18')
        lines.append(f'  {"-"*40}  {"-"*40}  {"-"*10}  -------  -------')
        for p in drift_pairs[:30]:
            lines.append(
                f'  {p["feature_a"]:<40}  {p["feature_b"]:<40}  '
                f'{p["correlated_in"]:<10}  '
                f'{p["pearson_2017"]:.4f}   {p["pearson_2018"]:.4f}'
            )
        if len(drift_pairs) > 30:
            lines.append(f'  ... and {len(drift_pairs) - 30} more (see drop_decisions.json → shifted_pairs)')

    h('OUTPUT FILES')
    lines.append('  For Script 4 (Preprocessing):')
    lines.append('    - output/3_correlation_comparison/drop_decisions.json')
    lines.append('  For Manual Review:')
    lines.append('    - results/3_correlation_comparison/pearson_min_heatmap.png')
    lines.append('    - results/3_correlation_comparison/spearman_min_heatmap.png')
    lines.append('    - results/3_correlation_comparison/diff_heatmap.png')
    lines.append('    - results/3_correlation_comparison/redundancy_groups.png')

    text = '\n'.join(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text)
    log.ok(f'Saved {out_path.name}')
    # also echo to console
    print(text)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    threshold    = Config3.CONSENSUS_THRESHOLD
    per_year_thr = Config3.PER_YEAR_THRESHOLD

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log = Logger(RESULTS_DIR / Config3.STEPS_FILE, step_prefix=3,
                 title='3_CORRELATION_COMPARISON STEPS LOG')

    # ── Step 1: load matrices ──────────────────────────────────────────────────
    log.step('Load correlation matrices from script 2 output')
    matrices: dict[str, dict] = {}
    for ds in DATASETS:
        for metric in ('pearson', 'spearman'):
            key  = f'{metric}_{ds}'
            path = Config3.INPUT_BASE / ds / f'{metric}_matrix.json'
            if not path.exists():
                log.warn(f'Missing: {path}  — run 2_correlation_analysis.py first')
                sys.exit(1)
            matrices[key] = load_matrix(path, log)
    log.ok('All 4 matrices loaded')

    # ── Step 2: align features ─────────────────────────────────────────────────
    log.step('Align to common feature set')
    features = align_features(matrices, log)

    pearson_17  = reindex_matrix(matrices['pearson_cicids2017'],  features)
    pearson_18  = reindex_matrix(matrices['pearson_cicids2018'],  features)
    spearman_17 = reindex_matrix(matrices['spearman_cicids2017'], features)
    spearman_18 = reindex_matrix(matrices['spearman_cicids2018'], features)
    log.ok(f'Matrices reindexed to {len(features)} common features')

    # ── Step 3: aggregate correlation matrices (avg for report, min for decision) ──
    log.step('Compute aggregated correlation matrices (average + intersection/min)')
    avg_pearson  = average_abs_matrix([pearson_17, pearson_18])
    avg_spearman = average_abs_matrix([spearman_17, spearman_18])
    min_pearson  = min_abs_matrix([pearson_17, pearson_18])
    min_spearman = min_abs_matrix([spearman_17, spearman_18])
    log.ok('Average + min matrices computed')

    # ── Step 4: save MIN Pearson matrix ────────────────────────────────────────
    log.step('Generate MIN Pearson matrix and heatmap')
    plot_min_matrix_heatmap(min_pearson, features, 'Pearson',
                            RESULTS_DIR / 'pearson_min_heatmap.png', log)
    save_min_matrix(min_pearson, features,
                    OUTPUT_DIR / 'min_pearson_matrix.json', log)
    log.ok('Saved min_pearson_matrix.json and pearson_min_heatmap.png')

    # ── Step 5: save MIN Spearman matrix ───────────────────────────────────────
    log.step('Generate MIN Spearman matrix and heatmap')
    plot_min_matrix_heatmap(min_spearman, features, 'Spearman',
                            RESULTS_DIR / 'spearman_min_heatmap.png', log)
    save_min_matrix(min_spearman, features,
                    OUTPUT_DIR / 'min_spearman_matrix.json', log)
    log.ok('Saved min_spearman_matrix.json and spearman_min_heatmap.png')

    # ── Step 6: build consensus graph and find redundancy groups ──────────────
    log.step(f'Build consensus correlation graph (threshold={threshold})')
    groups = build_redundancy_groups(min_pearson, min_spearman, features, threshold, log)
    for g in groups:
        log.info(f'  Group: {" | ".join(g)}')
    log.ok(f'{len(groups)} redundancy groups identified')

    # ── Step 7: classify pairs ─────────────────────────────────────────────────
    log.step('Classify pairs (stable / shifted / weak)')
    classified = classify_pairs(
        groups, pearson_17, pearson_18, spearman_17, spearman_18, features,
        avg_pearson, avg_spearman, threshold
    )
    by_status: dict[str, int] = defaultdict(int)
    for r in classified:
        by_status[r['status']] += 1
    for status, cnt in sorted(by_status.items()):
        log.info(f'  {status}: {cnt} pair(s)')
    log.ok('Pair classification complete')

    # ── Step 8: derive drop list ───────────────────────────────────────────────
    log.step('Derive final drop list')
    features_to_drop, features_to_monitor = derive_drop_list(classified)
    log.info(f'  Drop    : {len(features_to_drop)}  features')
    log.info(f'  Monitor : {len(features_to_monitor)} features')
    log.ok('Drop list finalized')

    # ── Step 9: correlation drift pairs ───────────────────────────────────────
    log.step('Detect correlation drift pairs (shifted between years)')
    drift = shifted_pairs(pearson_17, pearson_18, features, per_year_thr, threshold)
    log.info(f'  Drift pairs found: {len(drift)}')
    log.ok('Drift detection complete')

    # ── Step 9b: consensus-threshold sensitivity sweep ─────────────────────────
    log.step(f'Sweep consensus threshold {Config3.SENSITIVITY_THRESHOLDS} (sensitivity check, '
             f'does not change the official {threshold} drop list)')
    sensitivity = threshold_sensitivity_sweep(
        min_pearson, min_spearman, pearson_17, pearson_18, spearman_17, spearman_18,
        features, avg_pearson, avg_spearman, threshold, set(features_to_drop), log)
    (OUTPUT_DIR / 'threshold_sensitivity.json').write_text(
        json.dumps({'official_threshold': threshold, 'sweep': sensitivity}, indent=2),
        encoding='utf-8')
    log.ok('Saved threshold_sensitivity.json')

    # ── Step 10: plots ──────────────────────────────────────────────────────────
    log.step('Generate visualisations (diff heatmap + redundancy groups)')
    plot_diff_heatmap(pearson_17, pearson_18, features,
                      RESULTS_DIR / 'diff_heatmap.png', log)
    plot_redundancy_groups(groups, classified,
                           RESULTS_DIR / 'redundancy_groups.png', log)
    log.ok('Visualisations complete')

    # ── Step 11: save outputs ──────────────────────────────────────────────────
    log.step('Save JSON outputs (drop decisions + comparison matrices)')
    save_drop_decisions(
        features_to_drop, features_to_monitor, groups, classified, drift,
        threshold, per_year_thr, len(features),
        OUTPUT_DIR / 'drop_decisions.json', log
    )
    save_comparison_matrix(
        avg_pearson, avg_spearman, pearson_17, pearson_18, features,
        OUTPUT_DIR / 'comparison_matrix.json', log
    )
    log.ok('JSON outputs saved')

    # ── Step 12: write report ──────────────────────────────────────────────────
    log.step('Write human-readable report')
    write_report(
        features, groups, classified,
        features_to_drop, features_to_monitor, drift,
        threshold, per_year_thr,
        RESULTS_DIR / Config3.RESULTS_FILE, log
    )

    log.section('COMPLETE')
    log.info(f'Drop decisions : {OUTPUT_DIR / "drop_decisions.json"}')
    log.info(f'Report         : {RESULTS_DIR / Config3.RESULTS_FILE}')
    log.info(f'Diff heatmap   : {RESULTS_DIR / "diff_heatmap.png"}')
    log.close()


if __name__ == '__main__':
    main()
