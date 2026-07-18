"""
8_visualize.py — Per-feature raincloud visualizer for NIDS datasets.

PURPOSE:
  Render the human-facing picture for each feature as a PURE CONSUMER of the profiler (script 7).
  It computes no distribution facts itself — it reads the profile's scale, view range, modes,
  masses, separation and shape stats and draws them, so the picture and the numbers can never
  disagree. Reads the cleaned, raw-unit parquet from script 1 one column at a time.

SUB-STEPS:
  Sub-step 8.1 (Discover features):
    - List the features to plot from the profiler output (skips identifiers).
    Input: output/7_profile/<ds>/profiles.json
    Output: ordered list of feature names

  Sub-step 8.2 (Render plots):
    - For each feature (parallel, one process per feature; label read once per worker), draw four
      variants: binary + multiclass, each in a standard view (outliers clipped to the profile
      view range) and an *_extended view (full min..max so outliers are shown).
    Input: features from 8.1 + data/cc_data/<ds>_cleaned.parquet (one column at a time)
    Output: results/8_visualize/<ds>/{binary,binary_extended,multiclass,multiclass_extended}/*.png
            + results/8_visualize/<ds>/filename_manifest.json

  Sub-step 8.3 (Write report):
    - Write a brief human-readable summary (plot counts per variant, output paths).
    Input: render manifest from 8.2
    Output: results/8_visualize/<ds>/8_visualize_report.txt

GUARANTEES:
  - No source data is modified (read-only on the cleaned parquet).
  - Every drawn fact (scale, view range, modes, zero/outlier masses, shape) comes from the
    profile; the visualizer never re-derives distribution facts.
  - Filenames are sanitized (feature names contain '/'), output is flat per variant folder.

NOTES:
  - Memory model (2018 has 63M rows): the Label column is read ONCE per dataset (int16 codes),
    and each feature column is read ONCE and drawn for all four variants, then dropped.
  - The standard view hides extreme outliers so the body of the distribution is readable; the
    *_extended view spans the full range so the outliers the standard view clips are visible.
"""

import os
import sys
import gc
import json
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.stats import gaussian_kde

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unified_config import (
    Config8, Logger, ensure_results_dir, DATASETS, plan_workers,
    BENIGN_LABEL, profile_json_path, read_feature_only, read_labels_encoded, safe_filename,
)

_PALETTE = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12',
            '#1abc9c', '#e67e22', '#34495e', '#fd79a8', '#7f8c8d']


def _build_variants() -> list:
    """The (folder, class_mode, extended) variants to render. Folder is the on-disk subdir."""
    variants = []
    for cm in Config8.CLASS_MODES:
        variants.append((cm, cm, False))
        if Config8.INCLUDE_EXTENDED:
            variants.append((f'{cm}_extended', cm, True))
    return variants


def _extended_range(profile: dict, values: np.ndarray) -> tuple:
    """Full-range view for the *_extended plots: span min..max so clipped outliers are shown.
    Impossible (negative) values stay excluded — they are extraction bugs, not real outliers."""
    lo = float(profile.get('min', np.nanmin(values)))
    hi = float(profile.get('max', np.nanmax(values)))
    if profile.get('has_impossible_values'):
        lo = max(lo, 0.0)
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _apply_scale(ax, scale: str, scale_param):
    if scale == 'log':
        ax.set_xscale('log')
    elif scale == 'symlog':
        ax.set_xscale('symlog', linthresh=scale_param or 1e-6)
    # categorical/linear: leave default linear axis (categorical handled by integer ticks)


def _scale_transform(scale: str, scale_param):
    """(forward, inverse) pair matching the axis scale, so the KDE is estimated in the SAME
    space the axis displays. A KDE fit on native values over a linear grid but drawn on a
    log/symlog axis visually distorts the cloud (mass near zero stretched, tails compressed)."""
    if scale == 'log':
        return (lambda x: np.log10(x), lambda t: np.power(10.0, t))
    if scale == 'symlog':
        s = float(scale_param or 1e-6)
        return (lambda x: np.arcsinh(x / s), lambda t: np.sinh(t) * s)
    return (lambda x: x, lambda t: t)


def _kde_curve(v: np.ndarray, lo: float, hi: float, n: int = 220,
               scale: str = 'linear', scale_param=None):
    """KDE density over [lo, hi] estimated in the axis's transformed space, normalized to max 1.
    Returns (xs_native, dens) or None if degenerate (constant / too few points / singular).
    The density CLOUD is what makes bimodal blobs visible."""
    v = v[np.isfinite(v)]
    if scale == 'log':
        v = v[v > 0]
        if v.size:
            lo = max(lo, float(v.min()))      # log10 needs a positive lower bound
    if v.size < 20 or np.ptp(v) < 1e-12:
        return None
    fwd, inv = _scale_transform(scale, scale_param)
    tv = fwd(v.astype(np.float64))
    tv = tv[np.isfinite(tv)]
    if tv.size < 20 or np.ptp(tv) < 1e-12:
        return None
    try:
        kde = gaussian_kde(tv)
        t_lo, t_hi = float(fwd(np.float64(lo))), float(fwd(np.float64(hi)))
        if not np.isfinite(t_lo) or not np.isfinite(t_hi) or t_hi <= t_lo:
            return None
        ts = np.linspace(t_lo, t_hi, n)
        d = kde(ts)
        mx = float(np.max(d))
        if not np.isfinite(mx) or mx <= 0:
            return None
        return inv(ts), d / mx
    except Exception:
        return None


def _draw_continuous(ax, lanes, scale, scale_param, view_lo, view_hi, rain_cap):
    """
    TRUE raincloud per class lane: half-violin KDE density 'cloud' above the baseline, a box
    summary (Q1-Q3 + median) just below it, and the jittered raw-point 'rain' below that. The cloud
    is what reveals bimodal blobs that the old strip-only plot smeared into a single band.
    """
    rng = np.random.default_rng(Config8.SEED)
    cloud_h = 0.42
    kde_cap = Config8.KDE_SAMPLE_CAP
    for i, (name, vals, color) in enumerate(lanes):
        v = vals[np.isfinite(vals)]
        v = v[(v >= view_lo) & (v <= view_hi)]
        if v.size == 0:
            continue
        # cloud (KDE on a capped sample for a smooth curve, estimated in the axis's scale space)
        vk = v if v.size <= kde_cap else rng.choice(v, kde_cap, replace=False)
        curve = _kde_curve(vk, view_lo, view_hi, scale=scale, scale_param=scale_param)
        if curve is not None:
            xs, dens = curve
            ax.fill_between(xs, i + 0.08, i + 0.08 + cloud_h * dens,
                            color=color, alpha=0.28, linewidth=0)
        # box summary (Q1-Q3 bar + median tick) just below the baseline
        q1, med, q3 = (float(q) for q in np.quantile(v, [0.25, 0.5, 0.75]))
        ax.plot([q1, q3], [i - 0.06, i - 0.06], color=color, lw=4, alpha=0.75,
                solid_capstyle='butt')
        ax.plot([med, med], [i - 0.13, i + 0.01], color='black', lw=1.1, alpha=0.85)
        # rain (jittered raw points) below the box; lighter/smaller when dense so it stays readable
        vs = v if v.size <= rain_cap else rng.choice(v, rain_cap, replace=False)
        ps, pa = (7, 0.28) if vs.size <= 8000 else (4, 0.14)
        yj = (i - 0.28) + rng.normal(0, 0.04, size=vs.size)
        ax.scatter(vs, yj, s=ps, alpha=pa, color=color, edgecolors='none')
    _apply_scale(ax, scale, scale_param)
    ax.set_xlim(view_lo, view_hi)


def _draw_discrete(ax, lanes, view_lo, view_hi):
    """Discrete features on exact integer positions per lane -- no continuous smear."""
    finite_all = np.concatenate([v[np.isfinite(v)] for _, v, _ in lanes if v.size]) \
        if any(v.size for _, v, _ in lanes) else np.array([0.0])
    finite_all = finite_all[(finite_all >= view_lo) & (finite_all <= view_hi)]
    levels = np.unique(np.round(finite_all)).astype(int)
    if levels.size == 0 or levels.size > Config8.DISCRETE_MAX_LEVELS:
        levels = None
    for i, (name, vals, color) in enumerate(lanes):
        v = vals[np.isfinite(vals)]
        v = v[(v >= view_lo) & (v <= view_hi)]
        if levels is not None:
            vals_round = np.round(v).astype(int)
            uniq, cnt = np.unique(vals_round, return_counts=True)
            frac = cnt / cnt.sum() if cnt.sum() else cnt
            ax.scatter(uniq, np.full(uniq.size, i), s=40 + 400 * frac, alpha=0.7, color=color)
        else:
            rng = np.random.default_rng(Config8.SEED)
            y = i + rng.normal(0, 0.03, size=v.size)
            ax.scatter(v, y, s=8, alpha=0.3, color=color)
    if levels is not None:
        ax.set_xticks(levels)


def _build_lanes(values, class_mode, codes, y_bin, present, cat_index):
    """Build (classname, values_for_class, color) lanes from precomputed label codes."""
    if class_mode == 'binary':
        classes = [BENIGN_LABEL, 'Attack']
        masks = [y_bin == 0, y_bin == 1]
    else:
        classes = present
        masks = [codes == cat_index[c] for c in present]
    return [(c, values[m], _PALETTE[i % len(_PALETTE)])
            for i, (c, m) in enumerate(zip(classes, masks))]


def plot_one(ds, feature, profile, values, class_mode, extended,
             codes, y_bin, present, cat_index, out_dir, rain_cap):
    """Draw one feature for one variant using the already-loaded column + precomputed label codes."""
    classes = [BENIGN_LABEL, 'Attack'] if class_mode == 'binary' else present
    lanes = _build_lanes(values, class_mode, codes, y_bin, present, cat_index)

    scale = profile['recommended_scale']
    scale_param = profile.get('scale_param')
    det = profile['detected_type']
    if extended:
        view_lo, view_hi = _extended_range(profile, values)
    else:
        view_lo = profile.get('view_low', float(np.nanmin(values)))
        view_hi = profile.get('view_high', float(np.nanmax(values)))

    width = Config8.FIG_WIDTH_EXTENDED if extended else Config8.FIG_WIDTH
    dpi   = Config8.FIG_DPI_EXTENDED if extended else Config8.FIG_DPI
    fig, ax = plt.subplots(figsize=(width, max(3, 0.6 * len(classes) + 2)))

    discrete_like = det in ('binary', 'low-cardinality-discrete', 'discrete-count', 'nominal')
    if discrete_like:
        _draw_discrete(ax, lanes, view_lo, view_hi)
    else:
        _draw_continuous(ax, lanes, scale, scale_param, view_lo, view_hi, rain_cap)

    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel(f'{feature}  (native units, scale={scale})', fontsize=9)

    # Distribution-aware annotations, all sourced from the richer script-7 profile.
    notes = []
    zf = profile.get('zero_fraction', 0.0)
    if zf >= 0.10:
        notes.append(f'{zf*100:.0f}% = 0')
    if profile.get('has_impossible_values'):
        notes.append('impossible(neg) excluded')
    sk, ku = profile.get('skewness'), profile.get('kurtosis')
    if sk is not None and ku is not None:
        notes.append(f'skew={sk:.2f}, kurt={ku:.2f}')
    n_out = profile.get('n_clipped_low', 0) + profile.get('n_clipped_high', 0)
    ofrac = profile.get('outlier_fraction', 0.0)
    if extended:
        if n_out:
            notes.append(f'full range — {n_out:,} outliers shown ({ofrac*100:.2f}%)')
    elif n_out:
        notes.append(f'{n_out:,} outliers beyond view')
    title = f'{ds} — {feature}  [{det}]'
    if notes:
        title += '\n' + ' | '.join(notes)
    ax.set_title(title, fontsize=9)

    fig.tight_layout()
    out_path = out_dir / f'{safe_filename(feature)}.png'
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return out_path.name


# ── Parallel plotting (one process per feature; label + profiles read ONCE per worker) ──────────
_VS: dict = {}   # per-worker state, populated by the pool initializer


def _viz_init(ds, rdir_str, variants):
    """Each worker loads the profiles + dictionary-encoded label ONCE, then plots its features."""
    with open(profile_json_path(ds), encoding='utf-8') as f:
        profiles = json.load(f)
    codes, categories, y_bin = read_labels_encoded(ds)
    present = [c for c in [BENIGN_LABEL] + sorted(set(categories) - {BENIGN_LABEL})
               if c in categories]
    rdir = Path(rdir_str)
    out_dirs = {}
    for folder, _class_mode, _extended in variants:
        d = rdir / folder
        d.mkdir(parents=True, exist_ok=True)
        out_dirs[folder] = d
    rain_cap = Config8.RAIN_POINTS_PER_CLASS.get(ds, Config8.RAIN_POINTS_DEFAULT)
    _VS.update(ds=ds, profiles=profiles, codes=codes, y_bin=y_bin,
               cat_index={c: i for i, c in enumerate(categories)},
               present=present, out_dirs=out_dirs, variants=variants, rain_cap=rain_cap)


def _viz_one(feature):
    """Worker task: plot one feature (all variants) using the per-worker shared state."""
    profile = _VS['profiles'].get(feature)
    if profile is None or profile.get('is_identifier'):
        return feature, {}, None
    try:
        values = np.asarray(read_feature_only(_VS['ds'], feature), dtype=np.float64)
        entry = {}
        for folder, class_mode, extended in _VS['variants']:
            fname = plot_one(_VS['ds'], feature, profile, values, class_mode, extended,
                             _VS['codes'], _VS['y_bin'], _VS['present'], _VS['cat_index'],
                             _VS['out_dirs'][folder], _VS['rain_cap'])
            entry[folder] = fname
        del values
        gc.collect()
        return feature, entry, None
    except Exception as e:
        return feature, {}, f'{type(e).__name__}: {e}'


def discover_features(ds: str):
    """Read the profiler output and return (feature_list, profile_path); feats is None if missing."""
    profile_file = profile_json_path(ds)
    if not profile_file.exists():
        return None, profile_file
    with open(profile_file, encoding='utf-8') as f:
        feats = list(json.load(f).keys())
    return feats, profile_file


def render_features(ds, feats, rdir, variants, n_workers, log) -> tuple:
    """Render every feature in parallel; collect a folder -> {filename: feature} manifest."""
    manifest: dict = {}
    n = len(feats)
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_viz_init,
                             initargs=(ds, str(rdir), variants)) as ex:
        futs = {ex.submit(_viz_one, f): f for f in feats}
        for i, fut in enumerate(as_completed(futs), 1):
            feature, entry, err = fut.result()
            if err:
                log.warn(f'failed {feature}: {err}')
            for folder, fname in entry.items():
                manifest.setdefault(folder, {})[fname] = feature
            if i % 20 == 0:
                log.info(f'{i}/{n} features plotted')

    # safe-filename -> real-feature manifest (per variant folder) so nothing is ambiguous
    man_path = rdir / 'filename_manifest.json'
    with open(man_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    return manifest, man_path


def _write_visualize_report(ds: str, rdir: Path, manifest: dict, log: Logger) -> None:
    lines: list[str] = []

    def h(t: str) -> None:
        lines.extend(['', '=' * 70, t, '=' * 70])

    total = sum(len(v) for v in manifest.values())
    h(f'VISUALIZE REPORT  —  {ds}')
    lines.append(f'Generated : {datetime.now():%Y-%m-%d %H:%M:%S}')
    lines.append(f'Plots     : {total} PNGs across {len(manifest)} variant folders')

    h('OUTPUT LOCATIONS')
    for folder in sorted(manifest):
        lines.append(f'  {folder:<20} : {rdir / folder}  ({len(manifest[folder])} plots)')
    lines.append(f'  {"manifest":<20} : {rdir / "filename_manifest.json"}')
    lines.append('')
    lines.append('  Standard folders clip outliers to the profile view range (readable body);')
    lines.append('  *_extended folders span the full min..max so the clipped outliers are shown.')
    lines.append('  The manifest maps each safe filename back to its real feature name.')

    out_path = rdir / Config8.RESULTS_FILE
    out_path.write_text('\n'.join(lines), encoding='utf-8')
    log.ok(f'Saved {out_path.name}')


def main():
    ap = argparse.ArgumentParser(description='Script 8: visualize features from the profiler output.')
    ap.add_argument('--datasets', nargs='+', default=list(DATASETS))
    args = ap.parse_args()

    n_workers = plan_workers(os.cpu_count() or 4, Config8.MAX_WORKERS)
    variants = _build_variants()

    for ds in args.datasets:
        rdir = ensure_results_dir(8, ds)
        log = Logger(rdir / Config8.STEPS_FILE, step_prefix=8, title=f'SCRIPT 8 VISUALIZE — {ds}')
        rain = Config8.RAIN_POINTS_PER_CLASS.get(ds, Config8.RAIN_POINTS_DEFAULT)
        log.info(f'workers={n_workers}, variants={[v[0] for v in variants]}, '
                 f'rain/class={rain:,}')

        # 8.1 — discover features from the profiler output
        log.step('Discover features')
        feats, profile_file = discover_features(ds)
        if feats is None:
            log.warn(f'profiles missing: {profile_file} — run 7_profile.py first')
            log.step_end()
            log.close()
            continue
        log.ok(f'{len(feats)} features to plot')
        log.step_end()

        # 8.2 — render all variants in parallel
        log.step('Render plots')
        manifest, man_path = render_features(ds, feats, rdir, variants, n_workers, log)
        log.ok(f'plots written under {rdir} (+ {man_path.name})')
        log.step_end()

        # 8.3 — write the brief human-readable report
        log.step('Write report')
        _write_visualize_report(ds, rdir, manifest, log)
        log.step_end()

        log.close()


if __name__ == '__main__':
    main()
