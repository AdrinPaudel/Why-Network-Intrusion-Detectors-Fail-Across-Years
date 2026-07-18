"""
9_plan_comparison.py — Cross-dataset comparison planner for NIDS features (Branch B).

PURPOSE:
  Given the two per-feature profiles produced by script 7 (2017 and 2018), decide HOW each
  feature should be compared across years: which metric family, which CORROBORATION metrics
  (the Axis-1 verdict itself is always calibrated C2ST-AUC), mode alignment,
  normalization, and split. It does NOT compute any verdict — script 10 executes the plan.
  It is a PURE CONSUMER of script 7's profiles.json (never re-derives distribution facts).

SUB-STEPS:
  Sub-step 9.1 (Load profiles):
    - Read both datasets' profiles and confirm they exist.
    Input: output/7_profile/<ds1>/profiles.json, output/7_profile/<ds2>/profiles.json
    Output: two profile dicts in memory

  Sub-step 9.2 (Plan comparisons):
    - For each feature present in both years, route on the JOINT shape of the two years:
        * same family + same modality -> PMF corroboration (nominal / discrete-count),
          Wasserstein corroboration (both unimodal), or per-mode MMD corroboration (same
          multimodal structure);
        * any structural MISMATCH (cross-family type change, or modality change such as unimodal
          in one year vs bimodal in the other) -> 'structural_change'; step 10 stamps a
          'restructured' verdict when the C2ST evidence confirms it.
      The route only picks the CORROBORATION battery — the stable/shifted decision is always
      calibrated C2ST-AUC, the one metric computed identically for every route.
      The plan also carries the cross-year recommended_scale so step 10 compresses MMD's inputs
      for heavy-tailed/log features (the other metrics are rank/CDF-based and scale-invariant).
      Identifiers are skipped.
    Input: the two profile dicts from 9.1
    Output: a {feature: plan} dict

  Sub-step 9.3 (Save plans):
    - Write the machine-readable plans, the input contract for script 10.
    Input: plans from 9.2
    Output: output/9_plan_comparison/comparison_plans_<ds1>_<ds2>.json

  Sub-step 9.4 (Write report):
    - Write a brief human-readable summary (route / family / corroboration-metric counts, flags).
    Input: plans from 9.2
    Output: results/9_plan_comparison/9_plan_comparison_report.txt

GUARANTEES:
  - No source data is modified; only script-7 output is read.
  - Every routing decision comes from the profile (detected_type, n_modes, zero_inflated);
    the planner never re-measures a distribution.
  - Counts (discrete-count) are routed to PMF metrics, never Wasserstein, so their integer
    structure is respected. Their plan carries metric_family='nominal' so script 10's PMF path
    (Jensen-Shannon) executes them; the 'route' tag preserves their identity in the report.

NOTES:
  - Step 9 is CROSS-DATASET: one combined output folder, no per-dataset subfolders.
  - All routing tables (PMF types, type order, metric assignments) live in Config9 — the script
    body hardcodes no thresholds or metric names.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unified_config import Config9, Logger, DATASETS, profile_json_path


def _resolve_by_order(v1: str, v2: str, order: dict, default) -> str:
    """Shared template behind `_resolve_type`/`_resolve_scale`: pick whichever of v1/v2 ranks
    higher in `order` (ties keep v1), falling back to `default` for values `order` doesn't
    know about."""
    return v1 if order.get(v1, default) >= order.get(v2, default) else v2


def _resolve_type(det17: str, det18: str) -> str:
    """If the two years disagree on a feature's type (cardinality drift), route on the MORE
    GENERAL type (higher rank in Config9.TYPE_ORDER) so the metric is never under-specified."""
    return _resolve_by_order(det17, det18, Config9.TYPE_ORDER, 3)


def _family_of(det: str) -> str:
    """Coarse metric family of a detected_type: 'categorical' (PMF), 'count' (PMF), or
    'continuous' (distance metrics). A change of FAMILY between years is a structural change."""
    if det in Config9.NOMINAL_TYPES:
        return 'categorical'
    if det in Config9.DISCRETE_TYPES:
        return 'count'
    return 'continuous'


def _resolve_scale(scale17: str, scale18: str) -> str:
    """The more general scale wins (log > symlog > linear > categorical). Used by script 10 to
    compute the scale-sensitive secondaries (MMD/KS/Anderson-Darling) in the compressed space."""
    order = Config9.SCALE_ORDER
    floor = order[Config9.DEFAULT_SCALE]
    return _resolve_by_order(scale17, scale18, order, floor)


def _route_for(det17: str, det18: str, n_modes17: int, n_modes18: int) -> str:
    """Route on the JOINT shape of the two years (not just a single resolved type).

    Same family + same modality structure -> a shape-specific metric (PMF / Wasserstein /
    per-mode MMD). Any structural MISMATCH between the years -- a cross-family type change, or a
    modality change (e.g. unimodal in one year, bimodal in the other) -- has no shared template to
    align, so it routes to 'structural_change' where the shape-agnostic C2ST is the arbiter."""
    fam17, fam18 = _family_of(det17), _family_of(det18)
    if fam17 != fam18:
        return 'structural_change'          # cross-family change (e.g. binary <-> continuous)
    # (C2ST decides stable/shifted for every route; the route picks the corroboration battery.)
    if fam17 == 'categorical':
        return 'nominal'
    if fam17 == 'count':
        return 'discrete_count'
    # both continuous: route on modality compatibility
    if n_modes17 <= 1 and n_modes18 <= 1:
        return 'continuous_unimodal'
    if n_modes17 == n_modes18:
        return 'continuous_multimodal'      # same multimodal structure -> per-mode comparison
    return 'structural_change'              # modality mismatch -> shape changed between years


def plan_comparison(feature: str, p17: dict, p18: dict) -> dict:
    """Decide the comparison plan for one feature from its two yearly profiles."""
    det17 = p17.get('detected_type', Config9.DEFAULT_DETECTED_TYPE)
    det18 = p18.get('detected_type', Config9.DEFAULT_DETECTED_TYPE)
    det = _resolve_type(det17, det18)
    type_mismatch = det17 != det18
    zero_inflated = p17.get('zero_inflated', False) or p18.get('zero_inflated', False)
    n_modes17 = int(p17.get('n_modes', 1))
    n_modes18 = int(p18.get('n_modes', 1))
    modality_mismatch = n_modes17 != n_modes18
    recommended_scale = _resolve_scale(
        p17.get('recommended_scale', Config9.DEFAULT_SCALE),
        p18.get('recommended_scale', Config9.DEFAULT_SCALE))

    route = _route_for(det17, det18, n_modes17, n_modes18)
    m = Config9.METRICS[route]

    base = {
        'feature': feature, 'route': route,
        'corroboration_primary': m['corroboration_primary'],
        'corroboration_secondary': list(m['corroboration_secondary']),
        'split_strategy': Config9.DEFAULT_SPLIT_STRATEGY,
        'type_mismatch': type_mismatch, 'det_2017': det17, 'det_2018': det18,
        'n_modes_2017': n_modes17, 'n_modes_2018': n_modes18,
        'modality_mismatch': modality_mismatch,
        'recommended_scale': recommended_scale,
        'metric_family':      m['metric_family'],
        'alignment_strategy': m['alignment'],
        'normalization':      m['normalization'],
        'comparison_mode':    m['comparison_mode'],
    }

    # PMF routes (nominal + discrete-count): no separate zero treatment.
    if route in ('nominal', 'discrete_count'):
        return {**base, 'zero_mass_separate': False}

    # Continuous routes (incl. structural_change): zero-inflated tail handled separately by step 10.
    return {**base, 'zero_mass_separate': bool(zero_inflated)}


def load_profiles(ds1: str, ds2: str, log: Logger):
    """Load both datasets' profiles.json; return (p1, p2) or None if either is missing."""
    f1, f2 = profile_json_path(ds1), profile_json_path(ds2)
    if not f1.exists() or not f2.exists():
        missing = f1 if not f1.exists() else f2
        log.warn(f'profiles missing: {missing} — run 7_profile.py for both datasets first')
        return None
    with open(f1, encoding='utf-8') as f:
        p1 = json.load(f)
    with open(f2, encoding='utf-8') as f:
        p2 = json.load(f)
    log.ok(f'{len(p1)} / {len(p2)} feature profiles loaded ({ds1} / {ds2})')
    return p1, p2


def build_plans(p1: dict, p2: dict, ds1: str, ds2: str, log: Logger) -> dict:
    """Route every feature present in both years; skip identifiers and schema mismatches."""
    plans: dict = {}
    n_skip_schema = 0
    for feature in p1:
        if feature not in p2:
            n_skip_schema += 1
            continue
        if p1[feature].get('is_identifier'):
            continue
        plans[feature] = plan_comparison(feature, p1[feature], p2[feature])
    if n_skip_schema:
        log.warn(f'{n_skip_schema} feature(s) in {ds1} not in {ds2} (schema drift) — skipped')
    log.ok(f'{len(plans)} features planned')
    return plans


def save_plans(plans: dict, ds1: str, ds2: str, log: Logger) -> Path:
    """Write the machine-readable plan JSON (input contract for script 10)."""
    Config9.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Config9.OUTPUT_DIR / f'comparison_plans_{ds1}_{ds2}.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(plans, f, indent=2)
    log.ok(f'Saved {out.name} ({len(plans)} features) -> {out.parent}')
    return out


def write_report(plans: dict, ds1: str, ds2: str, json_out: Path, rdir: Path, log: Logger) -> None:
    """Brief human-readable summary: route / family / primary counts and flags."""
    lines: list[str] = []

    def h(t: str) -> None:
        lines.extend(['', '=' * 70, t, '=' * 70])

    h(f'PLAN COMPARISON REPORT — {ds1} <-> {ds2}')
    lines.append(f'Generated : {datetime.now():%Y-%m-%d %H:%M:%S}')
    lines.append(f'Features  : {len(plans)} planned')

    route_counts   = Counter(p.get('route', 'unknown') for p in plans.values())
    family_counts  = Counter(p.get('metric_family', 'unknown') for p in plans.values())
    primary_counts = Counter(p.get('corroboration_primary', 'unknown') for p in plans.values())
    mode_counts    = Counter(p.get('comparison_mode', 'unknown') for p in plans.values())

    h('ROUTE (detected_type -> corroboration path)')
    for k, n in sorted(route_counts.items()):
        lines.append(f'  {k:<24} : {n}')

    h('METRIC FAMILY')
    for k, n in sorted(family_counts.items()):
        lines.append(f'  {k:<24} : {n}')

    h('CORROBORATION PRIMARY (the verdict metric is always calibrated C2ST-AUC)')
    for k, n in sorted(primary_counts.items()):
        lines.append(f'  {k:<24} : {n}')

    h('COMPARISON MODE')
    for k, n in sorted(mode_counts.items()):
        lines.append(f'  {k:<24} : {n}')

    n_mismatch  = sum(1 for p in plans.values() if p.get('type_mismatch'))
    n_modemiss  = sum(1 for p in plans.values() if p.get('modality_mismatch'))
    n_struct    = sum(1 for p in plans.values() if p.get('route') == 'structural_change')
    n_zero      = sum(1 for p in plans.values() if p.get('zero_mass_separate'))
    h('FLAGS')
    lines.append(f'  type mismatch across years   : {n_mismatch}')
    lines.append(f'  modality mismatch across yrs : {n_modemiss}')
    lines.append(f'  routed to structural_change  : {n_struct}')
    lines.append(f'  zero-mass handled separately : {n_zero}')

    h('OUTPUT')
    lines.append(f'  plans JSON : {json_out}')
    lines.append(f'  report     : {rdir / Config9.RESULTS_FILE}')
    lines.append(f'  steps log  : {rdir / Config9.STEPS_FILE}')

    out_path = rdir / Config9.RESULTS_FILE
    out_path.write_text('\n'.join(lines), encoding='utf-8')
    log.ok(f'Saved {out_path.name}')


def main():
    ap = argparse.ArgumentParser(description='Script 9: plan cross-dataset feature comparisons.')
    ap.add_argument('--datasets', nargs='+', default=list(DATASETS))
    args = ap.parse_args()

    # Cross-dataset comparison needs exactly two datasets; fall back to the full default pair.
    dsets = args.datasets if len(args.datasets) >= 2 else list(DATASETS)
    ds1, ds2 = dsets[0], dsets[1]

    rdir = Config9.RESULTS_DIR
    rdir.mkdir(parents=True, exist_ok=True)
    log = Logger(rdir / Config9.STEPS_FILE, step_prefix=9,
                 title=f'SCRIPT 9 PLAN COMPARISON — {ds1} <-> {ds2}')

    # 9.1 — load both datasets' profiles
    log.step('Load profiles')
    loaded = load_profiles(ds1, ds2, log)
    if loaded is None:
        log.step_end()
        log.close()
        return
    p1, p2 = loaded
    log.step_end()

    # 9.2 — route every shared feature to a comparison plan
    log.step('Plan comparisons')
    plans = build_plans(p1, p2, ds1, ds2, log)
    log.step_end()

    # 9.3 — save the machine-readable plans (input for script 10)
    log.step('Save plans')
    json_out = save_plans(plans, ds1, ds2, log)
    log.step_end()

    # 9.4 — write the brief human-readable report
    log.step('Write report')
    write_report(plans, ds1, ds2, json_out, rdir, log)
    log.step_end()

    log.close()


if __name__ == '__main__':
    main()
