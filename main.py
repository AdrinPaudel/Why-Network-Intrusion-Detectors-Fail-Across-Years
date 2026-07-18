#!/usr/bin/env python3
"""
main.py — Master orchestrator for all NIDS data processing steps.

USAGE:
  python main.py --steps 0 1 2 --datasets cicids2017 cicids2018
  python main.py --steps 5 6
  python main.py --steps 7 8 9 10 11

Pipeline:
  Branch A (ML):    0 → 1 → 2 → 3 → 4 → 5 (train) → 6 (test)
  Branch B (DA):    7 (profile) → 8 (visualize) → 9 (plan) → 10 (execute)
  Joint:            11 (cross analysis)

  Engine: LightGBM RF-mode only (the sklearn path and the --algorithm flag were removed).
  Note: step 7 reads the cleaned parquet from step 1 directly (data/cc_data/<ds>_cleaned.parquet).
"""

import sys
import argparse
import subprocess
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent / 'scripts'

# (script filename, display label)
STEP_SCRIPTS = {
    0:  ('0_dataexplore.py',            'DATA_EXPLORE'),
    1:  ('1_load_clean_combine.py',     'LOAD_CLEAN_COMBINE'),
    2:  ('2_correlation_analysis.py',   'CORRELATION_ANALYSIS'),
    3:  ('3_correlation_comparison.py', 'CORRELATION_COMPARISON'),
    4:  ('4_preprocessing.py',          'PREPROCESSING'),
    5:  ('5_train.py',                  'TRAIN'),
    6:  ('6_test.py',                   'TEST'),
    7:  ('7_profile.py',                'PROFILE'),
    8:  ('8_visualize.py',              'VISUALIZE'),
    9:  ('9_plan_comparison.py',        'PLAN_COMPARISON'),
    10: ('10_execute_comparison.py',    'EXECUTE_COMPARISON'),
    11: ('11_cross_analysis.py',        'CROSS_ANALYSIS'),
}

# Steps that accept a --datasets filter. Step 3 is a cross-dataset comparison with no
# argparse (it always reads both datasets' step-2 outputs), so it is intentionally NOT
# here — passing --datasets to it would be silently dropped.
DATASET_STEPS  = {0, 1, 2, 4, 5, 6, 7, 8, 9, 10}


def run_step(step_num: int, datasets: list = None) -> bool:
    script_file, label = STEP_SCRIPTS[step_num]
    script_path = SCRIPTS_DIR / script_file

    if not script_path.exists():
        print(f"[ERROR] Script not found: {script_path}")
        return False

    print(f"\n{'='*70}")
    print(f"RUNNING STEP {step_num} — {label}")
    print(f"{'='*70}\n")

    cmd = [sys.executable, str(script_path)]
    if step_num in DATASET_STEPS and datasets:
        cmd += ['--datasets'] + datasets

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Step {step_num} exited with code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n[ERROR] Step {step_num} failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Master orchestrator for NIDS data processing pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --steps 0 1 2 3 --datasets cicids2017 cicids2018
  python main.py --steps 5 6 --datasets cicids2017
  python main.py --steps 7 8 9 10
  python main.py --steps 11
        """
    )
    parser.add_argument('--steps', type=int, nargs='+',
                        default=sorted(STEP_SCRIPTS.keys()),
                        metavar='N', help='Steps to run (0-11). Default: all steps.')
    parser.add_argument('--datasets', nargs='+', metavar='NAME',
                        help='Datasets to process (e.g., cicids2017 cicids2018)')
    args = parser.parse_args()

    valid_steps = set(STEP_SCRIPTS)
    invalid = [s for s in args.steps if s not in valid_steps]
    if invalid:
        print(f"[ERROR] Invalid step(s): {invalid}. Valid: 0-11")
        sys.exit(1)

    print(f"\n{'#'*70}")
    print(f"# NIDS PIPELINE ORCHESTRATOR")
    print(f"# Steps    : {args.steps}")
    print(f"# Datasets : {args.datasets or 'all'}")
    print(f"# Engine   : lightgbm (RF mode) — the pipeline's only engine")
    print(f"{'#'*70}\n")

    failed = []
    for step_num in args.steps:
        ok = run_step(step_num, datasets=args.datasets)
        if not ok:
            failed.append(step_num)

    print(f"\n{'='*70}")
    print("PIPELINE SUMMARY")
    print(f"{'='*70}\n")
    for s in args.steps:
        status = "[FAIL]" if s in failed else "[ OK ]"
        print(f"  {status} Step {s} — {STEP_SCRIPTS[s][1]}")

    if failed:
        print(f"\n[ERROR] {len(failed)} step(s) failed: {failed}")
        sys.exit(1)
    else:
        print("\n[ OK ] All steps completed.")
        sys.exit(0)


if __name__ == '__main__':
    main()
