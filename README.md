# NIDS Cross-Year Feature Importance & Drift Pipeline

Code for *"Why Network Intrusion Detectors Fail Across Years: Decomposing Cross-Year Failure into
Covariate Shift, Concept Change, and Prior-Probability Shift."* A tree-ensemble NIDS (LightGBM,
random-forest mode) is trained on the corrected CIC-IDS 2017 and 2018 datasets and evaluated across
years; per-feature importance is joined against two independent drift axes (covariate shift, concept
stability) plus class-prior shift, to test whether feature importance predicts instability and whether
selecting temporally stable features improves cross-year transfer.

## Repository contents

```
main.py              — orchestrator: runs any subset of steps 0-11 in order
scripts/             — one script per pipeline step, plus unified_config.py (shared config/utilities)
                        and 11_result_gen.py (writes results/11_cross_analysis/lightgbm/results.md)
requirements.txt
```

`output/` and `results/` are committed for verifiability (the JSON/CSV artifacts and generated
reports/figures every step produces, including `results.md`). Only the large, regeneratable
data files are excluded: the raw CSVs, the cleaned per-year parquet in `data/cc_data/`, and the
step-4 train/test split parquet — see **Setup** below for how to regenerate those locally.

## Requirements

Python 3.10+ (the code uses `X | None` union type hints), then:

```bash
pip install -r requirements.txt
```

Tested with: numpy 2.3, pandas 2.3, polars 1.41, pyarrow 22.0, scipy 1.17, scikit-learn 1.8,
lightgbm 4.6, matplotlib 3.10, joblib 1.5. Newer minor versions of any of these should work; nothing
here depends on a version-specific API.

## Setup: getting the dataset

This pipeline expects the **corrected 2022 re-extraction** of CIC-IDS 2017 and CSE-CIC-IDS 2018 (Liu,
Engelen, Lynar, Essam & Joosen, *"Error Prevalence in NIDS Datasets,"* IEEE CNS 2022) — **not** the
original CIC release. The corrected extraction fixes known labeling/feature bugs in the original tool
and adds an "Attempted" label category; using the original (uncorrected) CSVs will not match the column
names and label scheme the scripts expect. Download it from the paper's companion site:
https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/

Download the per-day CSV files for both years and place them here, one folder per dataset, CSVs
directly inside (any filenames — the loader picks up every `*.csv` in the folder):

```
data/raw_data/cicids2017/*.csv   (5 files: Monday .. Friday)
data/raw_data/cicids2018/*.csv   (10 files, one per capture day)
```

**Disk space:** the raw CSVs are large — roughly 1 GB for 2017 and **30+ GB for 2018**. Step 1 then
writes cleaned parquet copies to `data/cc_data/` (another ~13 GB for 2018), and the full pipeline
(steps 0-11, both years) adds further intermediate output under `output/` and `results/`. Budget at
least 60-70 GB free disk space to run everything end to end.

## Running the pipeline

```bash
# Run everything, both datasets:
python main.py

# Run specific steps:
python main.py --steps 0 1 2 3 --datasets cicids2017 cicids2018
python main.py --steps 5 6 --datasets cicids2017
python main.py --steps 7 8 9 10
python main.py --steps 11
```

Steps must run in order at least once (each step reads the previous step's output). `--datasets` only
applies to per-dataset steps (0-2, 4-10); steps 3 and 11 always operate on both datasets together and
ignore that flag.

### What each step does

| Step | Does | Depends on |
|---|---|---|
| 0 | Explores the raw per-day CSVs: column inventory, row counts, label distribution | raw CSVs |
| 1 | Cleans and merges the per-day CSVs into one parquet per year; consolidates attack labels into canonical families | raw CSVs |
| 2 | Pearson/Spearman correlation between every feature pair, per year | Step 1 |
| 3 | Flags feature pairs redundant in both years, drops one per pair | Step 2 |
| 4 | Z-score scaling, label encoding, train/test split | Step 1, 3 |
| 5 | Trains LightGBM (RF-mode) binary + multiclass models per year; records native and permutation importance | Step 4 |
| 6 | Evaluates each model same-year and cross-year (concept + covariate framings) | Step 5 |
| 7 | Per-feature distributional statistics (cardinality, MI, separation AUC) per year | Step 1 |
| 8 | Renders every feature's distribution to a PNG (visual inspection only, no downstream output) | Step 7 |
| 9 | Decides which statistical test(s) to run per feature in Step 10 | Step 7 |
| 10 | Runs the planned tests; produces the two-axis (covariate shift / concept stability) verdict per feature | Step 9 |
| 11 | Joins Step 5 importance with Step 10 drift verdicts, runs the C1-C18 hypothesis tests and the decisive ablation, writes `results.md` | Steps 5, 10 |

Engine is fixed to LightGBM RF-mode (`ALGORITHM = 'lightgbm'` in `unified_config.py`) — there is no
`--algorithm` flag. Every step's random seed is controlled by the `PIPELINE_SEED` environment variable
(default 42); set it before a run to reproduce a specific seed or to spot-check result stability across
seeds.

## Output layout

- `output/<step>/` — intermediate data artifacts (JSON, CSV, parquet, pickled caches); committed,
  except the large per-year parquet files listed in `.gitignore`
- `results/<step>/` — anything meant to be looked at directly (PNGs, text reports); committed
- `results/11_cross_analysis/lightgbm/results.md` — the full write-up step 11 generates (via
  `scripts/11_result_gen.py`), including a Pipeline Output Map at the end that lists every
  artifact every step produced and where it landed
