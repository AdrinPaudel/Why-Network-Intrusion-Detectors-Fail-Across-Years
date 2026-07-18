#!/usr/bin/env python3
"""
Unified configuration AND shared infrastructure for all pipeline scripts (the single
source of truth — there are no other common modules).

STRUCTURE:
  - COMMON: paths, dataset list, shared label/identifier constants, batch size, engine constant
  - Config0 .. Config11 (script-specific): workers, chunk sizes, thresholds, hyperparameters
  - SHARED UTILITIES: label mapping, ensure_results_dir, Logger
  - BRANCH-B DATA ACCESS (steps 7-10): cleaned-parquet readers, feature columns, safe_filename
  - ML PIPELINE HELPERS (steps 5/6/11): algorithm-versioned dirs, step-4 artifact loaders,
    memory-bounded Parquet readers, the per-class stratified training loader

Every script imports what it needs from here — no per-script worker counts, no hardcoded
tuning, no duplicate loggers. Per-step worker counts may differ (RAM-bound) but ALL live in
the ConfigN classes below.
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Force UTF-8 console output for every script that imports this module.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ============================================================================
# COMMON (shared by all scripts)
# ============================================================================

# Shared pipeline seed: every step's Config<N>.SEED reads this
# same value, so a single env var reruns the WHOLE pipeline at one alternate seed for a
# multi-seed robustness check, instead of every step being separately and silently fixed at 42.
# Usage: set PIPELINE_SEED=123 in the environment before running steps 4,5,7,9,10,11 in sequence,
# then compare the headline numbers in the regenerated reports/results.md against the seed-42 run.
PIPELINE_SEED = int(os.environ.get('PIPELINE_SEED', '42'))

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_ROOT = PROJECT_ROOT / "results"
OUTPUT_ROOT = PROJECT_ROOT / "output"
RAW_DATA_ROOT = DATA_ROOT / "raw_data"
CLEANED_DATA_ROOT = DATA_ROOT / "cc_data"          # <ds>_cleaned.parquet (raw units, from step 1)
PREPROCESS_ROOT = OUTPUT_ROOT / "4_preprocessing"  # step-4 train/test parquet + artifacts

# Datasets to process
DATASETS = {
    "cicids2017": RAW_DATA_ROOT / "cicids2017",
    "cicids2018": RAW_DATA_ROOT / "cicids2018",
}

# Dataset names as an ordered tuple (the cross-year comparison order for steps 9/10/11).
DATASET_NAMES = tuple(DATASETS.keys())

# ── Shared label / identifier constants ─────────────────────────────────────────
# Raw string label column in the cleaned parquet (steps 0-3, 7-10 read this).
LABEL_COL    = 'Label'
BENIGN_LABEL = 'Benign'

# Encoded target columns written by step 4 (steps 5/6/11 read these).
LABEL_BINARY     = 'label_binary'
LABEL_MULTICLASS = 'label_multiclass'
LABEL_COLS       = (LABEL_BINARY, LABEL_MULTICLASS)

# Identifier / non-feature columns dropped before correlation/preprocessing (steps 2, 4).
# Lower-cased; compared case-insensitively. Single source of truth for Config2/Config4.
ML_IDENTIFIER_COLS = frozenset({
    'id', 'flow id', 'src ip', 'src port', 'dst ip', 'dst port',
    'protocol', 'timestamp', 'label',
})

# Identifier / quasi-identifier columns excluded from the Branch-B per-feature analysis
# (steps 2-3, 7-10). Kept only as sanity controls.
# Deliberately narrower than ML_IDENTIFIER_COLS above — Protocol and Dst Port are
# NOT excluded here, unlike Config4.IDENTIFIER_COLS (= ML_IDENTIFIER_COLS), which strips them from
# the actual ML feature set in step 4. This is intentional, not a scope-alignment bug: Branch-B's
# correlation/profile/drift analysis (steps 2-3, 7-10) treats Protocol/Dst Port as two EXTRA
# sanity-control columns worth watching for drift in their own right, even though they were never
# going to become ML features — harmless (step 4 strips them regardless of what steps 2-3 decide),
# but if either ever shows up inside a reported high-correlation redundancy group, remember it was
# never eligible for the ML feature set anyway.
BRANCH_B_IDENTIFIERS = (
    'id', 'Flow ID', 'Src IP', 'Dst IP', 'Src Port', 'Timestamp',
)

# Streaming batch size shared by every Parquet row-batch reader. 500k rows x 71 float32
# cols ~ 140 MB/batch — the memory-bounded unit for steps 5/6 and the Branch-B readers.
DEFAULT_BATCH_ROWS = 500_000

# ML engine (steps 5/6/11). LightGBM RF-mode is the pipeline's ONLY engine — the sklearn
# RandomForest path was removed (2026-07-02): it could not train the full ~50M-row 2018 split
# without capping Benign (which mis-calibrated class_weight), and its outputs were never part of
# the reported results. The constant keeps the algorithm-suffixed folder layout
# (<ds>_lightgbm) so existing outputs stay addressable.
ALGORITHM = 'lightgbm'


def plan_workers(n_items: int, max_workers: int) -> int:
    """Single, shared worker-count rule for every parallel step.

    Fixed cap, scaled down by how much work there actually is — no RAM probing.
    A step with only a few files/features spins up only that many workers; the
    per-step ``max_workers`` keeps RAM-heavy steps (e.g. step 10) lower than the
    cheap ones. Always at least 1.
    """
    cpu = os.cpu_count() or 4
    return max(1, min(n_items, cpu, max_workers))

# ============================================================================
# SCRIPT-SPECIFIC CONFIGURATIONS
# ============================================================================

class Config0:
    """Configuration for script 0_dataexplore.py"""
    CHUNK_ROWS   = 500_000
    MAX_WORKERS  = 10
    RESULTS_FILE = '0_dataexplore_report.txt'
    STEPS_FILE   = '0_dataexplore_steps.log'
    RESULTS_DIR  = RESULTS_ROOT / '0_dataexplore'

    # Per-dataset chart configuration for write_visuals().
    # Keyed by dataset name; falls back to *_DEFAULT when the name is not present.

    # Chart 1 (labels per file): lower Y floor, tick step, axis number format.
    CHART1_YMIN               = {'cicids2017': 250_000,   'cicids2018': 5_000_000}
    CHART1_YTICK_STEP         = {'cicids2017': 250_000,   'cicids2018': 250_000}
    CHART1_YSCALE             = {'cicids2017': 'K',        'cicids2018': 'M'}
    CHART1_YMIN_DEFAULT       = 0
    CHART1_YTICK_STEP_DEFAULT = 100_000
    CHART1_YSCALE_DEFAULT     = 'comma'

    # Chart 3 (data quality per file): tick step and axis number format. Y starts at 0.
    # Omit a dataset key to let matplotlib auto-select tick positions.
    CHART3_YTICK_STEP         = {'cicids2017': 100_000}
    CHART3_YSCALE             = {'cicids2017': 'K',  'cicids2018': 'comma'}
    CHART3_YTICK_STEP_DEFAULT = None   # None = matplotlib auto
    CHART3_YSCALE_DEFAULT     = 'comma'

class Config1:
    """Configuration for script 1_load_clean_combine.py"""
    MAX_WORKERS        = 10            # cap for sub-step 1.1 (cleaning); actual = plan_workers(n_files, MAX_WORKERS)
    MAX_SHARD_WORKERS  = 5             # cap for sub-step 1.3 (shard read); lower because temp CSVs are huge
    CHUNK_ROWS         = 500_000       # fixed rows per cleaning chunk
    MIN_CLASS_ROWS     = 100
    DROP_COL_SET       = frozenset({'attempted category'})
    HASH_EXCLUDE       = frozenset({'id'})
    RESULTS_FILE       = '1_load_clean_combine_report.txt'
    STEPS_FILE         = '1_load_clean_combine_steps.log'
    RESULTS_DIR        = RESULTS_ROOT / '1_load_clean_combine'

class Config2:
    """Configuration for script 2_correlation_analysis.py

    NOTE: step 2's actual feature scope comes from feature_columns() / BRANCH_B_IDENTIFIERS
    (Protocol and Dst Port stay in the correlation analysis as sanity controls — see the
    note above); there is no separate identifier list here.
    """
    # Correlation threshold for flagging high-correlation pairs
    CORR_THRESHOLD = 0.90

    # Spearman subsampling (adaptive based on dataset size).
    # If actual rows < THRESHOLD: cap at SMALL (effectively 100% of a small dataset).
    # If actual rows >= THRESHOLD: cap at LARGE (~10% subsample for large datasets).
    # Actual row counts are reported in results/1_load_clean_combine/<ds>/report.
    SPEARMAN_SAMPLE_SMALL = 6_000_000
    SPEARMAN_SAMPLE_LARGE = 6_500_000
    SPEARMAN_SAMPLE_THRESHOLD = 6_000_000
    # Intentionally NOT wired to PIPELINE_SEED, unlike every other Config<N>.SEED.
    # Step 2's correlation-drop decision is not part of the multi-seed H1/H2 robustness claim (see
    # the "steps 4,5,7,9,10,11" list in the PIPELINE_SEED comment above, which deliberately
    # excludes step 2) — it only determines WHICH features are eligible to survive into the
    # feature set, a one-time, threshold-based (|r| >= 0.90) decision, not a stochastic result
    # that a multi-seed check needs to reproduce at an alternate seed.
    SPEARMAN_SEED = 42

    # Heatmap rendering parameters
    HEATMAP_CELL_INCHES = 0.42
    HEATMAP_MIN_INCHES = 14.0
    HEATMAP_DPI = 200
    HEATMAP_ANNOT_FONTSZ = 5.0
    HEATMAP_TICK_FONTSZ = 6.0

    OUTPUT_BASE = PROJECT_ROOT / 'output' / '2_correlation_analysis'

    # Output file names
    RESULTS_FILE = '2_correlation_analysis_report.txt'
    STEPS_FILE = '2_correlation_analysis_steps.log'

    @staticmethod
    def get_spearman_sample_rows(estimated_total_rows: int) -> int:
        """Adaptive Spearman sample size: scales with dataset size."""
        if estimated_total_rows < Config2.SPEARMAN_SAMPLE_THRESHOLD:
            return Config2.SPEARMAN_SAMPLE_SMALL
        else:
            return Config2.SPEARMAN_SAMPLE_LARGE

class Config3:
    """Configuration for script 3_correlation_comparison.py"""
    # Consensus threshold: a pair is marked for drop ONLY if |r| >= this value
    # in BOTH years for BOTH metrics (Pearson AND Spearman).
    # Intersection rule (MIN of both years) must clear this bar.
    # 0.95 is the standard NIDS feature-selection cutoff.
    CONSENSUS_THRESHOLD = 0.95

    # Per-year threshold for "shifted" classification.
    # A pair counts as "correlated in year Y" if Pearson |r| in year Y >= this value.
    # Used to detect correlation drift (features correlated only in one year).
    PER_YEAR_THRESHOLD = 0.95

    # Sensitivity sweep: CONSENSUS_THRESHOLD was never swept even though it
    # determines the final feature set every downstream step trains/tests on. These are run
    # alongside the official 0.95 cutoff (which they do NOT replace) purely to report how much the
    # dropped-feature set would change at a stricter or looser bar.
    SENSITIVITY_THRESHOLDS = (0.90, 0.925, 0.95, 0.975, 0.99)

    # Output file names
    RESULTS_FILE = '3_correlation_comparison_report.txt'
    STEPS_FILE = '3_correlation_comparison_steps.log'

    # Output directories (in output/ not results/)
    OUTPUT_DIR = PROJECT_ROOT / 'output' / '3_correlation_comparison'
    RESULTS_DIR = RESULTS_ROOT / '3_correlation_comparison'
    INPUT_BASE = PROJECT_ROOT / 'output' / '2_correlation_analysis'

class Config4:
    """Configuration for script 4_preprocessing.py"""
    # Streaming configuration
    CHUNK_ROWS = 500_000

    # Stratified train/test split
    TEST_SIZE = 0.20
    SEED = PIPELINE_SEED
    SPLIT_MODULUS = 1000

    # Parquet compression (for storage efficiency)
    COMPRESSION = 'snappy'

    # Data type for scaled features (float32 saves 50% disk vs float64)
    OUTPUT_FLOAT_DTYPE = 'float32'

    # Columns to exclude from features (identifiers + target) — shared constant
    IDENTIFIER_COLS = ML_IDENTIFIER_COLS

    # Output file names
    RESULTS_FILE = '4_preprocessing_report.txt'
    STEPS_FILE = '4_preprocessing_steps.log'

    PHASE_HASH_MULTIPLIER = 131   # StratifiedSplitter per-class phase hash multiplier
    LOG_INTERVAL          = 50    # log progress every N batches

    # Output directories
    RESULTS_DIR = RESULTS_ROOT / '4_preprocessing'
    OUTPUT_DIR = PROJECT_ROOT / 'output' / '4_preprocessing'


class Config5:
    """Configuration for script 5_train.py (LightGBM RF-mode training — the only engine).

    Outputs land under results/5_training/<ds>_lightgbm/ and output/5_training/<ds>_lightgbm/
    (see training_output_dir / training_results_dir). All hyperparameters are fixed,
    individually justified constants — there is NO hyperparameter search and no tuning flags.
    Native importance is recorded BOTH ways: gain (primary; total split gain, less
    cardinality-biased) and split-count (secondary diagnostic; Strobl 2007 bias caveat).
    """
    N_CLASSES_MULTI   = 8        # canonical multiclass label space (union of both years)
    N_ESTIMATORS      = 200      # number of trees
    MAX_DEPTH         = None     # None = grow fully (bagging + feature subsampling de-correlate)
    SEED              = PIPELINE_SEED       # random state
    N_JOBS            = -1       # parallel jobs (-1 = all cores)

    # LightGBM RF-mode params (boosting_type='rf').
    LGBM_NUM_LEAVES       = 255    # allow deep-ish trees (RF de-correlates anyway)
    LGBM_MIN_CHILD        = 20     # min_child_samples (leaf regularisation)
    LGBM_BAGGING_FRACTION = 0.632  # bootstrap-like row subsample (RF mode requires <1)
    LGBM_BAGGING_FREQ     = 1      # bag every iteration (required for rf mode)
    LGBM_FEATURE_FRACTION = 0.5    # per-split column subsample (~sqrt-style de-correlation)

    TOP_K_IMPORTANCES = 25       # features shown in the importance plot

    # Permutation importance (computed on a HELD-OUT sample; unbiased counterpart to Gini/MDI).
    # Sample size and repeats raised from the original 10_000/3 (too small/noisy a
    # sample for an estimate this load-bearing — H1's native-vs-permutation contrast hinges on it).
    # 10 repeats left ~40/71 features with importance
    # indistinguishable from zero (|mean|/SE < 2) and 33 features at EXACTLY zero, so the H1
    # permutation-importance Spearman cells were estimated over a majority-noise tail ranking.
    # Raised 10 -> 30 to reach the low end of the conventional 30-100 range for an estimate feeding
    # a hypothesis test; the 25_000 held-out sample keeps this tractable (perm cost ~ linear in
    # repeats). To apply this WITHOUT retraining the models (and thus without drifting Table 2 /
    # H1-native / H2), set REUSE_EXISTING_MODEL=True and re-run step 5: it reloads the saved
    # rf_*.joblib and only recomputes importance, leaving every trained model bit-identical.
    PERM_IMPORTANCE   = True
    PERM_SAMPLE       = 25_000   # per-class-capped held-out rows
    PERM_REPEATS      = 30
    PERM_SCORING      = 'balanced_accuracy'   # honest under heavy class imbalance

    # When True, train_one() reuses an existing rf_<task>.joblib (skips fitting) and only recomputes
    # native + permutation importance. Lets a permutation-repeat change (above) update the H1 perm
    # cells in isolation, with zero model drift. Default False = normal fresh-train behavior.
    # Set True to reuse existing rf_<task>.joblib and recompute importance only (used once for the
    # audit 30-repeat permutation re-run so the H1 perm cells updated with zero model drift; reset
    # to False afterward — leaving it True makes step 5 never retrain, so a future genuine retrain
    # with new data / changed hyperparameters would silently be skipped).
    REUSE_EXISTING_MODEL = False

    # Output file names (per dataset/algorithm folder).
    RESULTS_FILE = '5_training_report.txt'
    STEPS_FILE   = '5_training_steps.log'


class Config6:
    """Configuration for script 6_test.py (evaluate trained LightGBM RF-mode models on the held-out split).

    Outputs are algorithm-versioned under results/6_testing/<ds>_<algorithm>/ and
    output/6_testing/<ds>_<algorithm>/ (see testing_output_dir / testing_results_dir).
    """
    TASKS             = ('binary', 'multiclass')   # both tasks evaluated when a model exists
    PLOT_CURVE_POINTS = 2000     # cap drawn points on ROC/PR curves (AUC stays exact)

    # Output file names (per dataset/algorithm folder).
    RESULTS_FILE = '6_testing_report.txt'
    STEPS_FILE   = '6_testing_steps.log'


class Config7:
    """Configuration for script 7_profile.py (per-feature distribution profiler)."""
    # Parallelism: one process per feature, capped.
    MAX_WORKERS = 8

    # Sampling caps for the expensive estimators. Exact stats + view range use the FULL
    # column; only AUC/MI/modality run on a sample so a 63M-row column costs seconds.
    SEP_SAMPLE_CAP = 60_000      # per-class cap for the separation/MI stratified sample
    MI_SAMPLE_MAX  = 80_000      # hard cap on rows fed to the KSG MI estimator (super-linear)
    GMM_SAMPLE_MAX = 50_000      # rows fed to GMM/dip for modality (marginal shape)

    # Modality / separation tuning.
    GMM_MAX_K          = 4       # max GMM components tried; best chosen by BIC
    DIRECTION_AUC_BAND = 0.05    # AUC deadband before a separation direction is trusted

    # New distribution metrics.
    ENTROPY_BINS     = 50        # histogram bins for Shannon entropy (bits)
    OUTLIER_IQR_MULT = 1.5       # Tukey fence multiplier (Q1-k*IQR, Q3+k*IQR)
    PERCENTILES      = (1, 5, 10, 25, 50, 75, 90, 95, 99)

    # Reproducibility + feature scope.
    SEED = PIPELINE_SEED
    INCLUDE_IDENTIFIERS = False  # exclude id/IP/port/timestamp from analysis

    # Type detection thresholds (detect_type)
    SENTINEL_MASS_FRAC    = 0.01   # min-value count must exceed n_total * this to be a sentinel
    ZERO_INFLATION_THRESH = 0.10   # zero fraction >= this → zero_inflated
    TYPE_N_UNIQUE_NOMINAL = 10     # n_unique <= this → low-cardinality-discrete
    TYPE_N_UNIQUE_DISCRETE = 100   # n_unique <= this (int-like) → discrete-count
    LOG_RATIO_THRESH      = 100    # pos.max()/pos.min() > this → log scale

    # Robust view range quantiles (robust_view_range)
    VIEW_LO_Q = 0.005
    VIEW_HI_Q = 0.995

    # GMM / dip modality detection
    MIN_SAMPLES_GMM   = 200   # fewer rows → skip GMM, return single mode
    MIN_SAMPLES_CONT  = 100   # fewer rows → skip modality analysis
    MIN_GMM_MODE_MASS = 0.02  # GMM modes below this mass are pruned
    DIP_TEST_PVALUE   = 0.05  # pval > this → unimodal (skip GMM)

    # Progress logging interval (log every N features)
    LOG_INTERVAL = 10

    # Column name hints for non-negative semantics and sentinel detection
    NONNEG_HINTS = ('length', 'count', 'duration', 'bytes', 'packet', 'iat', 'size',
                    'flags', 'total', 'mean', 'max', 'min', 'std', 'rate', 'win',
                    'active', 'idle', 'subflow', 'segment', 'bulk')
    SENTINEL_COLS = ('icmp code', 'icmp type')

    # Output file names + directories.
    RESULTS_FILE = '7_profile_report.txt'
    STEPS_FILE   = '7_profile_steps.log'
    RESULTS_DIR  = RESULTS_ROOT / '7_profile'
    OUTPUT_DIR   = PROJECT_ROOT / 'output' / '7_profile'


class Config8:
    """Configuration for script 8_visualize.py (per-feature raincloud visualizer)."""
    # Parallelism: one process per feature, capped.
    MAX_WORKERS = 8

    # Class-split modes. Each is rendered twice: a standard view (outliers clipped to the
    # profile's robust view range) and, when INCLUDE_EXTENDED, an *_extended view that spans
    # the full min..max so the outliers ARE shown. -> 4 output folders per dataset:
    #   <ds>/binary  <ds>/binary_extended  <ds>/multiclass  <ds>/multiclass_extended
    CLASS_MODES      = ('multiclass', 'binary')
    INCLUDE_EXTENDED = True

    # Rain (raw scatter) points drawn per class lane. The old fixed 4k was too sparse.
    # 2018 has ~30x the rows of 2017, so it gets a larger cap; both stay sane so the PNGs
    # render fast and stay light. Tune per dataset here.
    RAIN_POINTS_PER_CLASS = {
        'cicids2017': 20_000,
        'cicids2018': 50_000,
    }
    RAIN_POINTS_DEFAULT = 20_000

    # KDE density cloud is smooth well before this; the cap keeps the density fit cheap.
    KDE_SAMPLE_CAP = 30_000

    # Figure sizing. The extended view is wider + higher-dpi so the long tails resolve.
    FIG_WIDTH          = 12
    FIG_WIDTH_EXTENDED = 16
    FIG_DPI            = 100
    FIG_DPI_EXTENDED   = 130

    # Discrete features: max distinct integer levels before falling back to jittered scatter.
    DISCRETE_MAX_LEVELS = 40

    # Reproducibility (subsampling for clouds/rain is deterministic).
    SEED = PIPELINE_SEED

    # Output file names + directory.
    RESULTS_FILE = '8_visualize_report.txt'
    STEPS_FILE   = '8_visualize_steps.log'
    RESULTS_DIR  = RESULTS_ROOT / '8_visualize'


class Config9:
    """Configuration for script 9_plan_comparison.py (cross-dataset comparison planner)."""
    # ── Metric ROUTING (detected_type -> which metric family) ──────────────────────
    # PMF (probability-mass) types: compared with distribution-distance metrics on the
    # probability mass, NOT Wasserstein. 'low-cardinality-discrete' was already here.
    NOMINAL_TYPES = ('identifier', 'nominal', 'binary', 'low-cardinality-discrete')
    # 'discrete-count' (integer counts: flags, packets, windows) is now routed to PMF too,
    # so its integer-count structure is respected instead of being treated as continuous.
    DISCRETE_TYPES = ('discrete-count',)

    # Cross-year type resolution: if the two years disagree on a feature's type (cardinality
    # drift), route on the MORE GENERAL type (higher rank) so the metric is never under-specified.
    TYPE_ORDER = {
        'identifier': 0, 'nominal': 1, 'binary': 2,
        'low-cardinality-discrete': 3, 'discrete-count': 4, 'continuous': 5,
    }

    DEFAULT_DETECTED_TYPE  = 'continuous'
    DEFAULT_SPLIT_STRATEGY = 'benign_and_each_attack'

    # Cross-year SCALE resolution: the more general scale (log > symlog > linear) wins. Step 10
    # uses it to compress MMD's inputs (its RBF median-heuristic bandwidth is the one Axis-1 metric
    # that degrades on raw heavy-tailed data). Wasserstein-qn, energy, KS and Anderson-Darling are
    # rank/CDF-based and monotone-invariant, so this never touches the headline shift scalar.
    SCALE_ORDER   = {'categorical': 0, 'linear': 1, 'symlog': 2, 'log': 3}
    DEFAULT_SCALE = 'linear'

    # CORROBORATION-metric assignment per route. The Axis-1 VERDICT is always decided by
    # calibrated C2ST-AUC (Config10.C2ST_SHIFT_THRESHOLD) — the entries here
    # only say which distance FAMILY step 10 computes as pooled corroboration/E1 evidence for
    # each route ('corroboration_primary' = the route's most shape-appropriate distance,
    # 'corroboration_secondary' = the rest of the battery). Only metrics step 10 actually
    # computes are listed. discrete_count and nominal both use metric_family='nominal' (the PMF
    # path -> Jensen-Shannon). Each entry also carries the routing strings (metric_family,
    # alignment, normalization, comparison_mode) so plan_comparison() reads them from here.
    METRICS = {
        'nominal': {
            'corroboration_primary': 'jensen_shannon', 'corroboration_secondary': [],
            'metric_family': 'nominal', 'alignment': 'none',
            'normalization': 'pmf', 'comparison_mode': 'whole_distribution',
        },
        'discrete_count': {
            'corroboration_primary': 'jensen_shannon', 'corroboration_secondary': [],
            'metric_family': 'nominal', 'alignment': 'none',
            'normalization': 'pmf', 'comparison_mode': 'whole_distribution',
        },
        'continuous_unimodal': {
            'corroboration_primary': 'wasserstein_qn',
            'corroboration_secondary': ['energy_distance', 'anderson_darling', 'ks_statistic', 'mmd'],
            'metric_family': 'continuous', 'alignment': 'none',
            'normalization': 'quantile_normalization', 'comparison_mode': 'whole_distribution',
        },
        'continuous_multimodal': {
            'corroboration_primary': 'mmd',
            'corroboration_secondary': ['wasserstein_qn', 'energy_distance', 'anderson_darling',
                                        'ks_statistic'],
            'metric_family': 'continuous', 'alignment': 'semantic_or_nearest_center',
            'normalization': 'quantile_normalization', 'comparison_mode': 'per_mode',
        },
        # STRUCTURAL CHANGE: the two years have a DIFFERENT distributional structure — a modality
        # mismatch (e.g. unimodal in one year, bimodal in the other) or a cross-family type change.
        # There is no shared template to align mode-for-mode. C2ST (shape-agnostic) decides, as it
        # does for every route; step 10 stamps the distinct 'restructured' verdict when the C2ST
        # evidence confirms the mismatch is a real distributional change (not GMM/type noise).
        'structural_change': {
            'corroboration_primary': 'wasserstein_qn',
            'corroboration_secondary': ['mmd', 'energy_distance', 'anderson_darling',
                                        'ks_statistic'],
            'metric_family': 'continuous', 'alignment': 'none',
            'normalization': 'quantile_normalization', 'comparison_mode': 'whole_distribution',
        },
    }

    # Output file names + directories. Step 9 is CROSS-DATASET: a SINGLE combined folder, no
    # per-dataset subfolders. OUTPUT_DIR holds the plan JSON, which is
    # the input contract for script 10 — kept in sync with the module-level COMPARE_OUTPUT_DIR.
    RESULTS_FILE = '9_plan_comparison_report.txt'
    STEPS_FILE   = '9_plan_comparison_steps.log'
    RESULTS_DIR  = RESULTS_ROOT / '9_plan_comparison'
    OUTPUT_DIR   = PROJECT_ROOT / 'output' / '9_plan_comparison'


class Config10:
    """Configuration for script 10_execute_comparison.py (cross-dataset comparison executor)."""
    # Parallelism: one process per feature. 4 workers × ~480 MB per 2018 column ≈ 1.9 GB peak;
    # 8 workers OOMed on the test machine (too much concurrent RAM).
    MAX_WORKERS = 4

    # ── Sampling caps for the distance / divergence estimators ──────────────────────
    # MATCHED-N RULE: every metric's null floor MUST be computed at the SAME
    # per-side sample size as its actual statistic. Finite-sample two-sample distances are biased
    # upward at small n, so a null computed at a smaller n than the actual reads systematically
    # HIGH and under-detects real shift. Each *_SAMPLE below is therefore shared verbatim by the
    # actual computation AND its null re-splits (the null caps alias them; never set independently).
    # Wasserstein's actual cap was reduced 1M -> 100k to make the match affordable: for a 1-D
    # rank-normalized distance, 100k/side estimates are far more precise than its E1 corroboration
    # role needs (Wasserstein never decides the verdict — that is calibrated C2ST, FIX-1/FIX-3).
    MAX_DIST_SAMPLE   = 100_000    # per side, 1-D Wasserstein-qn (actual AND null re-splits)
    KS_SAMPLE         = 10_000     # per side, KS + Anderson-Darling (actual AND null)
    MMD_SAMPLE        = 2_000      # per side, O(n^2) MMD (actual AND null)
    MMD_MEDIAN_SAMPLE = 2_000      # pooled cap for the RBF median-heuristic bandwidth
    ENERGY_SAMPLE     = 2_000      # per side, O(n^2) energy distance (actual AND null)
    C2ST_SAMPLE       = 2_000      # per side, C2ST (actual AND null); CV on ~4k points is fast
    OVERLAP_SAMPLE    = 8_000      # points per class lane in the overlap scatter
    QQ_SAMPLE         = 500_000    # per side cap for the Q-Q quantile estimate
    NULL_SAMPLE       = MAX_DIST_SAMPLE   # Wasserstein null re-splits — matched by definition
    NULL_RESPLITS     = 50         # per-feature null re-splits to calibrate the shift threshold
    NULL_PERCENTILE   = 95         # percentile of the null shift distribution used as the floor

    # ── Per-metric null calibration. EACH metric gets its own resplit budget, sized to its own
    # cost; the null SAMPLE caps alias the actual caps above (matched-n rule).
    C2ST_NULL_RESPLITS  = 20        # null re-splits for C2ST-AUC (pooled + per-slice)
    C2ST_NULL_SAMPLE    = C2ST_SAMPLE
    HEAVY_NULL_RESPLITS = 15        # null re-splits for O(n^2) metrics (MMD, energy distance)
    HEAVY_NULL_SAMPLE   = MMD_SAMPLE      # matched-n (= ENERGY_SAMPLE too)
    LIGHT_NULL_RESPLITS = 30        # null re-splits for O(n log n) metrics (KS, Anderson-Darling)
    LIGHT_NULL_SAMPLE   = KS_SAMPLE       # matched-n
    MI_NULL_RESPLITS    = 30        # null re-splits for the MI_STRONG floor (k-NN MI estimator)
    MI_NULL_SAMPLE      = 5_000     # per-side cap for MI null re-splits (k-NN cost grows with n)

    # calibrate_excess() (MMD/energy/KS/AD) is a ratio-to-null-floor, unlike
    # calibrate_c2st()'s bounded [0,1] transform — near-zero null floors blow the ratio up
    # unboundedly (e.g. raw=0.001, floor=1e-6 -> ~999). These 4 metrics are diagnostic-only
    # (E1), but capping keeps them commensurable/rankable across features instead of a few
    # noisy-floor features dominating the scale.
    EXCESS_RATIO_CAP = 10.0         # calibrated excess ratio clip for MMD/energy/KS/AD

    # ── C2ST (classifier two-sample test) ──────────────────────────────────────────
    C2ST_CV_FOLDS      = 5         # stratified CV folds (the per-fold AUCs feed the CI)
    C2ST_TREE_DEPTH    = 6         # shallow decision-tree depth
    C2ST_CI_CONFIDENCE = 0.95      # confidence level for the AUC CI from the fold scores

    # ── Q-Q probability grid (clip 1% / 99% tails) + shape-classification tuning ─────
    QQ_Q_LOW   = 0.01
    QQ_Q_HIGH  = 0.99
    QQ_Q_COUNT = 99
    QQ_R2_SHAPE      = 0.97        # R^2 below this -> 'shape_change'
    QQ_SLOPE_TOL     = 0.15        # |slope-1| above this -> 'scale_change'
    QQ_INTERCEPT_TOL = 0.05        # |intercept| > tol*span -> 'location_shift'

    # Quantile-shift breakdown: where in the distribution the shift concentrates.
    QUANTILE_BREAKPOINTS = (0.25, 0.50, 0.75)

    # ── Verdict thresholds ──────────────────────────────────────────────────────────
    # The Axis-1 stable/shifted verdict is decided by CALIBRATED C2ST-AUC:
    # calibrated = (raw_cv_auc - null_95th_pct) / (1 - null_95th_pct); > C2ST_SHIFT_THRESHOLD
    # means the two years are more classifier-distinguishable than chance -> 'shifted'.
    # C2ST is the one metric computed identically for EVERY feature type (nominal, count,
    # continuous, multimodal), which is exactly why it — and not Wasserstein/MMD — decides.
    C2ST_SHIFT_THRESHOLD   = 0.0   # calibrated C2ST-AUC excess above this -> 'shifted'
    SEPARATION_STRONG      = 0.55  # folded-AUC "strong separation" bar
    MI_STRONG              = 0.05  # normalized-MI "strong separation" bar (non-monotonic blobs)
    SENSITIVITY_THRESHOLDS = (0.02, 0.05, 0.10)   # sweep for the C2ST verdict threshold

    # Reproducibility.
    SEED = PIPELINE_SEED

    # Output file names + directories. Step 10 is CROSS-DATASET: a SINGLE combined folder, no
    # per-dataset subfolders. OUTPUT_DIR holds the verdict JSONs (Layer A/B) read by step 11;
    # the E1 cross-metric agreement lives INSIDE those as per-feature 'e1_agreement' keys
    # (there is no separate agreement JSON file). Dir renamed from 10_verdict.
    RESULTS_FILE = '10_execute_comparison_report.txt'
    STEPS_FILE   = '10_execute_comparison_steps.log'
    RESULTS_DIR  = RESULTS_ROOT / '10_execute_comparison'
    OUTPUT_DIR   = PROJECT_ROOT / 'output' / '10_execute_comparison'


class Config11:
    """Configuration for script 11_cross_analysis.py (importance × distribution-shift cross analysis).

    Step 11 is CROSS-DATASET (both years, no --datasets) AND algorithm-versioned: importance is
    algorithm-specific, so outputs land under results/11_cross_analysis/<algorithm>/ and
    output/11_cross_analysis/<algorithm>/ (the <algorithm> dimension is required, not a per-dataset
    split). All numeric tuning below — no hardcoded values in the script body.
    """
    # Reproducibility: base seed for every bootstrap / permutation / model fit
    # (the H2 ablation derives its replicate seeds as SEED+0..SEED+ABLATION_SEEDS-1).
    SEED = PIPELINE_SEED

    # Parallel jobs for the ablation LightGBM fits / permutation (-1 = all cores).
    N_JOBS = -1

    # ── Sub-step 11.1 join ──────────────────────────────────────────────────────────
    JOIN_MIN_FEATURES = 60         # warn if fewer than this many features join (expected ~71)

    # ── Sub-step 11.2 headline rank statistics ──────────────────────────────────────
    BOOTSTRAP_N            = 2000   # bootstrap resamples for the Spearman 95% CI
    CLUSTER_CORR_THRESHOLD = 0.80   # |avg Pearson| >= this groups two kept features into one
                                    # collinearity cluster for the cluster bootstrap

    # ── Sub-step 11.4 bidirectional K rank test ─────────────────────────────────────
    BIDIR_K_VALUES = (5, 10, 15)

    # ── Sub-step 11.3 drift exposure ────────────────────────────────────────────────
    DRIFT_N_PERM = 5000            # permutation-null draws for the drift-exposure percentile

    # ── Sub-step 11.7 cross-domain ablation (the decisive experiment) ───────────────
    # The ablation retrain engine is LightGBM RF-mode (boosting_type='rf', mirroring
    # 5_train.build_lgbm_rf) — the pipeline's only engine.
    RUN_ABLATION       = True      # master toggle (replaces the old --skip-ablation flag)
    ABLATION_ONLY      = False     # only the ablation, skip stats/visuals (old --ablation-only)
    # 6 K values (5,10,20,30,50 here + len(canonical)=71 appended by run_ablation): trimmed from the
    # previous 8-value sweep (Option B) to keep the sweep tractable while still covering
    # small/medium/large feature-subset sizes. Run find_peak_k after to identify a fine-search
    # window if a finer sweep is ever needed.
    ABLATION_K_VALUES  = (5, 10, 20, 30, 50)
    TRAIN_CAP          = 500_000   # per-class row cap for ablation training
    TEST_CAP_DIVISOR   = 2         # in-domain test cap = max(TRAIN_CAP // DIVISOR, TEST_CAP_FLOOR)
    TEST_CAP_FLOOR     = 100_000   # min in-domain test rows (bumped)
    IN_DOMAIN_TEST_SIZE = 0.2      # split fraction when a dataset has no test.parquet
    ABLATION_TREES     = 100       # ablation is a comparison not production — 100 trees gives same ranking at half cost
    ABLATION_DEPTH     = 30        # max_depth (LightGBM uses -1 when 0/None)
    # SEED REPLICATES: EVERY policy (top_importance / axis1_stable /
    # axis2_stable / random / all_features) is retrained ABLATION_SEEDS times per
    # (K, direction) cell, and the H2 verdict is read from the per-seed means + a paired
    # significance test — a single-seed point comparison is retraining noise, not evidence.
    ABLATION_SEEDS     = 5         # seeds per (K, policy, direction); seed_i = SEED + i
    # LightGBM RF-mode ablation params; mirror step 5.
    ABLATION_LGBM_NUM_LEAVES       = 255    # allow deep-ish trees (RF de-correlates anyway)
    ABLATION_LGBM_MIN_CHILD        = 20     # min_child_samples (leaf regularisation)
    ABLATION_LGBM_BAGGING_FRACTION = 0.632  # bootstrap-like row subsample (RF mode requires <1)
    ABLATION_LGBM_FEATURE_FRACTION = 0.5    # per-split column subsample (~sqrt-style de-correlation)

    # ── Inputs / attack handling ────────────────────────────────────────────────────
    ATTACK_2017_ONLY  = ('PortScan',)   # 0 rows in 2018 -> never a shared cross comparison
    COMPARISON_MATRIX = PROJECT_ROOT / 'output' / '3_correlation_comparison' / 'comparison_matrix.json'

    # ── Sub-step 11.9 rank stability (Test C) ───────────────────────────────────
    RANK_STABILITY_K_VALUES = (5, 10, 15, 20)   # top-K overlap check

    # ── Sub-steps 11.11 MI preservation (Test F) + 11.12 univariate transfer (Test A) ─
    RUN_UNIVARIATE   = True     # toggle (trains one tiny tree per feature — ~seconds total)
    UNIVARIATE_DEPTH = 3        # shallow tree for univariate evaluation
    UNIVARIATE_CAP   = 20_000   # per-class row cap (8 classes × 20k = 160k rows max)

    # Output file names. Combined results land in results/11_cross_analysis/<algorithm>/ (no
    # per-dataset subfolders). Directory 11_cross_analysis matches the script filename (the
    # cross_output_dir / cross_results_dir helpers point at the same base path).
    RESULTS_FILE  = '11_cross_analysis_report.txt'
    STEPS_FILE    = '11_cross_analysis_steps.log'
    # Markdown output: results.md — detailed reference tables (every value where practical).
    RESULTS_MD_FILE = 'results.md'


# ============================================================================
# SHARED UTILITIES (label mapping, used by multiple steps)
# ============================================================================

# Exact label -> canonical family mapping
LABEL_MAP_EXACT = {
    'BENIGN': 'Benign', 'Benign': 'Benign',
    'DoS Hulk': 'DoS', 'DoS GoldenEye': 'DoS', 'DoS slowloris': 'DoS',
    'DoS Slowhttptest': 'DoS',
    'DoS attacks-GoldenEye': 'DoS', 'DoS attacks-Hulk': 'DoS',
    'DoS attacks-SlowHTTPTest': 'DoS', 'DoS attacks-Slowloris': 'DoS',
    'DDoS': 'DDoS',
    'DDOS attack-HOIC': 'DDoS', 'DDOS attack-LOIC-HTTP': 'DDoS',
    'DDoS attacks-LOIC-HTTP': 'DDoS',
    'SSH-Patator': 'BruteForce', 'FTP-Patator': 'BruteForce',
    'SSH-BruteForce': 'BruteForce', 'FTP-BruteForce': 'BruteForce',
    'Web Attack -- Brute Force': 'WebAttack', 'Web Attack -- XSS': 'WebAttack',
    'Web Attack -- Sql Injection': 'WebAttack',
    'Brute Force -Web': 'WebAttack', 'Brute Force -XSS': 'WebAttack',
    'SQL Injection': 'WebAttack', 'XSS': 'WebAttack',
    'Bot': 'Botnet', 'Botnet': 'Botnet',
    'PortScan': 'PortScan', 'Portscan': 'PortScan',
    'Infiltration': 'Infiltration', 'Infiltration-CoolIOT': 'Infiltration',
    'Infiltration-Dropbox': 'Infiltration', 'Infiltration-ARES': 'Infiltration',
    # These raw strings were previously resolved only via the substring fallback
    # below (traced and confirmed correct, but the fallback is brittle — a future new label
    # variant that doesn't hit a fallback branch would return None and be silently dropped with
    # only a log.warn). Added directly here from the real raw label inventories in
    # results/1_load_clean_combine/{cicids2017,cicids2018}/1_load_clean_combine_report.txt so the
    # "exact map" is actually exhaustive for every raw label this pipeline has ever seen.
    'DDoS-HOIC': 'DDoS', 'DDoS-LOIC-HTTP': 'DDoS', 'DDoS-LOIC-UDP': 'DDoS',
    'Botnet Ares': 'Botnet',
    'Infiltration - NMAP Portscan': 'Infiltration',
    'Infiltration - Communication Victim Attacker': 'Infiltration',
    'Infiltration - Dropbox Download': 'Infiltration',
    'Infiltration - Portscan': 'Infiltration',
    'Web Attack - Brute Force': 'WebAttack', 'Web Attack - XSS': 'WebAttack',
    'Web Attack - SQL': 'WebAttack', 'Web Attack - SQL Injection': 'WebAttack',
    # Heartbleed is intentionally NOT mapped: it is not a canonical multiclass family
    # (see CANONICAL_MULTICLASS) and CICIDS2017 has only ~11 Heartbleed flows — far below
    # Config1.MIN_CLASS_ROWS, so it is dropped in step 1. map_label() returns None for it,
    # which step 1 already handles as "unmapped". Mapping it to a bare 'Heartbleed' string
    # would crash step 4 (unknown label) if the floor were ever lowered.
}

# Case-insensitive version for variant matching
LABEL_MAP_LOWER = {k.lower(): v for k, v in LABEL_MAP_EXACT.items()}

def map_label(raw: str) -> 'str | None':
    """Map a raw label string to its canonical family. Returns None if unmapped."""
    raw = raw.strip()
    # 1. Exact match (fastest)
    if raw in LABEL_MAP_EXACT:
        return LABEL_MAP_EXACT[raw]
    # 2. Case-insensitive match
    low = raw.lower()
    if low in LABEL_MAP_LOWER:
        return LABEL_MAP_LOWER[low]
    # 3. Pattern fallback
    if 'infiltration' in low:
        return 'Infiltration'
    if 'portscan' in low or 'port scan' in low:
        return 'PortScan'
    if low.startswith('ddos') or low.startswith('dddos'):
        return 'DDoS'
    if low.startswith('dos'):
        return 'DoS'
    if 'web attack' in low:
        return 'WebAttack'
    if 'brute force' in low or 'bruteforce' in low or 'patator' in low:
        if any(k in low for k in ('web', 'xss', 'sql')):
            return 'WebAttack'
        return 'BruteForce'
    if 'botnet' in low or low == 'bot':
        return 'Botnet'
    return None


# ── Canonical multiclass label -> integer code (shared by scripts 4 and 5) ───
CANONICAL_MULTICLASS: dict[str, int] = {
    'Benign':       0,
    'Botnet':       1,
    'BruteForce':   2,
    'DDoS':         3,
    'DoS':          4,
    'Infiltration': 5,
    'PortScan':     6,
    'WebAttack':    7,
}

# Config5.N_CLASSES_MULTI (defined above CANONICAL_MULTICLASS in this file, so it
# can't just read len(CANONICAL_MULTICLASS) directly at class-definition time) is an independent
# hardcoded literal that must stay in sync with this dict's size. Loud, import-time failure instead
# of a silent out-of-bounds/mis-sized-array bug if a family is ever added/removed here without
# updating that constant.
assert Config5.N_CLASSES_MULTI == len(CANONICAL_MULTICLASS), (
    f'Config5.N_CLASSES_MULTI ({Config5.N_CLASSES_MULTI}) != len(CANONICAL_MULTICLASS) '
    f'({len(CANONICAL_MULTICLASS)}) — update Config5.N_CLASSES_MULTI to match.')


# ── Label display normalisation (shared by script 0) ─────────────────────────
# Maps lowercase-stripped raw labels to their canonical display form.
LABEL_STANDARDIZATION: dict[str, str] = {
    'benign': 'Benign',
}


# ── Data quality categories (shared by script 0) ─────────────────────────────
DATA_QUALITY_TYPES:  list[str] = ['clean', 'nan_only', 'inf_only', 'both']
DATA_QUALITY_COLORS: list[str] = ['#4CAF50', '#FFC107', '#F44336', '#9C27B0']


# ── Column-name helpers (shared by script 0) ─────────────────────────────────
def normalize_column_name(name: str) -> str:
    """Strip leading/trailing whitespace from a column name."""
    return name.strip()


def is_label_column(name: str) -> bool:
    """Return True if the column (after stripping) is the target label column."""
    return normalize_column_name(name) == 'Label'


# ============================================================================
# Common Paths (used by multiple steps)
# ============================================================================

_STEP_RESULT_DIRS = {
    0:  RESULTS_ROOT / '0_dataexplore',
    1:  RESULTS_ROOT / '1_load_clean_combine',
    2:  RESULTS_ROOT / '2_correlation_analysis',
    3:  RESULTS_ROOT / '3_correlation_comparison',
    4:  RESULTS_ROOT / '4_preprocessing',
    5:  RESULTS_ROOT / '5_training',
    6:  RESULTS_ROOT / '6_testing',
    7:  RESULTS_ROOT / '7_profile',
    8:  RESULTS_ROOT / '8_visualize',
    9:  RESULTS_ROOT / '9_plan_comparison',
    10: RESULTS_ROOT / '10_execute_comparison',
    11: RESULTS_ROOT / '11_cross_analysis',
}


def ensure_results_dir(step_num: int, dataset_name: str) -> Path:
    """Create and return the results directory for a given step and dataset."""
    if step_num not in _STEP_RESULT_DIRS:
        raise ValueError(f"Unknown step: {step_num}")
    results_dir = _STEP_RESULT_DIRS[step_num] / dataset_name
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


# ============================================================================
# SHARED LOGGER (used by all scripts)
# ============================================================================

class Logger:
    """Dual-output logger: console + file. step_prefix sets the N in [SUB-STEP N.x]."""

    def __init__(self, path: Path, step_prefix: int, title: str = ''):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, 'w', encoding='utf-8')
        self._n = 0
        self._prefix = step_prefix
        header = title or f'SCRIPT {step_prefix} STEPS LOG'
        self._emit(
            f"{'='*70}\n"
            f"{header}\n"
            f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
            f"{'='*70}"
        )

    def step(self, title: str):
        self._n += 1
        ts = datetime.now().strftime('%H:%M:%S')
        self._emit(f'\n[SUB-STEP {self._prefix}.{self._n}] {title} (started: {ts})')

    def info(self, msg: str):
        self._emit(f'  {msg}')

    def ok(self, msg: str = 'Done'):
        self._emit(f'  [OK] {msg}')

    def step_end(self):
        """Log the end of current sub-step with timestamp."""
        ts = datetime.now().strftime('%H:%M:%S')
        self._emit(f'  [✓] Sub-step {self._prefix}.{self._n} completed at {ts}')

    def warn(self, msg: str):
        self._emit(f'  [!] {msg}')

    def section(self, title: str):
        self._emit(f'\n{"=" * 60}\n{title}\n{"=" * 60}')

    def close(self):
        self._emit(
            f"\n{'='*70}\n"
            f"Completed: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
            f"{'='*70}"
        )
        self._f.close()

    def _emit(self, msg: str):
        print(msg)
        self._f.write(msg + '\n')
        self._f.flush()


# ============================================================================
# BRANCH-B DATA ACCESS (steps 7, 8, 9, 10)
# ============================================================================
# Branch B reads the CLEANED, RAW-UNIT, FULL data written by step 1
# (data/cc_data/<ds>_cleaned.parquet) — never the z-scored step-4 output.

# Per-step output dirs consumed across Branch B (mirror Config7.OUTPUT_DIR / Config9.OUTPUT_DIR).
PROFILE_OUTPUT_DIR = OUTPUT_ROOT / '7_profile'      # step 7: profiles.json
COMPARE_OUTPUT_DIR = OUTPUT_ROOT / '9_plan_comparison'  # step 9: comparison plans


def raw_parquet_path(ds: str) -> Path:
    """The cleaned, raw-unit columnar parquet from step 1 — the input for steps 7/8/9/10."""
    return CLEANED_DATA_ROOT / f'{ds}_cleaned.parquet'


def profile_json_path(ds: str) -> Path:
    """Step 7 output, per-dataset subfolder: output/7_profile/<ds>/profiles.json."""
    return PROFILE_OUTPUT_DIR / ds / 'profiles.json'


def require_raw_parquet(ds: str) -> Path:
    path = raw_parquet_path(ds)
    if not path.exists():
        raise FileNotFoundError(
            f'{path} missing -- run step 1 (1_load_clean_combine.py) for {ds} first '
            f'(it writes the cleaned, raw-unit columnar parquet that 7/8/9/10 read).')
    return path


def schema_columns(ds: str) -> list[str]:
    """All column names in the raw parquet, in file order (cheap: reads metadata only)."""
    return list(pq.ParquetFile(require_raw_parquet(ds)).schema.names)


def feature_columns(ds: str, include_identifiers: bool = False) -> list[str]:
    """Analyzable feature columns: every column except Label and (by default) the
    identifier/quasi-identifier columns."""
    cols = [c for c in schema_columns(ds) if c != LABEL_COL]
    if not include_identifiers:
        skip = {c.lower() for c in BRANCH_B_IDENTIFIERS}
        cols = [c for c in cols if c.lower() not in skip]
    return cols


def is_identifier(feature: str) -> bool:
    return feature.lower() in {c.lower() for c in BRANCH_B_IDENTIFIERS}


def read_labels_encoded(ds: str) -> 'tuple[np.ndarray, list[str], np.ndarray]':
    """Read the Label column ONCE, dictionary-encoded — the memory-safe path for large datasets.

    Returns (codes int16, categories list[str], y_bin int8 with 0=Benign / 1=Attack).
    """
    path = require_raw_parquet(ds)
    ca = pq.read_table(path, columns=[LABEL_COL]).column(LABEL_COL)
    dic = ca.dictionary_encode().unify_dictionaries()
    chunks = dic.chunks
    if not chunks:
        return np.empty(0, np.int16), [], np.empty(0, np.int8)
    categories = [str(x) for x in chunks[0].dictionary.to_pylist()]
    codes = np.concatenate(
        [c.indices.to_numpy(zero_copy_only=False) for c in chunks]).astype(np.int16)
    benign_code = categories.index(BENIGN_LABEL) if BENIGN_LABEL in categories else -1
    y_bin = (codes != benign_code).astype(np.int8)
    return codes, categories, y_bin


def read_feature_only(ds: str, feature: str) -> np.ndarray:
    """Read ONE feature column with NO label column (pairs with read_labels_encoded()).
    Peak memory is one column, not column + full label list."""
    path = require_raw_parquet(ds)
    return pq.read_table(path, columns=[feature]).column(feature).to_numpy(zero_copy_only=False)


# ── Filename sanitization (feature names contain '/', e.g. 'Flow Bytes/s') ───────
_PATH_UNSAFE = '/\\:*?"<>|'


def safe_filename(name: str) -> str:
    """Make a feature name safe to use as a filename by replacing path-unsafe chars with '_'."""
    out = name
    for ch in _PATH_UNSAFE:
        out = out.replace(ch, '_')
    return out.strip()


# ============================================================================
# ML PIPELINE HELPERS (steps 5, 6, 11)
# ============================================================================
# Step-4 artifacts (train/test parquet, feature list, label map) + memory-bounded
# Parquet readers + the per-class stratified training loader.


def algo_suffix(algorithm: str) -> str:
    """'lightgbm' -> '_lightgbm', '' -> ''."""
    return f'_{algorithm}' if algorithm else ''


def save_fig(fig, out_path: Path, log: Logger) -> None:
    """Shared tail for every plot-saving call site in steps 5/6: tight_layout, mkdir -p the
    parent, save at dpi=150 with tight bbox, close the figure, log it. Identical across all
    six call sites in 5_train.py/6_test.py before this helper existed — factored out verbatim,
    no change to any argument or default. Imports pyplot lazily so scripts that don't plot
    never pay for matplotlib, and so callers that set matplotlib.use('Agg') before importing
    this module keep controlling the backend (pyplot is already configured by the time a
    plotting script actually calls this function)."""
    import matplotlib.pyplot as plt
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.ok(f'Saved {out_path.name}')


def _algo_dir(root: Path, kind: str, step: str, algorithm: str, ds: 'str | None' = None) -> Path:
    """Shared template behind the six `*_output_dir`/`*_results_dir` functions below:
    root / kind / step / f'{ds}{algo_suffix(algorithm)}' when ds is given (training/testing
    variants), or root / kind / step / algorithm when there's no dataset (cross_* variants,
    which are already algorithm-only directories like 'output/11_cross_analysis/lightgbm/')."""
    if ds is None:
        return root / kind / step / algorithm
    return root / kind / step / f'{ds}{algo_suffix(algorithm)}'


def training_output_dir(root: Path, ds: str, algorithm: str) -> Path:
    """output/5_training/cicids2017_lightgbm/."""
    return _algo_dir(root, 'output', '5_training', algorithm, ds)


def training_results_dir(root: Path, ds: str, algorithm: str) -> Path:
    """results/5_training/cicids2017_lightgbm/."""
    return _algo_dir(root, 'results', '5_training', algorithm, ds)


def testing_output_dir(root: Path, ds: str, algorithm: str) -> Path:
    """output/6_testing/cicids2017_lightgbm/."""
    return _algo_dir(root, 'output', '6_testing', algorithm, ds)


def testing_results_dir(root: Path, ds: str, algorithm: str) -> Path:
    """results/6_testing/cicids2017_lightgbm/ (mirrors training_results_dir layout)."""
    return _algo_dir(root, 'results', '6_testing', algorithm, ds)


def cross_output_dir(root: Path, algorithm: str) -> Path:
    """output/11_cross_analysis/lightgbm/."""
    return _algo_dir(root, 'output', '11_cross_analysis', algorithm)


def cross_results_dir(root: Path, algorithm: str) -> Path:
    """results/11_cross_analysis/lightgbm/."""
    return _algo_dir(root, 'results', '11_cross_analysis', algorithm)


def dataset_dir(ds: str) -> Path:
    return PREPROCESS_ROOT / ds


def load_feature_names(ds: str) -> list[str]:
    """Return the ordered feature column list step 4 persisted for this dataset."""
    path = dataset_dir(ds) / 'feature_names.json'
    if not path.exists():
        raise FileNotFoundError(
            f'{path} missing — run 4_preprocessing.py for {ds} before training.')
    with open(path, encoding='utf-8') as f:
        return list(json.load(f)['features'])


def load_label_mapping(ds: str) -> dict:
    """Return the label_mapping.json dict (binary + multiclass maps)."""
    path = dataset_dir(ds) / 'label_mapping.json'
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def multiclass_names(label_map: dict) -> 'dict[int, str]':
    """int code -> class name, from the multiclass map in label_mapping.json."""
    return {int(v): k for k, v in label_map['multiclass'].items()}


def binary_names() -> 'dict[int, str]':
    return {0: 'Benign', 1: 'Attack'}


def train_path(ds: str) -> Path:
    return dataset_dir(ds) / 'train.parquet'


def test_path(ds: str) -> Path:
    return dataset_dir(ds) / 'test.parquet'


def _batch_to_X(batch, features: list[str]) -> np.ndarray:
    """RecordBatch -> X float32 [rows x n_features], columns in `features` order."""
    cols = [batch.column(f).to_numpy(zero_copy_only=False) for f in features]
    return np.column_stack(cols).astype(np.float32, copy=False)


def iter_feature_batches(path: Path, features: list[str],
                         batch_rows: int = DEFAULT_BATCH_ROWS):
    """Yield (X float32, y_bin int8, y_multi int8) over a Parquet file, batch by batch.
    Only the feature columns + the two label columns are read; memory is bounded to one batch."""
    pf = pq.ParquetFile(path)
    cols = list(features) + list(LABEL_COLS)
    for batch in pf.iter_batches(batch_size=batch_rows, columns=cols):
        X = _batch_to_X(batch, features)
        y_bin   = batch.column(LABEL_BINARY).to_numpy(zero_copy_only=False).astype(np.int8, copy=False)
        y_multi = batch.column(LABEL_MULTICLASS).to_numpy(zero_copy_only=False).astype(np.int8, copy=False)
        yield X, y_bin, y_multi


def hash_feature_matrix(X: np.ndarray, features: list[str]) -> np.ndarray:
    """uint64 hash per row of an in-memory float32 feature matrix (labels excluded).

    Must stay representation-identical to feature_row_hashes() below (same float32 columns,
    same column order) so in-memory samples can be matched against streamed Parquet rows.
    """
    return pd.util.hash_pandas_object(
        pd.DataFrame(X, columns=features), index=False).to_numpy()


def feature_row_hashes(path: Path, features: list[str],
                       batch_rows: int = DEFAULT_BATCH_ROWS) -> np.ndarray:
    """uint64 hash per row over the modeling FEATURE columns only, streamed batch by batch.

    Shared by step 6 (train/test overlap check + overlap-free metrics) and step 5 (removing
    exact train-duplicates from the permutation-importance holdout). Both
    sides hash the identical float32 representation produced by iter_feature_batches(), so
    hash equality == exact feature-value duplication.
    """
    parts = []
    for X, _y_bin, _y_multi in iter_feature_batches(path, features, batch_rows):
        parts.append(hash_feature_matrix(X, features))
    return np.concatenate(parts) if parts else np.empty(0, dtype='uint64')


def count_multiclass(path: Path, n_classes: int,
                     batch_rows: int = DEFAULT_BATCH_ROWS) -> np.ndarray:
    """One cheap pass reading ONLY the multiclass label column -> per-class row counts."""
    pf = pq.ParquetFile(path)
    counts = np.zeros(n_classes, dtype=np.int64)
    for batch in pf.iter_batches(batch_size=batch_rows, columns=[LABEL_MULTICLASS]):
        y = batch.column(LABEL_MULTICLASS).to_numpy(zero_copy_only=False)
        counts += np.bincount(y, minlength=n_classes).astype(np.int64)
    return counts


def load_capped_subsample(path: Path, features: list[str], n_classes: int,
                          per_class_cap: int, log: 'Logger',
                          batch_rows: int = DEFAULT_BATCH_ROWS):
    """Load a STRATIFIED, memory-bounded training subsample (per-class cap).

    Keeps every row of any class at/under the cap; for classes above the cap, systematically
    keeps ~1 row in `stride` (stride = count // cap). Deterministic and reproducible.
    Returns (X float32 [n x f], y_bin int8 [n], y_multi int8 [n], kept_counts dict[int,int]).
    """
    counts = count_multiclass(path, n_classes, batch_rows)
    total = int(counts.sum())
    no_cap = (per_class_cap is None) or (per_class_cap <= 0)
    if no_cap:
        stride = np.ones(n_classes, dtype=np.int64)
        target = counts.copy()
        est_gb = total * len(features) * 4 / 1e9
        log.info(f'  source rows: {total:,}  ->  NO CAP (keeping all rows; '
                 f'~{est_gb:.1f} GB float32 in RAM). Ensure enough memory / use LightGBM engine.')
    else:
        stride = np.maximum(1, counts // max(per_class_cap, 1)).astype(np.int64)
        target = np.minimum(counts, per_class_cap)
        log.info(f'  source rows: {total:,}  ->  capped target ~{int(target.sum()):,} '
                 f'(per-class cap {per_class_cap:,})')

    X_parts: list = []
    yb_parts: list = []
    ym_parts: list = []
    running = np.zeros(n_classes, dtype=np.int64)   # per-class index seen so far (across batches)
    kept    = np.zeros(n_classes, dtype=np.int64)   # per-class rows kept so far

    for X, y_bin, y_multi in iter_feature_batches(path, features, batch_rows):
        if len(y_multi) == 0:
            continue
        s = pd.Series(y_multi)
        within = s.groupby(s, sort=False).cumcount().to_numpy()
        gidx = running[y_multi] + within

        keep = (gidx % stride[y_multi]) == 0
        if keep.any():
            for c in np.unique(y_multi[keep]):
                room = target[c] - kept[c]
                idx_c = np.where(keep & (y_multi == c))[0]
                if len(idx_c) > room:
                    keep[idx_c[room:]] = False
                kept[c] += min(len(idx_c), max(room, 0))

        running += np.bincount(y_multi, minlength=n_classes).astype(np.int64)

        if keep.any():
            X_parts.append(X[keep])
            yb_parts.append(y_bin[keep])
            ym_parts.append(y_multi[keep])

    X_out  = np.concatenate(X_parts, axis=0) if X_parts else np.empty((0, len(features)), np.float32)
    yb_out = np.concatenate(yb_parts) if yb_parts else np.empty((0,), np.int8)
    ym_out = np.concatenate(ym_parts) if ym_parts else np.empty((0,), np.int8)
    kept_counts = {int(c): int(n) for c, n in enumerate(kept) if n > 0}
    log.info(f'  loaded subsample: {X_out.shape[0]:,} rows x {X_out.shape[1]} features '
             f'({X_out.nbytes / 1e9:.2f} GB in RAM)')
    return X_out, yb_out, ym_out, kept_counts
