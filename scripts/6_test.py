"""
6_test.py — Test (evaluate) the trained LightGBM RF-mode models on the held-out CIC-IDS test split.

Purpose:
  Load the models persisted by 5_train.py and score them on test.parquet (the 20% the model
  never saw). This is the counterpart to 5_train.py: training and evaluation are separate so
  the test set is touched ONLY here, and re-scoring never requires re-fitting. For each dataset
  and each available task (binary / multiclass) it reports a full picture, not just accuracy
  (which is misleading under 76-94% Benign): per-class precision/recall/F1, macro & weighted
  averages, balanced accuracy, MCC, the confusion matrix, and — for the binary task — ROC-AUC
  and PR-AUC with curves.

Why metrics are computed from a streamed confusion matrix:
  2018's test split is ~12.6M rows. Rather than hold all predictions in RAM, we stream the
  Parquet file in batches, predict each batch, and accumulate one small confusion matrix. Every
  count-based metric (precision, recall, F1, accuracy, balanced accuracy, MCC) is derived
  EXACTLY from that matrix afterwards — no per-row arrays needed. Only the binary task keeps the
  positive-class probabilities (one float per row) so ROC/PR curves can be drawn; that is ~50 MB
  even for 2018 and is skipped for multiclass.

Why these metrics:
  - Accuracy alone rewards always-predict-Benign. Balanced accuracy (mean recall across classes)
    and macro-F1 expose failure on rare attacks (WebAttack, Infiltration, Botnet).
  - MCC is a single robust summary that stays honest under heavy imbalance.
  - ROC-AUC / PR-AUC (binary) are threshold-independent; PR-AUC is the more informative of the
    two when the positive (Attack) class is the minority.

Train/test overlap handling (check_train_test_overlap(), runs first, per dataset):
  Hashes train.parquet/test.parquet on the modeling feature columns and reports what fraction of
  TEST rows are exact duplicates of a TRAIN row (23.16% for cicids2017). Beyond the [CAVEAT]
  report line, the per-row duplicate mask now feeds evaluate_dataset() so every within-year
  metric is reported TWICE: once on the full test split (comparable to prior
  runs) and once on the OVERLAP-FREE subset (test rows that are not exact feature-duplicates of
  any train row — the honest generalization number).

Reads (per dataset <ds> and algorithm <algo>):
  output/5_training/<ds>_<algo>/rf_binary.joblib       (if present)
  output/5_training/<ds>_<algo>/rf_multiclass.joblib   (if present)
  output/4_preprocessing/<ds>/train.parquet
  output/4_preprocessing/<ds>/test.parquet
  output/4_preprocessing/<ds>/feature_names.json

Writes (per dataset <ds> and algorithm <algo>):
  output/4_preprocessing/<ds>/train_test_overlap.json
  output/6_testing/<ds>_<algo>/metrics_binary.json
  output/6_testing/<ds>_<algo>/metrics_multiclass.json
  results/6_testing/<ds>_<algo>/6_testing_steps.log
  results/6_testing/<ds>_<algo>/6_testing_report.txt
  results/6_testing/<ds>_<algo>/6_testing_confusion_binary.png
  results/6_testing/<ds>_<algo>/6_testing_confusion_multiclass.png
  results/6_testing/<ds>_<algo>/6_testing_roc_binary.png
  results/6_testing/<ds>_<algo>/6_testing_pr_binary.png
  results/6_testing/<ds>_<algo>/6_testing_per_class_f1_multiclass.png
"""

import sys
import time
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import joblib

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             precision_recall_curve, average_precision_score)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unified_config import (
    Config6, Logger, PROJECT_ROOT, RESULTS_ROOT, OUTPUT_ROOT, DATASETS, ALGORITHM,
    DEFAULT_BATCH_ROWS,
    binary_names, multiclass_names, iter_feature_batches, load_feature_names, load_label_mapping,
    train_path, test_path, testing_output_dir, testing_results_dir, training_output_dir,
    dataset_dir, CANONICAL_MULTICLASS, feature_row_hashes, BENIGN_LABEL, save_fig,
)

TASKS = Config6.TASKS
PLOT_CURVE_POINTS = Config6.PLOT_CURVE_POINTS   # cap drawn points on ROC/PR curves (AUC stays exact)


# ── Metrics derived exactly from a confusion matrix ─────────────────────────────
def metrics_from_cm(cm: np.ndarray, names: list[str]) -> dict:
    """Compute accuracy, per-class P/R/F1, macro/weighted averages, balanced acc, MCC.

    cm[i, j] = #(true class i predicted as class j). Rows are truth, cols are predictions.
    """
    cm = cm.astype(np.float64)
    total = cm.sum()
    tp      = np.diag(cm)
    support = cm.sum(axis=1)                 # true count per class
    pred    = cm.sum(axis=0)                 # predicted count per class

    precision = np.divide(tp, pred,    out=np.zeros_like(tp), where=pred > 0)
    recall    = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)

    present = support > 0                     # classes that actually occur in the test set
    accuracy      = float(tp.sum() / total) if total else 0.0
    macro_f1      = float(f1[present].mean()) if present.any() else 0.0
    macro_prec    = float(precision[present].mean()) if present.any() else 0.0
    macro_rec     = float(recall[present].mean()) if present.any() else 0.0
    weighted_f1   = float((f1 * support).sum() / total) if total else 0.0
    balanced_acc  = float(recall[present].mean()) if present.any() else 0.0

    # Multiclass Matthews correlation coefficient straight from the confusion matrix.
    s = total
    c = tp.sum()
    cov_ytyp = c * s - (pred * support).sum()
    cov_ypyp = s * s - (pred * pred).sum()
    cov_ytyt = s * s - (support * support).sum()
    mcc_denom = np.sqrt(cov_ypyp * cov_ytyt)
    mcc = float(cov_ytyp / mcc_denom) if mcc_denom > 0 else 0.0

    per_class = {
        names[i]: {
            'precision': float(precision[i]),
            'recall':    float(recall[i]),
            'f1':        float(f1[i]),
            'support':   int(support[i]),
        }
        for i in range(len(names))
    }
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'macro_precision': macro_prec,
        'macro_recall': macro_rec,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'mcc': mcc,
        'per_class': per_class,
        'confusion_matrix': cm.astype(np.int64).tolist(),
        'labels': names,
    }


# ── Streaming prediction over the test Parquet ──────────────────────────────────
def evaluate_dataset(ds: str, tasks: list[str], log: Logger,
                     test_dup_mask: 'np.ndarray | None' = None) -> dict:
    """Score the trained models on the held-out split. If `test_dup_mask` is given (True where a
    test row exactly duplicates a train row, from check_train_test_overlap), every metric is
    computed TWICE: on the full split and on the overlap-free subset."""
    te_path = test_path(ds)
    if not te_path.exists():
        log.warn(f'Missing {te_path} — run 4_preprocessing.py for {ds} first')
        return {}

    features = load_feature_names(ds)
    label_map = load_label_mapping(ds)
    inv_multi = multiclass_names(label_map)

    # Algorithm-suffixed folder: reads models from output/5_training/cicids2017_lightgbm/
    train_dir   = training_output_dir(PROJECT_ROOT, ds, ALGORITHM)
    out_dir     = testing_output_dir(PROJECT_ROOT, ds, ALGORITHM)
    results_dir = testing_results_dir(PROJECT_ROOT, ds, ALGORITHM)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load whichever requested models exist.
    models: dict[str, object] = {}
    for task in tasks:
        mpath = train_dir / f'rf_{task}.joblib'
        if mpath.exists():
            models[task] = joblib.load(mpath)
            log.info(f'  loaded {mpath.name}  (from {train_dir.name})')
        else:
            log.warn(f'  no model for {task}: {mpath} (skipping — run 5_train.py first)')
    if not models:
        log.warn(f'  no models found for {ds} [{ALGORITHM}]; skipping')
        return {}

    # Per-task accumulators keyed on the model's own class set. The *_clean twins accumulate
    # only overlap-free rows (test rows that are NOT exact feature-duplicates of a train row).
    cms = {t: np.zeros((len(models[t].classes_), len(models[t].classes_)), dtype=np.int64)
           for t in models}
    cms_clean = ({t: np.zeros_like(cms[t]) for t in models}
                 if test_dup_mask is not None else None)
    class_codes = {t: list(models[t].classes_) for t in models}
    bin_true_parts: list[np.ndarray] = []
    bin_score_parts: list[np.ndarray] = []
    pos_idx = None
    if 'binary' in models:
        pos_idx = list(models['binary'].classes_).index(1)

    log.step(f'[{ds}] stream test.parquet and predict ({len(models)} model(s))')
    n_rows = 0
    t0 = time.time()
    for X, y_bin, y_multi in iter_feature_batches(te_path, features):
        start = n_rows
        n_rows += X.shape[0]
        # Overlap-free row selector for this batch (streaming order matches the hash pass).
        keep = None
        if test_dup_mask is not None:
            keep = ~test_dup_mask[start:n_rows]
        if 'binary' in models:
            proba = models['binary'].predict_proba(X)
            y_pred = models['binary'].classes_[proba.argmax(axis=1)]
            cms['binary'] += confusion_matrix(y_bin, y_pred, labels=class_codes['binary'])
            if keep is not None:
                cms_clean['binary'] += confusion_matrix(
                    y_bin[keep], y_pred[keep], labels=class_codes['binary'])
            bin_true_parts.append(y_bin.astype(np.int8, copy=False))
            bin_score_parts.append(proba[:, pos_idx].astype(np.float32, copy=False))
        if 'multiclass' in models:
            y_pred = models['multiclass'].predict(X)
            cms['multiclass'] += confusion_matrix(y_multi, y_pred, labels=class_codes['multiclass'])
            if keep is not None:
                cms_clean['multiclass'] += confusion_matrix(
                    y_multi[keep], y_pred[keep], labels=class_codes['multiclass'])
        if n_rows % (DEFAULT_BATCH_ROWS * 8) < DEFAULT_BATCH_ROWS:
            rate = n_rows / max(time.time() - t0, 1e-9)
            log.info(f'    predicted {n_rows:,} rows ({rate:,.0f} rows/s)')
    log.ok(f'scored {n_rows:,} test rows in {time.time() - t0:.1f}s')

    ds_metrics: dict[str, dict] = {}

    for task in models:
        names = ([binary_names()[c] for c in class_codes[task]] if task == 'binary'
                 else [inv_multi[c] for c in class_codes[task]])
        m = metrics_from_cm(cms[task], names)

        # Overlap-free twin of every count-based metric (the honest generalization numbers).
        if cms_clean is not None:
            m_clean = metrics_from_cm(cms_clean[task], names)
            m_clean['n_rows'] = int((~test_dup_mask).sum())
            m['overlap_free'] = m_clean

        if task == 'binary':
            y_true = np.concatenate(bin_true_parts)
            y_score = np.concatenate(bin_score_parts)
            fpr, tpr, _ = roc_curve(y_true, y_score)
            m['roc_auc'] = float(auc(fpr, tpr))
            m['pr_auc']  = float(average_precision_score(y_true, y_score))
            prec, rec, _ = precision_recall_curve(y_true, y_score)
            plot_roc(ds, fpr, tpr, m['roc_auc'], results_dir / '6_testing_roc_binary.png', log)
            plot_pr(ds, rec, prec, m['pr_auc'], results_dir / '6_testing_pr_binary.png', log)
            if cms_clean is not None:
                keep_all = ~test_dup_mask
                yt_c, ys_c = y_true[keep_all], y_score[keep_all]
                if np.unique(yt_c).size == 2:
                    fpr_c, tpr_c, _ = roc_curve(yt_c, ys_c)
                    m['overlap_free']['roc_auc'] = float(auc(fpr_c, tpr_c))
                    m['overlap_free']['pr_auc']  = float(average_precision_score(yt_c, ys_c))
                del yt_c, ys_c
            del y_true, y_score

        plot_confusion(ds, task, cms[task], names,
                       results_dir / f'6_testing_confusion_{task}.png', log)
        if task == 'multiclass':
            plot_per_class_f1(ds, m['per_class'],
                              results_dir / '6_testing_per_class_f1_multiclass.png', log)

        with open(out_dir / f'metrics_{task}.json', 'w', encoding='utf-8') as f:
            json.dump(m, f, indent=2)
        clean_note = ''
        if 'overlap_free' in m:
            clean_note = (f'  overlap_free: acc={m["overlap_free"]["accuracy"]:.4f} '
                          f'macro_f1={m["overlap_free"]["macro_f1"]:.4f}')
        log.ok(f'Saved metrics_{task}.json  '
               f'(acc={m["accuracy"]:.4f}  macro_f1={m["macro_f1"]:.4f}  mcc={m["mcc"]:.4f})'
               + clean_note)
        ds_metrics[task] = m

    return {'dataset': ds, 'n_test_rows': n_rows, 'metrics': ds_metrics}


# ── Cross-dataset (cross-year) evaluation ───────────────────────────────────────
# The headline of the whole project: a model trained on ONE year applied to the OTHER year.
# evaluate_dataset() above gives the inflated within-year baseline (~99.9%); this is the cross-year
# transfer that actually collapses and which the step-11 ablation compares feature subsets against.
# Two framings are reported from the SAME model (tree splits are scale-invariant, so only the test
# INPUT differs — the model is identical):
#   CONCEPT   : the target year keeps its OWN per-year z-scaler, so the between-year location/scale
#               (covariate) shift is normalized away -> isolates concept transfer.
#   COVARIATE : the target year is re-expressed in the TRAIN year's scaler frame
#               (native = z*scale_te + mean_te ; z' = (native - mean_tr)/scale_tr), preserving the
#               covariate shift -> the deployment-like failure mode (matches the VM story and the
#               step-11 ablation 'covariate' framing exactly).

def load_scaler(ds: str) -> dict:
    """feature -> (mean, scale) from step-4 scaler.json (per-year z-score params). Mirrors
    11_cross_analysis._load_scaler so step-6 cross numbers match the step-11 ablation baseline."""
    p = dataset_dir(ds) / 'scaler.json'
    d = json.loads(p.read_text(encoding='utf-8'))
    return {f: (float(m), float(s)) for f, m, s in zip(d['features'], d['mean'], d['scale'])}


def rebase_covariate(X: np.ndarray, features: list[str], sc_te: dict, sc_tr: dict) -> np.ndarray:
    """Re-express test-year z-scored columns in the TRAIN year's scaler frame, column by column:
    native = z*scale_te + mean_te ; z' = (native - mean_tr)/scale_tr. Preserves the covariate shift."""
    Xc = np.empty_like(X, dtype=np.float64)
    for j, f in enumerate(features):
        m_te, s_te = sc_te.get(f, (0.0, 1.0))
        m_tr, s_tr = sc_tr.get(f, (0.0, 1.0))
        s_tr = s_tr if s_tr else 1.0
        Xc[:, j] = (X[:, j].astype(np.float64) * s_te + m_te - m_tr) / s_tr
    return Xc


def _cross_summary(task: str, m: dict) -> dict:
    """Compact scalars step 11 (and results.md's Step-6 table) need as the real full-feature-model
    cross-year baseline. For binary, this mirrors 11_cross_analysis.py's _full_metrics() naming
    exactly (benign_f1/sensitivity/fpr/precision/specificity) so the ablation's K-feature policies
    (C18) and this real full-feature baseline are directly comparable on the SAME metric set —
    macro_f1/balanced_accuracy/mcc were already equivalent (both derived from the same per-class
    confusion-matrix counts), only the finer-grained breakdown was missing here."""
    out = {'accuracy': m['accuracy'], 'balanced_accuracy': m['balanced_accuracy'],
           'macro_f1': m['macro_f1'], 'mcc': m['mcc']}
    if task == 'binary':
        attack = m['per_class'].get('Attack', {})
        benign = m['per_class'].get('Benign', {})
        out['attack_f1'] = attack.get('f1', 0.0)          # pos-class F1 = ablation metric
        out['benign_f1'] = benign.get('f1', 0.0)
        out['sensitivity'] = attack.get('recall', 0.0)     # attack recall
        out['precision'] = attack.get('precision', 0.0)    # attack precision
        out['specificity'] = benign.get('recall', 0.0)     # benign recall
        out['fpr'] = 1.0 - out['specificity']
        out['roc_auc'] = m.get('roc_auc')
        out['pr_auc'] = m.get('pr_auc')
    return out


# ── Prior / threshold recalibration (threshold recalibration, not previously tested) ─────────────
# The cross-year collapse in the concept framing is a THRESHOLD failure, not a ranking failure
# (ROC-AUC stays high while recall at the implicit 0.5 threshold goes to ~0). This block takes the
# already-computed target-year attack probabilities and re-thresholds them under the known / an
# estimated prior, to measure how much attack-F1 a deployment-time recalibration recovers — with NO
# retraining. Reported strategies:
#   baseline_0.5      : the implicit argmax(0.5) threshold (== the number in Table 2).
#   prior_ratio_known : Saerens posterior adjustment using the TRUE target prior (upper bound on a
#                       prior-only correction; assumes the deployment prior is known/estimated).
#   sld_em            : Saerens-Latinne-Decaestecker EM — estimates the target prior from the
#                       UNLABELED target scores, then adjusts. The deployable, label-free method.
#   oracle_best_f1    : the threshold that maximizes attack-F1 on the target labels (ceiling: the
#                       most any single global threshold could recover; uses labels, not deployable).
# NOTE (class_weight='balanced'): the models are inverse-frequency reweighted, so their posteriors
# are not perfectly calibrated to the natural training prior used here as p_src; this is the standard
# Saerens assumption and the correction direction is validated empirically (2018->2017 recovers
# recall as expected). Kept as p_src = natural training-year attack prior for interpretability.

def train_attack_prior(ds: str) -> float:
    """P(attack) in the training-year TRAIN split, from step-4 preprocessing_meta.json."""
    p = dataset_dir(ds) / 'preprocessing_meta.json'
    d = json.loads(p.read_text(encoding='utf-8'))
    total = int(d.get('rows_train', 0))
    benign = int(d.get('class_counts_train', {}).get(BENIGN_LABEL, 0))
    return (1.0 - benign / total) if total else 0.5


def _binary_metrics_at(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compact attack-centric binary metrics from hard predictions (reuses metrics_from_cm)."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    m = metrics_from_cm(cm, ['Benign', 'Attack'])
    at = m['per_class'].get('Attack', {})
    bn = m['per_class'].get('Benign', {})
    return {
        'attack_f1': at.get('f1', 0.0), 'benign_f1': bn.get('f1', 0.0),
        'macro_f1': m['macro_f1'], 'precision': at.get('precision', 0.0),
        'recall': at.get('recall', 0.0), 'specificity': bn.get('recall', 0.0),
        'fpr': 1.0 - bn.get('recall', 0.0), 'mcc': m['mcc'],
        'balanced_accuracy': m['balanced_accuracy'], 'accuracy': m['accuracy'],
    }


def _sld_adjust(s: np.ndarray, p_src: float, p_tgt: float) -> np.ndarray:
    """Saerens posterior adjustment for binary: reweight P(attack) from prior p_src to p_tgt."""
    eps = 1e-12
    p_src = float(min(max(p_src, eps), 1 - eps))
    p_tgt = float(min(max(p_tgt, eps), 1 - eps))
    num = s * (p_tgt / p_src)
    den = num + (1.0 - s) * ((1.0 - p_tgt) / (1.0 - p_src))
    return num / (den + eps)


def _sld_em(s: np.ndarray, p_src: float, iters: int = 200, tol: float = 1e-9) -> float:
    """Saerens-Latinne-Decaestecker EM: estimate the target attack prior from unlabeled scores."""
    p = float(p_src)
    for _ in range(iters):
        p_new = float(np.clip(_sld_adjust(s, p_src, p).mean(), 1e-6, 1 - 1e-6))
        if abs(p_new - p) < tol:
            p = p_new
            break
        p = p_new
    return p


def recalibrate_binary(y_true: np.ndarray, y_score: np.ndarray, p_src: float) -> dict:
    """Compare baseline-0.5 vs prior/threshold recalibration strategies on target-year scores."""
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    p_tgt_true = float(y_true.mean())
    out = {'p_src_train_attack': round(float(p_src), 6),
           'p_tgt_true_attack': round(p_tgt_true, 6)}

    out['baseline_0.5'] = {**_binary_metrics_at(y_true, (y_score >= 0.5).astype(int)),
                           'threshold': 0.5}

    adj_known = _sld_adjust(y_score, p_src, p_tgt_true)
    out['prior_ratio_known'] = {**_binary_metrics_at(y_true, (adj_known >= 0.5).astype(int)),
                                'p_tgt_used': round(p_tgt_true, 6)}

    p_em = _sld_em(y_score, p_src)
    adj_em = _sld_adjust(y_score, p_src, p_em)
    out['sld_em'] = {**_binary_metrics_at(y_true, (adj_em >= 0.5).astype(int)),
                     'p_tgt_estimated': round(p_em, 6)}

    prec, rec, thr = precision_recall_curve(y_true, y_score)
    f1s = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    bi = int(np.nanargmax(f1s))
    bt = float(thr[min(bi, len(thr) - 1)]) if len(thr) else 0.5
    out['oracle_best_f1'] = {**_binary_metrics_at(y_true, (y_score >= bt).astype(int)),
                             'threshold': round(bt, 6)}
    return out


def evaluate_cross(train_ds: str, test_ds: str, tasks: list[str], log: Logger) -> dict:
    """Apply train_ds's models to test_ds's held-out split (cross-year transfer) under BOTH the
    CONCEPT and COVARIATE framings. Streams test_ds's 20% held-out split (test.parquet) in train_ds's
    feature ORDER so the model receives columns exactly as it was trained on.

    Design — clean 2x2 (the SAME per-year 20% test split in every cell, so only the MODEL differs and
    the four numbers are directly comparable):
        test 2017/20% on 2017-model   test 2018/20% on 2017-model   <- this fn (cross)
        test 2017/20% on 2018-model   test 2018/20% on 2018-model
    The within-year diagonal is produced by evaluate_dataset(); this function produces the cross
    off-diagonal. No leakage on the cross cells: the model never saw that year's 80% train OR its
    20% test."""
    te_path = test_path(test_ds)
    if not te_path.exists():
        log.warn(f'Missing {te_path} — run 4_preprocessing.py for {test_ds} first')
        return {}

    feats_tr = load_feature_names(train_ds)          # the model's expected feature ORDER
    feats_te_set = set(load_feature_names(test_ds))
    missing = [f for f in feats_tr if f not in feats_te_set]
    if missing:
        log.warn(f'  {test_ds} missing {len(missing)} of {train_ds} features (e.g. {missing[:3]}) '
                 f'-> cannot align; skipping cross-test')
        return {}

    train_dir = training_output_dir(PROJECT_ROOT, train_ds, ALGORITHM)
    tag = f'cross_{train_ds}_to_{test_ds}'
    out_dir     = testing_output_dir(PROJECT_ROOT, tag, ALGORITHM)
    results_dir = testing_results_dir(PROJECT_ROOT, tag, ALGORITHM)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    models: dict[str, object] = {}
    for task in tasks:
        mpath = train_dir / f'rf_{task}.joblib'
        if mpath.exists():
            models[task] = joblib.load(mpath)
            log.info(f'  loaded {mpath.name} (trained on {train_ds})')
        else:
            log.warn(f'  no {task} model at {mpath} (skipping that task)')
    if not models:
        log.warn(f'  no models for {train_ds} [{ALGORITHM}]; skipping cross-test')
        return {}

    sc_tr, sc_te = load_scaler(train_ds), load_scaler(test_ds)

    # Label spaces: binary = [0,1]; multiclass = the FULL canonical code set (0..7) so an attack
    # family present in the test year but never seen in the train year still counts as recall=0
    # (the model literally cannot detect it) instead of being silently dropped from the matrix.
    canon_codes = sorted(CANONICAL_MULTICLASS.values())
    canon_names = {v: k for k, v in CANONICAL_MULTICLASS.items()}
    label_codes = {'binary': [0, 1], 'multiclass': canon_codes}
    label_names = {'binary': [binary_names()[c] for c in (0, 1)],
                   'multiclass': [canon_names[c] for c in canon_codes]}

    framings = ('concept', 'covariate')
    cms = {t: {fr: np.zeros((len(label_codes[t]), len(label_codes[t])), dtype=np.int64)
               for fr in framings} for t in models}
    bin_true: list[np.ndarray] = []
    bin_score: dict[str, list] = {fr: [] for fr in framings}
    pos_idx = list(models['binary'].classes_).index(1) if 'binary' in models else None

    log.step(f'[{train_ds} -> {test_ds}] stream 20% test.parquet and cross-predict '
             f'({len(models)} model(s))')
    n_rows = 0
    t0 = time.time()
    for X, y_bin, y_multi in iter_feature_batches(te_path, feats_tr):
        n_rows += X.shape[0]
        batch_X = {'concept': X, 'covariate': rebase_covariate(X, feats_tr, sc_te, sc_tr)}
        if 'binary' in models:
            bin_true.append(y_bin.astype(np.int8, copy=False))
            for fr in framings:
                proba = models['binary'].predict_proba(batch_X[fr])
                y_pred = models['binary'].classes_[proba.argmax(axis=1)]
                cms['binary'][fr] += confusion_matrix(y_bin, y_pred, labels=label_codes['binary'])
                bin_score[fr].append(proba[:, pos_idx].astype(np.float32, copy=False))
        if 'multiclass' in models:
            for fr in framings:
                y_pred = models['multiclass'].predict(batch_X[fr])
                cms['multiclass'][fr] += confusion_matrix(
                    y_multi, y_pred, labels=label_codes['multiclass'])
        if n_rows % (DEFAULT_BATCH_ROWS * 8) < DEFAULT_BATCH_ROWS:
            log.info(f'    cross-predicted {n_rows:,} rows '
                     f'({n_rows/max(time.time()-t0,1e-9):,.0f} rows/s)')
    log.ok(f'cross-scored {n_rows:,} rows in {time.time()-t0:.1f}s')

    cross_metrics: dict[str, dict] = {}
    summary: dict[str, dict] = {}
    recal_out: dict[str, dict] = {}          # prior/threshold recalibration, binary only
    p_src_attack = train_attack_prior(train_ds)
    for task in models:
        cross_metrics[task], summary[task] = {}, {}
        for fr in framings:
            m = metrics_from_cm(cms[task][fr], label_names[task])
            if task == 'binary':
                y_true = np.concatenate(bin_true)
                y_score = np.concatenate(bin_score[fr])
                fpr, tpr, _ = roc_curve(y_true, y_score)
                m['roc_auc'] = float(auc(fpr, tpr))
                m['pr_auc']  = float(average_precision_score(y_true, y_score))
                prec, rec, _ = precision_recall_curve(y_true, y_score)
                plot_roc(f'{tag} [{fr}]', fpr, tpr, m['roc_auc'],
                         results_dir / f'6_cross_roc_binary_{fr}.png', log)
                plot_pr(f'{tag} [{fr}]', rec, prec, m['pr_auc'],
                        results_dir / f'6_cross_pr_binary_{fr}.png', log)
                # Prior/threshold recalibration on the target-year scores (no retraining).
                recal = recalibrate_binary(y_true, y_score, p_src_attack)
                m['recalibration'] = recal
                recal_out[fr] = recal
                with open(out_dir / f'recalibration_binary_{fr}.json', 'w', encoding='utf-8') as f:
                    json.dump(recal, f, indent=2)
                b05, em, orc = recal['baseline_0.5'], recal['sld_em'], recal['oracle_best_f1']
                log.ok(f'  [recalibration/{fr}] attack_f1 0.5={b05["attack_f1"]:.4f} -> '
                       f'sld_em={em["attack_f1"]:.4f} (p_hat={em["p_tgt_estimated"]:.3f}) '
                       f'| oracle={orc["attack_f1"]:.4f}')
                del y_true, y_score
            plot_confusion(f'{tag} [{fr}]', task, cms[task][fr], label_names[task],
                           results_dir / f'6_cross_confusion_{task}_{fr}.png', log)
            with open(out_dir / f'metrics_{task}_{fr}.json', 'w', encoding='utf-8') as f:
                json.dump(m, f, indent=2)
            cross_metrics[task][fr] = m
            summary[task][fr] = _cross_summary(task, m)
            log.ok(f'  [{task}/{fr}] acc={m["accuracy"]:.4f} macro_f1={m["macro_f1"]:.4f} '
                   f'mcc={m["mcc"]:.4f}'
                   + (f' pr_auc={m["pr_auc"]:.4f}' if task == 'binary' else ''))
    return {'direction': f'{train_ds}->{test_ds}', 'train_ds': train_ds, 'test_ds': test_ds,
            'n_test_rows': n_rows, 'metrics': cross_metrics, 'summary': summary,
            'recalibration': recal_out, 'results_dir': str(results_dir)}


def write_cross_report(cross_results: list[dict], out_path: Path, log: Logger):
    """Human-readable cross-year transfer report: within-year vs cross-year, both framings."""
    lines: list[str] = []

    def h(t):
        lines.extend(['', '=' * 70, t, '=' * 70])

    h('CROSS-YEAR TRANSFER REPORT  --  train on one year, test on the other')
    lines.append(f'Generated : {datetime.now():%Y-%m-%d %H:%M:%S}')
    lines += [
        '',
        '  Framings (same model, different test input):',
        '    CONCEPT   = target year self-normalized (covariate shift removed) -> concept transfer',
        '    COVARIATE = target re-based into train-year scaler (shift preserved) -> deployment failure',
        '  COVARIATE is the deployment-realistic number; the gap CONCEPT - COVARIATE is the share of',
        '  the collapse attributable to covariate (location/scale) shift vs concept change.',
    ]
    for r in cross_results:
        if not r:
            continue
        h(f'DIRECTION: {r["direction"]}   (test rows: {r["n_test_rows"]:,})')
        for task, fr_map in r['summary'].items():
            lines.append('')
            lines.append(f'  [{task}]')
            for fr, s in fr_map.items():
                extra = (f'  attack_f1={s.get("attack_f1", float("nan")):.4f}'
                         f'  benign_f1={s.get("benign_f1", float("nan")):.4f}'
                         f'  sensitivity={s.get("sensitivity", float("nan")):.4f}'
                         f'  fpr={s.get("fpr", float("nan")):.4f}'
                         f'  precision={s.get("precision", float("nan")):.4f}'
                         f'  specificity={s.get("specificity", float("nan")):.4f}'
                         f'  pr_auc={s.get("pr_auc") if s.get("pr_auc") is not None else float("nan"):.4f}'
                         if task == 'binary' else '')
                lines.append(f'    {fr:<10} accuracy={s["accuracy"]:.4f}  '
                             f'balanced_acc={s["balanced_accuracy"]:.4f}  '
                             f'macro_f1={s["macro_f1"]:.4f}  mcc={s["mcc"]:.4f}{extra}')

    text = '\n'.join(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text)
    log.ok(f'Saved {out_path.name}')
    print(text)


# ── Plots ───────────────────────────────────────────────────────────────────────
def _downsample(x: np.ndarray, y: np.ndarray):
    """Thin a curve to at most PLOT_CURVE_POINTS points (keeps endpoints)."""
    if len(x) <= PLOT_CURVE_POINTS:
        return x, y
    idx = np.linspace(0, len(x) - 1, PLOT_CURVE_POINTS).astype(int)
    return x[idx], y[idx]


def plot_confusion(ds, task, cm, names, out_path: Path, log: Logger):
    cm = cm.astype(np.float64)
    row = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)   # row-normalised (recall view)
    fig, ax = plt.subplots(figsize=(max(5, len(names) * 0.9), max(4, len(names) * 0.8)))
    im = ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'{ds} — {task} confusion (row-normalised)', fontsize=10)
    thresh = 0.5
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f'{norm[i, j]:.2f}\n{int(cm[i, j]):,}',
                    ha='center', va='center', fontsize=6,
                    color='white' if norm[i, j] > thresh else 'black')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_fig(fig, out_path, log)


def plot_roc(ds, fpr, tpr, roc_auc, out_path: Path, log: Logger):
    fpr, tpr = _downsample(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color='#c0392b', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='#7f8c8d', lw=1, ls='--', label='chance')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{ds} — binary ROC', fontsize=10)
    ax.legend(loc='lower right', fontsize=8)
    save_fig(fig, out_path, log)


def plot_pr(ds, rec, prec, pr_auc, out_path: Path, log: Logger):
    rec, prec = _downsample(rec, prec)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(rec, prec, color='#27ae60', lw=2, label=f'PR (AP = {pr_auc:.4f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'{ds} — binary Precision-Recall', fontsize=10)
    ax.legend(loc='lower left', fontsize=8)
    save_fig(fig, out_path, log)


def plot_per_class_f1(ds, per_class, out_path: Path, log: Logger):
    names = list(per_class.keys())
    f1s   = [per_class[n]['f1'] for n in names]
    recs  = [per_class[n]['recall'] for n in names]
    x = np.arange(len(names))
    w = 0.4
    fig, ax = plt.subplots(figsize=(max(7, len(names) * 1.0), 4.5))
    ax.bar(x - w / 2, f1s,  w, label='F1',     color='#8e44ad')
    ax.bar(x + w / 2, recs, w, label='recall', color='#e67e22')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('score')
    ax.set_title(f'{ds} — multiclass per-class F1 & recall', fontsize=10)
    ax.legend(fontsize=8)
    for xi, v in enumerate(f1s):
        ax.text(xi - w / 2, v, f'{v:.2f}', ha='center', va='bottom', fontsize=6)
    save_fig(fig, out_path, log)


# ── Train/test overlap check ─────────────────────────────────────────────────────
def check_train_test_overlap(ds: str, log: Logger) -> 'tuple[dict, np.ndarray]':
    """How many TEST rows are exact feature-value duplicates of a TRAIN row?

    Restored (2026-07-02) after being found deleted from scripts/ with no replacement — its
    prior finding (23.16% of cicids2017 test rows were exact duplicates of a train row) was
    still recorded in train_test_overlap.json but had become unreproducible, and undisclosed
    anywhere in active code. Lives here (not as a standalone script) because this is exactly
    where the within-year metrics that need this caveat are computed and reported.

    Hashes each row of train.parquet / test.parquet on the modeling feature columns only
    (excludes label_binary/label_multiclass), via the shared feature_row_hashes() helper —
    the same hashing 5_train.py uses to overlap-filter its permutation-importance sample.

    Returns (result dict, test_dup_mask): test_dup_mask[i] is True when test row i (in
    streaming order) exactly duplicates some train row. evaluate_dataset() uses the mask to
    report overlap-free metrics alongside the full-split ones.
    """
    features = load_feature_names(ds)
    train_distinct = np.unique(feature_row_hashes(train_path(ds), features))
    test_hashes = feature_row_hashes(test_path(ds), features)
    test_dup_mask = np.isin(test_hashes, train_distinct)

    result = {
        'dataset': ds,
        'n_features': len(features),
        'n_train_distinct_feature_hashes': int(train_distinct.size),
        'n_test_rows': int(test_hashes.size),
        'n_test_rows_matching_train': int(test_dup_mask.sum()),
        'frac_test_rows_matching_train': float(test_dup_mask.sum() / test_hashes.size) if test_hashes.size else 0.0,
    }
    out_path = dataset_dir(ds) / 'train_test_overlap.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    pct = 100.0 * result['frac_test_rows_matching_train']
    log.info(f'train/test overlap: {result["n_test_rows_matching_train"]:,}/{result["n_test_rows"]:,} '
             f'({pct:.2f}%) test rows are exact feature-value duplicates of a train row -> {out_path.name}')
    return result, test_dup_mask


# ── Report ──────────────────────────────────────────────────────────────────────
def _overlap_caveat_line(ds: str) -> str:
    """One-line disclosure of the train/test exact-duplicate-row overlap
    (check_train_test_overlap(), computed earlier in this same step), so the within-year metrics
    printed below are never read without this caveat. This finding previously regressed from
    'measured and known' to 'known only if you already know to look for it'."""
    p = dataset_dir(ds) / 'train_test_overlap.json'
    if not p.exists():
        return ('  [CAVEAT] Train/test row overlap not measured for this run — '
                'check_train_test_overlap() did not run before quoting the within-year metrics '
                'below as pure generalization.')
    try:
        d = json.loads(p.read_text(encoding='utf-8'))
        pct = 100.0 * d.get('frac_test_rows_matching_train', 0.0)
        return (f'  [CAVEAT] {pct:.1f}% of test rows are exact feature-value duplicates of a '
                f'train row (check_train_test_overlap()) — within-year metrics below are '
                f'partly measuring memorization, not generalization, to that extent.')
    except Exception:
        return ''


def write_report(results: list[dict], out_path: Path, log: Logger):
    lines: list[str] = []

    def h(t):
        lines.extend(['', '=' * 70, t, '=' * 70])

    h('TEST REPORT  --  LightGBM RF-mode on CIC-IDS 2017 / 2018 (held-out test split)')
    lines.append(f'Generated : {datetime.now():%Y-%m-%d %H:%M:%S}')

    for r in results:
        if not r:
            continue
        h(f'DATASET: {r["dataset"]}   (test rows: {r["n_test_rows"]:,})')
        lines.append(_overlap_caveat_line(r['dataset']))
        for task, m in r['metrics'].items():
            lines.append('')
            lines.append(f'  [{task}]  accuracy={m["accuracy"]:.4f}  '
                         f'balanced_acc={m["balanced_accuracy"]:.4f}  '
                         f'macro_f1={m["macro_f1"]:.4f}  weighted_f1={m["weighted_f1"]:.4f}  '
                         f'mcc={m["mcc"]:.4f}')
            if 'roc_auc' in m:
                lines.append(f'           roc_auc={m["roc_auc"]:.4f}  pr_auc={m["pr_auc"]:.4f}')
            if 'overlap_free' in m:
                c = m['overlap_free']
                extra = ''
                if 'roc_auc' in c:
                    extra = f'  roc_auc={c["roc_auc"]:.4f}  pr_auc={c["pr_auc"]:.4f}'
                lines.append(f'    [overlap-free subset: {c["n_rows"]:,} rows — test rows that are '
                             'NOT exact feature-duplicates of a train row]')
                lines.append(f'           accuracy={c["accuracy"]:.4f}  '
                             f'balanced_acc={c["balanced_accuracy"]:.4f}  '
                             f'macro_f1={c["macro_f1"]:.4f}  weighted_f1={c["weighted_f1"]:.4f}  '
                             f'mcc={c["mcc"]:.4f}{extra}')
            lines.append(f'    {"class":<14} {"precision":>10} {"recall":>10} {"f1":>10} {"support":>12}')
            lines.append(f'    {"-"*14} {"-"*10} {"-"*10} {"-"*10} {"-"*12}')
            for cname, pc in m['per_class'].items():
                lines.append(f'    {cname:<14} {pc["precision"]:>10.4f} {pc["recall"]:>10.4f} '
                             f'{pc["f1"]:>10.4f} {pc["support"]:>12,}')

    text = '\n'.join(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text)
    log.ok(f'Saved {out_path.name}')
    print(text)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate the trained LightGBM RF-mode models on the held-out CIC-IDS test split.')
    parser.add_argument('--datasets', nargs='+', metavar='NAME', default=list(DATASETS),
                        help=f'Datasets to evaluate (default: {" ".join(DATASETS)})')
    args = parser.parse_args()

    tasks = TASKS

    for ds in args.datasets:
        # Per-dataset results folder: results/6_testing/cicids2017_lightgbm/
        ds_results_dir = testing_results_dir(PROJECT_ROOT, ds, ALGORITHM)
        ds_results_dir.mkdir(parents=True, exist_ok=True)

        # Logs and reports named per consistency standard
        ds_log = Logger(ds_results_dir / Config6.STEPS_FILE,
                        step_prefix=6,
                        title=f'6_TESTING STEPS LOG  [{ds} / {ALGORITHM}]')
        ds_log.info(f'Dataset   : {ds}')
        ds_log.info(f'Engine    : {ALGORITHM} (the pipeline\'s only engine)')
        ds_log.info(f'Tasks     : {tasks}')

        ds_log.section(f'DATASET: {ds}  [engine: {ALGORITHM}]')

        ds_log.step('Check train/test row overlap (full + overlap-free metrics)')
        test_dup_mask = None
        try:
            _, test_dup_mask = check_train_test_overlap(ds, ds_log)
        except Exception as e:
            ds_log.warn(f'check_train_test_overlap failed ({type(e).__name__}: {e}) — '
                        'overlap-free metrics will be unavailable this run')
        ds_log.step_end()

        result = evaluate_dataset(ds, tasks, ds_log, test_dup_mask)

        ds_log.step('Write evaluation report')
        write_report([result], ds_results_dir / Config6.RESULTS_FILE, ds_log)

        ds_log.close()
        if result:
            print(f'✓ {ds} [{ALGORITHM}]: {result["n_test_rows"]:,} test rows evaluated')
        else:
            print(f'✗ {ds} [{ALGORITHM}]: no results (models missing?)')

    # ── Cross-year transfer: train on one year, test on the other (both directions) ──
    # The headline collapse (step-11 ablation reads cross_year_baseline_<algo>.json as the
    # 'all_features' baseline). Needs >= 2 datasets with trained models on both sides.
    # No overlap handling here: train and test are different YEARS, exact-duplicate leakage
    # between them is not the within-year memorization issue FIX-6 addresses.
    if len(args.datasets) >= 2:
        a, b = args.datasets[0], args.datasets[1]
        cross_dir = RESULTS_ROOT / '6_testing' / f'cross_{ALGORITHM}'
        cross_dir.mkdir(parents=True, exist_ok=True)
        x_log = Logger(cross_dir / Config6.STEPS_FILE, step_prefix=6,
                       title=f'6_CROSS-YEAR TRANSFER  [{a} <-> {b} / {ALGORITHM}]')
        x_log.section(f'CROSS-YEAR: {a} <-> {b}  [engine: {ALGORITHM}]')
        cross_results = [
            evaluate_cross(a, b, tasks, x_log),
            evaluate_cross(b, a, tasks, x_log),
        ]
        cross_results = [r for r in cross_results if r]
        if cross_results:
            x_log.step('Write cross-year baseline JSON (consumed by step 11 ablation)')
            baseline = {
                'algorithm': ALGORITHM,
                'generated': datetime.now().isoformat(timespec='seconds'),
                'directions': {r['direction']: r['summary'] for r in cross_results},
                # Prior/threshold recalibration per direction (binary), for step 11.
                'recalibration': {r['direction']: r.get('recalibration', {}) for r in cross_results},
            }
            base_path = OUTPUT_ROOT / '6_testing' / f'cross_year_baseline_{ALGORITHM}.json'
            base_path.parent.mkdir(parents=True, exist_ok=True)
            with open(base_path, 'w', encoding='utf-8') as f:
                json.dump(baseline, f, indent=2)
            x_log.ok(f'Saved {base_path}')
            x_log.step('Write cross-year report')
            write_cross_report(cross_results, cross_dir / '6_cross_year_report.txt', x_log)
            for r in cross_results:
                print(f'✓ cross {r["direction"]} [{ALGORITHM}]: {r["n_test_rows"]:,} rows')
        else:
            x_log.warn('no cross-year results (models or test splits missing)')
        x_log.close()


if __name__ == '__main__':
    main()
