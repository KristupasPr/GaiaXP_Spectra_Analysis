#!/usr/bin/env python
"""
14 — ROCKET / MiniROCKET + linear classifiers on L2-normalised spectra (RSKF)

Same inputs as 12_fpca_classifier.py. ROCKET / MiniROCKET (sktime) features →
StandardScaler → classifier (tuned on training indices only). PR/ROC-AUC from
decision_function; F1, sensitivity, precision, etc. use a threshold on
train-normalised scores: --threshold youden | f1.

Classifiers (--clf): ridge (RidgeClassifierCV), logreg (LogisticRegressionCV on C),
linear_svc (GridSearchCV on C for LinearSVC, dual=False). All use train-only CV
where applicable.

Resume key: (method, split_name, n_kernels, classifier, threshold_policy).

Usage:
  python 14_rocket_poc.py --method minirocket --clf logreg --threshold f1
  python 14_rocket_poc.py --method both --clf ridge --threshold youden

Outputs:
  - results/rocket_rskf_metrics.csv
  - results/rocket_rskf_summary.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

try:
    from sktime.transformations.panel.rocket import MiniRocket, Rocket
except ImportError as e:
    raise SystemExit(
        "Missing sktime. Use the project venv, e.g. from transformation_experiment/:\n"
        "  source ../.venv/bin/activate && pip install -r ../requirements.txt\n"
        "Or: ../.venv/bin/python 14_rocket_poc.py ...\n"
    ) from e

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DEFAULT_N_KERNELS_ROCKET = 10_000
DEFAULT_N_KERNELS_MINI = 10_000
KERNEL_COUNTS_CONVERGENCE = [100, 500, 1_000, 5_000, 10_000]

METRICS_CSV = "rocket_rskf_metrics.csv"
SUMMARY_CSV = "rocket_rskf_summary.csv"

parser = argparse.ArgumentParser(description="ROCKET / MiniROCKET on L2 Gaia XP spectra")
parser.add_argument("--smoke", action="store_true", help="Use rep0 folds only (5 splits)")
parser.add_argument(
    "--method",
    choices=["rocket", "minirocket", "both"],
    default="both",
    help="Which transform to run (default: both)",
)
parser.add_argument(
    "--n-kernels",
    type=int,
    default=DEFAULT_N_KERNELS_ROCKET,
    help=f"num_kernels for ROCKET (default {DEFAULT_N_KERNELS_ROCKET})",
)
parser.add_argument(
    "--minirocket-kernels",
    type=int,
    default=DEFAULT_N_KERNELS_MINI,
    help=f"num_kernels for MiniROCKET (default {DEFAULT_N_KERNELS_MINI})",
)
parser.add_argument(
    "--convergence",
    action="store_true",
    help="Run ROCKET with kernel counts "
    + str(KERNEL_COUNTS_CONVERGENCE)
    + " (use with --method rocket or both)",
)
parser.add_argument(
    "--n-jobs",
    type=int,
    default=-1,
    help="n_jobs for sktime Rocket/MiniRocket (-1 = all cores)",
)
parser.add_argument(
    "--clf",
    choices=["ridge", "logreg", "linear_svc"],
    default="ridge",
    help="Classifier on ROCKET features (default ridge). Each choice uses train-only CV for strength/C where applicable.",
)
parser.add_argument(
    "--threshold",
    choices=["youden", "f1"],
    default="youden",
    help="How to pick score threshold on train (normalised decision scores): youden = max sens+spec-1; f1 = max F1.",
)
args = parser.parse_args()

EXPERIMENT_DIR = Path.cwd() if Path("data").exists() else Path("transformation_experiment")
DATA_DIR = EXPERIMENT_DIR / "data"
RESULTS_DIR = EXPERIMENT_DIR / "results"
POC_DIR = Path("transformation_poc") if Path("transformation_poc").exists() else Path("..") / "transformation_poc"
RESULTS_DIR.mkdir(exist_ok=True)
METRICS_PATH = RESULTS_DIR / METRICS_CSV
SUMMARY_PATH = RESULTS_DIR / SUMMARY_CSV

print("Data dir:   ", DATA_DIR.resolve())
print("Results dir:", RESULTS_DIR.resolve())
print("POC dir:    ", POC_DIR.resolve())
print(f"Classifier: {args.clf}  |  Train threshold policy: {args.threshold}")


def to_sktime_panel(X_2d: np.ndarray) -> np.ndarray:
    return X_2d[:, np.newaxis, :]


def normalize_scores_train_ref(scores_te: np.ndarray, scores_tr: np.ndarray) -> np.ndarray:
    lo, hi = float(scores_tr.min()), float(scores_tr.max())
    if hi == lo:
        return np.full_like(scores_te, 0.5, dtype=np.float64)
    return ((scores_te - lo) / (hi - lo)).astype(np.float64)


def pick_youden_threshold(y_true: np.ndarray, y_prob: np.ndarray, grid_size: int = 200) -> float:
    thresholds = np.linspace(0, 1, grid_size)
    best_j, best_thr = -1.0, 0.5
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        j = sens + spec - 1.0
        if j > best_j:
            best_j, best_thr = j, float(thr)
    return best_thr


def pick_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray, grid_size: int = 200) -> float:
    thresholds = np.linspace(0, 1, grid_size)
    best_f1, best_thr = -1.0, 0.5
    z = 0
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        f1v = f1_score(y_true, y_pred, pos_label=1, zero_division=z)
        if f1v > best_f1:
            best_f1, best_thr = f1v, float(thr)
    return best_thr


def fold_metrics(
    y_true_te: np.ndarray,
    y_score_te: np.ndarray,
    y_true_tr: np.ndarray,
    y_score_tr: np.ndarray,
    threshold_policy: str,
) -> dict:
    out: dict = {"pr_auc": average_precision_score(y_true_te, y_score_te)}
    try:
        out["roc_auc"] = float(roc_auc_score(y_true_te, y_score_te))
    except ValueError:
        out["roc_auc"] = np.nan

    prob_tr = normalize_scores_train_ref(y_score_tr, y_score_tr)
    prob_te = normalize_scores_train_ref(y_score_te, y_score_tr)
    if threshold_policy == "youden":
        thr = pick_youden_threshold(y_true_tr, prob_tr)
    elif threshold_policy == "f1":
        thr = pick_f1_threshold(y_true_tr, prob_tr)
    else:
        raise ValueError(threshold_policy)
    y_pred = (prob_te >= thr).astype(np.int64)
    out["youden_threshold"] = thr
    out["threshold_policy"] = threshold_policy

    z = 0
    out["sensitivity"] = recall_score(y_true_te, y_pred, pos_label=1, zero_division=z)
    out["precision"] = precision_score(y_true_te, y_pred, pos_label=1, zero_division=z)
    out["specificity"] = recall_score(y_true_te, y_pred, pos_label=0, zero_division=z)
    out["accuracy"] = accuracy_score(y_true_te, y_pred)
    out["f1"] = f1_score(y_true_te, y_pred, pos_label=1, zero_division=z)
    tn, fp, fn, tp = confusion_matrix(y_true_te, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    out["youden_j"] = sens + spec - 1.0
    out["tp"], out["fp"], out["fn"], out["tn"] = int(tp), int(fp), int(fn), int(tn)
    return out


def _to_numpy_2d(Xt) -> np.ndarray:
    if hasattr(Xt, "values"):
        return np.asarray(Xt.values, dtype=np.float64)
    return np.asarray(Xt, dtype=np.float64)


def build_transformer(method: str, n_kernels: int, random_state: int, n_jobs: int):
    if method == "rocket":
        return Rocket(num_kernels=n_kernels, random_state=random_state, n_jobs=n_jobs)
    if method == "minirocket":
        return MiniRocket(num_kernels=n_kernels, random_state=random_state, n_jobs=n_jobs)
    raise ValueError(method)


def fit_classifier(
    clf_kind: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    random_state: int,
    clf_n_jobs: int,
):
    """Fit classifier on scaled train features; hyperparameters chosen by CV on train only."""
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    meta: dict = {"ridge_alpha": np.nan, "clf_C": np.nan}

    if clf_kind == "ridge":
        clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), class_weight="balanced")
        clf.fit(X_tr, y_tr)
        meta["ridge_alpha"] = float(clf.alpha_)
        return clf, meta

    if clf_kind == "logreg":
        clf = LogisticRegressionCV(
            Cs=10,
            cv=inner_cv,
            scoring="average_precision",
            class_weight="balanced",
            max_iter=3000,
            solver="saga",
            penalty="l2",
            n_jobs=clf_n_jobs,
            random_state=random_state,
        )
        clf.fit(X_tr, y_tr)
        meta["clf_C"] = float(np.asarray(clf.C_).ravel()[0])
        return clf, meta

    if clf_kind == "linear_svc":
        base = LinearSVC(
            class_weight="balanced",
            dual=False,
            max_iter=10_000,
            random_state=random_state,
        )
        grid = {"C": np.logspace(-2, 2, 5)}
        gscv = GridSearchCV(
            base,
            grid,
            cv=inner_cv,
            scoring="average_precision",
            n_jobs=clf_n_jobs,
            refit=True,
        )
        gscv.fit(X_tr, y_tr)
        meta["clf_C"] = float(gscv.best_params_["C"])
        return gscv, meta

    raise ValueError(clf_kind)


def run_one_fold(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    method: str,
    n_kernels: int,
    random_state: int,
    n_jobs: int,
    clf_kind: str,
    threshold_policy: str,
    clf_n_jobs: int,
) -> dict:
    t0 = time.time()
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    X_train_3d = to_sktime_panel(X_train)
    X_test_3d = to_sktime_panel(X_test)

    transformer = build_transformer(method, n_kernels, random_state, n_jobs)
    X_tr_f = _to_numpy_2d(transformer.fit_transform(X_train_3d))
    X_te_f = _to_numpy_2d(transformer.transform(X_test_3d))
    n_features = X_tr_f.shape[1]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_f)
    X_te_s = scaler.transform(X_te_f)

    clf, cmeta = fit_classifier(clf_kind, X_tr_s, y_train, random_state, clf_n_jobs)

    y_score_tr = clf.decision_function(X_tr_s)
    y_score_te = clf.decision_function(X_te_s)

    m = fold_metrics(y_test, y_score_te, y_train, y_score_tr, threshold_policy)
    m["classifier"] = clf_kind
    m["ridge_alpha"] = cmeta["ridge_alpha"]
    m["clf_C"] = cmeta["clf_C"]
    m["alpha_best"] = m["ridge_alpha"]
    m["n_features"] = int(n_features)
    m["n_train"] = int(len(train_idx))
    m["n_test"] = int(len(test_idx))
    m["time_s"] = time.time() - t0
    return m


def save_metrics_atomic(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    fd, tmp = tempfile.mkstemp(suffix=".csv", dir=str(RESULTS_DIR))
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, METRICS_PATH)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def write_summary(rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    metric_cols = [
        "pr_auc",
        "roc_auc",
        "f1",
        "sensitivity",
        "precision",
        "specificity",
        "youden_j",
        "time_s",
    ]
    gcols = ["method", "n_kernels"]
    if "classifier" in df.columns and "threshold_policy" in df.columns:
        gcols = ["method", "n_kernels", "classifier", "threshold_policy"]
    agg = df.groupby(gcols, dropna=False)[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    agg.reset_index().to_csv(SUMMARY_PATH, index=False)


df_og = pd.read_csv(DATA_DIR / "og_xp.csv")
df_spec = pd.read_csv(POC_DIR / "xp_sampled_spectra.csv")
wl_cols = [c for c in df_spec.columns if c.startswith("wl_")]

df_m = df_og[["source_id", "y"]].merge(
    df_spec[["source_id"] + wl_cols], on="source_id", how="inner", validate="one_to_one"
)
assert len(df_m) == len(df_og) == len(df_spec), "Row alignment failed"

y = df_m["y"].to_numpy(dtype=np.int64)
F_raw = df_m[wl_cols].to_numpy(dtype=np.float64)
norms = np.linalg.norm(F_raw, axis=1, keepdims=True)
norms = np.maximum(norms, 1e-15)
X = F_raw / norms

print(f"N = {len(y)}, p = {X.shape[1]} bins; positive rate = {y.mean():.1%}")

with open(DATA_DIR / "splits_rskf.json") as f:
    splits = json.load(f)
if args.smoke:
    splits = {k: v for k, v in splits.items() if k.startswith("rep0_")}
    print(f"SMOKE: {len(splits)} splits")
else:
    print(f"Loaded {len(splits)} splits")


def split_sort_key(k: str):
    rep = int(k.split("_")[0].replace("rep", ""))
    fold = int(k.split("_")[1].replace("fold", ""))
    return (rep, fold)


split_names = sorted(splits.keys(), key=split_sort_key)

completed: set[tuple[str, str, int, str, str]] = set()
all_results: list[dict] = []
if METRICS_PATH.exists():
    df_prev = pd.read_csv(METRICS_PATH)
    for _, row in df_prev.iterrows():
        nk = row["n_kernels"]
        nk_int = int(nk) if pd.notna(nk) else -1
        clf_r = row["classifier"] if "classifier" in row.index and pd.notna(row.get("classifier")) else "ridge"
        pol_r = (
            row["threshold_policy"]
            if "threshold_policy" in row.index and pd.notna(row.get("threshold_policy"))
            else "youden"
        )
        completed.add((str(row["method"]), str(row["split_name"]), nk_int, str(clf_r), str(pol_r)))
    all_results = df_prev.to_dict("records")
    print(f"Resume: {len(completed)} fold-rows done, will skip those keys.")

jobs: list[tuple[str, int]] = []
if args.convergence:
    if args.method in ("rocket", "both"):
        for nk in KERNEL_COUNTS_CONVERGENCE:
            jobs.append(("rocket", nk))
    if args.method in ("minirocket", "both"):
        jobs.append(("minirocket", int(args.minirocket_kernels)))
    if args.method == "minirocket" and not jobs:
        print("ERROR: --convergence is for ROCKET; with --method minirocket omit --convergence.", file=sys.stderr)
        sys.exit(1)
else:
    if args.method in ("rocket", "both"):
        jobs.append(("rocket", int(args.n_kernels)))
    if args.method in ("minirocket", "both"):
        jobs.append(("minirocket", int(args.minirocket_kernels)))

if not jobs:
    print("ERROR: no jobs to run.", file=sys.stderr)
    sys.exit(1)

print(f"Job configs: {jobs}")
pending_count = sum(
    1
    for method, nk in jobs
    for sname in split_names
    if (method, sname, nk, args.clf, args.threshold) not in completed
)
print(f"Pending fold-runs: {pending_count}")

t_start = time.time()
done_this_session = 0

for method, n_kernels in jobs:
    for split_name in split_names:
        key = (method, split_name, n_kernels, args.clf, args.threshold)
        if key in completed:
            continue

        tr = np.array(splits[split_name]["train"], dtype=int)
        te = np.array(splits[split_name]["test"], dtype=int)

        metrics = run_one_fold(
            X,
            y,
            tr,
            te,
            method=method,
            n_kernels=n_kernels,
            random_state=RANDOM_STATE,
            n_jobs=args.n_jobs,
            clf_kind=args.clf,
            threshold_policy=args.threshold,
            clf_n_jobs=args.n_jobs,
        )
        row = {"split_name": split_name, "method": method, "n_kernels": n_kernels, **metrics}
        all_results.append(row)
        completed.add(key)
        done_this_session += 1

        save_metrics_atomic(all_results)
        write_summary(all_results)

        remaining = pending_count - done_this_session
        elapsed = time.time() - t_start
        eta = (elapsed / done_this_session) * remaining if done_this_session and remaining > 0 else 0.0
        print(
            f"  [{method} nk={n_kernels} {args.clf} thr={args.threshold}] {split_name}  "
            f"PR-AUC={metrics['pr_auc']:.4f}  F1={metrics['f1']:.4f}  "
            f"Sens={metrics['sensitivity']:.4f}  Prec={metrics['precision']:.4f}  "
            f"({metrics['time_s']:.1f}s)  ETA~{eta/60:.1f}min",
            flush=True,
        )

print(f"\nFinished in {(time.time() - t_start)/60:.2f} min (this session).")
print(f"Wrote {METRICS_PATH} ({len(all_results)} rows)")
print(f"Summary → {SUMMARY_PATH}")
