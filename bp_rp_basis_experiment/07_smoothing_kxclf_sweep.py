#!/usr/bin/env python3
"""Smoothing x K x Classifier sweep.

Tests the hypothesis: 'explicit Gaussian smoothing has measurable effect on
classification only when K is large enough for the basis to encode
high-frequency content.'

Grid:
    K          : [10, 50]     (20 or 100 total features)
    basis      : chebyshev, legendre, bspline
    sigma      : 0 (none), 0.5, 1, 2, 3, 5, 10, 20
    classifier : LogisticRegression, XGBoost
    splits     : 50-fold RSKF (10 repeats x 5 folds)

Sigma is in raw-flux pixel units (BP ~2.2 nm/pixel, RP ~2.0 nm/pixel).
Smoothing is applied to the raw flux *before* basis fitting, so sigma
meaning is K-independent.

Note: inner-CV HPO uses scoring='roc_auc' while the headline metric is
PR-AUC. This matches the protocol in 05_classify_focused.py / 06 notebook.

Outputs:
    results/smoothing_kxclf_raw.csv       -- one row per cell
    results/smoothing_kxclf_summary.csv   -- mean/std aggregated over splits

Usage:
    python 07_smoothing_kxclf_sweep.py               # full run (~10-11 h)
    python 07_smoothing_kxclf_sweep.py --smoke        # rep0 only (5 folds)
    python 07_smoothing_kxclf_sweep.py --clf LR       # LR only
    python 07_smoothing_kxclf_sweep.py --clf XGB      # XGBoost only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from importlib.util import module_from_spec, spec_from_file_location
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*lbfgs.*")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", message=".*max_iter was reached.*")

ROOT = Path(__file__).resolve().parent
sys.modules.pop("bp_basis_step02", None)
_spec = spec_from_file_location("bp_basis_step02", ROOT / "02_generate_basis_features.py")
step02 = module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = step02
_spec.loader.exec_module(step02)

from _common import (  # noqa: E402
    BP_SAMPLED_CSV,
    DATA_DIR,
    RESULTS_DIR,
    RP_SAMPLED_CSV,
    flatten_feature_blocks,
    l2_normalize,
)

RANDOM_STATE = 42
N_JOBS = 8
BASES = ["chebyshev", "legendre", "bspline"]
SIGMAS = [0, 0.5, 1, 2, 3, 5, 10, 20]
K_VALUES = [10, 50]

RAW_CSV = RESULTS_DIR / "smoothing_kxclf_raw.csv"
SUMMARY_CSV = RESULTS_DIR / "smoothing_kxclf_summary.csv"

METRIC_COLS = [
    "roc_auc", "pr_auc", "f1", "sensitivity", "precision",
    "specificity", "accuracy", "youden_j", "brier", "log_loss",
]
GROUP_COLS = ["K", "basis", "sigma", "classifier"]


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true", help="Use rep0 folds only (5 splits)")
    p.add_argument("--clf", nargs="+", default=["LR", "XGB"], choices=["LR", "XGB"],
                   help="Which classifiers to run (default: both)")
    p.add_argument("--k-values", nargs="+", type=int, default=K_VALUES,
                   help="K values to sweep (default: 10 50)")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# Feature generation
# ═══════════════════════════════════════════════════════════════════════

def generate_features(bp, rp, basis: str, K: int, sigma: float):
    """Smooth -> fit basis -> L2-normalise, return (X, y) arrays."""
    smoothing = "none" if sigma == 0 else "gaussian"
    smooth_kwargs = {} if sigma == 0 else {"sigma": sigma}

    bp_fit = step02.build_block_fit(bp, basis, smoothing, K, **smooth_kwargs)
    rp_fit = step02.build_block_fit(rp, basis, smoothing, K, **smooth_kwargs)
    feat_df = flatten_feature_blocks(bp.source_ids, bp.labels, bp_fit.coeffs, rp_fit.coeffs)
    coeff_cols = [c for c in feat_df.columns if c.startswith("c")]
    feat_df = l2_normalize(feat_df, coeff_cols=coeff_cols)
    X = feat_df[coeff_cols].to_numpy(dtype=np.float64)
    y = feat_df["y"].astype(int).to_numpy()
    return X, y


# ═══════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ═══════════════════════════════════════════════════════════════════════

def pick_youden_threshold(y_true, y_prob, grid_size=200):
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


def evaluate(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)
    f1 = (2 * prec * sens) / (prec + sens) if (prec + sens) else 0.0
    return {
        "threshold": threshold,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "accuracy": acc,
        "f1": f1,
        "youden_j": sens + spec - 1.0,
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
    }


# ═══════════════════════════════════════════════════════════════════════
# Classifier runners
# ═══════════════════════════════════════════════════════════════════════

def run_lr(X_tr, y_tr, X_te, y_te):
    """Logistic regression with RandomizedSearchCV (n_iter=30)."""
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
    ])
    param_dist = {
        "clf__C": loguniform(1e-3, 1e3),
        "clf__penalty": ["l1", "l2"],
        "clf__solver": ["saga"],
        "clf__class_weight": [None, "balanced"],
    }
    search = RandomizedSearchCV(
        pipeline, param_dist, n_iter=30, cv=inner_cv,
        scoring="roc_auc", random_state=RANDOM_STATE, n_jobs=N_JOBS,
        error_score="raise",
    )
    search.fit(X_tr, y_tr)
    best_pipe = search.best_estimator_

    oof_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    y_prob_oof = cross_val_predict(
        best_pipe, X_tr, y_tr, cv=oof_cv, method="predict_proba", n_jobs=N_JOBS,
    )[:, 1]
    thr = pick_youden_threshold(y_tr, y_prob_oof)

    y_prob_te = best_pipe.predict_proba(X_te)[:, 1]
    metrics = evaluate(y_te, y_prob_te, thr)
    metrics["best_cv_roc_auc"] = search.best_score_
    best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    metrics["best_params"] = json.dumps({k: _json_safe(v) for k, v in best_params.items()})
    return metrics


def run_xgb(X_tr, y_tr, X_te, y_te):
    """XGBoost with RandomizedSearchCV (n_iter=20).

    XGBoost is sensitive to feature-space geometry (unlike RF which splits
    on thresholds), making it appropriate for detecting whether smoothing
    changes the basis coefficient distribution.
    """
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    pipeline = Pipeline([
        ("clf", XGBClassifier(
            eval_metric="logloss", random_state=RANDOM_STATE,
            n_jobs=1, verbosity=0,
        )),
    ])
    param_dist = {
        "clf__n_estimators": [100, 300, 500],
        "clf__max_depth": [3, 5, 7, 10],
        "clf__learning_rate": loguniform(0.01, 0.3),
        "clf__subsample": uniform(0.6, 0.4),
        "clf__colsample_bytree": uniform(0.5, 0.5),
        "clf__scale_pos_weight": [1, 3, 4],
    }
    search = RandomizedSearchCV(
        pipeline, param_dist, n_iter=20, cv=inner_cv,
        scoring="roc_auc", random_state=RANDOM_STATE, n_jobs=N_JOBS,
        error_score="raise",
    )
    search.fit(X_tr, y_tr)
    best_pipe = search.best_estimator_

    oof_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    y_prob_oof = cross_val_predict(
        best_pipe, X_tr, y_tr, cv=oof_cv, method="predict_proba", n_jobs=N_JOBS,
    )[:, 1]
    thr = pick_youden_threshold(y_tr, y_prob_oof)

    y_prob_te = best_pipe.predict_proba(X_te)[:, 1]
    metrics = evaluate(y_te, y_prob_te, thr)
    metrics["best_cv_roc_auc"] = search.best_score_
    best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    metrics["best_params"] = json.dumps({k: _json_safe(v) for k, v in best_params.items()})
    return metrics


def _json_safe(v):
    if isinstance(v, (np.integer, np.int64)):
        return int(v)
    if isinstance(v, (np.floating, np.float64)):
        return round(float(v), 6)
    return v


CLF_RUNNERS = {
    "LR": run_lr,
    "XGB": run_xgb,
}


# ═══════════════════════════════════════════════════════════════════════
# Resume support
# ═══════════════════════════════════════════════════════════════════════

def load_completed(raw_path: Path) -> set[tuple]:
    """Return set of (K, basis, sigma, classifier, split) already done."""
    if not raw_path.exists():
        return set()
    df = pd.read_csv(raw_path)
    return set(
        df[["K", "basis", "sigma", "classifier", "split"]].itertuples(index=False, name=None)
    )


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    clf_names = [c.upper() for c in args.clf]
    k_values = args.k_values

    print("=" * 70)
    print("  07 - Smoothing x K x Classifier sweep")
    print("=" * 70)

    # Load spectra
    bp = step02.load_block(BP_SAMPLED_CSV)
    rp = step02.load_block(RP_SAMPLED_CSV)
    step02.check_alignment(bp, rp)
    print(f"BP shape: {bp.flux.shape} (~{bp.flux.shape[1] * 2.2:.0f} nm span, "
          f"{bp.flux.shape[1]} pixels -> sigma in ~2.2 nm/pixel units)")
    print(f"RP shape: {rp.flux.shape} (~{rp.flux.shape[1] * 2.0:.0f} nm span, "
          f"{rp.flux.shape[1]} pixels -> sigma in ~2.0 nm/pixel units)")

    # Load splits
    splits_path = DATA_DIR / "splits_rskf.json"
    if not splits_path.exists():
        raise FileNotFoundError(
            f"Missing {splits_path}. Copy from transformation_experiment/data/splits_rskf.json."
        )
    with splits_path.open() as fh:
        splits_dict = json.load(fh)

    if args.smoke:
        splits_dict = {k: v for k, v in splits_dict.items() if k.startswith("rep0_")}
        print(f"SMOKE MODE: {len(splits_dict)} splits (rep0 only)")
    else:
        print(f"Loaded {len(splits_dict)} splits (10 repeats x 5 folds)")

    split_names = sorted(splits_dict.keys())

    # Resume
    completed = load_completed(RAW_CSV)
    print(f"Already completed: {len(completed)} cells")

    # Build work list
    work = []
    for K in k_values:
        for sigma in SIGMAS:
            for basis in BASES:
                for clf_name in clf_names:
                    for sname in split_names:
                        key = (K, basis, sigma, clf_name, sname)
                        if key not in completed:
                            work.append(key)

    total = len(work)
    print(f"Remaining work: {total} cells")
    print(f"Grid: K={k_values}, bases={BASES}, sigmas={SIGMAS}, clf={clf_names}")
    print(f"Splits: {len(split_names)}")
    print()

    if total == 0:
        print("Nothing to do.")
        _write_summary()
        return

    csv_header_written = RAW_CSV.exists() and RAW_CSV.stat().st_size > 0

    done = 0
    t_start = time.time()
    clf_times: dict[str, list[float]] = {c: [] for c in clf_names}
    clf_total: dict[str, int] = {c: sum(1 for w in work if w[3] == c) for c in clf_names}

    # Group work by (K, sigma, basis) to reuse features across clf/splits
    work.sort(key=lambda x: (x[0], x[2], x[1]))
    for (K, sigma, basis), group_iter in groupby(work, key=lambda x: (x[0], x[2], x[1])):
        group = list(group_iter)
        t_feat = time.time()
        X, y = generate_features(bp, rp, basis, K, sigma)
        feat_seconds = time.time() - t_feat
        print(f"  >> features: K={K} {basis} sigma={sigma} -> "
              f"{X.shape[1]}D in {feat_seconds:.1f}s", flush=True)

        for (_, _, _, clf_name, sname) in group:
            split = splits_dict[sname]
            train_idx = np.asarray(split["train"], dtype=int)
            test_idx = np.asarray(split["test"], dtype=int)
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te, y_te = X[test_idx], y[test_idx]

            t_cell = time.time()
            runner = CLF_RUNNERS[clf_name]
            metrics = runner(X_tr, y_tr, X_te, y_te)
            cell_seconds = time.time() - t_cell
            clf_times[clf_name].append(cell_seconds)

            row = {
                "K": K,
                "basis": basis,
                "sigma": sigma,
                "classifier": clf_name,
                "split": sname,
                **metrics,
            }

            row_df = pd.DataFrame([row])
            row_df.to_csv(RAW_CSV, mode="a", header=not csv_header_written, index=False)
            csv_header_written = True

            done += 1
            elapsed = time.time() - t_start

            remaining_by_clf = {}
            for c in clf_names:
                n_done_c = len(clf_times[c])
                if n_done_c > 0:
                    avg_c = np.mean(clf_times[c])
                    remaining_by_clf[c] = avg_c * (clf_total[c] - n_done_c)
                else:
                    remaining_by_clf[c] = 0
            eta = sum(remaining_by_clf.values())

            print(
                f"  [{done}/{total}] K={K:2d} {basis:10s} sigma={sigma:5.1f} "
                f"{clf_name:3s} {sname:12s}  "
                f"PR-AUC={metrics['pr_auc']:.4f}  "
                f"{cell_seconds:.1f}s  "
                f"({elapsed:.0f}s elapsed, ~{eta:.0f}s left)",
                flush=True,
            )

    elapsed_total = time.time() - t_start
    print(f"\nFinished {done} cells in {elapsed_total / 60:.1f} minutes.")
    for c in clf_names:
        if clf_times[c]:
            print(f"  {c}: {len(clf_times[c])} cells, "
                  f"avg {np.mean(clf_times[c]):.1f}s/cell")
    _write_summary()


def _write_summary():
    """Aggregate raw results into summary CSV."""
    if not RAW_CSV.exists():
        return
    df = pd.read_csv(RAW_CSV)
    agg = {}
    for col in METRIC_COLS:
        if col in df.columns:
            agg[col] = ["mean", "std"]
    summary = (
        df.groupby(GROUP_COLS, sort=True)
        .agg(agg)
    )
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Summary written to {SUMMARY_CSV} ({len(summary)} rows)")


if __name__ == "__main__":
    main()
