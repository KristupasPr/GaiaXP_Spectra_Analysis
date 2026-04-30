#!/usr/bin/env python3
"""LogReg benchmark on ALL feature variants (basis-only + smoothed).

Mirrors 03_logreg_basis_only.py with full HPO:
  - RandomizedSearchCV (n_iter=50) over C, penalty (l1/l2), class_weight
  - Pipeline: StandardScaler -> LogisticRegression (saga, max_iter=2000)
  - Inner 3-fold stratified CV, scoring=roc_auc
  - Youden threshold on OOF train-set predictions
  - Resume: skips (representation, split) already in the results CSV

Usage:
    python3 05_logreg_smoothed.py                        # default: *_L2.csv
    python3 05_logreg_smoothed.py --pattern '*.csv'      # custom pattern
    python3 05_logreg_smoothed.py --smoke                # first seed only

Outputs (results/):
  - logreg_smoothed_raw.csv      per-fold metrics
  - logreg_smoothed_summary.csv  aggregated stats

Visualize results in 05_logreg_smoothed_results.ipynb.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform
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

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_SPLITS = DATA_DIR / "splits.json"

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

N_ITER_HPO = 50
RESULTS_CSV = RESULTS_DIR / "logreg_smoothed_raw.csv"
SUMMARY_CSV = RESULTS_DIR / "logreg_smoothed_summary.csv"


def load_split_records() -> dict:
    if not LOCAL_SPLITS.exists():
        raise FileNotFoundError(f"Split file not found: {LOCAL_SPLITS}. Run 01_prepare_inputs.py first.")
    with LOCAL_SPLITS.open() as fh:
        return json.load(fh)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LogReg benchmark for all feature variants.")
    parser.add_argument(
        "--pattern",
        default="*_L2.csv",
        help="Glob pattern inside data/features/.",
    )
    parser.add_argument("--smoke", action="store_true", help="Only first seed (quick test)")
    return parser.parse_args()


def pick_youden_threshold(y_true: np.ndarray, y_prob: np.ndarray, grid_size: int = 200) -> float:
    """Threshold in [0, 1] maximising Youden's J = sensitivity + specificity - 1."""
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


def get_oof_probabilities(pipeline: Pipeline, X_tr: np.ndarray, y_tr: np.ndarray,
                          cv: int = 3, n_jobs: int = 1) -> np.ndarray:
    """Out-of-fold probability predictions on training set."""
    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_predict(
        pipeline, X_tr, y_tr, cv=cv_obj,
        method="predict_proba", n_jobs=n_jobs,
    )[:, 1]


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    """Compute all metrics at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)
    f1 = (2 * prec * sens) / (prec + sens) if (prec + sens) else 0.0
    youden = sens + spec - 1.0
    return {
        "threshold": threshold,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "accuracy": acc,
        "f1": f1,
        "youden_j": youden,
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
    }


def parse_feature_stem(stem: str) -> dict:
    """Extract basis, smoothing, n_coeffs from filename like 'bspline_gaussian_20_L2'."""
    parts = stem.replace("_L2", "").split("_")
    basis = parts[0]
    k_str = parts[-1]
    smoothing = "_".join(parts[1:-1])
    return {
        "basis": basis,
        "smoothing": smoothing,
        "n_coeffs_per_arm": int(k_str),
    }


def process_cell(
    X_all: np.ndarray,
    y_all: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    n_iter: int,
) -> tuple[dict, dict]:
    """Run RandomizedSearchCV + Youden evaluation for one split."""
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    X_tr, y_tr = X_all[train_idx], y_all[train_idx]
    X_te, y_te = X_all[test_idx], y_all[test_idx]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])
    param_dist = {
        "clf__C": loguniform(1e-3, 1e3),
        "clf__penalty": ["l1", "l2"],
        "clf__solver": ["saga"],
        "clf__class_weight": [None, "balanced"],
    }

    search = RandomizedSearchCV(
        pipeline,
        param_dist,
        n_iter=n_iter,
        cv=inner_cv,
        scoring="roc_auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        error_score="raise",
    )
    search.fit(X_tr, y_tr)
    best_pipe = search.best_estimator_
    best_cv_score = search.best_score_

    best_params = {
        k.replace("clf__", ""): v
        for k, v in search.best_params_.items()
    }

    y_prob_oof = get_oof_probabilities(best_pipe, X_tr, y_tr, cv=3, n_jobs=-1)
    thr = pick_youden_threshold(y_tr, y_prob_oof)

    y_prob_te = best_pipe.predict_proba(X_te)[:, 1]
    metrics = evaluate(y_te, y_prob_te, thr)
    metrics["best_cv_roc_auc"] = best_cv_score

    return metrics, best_params


def main() -> None:
    args = parse_args()
    n_iter = 5 if args.smoke else N_ITER_HPO

    splits_dict = load_split_records()
    split_names = sorted(splits_dict.keys())
    if args.smoke:
        split_names = split_names[:1]
        print(f"SMOKE: using 1/{len(splits_dict)} splits, {n_iter} HPO iters")

    files = sorted(FEATURES_DIR.glob(args.pattern))
    files = [f for f in files if not f.name.startswith("fourier_")]
    if not files:
        raise FileNotFoundError(f"No files matched {args.pattern!r} in {FEATURES_DIR}")

    total_cells = len(files) * len(split_names)
    print(f"Found {len(files)} feature files (fourier excluded), {len(split_names)} splits")
    print(f"Total cells: {total_cells}")

    completed: set[tuple[str, str]] = set()
    existing_rows: list[dict] = []
    csv_columns: list[str] | None = None
    if RESULTS_CSV.exists():
        df_old = pd.read_csv(RESULTS_CSV)
        existing_rows = df_old.to_dict("records")
        csv_columns = list(df_old.columns)
        for r in existing_rows:
            completed.add((str(r["representation"]), str(r["split"])))
        print(f"Resume: {len(completed)} cells done, will skip them.")

    all_results = list(existing_rows)
    done_count = len(completed)
    t_start = time.time()

    for file_i, path in enumerate(files, 1):
        stem = path.stem
        meta = parse_feature_stem(stem)

        df = pd.read_csv(path)
        coeff_cols = [c for c in df.columns if c.startswith("c")]
        X_all = df[coeff_cols].to_numpy(dtype=np.float64)
        y_all = df["y"].astype(int).to_numpy()
        n_features = X_all.shape[1]

        pending = [s for s in split_names if (stem, s) not in completed]
        if not pending:
            continue

        for sname in pending:
            split = splits_dict[sname]
            train_idx = np.asarray(split["train"], dtype=int)
            test_idx = np.asarray(split["test"], dtype=int)

            t0 = time.time()
            metrics, best_params = process_cell(X_all, y_all, train_idx, test_idx, n_iter)
            cell_time = time.time() - t0

            metrics.update({
                "representation": stem,
                "basis": meta["basis"],
                "smoothing": meta["smoothing"],
                "n_coeffs_per_arm": meta["n_coeffs_per_arm"],
                "n_features": n_features,
                "split": sname,
                "time_seconds": cell_time,
                "best_C": best_params.get("C"),
                "best_penalty": best_params.get("penalty"),
                "best_class_weight": best_params.get("class_weight"),
            })
            all_results.append(metrics)
            done_count += 1

            if csv_columns is None:
                csv_columns = list(metrics.keys())
                if not RESULTS_CSV.exists() or RESULTS_CSV.stat().st_size == 0:
                    with open(RESULTS_CSV, "w", newline="") as f:
                        csv.DictWriter(f, fieldnames=csv_columns).writeheader()
            with open(RESULTS_CSV, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=csv_columns).writerow(metrics)

            elapsed = time.time() - t_start
            cells_this_run = done_count - len(completed)
            remaining = total_cells - done_count
            eta = (elapsed / cells_this_run) * remaining if cells_this_run > 0 else 0

            print(
                f"[{done_count:4d}/{total_cells}]  {stem}  {sname}  "
                f"ROC-AUC={metrics['roc_auc']:.4f}  "
                f"PR-AUC={metrics['pr_auc']:.4f}  "
                f"Sens={metrics['sensitivity']:.4f}  "
                f"Prec={metrics['precision']:.4f}  "
                f"best_C={metrics['best_C']:.4g}  {metrics['best_penalty']}  "
                f"({cell_time:.1f}s)  ETA={eta / 60:.1f}min",
                flush=True,
            )

    if not all_results:
        print("No results produced.")
        return

    df_all = pd.DataFrame(all_results)
    df_all.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved {len(df_all)} rows -> {RESULTS_CSV}")

    metric_cols = [
        "roc_auc", "pr_auc", "youden_j", "f1",
        "sensitivity", "specificity", "precision", "accuracy",
        "brier", "log_loss",
    ]
    group_cols = ["representation", "basis", "smoothing", "n_coeffs_per_arm", "n_features"]
    available_metrics = [c for c in metric_cols if c in df_all.columns]

    summary = (
        df_all.groupby(group_cols, as_index=False)[available_metrics]
        .agg(["mean", "std"])
    )
    summary.columns = [
        col if stat == "" else f"{col}_{stat}"
        for col, stat in summary.columns.to_flat_index()
    ]
    summary = summary.reset_index().sort_values("roc_auc_mean", ascending=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved summary -> {SUMMARY_CSV}")

    print("\n── Summary (top 10 by ROC-AUC) ──")
    for _, row in summary.head(10).iterrows():
        print(
            f"  {row['representation']:30s}  "
            f"ROC-AUC={row['roc_auc_mean']:.4f}±{row['roc_auc_std']:.4f}  "
            f"PR-AUC={row['pr_auc_mean']:.4f}±{row['pr_auc_std']:.4f}"
        )

    elapsed_total = time.time() - t_start
    print(f"\nFinished in {elapsed_total / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
