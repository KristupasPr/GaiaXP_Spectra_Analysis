#!/usr/bin/env python
"""
05 — Focused Classification Experiment (Chebyshev & Legendre)

Evaluates Chebyshev and Legendre L2-normalised polynomial representations
at n_coeffs = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50] plus the og_xp_110
baseline, using 4 classifiers and 10×5 Repeated Stratified K-Fold.

HPO strategy:
  - Optuna (TPE + MedianPruner) for SVM_RBF, XGBoost, RandomForest
  - RandomizedSearchCV for LogisticRegression

Parallelism strategy (12-core machine):
  - SVM_RBF:           4 splits in parallel × 3 CV folds = 12 threads
  - LogisticRegression: 1 split  (RandomizedSearchCV n_jobs=-1 uses all cores)
  - RandomForest:       1 split  (estimator n_jobs=-1 uses all cores)
  - XGBoost:            1 split  (estimator n_jobs=-1 uses all cores)

Resume: skips (representation, classifier, split) already in the results CSV.

Outputs:
  - results/focused_experiment_results.csv
  - results/focused_summary.csv
  - models_focused/*.joblib + model_manifest.json
"""

import argparse
import csv
import json
import os
import time
import warnings
from pathlib import Path

parser = argparse.ArgumentParser(description="Focused classification experiment")
parser.add_argument("--smoke", action="store_true",
                    help="Quick smoke test: 2 reprs, 5 splits, 5 HPO trials")
parser.add_argument("--only", nargs="+", default=None,
                    help="Run only these representations, e.g. --only legendre_45_L2 legendre_50_L2")
args = parser.parse_args()
SMOKE = args.smoke

import numpy as np
import pandas as pd
import joblib as jl
import optuna

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
)
from scipy.stats import uniform, loguniform

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

N_CPU = os.cpu_count() or 4

# ── Paths ──
EXPERIMENT_DIR = Path.cwd() if Path("data").exists() else Path("transformation_experiment")
DATA_DIR = EXPERIMENT_DIR / "data"
RESULTS_DIR = EXPERIMENT_DIR / "results"
MODELS_DIR = EXPERIMENT_DIR / "models_focused"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

print(f"CPU cores:   {N_CPU}")
print("Data dir:   ", DATA_DIR.resolve())
print("Results dir:", RESULTS_DIR.resolve())
print("Models dir: ", MODELS_DIR.resolve())

# ── Load splits ──
with open(DATA_DIR / "splits_rskf.json") as f:
    splits = json.load(f)
if SMOKE:
    splits = {k: v for k, v in splits.items() if k.startswith("rep0_")}
    print(f"SMOKE MODE: using {len(splits)} splits (rep0 only)")
else:
    print(f"Loaded {len(splits)} splits (10 repeats × 5 folds)")


# ═══════════════════════════════════════════════════════════════════════
# 1. Representations
# ═══════════════════════════════════════════════════════════════════════

N_COEFFS_GRID = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

REPRESENTATIONS = [
    {"name": "og_xp_110", "file": "og_xp.csv", "n_features": 110},
]

for basis in ["chebyshev", "legendre"]:
    for n in N_COEFFS_GRID:
        REPRESENTATIONS.append({
            "name": f"{basis}_{n}_L2",
            "file": f"{basis}_{n}_L2.csv",
            "n_features": n,
        })

if SMOKE:
    REPRESENTATIONS = [r for r in REPRESENTATIONS if r["name"] in ("og_xp_110", "chebyshev_10_L2")]
elif args.only:
    REPRESENTATIONS = [r for r in REPRESENTATIONS if r["name"] in args.only]

print(f"\n{len(REPRESENTATIONS)} representations:")
for r in REPRESENTATIONS:
    print(f"  {r['name']:25s} ({r['n_features']:3d} features)  [{r['file']}]")


# ═══════════════════════════════════════════════════════════════════════
# 2. Classifier definitions
# ═══════════════════════════════════════════════════════════════════════

N_TRIALS = 5 if SMOKE else 50
OPTUNA_TIMEOUT = 60 if SMOKE else 300


def lr_pipeline(**kwargs):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE, solver="saga", **kwargs,
        )),
    ])


def svm_pipeline(**kwargs):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(probability=False, random_state=RANDOM_STATE, **kwargs)),
    ])


def rf_pipeline(**kwargs):
    return Pipeline([
        ("clf", RandomForestClassifier(
            random_state=RANDOM_STATE, n_jobs=-1, **kwargs,
        )),
    ])


def xgb_pipeline(**kwargs):
    return Pipeline([
        ("clf", XGBClassifier(
            eval_metric="logloss", random_state=RANDOM_STATE,
            n_jobs=-1, verbosity=0, **kwargs,
        )),
    ])


def svm_params(trial):
    return {
        "C": trial.suggest_float("C", 1e-2, 1e3, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 1e1, log=True),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }


def rf_params(trial):
    return {
        "n_estimators": trial.suggest_categorical("n_estimators", [100, 300, 500]),
        "max_depth": trial.suggest_categorical("max_depth", [None, 10, 20, 30]),
        "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [1, 2, 5]),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3]),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }


def xgb_params(trial):
    return {
        "n_estimators": trial.suggest_categorical("n_estimators", [100, 300, 500]),
        "max_depth": trial.suggest_categorical("max_depth", [3, 5, 7, 10]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "scale_pos_weight": trial.suggest_categorical("scale_pos_weight", [1, 3, 4]),
    }


# outer_n_jobs: how many splits to run in parallel for this classifier
# cv_n_jobs: n_jobs for cross_val_score / cross_val_predict inside each split
#
# Logic:
#   - SVM is single-threaded → run 4 splits in parallel, each with 3-fold CV
#     (4 × 3 = 12 threads ≈ N_CPU)
#   - LogReg: RandomizedSearchCV n_jobs=-1 already saturates all cores
#   - RF/XGBoost: estimator itself is multi-threaded (n_jobs=-1), so
#     inner CV should NOT parallelize folds (would oversubscribe)

CLASSIFIERS = [
    {
        "name": "LogisticRegression",
        "method": "randomized",
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "params": {
            "clf__C": loguniform(1e-3, 1e3),
            "clf__penalty": ["l1", "l2"],
            "clf__solver": ["saga"],
            "clf__class_weight": [None, "balanced"],
        },
        "n_iter": 5 if SMOKE else 50,
        "outer_n_jobs": 1,
        "cv_n_jobs": -1,
    },
    {
        "name": "SVM_RBF",
        "method": "optuna",
        "pipeline_fn": svm_pipeline,
        "param_fn": svm_params,
        "needs_calibration": True,
        "outer_n_jobs": max(1, N_CPU // 3),
        "cv_n_jobs": 3,
    },
    {
        "name": "RandomForest",
        "method": "optuna",
        "pipeline_fn": rf_pipeline,
        "param_fn": rf_params,
        "outer_n_jobs": 1,
        "cv_n_jobs": 1,
    },
    {
        "name": "XGBoost",
        "method": "optuna",
        "pipeline_fn": xgb_pipeline,
        "param_fn": xgb_params,
        "outer_n_jobs": 1,
        "cv_n_jobs": 1,
    },
]

print(f"\n{len(CLASSIFIERS)} classifiers:")
for c in CLASSIFIERS:
    print(f"  {c['name']:20s}  method={c['method']:10s}  "
          f"outer_n_jobs={c['outer_n_jobs']}  cv_n_jobs={c['cv_n_jobs']}")


# ═══════════════════════════════════════════════════════════════════════
# 3. Evaluation helpers
# ═══════════════════════════════════════════════════════════════════════

def pick_youden_threshold(y_true, y_prob, grid_size=200):
    """Pick threshold maximising Youden's J = Sensitivity + Specificity - 1."""
    thresholds = np.linspace(0, 1, grid_size)
    best_j, best_thr = -1, 0.5
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0
        spec = tn / (tn + fp) if (tn + fp) else 0
        j = sens + spec - 1
        if j > best_j:
            best_j, best_thr = j, thr
    return best_thr


def get_oof_probabilities(pipeline, X_tr, y_tr, cv=3, n_jobs=1):
    """Out-of-fold probability predictions on training set."""
    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    try:
        y_prob_oof = cross_val_predict(
            pipeline, X_tr, y_tr, cv=cv_obj,
            method="predict_proba", n_jobs=n_jobs,
        )[:, 1]
    except Exception:
        y_prob_oof = cross_val_predict(
            pipeline, X_tr, y_tr, cv=cv_obj,
            method="decision_function", n_jobs=n_jobs,
        )
        y_prob_oof = (y_prob_oof - y_prob_oof.min()) / (y_prob_oof.max() - y_prob_oof.min())
    return y_prob_oof


def evaluate(y_true, y_prob, threshold):
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


# ═══════════════════════════════════════════════════════════════════════
# 4. Single-cell worker (one repr × clf × split)
# ═══════════════════════════════════════════════════════════════════════

def process_cell(repr_cfg, clf_cfg, split_name, split_idx, X_all, y_all):
    """Run HPO + evaluation for one (representation, classifier, split).

    Returns (metrics_dict, model_bytes, best_params) — model_bytes is the
    joblib-serialised pipeline so the caller can save it to disk.
    """
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    cv_n_jobs = clf_cfg["cv_n_jobs"]

    train_idx = np.array(split_idx["train"])
    test_idx = np.array(split_idx["test"])
    X_tr, y_tr = X_all[train_idx], y_all[train_idx]
    X_te, y_te = X_all[test_idx], y_all[test_idx]

    best_params = {}

    if clf_cfg["method"] == "randomized":
        search = RandomizedSearchCV(
            clf_cfg["pipeline"],
            clf_cfg["params"],
            n_iter=clf_cfg.get("n_iter", 50),
            cv=inner_cv,
            scoring="roc_auc",
            random_state=RANDOM_STATE,
            n_jobs=cv_n_jobs,
            error_score="raise",
        )
        search.fit(X_tr, y_tr)
        best_pipe = search.best_estimator_
        best_cv_score = search.best_score_
        best_params = {
            k.replace("clf__", ""): v
            for k, v in search.best_params_.items()
        }
    else:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = clf_cfg["param_fn"](trial)
            pipe = clf_cfg["pipeline_fn"](**params)
            scores = cross_val_score(
                pipe, X_tr, y_tr, cv=inner_cv,
                scoring="roc_auc", n_jobs=cv_n_jobs,
            )
            return scores.mean()

        sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=N_TRIALS, timeout=OPTUNA_TIMEOUT)
        best_params = study.best_trial.params
        best_cv_score = study.best_value
        best_pipe = clf_cfg["pipeline_fn"](**best_params)
        best_pipe.fit(X_tr, y_tr)

    # SVM calibration
    if clf_cfg.get("needs_calibration"):
        cal_pipe = CalibratedClassifierCV(best_pipe, cv=3, method="sigmoid", n_jobs=cv_n_jobs)
        cal_pipe.fit(X_tr, y_tr)
        save_pipe = cal_pipe
    else:
        save_pipe = best_pipe

    # Youden threshold on OOF predictions
    y_prob_oof = get_oof_probabilities(save_pipe, X_tr, y_tr, cv=3, n_jobs=cv_n_jobs)
    thr = pick_youden_threshold(y_tr, y_prob_oof)

    # Evaluate on test set
    y_prob_te = save_pipe.predict_proba(X_te)[:, 1]
    metrics = evaluate(y_te, y_prob_te, thr)
    metrics.update({
        "representation": repr_cfg["name"],
        "n_features": repr_cfg["n_features"],
        "classifier": clf_cfg["name"],
        "split": split_name,
        "best_cv_roc_auc": best_cv_score,
    })

    clean_params = {
        k: (v if not isinstance(v, np.generic) else v.item())
        for k, v in best_params.items()
    }

    return metrics, save_pipe, thr, best_cv_score, clean_params


# ═══════════════════════════════════════════════════════════════════════
# 5. Resume: load already-completed cells
# ═══════════════════════════════════════════════════════════════════════

results_path = RESULTS_DIR / ("_smoke_results.csv" if SMOKE else "focused_experiment_results.csv")
completed = set()

if results_path.exists():
    df_done = pd.read_csv(results_path)
    for _, row in df_done.iterrows():
        completed.add((row["representation"], row["classifier"], row["split"]))
    print(f"\nResume: {len(completed)} cells already completed, will skip them.")
else:
    df_done = pd.DataFrame()

all_results = df_done.to_dict("records") if len(df_done) > 0 else []
model_manifest = {}

# Reload manifest if it exists
manifest_path = MODELS_DIR / "model_manifest.json"
if manifest_path.exists():
    with open(manifest_path) as f:
        model_manifest = json.load(f)

print("Ready.\n")


# ═══════════════════════════════════════════════════════════════════════
# 6. Run experiment
# ═══════════════════════════════════════════════════════════════════════

total_cells = len(REPRESENTATIONS) * len(CLASSIFIERS) * len(splits)
done_count = len(completed)
t_start = time.time()
csv_columns = None

for repr_i, repr_cfg in enumerate(REPRESENTATIONS, 1):
    repr_file = DATA_DIR / repr_cfg["file"]
    print(
        f"\n{'═' * 70}\n"
        f"  [{repr_i}/{len(REPRESENTATIONS)}]  {repr_cfg['name']}  "
        f"({repr_cfg['n_features']} features)\n"
        f"{'═' * 70}",
        flush=True,
    )
    if not repr_file.exists():
        print(f"  SKIP — file not found: {repr_file.name}")
        done_count += len(CLASSIFIERS) * len(splits)
        continue

    df = pd.read_csv(repr_file)
    feat_cols = [c for c in df.columns if c not in ("source_id", "y")]
    X_all = df[feat_cols].to_numpy(dtype=np.float64)
    y_all = df["y"].to_numpy(dtype=int)

    for clf_cfg in CLASSIFIERS:
        # Collect pending splits for this (repr, clf)
        pending = [
            (sname, sidx) for sname, sidx in splits.items()
            if (repr_cfg["name"], clf_cfg["name"], sname) not in completed
        ]

        if not pending:
            done_count += len(splits)
            print(f"  ── {clf_cfg['name']:20s}  all {len(splits)} splits already done, skipping ──", flush=True)
            continue

        outer_n_jobs = min(clf_cfg["outer_n_jobs"], len(pending))
        print(
            f"\n  ── {clf_cfg['name']:20s}  {len(pending)} pending  "
            f"(parallel={outer_n_jobs}) ──",
            flush=True,
        )
        t_block = time.time()

        if outer_n_jobs > 1:
            # Parallel execution across splits
            batch_results = jl.Parallel(n_jobs=outer_n_jobs, prefer="processes")(
                jl.delayed(process_cell)(
                    repr_cfg, clf_cfg, sname, sidx, X_all, y_all,
                )
                for sname, sidx in pending
            )
        else:
            # Sequential execution
            batch_results = [
                process_cell(repr_cfg, clf_cfg, sname, sidx, X_all, y_all)
                for sname, sidx in pending
            ]

        # Collect results from batch
        for (sname, _), (metrics, save_pipe, thr, best_cv_score, clean_params) in zip(pending, batch_results):
            done_count += 1
            all_results.append(metrics)

            # Incremental CSV append
            if csv_columns is None:
                csv_columns = list(metrics.keys())
                if not results_path.exists() or results_path.stat().st_size == 0:
                    with open(results_path, "w", newline="") as f:
                        csv.DictWriter(f, fieldnames=csv_columns).writeheader()
            with open(results_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=csv_columns).writerow(metrics)

            # Save model
            model_filename = f"{repr_cfg['name']}__{clf_cfg['name']}__{sname}.joblib"
            jl.dump(save_pipe, MODELS_DIR / model_filename)
            model_manifest[model_filename] = {
                "representation": repr_cfg["name"],
                "classifier": clf_cfg["name"],
                "split": sname,
                "roc_auc": round(metrics["roc_auc"], 6),
                "threshold": round(thr, 4),
                "best_cv_roc_auc": round(best_cv_score, 6),
                "best_params": clean_params,
            }

            elapsed = time.time() - t_start
            cells_done_this_run = done_count - len(completed)
            if cells_done_this_run > 0:
                eta = (elapsed / cells_done_this_run) * (total_cells - done_count)
            else:
                eta = 0
            print(
                f"  [{done_count:4d}/{total_cells}]  {repr_cfg['name']:25s}  "
                f"{clf_cfg['name']:20s}  {sname}  "
                f"ROC-AUC={metrics['roc_auc']:.4f}  "
                f"Sens={metrics['sensitivity']:.4f}  "
                f"Prec={metrics['precision']:.4f}  "
                f"ETA={eta / 60:.1f}min",
                flush=True,
            )

        block_time = time.time() - t_block
        block_aucs = [m["roc_auc"] for m in all_results[-len(pending):]]
        print(
            f"  >> {repr_cfg['name']} × {clf_cfg['name']}: "
            f"{len(pending)} folds in {block_time:.1f}s  "
            f"mean ROC-AUC={np.mean(block_aucs):.4f}",
            flush=True,
        )

        # Checkpoint manifest after each (repr, clf) block
        with open(manifest_path, "w") as f:
            json.dump(model_manifest, f, indent=2)

elapsed_total = time.time() - t_start
print(f"\nFinished {done_count} cells in {elapsed_total / 60:.1f} minutes.")


# ═══════════════════════════════════════════════════════════════════════
# 7. Save final results
# ═══════════════════════════════════════════════════════════════════════

df_results = pd.DataFrame(all_results)
final_csv = RESULTS_DIR / ("_smoke_results.csv" if SMOKE else "focused_experiment_results.csv")
df_results.to_csv(final_csv, index=False)
print(f"Saved {len(df_results)} rows → {final_csv.name}")

df_results["repeat"] = df_results["split"].str.extract(r"rep(\d+)").astype(int)

metric_cols = [
    "roc_auc", "pr_auc", "youden_j", "f1",
    "sensitivity", "specificity", "precision", "accuracy",
    "brier", "log_loss",
]

repeat_means = (
    df_results
    .groupby(["representation", "n_features", "classifier", "repeat"])[metric_cols]
    .mean()
    .reset_index()
)

agg = (
    repeat_means
    .groupby(["representation", "n_features", "classifier"])[metric_cols]
    .agg(["mean", "std"])
)
agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
agg = agg.reset_index().sort_values("roc_auc_mean", ascending=False)
summary_csv = RESULTS_DIR / ("_smoke_summary.csv" if SMOKE else "focused_summary.csv")
agg.to_csv(summary_csv, index=False)
print(f"Saved aggregated summary → {summary_csv.name}")

with open(manifest_path, "w") as f:
    json.dump(model_manifest, f, indent=2)

total_size_mb = sum(
    (MODELS_DIR / fn).stat().st_size
    for fn in model_manifest if (MODELS_DIR / fn).exists()
) / (1024 * 1024)

print(f"Saved {len(model_manifest)} models → models_focused/")
print(f"Total size: {total_size_mb:.1f} MB")
print(f"\nRun 06_analyze_focused.ipynb for analysis.")
