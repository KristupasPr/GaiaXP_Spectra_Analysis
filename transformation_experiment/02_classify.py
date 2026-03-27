#!/usr/bin/env python
# coding: utf-8

# # 02 — Classification Experiment
# 
# Compares **original Gaia XP coefficients (110-d)** against **polynomial-transformed features**
# (Chebyshev, Hermite, Laguerre, Legendre at 10–50 coefficients) using six classifiers.
# 
# **Experimental grid:**
# - 1 baseline (OG XP 110) + 4 bases × 5 dims × 2 normalisations (raw, L2) = **41 representations**
# - 6 classifiers: Logistic Regression, SVM (RBF), Random Forest, XGBoost, GaussianNB, k-NN
# - 10 stratified train/test splits → report mean ± std
# 
# **Methodology improvements over v1:**
# - Youden threshold selected on **out-of-fold** training predictions (not in-sample)
# - `N_ITER=50` for `RandomizedSearchCV` (was 30)
# - 10 splits for adequate Wilcoxon signed-rank power (min p ≈ 0.002)
# - Polynomial features tested with and without L2 normalization
# - Best model pipelines saved with `joblib` for reproducibility
# 
# **Outputs:**
# - `results/experiment_results.csv` — one row per (representation × classifier × split)
# - `results/summary.csv` — aggregated mean ± std
# - `models/{repr}__{clf}__{split}.joblib` — best pipeline per cell
# - `models/model_manifest.json` — metadata for all saved models

# In[1]:


import csv
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, cross_val_predict,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, log_loss,
)
from scipy.stats import uniform, loguniform

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# In[2]:


# ── Paths ──
EXPERIMENT_DIR = Path.cwd() if Path("data").exists() else Path("transformation_experiment")
DATA_DIR = EXPERIMENT_DIR / "data"
RESULTS_DIR = EXPERIMENT_DIR / "results"
MODELS_DIR = EXPERIMENT_DIR / "models"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

print("Data dir:   ", DATA_DIR.resolve())
print("Results dir:", RESULTS_DIR.resolve())
print("Models dir: ", MODELS_DIR.resolve())


# In[3]:


# ── Load splits ──
with open(DATA_DIR / "splits.json") as f:
    splits = json.load(f)
print(f"Loaded {len(splits)} splits")


# ## 1. Define representations and classifiers

# In[4]:


# ── Representations ──
REPRESENTATIONS = [
    {"name": "og_xp_110", "file": "og_xp.csv", "n_features": 110},
]

for basis in ["chebyshev", "hermite", "laguerre", "legendre"]:
    for n in [10, 20, 30, 40, 50]:
        for norm in ["raw", "L2"]:
            REPRESENTATIONS.append({
                "name": f"{basis}_{n}_{norm}",
                "file": f"{basis}_{n}_{norm}.csv",
                "n_features": n,
            })

print(f"{len(REPRESENTATIONS)} representations:")
for r in REPRESENTATIONS:
    print(f"  {r['name']:25s} ({r['n_features']:3d} features)  [{r['file']}]")


# In[5]:


# ── Classifier definitions ──
N_ITER = 50
INNER_CV = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

CLASSIFIERS = [
    {
        "name": "LogisticRegression",
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
        "n_iter": N_ITER,
    },
    {
        "name": "SVM_RBF",
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=False, random_state=RANDOM_STATE)),
        ]),
        "params": {
            "clf__C": loguniform(1e-2, 1e3),
            "clf__gamma": loguniform(1e-4, 1e1),
            "clf__class_weight": [None, "balanced"],
        },
        "needs_calibration": True,
        "n_iter": N_ITER,
    },
    {
        "name": "RandomForest",
        "pipeline": Pipeline([
            ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "params": {
            "clf__n_estimators": [100, 300, 500],
            "clf__max_depth": [None, 10, 20, 30],
            "clf__min_samples_leaf": [1, 2, 5],
            "clf__max_features": ["sqrt", "log2", 0.3],
            "clf__class_weight": [None, "balanced"],
        },
        "n_iter": N_ITER,
    },
    {
        "name": "XGBoost",
        "pipeline": Pipeline([
            ("clf", XGBClassifier(
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=0,
            )),
        ]),
        "params": {
            "clf__n_estimators": [100, 300, 500],
            "clf__max_depth": [3, 5, 7, 10],
            "clf__learning_rate": loguniform(0.01, 0.3),
            "clf__subsample": uniform(0.6, 0.4),
            "clf__colsample_bytree": uniform(0.5, 0.5),
            "clf__scale_pos_weight": [1, 3, 4],
        },
        "n_iter": N_ITER,
    },
    {
        "name": "GaussianNB",
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GaussianNB()),
        ]),
        "params": {
            "clf__var_smoothing": loguniform(1e-12, 1e-6),
        },
        "n_iter": 10,
    },
    {
        "name": "kNN",
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier()),
        ]),
        "params": {
            "clf__n_neighbors": [3, 5, 7, 11, 15, 21],
            "clf__weights": ["uniform", "distance"],
            "clf__metric": ["euclidean", "manhattan"],
        },
        "n_iter": 20,
    },
]

print(f"{len(CLASSIFIERS)} classifiers: {[c['name'] for c in CLASSIFIERS]}")


# ## 2. Evaluation helpers

# In[6]:


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


def get_oof_probabilities(pipeline, X_tr, y_tr, cv=3):
    """Get out-of-fold probability predictions on the training set.

    Uses cross_val_predict so the threshold is not tuned on in-sample
    (potentially overfit) predictions.
    """
    try:
        y_prob_oof = cross_val_predict(
            pipeline, X_tr, y_tr,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE),
            method="predict_proba",
        )[:, 1]
    except Exception:
        y_prob_oof = cross_val_predict(
            pipeline, X_tr, y_tr,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE),
            method="decision_function",
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
    acc  = (tp + tn) / (tp + tn + fp + fn)
    f1   = (2 * prec * sens) / (prec + sens) if (prec + sens) else 0.0
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


print("Evaluation helpers ready.")


# ## 3. Run experiment
# 
# For each (representation × classifier × split):
# 1. Load features, apply train/test split
# 2. `RandomizedSearchCV` on training set (3-fold CV, ROC-AUC)
# 3. Pick Youden threshold on **out-of-fold** training predictions
# 4. Evaluate on held-out test set
# 5. Save best pipeline with `joblib`

# In[7]:


all_results = []
model_manifest = {}

total_cells = len(REPRESENTATIONS) * len(CLASSIFIERS) * len(splits)
cell_idx = 0
t_start = time.time()

results_path = RESULTS_DIR / "experiment_results.csv"
csv_columns = None

for repr_cfg in REPRESENTATIONS:
    repr_file = DATA_DIR / repr_cfg["file"]
    if not repr_file.exists():
        print(f"  SKIP {repr_cfg['name']} — file not found: {repr_file.name}")
        continue

    df = pd.read_csv(repr_file)
    feat_cols = [c for c in df.columns if c not in ("source_id", "y")]
    X_all = df[feat_cols].to_numpy(dtype=np.float64)
    y_all = df["y"].to_numpy(dtype=int)

    for clf_cfg in CLASSIFIERS:
        for split_name, split_idx in splits.items():
            cell_idx += 1
            train_idx = np.array(split_idx["train"])
            test_idx  = np.array(split_idx["test"])

            X_tr, y_tr = X_all[train_idx], y_all[train_idx]
            X_te, y_te = X_all[test_idx],  y_all[test_idx]

            # Hyperparameter search
            search = RandomizedSearchCV(
                clf_cfg["pipeline"],
                clf_cfg["params"],
                n_iter=clf_cfg.get("n_iter", N_ITER),
                cv=INNER_CV,
                scoring="roc_auc",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                error_score="raise",
            )
            search.fit(X_tr, y_tr)
            best_pipe = search.best_estimator_

            # For SVM: calibrate probabilities
            if clf_cfg.get("needs_calibration"):
                cal_pipe = CalibratedClassifierCV(best_pipe, cv=3, method="sigmoid")
                cal_pipe.fit(X_tr, y_tr)
                save_pipe = cal_pipe
            else:
                save_pipe = best_pipe

            # Youden threshold on OUT-OF-FOLD predictions (not in-sample)
            y_prob_oof = get_oof_probabilities(save_pipe, X_tr, y_tr, cv=3)
            thr = pick_youden_threshold(y_tr, y_prob_oof)

            # Evaluate on test set
            y_prob_te = save_pipe.predict_proba(X_te)[:, 1]
            metrics = evaluate(y_te, y_prob_te, thr)
            metrics.update({
                "representation": repr_cfg["name"],
                "n_features": repr_cfg["n_features"],
                "classifier": clf_cfg["name"],
                "split": split_name,
                "best_cv_roc_auc": search.best_score_,
            })
            all_results.append(metrics)

            # Incremental CSV save
            if csv_columns is None:
                csv_columns = list(metrics.keys())
                with open(results_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=csv_columns)
                    writer.writeheader()
            with open(results_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=csv_columns).writerow(metrics)

            # Save model
            model_filename = f"{repr_cfg['name']}__{clf_cfg['name']}__{split_name}.joblib"
            model_path = MODELS_DIR / model_filename
            joblib.dump(save_pipe, model_path)
            model_manifest[model_filename] = {
                "representation": repr_cfg["name"],
                "classifier": clf_cfg["name"],
                "split": split_name,
                "roc_auc": round(metrics["roc_auc"], 6),
                "threshold": round(thr, 4),
                "best_cv_roc_auc": round(search.best_score_, 6),
            }

            elapsed = time.time() - t_start
            eta = (elapsed / cell_idx) * (total_cells - cell_idx)
            print(
                f"  [{cell_idx:4d}/{total_cells}]  {repr_cfg['name']:25s}  {clf_cfg['name']:20s}  "
                f"{split_name}  ROC-AUC={metrics['roc_auc']:.4f}  "
                f"Youden={metrics['youden_j']:.4f}  "
                f"Sens={metrics['sensitivity']:.4f}  "
                f"Prec={metrics['precision']:.4f}  ETA={eta/60:.1f}min",
                flush=True,
            )

elapsed_total = time.time() - t_start
print(f"\nFinished {cell_idx} cells in {elapsed_total/60:.1f} minutes.")


# ## 4. Save results

# In[ ]:


# ── Overwrite with clean version ──
df_results = pd.DataFrame(all_results)
df_results.to_csv(RESULTS_DIR / "experiment_results.csv", index=False)
print(f"Saved {len(df_results)} rows → results/experiment_results.csv")


# In[ ]:


# ── Aggregated summary ──
metric_cols = ["roc_auc", "pr_auc", "youden_j", "f1",
               "sensitivity", "specificity", "precision", "accuracy",
               "brier", "log_loss"]

agg = (
    df_results
    .groupby(["representation", "n_features", "classifier"])[metric_cols]
    .agg(["mean", "std"])
)
agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
agg = agg.reset_index().sort_values("roc_auc_mean", ascending=False)
agg.to_csv(RESULTS_DIR / "summary.csv", index=False)
print(f"Saved aggregated summary → results/summary.csv")


# In[ ]:


# ── Save model manifest ──
with open(MODELS_DIR / "model_manifest.json", "w") as f:
    json.dump(model_manifest, f, indent=2)

total_size_mb = sum(
    (MODELS_DIR / fn).stat().st_size for fn in model_manifest
) / (1024 * 1024)

print(f"Saved {len(model_manifest)} models → models/")
print(f"Total size: {total_size_mb:.1f} MB")
print(f"Manifest:   models/model_manifest.json")
print(f"\nRun 03_analyze_results.ipynb for visualisation and statistical analysis.")

