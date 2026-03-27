#!/usr/bin/env python
"""
Quick pipeline check: Logistic Regression on Chebyshev coefficients 1–10.

For n_coeffs = k, uses the first k columns (c000 … c{k-1}) from
chebyshev_10_L2.csv. Coefficients are identical regardless of the max
polynomial degree because Chebyshev polynomials are orthogonal.

Uses a single 5-fold stratified CV (rep0 only) for speed.
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
)
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

EXPERIMENT_DIR = Path.cwd() if Path("data").exists() else Path("transformation_experiment")
DATA_DIR = EXPERIMENT_DIR / "data"
RESULTS_DIR = EXPERIMENT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Load data (10 Chebyshev coefficients) ──
df = pd.read_csv(DATA_DIR / "chebyshev_10_L2.csv")
all_coeff_cols = [c for c in df.columns if c.startswith("c")]
y_all = df["y"].to_numpy(dtype=int)
X_all_10 = df[all_coeff_cols].to_numpy(dtype=np.float64)

print(f"Dataset: {len(df)} samples, {sum(y_all)} positives, "
      f"{len(y_all) - sum(y_all)} negatives")
print(f"Available coefficient columns: {all_coeff_cols}\n")

# ── Load splits (rep0 only — 5 folds) ──
with open(DATA_DIR / "splits_rskf.json") as f:
    all_splits = json.load(f)

splits = {k: v for k, v in all_splits.items() if k.startswith("rep0_")}
print(f"Using {len(splits)} splits (rep0, 5-fold)\n")


# ── Evaluation helpers ──

def pick_youden_threshold(y_true, y_prob, grid_size=200):
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


# ── Run experiment ──

results = []
t0 = time.time()

for n_coeffs in range(1, 11):
    X = X_all_10[:, :n_coeffs]
    repr_name = f"chebyshev_{n_coeffs}_L2"
    print(f"{'─'*60}")
    print(f"  n_coeffs={n_coeffs:2d}  ({repr_name})")

    fold_metrics = []

    for split_name, split_idx in splits.items():
        train_idx = np.array(split_idx["train"])
        test_idx = np.array(split_idx["test"])
        X_tr, y_tr = X[train_idx], y_all[train_idx]
        X_te, y_te = X[test_idx], y_all[test_idx]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ])
        param_dist = {
            "clf__C": loguniform(1e-3, 1e3),
            "clf__penalty": ["l1", "l2"],
            "clf__solver": ["saga"],
            "clf__class_weight": [None, "balanced"],
        }

        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        search = RandomizedSearchCV(
            pipe, param_dist,
            n_iter=20,
            cv=inner_cv,
            scoring="roc_auc",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        search.fit(X_tr, y_tr)
        best_pipe = search.best_estimator_

        # OOF predictions for threshold
        from sklearn.model_selection import cross_val_predict
        oof_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        y_prob_oof = cross_val_predict(
            best_pipe, X_tr, y_tr, cv=oof_cv, method="predict_proba", n_jobs=1
        )[:, 1]
        thr = pick_youden_threshold(y_tr, y_prob_oof)

        y_prob_te = best_pipe.predict_proba(X_te)[:, 1]
        m = evaluate(y_te, y_prob_te, thr)
        m.update({
            "representation": repr_name,
            "n_features": n_coeffs,
            "classifier": "LogisticRegression",
            "split": split_name,
            "best_cv_roc_auc": search.best_score_,
        })
        fold_metrics.append(m)
        results.append(m)

    aucs = [fm["roc_auc"] for fm in fold_metrics]
    sens = [fm["sensitivity"] for fm in fold_metrics]
    print(f"  ROC-AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}   "
          f"Sens = {np.mean(sens):.4f} ± {np.std(sens):.4f}")

elapsed = time.time() - t0
print(f"\n{'═'*60}")
print(f"Done in {elapsed:.1f}s\n")

# ── Save & display summary ──

df_res = pd.DataFrame(results)
out_csv = RESULTS_DIR / "lr_chebyshev_1to10.csv"
df_res.to_csv(out_csv, index=False)
print(f"Saved {len(df_res)} rows → {out_csv}\n")

summary = (
    df_res
    .groupby(["representation", "n_features"])[
        ["roc_auc", "pr_auc", "sensitivity", "specificity", "f1", "brier"]
    ]
    .agg(["mean", "std"])
)
summary.columns = [f"{c}_{s}" for c, s in summary.columns]
summary = summary.reset_index().sort_values("n_features")

summary_csv = RESULTS_DIR / "lr_chebyshev_1to10_summary.csv"
summary.to_csv(summary_csv, index=False)
print(f"Saved summary → {summary_csv}\n")

print(summary[["representation", "n_features",
               "roc_auc_mean", "roc_auc_std",
               "sensitivity_mean", "f1_mean"]].to_string(index=False))


# ── Visualisation ──

import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

s = summary.sort_values("n_features")
x = s["n_features"].values

metrics_to_plot = [
    ("roc_auc",      "ROC-AUC",      "#2166ac"),
    ("pr_auc",       "PR-AUC",       "#b2182b"),
    ("f1",           "F1",           "#1b7837"),
    ("sensitivity",  "Sensitivity",  "#762a83"),
]

fig, ax = plt.subplots(figsize=(8, 4.5))

for col, label, color in metrics_to_plot:
    mean = s[f"{col}_mean"].values
    std = s[f"{col}_std"].values
    ax.plot(x, mean, "o-", label=label, color=color, linewidth=2, markersize=5)
    ax.fill_between(x, mean - std, mean + std, alpha=0.12, color=color)

ax.set_xlabel("Number of Chebyshev coefficients")
ax.set_ylabel("Score")
ax.set_title("Logistic Regression: performance vs. number of Chebyshev coefficients")
ax.set_xticks(range(1, 11))
ax.set_ylim(0.35, 1.0)
ax.legend(loc="lower right", frameon=False)
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig_path = RESULTS_DIR / "lr_chebyshev_1to10.png"
fig.savefig(fig_path, bbox_inches="tight")
print(f"\nSaved figure → {fig_path}")
plt.close(fig)
