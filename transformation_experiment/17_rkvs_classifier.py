#!/usr/bin/env python
"""
17 — RKVS (Reproducing Kernel Variable Selection) + classifiers on L2-normalised spectra

Greedy impact-point selection: selects K wavelength indices that maximise the squared
Mahalanobis distance between class means under the within-class covariance (binary
RKHS variable selection, linear kernel — equivalent to RKVS from Ramos-Carreño et al. 2021).
Selected flux values fed to LDA, QDA, k-NN, LogisticRegression, and SVM classifiers.

Outputs (results/):
  rkvs_rskf_metrics.csv  — per-fold [split, K, classifier, pr_auc, roc_auc, ...]
  rkvs_rskf_summary.csv  — mean/std aggregated by [K, classifier]
  rkvs_impact_points_stability.png — heatmap of wavelength selection frequency
  rkvs_mean_spectra_with_impacts.png — class means with selected wavelengths

Exploration / extra plots: see 18_rkvs_results.ipynb.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.discriminant_analysis import (  # noqa: E402
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

K_VALUES = [1, 2, 3, 5, 8, 10, 15, 20, 30]

parser = argparse.ArgumentParser(description="RKVS + classifiers on L2 Gaia XP spectra")
parser.add_argument(
    "--smoke",
    action="store_true",
    help="Use rep0 folds only (5 splits) for a quick run",
)
args = parser.parse_args()
SMOKE = args.smoke

EXPERIMENT_DIR = Path.cwd() if Path("data").exists() else Path("transformation_experiment")
DATA_DIR = EXPERIMENT_DIR / "data"
RESULTS_DIR = EXPERIMENT_DIR / "results"
POC_DIR = (
    Path("transformation_poc")
    if Path("transformation_poc").exists()
    else Path("..") / "transformation_poc"
)
RESULTS_DIR.mkdir(exist_ok=True)

METRICS_CSV = "rkvs_rskf_metrics.csv"
SUMMARY_CSV = "rkvs_rskf_summary.csv"
METRICS_PATH = RESULTS_DIR / METRICS_CSV
SUMMARY_PATH = RESULTS_DIR / SUMMARY_CSV

print("Data dir:   ", DATA_DIR.resolve())
print("Results dir:", RESULTS_DIR.resolve())
print("POC dir:    ", POC_DIR.resolve())

# ── Load aligned spectra + labels ──
df_og = pd.read_csv(DATA_DIR / "og_xp.csv")
df_spec = pd.read_csv(POC_DIR / "xp_sampled_spectra.csv")
wl_cols = [c for c in df_spec.columns if c.startswith("wl_")]
wavelengths = np.array([float(c.split("_")[1]) for c in wl_cols], dtype=np.float64)

df_m = df_og[["source_id", "y"]].merge(
    df_spec[["source_id"] + wl_cols], on="source_id", how="inner", validate="one_to_one"
)
assert len(df_m) == len(df_og) == len(df_spec), "Row alignment failed"

y = df_m["y"].to_numpy(dtype=np.int64)
F_raw = df_m[wl_cols].to_numpy(dtype=np.float64)
n, p = F_raw.shape
print(f"N = {n}, p = {p} (wavelength bins)")
print(
    f"Class balance: single={int((y == 0).sum())}, binary={int((y == 1).sum())} "
    f"({(y == 1).mean():.1%} positive)"
)

# ── L2-normalise ──
norms = np.linalg.norm(F_raw, axis=1, keepdims=True)
norms = np.maximum(norms, 1e-15)
F = F_raw / norms

# ── Mean shape difference (diagnostic) ──
mu0 = F[y == 0].mean(axis=0)
mu1 = F[y == 1].mean(axis=0)
delta = mu1 - mu0
print(f"Mean shape difference L2 norm: {np.linalg.norm(delta):.6f}")

# ── RSKF CV ──
with open(DATA_DIR / "splits_rskf.json") as f:
    splits = json.load(f)
if SMOKE:
    splits = {k: v for k, v in splits.items() if k.startswith("rep0_")}
    print(f"SMOKE: using {len(splits)} splits")
else:
    print(f"Loaded {len(splits)} splits (10 repeats × 5 folds)")


def split_sort_key(k: str):
    rep = int(k.split("_")[0].replace("rep", ""))
    fold = int(k.split("_")[1].replace("fold", ""))
    return (rep, fold)


split_names = sorted(splits.keys(), key=split_sort_key)


# ── RKVS algorithm ──
def rkvs_select(
    F_tr: np.ndarray,
    y_tr: np.ndarray,
    K: int,
    ridge_eps: float = 1e-6,
) -> list[int]:
    """
    Greedy impact-point selection for binary classification.
    Selects K wavelength indices maximising squared Mahalanobis distance
    between class means under pooled within-class covariance (linear RKHS).

    Args:
        F_tr: training flux matrix (n_tr, p)
        y_tr: training labels (n_tr,), binary {0, 1}
        K: number of impact points to select
        ridge_eps: regularisation for near-singular covariance

    Returns:
        List of K integer column indices (wavelengths) in greedy order
    """
    idx0 = np.where(y_tr == 0)[0]
    idx1 = np.where(y_tr == 1)[0]
    n_tr = len(y_tr)

    mu0 = F_tr[idx0].mean(axis=0)
    mu1 = F_tr[idx1].mean(axis=0)

    # Pooled within-class covariance
    C0 = F_tr[idx0] - mu0
    C1 = F_tr[idx1] - mu1
    S_W = (C0.T @ C0 + C1.T @ C1) / (n_tr - 2)

    selected: list[int] = []
    remaining = list(range(F_tr.shape[1]))

    for _ in range(K):
        best_score = -np.inf
        best_j = remaining[0]

        for j in remaining:
            ix = selected + [j]
            S_sub = S_W[np.ix_(ix, ix)]
            S_sub += ridge_eps * np.eye(len(ix))
            delta = (mu1 - mu0)[ix]

            # solve S_sub @ w = delta
            w, *_ = np.linalg.lstsq(S_sub, delta, rcond=None)
            score = float(delta @ w)  # squared Mahalanobis distance

            if score > best_score:
                best_score = score
                best_j = j

        selected.append(best_j)
        remaining.remove(best_j)

    return selected


# ── Metric helpers (verbatim from 12_fpca_classifier.py) ──
def normalize_scores_train_ref(
    scores_te: np.ndarray, scores_tr: np.ndarray
) -> np.ndarray:
    """Map scores to [0, 1] using min/max from training fold."""
    lo, hi = float(scores_tr.min()), float(scores_tr.max())
    if hi == lo:
        return np.full_like(scores_te, 0.5, dtype=np.float64)
    return ((scores_te - lo) / (hi - lo)).astype(np.float64)


def pick_youden_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, grid_size: int = 200
) -> float:
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


def fold_metrics(
    y_true_te: np.ndarray,
    y_score_te: np.ndarray,
    y_true_tr: np.ndarray,
    y_score_tr: np.ndarray,
):
    """PR/ROC from test scores; F1, Youden, etc. at threshold chosen on train."""
    out: dict = {"pr_auc": average_precision_score(y_true_te, y_score_te)}
    try:
        out["roc_auc"] = float(roc_auc_score(y_true_te, y_score_te))
    except ValueError:
        out["roc_auc"] = np.nan

    prob_tr = normalize_scores_train_ref(y_score_tr, y_score_tr)
    prob_te = normalize_scores_train_ref(y_score_te, y_score_tr)
    thr = pick_youden_threshold(y_true_tr, prob_tr)
    y_pred = (prob_te >= thr).astype(np.int64)
    out["youden_threshold"] = thr

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

    return out


METRIC_COLS = [
    "pr_auc",
    "roc_auc",
    "sensitivity",
    "precision",
    "specificity",
    "accuracy",
    "f1",
    "youden_j",
    "youden_threshold",
]

# ── Classifier factory ──
def make_classifiers() -> list[tuple[str, object]]:
    """Return list of (name, classifier) tuples."""
    return [
        ("lda", LinearDiscriminantAnalysis()),
        ("qda", QuadraticDiscriminantAnalysis(reg_param=0.1)),
        ("knn", KNeighborsClassifier(n_neighbors=5, metric="euclidean")),
        (
            "lr",
            LogisticRegression(
                class_weight="balanced",
                max_iter=5000,
                solver="lbfgs",
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "svm",
            SVC(
                C=1.0,
                gamma="scale",
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ]


def get_scores(clf_name: str, clf, X_tr: np.ndarray, X_te: np.ndarray):
    """Return (score_tr, score_te) using decision_function or predict_proba."""
    if clf_name in ("lda", "lr", "svm"):
        return clf.decision_function(X_tr), clf.decision_function(X_te)
    else:  # qda, knn
        return clf.predict_proba(X_tr)[:, 1], clf.predict_proba(X_te)[:, 1]


# ── Atomic save helper ──
def _atomic_save(rows: list[dict], path: Path) -> None:
    """Atomically write DataFrame to CSV via tempfile."""
    df = pd.DataFrame(rows)
    fd, tmp = tempfile.mkstemp(suffix=".csv", dir=str(path.parent))
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# ── Resume safety ──
all_results: list[dict] = []
done_set: set[tuple[str, int, str]] = set()
impacts_store: dict[str, list[int]] = {}  # sname -> impact_points list

if METRICS_PATH.exists():
    df_done = pd.read_csv(METRICS_PATH)
    done_set = set(zip(df_done["split"], df_done["K"], df_done["classifier"]))
    all_results = df_done.to_dict("records")
    print(f"Resume: {len(done_set)} (split, K, classifier) triples done.")
else:
    print("Starting fresh (no prior results found)")

# ── Main loop ──
t_start = time.time()
done_this_session = 0

total_pending = sum(
    1
    for sname in split_names
    for K in K_VALUES
    for clf_name, _ in make_classifiers()
    if (sname, K, clf_name) not in done_set
)
print(f"Pending: {total_pending} (split x K x clf) triples")

for sname in split_names:
    tr_idx = np.array(splits[sname]["train"], dtype=int)
    te_idx = np.array(splits[sname]["test"], dtype=int)
    F_tr, F_te = F[tr_idx], F[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    # Compute RKVS impact points once per fold to K_MAX
    K_MAX = max(K_VALUES)
    t_rkvs = time.time()
    impact_points = rkvs_select(F_tr, y_tr, K_MAX)
    rkvs_time = time.time() - t_rkvs
    impacts_store[sname] = impact_points

    for K in K_VALUES:
        ix = impact_points[:K]  # ordered list of K wavelength indices
        X_tr = F_tr[:, ix]
        X_te = F_te[:, ix]

        for clf_name, clf_proto in make_classifiers():
            if (sname, K, clf_name) in done_set:
                continue

            clf = clone(clf_proto)
            clf.fit(X_tr, y_tr)
            sc_tr, sc_te = get_scores(clf_name, clf, X_tr, X_te)
            met = fold_metrics(y_te, sc_te, y_tr, sc_tr)

            row = {
                "split": sname,
                "K": K,
                "classifier": clf_name,
                "rkvs_time_s": rkvs_time,
                **met,
            }
            # Only store impacts at K_MAX to avoid redundancy
            if K == K_MAX:
                row["impacts_json"] = json.dumps(impact_points)
            all_results.append(row)
            done_set.add((sname, K, clf_name))
            done_this_session += 1

            # Atomic save after each completed row
            _atomic_save(all_results, METRICS_PATH)

            remaining = total_pending - done_this_session
            elapsed = time.time() - t_start
            eta = (elapsed / done_this_session) * remaining if done_this_session and remaining > 0 else 0.0
            print(
                f"  {sname} K={K:2d} {clf_name:5s}  "
                f"PR-AUC={met['pr_auc']:.4f}  F1={met['f1']:.4f}  "
                f"Sens={met['sensitivity']:.4f}  "
                f"ETA~{eta/60:.1f}min",
                flush=True,
            )

print(f"\nCompleted {done_this_session} new triples in {(time.time() - t_start)/60:.1f}min")

# ── Summary aggregation ──
def write_summary(rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    _named = {}
    for m in METRIC_COLS:
        _named[f"{m}_mean"] = pd.NamedAgg(column=m, aggfunc="mean")
        _named[f"{m}_std"] = pd.NamedAgg(column=m, aggfunc="std")
    df.groupby(["K", "classifier"]).agg(**_named).reset_index().to_csv(
        SUMMARY_PATH, index=False
    )


write_summary(all_results)
print(f"Saved summary: {SUMMARY_PATH}")

# ── Diagnostic figures ──
print("Generating diagnostic figures...")

# Figure 1: Impact point selection frequency heatmap
freq_map = np.zeros((K_MAX, p))
for sname in split_names:
    if sname in impacts_store:
        pts = impacts_store[sname]
        for rank, j in enumerate(pts):
            freq_map[: rank + 1, j] += 1.0
freq_map /= len(split_names)

fig, ax = plt.subplots(figsize=(14, 3))
im = ax.imshow(
    freq_map, aspect="auto", origin="upper", cmap="Blues",
    extent=[wavelengths[0], wavelengths[-1], K_MAX + 0.5, 0.5]
)
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Top-K impact points selected")
ax.set_title("RKVS impact point selection frequency across 50 RSKF folds")
plt.colorbar(im, ax=ax, label="Fraction of folds")
fig.tight_layout()
fig.savefig(RESULTS_DIR / "rkvs_impact_points_stability.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {RESULTS_DIR / 'rkvs_impact_points_stability.png'}")

# Figure 2: Class-mean spectra with impact points
mu0_viz = F[y == 0].mean(axis=0)
mu1_viz = F[y == 1].mean(axis=0)
delta_viz = mu1_viz - mu0_viz

# Get most frequently selected wavelengths
freq_by_wl = freq_map.sum(axis=0)
top_indices = np.argsort(-freq_by_wl)[: K_MAX]

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(wavelengths, mu0_viz, color="tab:blue", label="Single (class 0)", lw=1.5)
ax.plot(wavelengths, mu1_viz, color="tab:orange", label="Binary (class 1)", lw=1.5)
ax.axhline(0, color="gray", lw=0.5, linestyle="--", alpha=0.5)

# Mark most frequently selected wavelengths
for j in top_indices:
    ax.axvline(wavelengths[j], color="red", alpha=0.2, lw=0.8)
ax.plot([], [], color="red", alpha=0.2, lw=0.8, label="Top-30 impact points (all 50 folds)")

ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("L2-normalised flux")
ax.set_title("Class-mean spectra with RKVS impact point frequency")
ax.legend(loc="best")
fig.tight_layout()
fig.savefig(RESULTS_DIR / "rkvs_mean_spectra_with_impacts.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {RESULTS_DIR / 'rkvs_mean_spectra_with_impacts.png'}")

print("\nDone! Next: open 18_rkvs_results.ipynb for analysis.")
