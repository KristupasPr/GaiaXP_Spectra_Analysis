#!/usr/bin/env python
"""
12 — FPCA + classifiers on L2-normalised sampled spectra (data prep + models)

Loads calibrated flux from xp_sampled_spectra.csv; aligns labels via og_xp.csv
(only source_id + y — no XP coefficients used as features). Fits fold-wise SVD
and evaluates centroid / LR / SVM on RSKF splits.

Outputs (results/):
  - fpca_classifier_rskf_metrics.csv (per-fold PR/ROC-AUC; F1 & Youden J at Youden threshold on train)
  - fpca_classifier_rskf_pr_auc.csv (subset: split, J, classifier, pr_auc only — legacy)
  - fpca_classifier_summary.csv
  - fpca_vs_baselines_table.csv
  - fpca_*.png (diagnostic figures)

Exploration / extra plots: see 13_fpca_results.ipynb.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
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
from sklearn.svm import SVC  # noqa: E402

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

J_VALUES = [1, 2, 3, 5, 8, 10, 15, 20, 30]
N_BOOTSTRAP_VIZ = 8

parser = argparse.ArgumentParser(description="FPCA + centroid/LR/SVM on L2 spectra")
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
POC_DIR = Path("transformation_poc") if Path("transformation_poc").exists() else Path("..") / "transformation_poc"
RESULTS_DIR.mkdir(exist_ok=True)

print("Data dir:   ", DATA_DIR.resolve())
print("Results dir:", RESULTS_DIR.resolve())
print("POC dir:    ", POC_DIR.resolve())

# ── Load aligned spectra + labels (og_xp supplies only source_id / y) ──
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

# ── Figure: random raw spectra ──
rng = np.random.default_rng(RANDOM_STATE)
idx0 = rng.choice(np.where(y == 0)[0], size=min(N_BOOTSTRAP_VIZ, (y == 0).sum()), replace=False)
idx1 = rng.choice(np.where(y == 1)[0], size=min(N_BOOTSTRAP_VIZ, (y == 1).sum()), replace=False)

fig, ax = plt.subplots(figsize=(8, 4.5))
for i in idx0:
    ax.plot(wavelengths, F_raw[i], color="tab:blue", alpha=0.35, lw=0.8)
for i in idx1:
    ax.plot(wavelengths, F_raw[i], color="tab:orange", alpha=0.5, lw=0.8)
ax.plot([], [], color="tab:blue", label=f"single (n={len(idx0)})")
ax.plot([], [], color="tab:orange", label=f"binary (n={len(idx1)})")
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Calibrated flux")
ax.set_title("Random raw spectra per class (GaiaXPy sampled)")
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS_DIR / "fpca_sanity_raw_spectra.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Mean shape difference (L2) ──
mu0 = F[y == 0].mean(axis=0)
mu1 = F[y == 1].mean(axis=0)
delta = mu1 - mu0
fig, ax = plt.subplots(figsize=(8, 4))
ax.axhline(0, color="gray", lw=0.8)
ax.plot(wavelengths, delta, color="darkred", lw=1.5)
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Δ mean (L2-normalised spectra)")
ax.set_title("Mean shape difference: binary − single")
fig.tight_layout()
fig.savefig(RESULTS_DIR / "fpca_mean_shape_difference.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Full-sample SVD (visualisation only) ──
mu_all = F.mean(axis=0)
F_c = F - mu_all
_, S, Vt = np.linalg.svd(F_c, full_matrices=False)
ev = (S**2) / (n - 1)
evr = ev / ev.sum()
cum = np.cumsum(evr)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(np.arange(1, len(cum) + 1), cum, "-", color="steelblue")
for j in J_VALUES:
    if j <= len(cum):
        ax.axvline(j, color="gray", ls="--", alpha=0.4)
ax.set_xlabel("J (number of components)")
ax.set_ylabel("Cumulative explained variance ratio")
ax.set_title("FPCA on full sample (visualisation only)")
ax.set_xlim(0, min(60, len(cum)))
fig.tight_layout()
fig.savefig(RESULTS_DIR / "fpca_cumulative_variance_full.png", dpi=150, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
for k, ax in enumerate(axes.ravel()):
    ax.plot(wavelengths, Vt[k], color="black", lw=1.2)
    ax.set_ylabel(f"φ_{k+1}")
axes[1, 0].set_xlabel("Wavelength (nm)")
axes[1, 1].set_xlabel("Wavelength (nm)")
fig.suptitle("First four eigenspectra (full-sample SVD, viz. only)")
fig.tight_layout()
fig.savefig(RESULTS_DIR / "fpca_first_four_eigenspectra.png", dpi=150, bbox_inches="tight")
plt.close(fig)

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


def centroid_scores(xi_tr, y_tr, xi_te):
    mu_b = xi_tr[y_tr == 1].mean(axis=0)
    mu_s = xi_tr[y_tr == 0].mean(axis=0)
    return np.sum((xi_te - mu_s) ** 2, axis=1) - np.sum((xi_te - mu_b) ** 2, axis=1)


def normalize_scores_train_ref(scores_te: np.ndarray, scores_tr: np.ndarray) -> np.ndarray:
    """Map scores to [0, 1] using min/max from training fold (same idea as 05_classify_focused OOF fallback)."""
    lo, hi = float(scores_tr.min()), float(scores_tr.max())
    if hi == lo:
        return np.full_like(scores_te, 0.5, dtype=np.float64)
    return ((scores_te - lo) / (hi - lo)).astype(np.float64)


def pick_youden_threshold(y_true: np.ndarray, y_prob: np.ndarray, grid_size: int = 200) -> float:
    """Threshold in [0, 1] maximising Youden's J = sensitivity + specificity - 1 (see 05_classify_focused)."""
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
    """PR/ROC from test scores; F1, Youden, etc. at threshold chosen on train (Youden on min-max scores)."""
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

records = []

for sname in split_names:
    tr_idx = np.array(splits[sname]["train"], dtype=int)
    te_idx = np.array(splits[sname]["test"], dtype=int)
    F_tr, F_te = F[tr_idx], F[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    mu_train = F_tr.mean(axis=0)
    F_tr_c = F_tr - mu_train
    F_te_c = F_te - mu_train

    _, _, Vtf = np.linalg.svd(F_tr_c, full_matrices=False)

    for J in J_VALUES:
        Vj = Vtf[:J].T
        xi_tr = F_tr_c @ Vj
        xi_te = F_te_c @ Vj

        sc_c_tr = centroid_scores(xi_tr, y_tr, xi_tr)
        sc_c_te = centroid_scores(xi_tr, y_tr, xi_te)
        met_c = fold_metrics(y_te, sc_c_te, y_tr, sc_c_tr)

        lr = LogisticRegression(
            class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE
        )
        lr.fit(xi_tr, y_tr)
        score_lr_tr = lr.decision_function(xi_tr)
        score_lr_te = lr.decision_function(xi_te)
        met_lr = fold_metrics(y_te, score_lr_te, y_tr, score_lr_tr)

        svm = SVC(
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        svm.fit(xi_tr, y_tr)
        score_svm_tr = svm.decision_function(xi_tr)
        score_svm_te = svm.decision_function(xi_te)
        met_svm = fold_metrics(y_te, score_svm_te, y_tr, score_svm_tr)

        for clf, met in [
            ("centroid", met_c),
            ("lr", met_lr),
            ("svm", met_svm),
        ]:
            row = {"split": sname, "J": J, "classifier": clf, **met}
            records.append(row)

df_run = pd.DataFrame(records)
df_run.to_csv(RESULTS_DIR / "fpca_classifier_rskf_metrics.csv", index=False)
print("Saved", RESULTS_DIR / "fpca_classifier_rskf_metrics.csv")
df_run[["split", "J", "classifier", "pr_auc"]].to_csv(
    RESULTS_DIR / "fpca_classifier_rskf_pr_auc.csv", index=False
)
print("Saved", RESULTS_DIR / "fpca_classifier_rskf_pr_auc.csv (legacy columns)")

# ── Summary + baselines + comparison plot ──
summary_path = RESULTS_DIR / "focused_summary.csv"
df_sum = pd.read_csv(summary_path)


def baseline_row(mask):
    r = df_sum.loc[mask].sort_values("pr_auc_mean", ascending=False).iloc[0]
    return r


r_xp = baseline_row(
    (df_sum["representation"] == "og_xp_110") & (df_sum["classifier"] == "SVM_RBF")
)
ch_mask = (
    df_sum["representation"].str.startswith("chebyshev")
    & df_sum["representation"].str.endswith("_L2")
    & (df_sum["classifier"] == "SVM_RBF")
)
leg_mask = (
    df_sum["representation"].str.startswith("legendre")
    & df_sum["representation"].str.endswith("_L2")
    & (df_sum["classifier"] == "SVM_RBF")
)
r_ch = baseline_row(ch_mask)
r_leg = baseline_row(leg_mask)

m_svm_xp, s_svm_xp = r_xp["pr_auc_mean"], r_xp["pr_auc_std"]
m_ch, s_ch, k_ch, name_ch = r_ch["pr_auc_mean"], r_ch["pr_auc_std"], int(r_ch["n_features"]), r_ch["representation"]
m_leg, s_leg, k_leg, name_leg = r_leg["pr_auc_mean"], r_leg["pr_auc_std"], int(r_leg["n_features"]), r_leg["representation"]

print("Baselines (SVM RBF, focused experiment):")
print(f"  og_xp_110:  {m_svm_xp:.4f} ± {s_svm_xp:.4f}")
print(f"  {name_ch}: {m_ch:.4f} ± {s_ch:.4f} (K={k_ch})")
print(f"  {name_leg}: {m_leg:.4f} ± {s_leg:.4f} (K={k_leg})")

_named = {}
for m in METRIC_COLS:
    _named[f"{m}_mean"] = pd.NamedAgg(column=m, aggfunc="mean")
    _named[f"{m}_std"] = pd.NamedAgg(column=m, aggfunc="std")
df_agg = df_run.groupby(["J", "classifier"]).agg(**_named).reset_index()
df_agg.to_csv(RESULTS_DIR / "fpca_classifier_summary.csv", index=False)

colors = {"centroid": "#7b3294", "lr": "#008837", "svm": "#c2a5cf"}
labels_plot = {"centroid": "FPCA + Centroid", "lr": "FPCA + LR", "svm": "FPCA + SVM RBF"}


def plot_metric_vs_J(metric_key: str, ylabel: str, title: str, fname: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot_m = df_run.pivot_table(index="J", columns="classifier", values=metric_key, aggfunc="mean").reindex(
        J_VALUES
    )
    pivot_s = df_run.pivot_table(index="J", columns="classifier", values=metric_key, aggfunc="std").reindex(
        J_VALUES
    )
    for key in ["centroid", "lr", "svm"]:
        m = pivot_m[key].values
        s = pivot_s[key].values
        ax.plot(J_VALUES, m, "o-", color=colors[key], label=labels_plot[key], ms=4)
        ax.fill_between(J_VALUES, m - s, m + s, color=colors[key], alpha=0.15)

    col_mean = f"{metric_key}_mean"
    ax.axhline(
        r_xp[col_mean],
        color="black",
        ls="--",
        lw=1,
        label=f"XP 110-d + SVM ({r_xp[col_mean]:.3f})",
    )
    ax.axhline(
        r_ch[col_mean],
        color="tab:blue",
        ls=":",
        lw=1,
        label=f"Best Chebyshev L2 + SVM (K={k_ch})",
    )
    ax.axhline(
        r_leg[col_mean],
        color="tab:orange",
        ls=":",
        lw=1,
        label=f"Best Legendre L2 + SVM (K={k_leg})",
    )

    ax.set_xlabel("J (FPCA components)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xticks(J_VALUES)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


plot_metric_vs_J(
    "pr_auc",
    "PR-AUC (mean ± 1 SD over folds)",
    "FPCA representation vs. polynomial baselines",
    "fpca_pr_auc_vs_J.png",
)
plot_metric_vs_J(
    "roc_auc",
    "ROC-AUC (mean ± 1 SD over folds)",
    "FPCA vs. baselines — ROC-AUC",
    "fpca_roc_auc_vs_J.png",
)

# ── Results table (F1 & Youden at test set, threshold = Youden-optimal on train scores min-max scaled; cf. focused_summary) ──


def fmt_ms(series):
    m, s = series.mean(), series.std(ddof=1) if len(series) > 1 else 0.0
    return f"{m:.4f} ± {s:.4f}"


def baseline_fmt(r, key):
    return f"{r[key + '_mean']:.4f} ± {r[key + '_std']:.4f}"


rows = [
    {
        "Method": "Chebyshev + SVM",
        "Representation": "Fixed polynomial",
        "K or J": k_ch,
        "PR-AUC mean ± std": baseline_fmt(r_ch, "pr_auc"),
        "ROC-AUC mean ± std": baseline_fmt(r_ch, "roc_auc"),
        "Sensitivity mean ± std": baseline_fmt(r_ch, "sensitivity"),
        "Precision mean ± std": baseline_fmt(r_ch, "precision"),
        "F1 mean ± std": baseline_fmt(r_ch, "f1"),
        "Youden J mean ± std": baseline_fmt(r_ch, "youden_j"),
    },
    {
        "Method": "Legendre + SVM",
        "Representation": "Fixed polynomial",
        "K or J": k_leg,
        "PR-AUC mean ± std": baseline_fmt(r_leg, "pr_auc"),
        "ROC-AUC mean ± std": baseline_fmt(r_leg, "roc_auc"),
        "Sensitivity mean ± std": baseline_fmt(r_leg, "sensitivity"),
        "Precision mean ± std": baseline_fmt(r_leg, "precision"),
        "F1 mean ± std": baseline_fmt(r_leg, "f1"),
        "Youden J mean ± std": baseline_fmt(r_leg, "youden_j"),
    },
    {
        "Method": "Native XP + SVM",
        "Representation": "Hermite / BP+RP coeffs",
        "K or J": 110,
        "PR-AUC mean ± std": baseline_fmt(r_xp, "pr_auc"),
        "ROC-AUC mean ± std": baseline_fmt(r_xp, "roc_auc"),
        "Sensitivity mean ± std": baseline_fmt(r_xp, "sensitivity"),
        "Precision mean ± std": baseline_fmt(r_xp, "precision"),
        "F1 mean ± std": baseline_fmt(r_xp, "f1"),
        "Youden J mean ± std": baseline_fmt(r_xp, "youden_j"),
    },
]

for clf_key, label in [("centroid", "FPCA + Centroid"), ("lr", "FPCA + LR"), ("svm", "FPCA + SVM")]:
    sub = df_run[df_run["classifier"] == clf_key]
    best_j, best_m = None, -1.0
    for J in J_VALUES:
        m = sub[sub["J"] == J]["pr_auc"].mean()
        if m > best_m:
            best_m, best_j = m, J
    part = sub[sub["J"] == best_j]
    rows.append(
        {
            "Method": label,
            "Representation": "Data-adaptive (FPCA)",
            "K or J": best_j,
            "PR-AUC mean ± std": fmt_ms(part["pr_auc"]),
            "ROC-AUC mean ± std": fmt_ms(part["roc_auc"]),
            "Sensitivity mean ± std": fmt_ms(part["sensitivity"]),
            "Precision mean ± std": fmt_ms(part["precision"]),
            "F1 mean ± std": fmt_ms(part["f1"]),
            "Youden J mean ± std": fmt_ms(part["youden_j"]),
        }
    )

df_table = pd.DataFrame(rows)
print("\n", df_table.to_string(index=False))
df_table.to_csv(RESULTS_DIR / "fpca_vs_baselines_table.csv", index=False)
print("\nSaved", RESULTS_DIR / "fpca_vs_baselines_table.csv")
print("Done.")
