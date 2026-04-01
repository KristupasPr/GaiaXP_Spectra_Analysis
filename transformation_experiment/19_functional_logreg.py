#!/usr/bin/env python
"""
19 — Functional Logistic Regression on L2-normalised sampled spectra (RSKF)

Penalised logistic regression directly on 343-d L2-normalised calibrated
Gaia XP spectra for binary vs. single hot subdwarf classification.

Model:
    log P(binary|X) / P(single|X) = α + Σ_j β(λ_j) · X(λ_j) · Δλ

Key output: the weight function β(λ) — a discrimination curve showing which
wavelength regions drive the classification. L2 penalty yields a smooth β(λ);
L1 penalty yields a sparse β(λ) highlighting the most discriminative bins.

Methodology:
  - LogisticRegressionCV with inner 5-fold stratified CV, scoring=roc_auc
  - Cs = logspace(-4, 4, 10), class_weight='balanced'
  - StandardScaler fit per outer fold (no data leakage)
  - Youden threshold on train-set predict_proba (consistent with 12/14/17)
  - 10×5 RSKF outer CV from shared splits_rskf.json

Resume: skips (split, penalty_name) already in the results CSV.

Outputs (results/functional_logreg/):
  - functional_logreg_results.csv      per-fold metrics
  - beta_vectors_L2_ridge.npy          (50, 343) β(λ) curves
  - beta_vectors_L1_lasso.npy          (50, 343) β(λ) curves
  - beta_lambda.png / .pdf             thesis figure
  - mean_spectral_difference_l2.png    diagnostic
  - summary.txt                        aggregated stats

References:
  Araki et al. (2009); Wang, Huang & Cao (2024, WIREs Comp. Stat., §4.2)

Usage:
    python 19_functional_logreg.py --smoke   # rep0 only (5 folds)
    python 19_functional_logreg.py           # full 50 folds
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

N_JOBS = int(os.environ.get("FUNCTIONAL_LOGREG_N_JOBS", "-1"))

# Optional: override data paths with .npy files
SPECTRA_PATH = None
LABELS_PATH = None
WAVELENGTHS_PATH = None

CS_GRID = np.logspace(-4, 4, 10)
INNER_CV = 5
L1_NONZERO_TOL = 1e-6

PENALTIES = [
    {"name": "L2_ridge", "penalty": "l2", "solver": "lbfgs"},
    {"name": "L1_lasso", "penalty": "l1", "solver": "liblinear"},
]

REPRESENTATION_NAME = "sampled_spectra_L2"

parser = argparse.ArgumentParser(description="Functional LR on L2 XP spectra (RSKF)")
parser.add_argument("--smoke", action="store_true", help="Only rep0_* (5 folds)")
args = parser.parse_args()
SMOKE = args.smoke

EXPERIMENT_DIR = Path.cwd() if Path("data").exists() else Path("transformation_experiment")
DATA_DIR = EXPERIMENT_DIR / "data"
OUTPUT_DIR = EXPERIMENT_DIR / "results" / "functional_logreg"
POC_DIR = Path("transformation_poc") if Path("transformation_poc").exists() else Path("..") / "transformation_poc"
SPLITS_PATH = DATA_DIR / "splits_rskf.json"
RESULTS_CSV = OUTPUT_DIR / "functional_logreg_results.csv"
SUMMARY_TXT = OUTPUT_DIR / "summary.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Experiment dir:", EXPERIMENT_DIR.resolve())
print("Data dir:      ", DATA_DIR.resolve())
print("Output dir:    ", OUTPUT_DIR.resolve())
print("POC dir:       ", POC_DIR.resolve())


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def split_sort_key(k: str) -> tuple[int, int]:
    rep = int(k.split("_")[0].replace("rep", ""))
    fold = int(k.split("_")[1].replace("fold", ""))
    return (rep, fold)


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
    out = {
        "threshold": threshold,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "accuracy": acc,
        "f1": f1,
        "youden_j": youden,
        "brier": brier_score_loss(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out["roc_auc"] = float("nan")
    out["pr_auc"] = average_precision_score(y_true, y_prob)
    return out


def mean_inner_cv_roc_auc_at_best_c(clf: LogisticRegressionCV) -> float:
    """Extract mean inner-CV ROC-AUC at the selected best C."""
    sc = clf.scores_
    if sc is None:
        return float("nan")
    if isinstance(sc, dict):
        arr = np.asarray(next(iter(sc.values())))
    elif hasattr(sc, "ndim") and sc.ndim == 3:
        arr = sc[0]
    else:
        arr = np.asarray(sc)
    c_grid = np.asarray(clf.Cs_)
    best_c = float(clf.C_[0])
    j = int(np.argmin(np.abs(c_grid - best_c)))
    return float(np.mean(arr[:, j]))


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

def load_xy_wavelengths() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load L2-normalised spectra, labels, and wavelength grid."""
    if SPECTRA_PATH is not None and LABELS_PATH is not None:
        X = np.load(SPECTRA_PATH)
        y = np.load(LABELS_PATH)
        if WAVELENGTHS_PATH is None:
            raise ValueError("WAVELENGTHS_PATH required when using npy spectra")
        wavelengths = np.load(WAVELENGTHS_PATH)
        return X, y, wavelengths

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

    # L2-normalise: divide by row norm, leave near-zero rows as zero vectors
    row_norm = np.linalg.norm(F_raw, axis=1, keepdims=True)
    X = np.divide(F_raw, row_norm, out=np.zeros_like(F_raw), where=row_norm > 1e-20)

    return X, y, wavelengths


# ═══════════════════════════════════════════════════════════════════════
# Single fold worker
# ═══════════════════════════════════════════════════════════════════════

def run_one_split(X, y, train_idx, test_idx, penalty, solver, penalty_name):
    """Run LogisticRegressionCV + evaluation for one (split, penalty)."""
    t0 = time.time()

    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    # StandardScaler fit on training fold only
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # Inner CV for C selection
    inner_cv = StratifiedKFold(n_splits=INNER_CV, shuffle=True, random_state=RANDOM_STATE)
    lr_cv = LogisticRegressionCV(
        Cs=CS_GRID,
        cv=inner_cv,
        penalty=penalty,
        solver=solver,
        class_weight="balanced",
        scoring="roc_auc",
        max_iter=2000,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    lr_cv.fit(X_tr_s, y_tr)

    best_c = float(lr_cv.C_[0])
    best_cv_roc_auc = mean_inner_cv_roc_auc_at_best_c(lr_cv)

    # Youden threshold on train-set predictions (consistent with 12/14/17)
    y_prob_tr = lr_cv.predict_proba(X_tr_s)[:, 1]
    thr = pick_youden_threshold(y_tr, y_prob_tr)

    # Evaluate on test set
    y_prob_te = lr_cv.predict_proba(X_te_s)[:, 1]
    metrics = evaluate(y_te, y_prob_te, thr)

    # Extract β(λ) coefficients
    coef = lr_cv.coef_[0]
    n_nonzero = int(np.sum(np.abs(coef) > L1_NONZERO_TOL)) if penalty == "l1" else int(X.shape[1])

    row = {
        **metrics,
        "representation": REPRESENTATION_NAME,
        "n_features": X.shape[1],
        "classifier": penalty_name,
        "penalty_name": penalty_name,
        "best_C": best_c,
        "best_cv_roc_auc": best_cv_roc_auc,
        "n_nonzero_coefs": n_nonzero,
        "time_seconds": time.time() - t0,
    }
    return row, coef.copy()


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    X, y, wavelengths = load_xy_wavelengths()
    assert X.ndim == 2 and X.shape[1] == len(wavelengths), "Shape mismatch"

    # Verify L2 normalisation
    row_n = np.linalg.norm(X, axis=1)
    ok = row_n > 1e-10
    if ok.any() and not np.allclose(row_n[ok], 1.0, rtol=1e-4, atol=1e-4):
        raise AssertionError("Non-degenerate spectra must have L2 norm ~ 1")

    print(f"N = {X.shape[0]}, p = {X.shape[1]}; binary fraction = {(y == 1).mean():.1%}")

    # Load shared CV splits
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    if SMOKE:
        splits = {k: v for k, v in splits.items() if k.startswith("rep0_")}
        print(f"SMOKE: {len(splits)} splits")
    else:
        print(f"Loaded {len(splits)} RSKF splits")

    split_names = sorted(splits.keys(), key=split_sort_key)

    # Resume: load completed cells
    completed = set()
    existing_rows = []
    if RESULTS_CSV.exists():
        df_old = pd.read_csv(RESULTS_CSV)
        existing_rows = df_old.to_dict("records")
        for r in existing_rows:
            completed.add((str(r["split"]), str(r["penalty_name"])))
        print(f"Resume: {len(completed)} cells done, will skip them.")

    new_rows = []
    beta_by_penalty = {p["name"]: [] for p in PENALTIES}

    t_start = time.time()
    total_cells = len(split_names) * len(PENALTIES)
    done_count = len(completed)

    for sname in split_names:
        tr = np.array(splits[sname]["train"], dtype=int)
        te = np.array(splits[sname]["test"], dtype=int)
        rep, fold = split_sort_key(sname)

        for cfg in PENALTIES:
            pname = cfg["name"]
            if (sname, pname) in completed:
                continue

            row, beta = run_one_split(X, y, tr, te, cfg["penalty"], cfg["solver"], pname)
            row["split"] = sname
            row["repeat"] = rep
            row["fold"] = fold
            new_rows.append(row)
            beta_by_penalty[pname].append(beta)
            done_count += 1

            elapsed = time.time() - t_start
            cells_this_run = done_count - len(completed)
            remaining = total_cells - done_count
            eta = (elapsed / cells_this_run) * remaining if cells_this_run > 0 else 0

            print(
                f"  [{done_count:3d}/{total_cells}]  {sname}  {pname}  "
                f"PR-AUC={row['pr_auc']:.4f}  ROC-AUC={row['roc_auc']:.4f}  "
                f"best_C={row['best_C']:.4g}  ({row['time_seconds']:.1f}s)  "
                f"ETA={eta/60:.1f}min",
                flush=True,
            )

    if not new_rows and not existing_rows:
        print("No results produced.")
        return

    # Save results CSV
    if new_rows:
        df_new = pd.DataFrame(new_rows)
        df_all = pd.concat([pd.DataFrame(existing_rows), df_new], ignore_index=True) if existing_rows else df_new
        df_all.to_csv(RESULTS_CSV, index=False)
        print(f"\nSaved {len(df_all)} rows -> {RESULTS_CSV}")
    else:
        df_all = pd.DataFrame(existing_rows)
        print("\nNo new folds; CSV unchanged.")

    # Reload for consistent ordering
    df_all = pd.read_csv(RESULTS_CSV)
    n_splits_expected = len(split_names)

    # Save β(λ) vectors
    for cfg in PENALTIES:
        pname = cfg["name"]
        sub = df_all[df_all["penalty_name"] == pname]
        if len(sub) != n_splits_expected:
            print(f"Skip beta_vectors_{pname}.npy: {len(sub)}/{n_splits_expected} rows")
            continue
        if len(beta_by_penalty[pname]) != n_splits_expected:
            print(f"Skip beta .npy for {pname}: need fresh run without resume for coefficients")
            continue
        order = {s: i for i, s in enumerate(split_names)}
        sort_ix = np.argsort(sub["split"].map(order).to_numpy())
        arr = np.stack([beta_by_penalty[pname][int(i)] for i in sort_ix], axis=0)
        np.save(OUTPUT_DIR / f"beta_vectors_{pname}.npy", arr)
        print(f"Saved {arr.shape} -> beta_vectors_{pname}.npy")

    # ── β(λ) figure ──
    if all(len(beta_by_penalty[p["name"]]) == n_splits_expected for p in PENALTIES):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
        for ax, cfg in zip(axes, PENALTIES):
            pname = cfg["name"]
            betas = np.stack(beta_by_penalty[pname], axis=0)
            order = {s: i for i, s in enumerate(split_names)}
            sort_ix = np.argsort([order[s] for s in split_names])
            betas = betas[sort_ix]
            m, s = betas.mean(axis=0), betas.std(axis=0)

            ax.plot(wavelengths, m, color="navy", lw=1.5)
            ax.fill_between(wavelengths, m - s, m + s, alpha=0.2, color="steelblue")
            ax.axhline(0, color="gray", ls="--", lw=0.8)
            ax.axvline(700, color="red", ls=":", lw=0.8, label="700 nm")
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel(r"$\beta(\lambda)$ (scaled features)")
            ax.set_title(f"{pname}: discrimination curve")
            ax.legend(loc="best", fontsize=8)

        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "beta_lambda.png", dpi=200, bbox_inches="tight")
        fig.savefig(OUTPUT_DIR / "beta_lambda.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved beta_lambda.png / .pdf")

        # Mean spectral difference overlay
        d = X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0)
        fig2, axb = plt.subplots(figsize=(10, 4))
        axb.plot(wavelengths, d, color="darkred", lw=1.2, label="mean binary - single (L2)")
        axb.axhline(0, color="gray", ls="--", lw=0.8)
        axb.set_xlabel("Wavelength (nm)")
        axb.set_ylabel("Delta flux (shape)")
        axb.set_title("Mean L2 spectrum difference (visualisation)")
        axb.legend()
        fig2.tight_layout()
        fig2.savefig(OUTPUT_DIR / "mean_spectral_difference_l2.png", dpi=200, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved mean_spectral_difference_l2.png")

    # ── Summary ──
    metric_cols = ["roc_auc", "pr_auc", "f1", "sensitivity", "specificity",
                   "precision", "youden_j", "best_cv_roc_auc"]
    lines = ["Functional logistic regression - mean +/- std by penalty\n"]
    for cfg in PENALTIES:
        pname = cfg["name"]
        sub = df_all[df_all["penalty_name"] == pname]
        if sub.empty:
            continue
        lines.append(f"\n{pname}  (n={len(sub)} folds)\n")
        for col in metric_cols:
            if col not in sub.columns:
                continue
            mu, sd = float(sub[col].mean()), float(sub[col].std())
            lines.append(f"  {col}: {mu:.4f} +/- {sd:.4f}\n")
    SUMMARY_TXT.write_text("".join(lines))
    print("\n" + "".join(lines))
    print(f"Wrote {SUMMARY_TXT}")

    elapsed_total = time.time() - t_start
    print(f"\nFinished in {elapsed_total/60:.1f} minutes.")


if __name__ == "__main__":
    main()
