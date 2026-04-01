#!/usr/bin/env python
"""
21 — Functional SVM on L2-normalised sampled spectra (RSKF)

SVM with custom functional kernels computed directly on the 343-d
L2-normalised calibrated Gaia XP spectra for binary vs. single hot
subdwarf classification.

Unlike standard SVM on finite coefficient vectors (12, 05), this script
uses precomputed kernels that respect the L2 functional geometry and
incorporate spectral smoothness (derivatives).

Functional kernels:
  1. L2_linear    — functional inner product: K(Xi,Xj) = ∫ Xi(t)Xj(t) dt
  2. L2_rbf       — Gaussian functional kernel: exp(-γ · ∫ (Xi(t)-Xj(t))² dt)
  3. deriv1_rbf   — 1st-derivative RBF: exp(-γ · ∫ (Xi'(t)-Xj'(t))² dt)
  4. sobolev_rbf  — combined (Sobolev-type): exp(-γ · [α·L2² + (1-α)·deriv²])

Methodology:
  - SVC(kernel='precomputed', class_weight='balanced')
  - C tuned via inner 3-fold stratified CV, scoring=roc_auc
  - γ: median heuristic on training pairwise distances × {0.1, 0.5, 1, 2, 5}
  - Youden threshold on train-set decision_function (consistent with 12/14/17/19)
  - 10×5 RSKF outer CV from shared splits_rskf.json

Resume: skips (split, kernel_name) already in the results CSV.

Outputs (results/functional_svm/):
  - functional_svm_rskf_metrics.csv   per-fold metrics
  - functional_svm_summary.csv        aggregated by kernel type
  - summary.txt                       text summary

References:
  Rossi & Villa (2006); Ferraty & Vieu (2003);
  Wang, Huang & Cao (2024, WIREs Comp. Stat., §4.5)

Usage:
    python 21_functional_svm.py --smoke   # rep0 only (5 folds)
    python 21_functional_svm.py           # full 50 folds
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Sobolev kernel weight for L2 vs derivative components
SOBOLEV_ALPHA = 0.5

# C grid for inner CV
C_GRID = np.logspace(-2, 3, 8)

# γ multipliers of the median heuristic
GAMMA_MULTIPLIERS = [0.1, 0.5, 1.0, 2.0, 5.0]

INNER_CV_FOLDS = 3

parser = argparse.ArgumentParser(description="Functional SVM on L2 XP spectra (RSKF)")
parser.add_argument("--smoke", action="store_true", help="Only rep0_* (5 folds)")
args = parser.parse_args()
SMOKE = args.smoke

EXPERIMENT_DIR = Path.cwd() if Path("data").exists() else Path("transformation_experiment")
DATA_DIR = EXPERIMENT_DIR / "data"
OUTPUT_DIR = EXPERIMENT_DIR / "results" / "functional_svm"
POC_DIR = Path("transformation_poc") if Path("transformation_poc").exists() else Path("..") / "transformation_poc"
SPLITS_PATH = DATA_DIR / "splits_rskf.json"
RESULTS_CSV = OUTPUT_DIR / "functional_svm_rskf_metrics.csv"
SUMMARY_CSV = OUTPUT_DIR / "functional_svm_summary.csv"
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


def normalize_scores(scores_te: np.ndarray, scores_tr: np.ndarray) -> np.ndarray:
    """Map scores to [0, 1] using min/max from training fold."""
    lo, hi = float(scores_tr.min()), float(scores_tr.max())
    if hi == lo:
        return np.full_like(scores_te, 0.5, dtype=np.float64)
    return np.clip((scores_te - lo) / (hi - lo), 0.0, 1.0).astype(np.float64)


def evaluate(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    """Compute all metrics at a given threshold."""
    y_pred = (y_score >= threshold).astype(int)
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
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        out["roc_auc"] = float("nan")
    out["pr_auc"] = average_precision_score(y_true, y_score)
    return out


# ═══════════════════════════════════════════════════════════════════════
# Functional kernel computations
# ═══════════════════════════════════════════════════════════════════════

def compute_trapez_weights(wavelengths: np.ndarray) -> np.ndarray:
    """Trapezoidal quadrature weights for numerical integration over wavelength grid."""
    diffs = np.diff(wavelengths)
    w = np.zeros_like(wavelengths)
    w[0] = diffs[0] / 2.0
    w[-1] = diffs[-1] / 2.0
    w[1:-1] = (diffs[:-1] + diffs[1:]) / 2.0
    return w


def pairwise_L2_sq(A: np.ndarray, B: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pairwise squared L2 distances: ∫ (Ai(t) - Bj(t))² dt using trapezoidal weights.

    Returns (n_A, n_B) matrix.
    """
    # ||A - B||² = A²·w + B²·w - 2·A·diag(w)·B'
    Aw = A * np.sqrt(w)[np.newaxis, :]
    Bw = B * np.sqrt(w)[np.newaxis, :]
    A_sq = np.sum(Aw ** 2, axis=1)  # (n_A,)
    B_sq = np.sum(Bw ** 2, axis=1)  # (n_B,)
    cross = Aw @ Bw.T               # (n_A, n_B)
    D2 = A_sq[:, np.newaxis] + B_sq[np.newaxis, :] - 2.0 * cross
    np.maximum(D2, 0.0, out=D2)
    return D2


def functional_inner_product(A: np.ndarray, B: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pairwise functional inner product: ∫ Ai(t)·Bj(t) dt using trapezoidal weights.

    Returns (n_A, n_B) matrix.
    """
    Aw = A * w[np.newaxis, :]
    return Aw @ B.T


def finite_diff_first_derivative(X: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """Compute first derivatives via central finite differences.

    Returns array of shape (n, p-2) evaluated at interior points.
    """
    dw = wavelengths[2:] - wavelengths[:-2]  # (p-2,)
    dX = X[:, 2:] - X[:, :-2]                # (n, p-2)
    return dX / dw[np.newaxis, :]


def median_heuristic(D2_train: np.ndarray) -> float:
    """Compute γ = 1 / median(pairwise squared distances) on training set."""
    # Extract upper triangle (no diagonal)
    n = D2_train.shape[0]
    iu = np.triu_indices(n, k=1)
    d2_vals = D2_train[iu]
    med = float(np.median(d2_vals))
    if med < 1e-15:
        med = 1.0
    return 1.0 / med


def rbf_kernel_from_D2(D2: np.ndarray, gamma: float) -> np.ndarray:
    """Compute RBF kernel matrix from precomputed squared distances."""
    return np.exp(-gamma * D2)


# ═══════════════════════════════════════════════════════════════════════
# Kernel configurations
# ═══════════════════════════════════════════════════════════════════════

KERNEL_CONFIGS = [
    {"name": "L2_linear", "type": "linear"},
    {"name": "L2_rbf", "type": "rbf", "distance": "L2"},
    {"name": "deriv1_rbf", "type": "rbf", "distance": "deriv1"},
    {"name": "sobolev_rbf", "type": "rbf", "distance": "sobolev"},
]


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load L2-normalised spectra, labels, and wavelength grid."""
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

    row_norm = np.linalg.norm(F_raw, axis=1, keepdims=True)
    X = np.divide(F_raw, row_norm, out=np.zeros_like(F_raw), where=row_norm > 1e-20)

    return X, y, wavelengths


# ═══════════════════════════════════════════════════════════════════════
# Inner CV for hyperparameter selection
# ═══════════════════════════════════════════════════════════════════════

def inner_cv_select(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    wavelengths: np.ndarray,
    w_trap: np.ndarray,
    kernel_cfg: dict,
) -> tuple[float, float | None]:
    """Select best (C, γ) via inner stratified CV, return (best_C, best_gamma).

    For linear kernels, gamma is None.
    """
    inner_cv = StratifiedKFold(n_splits=INNER_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    ktype = kernel_cfg["type"]

    if ktype == "linear":
        # Just tune C; compute kernel once
        K_full = _compute_kernel_matrix(X_tr, X_tr, wavelengths, w_trap, kernel_cfg, gamma=None)
        best_score, best_c = -1.0, 1.0
        for C in C_GRID:
            scores = []
            for tr_i, val_i in inner_cv.split(X_tr, y_tr):
                K_tr = K_full[np.ix_(tr_i, tr_i)]
                K_val = K_full[np.ix_(val_i, tr_i)]
                svc = SVC(kernel="precomputed", C=C, class_weight="balanced", random_state=RANDOM_STATE)
                svc.fit(K_tr, y_tr[tr_i])
                dec = svc.decision_function(K_val)
                try:
                    scores.append(roc_auc_score(y_tr[val_i], dec))
                except ValueError:
                    scores.append(0.5)
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score, best_c = mean_score, C
        return best_c, None

    # RBF-type kernels: tune C and γ
    # Compute pairwise distances on full training set
    D2_full = _compute_D2(X_tr, X_tr, wavelengths, w_trap, kernel_cfg)
    gamma_base = median_heuristic(D2_full)

    best_score, best_c, best_gamma = -1.0, 1.0, gamma_base
    for gm in GAMMA_MULTIPLIERS:
        gamma = gamma_base * gm
        K_full = rbf_kernel_from_D2(D2_full, gamma)
        for C in C_GRID:
            scores = []
            for tr_i, val_i in inner_cv.split(X_tr, y_tr):
                K_tr = K_full[np.ix_(tr_i, tr_i)]
                K_val = K_full[np.ix_(val_i, tr_i)]
                svc = SVC(kernel="precomputed", C=C, class_weight="balanced", random_state=RANDOM_STATE)
                svc.fit(K_tr, y_tr[tr_i])
                dec = svc.decision_function(K_val)
                try:
                    scores.append(roc_auc_score(y_tr[val_i], dec))
                except ValueError:
                    scores.append(0.5)
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score, best_c, best_gamma = mean_score, C, gamma
    return best_c, best_gamma


def _compute_D2(
    A: np.ndarray,
    B: np.ndarray,
    wavelengths: np.ndarray,
    w_trap: np.ndarray,
    kernel_cfg: dict,
) -> np.ndarray:
    """Compute squared distance matrix for a given kernel config."""
    dist_type = kernel_cfg["distance"]
    if dist_type == "L2":
        return pairwise_L2_sq(A, B, w_trap)
    elif dist_type == "deriv1":
        wl_inner = wavelengths[1:-1]
        w_inner = compute_trapez_weights(wl_inner)
        dA = finite_diff_first_derivative(A, wavelengths)
        dB = finite_diff_first_derivative(B, wavelengths)
        return pairwise_L2_sq(dA, dB, w_inner)
    elif dist_type == "sobolev":
        D2_l2 = pairwise_L2_sq(A, B, w_trap)
        wl_inner = wavelengths[1:-1]
        w_inner = compute_trapez_weights(wl_inner)
        dA = finite_diff_first_derivative(A, wavelengths)
        dB = finite_diff_first_derivative(B, wavelengths)
        D2_deriv = pairwise_L2_sq(dA, dB, w_inner)
        return SOBOLEV_ALPHA * D2_l2 + (1.0 - SOBOLEV_ALPHA) * D2_deriv
    else:
        raise ValueError(f"Unknown distance type: {dist_type}")


def _compute_kernel_matrix(
    A: np.ndarray,
    B: np.ndarray,
    wavelengths: np.ndarray,
    w_trap: np.ndarray,
    kernel_cfg: dict,
    gamma: float | None,
) -> np.ndarray:
    """Compute kernel matrix for a given config."""
    ktype = kernel_cfg["type"]
    if ktype == "linear":
        return functional_inner_product(A, B, w_trap)
    elif ktype == "rbf":
        D2 = _compute_D2(A, B, wavelengths, w_trap, kernel_cfg)
        return rbf_kernel_from_D2(D2, gamma)
    else:
        raise ValueError(f"Unknown kernel type: {ktype}")


# ═══════════════════════════════════════════════════════════════════════
# Single fold worker
# ═══════════════════════════════════════════════════════════════════════

def run_one_split(
    X: np.ndarray,
    y: np.ndarray,
    wavelengths: np.ndarray,
    w_trap: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    kernel_cfg: dict,
) -> dict:
    """Run functional SVM for one (split, kernel) combination."""
    t0 = time.time()

    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    # Inner CV for hyperparameter selection
    best_c, best_gamma = inner_cv_select(X_tr, y_tr, wavelengths, w_trap, kernel_cfg)

    # Compute kernel matrices for train and test
    K_train = _compute_kernel_matrix(X_tr, X_tr, wavelengths, w_trap, kernel_cfg, best_gamma)
    K_test = _compute_kernel_matrix(X_te, X_tr, wavelengths, w_trap, kernel_cfg, best_gamma)

    # Fit final SVM
    svc = SVC(kernel="precomputed", C=best_c, class_weight="balanced", random_state=RANDOM_STATE)
    svc.fit(K_train, y_tr)

    # Decision function scores
    dec_tr = svc.decision_function(K_train)
    dec_te = svc.decision_function(K_test)

    # Normalise to [0,1] using train reference, pick Youden threshold
    prob_tr = normalize_scores(dec_tr, dec_tr)
    prob_te = normalize_scores(dec_te, dec_tr)
    thr = pick_youden_threshold(y_tr, prob_tr)

    # Evaluate on test set (use normalised scores for threshold, raw for AUC)
    metrics = evaluate(y_te, prob_te, thr)

    row = {
        **metrics,
        "kernel_name": kernel_cfg["name"],
        "best_C": best_c,
        "best_gamma": best_gamma if best_gamma is not None else float("nan"),
        "n_support_vectors": int(svc.n_support_.sum()),
        "time_seconds": time.time() - t0,
    }
    return row


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    X, y, wavelengths = load_data()
    w_trap = compute_trapez_weights(wavelengths)
    print(f"N = {X.shape[0]}, p = {X.shape[1]}; binary fraction = {(y == 1).mean():.1%}")
    print(f"Wavelength range: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm")

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
            completed.add((str(r["split"]), str(r["kernel_name"])))
        print(f"Resume: {len(completed)} cells done, will skip them.")

    new_rows = []
    t_start = time.time()
    total_cells = len(split_names) * len(KERNEL_CONFIGS)
    done_count = len(completed)

    for sname in split_names:
        tr = np.array(splits[sname]["train"], dtype=int)
        te = np.array(splits[sname]["test"], dtype=int)
        rep, fold = split_sort_key(sname)

        for kcfg in KERNEL_CONFIGS:
            kname = kcfg["name"]
            if (sname, kname) in completed:
                continue

            row = run_one_split(X, y, wavelengths, w_trap, tr, te, kcfg)
            row["split"] = sname
            row["repeat"] = rep
            row["fold"] = fold
            new_rows.append(row)
            done_count += 1

            elapsed = time.time() - t_start
            cells_this_run = done_count - len(completed)
            remaining = total_cells - done_count
            eta = (elapsed / cells_this_run) * remaining if cells_this_run > 0 else 0

            print(
                f"  [{done_count:3d}/{total_cells}]  {sname}  {kname}  "
                f"PR-AUC={row['pr_auc']:.4f}  ROC-AUC={row['roc_auc']:.4f}  "
                f"C={row['best_C']:.3g}  γ={row['best_gamma']:.3g}  "
                f"nSV={row['n_support_vectors']}  ({row['time_seconds']:.1f}s)  "
                f"ETA={eta/60:.1f}min",
                flush=True,
            )

        # Atomic save after each split (all kernels for this split)
        if new_rows:
            df_new = pd.DataFrame(new_rows)
            if existing_rows:
                df_all = pd.concat([pd.DataFrame(existing_rows), df_new], ignore_index=True)
            else:
                df_all = df_new
            df_all.to_csv(RESULTS_CSV, index=False)

    if not new_rows and not existing_rows:
        print("No results produced.")
        return

    # Final save
    df_all = pd.read_csv(RESULTS_CSV)
    print(f"\nSaved {len(df_all)} rows -> {RESULTS_CSV}")

    # ── Summary ──
    metric_cols = ["roc_auc", "pr_auc", "f1", "sensitivity", "specificity",
                   "precision", "youden_j"]
    summary_rows = []
    lines = ["Functional SVM - mean +/- std by kernel\n"]
    for kcfg in KERNEL_CONFIGS:
        kname = kcfg["name"]
        sub = df_all[df_all["kernel_name"] == kname]
        if sub.empty:
            continue
        lines.append(f"\n{kname}  (n={len(sub)} folds)\n")
        srow = {"kernel_name": kname, "n_folds": len(sub)}
        for col in metric_cols:
            if col not in sub.columns:
                continue
            mu, sd = float(sub[col].mean()), float(sub[col].std())
            lines.append(f"  {col}: {mu:.4f} +/- {sd:.4f}\n")
            srow[f"{col}_mean"] = mu
            srow[f"{col}_std"] = sd
        summary_rows.append(srow)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(SUMMARY_CSV, index=False)
    SUMMARY_TXT.write_text("".join(lines))
    print("\n" + "".join(lines))
    print(f"Wrote {SUMMARY_TXT}")
    print(f"Wrote {SUMMARY_CSV}")

    elapsed_total = time.time() - t_start
    print(f"\nFinished in {elapsed_total/60:.1f} minutes.")


if __name__ == "__main__":
    main()
