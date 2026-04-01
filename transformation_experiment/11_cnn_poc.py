#!/usr/bin/env python
"""
11 — 1D CNN POC (standalone script for overnight runs)

Equivalent to 11_cnn_poc.ipynb but runnable as a script.
Resume-safe: skips (representation, split) pairs already in the results CSV.

Usage:
    caffeinate -i python 11_cnn_poc.py          # prevent macOS sleep
    caffeinate -i python 11_cnn_poc.py 2>&1 | tee cnn_poc.log
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import optuna
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

EXPERIMENT_DIR = Path.cwd() if Path("data").exists() else Path("transformation_experiment")
DATA_DIR = EXPERIMENT_DIR / "data"
RESULTS_DIR = EXPERIMENT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CNN_RESULTS_PATH = RESULTS_DIR / "cnn_poc_results.csv"

device = torch.device("cpu")
print(f"PyTorch {torch.__version__}, device: {device}")
print(f"Data dir:    {DATA_DIR.resolve()}")
print(f"Results dir: {RESULTS_DIR.resolve()}")

ENSEMBLE_SIZE = 5


# ═══════════════════════════════════════════════════════════════════════
# 1. Splits & representations
# ═══════════════════════════════════════════════════════════════════════

with open(DATA_DIR / "splits_rskf.json") as f:
    all_splits = json.load(f)

splits = {k: v for k, v in all_splits.items() if k.startswith("rep0_")}
print(f"Using {len(splits)} folds (rep0 only)")

REPRESENTATIONS = []
for basis in ["chebyshev", "legendre"]:
    for n in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        REPRESENTATIONS.append({
            "name": f"{basis}_{n}_L2",
            "file": f"{basis}_{n}_L2.csv",
            "n_features": n,
        })

print(f"\n{len(REPRESENTATIONS)} representations:")
for r in REPRESENTATIONS:
    print(f"  {r['name']:25s} ({r['n_features']:3d} features)")


# ═══════════════════════════════════════════════════════════════════════
# 2. Model
# ═══════════════════════════════════════════════════════════════════════

class SpectralCNN(nn.Module):
    def __init__(self, n_features: int, n_filters: int = 32,
                 dropout: float = 0.3, n_layers: int = 2):
        super().__init__()
        k1 = min(5, n_features)
        k2 = min(3, n_features)

        layers = [
            nn.Conv1d(1, n_filters, kernel_size=k1, padding=k1 // 2),
            nn.BatchNorm1d(n_filters),
            nn.ReLU(),
        ]
        in_ch = n_filters
        for _ in range(n_layers - 1):
            out_ch = in_ch * 2
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=k2, padding=k2 // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
            ]
            in_ch = out_ch

        layers.append(nn.AdaptiveAvgPool1d(1))
        self.features = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.Linear(in_ch, in_ch // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_ch // 2, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(-1)
        return self.head(x).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════
# 3. Helpers
# ═══════════════════════════════════════════════════════════════════════

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


def _stratified_val_split(X, y, val_frac, rng):
    idx = rng.permutation(len(X))
    pos = idx[y[idx] == 1]
    neg = idx[y[idx] == 0]
    n_val_pos = max(1, int(len(pos) * val_frac))
    n_val_neg = max(1, int(len(neg) * val_frac))
    val_idx = np.concatenate([pos[:n_val_pos], neg[:n_val_neg]])
    tr_idx = np.concatenate([pos[n_val_pos:], neg[n_val_neg:]])
    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]


def to_tensor(X, y=None):
    t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    if y is not None:
        return t, torch.tensor(y, dtype=torch.float32)
    return t


def train_single_model(
    X_tr, y_tr, X_val, y_val, n_features,
    lr=1e-3, weight_decay=1e-4, max_epochs=200, patience=20,
    batch_size=64, noise_std=0.0, n_filters=32, dropout=0.3, n_layers=2,
    seed=42,
):
    torch.manual_seed(seed)
    X_tr_t, y_tr_t = to_tensor(X_tr, y_tr)
    X_val_t, y_val_t = to_tensor(X_val, y_val)

    n_pos = y_tr.sum()
    n_neg = len(y_tr) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)

    model = SpectralCNN(n_features, n_filters=n_filters, dropout=dropout, n_layers=n_layers)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(len(X_tr_t))
        for i in range(0, len(X_tr_t), batch_size):
            batch_idx = perm[i : i + batch_size]
            xb, yb = X_tr_t[batch_idx], y_tr_t[batch_idx]
            if noise_std > 0:
                xb = xb + torch.randn_like(xb) * noise_std
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_val_loss, epoch + 1


def hpo_and_ensemble_fold(
    X_all, y_all, train_idx, test_idx, n_features,
    n_hpo_trials=20, ensemble_size=ENSEMBLE_SIZE,
):
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_all[train_idx])
    X_te_scaled = scaler.transform(X_all[test_idx])
    y_tr = y_all[train_idx]
    y_te = y_all[test_idx]

    rng = np.random.RandomState(RANDOM_STATE)
    X_tr_inner, y_tr_inner, X_val, y_val = _stratified_val_split(
        X_tr_scaled, y_tr, val_frac=0.15, rng=rng,
    )

    def objective(trial):
        params = {
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
            "noise_std": trial.suggest_float("noise_std", 0.0, 0.1),
            "n_filters": trial.suggest_categorical("n_filters", [16, 32, 64]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "n_layers": trial.suggest_int("n_layers", 1, 3),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        }
        _, val_loss, _ = train_single_model(
            X_tr_inner, y_tr_inner, X_val, y_val, n_features,
            max_epochs=150, patience=15, seed=RANDOM_STATE, **params,
        )
        return val_loss

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_hpo_trials, timeout=120)
    best_params = study.best_trial.params

    models = []
    for i in range(ensemble_size):
        model, _, _ = train_single_model(
            X_tr_inner, y_tr_inner, X_val, y_val, n_features,
            max_epochs=200, patience=20, seed=RANDOM_STATE + i + 1, **best_params,
        )
        models.append(model)

    X_val_t = to_tensor(X_val)
    X_te_t = to_tensor(X_te_scaled)

    val_probs_list, te_probs_list = [], []
    for model in models:
        model.eval()
        with torch.no_grad():
            val_probs_list.append(torch.sigmoid(model(X_val_t)).numpy())
            te_probs_list.append(torch.sigmoid(model(X_te_t)).numpy())

    val_probs = np.mean(val_probs_list, axis=0)
    te_probs = np.mean(te_probs_list, axis=0)

    thr = pick_youden_threshold(y_val, val_probs)
    metrics = evaluate(y_te, te_probs, thr)
    metrics["best_params"] = str(best_params)
    metrics["n_hpo_trials"] = n_hpo_trials
    return metrics


# ═══════════════════════════════════════════════════════════════════════
# 4. Resume
# ═══════════════════════════════════════════════════════════════════════

completed = set()
if CNN_RESULTS_PATH.exists():
    df_prev = pd.read_csv(CNN_RESULTS_PATH)
    for _, row in df_prev.iterrows():
        completed.add((row["representation"], row["split"]))
    all_results = df_prev.to_dict("records")
    print(f"\nResume: {len(completed)} cells already done, will skip them.")
else:
    all_results = []

print(f"\nReady.\n")


# ═══════════════════════════════════════════════════════════════════════
# 5. Run experiment
# ═══════════════════════════════════════════════════════════════════════

total_cells = len(REPRESENTATIONS) * len(splits)
done_count = len(completed)
t_start = time.time()

for repr_cfg in REPRESENTATIONS:
    repr_file = DATA_DIR / repr_cfg["file"]
    if not repr_file.exists():
        print(f"SKIP — {repr_file.name} not found")
        continue

    pending = [
        s for s in splits.keys()
        if (repr_cfg["name"], s) not in completed
    ]
    if not pending:
        done_count_for_repr = len(splits)
        print(f"  {repr_cfg['name']:25s} — all folds done, skipping")
        continue

    df = pd.read_csv(repr_file)
    feat_cols = [c for c in df.columns if c not in ("source_id", "y")]
    X_all = df[feat_cols].to_numpy(dtype=np.float64)
    y_all = df["y"].to_numpy(dtype=int)

    print(f"\n{'═'*60}")
    print(f"  {repr_cfg['name']}  ({repr_cfg['n_features']} features)  [{len(pending)} folds pending]")
    print(f"{'═'*60}", flush=True)

    for split_name in pending:
        split_idx = splits[split_name]
        torch.manual_seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)

        train_idx = np.array(split_idx["train"])
        test_idx = np.array(split_idx["test"])

        t_fold = time.time()
        metrics = hpo_and_ensemble_fold(
            X_all, y_all, train_idx, test_idx,
            n_features=repr_cfg["n_features"],
            n_hpo_trials=20,
            ensemble_size=ENSEMBLE_SIZE,
        )
        metrics["representation"] = repr_cfg["name"]
        metrics["n_features"] = repr_cfg["n_features"]
        metrics["split"] = split_name
        all_results.append(metrics)
        done_count += 1

        elapsed = time.time() - t_start
        remaining = total_cells - done_count
        cells_this_run = done_count - len(completed)
        eta = (elapsed / cells_this_run) * remaining if cells_this_run > 0 else 0

        print(
            f"  [{done_count:3d}/{total_cells}]  {split_name}  "
            f"ROC-AUC={metrics['roc_auc']:.4f}  "
            f"Sens={metrics['sensitivity']:.4f}  "
            f"Prec={metrics['precision']:.4f}  "
            f"({time.time()-t_fold:.0f}s)  "
            f"ETA={eta/60:.1f}min",
            flush=True,
        )

    pd.DataFrame(all_results).to_csv(CNN_RESULTS_PATH, index=False)
    print(f"  >> Saved {len(all_results)} rows → {CNN_RESULTS_PATH.name}", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# 6. Summary
# ═══════════════════════════════════════════════════════════════════════

elapsed_total = time.time() - t_start
df_cnn = pd.DataFrame(all_results)
df_cnn.to_csv(CNN_RESULTS_PATH, index=False)

metric_cols = ["roc_auc", "pr_auc", "sensitivity", "specificity", "precision", "f1", "youden_j"]

cnn_summary = (
    df_cnn
    .groupby(["representation", "n_features"])[metric_cols]
    .agg(["mean", "std"])
)
cnn_summary.columns = [f"{col}_{stat}" for col, stat in cnn_summary.columns]
cnn_summary = cnn_summary.reset_index().sort_values("roc_auc_mean", ascending=False)

summary_path = RESULTS_DIR / "cnn_poc_summary.csv"
cnn_summary.to_csv(summary_path, index=False)

print(f"\n{'═'*60}")
print(f"  Finished in {elapsed_total/60:.1f} minutes")
print(f"  {len(df_cnn)} results → {CNN_RESULTS_PATH.name}")
print(f"  Summary → {summary_path.name}")
print(f"{'═'*60}")
print(f"\nTop 10 by ROC AUC:")
print(cnn_summary[["representation", "n_features", "roc_auc_mean", "roc_auc_std",
                    "sensitivity_mean", "precision_mean", "f1_mean"]].head(10).to_string(index=False))
