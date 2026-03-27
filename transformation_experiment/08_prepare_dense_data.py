#!/usr/bin/env python
"""
08 — Prepare Dense Polynomial Feature Datasets (n=1..25)

Generates L2-normalised Chebyshev and Legendre polynomial feature CSVs
for n_coeffs = 1, 2, ..., 25.  Skips files that already exist for
idempotent re-runs.

Source data (from transformation_poc/):
  - xp_sampled_spectra.csv — calibrated sampled spectra (2815 stars, 343 bins)

Outputs (to data/):
  - {basis}_{n}_L2.csv  (columns: source_id, y, c000 .. c{n-1})

Usage:
  python 08_prepare_dense_data.py            # generate all missing files
  python 08_prepare_dense_data.py --force     # regenerate even if files exist
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from clustertools.spectra.xp import l2_normalize
from clustertools.spectra.polynomial import fit_polynomial

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Generate dense polynomial features")
parser.add_argument("--force", action="store_true",
                    help="Regenerate files even if they already exist")
args = parser.parse_args()


# ── Configuration ──
EXPERIMENT_DIR = Path.cwd() if Path("data").exists() else Path("transformation_experiment")
POC_DIR = Path("transformation_poc") if Path("transformation_poc").exists() else Path("..") / "transformation_poc"
DATA_OUT = EXPERIMENT_DIR / "data"
DATA_OUT.mkdir(exist_ok=True)

BASES = ["chebyshev", "legendre"]
N_COEFFS_RANGE = range(1, 26)

print("Experiment dir:", EXPERIMENT_DIR.resolve())
print("POC dir:       ", POC_DIR.resolve())
print("Output dir:    ", DATA_OUT.resolve())
print(f"Bases:          {BASES}")
print(f"n_coeffs range: 1..25")
print()


# ── Load sampled spectra ──
spectra_path = POC_DIR / "xp_sampled_spectra.csv"
print(f"Loading spectra from {spectra_path.name} ...")
df_spectra = pd.read_csv(spectra_path)

wl_cols = [c for c in df_spectra.columns if c.startswith("wl_")]
wavelengths = np.array([float(c.split("_")[1]) for c in wl_cols])
source_ids = df_spectra["source_id"].values
labels = df_spectra["y"].values
spectra_matrix = df_spectra[wl_cols].to_numpy(dtype=np.float64)

print(f"  {spectra_matrix.shape[0]} stars, {spectra_matrix.shape[1]} wavelength bins")
print(f"  Wavelength range: {wavelengths[0]:.0f} – {wavelengths[-1]:.0f} nm")
print()


# ── Helpers ──
def fit_all_stars(wavelengths, spectra_matrix, basis_name, n_coeffs):
    """Fit polynomial to every star."""
    n_stars = spectra_matrix.shape[0]
    coeff_matrix = np.zeros((n_stars, n_coeffs), dtype=np.float64)
    r2_values = np.zeros(n_stars, dtype=np.float64)

    for i in range(n_stars):
        result = fit_polynomial(
            wavelengths, spectra_matrix[i],
            basis=basis_name, n_coeffs=n_coeffs,
        )
        coeff_matrix[i] = result["coefficients"][:n_coeffs]
        r2_values[i] = result["metrics"]["R2"]

    return coeff_matrix, r2_values


def build_feature_df(source_ids, labels, coeff_matrix, n_coeffs):
    """Assemble a feature DataFrame with source_id, y, and coefficient columns."""
    col_names = [f"c{i:03d}" for i in range(n_coeffs)]
    df = pd.DataFrame(coeff_matrix, columns=col_names)
    df.insert(0, "y", labels)
    df.insert(0, "source_id", source_ids)
    return df


# ── Generate features ──
generated = 0
skipped = 0

for basis in BASES:
    for n_coeffs in N_COEFFS_RANGE:
        l2_path = DATA_OUT / f"{basis}_{n_coeffs}_L2.csv"

        if l2_path.exists() and not args.force:
            print(f"  {basis:10s} n={n_coeffs:3d} — already exists, skipping")
            skipped += 1
            continue

        print(f"  {basis:10s} n={n_coeffs:3d} ... ", end="", flush=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coeff_raw, r2_values = fit_all_stars(
                wavelengths, spectra_matrix, basis.capitalize(), n_coeffs,
            )

        median_r2 = np.median(r2_values)

        df_feat_raw = build_feature_df(source_ids, labels, coeff_raw, n_coeffs)
        coeff_cols = [f"c{i:03d}" for i in range(n_coeffs)]
        df_feat_l2 = l2_normalize(df_feat_raw, coeff_cols=coeff_cols)
        df_feat_l2.to_csv(l2_path, index=False)

        print(f"R²={median_r2:.6f}  → {l2_path.name}")
        generated += 1

print(f"\nDone. Generated {generated} files, skipped {skipped} existing.")
