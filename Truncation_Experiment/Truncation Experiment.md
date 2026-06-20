# Truncation Experiments

This repository provides notebooks and result tables for evaluating coefficient truncation in Gaia DR3 XP binary-classification models. The central question is how model performance changes when the BP and RP coefficient vectors are shortened, either symmetrically or with independent truncation depths for each arm.

The workflow is organized into three stages:

1. Construction of the normalized Gaia XP coefficient dataset.
2. One-dimensional `K` truncation experiments, where the same number of BP and RP coefficients is retained.
3. Two-dimensional `55x55` truncation experiments, where BP and RP truncation depths are varied independently.

## Repository Structure

```text
README.md
requirements.txt
01_build_gaia_xp_coefficient_dataset.ipynb

preparations/
  VOSA_labels_training.csv

k55_experiments/
  <model>_k55_truncation.ipynb
  <model>_k55_truncation_out/
    truncation_<model>_raw.*
    truncation_<model>_summary_byK*.csv

55x55_experiments/
  combined_55x55_visuals.ipynb
  <model>_55x55_truncation.ipynb
  <model>_55x55_truncation_out/
    truncation_<model>_bp_rp_grid_raw.parquet
    truncation_<model>_bp_rp_grid_summary_by_pair.*

out_data/
  gaia_dr3_xp_c110_l2_binary.csv
  gaia_dr3_xp_c110_binary.npz
```

The `k55` notebooks cover `cnn`, `knn`, `lr`, `rf`, and `svm`. The `55x55` notebooks cover those models plus `lda` and `lightgbm`. Precomputed experiment tables are stored next to their corresponding notebooks in `*_truncation_out/` folders.

## Input Dataset

The starting supervised-label file is:

- `preparations/VOSA_labels_training.csv`

This table defines the classification target used throughout the repository. It contains Gaia DR3 source identifiers in `GaiaDR3` and binary VOSA labels in `VOSA`. The dataset notebook uses these source identifiers to retrieve or load Gaia DR3 XP continuous mean-spectrum coefficients, then joins the coefficients to the labels.

The dataset notebook builds the base coefficient table used by all truncation experiments:

- `out_data/gaia_dr3_xp_c110_l2_binary.csv`
- `out_data/gaia_dr3_xp_c110_binary.npz`

The CSV contains Gaia source identifiers, binary labels, and normalized XP coefficient columns `c000` through `c109`. The NPZ file stores the same feature matrix in array form for faster loading.

The notebook can use locally cached Gaia XP continuous mean-spectrum files when available. If the cache is incomplete, it queries Gaia@AIP and therefore requires valid Gaia Archive credentials. The legacy label-file location `data/VOSA_labels_training.csv` is also accepted, but `preparations/VOSA_labels_training.csv` is the repository location.

## K55 Experiments

The notebooks in `k55_experiments/` evaluate a single truncation depth `K`. For each run, the feature vector contains the first `K` BP coefficients and the first `K` RP coefficients.

These notebooks produce model-specific output folders:

```text
k55_experiments/svm_k55_truncation_out/
k55_experiments/rf_k55_truncation_out/
k55_experiments/lr_k55_truncation_out/
k55_experiments/knn_k55_truncation_out/
k55_experiments/cnn_k55_truncation_out/
```

The main reusable outputs are:

- `truncation_<model>_raw.parquet`
- `truncation_<model>_raw.csv`
- `truncation_<model>_summary_byK.csv`
- `truncation_<model>_summary_byK_long.csv`, when generated

The raw table stores one row per completed repeat and truncation depth. The summary tables aggregate metrics by `K`, including means and confidence intervals used for the metric plots.

## 55x55 Experiments

The notebooks in `55x55_experiments/` evaluate BP and RP truncation depths independently. Each point in the grid corresponds to one `(K_BP, K_RP)` coefficient pair.

These notebooks produce model-specific output folders:

```text
55x55_experiments/svm_55x55_truncation_out/
55x55_experiments/rf_55x55_truncation_out/
55x55_experiments/lr_55x55_truncation_out/
55x55_experiments/knn_55x55_truncation_out/
55x55_experiments/lda_55x55_truncation_out/
55x55_experiments/lightgbm_55x55_truncation_out/
55x55_experiments/cnn_55x55_truncation_out/
```

The main reusable outputs are:

- `truncation_<model>_bp_rp_grid_raw.parquet`
- `truncation_<model>_bp_rp_grid_summary_by_pair.parquet`
- `truncation_<model>_bp_rp_grid_summary_by_pair.csv`

The raw table stores repeat-level results for each BP/RP coefficient pair. It is stored as parquet because the full raw CSV exports exceed GitHub's regular file-size limit for several models. The summary table stores aggregated metrics by `(K_BP, K_RP)` and is the preferred file for plotting or comparing model surfaces. The model grids contain 100 repeats per coefficient pair except for the CNN grid, which covers the same 55x55 coefficient-pair grid with 7-8 repeats per pair because of higher training cost.

The notebook `combined_55x55_visuals.ipynb` builds combined heatmap, contour, surface, and k55 companion figures from the precomputed output tables. Its default four-panel maps compare Logistic Regression, KNN, SVM, and Random Forest. When figures are exported, the notebook writes them to `55x55_experiments/combined_55x55_visuals_out/figures/`.

## Running Order

Install the Python requirements before running the notebooks:

```bash
pip install -r requirements.txt
```

Run the dataset notebook before running any model experiments:

`01_build_gaia_xp_coefficient_dataset.ipynb`

After the dataset files exist, the notebooks in `k55_experiments/` and `55x55_experiments/` can be run in either order.

The experiment notebooks first try to load existing consolidated result tables from their output folders. Both experiment families use the same two-control pattern for each workflow:

- `k55`: `USE_PRECOMPUTED_K55_RESULTS = True` and `RUN_K55_SWEEP = False`
- `55x55`: `USE_PRECOMPUTED_BP_RP_GRID_RESULTS = True` and `RUN_BP_RP_GRID_SWEEP = False`

The first flag controls whether the notebook searches for an existing consolidated result table before fitting models. The second flag controls whether the notebook is allowed to start a new sweep. If no complete precomputed result table is available and the run flag is still `False`, the notebook reports the missing result and stops before any re-computation.
