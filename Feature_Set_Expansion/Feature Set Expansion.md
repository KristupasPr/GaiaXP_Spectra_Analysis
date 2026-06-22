# Feature Set Expansion

This experiment studies whether Gaia DR3 XP information beyond the normalized XP coefficient vector improves binary classification. The baseline feature set is the 110-dimensional L2-normalized coefficient vector, denoted `c`, built from 55 BP and 55 RP continuous mean-spectrum coefficients.

The expansion adds three main feature families: derivative coefficients, errors of coefficients (coefficient uncertaintie), and coefficient signal-to-noise ratios. Gaia XP summary indicators are also retained in the fully expanded dataset as quality metadata.

## Baseline Coefficients

Gaia XP spectra are represented by coefficients in BP and RP basis-function systems. The baseline dataset concatenates the BP and RP coefficient vectors into one 110-dimensional feature vector:

```text
c000 ... c054  = BP coefficients
c055 ... c109  = RP coefficients
```

Each row is L2-normalized so that the classifier sees the shape of the coefficient vector rather than being dominated by the absolute scale of the spectrum. This `c110` representation is the reference point for all comparisons.

## Derivative Features

The derivative block `d` is intended to describe how the reconstructed XP spectrum changes across the sampled pseudo-wavelength grid. Conceptually, it is built in four steps:

1. Use GaiaXPy to evaluate the BP and RP XP basis functions on a shared sampling grid.
2. Reconstruct the BP and RP sampled spectra from their original XP coefficients.
3. Compute the numerical derivative of the sampled spectrum along the grid.
4. Project the derivative signal back into the same BP/RP coefficient-space size, producing 55 derivative BP coefficients and 55 derivative RP coefficients.

The resulting derivative vector is again L2-normalized and stored as:

```text
d000 ... d054  = BP derivative coefficients
d055 ... d109  = RP derivative coefficients
```


## Uncertainty Features

The uncertainty block `err` comes from the coefficient-error arrays stored with Gaia DR3 `xp_continuous_mean_spectrum`. For each source, Gaia provides BP and RP coefficient uncertainties corresponding to the XP coefficient arrays.

The notebook expands these arrays into per-coefficient columns:

```text
bp_err_00 ... bp_err_54
rp_err_00 ... rp_err_54
```

These features encode where the XP coefficient representation is better or worse constrained. A classifier can therefore learn not only from the spectrum-like representation, but also from the structure of the measurement uncertainty.

## Signal-to-Noise Ratio Features

The SNR block `snr` is derived from the XP coefficients and their coefficient uncertainties. For each BP/RP coefficient, the we compute an absolute coefficient-to-error ratio:

```text
snr = abs(coefficient) / (coefficient_error + epsilon)
```

The expanded columns are:

```text
bp_snr_00 ... bp_snr_54
rp_snr_00 ... rp_snr_54
```

These features describe how strongly each coefficient is measured relative to its uncertainty. They are useful because two spectra with similar coefficient values may differ substantially in how reliable those coefficients are.

## XP Summary Indicators

The fully expanded dataset also includes scalar fields from Gaia DR3 `xp_summary`, prefixed as `xps_`. These include quantities such as numbers of measurements/transits, rejected measurements, standard deviations, chi-squared values, contamination counts, and blended-transit counts.

They are included to preserve information about how the XP spectrum was observed and fit, and to support later quality-control or diagnostic analyses.

## Included Data Products

The prepared datasets are:

- `out_data/gaia_dr3_xp_c110_l2_binary.csv`: baseline `c` features.
- `out_data/gaia_dr3_xp_c110_d110_l2_binary.csv`: baseline `c` plus derivative `d` features.
- `out_data/gaia_dr3_xp_c110_d110_errors_snr_binary.csv`: expanded dataset containing `c`, `d`, uncertainty, SNR, and XP-summary columns.

The model-comparison notebooks use feature-block combinations of `c`, `d`, `err`, and `snr` and compare them against the baseline `c` representation using matched repeated runs.

## Notebooks

- `01_build_gaia_xp_coefficient_dataset.ipynb`: builds the baseline coefficient dataset from VOSA labels and Gaia XP coefficients.
- `02_add_gaia_xp_derivative_error_snr_features.ipynb`: adds derivative, uncertainty, SNR, and XP-summary features.
- `svm_feature_set_expansion.ipynb`, `rf_feature_set_expansion.ipynb`, `logreg_feature_set_expansion.ipynb`, `cnn_feature_set_expansion.ipynb`: summarize the precomputed model runs and regenerate comparison figures.

Gaia@AIP credentials are needed only when rebuilding the datasets from remote Gaia tables.
