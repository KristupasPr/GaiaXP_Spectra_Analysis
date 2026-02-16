# Gaia-Spectrum-Analysis

Building Blocks/

  - gaia_aip_pavyzdinis.ipynb: šablonas sąsajai su Gaia AIP;

  - xpcoeff_feature_build_binary.ipynb: duomenų paruošimas binarinei klasifikacijai;

  - xpcoeff_feature_derivative.ipynb: spektrų išvestinių bazės koeficientų gavimas ir įtraukimas į dataset;

  - xpcoeff_feature_errors_snr.ipynb: paklaidų ir snr koeficientų gavimas ir įtraukimas į dataset;

  (testuota su python 3.9.6 ir 3.11.9 versijomis (win, mac))

---

Methods (Binary)/

Kai kurie metodai binarinių/singuliarių sistemų klasifikacijai. Dar neištobulinti, tik bendrai idėjai. Input'as tik koef. iš `xpcoeff_feature_build_binary.ipynb`

---

- gaia_api_spectra.ipynb: duomenų gavimas per Gaia@AIP TAP API (SQL užklausos / SJS pagal source_id) ir XP spektrų kalibravimas bei braižymas su gaiaxpy;

- cluster_analysis.ipynb: atviro spiečiau erdvinės struktūros analizė – tarpžvaigždinių atstumų metrikos ir PCA formos analizė pagal Viscasillas Vázquez et al. (2024);

