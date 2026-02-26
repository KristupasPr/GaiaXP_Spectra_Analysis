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

Coordinates matching/

  - __init__.py: paketo inicializacija, leidžia naudoti astroflow kaip Python modulį;
  - gaia_tap.py: funkcijos užklausoms į Gaia DR3 TAP servisą (coordinates cross-matching);
  - cli_tap.py: komandų eilutės sąsaja darbui su gaia_tap moduliu (užklausų vykdymas iš terminalo).

  analizė pagal Viscasillas Vázquez et al. (2024);

