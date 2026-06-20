# Gaia-Spectrum-Analysis


---

Truncation Experiment/

Pilnas detalus aprašas pateiktas "Wiki" skyrelyje.

---

GaiaAIP/

  - 00_gaia_aip_access_example.ipynb: minimal authenticated Gaia@AIP TAP/SJS access example for XP continuous mean-spectrum coefficients;
  - 01_build_gaia_xp_coefficient_dataset.ipynb: example construction of the normalized Gaia DR3 XP coefficient CSV used by downstream classification experiments;

  (testuota su python 3.9.6 ir 3.11.9 versijomis (win, mac))

---


Coordinates matching/

  - __init__.py: paketo inicializacija, leidžia naudoti astroflow kaip Python modulį;
  - gaia_tap.py: funkcijos užklausoms į Gaia DR3 TAP servisą (coordinates cross-matching);
  - cli_tap.py: komandų eilutės sąsaja darbui su gaia_tap moduliu (užklausų vykdymas iš terminalo).

  analizė pagal Viscasillas Vázquez et al. (2024);
