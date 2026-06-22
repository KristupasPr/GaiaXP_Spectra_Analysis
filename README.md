# Gaia-Spectrum-Analysis


---

Truncation Experiment/

Koeficientų mažinimo analizė. Pilnas detalus aprašas pateiktas "Wiki" skyrelyje.

---

Feature_Set_Expansion/

Gaia XP požymių rinkinio išplėtimo analizė. Aplanke yra bazinio `c110` duomenų rinkinio sudarymas, išvestinių, paklaidų ir SNR požymių pridėjimas, bei SVM, RF, logistinės regresijos ir CNN modelių palyginimo sąsiuviniai.

---

GaiaAIP/

  - 00_gaia_aip_access_example.ipynb: minimalus autentifikuotos prieigos prie Gaia@AIP TAP/SJS pavyzdys;
  - 01_build_gaia_xp_coefficient_dataset.ipynb: Gaia DR3 XP koeficientų CSV failo, naudojamo tolimesniuose klasifikacijos eksperimentuose, sudarymo pavyzdys;

---


Coordinates matching/

  - __init__.py: paketo inicializacija, leidžia naudoti astroflow kaip Python modulį;
  - gaia_tap.py: funkcijos užklausoms į Gaia DR3 TAP servisą (coordinates cross-matching);
  - cli_tap.py: komandų eilutės sąsaja darbui su gaia_tap moduliu (užklausų vykdymas iš terminalo).

  analizė pagal Viscasillas Vázquez et al. (2024);
