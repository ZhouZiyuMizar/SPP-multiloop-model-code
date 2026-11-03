# SPP multi-loop model code (v1.2)

This repository provides the numerical models and analysis code used to study harmonic contributions and self-intersection structures in SPP trajectories under large-amplitude oscillatory shear (LAOS). It includes:

- the parallel multi-threshold viscoplastic model, dynamic-yielding model and reversible-elastic model;
- the SPP reconstruction, complex Fourier coefficients and self-intersection counting with an area criterion;
- the three counting measures (`N_onset`, `N_flow`, `N_cross`);
- numerical checks and the parameter scans used in the manuscript.

## Contents

- `source/spp_model.py` — model solver, SPP/Fourier analysis, counting measures, checks, scans.
- `configs/` — explicit parameters of the six manuscript model cases.
- `run_all.py` — runs `check`, `cases`, `sweep` and `harmonics`.
- `results/` — generated at run time in CSV and JSON formats.

## Requirements

Python 3.8+ with NumPy and SciPy.

```powershell
python -m pip install -r requirements.txt
```

## Reproduce

```powershell
python source/spp_model.py check --out results
python source/spp_model.py cases --out results
python source/spp_model.py sweep --out results
python source/spp_model.py harmonics --out results
```

or simply

```powershell
python run_all.py
```

Generated numerical results are written to `results/` as CSV and JSON files.

## Expected key values

- `groups_1`: N_onset = 1, N_flow = 1, N_cross = 0, R5 = 1.1139, work = 1.2134 Pa.
- `groups_2`: N_onset = 2, N_flow = 2, N_cross = 5, R5 = 4.1671, work = 0.6245 Pa.
- `groups_3`: N_onset = 3, N_flow = 1, N_cross = 12, R5 = 0.9613, work = 1.0329 Pa.
- `groups_4`: N_onset = 4, N_flow = 1, N_cross = 19, R5 = 0.6913, work = 0.8810 Pa.
- `recovery_two`: N_onset = 2, N_flow = 1, N_cross = 5, R5 = 1.5038, work = 0.8509 Pa.
- `elastic_control`: N_onset = 0, N_flow = 0, N_cross = 3, work = 1.4e-15 Pa.
- Verification (dynamic model, yield strains 0.20/0.30, breakdown 1.0): refinement error 4.55e-7 Pa, N_cross = 5.

## Notes

- All calculations are synthetic; no experimental data are used.
- The default `Config` values correspond to the dynamic-yielding model used in the manuscript (`yield_strain = (0.20, 0.30)`, `breakdown = 1.0`, `ymin_fraction = 0.35`).

## Single configuration

Any configuration JSON in `configs/` can be evaluated directly:

```powershell
python source/spp_model.py case --config configs/recovery_two.json --out results
```
