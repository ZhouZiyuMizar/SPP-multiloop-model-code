# Data dictionary (model outputs)

Each `results/<case>/` folder contains:

- `metadata.json`: model parameters, summary counts/metrics and yield-onset events.
- `geometry_cutoff.csv`: harmonic-order scan of SPP reconstruction error and self-intersection counts.
- `harmonics.csv`: complex Fourier coefficients as G' and G'' components, amplitude and phase.
- `waveform.csv`: phase, strain, stress, transient moduli, dissipation and per-unit state variables.

Top-level outputs:

- `checks.json`: numerical verification report (PASS status, refinement error, cross counts).
- `cases_summary.csv`: N_onset, N_flow, N_cross, R5, R7, R9, C3, work and related metrics for the six manuscript cases.
- `flow_peak_sensitivity.csv`: N_flow versus peak-prominence and peak-separation criteria.
- `parameter_sweep.csv`: two-unit parameter grid (modulus fraction, yield-strain gap) and derived metrics.
- `harmonic_geometry_map.csv`: self-intersection counts on the third–fifth and third–seventh harmonic grids.
- `harmonic_transition_brackets.csv`: count-change brackets on the sampled grids.
- `fifth_harmonic_thresholds.csv`: exact and operational critical ratios for 65 phases.
- `fifth_harmonic_threshold_summary.csv`: exact and 1024-point operational bounds and their sensitivity to the area criterion.

Units: stress in Pa, strain dimensionless, phase in rad (degrees in `*_deg` columns), energy density in Pa (equivalent to J/m3).
