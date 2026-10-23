# PGS zero-step Jacobian/VJP diagnostic (2026-09-15)

This read-only diagnostic used the frozen BND ON+CICA step-14999 checkpoints
for all four scenes. No optimizer step, parameter mutation, geometry change,
or renderer change was performed. For three deterministic training views per
scene, gradients were computed for:

- full underwater photometric loss, `L_full = ||pred - GT||²`;
- nuisance/medium path, `L_nuis = ||pred - rgb_object||²`;
- direct-object loss, `L_obj = ||rgb_object - GT||²`.

The nuisance VJP was compared with the full VJP per parameter group. The
shieldable fraction is the squared cosine between the two vectors.

| Parameter group | Full grad L2 | Nuisance grad L2 | Mean cosine | Shieldable fraction |
|---|---:|---:|---:|---:|
| features_dc | 2.33e-6 | 0 | 0.000 | 0.000 |
| features_rest | 9.02e-6 | 0 | 0.000 | 0.000 |
| means | 2.09e-3 | 1.32e-3 | −0.006 | 0.00015 |
| scales | 1.84e-5 | 5.43e-4 | −0.017 | 0.00823 |
| quats | 2.81e-5 | 8.50e-4 | 0.015 | 0.0331 |
| opacities | 4.35e-6 | 9.83e-5 | −0.043 | 0.0107 |

The per-view raw records are in
`outputs/pgs_zero_step_vjp_20260915_v2/*/pgs_vjp.json`.

## Interpretation

The current renderer cleanly separates Gaussian appearance from the nuisance
path: the medium/tail residual produces zero VJP on both DC and SH residual
parameters in all 12 probes. Geometry overlap is also small on average (about
3.3% for quaternion parameters and below 1.1% for the other groups). Thus a
PGS module that shields appearance gradients from medium-induced error has no
measurable target at the current checkpoint. Applying such a projection would
mostly remove valid object gradients and is not justified by this diagnostic.

## Decision

PGS training is rejected for the current model and data. The next candidate
should be VAB only after a frozen angular-support audit demonstrates a stable
relationship between view-dependent support and held-out error. CVC remains
deferred until a front-surface stability statistic is defined; simple visibility
counts are insufficient based on prior GRAC/GCRC/BRASH results.
