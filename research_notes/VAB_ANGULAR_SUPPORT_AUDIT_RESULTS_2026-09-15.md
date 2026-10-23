# VAB frozen angular-support audit (2026-09-15)

This is a read-only audit on the frozen BND ON+CICA step-14999 checkpoints.
No parameters, renderer settings, or checkpoints were changed. Three or four
official held-out views per scene were evaluated after constructing the
angle-only predictors; held-out RGB was not used to select a bound.

| Scene | Held-out views | nearest-angle rho vs RMSE | 3NN-angle rho vs RMSE |
|---|---:|---:|---:|
| Curasao | 3 | +0.500 | +1.000 |
| IUI3-RedSea | 4 | −0.200 | +0.200 |
| JapaneseGradens-RedSea | 3 | −0.500 | +1.000 |
| Panama | 3 | +0.500 | +1.000 |

Per-view angle, RMSE, transmission, accumulation, and visible-Gaussian
fractions are stored in
`outputs/vab_angular_support_audit_20260915/*/vab_angular_support.json`.

## Interpretation

Nearest-view angular novelty is not directionally consistent across scenes.
The apparent 3NN positive correlations are based on only 3–4 points per scene
and are therefore rank ties rather than reliable evidence. The diagnostic does
not establish that angularly novel views need a looser appearance bound, nor
that a view-adaptive bound would improve underwater rendering.

## Decision

VAB training is deferred. A valid VAB experiment would require a larger,
outcome-blind held-out split or leave-one-view-out support statistics and a
pre-registered bound rule independent of RGB error. The current evidence is
insufficient to justify changing BND capacity per view.

The next candidate remains CVC only if a front-surface stability measure can
be defined from training views without reusing simple visibility counts.
