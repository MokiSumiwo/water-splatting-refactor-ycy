# CVC front-surface stability audit (2026-09-15)

This read-only audit uses frozen BND ON+CICA step-14999 checkpoints. Training
views construct a per-Gaussian stability score from four detached quantities:
front-surface occurrence (fraction of visible views in the nearest 30% depth),
coefficient of variation of projected depth, coefficient of variation of
footprint radius, and cross-view Gaussian color standard deviation. No
optimizer step or parameter update is performed.

| Scene | Train views | Front occurrence mean | Depth CV | Footprint CV | Color std | Stability mean | Stability p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Curasao | 18 | 0.372 | 0.087 | 0.089 | 0.0066 | 0.321 | 0.094 |
| IUI3-RedSea | 25 | 0.478 | 0.119 | 0.091 | 0.0052 | 0.391 | 0.312 |
| JapaneseGradens-RedSea | 17 | 0.468 | 0.103 | 0.089 | 0.0094 | 0.392 | 0.343 |
| Panama | 18 | 0.448 | 0.105 | 0.098 | 0.0097 | 0.393 | 0.188 |

Held-out association between mean visible-Gaussian stability and RGB RMSE:

| Scene | Stability–RMSE Spearman rho | Support–RMSE rho |
|---|---:|---:|
| Curasao | −0.500 | +0.500 |
| IUI3-RedSea | +0.800 | −1.000 |
| JapaneseGradens-RedSea | +0.500 | −0.500 |
| Panama | −0.500 | +0.500 |

Per-view and per-scene records are in
`outputs/cvc_front_surface_audit_20260915/*/cvc_front_surface.json`.

## Interpretation

The stability-to-error direction changes across scenes. In IUI3 and
JapaneseGradens, views with higher stability can still have higher error,
while Curasao and Panama show the opposite tendency. The support-only control
also flips sign. The small held-out populations (3–4 views per scene) are
insufficient for a reliable causal or predictive rule.

## Decision

CVC training is rejected for the current protocol. A visibility-calibrated
appearance gate would require a larger outcome-blind split and a validated
front-surface definition before changing Gaussian appearance capacity. The
current BND ON best remains FSB; no PGS, VAB, or CVC module is promoted.
