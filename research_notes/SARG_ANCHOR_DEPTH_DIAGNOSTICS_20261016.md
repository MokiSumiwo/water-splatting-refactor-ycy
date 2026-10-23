# Anchor/depth/medium diagnostics

Frozen v6 checkpoints; all ordinary training cameras, no test cameras, no
parameter updates. Code and per-view results/images:
`research_runs/sarg_anchor_depth_diagnostics_20261016/`.
Audited six previously selected unprotected IUI3 candidates and the eight
highest-scoring protected Gaussians in each scene. Selection scores and split
come from the stride-three single-Gaussian screen. This is post-hoc diagnosis,
not a new independent validation or a new admission decision.

## Measurements

Recorded nearest-eight-anchor distances, exact protection ellipsoid distances,
projected anchor overlap within Gaussian radius, anchor relative depth gaps,
and supported-pixel rendered-depth agreement with candidate centre depth.
The latter uses the model's rendered depth, not external geometry truth.
For every projected intervention, recorded native omission ROI metrics,
medium replacement and clear-image change. Reserved-camera clear full/drop,
support and signed photometric error maps are saved per Gaussian.

## Findings

Panama's eight highest protected scores range from 0.00068 to 0.00165, below
the previous screening threshold 0.003. Protection alone therefore cannot
explain that screen's empty result. Some protected Gaussians pass average ROI
checks across views, but this is not evidence that they can be unprotected:
local safety, real geometry and semantic identity remain unresolved.

The six IUI3 unprotected candidates have median medium replacement fractions
0.886--1.118 on active views. Thus insufficient medium replacement is not
the only blocker. Candidates 725438, 640168 and 648001 pass average ROI
photometric checks in all 25 active cameras, while the previous screen still
rejects them locally. Candidates 504513, 646466 and 723919 have respectively
2, 3 and 5 failing average-ROI cameras as well.

Candidate 504513 agrees within 5% with rendered surface depth on about 90%
of affected supported pixels (view-averaged); the remaining five range roughly
0.3%--18.5%. They are not one uniform geometric failure type. The protected
IUI3 examples often have high surface-depth agreement, and some fail ROI
checks in many views (237698: 12/25), supporting continued conservative
protection. These correlations do not prove object/water labels.

## Decision and next mechanism

Do not remove protection or increase deletion strength. There is no useful
safe deletion set established for both scenes. Full deletion can expose
background surfaces and shift medium compositing even when medium replacement
is near one. A justified next frozen experiment is a bounded attenuation
response curve for the six IUI3 candidates (e.g. 0%, 10%, 25%, 50%, 100%
opacity attenuation), with cumulative local checks, fixed support reference,
and a minimum signed target-region benefit. Include high-scoring unprotected
Panama candidates using contribution quantiles as a *separate exploratory arm*,
not silently lowering the earlier fixed threshold. Keep calibration selection
separate from final audit; test on another checkpoint before training.

This experiment diagnoses rendering/photometric constraints; preservation of
actual object geometry remains unproven without independent geometric evidence.
