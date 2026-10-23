# Frozen opacity response experiment

Code and complete per-tile results:
`research_runs/sarg_opacity_response_20261016/code/run.py`
and scene `results.json` files in that run directory.

## Protocol

Frozen v6 15k checkpoints; ordinary training views only; no model updates.
Reuse stride-three fitting/calibration/audit split. IUI3 uses the six previous
unprotected candidates. Panama is a separately labelled exploratory arm using
the top eight unprotected scores, bypassing the previous 0.003 score filter.
Thus this is not independent confirmation of the earlier selection procedure.

Native rasterizer opacity is multiplied by 1-r, for r=0, .1, .25, .5, 1.
No opacity-logit scaling. Parameters and optimizer state are untouched.
For each candidate/camera, fix the target far-blue/green region from the full
model. Fix local support from complete deletion (clear change >1e-5, 5x5
dilation), then use the identical support for every rate. Evaluate 32x32
tiles with >=16 support pixels, inherited 1% L1/MSE tolerances and floors/caps.
All tiles must pass. This run gates on local photometric safety; it does not
certify same-ray replacement or independent geometric safety at every rate.

Select the largest rate passing every fitting/calibration camera and producing
target signed mean RGB removal >=1e-5 in at least three fitting cameras. Audit
does not select rates. Check the cumulative selected intervention on all
cameras using its full-deletion union support. Here each scene selects only
one Gaussian, so the joint check is a singleton repetition, not evidence for
multi-Gaussian interaction safety. Empty support tiles are vacuous checks;
the minimum fitting effect prevents all-zero admission.

## Results

| Scene | Selected ID | Attenuation | Fit/calibration/audit local pass |
| --- | --- | --- | --- |
| Panama | 1038610 | 10% | 10/10, 3/3, 2/2 |
| IUI3 | 640168 | 10% | 16/16, 5/5, 4/4 |

Audit signed residual-region RGB removal fractions:
Panama 0.0092%, 0.0018%; IUI3 0.0035%, 0.0031%, 0.0030%, 0.0014%.
These are small causal RGB changes, not semantic water removal percentages.
Some IUI3 fitting views have slightly negative signed regional removal.
The IUI3 audit signed mean change (4.0e-6--9.4e-6) is below the fitting
usefulness threshold: safety transfers here, useful magnitude does not.

Across all candidate-camera interventions, local failures increase with rate:

| Rate | Panama failures /120 | IUI3 failures /150 |
| --- | --- | --- |
| 10% | 17 | 26 |
| 25% | 31 | 76 |
| 50% | 46 | 115 |
| 100% | 56 | 135 |

Zero attenuation is the no-change control. Both final model hashes match.

## Decision

Bounded opacity attenuation has a small safe photometric region, unlike full
deletion for these selected candidates. This supports investigation of
reversible attenuation budgets, but not continuous pressure until pruning.
Repeated 10% attenuation compounds and may leave the measured safe interval.
Original pruning is a topology action, not a geometric safety guarantee.

Do not launch fresh 15k training yet: selected effects are too small and IUI3
does not preserve minimum useful benefit in audit. Next test a larger pool
with a fixed total attenuation budget, repeated cumulative local checks and
minimum audit benefit as a reporting criterion, not a tuning loop. Then use a
short training pilot to measure rebound, contribution transfer and pruning.
Retain cumulative opacity caps relative to the pre-intervention model, record
pre/post-pruning safety, and stop pressure when local gates fail. Do not assume
that a safe 10% change can be repeated without revalidation.
