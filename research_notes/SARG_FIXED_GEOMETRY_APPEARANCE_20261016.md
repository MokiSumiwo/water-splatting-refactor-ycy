# Fixed-geometry appearance control

Artifacts: `research_runs/sarg_fixed_geometry_appearance_20261016/` with code,
scene results and audit clear/error/support PNGs. Frozen v6 final checkpoints;
8 Panama and 6 IUI3 candidates identical to the prior opacity response run.
Same stride-three ordinary-training split. No test views, parameter changes,
training or pruning. Audit is excluded from strength selection but these
training cameras have been inspected previously: not independent validation.

## Mechanism

Mean per-Gaussian view colour is estimated on fitting cameras. Compute its
Rec.709 luminance and a correction toward an achromatic colour with that
luminance. Apply the same correction vector to the candidate's view-dependent
RGB, with strengths 0, .1, .25, .5, 1 and clamp to [0,1]. No geometry or opacity
changes. This deliberately simple neutral-chroma direction is an appearance
control, not a recovered water-free colour or a physically identified two-part
representation. Luminance is preserved before clipping; clipping can change it.

Opacity arm uses alpha'=(1-strength)*alpha. These two strength coefficients
are not equal perturbation magnitudes or matched-benefit interventions.

Both arms use the identical fixed support: union of full opacity deletion and
full appearance intervention supports (>1e-5 mean absolute clear difference),
5x5 dilation. Same 32px tiles, >=16 support pixels and inherited 1% local L1/MSE
floors/caps. Target is the full model's supported far blue/green region.
Benefit is reduced positive B-R plus G-R chroma, not signed RGB removal used in
the previous run. Require >=1e-5 mean chroma reduction in >=3 fitting views and
every fitting/calibration tile passing. Audit only evaluates selected strengths.

## Results

Failed candidate-camera interventions (all ordinary training cameras):

| Scene/arm | 10% | 25% | 50% | 100% |
| --- | --- | --- | --- | --- |
| Panama opacity (/120) | 17 | 31 | 46 | 56 |
| Panama appearance (/120) | 27 | 43 | 61 | 66 |
| IUI3 opacity (/150) | 26 | 76 | 115 | 135 |
| IUI3 appearance (/150) | 20 | 42 | 66 | 118 |

No appearance candidate simultaneously meets fitting/calibration safety and
useful benefit. IUI3 opacity candidate 640168 at 10% passes selection and all
four audit cameras; mean audit chroma reductions range 9.39e-5--1.43e-4.
Panama has no candidate admitted under this experiment's chroma benefit rule.
This is not inconsistent with the prior opacity experiment's signed RGB rule:
the definition of useful benefit differs and is recorded explicitly.

Appearance interventions preserve rendered accumulation/depth within max
8.35e-7 (Panama), exactly in IUI3; medium RGB within 1.35e-7 (Panama), exactly
in IUI3. Model hashes unchanged. Zero-strength output control is unchanged.

## Interpretation

Keeping occlusion fixed isolates appearance changes and can reduce IUI3 local
photometric failures, but neutralizing colour is not sufficient for useful safe
decoupling. Panama worsens under this direction. Same-coefficient comparisons
cannot establish superior safety: appearance produces larger chroma change
than opacity at the same coefficient. Do not conclude that all appearance
decoupling fails, or that fixed geometry proves real geometry is correct.

No training warranted from this control. Next compare curves at matched target
chroma reduction or matched clear RGB perturbation, then estimate a constrained
appearance direction using multi-view fitting residuals and the existing medium
model instead of assuming neutrality. Retain an unchanged-appearance control and
independent calibration/audit separation. A physically coupled mechanism must
explain underwater observations; merely editing clear outputs is insufficient.
