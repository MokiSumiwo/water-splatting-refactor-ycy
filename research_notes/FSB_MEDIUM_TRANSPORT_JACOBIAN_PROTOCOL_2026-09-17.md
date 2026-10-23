# FSB Medium Transport Jacobian Diagnostic Protocol

This is a frozen, no-training diagnostic for the remaining far-field
underwater residual. It asks whether a small, low-dimensional change to the
medium transport can improve the underwater image while Gaussian positions,
opacity, colors, projection, ordering, support, and bounded appearance remain
fixed.

## Fixed source and controls

The source is the retained FSB+CICA `R0/C0a` step-14999 checkpoint for each
scene. The production forward rasterizer is used. For each view the script
captures the native rasterizer arguments once, then renders symmetric
counterfactuals with no gradients:

* `beta_D[c] <- beta_D[c] * exp(q[c])` for direct attenuation;
* `beta_B[c] <- beta_B[c] * exp(q[3+c])` for backscatter attenuation;
* `B_inf[c] <- sigmoid(logit(B_inf[c]) + q[6+c])` for asymptotic medium color.

The nine-dimensional parameter `q` is measured by a fixed symmetric finite
difference epsilon `0.01`. All medium outputs remain positive or in `[0,1]`.
Gaussian geometry, opacity, bounded Gaussian colors, ordering, and renderer
early termination are untouched. The native base output is checked against the
retained FSB underwater PNG and the existing stride-4 maps.

## Two predeclared fits

`all_fit` accumulates a ridge least-squares solution for `GT-underwater` on a
deterministic subset of every training image. `far_fit` uses only the existing
stable far-field candidate mask (`A>=0.9`, depth CV `<=0.1`, positive depth) and
the scene's frozen upper depth boundary. No test image or clear render is used
to estimate either vector. Coefficients are clipped to `[-0.25,0.25]` before
the nonlinear counterfactual render; clipping frequency and fit residuals are
reported.

The test evaluation renders the frozen baseline and both fixed correction
vectors on every original test view. Metrics are native full-image PSNR, SSIM,
LPIPS, and MSE. Region errors use the existing stride-4 maps at identical grid
locations. The far/near summaries are diagnostic subsets, not whole-image
quality claims. No clear GT is available; clear output is expected to remain
exactly unchanged because Gaussian color and geometry are frozen.

## Promotion rule fixed before results

This diagnostic does not authorize training by itself. A transport-only
follow-up may be designed only if at least three scenes satisfy all of:

1. `far_fit` improves far-candidate RMSE and whole-image PSNR over the frozen
   baseline on the complete test split;
2. mean PSNR gain is at least `0.05 dB`, no scene loses more than `0.10 dB`,
   and mean LPIPS does not worsen by more than `0.001`;
3. `all_fit` and `far_fit` have the same sign for at least six of nine controls,
   and no more than 25% of fitted coefficients are clipped;
4. the near-candidate RMSE does not worsen by more than 5% in any scene.

These are screening thresholds, not statistical significance or a SOTA claim.
If the gate fails, no transport training is started and no extra Gaussian,
color, exposure, or clear-image postprocessing is added to hide the result.

## Evidence boundaries

The Jacobian is local and the medium is a view-conditioned field. It does not
prove physical medium identifiability, and a train-fitted correction can still
exploit camera-specific bias. `B_inf=medium_rgb` is the current tied-tail
semantics; the script checks that direct rasterizer output equals the model
output at the base point. The diagnostic tests underwater novel-view fit only;
it does not claim clear restoration accuracy.

## Implementation audit and corrected v2 (before valid formal run)

The first implementation is invalid and retained under
`outputs/fsb_medium_transport_jacobian_20260917_v1/INVALID_RESULT.json`.
Its normalized normal-equation solution omitted the division by column norm;
its replay compared raw RGB to a clamped reference, and its regional metrics
used per-view depth quantiles. Its apparent coefficient saturation or degradation
must not be interpreted as physical evidence. No model updates occurred.

V2 corrects these implementation issues without changing epsilon, ridge (1e-3
in column-normalized coordinates), coefficient bound, or promotion thresholds.
The solve uses `D=sqrt(diag(H))`, solves for `z=D*q`, then returns `q=z/D`.
Both Jacobian and residual refer to the evaluated `clip(RGB,0,1)`; the nonlinear
renderer still computes full raw RGB internally. All regions use frozen scene
boundaries. Fit sampling is at native pixel locations on stride-4 grids, at
most 12000 pixels per view/fit, with deterministic seeds. Zero B_inf shift is
exact via its odds-ratio form, including endpoints. B_inf changes also update
the tied background argument. Native support and both raw/display clear are
measured for every perturbation and correction; >1e-6 aborts interpretation.

Before evaluation, coefficients and normal equations are written to disk.
Evaluation uses the exact original FSB metric implementation (model PSNR,
SSIM, LPIPS); baseline PNGs and metrics are verified against retained assets.
The previous local VGG metric helper was different and is not used for V2.
No image appearance, exposure, Gaussian parameters, or production code changes.
Shared initialization constructs optimizer objects but no step is invoked.
The two 9-D diagnostic vectors are fitted; this is not a neural-network training
run or a claim that there was no data fitting.

Additional validity checks: known-solution normal-equation test, finite-difference
halving on the first train and first eval view, first-train directional prediction,
training-only leave-one-view-out linear fits, normalized condition spectrum,
single-family attenuation/scatter/B_inf linear ablations, and actual nonlinear
rendering on all train/eval views. Singular values and linear gains are not
physical identifiability or causal responsibility estimates.

Interpret the original promotion rule as: at least three scenes have far RMSE
and full-image PSNR gains, >=6/9 coefficient sign agreement, <=25% clipping for
both fits; in addition, four-scene mean PSNR gain >=0.05 dB, minimum scene gain
>=-0.10 dB, mean LPIPS delta <=0.001, and every scene's near RMSE ratio <=1.05.
Use the far_fit arm for these gates; all_fit is a control. A failed FD or replay
check is an invalid experiment, not negative evidence for the hypothesis.
