# Frozen BND Direct-Light Gain Protocol

This protocol is fixed before new fitting. Results belong in a separate file.

## Sources and Interpretation

Use CRCC E0 bounded_sh3 checkpoints at step 14999 for all four scenes. Freeze
all source parameters, OCMC, camera, Gaussian topology, appearance and medium.
Historical legacy_ocmc_cica_no_auxiliary_20260905 OFF is an image reference and
conditional teacher, not a matched causal BND ablation. Existing model-LPIPS
results from other runs are not comparable to the project-local VGG metrics.

The adapter multiplies only each accepted Gaussian's underwater direct-light
contribution by exp(0.35*tanh(f(x))). Its clear output is the original BND
render, including original support and background semantics. This is an
effective underwater illumination factor; effective radiance can exceed one.
It does not prove recovery of a single physically correct intrinsic radiance.

## Step 1: Shared Regions

Use all 13 existing test views, common GT, clamped ON/OFF RGB, and identical
masks for both models. Object support proxy: ON and OFF accumulation >=0.5.
Boundaries: GT adjacent RGB difference >=0.03, dilated by two pixels, within
object support. Interiors: remaining object support. Bright objects have GT
mean RGB >=0.65; ordinary objects are the complement within support. Low/high
transmission: original ON mean transmission <0.35 / >=0.7 within support.
OFF overflow: any raw composited clear channel >1 (not individual SH values).
Report mask counts, regional MSE/PSNR and shares of total error; empty masks
have null metrics. Masks overlap and are proxies, not semantic object labels.
Also decompose residual into achromatic/chromatic and 16x16 block-mean/detail
energies. These decompositions are descriptive, not causal illumination tests.

## Step 2: Frozen Direct Gain

- B0: unchanged BND source.
- scalar: one scene-wide coefficient.
- spatial: ten quadratic coefficients [1,x,y,z,xy,xz,yz,x^2,y^2,z^2].

Normalize world Gaussian centers using their 2%/98% coordinate quantiles,
clipping normalized coordinates to [-1,1]. No camera IDs or view parameters.
All coefficients start at zero. Gains stay in [exp(-0.35),exp(0.35)].
Only sampled RGB MSE trains the adapter; no color or geometry regularizers.

Sort training view IDs; every fifth camera (rank modulo 5 equals 2) is adapter
validation, others fit. Base checkpoints already saw these validation cameras:
this tests adapter generalization, not independently unseen base-model views.
Per camera fix 8192 fit or 16384 validation pixels using seed 91400+camera index.
Use Adam lr=0.01, batch 1024, 2500 updates, continuation seeds 42/123/3407.
Seeds change minibatches, not pretrained source models. All arms share batches.
Select checkpoint using mean camera validation PSNR at 0 and every 250 steps;
identity is an explicit candidate. All selected checkpoints are then evaluated
on complete validation images before a global architecture gate is calculated.

Enter step 3 for the architecture with greatest full-validation improvement
only if macro PSNR improves >=0.03 dB, at least three scenes improve >=0.01 dB,
no scene loses >0.05 dB, each seed macro improves, and macro LPIPS does not
worsen. Exact source/clear preservation and fixed/native agreement <=2e-6 are
required. These are screening gates, not significance or SOTA claims.
Official test predictions are evaluated after this selection and never select
architecture, step or teacher weight. Test safety is reported separately.

## Step 3: Conditional Teacher Comparison

If eligible, restart the selected architecture at identity with the same
seeds, batches and checkpoint selection. Compare GT MSE against GT MSE plus
0.1 times MSE to the historical OFF underwater prediction clamped to [0,1].
Teacher uses only fitting cameras. Render at the original ON camera and
verify matching GT. No Gaussian correspondence is assumed. Freeze teacher.
No weight search. Report teacher benefit over identical GT-only architecture,
both full validation and all official test images, and added rendering cost.
If step 2 gate fails, step 3 is not triggered; do not conceal failure by
changing exposure, colors, geometry, point count or the gain bound.

## Engineering and Artifacts

Use only conda water_splatting and physical GPUs 6-9. No installation,
commit/push or overwritten assets. Save source hashes, code snapshots, pixel
manifests, optimizer/checkpoint states, curves, validation selection, raw
float predictions, native per-view metrics, timing/memory and comparison
sheets. Existing fixed-ray helper must match native direct contributions,
including culling and termination. Re-render with modified colors only in an
auxiliary direct-light call; final underwater = original prediction + changed
direct - original direct. Return original clear, never the auxiliary clear.
Check original images against archived floats and source state before/after.
All displays use clamp [0,1], fixed-scale errors, no exposure/white balance.
No independent clear GT exists: preservation is not a clear-accuracy claim.
