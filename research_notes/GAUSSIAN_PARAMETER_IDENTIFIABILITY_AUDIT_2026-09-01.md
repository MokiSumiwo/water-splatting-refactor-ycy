# Gaussian Parameter Identifiability Audit Under OCMC

Date: 2026-09-01
Experiment: `GAUSSIAN_PARAMETER_IDENTIFIABILITY_AUDIT_OCMC`
Classification: `SUPPORTED`

## Hypothesis

Medium, bounded-SH appearance, opacity, and screen-space geometry responses may occupy overlapping local RGB observation subspaces. Such overlap is a representation ambiguity only if it is substantial, predicts heldout error after controls, remains independent of OCMC, and recurs over checkpoint populations.

## Frozen Protocol

All 20 registered C0 checkpoints use OCMC on, RAOC off, runtime `bounded_sh3`, SH degree 3, `dir_xy_camera`, `b_inf_mode=tied`, `infinite_water_enabled=false`, and classic rasterization. Historical YAML uses the old `sigmoid_sh` name, which the protected setup normalizes to `bounded_sh3`. The audit uses detached analytic forward sensitivities only: zero backward, JVP, VJP, optimizer steps, checkpoint writes, and render writes.

Sampling and every ambiguity/removal metric use training visibility only and are frozen before heldout GT access. Heldout GT is used only as the projected-footprint error outcome.

## Metric Definition

For each Gaussian, local RGB responses are evaluated at five fixed footprint offsets over all visible training cameras and stacked into a common observation matrix. The isolated signal relative to tied pure-medium background is `alpha*(exp(-beta_D*d)*c_SH-exp(-beta_B*d)*B_inf)`. Analytic response groups are physical 9-D medium, the full 48-D bounded SH3 appearance coefficient group (including DC), raw-opacity logit, and 2-D screen-center displacement with conic/depth fixed. The separate SH removal proxy replaces full bounded RGB with bounded DC RGB and therefore removes only the non-DC contribution.

Each response matrix is truncated to the effective left-singular subspace retaining 99.9% energy with relative singular values at least `1e-4`. Normalized overlap is `||U_A^T U_B||_F/sqrt(min(rank_A,rank_B))`; principal angle is the minimum angle. Obvious overlap requires population median maximum overlap at least `0.8` and median minimum angle at most `20 degrees`. Ambiguity score is maximum overlap times total response sensitivity. PSNR error is `10*log10(MSE)`, so its rank correlation is expected to equal the heldout-MSE rank correlation and is recorded explicitly in the formal outputs.

## Final Results

| Scene | max overlap median | min angle | largest pair | rho(A,MSE) | rho(A,PSNR error) | AUROC | depth/tau/T ctrl | OCMC ctrl | temporal | pass |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|:---:|
| Curasao | 0.997957 | 1.346 | SH-Opacity | 0.150555 | 0.150555 | 0.606317 | 0.106047 | 0.155057 | 4/5 | yes |
| IUI3-RedSea | 0.999628 | 0.899 | SH-Opacity | 0.204896 | 0.204896 | 0.610215 | 0.255509 | 0.227626 | 5/5 | yes |
| JapaneseGradens-RedSea | 0.999622 | 0.342 | SH-Opacity | 0.099278 | 0.099278 | 0.552490 | 0.115726 | 0.106120 | 5/5 | yes |
| Panama | 0.999582 | 0.442 | SH-Opacity | 0.040450 | 0.040450 | 0.523898 | 0.160353 | 0.056878 | 5/5 | yes |

## Pairwise Overlap

| Scene | Medium-SH | Medium-Opacity | Medium-Geometry | SH-Opacity | SH-Geometry | Opacity-Geometry |
|---|---:|---:|---:|---:|---:|---:|
| Curasao | 0.974839 | 0.994031 | 0.000000 | 0.997829 | 0.000000 | 0.000000 |
| IUI3-RedSea | 0.980806 | 0.996614 | 0.000000 | 0.999444 | 0.000000 | 0.000000 |
| JapaneseGradens-RedSea | 0.996010 | 0.997373 | 0.000000 | 0.999475 | 0.000000 | 0.000000 |
| Panama | 0.998504 | 0.996962 | 0.000000 | 0.999264 | 0.000000 | 0.000000 |

## Counterfactual Removal Alignment

| Scene | remove medium | remove SH | remove opacity | remove geometry |
|---|---:|---:|---:|---:|
| Curasao | 0.187280 | 0.210738 | 0.164796 | 0.165323 |
| IUI3-RedSea | 0.112664 | 0.215461 | 0.305224 | 0.305114 |
| JapaneseGradens-RedSea | -0.070102 | 0.231555 | 0.395594 | 0.395432 |
| Panama | -0.010067 | 0.222347 | 0.326911 | 0.326549 |

Removal/error alignment uses absolute local removal-delta RGB RMS. Baseline-relative sensitivity is also retained per Gaussian, but opacity alignment cannot use it because removing opacity is identically the full isolated signal and therefore has relative magnitude one. These are frozen read-only counterfactual proxies, not retraining, not physical component ground truth, and not causal error attribution.

## Cross-view Stability

Response stability is the coefficient of variation (CV) of each group response RMS over visible training cameras. Lower CV means more stable training-view sensitivity magnitude. Its correlation with heldout error is descriptive; no post-hoc CV threshold changes the classification gate.

| Scene | medium CV | SH CV | opacity CV | geometry CV | mean-CV/error rho |
|---|---:|---:|---:|---:|---:|
| Curasao | 0.086798 | 0.112853 | 0.152827 | 0.144147 | -0.177916 |
| IUI3-RedSea | 0.039002 | 0.086003 | 0.108498 | 0.111013 | -0.150305 |
| JapaneseGradens-RedSea | 0.044922 | 0.067312 | 0.121872 | 0.112609 | -0.269485 |
| Panama | 0.038519 | 0.089204 | 0.299483 | 0.225733 | -0.431733 |

The stacked subspace audit asks whether parameter explanations overlap across the training-view observation set; CV separately tests whether their response magnitudes are stable across those views. Thus a training-view-stable but novel-view-ambiguous pattern requires low response CV together with positive ambiguity/error alignment, rather than overlap alone.

## Temporal Stability

Checkpoint populations have no persistent Gaussian lineage. Temporal recurrence is distribution-level only; array index and nearest-geometry identity matching were not used.

| Scene | 5k | 8k | 10k | 13k | 14999 | checkpoint passes |
|---|:---:|:---:|:---:|:---:|:---:|---:|
| Curasao | yes | no | yes | yes | yes | 4/5 |
| IUI3-RedSea | yes | yes | yes | yes | yes | 5/5 |
| JapaneseGradens-RedSea | yes | yes | yes | yes | yes | 5/5 |
| Panama | yes | yes | yes | yes | yes | 5/5 |

## OCMC Independence

OCMC independence jointly rank-residualizes ambiguity and heldout error against projected OCMC active magnitude and suppressed medium residual. The all-required control additionally includes depth, tau, transmission, accumulation, opacity, and footprint. A scene cannot pass based on overlap alone.

## Limitations

This is local first-order identifiability, not global optimization equivalence. The isolated-Gaussian observation ignores other-Gaussian occlusion, uses an isotropic radius-based footprint for legal screen-displacement sensitivity, and associates heldout error through overlapping projected boxes. Medium parameters are local physical activations rather than all MLP weights. Geometry covers screen translation only, not full 3-D position/scale/rotation. Pair medians use finite overlaps only; zero-rank groups remain undefined and their pairwise finite fractions are retained in `final_summary.json`. No true parameter or medium labels exist.

## Classification

The formal result is `SUPPORTED` with 4/4 supported scenes and 4/4 partial candidate scenes. Module design authorization is `true`.

The next and only authorized task is a separate Gaussian-identifiability module-design phase. No module was implemented here.

## Integrity

Analyzed 5598 Gaussian-checkpoint rows, 33588 pairwise-overlap rows, and 65 heldout camera-checkpoint rows. All 20 checkpoint and protected source hashes matched before and after execution. Backward, JVP, VJP, optimizer-step, checkpoint-write, and render-write counts were zero.
