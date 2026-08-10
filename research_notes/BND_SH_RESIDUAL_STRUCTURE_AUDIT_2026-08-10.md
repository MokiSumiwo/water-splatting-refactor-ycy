# BND SH Residual Structure Audit

Date: 2026-08-10

Branch: `research/m1-bounded-intrinsic`

Start HEAD: `4879398ad6888f6780515ce5b53ebfa5a5d56c2e`

Diagnostic outputs:

- Metrics: `outputs/bnd_shstruct_audit_20260810/`
- Visuals: `renders/bnd_shstruct_audit_20260810/`
- Visual index: `renders/bnd_shstruct_audit_20260810/VISUAL_COMPARE_INDEX.md`

This audit is read-only. No model was trained, no optimizer step was run, and no checkpoint state was modified.

## Motivation

HYPOTHESIS:

The previous BND appearance optimizer test showed that increasing bounded-SH appearance LR can increase RGB-space SH residual amplitude, but does not materially recover the Panama RGB gap. This audit tests whether the missing M1 capacity is mostly valid in-range view-dependent residual, range-violating compensation residual, or a bounded from-scratch object-medium recomposition issue.

## Metric-Definition Reconciliation

CODE FACT:

The earlier cross-scene BND summary used the archived diagnostic path:

- `research-snapshot-20260808-gmvc-dewatering-full:scripts/diagnostics/diagnose_dewater_optical_depth.py`
- `research-snapshot-20260808-gmvc-dewatering-full:scripts/diagnostics/summarize_bounded_sh3_cross_scene.py`

That path used `outputs["tau_D_effective"]` and `outputs["clear_object_fullsh_raw"]`, applied object support `outputs["accumulation"] > 0.01`, pooled all eval-view supported pixels per RGB channel, computed p90/p99 per channel, then averaged the three channel scalars.

CODE FACT:

The AOPT/tradeoff summary used the current diagnostic path:

- `scripts/diagnostics/summarize_bnd_aopt_panama.py`

For `tau_D_all_p90` / `J_all_p99`, that path flattened each eval view's H x W x C tensor over all pixels/channels, computed the per-view quantile, then averaged those view scalars. It did not use the object-support mask.

QUANTITATIVE RESULT:

The new audit reproduces both old Panama values:

| Run | Old cross-scene tau p90 | Old cross-scene J p99 | Old AOPT tau p90 | Old AOPT J p99 |
| --- | ---: | ---: | ---: | ---: |
| M1 | 1.769849 | 1.311801 | 1.594246 | 1.293632 |
| BND-K1 | 0.999069 | 0.838911 | 0.909361 | 0.829595 |

QUANTITATIVE CONCLUSION:

The discrepancy is not random sampling. It is caused by aggregation/support semantics:

- Cross-scene: object-support masked, pooled across all eval pixels before quantile, channel mean.
- AOPT: all pixels, per-view flattened RGB quantile, then view mean.

CANONICAL_TAU_METRIC:

`tau_eval_object_support_pooled_channel_mean_p90`

CANONICAL_J_METRIC:

`J_clear_eval_object_support_pooled_channel_mean_p99`

Primary mask:

`outputs["accumulation"] > 0.01`

## Checkpoint Audit

EXPERIMENTAL FACT:

All checkpoints loaded at actual step `14999`, nominal `15000`, with seed `42`, SH degree `3`, `medium_context_mode=dir_xy_camera`, `b_inf_mode=tied`.

| Scene | Run | Eval views | Intrinsic mode |
| --- | --- | ---: | --- |
| Curasao | M1 | 3 | legacy |
| Curasao | BND-K1 | 3 | bounded_sh3 |
| JapaneseGradens | M1 | 3 | legacy |
| JapaneseGradens | BND-K1 | 3 | bounded_sh3 |
| IUI3 | M1 | 4 | legacy |
| IUI3 | BND-K1 | 4 | bounded_sh3 |
| Panama | M1 | 3 | legacy |
| Panama | BND-K1 | 3 | bounded_sh3 |
| Panama | K2 | 3 | bounded_sh3 |
| Panama | K4 | 3 | bounded_sh3 |

Full checkpoint paths are recorded in:

`outputs/bnd_shstruct_audit_20260810/checkpoint_manifest.csv`

## Exact SH Color Semantics

CODE FACT:

Legacy color path:

`water_splatting/fields/gaussian_appearance.py::compute_gaussian_colors`

For SH degree > 0:

`c_dc_legacy = clamp(spherical_harmonics(0, viewdirs, features_dc) + 0.5, min=0)`

`c_full_legacy(v) = clamp(spherical_harmonics(active_sh_degree, viewdirs, [features_dc, features_rest]) + 0.5, min=0)`

There is no upper clamp in the legacy SH path.

CODE FACT:

BND color path:

`water_splatting/fields/gaussian_appearance.py::compute_bounded_gaussian_colors`

`s_dc = spherical_harmonics(0, viewdirs, features_dc)`

`s_full(v) = spherical_harmonics(active_sh_degree, viewdirs, [features_dc, features_rest])`

`c_dc_bnd = sigmoid(s_dc)`

`c_full_bnd(v) = sigmoid(s_full(v))`

Thus BND residual has two distinct forms:

- logit residual: `Delta_s_SH(v) = s_full(v) - s_dc`
- RGB residual: `Delta_c_SH(v) = c_full_bnd(v) - c_dc_bnd`

All final checkpoints used active SH3.

## Gaussian-Level SH Residual

QUANTITATIVE RESULT:

Visible Gaussian observations are unweighted observations using `outputs["gaussian_visible_mask"] = radii > 0`. No cross-run Gaussian index matching was used.

| Scene | M1 R_SH p50 | BND R_SH p50 | BND/M1 |
| --- | ---: | ---: | ---: |
| Curasao | 0.103558 | 0.030497 | 0.294491 |
| JapaneseGradens | 0.092242 | 0.033742 | 0.365795 |
| IUI3 | 0.081024 | 0.040943 | 0.505322 |
| Panama | 0.102911 | 0.035256 | 0.342590 |

QUANTITATIVE CONCLUSION:

BND-K1 keeps substantially smaller RGB-space SH residual distributions than M1 in all four scenes.

## Legacy RGB-Range Violations

QUANTITATIVE RESULT:

Panama M1 visible Gaussian channel-observation fractions:

| Class | Channel fraction | Gaussian observation any-channel fraction |
| --- | ---: | ---: |
| VALID_TO_VALID | 0.888993 | 0.931628 |
| VALID_TO_OVERFLOW | 0.032546 | 0.067045 |
| VALID_TO_UNDERFLOW | 0.000000 | 0.000000 |
| BASE_ALREADY_OVERFLOW | 0.078461 | 0.102490 |

Legacy lower underflow is zero by construction in this path.

## Residual Energy

QUANTITATIVE RESULT:

| Scene | Legal SH energy | Valid-to-overflow energy | Base-invalid energy |
| --- | ---: | ---: | ---: |
| Curasao | 0.511652 | 0.083436 | 0.404912 |
| JapaneseGradens | 0.732318 | 0.100150 | 0.167532 |
| IUI3 | 0.485124 | 0.059724 | 0.455152 |
| Panama | 0.602457 | 0.150581 | 0.246963 |

QUANTITATIVE CONCLUSION:

Panama M1 contains a majority legal in-range SH residual energy fraction, but also a substantial range-related fraction: valid-to-overflow plus base-invalid energy is `0.397543`.

## Headroom

QUANTITATIVE RESULT:

| Scene | M1 P(u>1) | BND P(u>1) |
| --- | ---: | ---: |
| Curasao | 0.022235 | 0.000000 |
| JapaneseGradens | 0.018820 | 0.000000 |
| IUI3 | 0.024672 | 0.000000 |
| Panama | 0.035381 | 0.000000 |

Panama M1 headroom utilization:

- `u p50 = 0.104420`
- `u p90 = 0.496191`
- `u p95 = 0.917869`
- `u p99 = 3.187996`
- `P(u>0.5) = 0.099086`
- `P(u>2) = 0.016378`

## DC-Only vs Full-SH Image-Space Contribution

QUANTITATIVE RESULT:

| Scene | M1 SH_RGB_GAIN_PSNR | BND SH_RGB_GAIN_PSNR |
| --- | ---: | ---: |
| Curasao | 2.417330 | 0.436658 |
| JapaneseGradens | 0.787968 | 0.242116 |
| IUI3 | 1.835103 | 0.829632 |
| Panama | 2.591460 | 1.139380 |

QUANTITATIVE CONCLUSION:

M1 obtains larger underwater RGB PSNR gain from full-SH over DC-only in every scene.

## Frozen M1 Bounded-Projection Counterfactual

CODE FACT:

M1-PROJ is a diagnostic-only render. It keeps M1 geometry, opacity, medium, beta terms, B_inf, camera, and renderer fixed, and replaces only the current-view legacy Gaussian RGB:

`c_projected(v) = clip(c_full_legacy(v), 0, 1)`

Projection forward audit max absolute differences between normal full forward and diagnostic full forward were at numerical precision:

- Curasao aggregate max: `0`
- JapaneseGradens aggregate max: `0`
- IUI3 aggregate max: `1.19e-7`
- Panama aggregate max: `1.34e-7`

QUANTITATIVE RESULT:

| Scene | M1 FULL PSNR | M1-PROJ PSNR | BND-K1 PSNR | Projection MSE fraction | PSNR PROJ - BND |
| --- | ---: | ---: | ---: | ---: | ---: |
| Curasao | 32.165164 | 25.919282 | 32.188016 | -412.101346 | -6.268734 |
| JapaneseGradens | 24.756484 | 24.024287 | 24.698505 | -7.840137 | -0.674218 |
| IUI3 | 30.874542 | 30.443835 | 30.963132 | 0.535920 | -0.519297 |
| Panama | 32.308910 | 28.095669 | 31.498353 | 11.598585 | -3.402683 |

QUANTITATIVE CONCLUSION:

For Panama, naive frozen M1 clipping is much worse than trained BND-K1. Therefore `FROZEN_M1_BOUNDED_COUNTERFACTUAL_STRONG = FALSE`.

QUANTITATIVE CONCLUSION:

The Panama projection fraction is greater than 1, meaning the frozen clipping counterfactual loses more MSE than the trained BND gap. This is not the predefined `RANGE_REMOVAL_EXPLAINS_RGB_GAP` case, which required projection to be close to BND.

## Projection-vs-BND Residual Overlap

QUANTITATIVE RESULT:

| Scene | Pearson | Spearman | Top10 overlap | Top20 overlap |
| --- | ---: | ---: | ---: | ---: |
| Curasao | 0.483164 | 0.045397 | 0.262063 | 0.257460 |
| JapaneseGradens | 0.156189 | 0.115349 | 0.250603 | 0.363308 |
| IUI3 | 0.180074 | 0.114013 | 0.288350 | 0.358939 |
| Panama | 0.458037 | 0.094353 | 0.310995 | 0.340005 |

## Compensation-Region Attribution

QUANTITATIVE RESULT:

COMP mask definition:

`J1 OR J95 OR TAU90 OR TLOW`

M1-defined maps:

- `J1`: `max_rgb(M1 clear_object_fullsh_raw) > 1`
- `J95`: top 5% of `max_rgb(M1 clear_object_fullsh_raw)`
- `TAU90`: top 10% of mean-RGB `M1 tau_D`
- `TLOW`: min-RGB `M1 transmission < 0.1`

| Scene | Projection-change enrichment in COMP | BND-excess enrichment in COMP |
| --- | ---: | ---: |
| Curasao | 4.409581 | 3.202706 |
| JapaneseGradens | 5.641655 | 1.893017 |
| IUI3 | 6.367938 | 1.875158 |
| Panama | 6.358184 | 3.197804 |

## Cross-Scene Comparison

QUANTITATIVE RESULT:

Panama has the largest BND-K1 RGB gap among the audited scenes:

- Curasao: `+0.022852 dB`
- JapaneseGradens: `-0.057979 dB`
- IUI3: `+0.088590 dB`
- Panama: `-0.810558 dB`

QUANTITATIVE CONCLUSION:

Panama is distinct in RGB gap size. It is not distinct by legal SH energy alone: JapaneseGradens has higher legal SH energy fraction, and Curasao/IUI3 have comparable or larger base-invalid fractions.

## Panama K1/K2/K4 Secondary Analysis

QUANTITATIVE RESULT:

| Run | PSNR full | PSNR DC | SH RGB gain | R_SH p50 | R_SH p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BND-K1 | 31.498353 | 30.358973 | 1.139380 | 0.035256 | 0.197820 |
| K2 | 31.529617 | 30.180414 | 1.349203 | 0.045453 | 0.252564 |
| K4 | 31.393145 | 29.761049 | 1.632097 | 0.059081 | 0.331701 |

QUANTITATIVE CONCLUSION:

K2/K4 increase RGB-space residual amplitude and DC-vs-full PSNR gain, but do not recover the Panama M1 RGB gap. This supports the earlier AOPT conclusion that more SH residual amplitude alone is not the primary missing factor.

## Final Classification

QUANTITATIVE RESULT:

Based on Panama:

- `VALID_SH_RESIDUAL_WORTH_RECOVERING = TRUE`
- `LEGACY_SH_OVERFLOW_DOMINANT = FALSE`
- `FROZEN_M1_BOUNDED_COUNTERFACTUAL_STRONG = FALSE`
- `RANGE_REMOVAL_EXPLAINS_RGB_GAP = FALSE`
- `MIXED_RANGE_AND_RECOMPOSITION = FALSE`

Numeric basis:

- `LEGAL_SH_ENERGY_FRACTION = 0.602457`
- `OVERFLOW_PLUS_BASE_INVALID_SH_ENERGY_FRACTION = 0.397543`
- `BND_over_M1_R_SH_p50 = 0.342590`
- `PROJECTION_MSE_FRACTION = 11.598585`
- `PSNR_PROJ_minus_BND = -3.402683 dB`

INFERENCE:

There is valid in-range legacy SH residual worth understanding, but frozen M1 clipping is too destructive to serve as evidence that a simple bounded current-view projection preserves M1 RGB. The data therefore does not support immediately implementing BND-v2 solely to restore residual amplitude.

## Next Single-Factor Recommendation

NEXT_SINGLE_FACTOR_EXPERIMENT:

`bounded object-medium recomposition diagnostic`

Reason:

The strongest new quantitative fact is that M1-PROJ is much worse than trained BND-K1 on Panama, while K2/K4 show that increasing bounded SH residual amplitude alone does not recover the RGB gap. A recomposition diagnostic should test whether BND from-scratch changes object/medium allocation in a way that prevents recovery of the legacy M1 RGB solution under bounded colors.

Do not start this experiment automatically from this note.

## Visual Assets

Four-scene visual contact sheets:

- `renders/bnd_shstruct_audit_20260810/Curasao/`
- `renders/bnd_shstruct_audit_20260810/JapaneseGradens/`
- `renders/bnd_shstruct_audit_20260810/IUI3/`
- `renders/bnd_shstruct_audit_20260810/Panama/`

Panama extra K1/K2/K4 sheets:

- `renders/bnd_shstruct_audit_20260810/Panama/panama_underwater_m1_proj_k1_k2_k4.png`
- `renders/bnd_shstruct_audit_20260810/Panama/panama_clear_dc_full_projected_k1.png`
- `renders/bnd_shstruct_audit_20260810/Panama/panama_sh_luma_m1_k1_k2_k4.png`
- `renders/bnd_shstruct_audit_20260810/Panama/panama_compensation_overlay.png`

Complete index:

`renders/bnd_shstruct_audit_20260810/VISUAL_COMPARE_INDEX.md`

Visual assets are ready for external/manual analysis.

No subjective clear-image correctness judgment was made.
