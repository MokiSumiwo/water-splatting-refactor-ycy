# BND Staged Object-Medium Optimization - Panama

## 1. Motivation

HYPOTHESIS: Panama BND-K1 keeps bounded intrinsic appearance and lower optical depth, but the remaining RGB gap may come from an optimization-path issue. A temporary medium hold after the 10k BND checkpoint may let the bounded object representation adapt before joint object/medium catch-up.

This experiment tests only one intervention: medium parameter hold followed by joint catch-up. It does not use GMVC, D010/gamma scaling, BG/BGI supervision, FAW/OAW, pseudo-depth, SH0, renderer changes, loss changes, medium architecture changes, densification changes, or opacity-policy changes.

## 2. Prior Evidence

EXPERIMENTAL FACT: Prior Panama final metrics were:

| Run | PSNR | SSIM | LPIPS | canonical tau p90 | canonical J p99 | P(J>1) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 32.308910 | 0.949487 | 0.073979 | 1.769849 | 1.311801 | 0.037758 |
| BND-K1-HIST | 31.498353 | 0.948783 | 0.075521 | 0.999069 | 0.838911 | 0.000000 |

EXPERIMENTAL FACT: BND recomposition diagnosis classified Panama as `COUPLED_OBJECT_MEDIUM_REMAINDER=TRUE`, with strong global D/B recomposition and localized residual concentration in M1 high-J / bright / bottom-image regions.

## 3. Relation To SeaFree-Inspired Staged Refinement

INFERENCE: This experiment is motivated by staged/post-densification refinement behavior, but it is not a SeaFree-GS reproduction.

CODE FACT: No SeaFree-GS code was copied. SeaFree-specific mechanisms such as pseudo-depth, content-based loss, background supervision, distance scaling, antialiasing changes, and SH0 are not enabled in this experiment.

## 4. Important Distinction From GMVC

CODE FACT: This experiment uses M1+BND only. It does not re-enable GMVC profile calibration, GMVC object auxiliary loss, target/current camera tracking, profile banks, or alternating GMVC schedules.

## 5. Restart Determinism Audit

CODE FACT: The shared starting checkpoint is:

`outputs/dewater_bounded_sh3_cross_scene_20260808/dewater_bnd_cross_scene_panama_seed42_step0_to_15000/water-splatting/dewater_bnd_cross_scene_panama_seed42_step0_to_15000_20260808_bounded_sh3_cross_scene_full_panama_bnd_g1p00/nerfstudio_models/step-000010000.ckpt`

EXPERIMENTAL FACT: The 10k checkpoint internal step is `10000`; Nerfstudio resume starts the next update at `10001`.

EXPERIMENTAL FACT: A no-hold restart audit at 11k was not tensor-shape equivalent to historical uninterrupted K1 11k:

| Check | Historical K1 11k | Restart audit 11k |
| --- | ---: | ---: |
| checkpoint step | 11000 | 11000 |
| features_dc shape | [1193866, 3] | [1193913, 3] |
| means shape | [1193866, 3] | [1193913, 3] |

QUANTITATIVE RESULT: `RESTART_EQUIVALENCE = FAIL`; matched restart control `BND-K1-RST` is required and was run. Historical K1 is retained only as a reference.

## 6. Medium Parameter And Optimizer Audit

CODE FACT: The medium parameter set held by the schedule is:

| Group | Parameter key | Shape | Optimizer state | Scheduler state |
| --- | --- | ---: | --- | --- |
| medium_mlp | `_model.medium_mlp.tcnn_encoding.params` | [6144] | present, 1 entry, 12289 state elements | present |
| direction_encoding | `_model.direction_encoding.tcnn_encoding.params` | [0] | present, 1 entry, 1 state element | present |

CODE FACT: Archived source checkpoints contain an extra legacy optimizer/scheduler group `gmvc_bounded_medium`. `WaterSplattingTrainer` filters optimizer/scheduler groups not present in the current clean model when resuming archived checkpoints.

## 7. Schedule Definition

CODE FACT: `medium_hold_start_step=10000`, `medium_hold_end_step=12500`.

| Boundary | Update |
| --- | ---: |
| HOLD_FIRST_UPDATE | 10001 |
| HOLD_LAST_UPDATE | 12500 |
| JOINT_FIRST_UPDATE | 12501 |
| FINAL_UPDATE | 14999 |

CODE FACT: The hold freezes only `medium_mlp` and `direction_encoding` by setting `requires_grad_(False)` during held updates. It does not reset Adam state, scheduler state, or medium parameters.

## 8. K1 Control Definition

EXPERIMENTAL FACT: Because restart equivalence failed, formal causal comparison is:

`BND-K1-RST` vs `BND-STAGE-MH2500`

Historical `BND-K1-HIST` remains a reference run.

## 9. Medium-Hold Validation

EXPERIMENTAL FACT: Short smoke audit:

| Step | Phase | medium param max delta | exp_avg max delta | exp_avg_sq max delta | optimizer step delta | medium grad l2 | features_dc delta |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10001 | MEDIUM_HOLD | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 2.103837 |
| 10010 | MEDIUM_HOLD | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.813747 |

QUANTITATIVE RESULT: Hold-phase medium parameter freeze and Adam-state pause passed in the smoke audit.

## 10. Unfreeze Validation

EXPERIMENTAL FACT: Short unfreeze smoke audit:

| Step | Phase | medium param max delta | exp_avg max delta | exp_avg_sq max delta | optimizer step delta | medium grad l2 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 10011 | JOINT | 0.000892 | 0.0000158 | 2.50e-11 | 1.000000 | 0.001000 |
| 10012 | JOINT | 0.001199 | 0.0000165 | 2.44e-11 | 1.000000 | 0.001000 |

QUANTITATIVE RESULT: Unfreeze restored medium gradients and parameter updates without optimizer-state reset.

## 11. Hold-Phase Trajectory

QUANTITATIVE RESULT: During STAGE medium hold, medium parameters remained unchanged relative to 10k through 12.5k:

| Run | Step | medium_mlp L2 delta | medium_mlp max abs delta |
| --- | ---: | ---: | ---: |
| STAGE | 10000 | 0.000000 | 0.000000 |
| STAGE | 12500 | 0.000000 | 0.000000 |
| K1-RST | 12500 | 4.091972 | 0.675051 |

EXPERIMENTAL FACT: At 12.5k:

| Run | PSNR | SSIM | LPIPS | tau p90 | J p99 | Gaussians |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| K1-RST | 31.467365 | 0.948980 | 0.075830 | 1.040067 | 0.836757 | 1185384 |
| STAGE | 31.386793 | 0.948952 | 0.075876 | 1.149096 | 0.856987 | 1186851 |

INFERENCE: The held trajectory diverged from matched K1-RST in rendered metrics and Gaussian population while medium parameters stayed fixed. Per-Gaussian object delta cannot be reliably aligned after 10k because continuation checkpoints do not contain `_model.gaussian_lineage_ids`; this is recorded in `object_parameter_trajectory.csv`.

## 12. Joint Catch-Up Trajectory

EXPERIMENTAL FACT: STAGE medium updates resumed after 12.5k:

| Step | Phase | medium_mlp LR | requires grad | medium param max delta | optimizer step delta | medium grad l2 | Gaussians |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: |
| 12500 | MEDIUM_HOLD | 0.000205757 | False | 0.000000 | 0.000000 | 0.000000 | 1187167 |
| 12501 | JOINT | 0.000205731 | True | 0.000651 | 1.000000 | 0.001000 | 1186851 |
| 14999 | JOINT | 0.000150000 | True | 0.000309 | 1.000000 | 0.001000 | 1179287 |

QUANTITATIVE RESULT: STAGE PSNR increased from 31.386793 at 12.5k to 31.427324 at 15k, but stayed below K1-RST final PSNR.

## 13. Final RGB Metrics

QUANTITATIVE RESULT:

| Run | PSNR | SSIM | LPIPS | MSE | Gaussians |
| --- | ---: | ---: | ---: | ---: | ---: |
| M1 | 32.308910 | 0.949487 | 0.073979 | 0.000595238 | 1173293 |
| K1-HIST | 31.498353 | 0.948783 | 0.075521 | 0.000714139 | 1177886 |
| K1-RST | 31.495648 | 0.948784 | 0.075578 | 0.000714753 | 1177908 |
| STAGE | 31.427324 | 0.948752 | 0.075697 | 0.000726130 | 1179287 |

QUANTITATIVE RESULT: Relative to matched K1-RST:

- `STAGE_PSNR_GAIN = -0.068324 dB`
- `GLOBAL_MSE_GAP_RECOVERY = -0.095192`
- STAGE improved 0 of 3 eval views and degraded 3 of 3 by PSNR.
- `RGB_SAFETY = FAIL` relative to M1 because PSNR remains more than 0.15 dB below M1.

## 14. Decomposition Retention

QUANTITATIVE RESULT:

| Run | canonical tau p90 | canonical J p99 | P(J>1) | P(T<0.1) |
| --- | ---: | ---: | ---: | ---: |
| M1 | 1.769849 | 1.311801 | 0.037758 | 0.007454 |
| K1-RST | 0.997965 | 0.838074 | 0.000000 | 0.000497 |
| STAGE | 1.052825 | 0.847281 | 0.000000 | 0.001056 |

QUANTITATIVE RESULT: `TAU_BENEFIT_RETENTION = 0.928927`.

INFERENCE: STAGE preserved most of the lower-optical-depth decomposition proxy benefit and kept bounded intrinsic values, but it did not improve RGB relative to matched K1-RST.

## 15. Boundary Safety

QUANTITATIVE RESULT:

| Run | c p99 | P(c>0.95) | P(c>0.99) | P(|s|>5) | P(|s|>8) | sigmoid deriv p50 | sigmoid deriv p10 | BOUNDARY_ESCAPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| K1-RST | 1.000000 | 0.020804 | 0.017510 | 0.017137 | 0.015563 | 0.210309 | 0.137268 | FALSE |
| STAGE | 1.000000 | 0.021352 | 0.017816 | 0.017413 | 0.015765 | 0.210582 | 0.136689 | FALSE |

QUANTITATIVE RESULT: No sigmoid boundary escape was detected under the preregistered threshold.

## 16. High-J Region Recovery

CODE FACT: High-J mask is fixed from M1 eval output: object support with `max_channel(M1 clear_object_fullsh_raw) > 1`.

QUANTITATIVE RESULT:

| Region | M1 MSE | K1-RST MSE | STAGE MSE |
| --- | ---: | ---: | ---: |
| M1_J_gt_1 | 0.002617530 | 0.004828606 | 0.005019723 |

QUANTITATIVE RESULT: `HIGH_J_MSE_GAP_RECOVERY = -0.086436`.

INFERENCE: The staged schedule did not recover the localized M1 high-J region gap; it slightly increased MSE in that fixed region relative to K1-RST.

## 17. Bright-Region / Bottom-Region Controls

QUANTITATIVE RESULT:

| Region | M1 MSE | K1-RST MSE | STAGE MSE |
| --- | ---: | ---: | ---: |
| M1_J_le_1 | 0.000492933 | 0.000497473 | 0.000499350 |
| GT_brightness_Q5 | 0.001681953 | 0.002247948 | 0.002299306 |
| bottom20_image_y | 0.001262867 | 0.001606351 | 0.001641894 |

QUANTITATIVE RESULT: `LOW_J_DAMAGE = 0.000001877`.

INFERENCE: STAGE did not produce targeted high-J recovery and did not provide improvement in the registered bright or bottom-image control regions.

## 18. Object Parameter Trajectory

CODE FACT: Object groups audited: `features_dc`, `features_rest`, `means`, `scales`, and `opacities`.

EXPERIMENTAL FACT: The 10k source checkpoint has `_model.gaussian_lineage_ids`; continuation checkpoints produced by this branch do not carry `_model.gaussian_lineage_ids`. Therefore exact per-Gaussian object delta after 10k is not reliably available.

QUANTITATIVE RESULT: Gaussian population trajectory:

| Step | K1-RST Gaussians | STAGE Gaussians |
| ---: | ---: | ---: |
| 10000 | 1219898 | 1219898 |
| 12500 | 1185384 | 1186851 |
| 15000 | 1177908 | 1179287 |

INFERENCE: Object/geometry population differs under the held schedule, but direct per-Gaussian parameter displacement is marked unavailable rather than inferred from tensor order.

## 19. Medium Parameter Trajectory

QUANTITATIVE RESULT:

| Run | Step | medium_mlp L2 delta | medium_mlp normalized L2 delta | max abs delta |
| --- | ---: | ---: | ---: | ---: |
| K1-RST | 10000 | 0.000000 | 0.000000 | 0.000000 |
| K1-RST | 12500 | 4.091972 | 0.161658 | 0.675051 |
| K1-RST | 15000 | 6.475482 | 0.255821 | 0.952112 |
| STAGE | 10000 | 0.000000 | 0.000000 | 0.000000 |
| STAGE | 12500 | 0.000000 | 0.000000 | 0.000000 |
| STAGE | 15000 | 3.185677 | 0.125854 | 0.502441 |

EXPERIMENTAL FACT: STAGE medium parameters were paused through 12.5k and then moved during joint catch-up.

## 20. Recomposition Analysis

QUANTITATIVE RESULT: STAGE minus K1-RST, object support:

| Metric | Value |
| --- | ---: |
| mean_abs_DeltaD_l1 | 0.011890916 |
| mean_abs_DeltaB_l1 | 0.009578661 |
| mean_abs_DeltaI_l1 | 0.005140543 |
| deltaD_deltaB_pearson_luma | -0.623911 |
| flattened cosine similarity | -0.730439 |
| cos_theta_p50 | -0.980652 |
| P(cos<-0.9) | 0.720598 |
| r_DB p50 | 1.074034 |
| RECOMP_EFFICIENCY | 0.743933 |

INFERENCE: STAGE forms a different co-adapted D/B pair from K1-RST, but the new pair does not reduce final RGB residual.

## 21. Exact MSE Attribution

QUANTITATIVE RESULT: Relative to M1:

| Run | DeltaMSE | C_direct | C_medium | C_cross | closure error |
| --- | ---: | ---: | ---: | ---: | ---: |
| K1-RST | 0.000118811 | 0.004311470 | 0.004156530 | -0.008349189 | 3.88e-10 |
| STAGE | 0.000130188 | 0.003949503 | 0.003816889 | -0.007636204 | 5.82e-10 |

INFERENCE: STAGE reduces the magnitudes of direct and medium terms and their negative cross compensation, but the net DeltaMSE relative to M1 is larger than K1-RST.

## 22. Per-View Results

QUANTITATIVE RESULT: STAGE vs K1-RST final PSNR:

| View | STAGE PSNR | Delta PSNR vs K1-RST |
| --- | ---: | ---: |
| MTN_1539 | 31.048641 | -0.040413 |
| MTN_1529 | 32.247086 | -0.066338 |
| MTN_1547 | 30.986244 | -0.098223 |

QUANTITATIVE RESULT: improved view count = 0; degraded view count = 3.

## 23. Final Classification And Next Step

QUANTITATIVE RESULT:

| Flag | Value |
| --- | --- |
| OBJECT_HOLD_PHASE_MOVED_BASIN | TRUE, with direct object-delta caveat due missing lineage IDs |
| JOINT_CATCHUP_RECOVERS_RGB | TRUE for STAGE 12.5k -> 15k PSNR, but not beyond K1-RST |
| HIGH_J_TARGETED_RECOVERY | FALSE |
| COUPLED_PAIR_IMPROVED | FALSE |
| DECOMPOSITION_REGRESSION | FALSE |
| STRONG_STAGED_RECOVERY | FALSE |
| PARTIAL_STAGED_RECOVERY | FALSE |
| NO_STAGED_RECOVERY | FALSE |
| HARMFUL_STAGED_OPTIMIZATION | FALSE |

FINAL CLASSIFICATION: `HYPOTHESIS_SUPPORT = NOT_SUPPORTED`.

INFERENCE: The staged medium hold preserves much of the BND decomposition proxy benefit and remains boundary-safe, but it does not recover Panama RGB and does not improve the fixed M1 high-J region. The current evidence does not support "temporarily decoupling medium updates allows BND to find a better bounded object-medium co-adapted basin" for Panama under this exact MH2500 schedule.

NEXT SINGLE-FACTOR RECOMMENDATION: Do not continue medium-hold schedule tuning immediately. The next single-factor experiment should target the residual structure with a bounded residual appearance parameterization (BND-v2) while keeping the renderer, medium schedule, and RGB loss fixed. Do not run it automatically without a new experiment request.

## Outputs

Metric tables:

- `outputs/bnd_stage_panama_20260810/bnd_stage_final_summary.json`
- `outputs/bnd_stage_panama_20260810/training_trajectory.csv`
- `outputs/bnd_stage_panama_20260810/final_rgb_metrics.csv`
- `outputs/bnd_stage_panama_20260810/decomposition_metrics.csv`
- `outputs/bnd_stage_panama_20260810/high_j_region_metrics.csv`
- `outputs/bnd_stage_panama_20260810/region_control_metrics.csv`
- `outputs/bnd_stage_panama_20260810/recomposition_metrics.csv`
- `outputs/bnd_stage_panama_20260810/mse_attribution.csv`
- `outputs/bnd_stage_panama_20260810/per_view_metrics.csv`
- `outputs/bnd_stage_panama_20260810/object_parameter_trajectory.csv`
- `outputs/bnd_stage_panama_20260810/medium_parameter_trajectory.csv`

Visual assets:

- Final underwater: `renders/bnd_stage_panama_20260810/contact_sheet_final_underwater_m1_k1rst_stage.png`
- Final clear raw display: `renders/bnd_stage_panama_20260810/contact_sheet_final_clear_raw_m1_k1rst_stage.png`
- Direct comparison: `renders/bnd_stage_panama_20260810/contact_sheet_final_direct_k1rst_stage_delta.png`
- Medium comparison: `renders/bnd_stage_panama_20260810/contact_sheet_final_medium_k1rst_stage_delta.png`
- Residual comparison: `renders/bnd_stage_panama_20260810/contact_sheet_final_residual_m1_k1rst_stage.png`
- High-J overlay: `renders/bnd_stage_panama_20260810/contact_sheet_high_j_mask_overlay_k1rst_stage.png`
- Phase trajectory: `renders/bnd_stage_panama_20260810/contact_sheet_phase_trajectory_MTN_1539_k1rst_stage.png`
- Visual index: `renders/bnd_stage_panama_20260810/VISUAL_COMPARE_INDEX.md`

No subjective clear-image correctness judgment was made.
