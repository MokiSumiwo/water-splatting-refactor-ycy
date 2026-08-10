# BND-UNORM Photometric Loss - 2026-08-10

## 1. Motivation

HYPOTHESIS: Removing inverse-prediction photometric normalization may increase optimization responsibility for localized bright / legacy-high-J Panama failure regions.

The causal question is whether the Panama BND-K1 RGB gap is partly caused by the current relative photometric objective under-emphasizing bright or legacy-high-J regions after intrinsic appearance is bounded.

## 2. Current Panama Failure Localization

EXPERIMENTAL FACT: Formal Panama references before BND-UNORM:

| Run | PSNR | SSIM | LPIPS | MSE | tau p90 | J p99 | P(J>1) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 32.308910 | 0.949487 | 0.073979 | 0.000595238 | 1.769849 | 1.311801 | 0.037758 |
| BND-K1 | 31.498353 | 0.948783 | 0.075521 | 0.000714139 | 0.999069 | 0.838911 | 0.000000 |

QUANTITATIVE RESULT: BND-K1 is decomposition-successful on the bounded metrics, but remains RGB-incomplete on Panama with Delta PSNR `-0.810558 dB` vs M1.

## 3. LOSSRESP-AUDIT Evidence

QUANTITATIVE RESULT: On Panama eval views, fixed `M1_HIGH_J` covers about `5.05%` of pixels and accounts for about `33.78%` of K1 MSE, but receives only `2.00%` of formal total image-gradient responsibility under the current relative objective.

QUANTITATIVE RESULT: On Panama train views, `M1_HIGH_J` covers `0.051671` of pixels, accounts for `0.511054` of MSE, has MSE enrichment `9.890480`, and receives relative-objective total image-gradient share `0.021958` with responsibility ratio `0.042965`.

## 4. Why SeaFree CB Is Not Used

EXPERIMENTAL FACT: Prior SeaFree content-based weighting audit found that SeaFree-style inverse-intensity weighting does not align with Panama high-J / bright failure regions. It was not used in this training run.

## 5. Exact Current Loss Semantics

CODE FACT: Current K1 default remains:

```text
L_rel =
0.8 * mean(abs((GT - pred) / (stopgrad(pred) + 1e-3)))
+
0.2 * (1 - SSIM(GT / (stopgrad(pred)+1e-3),
                pred / (stopgrad(pred)+1e-3)))
```

CODE FACT: The denominator is per-channel `pred.detach() + 1e-3`. There is no pred clamp before the loss. SSIM uses data range `1.0`. No extra training loss was added in `get_loss_dict`.

## 6. BND-UNORM Formulation

CODE FACT: BND-UNORM adds `photometric_normalization_mode: absolute` and uses:

```text
L_abs =
0.8 * mean(abs(GT - pred))
+
0.2 * (1 - SSIM(GT, pred))
```

CODE FACT: Default mode is `relative_pred_detached`, preserving historical behavior. The absolute mode does not add foreground masks, residual weighting, brightness weighting, pseudo-depth, SeaFree CB, medium supervision, or any new regularizer.

## 7. Single-Factor Experimental Design

CODE FACT: The only intended training variable is:

```text
photometric_normalization_mode:
relative_pred_detached -> absolute
```

CODE FACT: Fixed configuration: Panama, from scratch, seed 42, SH degree 3, `intrinsic_color_parameterization=bounded_sh3`, `rasterize_mode=classic`, `medium_context_mode=dir_xy_camera`, `b_inf_mode=tied`, `infinite_water_enabled=False`, `appearance_lr_scale=1.0`, no medium hold.

## 8. Default Behavior Equivalence

QUANTITATIVE RESULT: `DEFAULT_LOSS_EQUIVALENCE = True`; max absolute loss difference `0.0`.

## 9. Train-View Responsibility Replication

QUANTITATIVE RESULT: `TRAIN_HIGHJ_UNDEREMPHASIS_REPLICATED = True`.

Train `M1_HIGH_J`: pixel fraction `0.051671`, MSE share `0.511054`, MSE enrichment `9.890480`, total gradient share `0.021958`, responsibility ratio `0.042965`.

## 10. Fixed-State REL-vs-ABS Responsibility Shift

QUANTITATIVE RESULT: At the same K1 final state with no optimizer step:

| Region | REL grad share | ABS grad share | Direction |
| --- | ---: | ---: | --- |
| M1_HIGH_J | 0.021958 | 0.040891 | increased |
| BRIGHT_Q5 | 0.112882 | 0.167512 | increased |
| DARK_BOTTOM_QUINTILE | 0.313488 | 0.214330 | decreased |
| LOW_TRANSMISSION | 0.010796 | 0.012493 | increased |

QUANTITATIVE RESULT: `HIGHJ_GRAD_SHARE_GAIN = 1.862352`; `RESP_RATIO_GAIN = 1.862352`.

## 11. Global Gradient Scale Audit

QUANTITATIVE RESULT: Image-gradient L2 changed from `0.038036` under REL to `0.002476` under ABS in the fixed-state no-step audit.

Parameter-group ABS/REL gradient norm ratios:

| Group | Ratio |
| --- | ---: |
| features_dc | 0.214088 |
| features_rest | 0.214088 |
| means | 0.062600 |
| scales | 0.223418 |
| opacities | 0.238108 |

EXPERIMENTAL FACT: This global scale change was recorded but not manually compensated, preserving the one-factor training test.

## 12. Gradient Direction Audit

QUANTITATIVE RESULT: Fixed-state REL-vs-ABS parameter gradient cosine:

| Group | Cosine |
| --- | ---: |
| features_dc | 0.795227 |
| features_rest | 0.795225 |
| means | 0.852851 |
| scales | 0.795258 |
| opacities | 0.875159 |
| medium_mlp | 0.891638 |

## 13. High-J/Low-J Conflict Audit

QUANTITATIVE RESULT: `ABS_HIGHJ_GRADIENT_CONFLICT = False`.

Fixed-state ABS high-J vs low-J gradient cosines:

| Group | Cosine |
| --- | ---: |
| features_dc | 0.090329 |
| features_rest | 0.090329 |
| means | 0.171498 |
| medium_mlp | 0.408289 |

## 14. Initialization Equivalence

QUANTITATIVE RESULT: `INIT_PARAMETER_EQUIVALENCE = True`, max parameter diff `0.0`.

QUANTITATIVE RESULT: `INIT_FORWARD_EQUIVALENCE = True`, max forward diff `0.0` for `pred_image`, `direct_object_signal`, `rgb_medium`, `depth`, `accumulation`, `clear_object_fullsh_raw`, `transmission`, and `tau_D`.

## 15. Training Configuration

EXPERIMENTAL FACT: BND-UNORM was trained from scratch:

```text
outputs/bnd_unorm_panama_20260810/
panama_bnd_unorm_seed42_step0_to_15000/
water-splatting/20260810_bnd_unorm/
```

Final checkpoint: `nerfstudio_models/step-000014999.ckpt`.

EXPERIMENTAL FACT: Formal comparison uses M1 reused, K1 reused, and BND-UNORM new 0->15000. The summary scaffold labels BND-UNORM as `STAGE`; in this note `STAGE` means the absolute photometric normalization run.

## 16. RGB Trajectory

QUANTITATIVE RESULT:

| Step | K1 PSNR | UNORM PSNR | K1 SSIM | UNORM SSIM | K1 LPIPS | UNORM LPIPS | K1 Gauss | UNORM Gauss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 22.563314 | 22.592882 | 0.688374 | 0.685170 | 0.436505 | 0.439270 | 60112 | 35225 |
| 3000 | 20.869651 | 21.659102 | 0.597971 | 0.659706 | 0.440031 | 0.432973 | 566332 | 81228 |
| 5000 | 26.459814 | 27.205816 | 0.822805 | 0.842691 | 0.252952 | 0.239163 | 998668 | 106162 |
| 8000 | 31.008972 | 30.913612 | 0.943347 | 0.922150 | 0.086439 | 0.146630 | 1223420 | 124263 |
| 10000 | 31.154767 | 31.232512 | 0.945652 | 0.926635 | 0.082064 | 0.136501 | 1219898 | 126171 |
| 13000 | 31.540089 | 31.557959 | 0.949397 | 0.930239 | 0.074719 | 0.129748 | 1183679 | 126022 |
| 15000 | 31.498353 | 31.457218 | 0.948783 | 0.930509 | 0.075521 | 0.128953 | 1177886 | 125970 |

## 17. Final RGB Metrics

QUANTITATIVE RESULT:

| Run | PSNR | SSIM | LPIPS | MSE | Gaussian count |
| --- | ---: | ---: | ---: | ---: | ---: |
| M1 | 32.308910 | 0.949487 | 0.073979 | 0.000595238 | 1173293 |
| K1 | 31.498353 | 0.948783 | 0.075521 | 0.000714139 | 1177886 |
| UNORM | 31.457218 | 0.930509 | 0.128953 | 0.000717197 | 125970 |

QUANTITATIVE RESULT: `UNORM_PSNR_GAIN = -0.041135 dB` vs K1. `GLOBAL_MSE_GAP_RECOVERY = -0.025712`.

QUANTITATIVE RESULT: `RGB_SAFETY = False` because SSIM and LPIPS are below the gate relative to M1, and LPIPS is substantially worse than K1.

## 18. Canonical Decomposition

QUANTITATIVE RESULT:

| Run | tau p90 | T mean | P(T<0.1) | J p99 | P(J>1) | beta_D mean | beta_B mean | medium_rgb mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 1.769849 | 0.384434 | 0.007454 | 1.311801 | 0.037758 | 0.396922 | 0.296831 | 0.215950 |
| K1 | 0.999069 | 0.685939 | 0.000477 | 0.838911 | 0.000000 | 0.143104 | 0.084423 | 0.190539 |
| UNORM | 1.244436 | 0.635350 | 0.002840 | 0.780398 | 0.000000 | 0.175677 | 3.053223 | 0.127139 |

QUANTITATIVE RESULT: `TAU_BENEFIT_RETENTION = 0.681664`, below the `0.75` retention threshold.

QUANTITATIVE CONCLUSION: UNORM keeps bounded J saturation removed, but regresses part of the K1 low-tau decomposition benefit.

## 19. Boundary Audit

QUANTITATIVE RESULT:

| Run | P(c>0.95) | P(c>0.99) | P(|s|>5) | P(|s|>8) | BOUNDARY_ESCAPE |
| --- | ---: | ---: | ---: | ---: | --- |
| K1 | 0.020831 | 0.017517 | 0.017149 | 0.015563 | False |
| UNORM | 0.046866 | 0.039263 | 0.038917 | 0.034066 | False |

QUANTITATIVE RESULT: Boundary escape is false, but UNORM has higher near-boundary occupancy than K1.

## 20. High-J Targeted Recovery

QUANTITATIVE RESULT: Fixed eval `M1_HIGH_J` aggregate MSE:

| Run | MSE | Mean abs residual L1 |
| --- | ---: | ---: |
| M1 | 0.002617530 | 0.092595 |
| K1 | 0.004818396 | 0.127099 |
| UNORM | 0.003691699 | 0.115301 |

QUANTITATIVE RESULT: `HIGH_J_MSE_GAP_RECOVERY = 0.511934`.

QUANTITATIVE CONCLUSION: UNORM causally improves the targeted legacy-high-J region relative to K1.

## 21. Brightness Q5 Recovery

QUANTITATIVE RESULT: Fixed eval `GT_brightness_Q5` aggregate MSE:

| Run | MSE |
| --- | ---: |
| M1 | 0.001681953 |
| K1 | 0.002242602 |
| UNORM | 0.001980643 |

QUANTITATIVE RESULT: Bright Q5 recovery is `0.467242` by the same gap-recovery form.

## 22. Low-J Control

QUANTITATIVE RESULT: Fixed eval `M1_J_le_1` aggregate MSE:

| Run | MSE |
| --- | ---: |
| M1 | 0.000492933 |
| K1 | 0.000497582 |
| UNORM | 0.000571824 |

QUANTITATIVE RESULT: `LOW_J_DAMAGE = 0.000074241`, about `+14.84%` relative to K1 in the final responsibility audit.

## 23. Dark-Region Control

QUANTITATIVE RESULT: Fixed eval dark bottom-quintile responsibility audit MSE:

| Run | MSE | MSE share | Gradient share |
| --- | ---: | ---: | ---: |
| K1 | 0.000386340 | 0.108171 | 0.305058 |
| UNORM | 0.000394029 | 0.109843 | 0.199052 |

QUANTITATIVE RESULT: Dark-region MSE delta is `+0.000007690`, about `+1.99%` relative to K1.

## 24. Low-Transmission Control

QUANTITATIVE RESULT: Fixed M1 low-transmission responsibility audit MSE:

| Run | MSE | MSE share | Gradient share |
| --- | ---: | ---: | ---: |
| K1 | 0.000152200 | 0.002255 | 0.007344 |
| UNORM | 0.000103428 | 0.001526 | 0.009067 |

## 25. Final Responsibility Audit

QUANTITATIVE RESULT: Final eval `M1_HIGH_J` responsibility:

| Run | MSE share | L1 loss share | Gradient share | Responsibility ratio |
| --- | ---: | ---: | ---: | ---: |
| K1 | 0.337810 | 0.069292 | 0.020002 | 0.059211 |
| UNORM | 0.242854 | 0.119529 | 0.037006 | 0.152379 |

QUANTITATIVE RESULT: Final high-J gradient share increased by `1.8501x` and responsibility ratio increased by `2.5737x` relative to K1.

## 26. Responsibility Trajectory

QUANTITATIVE RESULT: Fixed eval `M1_HIGH_J` responsibility trajectory:

| Step | Run | MSE share | Gradient share | Responsibility ratio |
| ---: | --- | ---: | ---: | ---: |
| 1000 | K1 | 0.405696 | 0.020415 | 0.050321 |
| 3000 | K1 | 0.410103 | 0.019760 | 0.048184 |
| 5000 | K1 | 0.386107 | 0.019845 | 0.051398 |
| 8000 | K1 | 0.374729 | 0.020544 | 0.054825 |
| 10000 | K1 | 0.364986 | 0.020411 | 0.055923 |
| 13000 | K1 | 0.355774 | 0.019986 | 0.056175 |
| 15000 | K1 | 0.337810 | 0.020002 | 0.059211 |
| 1000 | UNORM | 0.390855 | 0.051028 | 0.130555 |
| 3000 | UNORM | 0.423621 | 0.047333 | 0.111734 |
| 5000 | UNORM | 0.336578 | 0.046010 | 0.136700 |
| 8000 | UNORM | 0.251309 | 0.037816 | 0.150477 |
| 10000 | UNORM | 0.244283 | 0.037486 | 0.153452 |
| 13000 | UNORM | 0.238804 | 0.037028 | 0.155054 |
| 15000 | UNORM | 0.242854 | 0.037006 | 0.152379 |

QUANTITATIVE CONCLUSION: UNORM consistently increases fixed high-J formal image-gradient responsibility during training.

## 27. Per-View Results

QUANTITATIVE RESULT:

| View | K1 PSNR | UNORM PSNR | Delta PSNR |
| --- | ---: | ---: | ---: |
| MTN_1539 | 31.089798 | 31.059282 | -0.030516 |
| MTN_1529 | 32.304523 | 31.900122 | -0.404402 |
| MTN_1547 | 31.100737 | 31.412249 | +0.311512 |

QUANTITATIVE RESULT: UNORM improves 1 eval view and degrades 2 eval views. Delta PSNR mean `-0.041135`, median `-0.030516`, min `-0.404402`, max `+0.311512`.

## 28. Recomposition

QUANTITATIVE RESULT: UNORM vs K1 recomposition on K1 object support:

| Metric | Value |
| --- | ---: |
| mean_abs_DeltaD_l1 | 0.212474 |
| mean_abs_DeltaB_l1 | 0.210841 |
| mean_abs_DeltaI_l1 | 0.034755 |
| flattened cosine similarity | -0.957735 |
| cancellation residual ratio | 0.082377 |
| recomposition efficiency | 0.917623 |

QUANTITATIVE RESULT: MSE attribution relative to M1:

| Run | C_direct | C_medium | C_cross |
| --- | ---: | ---: | ---: |
| K1 | 0.004331 | 0.004166 | -0.008379 |
| UNORM | 0.000920 | 0.000504 | -0.001303 |

## 29. Gaussian Population

QUANTITATIVE RESULT: K1 and UNORM Gaussian counts:

| Step | K1 | UNORM |
| ---: | ---: | ---: |
| 1000 | 60112 | 35225 |
| 3000 | 566332 | 81228 |
| 5000 | 998668 | 106162 |
| 8000 | 1223420 | 124263 |
| 10000 | 1219898 | 126171 |
| 13000 | 1183679 | 126022 |
| 15000 | 1177886 | 125970 |

REASONABLE INFERENCE: Removing inverse-prediction normalization substantially changes optimization and densification dynamics, not only the final pixel weighting. This is a measured downstream effect of the one-factor loss change.

## 30. Training Stability

EXPERIMENTAL FACT: Training reached nominal step `15000` with actual checkpoint `14999`.

EXPERIMENTAL FACT: `missing_checkpoints.csv` is empty. Eval image technical checks reported finite outputs for M1, K1, and UNORM on all three eval views. Final images use width `1795`, height `1188`.

## 31. Final Classification

QUANTITATIVE RESULT:

| Flag | Value |
| --- | --- |
| STRONG_UNORM_RECOVERY | False |
| PARTIAL_UNORM_RECOVERY | False |
| RGB_SAFETY | False |
| BOUNDARY_ESCAPE | False |
| DECOMPOSITION_REGRESSION | True |
| UNORM_HARMFUL | True by LPIPS gate |
| PANAMA_PARETO_CLOSED | False |

QUANTITATIVE CONCLUSION: BND-UNORM gives targeted high-J and bright-region recovery, but does not close the Panama Pareto gap because global RGB safety fails, LPIPS degrades, low-J MSE increases, and tau benefit retention is below threshold.

## 32. Causal Hypothesis Assessment

FINAL CLASSIFICATION: `NOT_SUPPORTED` for the full hypothesis:

```text
Removing inverse-prediction normalization from the bounded WaterSplatting
photometric objective increases responsibility for localized bright /
legacy-high-J regions and causally recovers part of the Panama RGB gap
without reopening decomposition trade-offs.
```

QUANTITATIVE CONCLUSION: The responsibility mechanism is supported locally: high-J gradient share and high-J MSE improve. The full training hypothesis is not supported because the local recovery does not translate into global RGB recovery and does not retain the K1 decomposition benefit.

REASONABLE INFERENCE: Simple removal of inverse-prediction normalization is not a deployable replacement for K1 on Panama. The experiment indicates a spatial responsibility trade-off rather than a clean correction.

## 33. Next Single-Factor Recommendation

RECOMMENDATION: Close the simple BND-UNORM training candidate. The next step should be read-only hard-region / residual-responsibility diagnostics rather than another training run. Specifically, identify whether a minimal residual-responsibility objective can target the high-J failure region without reducing low-J and dark-region protection.

This recommendation is a mechanism diagnostic, not a visual-quality conclusion.

## Outputs

Metric outputs:

- `outputs/bnd_unorm_panama_20260810/bnd_unorm_final_summary.json`
- `outputs/bnd_unorm_panama_20260810/bnd_unorm_final_summary.csv`
- `outputs/bnd_unorm_panama_20260810/final_rgb_metrics.csv`
- `outputs/bnd_unorm_panama_20260810/decomposition_metrics.csv`
- `outputs/bnd_unorm_panama_20260810/boundary_metrics.csv`
- `outputs/bnd_unorm_panama_20260810/high_j_region_metrics.csv`
- `outputs/bnd_unorm_panama_20260810/region_control_metrics.csv`
- `outputs/bnd_unorm_panama_20260810/final_responsibility_audit.csv`
- `outputs/bnd_unorm_panama_20260810/responsibility_trajectory.csv`
- `outputs/bnd_unorm_panama_20260810/recomposition_metrics.csv`
- `outputs/bnd_unorm_panama_20260810/image_technical_checks.csv`

Visual assets:

- `renders/bnd_unorm_panama_20260810/contact_sheet_final_underwater_m1_k1rst_stage.png`
- `renders/bnd_unorm_panama_20260810/contact_sheet_final_clear_raw_m1_k1rst_stage.png`
- `renders/bnd_unorm_panama_20260810/contact_sheet_final_residual_m1_k1rst_stage.png`
- `renders/bnd_unorm_panama_20260810/contact_sheet_high_j_mask_overlay_k1rst_stage.png`
- `renders/bnd_unorm_panama_20260810/contact_sheet_final_direct_k1rst_stage_delta.png`
- `renders/bnd_unorm_panama_20260810/contact_sheet_final_medium_k1rst_stage_delta.png`
- `renders/bnd_unorm_panama_20260810/VISUAL_COMPARE_INDEX.md`

EXPERIMENTAL FACT: Visual assets are for external/manual analysis only. No subjective clear-image correctness judgment was made.
