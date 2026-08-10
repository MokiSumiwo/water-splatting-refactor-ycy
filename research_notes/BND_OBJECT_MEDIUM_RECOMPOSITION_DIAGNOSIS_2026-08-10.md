# BND Object-Medium Recomposition Diagnosis

Date: 2026-08-10

Branch: `research/m1-bounded-intrinsic`

Start HEAD: `dde3033c59e5c1cb3623aad3e34bbb9c1dc5d27e`

Diagnostic code:

- `scripts/diagnostics/diagnose_bnd_object_medium_recomposition.py`

Diagnostic outputs:

- Metrics: `outputs/bnd_object_medium_recomposition_20260810/`
- Visuals: `renders/bnd_object_medium_recomposition_20260810/`
- Visual index: `renders/bnd_object_medium_recomposition_20260810/VISUAL_COMPARE_INDEX.md`

This diagnostic is read-only. No model was trained, no optimizer step was run, and no checkpoint state was modified.

## 1. Motivation

HYPOTHESIS:

Bounded SH3 constrains the image-space clear-object proxy and reduces high-J / high-optical-depth compensation. The remaining question is how the direct object contribution and medium contribution recombine after moving from M1 to BND, and why Panama keeps a larger RGB gap than Curasao and IUI3.

QUESTION:

Does Panama fail because a single branch is missing fit, or because object and medium changes must be jointly coordinated?

## 2. Current Evidence

EXPERIMENTAL FACT:

Previous BND experiments established that bounded SH3 reduces canonical tau and J tails across Curasao, JapaneseGradens, IUI3, and Panama. Panama still has a `-0.810558 dB` PSNR gap relative to M1, while Curasao and IUI3 are RGB-safe under the same bounded parameterization.

EXPERIMENTAL FACT:

Previous frozen M1 bounded-projection diagnostics showed that clipping M1 current-view intrinsic color without retraining causes much larger losses than final BND, so the remaining Panama gap is not the full frozen-projection loss. Joint object-medium recomposition recovers most of that loss.

## 3. Exact Forward Component Semantics

CODE FACT:

Forward path audited:

- `water_splatting/water_splatting.py::WaterSplattingModel.get_outputs`
- `water_splatting/rendering/underwater_rasterizer.py::UnderwaterRasterizer.rasterize`

CODE FACT:

For the audited checkpoints, `b_inf_mode=tied`. The returned additive closure is:

`I_PRED = D_DIRECT + B_MEDIUM`

with:

- `I_PRED`: `outputs["pred_image"]` / `outputs["rgb"]`
- `D_DIRECT`: `outputs["direct_object_signal"] = outputs["rgb_object"] = render.rgb_object`
- `B_MEDIUM`: `outputs["rgb_medium"]`, recomposed from finite medium contribution plus tied B_inf tail handling
- `B_MEDIUM_FINITE`: `outputs["rgb_medium_finite"]`
- `B_TAIL`: `outputs["rgb_tail"] = tail_weight * b_inf`
- `J_CLEAR`: `outputs["clear_object_fullsh_raw"] = render.j_raw`, an image-space alpha-composited full-SH clear-object proxy, not clear GT
- `TAU_DIRECT`: `outputs["tau_D"] = outputs["medium_attn"] * render.depth`
- `T_DIRECT`: `outputs["transmission"] = exp(-outputs["tau_D"].clamp_min(0)).clamp(0, 1)`

CODE FACT:

Direct/medium hybrids are marked `IMAGE_SPACE_COUNTERFACTUAL` after the `D+B` closure audit passes. Cross-run Gaussian color swaps were not performed because independently trained runs do not have Gaussian index correspondence.

## 4. Checkpoint Audit

EXPERIMENTAL FACT:

All primary M1 and BND-K1 checkpoints loaded at actual step `14999`, nominal step `15000`, with seed `42`, SH degree `3`, `medium_context_mode=dir_xy_camera`, `b_inf_mode=tied`, and `infinite_water_enabled=False`.

| Scene | Run | Role | Loaded step | Seed | Intrinsic mode | SH | Eval views | Gaussian count |
| --- | --- | --- | ---: | ---: | --- | ---: | --- | ---: |
| Curasao | M1 | primary | 14999 | 42 | legacy | 3 | MTN_1288; MTN_1296; MTN_1304 | 1106714 |
| Curasao | BND-K1 | primary | 14999 | 42 | bounded_sh3 | 3 | MTN_1288; MTN_1296; MTN_1304 | 1081672 |
| JapaneseGradens | M1 | primary | 14999 | 42 | legacy | 3 | MTN_1090; MTN_1098; MTN_1106 | 861508 |
| JapaneseGradens | BND-K1 | primary | 14999 | 42 | bounded_sh3 | 3 | MTN_1090; MTN_1098; MTN_1106 | 870028 |
| IUI3 | M1 | primary | 14999 | 42 | legacy | 3 | MTN_5903; MTN_5894; MTN_5911; MTN_5928 | 808747 |
| IUI3 | BND-K1 | primary | 14999 | 42 | bounded_sh3 | 3 | MTN_5903; MTN_5894; MTN_5911; MTN_5928 | 797196 |
| Panama | M1 | primary | 14999 | 42 | legacy | 3 | MTN_1539; MTN_1529; MTN_1547 | 1173293 |
| Panama | BND-K1 | primary | 14999 | 42 | bounded_sh3 | 3 | MTN_1539; MTN_1529; MTN_1547 | 1177886 |
| Panama | K2 | secondary | 14999 | 42 | bounded_sh3 | 3 | MTN_1539; MTN_1529; MTN_1547 | 1179946 |
| Panama | K4 | secondary | 14999 | 42 | bounded_sh3 | 3 | MTN_1539; MTN_1529; MTN_1547 | 1175279 |

Full checkpoint paths are recorded in:

- `outputs/bnd_object_medium_recomposition_20260810/checkpoint_manifest.csv`
- `outputs/bnd_object_medium_recomposition_20260810/checkpoint_manifest.json`

## 5. Forward Closure

QUANTITATIVE RESULT:

For all primary scene/run pairs, `pred_image - (direct_object_signal + rgb_medium)` is exactly zero in the saved audit tensors.

| Scene | Run | Mean abs | P95 abs | Max abs | PASS |
| --- | --- | ---: | ---: | ---: | --- |
| Curasao | M1 | 0.0 | 0.0 | 0.0 | TRUE |
| Curasao | BND-K1 | 0.0 | 0.0 | 0.0 | TRUE |
| JapaneseGradens | M1 | 0.0 | 0.0 | 0.0 | TRUE |
| JapaneseGradens | BND-K1 | 0.0 | 0.0 | 0.0 | TRUE |
| IUI3 | M1 | 0.0 | 0.0 | 0.0 | TRUE |
| IUI3 | BND-K1 | 0.0 | 0.0 | 0.0 | TRUE |
| Panama | M1 | 0.0 | 0.0 | 0.0 | TRUE |
| Panama | BND-K1 | 0.0 | 0.0 | 0.0 | TRUE |

QUANTITATIVE CONCLUSION:

Direct/medium additive attribution is valid for the audited outputs.

## 6. M1->BND Component Deltas

QUANTITATIVE RESULT:

Object-support, RGB-pooled mean absolute component changes:

| Scene | mean abs DeltaJ | mean abs DeltaT | mean abs DeltaD | mean abs DeltaB | mean abs DeltaI |
| --- | ---: | ---: | ---: | ---: | ---: |
| Curasao | 0.151115 | 0.243325 | 0.036488 | 0.036642 | 0.007706 |
| JapaneseGradens | 0.101851 | 0.394480 | 0.095888 | 0.091191 | 0.016237 |
| IUI3 | 0.059456 | 0.157916 | 0.038175 | 0.037198 | 0.008734 |
| Panama | 0.110802 | 0.301607 | 0.062901 | 0.063240 | 0.008954 |

QUANTITATIVE CONCLUSION:

All scenes show larger direct and medium branch changes than final predicted RGB change, meaning the BND transition is a recomposition rather than a small perturbation of final RGB.

## 7. Direct-Medium Cancellation

QUANTITATIVE RESULT:

Object-support aggregate cancellation metrics:

| Scene | RGB flat corr | Cos p50 | P(cos<-0.9) | r_DB p50 | Cancellation efficiency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Curasao | -0.815874 | -0.993461 | 0.920166 | 0.971814 | 0.881859 |
| JapaneseGradens | -0.786707 | -0.999343 | 0.942392 | 0.999639 | 0.900560 |
| IUI3 | -0.847829 | -0.996907 | 0.867835 | 0.985627 | 0.888840 |
| Panama | -0.673177 | -0.999543 | 0.972134 | 0.998721 | 0.928829 |

QUANTITATIVE CONCLUSION:

Panama does not show a lower global cancellation efficiency than Curasao/IUI3. Its direct and medium changes are strongly opposing and magnitude-balanced in aggregate.

## 8. Exact MSE Attribution

CODE FACT:

The exact MSE expansion uses:

- `DeltaMSE = MSE_BND - MSE_M1`
- `C_direct = mean(2 * e0 dot DeltaD + ||DeltaD||^2)`
- `C_medium = mean(2 * e0 dot DeltaB + ||DeltaB||^2)`
- `C_cross = mean(2 * DeltaD dot DeltaB)`

where `e0 = I_M1 - GT`.

QUANTITATIVE RESULT:

Aggregate attribution:

| Scene | DeltaMSE | C_direct | C_medium | C_cross | Closure error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Curasao | -0.00008992 | 0.00105558 | 0.00185941 | -0.00300491 | 1.16e-10 |
| JapaneseGradens | 0.00025685 | 0.01049174 | 0.01229587 | -0.02253076 | 9.70e-10 |
| IUI3 | -0.00008248 | 0.00224037 | 0.00189854 | -0.00422139 | 4.84e-10 |
| Panama | 0.00011820 | 0.00433061 | 0.00416626 | -0.00837868 | 6.40e-10 |

QUANTITATIVE CONCLUSION:

The negative cross term is the dominant cancellation term in every scene. Panama's positive DeltaMSE occurs because the cross-term cancellation is slightly insufficient relative to the direct and medium terms, not because one branch alone explains the gap.

## 9. Cross-Scene Recomposition Efficiency

QUANTITATIVE RESULT:

Primary aggregate RGB/decomposition metrics:

| Scene | Delta PSNR | M1 tau p90 | BND tau p90 | M1 J p99 | BND J p99 | Recomp efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Curasao | +0.022852 | 2.504676 | 1.533122 | 2.350653 | 0.900398 | 0.881859 |
| JapaneseGradens | -0.057979 | 1.947181 | 0.712033 | 1.348906 | 0.966225 | 0.900560 |
| IUI3 | +0.088590 | 1.497287 | 0.928444 | 1.137097 | 0.904771 | 0.888840 |
| Panama | -0.810558 | 1.769849 | 0.999069 | 1.311801 | 0.838911 | 0.928829 |

QUANTITATIVE CONCLUSION:

`RECOMPOSITION_STRONG_CROSS_SCENE = TRUE`.

QUANTITATIVE CONCLUSION:

`PANAMA_RECOMPOSITION_INCOMPLETE = FALSE` under the current global-efficiency threshold, because Panama's global recomposition efficiency is not lower than the Curasao/IUI3 control mean (`0.885350`). Panama's remaining gap must therefore be localized or coupled rather than a global failure of D/B cancellation.

## 10. Depth Stratification

QUANTITATIVE RESULT:

Panama depth-quintile recomposition using M1 depth:

| Bin | Recomp efficiency | DeltaMSE | BND excess residual | M1 tau mean | M1 T min mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q1 | 0.882691 | 0.00047993 | 0.011387 | 0.737353 | 0.450839 |
| Q2 | 0.901782 | 0.00013192 | 0.008876 | 0.765562 | 0.436098 |
| Q3 | 0.933143 | 0.00005512 | 0.006691 | 0.846289 | 0.401296 |
| Q4 | 0.958146 | -0.00002890 | 0.003384 | 1.093976 | 0.314821 |
| Q5 | 0.960287 | -0.00004706 | 0.001757 | 1.724074 | 0.172903 |

INFERENCE:

Panama's positive DeltaMSE is concentrated more in lower-depth bins than in far/high-depth bins under this diagnostic binning.

## 11. J/Tau/Transmission Stratification

QUANTITATIVE RESULT:

Panama M1-J masks:

| Mask | Recomp efficiency | DeltaMSE | BND excess residual | mean abs DeltaI |
| --- | ---: | ---: | ---: | ---: |
| J > 1 | 0.761053 | 0.00225984 | 0.031600 | 0.087204 |
| J <= 1 | 0.937610 | 0.00000438 | 0.005081 | 0.023654 |

QUANTITATIVE RESULT:

Panama tau/transmission masks:

| Mask | Recomp efficiency | DeltaMSE | BND excess residual |
| --- | ---: | ---: | ---: |
| tau top10 | 0.964917 | -0.00002498 | 0.002414 |
| T < 0.3 | 0.957484 | -0.00004530 | 0.002444 |
| T < 0.2 | 0.950145 | -0.00005442 | 0.002036 |
| T < 0.1 | 0.946162 | -0.00000416 | 0.003221 |

QUANTITATIVE CONCLUSION:

The Panama remaining positive DeltaMSE is much stronger in the M1 `J > 1` support than in low-transmission or top-tau masks.

## 12. Brightness/Spatial Stratification

QUANTITATIVE RESULT:

Panama GT-luminance quintiles:

| Bin | Recomp efficiency | DeltaMSE | BND excess residual |
| --- | ---: | ---: | ---: |
| Q1 | 0.939122 | 0.00001560 | 0.005459 |
| Q2 | 0.948749 | 0.00000071 | 0.004228 |
| Q3 | 0.953574 | -0.00001462 | 0.003474 |
| Q4 | 0.939152 | -0.00001752 | 0.004211 |
| Q5 | 0.855004 | 0.00060681 | 0.014723 |

QUANTITATIVE RESULT:

Panama image-y bins:

| Bin | Recomp efficiency | DeltaMSE | BND excess residual |
| --- | ---: | ---: | ---: |
| top20 | 0.961472 | -0.00001796 | 0.002178 |
| middle60 | 0.936008 | 0.00008924 | 0.006013 |
| bottom20 | 0.873247 | 0.00034069 | 0.011865 |

INFERENCE:

The Panama remainder is associated with high M1-J support, high luminance, and bottom-image bins in this image-space audit. This is a proxy association, not medium GT evidence.

## 13. Per-View Recomposition

QUANTITATIVE RESULT:

Panama per-view metrics:

| View | Delta PSNR | Recomp efficiency | DeltaMSE | RGB flat corr |
| --- | ---: | ---: | ---: | ---: |
| MTN_1539 | -0.252670 | 0.922546 | 0.00004319 | -0.465879 |
| MTN_1529 | -0.547436 | 0.927717 | 0.00006898 | -0.873179 |
| MTN_1547 | -1.631567 | 0.936224 | 0.00024242 | -0.680473 |

QUANTITATIVE RESULT:

For Panama, `DeltaPSNR_vs_cancellation_efficiency_pearson = -0.983961` across the three eval views.

INFERENCE:

The worst Panama PSNR view is not the lowest cancellation-efficiency view. The view-level evidence does not support a simple "lower global cancellation efficiency causes larger RGB gap" explanation.

## 14. Direct-Medium Hybrid Counterfactuals

CODE FACT:

The legal additive hybrids are image-space diagnostics:

- `H_D1_B0 = D_BND + B_M1`
- `H_D0_B1 = D_M1 + B_BND`

They are not renderer-level physical rerenders.

QUANTITATIVE RESULT:

Aggregate hybrid metrics:

| Scene | Hybrid | Type | PSNR | SSIM | LPIPS | MSE | Gap recovery |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Curasao | M1 | renderer | 32.165164 | 0.956004 | 0.108141 | 0.00086598 | n/a |
| Curasao | BND | renderer | 32.188016 | 0.958728 | 0.109208 | 0.00077851 | n/a |
| Curasao | H_D1_B0 | image-space | 27.349759 | 0.939122 | 0.127556 | 0.00192401 | 75.035882 |
| Curasao | H_D0_B1 | image-space | 25.657115 | 0.906277 | 0.130993 | 0.00272655 | 135.986499 |
| JapaneseGradens | M1 | renderer | 24.756484 | 0.899500 | 0.120356 | 0.00514719 | n/a |
| JapaneseGradens | BND | renderer | 24.698505 | 0.896300 | 0.117283 | 0.00541576 | n/a |
| JapaneseGradens | H_D1_B0 | image-space | 18.141029 | 0.850184 | 0.191748 | 0.01537256 | 339.264411 |
| JapaneseGradens | H_D0_B1 | image-space | 17.688232 | 0.786700 | 0.211062 | 0.01745896 | 350.753451 |
| IUI3 | M1 | renderer | 30.874542 | 0.912143 | 0.174617 | 0.00154450 | n/a |
| IUI3 | BND | renderer | 30.963132 | 0.911644 | 0.177082 | 0.00146455 | n/a |
| IUI3 | H_D1_B0 | image-space | 25.218780 | 0.897697 | 0.195856 | 0.00377526 | 206.288389 |
| IUI3 | H_D0_B1 | image-space | 25.237069 | 0.896244 | 0.197128 | 0.00344510 | 80.367326 |
| Panama | M1 | renderer | 32.308910 | 0.949487 | 0.073979 | 0.00059524 | n/a |
| Panama | BND | renderer | 31.498353 | 0.948783 | 0.075521 | 0.00071414 | n/a |
| Panama | H_D1_B0 | image-space | 23.076890 | 0.911594 | 0.113730 | 0.00492647 | -57.318957 |
| Panama | H_D0_B1 | image-space | 23.437222 | 0.887729 | 0.115309 | 0.00476210 | -52.768581 |

QUANTITATIVE CONCLUSION:

For Panama, neither single-branch hybrid recovers the BND-to-M1 MSE gap; both have negative recovery fractions. This supports a coupled object-medium remainder rather than a one-branch substitution explanation.

## 15. J/T Hybrid Audit

CODE FACT:

J/T hybrid was not generated because `clear_object_fullsh_raw * transmission` did not reconstruct `direct_object_signal` at floating-point closure level. The direct branch includes alpha/raster compositing semantics.

QUANTITATIVE RESULT:

`JT_HYBRID_NOT_SEMANTICALLY_VALID` for every primary scene/run. Representative mean/max absolute differences:

| Scene | Run | Mean abs diff | P99 abs diff | Max abs diff |
| --- | --- | ---: | ---: | ---: |
| Curasao | M1 | 0.000978 | 0.014018 | 0.091760 |
| Curasao | BND-K1 | 0.000308 | 0.004585 | 0.036065 |
| JapaneseGradens | M1 | 0.002380 | 0.047931 | 0.225444 |
| JapaneseGradens | BND-K1 | 0.000353 | 0.005901 | 0.079610 |
| IUI3 | M1 | 0.002133 | 0.038368 | 0.370755 |
| IUI3 | BND-K1 | 0.000904 | 0.016205 | 0.116559 |
| Panama | M1 | 0.000839 | 0.010257 | 0.151161 |
| Panama | BND-K1 | 0.000297 | 0.003661 | 0.033550 |

QUANTITATIVE CONCLUSION:

Image-space J/T swapping is unavailable in this diagnostic. Direct/medium additive hybrids remain valid because the additive `D+B` closure passed.

## 16. Required Medium/Direct Residual Analysis

CODE FACT:

Given direct branch `D`, the required medium target is `GT - D`; given medium branch `B`, the required direct target is `GT - B`. The audit reports the corresponding target error, which equals the final residual under the additive closure.

QUANTITATIVE RESULT:

Panama aggregate target errors:

| Run | Direct target error mean | Medium target error mean | Required correction mean abs | corr(required luma, J) | corr(required luma, image y) |
| --- | ---: | ---: | ---: | ---: | ---: |
| M1 | 0.024784 | 0.024784 | 0.012321 | 0.007991 | 0.077440 |
| BND-K1 | 0.026141 | 0.026141 | 0.013020 | 0.158161 | 0.094556 |

INFERENCE:

The BND-K1 Panama residual proxy has stronger association with J than M1 in this aggregate audit, but the correlation remains a proxy and is not evidence of physical correctness.

## 17. Frequency Analysis

QUANTITATIVE RESULT:

BND residual `GT - I_BND` energy fractions:

| Scene | Sigma | Low-frequency fraction | High-frequency fraction |
| --- | ---: | ---: | ---: |
| Curasao | 3 | 0.817686 | 0.182314 |
| Curasao | 9 | 0.688230 | 0.311770 |
| JapaneseGradens | 3 | 0.836102 | 0.163898 |
| JapaneseGradens | 9 | 0.666037 | 0.333963 |
| IUI3 | 3 | 0.533552 | 0.466448 |
| IUI3 | 9 | 0.378375 | 0.621625 |
| Panama | 3 | 0.617868 | 0.382132 |
| Panama | 9 | 0.371264 | 0.628736 |

INFERENCE:

At sigma 9, Panama's BND residual proxy has a high-frequency fraction similar to IUI3 and higher than Curasao/JapaneseGradens. This supports treating the Panama remainder as not purely low-frequency medium-like.

## 18. Edge Alignment

QUANTITATIVE RESULT:

Edge proxy uses top 20 percent GT luminance gradient magnitude.

| Scene | Residual-edge Pearson | Residual energy in top20 edge pixels |
| --- | ---: | ---: |
| Curasao | 0.086160 | 0.253830 |
| JapaneseGradens | 0.352795 | 0.458493 |
| IUI3 | 0.335710 | 0.427001 |
| Panama | 0.329556 | 0.512297 |

INFERENCE:

Panama has the largest fraction of BND residual energy in the top20 edge proxy among the audited scenes. This does not prove an object-only issue because direct/medium hybrid replacement failed, but it rules against a simple low-frequency medium-only explanation.

## 19. Panama Root-Cause Analysis

QUANTITATIVE RESULT:

Panama:

- Delta PSNR: `-0.810558 dB`
- Recomposition efficiency: `0.928829`
- RGB flat DeltaD/DeltaB correlation: `-0.673177`
- Cos p50: `-0.999543`
- P(cos<-0.9): `0.972134`
- r_DB p50: `0.998721`
- MSE terms: `C_direct=0.00433061`, `C_medium=0.00416626`, `C_cross=-0.00837868`, final `DeltaMSE=0.00011820`
- `J>1` mask DeltaMSE: `0.00225984`
- `J<=1` mask DeltaMSE: `0.00000438`
- `T<0.1` mask DeltaMSE: `-0.00000416`
- H_D1_B0 recovery: `-57.318957`
- H_D0_B1 recovery: `-52.768581`
- Sigma-9 high-frequency residual fraction: `0.628736`
- Top20 edge residual energy fraction: `0.512297`

QUANTITATIVE CONCLUSION:

`COUPLED_OBJECT_MEDIUM_REMAINDER = TRUE`.

INFERENCE:

Panama's remaining gap is best described as coupled object-medium recomposition remainder with localized high-J / high-luminance / edge-associated residual structure. The evidence does not support a one-branch object-only or medium-only diagnosis.

## 20. Curasao/IUI3 Success Controls

QUANTITATIVE RESULT:

Curasao and IUI3 both show large component redistribution with small final RGB change:

| Scene | Delta PSNR | mean abs DeltaD | mean abs DeltaB | mean abs DeltaI | Recomp efficiency | C_cross |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Curasao | +0.022852 | 0.036488 | 0.036642 | 0.007706 | 0.881859 | -0.00300491 |
| IUI3 | +0.088590 | 0.038175 | 0.037198 | 0.008734 | 0.888840 | -0.00422139 |

QUANTITATIVE CONCLUSION:

The same BND bounded intrinsic parameterization can support RGB-safe recomposition when direct and medium terms jointly settle into a matched solution.

## 21. Japanese Structural Trade-Off

QUANTITATIVE RESULT:

JapaneseGradens:

- Delta PSNR: `-0.057979 dB`
- Delta SSIM: `0.896300 - 0.899500 = -0.003200`
- Delta LPIPS: `0.117283 - 0.120356 = -0.003073`
- Recomposition efficiency: `0.900560`
- Sigma-9 high-frequency fraction: `0.333963`
- Top20 edge residual energy fraction: `0.458493`

INFERENCE:

JapaneseGradens is not the same failure mode as Panama in this audit. The PSNR change is small, LPIPS improves numerically, and edge/high-frequency proxies are elevated relative to Curasao but below Panama's sigma-9 high-frequency and top-edge residual energy fractions.

## 22. Final Classification

QUANTITATIVE RESULT:

Classification flags:

| Flag | Value |
| --- | --- |
| RECOMPOSITION_STRONG_CROSS_SCENE | TRUE |
| PANAMA_RECOMPOSITION_INCOMPLETE | FALSE |
| OBJECT_DOMINATED_REMAINDER | FALSE |
| MEDIUM_DOMINATED_REMAINDER | FALSE |
| COUPLED_OBJECT_MEDIUM_REMAINDER | TRUE |
| VIEW_CONTEXT_REMAINDER | FALSE |
| MIXED_RECOMPOSITION_REMAINDER | FALSE |

NUMERIC BASIS:

- Panama cancellation efficiency: `0.928829`
- Curasao/IUI3 success-control efficiency mean: `0.885350`
- Panama Delta PSNR: `-0.810558`
- Panama H_D1_B0 recovery: `-57.318957`
- Panama H_D0_B1 recovery: `-52.768581`
- Panama low-frequency energy fraction at sigma 9: `0.371264`
- Panama edge top20 residual energy fraction: `0.512297`
- Panama view efficiency range: `0.013678`

QUANTITATIVE CONCLUSION:

Panama is not classified as globally recomposition-incomplete by the current efficiency rule. The strongest supported classification is coupled object-medium remainder.

## 23. Next Single-Factor Experiment Recommendation

NEXT_SINGLE_FACTOR_EXPERIMENT:

`Panama BND staged object-medium optimization test`

INFERENCE:

The next single-factor experiment should test optimization coupling rather than immediately changing appearance representation or medium representation. The direct/medium cross term dominates the exact MSE attribution, single-branch image-space hybrids do not recover Panama's gap, and global recomposition efficiency is high. This makes a staged/asymmetric object-medium optimization diagnostic the highest-information next test under the current evidence.

UNVERIFIED HYPOTHESIS:

A staged object-medium procedure may help Panama reach a better matched bounded decomposition without changing renderer physics or reintroducing archived GMVC modules. This remains untested.

## Visual Assets

EXPERIMENTAL FACT:

Visual outputs were generated for external/manual analysis only.

Visual index:

- `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_object_medium_recomposition_20260810/VISUAL_COMPARE_INDEX.md`

Curasao:

- `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_object_medium_recomposition_20260810/Curasao/contact_sheet_component_decomposition.png`
- `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_object_medium_recomposition_20260810/Curasao/contact_sheet_cancellation.png`
- `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_object_medium_recomposition_20260810/Curasao/contact_sheet_residual_frequency.png`
- `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_object_medium_recomposition_20260810/Curasao/contact_sheet_hybrid_counterfactuals.png`

JapaneseGradens:

- `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_object_medium_recomposition_20260810/JapaneseGradens/contact_sheet_component_decomposition.png`
- `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_object_medium_recomposition_20260810/JapaneseGradens/contact_sheet_cancellation.png`
- `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_object_medium_recomposition_20260810/JapaneseGradens/contact_sheet_residual_frequency.png`
- `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_object_medium_recomposition_20260810/JapaneseGradens/contact_sheet_hybrid_counterfactuals.png`

IUI3:

- `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_object_medium_recomposition_20260810/IUI3/contact_sheet_component_decomposition.png`
- `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_object_medium_recomposition_20260810/IUI3/contact_sheet_cancellation.png`
- `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_object_medium_recomposition_20260810/IUI3/contact_sheet_residual_frequency.png`
- `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_object_medium_recomposition_20260810/IUI3/contact_sheet_hybrid_counterfactuals.png`

Panama:

- `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_object_medium_recomposition_20260810/Panama/contact_sheet_component_decomposition.png`
- `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_object_medium_recomposition_20260810/Panama/contact_sheet_cancellation.png`
- `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_object_medium_recomposition_20260810/Panama/contact_sheet_residual_frequency.png`
- `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_object_medium_recomposition_20260810/Panama/contact_sheet_hybrid_counterfactuals.png`
- `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_object_medium_recomposition_20260810/Panama/contact_sheet_panama_stratification_maps.png`

No subjective clear-image correctness judgment was made.
