# BND Loss Responsibility Alignment Audit - 2026-08-10

## 1. Motivation

HYPOTHESIS: The remaining Panama BND RGB gap may be materially limited by spatial loss-responsibility allocation. The suspected regions are localized legacy-high-J / bright / edge or bottom-image areas rather than the full image.

This audit is read-only. It uses existing checkpoints, forward passes, no-step autograd, region masks, counterfactual loss maps, and visual contact sheets. It does not train, change an optimizer, change a scheduler, densify, prune, or modify any checkpoint.

## 2. Current Panama Evidence

EXPERIMENTAL FACT: Prior Panama M1 vs BND evidence showed the RGB gap is localized. This audit therefore separates pixel error, formal loss contribution, image-space gradient responsibility, and parameter-space regional gradients.

## 3. Closed / Deprioritized Mechanisms

EXPERIMENTAL FACT: This audit did not reopen appearance LR, K2/K4, D010, medium LR, medium hold, GMVC, SH0, AA, HR, BG/BGI, or new renderer/loss modules.

## 4. Loss-Responsibility Hypothesis

HYPOTHESIS: A region can be small in pixel count but large in error. That alone does not prove under-weighting. The required comparison is:

```text
ERROR SHARE
vs
FORMAL LOSS SHARE
vs
GRADIENT SHARE
```

## 5. Formal WaterSplatting Loss Semantics

CODE FACT: `water_splatting/water_splatting.py:get_loss_dict` uses:

```text
gt_img = composite_with_background(get_gt_img(batch["image"]), outputs["background"])
pred_img = outputs["pred_image"]
```

If `batch["mask"]` exists, both GT and prediction are multiplied by the mask before loss. Formal M1/BND config uses:

```text
main_loss = reg_l1
ssim_loss = reg_ssim
ssim_lambda = 0.2
```

## 6. Formal Loss Equation

CODE FACT:

```text
L_reg_l1 =
mean_{x,c} |(GT[x,c] - pred[x,c]) / (stopgrad(pred[x,c]) + 1e-3)|

L_reg_ssim =
1 - SSIM(
    GT / (stopgrad(pred) + 1e-3),
    pred / (stopgrad(pred) + 1e-3)
)

L_main = 0.8 * L_reg_l1 + 0.2 * L_reg_ssim
```

`L_reg_l1` is pixel/channel-decomposable. `L_reg_ssim` is not treated as an independent per-pixel loss; spatial responsibility is measured through `dL/dI_pred`.

## 7. Region Definitions

EXPERIMENTAL FACT:

- `M1_HIGH_J`: M1 object support (`accumulation > 0.01`) and `clear_object_fullsh_raw.max_rgb > 1.0`.
- `M1_LOW_J`: M1 object support and `clear_object_fullsh_raw.max_rgb <= 1.0`.
- `BRIGHT_Q5`: top 20 percent GT luminance across Panama eval views.
- `BOTTOM20`: image rows with normalized y >= 0.8.
- `EDGE_TOP20`: top 20 percent GT luminance gradient magnitude.
- `LOW_TRANSMISSION`: M1 object support and min-RGB transmission < 0.1.

Eval views: `MTN_1539`, `MTN_1529`, `MTN_1547`.

## 8. Pixel-Error Localization

QUANTITATIVE RESULT:

| Region | Pixel Fraction | L1 Error Share | MSE Error Share | MSE Enrichment |
| --- | ---: | ---: | ---: | ---: |
| M1_HIGH_J | 0.050461 | 0.152593 | 0.337810 | 6.694488 |
| M1_LOW_J | 0.949481 | 0.847372 | 0.662183 | 0.697416 |
| BRIGHT_Q5 | 0.200000 | 0.368983 | 0.602325 | 3.011628 |
| BOTTOM20 | 0.200337 | 0.307071 | 0.449927 | 2.245856 |
| EDGE_TOP20 | 0.200001 | 0.352233 | 0.511382 | 2.556898 |
| LOW_TRANSMISSION | 0.010583 | 0.007538 | 0.002255 | 0.213124 |

QUANTITATIVE CONCLUSION: `ERROR_LOCALIZED = TRUE` because `M1_HIGH_J` MSE enrichment is 6.694488.

## 9. Formal Loss Contribution

QUANTITATIVE RESULT:

| Region | reg_l1 Loss Share | reg_l1 Loss Enrichment |
| --- | ---: | ---: |
| M1_HIGH_J | 0.069292 | 1.373173 |
| M1_LOW_J | 0.930683 | 0.980202 |
| BRIGHT_Q5 | 0.225037 | 1.125188 |
| BOTTOM20 | 0.337082 | 1.682579 |
| EDGE_TOP20 | 0.264161 | 1.320799 |
| LOW_TRANSMISSION | 0.006213 | 0.587107 |

INFERENCE: The high-J region receives more decomposable `reg_l1` loss than its pixel fraction, but much less than its MSE error share.

## 10. Image-Space Gradient Responsibility

QUANTITATIVE RESULT for formal total image gradient (`0.8*reg_l1 + 0.2*reg_ssim`, RGB L2 magnitude):

| Region | Gradient Share | Gradient Enrichment |
| --- | ---: | ---: |
| M1_HIGH_J | 0.020002 | 0.396388 |
| M1_LOW_J | 0.979966 | 1.032107 |
| BRIGHT_Q5 | 0.107012 | 0.535059 |
| BOTTOM20 | 0.222505 | 1.110656 |
| EDGE_TOP20 | 0.143955 | 0.719771 |
| LOW_TRANSMISSION | 0.007344 | 0.693945 |

For `M1_HIGH_J`, `reg_l1` gradient share is 0.021467 and `reg_ssim` gradient share is 0.020009. Both are far below MSE error share 0.337810.

## 11. Parameter-Space Regional Gradients

QUANTITATIVE RESULT: Natural-mass `M1_HIGH_J` regional gradient norm relative to total `reg_l1` gradient, averaged over Panama eval views:

| Parameter Group | Norm / Total | Cosine With Total |
| --- | ---: | ---: |
| features_dc | 0.065672 | 0.147270 |
| features_rest | 0.065780 | 0.147622 |
| means | 0.238685 | 0.323822 |
| scales | 0.158985 | 0.292332 |
| opacities | 0.145853 | 0.295485 |
| medium_mlp | 0.085462 | 0.475179 |

EXPERIMENTAL FACT: `direction_encoding` had zero trainable gradient mass in this audit.

## 12. High-J vs Low-J Gradient Conflict

QUANTITATIVE RESULT: Panama `M1_HIGH_J` vs `M1_LOW_J` natural-mass gradient cosines:

| Parameter Group | Cosine | High-J Norm / Total | Low-J Norm / Total |
| --- | ---: | ---: | ---: |
| features_dc | 0.082363 | 0.065672 | 0.991332 |
| features_rest | 0.082549 | 0.065780 | 0.991417 |
| means | 0.089652 | 0.238685 | 0.948302 |
| medium_mlp | 0.447036 | 0.085462 | 0.934442 |

QUANTITATIVE CONCLUSION: `HIGHJ_GRADIENT_CONFLICT = FALSE` under the preregistered threshold because no appearance cosine is <= -0.20 with non-negligible norms.

## 13. Bright vs Non-Bright Gradient Conflict

QUANTITATIVE RESULT: Panama `BRIGHT_Q5` vs non-bright natural-mass gradient cosines:

| Parameter Group | Cosine | Bright Norm / Total | Non-Bright Norm / Total |
| --- | ---: | ---: | ---: |
| features_dc | 0.068696 | 0.371981 | 0.883945 |
| features_rest | 0.068870 | 0.372416 | 0.884306 |
| means | 0.117554 | 0.509201 | 0.798318 |
| medium_mlp | 0.718742 | 0.302879 | 0.742701 |

INFERENCE: This audit does not identify a strong negative-gradient conflict as the primary explanation.

## 14. Responsibility Ratio

QUANTITATIVE RESULT:

| Region | MSE Error Share | Formal Total Gradient Share | Responsibility Ratio |
| --- | ---: | ---: | ---: |
| M1_HIGH_J | 0.337810 | 0.020002 | 0.059211 |
| M1_LOW_J | 0.662183 | 0.979966 | 1.479903 |
| BRIGHT_Q5 | 0.602325 | 0.107012 | 0.177664 |
| BOTTOM20 | 0.449927 | 0.222505 | 0.494536 |
| EDGE_TOP20 | 0.511382 | 0.143955 | 0.281502 |
| LOW_TRANSMISSION | 0.002255 | 0.007344 | 3.256058 |

QUANTITATIVE CONCLUSION: `FAILURE_REGION_UNDER_EMPHASIZED = TRUE` for `M1_HIGH_J`: MSE enrichment is 6.694488, responsibility ratio is 0.059211, and gradient enrichment 0.396388 is below MSE enrichment.

## 15. SeaFree Content-Based Loss Semantics

CODE FACT from SeaFree-GS commit `7797e97dae831029ac89ae9f37b3c3d69ec2cf6c`, `seafree_gs/seafree_model.py`:

- `batch["depth_image"]` is normalized by its max.
- Foreground mask uses threshold `1e-2`, inverse binary threshold, largest contour fill, and binarization.
- Reconstruction weight is `1 / (rendered_underwater_image.detach() + 1e-3)` on foreground pixels, and `1` on background pixels.
- Weighted L1 is `abs((GT - rendered) * weight).mean()`.
- Weighted DSSIM is `1 - SSIM(GT*weight, rendered*weight)`.
- Background-water supervision and coarse depth loss exist in SeaFree, but are not included in this audit's CB-weight counterfactual.

EXPERIMENTAL FACT: Panama WaterSplatting batches did not provide SeaFree-compatible `depth_image`. Accumulation was not substituted. Therefore the SeaFree counterfactual is marked intensity-only reference, not full SeaFree foreground reference.

## 16. SeaFree Weight Alignment

QUANTITATIVE RESULT:

| Region | SeaFree Intensity Weight Enrichment |
| --- | ---: |
| M1_HIGH_J | 0.434562 |
| M1_LOW_J | 1.030292 |
| BRIGHT_Q5 | 0.557357 |
| BOTTOM20 | 1.201642 |
| EDGE_TOP20 | 0.715388 |

QUANTITATIVE CONCLUSION:

- `SEAFREE_HIGHJ_ALIGNED = FALSE`
- `SEAFREE_BRIGHT_ALIGNED = FALSE`
- `SEAFREE_HIGHJ_DOWNWEIGHTS = TRUE`
- `SEAFREE_BRIGHT_DOWNWEIGHTS = TRUE`

## 17. SeaFree Counterfactual Responsibility

QUANTITATIVE RESULT:

| Region | Formal Gradient Share | SeaFree-Weighted Gradient Share | Relative Change |
| --- | ---: | ---: | ---: |
| M1_HIGH_J | 0.020002 | 0.020034 | +0.001608 |
| BRIGHT_Q5 | 0.107012 | 0.109076 | +0.019290 |

QUANTITATIVE CONCLUSION: `SEAFREE_CB_ALIGNED = FALSE` because neither high-J nor bright gradient share increases by >=25%. `SEAFREE_CB_ANTI_ALIGNED = FALSE` under the strict preregistered rule because the gradient share did not decrease, although the raw intensity weights downweight both regions.

## 18. Oracle 2x Diagnostic

QUANTITATIVE RESULT for `M1_HIGH_J` 2x oracle weighting:

| Parameter Group | Formal-vs-Oracle Cosine | Oracle / Formal Magnitude |
| --- | ---: | ---: |
| features_dc | 0.999769 | 1.003554 |
| features_rest | 1.000000 | 1.003544 |
| medium_mlp | 0.999142 | 1.064997 |

Image-space `M1_HIGH_J` gradient share changes from 0.020002 to 0.022087.

INFERENCE: Oracle 2x reweighting slightly increases failure-region image-gradient share but does not materially rotate the appearance-parameter gradient in this no-step diagnostic.

## 19. Deployable Proxy Alignment

QUANTITATIVE RESULT against diagnostic target `M1_HIGH_J`:

| Proxy | AUROC | AUPRC | Top20 Enrichment |
| --- | ---: | ---: | ---: |
| GT/input brightness | 0.977303 | 0.822426 | 4.830698 |
| current prediction brightness | 0.979758 | 0.850316 | 4.844618 |
| current abs residual | 0.776140 | 0.234752 | 3.040457 |
| GT edge magnitude | 0.828426 | 0.248896 | 3.428077 |
| pseudo-depth foreground | unavailable | unavailable | unavailable |

QUANTITATIVE CONCLUSION: `DEPLOYABLE_PROXY_EXISTS = TRUE` by the preregistered proxy threshold. The strongest non-oracle proxy in this audit is brightness. Residual is recorded separately and should not be interpreted as SeaFree CB evidence.

## 20. Cross-Scene Controls

QUANTITATIVE RESULT for `M1_HIGH_J`:

| Scene | Pixel Fraction | MSE Enrichment | Formal Total Grad Enrichment | Responsibility Ratio |
| --- | ---: | ---: | ---: | ---: |
| Panama | 0.050461 | 6.694488 | 0.396388 | 0.059211 |
| Curasao | 0.048568 | 3.715134 | 0.418875 | 0.112748 |
| IUI3 | 0.028868 | 3.352622 | 0.442398 | 0.131956 |

INFERENCE: Panama shows stronger high-J error localization and lower high-J responsibility ratio than the two control scenes in this audit.

## 21. Final Loss-Responsibility Classification

QUANTITATIVE CONCLUSION:

```text
ERROR_LOCALIZED = TRUE
FAILURE_REGION_UNDER_EMPHASIZED = TRUE
HIGHJ_GRADIENT_CONFLICT = FALSE
OVERALL_HYPOTHESIS = PARTIALLY_SUPPORTED
```

Reason for `PARTIALLY_SUPPORTED` rather than `SUPPORTED`: the failure region is localized and under-emphasized in image-gradient responsibility, but oracle 2x weighting does not materially rotate appearance-parameter gradients in this no-step diagnostic.

## 22. SeaFree-Specific Classification

QUANTITATIVE CONCLUSION:

```text
SEAFREE_CB_ALIGNED = FALSE
SEAFREE_CB_ANTI_ALIGNED = FALSE
SEAFREE_SPECIFIC_HYPOTHESIS = NOT_SUPPORTED
```

Reason: SeaFree intensity weighting downweights both `M1_HIGH_J` and `BRIGHT_Q5`, and its counterfactual gradient-share increase is far below the preregistered 25% alignment threshold.

## 23. Next Single-Factor Recommendation

RECOMMENDATION: Do not start a weighted-loss training run from this audit alone. The next single-factor step should be a no-training, no-step brightness-proxy gradient projection diagnostic:

```text
W_bright = detached fixed brightness-derived proxy
compare formal vs W_bright objective:
  image-gradient share
  appearance/medium parameter-gradient cosine
  oracle/high-J alignment
```

This is recommended because brightness is deployable and strongly aligned with the diagnostic high-J target, while original SeaFree inverse-intensity CB weighting is not aligned for Panama.

## Outputs

- Output manifest: `outputs/lossresp_audit_20260810/manifest.json`
- Formal loss equation: `outputs/lossresp_audit_20260810/formal_loss_equation.md`
- Visual index: `renders/lossresp_panama_20260810/VISUAL_COMPARE_INDEX.md`
- Failure localization sheet: `renders/lossresp_panama_20260810/contact_sheet_failure_localization.png`
- Formal loss map sheet: `renders/lossresp_panama_20260810/contact_sheet_formal_loss_map.png`
- Formal image-gradient sheet: `renders/lossresp_panama_20260810/contact_sheet_formal_image_gradients.png`
- Responsibility overlay sheet: `renders/lossresp_panama_20260810/contact_sheet_responsibility_overlay.png`
- SeaFree weight sheet: `renders/lossresp_panama_20260810/contact_sheet_seafree_weight_alignment.png`
- Oracle 2x sheet: `renders/lossresp_panama_20260810/contact_sheet_oracle_2x_diagnostic.png`
- Proxy alignment sheet: `renders/lossresp_panama_20260810/contact_sheet_proxy_alignment.png`
- Cross-scene control sheet: `renders/lossresp_panama_20260810/contact_sheet_cross_scene_control.png`

## Technical Validation

EXPERIMENTAL FACT: `NO_PARAMETER_DELTA_AUDIT` reported max absolute parameter delta 0.0 for `features_dc`, `features_rest`, `means`, `opacities`, and `medium_mlp_flat`.

Visual assets are ready for external/manual analysis.
No subjective clear-image correctness judgment was made.
