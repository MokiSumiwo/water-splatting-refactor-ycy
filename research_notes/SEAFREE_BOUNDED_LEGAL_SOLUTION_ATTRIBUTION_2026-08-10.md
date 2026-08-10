# SeaFree-GS Bounded Legal-Solution Attribution Audit

## 1. Motivation

HYPOTHESIS: SeaFree-GS is an important bounded-intrinsic reference because it uses bounded SH0 intrinsic colors, yet may recover part of the Panama RGB gap that WaterSplatting BND-K1 leaves after removing the legacy M1 high-J route.

EXPERIMENTAL DISCIPLINE: This stage is a reference-solution audit. No WaterSplatting training was run, no WaterSplatting optimizer step was executed, and no M1/BND-K1/AA/HR/UNORM checkpoint was modified.

## 2. Repository State

CODE FACT: WaterSplatting repository:

- Path: `/mnt/new/home_old/ycy/water-splatting-refactor`
- Branch at start of audit: `research/m1-bounded-intrinsic`
- Start HEAD: `25ee929de73a967ea91fc3c76a4e31ca628cf22a`
- Start status contained only the historical untracked GMVC scripts:
  - `scripts/diagnostics/render_gmvc_curasao_contact_sheet.py`
  - `scripts/diagnostics/summarize_gmvc_four_scene_visual_audit.py`

CODE FACT: SeaFree-GS reference repository:

- Path: `/mnt/new/home_old/ycy/reference_repos/SeaFree-GS`
- Reference commit: `7797e97dae831029ac89ae9f37b3c3d69ec2cf6c`
- Reference repo status at audit: clean
- SeaFree reference repository was read-only for this task.

## 3. Current WaterSplatting Evidence

EXPERIMENTAL FACT: Prior Panama references used in this audit:

| Run | PSNR | SSIM | LPIPS | MSE |
| --- | ---: | ---: | ---: | ---: |
| M1 | 32.308910 | 0.949487 | 0.073979 | 0.000595238 |
| BND-K1 | 31.498353 | 0.948783 | 0.075521 | 0.000714139 |

EXPERIMENTAL FACT: Prior Panama decomposition proxies:

| Run | canonical tau p90 | canonical J p99 | P(J>1) |
| --- | ---: | ---: | ---: |
| M1 | 1.769849 | 1.311801 | 0.037758 |
| BND-K1 | 0.999069 | 0.838911 | 0.000000 |

INTERPRETATION: `M1_HIGH_J` is a WaterSplatting M1 solution property, not a Panama clear-GT label. It identifies the pixels where M1 chose a high-radiance compensation route.

## 4. SeaFree Source Facts

CODE FACT: SeaFree official config registers:

- `max_num_iterations = 30000`
- `steps_per_save = 2000`
- `save_only_latest_checkpoint = true` in the produced run config
- `rasterize_mode = antialiased`
- `sh_degree = 0`
- `stop_split_at = 15000`
- `reset_alpha_value = 0.5`
- `cull_alpha_thresh = 0.5`
- `cull_alpha_thresh_post = 0.1`
- `enable_coarse_grained_depth_loss = true`
- `enable_background_water_supervision = true`

CODE FACT: SeaFree bounded intrinsic appearance:

- In SH0 mode, SeaFree applies `torch.sigmoid(colors_crop).squeeze(1)` to the Gaussian DC color parameters before degradation/rasterization.
- The sigmoid output is then degraded by direct attenuation and backscatter before the underwater RGB channels are rendered.
- The intrinsic render channels are rendered as channels 3:6 and returned as `intrinsic_color_render` after clamp to `[0,1]`.

CODE FACT: SeaFree direct/backscatter degradation:

- Gaussian LOS distance is `norm(mean - camera_center)`.
- SeaFree uses `normalized_gaussian_line_of_sight_distances = gaussian_line_of_sight_distances / 10`.
- Degraded Gaussian color is:
  - `colors * exp(-beta_D * distance/10) + A * (1 - exp(-beta_B * distance/10))`
- Pixel water background is the WPP ambient output for pixel LOS directions.
- Final underwater RGB is `render[..., :3] + (1 - alpha) * water_background_image`, clamped to `[0,1]`.

CODE FACT: SeaFree foreground-aware content loss:

- Builds a foreground mask from normalized `depth_image` using threshold `1e-2`, inverse thresholding, largest contour fill, and binarization.
- Uses weight `1 / (rendered_underwater_image.detach() + 1e-3)` on foreground pixels and `1` on background pixels.
- Uses weighted L1 plus weighted DSSIM with `ssim_lambda = 0.2`.

CODE FACT: SeaFree background-water supervision:

- Enabled when `step < 15000` and `background_pixel_ratio > 0.05`.
- Supervises `water_background_image` against GT underwater pixels on the background side of the foreground mask.
- Uses weight `1 / (background_ambient_light_pixels.detach() + 1e-3)`.
- Enters content-based reconstruction loss with coefficient `0.01`.

CODE FACT: SeaFree coarse depth loss:

- Pseudo-depth source is `depth_image`, normalized by its own max.
- Rendered depth is expected depth from rasterization.
- Approximate rendered disparity is `1 / (rendered_depth * 10 + 1)`.
- Loss is `0.1 * (1 - Pearson(pseudo_depth, approximate_rendered_disparity))`.

## 5. SeaFree Reference Validity

EXPERIMENTAL FACT: No existing valid SeaFree Panama checkpoint was found before this audit. Available SeaFree checkpoints outside this repo were not Panama or could not be tied to the fixed SeaFree reference commit.

EXPERIMENTAL FACT: A one-step smoke run confirmed that fixed commit `7797e97dae831029ac89ae9f37b3c3d69ec2cf6c`, the Panama dataset, and the official SeaFree config can load and train without modifying the SeaFree reference repository.

EXPERIMENTAL FACT: One official SeaFree Panama reference reproduction was run with the fixed source and unmodified method settings. Only output directory, experiment name, timestamp, vis backend, dataset path, and dataparser paths were provided on the CLI.

SeaFree reference:

- Config: `outputs/seafree_legal_panama_20260810/reference_reproduction/panama_seafree_gs_reference_30k/seafree-gs/20260810_reference/config.yml`
- Checkpoint: `outputs/seafree_legal_panama_20260810/reference_reproduction/panama_seafree_gs_reference_30k/seafree-gs/20260810_reference/nerfstudio_models/step-000029999.ckpt`
- Loaded step: `29999`
- Dataset/images: `undistorted_data/undistorted_Panama/images/ColorImage`
- Depth path: `undistorted_data/undistorted_Panama/depthAnything_u16`

QUANTITATIVE CONCLUSION: `SEAFREE_REFERENCE_VALID = TRUE`.

## 6. Evaluation Alignment

EXPERIMENTAL FACT: Common eval views:

- `MTN_1529`
- `MTN_1539`
- `MTN_1547`

EXPERIMENTAL FACT: SeaFree `val_list.txt` and `test_list.txt` both contain these three views. WaterSplatting test loading resolves to the same three view IDs.

EXPERIMENTAL FACT: GT alignment checks:

- M1 vs BND-K1: same shape for all views, max absolute GT difference `0.0`
- M1 vs SeaFree: same shape for all views, max absolute GT difference `0.0`

QUANTITATIVE CONCLUSION: `GLOBAL_RGB_METRICS_DIRECTLY_COMPARABLE = TRUE`.

## 7. Global RGB Comparison

QUANTITATIVE RESULT:

| Run | Step | PSNR | SSIM | LPIPS | MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| M1 | 14999 | 32.308910 | 0.949487 | 0.073979 | 0.000595238 |
| BND-K1 | 14999 | 31.498353 | 0.948783 | 0.075521 | 0.000714139 |
| SeaFree | 29999 | 31.725087 | 0.944203 | 0.088957 | 0.000676841 |

QUANTITATIVE CONCLUSION: SeaFree's global MSE is lower than BND-K1 on the aligned Panama eval views, but higher than M1. SeaFree's SSIM and LPIPS are lower than both WaterSplatting references in this reproduction.

## 8. Fixed M1_HIGH_J Local Recovery

CODE FACT: Fixed `M1_HIGH_J` mask definition:

- WaterSplatting M1 object support: `accumulation > 0.01`
- and `max_rgb(clear_object_fullsh_raw) > 1.0`

QUANTITATIVE RESULT:

| Region | Pixel fraction | M1 MSE | BND-K1 MSE | SeaFree MSE |
| --- | ---: | ---: | ---: | ---: |
| M1_HIGH_J | 0.050461 | 0.002520954 | 0.004780798 | 0.003383297 |

QUANTITATIVE RESULT:

`SEAFREE_HIGHJ_GAP_RECOVERY = 0.618406`

where:

`(MSE_K1_highJ - MSE_SeaFree_highJ) / (MSE_K1_highJ - MSE_M1_highJ)`.

QUANTITATIVE CONCLUSION: `SEAFREE_HIGHJ_LOCAL_RECOVERY = TRUE` by the registered threshold (`>= 0.25` and SeaFree high-J MSE below K1).

## 9. Control Regions

QUANTITATIVE RESULT:

| Region | Pixel fraction | Run | MSE | L1 |
| --- | ---: | --- | ---: | ---: |
| M1_LOW_J | 0.949481 | M1 | 0.000493670 | 0.012323 |
| M1_LOW_J | 0.949481 | BND-K1 | 0.000498052 | 0.012478 |
| M1_LOW_J | 0.949481 | SeaFree | 0.000533044 | 0.013499 |
| Bright Q5 | 0.200000 | M1 | 0.001543909 | 0.021943 |
| Bright Q5 | 0.200000 | BND-K1 | 0.002150722 | 0.025794 |
| Bright Q5 | 0.200000 | SeaFree | 0.001814401 | 0.025333 |
| Bright-not-Q5 | 0.800000 | M1 | 0.000358951 | 0.011028 |
| Bright-not-Q5 | 0.800000 | BND-K1 | 0.000354994 | 0.011028 |
| Bright-not-Q5 | 0.800000 | SeaFree | 0.000392452 | 0.012060 |

QUANTITATIVE CONCLUSION: SeaFree's recovery is concentrated in the fixed high-J and bright control regions relative to BND-K1, while it is not lower-MSE than BND-K1 in the M1_LOW_J or Bright-not-Q5 controls.

## 10. Intrinsic Appearance and Boundary Usage

QUANTITATIVE RESULT: Fixed M1_HIGH_J intrinsic render statistics:

| Run | Source | Mean | p99 | P(value>0.99) |
| --- | --- | ---: | ---: | ---: |
| M1 | `clear_object_fullsh_raw` | 1.142377 | 1.939002 | 0.764468 |
| BND-K1 | `clear_object_fullsh_raw` | 0.716139 | 0.999732 | 0.036059 |
| SeaFree | `intrinsic_color_render` | 0.618737 | 0.999511 | 0.023889 |

QUANTITATIVE RESULT: Visible Gaussian color boundary usage:

| Run | P(c>0.95) | P(c>0.99) | P(c<0.01) |
| --- | ---: | ---: | ---: |
| BND-K1 | 0.020831 | 0.017517 | 0.000001 |
| SeaFree | 0.067824 | 0.054160 | 0.002008 |

QUANTITATIVE CONCLUSION: `SEAFREE_BOUNDARY_HEAVY = TRUE` by the registered rule `P(c>0.99) > 0.05` on visible Gaussian colors.

INFERENCE: SeaFree uses legal bounded color boundary more heavily than BND-K1 at the visible-Gaussian distribution level. This is a solution-mechanism fact, not a correctness judgment.

## 11. Geometry / Pseudo-Depth Agreement

CODE FACT: Pseudo-depth is `depthAnything_u16` normalized per image. It is a diagnostic reference only, not GT geometry.

CODE FACT: Depth comparison uses SeaFree-style approximate disparity:

`1 / (rendered_depth * 10 + 1)`.

QUANTITATIVE RESULT:

| Region | Run | Spearman | Pearson | aligned MAE | aligned RMSE | gradient Pearson |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| M1_HIGH_J | BND-K1 | 0.985837 | 0.985610 | 0.029016 | 0.040947 | 0.261865 |
| M1_HIGH_J | SeaFree | 0.990782 | 0.990268 | 0.025448 | 0.033772 | 0.401077 |
| M1_LOW_J | BND-K1 | 0.968824 | 0.966485 | 0.044217 | 0.067071 | 0.171246 |
| M1_LOW_J | SeaFree | 0.989379 | 0.988182 | 0.029958 | 0.040636 | 0.285697 |

QUANTITATIVE CONCLUSION: SeaFree has stronger agreement with the same pseudo-depth diagnostic reference than BND-K1 in both high-J and low-J regions under the registered metrics.

INFERENCE: Geometry/depth is an eligible mechanism candidate. This does not prove that SeaFree geometry is physically more accurate, because pseudo-depth is not GT geometry.

## 12. Alpha / Coverage / Population

QUANTITATIVE RESULT: High-J region alpha accumulation:

| Run | mean | p50 | p99 | P(acc>0.99) |
| --- | ---: | ---: | ---: | ---: |
| M1 | 0.997369 | 0.999749 | 0.999900 | 0.943058 |
| BND-K1 | 0.996633 | 0.999777 | 0.999900 | 0.945753 |
| SeaFree | 0.994101 | 0.999537 | 0.999899 | 0.881181 |

QUANTITATIVE RESULT: Gaussian population:

| Run | Gaussian count | opacity mean | scale p50 | scale p99 |
| --- | ---: | ---: | ---: | ---: |
| M1 | 1173293 | 0.708701 | 0.000795 | 0.011846 |
| BND-K1 | 1177886 | 0.726313 | 0.000793 | 0.013251 |
| SeaFree | 1693923 | 0.893089 | 0.000494 | 0.022172 |

EXPERIMENTAL FACT: SeaFree projected radius statistics were exported from gsplat info. WaterSplatting projected radius was not exported by this script, so direct projected-radius enrichment is not evaluated.

INFERENCE: SeaFree differs materially in population size and opacity distribution, but this audit does not isolate population as a dominant factor.

## 13. Medium Diagnostics

CODE FACT: SeaFree exact per-Gaussian direct tau is not returned by `get_outputs`. This audit records a SeaFree pixel-depth proxy:

`background_attenuation_coefficients * rendered_expected_depth / 10`.

This is not the exact per-Gaussian direct tau used internally for degraded Gaussian colors.

QUANTITATIVE RESULT: High-J region medium proxies:

| Run | tau source | tau p90 | tau p99 | T mean | P(T<0.1) | medium mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BND-K1 | `tau_D` | 0.596510 | 1.279038 | 0.748182 | 0.000000 | 0.034166 |
| SeaFree | pixel-depth tau proxy | 0.293924 | 0.702795 | 0.873024 | 0.000000 | 0.180186 |

INFERENCE: SeaFree's exported medium proxy differs strongly from BND-K1 in the fixed high-J region, but medium cannot be classified dominant from this proxy alone.

## 14. Rasterization Evidence

CODE FACT: SeaFree official reference uses `rasterize_mode = antialiased`.

EXPERIMENTAL FACT: Prior WaterSplatting BND-AA causal evidence:

- Global PSNR gain over BND-K1: `+0.304330 dB`
- High-J gap recovery: approximately `0.271191`
- LPIPS and boundary-pressure side effects were observed in the prior BND-AA audit.

INFERENCE: Rasterization has already shown causal positive signal in WaterSplatting, but it is not a clean complete explanation by itself.

## 15. Trajectory / Post-15k

EXPERIMENTAL FACT: The official reproduced SeaFree config has `save_only_latest_checkpoint = true`. The run ended with only `step-000029999.ckpt` retained. Intermediate 10k/15k/20k/25k checkpoints were not available after training.

QUANTITATIVE CONCLUSION: SeaFree training trajectory and post-15k high-J recovery are `NOT_EVALUABLE` from retained checkpoints in this audit.

## 16. Coarse Depth Supervision

CODE FACT: SeaFree coarse depth supervision was enabled in the reference reproduction. Exact formula:

`0.1 * (1 - Pearson(pseudo_depth, 1 / (rendered_depth * 10 + 1)))`

QUANTITATIVE CONCLUSION: `DEPTH_FACTOR_ELIGIBLE = TRUE` because SeaFree shows stronger pseudo-depth agreement than BND-K1 in the fixed high-J region and low-J control under Spearman/Pearson, aligned RMSE, and gradient-correlation diagnostics.

## 17. Loss Responsibility

EXPERIMENTAL FACT: Prior LOSSRESP result:

- SeaFree high-J weight enrichment: `0.434562`
- Brightness-Q5 weight enrichment: `0.557357`

QUANTITATIVE CONCLUSION: SeaFree Content-Based loss weighting remains evidence against a direct high-J emphasis explanation. SeaFree fits the fixed high-J region better than BND-K1 despite this prior anti-aligned weighting.

## 18. Factor Scorecard

| Factor | Score | Evidence |
| --- | --- | --- |
| AA / rasterization | MODERATE_EVIDENCE | SeaFree official mode is antialiased; prior BND-AA recovered +0.304330 dB. |
| geometry / depth | MODERATE_EVIDENCE | High-J pseudo-depth Spearman 0.985837 -> 0.990782; aligned RMSE 0.040947 -> 0.033772; gradient Pearson 0.261865 -> 0.401077. |
| Gaussian population / coverage | WEAK_EVIDENCE | SeaFree population and opacity differ, but high-J accumulation is only modestly different and projected-radius matching is unavailable for WaterSplatting in this script. |
| late refinement | NOT_EVALUABLE | Intermediate SeaFree checkpoints were not retained. |
| medium | WEAK_EVIDENCE | SeaFree tau is only a pixel-depth proxy in this audit. |
| appearance boundary | MODERATE_EVIDENCE | SeaFree visible `P(c>0.99)=0.054160`, above the boundary-heavy threshold. |
| Content-Based loss | EVIDENCE_AGAINST | Prior LOSSRESP found anti-aligned high-J/bright weighting. |
| degradation/compositing | EVIDENCE_AGAINST | Prior DCOMP found restricted-condition formula equivalence. |

## 19. Dominant Interpretation

QUANTITATIVE CONCLUSION: `Dominant Interpretation = MIXED`.

INFERENCE: The audit supports that SeaFree has a bounded legal solution that recovers a substantial part of the fixed M1_HIGH_J local MSE gap relative to BND-K1. The strongest supported mechanisms are a combination of:

- stronger pseudo-depth agreement,
- antialiased rasterization with prior causal support,
- heavier legal bounded-color boundary usage,
- and different Gaussian population/medium state.

No single factor is isolated as dominant by this read-only audit.

## 20. Main Scientific Conclusion

QUANTITATIVE CONCLUSION: SeaFree does recover the fixed WaterSplatting M1 high-J region relative to BND-K1:

`SEAFREE_HIGHJ_GAP_RECOVERY = 0.618406`.

REASONABLE INFERENCE: The Panama BND-K1 gap is not explained by bounded intrinsic appearance alone. A bounded legal alternative exists in the SeaFree reference solution, but it appears to arise from a different coupled solution involving geometry/depth, antialiased rasterization, boundary usage, and population/medium differences.

BOUNDARY: This audit does not show that SeaFree intrinsic appearance is physically correct, nor that pseudo-depth agreement is true geometry accuracy.

## 21. Next Single-Factor Experiment

RECOMMENDATION: Run exactly one next WaterSplatting causal experiment:

`BND-K1 + SeaFree-style coarse-depth supervision`

Scope:

- Panama only
- from scratch to 15k
- keep BND-K1 bounded SH3 appearance
- keep WaterSplatting renderer/degradation unchanged
- add only the SeaFree-style coarse-depth Pearson loss using existing `depthAnything_u16`
- no AA, no medium supervision, no loss reweighting, no GMVC

Reason: geometry/depth is the strongest not-yet-isolated SeaFree-linked factor in this audit. AA already has a WaterSplatting causal test, CB loss is evidence-against, and degradation/compositing is evidence-against under the prior restricted-condition audit.

## 22. Outputs

Diagnostic source:

- `scripts/diagnostics/audit_seafree_panama_legal_solution.py`

Output metrics:

- `outputs/seafree_legal_panama_20260810/global_rgb_comparison.csv`
- `outputs/seafree_legal_panama_20260810/high_j_local_recovery.csv`
- `outputs/seafree_legal_panama_20260810/depth_alignment_metrics.csv`
- `outputs/seafree_legal_panama_20260810/intrinsic_boundary_statistics.csv`
- `outputs/seafree_legal_panama_20260810/legal_solution_factor_scorecard.csv`
- `outputs/seafree_legal_panama_20260810/seafree_legal_final_summary.json`
- `outputs/seafree_legal_panama_20260810/manifest.json`

Visual assets:

- `renders/seafree_legal_panama_20260810/contact_sheet_underwater_m1_k1_seafree.png`
- `renders/seafree_legal_panama_20260810/contact_sheet_fixed_m1_high_j_residual.png`
- `renders/seafree_legal_panama_20260810/contact_sheet_low_j_control_residual.png`
- `renders/seafree_legal_panama_20260810/contact_sheet_brightness_q5_residual.png`
- `renders/seafree_legal_panama_20260810/contact_sheet_intrinsic_m1_k1_seafree.png`
- `renders/seafree_legal_panama_20260810/contact_sheet_boundary_use_k1_seafree.png`
- `renders/seafree_legal_panama_20260810/contact_sheet_depth_pseudo_k1_seafree.png`
- `renders/seafree_legal_panama_20260810/contact_sheet_alpha_coverage_k1_seafree.png`
- `renders/seafree_legal_panama_20260810/contact_sheet_medium_k1_seafree.png`
- `renders/seafree_legal_panama_20260810/contact_sheet_factor_summary.png`
- `renders/seafree_legal_panama_20260810/VISUAL_COMPARE_INDEX.md`

VISUAL DISCIPLINE: Visual assets are prepared for external/manual analysis. No subjective clear-image correctness judgment was made.
