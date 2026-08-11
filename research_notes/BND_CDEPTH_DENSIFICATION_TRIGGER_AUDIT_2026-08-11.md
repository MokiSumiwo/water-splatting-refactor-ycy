# BND-CDEPTH Densification Trigger Audit

## 1. Motivation

HYPOTHESIS: BND-CDEPTH may alter early Gaussian refinement / densification eligibility before the later high-J RGB recovery appears.

This audit tests that hypothesis with a read-only fixed-state intervention:

`same checkpoint + same Gaussians + same train cameras + same renderer`, comparing `R` against `RD`.

`R` is the formal BND-K1 RGB objective. `RD` is `R` plus the exact WaterSplatting implementation of the SeaFree-style coarse-depth term.

## 2. Current Bounded Reconstruction Trade-Off

EXPERIMENTAL FACT: Prior Panama BND-CDEPTH results remain the context for this audit:

- BND-CDEPTH improved PSNR over BND-K1 by `+0.254946 dB`.
- SSIM changed by `-0.002492`.
- LPIPS changed by `+0.005410`.

INFERENCE: CDEPTH is treated here as a partial mitigation mechanism, not as a final Pareto solution.

## 3. Negative Final-State Structural Findings

EXPERIMENTAL FACT: Previous audits did not support these final-state explanations:

- Better pseudo-depth geometry: not supported.
- Final finer projected footprint: `EVIDENCE_AGAINST`.
- Final higher local density: `EVIDENCE_AGAINST`.
- Robust final alpha rebalance: `WEAK`.
- NEW_LOW_T harm explanation: `LOWT_HARM_ALIGNED = FALSE`.
- Edge-localized harm explanation: `EDGE_HARM_ALIGNED = FALSE`.

## 4. Why Early Trigger Dynamics Are Tested

EXPERIMENTAL FACT: Previous trajectory timing:

- Population divergence onset: `1000`.
- Scale divergence onset: `1000`.
- Projected-footprint divergence onset: `1000`.
- Anisotropy divergence onset: `3000`.
- Alpha divergence onset: `3000`.
- High-J recovery onset: `5000`.
- Global recovery onset: `8000`.

INFERENCE: This timing makes early refinement / densification eligibility worth testing, but timing alone is not causal evidence.

## 5. Exact WaterSplatting Densification Source Logic

CODE FACT:

- Source files audited:
  - `water_splatting/water_splatting.py`
  - `water_splatting/rasterize.py`
  - `water_splatting/rendering/underwater_rasterizer.py`
- Callback path:
  - `WaterSplattingModel.get_training_callbacks`
  - `WaterSplattingModel.after_train`
  - `WaterSplattingModel.refinement_after`
  - `WaterSplattingModel.split_gaussians`
  - `WaterSplattingModel.dup_gaussians`
  - `WaterSplattingModel.cull_gaussians`

CODE FACT: The true grow trigger statistic is:

```text
avg_grad_norm_i =
    (xys_grad_norm_i / vis_counts_i)
    * 0.5 * max(last_size)
```

CODE FACT: In the active config, `abs_grad_densification=True`, so the accumulator input is:

```text
self.xys_grad_abs.detach().norm(dim=-1)
```

This is a screen-space projected `xys` absolute gradient statistic, not a 3D means-gradient statistic.

CODE FACT: Visibility handling:

- `visible_mask = (self.radii > 0).flatten()`.
- On first accumulator initialization, `vis_counts = ones_like(xys_grad_norm)`.
- On later views, `vis_counts[visible_mask] += 1`.
- `xys_grad_norm` and `depths_accum` are updated only for visible Gaussians after initialization.

CODE FACT: Gates:

- High-gradient gate: `avg_grad_norm > densify_grad_thresh`.
- `densify_grad_thresh = 0.0008`.
- Split gate: `exp(scales).max(dim=-1) > densify_size_thresh`.
- Duplicate gate: `exp(scales).max(dim=-1) <= densify_size_thresh`.
- `densify_size_thresh = 0.001`.
- 2D screen-size split gate is only active if `step < stop_screen_size_at`; the current default is `stop_screen_size_at = 0`, so it is inactive for positive steps.
- Opacity does not participate in grow eligibility.

CODE FACT: Pruning:

- Before `stop_split_at`, opacity cull threshold is `cull_alpha_thresh = 0.5`.
- At/after `stop_split_at`, opacity cull threshold is `cull_alpha_thresh_post = 0.1`.
- After `step > refine_every * reset_alpha_every`, 3D too-large cull uses `exp(scales).max > cull_scale_thresh = 10.0`.
- Current-loss gradients do not directly enter the pruning gate in a fixed-state comparison.

CODE FACT: Timing:

- `warmup_length = 500`.
- `refine_every = 100`.
- `reset_alpha_every = 5`.
- `reset_interval = 500`.
- `stop_split_at = 10000`.
- Densification condition:

```text
step < stop_split_at
and
step % reset_interval > num_train_data + refine_every
```

- Opacity reset occurs when:

```text
step < stop_split_at
and
step % reset_interval == refine_every
```

## 6. Trigger Statistic and Gradient Path

CODE FACT: The WaterSplatting CDEPTH term is:

```text
coarse_depth_loss =
    0.1 * (1 - pearson_corrcoef(
        normalized_pseudo_depth.flatten(),
        (1 / (10 * rendered_depth + 1)).flatten()
    ))
```

CODE FACT: `outputs["depth"]` is `UnderwaterRasterizer.depth`, computed as `depth_im / alpha` for alpha-supported pixels and a detached fill value where `alpha == 0`.

CODE FACT: The custom rasterizer Python backward currently forwards `v_out_img`, `v_out_medium`, and `v_out_alpha` into CUDA backward. It does not directly forward `v_depth_im`, but `depth_expected` depends on `alpha`, so the coarse-depth term can reach the rasterization path through alpha-gradient coupling.

QUANTITATIVE RESULT: No-step `D_ONLY` backward produced nonzero trigger statistic response in the fixed-bank audit.

```text
DEPTH_GRAD_CAN_REACH_TRIGGER_STATISTIC = TRUE
```

## 7. Historical Trigger-State Availability

EXPERIMENTAL FACT:

- `HISTORICAL_TRIGGER_STATE_AVAILABLE = FALSE`.
- `HISTORICAL_VISIBILITY_ACCUMULATOR_AVAILABLE = FALSE`.
- `HISTORICAL_EXACT_ELIGIBILITY_AVAILABLE = FALSE`.
- `HISTORICAL_SPLIT_COUNTS_AVAILABLE = TRUE` for BND-K1 log rows.
- `HISTORICAL_PRUNE_COUNTS_AVAILABLE = TRUE` for BND-K1 log rows.
- Matching CDEPTH event logs were not found under `logs/`.

INFERENCE: The exact historical candidate sets at 1k/3k/5k cannot be recovered from checkpoints. The main outputs in this audit are therefore named `FIXED_BANK_TRIGGER_RESPONSE`, not historical densification eligibility.

## 8. Fixed-State Intervention Design

CONFIG FACT:

- Scene: `Panama`.
- Runs: `BND-K1`, `CDEPTH`.
- Primary steps: `1000`, `3000`, `5000`.
- Post-recovery control step: `8000`.
- SH degree remains `3`.
- Rasterization mode remains `classic`.
- Intrinsic color parameterization remains `bounded_sh3`.

FIXED_TRIGGER_CAMERA_BANK:

- Count: `15`.
- Selection rule: all Panama training views in dataset order.
- Views: `MTN_1538`, `MTN_1541`, `MTN_1540`, `MTN_1534`, `MTN_1535`, `MTN_1536`, `MTN_1533`, `MTN_1542`, `MTN_1537`, `MTN_1532`, `MTN_1546`, `MTN_1543`, `MTN_1544`, `MTN_1545`, `MTN_1548`.
- Pseudo-depth source: `batch["depth_image"]` loaded by `DepthDataset` from `depthAnything_u16`, normalized per image inside the loss.

## 9. Safety / Equivalence

EXPERIMENTAL FACT:

- No training was run.
- No optimizer or scheduler step was called.
- No split / duplicate / prune / opacity reset mutation was called.
- Parameter safety rows: `64`.
- Maximum parameter absolute delta after all fixed-state backward audits: `0.0`.

```text
AUDIT_PARAMETER_SAFETY = PASS
```

## 10. K1@1k Trigger Response

QUANTITATIVE RESULT:

| Condition | Eligible | Split | Duplicate | Score Mean | Score p50 | Score p90 | Score p99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R | 49,785 | 47,890 | 1,895 | 0.003292742 | 0.002251960 | 0.007257262 | 0.016521133 |
| RD | 49,845 | 47,948 | 1,897 | 0.003303787 | 0.002257684 | 0.007271227 | 0.016563859 |

Eligibility flips:

- `N_visible = 60,111`.
- `N_depth_added = 74`.
- `N_depth_removed = 14`.
- `ELIGIBLE_COUNT_RATIO = 1.001205`.
- `ADDED_RATE = 0.001231`.
- `near_threshold_median_abs_shift / theta = 0.001334`.
- Added type: `72` split, `2` duplicate.

## 11. K1@3k Trigger Response

QUANTITATIVE RESULT:

| Condition | Eligible | Split | Duplicate | Score Mean | Score p50 | Score p90 | Score p99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R | 165,233 | 84,175 | 81,058 | 0.000686013 | 0.000442006 | 0.001565430 | 0.003775771 |
| RD | 166,369 | 84,880 | 81,489 | 0.000689732 | 0.000444735 | 0.001573724 | 0.003790435 |

Eligibility flips:

- `N_visible = 566,253`.
- `N_depth_added = 1,324`.
- `N_depth_removed = 188`.
- `ELIGIBLE_COUNT_RATIO = 1.006875`.
- `ADDED_RATE = 0.002338`.
- `near_threshold_median_abs_shift / theta = 0.000908`.
- Added type: `797` split, `527` duplicate.

## 12. K1@5k Trigger Response

QUANTITATIVE RESULT:

| Condition | Eligible | Split | Duplicate | Score Mean | Score p50 | Score p90 | Score p99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R | 184,370 | 80,902 | 103,468 | 0.000567229 | 0.000370672 | 0.001154723 | 0.003918932 |
| RD | 184,679 | 81,147 | 103,532 | 0.000567749 | 0.000371017 | 0.001155701 | 0.003924451 |

Eligibility flips:

- `N_visible = 998,452`.
- `N_depth_added = 414`.
- `N_depth_removed = 105`.
- `ELIGIBLE_COUNT_RATIO = 1.001676`.
- `ADDED_RATE = 0.000415`.
- `near_threshold_median_abs_shift / theta = 0.000037`.
- Added type: `320` split, `94` duplicate.

## 13. K1@8k Post-Recovery Control

QUANTITATIVE RESULT:

| Condition | Eligible | Split | Duplicate | Score Mean | Score p50 | Score p90 | Score p99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R | 168,809 | 114,766 | 54,043 | 0.000481437 | 0.000373961 | 0.000920208 | 0.002601114 |
| RD | 169,156 | 115,077 | 54,079 | 0.000481886 | 0.000374521 | 0.000920957 | 0.002601919 |

Eligibility flips:

- `N_visible = 1,223,067`.
- `N_depth_added = 481`.
- `N_depth_removed = 134`.
- `ELIGIBLE_COUNT_RATIO = 1.002056`.
- `ADDED_RATE = 0.000393`.
- `near_threshold_median_abs_shift / theta = 0.000025`.
- Added type: `418` split, `63` duplicate.

## 14. CDEPTH-State Robustness

QUANTITATIVE RESULT:

| Step | Eligible R | Eligible RD | Depth Added | Depth Removed | Eligible Ratio | Added Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 37,306 | 37,304 | 2 | 4 | 0.999946 | 0.000046 |
| 3000 | 132,237 | 132,241 | 6 | 2 | 1.000030 | 0.000013 |
| 5000 | 186,356 | 186,356 | 2 | 2 | 1.000000 | 0.000002 |
| 8000 | 173,058 | 173,055 | 1 | 4 | 0.999983 | 0.000001 |

INFERENCE: In already-trained CDEPTH states, adding the same depth term at fixed state changes eligibility only negligibly.

## 15. Candidate Eligibility Flips

QUANTITATIVE RESULT:

| Step | `N_depth_added / N_eligible_R` | `|eligible ratio - 1|` | Nontrivial trigger shift gate |
| --- | ---: | ---: | --- |
| 1000 | 0.001486 | 0.001205 | false |
| 3000 | 0.008013 | 0.006875 | false |
| 5000 | 0.002246 | 0.001676 | false |
| 8000 | 0.002850 | 0.002056 | post-recovery control |

The pre-registered nontrivial trigger-shift criteria were not met at 1k or 3k.

## 16. Near-Threshold Response

QUANTITATIVE RESULT:

Near-threshold definition: `0.8 <= score_R / theta <= 1.2`, where `theta = 0.0008`.

| Step | Near-threshold median abs shift / theta |
| --- | ---: |
| 1000 | 0.001334 |
| 3000 | 0.000908 |
| 5000 | 0.000037 |
| 8000 | 0.000025 |

INFERENCE: The depth term reaches the trigger statistic, but its fixed-state margin shift is far below the pre-registered `5% of threshold` nontrivial criterion.

## 17. Split / Duplicate Candidate Type

QUANTITATIVE RESULT:

| Step | Depth-Added Split | Depth-Added Duplicate | Depth-Removed Split | Depth-Removed Duplicate |
| --- | ---: | ---: | ---: | ---: |
| 1000 | 72 | 2 | 14 | 0 |
| 3000 | 797 | 527 | 92 | 96 |
| 5000 | 320 | 94 | 75 | 30 |
| 8000 | 418 | 63 | 107 | 27 |

Candidate type label:

```text
LARGE_SCALE_SPLIT_BIASED
```

## 18. Candidate Physical Attributes

QUANTITATIVE RESULT: Selected K1 fixed-state candidate attributes.

| Step | Group | Scale p50 | Scale p90 | Anisotropy p50 | Opacity p50 | Visibility p50 | Radius-fraction p50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | DEPTH_ADDED | 0.006555 | 0.027289 | 3.142459 | 0.983064 | 8 | 0.013393 |
| 1000 | UNCHANGED_ELIGIBLE | 0.005144 | 0.015831 | 4.872568 | 0.979547 | 10 | 0.011161 |
| 3000 | DEPTH_ADDED | 0.001411 | 0.006076 | 11.283659 | 0.922300 | 7 | 0.003344 |
| 3000 | UNCHANGED_ELIGIBLE | 0.001024 | 0.007805 | 19.490410 | 0.934142 | 8 | 0.003344 |
| 5000 | DEPTH_ADDED | 0.002360 | 0.047306 | 19.782286 | 0.912679 | 3 | 0.005574 |
| 5000 | UNCHANGED_ELIGIBLE | 0.000947 | 0.002893 | 31.318148 | 0.909810 | 8 | 0.003344 |

INFERENCE: Candidate type/attribute comparisons are descriptive fixed-state properties, not evidence that these candidates historically split.

## 19. Spatial Projection to Future HJ_GAIN/HARM

CODE FACT: Spatial masks are post-hoc outcome masks from final 15k M1/K1/CDEPTH eval views:

- `M1_HIGH_J`: final M1 accumulation `> 0.01` and final M1 `clear_object_fullsh_raw` max channel `> 1.0`.
- `HJ_GAIN`: `M1_HIGH_J` and final RGB MSE(K1) - RGB MSE(CDEPTH) `> 0`.
- `HJ_HARM`: `M1_HIGH_J` and final RGB MSE(K1) - RGB MSE(CDEPTH) `< 0`.

These masks are not online-observable training signals.

QUANTITATIVE RESULT: DEPTH_ADDED projected-center enrichment.

| Step | View | HJ_GAIN enrich | HJ_HARM enrich | Gain/Harm ratio | Gain > Harm |
| --- | --- | ---: | ---: | ---: | --- |
| 1000 | MTN_1529 | 0.288020 | 0.309994 | 0.929115 | false |
| 1000 | MTN_1539 | 0.000000 | 0.422417 | 0.000000 | false |
| 1000 | MTN_1547 | 0.000000 | 0.627773 | 0.000000 | false |
| 3000 | MTN_1529 | 0.599613 | 0.441674 | 1.357594 | true |
| 3000 | MTN_1539 | 0.514120 | 0.405013 | 1.269390 | true |
| 3000 | MTN_1547 | 0.127449 | 0.442755 | 0.287854 | false |
| 5000 | MTN_1529 | 0.195034 | 1.150295 | 0.169552 | false |
| 5000 | MTN_1539 | 0.851029 | 0.241486 | 3.524135 | true |
| 5000 | MTN_1547 | 0.000000 | 0.518868 | 0.000000 | false |
| 8000 | MTN_1529 | 0.540428 | 0.000000 | 54042799.142242 | true |
| 8000 | MTN_1539 | 0.860796 | 0.162292 | 5.303985 | true |
| 8000 | MTN_1547 | 0.000000 | 0.000000 | 0.000000 | false |

Support-proxy disk overlap was not computed; this run reports projected-center proxy only.

## 20. Cross-View Consistency

QUANTITATIVE RESULT:

- At 1k, `0/3` views have DEPTH_ADDED HJ_GAIN enrichment greater than HJ_HARM enrichment.
- At 3k, `2/3` views have DEPTH_ADDED HJ_GAIN enrichment greater than HJ_HARM enrichment.
- Pooled classification flag:

```text
SPATIAL_TRIGGER_CROSS_VIEW_CONSISTENT = TRUE
```

## 21. Temporal Ordering

QUANTITATIVE RESULT:

- 1k pre-recovery trigger shift: weak by gate.
- 3k pre-recovery trigger shift: weak by gate.
- 5k high-J recovery onset: prior trajectory reference.
- 8k global recovery onset: prior trajectory reference and post-recovery control.

```text
PRE_RECOVERY_TRIGGER_REDISTRIBUTION = FALSE
```

INFERENCE: Although 3k center enrichment direction is partially aligned in 2/3 eval views, the actual eligibility / score shift is too small to pass the pre-registered trigger redistribution gate.

## 22. Historical Corroboration

EXPERIMENTAL FACT:

```text
HISTORICAL_TRIGGER_CORROBORATION = NOT_AVAILABLE
```

Reason: exact historical accumulators are absent from checkpoints, and matching CDEPTH split/prune event logs were not found.

## 23. Pruning Context

CODE FACT: In a same-state R vs RD comparison, current loss gradients cannot directly affect pruning eligibility because pruning reads opacity, 3D scale, optional screen-size, and optional extra split mask.

EXPERIMENTAL FACT: Pruning context output was written to:

`outputs/bnd_cdepth_densify_trigger_panama_20260811/pruning_context.json`

## 24. Mechanism Classification

QUANTITATIVE CONCLUSION:

```text
DEPTH_GRAD_CAN_REACH_TRIGGER_STATISTIC = TRUE
PRE_RECOVERY_TRIGGER_REDISTRIBUTION = FALSE
SPATIAL_TRIGGER_CROSS_VIEW_CONSISTENT = TRUE
Mechanism Classification = HJ_ALIGNMENT_WITH_WEAK_TRIGGER_RESPONSE
Candidate Type = LARGE_SCALE_SPLIT_BIASED
```

INFERENCE: The depth loss can reach the true WaterSplatting trigger statistic, but the measured fixed-state eligibility effect at 1k/3k is below the pre-registered nontrivial threshold. Spatial alignment exists in part of the pre-recovery window, but the weak trigger magnitude prevents classifying this as `DENSIFICATION_TRIGGER_SUPPORTED`.

## 25. Scientific Interpretation

Main answers:

1. Coarse-depth loss directly changes the exact WaterSplatting trigger statistic: `TRUE`.
2. It produces new eligible candidates: `TRUE`, but the rates are small (`0.001231` at 1k and `0.002338` at 3k).
3. DEPTH_ADDED candidates are not consistently enriched toward future HJ_GAIN at 1k; at 3k they align in `2/3` views, but the trigger shift remains weak.
4. Current evidence does not support early densification-trigger redistribution as the main CDEPTH partial-mitigation mechanism.
5. The spatial association is post-hoc and remains association-only.

## 26. Next Single-Factor Recommendation

Recommended next experiment:

```text
DIRECT-OBJECT CONTINUOUS OPTIMIZATION PATH AUDIT
```

Do not prioritize BND-DTRIG from the current evidence, because `PRE_RECOVERY_TRIGGER_REDISTRIBUTION = FALSE`.

## 27. Outputs

Primary outputs:

- Source audit: `outputs/bnd_cdepth_densify_trigger_panama_20260811/densification_source_audit.json`.
- Historical availability: `outputs/bnd_cdepth_densify_trigger_panama_20260811/historical_state_availability.json`.
- Camera bank: `outputs/bnd_cdepth_densify_trigger_panama_20260811/fixed_trigger_camera_bank.json`.
- Trigger summary: `outputs/bnd_cdepth_densify_trigger_panama_20260811/densify_trigger_final_summary.json`.
- Trigger all rows: `outputs/bnd_cdepth_densify_trigger_panama_20260811/trigger_response_all.csv`.
- Candidate flips: `outputs/bnd_cdepth_densify_trigger_panama_20260811/candidate_flip_counts.csv`.
- Candidate types: `outputs/bnd_cdepth_densify_trigger_panama_20260811/candidate_type_metrics.csv`.
- Candidate attributes: `outputs/bnd_cdepth_densify_trigger_panama_20260811/candidate_attribute_metrics.csv`.
- Spatial enrichment: `outputs/bnd_cdepth_densify_trigger_panama_20260811/candidate_spatial_enrichment.csv`.
- Cross-view metrics: `outputs/bnd_cdepth_densify_trigger_panama_20260811/candidate_cross_view_metrics.csv`.
- Mechanism classification: `outputs/bnd_cdepth_densify_trigger_panama_20260811/mechanism_classification.json`.
- Output manifest: `outputs/bnd_cdepth_densify_trigger_panama_20260811/manifest.json`.

Visual assets:

- Trigger score distributions: `renders/bnd_cdepth_densify_trigger_panama_20260811/plot_trigger_score_distributions.png`.
- Eligibility counts: `renders/bnd_cdepth_densify_trigger_panama_20260811/plot_eligibility_counts.png`.
- Candidate type counts: `renders/bnd_cdepth_densify_trigger_panama_20260811/plot_candidate_type_counts.png`.
- Candidate projected maps: `renders/bnd_cdepth_densify_trigger_panama_20260811/contact_sheet_projected_candidate_maps.png`.
- Strong gain/harm overlay: `renders/bnd_cdepth_densify_trigger_panama_20260811/contact_sheet_strong_gain_overlay.png`.
- Temporal summary: `renders/bnd_cdepth_densify_trigger_panama_20260811/contact_sheet_temporal_summary.png`.
- Compact causal-chain sheet: `renders/bnd_cdepth_densify_trigger_panama_20260811/contact_sheet_compact_causal_chain.png`.
- Visual manifest: `renders/bnd_cdepth_densify_trigger_panama_20260811/manifest.json`.
- Visual index: `renders/bnd_cdepth_densify_trigger_panama_20260811/VISUAL_COMPARE_INDEX.md`.

Visual assets are ready for external/manual analysis.
No subjective clear-image correctness judgment was made.
