# BND Medium Context Utilization Audit

Date: 2026-08-12

## Scope

CODE FACT: This audit is read-only. It loads existing Panama M1 and BND-K1 checkpoints, queries/render-checks medium context counterfactuals, writes CSV/JSON/PNG diagnostics, and does not call optimizer steps, scheduler steps, scaler updates, densification, pruning, opacity reset, or checkpoint writes.

CONFIG FACT: Repository branch was `research/m1-bounded-intrinsic` at HEAD `135d85c74e91ed29459aac28336047e386972a66` when the audit was run.

CONFIG FACT: Scene is `Panama`; formal M1/BND medium configuration is `medium_context_mode=dir_xy_camera`, `b_inf_mode=tied`, and `infinite_water_enabled=False`.

CONFIG FACT: The GPU policy for this task allows only physical GPUs `6,7,8,9`; the generated run used `CUDA_VISIBLE_DEVICES=6`.

## Source Semantics

CODE FACT: The medium MLP input has total dimension `22`.

CODE FACT: The input layout is:

- Direction encoding: dimension `16`.
- XY context: dimension `3`, containing normalized image coordinates plus radial image term.
- Camera context: dimension `3`, containing scene-box normalized camera center.

CODE FACT: The medium MLP output layout is:

- `0:3`: `B_inf` / `medium_rgb`, activated with sigmoid.
- `3:6`: `beta_B` / `medium_bs`, activated with softplus.
- `6:9`: `beta_D` / `medium_attn`, activated with softplus.

CODE FACT: `B_inf` is tied to `medium_rgb` under `b_inf_mode=tied`.

## Baseline Medium Statistics

EXPERIMENTAL FACT: Medium output statistics were computed using deterministic pixel samples from all Panama train/held-out views, with `max_samples_per_view=4096`.

QUANTITATIVE RESULT: BND-K1 sampled mean medium outputs:

| Split | B_inf mean | beta_B mean | beta_D mean |
| --- | ---: | ---: | ---: |
| train | 0.201598 | 0.097159 | 0.151396 |
| heldout | 0.190310 | 0.084339 | 0.142964 |

## Jacobian Sensitivity

EXPERIMENTAL FACT: Local Jacobian sensitivity is reported as raw Frobenius norm multiplied by the training-development RMS standard deviation for the input group.

QUANTITATIVE RESULT: BND-K1 scale-normalized `beta_D` sensitivity:

| Split | direction | XY | camera |
| --- | ---: | ---: | ---: |
| train | 0.075148 | 0.073604 | 0.026847 |
| heldout | 0.067965 | 0.069235 | 0.024739 |

INFERENCE: Under this diagnostic, BND-K1 uses direction and XY context more strongly than camera context for `beta_D` local sensitivity.

## Camera And XY Swap Response

EXPERIMENTAL FACT: Camera swap and XY swap responses were measured over 18 Panama views.

QUANTITATIVE RESULT: BND-K1 camera-swap mean absolute deltas:

| Output | Mean abs delta |
| --- | ---: |
| B_inf | 0.004884 |
| beta_B | 0.006190 |
| beta_D | 0.007579 |

QUANTITATIVE RESULT: BND-K1 XY-swap mean absolute deltas:

| Output | Mean abs delta |
| --- | ---: |
| B_inf | 0.014967 |
| beta_B | 0.012587 |
| beta_D | 0.024233 |

QUANTITATIVE RESULT: For `beta_D`, XY-swap response is larger than camera-swap response in BND-K1.

## Matched-LOS Analysis

EXPERIMENTAL FACT: Matched line-of-sight pairs used a locked angular threshold of `1.0` degree because the 1-degree candidate count was `24019`, exceeding the pre-registered threshold requirement.

QUANTITATIVE RESULT: BND-K1 matched-LOS pair counts:

| Pair type | Count |
| --- | ---: |
| cross-camera | 20000 |
| within-camera | 23 |

QUANTITATIVE RESULT: BND-K1 matched-LOS mean absolute deltas:

| Output | Cross-camera delta | Within-camera delta | Cross/within ratio |
| --- | ---: | ---: | ---: |
| B_inf | 0.009500 | 0.001866 | 5.089664 |
| beta_B | 0.016009 | 0.001547 | 10.349840 |
| beta_D | 0.022420 | 0.003134 | 7.153684 |

INFERENCE: Similar LOS rays across different cameras can yield different predicted medium parameters, and this difference is larger than the within-camera direction-control baseline for `beta_D`.

## Hard-Region Enrichment

EXPERIMENTAL FACT: Hard-region enrichment was computed by comparing region-specific context response against `OBJECT_SUPPORT` as the independent reference; corrected computation uses separate deterministic region samples rather than sampling only from the hard core for both numerator and denominator.

QUANTITATIVE RESULT: BND_HARD_CORE enrichment versus object support:

| Counterfactual | Output | Enrichment | Views >= 1.5 |
| --- | --- | ---: | ---: |
| camera swap | beta_D | 0.856358 | 0 |
| XY swap | beta_D | 0.869697 | 0 |
| extra context fixed | beta_D | 1.033296 | 0 |

QUANTITATIVE RESULT: Context-error correlations:

| Pairing | Spearman |
| --- | ---: |
| camera-swap beta_D vs RGB MSE | 0.007742 |
| XY-swap beta_D vs RGB MSE | -0.114742 |
| extra-context-fixed beta_D vs RGB MSE | -0.041168 |
| camera-swap beta_D vs hard-core label | -0.051816 |
| XY-swap beta_D vs hard-core label | -0.035866 |
| extra-context-fixed beta_D vs hard-core label | 0.013931 |

QUANTITATIVE RESULT: The hard-region association gate did not pass.

## M1 vs BND Context Response

EXPERIMENTAL FACT: M1 and BND-K1 were audited with the same Panama view set and context perturbation definitions.

QUANTITATIVE RESULT: BND/M1 response ratios:

| Counterfactual | Output | BND/M1 ratio |
| --- | --- | ---: |
| camera swap | beta_D | 0.183694 |
| XY swap | beta_D | 0.547753 |
| matched-LOS cross-camera | beta_D | 0.385016 |
| extra context fixed | beta_D | 0.495578 |
| camera swap | B_inf | 0.796535 |
| XY swap | B_inf | 1.821170 |

INFERENCE: BND-K1 does not show stronger extra-context dependence than M1 for `beta_D` under these probes; its `beta_D` response ratios are all below `1.0`.

## Counterfactual RGB Response

EXPERIMENTAL FACT: The full medium-query forward check reported zero MSE for `B_inf`, `beta_B`, and `beta_D`, confirming the counterfactual query path reproduces the normal medium outputs before perturbation.

QUANTITATIVE RESULT: Held-out RGB MSE means:

| Render mode | RGB MSE mean |
| --- | ---: |
| FULL | 0.000714 |
| CAM_CONTEXT_FIXED_CF | 0.000783 |
| EXTRA_CONTEXT_FIXED_CF | 0.001135 |

INFERENCE: This table records technical counterfactual response only; it is not a subjective image-quality judgment.

## Held-Out Validation

EXPERIMENTAL FACT: Held-out audit used Panama held-out views `MTN_1529`, `MTN_1539`, and `MTN_1547`.

QUANTITATIVE RESULT: Held-out BND_HARD_CORE views with enrichment >= `1.5`:

| Counterfactual | Output | Count |
| --- | --- | ---: |
| camera swap | beta_D | 0 |
| XY swap | beta_D | 0 |

## Classification

QUANTITATIVE RESULT: Final MEDCTX classification is `EXTRA_CONTEXT_USED_WITHOUT_HARD_REGION_ASSOCIATION`.

QUANTITATIVE RESULT: Gate values:

- `extra_context_used=True`.
- `camera_vs_xy_stronger=xy`.
- `matched_los_variation=True`.
- `hard_region_association=False`.
- `bound_compatible_escalation=False`.
- `supports_context_reduced_m1_training=False`.
- `close_medctx_concern=True`.

INFERENCE: The diagnostic supports closing the specific MEDCTX concern for the current BND branch: extra context is used, but the corrected hard-region and BND-vs-M1 gates do not support it as the dominant bounded-hard-region compensation mechanism.

## Main Scientific Conclusion

QUANTITATIVE CONCLUSION: Panama BND-K1 uses extra XY/camera context, and matched-LOS rays across cameras show measurable medium-output variation.

QUANTITATIVE CONCLUSION: The strongest BND-K1 `beta_D` context response is XY rather than camera context.

QUANTITATIVE CONCLUSION: Corrected region enrichment and context-error correlation do not associate the extra-context response with bounded hard regions.

QUANTITATIVE CONCLUSION: BND-K1 is not more context-dependent than M1 for `beta_D` under the measured camera, XY, matched-LOS, or extra-context-fixed probes.

INFERENCE: The current evidence does not support launching context-reduced M1 training as the next controlled experiment.

## Next Decision

INFERENCE: Close the MEDCTX concern under this audit and proceed to new-mechanism development rather than context-reduction training.

HYPOTHESIS: Candidate new mechanisms remain:

- OceanSplat-style object multi-view consistency focused on bounded intrinsic/direct-object consistency.
- SeaFree CB-Loss split into foreground inverse-intensity weighting and background-water anchoring.
- Cross-scene BG-anchor readiness audit, especially Curasao and IUI3.

HYPOTHESIS: These are future controlled experiments and were not trained or evaluated in this MEDCTX audit.

## Output Paths

EXPERIMENTAL FACT: Output directory: `outputs/bnd_medctx_panama_20260812`.

EXPERIMENTAL FACT: Render directory: `renders/bnd_medctx_panama_20260812`.

EXPERIMENTAL FACT: Visual index: `renders/bnd_medctx_panama_20260812/VISUAL_COMPARE_INDEX.md`.

EXPERIMENTAL FACT: Main output files:

- `outputs/bnd_medctx_panama_20260812/repo_manifest.json`
- `outputs/bnd_medctx_panama_20260812/medium_context_source_audit.json`
- `outputs/bnd_medctx_panama_20260812/medium_context_source_audit.md`
- `outputs/bnd_medctx_panama_20260812/medium_input_distribution.csv`
- `outputs/bnd_medctx_panama_20260812/medium_output_baseline.csv`
- `outputs/bnd_medctx_panama_20260812/jacobian_sensitivity_summary.json`
- `outputs/bnd_medctx_panama_20260812/matched_los_summary.json`
- `outputs/bnd_medctx_panama_20260812/camera_swap_response.json`
- `outputs/bnd_medctx_panama_20260812/xy_swap_response.json`
- `outputs/bnd_medctx_panama_20260812/counterfactual_rgb_metrics.json`
- `outputs/bnd_medctx_panama_20260812/m1_vs_bnd_context_response.json`
- `outputs/bnd_medctx_panama_20260812/medctx_classification.json`
- `outputs/bnd_medctx_panama_20260812/bnd_medctx_final_summary.json`
- `outputs/bnd_medctx_panama_20260812/manifest.json`

EXPERIMENTAL FACT: Visual files:

- `renders/bnd_medctx_panama_20260812/contact_sheet_baseline_medium_components.png`
- `renders/bnd_medctx_panama_20260812/contact_sheet_context_sensitivity_maps.png`
- `renders/bnd_medctx_panama_20260812/contact_sheet_counterfactual_rgb_response.png`
- `renders/bnd_medctx_panama_20260812/contact_sheet_hard_region_overlays.png`
- `renders/bnd_medctx_panama_20260812/contact_sheet_sensitivity_vs_hard_region.png`
- `renders/bnd_medctx_panama_20260812/heldout_summary_medctx.png`
- `renders/bnd_medctx_panama_20260812/plot_bnd_vs_m1_context_response.png`
- `renders/bnd_medctx_panama_20260812/plot_matched_los_cross_camera_variation.png`
- `renders/bnd_medctx_panama_20260812/plot_per_view_camera_sensitivity.png`
- `renders/bnd_medctx_panama_20260812/plot_per_view_xy_sensitivity.png`
- `renders/bnd_medctx_panama_20260812/scorecard_final_medctx.png`

## Validation

EXPERIMENTAL FACT: The generated PNG files were opened with PIL and had nonzero dimensions.

EXPERIMENTAL FACT: The tracked large-output check returned `0`; `outputs/`, `renders/`, `logs/`, `common_masks/`, and `checkpoints/` files are not tracked by Git.

EXPERIMENTAL FACT: No subjective visual-quality conclusion was made in this audit.
