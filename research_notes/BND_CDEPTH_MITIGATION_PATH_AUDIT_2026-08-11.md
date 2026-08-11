# BND-CDEPTH Partial-Mitigation Optimization-Path Audit

## Motivation

**Code Fact**

- This stage is a read-only model analysis of existing Panama BND-K1 and BND-CDEPTH checkpoints.
- No new training, optimizer step, scheduler step, densification, pruning, parameter update, or checkpoint modification was performed.
- The target comparison is BND-K1 versus BND-CDEPTH on the fixed Panama eval views.

**Experimental Fact**

- BND-CDEPTH was previously measured as a partial bounded-intrinsic mitigation mechanism: PSNR improved by `+0.254946 dB` over BND-K1 at final evaluation, while SSIM decreased by `-0.002492` and LPIPS increased by `+0.005410`.
- Fixed M1_HIGH_J local MSE gap recovery was `0.383953`.
- Brightness-Q5 gap recovery was `0.358741`.

**Inference**

- The goal is to explain which optimization-path changes are associated with the RGB/high-J recovery, and whether the SSIM/LPIPS cost has a separable spatial or physical proxy.

## Revised Interpretation Of SeaFree

**Experimental Fact**

- Panama M1: PSNR `32.308910`, SSIM `0.949487`, LPIPS `0.073979`.
- Panama BND-K1: PSNR `31.498353`, SSIM `0.948783`, LPIPS `0.075521`.
- Panama SeaFree context: PSNR `31.725087`, SSIM `0.944203`, LPIPS `0.088957`.

**Inference**

- SeaFree-GS is not treated as a complete solution to the bounded reconstruction trade-off.
- SeaFree-style terms are treated as sources of partial mitigation ideas.
- CDEPTH is treated as one experimentally validated partial mitigation mechanism, not as the final answer.

## Existing Negative Control

**Experimental Fact**

Fixed M1_HIGH_J pseudo-depth diagnostic:

| Run | Spearman | Pearson | Aligned RMSE | Depth-gradient Pearson |
| --- | ---: | ---: | ---: | ---: |
| BND-K1 | 0.985837 | 0.985610 | 0.040947 | 0.261865 |
| CDEPTH | 0.985512 | 0.984041 | 0.043154 | 0.261786 |
| SeaFree context | n/a | n/a | 0.033772 | 0.401077 |

**Quantitative Conclusion**

- `GEOMETRY_TARGET_IMPROVED = FALSE` for CDEPTH versus BND-K1 on the fixed M1_HIGH_J diagnostic.
- The simple explanation "CDEPTH improves pseudo-depth accuracy, therefore RGB improves" is not supported by the current diagnostic.

## Gradient-Routing Clue

**Experimental Fact**

Fixed K1 state no-step gradient audit, depth-gradient norm divided by RGB-gradient norm:

| Parameter group | Ratio |
| --- | ---: |
| means | 0.016094 |
| scales | 0.471700 |
| quats | 0.421534 |
| opacities | 0.505668 |
| features_dc | 0 |
| features_rest | 0 |
| medium | 0 |

**Inference**

- CDEPTH is more plausibly routed through Gaussian scale, orientation, opacity, coverage, and population trajectory than through direct large movement of Gaussian centers.

## Recovery Status

**Code Fact**

- Start branch: `research/m1-bounded-intrinsic`.
- Start HEAD: `d17d7539099c2bc87137081f079ccf0d5281964c`.
- Start status contained untracked CDEPTH audit files plus two historical untracked GMVC scripts:
  - `scripts/diagnostics/audit_bnd_cdepth_optimization_path.py`
  - `research_notes/BND_CDEPTH_MITIGATION_PATH_AUDIT_2026-08-11.md`
  - `scripts/diagnostics/render_gmvc_curasao_contact_sheet.py`
  - `scripts/diagnostics/summarize_gmvc_four_scene_visual_audit.py`
- Historical GMVC scripts were not modified, deleted, or staged.
- Stale CDEPTH process check: `NO_STALE_CDEPTH_PROCESS = true`.

**Experimental Fact**

- Previous partial directories were kept but not used as the formal output:
  - `outputs/bnd_cdepth_path_panama_20260811/`
  - `renders/bnd_cdepth_path_panama_20260811/`
- Formal recomputed outputs were written to:
  - `outputs/bnd_cdepth_mitigation_path_panama_20260811/`
  - `renders/bnd_cdepth_mitigation_path_panama_20260811/`
  - `logs/bnd_cdepth_mitigation_path_panama_20260811/`

## Trajectory Availability

**Experimental Fact**

| Run | Target steps | Actual final step |
| --- | --- | ---: |
| BND-K1 | 1000, 3000, 5000, 8000, 10000, 13000, 15000 | 14999 |
| CDEPTH | 1000, 3000, 5000, 8000, 10000, 13000, 15000 | 14999 |

Common trajectory steps: `[1000, 3000, 5000, 8000, 10000, 13000, 15000]`.

Eval views: `MTN_1539`, `MTN_1529`, `MTN_1547`.

## RGB And High-J Mitigation Trajectory

**Quantitative Result**

| Step | K1 PSNR | CDEPTH PSNR | K1 MSE | CDEPTH MSE | Global MSE Gain | K1 High-J MSE | CDEPTH High-J MSE | High-J MSE Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 22.563314 | 22.592255 | 0.005637601 | 0.005658584 | -0.000020983 | 0.047622857 | 0.046762552 | 0.000860305 |
| 3000 | 20.869651 | 20.877951 | 0.008590163 | 0.008585184 | 0.000004979 | 0.068755445 | 0.071095886 | -0.002340441 |
| 5000 | 26.459814 | 26.427294 | 0.002286107 | 0.002305323 | -0.000019216 | 0.018601283 | 0.017951227 | 0.000650056 |
| 8000 | 31.008972 | 31.219704 | 0.000801345 | 0.000761458 | 0.000039886 | 0.006075121 | 0.005119010 | 0.000956111 |
| 10000 | 31.154767 | 31.326232 | 0.000776138 | 0.000745957 | 0.000030181 | 0.005741542 | 0.004979898 | 0.000761644 |
| 13000 | 31.540089 | 31.779425 | 0.000712338 | 0.000672253 | 0.000040085 | 0.005109404 | 0.004283457 | 0.000825947 |
| 15000 | 31.498353 | 31.753299 | 0.000714139 | 0.000672041 | 0.000042098 | 0.004818396 | 0.004015234 | 0.000803162 |

- `HIGHJ_RECOVERY_ONSET_STEP = 5000`.
- `GLOBAL_RECOVERY_ONSET_STEP = 8000`.

**Quantitative Conclusion**

- High-J recovery becomes stable earlier than global MSE recovery in this trajectory.

## Gaussian Population

**Quantitative Result**

| Step | K1 count | CDEPTH count | Delta | Ratio |
| ---: | ---: | ---: | ---: | ---: |
| 1000 | 60112 | 43198 | -16914 | 0.718625 |
| 3000 | 566332 | 453172 | -113160 | 0.800188 |
| 5000 | 998668 | 941413 | -57255 | 0.942669 |
| 8000 | 1223420 | 1213251 | -10169 | 0.991688 |
| 10000 | 1219898 | 1219880 | -18 | 0.999985 |
| 13000 | 1183679 | 1184186 | 507 | 1.000428 |
| 15000 | 1177886 | 1178513 | 627 | 1.000532 |

- `POPULATION_DIVERGENCE_ONSET = 1000`.

**Inference**

- Population trajectory diverges before high-J recovery and is consistent with a population/densification pathway, but this is association evidence only.

## Scale And Anisotropy

**Code Fact**

- Scale values are activated physical scales from checkpoints, not raw pre-activation parameters.
- Anisotropy is `scale_max / (scale_min + eps)`.

**Quantitative Result**

| Step | K1 scale max p50 | CDEPTH scale max p50 | K1 scale max p90 | CDEPTH scale max p90 | K1 scale max p99 | CDEPTH scale max p99 | K1 anisotropy p90 | CDEPTH anisotropy p90 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 0.004949613 | 0.004259551 | 0.015186284 | 0.012012898 | 0.044689540 | 0.031223658 | 16.242 | 15.631 |
| 3000 | 0.000761183 | 0.000741410 | 0.003817119 | 0.003172547 | 0.019126071 | 0.016894629 | 343.313 | 399.248 |
| 5000 | 0.001008444 | 0.000987147 | 0.004116749 | 0.003812137 | 0.020950012 | 0.027987199 | 835.326 | 1106.881 |
| 8000 | 0.001317858 | 0.001316073 | 0.004381600 | 0.004537065 | 0.021286670 | 0.035517562 | 1099.431 | 2050.171 |
| 10000 | 0.001341727 | 0.001357092 | 0.004575044 | 0.004895472 | 0.022760782 | 0.037069403 | 3030.150 | 6660.712 |
| 13000 | 0.001569663 | 0.001584750 | 0.005086875 | 0.005482017 | 0.025594912 | 0.042736176 | 4921.004 | 10062.038 |
| 15000 | 0.001640223 | 0.001656369 | 0.005324855 | 0.005749525 | 0.026764009 | 0.045080625 | 6139.862 | 12143.399 |

- `SCALE_DIVERGENCE_ONSET = 1000`.
- `ANISOTROPY_DIVERGENCE_ONSET = 3000`.

## Projected Footprint

**Code Fact**

- Projected radius is read from `model.radii` after projection.
- Semantics: `project_gaussians` screen-space radius; treated as projected-radius proxy in pixels.

**Quantitative Result**

| Step | K1 radius p50 | CDEPTH radius p50 | K1 radius p90 | CDEPTH radius p90 | K1 radius p99 | CDEPTH radius p99 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 11.667 | 10.667 | 33.333 | 28.667 | 89.000 | 78.333 |
| 3000 | 3.000 | 3.000 | 9.667 | 9.000 | 44.000 | 54.667 |
| 5000 | 4.333 | 4.333 | 11.000 | 11.000 | 49.333 | 66.333 |
| 8000 | 5.000 | 5.000 | 12.000 | 12.000 | 50.000 | 64.667 |
| 10000 | 5.000 | 5.000 | 12.000 | 12.667 | 53.000 | 65.000 |
| 13000 | 5.333 | 5.333 | 13.000 | 13.333 | 57.667 | 75.000 |
| 15000 | 5.333 | 5.333 | 13.333 | 14.333 | 59.667 | 79.000 |

- `FOOTPRINT_DIVERGENCE_ONSET = 1000`.

## Opacity And Alpha Coverage

**Code Fact**

- Opacity is physical opacity, `sigmoid(raw opacity)`.
- Alpha coverage is renderer `accumulation`.

**Quantitative Result**

| Step | K1 opacity p90 | CDEPTH opacity p90 | K1 opacity p99 | CDEPTH opacity p99 | K1 P(opacity>0.95) | CDEPTH P(opacity>0.95) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 0.999918 | 0.999893 | 0.999998 | 0.999998 | 0.561652 | 0.509561 |
| 3000 | 0.982574 | 0.984265 | 0.998788 | 0.999289 | 0.272621 | 0.277519 |
| 5000 | 0.979686 | 0.980535 | 0.996504 | 0.997650 | 0.257419 | 0.259155 |
| 8000 | 0.974362 | 0.974805 | 0.994763 | 0.995353 | 0.214422 | 0.215666 |
| 10000 | 0.970820 | 0.971581 | 0.993354 | 0.993965 | 0.194844 | 0.198131 |
| 13000 | 0.992721 | 0.993526 | 0.999201 | 0.999335 | 0.277581 | 0.289635 |
| 15000 | 0.996633 | 0.997151 | 0.999708 | 0.999766 | 0.304236 | 0.319382 |

Final alpha coverage:

| Run | Region | Alpha mean | Alpha p50 | Alpha p90 | P(alpha>0.99) |
| --- | --- | ---: | ---: | ---: | ---: |
| BND-K1 | WHOLE_IMAGE | 0.968664 | 0.999519 | 0.999889 | 0.804864 |
| CDEPTH | WHOLE_IMAGE | 0.998990 | 0.999839 | 0.999896 | 0.980176 |
| BND-K1 | M1_HIGH_J | 0.996633 | 0.999777 | 0.999893 | 0.945753 |
| CDEPTH | M1_HIGH_J | 0.999514 | 0.999853 | 0.999897 | 0.993585 |
| BND-K1 | M1_LOW_J | 0.967180 | 0.999487 | 0.999889 | 0.797426 |
| CDEPTH | M1_LOW_J | 0.998963 | 0.999838 | 0.999896 | 0.979473 |
| BND-K1 | BRIGHT_Q5 | 0.935641 | 0.999556 | 0.999890 | 0.755643 |
| CDEPTH | BRIGHT_Q5 | 0.998602 | 0.999828 | 0.999895 | 0.967517 |
| BND-K1 | EDGE_TOP20 | 0.994810 | 0.999725 | 0.999892 | 0.907745 |
| CDEPTH | EDGE_TOP20 | 0.999561 | 0.999854 | 0.999896 | 0.995727 |

- `OPACITY_DIVERGENCE_ONSET = None`.
- `ALPHA_DIVERGENCE_ONSET = 3000`.

## Gain And Harm Spatial Decomposition

**Code Fact**

- `GAIN(x) = E_K1(x) - E_CDEPTH(x)`, where `E` is mean RGB squared error to GT.
- `GAIN > 0` means lower RGB MSE for CDEPTH; `GAIN < 0` means higher RGB MSE for CDEPTH.

**Quantitative Result**

- `TOTAL_GAIN_MASS = 1636.515991`.
- `TOTAL_HARM_MASS = 1367.197510`.
- `P_HARM_PIXELS = 0.501412`.

| Region | Pixel fraction | Gain enrichment | Harm enrichment | P(gain|region) | P(harm|region) |
| --- | ---: | ---: | ---: | ---: | ---: |
| M1_HIGH_J | 0.050461 | 6.144235 | 3.294542 | 0.564237 | 0.435763 |
| M1_LOW_J | 0.949481 | 0.726656 | 0.878116 | 0.495069 | 0.504931 |
| BRIGHT_Q5 | 0.200000 | 2.769466 | 2.296404 | 0.468074 | 0.531926 |
| DARK_Q5 | 0.200009 | 0.581635 | 0.723610 | 0.525943 | 0.474057 |
| EDGE_TOP20 | 0.200000 | 2.448215 | 2.368171 | 0.506709 | 0.493291 |
| BOTTOM20 | 0.200337 | 2.376083 | 1.849110 | 0.529111 | 0.470889 |

## Extreme Low-T Redistribution

**Code Fact**

- Low-T masks use RGB-channel-mean transmission with threshold `T < 0.1`.
- `NEW_LOW_T = (T_CDEPTH < 0.1) and (T_K1 >= 0.1)`.

**Quantitative Result**

| Mask | Pixel fraction | P(harm|mask) | Harm enrichment | Mean gain | High-J overlap | Bright overlap | Edge overlap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| K1_LOW_T | 0.000000 | 0.000000 | 0.000000 | 0.000000000 | 0 | 0 | 0 |
| CDEPTH_LOW_T | 0.005707 | 0.236689 | 0.472046 | 0.000049645 | 542 | 18104 | 13 |
| NEW_LOW_T | 0.005707 | 0.236689 | 0.472046 | 0.000049645 | 542 | 18104 | 13 |
| PERSISTENT_LOW_T | 0.000000 | 0.000000 | 0.000000 | 0.000000000 | 0 | 0 | 0 |
| RESOLVED_LOW_T | 0.000000 | 0.000000 | 0.000000 | 0.000000000 | 0 | 0 | 0 |

- `NEW_LOWT_HARM_ENRICHMENT = 0.472046`.
- `LOWT_HARM_ALIGNED = false`.

**Quantitative Conclusion**

- CDEPTH creates a small new low-transmission tail, but this tail is not enriched for RGB harm under the defined harm-pixel criterion.

## Attenuation Distribution

**Quantitative Result**

Final whole-image attenuation:

| Run | tau p50 | tau p90 | tau p99 | T p01 | T p10 | T p50 | P(T<0.1) | P(T<0.05) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BND-K1 | 0.282424 | 1.004371 | 1.905748 | 0.150237 | 0.367282 | 0.755299 | 0.000000 | 0.000000 |
| CDEPTH | 0.217652 | 0.851835 | 2.075097 | 0.127518 | 0.427416 | 0.805734 | 0.005707 | 0.000506 |

Final M1_HIGH_J attenuation:

| Run | tau p50 | tau p90 | tau p99 | T p01 | T p10 | T p50 | P(T<0.1) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BND-K1 | 0.233851 | 0.600773 | 1.236567 | 0.292140 | 0.548984 | 0.792111 | 0.000000 |
| CDEPTH | 0.170250 | 0.465392 | 1.075847 | 0.342405 | 0.628580 | 0.844506 | 0.001679 |

**Quantitative Conclusion**

- CDEPTH shifts the bulk optical-depth distribution lower at p50/p90 and increases median transmission.
- CDEPTH also introduces a heavier extreme low-T tail, visible in p99 tau and P(T<0.1).
- This is a distributional bulk/tail redistribution, not evidence of physical correctness.

## Edge And Structure Proxy

**Code Fact**

- EDGE_TOP20 uses the top 20 percent of GT gradient magnitude.
- Pseudo-depth-gradient top-20 is a separate proxy.
- No per-pixel LPIPS attribution was generated.

**Quantitative Result**

| Region | Pixel fraction | P(harm|region) | Harm enrichment | Harm-mass enrichment | Aligned flag |
| --- | ---: | ---: | ---: | ---: | --- |
| EDGE_TOP20 | 0.200000 | 0.493291 | 0.983805 | 2.368171 | false |
| PSEUDO_DEPTH_GRAD_TOP20 | 0.199984 | 0.496143 | 0.989493 | 2.231193 | false |

**Quantitative Conclusion**

- The tested edge/structure proxies do not pass the predefined harm-alignment threshold.

## Direct And Medium Recomposition

**Code Fact**

- Direct/medium attribution uses true renderer outputs `direct_object_signal` and `rgb_medium`.
- It does not reconstruct direct as `J_image * T_image`.

**Quantitative Result**

| Region | Direct delta mean | Medium delta mean | Pred delta mean | GT error change mean |
| --- | ---: | ---: | ---: | ---: |
| GAIN_PIXELS | 0.009711 | 0.003747 | 0.007492 | -0.000513 |
| HARM_PIXELS | 0.010546 | 0.004859 | 0.007651 | 0.000426 |
| M1_HIGH_J | 0.016844 | 0.001832 | 0.016423 | -0.000868 |
| NEW_LOW_T | 0.014894 | 0.014228 | 0.003751 | -0.000034 |

**Inference**

- High-J recovery is more associated with direct-object recomposition than medium-only changes.
- NEW_LOW_T pixels show coupled direct and medium changes, but they are not enriched for RGB harm by the predefined threshold.

## Temporal Ordering

**Quantitative Result**

| Metric | Onset | Ordering vs high-J recovery |
| --- | ---: | --- |
| HIGHJ_RECOVERY_ONSET_STEP | 5000 | REFERENCE |
| GLOBAL_RECOVERY_ONSET_STEP | 8000 | FOLLOWS |
| POPULATION_DIVERGENCE_ONSET | 1000 | PRECEDES |
| SCALE_DIVERGENCE_ONSET | 1000 | PRECEDES |
| ANISOTROPY_DIVERGENCE_ONSET | 3000 | PRECEDES |
| FOOTPRINT_DIVERGENCE_ONSET | 1000 | PRECEDES |
| OPACITY_DIVERGENCE_ONSET | None | NO_CLEAR_DIVERGENCE |
| ALPHA_DIVERGENCE_ONSET | 3000 | PRECEDES |
| LOWT_DIVERGENCE_ONSET | 1000 | PRECEDES |

## Pathway Flags And Classification

**Quantitative Result**

| Flag | Value |
| --- | --- |
| `SCALE_FOOTPRINT_PATH_SUPPORTED` | true |
| `OPACITY_COVERAGE_PATH_SUPPORTED` | true |
| `POPULATION_DENSIFICATION_PATH_SUPPORTED` | true |
| `CONTRIBUTOR_DIAGNOSTIC_AVAILABLE` | false |
| `LOWT_HARM_ALIGNED` | false |
| `LOWT_HARM_PATH_SUPPORTED` | false |
| `EDGE_HARM_ALIGNED` | false |
| `STRUCTURE_HARM_PATH_SUPPORTED` | false |

**Quantitative Conclusion**

- Beneficial mechanism: `MIXED_GAUSSIAN_STRUCTURE`.
- Harmful mechanism: `NO_CLEAR_HARM_PATHWAY`.
- Pathway relation: `UNRESOLVED`.

## Main Scientific Conclusion

**Quantitative Conclusion**

- CDEPTH partially mitigates the bounded reconstruction trade-off at final Panama evaluation: `+0.254946 dB` PSNR and fixed M1_HIGH_J MSE improvement from `0.004818396` to `0.004015234`.
- The most supported beneficial pathway is not improved pseudo-depth accuracy; it is a mixed Gaussian-structure optimization path involving population trajectory, activated scale / projected footprint, anisotropy, and alpha coverage changes that precede or coincide high-J recovery.
- The SSIM/LPIPS degradation is a confirmed global metric cost, but this audit did not isolate a clear spatial harm pathway using NEW_LOW_T or edge/structure proxies.

**Inference**

- The evidence supports CDEPTH as a partial mitigation mechanism.
- The evidence does not prove causal responsibility of a single Gaussian factor, and it does not prove that lower optical depth is physically correct.

## Next Single-Factor Experiment

**Hypothesis**

- Recommended next step: `region-conditioned read-only footprint/attenuation-tail diagnostic`.
- This remains read-only and is preferred before a new training intervention because current benefit evidence is mixed across scale, footprint, alpha, and population, while the harmful pathway remains unresolved.

## Outputs

**Code Fact**

- Diagnostic script: `scripts/diagnostics/audit_bnd_cdepth_optimization_path.py`.
- Final summary: `outputs/bnd_cdepth_mitigation_path_panama_20260811/cdepth_mitigation_final_summary.json`.
- Output manifest: `outputs/bnd_cdepth_mitigation_path_panama_20260811/manifest.json`.
- Visual manifest: `renders/bnd_cdepth_mitigation_path_panama_20260811/manifest.json`.
- Visual index: `renders/bnd_cdepth_mitigation_path_panama_20260811/VISUAL_COMPARE_INDEX.md`.

Visual assets:

- `renders/bnd_cdepth_mitigation_path_panama_20260811/contact_sheet_rgb_trajectory.png`
- `renders/bnd_cdepth_mitigation_path_panama_20260811/contact_sheet_gain_harm.png`
- `renders/bnd_cdepth_mitigation_path_panama_20260811/contact_sheet_high_j_gain_harm.png`
- `renders/bnd_cdepth_mitigation_path_panama_20260811/contact_sheet_alpha_delta.png`
- `renders/bnd_cdepth_mitigation_path_panama_20260811/contact_sheet_transmission_new_low_t.png`
- `renders/bnd_cdepth_mitigation_path_panama_20260811/contact_sheet_new_low_t_harm.png`
- `renders/bnd_cdepth_mitigation_path_panama_20260811/contact_sheet_edge_harm.png`
- `renders/bnd_cdepth_mitigation_path_panama_20260811/contact_sheet_direct_medium_attribution.png`
- `renders/bnd_cdepth_mitigation_path_panama_20260811/contact_sheet_factor_summary.png`
- `renders/bnd_cdepth_mitigation_path_panama_20260811/plot_rgb_highj_trajectory.png`
- `renders/bnd_cdepth_mitigation_path_panama_20260811/plot_population_trajectory.png`
- `renders/bnd_cdepth_mitigation_path_panama_20260811/plot_scale_trajectory.png`
- `renders/bnd_cdepth_mitigation_path_panama_20260811/plot_anisotropy_trajectory.png`
- `renders/bnd_cdepth_mitigation_path_panama_20260811/plot_projected_radius_trajectory.png`
- `renders/bnd_cdepth_mitigation_path_panama_20260811/plot_opacity_trajectory.png`
- `renders/bnd_cdepth_mitigation_path_panama_20260811/plot_alpha_trajectory.png`

No subjective clear-image correctness judgment was made.
