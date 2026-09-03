# Full Medium Degradation Responsibility Diagnostic

Date: 2026-09-03
Experiment: `FULL_MEDIUM_DEGRADATION_RESPONSIBILITY_DIAGNOSTIC`
Classification: `FULL_MEDIUM_DEGRADATION_RESPONSIBILITY_NOT_SUPPORTED`
Module design authorized: `False`

## Previous Formal Result

The preceding frozen four-scene audit was classified `MULTIVIEW_DEGRADATION_RESPONSIBILITY_TENTATIVE`: alignment 3/4, strict heldout prediction 2/4, B-R/B-G color 2/4, primary controls 3/4, and temporal stability 2/4. It used the same C0 OCMC-on / RAOC-off state and an attenuation-only medium response.

## Motivation And Scientific Boundary

This targeted read-only Stage-A diagnostic tests one limitation only: the earlier response omitted additive finite-interval and terminal-tail medium terms that can produce blue or cyan image residuals. It tests whether the complete renderer medium response explains cross-view RGB residual differentials after OCMC controls camera-conditioned medium ambiguity. It does not test physical water/surface identity.

All analyses use frozen C0 checkpoints with OCMC on and RAOC off. No training, backward call, optimizer step, checkpoint write, module design, Gaussian identity state, water/surface label, pruning, or densification was used. Stage B was skipped because it is authorized only after all preregistered Stage-A criteria pass.

## Difference From Gaussian Identity Classification

The unit of analysis is a residual-explanation relation for a frozen Gaussian/view/pixel contribution. The score `q` is not a water probability, permanent Gaussian label, identity archive, or pruning/densification decision. No Gaussian is assigned a water or surface state.

## Exact Renderer Equation

The registered bounded-SH3 Gaussian color is `J_i(v)`. With `w_i,p=T_i,p*alpha_i,p`, the actual classic renderer is:

`I_pred(p,v) = sum_i w_i,p exp(-medium_attn_p(v)*depth_i(v)) J_i(v) + M_finite(p,v) + M_tail(p,v)`.

The finite term is `M_finite=sum_i T_i[exp(-medium_bs_p*prev_depth_i)-exp(-medium_bs_p*depth_i)] medium_rgb_p`. The terminal term is `M_tail=T_final exp(-medium_bs_p*last_depth) b_inf,p`; in the registered tied mode `b_inf=medium_rgb`. These are code-derived image terms. Finite and tail terms are not owned by any Gaussian.

## Response Definitions

The attenuation response is `D_att_i,p=(exp(-tau_i,p)-1)J_i(v)`, where `tau_i,p=medium_attn_p*depth_i`. The additive response is the local footprint response obtained by applying the same Gaussian responsibility aggregation to `M_finite+M_tail`: `d_add=d_finite+d_tail`. The registered full response is `d_full=d_att+d_add`, with no learned or tuned weights.

The local additive response is an image-level renderer response sampled over a Gaussian compositing footprint. It is not a claim that finite or tail medium belongs to Gaussian `i`, and it is not Gaussian water ownership.

## Numerical Equivalence

For every audited renderer row, `pred_image` was reconstructed as `rgb_object + rgb_medium_finite + rgb_tail`; all rows passed the preregistered `max_abs <= 1e-5` gate. The largest observed reconstruction error across scenes was `1.1920929e-07`.

| Scene | finite L1 mean | finite RMS mean | tail L1 mean | tail RMS mean | additive L1 mean | additive RMS mean | rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| Curasao | 0.055617 | 0.073307 | 0.015690 | 0.050041 | 0.071307 | 0.094768 | 90 |
| IUI3-RedSea | 0.182979 | 0.223951 | 0.070073 | 0.150267 | 0.253051 | 0.278475 | 125 |
| JapaneseGradens-RedSea | 0.082178 | 0.100627 | 0.025676 | 0.072789 | 0.107854 | 0.136977 | 85 |
| Panama | 0.068201 | 0.091331 | 0.006796 | 0.021838 | 0.074997 | 0.102864 | 75 |

The finite and tail magnitudes above are image-level L1/RMS summaries, not ownership scores. Finite magnitude exceeds tail magnitude in all four scene means, but this does not identify which component caused any q generalization change because the registered q_add predictor combines finite and tail.

## Responsibility And Residual Protocol

Responsibility is the unchanged classic alpha-compositing weight `w_i,p=T_i,p*alpha_i,p`. A diagnostic-only CUDA hook reuses the existing sorted tile bins, alpha threshold, medium-aware eligibility test, and early termination, and was checked against native `out_clr` with unit colors. It does not modify the repository renderer.

For a training camera, `e_i,v=sum_p w_i,p*(I_GT-I_pred)/(sum_p w_i,p+eps)` and `d_i,v=sum_p w_i,p*D_i,p/(sum_p w_i,p+eps)`. For camera pairs, `Delta e=e_i,a-e_i,b`, `Delta d= d_i,a-d_i,b`, and `q_i^(a,b)=cos(Delta e,Delta d)`, summarized by the contribution-weighted median. The same camera bank, support floor, pair filtering, view diversity, 200-replicate matched null, heldout protocol, color criterion, controls, and temporal criterion as the previous audit were reused. q predictors were frozen before heldout GT was read.

## Historical q_att Regression

The final-checkpoint q_att calculation reused the previous audit's `_pair_rows` implementation. The fixed tolerance was 1e-5; it covers float32 CUDA atomic reduction ordering and is far below a substantive score change.

| Scene | historical q_att | rerun q_att | absolute error | pass |
|---|---:|---:|---:|:---:|
| Curasao | 0.164669125 | 0.164669756 | 6.32e-07 | yes |
| IUI3-RedSea | 0.400278732 | 0.400278369 | 3.63e-07 | yes |
| JapaneseGradens-RedSea | 0.474325228 | 0.474326775 | 1.55e-06 | yes |
| Panama | -0.023857838 | -0.023857500 | 3.38e-07 | yes |

All four historical regressions passed, so the full-medium comparison was not substituted for a failed baseline.

## Four-Scene q Results

Formal decisions use q_full only. q_att and q_add are reported for mechanism attribution.

| Scene | q_att | q_add | q_full | full null p95 | heldout q_full rho | heldout null p95 | alignment | heldout | color | controls | temporal |
|---|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|
| Curasao | 0.164670 | 0.038167 | 0.113824 | 0.021942 | 0.171568 | 0.110482 | yes | yes | yes | yes | yes |
| IUI3-RedSea | 0.400278 | -0.182518 | 0.220905 | 0.028828 | 0.161609 | 0.101722 | yes | yes | no | yes | yes |
| JapaneseGradens-RedSea | 0.474327 | -0.193277 | 0.084097 | 0.023387 | 0.092128 | 0.108652 | yes | no | no | yes | yes |
| Panama | -0.023858 | 0.142046 | 0.129603 | 0.024355 | 0.086139 | 0.060452 | yes | yes | yes | yes | yes |

## Heldout Prediction And Color Residual

Training-only q_full maps were projected to frozen heldout views using responsibility weights. The strict heldout criterion requires positive scene-mean Spearman rho above the scene-specific 200-replicate Gaussian-q permutation p95; `3/4 scenes pass` passed. Curasao changed from the previous attenuation-only rho `-0.062056` to full-response rho `0.171568`, above its null p95 `0.110482`, so its negative correlation was corrected. JapaneseGradens-RedSea reached rho `0.092128`, below its null p95 `0.108652`, so it did not pass. Panama's prior negative attenuation alignment was corrected by q_full `0.129603` versus full null p95 `0.024355`.

The preregistered color criterion requires both B-R and B-G enrichment to be positive in at least 3/4 scenes. The result was `2/4 scenes pass`. Curasao and Panama pass; IUI3-RedSea and JapaneseGradens-RedSea fail because both required enrichments are negative.

## Component Attribution

q_add is the fixed joint finite-plus-tail local response; no finite-only or tail-only q was selected after looking at heldout results. The full-response changes are mixed: q_full improves heldout rho over q_att in Curasao and Panama, while it is lower than q_att in IUI3-RedSea and JapaneseGradens-RedSea. The data therefore show a partial generalization rescue, not a uniform improvement.

The registered experiment cannot attribute q generalization gains separately to attenuation, finite medium, or tail medium. It contains no q_finite/q_tail causal comparison, and image-level finite amplitude is not evidence of causal responsibility. The only defensible attribution is that the renderer-complete sum changed the result in a scene-dependent way; the finite-versus-tail mechanism remains unresolved under this closed formulation.

## Controls And Temporal Analysis

Single-variable rank residualization retained the previous depth, tau, transmission, opacity, scale, footprint, support, OCMC magnitude, medium-residual magnitude, and SH view-response controls. The preregistered primary control gate result was `4/4 scenes pass`.

The five checkpoints were analyzed at population level only; no Gaussian index was treated as a cross-checkpoint lineage. The fixed direction criterion was satisfied by `4/4 scenes pass`. No lineage claim was made.

## Optional Stage B

Stage B local routing counterfactual was not executed. It is authorized only if alignment, strict heldout prediction, B-R/B-G color, controls, temporal stability, and the training-only construction condition all pass. The full Stage-A result did not meet that gate, so no SM versus SG comparison exists and no training, backward pass, optimizer update, or checkpoint write was allowed.

## Current Best RGB Figure

The requested `current_best_rgb_dewatering_comparison.png`, `current_best_rgb_dewatering_comparison.pdf`, and `rgb_comparison_manifest.json` were not generated. The repository does not contain a complete four-scene auxiliary-regularization render set with verified checkpoint, camera, crop, resolution, and split provenance. The available same-C0 formal heldout renders are OCMC-only and cannot validly populate the `Current Best` auxiliary-regularization column. No incompatible assets were mixed and no post-enhancement comparison was fabricated.

## Final Questions Answered

1. Historical attenuation-only q was reproduced in all four scenes within the fixed 1e-5 tolerance.
2. Yes. Full degraded RGB was reconstructed from `rgb_object + rgb_medium_finite + rgb_tail`; the maximum absolute error was 1.1920929e-07.
3. Mean image-level finite/tail L1 magnitudes were Curasao 0.055617/0.015690, IUI3-RedSea 0.182979/0.070073, JapaneseGradens-RedSea 0.082178/0.025676, and Panama 0.068201/0.006796. These are not causal ownership measures.
4. Final q_add was Curasao 0.038167, IUI3-RedSea -0.182518, JapaneseGradens-RedSea -0.193277, and Panama 0.142046.
5. Final q_full was Curasao 0.113824, IUI3-RedSea 0.220905, JapaneseGradens-RedSea 0.084097, and Panama 0.129603.
6. q_full alignment exceeded the matched null p95 in 4/4 scenes.
7. Yes. Strict heldout prediction reached 3/4 scenes.
8. No. The B-R/B-G criterion reached 2/4 scenes.
9. Yes. Curasao changed from attenuation-only rho -0.062056 to full-response rho 0.171568, above its null p95 0.110482.
10. No. JapaneseGradens-RedSea rho 0.092128 remained below permutation null p95 0.108652.
11. Yes. Panama's negative attenuation alignment -0.023858 was corrected by q_full 0.129603, above full null p95 0.024355.
12. Yes. Primary controls passed in 4/4 scenes.
13. Yes. Temporal stability passed in 4/4 scenes.
14. Full response produced partial, scene-dependent improvement: heldout increased from 2/4 to 3/4 and corrected Curasao/Panama, but color remained 2/4 and the full Stage-A gate failed.
15. The gain cannot be assigned to attenuation, finite, or tail separately. q_add jointly combines finite and tail; finite image magnitude was larger, but that is not causal evidence.
16. No. Stage B was not executed because the Stage-A authorization gate failed.
17. Not applicable. No SM-versus-SG result exists.
18. No. The current MDRR mechanism is not supported.
19. No. `MODULE_DESIGN_AUTHORIZED` is `false`.
20. Yes. Under the hard stopping rule, the current MDRR formulation is permanently closed.

## Final Decision

The final classification is `FULL_MEDIUM_DEGRADATION_RESPONSIBILITY_NOT_SUPPORTED`. Alignment: `4/4 scenes pass`. Heldout: `3/4 scenes pass`. Color: `2/4 scenes pass`. Controls: `4/4 scenes pass`. Temporal stability: `4/4 scenes pass`. Counterfactual: `NOT_EXECUTED_STAGE_A_NOT_AUTHORIZED`. OCMC independence: `OCMC_ON_RAOC_OFF; OCMC magnitude included as control`. Current MDRR formulation: `CLOSED`.

Full medium response did not provide uniform generalization evidence: alignment passed in all four scenes, but the strict heldout and color criteria passed in only 3/4 and 2/4 scenes, respectively. Under the hard stopping rule, the current MDRR formulation is permanently closed for this line of work. MDRR module design is not authorized and no MDRR module was designed or implemented.
