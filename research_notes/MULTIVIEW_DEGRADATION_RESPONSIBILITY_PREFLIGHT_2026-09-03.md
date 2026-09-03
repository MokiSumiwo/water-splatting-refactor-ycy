# Multi-view Degradation Responsibility Preflight

Date: 2026-09-03
Experiment: `MULTIVIEW_DEGRADATION_RESPONSIBILITY_PREFLIGHT`
Classification: `MULTIVIEW_DEGRADATION_RESPONSIBILITY_TENTATIVE`
Module design authorized: `False`

## Motivation

This frozen Stage-A diagnostic tests whether cross-view RGB residual differentials retain a medium-aligned component after OCMC controls camera-conditioned medium ambiguity. It studies responsibility routing, not physical water/surface identity.

## Difference From Gaussian Identity Classification

The unit of analysis is a residual-explanation relation for a frozen Gaussian/view/pixel contribution. The score `q` is not a water probability, a permanent Gaussian label, an identity archive, or a pruning/densification decision. No Gaussian is assigned a water or surface state.

## Difference From Previous View-Consistency Audit

The earlier generic view-consistency analysis concerned absolute appearance variation or consistency. This audit instead compares the differential residual `Delta e` with the differential direct-medium response `Delta d` for the same Gaussian across real training-camera pairs, and tests that alignment against a within-Gaussian matched permutation null.

## Difference From OCMC

OCMC is the locked C0 mechanism that controls where camera-conditioned medium capacity is represented. This Stage-A audit leaves OCMC unchanged and asks a complementary question: after that control, is the remaining cross-view residual explainable by the medium response strongly enough to guide optimization responsibility? OCMC is ON and RAOC is OFF.

## Scientific Boundary

All analyses use C0: OCMC on and RAOC off. No training, backward call, optimizer step, checkpoint write, new module, Gaussian identity state, water/surface label, pruning, or densification was used. Stage B local counterfactual was skipped because it is authorized only after a formal Stage-A SUPPORTED result.

## Actual Renderer Formulation

The intrinsic Gaussian color is `J_i(v)`, computed by the registered bounded SH3 parameterization. The classic renderer uses `w_i,p=T_i,p*alpha_i,p`, direct object color `exp(-medium_attn_p*depth_i)*J_i`, finite medium intervals from `medium_bs` and `medium_rgb`, and a final tied tail `T_final*exp(-medium_bs*last_depth)*b_inf`. With tied mode, `b_inf=medium_rgb`.

The per-Gaussian medium response used here is the exact direct-object attenuation difference `D_i,p=exp(-tau_i,p)J_i-J_i`. The additive finite/tail medium term is not incorrectly assigned to a Gaussian. This is a renderer-derived response, not a claim of physical attenuation ground truth.

## Definition Of Medium Response

For a Gaussian contribution at pixel `p`, `tau_i,p=medium_attn_p*depth_i` and `D_i,p=(exp(-tau_i,p)-1)J_i(v)`. This is the difference between the renderer's direct degraded Gaussian color and its intrinsic color. It excludes the additive finite-interval and tail medium terms because those terms are image-level medium contributions rather than uniquely attributable Gaussian responses.

## Definition Of Rendering Responsibility

Responsibility is the classic alpha-compositing weight `w_i,p=T_i,p*alpha_i,p`, where `T_i,p` is the accumulated transmittance before Gaussian `i`. The diagnostic hook reproduces the existing tile order, alpha cutoff, medium-aware eligibility test, and early termination, then accumulates selected-Gaussian statistics without changing native forward output.

## Definition Of Residual Differential

For a training camera, `e_i,v=sum_p w_i,p*(I_GT-I_pred)/(sum_p w_i,p+eps)` and `d_i,v=sum_p w_i,p*D_i,p/(sum_p w_i,p+eps)`. For cameras `a,b`, `Delta e=e_i,a-e_i,b` and `Delta d=d_i,a-d_i,b`. Only pairs with fixed contribution floor and non-degenerate differentials are retained.

## Definition Of Q

`q_i^(a,b)=cos(Delta e,Delta d)` and `q_i` is its contribution-weighted median over valid pairs. Positive q means the residual change and direct-medium-response change point in the same RGB direction; it does not imply Gaussian identity.

## Responsibility Extraction

The public renderer does not expose per-Gaussian `T*alpha`. A separate diagnostic-only CUDA forward hook reuses the existing sorted tile bins and the classic kernel's alpha threshold and early termination, then accumulates selected-Gaussian weights and weighted residual/response statistics. It does not modify the repository renderer. Its pixel total is checked against native RGB `out_clr` with unit colors using the preregistered aggregate tolerance recorded in `renderer_decomposition_check.json`.

## Protocol

For training views, `e_i,v=sum_p w_i,p*(GT-pred)/sum_p w_i,p` and `d_i,v=sum_p w_i,p*D_i,p/sum_p w_i,p`. Only Gaussians visible in at least three training views enter the preregistered sample. Pairwise values are `q=cos(Delta_e,Delta_d)` for all valid training-camera pairs. The matched null independently permutes medium-response camera assignments within each Gaussian, preserving Gaussian view coverage.

The q predictor is frozen before heldout GT is read. Heldout metrics use a responsibility-weighted q map and report residual magnitude, Spearman correlation, AUROC for top-20% residual, and B-R/B-G chromatic residual contrasts. Criterion B requires positive scene-mean rho above a fixed 200-replicate Gaussian-q permutation p95. No heldout GT enters q construction.

## Four-Scene Results

The primary final-checkpoint alignment criterion is `median q > matched-null p95`. The table below also reports the strict heldout permutation comparison and both required chromatic contrasts.

## Results

| Scene | final median q | alignment null p95 | heldout rho | heldout null p95 | alignment | heldout | color | controls | temporal |
|---|---:|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|
| Curasao | 0.164669 | 0.039892 | -0.062056 | 0.109826 | yes | no | no | no | yes |
| IUI3-RedSea | 0.400279 | 0.030097 | 0.187549 | 0.100112 | yes | yes | no | yes | yes |
| JapaneseGradens-RedSea | 0.474325 | 0.061520 | 0.108117 | 0.119370 | yes | no | yes | yes | no |
| Panama | -0.023858 | 0.032083 | 0.060399 | 0.054294 | no | yes | yes | yes | no |

Temporal rows are population-level checkpoint statistics only. There is no cross-checkpoint Gaussian lineage assumption; array index matching is forbidden and was not used.

## Limitations

The selected-Gaussian responsibility hook is exact for the current classic forward rule, but the experiment samples a fixed, support-stratified Gaussian population for tractable frozen analysis. The direct attenuation response does not claim that additive finite/tail medium is physically attributable to individual Gaussians. Chromaticity is evaluated only after training q is frozen. Stage A does not establish physical water decomposition.

## Heldout Prediction

Heldout q maps are constructed only from training-camera q values and then projected with frozen responsibility weights. Criterion B uses fixed top-20% residual evaluation and a fixed 200-replicate permutation null over Gaussian q assignments. The final strict result is `2/4` scenes passing.

## Blue And Color Residual Analysis

For each heldout camera, the top-20% and bottom-20% q regions are compared using both B-R and B-G residual contrasts. Criterion C requires both final scene-mean contrasts to be positive; `2/4` scenes pass.

## Control Analysis

Single-variable rank residualization reports depth, tau, transmission, opacity, scale, footprint, support, OCMC magnitude, medium-residual magnitude, and SH view response. The preregistered primary control gate uses depth, tau, transmission, opacity, footprint, and OCMC magnitude; `3/4` scenes pass.

## Temporal Analysis

The five checkpoint rows are population-level statistics only. No Gaussian index is treated as a cross-checkpoint lineage. The fixed direction criterion is satisfied by at least 4/5 checkpoints in `2/4` scenes.

## Optional Local Counterfactual

Stage B was not executed. Its authorization is conditional on a formal Stage-A SUPPORTED result, which was not reached.

## Final Decision

The final classification is `MULTIVIEW_DEGRADATION_RESPONSIBILITY_TENTATIVE`. Alignment: `3/4 scenes pass`. Heldout: `2/4 scenes pass`. Color: `2/4 scenes pass`. Controls: `3/4 scenes pass`. Temporal stability: `2/4 scenes pass`. Counterfactual: `NOT_EXECUTED_STAGE_A_NOT_AUTHORIZED`. OCMC independence: `OCMC_ON_RAOC_OFF; OCMC magnitude included as control`.

MDRR module design is not authorized unless the preregistered supported criteria are met. No MDRR module was designed or implemented in this task.
