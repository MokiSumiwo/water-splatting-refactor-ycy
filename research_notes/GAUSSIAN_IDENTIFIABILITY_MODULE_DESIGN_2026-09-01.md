# Gaussian Identifiability Module Design

Date: 2026-09-01
Classification: `MODULE_READY`

## Motivation

The frozen four-scene audit supported a Gaussian appearance-density ambiguity: non-DC/full SH sensitivity and raw-opacity sensitivity occupy nearly identical training-view RGB subspaces, and the ambiguity score predicts heldout error after depth, medium, and OCMC controls.

## Candidate Comparison

| Candidate | OCMC compatibility | Cost | valid-appearance risk | Decision |
|---|---|---|---|---|
| A: detached SH-opacity tangent orthogonalization | Separate per-Gaussian appearance axis | Periodic training-only analytic sensitivity | Low: one of 45 non-DC directions | selected |
| B: opacity-conditioned SH capacity | Compatible but opacity is not observability | Low | High: suppresses legitimate SH | rejected |
| C: update decorrelation | Interacts with all losses | High: paired gradients/update hook | Medium-high; normally GT-dependent | rejected |

## Formulation

For each Gaussian, stack the non-DC bounded-SH Jacobian `J_i`, non-DC response `r_i0`, and raw-opacity tangent `j_alpha_i` over visible training views. With `u_i=normalize(j_alpha_i)`, the detached shared direction is `h_i=normalize(J_i^T u_i)`. At refresh anchor `theta_i0`, the target `tau_i=h_i^T theta_i0-(u_i^T r_i0)/||J_i^T u_i||` nulls the shared response under the local linearization. A detached overlap gate activates only when the opacity tangent lies in the effective non-DC SH response subspace. The proposed regularizer is `lambda*sum_i g_i(h_i^T theta_i-tau_i)^2/(2 sum_i g_i)`.

Only `features_rest` receives a direct gradient. DC color, opacity, geometry, medium parameters, OCMC, and Gaussian topology are unchanged. Forty-four orthogonal non-DC SH dimensions remain unpenalized for valid view-dependent effects. At inference, learned coefficients are rendered normally: no gate, Jacobian, or extra compute remains.

## Relation With OCMC

OCMC controls camera-conditioned medium context. The selected mechanism controls a per-Gaussian SH coefficient direction that locally duplicates opacity. Their parameters and gradient pathways are disjoint, so this module is complementary rather than an OCMC replacement.

## Preflight Results

| Scene | overlap | active | parallel reduction | response reduction | orth drift | SH var ratio | SH RMS ratio | direct RGB ratio | ready |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| Curasao | 0.997913 | 0.977 | 0.298097 | 0.326737 | 1.853e-07 | 0.981386 | 0.903925 | 0.998401 | yes |
| IUI3-RedSea | 0.999420 | 0.984 | 0.328303 | 0.331204 | 1.981e-07 | 0.954852 | 0.861520 | 0.998297 | yes |
| JapaneseGradens-RedSea | 0.999420 | 0.996 | 0.327495 | 0.328595 | 2.135e-07 | 0.947961 | 0.905949 | 0.998367 | yes |
| Panama | 0.999403 | 0.984 | 0.328287 | 0.329889 | 2.565e-07 | 0.973262 | 0.904085 | 0.996722 | yes |

Strength-zero coefficient output was exactly equivalent in every scene. The direct `features_rest` gradient was nonzero, direct opacity gradient was zero, and frozen model, opacity, DC, Gaussian count, and OCMC projector hashes were unchanged. The 50-step optimization affected cloned sampled SH tensors only; it was not model training.

## Implementation Feasibility

A later causal implementation can refresh detached directions and gates at low cadence from recent training cameras, retain them only for active/visible Gaussians, and add the scalar term to the existing objective. No renderer or inference modification is required. This task deliberately did not integrate the regularizer into the production loss or training loop.

## Risks

The controller is local and first-order; stale directions may become inaccurate. A non-DC SH direction can be opacity-equivalent on observed views yet useful outside them. The future causal experiment must therefore compare multiple small strengths, monitor novel-view quality and SH utilization, and include a zero-strength equivalence branch. Preflight readiness is engineering authorization, not evidence of causal quality improvement.

## Classification

The result is `MODULE_READY` with 4/4 ready scenes. Full causal training authorization is `true`.

The next and only authorized task is `IDENTIFIABILITY_MODULE_CAUSAL_EXPERIMENT`. No 15K training was run here.

## Integrity

All four registered C0 checkpoints used OCMC on, RAOC off, bounded SH3, `dir_xy_camera`, tied `B_inf`, and classic rasterization. No heldout view or GT entered sampling, tangent construction, optimization, or classification. Checkpoint writes, render writes, model training steps, renderer/CUDA/optimizer/loss/training-loop changes were zero.
