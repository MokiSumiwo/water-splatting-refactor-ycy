# LOCAL CONTEXTUAL SUPPORT PREDICTOR PREFLIGHT

## CODE FACT
This was a read-only diagnostic using the frozen C1 step-14999 checkpoint, mode_01, basis, and held-out banks.
No optimizer step, training, mode reselection, threshold fitting, classifier fitting, or projector change was performed.

## MOTIVATION
Global mode gating was rejected because the previous alignment audit found sparse context-dependent support and mixed full-view removal metrics.
This preflight tests whether inference-available local model evidence can identify the supportive subset without using labels at inference time.

## DEFINITION
For local RGB Jacobian J_p in standardized raw medium coordinates, selected mode vector v_i, and camera residual Delta_z_cam_std:
`a_i = v_i^T Delta_z_cam_std`, `d_i = a_i J_p v_i`, and `d_cam = J_p Delta_z_cam_std`.
`E_mode = ||d_i||_2`, `E_cam = ||d_cam||_2`, `R = E_mode / (E_cam + 1e-12)`.
The preregistered primary score is `LCS = R * max(cos_i, 0)` where `cos_i = dot(d_i, d_cam) / (||d_i||_2 ||d_cam||_2 + 1e-12)`.
LCS uses only the current ray, camera context, frozen medium residual, and local Jacobian; it uses no GT RGB, camera-swap error, C_utility, or C_rgb at inference.

## RESULT
GENERAL Q1 base rate: `0.4260546875`.
GENERAL LCS AUROC: `0.6689011296642718`; AUPRC: `0.5785622299117319`; AUPRC minus base rate: `0.15250754241173192`.
GENERAL LCS Spearman with C_rgb: `0.1489401370267113`.
First-order fidelity cosine mean: `0.7922019328967642`; relative L2 error median: `0.0391166307864035`.
M_SAFE LCS AUROC: `0.7567455877480127`; top20-bottom20 Q1 gap: `0.36640625`; top20-bottom20 C_rgb: `1.828747459797775e-07`.
Eval LCS Spearman with C_rgb: `-0.13071020721736343`; top20-bottom20 C_rgb: `-8.392266021728241e-06` (eval is mixed and does not independently reproduce the GENERAL trend).
Local Jacobian action method: exact closed-form forward-compositor action with no full 3x9 Jacobian materialization and no backward/JVP/VJP calls; no optimizer step was run.
GENERAL time per 1024 rays: `0.049298181906342504` seconds; relative overhead vs ordinary forward: `2.901127791833769`x; peak allocated memory: `1834910208` bytes.
Primary classification: `LOCAL_CONTEXTUAL_SUPPORT_SUPPORTED`.
Granularity classification: `RAY_CONTEXT_ADAPTIVE_CONTROL_SUPPORTED`.

## NEXT TASK
`Implement a generic ray/context-adaptive capacity mechanism using global observability prior plus local contextual support, only if the primary classification is SUPPORTED.`
