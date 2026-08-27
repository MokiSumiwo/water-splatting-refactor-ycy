# RAOC RAY-ADAPTIVE OBSERVABILITY PREFLIGHT

## CODE FACT
RAOC is a default-off, detached capacity-control path over the existing 9-D standardized raw medium residual.
It retains the existing camera context, medium MLP, activations, CUDA rasterizer, and tied B_inf semantics.

## MATHEMATICAL DEFINITION
`Delta_z_cam = z_full - z_base`, `a_p = V^T Delta_z_cam_std`, `s_i,p = ||J_p v_i||_2`, and `e_i,p = |a_i,p| s_i,p`.
`q_i` is the train-only median evidence, with the registered mean fallback and inactive zero rule.
`g_local=e^2/(e^2+q^2)` and `g_keep=1-(1-g_obs)(1-g_local)`.
The first prototype uses no LCS cosine/alignment term.

## RESULT
Disabled-path pass: `True`; OCMC reduction pass: `True`; identity limit pass: `True`.
Global rescued-energy fraction: `0.7323188194096749`; bottom-20 rescue fraction: `0.001519949468797945`; top-20 rescue fraction: `0.91494173182777`.
Direct RAOC residual metric gradient reached the medium MLP with L2 norm `0.020786929107460713`; direct Gaussian gradient sum was `0.0`.
Calibration was train-only over 25 cameras and 1024 rays per camera; all 9 modes were active and calibration reproducibility passed.
The read-only train/eval comparisons are mechanism safety context only; they are not causal RAOC performance results.

## ENGINEERING VALIDATION
All registered readiness checks passed, including disabled-path equivalence, OCMC reduction, identity limit, gate bounds and monotonicity, detached gate state, state save/load, old-checkpoint compatibility, deterministic calibration, decomposition finiteness, and no-step parameter safety.
The 20-step engineering smoke was finite with `parameter_delta_max=0.01297568529844284`; the smoke gate distribution had `mean_g_local=0.45731955766677856`, `mean_g_keep=0.7561123371124268`, and `std_g_keep=0.24881881475448608`.
The analytic compositor action uses a fixed `16384`-pixel chunk. The measured RAOC forward was `0.9511s` relative to `0.0171s` FULL and `0.0194s` OCMC on the preflight view, with peak allocated/reserved memory of `6055337984/11098128384` bytes.

## CLASSIFICATION
Primary classification: `RAOC_MODULE_READY`.
Ray-adaptive behavior classification: `RAY_ADAPTIVE_CAPACITY_ALLOCATION_BEHAVIOR_SUPPORTED`.
Next formal experiment: `M1-RAOC-CAUSAL-IUI3` with matched OCMC and RAOC arms.
