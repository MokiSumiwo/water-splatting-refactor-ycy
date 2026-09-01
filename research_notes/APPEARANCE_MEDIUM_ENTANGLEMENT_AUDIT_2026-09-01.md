# Appearance-Medium Entanglement Audit Under OCMC

Date: 2026-09-01
Experiment: `APPEARANCE_MEDIUM_ENTANGLEMENT_AUDIT_OCMC`
Classification: `NOT_SUPPORTED`

## Hypothesis

The frozen bounded-SH appearance residual and classic underwater medium representation may vary in opposite directions across training views. If this paired compensation predicts heldout RGB error after geometry, attenuation, opacity, footprint, and OCMC controls, it is a candidate failure mechanism. This audit does not identify physical ground-truth appearance or medium parameters.

## Frozen Protocol

All 20 registered C0 checkpoints use OCMC on, RAOC off, runtime-canonical `bounded_sh3`, `dir_xy_camera`, `b_inf_mode=tied`, `infinite_water_enabled=false`, and classic rasterization. Historical source YAML files serialize the same bounded sigmoid-SH parameterization under its old `sigmoid_sh` name; the protected formal setup normalizes it to `bounded_sh3` before checkpoint loading, and every worker verifies the canonical runtime value. The audit uses detached forward passes only. It does not train, call backward or optimizer.step, modify model code, write checkpoints, or write renders.

Sampling uses training visibility only with support at least two. SH/medium metrics and samples are frozen before heldout GT is accessed. Heldout GT is used only as the error outcome in projected Gaussian footprint boxes and camera RGB MSE.

## Metric Definition

For opacity-area weight `w`, `C_SH = w*(RGB_fullSH-RGB_DC)`. `C_medium` is the sum of the DC direct attenuation residual `w*(T_D-1)*RGB_DC` and the existing renderer-integrated `rgb_medium` sampled at the projected Gaussian center and weighted by `w`. The explicit SH-attenuation interaction is `w*(T_D-1)*(RGB_fullSH-RGB_DC)`. `VC_SH` and `VC_medium` are population RGB variances over visible training cameras.

`SH_medium_corr` is the centered vector correlation of paired `C_SH` and `C_medium` across training views. `compensation_score=-SH_medium_corr`, so a positive score means opposite variation. This is a representational association proxy, not a causal intervention.

## Final Results

| Scene | median SH-medium corr | rho(comp,error) | controlled depth/tau/T | controlled OCMC | rho VC_SH | AUROC SH | rho VC_med | AUROC med | temporal | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| Curasao | 0.272460 | -0.008408 | 0.042679 | 0.037972 | 0.127728 | 0.549192 | 0.080387 | 0.526960 | 0/5 | no |
| IUI3-RedSea | -0.112636 | -0.149159 | -0.118644 | -0.157144 | 0.090852 | 0.496576 | 0.118625 | 0.560192 | 0/5 | no |
| JapaneseGradens-RedSea | 0.128240 | -0.038138 | 0.002927 | -0.038541 | 0.186133 | 0.545651 | 0.040671 | 0.505386 | 0/5 | no |
| Panama | 0.274322 | 0.173252 | 0.184082 | 0.157589 | 0.069527 | 0.484825 | -0.094400 | 0.427692 | 0/5 | no |

Only IUI3 has a negative median SH-medium correlation at the final checkpoint, but its compensation-error association has the wrong direction (`rho=-0.149159`) and remains negative after every control set. Panama has a positive compensation-error association that survives the controls, but its median SH-medium correlation is positive at all five checkpoints; this is stable co-variation rather than systematic compensation. Curasao and JapaneseGradens satisfy neither part of the mechanism chain.

`VC_SH` has the stronger final error association in Curasao, JapaneseGradens, and Panama; `VC_medium` is stronger only in IUI3. Neither branch is a uniformly strong predictor: the largest final association is `rho=0.186133` for `VC_SH` in JapaneseGradens, and the AUROCs remain close to chance. This does not establish either stability metric as a cross-scene novel-view failure mechanism.

## Residual Decomposition

| Scene | rho SH align | rho medium align | rho interaction | rho joint | best single | rank R2 multivariate |
|---|---:|---:|---:|---:|---|---:|
| Curasao | 0.118117 | 0.047403 | 0.064409 | 0.068048 | heldout_SH_alignment | 0.028548 |
| IUI3-RedSea | 0.075379 | 0.073657 | -0.083966 | 0.092952 | heldout_joint_alignment | 0.057919 |
| JapaneseGradens-RedSea | 0.214623 | 0.104545 | 0.022627 | 0.203398 | heldout_SH_alignment | 0.071812 |
| Panama | 0.098118 | -0.088033 | -0.022558 | -0.048179 | heldout_SH_alignment | 0.098377 |

These decomposition values measure rank association with heldout local error; they are not causal MSE attribution and do not imply true component ownership.

## Temporal Stability

No checkpoint contains persistent Gaussian lineage IDs. Temporal recurrence is therefore population-level only; array-index and nearest-geometry identity matching were not used.

| Scene | 5k | 8k | 10k | 13k | 14999 | passes |
|---|:---:|:---:|:---:|:---:|:---:|---:|
| Curasao | no | no | no | no | no | 0/5 |
| IUI3-RedSea | no | no | no | no | no | 0/5 |
| JapaneseGradens-RedSea | no | no | no | no | no | 0/5 |
| Panama | no | no | no | no | no | 0/5 |

## OCMC Independence

The OCMC independence test jointly rank-residualizes compensation score and heldout error against OCMC active projected magnitude and suppressed medium residual. The stricter all-control result additionally includes depth, tau, transmission, opacity, and footprint. OCMC and model state hashes remain unchanged for every checkpoint.

Panama's compensation-error ranking survives OCMC controls, but Panama fails the prerequisite compensation-direction test. The other scenes do not preserve a complete positive mechanism chain. Therefore no SH-medium entanglement mechanism is established as independent of OCMC.

## Limitations

`rgb_medium` is an exact renderer-integrated ray contribution but not a per-Gaussian physical attribution; weighting its projected-center sample by the Gaussian opacity-area proxy only associates that ray contribution with a Gaussian. Projected footprint boxes overlap, occlusion is not assigned exactly, heldout camera counts are small, and no true appearance/medium labels exist. The audit is observational and cannot establish causal compensation.

## Classification

The formal result is `NOT_SUPPORTED` with 0/4 supported scenes. Module design authorization is `false`.

Appearance-medium entanglement is not supported as a novel-view failure mechanism under this protocol. Close this direction and return to failure-hypothesis selection; do not design a module from this signal.

## Integrity

Analyzed 31352 Gaussian-checkpoint rows and 65 heldout camera-checkpoint rows. All 20 checkpoint hashes and protected source hashes matched before and after execution. Backward calls, optimizer steps, checkpoint writes, and render writes were all zero.

## Disk Management

Available space increased from `43,269,959,680` to `46,639,509,504` bytes. Deleted only `outputs/raoc_q50_q80_causal_four_scene_20260828_attempt1_interrupted_tool_session` and `outputs/raoc_q50_q80_causal_four_scene_20260828_attempt2_empty_tool_cleanup`, totaling `3,369,260,269` logical bytes. The first directory contained only eight partial 3k/5k checkpoints and no summary/classification; the second was an explicitly named empty cleanup attempt. The later formal `outputs/raoc_q50_q80_causal_four_scene_20260828` directory remains intact with nine checkpoints. No formal OCMC or RAOC checkpoint was deleted.
