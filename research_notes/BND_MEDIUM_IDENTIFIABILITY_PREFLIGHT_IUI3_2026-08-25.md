# BND-MEDIUM-IDENTIFIABILITY-PREFLIGHT-IUI3

## Scope
CONFIG FACT: This is a read-only Phase-A preflight. No optimizer, parameter update, full training, new loss, new module, checkpoint write, CDEPTH, MEDCTX removal, CB loss, OMVC, or opacity/densification intervention is used.
CONFIG FACT: Primary scene is IUI3-RedSea with the established train/eval split; matched checkpoints are M1/BND nominal 5000, 10000, and 15000 where 15000 maps to actual 14999.

## Environment
EXPERIMENTAL FACT: CONDA_ENV `water_splatting`; Python `/opt/anaconda3/envs/water_splatting/bin/python`; Torch `2.1.2+cu118`.
EXPERIMENTAL FACT: CUDA_VISIBLE_DEVICES `6`; torch logical cuda:0 is physical GPU `6`; GPU `NVIDIA GeForce RTX 3080`.

## Exact Medium Semantics
CODE FACT: The medium MLP takes 22-D `dir_xy_camera` features: 16-D direction encoding, 3-D XY/r context, and 3-D camera context.
CODE FACT: The 9-D raw output is activated as `B_inf=medium_rgb=sigmoid(z[0:3])`, `beta_B=softplus(z[3:6]+medium_density_bias)`, and `beta_D=softplus(z[6:9]+medium_density_bias)`.
CODE FACT: Direct object RGB is attenuated by `exp(-beta_D*depth)`. Medium backscatter uses `medium_rgb` and `beta_B` across Gaussian depth intervals plus the final tail.
CODE FACT: With `b_inf_mode=tied`, Python recomposes the tail with `b_inf=medium_rgb`, preserving rendered RGB while exposing `b_inf` semantics.

## Pre-Activation Access
EXPERIMENTAL FACT: `AVAILABLE_DIAGNOSTIC_MANUAL_MEDIUM_MLP_FORWARD`.
QUANTITATIVE RESULT: source-equivalence max abs diffs `{'medium_rgb': 0.0, 'medium_bs': 0.0, 'medium_attn': 0.0, 'b_inf': 0.0, 'pred_image': 0.0, 'depth': 0.0, 'accumulation': 0.0, 'rgb_medium': 0.0}`.

## Deterministic Sampling
CONFIG FACT: GENERAL sampled rays `25600`; M_SAFE sampled rays `25600`.
CONFIG FACT: `M_SAFE` reuses the locked IUI3 pseudo-depth/background and BND@3000 low-accumulation candidate semantics; it is a diagnostic population, not training supervision.

## Aggregate Structured Jacobian
CONFIG FACT: The perturbation is shared across sampled rays as `z_med' = z_med + S*delta`, with `S_j=max(std(z_j),1e-3)` per checkpoint and population.
QUANTITATIVE RESULT: Phase-A classification `MEDIUM_IDENTIFIABILITY_SUPPORTED` using relevant population `M_SAFE`.
QUANTITATIVE RESULT: ill-conditioned BND steps `[5000, 10000, 15000]`.

## Weak Mode
QUANTITATIVE RESULT: weak-mode family `WEAK_MODE_BETAD`.
QUANTITATIVE RESULT: stability summary `{'checkpoint_adjacent_abs_cosines': [0.9999999555589678, 0.9999998415086103], 'checkpoint_adjacent_abs_cosine_min': 0.9999998415086103, 'checkpoint_adjacent_abs_cosine_mean': 0.999999898533789, 'weak_mode_families': ['WEAK_MODE_BETAD', 'WEAK_MODE_BETAD', 'WEAK_MODE_BETAD'], 'same_family_across_matched_steps': True, 'camera_abs_cosine_median': 0.999999892209994, 'camera_abs_cosine_p25': 0.9999997756383168, 'camera_abs_cosine_min': 2.2809066049918063e-12}`.

## Counterfactual Perturbation
QUANTITATIVE RESULT: BND counterfactual steps satisfying `RGB_change(v_min) <= 0.25*RGB_change(v_max)`: `[5000, 10000, 15000]`.
EXPERIMENTAL FACT: Counterfactuals use one fixed standardized step epsilon=0.25 and no parameter updates.

## M1 vs BND
QUANTITATIVE RESULT: M1-vs-BND summary `{'GENERAL': [{'step': 5000, 'M1_sigma_min_over_sigma_max': 0.054234064796035125, 'BND_sigma_min_over_sigma_max': 0.09843094539435022, 'BND_over_M1_sigma_ratio': 1.8149284174905915, 'M1_condition_number': 18.438595811706644, 'BND_condition_number': 10.159406637756405, 'M1_vmin_over_vmax_cf_rgb': 0.2002979475732029, 'BND_vmin_over_vmax_cf_rgb': 0.3548704370411802}, {'step': 10000, 'M1_sigma_min_over_sigma_max': 0.059455788843438766, 'BND_sigma_min_over_sigma_max': 0.0973769271939665, 'BND_over_M1_sigma_ratio': 1.6378039731401615, 'M1_condition_number': 16.819220120571234, 'BND_condition_number': 10.269373134028822, 'M1_vmin_over_vmax_cf_rgb': 0.20633135915193332, 'BND_vmin_over_vmax_cf_rgb': 0.35907057720533636}, {'step': 15000, 'M1_sigma_min_over_sigma_max': 0.05852221086737119, 'BND_sigma_min_over_sigma_max': 0.0977081837124065, 'BND_over_M1_sigma_ratio': 1.6695914638946645, 'M1_condition_number': 17.087529421373002, 'BND_condition_number': 10.234557250019018, 'M1_vmin_over_vmax_cf_rgb': 0.21841265000599622, 'BND_vmin_over_vmax_cf_rgb': 0.35356702859851635}], 'M_SAFE': [{'step': 5000, 'M1_sigma_min_over_sigma_max': 0.001378596163108359, 'BND_sigma_min_over_sigma_max': 0.0005455535077641334, 'BND_over_M1_sigma_ratio': 0.395731195518533, 'M1_condition_number': 725.375586237867, 'BND_condition_number': 1833.0007703521972, 'M1_vmin_over_vmax_cf_rgb': 0.002249726109603123, 'BND_vmin_over_vmax_cf_rgb': 0.0003461181522614505}, {'step': 10000, 'M1_sigma_min_over_sigma_max': 0.0015176772030397387, 'BND_sigma_min_over_sigma_max': 0.001640808614322395, 'BND_over_M1_sigma_ratio': 1.0811314889859567, 'M1_condition_number': 658.9016412693761, 'BND_condition_number': 609.4556008977136, 'M1_vmin_over_vmax_cf_rgb': 0.001589157299190614, 'BND_vmin_over_vmax_cf_rgb': 0.0010141513988193174}, {'step': 15000, 'M1_sigma_min_over_sigma_max': 0.0014045981805690208, 'BND_sigma_min_over_sigma_max': 0.0018402844183269106, 'BND_over_M1_sigma_ratio': 1.3101856771460345, 'M1_condition_number': 711.9473838381928, 'BND_condition_number': 543.3942656044152, 'M1_vmin_over_vmax_cf_rgb': 0.0019590153487440503, 'BND_vmin_over_vmax_cf_rgb': 0.003038414218608923}]}`.

## Interpretation
INFERENCE: BND does not remove a stable, actionable low-observability medium direction under this preflight gate.
HYPOTHESIS: A later single-factor BND-MIC experiment can target medium-local weak-mode variation without changing the bounded object representation.

## Phase-B Gate
INFERENCE: `ENTER_PHASE_B`.
CONFIG FACT: Phase B must not be entered for `MEDIUM_IDENTIFIABILITY_NOT_SUPPORTED`; for `MEDIUM_IDENTIFIABILITY_TENTATIVE`, only a design note is allowed.

## Output Files
EXPERIMENTAL FACT: Full quantitative tables are written under `outputs/bnd_medium_identifiability_preflight_iui3_20260825/` and are intentionally not committed.
