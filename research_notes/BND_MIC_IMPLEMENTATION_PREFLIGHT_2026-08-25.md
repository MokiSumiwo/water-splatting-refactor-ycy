# BND-MIC-IMPLEMENTATION-PREFLIGHT

## Mechanism
CONFIG FACT: Mechanism name `BND-MIC-BetaDVariance`.
INFERENCE: Phase A supported a stable low-observability BND medium direction dominated by beta_D, so this prototype targets beta_D raw contextual variance only.
CONFIG FACT: Equation `L_MIC = mean((z_beta_D - stopgrad(mean(z_beta_D)))^2)`.
CONFIG FACT: The bounded object representation, dir_xy_camera medium context, RGB loss, depth path, opacity, and densification logic are unchanged.

## Implementation
CODE FACT: `MediumFieldOutput.raw` carries the medium MLP pre-activation tensor.
CODE FACT: `WaterSplattingModelConfig.medium_identifiability_enabled` defaults to `False`; weight defaults to `0.0`.
CODE FACT: `medium_raw` is attached to outputs only when the flag is enabled.
CODE FACT: `get_loss_dict` adds `medium_identifiability_loss` only when enabled, nonzero weighted, and within the optional step schedule.

## Disabled-Path Equivalence
QUANTITATIVE RESULT: `{'run': 'BND', 'nominal_step': 5000, 'loaded_step': 5000, 'view_id': 'MTN_5906', 'disabled_repeat_max_abs_diffs': {'pred_image': 0.0, 'depth': 0.0, 'accumulation': 0.0, 'medium_rgb': 0.0, 'medium_bs': 0.0, 'medium_attn': 0.0, 'rgb_medium': 0.0}, 'enabled_zero_weight_forward_max_abs_diffs': {'pred_image': 0.0, 'depth': 0.0, 'accumulation': 0.0, 'medium_rgb': 0.0, 'medium_bs': 0.0, 'medium_attn': 0.0, 'rgb_medium': 0.0}, 'disabled_main_loss_max_abs_diff': 0.0, 'enabled_zero_weight_main_loss_max_abs_diff': 0.0, 'baseline_has_medium_raw': False, 'disabled_has_medium_raw': False, 'enabled_zero_weight_has_medium_raw': True, 'equivalence_pass': True}`.

## Coefficient Selection
QUANTITATIVE RESULT: selected lambda `1.5118506741538569` by gradient-scale rule, not PSNR.

## Gradient Pathway
QUANTITATIVE RESULT: object grad max `0.0`; medium branch grad `0.2900940813997351`.

## Smoke / Checkpoint
EXPERIMENTAL FACT: smoke rows `[{'relative_step': 0, 'absolute_step': 5000, 'view_id': 'MTN_5906', 'total_loss': 0.15007738769054413, 'main_loss': 0.09212450683116913, 'medium_identifiability_loss': 0.057952880859375, 'finite_loss': True, 'finite_gradients': True, 'medium_branch_grad_l2': 0.6334036432523413}, {'relative_step': 1, 'absolute_step': 5001, 'view_id': 'MTN_5898', 'total_loss': 0.15148809552192688, 'main_loss': 0.09631231427192688, 'medium_identifiability_loss': 0.05517578125, 'finite_loss': True, 'finite_gradients': True, 'medium_branch_grad_l2': 0.4528042875062496}]`.
EXPERIMENTAL FACT: checkpoint compatibility `{'old_bnd_checkpoint_loaded_with_disabled_flag': True, 'old_bnd_checkpoint_loaded_with_enabled_flag': True, 'new_state_checkpoint_path': 'outputs/bnd_mic_preflight_iui3_20260825/mic_smoke_model_state.pt', 'new_checkpoint_saved': True, 'new_checkpoint_reloaded': True, 'load_state_dict_returned': 'None', 'load_state_dict_note': 'WaterSplattingModel.load_state_dict overrides PyTorch and returns None; strict reload success is inferred from no exception.', 'missing_keys': [], 'unexpected_keys': [], 'checkpoint_compatibility_pass': True}`.

## Classification
INFERENCE: `MIC_IMPLEMENTATION_READY`.

## Next
HYPOTHESIS: The next formal experiment is one single-factor causal continuation: BND vs BND+MIC on IUI3 with matched start state, RNG, camera sequence, optimizer/scheduler, and densification; only MIC differs.
