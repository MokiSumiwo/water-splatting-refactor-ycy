# M1-OCMC-CAUSAL-IUI3

## CODE FACT
OCMC is implemented in `water_splatting/fields/medium_field.py` as a detached 9-D raw-medium projector on the camera-conditioned residual.
The current camera context is scene-normalized camera position, not a learned latent.
Implemented equations: `['z_full = f(dir, xy, camera_context)', 'z_base = f(dir, xy, zero_camera_context)', 'Delta_z_cam = z_full - z_base', 'delta_std = Delta_z_cam / scale if scale is available', 'Delta_projected = (delta_std @ P_obs.T) * scale', 'z_effective = z_base + Delta_z_cam + strength * (Delta_projected - Delta_z_cam)', 'with strength=1.0, z_effective = z_base + Delta_projected']`.

## CONFIG FACT
Both arms use `bounded_sh3`, SH degree 3, `medium_context_mode=dir_xy_camera`, `b_inf_mode=tied`, and `infinite_water_enabled=False`.
C0 keeps camera context and disables OCMC. C1 keeps the same camera context and enables OCMC with strength `1.0`.
Projector protocol: `LOW_FREQUENCY_PERIODIC_REFRESH_CURRENT_GENERAL` with refresh steps `[0, 5000, 10000]` and population `GENERAL`.

## EXPERIMENTAL FACT
Start-state equivalence: `True`.
Camera sequence match: `True`.
Outputs: `/mnt/new/home_old/ycy/water-splatting-refactor/outputs/m1_ocmc_causal_iui3_20260825`.

## QUANTITATIVE RESULT
Final train C1-C0: PSNR `-0.304683` dB, SSIM `-0.000208`, LPIPS `-0.000315`, MSE `0.00002213`.
Final eval C1-C0: PSNR `0.130078` dB, SSIM `-0.000974`, LPIPS `-0.000144`, MSE `-0.00007035`.
RGB safety classification: `RGB_IMPROVED`.
Mechanism checks: `{'weak_energy_fraction_mean_decreased_pairs': 6, 'weak_energy_fraction_mean_compared_pairs': 6, 'weak_projection_over_random_1over9_decreased_pairs': 6, 'weak_projection_over_random_1over9_compared_pairs': 6, 'suppressed_over_full_decreased_pairs': 12, 'suppressed_over_full_compared_pairs': 12, 'final_correct_context_utility_C0': 0.0002705680812390421, 'final_correct_context_utility_C1': 6.441651711394769e-05, 'correct_context_utility_preserved': False, 'final_remove_weak_delta_E_C0': 9.850494312328095e-05, 'final_remove_weak_delta_E_C1': 3.8089460099488548e-06, 'removable_low_observability_component_decreased': True, 'bounded_decomposition_safety_intact': True, 'camera_expressiveness_preserved': True}`.
Decomposition safety C1 P(J>1)=0: `True`.

## INFERENCE
OCMC classification: `OCMC_PARTIALLY_ACTIONABLE`.
Capacity-allocation classification: `CAMERA_CONTEXT_CAPACITY_ALLOCATION_TENTATIVE`.
No true-color, true-medium, or true-geometry claim is made.

## HYPOTHESIS
Next single experiment: One diagnostic to separate projector temporal mismatch from over-suppression of useful context, without a sweep.
