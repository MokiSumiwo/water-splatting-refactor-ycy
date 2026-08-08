# Dewatering / SeaFree-Factor Audit - 2026-08-08

This note records code facts, experimental facts, quantitative conclusions, reasonable inferences, and unverified hypotheses for the WaterSplatting dewatering / intrinsic appearance calibration line. It does not make subjective visual-quality claims.

## References

- WaterSplatting repository: `/mnt/new/home_old/ycy/water-splatting-refactor`
- WaterSplatting branch at start: `research/gmvc-medium-calibration`
- WaterSplatting HEAD at start: `347359bd04ae187344f02112bdf2859aac492d99`
- SeaFree-GS reference repository: `/mnt/new/home_old/ycy/reference_repos/SeaFree-GS`
- SeaFree-GS reference commit: `7797e97dae831029ac89ae9f37b3c3d69ec2cf6c`
- SeaFree-GS status at start: clean
- Source audit note: `research_notes/SEAFREE_WATERSPLATTING_IMPLEMENTATION_AUDIT_2026-08-08.md`

## Code Fact

- Current WaterSplatting D010 direct attenuation is `T_D = exp(-gamma_D * beta_D * d)`.
- Current D010 implementation scales only direct attenuation via `direct_optical_depth_scale`; it does not scale `medium_bs`, backscatter exponent, `medium_rgb`, or `B_inf`.
- Current full-SH clear-object image-space proxy is `outputs["J_gaussian_raw"]`: alpha-composited active SH=3 Gaussian appearance without direct medium attenuation or backscatter.
- `direct_object_signal` is `outputs["rgb_object"]`: the attenuated direct object component `J*T_D`; it is not a fully dewatered image.
- SH0 remains deferred in this round.

## Stage 0 - Source Audit

### Code Fact

- The SeaFree-GS source audit was completed before starting new SeaFree-factor experiments.
- The audit covers LOS distance scaling, intrinsic boundedness, foreground-aware content loss, background-water supervision, coarse depth loss, and SH capacity.

### Experimental Fact

- No new SeaFree-inspired training was started before the source audit.

### Quantitative Conclusion

- Not applicable; this stage is a source-code audit.

### Reasonable Inference

- SeaFree's stable intrinsic behavior cannot be attributed to a single code factor without controlled WaterSplatting-native experiments, because the public method combines SH0/sigmoid boundedness, `/10` LOS scaling, pseudo-depth foreground/background support, background supervision, and coarse depth loss.

### Unverified Hypothesis

- In WaterSplatting SH3, remaining pure-J instability after D010-SWITCH may be caused by one or more of: Gaussian-level full-SH color overflow, foreground reconstruction gradient allocation, weak/incorrect medium anchoring, or geometry/depth compensation.

## Planned Stage Gates

- A0 reports `DEPTH_COMPENSATION_TRIGGER` from Gaussian-level projected LOS distance p90 change between D010-SCRATCH and D100-SCRATCH.
- A1 reports whether D010-SWITCH retains lower tau / lower J saturation with fixed Gaussian population.
- B1 tests weak current-view SH3 intrinsic bounds chosen by gradient matching at 1%, 5%, and 10% appearance-gradient ratios.
- B2 tests foreground-aware reconstruction only if needed, with one 5% gradient-matched auxiliary.
- B3 audits old background supervision gradient strength before deciding whether to rerun BG-G05 or switch to an integrated-medium target.
- B4 runs only if A0 triggers geometry/depth compensation and local Curasao pseudo-depth resources already exist.

## Stage A0 - Three-Path LOS / Geometry Audit

### Experimental Fact

- Script: `scripts/diagnostics/diagnose_dewater_los_geometry.py`
- Output JSON: `outputs/dewater_seafree_factor_20260808/los_geometry_audit.json`
- Output CSV: `outputs/dewater_seafree_factor_20260808/los_geometry_audit.csv`
- Depth contact sheet: `renders/dewater_seafree_factor_20260808/los_geometry_audit/contact_sheet_expected_depth.png`
- Compared checkpoints:
  - D100-SCRATCH nominal 15000 from `outputs/cross_scene_curasao_m1_seed42_15000/.../step-000014999.ckpt`
  - D010-SWITCH nominal 15000 from `outputs/dewater_d010_persistence_20260807/.../step-000015000.ckpt`
  - D010-SCRATCH nominal 15000 from `outputs/dewater_d010_scratch_20260807/.../step-000014999.ckpt`

| Run | gamma_D | Gaussian count | Gaussian LOS p90 | Gaussian LOS p99 | beta_eff mean | tau p90 | tau/beta_eff p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D100-SCRATCH | 1.0 | 1106714 | 3.804852 | 5.742488 | 0.531716 | 2.503527 | 5.603681 |
| D010-SWITCH | 0.1 | 931117 | 3.799549 | 5.685966 | 0.422636 | 1.993455 | 5.375017 |
| D010-SCRATCH | 0.1 | 1114093 | 3.846989 | 5.812272 | 0.548262 | 3.414756 | 7.690711 |

### Quantitative Conclusion

- `distance_p90_change_scratch = +0.011075`.
- `DEPTH_COMPENSATION_TRIGGER = FALSE` under the predefined Gaussian-level direct LOS p90 gate.
- D010-SCRATCH's higher image-space tau is not explained by a >=20% increase in the Gaussian-level projected LOS depth p90 that is passed to the CUDA direct attenuation expression.

### Reasonable Inference

- D010-SCRATCH still shows high image-space `tau/beta_eff` p90, but the direct Gaussian LOS p90 audit does not trigger a coarse-depth training branch under the predefined rule.

### Unverified Hypothesis

- Remaining D010-SCRATCH tau differences may involve image-space accumulation/depth distribution, visibility weighting, or coefficient distribution rather than a simple global increase in projected Gaussian LOS p90.

## Stage A1 - NOREFINE Causal Control

### Code Fact

- Added default-off config `disable_population_refinement=False`.
- When enabled, `refinement_after()` skips split, duplicate, cull, and prune operations while leaving normal optimization of means, scales, quats, opacity, features, and medium parameters active.
- In the 10k->15k continuation interval, the existing opacity reset condition is not active because `stop_split_at=10000`.

### Experimental Fact

- Script: `scripts/experiments/dewater_d010_no_refine_control.sh`
- Summary JSON: `outputs/dewater_seafree_factor_20260808/no_refine_summary.json`
- Summary CSV: `outputs/dewater_seafree_factor_20260808/no_refine_summary.csv`
- Both runs resumed the same Curasao M1 step-10000 checkpoint.
- Gaussian count check:
  - NR-D100 step 14000: 1140794
  - NR-D100 step 15000: 1140794
  - NR-D010 step 14000: 1140794
  - NR-D010 step 15000: 1140794

| Run | PSNR | SSIM | LPIPS | beta_eff | tau p90 | P(T<0.1) | P(J>1) | J p99 | P(c>1) | c p99 | Gaussian count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NR-D100 | 32.353214 | 0.956768 | 0.107758 | 0.553056 | 2.606315 | 0.165954 | 0.052468 | 2.526107 | 0.085145 | 2.513475 | 1140794 |
| NR-D010 | 32.274242 | 0.957137 | 0.108555 | 0.381845 | 1.624880 | 0.000297 | 0.026414 | 1.743646 | 0.060779 | 2.225021 | 1140794 |

### Quantitative Conclusion

- Relative to NR-D100:
  - `Delta PSNR = -0.078972`
  - `Delta SSIM = +0.000368`
  - `Delta LPIPS = +0.000797`
  - `tau_p90_relative_reduction = 37.656%`
  - `P(J>1)_relative_reduction = 49.658%`
  - `J_p99_relative_reduction = 30.975%`
- RGB safety passed.
- Tau gate passed.
- J gate passed.
- `D010_RECALIBRATION_CAUSAL_SIGNAL = TRUE`.
- `STRUCTURE_COUPLED = FALSE` under this fixed-population control.

### Reasonable Inference

- D010 late-stage recalibration has an independent lower-tau/lower-J effect even when post-10k Gaussian population-changing operations are disabled.

### Unverified Hypothesis

- This does not prove the resulting optical depth is physically correct; it only shows a persistent lower-optical-depth / reduced intrinsic-saturation solution under the fixed-population control.

## Stage B1 - SH3 Soft Intrinsic Bound

### Code Fact

- Added default-off config fields:
  - `intrinsic_bound_lambda: float = 0.0`
  - `intrinsic_bound_visible_only: bool = True`
- The loss is applied to `outputs["gaussian_view_rgb"]`, the current-view full-SH evaluated Gaussian colors from `features_dc + features_rest`, not to `features_dc` alone and not to image-space `J_gaussian_raw`.
- The default visible support is `outputs["gaussian_visible_mask"]`, defined by projected Gaussian radii greater than zero and detached before weighting.
- `diagnose_dewater_optical_depth.py` now reports Gaussian-level current-view color statistics `P(c<0)`, `P(c>1)`, `P(c>1.5)`, `P(c>2)`, `c_p95`, and `c_p99`.

### Experimental Fact

- Gradient audit JSON: `outputs/dewater_seafree_factor_20260808/loss_gradient_audit.json`
- Gradient audit CSV: `outputs/dewater_seafree_factor_20260808/loss_gradient_audit.csv`
- Training script: `scripts/experiments/dewater_intrinsic_bound.sh`
- Summary JSON: `outputs/dewater_seafree_factor_20260808/intrinsic_bound_summary.json`
- Summary CSV: `outputs/dewater_seafree_factor_20260808/intrinsic_bound_summary.csv`
- Start checkpoint for all B1 runs: D010-SWITCH step 13000.
- End checkpoint for all B1 runs: step 15000.
- `gamma_D=0.1`, SH=3, GMVC off, foreground-aware loss off, background supervision off.
- No SH0, hard clamp, or sigmoid renderer change was used.

Gradient-matched weights:

| Run | lambda | Initial gradient ratio |
| --- | ---: | ---: |
| IB-G01 | 0.124520 | 0.010000 |
| IB-G05 | 0.622599 | 0.050000 |
| IB-G10 | 1.245199 | 0.100000 |

Step-15000 results:

| Run | PSNR | SSIM | LPIPS | beta_eff | tau p90 | P(T<0.1) | P(J>1) | J p99 | P(c>1) | c p99 | Gaussian count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D010-SWITCH | 32.338216 | 0.956792 | 0.108534 | 0.422636 | 1.991694 | 0.023508 | 0.029077 | 1.841920 | 0.050511 | 2.055816 | 931117 |
| IB-G01 | 32.254855 | 0.956680 | 0.108757 | 0.393872 | 1.922905 | 0.011582 | 0.025736 | 1.628856 | 0.037756 | 1.331572 | 927598 |
| IB-G05 | 32.220921 | 0.956317 | 0.109336 | 0.360030 | 1.822818 | 0.002793 | 0.021957 | 1.434686 | 0.022357 | 1.070015 | 923681 |
| IB-G10 | 32.190111 | 0.956167 | 0.109670 | 0.345470 | 1.768230 | 0.001725 | 0.020327 | 1.357434 | 0.017198 | 1.030857 | 921830 |

### Quantitative Conclusion

- IB-G01 passed RGB safety and tau-not-worse gate, but did not reach either BOUND_PASS J sub-gate:
  - `P(J>1)` relative reduction: 11.491%
  - `J_p99` relative reduction: 11.567%
- IB-G05 reduced `J_p99` by 22.109% but failed RGB safety under the predefined `Delta PSNR >= -0.10` gate:
  - `Delta PSNR = -0.117295`
- IB-G10 met both J sub-gates but failed RGB safety:
  - `Delta PSNR = -0.148105`
  - `P(J>1)` relative reduction: 30.093%
  - `J_p99` relative reduction: 26.303%
- `BOUND_PASS = FALSE` for IB-G01, IB-G05, and IB-G10.
- `STRONG_BOUND_CANDIDATE = FALSE` for all B1 runs.

### Reasonable Inference

- Current-view SH3 color bounding does reduce both image-space `J_gaussian_raw` saturation statistics and Gaussian-level `c_i(v)` overflow statistics as the gradient ratio increases.
- Under the predefined B1 gate, this single factor did not produce a candidate that is both RGB-safe and sufficiently improves the two image-space J criteria.

### Unverified Hypothesis

- A different bound form or schedule could change this tradeoff, but this round stops the bounded-intrinsic direction instead of sweeping larger ratios or switching to hard sigmoid/SH0.

## Stage B2 - Foreground-Aware Reconstruction

### Code Fact

- Added default-off config fields:
  - `foreground_aware_weighting_enabled: bool = False`
  - `foreground_aware_weighting_lambda: float = 0.0`
  - `foreground_aware_accumulation_threshold: float = 0.05`
  - `foreground_aware_weight_epsilon: float = 1e-3`
  - `foreground_aware_weight_cap: float = -1.0`
- The implemented training-only support is `outputs["accumulation"].detach() > foreground_aware_accumulation_threshold`.
- The auxiliary, when enabled, is `sum(M_fg * w * abs(pred - GT)) / sum(M_fg)` with `w = 1 / (pred.detach() + eps)` and optional fixed cap.

### Experimental Fact

- No FAW training was run.
- No-update gradient audit found:
  - raw `foreground_aware_weighted_l1_raw = 0.391353`
  - recommended `FAW-G05 lambda = 0.013724`
  - initial gradient ratio at this lambda: 0.050000
- Foreground mask coverage from the train-set audit:
  - mean: 0.979033
  - p10: 0.957627
  - p50: 0.981537
  - p90: 0.995281
- Mask contact sheet: `renders/dewater_seafree_factor_20260808/loss_gradient_audit/foreground_masks/contact_sheet_foreground_masks.png`

### Quantitative Conclusion

- `FAW-G05 = NOT_RUN_MASK_COVERAGE_ABNORMAL`.
- The mean coverage exceeded the predefined `>95%` abnormal threshold, so foreground-aware training was not started.

### Reasonable Inference

- The current renderer-accumulation foreground support is too broad for the planned SeaFree-style foreground gradient allocation test without an additional mask audit or different foreground-support definition.

### Unverified Hypothesis

- A more selective foreground mask may still be useful, but it was not introduced in this round to avoid adding another uncontrolled factor.

## Stage B3 - Background Supervision Audit and Integrated-Medium Target

### Code Fact

- Existing old BG supervision target is `outputs["medium_rgb"]` on a detached background-water mask with inverse-intensity weighting.
- Added default-off config `lambda_background_finite_medium_render: float = 0.0`.
- When enabled, this adds `background_finite_medium_render_loss = lambda * masked_rgb_l1_loss(outputs["rgb_medium_finite"], GT, M_water)`.
- The new BGI target uses a renderer output that contributes to the final RGB as the finite-distance medium component; it does not supervise `B_inf`/ambient color directly and does not change renderer physics.
- Diagnostics now report background residuals for:
  - `medium_rgb`
  - `rgb_medium_total`
  - `rgb_medium_finite`

### Experimental Fact

- Old BG gradient audit at D010 step 13000:
  - `old_bg_lambda_0p01_vs_rgb_medium = 0.053831`
  - recommended `BG-G05 lambda = 0.009288`
  - recommended `BGI-FINITE-G05 lambda = 0.037249`
- Because the old `lambda=0.01` BG gradient ratio was already above 5%, BG-G05 was not trained.
- Training script: `scripts/experiments/dewater_background_supervision_v2.sh`
- Summary JSON: `outputs/dewater_seafree_factor_20260808/background_supervision_summary.json`
- Summary CSV: `outputs/dewater_seafree_factor_20260808/background_supervision_summary.csv`
- BGI-G05 resumed D010-SWITCH step 13000 and continued to step 15000 with:
  - `gamma_D=0.1`
  - SH=3
  - GMVC off
  - intrinsic bound off
  - foreground-aware weighting off
  - `lambda_background_finite_medium_render=0.037249`
- Background mask source: `common_masks/dewater_curasao_m1_step10000_train_background_water_20260807`, key `water`.
- Eval background mask coverage:
  - mean: 0.116595
  - p10: 0.081321
  - p50: 0.104777
  - p90: 0.156596

Step-15000 comparison:

| Run | PSNR | SSIM | LPIPS | beta_eff | tau p90 | P(T<0.1) | P(J>1) | J p99 | bg medium L1 | bg finite-medium L1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D010-SWITCH | 32.338216 | 0.956792 | 0.108534 | 0.422636 | 1.991694 | 0.023508 | 0.029077 | 1.841920 | 0.092063 | 0.285010 |
| BGI-G05 | 32.389868 | 0.956943 | 0.108369 | 0.432324 | 2.028900 | 0.045005 | 0.030118 | 1.869544 | 0.091057 | 0.274196 |

### Quantitative Conclusion

- `OLD_BG_WAS_TOO_WEAK = FALSE`.
- `OLD_BG_TARGET_OR_MASK_PROBLEM = TRUE` under the predefined rule, because the old BG gradient ratio was already >=5% while the historical BG010 run did not materially change tau/J.
- BGI-G05 RGB safety passed relative to D010-SWITCH:
  - `Delta PSNR = +0.051652`
  - `Delta SSIM = +0.000150`
  - `Delta LPIPS = -0.000166`
- BGI-G05 did not improve tau/J proxies:
  - `tau_p90_relative_reduction = -1.868%`
  - `P(J>1)_relative_reduction = -3.581%`
  - `J_p99_relative_reduction = -1.500%`
  - `P(T<0.1)` increased from 0.023508 to 0.045005.
- Background residual changes were small:
  - `background_medium_l1` relative reduction: 1.092%
  - `background_integrated_medium_finite_l1` relative reduction: 3.794%
  - `background_integrated_medium_total_l1` relative reduction: 5.232%
- `BGI_PASS = FALSE` for direct attenuation / intrinsic compensation.

### Reasonable Inference

- Under this mask and finite-medium target, background supervision mildly changes background residuals and RGB metrics but does not reduce the direct optical-depth / intrinsic-saturation proxies.

### Unverified Hypothesis

- The background mask semantics or target may still differ structurally from SeaFree's pseudo-depth-supported background formulation; this was not resolved by increasing the old BG weight or by the finite-medium BGI target tested here.

## Stage B4 - Coarse Geometry / Depth Constraint

### Experimental Fact

- No depth training was run.
- A0 reported `DEPTH_COMPENSATION_TRIGGER = FALSE`.

### Quantitative Conclusion

- `DEPTH-G05 = NOT_TRIGGERED`.

### Reasonable Inference

- The predefined direct LOS p90 gate did not justify adding a pseudo-depth/depth-correlation factor in this round.

### Unverified Hypothesis

- Geometry/depth could still affect image-space tau through mechanisms not captured by Gaussian-level LOS p90, but this round did not introduce pseudo-depth resources or depth losses.

## Combination

### Experimental Fact

- No COMBO-1 run was started.
- B1 had no `BOUND_PASS`, B2 was skipped due abnormal mask coverage, B3 had no tau/J pass, and B4 was not triggered.

### Quantitative Conclusion

- `COMBO-1 = NOT_TRIGGERED`.

## Final Unified Table

### Experimental Fact

- Final summary JSON: `outputs/dewater_seafree_factor_20260808/final_candidate_summary.json`
- Final summary CSV: `outputs/dewater_seafree_factor_20260808/final_candidate_summary.csv`

| Run | PSNR | SSIM | LPIPS | LOS p90 | beta_eff | tau p90 | P(T<0.1) | P(J>1) | P(J>1.5) | J p99 | P(c>1) | c p99 | Gaussian count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D100-SCRATCH | 32.165164 | 0.956004 | 0.108141 | 3.804852 | 0.531716 | 2.504676 | 0.143831 | 0.044918 | 0.024378 | 2.350653 | 0.074724 | 2.413399 | 1106714 |
| D010-SWITCH | 32.338216 | 0.956792 | 0.108534 | 3.799549 | 0.422636 | 1.991694 | 0.023508 | 0.029077 | 0.015934 | 1.841920 | 0.050511 | 2.055816 | 931117 |
| NR-D100 | 32.353214 | 0.956768 | 0.107758 | not in summary | 0.553056 | 2.606315 | 0.165954 | 0.052468 | 0.027311 | 2.526107 | 0.085145 | 2.513475 | 1140794 |
| NR-D010 | 32.274242 | 0.957137 | 0.108555 | not in summary | 0.381845 | 1.624880 | 0.000297 | 0.026414 | 0.014180 | 1.743646 | 0.060779 | 2.225021 | 1140794 |
| IB-G01 | 32.254855 | 0.956680 | 0.108757 | not in summary | 0.393872 | 1.922905 | 0.011582 | 0.025736 | 0.012578 | 1.628856 | 0.037756 | 1.331572 | 927598 |
| IB-G05 | 32.220921 | 0.956317 | 0.109336 | not in summary | 0.360030 | 1.822818 | 0.002793 | 0.021957 | 0.008215 | 1.434686 | 0.022357 | 1.070015 | 923681 |
| IB-G10 | 32.190111 | 0.956167 | 0.109670 | not in summary | 0.345470 | 1.768230 | 0.001725 | 0.020327 | 0.005970 | 1.357434 | 0.017198 | 1.030857 | 921830 |
| BGI-G05 | 32.389868 | 0.956943 | 0.108369 | not in summary | 0.432324 | 2.028900 | 0.045005 | 0.030118 | 0.016542 | 1.869544 | 0.050608 | 2.058612 | 931090 |
| FAW-G05 | NOT_TRIGGERED | | | | | | | | | | | | |
| DEPTH-G05 | NOT_TRIGGERED | | | | | | | | | | | | |
| COMBO-1 | NOT_TRIGGERED | | | | | | | | | | | | |

### Quantitative Conclusion

- No post-D010 single factor passed the predefined strong dewatering candidate rule.
- Best RGB-safe intrinsic-bound endpoint by this round's gate was IB-G01, but it did not provide the required extra J reductions relative to D010-SWITCH.
- The run with largest J-statistic reductions among B1 endpoints was IB-G10, but it failed the predefined RGB safety gate.
- Final minimal effective mechanism for this round remains `D010-SWITCH` only, not `D010 + bound`, `D010 + foreground weighting`, `D010 + medium supervision`, or a combination.
- New four-scene fixed validation is not triggered by the post-D010 factors in this round.

### Reasonable Inference

- D010 late-stage direct-attenuation recalibration remains the only factor in this audit that passed the defined lower-tau/lower-J and RGB-safety criteria.
- SH3 bound has a measurable monotonic effect on Gaussian-level and image-space intrinsic bounds, but the tested gradient ratios expose an RGB/intrinsic tradeoff rather than a passing candidate.

### Unverified Hypothesis

- A future candidate may require a different boundedness schedule, a better foreground/background support definition, or a different medium supervision target. This round does not justify adding those changes without a new controlled design.

## Visual Assets

### Experimental Fact

- Final factor comparison directory: `renders/dewater_seafree_factor_20260808/final_factor_comparison_step_15000/`
- Manifest JSON: `renders/dewater_seafree_factor_20260808/final_factor_comparison_step_15000/manifest.json`
- Manifest CSV: `renders/dewater_seafree_factor_20260808/final_factor_comparison_step_15000/manifest.csv`
- Visual index: `renders/dewater_seafree_factor_20260808/final_factor_comparison_step_15000/VISUAL_COMPARE_INDEX.md`
- Runs included: D100-SCRATCH, D010-SWITCH, IB-G01, IB-G05, IB-G10, BGI-G05.
- View IDs: 0, 1, 2.
- Camera IDs: 0, 1, 2.
- Display logic: reused diagnostic PNG mappings; no auto exposure, no white balance, no manual gamma, no histogram equalization, and no per-image normalization.

Contact sheets:

| Output | Path |
| --- | --- |
| underwater | `renders/dewater_seafree_factor_20260808/final_factor_comparison_step_15000/contact_sheet_underwater_factor_candidates.png` |
| direct object signal | `renders/dewater_seafree_factor_20260808/final_factor_comparison_step_15000/contact_sheet_direct_object_signal_factor_candidates.png` |
| clear clamp01 | `renders/dewater_seafree_factor_20260808/final_factor_comparison_step_15000/contact_sheet_clear_clamp01_factor_candidates.png` |
| WS-tonemap clear | `renders/dewater_seafree_factor_20260808/final_factor_comparison_step_15000/contact_sheet_clear_ws_tonemap_factor_candidates.png` |
| transmission | `renders/dewater_seafree_factor_20260808/final_factor_comparison_step_15000/contact_sheet_transmission_factor_candidates.png` |
| tau_D | `renders/dewater_seafree_factor_20260808/final_factor_comparison_step_15000/contact_sheet_tau_d_factor_candidates.png` |
| D100 alpha sweep | `renders/dewater_seafree_factor_20260808/final_factor_comparison_step_15000/contact_sheet_alpha_sweep_d100_scratch.png` |
| D010 alpha sweep | `renders/dewater_seafree_factor_20260808/final_factor_comparison_step_15000/contact_sheet_alpha_sweep_d010_switch.png` |
| IB-G01 alpha sweep | `renders/dewater_seafree_factor_20260808/final_factor_comparison_step_15000/contact_sheet_alpha_sweep_ib_g01.png` |
| IB-G05 alpha sweep | `renders/dewater_seafree_factor_20260808/final_factor_comparison_step_15000/contact_sheet_alpha_sweep_ib_g05.png` |
| IB-G10 alpha sweep | `renders/dewater_seafree_factor_20260808/final_factor_comparison_step_15000/contact_sheet_alpha_sweep_ib_g10.png` |
| BGI-G05 alpha sweep | `renders/dewater_seafree_factor_20260808/final_factor_comparison_step_15000/contact_sheet_alpha_sweep_bgi_g05.png` |

### Quantitative Conclusion

- Visual assets were generated for external/manual review only.
- No subjective visual-quality conclusion was made in this note.
