# Dewatering / Direct Optical-Depth Calibration Experiment Report - 2026-08-07

This report is independent of the GMVC-V3 line. It records code facts, experiment facts, quantitative gates, and bounded mechanism inferences only. It does not make subjective visual-quality claims.

## Code Facts

- Baseline renderer decomposition used in this audit:
  - `underwater_rgb`: current renderer underwater prediction, `outputs["pred_image"]` / `outputs["rgb"]`, composed from attenuated object signal and medium/backscatter/tail terms under the existing WaterSplatting renderer.
  - `direct_object_signal`: `outputs["rgb_object"]`, the object branch after direct medium attenuation.
  - `clear_object_fullsh_raw`: `outputs["J_gaussian_raw"]`, alpha-composited current active SH=3 Gaussian colors rendered without medium direct attenuation and without backscatter. This is an image-space full-SH clear-object / intrinsic proxy, not a single-surface-point ground-truth clear image.
  - `clear_object_fullsh_clamp01`: `clear_object_fullsh_raw` clamped to `[0,1]` for display.
  - `clear_object_fullsh_ws_tonemap`: `clear_object_fullsh_raw / (clear_object_fullsh_raw + 1)`.
  - `gmvc_J_proxy_raw`: `outputs["J_proxy_raw"]` from the clear-proxy branch when that branch is enabled. It is unavailable for normal GMVC-off M1 checkpoints and was not forced on for this experiment.
- D0 support mask: renderer object support `outputs["accumulation"] > 0.01`; this same mask is used for beta/tau/T/J statistics and correlations.
- Direct optical-depth scale A:
  - Added default-off config `direct_optical_depth_scale = 1.0`.
  - The effective direct coefficient is `medium_attn_effective = direct_optical_depth_scale * medium_attn_raw`.
  - Only direct attenuation receives this scale. `medium_bs`, backscatter exponent, `medium_rgb`, and `B_inf` are not scaled.
  - Recorded values include `beta_D_raw`, `beta_D_effective`, `tau_D_raw`, `tau_D_effective`, and `T_D_effective`.
- Medium background supervision B:
  - Added default-off configs `medium_background_supervision_enabled=False`, `medium_background_supervision_lambda=0.0`, `medium_background_supervision_exclude_boundary=True`, and `medium_background_supervision_hit_exclusion_threshold=-1.0`.
  - Loss when enabled: `sum(M * abs(medium_rgb - GT) / (medium_rgb.detach() + 1e-3)) / sum(M)`.
  - `M` is detached, training-only, and loaded from fixed renderer-derived masks. It is not used at inference and does not modify the renderer.
- SeaFree-style distance normalization note: a fixed `distance / 10` factor cannot be interpreted as a guaranteed 10x attenuation reduction when `beta_D` is learnable. This experiment records whether `beta_D_raw` increases enough to compensate the imposed direct scale.

## D0 No-Training Audit

- Scene: Curasao.
- Checkpoint: `outputs/cross_scene_curasao_m1_seed42_15000/water-splatting/cross_scene_curasao_m1_seed42_15000_20260730_cross_scene/nerfstudio_models/step-000014999.ckpt`.
- Config: `outputs/cross_scene_curasao_m1_seed42_15000/water-splatting/cross_scene_curasao_m1_seed42_15000_20260730_cross_scene/config.yml`.
- Eval views: 3.
- Visual output root: `renders/dewater_optical_depth_20260807/`.
- D0 generated `underwater_rgb`, `direct_object_signal`, `clear_object_fullsh_raw_display`, `clear_object_fullsh_clamp01`, `clear_object_fullsh_ws_tonemap`, transmission, tau visualization, medium_rgb, backscatter, and alpha sweeps for `alpha = 0/0.25/0.5/0.75/1`.
- `gmvc_J_proxy_raw`: unavailable under GMVC-off M1 and not generated.

| Quantity | r | g | b |
| --- | ---: | ---: | ---: |
| beta_D_raw mean | 0.514099 | 0.541529 | 0.539522 |
| tau_D_effective p90 | 2.588030 | 2.492570 | 2.433428 |
| T_D_effective mean | 0.244431 | 0.229031 | 0.231884 |
| P(T<0.30) | 0.676330 | 0.807175 | 0.806559 |
| P(T<0.20) | 0.278485 | 0.310131 | 0.319124 |
| P(T<0.10) | 0.153874 | 0.145188 | 0.132431 |
| P(T<0.05) | 0.020079 | 0.007765 | 0.005981 |
| clear_object_fullsh_raw p99 | 2.141917 | 2.465533 | 2.444510 |
| P(J>1.0) | 0.043006 | 0.047114 | 0.044634 |
| P(J>1.5) | 0.021995 | 0.026031 | 0.025110 |
| P(J>2.0) | 0.012363 | 0.016768 | 0.016005 |
| Pearson corr(tau_D, J) | -0.110245 | 0.019500 | 0.097962 |
| Spearman corr(tau_D, J) | -0.151384 | -0.010977 | 0.072460 |

Quantitative conclusion: the full-SH clear-object tensor itself has nonzero `J>1` and `J>2` mass. The issue is therefore not reducible to the historical GMVC/DC `J_proxy_raw` definition. The D0 per-pixel tau/J correlations are weak and mixed in sign, so D0 alone does not prove the optical-depth causal chain.

## A Direct Optical-Depth Scale

- All runs resume from the same Curasao M1 step-10000 checkpoint:
  `outputs/cross_scene_curasao_m1_seed42_15000/water-splatting/cross_scene_curasao_m1_seed42_15000_20260730_cross_scene/nerfstudio_models/step-000010000.ckpt`.
- Training: step 10000 to 13000, SH=3, GMVC off, original RGB reconstruction loss and original optimizer/scheduler/scaler/densification/refinement.
- Checkpoints saved/evaluated at 11000, 12000, 13000.
- Summary: `outputs/dewater_optical_depth_20260807/direct_optical_depth_sweep_summary.json`.

Final step 13000:

| Run | gamma_D | PSNR | SSIM | LPIPS | beta_raw mean | beta_eff mean | compensation ratio | expected ratio | tau p90 | P(T<0.1) | P(J>1) | J p99 | A gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| D100 | 1.00 | 32.223931 | 0.956911 | 0.107947 | 0.538801 | 0.538801 | 1.0000 | 1.0 | 2.531823 | 0.148602 | 0.044772 | 2.326760 | false |
| D050 | 0.50 | 32.183111 | 0.956724 | 0.107739 | 1.036769 | 0.518384 | 1.9242 | 2.0 | 2.541450 | 0.146124 | 0.041627 | 2.221376 | false |
| D025 | 0.25 | 32.300425 | 0.957018 | 0.108065 | 1.897625 | 0.474406 | 3.5219 | 4.0 | 2.297814 | 0.099340 | 0.035301 | 2.051046 | false |
| D010 | 0.10 | 32.436670 | 0.957246 | 0.108737 | 3.839738 | 0.383974 | 7.1264 | 10.0 | 1.733984 | 0.001710 | 0.025737 | 1.688977 | true |

D010 gate details relative to D100:

- RGB safety: passed (`dPSNR=+0.212739`, `dSSIM=+0.000335`, `dLPIPS=+0.000789`).
- Effective optical-depth gate A1: passed (`tau p90` drop `31.51%`, `P(T<0.1)` drop `98.85%`).
- J saturation gate A2: passed (`P(J>1)` drop `42.52%`, `J p99` drop `27.41%`).
- Compensation check: `beta_D_raw` increased by `7.1264x`, below the full compensation value `10x`; the effective beta/tau did not return to D100.

Quantitative conclusion: Direct optical-depth scaling was not fully absorbed by learnable beta on Curasao. D010 is the only A run satisfying the predefined A mechanism gate.

## B Medium Background Direct Supervision

- Mask source: `common_masks/dewater_curasao_m1_step10000_train_background_water_20260807/`.
- Mask definition: far renderer depth + low accumulation + low full-SH clear-object luma, eroded, boundary/hit excluded, saved as detached train-only masks.
- Train cameras: 18.
- Coverage: mean `0.072180`, p10 `0.034624`, p50 `0.067527`, p90 `0.107342`, min `0.027106`, max `0.169551`.
- Coverage gate `[0.02, 0.80]`: passed.
- Summary: `outputs/dewater_optical_depth_20260807/medium_background_supervision_summary.json`.

Final step 13000:

| Run | lambda_bg | PSNR | SSIM | LPIPS | background_medium_l1 | residual drop | tau p90 | tau drop | P(J>1) | J drop | B gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BG000 | 0.00 | 32.220703 | 0.956908 | 0.107955 | 0.022687 | 0.0000 | 2.532363 | 0.0000 | 0.044786 | 0.0000 | false |
| BG010 | 0.01 | 32.313914 | 0.957085 | 0.107726 | 0.021884 | 0.0354 | 2.506835 | 0.0101 | 0.045429 | -0.0143 | false |

Quantitative conclusion: BG010 passed RGB safety but failed the background residual gate (`3.54%` drop vs required `20%`) and failed the decomposition proxy gate (`tau p90` drop `1.01%`, `P(J>1)` did not decrease). Under this first setting, medium direct supervision did not provide evidence that tau/J saturation naturally decreases without direct beta scaling.

## AB Single Combination

- Trigger rule: AB is allowed because A passed with D010. B did not pass, so only one combination was run: D010 + BG010.
- Run: `AB_D010_BG010`, `gamma_D=0.10`, `lambda_bg=0.01`.
- Same Curasao M1 step-10000 resume and same 10000 to 13000 schedule.
- Summary: `outputs/dewater_optical_depth_20260807/ab_d010_bg010_summary.json`.
- Visual root: `renders/dewater_optical_depth_20260807/AB/AB_D010_BG010/`.

Final step 13000:

| Run | PSNR | SSIM | LPIPS | beta_raw mean | beta_eff mean | comp. ratio | tau p90 | tau drop vs D100 | P(T<0.1) | P(J>1) | J drop vs D100 | J p99 | bg_l1 | bg_l1 drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AB_D010_BG010 | 32.405120 | 0.957354 | 0.108492 | 3.940428 | 0.394043 | 7.3133 | 1.726770 | 0.3180 | 0.002248 | 0.026865 | 0.4000 | 1.730364 | 0.022054 | 0.0279 |

AB gate notes:

- RGB safety vs D100: passed (`dPSNR=+0.181189`, `dSSIM=+0.000443`, `dLPIPS=+0.000544`).
- A-like gate vs D100: passed.
- Background residual gate vs BG000: failed (`2.79%` drop vs required `20%`).
- Relative to D010, AB changed `tau p90` by `+0.42%` drop, `P(J>1)` by `-4.38%` drop, and `J p99` by `-2.45%` drop; this does not add a separate positive B mechanism signal.

Quantitative conclusion: AB preserves the direct-scale mechanism signal but does not make the background-supervision mechanism pass.

## C / SH3 Intrinsic Soft Bound

C was not run in this stage. D010 and AB already pass the predefined A/A-like mechanism gate at SH=3, while the tested BG010 supervision did not pass. Introducing a new full-SH soft-bound training variable would require a separate no-training or short audit to set `lambda * L_bound` near 1% of the main RGB loss; that calibration was deferred rather than mixed into the completed D0/A/B/AB evidence.

SH0 remains a deferred hypothesis. Historical experiments recorded RGB degradation with SH0, so SH degree was not changed in this branch.

## Synthetic Clean GT

No directly runnable WaterSplatting clean-pair synthetic dataset was found in the repository scan for this stage. The available Curasao-style directories contain real underwater images plus DepthAnything/stereo folders, not a clear-image GT pair pipeline. Synthetic clear-GT evaluation is recorded as not evaluated.

## Visual Output Paths

- D0 per-view: `renders/dewater_optical_depth_20260807/Curasao/per_view/`.
- D0 contact sheets: `renders/dewater_optical_depth_20260807/Curasao/contact_sheets/`.
- A per-view/contact roots: `renders/dewater_optical_depth_20260807/A/{D100,D050,D025,D010}/step_{11000,12000,13000}/`.
- B eval roots: `renders/dewater_optical_depth_20260807/B/{BG000,BG010}/step_{11000,12000,13000}/eval/`.
- B train-background stats: `renders/dewater_optical_depth_20260807/B/{BG000,BG010}/step_{11000,12000,13000}/train_background/summary.json`.
- AB eval root: `renders/dewater_optical_depth_20260807/AB/AB_D010_BG010/step_13000/eval/`.
- AB step-13000 contact sheets:
  - `renders/dewater_optical_depth_20260807/AB/AB_D010_BG010/step_13000/eval/Curasao/contact_sheets/all_dewater_audit.png`
  - `renders/dewater_optical_depth_20260807/AB/AB_D010_BG010/step_13000/eval/Curasao/contact_sheets/all_medium_optical_depth.png`
  - `renders/dewater_optical_depth_20260807/AB/AB_D010_BG010/step_13000/eval/Curasao/contact_sheets/all_partial_deattenuation_alpha_sweep.png`
- Visual index: `renders/dewater_optical_depth_20260807/VISUAL_REVIEW_INDEX.md`.

## Bounded Answers to Current Questions

- Q1: The historical `J_proxy_raw` is not the only source of the issue. `gmvc_J_proxy_raw` is unavailable under GMVC-off M1, and true `clear_object_fullsh_raw` itself has nonzero `P(J>1/1.5/2)`.
- Q2: D0 pixel correlations alone are weak and mixed. The A intervention provides stronger causal evidence because lowering effective direct optical depth in D010 coincides with lower `P(J>1)` and lower `J p99` while passing RGB safety.
- Q3: Lowering `gamma_D` causes partial beta compensation, but D010 is not fully compensated (`7.1264x` raw beta increase vs `10x` full-compensation target). Effective optical depth decreases.
- Q4: BG010 does not show the required natural reduction in tau/J saturation under the first medium direct supervision setting.
- Q5: For this Curasao test, the positive mechanism signal is closer to direct optical-depth scaling than to the tested medium independent supervision. This is not a claim about the full SeaFree-GS method.
- Q6: At SH=3, D010 and AB reduce full-SH clear saturation metrics relative to D100 while passing RGB safety gates. No SH-degree reduction was used.
