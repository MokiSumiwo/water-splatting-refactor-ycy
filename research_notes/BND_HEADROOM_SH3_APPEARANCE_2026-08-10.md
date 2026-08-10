# BND Headroom SH3 Appearance Test

Date: 2026-08-10

Branch: `research/m1-bounded-intrinsic`

Start HEAD: `95919d153f9c5a5d8fad1983a12137a5a45255a0`

Experiment short name: `BND-HR`

Diagnostic outputs:

- Metrics: `outputs/bnd_hr_panama_20260810/`
- Visuals: `renders/bnd_hr_panama_20260810/`
- Visual index: `renders/bnd_hr_panama_20260810/VISUAL_COMPARE_INDEX.md`
- Summary: `outputs/bnd_hr_panama_20260810/bnd_hr_final_summary.json`

No subjective clear-image correctness judgment was made in this experiment.

## 1. Motivation

HYPOTHESIS:

Separating bounded base appearance from a Jacobian-matched asymmetric headroom SH residual may recover legal view-dependent fitting capacity in Panama without reopening the unbounded compensation route.

CODE FACT:

This experiment changes only `intrinsic_color_parameterization` from `bounded_sh3` to `bounded_headroom_sh3`. It keeps SH degree 3, classic rasterization, M1 medium settings, optimizer/LR, loss, densification, renderer, and water forward model unchanged.

## 2. Current Panama Evidence

EXPERIMENTAL FACT:

Panama M1 and BND-K1 at nominal step 15000:

| Run | PSNR | SSIM | LPIPS | tau p90 | J p99 | P(J>1) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 32.308910 | 0.949487 | 0.073979 | 1.769849 | 1.311801 | 0.037758 |
| BND-K1 | 31.498353 | 0.948783 | 0.075521 | 0.999069 | 0.838911 | 0.000000 |

QUANTITATIVE RESULT:

BND-K1 removes the high-J route under the bounded intrinsic parameterization but leaves a `-0.810558 dB` PSNR gap relative to M1 on Panama.

## 3. Why AA Was Informative But Not Accepted

EXPERIMENTAL FACT:

BND-AA is a secondary diagnostic reference only. It changed rasterization to antialiased and recovered `+0.304330 dB` over K1, but it is not the formal baseline for this appearance test.

CODE FACT:

BND-HR uses `rasterize_mode=classic`, not AA. No AA/HR combination was run.

## 4. Legacy SH Legal Residual Evidence

QUANTITATIVE RESULT:

Panama legacy SH residual energy reference:

| Metric | Value |
| --- | ---: |
| Legal SH energy fraction | 0.602457 |
| Overflow SH energy fraction | 0.150581 |
| Base-invalid SH energy fraction | 0.246963 |

QUANTITATIVE RESULT:

Legacy valid-to-valid headroom observations, aggregate over Panama eval views:

| Metric | Value |
| --- | ---: |
| Valid-to-valid channel observations | 9227231 |
| Positive residual fraction | 0.631655 |
| Negative residual fraction | 0.368340 |
| Positive residual energy fraction | 0.728482 |
| Negative residual energy fraction | 0.271518 |
| u_pos all p50 / p90 / p95 / p99 | 0.090500 / 0.393107 / 0.559745 / 0.859830 |
| u_neg all p50 / p90 / p95 / p99 | 0.108319 / 0.411697 / 0.562326 / 0.861382 |
| P(u_pos>0.75 / 0.90) | 0.020740 / 0.006814 |
| P(u_neg>0.75 / 0.90) | 0.020372 / 0.007020 |

INFERENCE:

The legacy SH residual contains substantial legal RGB-domain view-dependent variation. This supports testing an in-range representation, not re-enabling unbounded intrinsic RGB.

## 5. BND-v1 Full-Logit Sigmoid Limitation

CODE FACT:

BND-v1 uses one sigmoid over the active full-SH logit:

`c_BND1(v) = sigmoid(s_full(v))`

where:

`s_full(v) = s0 + r_SH(v)`

HYPOTHESIS:

When the base color is near a boundary, using one full-logit sigmoid may couple legal view-dependent residual fitting to sigmoid boundary pressure.

## 6. Exact Current SH Semantics Audit

CODE FACT:

Audited code paths:

- `water_splatting/fields/gaussian_appearance.py::compute_bounded_gaussian_colors`
- `water_splatting/fields/gaussian_appearance.py::compute_bounded_headroom_gaussian_colors`
- `water_splatting/water_splatting.py::WaterSplattingModel.get_outputs`
- `water_splatting/sh.py::spherical_harmonics`

CODE FACT:

The audited formulas are:

`s0 = spherical_harmonics(0, viewdirs, colors[:, :1, :])`

`s_full(v) = spherical_harmonics(active_sh_degree, viewdirs, cat(features_dc[:, None, :], features_rest))`

`r_SH(v) = s_full(v) - s0`

`c_BND1(v) = sigmoid(s_full(v))`

QUANTITATIVE RESULT:

`SH_LINEAR_DECOMPOSITION_CONFIRMED = true`.

## 7. BND-HR Formulation

CODE FACT:

Implemented in `water_splatting/fields/gaussian_appearance.py::compute_bounded_headroom_gaussian_colors`.

`c0 = sigmoid(s0)`

`r = s_full(v) - s0`

For `r >= 0`:

`c_HR = c0 + (1 - c0) * tanh(c0 * r)`

For `r < 0`:

`c_HR = c0 + c0 * tanh((1 - c0) * r)`

CODE FACT:

The implementation uses `torch.where(residual >= 0.0, positive_rgb, negative_rgb)` and does not use a ReLU residual split, residual gain, temperature, clamp-after-unbounded output, or renderer physics change.

## 8. Mathematical Boundedness

CODE FACT:

For `r >= 0`, `0 <= tanh(c0*r) < 1`, so `c0 <= c_HR < 1`.

For `r < 0`, `-1 < tanh((1-c0)*r) < 0`, so `0 < c_HR < c0`.

QUANTITATIVE RESULT:

At final nominal step 15000, canonical `P(J>1)=0.000000` for HR.

## 9. Zero-Residual Equivalence

CODE FACT:

At `r=0`, BND-HR gives `c_HR=c0`. BND-v1 also gives `sigmoid(s0+0)=c0`.

## 10. Jacobian Equivalence

CODE FACT:

At `r=0`, both positive and negative HR branches have `dc/dr = c0*(1-c0)`, matching BND-v1.

QUANTITATIVE RESULT:

The numeric initialization Jacobian audit passed:

| Parameter | K1 grad L2 | HR grad L2 | Relative L2 diff | Max abs diff |
| --- | ---: | ---: | ---: | ---: |
| features_dc | 0.002525006 | 0.002525007 | 3.277771e-07 | 6.984919e-10 |
| features_rest | 0.009778976 | 0.009778977 | 3.445624e-07 | 1.396984e-09 |

The audited loss was identical: `0.6226951479911804`.

## 11. Legacy Positive/Negative Headroom Audit

QUANTITATIVE RESULT:

Aggregate per-channel legacy valid-to-valid utilization:

| Channel | u_pos p90 | u_pos p99 | P(u_pos>0.75) | P(u_pos>0.90) | u_neg p90 | u_neg p99 | P(u_neg>0.75) | P(u_neg>0.90) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R | 0.368251 | 0.844242 | 0.018198 | 0.006010 | 0.428693 | 0.873623 | 0.022550 | 0.007821 |
| G | 0.403368 | 0.867121 | 0.021961 | 0.007270 | 0.401781 | 0.857802 | 0.019545 | 0.006682 |
| B | 0.406250 | 0.865723 | 0.021964 | 0.007129 | 0.402406 | 0.850986 | 0.018826 | 0.006482 |

EXPERIMENTAL FACT:

Brightness-Q5 and M1 high-J image masks were not mapped to per-Gaussian legacy observations in this audit. No cross-run or per-pixel Gaussian matching was forced.

## 12. Initialization Parameter Audit

QUANTITATIVE RESULT:

K1 and HR initialization matched exactly for audited parameters:

| Parameter | Shape | Max abs diff | Mean abs diff |
| --- | --- | ---: | ---: |
| means | [22501, 3] | 0.0 | 0.0 |
| scales | [22501, 3] | 0.0 | 0.0 |
| quats | [22501, 4] | 0.0 | 0.0 |
| opacities | [22501, 1] | 0.0 | 0.0 |
| features_dc | [22501, 3] | 0.0 | 0.0 |
| features_rest | [22501, 15, 3] | 0.0 | 0.0 |
| medium_mlp.tcnn_encoding.params | [6144] | 0.0 | 0.0 |

`INIT_PARAMETER_EQUIVALENCE = PASS`.

## 13. Initialization Forward Audit

QUANTITATIVE RESULT:

At audit step 3000 on view `MTN_1539`, K1 and HR max abs diff was `0.0` for:

- `gaussian_view_rgb`
- `pred_image`
- `direct_object_signal`
- `rgb_medium`
- `depth`
- `accumulation`
- `clear_object_fullsh_raw`
- `transmission`
- `tau_D`

`INIT_FORWARD_EQUIVALENCE = PASS`.

## 14. Initialization Gradient/Jacobian Audit

QUANTITATIVE RESULT:

Appearance gradients passed the registered tolerance:

- `features_dc relative_l2_diff = 3.277771e-07`, max abs diff `6.984919e-10`
- `features_rest relative_l2_diff = 3.445624e-07`, max abs diff `1.396984e-09`

Other audited parameter groups also had relative L2 differences below `3.1e-07`.

`INIT_APPEARANCE_JACOBIAN_EQUIVALENCE = PASS`.

## 15. Training Setup

EXPERIMENTAL FACT:

Runs:

| Run | Role | Status | Config |
| --- | --- | --- | --- |
| M1 | Reference M1 | Reused | `outputs/cross_scene_panama_m1_seed42_15000/water-splatting/cross_scene_panama_m1_seed42_15000_20260730_cross_scene/config.yml` |
| K1 | Formal bounded classic control | Reused | `outputs/dewater_bounded_sh3_cross_scene_20260808/dewater_bnd_cross_scene_panama_seed42_step0_to_15000/water-splatting/dewater_bnd_cross_scene_panama_seed42_step0_to_15000_20260808_bounded_sh3_cross_scene_full_panama_bnd_g1p00/config.yml` |
| AA | Secondary reference only | Reused | `outputs/bnd_aa_panama_20260810/panama_bnd_aa_seed42_step0_to_15000/water-splatting/20260810_bnd_aa/config.yml` |
| HR | BND-HR candidate | New 0->15k run | `outputs/bnd_hr_panama_20260810/panama_bnd_hr_seed42_step0_to_15000/water-splatting/20260810_bnd_hr/config.yml` |

CODE FACT:

HR config:

- `intrinsic_color_parameterization=bounded_headroom_sh3`
- `rasterize_mode=classic`
- `medium_context_mode=dir_xy_camera`
- `b_inf_mode=tied`
- `infinite_water_enabled=False`
- `sh_degree=3`
- `seed=42`
- `max_num_iterations=15000`

EXPERIMENTAL FACT:

The final nominal step 15000 loaded actual checkpoint step `14999`.

## 16. RGB Trajectory

QUANTITATIVE RESULT:

| Step | Run | PSNR | SSIM | LPIPS | MSE | Gaussians |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1000 | K1 | 22.563314 | 0.688374 | 0.436505 | 0.005637601 | 60112 |
| 1000 | HR | 22.844411 | 0.689354 | 0.433608 | 0.005292724 | 59676 |
| 3000 | K1 | 20.869651 | 0.597971 | 0.440031 | 0.008590163 | 566332 |
| 3000 | HR | 20.883020 | 0.597200 | 0.442577 | 0.008561076 | 566550 |
| 5000 | K1 | 26.459814 | 0.822805 | 0.252952 | 0.002286107 | 998668 |
| 5000 | HR | 26.562921 | 0.823077 | 0.255212 | 0.002227582 | 996212 |
| 8000 | K1 | 31.008972 | 0.943347 | 0.086439 | 0.000801345 | 1223420 |
| 8000 | HR | 31.122148 | 0.943605 | 0.087179 | 0.000779379 | 1216856 |
| 10000 | K1 | 31.154767 | 0.945652 | 0.082064 | 0.000776138 | 1219898 |
| 10000 | HR | 31.233619 | 0.945836 | 0.083304 | 0.000763412 | 1212655 |
| 13000 | K1 | 31.540089 | 0.949397 | 0.074719 | 0.000712338 | 1183679 |
| 13000 | HR | 31.555182 | 0.949265 | 0.076801 | 0.000710748 | 1177164 |
| 15000 | K1 | 31.498353 | 0.948783 | 0.075521 | 0.000714139 | 1177886 |
| 15000 | HR | 31.568158 | 0.948645 | 0.077401 | 0.000704527 | 1171504 |

## 17. Final RGB Metrics

QUANTITATIVE RESULT:

| Run | PSNR | SSIM | LPIPS | MSE | Gaussians |
| --- | ---: | ---: | ---: | ---: | ---: |
| M1 | 32.308910 | 0.949487 | 0.073979 | 0.000595238 | 1173293 |
| K1 | 31.498353 | 0.948783 | 0.075521 | 0.000714139 | 1177886 |
| HR | 31.568158 | 0.948645 | 0.077401 | 0.000704527 | 1171504 |
| AA secondary | 31.802683 | 0.948218 | 0.080714 | 0.000663285 | 1969585 |

QUANTITATIVE RESULT:

- `HR_PSNR_GAIN = +0.069805 dB` vs K1
- `GLOBAL_MSE_GAP_RECOVERY = 0.080846`
- `RGB_SAFETY = false` relative to M1
- `PANAMA_PARETO_CLOSED = false`

## 18. Decomposition Retention

QUANTITATIVE RESULT:

| Run | tau p90 | J p99 | P(J>1) | P(T<0.1) | beta_D mean | T mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 1.769849 | 1.311801 | 0.037758 | 0.007454 | 0.396922 | 0.384434 |
| K1 | 0.999069 | 0.838911 | 0.000000 | 0.000477 | 0.143104 | 0.685939 |
| HR | 1.004852 | 0.846604 | 0.000000 | 0.003047 | 0.141456 | 0.688611 |

QUANTITATIVE RESULT:

`TAU_BENEFIT_RETENTION = 0.992498`.

INFERENCE:

HR retains essentially all K1 low-optical-depth benefit by the registered tau-retention metric, while remaining bounded by the `P(J>1)` metric.

## 19. Base-Color Boundary

QUANTITATIVE RESULT:

HR base color `c0`, pooled:

| Metric | Value |
| --- | ---: |
| p50 | 0.325167 |
| p90 | 0.631264 |
| p95 | 0.744419 |
| p99 | 1.000000 |
| P(c0>0.95) | 0.018823 |
| P(c0>0.99) | 0.016403 |
| P(c0<0.05) | 0.000017 |
| P(c0<0.01) | 0.000004 |

## 20. Full-Color Boundary

QUANTITATIVE RESULT:

HR full color `c_HR`, pooled:

| Metric | Value |
| --- | ---: |
| p50 | 0.329657 |
| p90 | 0.651702 |
| p95 | 0.774612 |
| p99 | 1.000000 |
| P(c>0.95) | 0.020267 |
| P(c>0.99) | 0.017133 |
| P(c<0.05) | 0.000119 |
| P(c<0.01) | 0.000004 |

QUANTITATIVE RESULT:

`BOUNDARY_PRESSURE = false` under the registered rule because no pooled boundary fraction exceeds `0.05`.

## 21. Headroom Utilization

QUANTITATIVE RESULT:

HR pooled headroom utilization:

| Metric | u_pos | u_neg |
| --- | ---: | ---: |
| p50 | 0.033076 | 0.053135 |
| p90 | 0.152691 | 0.142164 |
| p95 | 0.235335 | 0.176737 |
| p99 | 0.518506 | 0.257866 |
| P(>0.75) | 0.004005 | 0.000004 |
| P(>0.90) | 0.001932 | 0.000000 |
| P(>0.99) | 0.000278 | 0.000000 |

QUANTITATIVE RESULT:

Per-channel threshold fractions:

| Channel | P(u_pos>0.75) | P(u_pos>0.90) | P(u_pos>0.99) | P(u_neg>0.75) | P(u_neg>0.90) | P(u_neg>0.99) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R | 0.002734 | 0.001278 | 0.000125 | 0.000007 | 0.000000 | 0.000000 |
| G | 0.003800 | 0.001945 | 0.000327 | 0.000004 | 0.000000 | 0.000000 |
| B | 0.005468 | 0.002562 | 0.000377 | 0.000000 | 0.000000 | 0.000000 |

## 22. Residual Saturation

QUANTITATIVE RESULT:

- `RESIDUAL_SATURATION = false`
- `BOUNDARY_PRESSURE = false`
- `raw |r| p90 = 0.275372`
- `raw |r| p99 = 0.698471`
- `P(|r|>5) = 0.0000076`

## 23. Color-Space SH Capacity

QUANTITATIVE RESULT:

HR color-domain residual magnitude `R_SH_COLOR = ||c_HR(v)-c0||_2`:

| Metric | Value |
| --- | ---: |
| p50 | 0.034982 |
| p90 | 0.097088 |
| p95 | 0.126875 |
| p99 | 0.198277 |

INFERENCE:

HR used nonzero color-domain SH residual capacity, but the registered RGB recovery thresholds were not reached.

## 24. Positive/Negative Residual Structure

QUANTITATIVE RESULT:

HR residual signs and energy:

| Metric | Value |
| --- | ---: |
| P(Delta c > 0) pooled | 0.529518 |
| P(Delta c < 0) pooled | 0.458759 |
| Positive energy fraction pooled | 0.675626 |
| Negative energy fraction pooled | 0.324374 |
| Luma positive fraction | 0.543348 |
| Luma negative fraction | 0.451078 |
| Chroma residual p90 | 0.064689 |

Per-channel positive energy fractions:

| Channel | Positive energy fraction | Negative energy fraction |
| --- | ---: | ---: |
| R | 0.649011 | 0.350989 |
| G | 0.710894 | 0.289106 |
| B | 0.669136 | 0.330864 |

## 25. High-J-Region Recovery

QUANTITATIVE RESULT:

Fixed M1 `J>1` mask aggregate:

| Run | MSE | Mean abs residual L1 | Pixel fraction |
| --- | ---: | ---: | ---: |
| M1 | 0.002617530 | 0.092595 | 0.050461 |
| K1 | 0.004818396 | 0.127099 | 0.050461 |
| HR | 0.004615808 | 0.125242 | 0.050461 |

QUANTITATIVE RESULT:

`HIGH_J_MSE_GAP_RECOVERY = 0.092049`.

## 26. Brightness / Bottom20 Controls

QUANTITATIVE RESULT:

Aggregate control regions:

| Mask | Run | MSE | Mean abs residual L1 |
| --- | --- | ---: | ---: |
| M1 J<=1 | K1 | 0.000497582 | 0.037324 |
| M1 J<=1 | HR | 0.000496113 | 0.036901 |
| Brightness Q5 | K1 | 0.002242602 | 0.077978 |
| Brightness Q5 | HR | 0.002224939 | 0.077626 |
| Bottom20 | K1 | 0.001603855 | 0.064290 |
| Bottom20 | HR | 0.001574341 | 0.063851 |

QUANTITATIVE RESULT:

`LOW_J_DAMAGE = -0.000001469`.

## 27. Per-View Metrics

QUANTITATIVE RESULT:

| View | K1 PSNR | HR PSNR | Delta PSNR | K1 SSIM | HR SSIM | K1 LPIPS | HR LPIPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MTN_1539 | 31.089798 | 31.030657 | -0.059141 | 0.936919 | 0.936706 | 0.072059 | 0.072041 |
| MTN_1529 | 32.304523 | 32.485432 | +0.180908 | 0.952028 | 0.952038 | 0.091365 | 0.094779 |
| MTN_1547 | 31.100737 | 31.188385 | +0.087648 | 0.957404 | 0.957192 | 0.063139 | 0.065384 |

QUANTITATIVE RESULT:

- HR vs K1 improved views: `2`
- HR vs K1 degraded views: `1`
- Delta PSNR mean / median / min / max: `0.069805 / 0.087648 / -0.059141 / 0.180908`

## 28. Recomposition Analysis

QUANTITATIVE RESULT:

Aggregate recomposition relative to M1:

| Run | C_direct | C_medium | C_cross | Recomp efficiency |
| --- | ---: | ---: | ---: | ---: |
| K1 | 0.004330614 | 0.004166260 | -0.008378678 | 0.928829 |
| HR | 0.004302684 | 0.004139882 | -0.008333982 | 0.928394 |

QUANTITATIVE RESULT:

MSE attribution closure error stayed at float-level scale:

- K1 aggregate absolute closure error: `6.402843e-10`
- HR aggregate absolute closure error: `4.462587e-10`

INFERENCE:

HR forms a similar object/medium recomposition pattern to K1. Recomposition efficiency is not the limiting positive signal for this specific candidate.

## 29. Gaussian Population

QUANTITATIVE RESULT:

| Step | K1 Gaussians | HR Gaussians |
| ---: | ---: | ---: |
| 1000 | 60112 | 59676 |
| 3000 | 566332 | 566550 |
| 5000 | 998668 | 996212 |
| 8000 | 1223420 | 1216856 |
| 10000 | 1219898 | 1212655 |
| 13000 | 1183679 | 1177164 |
| 15000 | 1177886 | 1171504 |

INFERENCE:

The population trajectory is close to K1; HR did not recover RGB via a large Gaussian-count shift.

## 30. Final Classification

QUANTITATIVE RESULT:

| Flag | Value |
| --- | --- |
| STRONG_HR_RECOVERY | false |
| PARTIAL_HR_RECOVERY | false |
| PANAMA_PARETO_CLOSED | false |
| HR_BOUNDARY_FAILURE | false |
| NO_HR_RECOVERY | false |
| HR_HARMFUL | false |

QUANTITATIVE RESULT:

The partial recovery gate was not met because `HR_PSNR_GAIN=0.069805 < 0.10` and `GLOBAL_MSE_GAP_RECOVERY=0.080846 < 0.20`, despite `TAU_BENEFIT_RETENTION=0.992498`, `BOUNDARY_PRESSURE=false`, and `LOW_J_DAMAGE=-0.000001469`.

## 31. Hypothesis Assessment

HYPOTHESIS:

Separating bounded base appearance from a Jacobian-matched asymmetric headroom SH residual recovers legal view-dependent fitting capacity in Panama without reopening the unbounded compensation route.

QUANTITATIVE CONCLUSION:

`HYPOTHESIS_ASSESSMENT = NOT_SUPPORTED` under the pre-registered recovery thresholds.

INFERENCE:

BND-HR is not harmful by the registered flags and it preserves the bounded decomposition proxy, but it does not produce enough RGB or high-J-region recovery to support this specific candidate as the Panama solution.

## 32. Next Single-Factor Recommendation

INFERENCE:

Do not continue an HR formula sweep from this result. The next single-factor step should be a read-only SeaFree-vs-WaterSplatting per-Gaussian degradation/compositing semantics audit, focused on whether the remaining Panama gap is caused by degradation/compositing hierarchy differences rather than the bounded SH headroom mapping itself.

No automatic next experiment was launched.

## Visual Assets

Generated contact sheets:

- Underwater: `renders/bnd_hr_panama_20260810/contact_sheet_underwater_m1_k1_hr.png`
- Clear raw display clamp: `renders/bnd_hr_panama_20260810/contact_sheet_clear_raw_m1_k1_hr.png`
- HR base/full/delta: `renders/bnd_hr_panama_20260810/contact_sheet_hr_base_full_delta.png`
- Signed HR residual: `renders/bnd_hr_panama_20260810/contact_sheet_hr_signed_residual.png`
- Headroom utilization: `renders/bnd_hr_panama_20260810/contact_sheet_hr_headroom_utilization.png`
- Fixed M1 high-J mask overlay: `renders/bnd_hr_panama_20260810/contact_sheet_high_j_region_k1_hr.png`
- Brightness Q5 overlay: `renders/bnd_hr_panama_20260810/contact_sheet_brightness_q5_k1_hr.png`
- Direct object signal delta: `renders/bnd_hr_panama_20260810/contact_sheet_direct_k1_hr_delta.png`
- Medium contribution delta: `renders/bnd_hr_panama_20260810/contact_sheet_medium_k1_hr_delta.png`
- Boundary pressure masks: `renders/bnd_hr_panama_20260810/contact_sheet_boundary_pressure_hr.png`

Visual manifest:

- `renders/bnd_hr_panama_20260810/manifest.json`
- `renders/bnd_hr_panama_20260810/manifest.csv`

Visual assets are ready for external/manual analysis.
No subjective clear-image correctness judgment was made.
