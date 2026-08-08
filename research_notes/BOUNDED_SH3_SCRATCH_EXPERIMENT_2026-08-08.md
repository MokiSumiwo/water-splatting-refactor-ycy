# Bounded SH3 Scratch Experiment

Date: 2026-08-08

## Motivation

### Experimental Fact

Existing Curasao scratch baselines show that `D010-SCRATCH` with `gamma_D=0.1` and legacy unbounded SH3 is absorbed by the learnable direct attenuation coefficient. The reported final raw beta ratio versus `D100-SCRATCH` is approximately 10.31, with higher effective optical depth and higher image-space clear-object high-value tail than `D100-SCRATCH`.

### Unverified Hypothesis

The failure mode may depend on the legacy unbounded SH3 intrinsic color route. If the actual current-view SH3 Gaussian intrinsic RGB is bounded by a sigmoid from step 0, the optimizer may have less access to the `high beta_D + low T + high J` compensation channel.

## Exact Bounded SH3 Definition

### Code Fact

New model config:

- `intrinsic_color_parameterization = "legacy"` by default.
- `intrinsic_color_parameterization = "sigmoid_sh"` enables the bounded SH3 branch.
- `bounded_sh_logit_eps = 1e-7` is used only for seed RGB logit initialization.

For `sigmoid_sh` and `sh_degree > 0`, WaterSplatting computes:

```text
s_i(v) = spherical_harmonics(active_sh_degree, viewdir, [features_dc, features_rest])
c_i(v) = sigmoid(s_i(v))
```

`c_i(v)` is the Gaussian RGB passed to the existing underwater rasterizer. It is therefore used by direct attenuated object contribution, clear/intrinsic render outputs derived from the rasterizer, and dependent forward outputs. The bounded branch keeps SH3 and keeps `features_rest` trainable.

### Code Fact

The bounded branch does not apply the legacy `+0.5` offset and does not clamp the post-SH value before sigmoid. It interprets the SH field as RGB logits.

## Legacy SH3 Code Path

### Code Fact

The legacy SH3 appearance path remains:

```text
colors = concat(features_dc[:, None, :], features_rest)
linear_rgb = spherical_harmonics(active_sh_degree, viewdirs, colors)
rgb = clamp(linear_rgb + 0.5, min=0.0)
```

The SH0 path remains `sigmoid(features_dc)`.

### Code Fact

`direct_optical_depth_scale` still multiplies only `medium_attn_raw` before direct attenuation. It does not modify `medium_bs`, `B_inf`, `medium_rgb`, or the backscatter exponent.

## Initialization-Equivalence Derivation

### Code Fact

WaterSplatting uses `C0 = 0.28209479177387814` for degree-0 SH conversion.

Legacy seeded SH3 DC initialization is:

```text
features_dc = (seed_rgb - 0.5) / C0
legacy_initial_rgb = C0 * features_dc + 0.5 = seed_rgb
features_rest = 0
```

Bounded seeded SH3 DC initialization is:

```text
features_dc = logit(seed_rgb, eps=1e-7) / C0
bounded_initial_rgb = sigmoid(C0 * features_dc) = seed_rgb, up to logit epsilon
features_rest = 0
```

This follows the SeaFree initialization principle for sigmoid color parameters while respecting WaterSplatting's SH coefficient convention. The fixed epsilon is `1e-7` rather than SeaFree's `1e-10` because Curasao seed RGB contains exact 0/255 values and float32 `logit(..., eps=1e-10)` does not keep the 1.0 endpoint strictly inside `(0,1)` after sigmoid.

## Initialization Audit

### Experimental Fact

Executed output files:

- `renders/dewater_bounded_sh3_scratch_20260808/bounded_sh3_initialization_audit.json`
- `renders/dewater_bounded_sh3_scratch_20260808/bounded_sh3_initialization_audit.csv`

### Experimental Fact

Initial run with `eps=1e-10` reached mean RGB error `1.3658597808330342e-08` and max RGB error `5.960464477539063e-08`, but strict boundedness was false because the upper endpoint rounded to `1.0` in float32. The experiment was not started with that epsilon.

Final fixed-epsilon audit used `BOUND_LOGIT_EPS=1e-7`:

| item | value |
| --- | ---: |
| seed points | 24455 |
| mean RGB error | 1.562371920726946e-08 |
| p95 RGB error | 2.9802322387695312e-08 |
| max RGB error | 1.1920928955078125e-07 |
| bounded RGB min | 1.000000082740371e-07 |
| bounded RGB max | 0.9999998807907104 |
| bounded RGB all finite | true |
| bounded RGB strictly inside `(0,1)` | true |
| legacy duplicate-forward max abs diff, pred/rgb_object/J | 0.0 / 0.0 / 0.0 |

## Training Protocol

### Experimental Fact

Only two new Curasao seed-42 from-scratch runs are allowed in this stage:

| Run | gamma_D | intrinsic parameterization | steps |
| --- | ---: | --- | ---: |
| BND-SCRATCH | 1.0 | bounded SH3 / `sigmoid_sh` | 0 -> 15000 |
| D010-BND-SCRATCH | 0.1 | bounded SH3 / `sigmoid_sh` | 0 -> 15000 |

All other training variables remain matched to the formal M1 settings: `medium_context_mode=dir_xy_camera`, `b_inf_mode=tied`, `infinite_water_enabled=False`, `SH degree=3`, `seed=42`, original RGB reconstruction loss, optimizer, scheduler, densification, pruning, opacity reset, and dataset split.

## Trajectory

### Experimental Fact

Completed numeric results are recorded in the `Completed Results` section below.

## Beta Compensation Ratio

### Experimental Fact

The central ratio is:

```text
R_beta_BND(step) =
mean_beta_D_raw(D010-BND-SCRATCH, step) /
mean_beta_D_raw(BND-SCRATCH, step)
```

For `gamma_D=0.1`, full direct-scale absorption corresponds to `R_beta_BND = 10`.

Completed numeric results are recorded in the `Completed Results` section below.

## Optical-Depth Results

### Experimental Fact

Completed numeric results are recorded in the `Completed Results` section below.

## Gaussian Saturation Analysis

### Experimental Fact

Bounded SH3 makes `P(c>1)` theoretically zero, so this experiment records:

```text
P(c<0.01)
P(c<0.05)
P(c>0.95)
P(c>0.99)
SATURATION_MASS_001 = P(c<0.01) + P(c>0.99)
```

Completed numeric results are recorded in the `Completed Results` section below.

## Pre-Sigmoid Saturation Analysis

### Experimental Fact

Diagnostics record:

```text
logit mean/p01/p05/p50/p95/p99/min/max
P(s > 4.595)
P(s < -4.595)
P(|s| > 5)
P(|s| > 8)
P(|s| > 10)
sigmoid derivative mean/p10/p50/p90
P(sigmoid'(s) < 0.01)
```

Completed numeric results are recorded in the `Completed Results` section below.

## Backscatter Compensation Audit

### Experimental Fact

Diagnostics record `medium_bs`, `medium_rgb`, `B_inf`, `backscatter`, and `rgb_medium`-related outputs when available. This is diagnostic only; this stage does not change backscatter scaling or backscatter loss.

Completed numeric results are recorded in the `Completed Results` section below.

## Final 2x2 Factorial Comparison

### Experimental Fact

The final comparison will use existing `D100-SCRATCH` and `D010-SCRATCH` baselines plus the two new bounded runs:

| gamma_D | Legacy unbounded SH3 | Bounded SH3 |
| ---: | --- | --- |
| 1.0 | D100-SCRATCH | BND-SCRATCH |
| 0.1 | D010-SCRATCH | D010-BND-SCRATCH |

Completed numeric results are recorded in the `Completed Results` section below.

## Classification

### Quantitative Conclusion

Completed classifications are recorded in the `Completed Results` section below:

- `BOUND_MAIN_EFFECT`
- `BOUND_SCALE_SYNERGY`
- `BOUND_ONLY_SUFFICIENT`
- `SCALE_STILL_FULLY_COMPENSATED`
- `SIGMOID_BOUNDARY_ESCAPE`
- `BOUNDED_PARAMETERIZATION_RGB_FAILURE`

## Remaining Hypotheses

### Unverified Hypothesis

If bounded SH3 does not prevent compensation, future branches may need to examine other SeaFree factors such as SH0/bounded DC plus limited SH residual, medium independent supervision, foreground-aware reconstruction, backscatter distance scaling, or medium capacity constraints. Those are not enabled or trained in this stage.

## Completed Results

### Experimental Fact

Two formal Curasao seed-42 from-scratch runs were completed:

- `BND-SCRATCH`: `gamma_D=1.0`, `intrinsic_color_parameterization=sigmoid_sh`, loaded final checkpoint `step-000014999.ckpt`.
- `D010-BND-SCRATCH`: `gamma_D=0.1`, `intrinsic_color_parameterization=sigmoid_sh`, loaded final checkpoint `step-000014999.ckpt`.

Smoke-gradient checks after 20 iterations passed for both allowed configurations. Loss and PSNR were finite, features_dc/features_rest/medium gradients were finite, bounded c/logits/derivatives were finite, and bounded c stayed strictly inside `(0,1)`. At 20 iterations the original SH schedule keeps active SH degree at 0, so `features_rest` grad norm was 0 but finite.

### Experimental Fact

Trajectory summary for the two new bounded runs:

| Step | Run | PSNR | SSIM | LPIPS | beta raw | beta eff | tau p90 | P(T<0.1) | J p99 | P(c>0.99) | logit p99 | P(|s|>5) | Gaussian count |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | BND | 27.010359 | 0.746974 | 0.444367 | 0.232398 | 0.232398 | 1.194166 | 0.000000 | 0.932839 | 0.014540 | 16.163535 | 0.014367 | 66021 |
| 1000 | D010-BND | 26.179892 | 0.741273 | 0.443488 | 1.238719 | 0.123872 | 0.783698 | 0.000000 | 0.814568 | 0.011841 | 12.513747 | 0.011895 | 56665 |
| 3000 | BND | 25.142677 | 0.664034 | 0.432751 | 0.224409 | 0.224409 | 1.191571 | 0.000000 | 0.852234 | 0.012532 | 12.545304 | 0.012232 | 527114 |
| 3000 | D010-BND | 23.234956 | 0.651142 | 0.436545 | 1.580649 | 0.158065 | 1.016652 | 0.000000 | 0.772613 | 0.010878 | 8.359234 | 0.010586 | 499229 |
| 5000 | BND | 30.308083 | 0.892572 | 0.228769 | 0.263260 | 0.263260 | 1.567133 | 0.000002 | 0.906163 | 0.009709 | 4.530442 | 0.009482 | 887894 |
| 5000 | D010-BND | 29.635226 | 0.891822 | 0.228903 | 2.146661 | 0.214666 | 1.610040 | 0.022432 | 0.899105 | 0.007889 | 2.768643 | 0.007720 | 870891 |
| 8000 | BND | 32.749229 | 0.956931 | 0.115314 | 0.272088 | 0.272088 | 1.606991 | 0.000010 | 0.926112 | 0.009753 | 5.152195 | 0.009566 | 1154871 |
| 8000 | D010-BND | 33.276488 | 0.957009 | 0.112002 | 2.315956 | 0.231596 | 1.883992 | 0.052504 | 0.979424 | 0.008080 | 3.180297 | 0.007933 | 1160824 |
| 10000 | BND | 32.773031 | 0.957764 | 0.112162 | 0.274260 | 0.274260 | 1.577177 | 0.000020 | 0.924174 | 0.010884 | 8.987834 | 0.010701 | 1118017 |
| 10000 | D010-BND | 33.350877 | 0.957925 | 0.109176 | 2.442068 | 0.244207 | 1.911889 | 0.053464 | 0.991592 | 0.009433 | 5.498993 | 0.009250 | 1121378 |
| 13000 | BND | 32.374568 | 0.959083 | 0.109315 | 0.263824 | 0.263824 | 1.567307 | 0.000002 | 0.906079 | 0.011115 | 9.626627 | 0.010938 | 1085683 |
| 13000 | D010-BND | 32.876338 | 0.959431 | 0.106068 | 2.378990 | 0.237899 | 1.858916 | 0.029835 | 0.993518 | 0.009678 | 6.153477 | 0.009499 | 1086823 |
| 15000 | BND | 32.188016 | 0.958728 | 0.109208 | 0.263348 | 0.263348 | 1.533122 | 0.000000 | 0.900398 | 0.011111 | 9.672927 | 0.010941 | 1081672 |
| 15000 | D010-BND | 32.966604 | 0.959689 | 0.105701 | 2.407370 | 0.240737 | 1.911318 | 0.032967 | 0.993158 | 0.009717 | 6.296670 | 0.009543 | 1082109 |

### Experimental Fact

`R_beta_BND` trajectory:

| Step | R_beta_BND | 10 - R_beta_BND |
| ---: | ---: | ---: |
| 1000 | 5.330156 | 4.669844 |
| 3000 | 7.043608 | 2.956392 |
| 5000 | 8.154152 | 1.845848 |
| 8000 | 8.511804 | 1.488196 |
| 10000 | 8.904193 | 1.095807 |
| 13000 | 9.017340 | 0.982660 |
| 15000 | 9.141406 | 0.858594 |

### Experimental Fact

Final 2x2 comparison:

| Run | PSNR | SSIM | LPIPS | beta raw | beta eff | tau p90 | P(T<0.1) | P(T<0.05) | J p99 | P(J>0.99) | P(J>1) | c p99 | P(c>0.95) | P(c>0.99) | SATURATION_MASS_001 | logit p99 | P(|s|>5) | P(|s|>8) | beta_B | medium mean | B_inf mean | Gaussian count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D100 | 32.165164 | 0.956004 | 0.108141 | 0.531716 | 0.531716 | 2.504676 | 0.143831 | 0.011275 | 2.350653 | 0.045673 | 0.044918 | 2.413399 | 0.084135 | 0.076425 | 0.119083 | N/A | N/A | N/A | 0.382016 | 0.168725 | 0.168725 | 1106714 |
| D010 | 32.326513 | 0.957235 | 0.108202 | 5.482617 | 0.548262 | 3.437896 | 0.241403 | 0.148904 | 4.496261 | 0.114424 | 0.113092 | 2.400391 | 0.088650 | 0.080599 | 0.099089 | N/A | N/A | N/A | 0.516811 | 0.146577 | 0.146577 | 1114093 |
| BND | 32.188016 | 0.958728 | 0.109208 | 0.263348 | 0.263348 | 1.533122 | 0.000000 | 0.000000 | 0.900398 | 0.003093 | 0.000000 | 0.985916 | 0.012693 | 0.011111 | 0.011132 | 9.672927 | 0.010941 | 0.010011 | 0.130190 | 0.201684 | 0.201684 | 1081672 |
| D010-BND | 32.966604 | 0.959689 | 0.105701 | 2.407370 | 0.240737 | 1.911318 | 0.032967 | 0.000924 | 0.993158 | 0.011339 | 0.000000 | 0.974604 | 0.011468 | 0.009717 | 0.009718 | 6.296670 | 0.009543 | 0.008771 | 0.165330 | 0.154766 | 0.154766 | 1082109 |

### Quantitative Conclusion

Factor effects at 15k:

| Comparison | Delta PSNR | Delta SSIM | Delta LPIPS | tau p90 relative drop | J p99 relative drop | P(J>1) relative drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BND - D100 | +0.022852 | +0.002724 | +0.001067 | +38.7896% | +61.6959% | +100.0000% |
| D010-BND - D010 | +0.640092 | +0.002454 | -0.002501 | +44.4044% | +77.9115% | +100.0000% |
| D010 - D100 | +0.161348 | +0.001231 | +0.000061 | -37.2591% | -91.2771% | -151.7755% |
| D010-BND - BND | +0.778588 | +0.000961 | -0.003507 | -24.6683% | -10.3022% | 0.0000% |

Classification:

- `BOUND_MAIN_EFFECT`: true. Numeric basis: BND vs D100 has PSNR `+0.022852`, tau p90 reduction `38.7896%`, J p99 reduction `61.6959%`, and P(J>1) reduction `100%`.
- `BOUND_SCALE_SYNERGY`: false under the implemented rule. `R_beta_BND@15k=9.141406`, and D010-BND tau p90 is higher than BND (`1.911318` vs `1.533122`).
- `BOUND_ONLY_SUFFICIENT`: true under the implemented rule because BND has the main decomposition effect and D010-BND does not lower tau/J further relative to BND.
- `SCALE_STILL_FULLY_COMPENSATED`: false under the implemented strict rule because `R_beta_BND@15k=9.141406` but tau p90 is not within 10% of BND.
- `SIGMOID_BOUNDARY_ESCAPE`: false under the implemented threshold. At 15k, saturation mass is `0.011132` for BND and `0.009718` for D010-BND; P(|s|>5) is `0.010941` and `0.009543`.
- `BOUNDED_PARAMETERIZATION_RGB_FAILURE`: false. BND PSNR is `+0.022852` vs D100, and D010-BND PSNR is `+0.640092` vs D010.

### Experimental Fact

Densification log summary:

| Run | densification events | cull events | split sum | duplicate sum | split children sum | culled sum | final logged remaining |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BND | 57 | 107 | 6487392 | 6465892 | 12974784 | 18384841 | 1081672 |
| D010-BND | 57 | 107 | 6405129 | 6706758 | 12810258 | 18460744 | 1082109 |

### Experimental Fact

Visual assets:

- `renders/dewater_bounded_sh3_scratch_20260808/visual_compare_2x2_step_15000/contact_sheet_underwater_2x2.png`
- `renders/dewater_bounded_sh3_scratch_20260808/visual_compare_2x2_step_15000/contact_sheet_direct_object_signal_2x2.png`
- `renders/dewater_bounded_sh3_scratch_20260808/visual_compare_2x2_step_15000/contact_sheet_clear_raw_2x2.png`
- `renders/dewater_bounded_sh3_scratch_20260808/visual_compare_2x2_step_15000/contact_sheet_clear_clamp01_2x2.png`
- `renders/dewater_bounded_sh3_scratch_20260808/visual_compare_2x2_step_15000/contact_sheet_clear_ws_tonemap_2x2.png`
- `renders/dewater_bounded_sh3_scratch_20260808/visual_compare_2x2_step_15000/contact_sheet_transmission_2x2.png`
- `renders/dewater_bounded_sh3_scratch_20260808/visual_compare_2x2_step_15000/contact_sheet_tau_d_2x2.png`
- `renders/dewater_bounded_sh3_scratch_20260808/visual_compare_2x2_step_15000/contact_sheet_alpha_sweep_d100.png`
- `renders/dewater_bounded_sh3_scratch_20260808/visual_compare_2x2_step_15000/contact_sheet_alpha_sweep_d010.png`
- `renders/dewater_bounded_sh3_scratch_20260808/visual_compare_2x2_step_15000/contact_sheet_alpha_sweep_bnd.png`
- `renders/dewater_bounded_sh3_scratch_20260808/visual_compare_2x2_step_15000/contact_sheet_alpha_sweep_d010_bnd.png`
- `renders/dewater_bounded_sh3_scratch_20260808/visual_compare_2x2_step_15000/manifest.json`
- `renders/dewater_bounded_sh3_scratch_20260808/visual_compare_2x2_step_15000/manifest.csv`
- `renders/dewater_bounded_sh3_scratch_20260808/visual_compare_2x2_step_15000/VISUAL_COMPARE_INDEX.md`

### Reasonable Inference

Q1: From-step0 bounded intrinsic range did prevent the legacy extreme high-J image-space tail in these Curasao runs: BND has J p99 `0.900398` and P(J>1) `0`, versus D100 J p99 `2.350653` and P(J>1) `0.044918`.

Q2: BND-SCRATCH alone improved the decomposition proxies relative to D100-SCRATCH while keeping RGB metrics inside the predefined safety range.

Q3: In bounded SH3, D010 raw beta still moved close to the theoretical 10x compensation ratio: `R_beta_BND@15k=9.141406`.

Q4: Bounded intrinsic did not make `/10` produce lower tau relative to BND at 15k. D010-BND beta_eff was lower (`0.240737` vs `0.263348`), but tau p90 was higher (`1.911318` vs `1.533122`) and P(T<0.1) was higher (`0.032967` vs `0.0`).

Q5: The bounded runs did not show large sigmoid boundary escape under the implemented diagnostic threshold. P(c>0.99) stayed near 1%, and P(|s|>5) stayed near 1%.

Q6: Backscatter/medium values changed with the parameterization. Relative to D100, BND beta_B decreased from `0.382016` to `0.130190` while medium/B_inf mean increased from `0.168725` to `0.201684`; relative to BND, D010-BND beta_B increased to `0.165330` and medium/B_inf mean decreased to `0.154766`. This is recorded as branch redistribution, not medium-GT correctness.

Q7: Current evidence supports: `Intrinsic range is the primary missing constraint` for the tested Curasao decomposition proxies, while additional identifiability constraints may still be necessary for the `/10` scaling factor because D010-BND approaches beta compensation and does not further reduce tau versus BND.

Q8: Based only on these numbers, the next controlled direction would be BND cross-scene validation before adding new SeaFree factors. D010+BND cross-scene validation is less directly supported as a decomposition improvement because it raises tau p90 relative to BND despite improving RGB metrics.

## Reference State

### Code Fact

WaterSplatting start HEAD:

```text
77c64bc0eb21da17680ad7fff39f1bd46c479561
Audit SeaFree-inspired dewatering factors
```

SeaFree-GS reference commit:

```text
7797e97dae831029ac89ae9f37b3c3d69ec2cf6c
```

SeaFree-GS status at start: clean.
