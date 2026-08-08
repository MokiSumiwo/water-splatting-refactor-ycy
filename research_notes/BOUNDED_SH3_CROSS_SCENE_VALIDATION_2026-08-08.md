# Bounded SH3 Cross-Scene Validation

Date: 2026-08-08

## Motivation

### Experimental Fact

Curasao previously showed a positive bounded-intrinsic result for `BND-SCRATCH`:

```text
SH degree = 3
intrinsic color = sigmoid(full active SH3 logits)
gamma_D = 1.0
```

The purpose of this stage is replication only: apply the fixed Curasao BND candidate to `JapaneseGradens`, `IUI3`, and `Panama`, each from scratch for 15000 steps, and compare to matched formal M1 baselines.

### Code Fact

WaterSplatting start HEAD:

```text
c5f2b08b08b739cd0e6edaeb146640a2cccd5d94
Test bounded SH3 from scratch
```

SeaFree-GS reference commit:

```text
7797e97dae831029ac89ae9f37b3c3d69ec2cf6c
```

SeaFree-GS was used as read-only reference state only in this stage.

## Fixed BND Definition

### Code Fact

The fixed candidate is:

```text
gamma_D = 1.0
SH degree = 3
c_i(v) = sigmoid(s_i(v))
s_i(v) = spherical_harmonics(active_sh_degree, viewdir, [features_dc, features_rest])
```

`c_i(v)` is passed into the existing underwater rasterizer and therefore affects underwater RGB, direct object rendering, and clear/intrinsic rendering. This stage did not change the sigmoid position, initialization epsilon, SH degree, optimizer, scheduler, densification, pruning, opacity behavior, RGB loss, medium architecture, renderer physics, GMVC, D010, or any SeaFree-inspired additional loss.

### Code Fact

Color-equivalent bounded initialization was reused:

```text
features_dc = logit(seed_rgb, eps=1e-7) / C0
features_rest = 0
BOUND_LOGIT_EPS = 1e-7
```

## Initialization And Smoke Checks

### Experimental Fact

Per-scene initialization sanity checks passed:

| Scene | seed points | mean RGB error | p95 RGB error | max RGB error | bounded RGB min | bounded RGB max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| JapaneseGradens | 20522 | 1.3060278192256192e-08 | 2.9802322387695312e-08 | 1.1920928955078125e-07 | 1.000000082740371e-07 | 0.9999998807907104 |
| IUI3 | 20707 | 1.413714389997267e-08 | 5.960464477539063e-08 | 1.1920928955078125e-07 | 1.000000082740371e-07 | 0.9999998807907104 |
| Panama | 21010 | 1.9912084425754983e-08 | 1.1920928955078125e-07 | 1.1920928955078125e-07 | 1.000000082740371e-07 | 0.9999998807907104 |

All three scenes had finite bounded logits and bounded RGB strictly inside `(0, 1)`. Legacy duplicate-forward equivalence for the loaded M1 configs had max absolute difference `0.0` for `pred_image`, `rgb_object`, and `J_gaussian_raw`.

### Experimental Fact

20-iteration smoke-gradient checks passed for all three new scenes. Loss and PSNR were finite; `features_dc`, `features_rest`, and medium gradients were finite; bounded current-view `c`, logits, and sigmoid derivatives were finite. As expected under the original active-SH schedule, `features_rest` grad norm was zero at step 19 because active SH was still degree 0.

## Training Protocol

### Experimental Fact

Exactly three new full runs were trained:

| Run | Scene | Steps | gamma_D | intrinsic |
| --- | --- | ---: | ---: | --- |
| BND-JAPANESE | JapaneseGradens | 0 -> 15000 | 1.0 | `sigmoid_sh` |
| BND-IUI3 | IUI3 | 0 -> 15000 | 1.0 | `sigmoid_sh` |
| BND-PANAMA | Panama | 0 -> 15000 | 1.0 | `sigmoid_sh` |

Final checkpoints use the repository convention `step-000014999.ckpt` for nominal step 15000.

## Baseline Matching Audit

### Experimental Fact

M1 baselines were reused. They matched BND on seed, max iterations, SH degree, medium context, B_inf mode, infinite-water disabled, direct optical-depth scale, dataset split settings, downscale factor, densification threshold, refine interval, and opacity reset interval. The only intended difference was `intrinsic_color_parameterization: legacy` vs `sigmoid_sh`; BND used `steps_per_save=1000` while baselines were saved every 5000 steps.

| Scene | M1 config | BND config | matched |
| --- | --- | --- | --- |
| Curasao | `outputs/cross_scene_curasao_m1_seed42_15000/water-splatting/cross_scene_curasao_m1_seed42_15000_20260730_cross_scene/config.yml` | `outputs/dewater_bounded_sh3_scratch_20260808/dewater_bounded_sh3_curasao_seed42_bnd-scratch_step0_to_15000/water-splatting/dewater_bounded_sh3_curasao_seed42_bnd-scratch_step0_to_15000_20260808_bounded_sh3_scratch_full_bnd-scratch_g1p00/config.yml` | true |
| JapaneseGradens | `outputs/cross_scene_japanesegradens_redsea_m1_seed42_15000/water-splatting/cross_scene_japanesegradens_redsea_m1_seed42_15000_20260730_cross_scene/config.yml` | `outputs/dewater_bounded_sh3_cross_scene_20260808/dewater_bnd_cross_scene_japanesegradens_seed42_step0_to_15000/water-splatting/dewater_bnd_cross_scene_japanesegradens_seed42_step0_to_15000_20260808_bounded_sh3_cross_scene_full_japanesegradens_bnd_g1p00/config.yml` | true |
| IUI3 | `outputs/gmvc_v3_four_scene_iui3_m1_seed42_15000/water-splatting/gmvc_v3_four_scene_iui3_m1_seed42_15000_20260806_gmvc_four_scene_p30_mhold_15k_m1_bootstrap/config.yml` | `outputs/dewater_bounded_sh3_cross_scene_20260808/dewater_bnd_cross_scene_iui3_seed42_step0_to_15000/water-splatting/dewater_bnd_cross_scene_iui3_seed42_step0_to_15000_20260808_bounded_sh3_cross_scene_full_iui3_bnd_g1p00/config.yml` | true |
| Panama | `outputs/cross_scene_panama_m1_seed42_15000/water-splatting/cross_scene_panama_m1_seed42_15000_20260730_cross_scene/config.yml` | `outputs/dewater_bounded_sh3_cross_scene_20260808/dewater_bnd_cross_scene_panama_seed42_step0_to_15000/water-splatting/dewater_bnd_cross_scene_panama_seed42_step0_to_15000_20260808_bounded_sh3_cross_scene_full_panama_bnd_g1p00/config.yml` | true |

Full audit files:

- `renders/dewater_bounded_sh3_cross_scene_20260808/four_scene_summary/cross_scene_bnd_baseline_audit.json`
- `renders/dewater_bounded_sh3_cross_scene_20260808/four_scene_summary/cross_scene_bnd_baseline_audit.csv`

## Four-Scene Quantitative Comparison

### Experimental Fact

| Scene | Run | PSNR | SSIM | LPIPS | beta eff | tau p90 | P(T<0.1) | P(T<0.05) | J p99 | P(J>1) | c p99 | P(c>0.99) | P(|s|>5) | SATURATION_MASS | beta_B | Gaussian count |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Curasao | M1 | 32.165164 | 0.956004 | 0.108141 | 0.531716 | 2.504676 | 0.143831 | 0.011275 | 2.350653 | 0.044918 | 2.413399 | 0.076425 | 0.000000 | 0.119083 | 0.382016 | 1106714 |
| Curasao | BND | 32.188016 | 0.958728 | 0.109208 | 0.263348 | 1.533122 | 0.000000 | 0.000000 | 0.900398 | 0.000000 | 0.985916 | 0.011111 | 0.010941 | 0.011132 | 0.130190 | 1081672 |
| JapaneseGradens | M1 | 24.756484 | 0.899500 | 0.120357 | 0.349011 | 1.947181 | 0.042124 | 0.013405 | 1.348906 | 0.049442 | 1.480809 | 0.067567 | 0.000000 | 0.105272 | 0.512010 | 861508 |
| JapaneseGradens | BND | 24.698505 | 0.896300 | 0.117283 | 0.074932 | 0.712033 | 0.004992 | 0.001324 | 0.966225 | 0.000000 | 0.991475 | 0.010224 | 0.009663 | 0.010231 | 0.196181 | 870028 |
| IUI3 | M1 | 30.874542 | 0.912143 | 0.174617 | 0.155611 | 1.497287 | 0.014075 | 0.009442 | 1.137097 | 0.021695 | 1.325924 | 0.064421 | 0.000000 | 0.101259 | 0.393459 | 808747 |
| IUI3 | BND | 30.963132 | 0.911644 | 0.177082 | 0.088740 | 0.928444 | 0.000000 | 0.000000 | 0.904771 | 0.000000 | 0.989654 | 0.010054 | 0.009746 | 0.010879 | 0.207127 | 797196 |
| Panama | M1 | 32.308910 | 0.949487 | 0.073979 | 0.396923 | 1.769849 | 0.007454 | 0.000000 | 1.311801 | 0.037758 | 1.644130 | 0.109174 | 0.000000 | 0.121111 | 0.296831 | 1173293 |
| Panama | BND | 31.498353 | 0.948783 | 0.075521 | 0.143104 | 0.999069 | 0.000477 | 0.000000 | 0.838911 | 0.000000 | 0.999999 | 0.017517 | 0.017149 | 0.017518 | 0.084423 | 1177886 |

## Scene Deltas And Gate Results

### Quantitative Conclusion

| Scene | dPSNR | dSSIM | dLPIPS | tau p90 reduction | P(T<0.1) reduction | J p99 reduction | dP(J>1) | RGB safety | Decomp. improvement | Boundary escape | Scene pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| Curasao | +0.022852 | +0.002724 | +0.001067 | 38.7896% | 100.0000% | 61.6959% | -0.044918 | true | true | false | true |
| JapaneseGradens | -0.057979 | -0.003200 | -0.003073 | 63.4326% | 88.1487% | 28.3697% | -0.049442 | false | true | false | false |
| IUI3 | +0.088590 | -0.000499 | +0.002465 | 37.9916% | 100.0000% | 20.4315% | -0.021695 | true | true | false | true |
| Panama | -0.810558 | -0.000704 | +0.001542 | 43.5506% | 93.6016% | 36.0489% | -0.037758 | false | true | false | false |

RGB safety gate:

```text
Delta PSNR >= -0.15 dB
Delta SSIM >= -0.0015
Delta LPIPS <= +0.003
```

JapaneseGradens failed RGB safety due to SSIM delta `-0.003200`, despite PSNR and LPIPS being within the stated thresholds. Panama failed RGB safety due to PSNR delta `-0.810558`.

## Boundary Saturation Analysis

### Quantitative Conclusion

No scene crossed the implemented boundary-escape threshold:

```text
P(c>0.99) <= 5%
P(|s|>5) <= 5%
```

BND final values:

| Scene | P(c>0.99) | P(|s|>5) | SATURATION_MASS |
| --- | ---: | ---: | ---: |
| Curasao | 0.011111 | 0.010941 | 0.011132 |
| JapaneseGradens | 0.010224 | 0.009663 | 0.010231 |
| IUI3 | 0.010054 | 0.009746 | 0.010879 |
| Panama | 0.017517 | 0.017149 | 0.017518 |

Thus this run did not show large-scale sigmoid boundary escape.

## Medium Compensation Redistribution

### Experimental Fact

The configured redistribution flag was false for all four scenes. BND reduced `beta_B` and `backscatter_mean` rather than increasing them under the script's redistribution criterion.

| Scene | beta_B relative change | medium_rgb relative change | backscatter relative change | redistribution flag |
| --- | ---: | ---: | ---: | --- |
| Curasao | -65.9203% | +19.5341% | -39.3993% | false |
| JapaneseGradens | -61.6841% | -5.8911% | -53.6823% | false |
| IUI3 | -47.3575% | +2.8416% | -25.4367% | false |
| Panama | -71.5586% | -11.7627% | -53.6459% | false |

### Reasonable Inference

Within the available diagnostics, this stage did not show compensation being moved into the backscatter branch. This is not a statement about physical correctness because no medium ground truth is available.

## Trajectory Summary For New BND Runs

### Experimental Fact

| Scene | Step | PSNR | tau p90 | J p99 | P(c>0.99) | P(|s|>5) | Gaussian count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| JapaneseGradens | 1000 | 20.417708 | 0.247100 | 0.981619 | 0.006070 | 0.005367 | 65683 |
| JapaneseGradens | 3000 | 18.921530 | 0.594460 | 0.990755 | 0.012319 | 0.011405 | 392036 |
| JapaneseGradens | 5000 | 22.925909 | 0.755131 | 0.993387 | 0.012457 | 0.011756 | 707251 |
| JapaneseGradens | 8000 | 24.818353 | 0.737648 | 0.985229 | 0.010775 | 0.010197 | 938188 |
| JapaneseGradens | 10000 | 24.723005 | 0.776678 | 0.982046 | 0.010896 | 0.010342 | 914966 |
| JapaneseGradens | 13000 | 24.720423 | 0.729494 | 0.969741 | 0.010575 | 0.010037 | 876937 |
| JapaneseGradens | 15000 | 24.698505 | 0.712033 | 0.966225 | 0.010224 | 0.009663 | 870028 |
| IUI3 | 1000 | 22.955635 | 0.386491 | 0.871241 | 0.018460 | 0.018637 | 47163 |
| IUI3 | 3000 | 20.621342 | 0.845004 | 0.883868 | 0.011978 | 0.011689 | 297659 |
| IUI3 | 5000 | 25.792911 | 1.062820 | 0.904885 | 0.008514 | 0.008022 | 614534 |
| IUI3 | 8000 | 30.381599 | 1.049679 | 0.927952 | 0.008527 | 0.007951 | 830834 |
| IUI3 | 10000 | 30.424920 | 1.053203 | 0.929279 | 0.009484 | 0.008904 | 837842 |
| IUI3 | 13000 | 30.842779 | 0.961969 | 0.910829 | 0.009886 | 0.009477 | 803473 |
| IUI3 | 15000 | 30.963132 | 0.928444 | 0.904771 | 0.010054 | 0.009746 | 797196 |
| Panama | 1000 | 22.563314 | 0.847200 | 0.884581 | 0.090981 | 0.088047 | 60112 |
| Panama | 3000 | 20.869651 | 0.967602 | 0.785301 | 0.040982 | 0.039732 | 566332 |
| Panama | 5000 | 26.459815 | 1.028014 | 0.816315 | 0.022169 | 0.021777 | 998668 |
| Panama | 8000 | 31.008972 | 1.051138 | 0.845535 | 0.018313 | 0.018012 | 1223420 |
| Panama | 10000 | 31.154767 | 1.094382 | 0.854751 | 0.018118 | 0.017773 | 1219898 |
| Panama | 13000 | 31.540089 | 1.042146 | 0.837378 | 0.017607 | 0.017258 | 1183679 |
| Panama | 15000 | 31.498353 | 0.999069 | 0.838911 | 0.017517 | 0.017149 | 1177886 |

## Cross-Scene Classification

### Quantitative Conclusion

Automatic gate results:

```text
CROSS_SCENE_BND_FULL = false
CROSS_SCENE_BND_STRONG = false
RGB_SAFE_BUT_SCENE_DEPENDENT = false
BOUNDARY_ESCAPE_CROSS_SCENE = false
BND_CROSS_SCENE_FAILURE = true
```

Reason: Curasao and IUI3 passed all scene gates, but JapaneseGradens and Panama did not pass RGB safety. All four scenes passed the decomposition-improvement and no-boundary-escape gates.

## Visual Assets

### Experimental Fact

Root:

```text
renders/dewater_bounded_sh3_cross_scene_20260808/
```

Per scene:

```text
<scene>/contact_sheet_underwater_m1_vs_bnd.png
<scene>/contact_sheet_clear_raw_m1_vs_bnd.png
<scene>/contact_sheet_clear_clamp01_m1_vs_bnd.png
<scene>/contact_sheet_direct_object_signal_m1_vs_bnd.png
<scene>/contact_sheet_transmission_m1_vs_bnd.png
<scene>/contact_sheet_tau_d_m1_vs_bnd.png
<scene>/boundary_saturation_mask_bnd.png
```

Four-scene summaries:

```text
renders/dewater_bounded_sh3_cross_scene_20260808/four_scene_summary/four_scene_clear_raw_summary.png
renders/dewater_bounded_sh3_cross_scene_20260808/four_scene_summary/four_scene_underwater_summary.png
```

Visual manifest and index:

```text
renders/dewater_bounded_sh3_cross_scene_20260808/manifest.json
renders/dewater_bounded_sh3_cross_scene_20260808/manifest.csv
renders/dewater_bounded_sh3_cross_scene_20260808/VISUAL_COMPARE_INDEX.md
```

No subjective visual-quality labels were added.

## Answers To Required Questions

### Quantitative Conclusion

Q1. JapaneseGradens M1 does show high-J/high-tau compensation by the defined proxies: tau p90 `1.947181`, J p99 `1.348906`, P(J>1) `0.049442`.

Q2. JapaneseGradens BND lowered tau p90 by `63.4326%`, P(T<0.1) by `88.1487%`, J p99 by `28.3697%`, and P(J>1) to `0`. It failed RGB safety due to SSIM delta `-0.003200`.

Q3. IUI3 repeated the positive mechanism and passed RGB safety: tau p90 reduction `37.9916%`, P(T<0.1) reduction `100%`, J p99 reduction `20.4315%`, P(J>1) to `0`, PSNR delta `+0.088590`.

Q4. Panama repeated the decomposition-proxy change but failed RGB safety: tau p90 reduction `43.5506%`, P(T<0.1) reduction `93.6016%`, J p99 reduction `36.0489%`, P(J>1) to `0`, PSNR delta `-0.810558`.

Q5. No. BND did not maintain underwater RGB safety in all three new scenes. IUI3 passed; JapaneseGradens and Panama failed at least one RGB safety criterion.

Q6. No scene showed boundary escape under the implemented thresholds.

Q7. No scene was flagged as redistribution to `beta_B` / `B_inf` / `rgb_medium` by the current diagnostics. `beta_B` decreased in every scene.

Q8. All four M1 baselines had J p99 above 1 and nonzero P(J>1), and all four BND runs reduced J p99 and P(J>1). The strongest tau reductions occurred in JapaneseGradens and Panama, but those did not pass RGB safety. This supports a decomposition-proxy trend, not a complete RGB-safe cross-scene upgrade.

Q9. Required classification: `BND_CROSS_SCENE_FAILURE` for the full gate suite, with auxiliary facts that decomposition improvement replicated in all four scenes and boundary escape did not occur.

Q10. Recommendation: `PARTIALLY`. BND is a strong mechanism candidate for reducing high-J/high-optical-depth compensation, but it is not ready to replace M1 as a cross-scene core default because two of the three new scenes failed RGB safety.

## Remaining Limitations

### Unverified Hypothesis

This stage does not establish physical correctness of the recovered clear appearance or medium parameters. There is no clear-image GT or medium GT in these real scenes.

### Reasonable Inference

The fixed bounded SH3 parameterization consistently closes the high-value intrinsic route in the measured diagnostics. The unresolved issue is RGB-domain compatibility across all scenes, not boundary escape or a measured shift into backscatter.
