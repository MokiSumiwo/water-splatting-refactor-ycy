# BND RGB Trade-off Diagnosis

Date: 2026-08-09

## Motivation

### Code Fact

This stage was diagnosis only. No model, renderer, loss, optimizer, scheduler, densification, pruning, opacity, SH degree, GMVC, D010, background supervision, depth loss, foreground-aware weighting, or bounded-intrinsic mechanism was changed.

Diagnostic script:

```text
scripts/diagnostics/diagnose_bnd_rgb_tradeoff.py
```

Outputs:

```text
outputs/bnd_rgb_tradeoff_diagnosis_20260809/
renders/bnd_rgb_tradeoff_diagnosis_20260809/
logs/bnd_rgb_tradeoff_diagnosis_20260809/
```

### Experimental Fact

Start branch and HEAD:

```text
branch = research/m1-bounded-intrinsic
START_HEAD = 62294d6940771552a88d4ec2234d2ff4db53b874
START_COMMIT = Build clean M1 bounded-intrinsic baseline
```

Two historical untracked GMVC scripts were present at start and were not modified:

```text
scripts/diagnostics/render_gmvc_curasao_contact_sheet.py
scripts/diagnostics/summarize_gmvc_four_scene_visual_audit.py
```

## Questions

### Experimental Fact

The prior cross-scene validation already established that BND reduces direct optical depth and high-J tails in all four scenes. This diagnosis therefore asks why RGB metrics are safe in Curasao/IUI3 but fail for JapaneseGradens SSIM and Panama PSNR.

## Checkpoint Audit

### Code Fact

BND historical configs stored `intrinsic_color_parameterization: sigmoid_sh`. The clean branch uses `bounded_sh3`; the diagnostic maps BND configs to `bounded_sh3` in memory after loading. Checkpoints are not edited.

### Experimental Fact

All final comparisons loaded nominal step 15000 as checkpoint step 14999. All compared runs used seed 42, max iterations 15000, SH degree 3, `medium_context_mode=dir_xy_camera`, `b_inf_mode=tied`, `infinite_water_enabled=False`, and `mlp_type=tcnn`.

| Scene | Run | loaded step | seed | save cadence | SH | medium | B_inf | Gaussians |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| Curasao | M1 | 14999 | 42 | 5000 | 3 | dir_xy_camera | tied | 1106714 |
| Curasao | BND | 14999 | 42 | 1000 | 3 | dir_xy_camera | tied | 1081672 |
| JapaneseGradens | M1 | 14999 | 42 | 5000 | 3 | dir_xy_camera | tied | 861508 |
| JapaneseGradens | BND | 14999 | 42 | 1000 | 3 | dir_xy_camera | tied | 870028 |
| IUI3 | M1 | 14999 | 42 | 5000 | 3 | dir_xy_camera | tied | 808747 |
| IUI3 | BND | 14999 | 42 | 1000 | 3 | dir_xy_camera | tied | 797196 |
| Panama | M1 | 14999 | 42 | 5000 | 3 | dir_xy_camera | tied | 1173293 |
| Panama | BND | 14999 | 42 | 1000 | 3 | dir_xy_camera | tied | 1177886 |

Full audit files:

- `outputs/bnd_rgb_tradeoff_diagnosis_20260809/bnd_checkpoint_audit.csv`
- `outputs/bnd_rgb_tradeoff_diagnosis_20260809/bnd_checkpoint_audit.json`

### Experimental Fact

M1 baseline trajectory checkpoints were only available at 5k, 10k, and 14999. Missing 1k, 3k, 8k, and 13k M1 checkpoints were recorded as `MISSING_CHECKPOINT` and were not regenerated.

Missing checkpoint file:

```text
outputs/bnd_rgb_tradeoff_diagnosis_20260809/bnd_missing_checkpoints.csv
```

## Training Trajectory

### Quantitative Result

| Scene | Step | Run | PSNR | SSIM | LPIPS | tau p90 | P(T<0.1) | J p99 | Gaussians |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Curasao | 10000 | M1 | 33.151569 | 0.956404 | 0.109539 | 2.750149 | 0.190019 | 1.771311 | 1140794 |
| Curasao | 10000 | BND | 32.773031 | 0.957764 | 0.112163 | 1.829468 | 0.025780 | 0.755971 | 1118017 |
| Curasao | 13000 | BND | 32.374568 | 0.959083 | 0.109315 | 1.705528 | 0.015351 | 0.746377 | 1085683 |
| Curasao | 15000 | M1 | 32.165164 | 0.956004 | 0.108141 | 2.532414 | 0.148127 | 1.833698 | 1106714 |
| Curasao | 15000 | BND | 32.188016 | 0.958728 | 0.109208 | 1.638588 | 0.010584 | 0.746688 | 1081672 |
| JapaneseGradens | 10000 | M1 | 24.825679 | 0.896288 | 0.124963 | 2.375206 | 0.118248 | 1.354051 | 904681 |
| JapaneseGradens | 10000 | BND | 24.723005 | 0.893117 | 0.121996 | 1.094450 | 0.025056 | 0.877354 | 914966 |
| JapaneseGradens | 13000 | BND | 24.720423 | 0.897221 | 0.116592 | 0.864054 | 0.016131 | 0.863678 | 876937 |
| JapaneseGradens | 15000 | M1 | 24.756484 | 0.899500 | 0.120356 | 1.903143 | 0.043504 | 1.236499 | 861508 |
| JapaneseGradens | 15000 | BND | 24.698505 | 0.896300 | 0.117283 | 0.834880 | 0.015076 | 0.856821 | 870028 |
| IUI3 | 10000 | M1 | 30.307353 | 0.902584 | 0.184881 | 2.927647 | 0.099306 | 1.293565 | 846230 |
| IUI3 | 10000 | BND | 30.424920 | 0.902108 | 0.188073 | 1.297502 | 0.000000 | 0.891095 | 837842 |
| IUI3 | 13000 | BND | 30.842778 | 0.911340 | 0.177821 | 1.111696 | 0.000000 | 0.878875 | 803473 |
| IUI3 | 15000 | M1 | 30.874542 | 0.912143 | 0.174617 | 1.475870 | 0.016779 | 1.170564 | 808747 |
| IUI3 | 15000 | BND | 30.963132 | 0.911644 | 0.177082 | 1.060867 | 0.000000 | 0.874592 | 797196 |
| Panama | 10000 | M1 | 32.148161 | 0.947382 | 0.078112 | 1.735044 | 0.023119 | 1.329932 | 1213095 |
| Panama | 10000 | BND | 31.154767 | 0.945652 | 0.082064 | 1.047273 | 0.003979 | 0.837913 | 1219898 |
| Panama | 13000 | BND | 31.540089 | 0.949397 | 0.074719 | 0.946699 | 0.001508 | 0.826056 | 1183679 |
| Panama | 15000 | M1 | 32.308910 | 0.949487 | 0.073979 | 1.594246 | 0.007454 | 1.293632 | 1173293 |
| Panama | 15000 | BND | 31.498353 | 0.948783 | 0.075521 | 0.909361 | 0.000477 | 0.829595 | 1177886 |

### Quantitative Conclusion

BND late-recovery flag:

| Scene | BND PSNR 13k -> 15k | POSSIBLE_UNDERCONVERGENCE |
| --- | ---: | --- |
| Curasao | -0.186551 | false |
| JapaneseGradens | -0.021918 | false |
| IUI3 | +0.120354 | true |
| Panama | -0.041737 | false |

Panama under-convergence is not supported by the available trajectory: BND PSNR did not recover from 13k to 15k, while tau p90 remained lower (`0.946699 -> 0.909361`) and J p99 stayed bounded (`0.826056 -> 0.829595`). IUI3 showed BND late PSNR recovery, but IUI3 already passes the RGB gate.

## Per-view RGB Loss

### Quantitative Result

| Scene | eval views | mean dPSNR | median dPSNR | min dPSNR | max dPSNR | worst views | best views |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Curasao | 3 | +0.022852 | +0.065632 | -0.812763 | +0.815687 | MTN_1304:-0.812763 | MTN_1288:+0.815687 |
| JapaneseGradens | 3 | -0.057979 | +0.063372 | -0.305780 | +0.068472 | MTN_1090:-0.305780 | MTN_1098:+0.068472 |
| IUI3 | 4 | +0.088590 | +0.137991 | -0.368431 | +0.446810 | MTN_5903:-0.368431 | MTN_5911:+0.446810 |
| Panama | 3 | -0.810558 | -0.547436 | -1.631567 | -0.252670 | MTN_1547:-1.631567 | MTN_1539:-0.252670 |

### Quantitative Conclusion

Panama loss is distributed across all three eval views in this split, not caused by a single isolated bad view. JapaneseGradens has one negative view and two slightly positive PSNR views; its scene-level RGB gate failure is primarily the SSIM delta, not a large PSNR collapse.

Full per-view file:

```text
outputs/bnd_rgb_tradeoff_diagnosis_20260809/bnd_per_view_rgb_delta.csv
```

## Compensation-region Residual Attribution

### Code Fact

Masks were defined only from M1 baseline tensors:

```text
J1    = max_rgb(M1 clear_object_fullsh_raw) > 1
J95   = top 5 percent of max_rgb(M1 clear_object_fullsh_raw)
TAU90 = top 10 percent of mean_rgb(M1 tau_D)
TLOW  = min_rgb(M1 transmission) < 0.1
COMP  = J1 OR TAU90 OR TLOW
```

`Delta e_+ = max(||BND-GT|| - ||M1-GT||, 0)` was used for positive excess residual. Raw residual/mask tensors were saved under:

```text
outputs/bnd_rgb_tradeoff_diagnosis_20260809/raw_maps/
```

### Quantitative Result

| Scene | Mask | mask area | excess fraction | enrichment |
| --- | --- | ---: | ---: | ---: |
| JapaneseGradens | J1 | 0.054585 | 0.132783 | 2.432603 |
| JapaneseGradens | J95 | 0.050000 | 0.138795 | 2.775884 |
| JapaneseGradens | TAU90 | 0.100001 | 0.046736 | 0.467361 |
| JapaneseGradens | TLOW | 0.072798 | 0.037503 | 0.515162 |
| JapaneseGradens | COMP | 0.164674 | 0.191702 | 1.164130 |
| Panama | J1 | 0.050461 | 0.235189 | 4.660811 |
| Panama | J95 | 0.050000 | 0.232301 | 4.645972 |
| Panama | TAU90 | 0.100001 | 0.039603 | 0.396024 |
| Panama | TLOW | 0.010583 | 0.006342 | 0.599275 |
| Panama | COMP | 0.147974 | 0.273159 | 1.845995 |

### Quantitative Conclusion

Panama shows strong enrichment in M1 high-J masks, but the COMP union captures only `27.3%` of BND positive excess residual. Therefore it meets `PARTIAL_CONCENTRATION`, not the stricter `LOSS_CONCENTRATED_IN_LEGACY_COMPENSATION_REGIONS` rule that requires excess fraction at least `0.35`. JapaneseGradens does not meet the COMP concentration rule.

## Direct-vs-medium MSE Decomposition

### Code Fact

The renderer additive closure check passed:

```text
pred_image = direct_object_signal + rgb_medium
aggregate mean abs closure = 0.0
component decomposition closure ~ 1e-10
```

The decomposition used:

```text
DeltaMSE = C_direct + C_medium + C_cross
```

### Quantitative Result

| Scene | Delta MSE | C_direct | C_medium | C_cross | closure |
| --- | ---: | ---: | ---: | ---: | ---: |
| Curasao | -0.000089917 | 0.001055583 | 0.001859411 | -0.003004910 | -0.000000000116 |
| JapaneseGradens | +0.000256851 | 0.010491742 | 0.012295871 | -0.022530761 | +0.000000000013 |
| IUI3 | -0.000082476 | 0.002240373 | 0.001898542 | -0.004221391 | +0.000000000224 |
| Panama | +0.000118197 | 0.004330614 | 0.004166260 | -0.008378677 | +0.000000000226 |

### Quantitative Conclusion

JapaneseGradens and Panama are not cleanly direct-dominated, medium-dominated, or cross-term-dominated under the specified dominance threshold. Both have positive direct and medium terms with a negative cross term. Panama's direct and medium positive terms are close (`0.004330614` vs `0.004166260`).

## Hybrid Counterfactuals

### Code Fact

Hybrid images are diagnostic counterfactuals only:

```text
Hybrid-D = D_BND + M_M1
Hybrid-M = D_M1 + M_BND
```

They are not new model outputs and were not used for training.

### Quantitative Result

| Scene | Image | PSNR | SSIM | LPIPS |
| --- | --- | ---: | ---: | ---: |
| Curasao | M1 | 32.165164 | 0.956004 | 0.108141 |
| Curasao | BND | 32.188016 | 0.958728 | 0.109208 |
| Curasao | Hybrid-D | 27.349759 | 0.939122 | 0.127556 |
| Curasao | Hybrid-M | 25.657115 | 0.906277 | 0.130993 |
| JapaneseGradens | M1 | 24.756484 | 0.899500 | 0.120356 |
| JapaneseGradens | BND | 24.698505 | 0.896300 | 0.117283 |
| JapaneseGradens | Hybrid-D | 18.141029 | 0.850184 | 0.191748 |
| JapaneseGradens | Hybrid-M | 17.688232 | 0.786700 | 0.211062 |
| IUI3 | M1 | 30.874542 | 0.912143 | 0.174617 |
| IUI3 | BND | 30.963132 | 0.911644 | 0.177082 |
| IUI3 | Hybrid-D | 25.218780 | 0.897697 | 0.195856 |
| IUI3 | Hybrid-M | 25.237069 | 0.896244 | 0.197128 |
| Panama | M1 | 32.308910 | 0.949487 | 0.073979 |
| Panama | BND | 31.498353 | 0.948783 | 0.075521 |
| Panama | Hybrid-D | 23.076890 | 0.911594 | 0.113730 |
| Panama | Hybrid-M | 23.437222 | 0.887729 | 0.115309 |

## Sigmoid Jacobian Audit

### Code Fact

For BND current-view full SH output:

```text
c = sigmoid(s)
dc/ds = c * (1 - c)
```

The reported `1 / median(dc/ds)` is a diagnostic scale only. It is not a recommended learning-rate multiplier.

### Quantitative Result

| Scene | derivative mean | p10 | p50 | p90 | 1/p50 diagnostic |
| --- | ---: | ---: | ---: | ---: | ---: |
| Curasao | 0.178231 | 0.100795 | 0.186083 | 0.246269 | 5.378187 |
| JapaneseGradens | 0.192665 | 0.125717 | 0.202287 | 0.247614 | 4.943648 |
| IUI3 | 0.196884 | 0.114891 | 0.215866 | 0.248751 | 4.634133 |
| Panama | 0.199595 | 0.139400 | 0.211913 | 0.248027 | 4.721074 |

### Reasonable Inference

Boundary saturation was not large, but the sigmoid Jacobian substantially changes the appearance optimization geometry relative to legacy RGB-space SH output. This supports `SIGMOID_JACOBIAN_OPTIMIZATION_LIMIT` as a mechanism to test, without implying a specific LR multiplier.

## SH3 Capacity Audit

### Code Fact

For each current view:

```text
R_SH = ||c_full(v) - c_dc||
```

For BND, `c_dc` uses the same bounded-logit DC path: `sigmoid(s_dc)`.

### Quantitative Result

| Scene | Run | R_SH visible mean | R_SH visible p50 | R_SH visible p90 | R_SH visible p99 | visible fraction |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Curasao | M1 | 0.137312 | 0.102851 | 0.279592 | 0.608775 | 0.660354 |
| Curasao | BND | 0.040510 | 0.030586 | 0.084360 | 0.159826 | 0.666040 |
| JapaneseGradens | M1 | 0.122203 | 0.090502 | 0.257267 | 0.513388 | 0.638619 |
| JapaneseGradens | BND | 0.046284 | 0.033130 | 0.099280 | 0.211947 | 0.636769 |
| IUI3 | M1 | 0.104041 | 0.081000 | 0.203082 | 0.406496 | 0.519724 |
| IUI3 | BND | 0.050892 | 0.041542 | 0.099208 | 0.182153 | 0.516152 |
| Panama | M1 | 0.134024 | 0.100796 | 0.278947 | 0.542537 | 0.520143 |
| Panama | BND | 0.047079 | 0.035793 | 0.096999 | 0.196431 | 0.521005 |

### Quantitative Result

| Scene | Run | features_rest p50 | features_rest p90 | features_dc p50 | rest/DC p50 |
| --- | --- | ---: | ---: | ---: | ---: |
| Curasao | M1 | 0.370509 | 0.748067 | 1.450312 | 0.246776 |
| Curasao | BND | 0.543294 | 1.067100 | 6.513093 | 0.087139 |
| JapaneseGradens | M1 | 0.277982 | 0.596536 | 1.439644 | 0.206521 |
| JapaneseGradens | BND | 0.460871 | 0.968485 | 5.939174 | 0.083492 |
| IUI3 | M1 | 0.267724 | 0.556143 | 1.328448 | 0.215185 |
| IUI3 | BND | 0.512668 | 1.053030 | 4.900678 | 0.109100 |
| Panama | M1 | 0.303418 | 0.638526 | 1.452577 | 0.217622 |
| Panama | BND | 0.495278 | 1.041163 | 5.655890 | 0.092470 |

### Reasonable Inference

BND has larger raw `features_rest` norms but much smaller RGB-space full-vs-DC residual. This is consistent with bounded-logit geometry compressing RGB-space SH residual amplitude.

## No-step Gradient Audit

### Code Fact

The diagnostic performed forward + backward only. It did not call `optimizer.step()` and did not update parameters. The current medium implementation uses a shared tcnn `medium_mlp`; branch-specific parameter rows are not separable for tcnn, so the audit records shared `medium_mlp` parameter gradients and per-output tensor gradients for `medium_rgb`, `medium_bs`, and `medium_attn`.

### Quantitative Result

All 24 selected no-step gradient rows completed with `status=OK`.

| Scene | Run | features_dc grad/param | features_rest grad/param | medium_mlp grad/param | medium_rgb output grad | medium_bs output grad | medium_attn output grad | dL/ds over est dL/dc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Curasao | M1 | 1.229870e-06 | 1.229870e-06 | 7.966894e-03 | 2.670473e-06 | 5.848626e-07 | 1.940422e-07 | n/a |
| Curasao | BND | 1.747685e-07 | 1.747685e-07 | 4.775273e-03 | 1.447631e-06 | 1.357652e-06 | 2.895603e-07 | 0.159019 |
| JapaneseGradens | M1 | 8.073438e-07 | 8.073439e-07 | 3.507163e-03 | 2.589569e-06 | 6.346845e-07 | 2.535933e-07 | n/a |
| JapaneseGradens | BND | 3.559917e-07 | 3.559917e-07 | 2.084979e-03 | 1.282329e-06 | 1.522350e-06 | 5.581535e-07 | 0.165992 |
| IUI3 | M1 | 8.191255e-07 | 8.191255e-07 | 3.972393e-03 | 2.641113e-06 | 6.807299e-07 | 3.077586e-07 | n/a |
| IUI3 | BND | 1.206785e-07 | 1.206785e-07 | 3.706415e-03 | 2.285937e-06 | 1.347810e-06 | 3.976567e-07 | 0.145618 |
| Panama | M1 | 6.994669e-07 | 6.994669e-07 | 3.754216e-03 | 1.512604e-06 | 6.976777e-07 | 2.686521e-07 | n/a |
| Panama | BND | 1.984004e-07 | 1.984004e-07 | 2.375333e-03 | 5.639583e-07 | 9.848991e-07 | 4.155688e-07 | 0.164753 |

### Reasonable Inference

BND appearance parameter gradients per parameter are lower than M1 in all four scenes at the selected views. The retained-logit diagnostic ratio is also substantially below 1.0. This supports an optimization-geometry difference, not boundary saturation.

## Stratified Residual Audit

### Quantitative Result

For JapaneseGradens, positive excess residual was largest in the top luminance bin and nearest depth bin:

```text
GT luminance bin 4: 47.7% of positive excess residual
M1 luminance bin 4: 47.5% of positive excess residual
depth bin 0: 44.9% of positive excess residual
```

For Panama:

```text
GT luminance bin 4: 60.5% of positive excess residual
M1 luminance bin 4: 59.8% of positive excess residual
depth bin 0: 47.2% of positive excess residual
```

Full stratification file:

```text
outputs/bnd_rgb_tradeoff_diagnosis_20260809/bnd_stratified_residual_audit.csv
```

## Success-scene Controls

### Quantitative Result

Curasao and IUI3 preserve RGB safety while showing the same broad mechanism:

| Scene | dPSNR | dSSIM | dLPIPS | tau p90 reduction | J p99 reduction | sigmoid limit flag |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Curasao | +0.022852 | +0.002724 | +0.001067 | 0.352954 | 0.592797 | true |
| IUI3 | +0.088590 | -0.000499 | +0.002465 | 0.281192 | 0.252845 | true |

### Reasonable Inference

The sigmoid-geometry signal appears in both passing and failing scenes. It is therefore a mechanism-level constraint, not by itself a complete explanation of Panama's RGB loss.

## Panama Root-cause Analysis

### Quantitative Conclusion

```text
PANAMA_ROOT_CAUSE_PRIMARY = MIXED
```

Supporting numbers:

```text
dPSNR = -0.810558
dSSIM = -0.000704
dLPIPS = +0.001542
POSSIBLE_UNDERCONVERGENCE = false
COMP mask area = 0.147974
COMP excess fraction = 0.273159
COMP enrichment = 1.845995
J1 enrichment = 4.660811
J95 enrichment = 4.645972
C_direct = 0.004330614
C_medium = 0.004166260
C_cross = -0.008378677
sigmoid derivative median = 0.211913
R_SH visible p50 BND/M1 = 0.355105
```

### Reasonable Inference

Panama has partial legacy compensation-region concentration and strong high-J enrichment, but the COMP excess fraction is below the predefined concentration threshold. The MSE attribution is balanced between direct and medium positive terms, with a negative cross term. The strongest supported label is therefore `MIXED`, with secondary evidence for both `LEGACY_COMPENSATION_REMOVAL` and `SIGMOID_JACOBIAN_OPTIMIZATION_LIMIT`.

## JapaneseGradens Root-cause Analysis

### Quantitative Conclusion

```text
JAPANESE_ROOT_CAUSE_PRIMARY = MIXED
```

Supporting numbers:

```text
dPSNR = -0.057979
dSSIM = -0.003200
dLPIPS = -0.003073
POSSIBLE_UNDERCONVERGENCE = false
COMP mask area = 0.164674
COMP excess fraction = 0.191702
COMP enrichment = 1.164130
J1 enrichment = 2.432603
J95 enrichment = 2.775884
C_direct = 0.010491742
C_medium = 0.012295871
C_cross = -0.022530761
sigmoid derivative median = 0.202287
R_SH visible p50 BND/M1 = 0.366066
```

### Reasonable Inference

JapaneseGradens is not a Panama-style large PSNR failure. Its PSNR is close to M1 and LPIPS improves, while SSIM fails the gate. The residual is not concentrated in the COMP union, and component attribution is mixed.

## Final Classification

### Quantitative Conclusion

| Scene | MECHANISM_STILL_VALID | POSSIBLE_UNDERCONVERGENCE | LOSS_CONCENTRATED | PARTIAL_CONCENTRATION | DIRECT_DOMINATED | MEDIUM_DOMINATED | SIGMOID_JACOBIAN_LIMIT | ROOT_CAUSE_PRIMARY |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Curasao | true | false | true | false | false | true | true | MIXED |
| JapaneseGradens | true | false | false | false | false | false | true | MIXED |
| IUI3 | true | true | false | false | false | false | true | MIXED |
| Panama | true | false | false | true | false | false | true | MIXED |

### Reasonable Inference

BND continues to improve the decomposition proxies in all scenes. The RGB trade-off is not explained by a single cause across all scenes. The most actionable common signal is that bounded SH3 changes the appearance optimization geometry: derivative medians are about `0.186-0.216`, RGB-space SH residual amplitude is lower under BND, and no-step BND appearance gradients per parameter are lower than M1.

## Next Single-factor Experiment Recommendation

### Proposed Controlled Experiment

```text
NEXT_SINGLE_FACTOR_EXPERIMENT = BND appearance-optimizer parameterization-equivalence test
```

Definition:

```text
Keep BND model, renderer, loss, SH degree, medium, densification, pruning, opacity, seed, splits, and training length fixed.
Change only the optimizer scale for bounded appearance parameters/features.
Choose the tested scale by a short no-training or very short calibration audit, not by directly asserting that 1/median sigmoid derivative is the correct multiplier.
Run first on Panama, with Curasao or IUI3 as a control if one additional scene is allowed.
```

Reason:

```text
Panama under-convergence is not supported at 13k->15k.
Panama is not cleanly direct- or medium-dominated.
The most consistent cross-scene diagnostic signal is reduced BND appearance update geometry and reduced RGB-space SH residual amplitude.
```

This recommendation is a future experiment only. It was not run in this stage.

## Visual Assets

### Experimental Fact

Visual assets were generated for external/manual inspection only. No subjective clear-image correctness judgment was made.

Index:

```text
renders/bnd_rgb_tradeoff_diagnosis_20260809/VISUAL_COMPARE_INDEX.md
```

Four-scene sheets:

```text
renders/bnd_rgb_tradeoff_diagnosis_20260809/four_scene_rgb_tradeoff_summary.png
renders/bnd_rgb_tradeoff_diagnosis_20260809/four_scene_decomposition_summary.png
```

Per-scene sheets:

```text
renders/bnd_rgb_tradeoff_diagnosis_20260809/Curasao/worst_views_underwater_residual.png
renders/bnd_rgb_tradeoff_diagnosis_20260809/Curasao/worst_views_clear_tau_transmission.png
renders/bnd_rgb_tradeoff_diagnosis_20260809/Curasao/worst_views_compensation_mask_overlays.png
renders/bnd_rgb_tradeoff_diagnosis_20260809/Curasao/worst_views_direct_medium_components.png
renders/bnd_rgb_tradeoff_diagnosis_20260809/Curasao/worst_views_hybrid_counterfactuals.png

renders/bnd_rgb_tradeoff_diagnosis_20260809/JapaneseGradens/worst_views_underwater_residual.png
renders/bnd_rgb_tradeoff_diagnosis_20260809/JapaneseGradens/worst_views_clear_tau_transmission.png
renders/bnd_rgb_tradeoff_diagnosis_20260809/JapaneseGradens/worst_views_compensation_mask_overlays.png
renders/bnd_rgb_tradeoff_diagnosis_20260809/JapaneseGradens/worst_views_direct_medium_components.png
renders/bnd_rgb_tradeoff_diagnosis_20260809/JapaneseGradens/worst_views_hybrid_counterfactuals.png

renders/bnd_rgb_tradeoff_diagnosis_20260809/IUI3/worst_views_underwater_residual.png
renders/bnd_rgb_tradeoff_diagnosis_20260809/IUI3/worst_views_clear_tau_transmission.png
renders/bnd_rgb_tradeoff_diagnosis_20260809/IUI3/worst_views_compensation_mask_overlays.png
renders/bnd_rgb_tradeoff_diagnosis_20260809/IUI3/worst_views_direct_medium_components.png
renders/bnd_rgb_tradeoff_diagnosis_20260809/IUI3/worst_views_hybrid_counterfactuals.png

renders/bnd_rgb_tradeoff_diagnosis_20260809/Panama/worst_views_underwater_residual.png
renders/bnd_rgb_tradeoff_diagnosis_20260809/Panama/worst_views_clear_tau_transmission.png
renders/bnd_rgb_tradeoff_diagnosis_20260809/Panama/worst_views_compensation_mask_overlays.png
renders/bnd_rgb_tradeoff_diagnosis_20260809/Panama/worst_views_direct_medium_components.png
renders/bnd_rgb_tradeoff_diagnosis_20260809/Panama/worst_views_hybrid_counterfactuals.png
```

Render manifest:

```text
renders/bnd_rgb_tradeoff_diagnosis_20260809/manifest.json
renders/bnd_rgb_tradeoff_diagnosis_20260809/manifest.csv
```

## Output Files

### Experimental Fact

Primary machine-readable outputs:

```text
outputs/bnd_rgb_tradeoff_diagnosis_20260809/bnd_trajectory_audit.csv
outputs/bnd_rgb_tradeoff_diagnosis_20260809/bnd_trajectory_delta.csv
outputs/bnd_rgb_tradeoff_diagnosis_20260809/bnd_missing_checkpoints.csv
outputs/bnd_rgb_tradeoff_diagnosis_20260809/bnd_per_view_rgb_delta.csv
outputs/bnd_rgb_tradeoff_diagnosis_20260809/bnd_residual_enrichment.csv
outputs/bnd_rgb_tradeoff_diagnosis_20260809/bnd_component_mse_attribution.csv
outputs/bnd_rgb_tradeoff_diagnosis_20260809/bnd_hybrid_metrics.csv
outputs/bnd_rgb_tradeoff_diagnosis_20260809/bnd_sigmoid_jacobian_audit.csv
outputs/bnd_rgb_tradeoff_diagnosis_20260809/bnd_sh_capacity_audit.csv
outputs/bnd_rgb_tradeoff_diagnosis_20260809/bnd_features_parameter_audit.csv
outputs/bnd_rgb_tradeoff_diagnosis_20260809/bnd_no_step_gradient_audit.csv
outputs/bnd_rgb_tradeoff_diagnosis_20260809/bnd_stratified_residual_audit.csv
outputs/bnd_rgb_tradeoff_diagnosis_20260809/bnd_checkpoint_audit.csv
outputs/bnd_rgb_tradeoff_diagnosis_20260809/bnd_rgb_tradeoff_final_summary.csv
outputs/bnd_rgb_tradeoff_diagnosis_20260809/manifest.json
```

JSON counterparts were also written for the same audits.

## Verification

### Experimental Fact

The diagnostic script was compiled before final execution:

```text
/opt/anaconda3/envs/water_splatting/bin/python -m py_compile scripts/diagnostics/diagnose_bnd_rgb_tradeoff.py
```

No training or optimizer step was executed by the diagnostic.
