# BND Appearance Optimizer Parameterization-Equivalence Test

Date: 2026-08-09

## 1. Motivation

### HYPOTHESIS

The tested hypothesis was that `bounded_sh3` preserves the decomposition mechanism but may reduce effective RGB-space SH3 appearance capacity under the legacy Gaussian appearance optimizer scale.

The controlled variable was:

```text
appearance_lr_scale = k
k in {1, 2, 4}
```

Only `features_dc` and `features_rest` LR trajectories were scaled. The original `features_dc : features_rest` LR ratio was preserved.

## 2. Prior BND Findings

### EXPERIMENTAL FACT

Prior Panama references:

| Run | PSNR | SSIM | LPIPS | tau p90 | J p99 | P(J>1) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 32.308910 | 0.949487 | 0.073979 | 1.594246 | 1.293632 | 0.037755 |
| BND-K1 | 31.498353 | 0.948783 | 0.075521 | 0.909361 | 0.829595 | 0.000000 |

The BND-K1 PSNR gap to M1 was `0.810558 dB`. BND-K1 retained the low-optical-depth/high-J-tail mechanism but failed the RGB safety gate because `Delta PSNR = -0.810558 dB`.

Prior trade-off diagnosis labeled Panama as `MIXED`; simple under-convergence was not supported.

## 3. Optimization-Geometry Hypothesis

### HYPOTHESIS

For bounded SH3:

```text
s(v) = full active SH3 logits
c(v) = sigmoid(s(v))
dc/ds = c(1-c)
```

The question was whether the bounded parameterization reduces actual RGB-space SH utilization unless the appearance optimizer strength is increased.

## 4. Optimizer Implementation Audit

### CODE FACT

Optimizer audit output:

```text
outputs/bnd_aopt_equivalence_panama_20260809/aopt_optimizer_audit.csv
outputs/bnd_aopt_equivalence_panama_20260809/aopt_optimizer_audit.json
```

Appearance optimizer groups:

| Group | Optimizer | base LR | final LR | Scheduler |
| --- | --- | ---: | ---: | --- |
| features_dc | Adam | 0.0025 | 0.0025 | ExponentialDecayScheduler |
| features_rest | Adam | 0.000125 | 0.000125 | ExponentialDecayScheduler |

Other optimizer groups were not scaled: `means`, `scales`, `quats`, `opacities`, `camera_opt`, `medium_mlp`, and `direction_encoding`.

### CODE FACT

Implementation:

```text
water_splatting/water_splatting.py
```

New config:

```text
appearance_lr_scale: float = 1.0
appearance_audit_log_dir: Optional[str] = None
appearance_lr_audit_steps
appearance_update_audit_steps
```

The callback scales only live optimizer param groups and scheduler `base_lrs` for `features_dc` and `features_rest`.

## 5. Experimental Definition

### EXPERIMENTAL FACT

Scene:

```text
Panama
```

Fixed settings:

```text
SH degree = 3
medium_context_mode = dir_xy_camera
b_inf_mode = tied
infinite_water_enabled = False
seed = 42
original renderer/loss/densification/pruning/opacity/medium settings
```

Runs:

| Run | Parameterization | appearance_lr_scale | Source |
| --- | --- | ---: | --- |
| M1 | legacy SH3 | 1.0 | reused historical checkpoint |
| K1 | bounded_sh3 | 1.0 | reused historical BND checkpoint |
| K2 | bounded_sh3 | 2.0 | trained from scratch |
| K4 | bounded_sh3 | 4.0 | trained from scratch |

K2/K4 checkpoint output:

```text
outputs/bnd_aopt_equivalence_panama_20260809/bnd_aopt_panama_seed42_k2_step0_to_15000/water-splatting/20260809_bnd_aopt_k2/
outputs/bnd_aopt_equivalence_panama_20260809/bnd_aopt_panama_seed42_k4_step0_to_15000/water-splatting/20260809_bnd_aopt_k4/
```

## 6. K1 Reference Verification

### EXPERIMENTAL FACT

K1 optimizer-equivalence audit:

```text
outputs/bnd_aopt_equivalence_panama_20260809/aopt_k1_equivalence_audit.json
```

Result:

```text
K1_OPTIMIZER_EQUIVALENCE = true
loss_abs_diff = 0.0
appearance_gradient_max_abs_diff = 3.055902e-10
appearance_post_step_max_abs_diff = 1.192093e-07
```

The equivalence scope is appearance optimizer groups. Non-appearance post-step differences were logged separately because independent CUDA/tcnn backward passes are not bit-identical.

### EXPERIMENTAL FACT

K2 vs K4 initialization audit:

```text
outputs/bnd_aopt_equivalence_panama_20260809/aopt_initialization_audit.csv
```

All audited initialization hashes matched for:

```text
means, features_dc, features_rest, opacities, scales, quats, medium_parameters
```

Initial bounded RGB error:

```text
mean = 5.289847e-08
p95  = 2.064765e-07
max  = 2.064765e-07
```

## 7. K2 Trajectory

### QUANTITATIVE RESULT

| Step | PSNR | SSIM | LPIPS | tau p90 | J p99 | R_SH visible p50 | Gaussians |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 23.121574 | 0.689398 | 0.429553 | 0.812261 | 0.822432 | 0.000125 | 60770 |
| 3000 | 20.974568 | 0.591834 | 0.439706 | 0.916511 | 0.757865 | 0.013314 | 599405 |
| 5000 | 26.508390 | 0.824384 | 0.256155 | 1.016019 | 0.793669 | 0.027982 | 1022993 |
| 8000 | 31.038760 | 0.944648 | 0.086428 | 0.960060 | 0.831662 | 0.035636 | 1226446 |
| 10000 | 31.216770 | 0.946468 | 0.082354 | 0.963612 | 0.838425 | 0.038843 | 1218660 |
| 13000 | 31.530948 | 0.949704 | 0.075475 | 0.863300 | 0.812362 | 0.042848 | 1185692 |
| 15000 | 31.529617 | 0.948871 | 0.076573 | 0.824820 | 0.811879 | 0.045156 | 1179946 |

## 8. K4 Trajectory

### QUANTITATIVE RESULT

| Step | PSNR | SSIM | LPIPS | tau p90 | J p99 | R_SH visible p50 | Gaussians |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 23.207272 | 0.689363 | 0.428444 | 1.065473 | 0.821769 | 0.000242 | 58476 |
| 3000 | 21.041851 | 0.586776 | 0.442049 | 1.026044 | 0.759039 | 0.017888 | 624064 |
| 5000 | 26.517975 | 0.823464 | 0.255256 | 1.179237 | 0.808207 | 0.037592 | 1022593 |
| 8000 | 30.981345 | 0.943060 | 0.085650 | 1.089604 | 0.855212 | 0.045972 | 1214350 |
| 10000 | 31.066493 | 0.944462 | 0.083084 | 1.068915 | 0.865333 | 0.049743 | 1214626 |
| 13000 | 31.472016 | 0.947425 | 0.077081 | 0.863756 | 0.821279 | 0.054567 | 1181720 |
| 15000 | 31.393145 | 0.946374 | 0.078388 | 0.817702 | 0.827589 | 0.057865 | 1175279 |

## 9. RGB Recovery

### QUANTITATIVE RESULT

| Run | PSNR | dPSNR vs M1 | gain vs K1 | gap recovery | RGB safety |
| --- | ---: | ---: | ---: | ---: | --- |
| M1 | 32.308910 | 0.000000 | - | - | PASS |
| K1 | 31.498353 | -0.810558 | - | - | FAIL |
| K2 | 31.529617 | -0.779293 | +0.031265 | 0.038572 | FAIL |
| K4 | 31.393145 | -0.915765 | -0.105207 | -0.129796 | FAIL |

### QUANTITATIVE CONCLUSION

K2 recovered only `3.857%` of the K1-to-M1 PSNR gap. K4 moved in the opposite direction for PSNR relative to K1. Neither candidate passed RGB safety.

## 10. Decomposition Retention

### QUANTITATIVE RESULT

| Run | tau p90 | tau benefit retention | J p99 | J p99 retention | P(T<0.1) | P(J>1) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 1.594246 | - | 1.293632 | - | 0.007454 | 0.037755 |
| K1 | 0.909361 | 1.000000 | 0.829595 | 1.000000 | 0.000477 | 0.000000 |
| K2 | 0.824820 | 1.123439 | 0.811879 | 1.038179 | 0.000000 | 0.000000 |
| K4 | 0.817702 | 1.133833 | 0.827589 | 1.004325 | 0.000000 | 0.000000 |

### QUANTITATIVE CONCLUSION

K2 and K4 retained the low-optical-depth/high-J-tail reduction relative to K1 by the predefined tau/J retention metrics.

## 11. Boundary Saturation

### QUANTITATIVE RESULT

Final boundary escape:

| Run | P(c>0.99) | P(|s|>5) | sigmoid derivative p50 | Boundary escape |
| --- | ---: | ---: | ---: | --- |
| K1 | 0.019376 | 0.018972 | 0.211913 | false |
| K2 | 0.018404 | 0.017967 | 0.211765 | false |
| K4 | 0.019439 | 0.018994 | 0.210011 | false |

### QUANTITATIVE CONCLUSION

No final candidate crossed the boundary escape gate (`P(c>0.99)>0.05` or `P(|s|>5)>0.05`).

## 12. SH Capacity Recovery

### QUANTITATIVE RESULT

| Run | R_SH visible p50 | ratio vs M1 | recovery over K1 |
| --- | ---: | ---: | ---: |
| M1 | 0.100796 | 1.000000 | 2.816070 |
| K1 | 0.035793 | 0.355105 | 1.000000 |
| K2 | 0.045156 | 0.447997 | 1.261592 |
| K4 | 0.057865 | 0.574075 | 1.616635 |

### QUANTITATIVE CONCLUSION

Increasing appearance LR scale increased RGB-space SH utilization. K2 increased `R_SH visible p50` by `26.2%` over K1, and K4 increased it by `61.7%` over K1.

## 13. Parameter Update Audit

### CODE FACT

K2/K4 update logs:

```text
logs/bnd_aopt_equivalence_panama_20260809/bnd_aopt_panama_seed42_k2_step0_to_15000_20260809_bnd_aopt_k2/aopt_parameter_updates.jsonl
logs/bnd_aopt_equivalence_panama_20260809/bnd_aopt_panama_seed42_k4_step0_to_15000_20260809_bnd_aopt_k4/aopt_parameter_updates.jsonl
outputs/bnd_aopt_equivalence_panama_20260809/aopt_parameter_updates.csv
```

K1 update trajectory is unavailable because the reused historical K1 checkpoint was trained before AOPT update-audit JSONL logging existed.

### QUANTITATIVE RESULT

At the audited steps, K4 produced larger appearance update norms than K2 in the expected direction. For example:

| Step | Run | features_dc update L2 | features_rest update L2 |
| ---: | --- | ---: | ---: |
| 5000 | K2 | 3.702117 | 0.734043 |
| 5000 | K4 | 7.291474 | 1.443445 |
| 10000 | K2 | 4.533096 | 0.896521 |
| 10000 | K4 | 9.241035 | 1.819991 |
| 14999 | K2 | 1.697219 | 0.323302 |
| 14999 | K4 | 3.567555 | 0.673317 |

## 14. Medium Redistribution

### QUANTITATIVE RESULT

Final medium/backscatter aggregate:

| Run | beta_D mean | beta_B mean | T mean |
| --- | ---: | ---: | ---: |
| M1 | 0.396922 | 0.296831 | 0.384434 |
| K1 | 0.143104 | 0.084423 | 0.685939 |
| K2 | 0.144634 | 0.085026 | 0.689160 |
| K4 | 0.144560 | 0.095230 | 0.688092 |

### QUANTITATIVE CONCLUSION

K2 stayed close to K1 in medium/backscatter aggregate values. K4 had higher `beta_B mean` than K1/K2 but did not lose tau benefit retention or trigger boundary escape.

## 15. Per-view Analysis

### QUANTITATIVE RESULT

| View | Run | PSNR | dPSNR vs M1 | dPSNR vs K1 |
| --- | --- | ---: | ---: | ---: |
| MTN_1539 | K1 | 31.089798 | -0.252670 | 0.000000 |
| MTN_1539 | K2 | 30.970440 | -0.372028 | -0.119358 |
| MTN_1539 | K4 | 30.648653 | -0.693815 | -0.441145 |
| MTN_1529 | K1 | 32.304523 | -0.547436 | 0.000000 |
| MTN_1529 | K2 | 32.375992 | -0.475967 | +0.071468 |
| MTN_1529 | K4 | 32.252956 | -0.599003 | -0.051567 |
| MTN_1547 | K1 | 31.100737 | -1.631567 | 0.000000 |
| MTN_1547 | K2 | 31.242420 | -1.489883 | +0.141684 |
| MTN_1547 | K4 | 31.277826 | -1.454477 | +0.177090 |

### QUANTITATIVE CONCLUSION

K2 improved PSNR over K1 on 2 of 3 eval views and declined on 1 of 3. K4 improved PSNR over K1 on 1 of 3 eval views and declined on 2 of 3.

## 16. Residual Enrichment

### CODE FACT

Masks were defined from M1 only:

```text
J1 = max_rgb(M1 clear_object_fullsh_raw) > 1
COMP = J1 OR top-10%-tau OR min-transmission<0.1
```

### QUANTITATIVE RESULT

Aggregate positive excess residual enrichment:

| Run | Mask | mask area | excess fraction | enrichment |
| --- | --- | ---: | ---: | ---: |
| K1 | J1 | 0.050461 | 0.235189 | 4.820624 |
| K1 | COMP | 0.147974 | 0.273159 | 1.806848 |
| K2 | J1 | 0.050461 | 0.236856 | 4.898472 |
| K2 | COMP | 0.147974 | 0.263337 | 1.728791 |
| K4 | J1 | 0.050461 | 0.207903 | 4.276011 |
| K4 | COMP | 0.147974 | 0.235324 | 1.549770 |

## 17. Final Classification

### QUANTITATIVE CONCLUSION

Gate results:

```text
K2 STRONG_PARAMETERIZATION_RECOVERY = false
K2 PARTIAL_PARAMETERIZATION_RECOVERY = false
K4 STRONG_PARAMETERIZATION_RECOVERY = false
K4 PARTIAL_PARAMETERIZATION_RECOVERY = false
K4_OVER_OPTIMIZED = true
```

### INFERENCE

The optimizer-scale hypothesis is partially supported for the mechanism metric `R_SH`: increasing appearance LR scale did increase RGB-space SH utilization.

The hypothesis is not supported as the primary explanation for the Panama RGB gap, because K2 recovered only `0.031265 dB` over K1 (`3.857%` of the PSNR gap), and K4 reduced RGB metrics despite further increasing SH capacity.

Best safe AOPT candidate by the predefined ordering is K2, because it preserves decomposition benefit, avoids boundary escape, and has higher PSNR than K4. This does not make K2 a successful RGB recovery candidate.

## 18. Recommendation

### REASONABLE INFERENCE

Do not expand `appearance_lr_scale` into a new broad sweep at this stage. The controlled K2/K4 test shows that stronger appearance optimization can recover part of the SH-capacity proxy, but it does not recover the Panama RGB gap under the predefined gates.

Future work should treat appearance optimizer scale as a secondary tuning axis, not the main fix for the Panama BND trade-off.

## Visual Assets

### EXPERIMENTAL FACT

Visual outputs:

```text
renders/bnd_aopt_equivalence_panama_20260809/contact_sheet_underwater_m1_k1_k2_k4.png
renders/bnd_aopt_equivalence_panama_20260809/contact_sheet_clear_raw_m1_k1_k2_k4.png
renders/bnd_aopt_equivalence_panama_20260809/contact_sheet_direct_object_signal_m1_k1_k2_k4.png
renders/bnd_aopt_equivalence_panama_20260809/contact_sheet_transmission_m1_k1_k2_k4.png
renders/bnd_aopt_equivalence_panama_20260809/contact_sheet_tau_d_m1_k1_k2_k4.png
renders/bnd_aopt_equivalence_panama_20260809/contact_sheet_underwater_abs_residual_m1_k1_k2_k4.png
renders/bnd_aopt_equivalence_panama_20260809/contact_sheet_saturation_mask_k1_k2_k4.png
renders/bnd_aopt_equivalence_panama_20260809/VISUAL_COMPARE_INDEX.md
renders/bnd_aopt_equivalence_panama_20260809/manifest.json
```

Visual assets use eval views:

```text
MTN_1539
MTN_1529
MTN_1547
```

No subjective clear-image correctness judgment was made.

## Output Files

### EXPERIMENTAL FACT

Primary output directory:

```text
outputs/bnd_aopt_equivalence_panama_20260809/
```

Key files:

```text
aopt_optimizer_audit.json/csv
aopt_initialization_audit.json/csv
aopt_k1_equivalence_audit.json/csv
aopt_lr_trajectory.json/csv
aopt_training_trajectory.json/csv
aopt_rgb_metrics.json/csv
aopt_decomposition_metrics.json/csv
aopt_sh_capacity.json/csv
aopt_boundary_saturation.json/csv
aopt_parameter_updates.json/csv
aopt_medium_redistribution.json/csv
aopt_per_view_metrics.json/csv
aopt_residual_enrichment.json/csv
aopt_final_summary.json/csv
manifest.json
```
