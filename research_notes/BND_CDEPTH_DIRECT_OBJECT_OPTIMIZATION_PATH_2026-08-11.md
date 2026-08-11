# BND-CDEPTH Continuous Direct-Object Optimization Path Audit

## CODE FACT

- Repo branch at audit start: `research/m1-bounded-intrinsic`.
- Start HEAD: `bdde75883282e0f3926cd1e3d45236bb06824ddb`.
- This audit is read-only with respect to training artifacts: no persistent optimizer step, scheduler step, densification, pruning, opacity reset, checkpoint save, or renderer physics change was used.
- Current trainable groups from `WaterSplattingModel.get_param_groups()` are `means`, `scales`, `quats`, `features_dc`, `features_rest`, `opacities`, `medium_mlp`, and `direction_encoding`.
- `camera_opt` is configured in the method config but is not returned by the current model param-group implementation.
- Nerfstudio training order is zero-grad, forward/loss, backward, optimizer step, then scheduler step.
- Analytic Adam virtual updates use checkpoint current LR and saved `step`, `exp_avg`, and `exp_avg_sq`; medium groups apply the configured `max_norm=0.001` clipping.

## CONFIG FACT

- Formal Panama train camera bank: `MTN_1538`, `MTN_1541`, `MTN_1540`, `MTN_1534`, `MTN_1535`, `MTN_1536`, `MTN_1533`, `MTN_1542`, `MTN_1537`, `MTN_1532`, `MTN_1546`, `MTN_1543`, `MTN_1544`, `MTN_1545`, `MTN_1548`.
- Formal eval views: `MTN_1529`, `MTN_1539`, `MTN_1547`.
- Outcome masks are post-hoc diagnostics: `M1_HIGH_J` is final M1 accumulation > 0.01 and final M1 clear-object full-SH raw max RGB > 1.0; `HJ_GAIN/HJ_HARM` are defined from final K1 vs CDEPTH RGB MSE change inside `M1_HIGH_J`.

## EXPERIMENTAL FACT

- Optimizer-aware virtual step valid: `True`.
- Virtual update equivalence: `{'group': 'means', 'sample_count': 1024, 'max_abs_difference': 2.3770189727656543e-07, 'tolerance': 1e-06, 'pass': True, 'semantics': 'isolated tensor torch.optim.Adam.step compared with analytic Adam delta; no model/checkpoint mutation'}`.
- Optimizer state tier: BND-K1 1k/3k/5k/8k were all `EXACT_OPTIMIZER_STATE` for `means`, `scales`, `quats`, `features_dc`, `features_rest`, `opacities`, `medium_mlp`, and `direction_encoding`.
- Pre-backward forward equivalence passed for all 45 checkpoint/train-camera branches. Max recorded forward difference over `pred`, `D`, `B`, `depth`, `J`, `T`, `tau`, and `alpha` was `3.725290298461914e-09`.
- Future HJ_GAIN alignment: `False`.
- Direct response dominant: `False` with pre-recovery HJ_GAIN ratio `1.4452890826922307`.
- Training camera robust: `True`.
- Group additivity approximate: `True` with mean nonadditivity `0.05275598874108659`.
- Persistent safety passed: checkpoint `mtime`, `size`, and `sha256` were unchanged; all in-memory virtual-parameter restore checks had max abs delta `0.0`.

### Historical Direct/Medium Trajectory

| Step | Region | RGB MSE Gain | mean abs Delta D | mean abs Delta B | D/B Ratio |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1000 | HJ_GAIN | 0.001351 | 0.080832 | 0.025654 | 3.193865 |
| 1000 | HJ_HARM | 0.000177 | 0.083766 | 0.030259 | 2.857388 |
| 3000 | HJ_GAIN | -0.002110 | 0.097133 | 0.013737 | 7.348453 |
| 3000 | HJ_HARM | -0.002703 | 0.095508 | 0.015359 | 6.786219 |
| 5000 | HJ_GAIN | 0.001789 | 0.045946 | 0.005053 | 10.340723 |
| 5000 | HJ_HARM | -0.000814 | 0.044068 | 0.005456 | 9.183120 |
| 8000 | HJ_GAIN | 0.002264 | 0.023864 | 0.002884 | 9.864138 |
| 8000 | HJ_HARM | -0.000686 | 0.020886 | 0.002801 | 8.391344 |

### Raw Gradient Increment Ratio

Mean over 15 train-camera branches.

| Step | means | scales | quats | opacities | features_dc | features_rest | medium_mlp |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 0.358469 | 20.010056 | 16.592679 | 11.347620 | 2.20e-07 | 2.15e-07 | 3.64e-07 |
| 3000 | 0.055673 | 3.571407 | 3.163398 | 2.435394 | 1.98e-07 | 1.97e-07 | 0 |
| 5000 | 0.020627 | 1.681822 | 3.250914 | 0.610550 | 8.08e-08 | 8.02e-08 | 0 |

### Optimizer-Aware Update Increment Ratio

Mean over 15 train-camera branches after saved Adam state and group LR.

| Step | means | scales | quats | opacities | features_dc | features_rest | medium_mlp |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 0.088492 | 0.114666 | 0.088157 | 0.103613 | 2.29e-08 | 1.99e-07 | 1.56e-06 |
| 3000 | 0.050074 | 0.067607 | 0.052547 | 0.056636 | 2.47e-08 | 8.54e-08 | 0 |
| 5000 | 0.036005 | 0.047880 | 0.034748 | 0.040071 | 1.72e-08 | 1.60e-08 | 0 |

### Adam / Momentum Interaction

Mean cosine between depth-increment gradient and saved Adam `exp_avg`.

| Step | means | scales | quats | opacities |
| ---: | ---: | ---: | ---: | ---: |
| 1000 | -0.001242 | 0.106578 | 0.023969 | 0.661654 |
| 3000 | 0.000207 | -0.118526 | 0.182729 | -0.031820 |
| 5000 | -0.000394 | 0.020060 | 0.243509 | 0.011025 |

### Full One-Step Virtual Response

Mean over 15 train-camera branches and 3 eval views.

| Step | Region | Virtual RGB Gain | mean abs D Response | mean abs B Response | D/B Ratio | mean Delta tau |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1000 | HJ_GAIN | 7.682625e-06 | 1.683392e-04 | 1.025736e-04 | 1.740669 | 2.269685e-04 |
| 1000 | HJ_HARM | 6.367515e-06 | 1.770586e-04 | 1.043666e-04 | 1.753796 | 2.232175e-04 |
| 3000 | HJ_GAIN | 9.197328e-07 | 7.265575e-05 | 6.568251e-05 | 1.149909 | 6.227511e-05 |
| 3000 | HJ_HARM | 1.128597e-06 | 6.890140e-05 | 5.846987e-05 | 1.170503 | 5.092187e-05 |
| 5000 | HJ_GAIN | 3.899137e-08 | 2.888333e-06 | 2.555491e-06 | 1.347885 | 2.105636e-06 |
| 5000 | HJ_HARM | 3.503842e-08 | 2.689101e-06 | 2.183744e-06 | 1.332679 | 1.719665e-06 |

### Future HJ Alignment

| Step | Eval View | HJ_GAIN Gain | HJ_HARM Gain | Aligned |
| ---: | --- | ---: | ---: | --- |
| 1000 | MTN_1529 | -8.348127e-06 | -8.784731e-06 | False |
| 1000 | MTN_1539 | 5.124013e-06 | 5.221367e-06 | False |
| 1000 | MTN_1547 | 2.627199e-05 | 2.266591e-05 | True |
| 3000 | MTN_1529 | -3.720323e-06 | -2.547602e-06 | False |
| 3000 | MTN_1539 | 5.470216e-06 | 4.541626e-06 | True |
| 3000 | MTN_1547 | 1.009305e-06 | 1.391768e-06 | False |
| 5000 | MTN_1529 | -3.303091e-08 | -4.271666e-08 | False |
| 5000 | MTN_1539 | 9.275973e-08 | 8.052836e-08 | True |
| 5000 | MTN_1547 | 5.724529e-08 | 6.730358e-08 | False |

The formal future-HJ alignment gate requires at least 2/3 aligned eval views at 1k or 3k plus pooled same direction. This gate did not pass.

### Training-Camera Robustness

| Step | HJ-positive Cameras | Rate |
| ---: | ---: | ---: |
| 1000 | 12 / 15 | 0.800000 |
| 3000 | 9 / 15 | 0.600000 |
| 5000 | 13 / 15 | 0.866667 |

### Group-Isolated HJ_GAIN Response

Mean over 15 train-camera branches and 3 eval views.

| Step | Group | Virtual RGB Gain | mean abs D Response | mean abs B Response | D/B Ratio |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1000 | means | 7.475416e-07 | 1.775030e-05 | 6.582827e-06 | 2.529111 |
| 1000 | scales | 1.302279e-06 | 7.173702e-05 | 4.739573e-05 | 1.622989 |
| 1000 | quats | 2.894964e-07 | 2.384285e-05 | 1.637161e-05 | 1.512651 |
| 1000 | opacities | 5.583382e-06 | 8.346226e-05 | 4.727489e-05 | 2.016988 |
| 3000 | means | 1.357661e-07 | 1.388380e-05 | 7.422799e-06 | 1.743301 |
| 3000 | scales | 8.377764e-08 | 3.841390e-05 | 3.602895e-05 | 1.090303 |
| 3000 | quats | -1.655685e-08 | 1.976384e-05 | 1.767620e-05 | 1.149114 |
| 3000 | opacities | 7.197261e-07 | 1.812514e-05 | 1.796268e-05 | 1.100692 |
| 5000 | means | 2.980232e-09 | 4.573920e-07 | 2.403715e-07 | 2.437009 |
| 5000 | scales | 2.756715e-08 | 1.814436e-06 | 1.728403e-06 | 1.281923 |
| 5000 | quats | -3.849467e-09 | 6.275844e-07 | 5.347643e-07 | 1.387053 |
| 5000 | opacities | 1.320408e-08 | 8.430038e-07 | 7.314622e-07 | 1.355350 |

### Physical Update

Mean over 15 train-camera branches.

| Step | Group | Physical Metric | Mean | P90 | P99 |
| ---: | --- | --- | ---: | ---: | ---: |
| 1000 | means | world-space displacement norm | 1.533218e-06 | 1.866793e-06 | 2.839440e-05 |
| 1000 | scales | activated exp-scale relative abs delta | 5.581576e-05 | 7.305443e-05 | 1.110034e-03 |
| 1000 | quats | normalized quaternion angle radians | 2.525854e-04 | 9.765625e-04 | 1.208375e-03 |
| 1000 | opacities | sigmoid-opacity abs delta | 4.244556e-05 | 2.584457e-05 | 8.469780e-04 |
| 3000 | means | world-space displacement norm | 3.399531e-07 | 1.833044e-07 | 5.793250e-06 |
| 3000 | scales | activated exp-scale relative abs delta | 1.737868e-05 | 9.289418e-06 | 3.375708e-04 |
| 3000 | quats | normalized quaternion angle radians | 2.536089e-04 | 9.765625e-04 | 1.196040e-03 |
| 3000 | opacities | sigmoid-opacity abs delta | 1.094001e-05 | 4.199147e-06 | 1.680374e-04 |
| 5000 | means | world-space displacement norm | 9.822256e-08 | 9.435342e-09 | 1.039067e-06 |
| 5000 | scales | activated exp-scale relative abs delta | 4.323817e-06 | 4.510234e-07 | 4.638846e-05 |
| 5000 | quats | normalized quaternion angle radians | 2.618406e-04 | 9.765625e-04 | 1.196040e-03 |
| 5000 | opacities | sigmoid-opacity abs delta | 4.892912e-06 | 7.251898e-07 | 3.906886e-05 |

## QUANTITATIVE RESULT

- Continuous path classification: `LOCAL_RESPONSE_WITHOUT_FUTURE_ALIGNMENT`.
- Dominant group classification: `NOT_EVALUABLE`.
- Group scores: `[{'group': 'means', 'gain': 2.9542931803950557e-07, 'fraction': 0.102563575674629}, {'group': 'scales', 'gain': 4.712078306410048e-07, 'fraction': 0.1635882326004077}, {'group': 'quats', 'gain': 8.969671196407743e-08, 'fraction': 0.031139819048232917}, {'group': 'opacities', 'gain': 2.1054374950903432e-06, 'fraction': 0.7309403118448236}]`.

## INFERENCE

- The local optimizer-aware measurements are evidence about one-step response from fixed K1 checkpoints. They do not by themselves prove the final 15k RGB gain causal mechanism.
- Renderer branch responses use true `direct_object_signal` and `rgb_medium` outputs; no direct-object GT is assumed.

## HYPOTHESIS

- If the local response and historical trajectory are compatible, the next causal test should isolate the continuous path in training rather than sweep depth weights.
- Next single-factor recommendation: `read-only optimization-basin diagnostic`.

## Artifacts

- Output manifest: `/mnt/new/home_old/ycy/water-splatting-refactor/outputs/bnd_cdepth_direct_path_panama_20260811/manifest.json`.
- Visual manifest: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_cdepth_direct_path_panama_20260811/manifest.json`.
- Visual index: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_cdepth_direct_path_panama_20260811/VISUAL_COMPARE_INDEX.md`.
