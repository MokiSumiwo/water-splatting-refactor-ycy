# BND-CDEPTH Early-Off Final Experiment

## Motivation

CODE FACT: This is the final CDEPTH GO/CLOSE experiment for the current bounded intrinsic appearance study.

EXPERIMENTAL QUESTION: After the early CDEPTH state has already formed by step 3k, is continued coarse-depth supervision still necessary from 3k to the final 15k endpoint?

PRE-REGISTERED DECISION RULE: `CDEPTH_DECISION = KEEP` only if the formal classification is `EARLY_GUIDANCE_PARETO_IMPROVEMENT`. All other valid classifications close the CDEPTH line for the current study.

## Repository State

CODE FACT:

- Branch: `research/m1-bounded-intrinsic`
- Start HEAD: `f83781b38ad46f293c796add0723cd2f51998a3f`
- Start commit: `f83781b Test fixed-population CDEPTH rollout`
- Added runner: `scripts/diagnostics/run_bnd_cdepth_early_off.py`
- Added shell entrypoint: `scripts/experiments/bnd_cdepth_early_off_panama_3k_to_15k.sh`
- No production renderer, model, densification, optimizer, scheduler, or loss source was modified.

## Why This Is The Final CDEPTH Experiment

EXPERIMENTAL FACT: Previous CDEPTH experiments showed partial positive RGB recovery but did not establish a Pareto-complete solution.

EXPERIMENTAL FACT: Existing formal Panama context:

| Run | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| M1 | 32.308910 | 0.949487 | 0.073979 |
| BND-K1 | 31.498353 | 0.948783 | 0.075521 |
| Historical Full BND-CDEPTH | 31.753299 | 0.946292 | 0.080931 |

QUANTITATIVE RESULT: Historical Full BND-CDEPTH improves PSNR over K1 by `+0.254946 dB`, but worsens SSIM by `-0.002492` and LPIPS by `+0.005410`.

## Mechanisms Already Rejected Or Weakened

EXPERIMENTAL FACT: The current CDEPTH research line had already weakened or rejected the following explanations before this final test:

- Better selected pseudo-depth geometry: not supported.
- Final finer projected footprint: evidence against.
- Final higher local Gaussian density: evidence against.
- Robust final alpha / coverage rebalance: weak.
- NEW_LOW_T causes perceptual harm: not supported.
- Edge-localized harm: not supported.
- Densification-trigger redistribution: insufficient as a primary explanation.
- One-step continuous optimizer response: local response without future alignment.
- K1@3k fixed-topology RGB+depth rollout: no meaningful high-J recovery.

REASONABLE INFERENCE: The only remaining CDEPTH hypothesis worth one final test was early-basin formation from step 0 to 3k.

## CDEPTH@3k Start State

CONFIG FACT:

- Scene: `Panama`
- Seed: `42`
- Start checkpoint: `/mnt/new/home_old/ycy/water-splatting-refactor/outputs/bnd_cdepth_panama_20260811/panama_bnd_cdepth_seed42_step0_to_15000/water-splatting/20260811_bnd_cdepth/nerfstudio_models/step-000003000.ckpt`
- Config: `/mnt/new/home_old/ycy/water-splatting-refactor/outputs/bnd_cdepth_panama_20260811/panama_bnd_cdepth_seed42_step0_to_15000/water-splatting/20260811_bnd_cdepth/config.yml`
- Checkpoint SHA256: `951e2af1643b39b70806dd9d420bda0e297404ed9609c675e1c2c06b8286bd3f`
- Config SHA256: `6b72476b72f8b6c861efed2db38f2b4a70e51bf031d83df8105175dbb85d97f7`
- Actual start step: `3000`
- Actual final step: `14999`
- Continuation optimizer steps: `11999`
- Start Gaussian count: `453172`
- Intrinsic parameterization: `bounded_sh3`
- Rasterizer: `classic`
- Optimizer state available: `true`
- Scheduler state available: `true`
- Scaler state available: `true`

## Matched EON / EOFF Continuation

CONFIG FACT:

| Branch | 3k -> 14999 objective | Normal topology | load_depths |
| --- | --- | --- | --- |
| EON | RGB plus existing CDEPTH coarse-depth supervision | on | true |
| EOFF | RGB only; coarse-depth supervision disabled | on | true |

CONFIG FACT: The only intended scientific intervention was `coarse_depth_supervision_enabled`.

CODE FACT: EOFF set `weighted_L_depth = 0` for all `11999` continuation steps. EON had nonzero weighted depth loss for all `11999` continuation steps.

## Camera-Sequence Control

CODE FACT: A single explicit future training-camera sequence was generated from the restored CDEPTH@3k datamanager state and replayed in both branches.

CONFIG FACT:

- Training sequence length: `11999`
- Eval views: `MTN_1529`, `MTN_1539`, `MTN_1547`
- `CAMERA_SEQUENCE_EXACT_MATCH = true`
- `CAMERA_SEQUENCE_MISMATCH_COUNT = 0`

## Optimizer / Scheduler Equivalence

CODE FACT:

- `EXACT_OPTIMIZER_RESTORE = true`
- `EXACT_SCHEDULER_RESTORE = true`
- Restored optimizer groups: `means`, `scales`, `quats`, `features_dc`, `features_rest`, `opacities`, `medium_mlp`, `direction_encoding`

INITIAL EQUIVALENCE:

| Check | Result |
| --- | --- |
| `INITIAL_PARAMETER_EQUIVALENCE` | `PASS` |
| `INITIAL_OPTIMIZER_EQUIVALENCE` | `PASS` |
| `INITIAL_FORWARD_EQUIVALENCE` | `PASS` |
| `CONFIG_SINGLE_FACTOR_VALID` | `true` |
| `DEFAULT_COMPATIBILITY` | `true` |
| `TRAINING_STABLE` | `true` |
| `EARLY_OFF_CAUSAL_VALID` | `true` |

CODE FACT: Initial parameter max absolute difference was `0.0` for all tracked trainable groups. Initial optimizer/scheduler max absolute difference was `0.0` for all tracked optimizer state entries. Initial forward max absolute difference was `0.0` for prediction, direct object signal, medium RGB, depth, clear object, transmission, tau, accumulation, and main RGB loss.

## Normal Topology Continuation

CODE FACT: Normal WaterSplatting topology evolution was enabled for both branches. Split, duplicate/grow, prune, and opacity reset lifecycle were not disabled.

EXPERIMENTAL FACT: Topology schedule was matched, but topology results were allowed to diverge as a valid consequence of depth ON/OFF.

| Step | EON Gaussians | EOFF Gaussians | EOFF - EON |
| ---: | ---: | ---: | ---: |
| 3000 | 453172 | 453172 | 0 |
| 4000 | 839560 | 839313 | -247 |
| 5000 | 940462 | 942436 | 1974 |
| 8000 | 1212844 | 1210679 | -2165 |
| 10000 | 1219265 | 1214795 | -4470 |
| 13000 | 1183750 | 1178904 | -4846 |
| 14999 | 1177895 | 1173163 | -4732 |

CODE FACT: Each branch logged `120` topology rows. Each branch had `119` refinement calls and `14` opacity reset steps in the logged continuation events. Split/duplicate/prune subcounts were not separately instrumented; the event logs record net population changes.

## RGB Trajectory

QUANTITATIVE RESULT:

| Step | EON PSNR | EOFF PSNR | dPSNR OFF-ON | EON SSIM | EOFF SSIM | dSSIM OFF-ON | EON LPIPS | EOFF LPIPS | dLPIPS OFF-ON |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3000 | 20.877951 | 20.877951 | 0.000000 | 0.622060 | 0.622060 | 0.000000 | 0.473023 | 0.473023 | 0.000000 |
| 4000 | 26.394046 | 26.302177 | -0.091869 | 0.818291 | 0.817854 | -0.000438 | 0.302991 | 0.303237 | 0.000246 |
| 5000 | 26.495057 | 26.405650 | -0.089406 | 0.820406 | 0.820087 | -0.000320 | 0.278394 | 0.278900 | 0.000506 |
| 8000 | 31.257939 | 31.188369 | -0.069570 | 0.939975 | 0.939200 | -0.000775 | 0.095942 | 0.097642 | 0.001700 |
| 10000 | 31.399134 | 31.338401 | -0.060733 | 0.943371 | 0.942768 | -0.000603 | 0.087997 | 0.089787 | 0.001790 |
| 13000 | 31.787561 | 31.734819 | -0.052743 | 0.947334 | 0.946708 | -0.000627 | 0.080024 | 0.081973 | 0.001948 |
| 14999 | 31.817379 | 31.718239 | -0.099140 | 0.946736 | 0.946033 | -0.000703 | 0.080365 | 0.082565 | 0.002200 |

## K1 / Historical Context

QUANTITATIVE RESULT:

| Run | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| BND-K1 final | 31.498353 | 0.948783 | 0.075521 |
| Historical Full BND-CDEPTH final | 31.753299 | 0.946292 | 0.080931 |
| Matched EON final | 31.817379 | 0.946736 | 0.080365 |
| EOFF final | 31.718239 | 0.946033 | 0.082565 |

QUANTITATIVE RESULT: EOFF remained above K1 by `+0.219886 dB` PSNR, but was below matched EON by `-0.099140 dB`.

## M1_HIGH_J Trajectory

CODE FACT: `M1_HIGH_J` is a diagnostic mask defined as final M1 accumulation `> 0.01` and final M1 `clear_object_fullsh_raw` max RGB channel `> 1.0`. It is not GT and was not used as an online training mask.

QUANTITATIVE RESULT:

| Step | EON M1_HIGH_J MSE | EOFF M1_HIGH_J MSE | EOFF - EON | EON M1_HIGH_J L1 | EOFF M1_HIGH_J L1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3000 | 0.071096 | 0.071096 | 0.000000 | 0.210717 | 0.210717 |
| 4000 | 0.017992 | 0.018393 | 0.000401 | 0.094979 | 0.096988 |
| 5000 | 0.017651 | 0.018017 | 0.000367 | 0.094600 | 0.095725 |
| 8000 | 0.004984 | 0.005075 | 0.000091 | 0.045443 | 0.045689 |
| 10000 | 0.004905 | 0.004871 | -0.000034 | 0.044738 | 0.044302 |
| 13000 | 0.004294 | 0.004362 | 0.000068 | 0.041453 | 0.041489 |
| 14999 | 0.003970 | 0.004060 | 0.000090 | 0.039346 | 0.039554 |

## Benefit Retention

QUANTITATIVE RESULT:

- `EOFF_PSNR_GAIN_OVER_K1 = 0.21988646634928344`
- `EON_PSNR_GAIN_OVER_K1 = 0.3190259978027328`
- `PSNR_BENEFIT_RETENTION = 0.68924309574469`
- `HJ_GAIN_EON_OVER_K1 = 0.0008483500375101967`
- `HJ_GAIN_EOFF_OVER_K1 = 0.0007579104664425058`
- `HJ_BENEFIT_RETENTION = 0.8933935663791406`

QUANTITATIVE CONCLUSION: EOFF retained more than half of the matched EON PSNR benefit and more than half of the matched EON M1_HIGH_J benefit relative to K1.

## Perceptual Recovery

PRE-REGISTERED RULE: `PERCEPTUAL_IMPROVEMENT = true` required either `SSIM_RECOVERY_OFF_ON >= 0.0010` or `LPIPS_RECOVERY_OFF_ON >= 0.0020`, with the other metric not clearly worsening beyond its tolerance.

QUANTITATIVE RESULT:

- `SSIM_RECOVERY_OFF_ON = -0.0007033546765645715`
- `LPIPS_RECOVERY_OFF_ON = -0.0021999950210253444`
- `PERCEPTUAL_IMPROVEMENT = false`

QUANTITATIVE CONCLUSION: Turning CDEPTH off after 3k did not improve the SSIM/LPIPS trade-off relative to matched continued CDEPTH.

## Per-View Final Metrics

QUANTITATIVE RESULT:

| Branch | View | PSNR | SSIM | LPIPS | M1_HIGH_J MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| EON | MTN_1529 | 32.543976 | 0.949345 | 0.099861 | 0.004444 |
| EON | MTN_1539 | 31.373077 | 0.935499 | 0.073080 | 0.003629 |
| EON | MTN_1547 | 31.535084 | 0.955365 | 0.068152 | 0.003837 |
| EOFF | MTN_1529 | 32.412724 | 0.948322 | 0.103411 | 0.004516 |
| EOFF | MTN_1539 | 31.237600 | 0.935555 | 0.072556 | 0.003862 |
| EOFF | MTN_1547 | 31.504395 | 0.954222 | 0.071726 | 0.003803 |

## Decomposition Safety

QUANTITATIVE RESULT: Final EOFF M1_HIGH_J aggregate:

- `J_p99 = 0.9963003396987915`
- `P(J>1) = 0.0`
- `tau_p90 = 0.4546791563431422`
- `tau_p99 = 2.1353471080462136`
- `P(T<0.1) = 0.00908866710960865`
- `P(c>0.99) = 0.009928714173535505`
- `P(|s_full|>5) = 0.009442942837874094`
- `TAU_SAFETY = true`
- `BOUNDARY_PRESSURE_REGRESSION = false`

QUANTITATIVE CONCLUSION: EOFF did not reopen the unbounded intrinsic route under the tracked `P(J>1)` criterion.

## Direct / Medium Context

EXPERIMENTAL FACT: At final step 14999, averaged over eval views:

| Region | mean abs dD | mean abs dB | mean abs dI | D/B response ratio | Direct share |
| --- | ---: | ---: | ---: | ---: | ---: |
| global | 0.009108 | 0.004240 | 0.006514 | 2.572383 | 0.701277 |
| M1_HIGH_J | 0.012796 | 0.002857 | 0.011724 | 5.518370 | 0.816096 |

REASONABLE INFERENCE: Direct/medium differences exist between EON and EOFF, but prior audits showed direct-path divergence alone is not sufficient to explain beneficial high-J recovery. This experiment records the context but does not open a new direct-path mechanism branch.

## Optimizer-Memory Context

EXPERIMENTAL FACT: Initial optimizer memory was exactly equivalent. After normal topology evolution diverged, elementwise optimizer comparisons for per-Gaussian groups became unavailable because shapes differed, which is expected under normal topology.

EXPERIMENTAL FACT: At final step 14999:

- `medium_mlp` `exp_avg` norm: EON `0.0000900165`, EOFF `0.0000911016`, relative norm delta `+0.012055`, elementwise relative diff `0.327236`.
- Per-Gaussian groups had different shapes due to normal topology divergence.

## Formal Classification

QUANTITATIVE RESULT:

```text
EARLY_OFF_CLASSIFICATION = EARLY_GUIDANCE_RGB_ONLY_SUPPORTED
CDEPTH_DECISION = CLOSE
```

RATIONALE:

- Causal validity passed.
- EOFF retained substantial PSNR benefit over K1: `+0.219886 dB`.
- PSNR benefit retention passed: `0.689243`.
- HJ benefit retention passed: `0.893394`.
- Decomposition safety passed: `P(J>1)=0`, `TAU_SAFETY=true`, `BOUNDARY_PRESSURE_REGRESSION=false`.
- Perceptual improvement failed: `SSIM_RECOVERY_OFF_ON=-0.000703`, `LPIPS_RECOVERY_OFF_ON=-0.002200`.

## CDEPTH GO / CLOSE Decision

QUANTITATIVE CONCLUSION: `CDEPTH_DECISION = CLOSE`.

INFERENCE: The fixed 0->3k early-guidance schedule was sufficient to retain a substantial portion of matched CDEPTH RGB/high-J benefit after late depth supervision was removed, but it did not improve the SSIM/LPIPS trade-off required for a Pareto-positive KEEP decision.

## Scientific Interpretation

REASONABLE INFERENCE: CDEPTH's effect can be partly explained by early bounded-basin formation, because EOFF inherited the CDEPTH 0->3k history and retained `68.9%` of the matched PSNR benefit and `89.3%` of the matched M1_HIGH_J benefit relative to K1.

REASONABLE INFERENCE: Continued depth supervision after 3k is not strictly necessary for retaining much of the RGB/high-J recovery, but the evidence does not show that removing it solves the perceptual trade-off.

REASONABLE INFERENCE: For Panama, bounded intrinsic appearance introduces a scene-dependent reconstruction trade-off under the current WaterSplatting representation and optimization framework. This does not imply that sigmoid boundedness inevitably causes metric degradation; AA and CDEPTH both show partial recoverability.

## If Closed

QUANTITATIVE CONCLUSION: The CDEPTH line is closed for the current study.

REASONABLE INFERENCE: Full CDEPTH remains useful as a mechanistically informative partial-mitigation baseline, not as the active optimization direction.

DEFERRED / CLOSED CDEPTH DIRECTIONS:

- depth-weight sweep
- depth cutoff-step sweep
- depth-start-step sweep
- CDEPTH + AA
- CDEPTH + new loss
- CDEPTH + group restriction
- CDEPTH + densification tricks
- CDEPTH + optimizer tricks
- CDEPTH 30k

## Next Non-CDEPTH Direction

PROPOSED CONTROLLED EXPERIMENT: Prioritize bounded-aware representation/refinement capacity.

RATIONALE: The accumulated evidence supports preserving bounded decomposition control while adding legal representation capacity in bounded-hard regions, instead of continuing to tune CDEPTH schedules or weights.

## Outputs And Visual Assets

CODE FACT:

- Output directory: `/mnt/new/home_old/ycy/water-splatting-refactor/outputs/bnd_cdepth_early_off_panama_20260811`
- Render directory: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_cdepth_early_off_panama_20260811`
- Visual index: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_cdepth_early_off_panama_20260811/VISUAL_COMPARE_INDEX.md`
- Render manifest: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_cdepth_early_off_panama_20260811/manifest.json`
- Output manifest: `/mnt/new/home_old/ycy/water-splatting-refactor/outputs/bnd_cdepth_early_off_panama_20260811/manifest.json`

VISUAL ASSETS:

- Final RGB comparison: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_cdepth_early_off_panama_20260811/contact_sheet_final_rgb_comparison.png`
- M1_HIGH_J residual comparison: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_cdepth_early_off_panama_20260811/contact_sheet_m1_highj_residual_comparison.png`
- Per-view final comparison: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_cdepth_early_off_panama_20260811/contact_sheet_per_view_final_comparison.png`
- RGB metric trajectory: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_cdepth_early_off_panama_20260811/plot_rgb_metric_trajectory.png`
- SSIM/LPIPS trajectory: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_cdepth_early_off_panama_20260811/plot_ssim_lpips_trajectory.png`
- High-J MSE trajectory: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_cdepth_early_off_panama_20260811/plot_highj_mse_trajectory.png`
- Benefit retention: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_cdepth_early_off_panama_20260811/plot_benefit_retention.png`
- Perceptual recovery: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_cdepth_early_off_panama_20260811/plot_perceptual_recovery.png`
- Direct/medium context: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_cdepth_early_off_panama_20260811/plot_direct_medium_context.png`
- Gaussian count trajectory: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_cdepth_early_off_panama_20260811/plot_gaussian_count_trajectory.png`
- Optimizer-memory context: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_cdepth_early_off_panama_20260811/plot_optimizer_memory_context.png`
- Decomposition controls: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_cdepth_early_off_panama_20260811/plot_decomposition_controls.png`
- Final decomposition safety: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_cdepth_early_off_panama_20260811/plot_final_decomposition_safety.png`
- Training RGB loss context: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_cdepth_early_off_panama_20260811/plot_training_rgb_loss_context.png`
- Final GO/CLOSE summary sheet: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_cdepth_early_off_panama_20260811/final_go_close_summary_sheet.png`

No subjective clear-image correctness judgment was made.
