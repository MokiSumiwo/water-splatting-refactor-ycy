# BND-CDEPTH Fixed-Population Rollout

## CODE FACT

- Branch at experiment start: `research/m1-bounded-intrinsic`.
- Start HEAD: `f8aa217abb93a531b1ad52fc9f3f73ae54ecd25b`.
- Diagnostic runner: `scripts/diagnostics/run_bnd_cdepth_fixpop_rollout.py`.
- Shell entry point: `scripts/experiments/bnd_cdepth_fixpop_rollout_panama_3k_to_5k.sh`.
- No production renderer, model, densification, loss, or trainer source was modified.
- The runner creates two matched branches from the same formal Panama BND-K1 checkpoint at nominal step 3000: `FP-R` and `FP-RD`.
- `FP-R` uses the bounded K1 RGB objective only.
- `FP-RD` uses the same RGB objective plus the existing SeaFree-style coarse-depth term.
- The only branch intervention is `coarse_depth_supervision_enabled`.
- Gaussian population mutation paths are not invoked: no split, duplicate, grow, prune, cull, insertion, or deletion.
- Scheduled opacity reset is retained for both branches at absolute steps `3100`, `3600`, `4100`, and `4600`; the opacity clamp and opacity Adam-moment reset are applied symmetrically.
- Densification statistics are not used to mutate topology in this rollout.

## CONFIG FACT

- Start checkpoint: `/mnt/new/home_old/ycy/water-splatting-refactor/outputs/dewater_bounded_sh3_cross_scene_20260808/dewater_bnd_cross_scene_panama_seed42_step0_to_15000/water-splatting/dewater_bnd_cross_scene_panama_seed42_step0_to_15000_20260808_bounded_sh3_cross_scene_full_panama_bnd_g1p00/nerfstudio_models/step-000003000.ckpt`.
- Config path: `/mnt/new/home_old/ycy/water-splatting-refactor/outputs/dewater_bounded_sh3_cross_scene_20260808/dewater_bnd_cross_scene_panama_seed42_step0_to_15000/water-splatting/dewater_bnd_cross_scene_panama_seed42_step0_to_15000_20260808_bounded_sh3_cross_scene_full_panama_bnd_g1p00/config.yml`.
- Checkpoint actual step: `3000`.
- Checkpoint step field: `3000`.
- Start Gaussian count: `566332`.
- Optimizer state available/restored: `true`.
- Scheduler state available/restored: `true`.
- Runtime intrinsic parameterization: `bounded_sh3`.
- Medium context mode: `dir_xy_camera`.
- `b_inf_mode`: `tied`.
- Rasterizer: `classic`.
- Coarse depth at start checkpoint: `false`.
- Rollout length: `2000` optimizer steps, nominal absolute `3000 -> 5000`.
- Snapshots: relative `0`, `100`, `250`, `500`, `1000`, `1500`, `2000`.
- Eval views: `MTN_1529`, `MTN_1539`, `MTN_1547`.
- Camera sequence file: `/mnt/new/home_old/ycy/water-splatting-refactor/outputs/bnd_cdepth_fixpop_rollout_panama_20260811/paired_camera_sequence.json`.
- Output manifest: `/mnt/new/home_old/ycy/water-splatting-refactor/outputs/bnd_cdepth_fixpop_rollout_panama_20260811/manifest.json`.
- Visual index: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_cdepth_fixpop_rollout_panama_20260811/VISUAL_COMPARE_INDEX.md`.

## EXPERIMENTAL FACT

- `DEFAULT_COMPATIBILITY = true`.
- `FIXPOP_MUTATION_BLOCK = true`.
- `INITIAL_PARAMETER_EQUIVALENCE = true`.
- `INITIAL_OPTIMIZER_EQUIVALENCE = true`.
- `INITIAL_FORWARD_EQUIVALENCE = true`.
- `CAMERA_SEQUENCE_EXACT_MATCH = true`.
- `FIXED_TOPOLOGY_INVARIANCE = true`.
- `ROLL_STABLE = true`.
- `PERSISTENT_OUTPUT_SAFETY = true`.
- `ROLLOUT_CAUSAL_VALID = true`.
- Paired camera sequence length: `2000`.
- Paired camera mismatch count: `0`.
- Final `N_FP_R = 566332`.
- Final `N_FP_RD = 566332`.
- `I = D + B` max absolute error at recorded snapshots: `0.0`.

## RGB ROLLOUT

| Rel step | Abs step | FP-R PSNR | FP-RD PSNR | Delta PSNR | FP-R SSIM | FP-RD SSIM | FP-R LPIPS | FP-RD LPIPS | Global MSE gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3000 | 20.869651 | 20.869651 | +0.000000 | 0.597971 | 0.597971 | 0.440031 | 0.440031 | +0.000000e+00 |
| 100 | 3100 | 23.082726 | 23.095002 | +0.012277 | 0.750493 | 0.751103 | 0.402662 | 0.403141 | +2.079977e-05 |
| 250 | 3250 | 25.992416 | 25.944487 | -0.047930 | 0.814528 | 0.814984 | 0.304801 | 0.304115 | -3.016476e-05 |
| 500 | 3500 | 26.392229 | 26.427955 | +0.035726 | 0.824985 | 0.825514 | 0.271441 | 0.270587 | +2.176710e-05 |
| 1000 | 4000 | 26.523156 | 26.522398 | -0.000758 | 0.832161 | 0.832416 | 0.247024 | 0.246287 | -2.006845e-06 |
| 1500 | 4500 | 26.795144 | 26.815531 | +0.020387 | 0.834090 | 0.834436 | 0.238496 | 0.237513 | +8.464558e-06 |
| 2000 | 5000 | 26.790438 | 26.784211 | -0.006227 | 0.835643 | 0.835671 | 0.235006 | 0.233702 | -5.842652e-06 |

## M1_HIGH_J ROLLOUT

| Rel step | Abs step | MSE FP-R | MSE FP-RD | HJ MSE gain | HJ rel improvement |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3000 | 0.06875545 | 0.06875545 | +0.000000e+00 | +0.000000 |
| 100 | 3100 | 0.04669803 | 0.04679836 | -1.003233e-04 | -0.002148 |
| 250 | 3250 | 0.02133523 | 0.02163352 | -2.982877e-04 | -0.013981 |
| 500 | 3500 | 0.01899810 | 0.01873271 | +2.653853e-04 | +0.013969 |
| 1000 | 4000 | 0.01803338 | 0.01807312 | -3.973736e-05 | -0.002204 |
| 1500 | 4500 | 0.01647189 | 0.01638111 | +9.078346e-05 | +0.005511 |
| 2000 | 5000 | 0.01649055 | 0.01658075 | -9.019580e-05 | -0.005470 |

## HISTORICAL EFFECT CONTEXT

- `HIST_HJ_GAIN_5K = 6.500560169418653e-04`.
- `HIST_GLOBAL_MSE_GAIN_5K = -1.9215978682041168e-05`.
- `FIXPOP_HJ_EFFECT_FRACTION = -0.13875080767362774`.
- `FIXPOP_GLOBAL_EFFECT_FRACTION = 0.3040517617409005`.
- These values are effect-size context only. Historical K1/CDEPTH are not reproduced branches in this fixed-population rollout.

## PER-VIEW RESULT

| View | HJ gain at 1500 | HJ gain at 2000 |
| --- | ---: | ---: |
| `MTN_1529` | +1.258254e-04 | -1.032930e-04 |
| `MTN_1539` | +2.064575e-04 | +8.012820e-05 |
| `MTN_1547` | -5.993247e-05 | -2.474226e-04 |

- Late positive eval-view count: `1`.

## DIRECT / MEDIUM TRAJECTORY

M1_HIGH_J mean over eval views:

| Rel step | mean abs Delta D | mean abs Delta B | mean abs Delta I | D/B ratio | Direct share |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000 | 0.000 |
| 100 | 1.349931e-02 | 3.689286e-03 | 1.161167e-02 | 3.698 | 0.785 |
| 250 | 2.049502e-02 | 2.138956e-03 | 1.961867e-02 | 9.812 | 0.906 |
| 500 | 1.985203e-02 | 1.792970e-03 | 1.916918e-02 | 11.308 | 0.916 |
| 1000 | 1.874028e-02 | 1.869586e-03 | 1.793479e-02 | 10.622 | 0.909 |
| 1500 | 2.089952e-02 | 2.263810e-03 | 1.996730e-02 | 9.685 | 0.902 |
| 2000 | 2.015314e-02 | 2.098485e-03 | 1.923195e-02 | 10.501 | 0.907 |

- `ROLL_DIRECT_DOMINANCE_ONSET = 100`.
- `ROLL_HJ_RECOVERY_ONSET = null`.
- `ROLL_GLOBAL_RECOVERY_ONSET = null`.
- `TRAIN_RGB_ADVANTAGE_ONSET = null`.

## OPTIMIZER MEMORY

`exp_avg` relative divergence, 100 -> 2000:

| Group | Step 100 | Step 2000 | exp_avg_sq at 2000 |
| --- | ---: | ---: | ---: |
| `means` | 0.518710 | 0.989744 | 0.253065 |
| `scales` | 0.917277 | 1.025537 | 0.789043 |
| `quats` | 0.771143 | 1.016347 | 0.820926 |
| `opacities` | 0.000000 | 0.931050 | 0.999536 |
| `features_dc` | 0.503379 | 1.052630 | 1.137008 |
| `features_rest` | 0.498436 | 1.050516 | 1.112795 |
| `medium_mlp` | 0.687792 | 0.085457 | 0.043707 |
| `direction_encoding` | 0.000000 | 0.000000 | 0.000000 |

## PARAMETER DIVERGENCE

Parameter relative divergence, 100 -> 2000:

| Group | Step 100 | Step 2000 |
| --- | ---: | ---: |
| `means` | 0.000399 | 0.001688 |
| `scales` | 0.004395 | 0.064907 |
| `quats` | 0.016516 | 0.181088 |
| `opacities` | 0.152942 | 0.608252 |
| `features_dc` | 0.001191 | 0.017951 |
| `features_rest` | 0.028235 | 0.301579 |
| `medium_mlp` | 0.020437 | 0.114434 |
| `direction_encoding` | 0.000000 | 0.000000 |

## PHYSICAL GAUSSIAN DIVERGENCE

Final physical divergence at relative step 2000:

| Group | Metric | Mean | p90 | p99 |
| --- | --- | ---: | ---: | ---: |
| `means` | world displacement | 0.00262249 | 0.00629651 | 0.0152597 |
| `scales` | activated scale relative abs diff | 0.501376 | 0.738955 | 4.25755 |
| `quats` | quat angle radians | 0.252687 | 0.586770 | 1.23171 |
| `opacities` | sigmoid opacity abs diff | 0.113422 | 0.310588 | 0.794136 |

## DECOMPOSITION / BOUNDARY CONTROLS

Final M1_HIGH_J mean over eval views:

| Branch | J p99 | P(J>1) | tau p90 | P(T<0.1) | T mean | P(c>0.99) | P(|s|>5) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `FP-R` | 0.983621 | 0.000000 | 0.636391 | 0.000000 | 0.719609 | 0.033649 | 0.032669 |
| `FP-RD` | 0.984997 | 0.000000 | 0.645862 | 0.000000 | 0.714943 | 0.033632 | 0.032669 |

## QUANTITATIVE CONCLUSION

- `ROLL_HJ_RECOVERY_ONSET = null`.
- `ROLL_GLOBAL_RECOVERY_ONSET = null`.
- `ROLL_DIRECT_DOMINANCE_ONSET = 100`.
- Final HJ relative improvement: `-0.005469543253903957`.
- Final HJ MSE gain: `-9.019579738378525e-05`.
- Final PSNR gain: `-0.0062274932861328125`.
- Late positive eval-view count: `1`.
- `LONG_HORIZON_BUILDUP = false`.
- `FIXPOP_ROLLOUT_CLASSIFICATION = NO_MEANINGFUL_FIXPOP_RECOVERY`.
- `ONE_STEP_INSUFFICIENT_BUT_MULTISTEP_SUPPORTED = false`.

## REASONABLE INFERENCE

- Under matched camera sampling and fixed Gaussian topology, adding the coarse-depth term at the K1@3k state did not produce stable multi-step M1_HIGH_J recovery over the 3k->5k window.
- The rollout did produce direct-object-dominant branch divergence by relative step 100, but this did not translate into stable HJ or global recovery at the evaluation snapshots.
- The result does not imply that continuous optimization is irrelevant. Because the rollout starts from K1@3k, it does not include the historical 0->3k CDEPTH optimizer-memory path or early topology interactions.
- The appropriate wording is: starting from K1@3k, the continuous depth-supervised path under fixed topology was insufficient to reproduce the historical recovery over the 3k->5k window.

## UNVERIFIED HYPOTHESIS

- Historical CDEPTH recovery may depend on optimizer-memory and parameter-direction divergence accumulated before 3k.
- The next single-factor diagnostic should be a read-only historical optimizer-memory divergence audit for the 0->3k history.

## OUTPUTS

- Output directory: `/mnt/new/home_old/ycy/water-splatting-refactor/outputs/bnd_cdepth_fixpop_rollout_panama_20260811/`.
- Render directory: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_cdepth_fixpop_rollout_panama_20260811/`.
- Manifest: `/mnt/new/home_old/ycy/water-splatting-refactor/outputs/bnd_cdepth_fixpop_rollout_panama_20260811/manifest.json`.
- Visual index: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_cdepth_fixpop_rollout_panama_20260811/VISUAL_COMPARE_INDEX.md`.
