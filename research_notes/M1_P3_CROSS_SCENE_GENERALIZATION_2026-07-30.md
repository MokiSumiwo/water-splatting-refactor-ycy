# M1 / P3 Cross-Scene Generalization

Date: 2026-07-30
Branch: `refactor/core-framework`
Initial commit: `979b3aa Record T-series support diagnostics`

## Goal

Run a frozen cross-scene validation for the three remaining scenes under:

```text
/mnt/new/home_old/ycy/water-splatting-refactor/undistorted_data
```

This is not a tuning round. The goal is to determine whether:

1. M1 remains stable across non-IUI3 scenes.
2. P3 improves or preserves underwater reconstruction while reducing scene-medium residuals across scenes.

## Scenes

| Scene | Data directory | Train | Val | Test | Notes |
|---|---|---:|---:|---:|---|
| Curasao | `undistorted_data/undistorted_Curasao` | 17 | 3 | 2 | COLMAP sparse/0 present |
| JapaneseGradens-RedSea | `undistorted_data/undistorted_JapaneseGradens-RedSea` | 16 | 2 | 2 | Directory keeps original `Gradens` typo |
| Panama | `undistorted_data/undistorted_Panama` | 14 | 2 | 2 | COLMAP sparse/0 present |

## Frozen Protocol

- Seed: `42`
- Full training: `0 -> 14999`, `max_num_iterations=15000`
- Save checkpoints every `5000` steps
- `save_only_latest_checkpoint=False`
- M1 and P3 both train from scratch
- No scene-specific hyperparameter changes
- P3 uses scene-specific M1-derived masks for diagnostics only; masks are not training supervision

## Frozen M1 Config

```text
medium_context_mode = dir_xy_camera
b_inf_mode = tied
infinite_water_enabled = False
foreground transmission loss = off
medium explainability = off
budgeted capacity = off
clear proxy chroma = off
gradient routing = off
```

## Frozen Original Baseline Config

```text
medium_context_mode = dir_only
b_inf_mode = implicit
infinite_water_enabled = False
foreground transmission loss = off
medium explainability = off
budgeted capacity = off
clear proxy chroma = off
gradient routing = off
```

This matches the historical original WaterSplatting definition used for IUI3-RedSea.

## Frozen P3 Config

```text
medium_context_mode = dir_xy_camera
b_inf_mode = tied

medium_explainability_enabled = True
lambda_medium_explainability = 0.005
medium_explainability_start_step = 2000
medium_explainability_ramp_steps = 2000

training_gradient_routing_enabled = False

budgeted_capacity_enabled = True
lambda_budgeted_capacity = 0.0002
budgeted_capacity_value = 0.05
budgeted_capacity_temperature = 0.02
budgeted_capacity_start_step = 4000
budgeted_capacity_ramp_steps = 1000
budgeted_capacity_post_scale = 0.5

lambda_background_clear_chroma = 0.0015
background_clear_chroma_start_step = 10000
background_clear_chroma_ramp_steps = 1000
background_clear_chroma_use_medium_support = True
clear_proxy_geometry_gradient_scale = 0.0
clear_proxy_opacity_gradient_scale = 0.50
clear_proxy_color_gradient_scale = 1.0

halo capacity = off
conflict gate = off
candidate surgery = off
old M2 ownership = off
```

## Scripts Added

Generic runners:

```text
scripts/experiments/cross_scene_m1_common.sh
scripts/experiments/cross_scene_p3_common.sh
scripts/experiments/run_cross_scene_remaining_m1_p3.sh
```

Scene-specific M1 scripts:

```text
scripts/experiments/cross_scene_curasao_m1_seed42_15000.sh
scripts/experiments/cross_scene_japanesegradens_m1_seed42_15000.sh
scripts/experiments/cross_scene_panama_m1_seed42_15000.sh
```

Scene-specific P3 scripts:

```text
scripts/experiments/cross_scene_curasao_p3_seed42_15000.sh
scripts/experiments/cross_scene_japanesegradens_p3_seed42_15000.sh
scripts/experiments/cross_scene_panama_p3_seed42_15000.sh
```

Scene-specific original baseline scripts:

```text
scripts/experiments/cross_scene_curasao_baseline_seed42_15000.sh
scripts/experiments/cross_scene_japanesegradens_baseline_seed42_15000.sh
scripts/experiments/cross_scene_panama_baseline_seed42_15000.sh
scripts/experiments/run_cross_scene_remaining_baseline.sh
```

## Execution Plan

Run short smoke first:

```bash
STAMP=20260730_cross_scene_smoke MAX_NUM_ITERATIONS=20 MODEL_NUM_STEPS=20 \
  RUN_EVAL=0 RUN_CLOSURE_DIAG=0 BUILD_MASKS=0 RUN_POST_MASK_DIAGS=0 \
  scripts/experiments/cross_scene_curasao_m1_seed42_15000.sh

STAMP=20260730_cross_scene_smoke MAX_NUM_ITERATIONS=20 MODEL_NUM_STEPS=20 \
  RUN_EVAL=0 RUN_CLOSURE_DIAG=0 RUN_FAR_DIAG=0 RUN_REGION_DIAG=0 \
  scripts/experiments/cross_scene_curasao_p3_seed42_15000.sh
```

Repeat smoke for JapaneseGradens-RedSea and Panama.

Smoke status:

```text
2026-07-30:
  Curasao M1/P3 20-step smoke: passed
  JapaneseGradens-RedSea M1/P3 20-step smoke: passed
  Panama M1/P3 20-step smoke: passed
  P3 manifest check: medium explainability on, budgeted capacity on, routing/halo off,
    clear proxy geometry grad 0.0, opacity grad 0.50, chroma weight 0.0015
  Smoke outputs/renders/logs removed after verification.

2026-07-30 baseline smoke:
  Curasao / JapaneseGradens-RedSea / Panama original baseline 20-step smoke: passed
  Manifest check: medium_context_mode=dir_only, b_inf_mode=implicit,
    medium explainability off, budgeted capacity off, routing off, clear chroma off.
  Smoke outputs/renders/logs removed after verification.
```

M1 execution status:

```text
2026-07-30:
  Curasao M1 training and ns-eval completed.
  JapaneseGradens-RedSea M1 training, ns-eval, M1 masks, and diagnostics completed.
  Panama M1 training and ns-eval completed.

  Curasao/Panama closure diagnostic initially failed in post-training diagnostics:
    file: scripts/diagnostics/diagnose_backscatter_closure.py
    function: _stats
    error: RuntimeError: quantile() input tensor is too large
    fix: replace torch.quantile p95 with nearest-rank kthvalue p95.

  After the fix, Curasao/Panama closure, M1-derived far masks, eval-region masks,
  far-water diagnostics, and eval-region diagnostics completed without retraining.
```

Full run:

```bash
STAMP=20260730_cross_scene scripts/experiments/run_cross_scene_remaining_m1_p3.sh
```

The launcher runs:

1. Curasao / JapaneseGradens / Panama M1 in parallel.
2. Builds M1-derived common far masks and eval-region masks per scene.
3. Runs M1 far-water and eval-region diagnostics.
4. Runs Curasao / JapaneseGradens / Panama P3 in parallel using the corresponding scene M1 masks.

## Output Paths

Outputs:

```text
outputs/cross_scene_<scene>_m1_seed42_15000/
outputs/cross_scene_<scene>_p3_seed42_15000/
```

Renders and diagnostics:

```text
renders/cross_scene_<scene>_m1_seed42_15000_20260730_cross_scene/
renders/cross_scene_<scene>_p3_seed42_15000_20260730_cross_scene/
```

Masks:

```text
common_masks/cross_scene_<scene>_m1_q90_seed42_20260730_cross_scene/
common_masks/cross_scene_<scene>_m1_eval_regions_seed42_20260730_cross_scene/
```

Logs:

```text
logs/cross_scene_<scene>_m1_seed42_15000_20260730_cross_scene/
logs/cross_scene_<scene>_p3_seed42_15000_20260730_cross_scene/
logs/cross_scene_launcher_20260730_cross_scene/
```

## Result Table

| Scene | Method | PSNR | SSIM | LPIPS | J Blue | Far Accum | Far Clear | Water Accum | Water J | Object Acc Ret | Object J Ret | Boundary Ret |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Curasao | M1 | 32.1652 | 0.9560 | 0.1081 | 0.0263 | 0.4429 | 0.2140 | 0.0499 | 0.0284 | 1.0000 | 1.0000 | 1.0000 |
| Curasao | P3 | 32.1584 | 0.9560 | 0.1086 | 0.0154 | 0.3138 | 0.1415 | 0.0151 | 0.0076 | 0.9815 | 0.9724 | 0.9642 |
| JapaneseGradens-RedSea | M1 | 24.7565 | 0.8995 | 0.1204 | 0.1395 | 0.4412 | 0.0542 | 0.0762 | 0.0159 | 1.0000 | 1.0000 | 1.0000 |
| JapaneseGradens-RedSea | P3 | 24.6643 | 0.8965 | 0.1206 | 0.1602 | 0.2759 | 0.0502 | 0.0692 | 0.0129 | 0.9736 | 1.0031 | 1.0960 |
| Panama | M1 | 32.3089 | 0.9495 | 0.0740 | 0.2448 | 0.9184 | 0.4406 | 0.1361 | 0.0608 | 1.0000 | 1.0000 | 1.0000 |
| Panama | P3 | 32.3073 | 0.9491 | 0.0736 | 0.2022 | 0.8523 | 0.4212 | 0.9990 | 0.7515 | 0.9818 | 0.9801 | 0.8815 |

Additional far connected residual:

| Scene | Method | Far BG Fraction | Far BG LCC Sum |
|---|---|---:|---:|
| Curasao | M1 | 0.5780 | 0.5636 |
| Curasao | P3 | 0.2979 | 0.2741 |
| JapaneseGradens-RedSea | M1 | 0.4742 | 0.2861 |
| JapaneseGradens-RedSea | P3 | 0.3576 | 0.2662 |
| Panama | M1 | 0.9900 | 0.6549 |
| Panama | P3 | 0.9898 | 0.6568 |

P3 vs M1 deltas:

| Scene | dPSNR | dSSIM | dLPIPS | J Blue | Far Accum | Far Clear | Far BG Frac | Water Accum | Water J | Obj Acc Ret | Obj J Ret | Boundary Ret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Curasao | -0.0067 | +0.0000 | +0.0005 | -41.2% | -29.2% | -33.8% | -48.5% | -69.8% | -73.2% | 0.9815 | 0.9724 | 0.9642 |
| JapaneseGradens-RedSea | -0.0921 | -0.0030 | +0.0003 | +14.8% | -37.5% | -7.3% | -24.6% | -9.2% | -18.9% | 0.9736 | 1.0031 | 1.0960 |
| Panama | -0.0016 | -0.0004 | -0.0004 | -17.4% | -7.2% | -4.4% | -0.0% | +634.0% | +1136.2% | 0.9818 | 0.9801 | 0.8815 |

Macro averages over the three new scenes:

| Method | PSNR | SSIM | LPIPS | J Blue |
|---|---:|---:|---:|---:|
| M1 | 29.7435 | 0.9350 | 0.1008 | 0.1368 |
| P3 | 29.7100 | 0.9339 | 0.1009 | 0.1259 |

## Notes on Mask Reliability

Panama's M1-derived eval-region water mask produced only one water pixel across the evaluated images. Therefore:

- Panama `Water Accum` and `Water J` are not reliable scene-level water-region diagnostics.
- Panama far-mask diagnostics remain usable because they are depth-quantile based and contain 639,741 far pixels.
- Panama object/boundary retention should still be treated cautiously because the water/object split is highly imbalanced.

## Decision Criteria

P3 should be considered cross-scene positive only if it preserves underwater quality relative to M1:

```text
PSNR drop <= 0.03 dB
SSIM drop <= 0.0005
LPIPS increase <= 0.0005
```

and improves at least one meaningful reconstruction or residual signal:

```text
PSNR improvement >= 0.05 dB
or LPIPS improvement >= 0.001
or SSIM improvement >= 0.001
or J Blue / Far Clear / Water J improves materially
```

If P3 fails this on most new scenes, M1 remains the cross-scene reconstruction candidate and P3 should be positioned as a residual-cleanup module with scene dependence.

## Current Interpretation

P3 is not a clean cross-scene replacement for M1 under the frozen RedSea configuration.

- Curasao is the strongest positive case: underwater quality is essentially preserved, while J Blue, Far Accum, Far Clear, Far BG Fraction, Water Accum, and Water J all drop materially.
- JapaneseGradens-RedSea is a negative reconstruction/generalization case: PSNR drops by 0.092 dB, SSIM drops by 0.0030, and J Blue increases despite lower Far Accum/Far Clear.
- Panama is mixed: underwater metrics are preserved and J Blue/Far Accum/Far Clear improve modestly, but far BG connected residual is unchanged and the eval-region water mask is too sparse to support water-region claims.

Provisional decision:

```text
M1 remains the safer cross-scene underwater reconstruction candidate.
P3 should be described as a scene-medium residual cleanup module that can work strongly
on some scenes, but its frozen RedSea weights are not yet robust enough to be the
default cross-scene method.
```

Recommended next step:

```text
Do not tune per scene. First inspect P3 support coverage and medium explainability
maps on JapaneseGradens and Panama to determine whether the failure is caused by:
  1. over-broad capacity support near textured reef / seafloor,
  2. weak medium explainability in non-RedSea water color,
  3. proxy chroma pushing color in the wrong direction,
  4. unreliable auto eval-region masks on Panama.
```
