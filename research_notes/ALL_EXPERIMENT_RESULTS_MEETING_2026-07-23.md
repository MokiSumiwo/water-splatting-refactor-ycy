# WaterSplatting Refactor/Ablation Results - Meeting Summary

Date: 2026-07-23

Repo: `/mnt/new/home_old/ycy/water-splatting-refactor`

Dataset: `undistorted_data/undistorted_IUI3-RedSea`

Training setting: 15000 iterations unless otherwise noted.

## Metric Notes

- Underwater reconstruction metrics: PSNR/SSIM higher is better; LPIPS lower is better.
- Dewatered image output: `J = clamp(raw Gaussian-only clear render, 0, 1)`.
- Legacy `rgb_clear = raw / (raw + 1)` is kept only for compatibility and is not treated as the dewatered image.
- `J_*_ratio` metrics are diagnostics for dewatered-image quality; lower is generally better.

## Main Results

| Experiment | Mechanism | PSNR | SSIM | LPIPS | Delta PSNR vs Baseline | J White | J Saturation | J Red Dom. | J Blue Dom. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline original | Original WaterSplatting | 29.8790 | 0.9105 | 0.1810 | 0.0000 | - | - | - | - |
| Step1 refactor sanity | No-numeric-change refactor | 29.8790 | 0.9105 | 0.1810 | 0.0000 | - | - | - | - |
| M1 | Context-aware medium, `dir_xy_camera` | 31.1314 | 0.9120 | 0.1750 | +1.2524 | - | - | - | - |
| M2 | M1 + infinite-water ownership, `alpha_depth` | 31.0696 | 0.9129 | 0.1771 | +1.1905 | 0.0210 | 0.0265 | 0.0426 | 0.0514 |
| M3 diagnostic | M1 + M2 + cleanup dry-run diagnostics | 31.1794 | 0.9149 | 0.1775 | +1.3003 | - | - | - | - |
| M4 main | M1 + M2 + delayed SH + SH residual + DC softclip | 31.0922 | 0.9157 | 0.1779 | +1.2131 | 0.0145 | 0.0168 | 0.0529 | 0.0358 |
| M4b | M4 + DC channel balance `0.001` | 31.0142 | 0.9132 | 0.1801 | +1.1352 | 0.0145 | 0.0171 | 0.0468 | 0.0203 |
| M4c | M4 + weak DC channel balance `0.0003` | 31.1179 | 0.9132 | 0.1774 | +1.2389 | 0.0137 | 0.0162 | 0.0394 | 0.0246 |

## Experiment Paths

| Experiment | Checkpoint | Eval / Metrics |
|---|---|---|
| Baseline original | `outputs/baseline_original_watersplatting_iui3_redsea/water-splatting/orig_watersplatting_iui3_redsea_15000_20260723_063201/nerfstudio_models/step-000014999.ckpt` | `renders/baseline_original_watersplatting_iui3_redsea/eval_metrics.json` |
| Step1 refactor sanity | Reused baseline checkpoint / equivalent sanity eval | `renders/step1_refactor_sanity_iui3_redsea_20260723_072224/output.json` |
| M1 | `outputs/m1_dir_xy_camera_iui3_redsea_15000/water-splatting/m1_dir_xy_camera_iui3_redsea_15000_20260723_072412/nerfstudio_models/step-000014999.ckpt` | `renders/m1_dir_xy_camera_iui3_redsea_15000_20260723_072412/output.json` |
| M2 | `outputs/m2_alpha_depth_dir_xy_camera_iui3_redsea_15000/water-splatting/m2_alpha_depth_dir_xy_camera_iui3_redsea_15000_20260723_080936/nerfstudio_models/step-000014999.ckpt` | `renders/m2_alpha_depth_dir_xy_camera_iui3_redsea_15000_20260723_080936_pureJ_eval/output.json` |
| M3 diagnostic | `outputs/m3_cleanup_diag_alpha_depth_dir_xy_camera_iui3_redsea_15000/water-splatting/m3_cleanup_diag_alpha_depth_dir_xy_camera_iui3_redsea_15000_20260723_083104/nerfstudio_models/step-000014999.ckpt` | `renders/m3_cleanup_diag_alpha_depth_dir_xy_camera_iui3_redsea_15000_20260723_083104/output.json` |
| M4 main | `outputs/m4_constrained_appearance_alpha_depth_dir_xy_camera_iui3_redsea_15000/water-splatting/m4_constrained_appearance_alpha_depth_dir_xy_camera_iui3_redsea_15000_20260723_095026/nerfstudio_models/step-000014999.ckpt` | `renders/m4_constrained_appearance_alpha_depth_dir_xy_camera_iui3_redsea_15000_20260723_095026_pureJ_eval/output.json` |
| M4b | `outputs/m4b_dc_balance_constrained_appearance_alpha_depth_dir_xy_camera_iui3_redsea_15000/water-splatting/m4b_dc_balance_constrained_appearance_alpha_depth_dir_xy_camera_iui3_redsea_15000_20260723_101241/nerfstudio_models/step-000014999.ckpt` | `renders/m4b_dc_balance_constrained_appearance_alpha_depth_dir_xy_camera_iui3_redsea_15000_20260723_101241/output.json` |
| M4c | `outputs/m4c_dc_balance0003_constrained_appearance_alpha_depth_dir_xy_camera_iui3_redsea_15000/water-splatting/m4c_dc_balance0003_constrained_appearance_alpha_depth_dir_xy_camera_iui3_redsea_15000_20260723_103115/nerfstudio_models/step-000014999.ckpt` | `renders/m4c_dc_balance0003_constrained_appearance_alpha_depth_dir_xy_camera_iui3_redsea_15000_20260723_103115/output.json` |

## Reproduction Commands

### Baseline

Original baseline was already reproduced with:

```text
PSNR=29.8790, SSIM=0.9105, LPIPS=0.1810
```

### M1

```bash
GPU=6 MEDIUM_CONTEXT_MODE=dir_xy_camera MAX_NUM_ITERATIONS=15000 \
  EXPERIMENT_NAME=m1_dir_xy_camera_iui3_redsea_15000 \
  bash scripts/experiments/m1_context_medium_iui3_redsea.sh
```

### M2

```bash
GPU=7 OWNERSHIP_MODE=alpha_depth MEDIUM_CONTEXT_MODE=dir_xy_camera MAX_NUM_ITERATIONS=15000 \
  EXPERIMENT_NAME=m2_alpha_depth_dir_xy_camera_iui3_redsea_15000 \
  bash scripts/experiments/m2_infinite_water_iui3_redsea.sh
```

### M3 Diagnostic

```bash
GPU=8 MAX_NUM_ITERATIONS=15000 RUN_EVAL=1 CLEANUP_DRY_RUN=True \
  CLEANUP_START_STEP=12000 CLEANUP_INTERVAL=500 \
  EXPERIMENT_NAME=m3_cleanup_diag_alpha_depth_dir_xy_camera_iui3_redsea_15000 \
  bash scripts/experiments/m3_cleanup_diagnostic_iui3_redsea.sh
```

### M4 Main

```bash
GPU=9 MAX_NUM_ITERATIONS=15000 RUN_EVAL=1 \
  EXPERIMENT_NAME=m4_constrained_appearance_alpha_depth_dir_xy_camera_iui3_redsea_15000 \
  bash scripts/experiments/m4_constrained_appearance_iui3_redsea.sh
```

### M4b

```bash
GPU=9 MAX_NUM_ITERATIONS=15000 RUN_EVAL=1 \
  bash scripts/experiments/m4b_dc_balance_iui3_redsea.sh
```

### M4c

```bash
GPU=9 MAX_NUM_ITERATIONS=15000 RUN_EVAL=1 \
  bash scripts/experiments/m4c_dc_balance0003_iui3_redsea.sh
```

## Visual / Diagnostic Files

| File | Content |
|---|---|
| `renders/m4_vs_m2_pureJ_contact_sheet_20260723.png` | M2 vs M4 pure `J` visual comparison |
| `renders/m2_m4_m4b_pureJ_contact_sheet_20260723.png` | M2 vs M4 vs M4b pure `J` visual comparison |
| `renders/m2_m4_m4b_m4c_pureJ_contact_sheet_20260723.png` | M2 vs M4 vs M4b vs M4c pure `J` visual comparison |

## Diagnostics

| Item | Result |
|---|---|
| Step1 no-numeric-change check | Matches baseline metrics exactly |
| M3 cleanup dry-run candidates | 0 candidates at logged steps 12000, 12500, 13000, 13500, 14000, 14500 |
| M2 far-water residual check | Far `accumulation` mean `0.000480`; far `rgb_object` luma mean `0.0000249`; object-luma > `0.03` fraction `0.0234%` |
| M4b visual issue | DC balance `0.001` reduces color dominance metrics but shows visible boundary streak/light-spike artifacts |
| M4c status | Best current balance among tested M4 variants; no obvious M4b-style boundary streaking in the contact sheet |

## Smoke / CLI Checks

These were used only to validate code paths and CLI flags, not as formal quantitative experiments.

| Check | Status / Log |
|---|---|
| M1 `dir_xy_camera` smoke | Passed; `logs/m1_smoke_dir_xy_camera_iui3_redsea_20260723_072056/run_manifest.txt` |
| M1 `dir_xy_depth_camera` smoke | Passed; `logs/m1_smoke_dir_xy_depth_camera_iui3_redsea_20260723_072121/run_manifest.txt` |
| M2 smoke | Passed; `logs/m2_smoke_alpha_depth_iui3_redsea_20260723_075927/run_manifest.txt` |
| M3 smoke | Passed; `logs/m3_smoke_cleanup_diag_iui3_redsea_20260723_083023/run_manifest.txt` |
| M3 CLI ownership-source smoke | Passed; `logs/m3_cli_ownership_source_smoke_iui3_redsea_20260723_084603/run_manifest.txt` |
| M4 smoke | Passed; `logs/m4_smoke_constrained_appearance_iui3_redsea_20260723_094843/run_manifest.txt` |
| M4 channel-constraint smoke | Passed; `logs/m4_smoke_channel_constraints_iui3_redsea_20260723_100909/run_manifest.txt` |
| M4 active channel-constraint smoke | Passed; `logs/m4_smoke_channel_constraints_active_iui3_redsea_20260723_100936/run_manifest.txt` |

