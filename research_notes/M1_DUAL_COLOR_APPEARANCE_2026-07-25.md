# M1 Dual-Color Appearance Experiments

Date: 2026-07-25

## Objective

Evaluate the new M1-based intrinsic-underwater dual-color appearance direction from:

`plan/WaterSplatting_M1_DualColor新模块实验方案.md`

The goal was to keep M1 `dir_xy_camera` underwater reconstruction quality while replacing the clear `J` branch with:

```text
DC intrinsic color + eta_l * SH luminance residual + eta_c * SH chroma residual
```

M2 ownership/capacity mechanisms were not used.

## Code Changes

- Added `DualColorOutput` and `compute_dual_gaussian_colors()` in `water_splatting/fields/gaussian_appearance.py`.
- Added model flags in `water_splatting/water_splatting.py`:
  - `dual_color_enabled`
  - `clear_sh_luminance_scale`
  - `clear_sh_chroma_scale`
  - `lambda_intrinsic_near_anchor`
  - `lambda_view_residual_mean`
  - `lambda_clear_chroma`
  - `dual_color_loss_start_step`
  - `dual_color_loss_ramp_steps`
  - `dual_color_near_transmission_threshold`
  - `dual_color_near_transmission_temp`
  - `dual_color_freeze_geometry`
  - `dual_color_freeze_medium`
- Implemented the no-CUDA prototype: underwater RGB still uses full SH; clear `J` can use a second rasterization with intrinsic colors over the same geometry/opacities/medium.
- Added DualColor losses:
  - near-transmission intrinsic anchor
  - visible view-residual mean anchor
  - visible chroma residual penalty
- Added scripts:
  - `scripts/diagnostics/diagnose_dc_sh_clear_appearance.py`
  - `scripts/experiments/dual_color_phase1_dc_sh_diagnostic_iui3.sh`
  - `scripts/experiments/dual_color_stage1_frozen_geometry_iui3.sh`
  - `scripts/experiments/dual_color_stage2_joint_refine_iui3.sh`
  - `scripts/experiments/dual_color_funa_clear_gt.sh`

Default behavior remains unchanged because `dual_color_enabled=False`.

## Baseline

Fixed M1 baseline:

```text
M1 dir_xy_camera
PSNR = 31.1314
SSIM = 0.9120
LPIPS = 0.1750
J blue dominance = 0.1691
water J luma = 0.000928
far accumulation = 0.407096
far clear luma = 0.083962
```

Reference config:

```text
outputs/m1_dir_xy_camera_iui3_redsea_15000/water-splatting/m1_dir_xy_camera_iui3_redsea_15000_20260723_072412/config.yml
```

## Phase 1: Post-hoc DC/SH Clear Appearance

Command:

```bash
GPU=6 MAX_IMAGES=4 VARIANTS=A0,A1,A2,A3,A4 SAVE_IMAGES=1 \
STAMP=20260725_dual_phase1_A0_A4 \
scripts/experiments/dual_color_phase1_dc_sh_diagnostic_iui3.sh
```

Output:

```text
renders/dual_color_phase1_dc_sh_iui3_redsea_20260725_dual_phase1_A0_A4/
```

| Variant | Clear branch | water J luma | water B-G | far J luma | J blue dom |
|---|---|---:|---:|---:|---:|
| A0 | full_sh | 0.000928 | 0.001864 | 0.083962 | 0.1691 |
| A1 | dc_only | 0.002156 | 0.001505 | 0.091751 | 0.2158 |
| A2 | dc_luma | 0.001035 | 0.001918 | 0.082902 | 0.2070 |
| A3 | dc_luma_chroma005 | 0.001024 | 0.001923 | 0.082940 | 0.2052 |
| A4 | dc_luma_chroma010 | 0.001013 | 0.001927 | 0.082978 | 0.2035 |

Interpretation:

- DC-only reduces water-region blue-minus-green slightly, but it worsens water J luma, far J luma, and global J blue dominance.
- DC + luminance slightly lowers far J luma versus A0, but J blue dominance rises materially from `0.1691` to `0.2070`.
- The clear-color problem is not simply “high-order SH chroma residual contaminates J”.

## Phase 2/3: Frozen-Geometry DualColor Fine-Tuning

Smoke:

```bash
GPU=7 FINETUNE_STEPS=10 RUN_EVAL=0 RUN_DIAGNOSTICS=0 \
STAMP=smoke_dual_stage1_C2_v3 \
EXPERIMENT_NAME=smoke_dual_stage1_C2_iui3_v3 \
CLEAR_SH_LUMINANCE_SCALE=1.0 CLEAR_SH_CHROMA_SCALE=0.0 \
scripts/experiments/dual_color_stage1_frozen_geometry_iui3.sh
```

The first smoke exposed that Nerfstudio treats `max_num_iterations` as additional iterations after resume. The script was fixed to use:

```text
TRAIN_MAX_NUM_ITERATIONS = FINETUNE_STEPS
MODEL_NUM_STEPS = BASE_STEP + FINETUNE_STEPS
```

Main configs:

```text
C1: DC only
C2: DC + luminance
C3: C2 + near anchor 1e-4
C4: C3 + residual mean 1e-4
C5: C4 + chroma 1e-4
C6: DC + luminance + 0.05 chroma + all anchors 1e-4
```

| Config | PSNR | SSIM | LPIPS | J blue dom | water J | object J ret | boundary ret | far accum | far clear |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1_dc_only | 31.1314 | 0.9120 | 0.1750 | 0.2158 | 0.002156 | 1.0013 | 0.8744 | 0.407096 | 0.091751 |
| C2_dc_luma | 31.1314 | 0.9120 | 0.1750 | 0.2070 | 0.001035 | 0.9951 | 0.9953 | 0.407096 | 0.082902 |
| C3_dc_luma_near1e4 | 31.1314 | 0.9120 | 0.1750 | 0.2070 | 0.001035 | 0.9951 | 0.9953 | 0.407096 | 0.082902 |
| C4_dc_luma_near_mean1e4 | 31.1314 | 0.9120 | 0.1750 | 0.2070 | 0.001035 | 0.9951 | 0.9953 | 0.407096 | 0.082902 |
| C5_dc_luma_near_mean_chroma1e4 | 31.1314 | 0.9120 | 0.1750 | 0.2070 | 0.001035 | 0.9951 | 0.9953 | 0.407096 | 0.082902 |
| C6_dc_luma_chroma005_all1e4 | 31.1314 | 0.9120 | 0.1750 | 0.2052 | 0.001024 | 0.9953 | 0.9956 | 0.407096 | 0.082940 |

Key output paths:

```text
renders/dual_color_stage1_C1_dc_only_seed42_iui3_redsea_2000_20260725_stage1_C1_dc_only/
renders/dual_color_stage1_C2_dc_luma_seed42_iui3_redsea_2000_20260725_stage1_C2_dc_luma/
renders/dual_color_stage1_C3_dc_luma_near1e4_seed42_iui3_redsea_2000_20260725_stage1_C3_dc_luma_near1e4/
renders/dual_color_stage1_C4_dc_luma_near_mean1e4_seed42_iui3_redsea_2000_20260725_stage1_C4_dc_luma_near_mean1e4/
renders/dual_color_stage1_C5_dc_luma_near_mean_chroma1e4_seed42_iui3_redsea_2000_20260725_stage1_C5_dc_luma_near_mean_chroma1e4/
renders/dual_color_stage1_C6_dc_luma_chroma005_all1e4_seed42_iui3_redsea_2000_20260725_stage1_C6_dc_luma_chroma005_all1e4/
```

Checkpoints are under the corresponding ignored `outputs/dual_color_stage1_*/.../nerfstudio_models/step-000016999.ckpt`.

## Decision

No DualColor config is promoted to stage2.

Reasons:

- Underwater metrics are preserved, but only because the underwater branch remains full SH and geometry/medium are frozen.
- J blue dominance worsens versus M1/A0 in every dual clear branch:
  - M1/A0: `0.1691`
  - Best dual result C6: `0.2052`
- Water J leakage is not improved:
  - M1/A0: `0.000928`
  - Best dual result C6: `0.001024`
- Far clear luma improves only marginally in C2/C6 (`~0.0830` vs `0.0840`) and does not compensate for the worse blue dominance.
- C1 DC-only hurts boundary retention (`0.8744`) and should be discarded.

## Next Step

Do not run stage2 medium joint refinement or multi-seed validation for these configs.

Recommended next technical direction:

1. Treat the negative result as evidence that M1 clear residual is not primarily caused by SH chroma leakage.
2. Revisit geometry/opacity or medium-field identifiability before adding more appearance-only constraints.
3. If continuing DualColor, add a real external clear-color supervision signal or pseudo-depth/surface support; do not rely only on unsupervised DC/SH decomposition.

## FUNA Status

FUNA clear-GT validation was not run in this repository because no FUNA dataset layout / clean GT path is configured here. The script `scripts/experiments/dual_color_funa_clear_gt.sh` intentionally fails unless explicit FUNA inputs are supplied, so it does not guess paths or use old-repo workflows.
