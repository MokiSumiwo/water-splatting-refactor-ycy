# M2 Refinement E1-E4 Results - 2026-07-24

## Scope

First-stage M2 refinement on IUI3-RedSea, without CUDA changes. The stable mainline remains M1 `dir_xy_camera`. This pass only adds modular M2 controls and runs E1-E4 short sanity plus 15k main experiments.

## Code Changes

- Added `infinite_water_compose_mode={rgb_mix,tail_approx}` in `water_splatting/water_splatting.py`.
- Preserved current M2 behavior as `rgb_mix`.
- Added `tail_approx` composition:

```python
tail_gate = (1.0 - render.accumulation).detach().clamp(0.0, 1.0)
rgb = render.rgb + ownership.m_inf * tail_gate * (medium.b_inf - medium_rgb)
```

- Added `ownership_mode=off` support in `water_splatting/ownership/infinite_water_ownership.py`.
- Confirmed/used independent flags:
  - `medium_context_mode=dir_xy_camera`
  - `ownership_mode=off/alpha_only/alpha_depth`
  - `infinite_water_occupancy_limited`
  - `lambda_binf_rgb`
  - `lambda_accumulation_zero`
  - `lambda_near_zero`
  - `infinite_water_loss_start_step`
  - `infinite_water_loss_ramp_steps`
  - `infinite_water_compose_mode=rgb_mix/tail_approx`
- Extended `scripts/experiments/m2_infinite_water_iui3_redsea.sh` to log and pass all key flags.
- Added E1/E2/E3/E4 scripts under `scripts/experiments/`.
- Added `*.tif` and `*.tiff` to `.gitignore`; existing ignored data/output dirs remain protected.

## Validation

- Python compile passed for edited model/ownership files.
- `bash -n` passed for experiment scripts.
- CUDA extension import passed in `/opt/anaconda3/envs/water_splatting`.
- `ns-train water-splatting --help` exposes `--pipeline.model.infinite-water-compose-mode {rgb_mix,tail_approx}`.
- Short sanity runs completed for E1, E4 tail, and E4 tail+delay.
- 15k train/eval completed for E1, E2 sweep, E3, E4 tail, and E4 tail+delay.
- J images are generated as `J = clamp(J_raw, 0, 1)` in eval render directories; legacy clear outputs remain diagnostic only.

## Baselines

| Experiment | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Baseline original | 29.8790 | 0.9105 | 0.1810 |
| M1 `dir_xy_camera` | 31.1314 | 0.9120 | 0.1750 |
| Old M2 `alpha_depth` | 31.0696 | 0.9129 | 0.1771 |

## Main Metrics And Diagnostics

`dPSNR`, `dSSIM`, and `dLPIPS` are relative to M1 `dir_xy_camera`.

| Experiment | Key flags | PSNR | dPSNR | SSIM | dSSIM | LPIPS | dLPIPS | J blue dom | far accum mean | far object luma | alpha>0.05 | object>0.03 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M1 baseline | `dir_xy_camera`, M2 off | 31.1314 | +0.0000 | 0.9120 | +0.0000 | 0.1750 | +0.0000 | n/a | 0.407096 | 0.017047 | 0.654392 | 0.139857 |
| Old M2 | `binf=0.01`, `accum=0.004`, `near=0.001`, `rgb_mix` | 31.0696 | -0.0619 | 0.9129 | +0.0009 | 0.1771 | +0.0021 | n/a | 0.000480 | 0.000025 | 0.001199 | 0.000234 |
| E1 no-near | `binf=0.01`, `accum=0.004`, `near=0`, `rgb_mix` | 31.0598 | -0.0716 | 0.9135 | +0.0014 | 0.1786 | +0.0036 | 0.0540 | 0.002045 | 0.000096 | 0.005555 | 0.000530 |
| E2 accum=0 | `binf=0.005`, `accum=0`, `near=0`, `rgb_mix` | 31.1485 | +0.0170 | 0.9148 | +0.0028 | 0.1767 | +0.0017 | 0.0793 | 0.363098 | 0.000228 | 0.572066 | 0.000000 |
| E2 accum=0.0005 | `binf=0.005`, `accum=0.0005`, `near=0`, `rgb_mix` | 31.2029 | +0.0714 | 0.9136 | +0.0016 | 0.1738 | -0.0012 | 0.1208 | 0.043682 | 0.001779 | 0.081980 | 0.017608 |
| E2 accum=0.001 | `binf=0.005`, `accum=0.001`, `near=0`, `rgb_mix` | 31.0871 | -0.0443 | 0.9124 | +0.0004 | 0.1761 | +0.0011 | 0.0794 | 0.038917 | 0.002999 | 0.081259 | 0.022459 |
| E2 accum=0.002 | `binf=0.005`, `accum=0.002`, `near=0`, `rgb_mix` | 31.2206 | +0.0891 | 0.9140 | +0.0020 | 0.1765 | +0.0015 | 0.0565 | 0.004166 | 0.000149 | 0.008423 | 0.000976 |
| E2 accum=0.004 | `binf=0.005`, `accum=0.004`, `near=0`, `rgb_mix` | 31.0085 | -0.1230 | 0.9130 | +0.0010 | 0.1795 | +0.0045 | 0.0671 | 0.001070 | 0.000183 | 0.005161 | 0.002267 |
| E3 delay/ramp | `binf=0.005`, `accum=0.004`, `near=0`, `start=5000`, `ramp=5000`, `rgb_mix` | 30.9759 | -0.1555 | 0.9125 | +0.0005 | 0.1757 | +0.0007 | 0.0596 | 0.000651 | 0.000048 | 0.002462 | 0.000228 |
| E4 tail | `binf=0.01`, `accum=0.004`, `near=0.001`, `tail_approx` | 31.0078 | -0.1236 | 0.9132 | +0.0012 | 0.1771 | +0.0020 | 0.0230 | 0.000547 | 0.000045 | 0.000943 | 0.000782 |
| E4 tail+delay | `binf=0.005`, `accum=0.004`, `near=0`, `start=5000`, `ramp=5000`, `tail_approx` | 30.9619 | -0.1695 | 0.9116 | -0.0004 | 0.1767 | +0.0017 | 0.0519 | 0.000225 | 0.000007 | 0.000259 | 0.000105 |

## Commands And Artifacts

### E1 no-near

Command:

```bash
GPU=7 MAX_NUM_ITERATIONS=15000 RUN_EVAL=1 EXPERIMENT_NAME=m2_e1_no_near_alpha_depth_dir_xy_camera_iui3_redsea_15000 bash scripts/experiments/m2_e1_no_near_iui3_redsea.sh
```

- Checkpoint: `outputs/m2_e1_no_near_alpha_depth_dir_xy_camera_iui3_redsea_15000/water-splatting/m2_e1_no_near_alpha_depth_dir_xy_camera_iui3_redsea_15000_20260724_024441/nerfstudio_models/step-000014999.ckpt`
- Eval: `renders/m2_e1_no_near_alpha_depth_dir_xy_camera_iui3_redsea_15000_20260724_024441/output.json`
- Diagnostic: `logs/diagnostics/m2_e1_no_near_far_water_20260724/far_water_residual_diagnostic.json`

### E2 accumulation sweep

Command:

```bash
GPU=9 MAX_NUM_ITERATIONS=15000 RUN_EVAL=1 ACCUM_ZERO_WEIGHTS="0 0.0005 0.001 0.002" bash scripts/experiments/m2_e2_accum_sweep_iui3_redsea.sh
GPU=9 MAX_NUM_ITERATIONS=15000 RUN_EVAL=1 ACCUM_ZERO_WEIGHTS="0.004" STAMP=20260724_040937 bash scripts/experiments/m2_e2_accum_sweep_iui3_redsea.sh
```

- Shared flags: `BINF_RGB_WEIGHT=0.005`, `NEAR_ZERO_WEIGHT=0`, `LOSS_START_STEP=1000`, `LOSS_RAMP_STEPS=3000`, `INFINITE_WATER_COMPOSE_MODE=rgb_mix`.
- Best current checkpoint: `outputs/m2_e2_accum0p002_alpha_depth_dir_xy_camera_iui3_redsea_15000/water-splatting/m2_e2_accum0p002_alpha_depth_dir_xy_camera_iui3_redsea_15000_20260724_025830/nerfstudio_models/step-000014999.ckpt`
- Best current eval: `renders/m2_e2_accum0p002_alpha_depth_dir_xy_camera_iui3_redsea_15000_20260724_025830/output.json`
- Best current diagnostic: `logs/diagnostics/m2_e2_accum0p002_far_water_20260724/far_water_residual_diagnostic.json`

### E3 delay/ramp

Command:

```bash
GPU=8 MAX_NUM_ITERATIONS=15000 RUN_EVAL=1 EXPERIMENT_NAME=m2_e3_delay5000_ramp5000_alpha_depth_dir_xy_camera_iui3_redsea_15000 bash scripts/experiments/m2_e3_delay_ramp_iui3_redsea.sh
```

- Checkpoint: `outputs/m2_e3_delay5000_ramp5000_alpha_depth_dir_xy_camera_iui3_redsea_15000/water-splatting/m2_e3_delay5000_ramp5000_alpha_depth_dir_xy_camera_iui3_redsea_15000_20260724_040937/nerfstudio_models/step-000014999.ckpt`
- Eval: `renders/m2_e3_delay5000_ramp5000_alpha_depth_dir_xy_camera_iui3_redsea_15000_20260724_040937/output.json`
- Diagnostic: `logs/diagnostics/m2_e3_delay5000_ramp5000_far_water_20260724/far_water_residual_diagnostic.json`

### E4 tail approximation

Commands:

```bash
GPU=8 MAX_NUM_ITERATIONS=15000 RUN_EVAL=1 EXPERIMENT_NAME=m2_e4_tail_approx_alpha_depth_dir_xy_camera_iui3_redsea_15000 bash scripts/experiments/m2_e4_tail_approx_iui3_redsea.sh
GPU=8 MAX_NUM_ITERATIONS=15000 RUN_EVAL=1 EXPERIMENT_NAME=m2_e4_tail_delay_recommended_alpha_depth_dir_xy_camera_iui3_redsea_15000 bash scripts/experiments/m2_e4_tail_delay_recommended_iui3_redsea.sh
```

- Tail checkpoint: `outputs/m2_e4_tail_approx_alpha_depth_dir_xy_camera_iui3_redsea_15000/water-splatting/m2_e4_tail_approx_alpha_depth_dir_xy_camera_iui3_redsea_15000_20260724_024441/nerfstudio_models/step-000014999.ckpt`
- Tail eval: `renders/m2_e4_tail_approx_alpha_depth_dir_xy_camera_iui3_redsea_15000_20260724_024441/output.json`
- Tail diagnostic: `logs/diagnostics/m2_e4_tail_approx_far_water_20260724/far_water_residual_diagnostic.json`
- Tail+delay checkpoint: `outputs/m2_e4_tail_delay_recommended_alpha_depth_dir_xy_camera_iui3_redsea_15000/water-splatting/m2_e4_tail_delay_recommended_alpha_depth_dir_xy_camera_iui3_redsea_15000_20260724_024441/nerfstudio_models/step-000014999.ckpt`
- Tail+delay eval: `renders/m2_e4_tail_delay_recommended_alpha_depth_dir_xy_camera_iui3_redsea_15000_20260724_024441/output.json`
- Tail+delay diagnostic: `logs/diagnostics/m2_e4_tail_delay_recommended_far_water_20260724/far_water_residual_diagnostic.json`

## Visual Review

- Contact sheet: `renders/m2_refinement_e1_e4_key_candidates_view0000_contact_sheet_20260724.png`.
- E2 `accum=0.002` keeps RGB structure close to M1/old M2 while strongly reducing far Gaussian leakage.
- E2 `accum=0.0005` has the best LPIPS and PSNR among this sweep, but leaves noticeably higher far accumulation and J blue dominance.
- E3 and E4 tail+delay clear far residual most aggressively, but reconstruction metrics and/or visible far structure degrade too much for primary candidate status.
- E4 `tail_approx` gives the lowest J blue dominance among new main runs (`0.0230`) and is useful as a composition option, but its PSNR drop is too large to promote as current default.

## Conclusion

Promote E2 `accum=0.002` as the current M2 candidate:

- `medium_context_mode=dir_xy_camera`
- `ownership_mode=alpha_depth`
- `infinite_water_compose_mode=rgb_mix`
- `infinite_water_occupancy_limited=True`
- `lambda_binf_rgb=0.005`
- `lambda_accumulation_zero=0.002`
- `lambda_near_zero=0`
- `infinite_water_loss_start_step=1000`
- `infinite_water_loss_ramp_steps=3000`

This candidate improves over M1 on PSNR and SSIM and improves over old M2 on PSNR/SSIM while keeping far accumulation far below M1. It misses the strict LPIPS success threshold by about `0.0005` (`+0.0015` vs allowed `+0.0010`), so it should be treated as the best first-stage candidate rather than final.

## Next Steps

1. Run E2 `accum=0.002` on at least one additional seed or repeat run to check variance.
2. Sweep a narrower range around the candidate: `lambda_accumulation_zero in {0.0015, 0.0020, 0.0025}` with `lambda_binf_rgb=0.005`, `lambda_near_zero=0`.
3. Keep `tail_approx` available, but do not make it default until a lower-PSNR-loss variant exists.
4. Start second-stage ownership split only after E2 candidate repeat is stable: `m_support`, `m_render`, `m_capacity`.
5. Implement pseudo-depth teacher as mask-loading plus diagnostics first; do not hardcode pseudo-label paths.
