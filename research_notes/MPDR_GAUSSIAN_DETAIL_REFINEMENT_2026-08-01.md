# MPDR Gaussian Detail Refinement - 2026-08-01

## Objective

Return to M1 underwater novel-view quality as the primary objective and test whether medium-detached, object-safe high-frequency residual evidence can improve Gaussian refinement without changing the underwater renderer, inference outputs, or adding dewatered-J supervision.

M1 base setting:

- `medium_context_mode=dir_xy_camera`
- `b_inf_mode=tied`
- `B_inf = medium_rgb = A`
- Old TACMD/TMICA/TBAP/J/cleanup/capacity/pruning directions explicitly disabled in experiment wrappers.

## Code Changes

- Added **Medium-Detached Persistent Detail Refinement (MPDR)** flags to `WaterSplattingModelConfig`.
- Added detached Sobel + 5x5 high-pass detail evidence in `get_loss_dict`, using `outputs["pred_image"]` and the masked ground-truth image.
- Added object-safe support:
  - `q_hit = hit_confidence`
  - `q_conc = exp(-depth_std_relative / kappa)`
  - `q_acc = sigmoid((accumulation - a0) / t_a)`
  - `q_obj = q_hit * q_conc * q_acc`
- Sampled `q_obj * E_detail` and `q_obj` at Gaussian projected centers via existing `sample_pixel_map_at_gaussians`.
- Maintained per-Gaussian EMA buffers:
  - `mpdr_detail_ema`
  - `mpdr_obj_ema`
  - `mpdr_visibility_count`
- Synchronized MPDR buffers across split, duplicate, and cull operations.
- Added MPDR refinement candidates by OR-ing detail candidates into existing `high_grads`, preserving the original split/duplicate/cull rules.
- Added JSONL diagnostics at refinement events via `mpdr_log_path`.
- Reused existing constrained appearance SH delay path for G3 SH curriculum instead of adding a new SH scheduler.

## Config Flags

```text
mpdr_enabled: bool = False
mpdr_diagnostic_only: bool = False
mpdr_start_step: int = 500
mpdr_stop_step: int = 10000
mpdr_detail_score_weight: float = 0.25
mpdr_detail_ema_decay: float = 0.90
mpdr_highpass_weight: float = 0.35
mpdr_min_visibility_count: int = 4
mpdr_object_support_threshold: float = 0.20
mpdr_detail_threshold_quantile: float = 0.75
mpdr_top_fraction: float = 0.05
mpdr_max_extra_fraction_per_refine: float = 0.02
mpdr_obj_accum_mid: float = 0.35
mpdr_obj_accum_temp: float = 0.08
mpdr_depth_concentration_kappa: float = 0.25
mpdr_log_path: Optional[str] = None
```

## Diagnostic Method

Added read-only diagnostic:

```text
scripts/diagnostics/diagnose_gaussian_detail_quality.py
```

Inputs:

- `--load-config`
- `--load-step`
- `--test-mode`
- `--max-images`
- `--scene-name`
- `--output-dir`
- optional `--log-dir`
- optional `--save-images`

Statistics:

- Capacity: total Gaussians and parsed split/duplicate/cull counts where logs are available.
- Scale: world scale max/min axis, projected radii, max-axis/min-axis ratio.
- Visibility: visible Gaussian count, mean visibility fraction, low-visibility ratio.
- Geometry: depth, depth concentration, hit confidence, far accumulation.
- Detail: Sobel residual, high-pass residual, combined detail residual, object-safe detail residual.
- Appearance: full RGB residual, `features_dc` norm, `features_rest` norm, feature-level full-SH-minus-DC RGB proxy.
- Correlations: detail/object-safe detail versus scale, projected radius, depth, hit confidence.

Limitations:

- Frozen eval has no active `xys_grad_norm` buffer, so `object_safe_detail_vs_current_densification_gradient` is reported unavailable.
- DC-only RGB is a feature-level proxy; a true rendered DC-only image was not implemented because the eval rasterizer path does not expose an easy post-hoc DC-only render without changing the model path.

## Phase 0 Audit

Initial four-scene diagnostic comparing baseline WaterSplatting and M1:

| Scene | Variant | Gaussians | Detail | Obj-Safe Detail | Radius p95 | Scale p95 | SH Delta | Corr Safe/Depth |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Curasao | Baseline | 1,096,944 | 0.028621 | 0.021320 | 17.0 | 0.009335 | 0.190172 | -0.2153 |
| Curasao | M1 | 1,106,714 | 0.029355 | 0.018865 | 13.0 | 0.007631 | 0.147504 | -0.1481 |
| IUI3 | Baseline | 801,947 | 0.056602 | 0.024343 | 13.0 | 0.010408 | 0.112602 | -0.1045 |
| IUI3 | M1 | 807,502 | 0.056598 | 0.021914 | 13.0 | 0.009797 | 0.107083 | -0.0633 |
| JapaneseGradens | Baseline | 857,831 | 0.058229 | 0.031094 | 14.0 | 0.008209 | 0.122403 | -0.3054 |
| JapaneseGradens | M1 | 861,508 | 0.058598 | 0.024921 | 13.0 | 0.007497 | 0.124334 | -0.2656 |
| Panama | Baseline | 1,144,343 | 0.038810 | 0.020013 | 17.0 | 0.008926 | 0.124460 | -0.1697 |
| Panama | M1 | 1,173,293 | 0.037703 | 0.022744 | 16.0 | 0.007793 | 0.137460 | -0.1830 |

Phase 0 interpretation:

- JapaneseGradens M1 does **not** show lower Gaussian count than baseline; it has slightly more Gaussians.
- JapaneseGradens M1 total detail residual is slightly worse than baseline, but object-safe detail residual is lower.
- JapaneseGradens M1 SH delta is slightly higher than baseline.
- This weakens a simple Gaussian capacity shortage hypothesis and leaves SH behavior as a plausible contributor.

## Smoke Tests

MPDR smoke:

```bash
STAMP=20260801_mpdr_smoke MAX_NUM_ITERATIONS=800 MODEL_NUM_STEPS=800 RUN_EVAL=0 STEPS_PER_SAVE=800 SAVE_ONLY_LATEST_CHECKPOINT=True GPU=6 scripts/experiments/mpdr_g1_detail025_japanesegradens_5k.sh
```

Result:

- Passed startup, training, densification, split/dup/cull buffer sync, and JSONL logging.
- No `MPDR warning`, `Traceback`, `RuntimeError`, CUDA, or autograd error found.
- Final smoke MPDR log row at step 700:
  - `total_gaussians=21140`
  - `mpdr_eligible_count=5254`
  - `mpdr_extra_candidate_count=423`
  - `mpdr_split_count=423`
  - `mpdr_duplicate_count=0`
  - `post_refine_gaussians=24430`
  - `culled_count=31571`

SH curriculum smoke:

```bash
STAMP=20260801_mpdr_smoke MAX_NUM_ITERATIONS=120 MODEL_NUM_STEPS=120 RUN_EVAL=0 STEPS_PER_SAVE=120 SAVE_ONLY_LATEST_CHECKPOINT=True GPU=7 scripts/experiments/mpdr_g3_shcurr_japanesegradens_5k.sh
```

Result:

- Passed startup and short training run.

## Experiment Scripts

Common wrapper:

```text
scripts/experiments/mpdr_5k_common.sh
```

JapaneseGradens scripts:

```text
scripts/experiments/mpdr_g0_m1_japanesegradens_5k.sh
scripts/experiments/mpdr_g1_detail025_japanesegradens_5k.sh
scripts/experiments/mpdr_g2_detail050_japanesegradens_5k.sh
scripts/experiments/mpdr_g3_shcurr_japanesegradens_5k.sh
scripts/experiments/mpdr_g4_best_combo_japanesegradens_5k.sh
```

IUI3 scripts:

```text
scripts/experiments/mpdr_g0_m1_iui3_5k.sh
scripts/experiments/mpdr_g1_detail025_iui3_5k.sh
scripts/experiments/mpdr_g2_detail050_iui3_5k.sh
scripts/experiments/mpdr_g3_shcurr_iui3_5k.sh
scripts/experiments/mpdr_g4_best_combo_iui3_5k.sh
```

5k run commands:

```bash
STAMP=20260801_mpdr_5k RUN_EVAL=1 GPU=6 scripts/experiments/mpdr_g0_m1_japanesegradens_5k.sh
STAMP=20260801_mpdr_5k RUN_EVAL=1 GPU=7 scripts/experiments/mpdr_g1_detail025_japanesegradens_5k.sh
STAMP=20260801_mpdr_5k RUN_EVAL=1 GPU=8 scripts/experiments/mpdr_g2_detail050_japanesegradens_5k.sh
STAMP=20260801_mpdr_5k RUN_EVAL=1 GPU=9 scripts/experiments/mpdr_g3_shcurr_japanesegradens_5k.sh

STAMP=20260801_mpdr_5k RUN_EVAL=1 GPU=6 scripts/experiments/mpdr_g0_m1_iui3_5k.sh
STAMP=20260801_mpdr_5k RUN_EVAL=1 GPU=7 scripts/experiments/mpdr_g1_detail025_iui3_5k.sh
STAMP=20260801_mpdr_5k RUN_EVAL=1 GPU=8 scripts/experiments/mpdr_g2_detail050_iui3_5k.sh
STAMP=20260801_mpdr_5k RUN_EVAL=1 GPU=9 scripts/experiments/mpdr_g3_shcurr_iui3_5k.sh
```

Gaussian detail diagnostics:

```bash
CUDA_VISIBLE_DEVICES=6 /opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/diagnose_gaussian_detail_quality.py --load-config outputs/mpdr_g0_m1_japanesegradens_5k/water-splatting/mpdr_g0_m1_japanesegradens_5k_20260801_mpdr_5k/config.yml --load-step 4999 --test-mode inference --max-images 3 --scene-name japanesegradens_g0_m1_5k --output-dir renders/gaussian_detail_quality_20260801_5k/japanesegradens_g0_m1 --log-dir logs/mpdr_g0_m1_japanesegradens_5k_20260801_mpdr_5k
CUDA_VISIBLE_DEVICES=7 /opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/diagnose_gaussian_detail_quality.py --load-config outputs/mpdr_g1_detail025_japanesegradens_5k/water-splatting/mpdr_g1_detail025_japanesegradens_5k_20260801_mpdr_5k/config.yml --load-step 4999 --test-mode inference --max-images 3 --scene-name japanesegradens_g1_detail025_5k --output-dir renders/gaussian_detail_quality_20260801_5k/japanesegradens_g1_detail025 --log-dir logs/mpdr_g1_detail025_japanesegradens_5k_20260801_mpdr_5k
CUDA_VISIBLE_DEVICES=8 /opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/diagnose_gaussian_detail_quality.py --load-config outputs/mpdr_g2_detail050_japanesegradens_5k/water-splatting/mpdr_g2_detail050_japanesegradens_5k_20260801_mpdr_5k/config.yml --load-step 4999 --test-mode inference --max-images 3 --scene-name japanesegradens_g2_detail050_5k --output-dir renders/gaussian_detail_quality_20260801_5k/japanesegradens_g2_detail050 --log-dir logs/mpdr_g2_detail050_japanesegradens_5k_20260801_mpdr_5k
CUDA_VISIBLE_DEVICES=9 /opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/diagnose_gaussian_detail_quality.py --load-config outputs/mpdr_g3_shcurr_japanesegradens_5k/water-splatting/mpdr_g3_shcurr_japanesegradens_5k_20260801_mpdr_5k/config.yml --load-step 4999 --test-mode inference --max-images 3 --scene-name japanesegradens_g3_shcurr_5k --output-dir renders/gaussian_detail_quality_20260801_5k/japanesegradens_g3_shcurr --log-dir logs/mpdr_g3_shcurr_japanesegradens_5k_20260801_mpdr_5k

CUDA_VISIBLE_DEVICES=6 /opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/diagnose_gaussian_detail_quality.py --load-config outputs/mpdr_g0_m1_iui3_5k/water-splatting/mpdr_g0_m1_iui3_5k_20260801_mpdr_5k/config.yml --load-step 4999 --test-mode inference --max-images 3 --scene-name iui3_g0_m1_5k --output-dir renders/gaussian_detail_quality_20260801_5k/iui3_g0_m1 --log-dir logs/mpdr_g0_m1_iui3_5k_20260801_mpdr_5k
CUDA_VISIBLE_DEVICES=7 /opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/diagnose_gaussian_detail_quality.py --load-config outputs/mpdr_g1_detail025_iui3_5k/water-splatting/mpdr_g1_detail025_iui3_5k_20260801_mpdr_5k/config.yml --load-step 4999 --test-mode inference --max-images 3 --scene-name iui3_g1_detail025_5k --output-dir renders/gaussian_detail_quality_20260801_5k/iui3_g1_detail025 --log-dir logs/mpdr_g1_detail025_iui3_5k_20260801_mpdr_5k
CUDA_VISIBLE_DEVICES=8 /opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/diagnose_gaussian_detail_quality.py --load-config outputs/mpdr_g2_detail050_iui3_5k/water-splatting/mpdr_g2_detail050_iui3_5k_20260801_mpdr_5k/config.yml --load-step 4999 --test-mode inference --max-images 3 --scene-name iui3_g2_detail050_5k --output-dir renders/gaussian_detail_quality_20260801_5k/iui3_g2_detail050 --log-dir logs/mpdr_g2_detail050_iui3_5k_20260801_mpdr_5k
CUDA_VISIBLE_DEVICES=9 /opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/diagnose_gaussian_detail_quality.py --load-config outputs/mpdr_g3_shcurr_iui3_5k/water-splatting/mpdr_g3_shcurr_iui3_5k_20260801_mpdr_5k/config.yml --load-step 4999 --test-mode inference --max-images 3 --scene-name iui3_g3_shcurr_5k --output-dir renders/gaussian_detail_quality_20260801_5k/iui3_g3_shcurr --log-dir logs/mpdr_g3_shcurr_iui3_5k_20260801_mpdr_5k
```

## 5k Results

JapaneseGradens:

| Run | PSNR | dPSNR | SSIM | dSSIM | LPIPS | dLPIPS | Gaussians | Growth vs G0 | Final Iter Time | Detail | Obj-Safe Detail | MPDR Extras | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| G0 M1 | 22.7388 | +0.0000 | 0.799511 | +0.000000 | 0.238191 | +0.000000 | 722,008 | +0.0% | 27.416 ms | 0.099586 | 0.040197 | 0 | control |
| G1 MPDR 0.25 | 22.9237 | +0.1850 | 0.801453 | +0.001942 | 0.237927 | -0.000265 | 804,538 | +11.4% | 30.996 ms | 0.098947 | 0.032277 | 230,953 | fail: LPIPS gain below 0.0015 |
| G2 MPDR 0.50 | 22.7985 | +0.0598 | 0.798200 | -0.001311 | 0.239957 | +0.001765 | 891,292 | +23.4% | 33.196 ms | 0.100344 | 0.034975 | 240,559 | fail: PSNR/SSIM/LPIPS/growth |
| G3 SH curr | 22.9312 | +0.1925 | 0.801366 | +0.001855 | 0.239613 | +0.001422 | 707,251 | -2.0% | 26.039 ms | 0.099156 | 0.030540 | 0 | fail: LPIPS worsened |

IUI3:

| Run | PSNR | dPSNR | SSIM | dSSIM | LPIPS | dLPIPS | Gaussians | Growth vs G0 | Final Iter Time | Detail | Obj-Safe Detail | MPDR Extras | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| G0 M1 | 25.5416 | +0.0000 | 0.764780 | +0.000000 | 0.308659 | +0.000000 | 640,113 | +0.0% | 25.289 ms | 0.112163 | 0.029278 | 0 | control |
| G1 MPDR 0.25 | 25.4668 | -0.0748 | 0.764452 | -0.000328 | 0.305991 | -0.002668 | 711,238 | +11.1% | 28.531 ms | 0.112429 | 0.028623 | 191,817 | fail: PSNR safety |
| G2 MPDR 0.50 | 25.7990 | +0.2574 | 0.764051 | -0.000729 | 0.305910 | -0.002750 | 741,910 | +15.9% | 32.408 ms | 0.112587 | 0.027306 | 191,790 | fail: growth/time |
| G3 SH curr | 25.8528 | +0.3112 | 0.763462 | -0.001318 | 0.305873 | -0.002786 | 630,944 | -1.4% | 23.510 ms | 0.112740 | 0.031303 | 0 | safety pass by PSNR/LPIPS/growth, but visual artifact observed |

Notes:

- Final iteration time is used as a training-time proxy because logs do not expose a clean total wall-clock summary line.
- JapaneseGradens G1 is the best MPDR candidate numerically, with better PSNR/SSIM and lower object-safe detail residual, but it fails the user-defined LPIPS gate.
- IUI3 G1 is visually clean and lowers LPIPS, but violates the PSNR safety limit by -0.0748 dB.
- IUI3 G3 improves PSNR/LPIPS and has no growth issue, but one eval view shows a visible yellow/green artifact, so it is not a safe cross-scene recommendation.
- G2 is too aggressive: JapaneseGradens growth is +23.4%, IUI3 growth is +15.9%, and final iteration time overhead exceeds 15%.

## Visual Outputs

Generated contact sheets under ignored render output:

```text
renders/mpdr_contact_sheets_20260801/japanesegradens_g0_g3_5k_eval_rgb_contact.png
renders/mpdr_contact_sheets_20260801/iui3_g0_g3_5k_eval_rgb_contact.png
```

Visual check:

- JapaneseGradens differences are subtle; no obvious new open-water floaters in G1/G3.
- IUI3 G1 is clean in sampled views.
- IUI3 G3 introduces a visible yellow/green artifact in one sampled view.

## Checkpoints and Outputs

5k checkpoints:

```text
outputs/mpdr_g0_m1_japanesegradens_5k/water-splatting/mpdr_g0_m1_japanesegradens_5k_20260801_mpdr_5k/nerfstudio_models/step-000004999.ckpt
outputs/mpdr_g1_detail025_japanesegradens_5k/water-splatting/mpdr_g1_detail025_japanesegradens_5k_20260801_mpdr_5k/nerfstudio_models/step-000004999.ckpt
outputs/mpdr_g2_detail050_japanesegradens_5k/water-splatting/mpdr_g2_detail050_japanesegradens_5k_20260801_mpdr_5k/nerfstudio_models/step-000004999.ckpt
outputs/mpdr_g3_shcurr_japanesegradens_5k/water-splatting/mpdr_g3_shcurr_japanesegradens_5k_20260801_mpdr_5k/nerfstudio_models/step-000004999.ckpt

outputs/mpdr_g0_m1_iui3_5k/water-splatting/mpdr_g0_m1_iui3_5k_20260801_mpdr_5k/nerfstudio_models/step-000004999.ckpt
outputs/mpdr_g1_detail025_iui3_5k/water-splatting/mpdr_g1_detail025_iui3_5k_20260801_mpdr_5k/nerfstudio_models/step-000004999.ckpt
outputs/mpdr_g2_detail050_iui3_5k/water-splatting/mpdr_g2_detail050_iui3_5k_20260801_mpdr_5k/nerfstudio_models/step-000004999.ckpt
outputs/mpdr_g3_shcurr_iui3_5k/water-splatting/mpdr_g3_shcurr_iui3_5k_20260801_mpdr_5k/nerfstudio_models/step-000004999.ckpt
```

Metrics JSONs:

```text
renders/mpdr_g0_m1_japanesegradens_5k_20260801_mpdr_5k/output.json
renders/mpdr_g1_detail025_japanesegradens_5k_20260801_mpdr_5k/output.json
renders/mpdr_g2_detail050_japanesegradens_5k_20260801_mpdr_5k/output.json
renders/mpdr_g3_shcurr_japanesegradens_5k_20260801_mpdr_5k/output.json

renders/mpdr_g0_m1_iui3_5k_20260801_mpdr_5k/output.json
renders/mpdr_g1_detail025_iui3_5k_20260801_mpdr_5k/output.json
renders/mpdr_g2_detail050_iui3_5k_20260801_mpdr_5k/output.json
renders/mpdr_g3_shcurr_iui3_5k_20260801_mpdr_5k/output.json
```

MPDR JSONL logs:

```text
logs/mpdr_g0_m1_japanesegradens_5k_20260801_mpdr_5k/mpdr.jsonl
logs/mpdr_g1_detail025_japanesegradens_5k_20260801_mpdr_5k/mpdr.jsonl
logs/mpdr_g2_detail050_japanesegradens_5k_20260801_mpdr_5k/mpdr.jsonl
logs/mpdr_g3_shcurr_japanesegradens_5k_20260801_mpdr_5k/mpdr.jsonl

logs/mpdr_g0_m1_iui3_5k_20260801_mpdr_5k/mpdr.jsonl
logs/mpdr_g1_detail025_iui3_5k_20260801_mpdr_5k/mpdr.jsonl
logs/mpdr_g2_detail050_iui3_5k_20260801_mpdr_5k/mpdr.jsonl
logs/mpdr_g3_shcurr_iui3_5k_20260801_mpdr_5k/mpdr.jsonl
```

## Decision

Do **not** run G4 or 15k in this round.

Rationale:

- No JapaneseGradens variant passes the 5k gate.
- G1 MPDR 0.25 is the closest candidate but fails LPIPS on JapaneseGradens and PSNR safety on IUI3.
- G2 MPDR 0.50 is too aggressive in Gaussian growth and time overhead.
- G3 SH curriculum is promising for PSNR but worsens JapaneseGradens LPIPS and shows an IUI3 visual artifact.

## Conclusion

Current evidence does **not** confirm that M1's JapaneseGradens loss is primarily caused by insufficient Gaussian detail/refinement allocation.

MPDR can move object-safe detail residual down and improve JapaneseGradens PSNR/SSIM at `lambda_detail_score=0.25`, but it does not clear the perceptual gate and is not safely neutral on IUI3. The next branch should treat MPDR as diagnostic/ablation evidence, not as a validated 15k candidate. The stronger next question is likely SH/medium learning interaction, scene scale/datamanager/camera behavior, or a more conservative appearance curriculum rather than more densification pressure.
