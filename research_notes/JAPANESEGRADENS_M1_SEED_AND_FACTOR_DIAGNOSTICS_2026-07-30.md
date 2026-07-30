# JapaneseGradens M1 Seed and Factor Diagnostics

Date: 2026-07-30
Branch: `refactor/core-framework`
Starting commit: `0569701 Record cross-scene baseline results`

## Goal

JapaneseGradens-RedSea is the only new scene where M1 underperforms the original WaterSplatting baseline in underwater reconstruction. This round follows the proposed order:

```text
checkpoint trajectory check
-> paired Baseline/M1 multi-seed
-> M1 factor split only if degradation is stable
```

This is not a JapaneseGradens-specific tuning round, and it does not select a better M1 seed against the seed-42 baseline.

## Existing Seed-42 Results

| Method | PSNR | SSIM | LPIPS | J Blue | Far Accum | Far Clear |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 24.9098 | 0.9001 | 0.1171 | 0.2043 | 0.6343 | 0.2359 |
| M1 | 24.7565 | 0.8995 | 0.1204 | 0.1395 | 0.4412 | 0.0542 |

Seed-42 M1 improves residual metrics but hurts underwater reconstruction:

```text
dPSNR  = -0.1533 dB
dSSIM  = -0.0006
dLPIPS = +0.0032
```

## Step-Trajectory Check

Checkpoints evaluated:

```text
step-000005000.ckpt
step-000010000.ckpt
step-000014999.ckpt
```

Script:

```text
scripts/diagnostics/evaluate_per_image_metrics.py
```

Outputs:

```text
renders/japanesegradens_step_diagnostics_20260730/
```

| Step | Method | Mean PSNR | Mean SSIM | Mean LPIPS | Image0 PSNR | Image1 PSNR | Image2 PSNR |
|---:|---|---:|---:|---:|---:|---:|---:|
| 5000 | Baseline | 23.0190 | 0.8029 | 0.2379 | 19.2671 | 26.2292 | 23.5606 |
| 5000 | M1 | 23.1042 | 0.8032 | 0.2438 | 19.3271 | 26.1981 | 23.7875 |
| 10000 | Baseline | 24.9152 | 0.8968 | 0.1221 | 19.4251 | 28.7676 | 26.5531 |
| 10000 | M1 | 24.8257 | 0.8963 | 0.1250 | 19.3950 | 28.6553 | 26.4268 |
| 14999 | Baseline | 24.9098 | 0.9001 | 0.1171 | 19.3326 | 28.7084 | 26.6884 |
| 14999 | M1 | 24.7565 | 0.8995 | 0.1204 | 19.2839 | 28.4479 | 26.5376 |

M1 minus baseline:

| Step | dPSNR | dSSIM | dLPIPS | dPSNR Image0 | dPSNR Image1 | dPSNR Image2 |
|---:|---:|---:|---:|---:|---:|---:|
| 5000 | +0.0853 | +0.0004 | +0.0058 | +0.0600 | -0.0312 | +0.2269 |
| 10000 | -0.0896 | -0.0005 | +0.0029 | -0.0301 | -0.1123 | -0.1263 |
| 14999 | -0.1533 | -0.0006 | +0.0032 | -0.0487 | -0.2605 | -0.1508 |

Interpretation:

- M1 is not weak from the start. It has a small PSNR advantage at step 5000.
- M1 becomes worse by step 10000 and degrades further by step 14999.
- At step 14999, all three eval images have lower PSNR under M1 than under baseline.
- This supports a late-training / model-competition explanation more than an immediate structural mismatch.

## Paired Multi-Seed Results

Completed paired JapaneseGradens experiments:

| Seed | Baseline PSNR | M1 PSNR | dPSNR | Baseline SSIM | M1 SSIM | dSSIM | Baseline LPIPS | M1 LPIPS | dLPIPS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 24.9098 | 24.7565 | -0.1533 | 0.9001 | 0.8995 | -0.0006 | 0.1171 | 0.1204 | +0.0033 |
| 123 | 24.9055 | 24.5965 | -0.3089 | 0.8989 | 0.8963 | -0.0026 | 0.1195 | 0.1191 | -0.0004 |
| 3407 | 24.9386 | 24.6383 | -0.3003 | 0.8997 | 0.8954 | -0.0042 | 0.1190 | 0.1175 | -0.0015 |

Aggregate paired deltas:

```text
dPSNR mean  = -0.2542 dB
dPSNR std   =  0.0714 dB
dSSIM mean  = -0.0025
dSSIM std   =  0.0015
dLPIPS mean = +0.0005
dLPIPS std  =  0.0021
```

Interpretation:

- M1 is consistently worse than baseline in PSNR and SSIM across seeds 42, 123, and 3407.
- LPIPS is mixed, so the degradation should not be described as uniformly worse across every metric.
- The stable PSNR/SSIM degradation is enough to run the planned factor split, because M1 changes both `medium_context_mode` and `b_inf_mode`.

## Factor Split Plan

Existing references:

```text
J0 Baseline: medium_context_mode=dir_only,      b_inf_mode=implicit
J3 M1:       medium_context_mode=dir_xy_camera, b_inf_mode=tied
```

New split experiments:

| ID | Medium context | `b_inf_mode` | Script | Status |
|---|---|---|---|---|
| J1 | `dir_xy_camera` | `implicit` | `scripts/experiments/japanesegradens_j1_context_implicit_seed42_15000.sh` | 20-step smoke passed |
| J2 | `dir_only` | `tied` | `scripts/experiments/japanesegradens_j2_dironly_tied_seed42_15000.sh` | 20-step smoke passed |

Smoke settings:

```text
STAMP=20260730_jgradens_factor_smoke
MAX_NUM_ITERATIONS=20
MODEL_NUM_STEPS=20
RUN_EVAL=0
RUN_CLOSURE_DIAG=0
RUN_FAR_DIAG=0
RUN_REGION_DIAG=0
```

The smoke outputs, renders, and logs were removed after validation. Full 15k factor split should use:

```text
STAMP=20260730_jgradens_factor_split
```
