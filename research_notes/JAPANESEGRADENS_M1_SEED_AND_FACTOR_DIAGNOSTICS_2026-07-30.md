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

## Paired Multi-Seed Plan

Run paired JapaneseGradens experiments:

| Seed | Baseline | M1 | Purpose |
|---:|---|---|---|
| 42 | complete | complete | existing reference |
| 123 | pending | pending | paired seed check |
| 3407 | pending | pending | paired seed check |

For speed and to keep the scope aligned with reconstruction stability, seed-123 and seed-3407 runs initially collect `ns-eval` reconstruction metrics only. Far/region diagnostics can be added after the paired deltas are known.

Judgment gate:

```text
If all three seeds show negative M1-vs-baseline PSNR and mean drop >= 0.08 dB
with mean LPIPS degradation >= 0.001, treat JapaneseGradens M1 degradation as stable.
Only then run the J1/J2 factor split:
  J1: dir_xy_camera + implicit
  J2: dir_only + tied
```

