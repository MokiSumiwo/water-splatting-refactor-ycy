# SBR strict four-scene summary (2026-09-15)

The corrected run uses the exact scene-specific BND ON+CICA C0 sources,
identical camera sequences, seed 42, CICA/OCMC ON, Auxiliary OFF, and the
same topology callbacks. Only the selective bounded appearance map changes.

| Scene | Baseline PSNR | SBR PSNR | ΔPSNR | ΔSSIM | ΔLPIPS | ΔMSE |
|---|---:|---:|---:|---:|---:|---:|
| Curasao | 32.493907 | 32.460892 | −0.033014 | −0.000140 | −0.000043 | +0.0000078 |
| IUI3-RedSea | 30.732962 | 30.662860 | −0.070102 | +0.000315 | +0.000076 | +0.0000564 |
| JapaneseGradens-RedSea | 24.796497 | 24.801765 | +0.005269 | −0.000114 | −0.001494 | −0.0000221 |
| Panama | 31.501145 | 31.535857 | +0.034712 | −0.000148 | −0.000978 | −0.0000068 |
| **Macro** | **29.881128** | **29.865344** | **−0.015784** | **−0.000022** | **−0.000610** | **+0.0000088** |

The strict per-view render comparison is available at
`outputs/sbr_bnd_on_four_scene_strict_20260915/visualizations/sbr_strict_vs_bnd_on_cica_underwater.jpg`.
Differences are localized to textured edges and high-frequency regions; there
is no consistent black-hole or exposure improvement. Clear outputs are
diagnostic only because no paired clear ground truth exists.

## Decision

SBR does not provide a four-scene PSNR gain and is not combined with FSB. The
small LPIPS reduction is insufficient to offset PSNR/MSE regressions and does
not establish better clear reconstruction. The next candidate is PGS, first as
a zero-step Jacobian/VJP audit on the frozen BND ON+CICA checkpoints. VAB and
CVC remain deferred until angular-support and front-surface stability signals
are measured without test-view selection.
