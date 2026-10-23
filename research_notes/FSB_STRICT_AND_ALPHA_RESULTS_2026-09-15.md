# Strict FSB and FSB-alpha results (2026-09-15)

The strict FSB run starts from the registered scene-specific BND-ON+CICA C0
checkpoint at step 3000, uses seed 42, OCMC+CICA ON, auxiliary appearance
OFF, and `bounded_headroom_sh3`. All four 5k smoke runs passed start-state and
checkpoint reload audits. The 15k run is in
`outputs/fsb_bnd_on_four_scene_strict_20260915_v1`.

| Scene | ΔPSNR | ΔSSIM | ΔLPIPS | ΔMSE |
|---|---:|---:|---:|---:|
| Curasao | +0.002160 | −0.000105 | −0.000137 | −0.00000122 |
| IUI3-RedSea | −0.019125 | −0.000554 | −0.000550 | −0.00000663 |
| JapaneseGradens-RedSea | +0.038695 | +0.000078 | −0.001345 | −0.00000485 |
| Panama | +0.061419 | −0.000358 | +0.000077 | −0.00001075 |
| **Macro** | **+0.020787** | **−0.000235** | **−0.000489** | **−0.00000586** |

The strict FSB result is positive on PSNR/MSE/LPIPS in the macro average but
does not improve every scene and slightly lowers SSIM. It is a valid bounded
variant baseline, not SOTA evidence.

FSB-alpha was implemented as a fixed convex blend of bounded SH and headroom
RGB (`alpha=0.25, 0.50, 0.75, 1.0`), leaving the underwater renderer and losses
unchanged. All three alpha values passed four-scene 5k smoke/reload audits.
The alpha=0.50 15k run is in
`outputs/fsb_alpha050_four_scene_strict_20260915_v1` and gives:

| Metric | BND ON+CICA | α=0.50 | Δ |
|---|---:|---:|---:|
| PSNR | 29.881128 | 29.870787 | −0.010340 |
| SSIM | 0.927658 | 0.927561 | −0.000097 |
| LPIPS | 0.120722 | 0.120630 | −0.000093 |
| MSE | 0.00205675 | 0.00207519 | +0.00001844 |

Therefore α=0.50 is rejected for promotion. The α=0.25 and α=0.75 runs are
diagnostic 5k smoke only and were not promoted to 15k because no early metric
supports replacing the strict FSB baseline. The strict visualization is
`outputs/fsb_bnd_on_four_scene_strict_20260915_v1/visualizations/fsb_strict_vs_bnd_on_cica_underwater.jpg`.

Clear images remain diagnostic because no paired clear ground truth exists.
