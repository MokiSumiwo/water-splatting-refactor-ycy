# WaterSplatting Improvement Metric Summary

Date: 2026-09-05

## Scope

This summary uses the original WaterSplatting legacy SH3 implementation as the reference. The legacy and bounded runs were newly evaluated from scratch with seed 42, 15,000 updates, the same four-scene camera sequence, `dir_xy_camera`, tied `B_inf`, OCMC off, and RAOC off. This provides the missing same-sequence legacy MSE values.

The historical BND validation note contains an older legacy endpoint with PSNR/SSIM/LPIPS but no retained same-run MSE. It is kept as archival evidence only. It is not mixed with the new MSE values below; all legacy-versus-bounded rows in this summary use the new same-sequence reruns for all four metrics.

PSNR and SSIM improve upward. LPIPS and MSE improve downward. A positive signed delta is therefore not automatically an improvement for every metric.

## Direct Matched Effects

Each row is treatment minus its own matched causal baseline. These are the valid module-level effects and must not be added across rows.

| Method | Scene | dPSNR | dSSIM | dLPIPS | dMSE |
|---|---|---:|---:|---:|---:|
| Auxiliary Appearance Regularization | Curasao | -0.045276 | -0.000012 | +0.000199 | +0.00001704 |
| Auxiliary Appearance Regularization | IUI3-RedSea | +0.090387 | +0.000146 | -0.000192 | -0.00006255 |
| Auxiliary Appearance Regularization | JapaneseGradens-RedSea | +0.027695 | +0.000018 | +0.000473 | +0.00002576 |
| Auxiliary Appearance Regularization | MEAN | +0.036275 | +0.000206 | +0.000203 | -0.00000798 |
| Auxiliary Appearance Regularization | Panama | +0.072294 | +0.000673 | +0.000332 | -0.00001218 |
| Bounded SH3 | Curasao | +0.404679 | +0.002562 | +0.001498 | -0.00025786 |
| Bounded SH3 | IUI3-RedSea | -0.325708 | -0.003294 | +0.001309 | +0.00011566 |
| Bounded SH3 | JapaneseGradens-RedSea | +0.091778 | +0.002463 | -0.001217 | -0.00027206 |
| Bounded SH3 | MEAN | -0.160320 | +0.000246 | +0.001314 | -0.00007379 |
| Bounded SH3 | Panama | -0.812031 | -0.000748 | +0.003667 | +0.00011912 |
| CICA | Curasao | +0.135233 | +0.000295 | -0.000469 | -0.00003302 |
| CICA | IUI3-RedSea | +0.001946 | -0.000451 | +0.000146 | -0.00000653 |
| CICA | JapaneseGradens-RedSea | +0.029947 | +0.000150 | -0.000993 | -0.00004683 |
| CICA | MEAN | +0.041539 | +0.000029 | -0.000749 | -0.00002148 |
| CICA | Panama | -0.000969 | +0.000121 | -0.001678 | +0.00000047 |
| OCMC | Curasao | +0.155507 | +0.000439 | -0.002179 | -0.00001060 |
| OCMC | IUI3-RedSea | +0.085546 | +0.000096 | +0.001465 | -0.00001770 |
| OCMC | JapaneseGradens-RedSea | +0.006337 | -0.002479 | -0.000719 | +0.00005747 |
| OCMC | MEAN | +0.084211 | -0.000355 | -0.000160 | +0.00000352 |
| OCMC | Panama | +0.089454 | +0.000526 | +0.000794 | -0.00001511 |

## Complete Configurations Versus Legacy

These rows compare complete trained configurations to the same-sequence legacy run. They are descriptive endpoint differences, not additive causal effects, because the configurations are nested and some use different continuation baselines.

| Configuration | Scene | dPSNR | dSSIM | dLPIPS | dMSE |
|---|---|---:|---:|---:|---:|
| Bounded SH3 | Curasao | +0.404679 | +0.002562 | +0.001498 | -0.00025786 |
| Bounded SH3 | IUI3-RedSea | -0.325708 | -0.003294 | +0.001309 | +0.00011566 |
| Bounded SH3 | JapaneseGradens-RedSea | +0.091778 | +0.002463 | -0.001217 | -0.00027206 |
| Bounded SH3 | MEAN | -0.160320 | +0.000246 | +0.001314 | -0.00007379 |
| Bounded SH3 | Panama | -0.812031 | -0.000748 | +0.003667 | +0.00011912 |
| OCMC | Curasao | +0.560186 | +0.003001 | -0.000681 | -0.00026846 |
| OCMC | IUI3-RedSea | -0.240161 | -0.003199 | +0.002774 | +0.00009796 |
| OCMC | JapaneseGradens-RedSea | +0.098116 | -0.000017 | -0.001937 | -0.00021459 |
| OCMC | MEAN | -0.076109 | -0.000109 | +0.001154 | -0.00007027 |
| OCMC | Panama | -0.722577 | -0.000223 | +0.004461 | +0.00010401 |
| OCMC + Auxiliary | Curasao | +0.526328 | +0.002875 | -0.000769 | -0.00026168 |
| OCMC + Auxiliary | IUI3-RedSea | -0.330550 | -0.003261 | +0.002920 | +0.00015598 |
| OCMC + Auxiliary | JapaneseGradens-RedSea | +0.151467 | +0.000781 | -0.001356 | -0.00020426 |
| OCMC + Auxiliary | MEAN | -0.082656 | +0.000126 | +0.001369 | -0.00005354 |
| OCMC + Auxiliary | Panama | -0.677867 | +0.000111 | +0.004681 | +0.00009580 |
| OCMC + Auxiliary + CICA | Curasao | +0.661561 | +0.003170 | -0.001238 | -0.00029470 |
| OCMC + Auxiliary + CICA | IUI3-RedSea | -0.328604 | -0.003712 | +0.003066 | +0.00014945 |
| OCMC + Auxiliary + CICA | JapaneseGradens-RedSea | +0.181414 | +0.000931 | -0.002349 | -0.00025109 |
| OCMC + Auxiliary + CICA | MEAN | -0.041116 | +0.000155 | +0.000620 | -0.00007502 |
| OCMC + Auxiliary + CICA | Panama | -0.678836 | +0.000232 | +0.003003 | +0.00009627 |

## Interpretation

- Bounded SH3 reduces decomposition tails and improves average MSE in this matched rerun, but its average PSNR decreases and its RGB effect remains scene-dependent.
- OCMC is the retained first core mechanism. Its matched RGB effect is mixed: mean PSNR increases by the listed amount, while mean SSIM and MSE do not improve under this exact comparison.
- Auxiliary appearance regularization has a small mean held-out PSNR gain, but its mechanism audit was not supported; it remains baseline-only.
- CICA is the retained second mechanism. Relative to the auxiliary baseline A0, it improves mean PSNR, LPIPS, and MSE, with a very small SSIM increase. It must not be reported as the sum of all previous deltas.
- The endpoint rows versus legacy are not additive causal effects. In particular, OCMC, auxiliary regularization, and CICA use their own matched continuation baselines; the direct-effect table is the correct table for module attribution.
- RAOC, MDRR, and MDRR+CICA are excluded from the retained-positive table because the project decisions classify them as not retained or interfering.

## Data Provenance

- Legacy: `outputs/legacy_baseline_four_scene_20260905/<scene>/final_summary.json`
- Bounded SH3 / OCMC-off: `outputs/ocmc_bounded_baseline_four_scene_20260905_v3/<scene>/final_summary.json`
- OCMC-on: `outputs/m1_raoc_causal_four_scene_20260827/<scene>/eval_metrics.csv`
- Auxiliary causal result: `outputs/identifiability_module_causal_iui3_20260902/final_metrics.csv`
- CICA causal result: `outputs/direct_mdrr_cica_four_scene_20260903/final_metrics.csv`

There is no paired real clear-ground-truth dataset in this project. The metrics above are underwater/held-out RGB reconstruction metrics; they do not quantify true clear/dewatered rendering quality.
