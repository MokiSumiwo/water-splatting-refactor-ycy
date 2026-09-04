# Direct MDRR/CICA Four-Scene Experiment

## Current Formal Baseline
A0 is the historical OCMC-on, RAOC-off continuation with `A_DETACHED_SH_OPACITY_TANGENT_ORTHOGONALIZATION`. Its prior causal audit favored a generic regularization effect, so it remains auxiliary appearance regularization and is not claimed as an identifiability innovation.

## Configurations And Matching
A1 is A0 plus MDRR, A2 is A0 plus CICA, and A3 is A0 plus both. Each arm starts from the same scene-specific OCMC C0 checkpoint at step 3000, restores identical model/optimizer/scheduler/scaler state and seed-42 continuation RNG, consumes the same 11,999-camera sequence, and ends at step 14999. OCMC remains locked and RAOC remains disabled.

Start-state equivalence: `True`. Camera sequence exact match: `True`. Partner mapping match: `True`. CICA bank match: `True`.

## MDRR Implementation
MDRR activates at step 5000. Every active update uses its fixed training-camera partner and current model state. Exact classic responsibilities form cross-view residual differences and the full medium response: direct attenuation, finite medium, and tail medium. Detached positive cosine responsibility forms `g_p`; appearance receives `(1-g_p)`, medium receives `g_p`, and geometry/opacity retain the base gradient.

## CICA Implementation
CICA activates at step 10000 and refreshes at 10000, 12000, and 14000 from at most six deterministic training cameras. A read-only CUDA accumulator follows classic alpha threshold and early termination to compute the DC-logit Jacobian normal equation. Gaussians with at least three views receive an information-weighted median detached log-chroma target. Huber loss acts only on `features_dc`; scale is calibrated once to 10% of first-activation photometric DC gradient norm.
A2 and A3 use the same deterministic camera bank rule and refresh schedule. The resolved scale is fixed after calibration; there is no sweep or heldout-driven adjustment. Direction audits permit both signs and report no color-prior-collapse warning. CICA and auxiliary gradient escape are zero in the formal logs.

## Heldout RGB Results

### PSNR

| Scene | A0 | A1 | A2 | A3 | A1-A0 | A2-A0 | A3-A0 | A3-A1 | A3-A2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Curasao | 32.415435 | 28.763816 | 32.550667 | 26.492842 | -3.651619 | +0.135233 | -5.922593 | -2.270974 | -6.057826 |
| IUI3-RedSea | 30.748682 | 29.603074 | 30.750628 | 29.537513 | -1.145608 | +0.001946 | -1.211169 | -0.065561 | -1.213115 |
| JapaneseGradens-RedSea | 24.627346 | 6.556266 | 24.657293 | 19.962578 | -18.071080 | +0.029947 | -4.664769 | +13.406312 | -4.694716 |
| Panama | 31.613832 | 26.817352 | 31.612864 | 30.001301 | -4.796481 | -0.000969 | -1.612532 | +3.183949 | -1.611563 |
| MEAN | 29.851324 | 22.935127 | 29.892863 | 26.498558 | -6.916197 | +0.041539 | -3.352766 | +3.563432 | -3.394305 |

### SSIM

| Scene | A0 | A1 | A2 | A3 | A1-A0 | A2-A0 | A3-A0 | A3-A1 | A3-A2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Curasao | 0.958855 | 0.951875 | 0.959151 | 0.944375 | -0.006980 | +0.000295 | -0.014480 | -0.007500 | -0.014776 |
| IUI3-RedSea | 0.908729 | 0.908802 | 0.908279 | 0.908729 | +0.000073 | -0.000451 | -0.000000 | -0.000073 | +0.000450 |
| JapaneseGradens-RedSea | 0.893082 | 0.516321 | 0.893232 | 0.851464 | -0.376761 | +0.000150 | -0.041618 | +0.335143 | -0.041768 |
| Panama | 0.948721 | 0.938865 | 0.948842 | 0.944441 | -0.009856 | +0.000121 | -0.004279 | +0.005576 | -0.004401 |
| MEAN | 0.927347 | 0.828966 | 0.927376 | 0.912252 | -0.098381 | +0.000029 | -0.015094 | +0.083287 | -0.015124 |

### LPIPS

| Scene | A0 | A1 | A2 | A3 | A1-A0 | A2-A0 | A3-A0 | A3-A1 | A3-A2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Curasao | 0.107533 | 0.127573 | 0.107064 | 0.143242 | +0.020039 | -0.000469 | +0.035709 | +0.015669 | +0.036178 |
| IUI3-RedSea | 0.179997 | 0.181706 | 0.180143 | 0.180940 | +0.001709 | +0.000146 | +0.000943 | -0.000765 | +0.000797 |
| JapaneseGradens-RedSea | 0.117853 | 0.673744 | 0.116860 | 0.204599 | +0.555890 | -0.000993 | +0.086746 | -0.469145 | +0.087739 |
| Panama | 0.077655 | 0.103035 | 0.075977 | 0.088562 | +0.025380 | -0.001678 | +0.010907 | -0.014473 | +0.012585 |
| MEAN | 0.120760 | 0.271514 | 0.120011 | 0.154336 | +0.150755 | -0.000749 | +0.033576 | -0.117179 | +0.034325 |

### MSE

| Scene | A0 | A1 | A2 | A3 | A1-A0 | A2-A0 | A3-A0 | A3-A1 | A3-A2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Curasao | 0.000745 | 0.001596 | 0.000712 | 0.003097 | +0.000851 | -0.000033 | +0.002352 | +0.001501 | +0.002385 |
| IUI3-RedSea | 0.001595 | 0.001852 | 0.001589 | 0.001790 | +0.000257 | -0.000007 | +0.000195 | -0.000062 | +0.000201 |
| JapaneseGradens-RedSea | 0.005569 | 0.258098 | 0.005522 | 0.015204 | +0.252529 | -0.000047 | +0.009635 | -0.242894 | +0.009682 |
| Panama | 0.000696 | 0.004020 | 0.000696 | 0.001057 | +0.003324 | +0.000000 | +0.000361 | -0.002963 | +0.000361 |
| MEAN | 0.002151 | 0.066391 | 0.002130 | 0.005287 | +0.064240 | -0.000021 | +0.003136 | -0.061105 | +0.003157 |

## Per-View Results
Per-view final values and all five paired deltas are in `per_view_metrics.csv`. A1 loses PSNR on every heldout view except one Panama view (+0.008 dB). A2 is close to A0 on every heldout view, with deltas ranging from -0.105 to +0.253 dB. A3 is below A0 on every heldout view except one Panama view that is effectively tied (-0.004 dB).

## Clear Rendering And Underwater Safety
No paired real clear ground truth was found. Clear conclusions are qualitative and distributional only; less blue is not automatically better. Panels are native outputs with no white balance, contrast, saturation, histogram matching, gamma change, or manual dehazing.

Native clear comparison: `/mnt/new/home_old/ycy/water-splatting-refactor/outputs/direct_mdrr_cica_four_scene_20260903/mdrr_cica_rgb_comparison.png` and `/mnt/new/home_old/ycy/water-splatting-refactor/outputs/direct_mdrr_cica_four_scene_20260903/mdrr_cica_rgb_comparison.pdf`. Underwater safety comparison: `/mnt/new/home_old/ycy/water-splatting-refactor/outputs/direct_mdrr_cica_four_scene_20260903/mdrr_cica_underwater_reconstruction.png`. Raw clear summaries are in `clear_color_statistics.csv`.
Visual audit: A2 stays visually close to A0 without a new color distortion. Blue-minus-red decreases on the selected native clear view in all four scenes, but the visible reduction of distant blue/cyan residual is subtle rather than decisive. A1 does not reduce water-like appearance contamination safely. It introduces conspicuous color or reconstruction failures in Curasao, JapaneseGradens-RedSea, and Panama. A3 retains substantial MDRR-related degradation and is visibly worse than A2; it does not add a usable combined benefit.
Underwater reconstruction safety: A2 preserves underwater reconstruction. A1 and A3 damage it, most conspicuously in JapaneseGradens-RedSea.

## Decomposition And Topology
Final heldout decomposition safety pass: `True`; every final configuration has `P(J > 1) = 0`. No module-specific topology rule was introduced. A2 population stays within 1% of A0 in every scene: `True`. MDRR changes the learned topology substantially despite an unchanged topology schedule, increasing the final population by about 56%, 65%, and 56% in Curasao, IUI3-RedSea, and Panama and reducing it by 24% in JapaneseGradens-RedSea. This accompanies RGB degradation rather than a gain.

| Scene | Arm | Final Gaussians | Splits | Duplicates | Prunes | Resets |
|---|---:|---:|---:|---:|---:|---:|
| Curasao | A0 | 1008554 | 6311753 | 2550124 | 14656896 | 14 |
| Curasao | A1 | 1573274 | 9700571 | 5488177 | 23807865 | 14 |
| Curasao | A2 | 1007813 | 6309952 | 2545762 | 14649673 | 14 |
| Curasao | A3 | 1527990 | 9515167 | 5418909 | 23413073 | 14 |
| IUI3-RedSea | A0 | 749347 | 3718614 | 2100344 | 9068926 | 14 |
| IUI3-RedSea | A1 | 1234381 | 6121485 | 5176268 | 16465558 | 14 |
| IUI3-RedSea | A2 | 749516 | 3714890 | 2092496 | 9053461 | 14 |
| IUI3-RedSea | A3 | 1240913 | 6097919 | 5159889 | 16395515 | 14 |
| JapaneseGradens-RedSea | A0 | 808177 | 4127054 | 3264000 | 11077310 | 14 |
| JapaneseGradens-RedSea | A1 | 617477 | 4653690 | 4224497 | 13281779 | 14 |
| JapaneseGradens-RedSea | A2 | 806159 | 4122456 | 3261708 | 11067840 | 14 |
| JapaneseGradens-RedSea | A3 | 670952 | 4665785 | 4812373 | 13840370 | 14 |
| Panama | A0 | 1096597 | 5915574 | 3241838 | 14511332 | 14 |
| Panama | A1 | 1707554 | 10049652 | 6665945 | 25592638 | 14 |
| Panama | A2 | 1096356 | 5922155 | 3233311 | 14516208 | 14 |
| Panama | A3 | 1707423 | 10073648 | 6645295 | 25620111 | 14 |

## Interaction And Recommendation
MDRR: `DROP`. It improves 0/4 scene means and changes mean PSNR by -6.916197 dB. CICA: `KEEP`. It improves 3/4 scene means, all four are non-worsening at the preregistered -0.05 dB tolerance, and mean PSNR changes by +0.041539 dB. Combined: `DROP` with mean PSNR -3.352766 dB versus A0.
Interaction: `INTERFERING`. A3 is +3.563432 dB versus A1 but -3.394305 dB versus the independently useful A2. It therefore loses CICA's benefit instead of preserving complementary advantages.
The recommended second innovation is CICA, with the explicit limitation that its native clear improvement is mild and not visually decisive without paired clear ground truth. The recommended method is `OCMC + auxiliary appearance regularization + CICA`; in the preregistered naming this is `BASE+CICA` (A2). MDRR and A3 are not retained. Auxiliary SH regularization remains in the baseline as auxiliary appearance regularization, not as an identifiability innovation.

## Required Final Answers
1. Same starting state: yes (`True`).
2. Identical primary camera sequence: yes (`True`).
3. Matrix completion: yes, all 12 formal runs reached step 14999 with 11,999 matched updates.
4. MDRR PSNR deltas: Curasao -3.651619 dB, IUI3-RedSea -1.145608 dB, JapaneseGradens-RedSea -18.071080 dB, Panama -4.796481 dB.
5. CICA PSNR deltas: Curasao +0.135233 dB, IUI3-RedSea +0.001946 dB, JapaneseGradens-RedSea +0.029947 dB, Panama -0.000969 dB.
6. Combined PSNR deltas: Curasao -5.922593 dB, IUI3-RedSea -1.211169 dB, JapaneseGradens-RedSea -4.664769 dB, Panama -1.612532 dB.
7. Combined versus MDRR mean PSNR: +3.563432 dB.
8. Combined versus CICA mean PSNR: -3.394305 dB.
9. MDRR improved 0/4 scene means.
10. CICA improved 3/4 scene means.
11. Combined improved 0/4 scene means.
12. Best mean PSNR: A2 (29.892863 dB for A2).
13. Best mean SSIM: A2 (0.927376 for A2).
14. Best mean LPIPS: A2 (0.120011 for A2).
15. CICA distant blue/cyan reduction: mild distributional shift, not a visually decisive reduction.
16. MDRR water-like appearance contamination: no; it causes conspicuous degradation.
17. A3 further improvement: no; it is 3.394305 dB below A2 in mean PSNR.
18. Real paired clear ground truth: no.
19. Quantitative clear result: not applicable because no paired real clear GT exists.
20. Clear limitation: all clear/dewatered conclusions are qualitative and distributional only.
21. Underwater reconstruction: preserved by A2; damaged by A1 and A3.
22. Decomposition safety: all pass (`True`).
23. Gaussian population: A2 tracks A0; MDRR arms change population strongly and adversely.
24. MDRR classification: DROP.
25. CICA classification: KEEP.
26. MDRR+CICA interaction: INTERFERING.
27. Recommended second innovation: CICA.
28. Recommended paper configuration: OCMC + CICA, with the baseline auxiliary appearance regularizer retained.
29. Auxiliary SH regularization: KEEP_AS_BASELINE_ONLY.
30. Current best version: BASE+CICA (A2).

## Limitations
This is one fixed four-scene, one-seed protocol with no sweep or rescue. The small CICA RGB gain does not by itself prove intrinsic purification, clear rendering has no paired reference, and qualitative preference cannot establish physical correctness. The auxiliary appearance regularizer remains baseline-only and cannot be presented as the second innovation.
