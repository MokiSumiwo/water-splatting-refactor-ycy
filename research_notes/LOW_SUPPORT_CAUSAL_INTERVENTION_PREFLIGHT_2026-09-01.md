# Low-Support Causal Intervention Preflight

Date: 2026-09-01
Experiment: `LOW-SUPPORT-CAUSAL-INTERVENTION-PREFLIGHT`
Classification: `LOW_SUPPORT_NOT_SUPPORTED`

## Frozen Protocol

The audit loads only the four registered C0 OCMC checkpoints at step 14999. It performs forward rendering only. Per-Gaussian opacity is scaled in a detached local tensor after OCMC medium prediction; model opacity, topology, renderer source, OCMC, and RAOC state are unchanged. Support counts distinct training cameras with `model.radii > 0`. Held-out GT is used only for evaluation and localization.

Primary criterion: T1 (`s <= 1`) at `alpha=0`; support requires positive held-out mean PSNR change and a larger change than both size-matched random non-low and high-support controls in at least 3/4 scenes. T2 is sensitivity-only.

## Primary Results

| Scene | low dPSNR | random dPSNR | high dPSNR | low dLPIPS | criterion |
|---|---:|---:|---:|---:|:---:|
| Curasao | -0.006661 | -0.290271 | -0.358768 | -0.000249 | no |
| IUI3-RedSea | -0.973782 | -0.717086 | -1.246602 | -0.028205 | no |
| JapaneseGradens-RedSea | -0.660028 | -0.328510 | -0.593578 | -0.008470 | no |
| Panama | -0.004952 | -0.853462 | -1.158386 | -0.000463 | no |

## Localization

Low-support indicator contribution (`> 1e-12`) is compared with each held-out view's top-20% baseline residual pixels. This diagnostic never feeds GT into group selection or intervention.

| Scene | mean IoU | mean precision | mean recall |
|---|---:|---:|---:|
| Curasao | 0.006826 | 0.171714 | 0.007345 |
| IUI3-RedSea | 0.085687 | 0.414111 | 0.125435 |
| JapaneseGradens-RedSea | 0.151951 | 0.206855 | 0.210670 |
| Panama | 0.015054 | 0.396907 | 0.015379 |

## Strength And Train-View Trade-Off

| Scene | zero eval dPSNR | zero eval dLPIPS | half eval dPSNR | zero train dPSNR |
|---|---:|---:|---:|---:|
| Curasao | -0.006661 | -0.000249 | 0.000802 | -0.626909 |
| IUI3-RedSea | -0.973782 | -0.028205 | -0.040419 | -0.626870 |
| JapaneseGradens-RedSea | -0.660028 | -0.008470 | -0.045507 | -0.936028 |
| Panama | -0.004952 | -0.000463 | 0.008870 | -1.542951 |

## Decision

T1 zero-opacity low-support suppression does not improve held-out PSNR in any scene.

Low-support suppression improves novel-view PSNR in 0/4 scenes and satisfies the full matched-control criterion in 0/4 scenes.

At the primary zero strength, low support beats random in 2/4 scenes and high support in 3/4, but it never improves over FULL. PSNR and LPIPS both worsen in all four scenes. All train splits also degrade, with dPSNR from -1.542951 to -0.626870.

Half suppression gives tiny heldout PSNR gains only in Curasao, Panama; this sensitivity result is not the preregistered primary comparison and does not authorize module design.

The direction is not causally supported across scenes. The negative alpha=0 result is stable in 4/4 scenes, while localization overlap is heterogeneous and therefore remains correlational evidence only. Detailed PSNR, SSIM, LPIPS, and MSE results are recorded in `per_scene_metrics.csv`, `group_comparison.csv`, and `strength_ablation.csv`.

The preregistered gate does not pass. Close the low-support module direction; do not begin reliability weighting, support-aware refinement, or uncertainty-module design from this evidence.

## Integrity

All alpha=1 counterfactuals reproduce FULL within `2e-06` absolute pixel tolerance. Model state and the OCMC projector remain hash-identical before and after every scene audit. No backward call, optimizer step, training run, topology edit, renderer edit, or checkpoint write occurred.
