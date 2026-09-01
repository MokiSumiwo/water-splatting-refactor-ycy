# Gaussian View Consistency Audit

Date: 2026-09-01
Experiment: `GAUSSIAN-VIEW-CONSISTENCY-AUDIT`
Classification: `VIEW_CONSISTENCY_TENTATIVE`

## Frozen Protocol

All 20 registered C0 checkpoints use OCMC on and RAOC off. The audit performs detached forward rendering only. It does not train, call backward or optimizer.step, modify the renderer/OCMC/RAOC, write checkpoints, or write renders.

The primary GT-free proxy is `r_i(v) = alpha_i(v) * gaussian_view_rgb_i(v)` on training cameras where Gaussian `i` is visible, with `alpha_i(v) = sigmoid(opacity_i) * pi * projected_radius_i(v)^2 / image_area`. `VC variance` is the population mean squared L2 deviation from the per-Gaussian training-view mean; support of at least two views is required and support count is controlled. The mean L2 deviation, opacity-color appearance score, and medium-attenuated direct score are also reported. The primary is an OCMC-independent, unoccluded projected opacity-area proxy, not exact per-pixel transmittance attribution.

Heldout error is added only after VC construction. It is approximated by mean heldout RGB MSE inside each Gaussian's clipped projected-radius bounding box. The AUROC label is the top 20% of sampled Gaussians by the fraction of their footprint box occupied by within-camera top-20% high-error pixels.

## Metric Estimability

| Scene | final analyzed Gaussians | exact-positive VC | median VC variance | minimum positive VC |
|---|---:|---:|---:|---:|
| Curasao | 3410 | 0.999413 | 5.719518e-12 | 6.306997e-23 |
| IUI3-RedSea | 4044 | 0.998022 | 2.886745e-11 | 7.456727e-23 |
| JapaneseGradens-RedSea | 4039 | 0.995048 | 1.765556e-11 | 3.717681e-20 |
| Panama | 3385 | 0.999409 | 1.507980e-11 | 6.101874e-18 |

VC is numerically estimable and nonzero in all four scene populations; this establishes measurable multi-view variation, but not by itself a failure mechanism.

## Final Gaussian-Level Results

| Scene | rho(VC,error) | AUROC | top-20% enrichment | all controls positive | temporal stable | pass |
|---|---:|---:|---:|:---:|:---:|:---:|
| Curasao | 0.172301 | 0.579790 | 1.512330 | yes | yes | yes |
| IUI3-RedSea | 0.082010 | 0.574492 | 1.223396 | yes | no | no |
| JapaneseGradens-RedSea | 0.006759 | 0.520362 | 1.237013 | yes | yes | yes |
| Panama | -0.061911 | 0.418703 | 0.869359 | no | no | no |

## Single-Factor Controls

| Scene | support | depth | tau | transmission | opacity | footprint | scale | OCMC active | medium suppressed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Curasao | 0.192265 | 0.193928 | 0.199194 | 0.198658 | 0.184602 | 0.214996 | 0.272741 | 0.171752 | 0.188201 |
| IUI3-RedSea | 0.163270 | 0.140598 | 0.142182 | 0.140761 | 0.102733 | 0.192664 | 0.288768 | 0.136094 | 0.139752 |
| JapaneseGradens-RedSea | 0.079234 | 0.032580 | 0.079336 | 0.077698 | 0.031834 | 0.064357 | 0.170723 | 0.056033 | 0.067970 |
| Panama | 0.108625 | 0.145994 | 0.093970 | 0.089028 | -0.046324 | -0.009010 | 0.160038 | -0.071365 | -0.060229 |

## Temporal Stability

| Scene | 5k | 8k | 10k | 13k | 14999 | positive | stable |
|---|---:|---:|---:|---:|---:|---:|:---:|
| Curasao | 0.021004 | 0.115623 | 0.097700 | 0.123933 | 0.172301 | 5/5 | yes |
| IUI3-RedSea | -0.108458 | -0.046514 | -0.115831 | 0.074495 | 0.082010 | 2/5 | no |
| JapaneseGradens-RedSea | 0.099347 | 0.032347 | 0.002898 | 0.006200 | 0.006759 | 5/5 | yes |
| Panama | -0.126677 | -0.104837 | -0.103504 | -0.049270 | -0.061911 | 0/5 | no |

Checkpoint populations have no persistent Gaussian lineage IDs. Temporal results therefore test distribution-level recurrence, not identity-level persistence; array index or nearest-geometry matching was not used.

## Camera-Level Results

| Scene | cameras | rho(camera VC, camera MSE) |
|---|---:|---:|
| Curasao | 3 | 0.500000 |
| IUI3-RedSea | 4 | -0.800000 |
| JapaneseGradens-RedSea | 3 | 1.000000 |
| Panama | 3 | 0.500000 |

Pooled within-scene-rank rho over 13 heldout cameras: `0.093931`. Per-scene camera correlations are descriptive because each scene has only 3-4 heldout cameras.

## OCMC Independence

OCMC active magnitude and suppressed medium residual are included as preregistered single-factor rank controls. A scene only passes when both controlled VC-error associations, together with all other major controls, remain positive.

The loaded config retains the dormant backend string `reference`; RAOC is effectively disabled because every worker verifies `camera_medium_ray_adaptive_observability_enabled=False` and an absent `raoc_state`. No RAOC path is executed.

## Disk Management

Available bytes before cleanup: `34224603136`. Deleted only `outputs/ocmc_candidate_c_resplit_replication_20260831_attempt1_interrupted_tool_session` (`9127403775` logical bytes). Available bytes after cleanup: `43352379392`; reclaimed available bytes: `9127776256`.

## Scientific Decision

The formal classification is `VIEW_CONSISTENCY_TENTATIVE` with 2/4 scene passes.

Curasao provides the only clearly positive final effect (`rho=0.172`, `AUROC=0.580`). JapaneseGradens passes the sign-based rule but its final effect is near zero (`rho=0.0068`, `AUROC=0.520`); IUI3 is not temporally stable and Panama is consistently negative. The pooled camera-level association is also weak. These effect sizes are why the tentative label does not authorize a mechanism claim.

Gaussian view inconsistency is not established as a valid second failure mechanism. New module design is not scientifically authorized.

## Integrity

Every worker reports unchanged model and OCMC projector hashes, zero backward calls, zero optimizer steps, and zero checkpoint writes. Protected historical GMVC, Q50/Q80, renderer, OCMC, and RAOC sources remain hash-identical.
