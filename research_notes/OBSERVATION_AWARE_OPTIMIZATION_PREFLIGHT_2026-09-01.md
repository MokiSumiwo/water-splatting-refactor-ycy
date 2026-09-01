# Observation-Aware Optimization Preflight

Date: 2026-09-01
Experiment: `OBSERVATION-AWARE-OPTIMIZATION-PREFLIGHT`
Classification: `OBSERVATION_UNDERCONSTRAINED_NOT_SUPPORTED`

## Frozen Protocol

The audit uses only the registered step-14999 C0 checkpoints with OCMC on and RAOC off. It samples heldout-visible Gaussians from T1, T2, middle, and high-support populations without using GT. Each sampled Gaussian receives isolated physical-opacity `+0.01/+0.05` without parameter-level clipping, relative `features_dc` `+1%/+5%`, and relative physical-scale `+1%/+5%` perturbations in detached render inputs. Scale perturbations are reprojected. The unchanged renderer retains its existing per-pixel alpha cap. No parameter, topology, optimizer, renderer, OCMC state, or checkpoint is changed.

Primary sensitivity is the L2 norm of scene-wise rank fractions for the three 1% image-RMS finite-difference sensitivities. Rank normalization prevents opacity, color, and scale coordinate units from dominating the composite. A scene passes only when T1 median sensitivity exceeds high-support median sensitivity, all single-factor rank controls for depth/opacity/scale/footprint retain a positive underconstraint association, and sensitivity correlates positively with projected heldout residual.

## Primary Results

| Scene | T1/high sensitivity | low > high | all controls positive | sensitivity-residual rho | scene pass |
|---|---:|:---:|:---:|---:|:---:|
| Curasao | 0.887073 | no | no | 0.108871 | no |
| IUI3-RedSea | 1.003141 | yes | no | 0.110818 | no |
| JapaneseGradens-RedSea | 1.015769 | yes | no | 0.139712 | no |
| Panama | 0.679390 | no | no | 0.014867 | no |

## Control Analysis

| Scene | depth | opacity | scale | footprint |
|---|---:|---:|---:|---:|
| Curasao | -0.176953 | -0.126506 | -0.135608 | -0.133425 |
| IUI3-RedSea | -0.022056 | 0.094479 | 0.114311 | 0.077764 |
| JapaneseGradens-RedSea | -0.107759 | 0.042797 | 0.054069 | 0.069668 |
| Panama | -0.230467 | -0.134251 | -0.156226 | -0.135221 |

## Numerical Quality

| Scene | finite | nonzero composite | opacity 5%/1% | color 5%/1% | scale 5%/1% | pass |
|---|:---:|---:|---:|---:|---:|:---:|
| Curasao | yes | 0.945312 | 1.000010 | 0.991583 | 1.003276 | yes |
| IUI3-RedSea | yes | 0.992188 | 1.000023 | 0.992401 | 1.011967 | yes |
| JapaneseGradens-RedSea | yes | 0.984375 | 1.000011 | 0.992559 | 1.010203 | yes |
| Panama | yes | 0.984375 | 1.000038 | 0.989994 | 1.006105 | yes |

## Scientific Interpretation

The numerical quality gate passes and the preregistered scientific criterion passes in 0/4 scenes. The frozen sensitivity evidence is not stable enough to establish observation-underconstrained Gaussian optimization as the second failure mechanism.

Observation-aware optimization module design is not authorized. Treat low support as a difficult-region indicator and search for another failure mechanism.

## Integrity

All detached no-op renders reproduce FULL within `2e-06` absolute tolerance. Every worker reports hash-identical model and OCMC projector state, exact OCMC forward-state equality, zero backward calls, zero optimizer steps, and zero checkpoint writes.
