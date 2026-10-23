# Strict BND-OFF Teacher Attempt (2026-09-14)

## Protocol

Four legacy-SH3 + OCMC + RAOC-OFF runs were trained from the registered
scene configurations with seed 42, 15,000 updates, the same train/eval camera
lists, and outputs saved under
`outputs/bnd_off_teacher_strict_20260914/`. Existing assets were not changed.

## Evaluation

| Scene | Train PSNR | Eval PSNR | Eval SSIM | Eval LPIPS |
| --- | ---: | ---: | ---: | ---: |
| JapaneseGradens-RedSea | 32.5283 | 24.7335 | 0.8956 | 0.1150 |
| Panama | 41.7148 | 32.2747 | 0.9486 | 0.0730 |
| Curasao | 37.4117 | 32.0084 | 0.9560 | 0.1083 |
| IUI3-RedSea | 36.9854 | 30.8336 | 0.9123 | 0.1761 |

The runs completed successfully. However, comparison of the common step-3000
checkpoints shows that Gaussian populations already differ from the BND-ON
sources. The independent legacy and bounded runs therefore have different
geometry/topology and cannot provide a valid pixel-aligned teacher residual.
Their final PSNR is useful as a separately trained BND-OFF reference only.

## Decision

UGTR residual distillation is **not started** from these checkpoints. A valid
next teacher must share the BND-ON Gaussian population. The safe construction is
to clone a BND-ON checkpoint, convert its bounded SH coefficients to an
RGB-equivalent legacy DC initialization, freeze Gaussian topology, and train
only a bounded underwater residual or color adapter. Any adapter must be
validated on held-out training views before using official test views.
