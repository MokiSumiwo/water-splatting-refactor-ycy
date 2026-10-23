# SBR BND-ON 结果（2026-09-15）

Selective Bounded Relaxation (SBR) was evaluated as a single-factor
continuation from an OCMC C0 checkpoint at step 3000. The run used seed 42,
the registered camera sequence, CICA ON, M1, OCMC strength 1, Auxiliary OFF,
unchanged topology callbacks, and four scenes through step 14999. The only
model change was a smooth mixture of bounded sigmoid and the existing
headroom-SH map for Gaussian channels whose sigmoid derivative was below 0.055.

## Evaluation

| Scene | BND ON+CICA PSNR | SBR PSNR | ΔPSNR | ΔSSIM | ΔLPIPS | ΔMSE |
|---|---:|---:|---:|---:|---:|---:|
| Curasao | 32.493907 | 32.456994 | −0.036913 | −0.000268 | +0.000575 | +0.0000105 |
| IUI3-RedSea | 30.732962 | 30.671668 | −0.061294 | −0.000477 | −0.000348 | +0.0000521 |
| JapaneseGradens-RedSea | 24.796497 | 24.624141 | −0.172356 | −0.002187 | −0.000246 | +0.0003448 |
| Panama | 31.501145 | 31.648637 | +0.147492 | +0.001294 | −0.000487 | −0.0000240 |
| **Macro mean** | **29.881128** | **29.850360** | **−0.030768** | **−0.000410** | **−0.000126** | **+0.0000958** |

The complete artifacts are in
`outputs/sbr_bnd_on_four_scene_20260914_v4/`. Checkpoints, gradient audits,
start-state equivalence files and per-view evaluation files are retained.

## Protocol correction

This run used `outputs/m1_raoc_causal_four_scene_20260827` as its source root.
The BND ON+CICA reference used for the table comes from scene-specific paired
source roots. The source model hashes therefore differ for JapaneseGradens and
Panama (and the source checkpoint hashes differ for all scenes). Consequently
the table is an implementation/screening result, not a strict causal SBR
ablation. A valid causal conclusion requires rerunning SBR from exactly the
same scene-specific C0 checkpoints as the reference.

## Decision

Because source parity is not satisfied, SBR is not promoted and is not
combined with FSB, CICA, or additional gates. The observed scene variation is
descriptive only; it does not establish that SBR itself caused the changes.

The clear outputs remain diagnostic because no paired clear ground truth is
available. No exposure, geometry, opacity, densification, or renderer changes
were used to mask the result.
