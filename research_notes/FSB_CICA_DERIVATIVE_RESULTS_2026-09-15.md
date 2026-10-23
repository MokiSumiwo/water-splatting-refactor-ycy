# FSB/CICA Derivative Consistency: Completed Frozen Diagnostic

The four-scene diagnostic is complete. FSB's DC derivative differs from the
sigmoid rule used by CICA, and replacing only that rule measurably changes
the hypothetical CICA refresh. A second, independent renderer backward
inconsistency was found in both BND and FSB. Neither finding establishes a
training gain or explains the full BND ON/OFF metric gap.

## Scope And Results

Eight frozen step-14999 checkpoints were evaluated: strict FSB and paired
BND ON+CICA in each scene. Each uses its six recorded CICA training-bank
cameras, giving 48 model-view combinations on 24 training cameras. Test
images and metrics were not used by the diagnostic; the standard datamanager
does cache the test split during initialization. No optimizer step or
parameter update occurred, and production code was not changed.

| Scene | FSB DC Jacobian relative error | CICA gradient vector change | Gradient cosine | Target-only gradient change | Weight-only gradient change |
|---|---:|---:|---:|---:|---:|
| Curasao | 1.1135% | 12.5090% | 0.992506 | 11.5731% | 2.7459% |
| IUI3-RedSea | 1.3877% | 2.9328% | 0.999579 | 2.7604% | 0.8712% |
| JapaneseGradens-RedSea | 1.2309% | 4.7109% | 0.998925 | 4.5824% | 0.8613% |
| Panama | 1.0550% | 4.0857% | 0.999166 | 3.8596% | 1.0970% |

Jacobian error is `||J_old-J_exact||_F / ||J_exact||_F`, accumulated over
accepted Gaussian/pixel/channel contributions and cameras. It includes the
native alpha-compositing and transmission factors. Gradient change is
`||g_corrected-g_old||/||g_old||`, not a norm increase or a training benefit.
The gradient norm itself changes by approximately +1.92%, -0.47%, -0.94%,
and +0.08% in scene order. The vectors remain strongly aligned.

Only 0.0655%-0.2017% of exact information mass belongs to contributions with
more than 10% relative color-derivative error. The total error is therefore
modest at these checkpoints, rather than a pervasive large Jacobian error.
The nonlinear correction, clipping and weighted-median target construction
can nonetheless yield measurable changes in the final CICA gradient. The
target-only substitution produces most of the observed vector change;
target-only and weight-only results are not additive decompositions.

The BND color-derivative negative control matches exactly. Recomputed BND
CICA gradients differ by at most 0.0331%, reflecting floating-point atomic
accumulation and target sensitivity. All FSB effects above exceed that
observed numerical floor, but statistical training significance is untested.

## Verification And The Additional Renderer Finding

The diagnostic reuses the existing CICA kernel in its linear-derivative mode
to obtain `sum(q*residual)` and `sum(q*q)`, where `q` includes SH_C0,
visibility, and transmission. Per-Gaussian color derivatives then factor
outside these sums for each camera. The original CICA result was independently
computed as a control, using the unmodified original kernel.

- Analytic DC derivatives match actual CUDA SH/color autograd on up to 8192
  deterministic visible Gaussians per view; maximum absolute error 2.24e-8.
- Factorization reproduces the old CICA accumulator; maximum relative L2
  discrepancy 1.225e-4, below the prespecified 1e-3 tolerance.
- Native forward adjoint checks pass in all 48 cotangent probes; maximum
  relative discrepancy 1.104e-5. Signed tangent colors are rendered through
  the existing native forward with fixed geometry, opacity and medium.
- Model, configuration, unused branch optimizer state, and checkpoint file
  hashes are unchanged. Native underwater, displayed clear, accumulation,
  and depth outputs at the first bank camera remain bit-identical. Parameter
  `.grad` fields remain unset. Historical optimizer states were not loaded
  for this no-update diagnostic; checkpoint file integrity was checked.

The original native-backward check FAILED even for BND. The first failed
Curasao run is preserved; the forward-reference addendum records why an
independent forward check was added without relaxing the original tolerance.
The completed run continues to report `native_backward_checks_pass=false`.

RGB forward caps alpha at 0.999, while RGB backward reconstructs it with a
0.99 cap. The existing native binary reproduces the discrepancy in a
one-pixel, one-Gaussian scene with zero medium:

| Opacity | Forward derivative with respect to color | Native backward derivative |
|---|---:|---:|
| 0.500 | 0.500 | 0.500 |
| 0.990 | 0.990 | 0.990 |
| 0.995 | 0.995 | 0.495 |
| 0.999 | 0.999 | 0.098999 |

On actual checkpoints, mean residual-VJP discrepancies against the corrected
forward-based accumulator range from 0.7594%-1.0064% for BND and
0.7945%-1.3363% for FSB across scenes. Fixed signed probes show larger
differences, up to 10.88% on an individual view. This is an additional
backward inconsistency, not the same quantity as the FSB/CICA color-derivative
error. The cap discrepancy is independently demonstrated, but a patched
comparison is still needed to determine how much of each real-scene VJP
error it explains. No kernel was modified in this diagnostic.

## Decision

Defer new FSB variants. Prioritize renderer forward/backward consistency and
the explicit adaptation of CICA to the current appearance DC derivative.
These are compatibility/correctness changes, not new research modules.

A subsequent scoped sequence should be:

1. Isolate the renderer correction and check unchanged forward images, native
   backward versus forward JVPs, and finite-difference gradients. Cover high
   opacity, early termination, multiple Gaussians, and real training views.
2. Make CICA consume the actual DC-feature derivative, preserving the BND
   and legacy controls. Verify normal-equation terms, targets and loss routing.
3. After frozen checks pass, use separate renderer and CICA switches in a
   paired 2x2 FSB experiment. Preserve the historical implementation as a
   control. Do not combine both changes and attribute any resulting gain to
   one of them; do not tune using existing test-view results.

The refreshed targets here are hypothetical step-14999 targets, not the
saved training run's step-14000 targets. Lambda is reused only for gradient
scale reporting, never recalibrated, and no Adam prediction or update is
performed. Positive PSNR or clear-quality claims require future training and
evaluation. Frozen renders being identical is the expected diagnostic result.

## Artifacts

- [Complete report and definitions](../outputs/fsb_cica_derivative_diagnostic_20260915_v2/summary/REPORT.md)
- [Diagnostic overview](../outputs/fsb_cica_derivative_diagnostic_20260915_v2/summary/diagnostic_overview.png)
- [Derivative-ratio information distribution](../outputs/fsb_cica_derivative_diagnostic_20260915_v2/summary/derivative_ratio_information.png)
- [Per-scene metrics](../outputs/fsb_cica_derivative_diagnostic_20260915_v2/summary/scene_metrics.csv)
- [Per-view records](../outputs/fsb_cica_derivative_diagnostic_20260915_v2/summary/per_view_metrics.csv)
- [Native forward/backward probes](../outputs/fsb_cica_derivative_diagnostic_20260915_v2/summary/native_forward_backward_probes.csv)
- [Controller gradient comparisons](../outputs/fsb_cica_derivative_diagnostic_20260915_v2/summary/controller_gradients.csv)
- [Source snapshots and hashes](../outputs/fsb_cica_derivative_diagnostic_20260915_v2/provenance/manifest.json)
- [Registered protocol](FSB_CICA_DERIVATIVE_PROTOCOL_2026-09-15.md)
- [Forward-reference addendum](FSB_CICA_DERIVATIVE_FORWARD_ADDENDUM_2026-09-15.md)

All diagnostic processes finished. Only conda `water_splatting` and physical
GPUs 6-9 were used; there was no dependency installation, commit/push, or
historical asset replacement.
