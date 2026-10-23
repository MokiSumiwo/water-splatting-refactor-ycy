# BRASH BND-ON Capacity Recovery

Date: 2026-09-09  
Branch: `research/brash-bnd-on`  
Environment: `water_splatting` (`/mnt/new/anaconda3_ycy/envs/WS_ycy/bin/python`)

## Scope

This round implemented and screened only the first BND-ON mechanism requested in
the research plan: **BRASH (Bounded Residual Appearance with Split Heads)**.
HCC, BARR, BGTR, BVAM, RAOC, and any third innovation were not implemented or
merged. The four earlier task branches remain separate from this branch.

The fixed protocol was:

```text
M1 + OCMC + CICA
Auxiliary OFF
BND ON
RAOC OFF
seed=42
final step=14999
local metrics.py evaluation
```

The screening scenes were `JapaneseGradens-RedSea` and `Panama`.

## Mechanism

The ordinary BND path computes a full SH3 logit and applies one sigmoid:

```text
s_full(g, v) = s_DC(g) + s_rest(g, v)
c_BND(g, v) = sigmoid(s_full(g, v))
```

BRASH keeps every color bounded, but separates the DC base and the view-
dependent residual:

```text
c_base(g) = sigmoid(s_DC(g))
c_view(g, v) = sigmoid(s_DC(g) + alpha * s_rest(g, v))
g(g) = sigmoid(gate_logit(g))
c_BRASH(g, v) = (1 - g(g)) * c_base(g) + g(g) * c_view(g, v)
```

Here `alpha` is `brash_residual_scale`, initialized to `1.0`; the gate is
initialized close to one (`gate_init_logit=5.0`). Both branches are strictly
inside `[0, 1]`, and their convex mixture is also strictly bounded. BRASH is
therefore a bounded parameterization change, not a return to the legacy
unbounded `SH + 0.5` path and not a render-time clamp.

When `brash_enabled=False`, the model takes the existing BND path exactly.
When BRASH is enabled on an old BND checkpoint, the missing gate parameter is
created with the configured safe initialization; all old model tensors and
optimizer/scheduler states are otherwise restored.

## Reliability Prior

The optional E3 prior uses only renderer-derived statistics from training
cameras: Gaussian support, depth variance, transmission, and view-direction
spread. The statistics are detached before constructing the confidence prior.
The prior is a weak gate regularizer and does not provide color pseudo-ground
truth. It cannot backpropagate into geometry, opacity, medium, SH residuals,
or the OCMC projector.

## Gradient Boundaries

The formal audit recorded:

```text
BRASH direct gradient -> brash_gate only
CICA direct gradient -> features_dc only
BRASH escape to appearance/geometry/medium = 0
CICA escape = 0
P(J > 1) = 0
```

The disabled-equivalence audit on a real CUDA run also produced exact hashes
for the prediction image, object/medium/tail RGB, Gaussian RGB/logits, DC
RGB/logits, and SH residual. Thus adding the BRASH code with the switch off
does not change the existing BND computation.

## Validation

Unit tests: `3 passed`.

Short smoke and checkpoint reload passed for E0, E1, E2, and E3. Formal
training completed for all eight scene/variant jobs at step 14999. The formal
jobs used compact output: their checkpoints were evaluated and removed. The
retained artifacts are the metric CSVs, training summaries, reliability and
gate statistics, safety files, manifests, and gradient-routing audits.

The machine-readable summary can be regenerated with:

```bash
CONDA_DEFAULT_ENV=water_splatting \
/mnt/new/anaconda3_ycy/envs/WS_ycy/bin/python \
scripts/experiments/summarize_brash_bnd_on.py \
  --output outputs/brash_bnd_on_screening_20260909/BRASH_SUMMARY.md
```

## Metrics

All numbers below are from the repository-local RGB protocol and are evaluated
at the same final step.

### JapaneseGradens-RedSea

| Variant | PSNR | SSIM | LPIPS | MSE |
|---|---:|---:|---:|---:|
| E0 BND fixed | 24.635557 | 0.892471 | 0.117406 | 0.005483938 |
| E1 fixed gate | 24.660160 | 0.892726 | 0.117604 | 0.005500870 |
| E2 learned gate | 24.638990 | 0.892557 | 0.118103 | 0.005475420 |
| E3 reliability prior | 24.603230 | 0.892847 | 0.117385 | 0.005614960 |

Relative to E0, E1 gives `+0.02460 dB` PSNR and `+0.000255` SSIM, but
slightly worsens LPIPS and MSE. E3 gives the best LPIPS/SSIM direction here,
but loses `0.03233 dB` PSNR and worsens MSE.

### Panama

| Variant | PSNR | SSIM | LPIPS | MSE |
|---|---:|---:|---:|---:|
| E0 BND fixed | 31.570630 | 0.948071 | 0.077268 | 0.000703210 |
| E1 fixed gate | 31.669580 | 0.948488 | 0.076991 | 0.000686800 |
| E2 learned gate | 31.581040 | 0.948046 | 0.077363 | 0.000701350 |
| E3 reliability prior | 31.632330 | 0.948510 | 0.076975 | 0.000692550 |

E1 improves all four reported Panama metrics. E3 also improves Panama in
PSNR, SSIM, LPIPS, and MSE relative to E0.

### Two-scene mean

| Variant | PSNR | SSIM | LPIPS | MSE |
|---|---:|---:|---:|---:|
| E0 BND fixed | 28.10310 | 0.920271 | 0.097337 | 0.00309357 |
| E1 fixed gate | 28.16487 | 0.920607 | 0.097298 | 0.00309384 |
| E2 learned gate | 28.11002 | 0.920301 | 0.097733 | 0.00308838 |
| E3 reliability prior | 28.11778 | 0.920679 | 0.097180 | 0.00315376 |

No candidate is a clean all-metric non-regression over E0. E1 has the highest
PSNR and SSIM mean; E3 has the highest SSIM and lowest LPIPS mean; E2 has the
lowest MSE mean. Each gives up at least one other metric.

## Diagnostic Interpretation

The main failure signal is the reliability prior, not numerical instability.
At the final refresh, E3 had high-confidence fraction `0` in both scenes;
confidence p90 was only about `0.042` for JapaneseGradens and `0.008` for
Panama. The support distribution is concentrated around only three to four
training views, and the measured view-direction spread is small. Consequently
the prior is conservative for nearly all Gaussians. It cannot provide a broad,
reliable opening of the residual head, and it does not recover BND capacity
uniformly. This explains why E3 helps Panama while harming JapaneseGradens
PSNR/MSE.

E2 is effectively neutral in mean PSNR/SSIM and worsens LPIPS. E1 shows that
the split representation itself is numerically viable and can help Panama,
but its small JapaneseGradens MSE/LPIPS regression means it is not yet a
paper-level replacement under the hard all-metric rule.

No clear/dewatered image comparison is claimed for this screening because the
formal jobs used `--compact-output`; no rendered images remain under the
screening root and no checkpoint remains there. The bounded safety audit is
the available visual-risk guard for this run, with `P(J > 1)=0` in every
formal job.

## Decision

BRASH is technically valid, bounded, compatible with old BND checkpoints, and
correctly isolated in the gradient graph. It does **not** pass the current
hard scientific screening rule as a default replacement: no variant improves
all four mean metrics over E0, and the most promising prior variant damages
JapaneseGradens PSNR/MSE while its confidence prior is nearly inactive.

Do not implement HCC, BARR, BGTR, or BVAM automatically from this result.
The next research step should be a narrow diagnostic or a new user-approved
single-factor experiment focused on why JapaneseGradens has weak reliable
cross-view support. Do not interpret this result as evidence for a broad
joint schedule search.
