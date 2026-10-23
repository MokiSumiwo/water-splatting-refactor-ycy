# TABA on the BND-ON Mainline: Chromatic-only Training and Four-scene Results

Completion update: all 10 registered training/evaluation runs finished at
step 14999 on 2026-09-10. Four-scene E2 versus E0 improves mean PSNR by
0.029097 dB and SSIM by 0.000315, but worsens mean LPIPS and MSE. Retain TABA
as a default-off research option; do not promote it to the mainline default.
The early diagnostic sections below are preserved historical records.
The current implementation and final results start at
"Chromatic-only training continuation" and "Completed four-scene results".

Date: 2026-09-10  
Protocol: M1 + OCMC + CICA, Auxiliary OFF, BND ON, RAOC/BRASH/BVAM/BARR/BHTC/BGTR OFF, seed 42.

## Audit and plan

The active bounded path is `compute_bounded_gaussian_colors` in
`water_splatting/fields/gaussian_appearance.py`. It evaluates the active SH
coefficients as logits and applies one sigmoid. `features_dc` supplies the SH0
base and `features_rest` supplies the view-dependent SH terms. The natural
decomposition is `delta = s_full - s_base`, with luminance
`mean(delta, channel)` and chromatic `delta - luminance`.

The renderer already exposes `rgb_object`, `rgb_medium`, `rgb_clear`,
`tau_D`, `transmission`, Gaussian visibility and projected depth. Transmission
is the detached renderer proxy `exp(-clamp(medium_attn * depth, min=0))`.
Support is the count of training cameras for which `(radii > 0)` is true.

TABA was implemented only as the standalone analysis script
`scripts/diagnostics/run_taba_bnd_diagnostic.py`. It does not call backward,
does not alter loss or optimizer state, and does not add a model parameter.
No E1/E2 training implementation was started.

## Data limitation

The retained BND-ON checkpoints for these scenes contain only step 14999;
the historical 5000 and 10000 BND-ON checkpoints were removed. The diagnostic
therefore records only step 14999 and writes the missing-step record rather
than fabricating intermediate states. BND-OFF uses the matched retained A2
checkpoints. All statistics use training views; test views are not used.

## Preliminary diagnostic result (not a valid D1 decision)

Machine-readable artifacts are under `outputs/taba_diagnostic_20260910/`.
They are retained as a debugging trace only. The first pass exposed a
protocol mismatch: the available BND-ON and BND-OFF retained checkpoints do
not form a complete matched 5000/10000/14999 CICA protocol, and the first
analysis pass did not reconstruct the exact per-pixel/per-Gaussian J statistic
for every branch. Therefore the table below must not be used as a formal D1
accept/reject result.

| Scene / variant | Bucket | mean T | object residual | medium residual | chromatic residual | corr(T, chroma) | corr(T, object) | g_luma | g_chroma | P(J>1) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| JapaneseGradens BND ON | low | 0.222 | 0.258 | 0.0057 | 0.067 | +0.463 | -0.110 | 0.942 | 0.861 | 0 |
| JapaneseGradens BND ON | middle | 0.517 | 0.202 | 0.0079 | 0.067 | +0.408 | -0.476 | 0.946 | 0.875 | 0 |
| JapaneseGradens BND ON | high | 0.897 | 0.070 | 0.0172 | 0.068 | -0.261 | -0.618 | 0.946 | 0.875 | 0 |
| Panama BND ON | low | 0.302 | 0.202 | 0.0052 | 0.092 | -0.002 | -0.397 | 0.916 | 0.782 | 0 |
| Panama BND ON | middle | 0.523 | 0.098 | 0.0051 | 0.094 | -0.684 | -0.739 | 0.922 | 0.796 | 0 |
| Panama BND ON | high | 0.782 | 0.026 | 0.0058 | 0.094 | -0.289 | -0.459 | 0.922 | 0.796 | 0 |

BND-OFF confirms a different decomposition regime: JapaneseGradens has
medium residuals about 0.059/0.107 in low/middle buckets, while Panama has
0.033/0.090. The BND-ON medium residual is much smaller, but this does not
produce a common transmission-conditioned chromatic pattern.

## Decision

## Strict paired 2026-09-10 run

To remove the source-checkpoint ambiguity, paired C0 sources were regenerated
for JapaneseGradens-RedSea and Panama with seed 42 and the registered camera
sequence.  The BND-ON source used the bounded parameterization and the BND-OFF
source used legacy SH; both used OCMC and RAOC OFF.  CICA continuation from the
common step-3000 source was started on GPUs 6--9 with Auxiliary OFF.  The
strict continuation produced matched step-5000 checkpoints for both scenes;
10000 and 14999 were not completed in this run and are explicitly absent from
the postprocess report.

The read-only paired postprocess is
`scripts/diagnostics/run_taba_paired_postprocess.py`.  It uses the same train
view IDs and deterministic 2048-pixel sample in both branches, buckets by the
mean ON/OFF transmission (`low < .33`, `middle < .66`, `high >= .66`), and
reports scene means and across-view standard deviations.  `S_med` is the
detached Frobenius norm of the analytic medium Jacobian for nine unit raw
medium directions.  Artifacts are under
`outputs/taba_paired_diagnostic_strict_20260910/`.

At step 5000, JapaneseGradens chromatic ON-minus-OFF residual increases from
0.0169 (low) to 0.0813 (middle) and 0.0960 (high); Panama gives 0.0504,
0.0606, and 0.0951.  JapaneseGradens low-bucket direction agreement is high
(94% of views), but Panama is near chance (53%).  Object and medium residuals
also vary strongly by bucket, and the medium sensitivity is largest in the
low bucket (0.306/0.324 versus 0.173/0.113 in high).  This is evidence that
the observed difference is not a stable low-transmission chromatic failure
shared by both scenes.

The conservative decision file is
`outputs/taba_paired_diagnostic_strict_20260910/D1_decision.json`; it reports
`D1_formally_passes=false` because the requested 14999 checkpoint is missing
and no eligible cross-scene gate was found.  No E1/E2 training was run, no
default configuration was changed, and no gate threshold was tuned.

The transmission-aware allocation hypothesis is **not yet validated**. The
preliminary traces suggest cross-scene instability, but the required matched
protocol and exact J audit are still missing. Gate means remain high, so this
trace does not indicate gate collapse. BND remains strictly bounded in the
native renderer; the helper tests also confirm gate bounds and detachment.

Per the stopping rule, this round remains diagnostic-only. No gate threshold
tuning, E1/E2 training, or four-scene validation is performed. The result is:

> TABA's transmission-conditioned appearance allocation assumption remains
> unestablished. Do not implement the training version from this trace.

## Chromatic-only training continuation, 2026-09-10

The preceding diagnostic record is retained as historical evidence. The new
user-authorized experiment explicitly implements and trains chromatic-only
TABA; the earlier diagnostic-only stopping decision does not govern this run.

### Implementation and mathematical definition

The default-off model fields are `taba_enabled`, `taba_diagnostic_only`,
`taba_gate_mode="detached_transmission"`, `taba_transmission_mid`,
`taba_transmission_temperature`, `taba_chroma_floor`, and `taba_log_path`.
The legacy support-based diagnostic helper API remains available to its old
callers. Training calls `compute_taba_allocation` without a support argument.
No support weighting or luminance allocation is applied in this experiment.

For each Gaussian and view:

```text
s_base = SH0(features_dc)
s_full = SH3(features_dc, features_rest, viewdir)
delta = s_full - s_base
delta_luma = mean(delta, RGB)
delta_chroma = delta - delta_luma
tau_D = clamp(medium_attn * depth, min=0)
T = mean(exp(-tau_D), RGB)
g = floor + (1-floor) * sigmoid((T-mid)/temperature)
s_taba = s_full + (detach(g)-1) * delta_chroma
rgb_taba = safe_sigmoid(s_taba)
```

The implemented expression is algebraically equal to
`s_base + delta_luma + detach(g)*delta_chroma` and preserves the original
logits exactly when the gate is one. Safe sigmoid clips only floating-point
endpoints to `[finfo.tiny, 1-finfo.eps]`. The existing BND path is untouched
when TABA is disabled. E0 can therefore retain sigmoid values rounded to
exactly one; E1/E2 remain strictly inside the open interval.

### Transmission proxy

A no-gradient prepass calls the unchanged underwater rasterizer using the
current projected Gaussians and OCMC-conditioned medium. It yields the
existing alpha-normalized expected `depth`. Depth is independent of the
Gaussian RGB, so there is no appearance/depth fixed-point iteration. The
RGB-mean transmission is sampled at each rounded projected Gaussian center,
using the same nearest-pixel convention as the existing CICA worker.
One scalar gate preserves the zero-mean chromatic subspace. Per-channel gates
would change luminance and are rejected.

Invisible, out-of-frame, empty, nonpositive-depth or nonfinite samples get
`g=1`. Projection coordinates, depth, medium attenuation, transmission, gate
and TABA diagnostic outputs are detached. The rasterizer formulas, OCMC
projector, CICA target computation/calibration, and optimizer groups are
unchanged. CICA continues to start at 10000 and refresh at 10000/12000/14000.

### Registered training configuration

| Variant | TABA | Mid | Temperature | Chroma floor |
|---|---|---:|---:|---:|
| E0 | OFF | 0.50 | 0.15 | 0.20 |
| E1, TABA-A | ON | 0.50 | 0.15 | 0.20 |
| E2, TABA-B | ON | 0.40 | 0.20 | 0.30 |

All runs use M1 + OCMC + CICA, Auxiliary OFF, BND ON, RAOC/BRASH/BVAM/
BARR/BHTC/BGTR OFF, seed 42, and the full continuation from 3000 to 14999.
Checkpoints at 5000, 10000, and 14999 and all evaluation-view underwater,
native clear, raw-J display and GT images are retained.

Source hashes, BND ON/OFF parameterizations, source seed, continuation camera
sequence, and train/test lists were verified against
`outputs/taba_paired_baseline_20260910_strict_v4`. Both scenes retain valid
step-3000 sources under `outputs/taba_paired_source_20260910_v2/bnd_on`.
JapaneseGradens uses 17 training and 3 test images; Panama uses 15 training
and 3 test images. No source regeneration or uncertain checkpoint mixing was
needed. Full machine-readable provenance is in `source_asset_audit.json`.

### Engineering validation

- 24 tests passed, including retained BRASH/BVAM/diagnostic tests and two
  focused local-metric tests.
- Both scenes load the old source state strictly, with identical model hashes.
- Disabled forward hashes equal the historical strict-v4 BND hashes exactly.
- Disabled and diagnostic-only forward and main loss are exactly equal.
- E1/E2 appearance gradients reach DC/rest only; gate/proxy/statistics detach.
- Existing CICA gradient routing audits pass for both scenes and both presets.
- OCMC projector detachment passes; CICA escape gradient norms are zero.
- TABA intrinsic RGB is strictly in (0,1); raw J and intrinsic exceedance are zero.
- Luminance logit roundoff is checked against a dtype- and scale-aware bound.
- The existing RAOC fused CUDA audit has pre-existing strict-tolerance failures.
  Running the exact same audit with the committed pre-TABA model yields
  identical forward and backward error values: no TABA regression. RAOC is OFF.

Engineering artifacts: `outputs/taba_chromatic_bnd_on_20260910/engineering_v2`
and `outputs/taba_chromatic_bnd_on_20260910/engineering/cuda_nonregression.json`.
The first engineering run's overly tight absolute luminance tolerance and its
failure record are retained in `engineering`; the corrected audit uses an
explicit roundoff bound rather than altering TABA mathematics.

### Metrics and visualization protocol

The active experiment root is `outputs/taba_chromatic_bnd_on_20260910`.
The collector `scripts/experiments/summarize_taba_bnd_on.py` writes
`report.md`, `summary.json`, `metrics_all_steps.csv`, `metrics_final.csv`,
`diagnostics_all_steps.csv`, and matched contact sheets under that root.
It uses project-local `metrics.py` SSIM/PSNR/VGG-LPIPS on native-resolution
RGB tensors with exact GT pairing; MSE uses the same tensor pair. No resizing,
cropping, or additional color transform is applied to metric inputs.

`P(J>1)` and `J p99` use **unclamped** `J_gaussian_raw`; the previously used
`J_gaussian` is already clamped and cannot establish the bound. Scene-level
quantiles are means of per-view quantiles. The existing native clear mapping
is `J/(1+J)`; raw-J contact sheets are also retained for direct inspection.
Object residual means `mean|rgb_object-GT|`; medium residual means
`mean|rgb_medium-(GT-rgb_object)|`. Neither has a separate observed component GT.

Final per-scene metrics, four-scene means, visual observations, failure cases,
the gap to BND OFF, and retention recommendation appear below. No true
clear-color recovery claim is made without clear ground truth.

### Engineering incidents retained

The first E0/E1 JapaneseGradens evaluation reached the local PSNR function
with a noncontiguous NCHW tensor. Its internal `.view()` rejected that layout.
The metric adapter now makes NCHW tensors contiguous without changing values,
shape, range or pairing. Two focused metric tests passed. Both finished runs
were re-evaluated from all three saved checkpoints with `--evaluate-only`;
the original error logs are retained alongside evaluation recovery logs.

Stopping the initial queue coordinator to reassign a job from occupied GPU 8
caused the execution environment to clean up its remaining descendant workers.
This unexpectedly interrupted JapaneseGradens E2 and Panama E1. Their complete
partial directories and worker logs are preserved under
`interrupted_scheduler_cleanup/`. Both experiments were restarted from the
same original strict paired C0@3000 with seed 42 and the identical registered
camera sequence. They were not resumed from partial checkpoints because those
checkpoints retain RNG hashes rather than complete RNG state. Final metrics
only use the complete restarted runs. The interruption was an execution error,
not a metric-based stopping decision.

### Completed two-scene selection

The six registered runs finished all 11999 continuation updates. Frozen
two-scene results are retained in `two_scene_summary.json` and
`two_scene_report.md` under the experiment root.

| Scene | Variant | PSNR | SSIM | VGG-LPIPS | MSE |
|---|---|---:|---:|---:|---:|
| JapaneseGradens-RedSea | E0 | 24.849467 | 0.894747 | 0.134311 | 0.005105518 |
| JapaneseGradens-RedSea | E1 | 24.773291 | 0.894861 | 0.134023 | 0.005279881 |
| JapaneseGradens-RedSea | E2 | 24.809669 | 0.895357 | 0.133939 | 0.005287252 |
| Panama | E0 | 31.521616 | 0.947028 | 0.103610 | 0.000709911 |
| Panama | E1 | 31.520214 | 0.947219 | 0.103217 | 0.000709988 |
| Panama | E2 | 31.495697 | 0.947405 | 0.103673 | 0.000713553 |
| Two-scene mean | E0 | 28.185541 | 0.920888 | 0.118960 | 0.002907715 |
| Two-scene mean | E1 | 28.146753 | 0.921040 | 0.118620 | 0.002994935 |
| Two-scene mean | E2 | 28.152683 | 0.921381 | 0.118806 | 0.003000403 |

Both TABA presets pass the engineering checks. E2 was selected by higher
two-scene mean final PSNR among E1/E2, after both finished, and recorded in
`four_scene_selection.json` before launching the extension. This is an
exploratory choice on the two evaluation scenes, not an independently
validated optimum. Neither preset improves the two-scene mean PSNR over E0.

The additional Curasao and IUI3-RedSea runs compare E0 against this fixed E2.
The strict-v4 directory contains sources for the original two scenes only.
The extension uses the registered BND-from-scratch C0@3000 sources from
`outputs/m1_raoc_causal_four_scene_20260827`, with their original SHA256
enforced and shared by both arms. The registered fallback config is
`outputs/m1_ocmc_causal_iui3_20260825/source_bnd_config.yml`.
`four_scene_source_audit.json` records hashes, seed 42, camera sequences and
the disjoint train/test splits (Curasao 18/3; IUI3-RedSea 25/4). Each pair
loads identical model, optimizer, scheduler, OCMC projector and continuation
RNG states; no BND-OFF counterpart is claimed for the extension sources.

### Completed four-scene results

All 10 registered runs completed training and evaluation at steps 5000,
10000 and 14999. The four extension processes exited normally with code 0.
Results below use final step 14999, regardless of earlier checkpoint scores.
Scene means are unweighted means over test views, and the four-scene mean
weights each scene equally. LPIPS is the local VGG-LPIPS metric, not the
separate training-log LPIPS implementation.

| Scene | Variant | PSNR | SSIM | VGG-LPIPS | MSE |
|---|---|---:|---:|---:|---:|
| Curasao | E0 | 32.435543 | 0.959069 | 0.125384 | 0.000733794 |
| Curasao | E2 | 32.533065 | 0.959169 | 0.125693 | 0.000706067 |
| IUI3-RedSea | E0 | 30.669319 | 0.908650 | 0.199086 | 0.001632153 |
| IUI3-RedSea | E2 | 30.753901 | 0.908823 | 0.199939 | 0.001582167 |
| Four-scene mean | E0 | 29.868986 | 0.927374 | 0.140597 | 0.002045344 |
| Four-scene mean | E2 | 29.898083 | 0.927688 | 0.140811 | 0.002072260 |
| E2 minus E0 | | +0.029097 | +0.000315 | +0.000213 | +0.000026916 |

E2 PSNR changes by -0.039798 dB on JapaneseGradens, -0.025919 dB on Panama,
+0.097521 dB on Curasao, and +0.084582 dB on IUI3-RedSea. SSIM increases
slightly in all four scenes. LPIPS improves only on JapaneseGradens and
worsens on the other three. The lower MSE on the extension scenes does not
offset the higher JapaneseGradens MSE in the arithmetic four-scene mean.
Mean PSNR can improve while mean MSE worsens because PSNR averages logarithms
of per-view MSE; it is not the PSNR of the final arithmetic mean MSE.

### Final bounded and allocation diagnostics

Every evaluated checkpoint, logged training diagnostic and saved snapshot
diagnostic has `P(J>1)=0` and `P(intrinsic_rgb>1)=0`. The checks use raw J.
All E1/E2 Gaussian intrinsic colors remain strictly inside (0,1); their
observed maximum is 0.9999998807907104. E0 retains the original sigmoid
floating-point endpoint of 1.0. Clear image pixels may be zero where no
object contributes; this is separate from strictly bounded Gaussian colors.

| Scene | Variant | P(J>1) | P(intrinsic>1) | J p99 | tau_D p90 |
|---|---|---:|---:|---:|---:|
| JapaneseGradens-RedSea | E0 | 0 | 0 | 0.846933 | 0.882759 |
| JapaneseGradens-RedSea | E1 | 0 | 0 | 0.847456 | 0.842865 |
| JapaneseGradens-RedSea | E2 | 0 | 0 | 0.846412 | 0.898767 |
| Panama | E0 | 0 | 0 | 0.821295 | 0.845909 |
| Panama | E1 | 0 | 0 | 0.805124 | 0.847060 |
| Panama | E2 | 0 | 0 | 0.806621 | 0.846031 |
| Curasao | E0 | 0 | 0 | 0.742011 | 1.788443 |
| Curasao | E2 | 0 | 0 | 0.742845 | 1.807369 |
| IUI3-RedSea | E0 | 0 | 0 | 0.869929 | 2.000304 |
| IUI3-RedSea | E2 | 0 | 0 | 0.868993 | 2.075945 |

E2 reduces the applied chromatic SH residual magnitude in every scene:
JapaneseGradens 0.095868 to 0.088737, Panama 0.126027 to 0.097613,
Curasao 0.132550 to 0.090458, IUI3-RedSea 0.123364 to 0.097853.
This confirms that allocation acts on the intended residual, but it does
not establish that water color has been separated from object appearance.
Luminance is preserved algebraically within each gated forward pass; the
final trained models can have different luminance residuals because their
optimization trajectories differ.

The full `report.md` contains final mean transmission; low/middle/high
fractions; gate mean/p10/p50/p90; invalid-gate fraction; chromatic and
luminance residual magnitudes; object and medium residuals; and clear and
intrinsic RGB extrema. `diagnostics_all_steps.csv` includes these quantities
at all three checkpoints. Transmission fractions are over image pixels,
whereas gate/residual statistics are over visible Gaussians. Invalid gate
fractions therefore should not be compared directly with pixel fractions.

### Visual observations and failure cases

The final contact sheets were inspected for all scenes, using underwater,
native clear and raw-J displays. Contact-sheet thumbnails are resized for
presentation only; all retained PNGs and metric inputs have native size.

- JapaneseGradens: fine foreground texture is largely retained. Differences
  among E0/E1/E2 are subtle; distant cyan and incomplete backgrounds persist.
  E2 loses 0.217651 dB on MTN_1090 while gaining 0.144445 dB on MTN_1106,
  so the effect is not consistent across views.
- Panama: no convincing general reduction of blue/cyan is visible. Fine
  texture remains, but the soft cyan background and upper-image holes in
  MTN_1529 persist. E2 loses 0.067043 dB there and 0.043642 dB on MTN_1547.
- Curasao: underwater appearances are very similar, and dense object texture
  and the color chart remain recognizable. Large black clear-background
  regions and a cyan fringe persist in both arms. The strongest PSNR gain is
  MTN_1288 (+0.234127 dB), but every Curasao test view has slightly worse LPIPS.
- IUI3-RedSea: underwater structure remains similar. Clear/raw-J images still
  show deep blue background patches and incomplete object coverage. E2 gains
  PSNR on three of four views, but LPIPS worsens on all four; MTN_5928 has the
  largest LPIPS increase (+0.002266).

No obvious new global greying, broad color banding, overexposure or systematic
edge softening is visible at contact-sheet scale. This is a qualitative
inspection, not a pixel-level sharpness claim. Native clear appears darker
than raw-J because of the existing `J/(1+J)` mapping. Bounded output does not
resolve the persistent clear-background artifacts or prove accurate clear
color recovery, and there is no true clear ground truth in this evaluation.

### Gap to the BND-OFF reference and recommendation

The supplied BND-OFF reference is PSNR 30.031764, SSIM 0.927403,
LPIPS 0.137838, MSE 0.002015746. Relative to it, E2 is -0.133681 dB PSNR,
+0.000285 SSIM, +0.002973 LPIPS, and +0.000056514 MSE. This is contextual
comparison only: the supplied reference is not a newly run paired BND-OFF
control, and its LPIPS network/protocol may differ from local VGG-LPIPS.
It does not support a causal improvement claim against BND OFF.

Retain the implemented, tested TABA feature as default OFF for research.
Do not promote E2 to the mainline default based on these results. The gain
in average PSNR is small and comes with worse average LPIPS and MSE, mixed
scene results, and no consistent qualitative clear improvement. Only seed 42
was tested, and E2 was selected on the two original evaluation scenes; no
statistical significance or universal capacity recovery is established.
The detached rasterizer prepass adds runtime cost. Logged median iteration
times on the original two scenes were approximately 9-10% higher for E2
than E0, but these are sparse training samples, not a controlled benchmark.

### Final artifact and engineering verification

`scripts/diagnostics/audit_taba_artifacts.py` completed successfully in the
existing `water_splatting` environment, with only GPUs 6,7,8,9 exposed.
It verified 30 checkpoint SHA256 values against the training manifests,
384 readable native-resolution PNGs (underwater, clear, raw-J, GT), and
18 current contact sheets. All 13 distinct test-view GT images match the
original dataset pixels exactly across arms and checkpoints. Additional
historical sheets and interrupted-run artifacts remain preserved.

All 10 gradient audits pass, all four scene pairing audits pass, all logged
CICA escape norms are zero, and every OCMC projector remains unchanged.
The 24-test suite passed before the report-only extension. The final report
collector and artifact auditor pass `py_compile`; final `git diff --check`
also passes. CUDA nonregression passes by matching committed-reference
errors exactly; the pre-existing strict RAOC tolerance failures remain
documented and must not be described as a strict CUDA-equivalence pass.

Current branch: `research/bvam-bnd-on`. The worktree remains modified and
untracked, including earlier BRASH/BVAM/TABA diagnostic assets. Existing
changes and results were preserved. No commit, push, package installation,
checkpoint deletion, or compact-output operation was performed.

Files changed for this chromatic-only training work:

- `water_splatting/fields/gaussian_appearance.py`: detached allocation and
  bounded-color helper; historical diagnostic helper signature retained.
- `water_splatting/taba.py`: detached transmission sampling and diagnostics.
- `water_splatting/water_splatting.py`: default-off configuration and bounded
  Gaussian color substitution with the detached rasterizer depth prepass.
- `scripts/experiments/run_direct_mdrr_cica_scene.py`: TABA configuration,
  audits, diagnostics, native metrics and checkpoint/render retention.
- `scripts/experiments/run_taba_bnd_on.py`: E0/E1/E2 presets and paired runner.
- `scripts/experiments/summarize_taba_bnd_on.py`: all-step metrics, final
  diagnostic tables, pairing checks and matched contact sheets.
- `scripts/diagnostics/audit_taba_training.py`: old-state/disabled-path and
  gradient-boundary audits.
- `scripts/diagnostics/audit_taba_cuda_nonregression.py`: committed-reference
  comparison for the existing CUDA audit.
- `scripts/diagnostics/audit_taba_artifacts.py`: hashes, image pairing and
  full registered-result completeness checks.
- `tests/test_taba.py` and `tests/test_taba_metrics.py`: focused tensor and
  local-metric verification.
- This research note: preserved history, implementation, results and limits.

### Reproduction and output paths

All paths below are relative to the confirmed repository root
`/mnt/new/home_old/ycy/water-splatting-refactor`. Let `<out>` denote
`outputs/taba_chromatic_bnd_on_20260910`.

| Artifact | Path |
|---|---|
| Complete readable results | `<out>/report.md` |
| Machine-readable results | `<out>/summary.json`, `metrics_final.csv`, `metrics_all_steps.csv`, `diagnostics_all_steps.csv` |
| Frozen two-scene selection | `<out>/two_scene_report.md`, `two_scene_summary.json`, `four_scene_selection.json` |
| Sources | `<out>/source_asset_audit.json`, `four_scene_source_audit.json` |
| Full checkpoints | `<out>/<scene>/<E0,E1,E2>/checkpoints/step-{000005000,000010000,000014999}.ckpt` |
| Native PNGs | `<out>/renders/<variant>/<scene>/step_{05000,10000,14999}/<view>/A2/{underwater,clear,clear_raw_display_clamp01,gt}.png` |
| Three-arm sheets | `<out>/contact_sheets/{JapaneseGradens-RedSea,Panama}_E0_vs_E1_vs_E2_{underwater,clear,clear_raw_display_clamp01}.jpg` |
| Four-scene paired sheets | `<out>/contact_sheets/<scene>_E0_vs_E2_{underwater,clear,clear_raw_display_clamp01}.jpg` |
| Evaluation details | `<out>/<scene>/<variant>/{evaluation_metrics.csv,per_view_metrics.csv}` |
| Training logs | `<out>/logs/<scene>_<variant>.log` and `<out>/<scene>/<variant>/taba_training.jsonl` |
| Snapshot diagnostics | `<out>/<scene>/<variant>/taba_statistics_step_{05000,10000,14999}.csv` |
| Gradient audit | `<out>/<scene>/<variant>/gradient_routing_audit.json` |
| Final integrity audit | `<out>/artifact_integrity_audit.json` |
| Engineering audits | `<out>/engineering_v2/<scene>/audit.json`, `<out>/engineering/{test_results,cuda_nonregression}.json` |

To collect and verify the retained results without new training:

```bash
export PATH=/opt/anaconda3/bin:$PATH
export CUDA_VISIBLE_DEVICES=6,7,8,9
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate water_splatting
printf 'CONDA_DEFAULT_ENV=%s CUDA_VISIBLE_DEVICES=%s\n' "$CONDA_DEFAULT_ENV" "$CUDA_VISIBLE_DEVICES"
python scripts/experiments/summarize_taba_bnd_on.py
python scripts/diagnostics/audit_taba_artifacts.py
```

New repetitions must use a fresh output directory and one allowed physical
GPU per job. The runner rejects occupied GPUs and nonempty run directories;
retained results must not be overwritten.
