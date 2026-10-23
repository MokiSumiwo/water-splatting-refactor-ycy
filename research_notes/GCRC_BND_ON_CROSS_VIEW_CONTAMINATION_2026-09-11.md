# GCRC BND ON: Gaussian Cross-View Residual Contamination Reweighting

Repository: `/mnt/new/home_old/ycy/water-splatting-refactor`.
Branch: `research/bvam-bnd-on`. Date: September 11, 2026.
Output root: `outputs/gcrc_bnd_on_20260911` (abbreviated `<out>`).
All prior GRAC/TABA and other dirty research work and assets are retained.
No install, commit, push, compact mode or asset deletion is authorized or used.
Python, tests, training and evaluation use conda `water_splatting` and only
physical GPUs 6,7,8,9, with at most one training worker per physical GPU.

## Registered mechanism

```text
delta = s_full - s_base
luma = mean_RGB(delta)
d = delta - luma
w_raw = detach(prepass.rgb_medium - prepass.rgb_object)
u_d = d / (norm(d) + 1e-8)
u_w = w_raw / (norm(w_raw) + 1e-8)
p_w_new = normalize(.95*p_w_old + .05*detach(u_w))
p_d_new = normalize(.95*p_d_old + .05*detach(u_d))
water_axis_consistency = clamp((1 + dot(u_w, normalize(p_w)))/2, 0, 1)
residual_consistency = clamp((1 + dot(u_d, p_d))/2, 0, 1)
residual_instability = 1 - residual_consistency
water_alignment = clamp(abs(dot(u_d, normalize(p_w))), 0, 1)
water_strength = clamp((norm(w_raw)-1e-4)/(.10-1e-4), 0, 1)
tau_D = clamp(detach(medium_attn*prepass.depth), min=0)
transmission = exp(-tau_D)
low_transmission = clamp(1-mean_RGB(transmission), 0, 1)
visibility_support = 1-exp(-visibility_count/8)
C = clamp(low_transmission * water_strength * water_axis_consistency
          * water_alignment * residual_instability * visibility_support, 0, 1)
alpha = .50*C
parallel = dot(d, detach(normalize(p_w))) * detach(normalize(p_w))
orthogonal = d-parallel
s_gcrc = s_full-detach(alpha)*parallel
rgb = existing_safe_sigmoid(s_gcrc) for corrected samples
```

The current gate reads prior completed-iteration history. Prototypes start
at zero with separate initialized=false masks. Only visible finite water
vectors above 1e-4 update the water prototype, with a valid projected sample;
only visible finite chromatic residuals above 1e-5 update residual history.
Both updates use normalized detached EMA. Water axis consistency is zero
without a prototype; residual consistency is one and instability zero
without a prototype. A near-zero/invalid current residual also falls back.

Visibility is exactly `radii > 0`, matching refinement. Count updates once
in `after_train`, after optimizer work and before refinement. Pending
observations are cleared in `step_cb`. Evaluation/controller/audit forwards
do not update history. Repeated after_train calls at a step do not double
count. Counts include repeated cameras and inherited observations, not
independent distinct viewpoints. Residual history also mixes view changes
with parameter evolution, so contamination is an uncalibrated pseudo-target.

Water RGB comes directly from the existing detached, ungated BND compositor
prepass. `sample_taba_transmission` maps its object/medium RGB, alpha-normalized
expected depth and RGB transmission to rounded projected Gaussian centers.
The prepass has no tau/transmission output field, so the existing helper
computes exp(-clamp(medium_attn*depth,0)) with the same model formula. No new
medium/depth estimator or rendering equation is introduced. Invalid,
empty, out-of-image or invisible center samples use unit transmission.

Before step3000, visibility_count<4, either prototype uninitialized, or any
required current sample invalid: C=alpha=0. Invalid auxiliary samples do
not poison the model. Underlying NaN appearance itself cannot be repaired
while also preserving the original BND result; the formal trainer retains
its NaN stopping condition.

All C factors, prototypes, counters and alpha are detached. Appearance
gradients flow to features_dc/rest. Ordinary photometric medium/geometry
gradients remain intact; the auxiliary prepass adds none. CICA is still
restricted to features_dc and OCMC remains detached. No new parameter,
expert, loss, renderer physics or BND safe_sigmoid change is introduced.

### Explicit mathematical and numerical boundaries

The requested raw RGB water prototype is generally not zero-mean. Scaling
its projection leaves the original luma term and raw-axis orthogonal vector
unscaled, but the resulting total RGB logit mean can change. Preserving that
mean simultaneously would require a different axis or formula. This run
uses the requested raw-axis formula and reports the change explicitly.

For alpha=0, return the original torch.sigmoid(s_full) bits exactly;
for alpha>0, use the unchanged safe_sigmoid. At extreme logits the baseline
can round to 0 or 1, so universal strict (0,1) and exact zero-alpha baseline
equivalence cannot both hold. We enforce closed [0,1] for all fallback RGB,
strict (0,1) for corrected samples, and P(J>1)=P(intrinsic>1)=0 everywhere.
The actual observed endpoint fractions and ranges are reported rather than
silently clipping baseline fallbacks. Raw J can be zero in uncovered pixels.

### Combination and modes

E3 applies GCRC first, then the unchanged TABA-DCU directional operator:

```text
s1 = s_full-alpha*P_history(d)
d1 = chroma(s1-s_base)
s2 = s1+(g_taba-1)*P_current(d1)
```

TABA's weak-axis fallback uses d1, as in its existing implementation.
Both gate statistics and both water axes derive from ungated BND data.
At alpha=0, E3 returns the exact existing TABA logits/RGB. No additional
combined floor is applied: GCRC has alpha<=.5 and TABA retains floor=.2.

| Arm | gcrc_enabled | gcrc_mode | TABA | GRAC |
|---|---|---|---|---|
| E0 | false | disabled | off | off |
| E1 | false | taba | existing directional E2 | off |
| E2 | true | gcrc | off | off |
| E3 | true | gcrc_taba | existing directional E2 | off |

The separate `grac` mode reproduces retained GRAC for historical ablations;
it is audited but not a new-method candidate. Default GCRC and GRAC remain
off. Existing GRAC configurations work when the new selector is inactive.
TABA fixed values: mid=.50, temperature=.15, floor=.20, lambda=0, axis_eps=1e-4.
All GCRC values are the requested defaults. No threshold search.

## State and checkpoints

`GaussianContaminationState` is an optional child module with persistent
buffers: water_direction_prototype and residual_direction_prototype (N,3),
their initialized masks (N,1 bool), visibility_count (N,1 int64), and scalar
last_update_step. Floating buffers follow appearance device/dtype; masks
and counters retain bool/int64. No optimizer parameters are added.
Persistent state is about 34 bytes/Gaussian in float32, plus one pending
observation. There is no accumulating frame-history list.

Old source checkpoints initialize zero history. Disabled modes load old
checkpoints normally and ignore optional GCRC history. Complete GCRC
checkpoints restore all buffers; partial/nonfinite/misaligned history is
rejected. All final checkpoints carry fixed GCRC config metadata.

Split/clone order is `[old, split_parents.repeat(n_samples), clone_parents]`,
exactly matching Gaussian parameter order. Pruning applies the same removal
mask to every history tensor; Adam state follows the existing helpers.
Tests exercise the actual refinement/cull functions with labelled lineage.

## Protocol and verification

All arms use BND ON, M1+OCMC+CICA, Auxiliary/RAOC/BRASH/BVAM/BARR/BHTC/BGTR
off, seed42. Start from the same registered BND ON step3000 source per scene
and run through14999. Preserve CICA start10000 and refresh10000/12000/14000.
Sources, model, optimizer, scheduler, projector, RNG manifest and camera
sequence are checked for exact pairing. Final small changes remain limited
by single-seed CUDA atomic nondeterminism.

Run all eight JapaneseGradens-RedSea/Panama E0-E3 jobs. Select the stable
E1/E2/E3 with highest two-scene mean final PSNR after all finish, freeze it,
then run Curasao/IUI3 E0 plus that single selected configuration. The screen
E0/selected runs form the other half of four-scene validation. This is an
exploratory selection on the two test scenes, not an independent benchmark.

82 tests passed in 7.68 seconds before launch. Real-source audits on both
screen scenes pass disabled historical hash/forward/loss/optimizer equality,
cold/diagnostic equivalence, state restore/corruption rejection, frozen
forward/backward/eval history, once-only callback updates, active correction,
appearance gradients and CICA/OCMC routing. Retained GRAC mode matches.
CUDA audit reproduces historical RAOC strict-tolerance failures exactly
against the committed reference: no regression, not a strict equivalence
pass. RAOC remains off. Evidence: `<out>/engineering/`.

## Metrics and artifacts

Project-local metrics.py evaluates native exactly paired float RGB in [0,1]
before PNG quantization: PSNR, SSIM, VGG-LPIPS and MSE. No resize, crop or
extra color conversion. Only contact-sheet thumbnails resize. Clear/dewatered
rendering has no clear-image GT. `clear.png` uses existing J/(1+J); raw J is
shown separately as `clear_raw_display_clamp01.png`.

GCRC diagnostics summarize visible Gaussians. Scene quantiles are means
of per-view quantiles, not pooled quantiles. E0/E1 GCRC fields are neutral
placeholders with history_available=0. TABA's `g_chroma` is a different
quantity from GCRC alpha. `gcrc_corrected_fraction` counts positive alpha,
while active_fraction counts eligible history even when C=0. The first
JSONL step5000 row is a source-state gradient audit; trained snapshots are
the separate step CSVs and checkpoints. Formal logs start at3500.

| Asset | Relative to `<out>` |
|---|---|
| Checkpoints | `<scene>/<arm>/checkpoints/step-{000005000,000010000,000014999}.ckpt` |
| Native renders | `renders/<arm>/<scene>/step_{05000,10000,14999}/<view>/A2/{underwater,clear,clear_raw_display_clamp01,gt}.png` |
| Metrics | `metrics_final.csv`, `metrics_all_steps.csv`, `<scene>/<arm>/{evaluation_metrics,per_view_metrics}.csv` |
| Statistics | `<scene>/<arm>/gcrc_training.jsonl`, `gcrc_statistics_step_{05000,10000,14999}.csv`, `diagnostics_all_steps.csv` |
| Gradient audits | `<scene>/<arm>/gradient_routing_audit.json` |
| Comparison images | `contact_sheets/` |
| Selection | `four_scene_selection.json`, `two_scene_summary.json`, `summary.json` |
| Complete audits | `artifact_integrity_audit.json`, `gcrc_artifact_audit.json` |
| Frozen source | `code_snapshot/manifest.json` |

## Results

### Two-scene screen, step14999

| Scene | Arm | PSNR | SSIM | LPIPS | MSE |
|---|---|---:|---:|---:|---:|
| JapaneseGradens-RedSea | E0 disabled | 24.708965 | 0.895309 | 0.134818 | 0.005427655 |
| JapaneseGradens-RedSea | E1 TABA | 24.905279 | 0.895460 | 0.134212 | 0.005105918 |
| JapaneseGradens-RedSea | E2 GCRC | 24.776872 | 0.895267 | 0.133356 | 0.005226061 |
| JapaneseGradens-RedSea | E3 GCRC+TABA | 24.825357 | 0.895266 | 0.135385 | 0.005189657 |
| Panama | E0 disabled | 31.562361 | 0.947441 | 0.103408 | 0.000702837 |
| Panama | E1 TABA | 31.533820 | 0.947416 | 0.103150 | 0.000707936 |
| Panama | E2 GCRC | 31.486682 | 0.946964 | 0.103093 | 0.000715548 |
| Panama | E3 GCRC+TABA | 31.510260 | 0.947144 | 0.103233 | 0.000711282 |
| Equal-scene mean | E0 disabled | 28.135663 | 0.921375 | 0.119113 | 0.003065246 |
| Equal-scene mean | E1 TABA | 28.219549 | 0.921438 | 0.118681 | 0.002906927 |
| Equal-scene mean | E2 GCRC | 28.131777 | 0.921116 | 0.118225 | 0.002970804 |
| Equal-scene mean | E3 GCRC+TABA | 28.167809 | 0.921205 | 0.119309 | 0.002950469 |

GCRC improves JapaneseGradens PSNR by 0.067907 dB, LPIPS by 0.001462
and MSE by 0.000201594 against its paired E0, while SSIM is slightly worse.
Panama PSNR falls by 0.075680 dB and SSIM/MSE worsen; LPIPS improves by
0.000315. Its three per-view PSNR differences are all negative:
MTN_1539 -0.048281, MTN_1529 -0.041988, MTN_1547 -0.136770 dB.
Japanese per-view differences are MTN_1090 +0.211243, MTN_1098 -0.031368,
MTN_1106 +0.023848 dB. Thus the Japanese gain is not uniform across views.

GCRC mean PSNR is 0.003886 dB below E0 and 0.087772 dB below TABA.
It has the best mean LPIPS of the four arms, but does not dominate E0 or
TABA. GCRC+TABA gains 0.036032 dB over GCRC, with better mean SSIM/MSE
but worse LPIPS; it is still 0.051741 dB below TABA. E1 is the registered
winner by mean PSNR. The extension is therefore TABA-only, not a GCRC
four-scene experiment. No scene-specific configuration was selected.

Historical GRAC screen PSNR was 28.196693, versus its own E0 28.208179;
the current GCRC screen PSNR is lower by 0.064916 dB. There is no matched
GRAC arm in this experiment, so this is historical context, not a causal
GCRC-versus-GRAC ablation. In particular Japanese E0 changed from historical
24.897843 to current 24.708965 despite source pairing and verified default
path equivalence. CUDA atomic nondeterminism and a single seed limit
interpretation of the small differences. No statistical significance or
consistent superiority over GRAC is established.

### Contamination strength

Final E2 diagnostics below average the three evaluation views. All are
over visible Gaussians, except the explicitly labelled image diagnostics.
Full mean/std/p10/p50/p90 tables for both E2 and E3 are in `report.md` and
`diagnostics_all_steps.csv`.

| Quantity | JapaneseGradens-RedSea E2 | Panama E2 |
|---|---:|---:|
| Center-sampled transmission mean | 0.857311 | 0.743087 |
| Transmission std | 0.119112 | 0.116992 |
| Transmission p10 / p50 / p90 | 0.698388 / 0.882897 / 0.979777 | 0.583819 / 0.761065 / 0.859024 |
| Low transmission mean | 0.142689 | 0.256913 |
| Water strength mean | 0.944490 | 0.966753 |
| Water axis consistency mean | 0.936863 | 0.975754 |
| Water alignment mean | 0.098159 | 0.103376 |
| Residual consistency mean | 0.899124 | 0.877070 |
| Residual instability mean | 0.100876 | 0.122931 |
| Visibility count mean | 9500.651530 | 9190.085940 |
| Visibility support mean | 0.999980 | 1.000000 |
| C mean | 0.000840395 | 0.002473266 |
| Alpha mean | 0.000420198 | 0.001236633 |
| Alpha p90 | 0.000798362 | 0.003023672 |
| Active fraction | 0.980893 | 0.973340 |
| Fallback fraction | 0.019107 | 0.026660 |
| Positive correction fraction | 0.979611 | 0.970962 |
| Parallel residual magnitude | 0.009994651 | 0.012521183 |
| Orthogonal residual magnitude | 0.095178465 | 0.124882882 |
| Chromatic residual magnitude | 0.096687724 | 0.126794644 |
| Luminance residual magnitude | 0.126492783 | 0.127997383 |
| Mean absolute logit-luma change | 0.000003403 | 0.000010645 |
| Decomposition error | 3.43e-10 | 4.66e-10 |
| rgb_object residual vs underwater GT | 0.110843976 | 0.059020770 |
| rgb_medium residual vs GT-rgb_object | 0.034776104 | 0.013783925 |
| Image raw-J p99 | 0.849262 | 0.809875 |
| Image tau_D p90 | 0.862760 | 0.839114 |
| Clear display RGB range | [0, 0.499974996] | [0, 0.499974996] |
| Intrinsic RGB range | [0.001427719, 1] | [0.004792469, 1] |
| Corrected intrinsic RGB range | [0.001864838, 0.999999881] | [0.007170534, 0.999999881] |
| Zero-alpha intrinsic endpoint fraction | 0.002662513 | 0.006021076 |
| P(J>1) / P(intrinsic>1) | 0 / 0 | 0 / 0 |

The actual GCRC correction is nonzero but weak. Alignment and residual
instability are about 0.1; multiplying six factors produces mean alpha
0.00042/0.00124 despite nearly complete history coverage. Visibility support
has saturated. Active fraction is therefore not evidence of strong correction
or accurate contamination detection. No real contamination labels exist.

### Visual checks completed for the screen

Inspected all six E0/E1/E2/E3 screen contact sheets: underwater, clear and
raw-J display for both scenes. Also inspected native final E0/E2
JapaneseGradens MTN_1090 raw-J and Panama MTN_1547 underwater files.
Interior appearance is very similar across arms. JapaneseGradens MTN_1090
retains stretched and blurred left/bottom boundaries, missing black upper
coverage and upper-right streaks; their shapes change without consistent
removal. Panama MTN_1529 retains smeared far-water geometry and holes in
clear/raw-J, while the native MTN_1547 foreground remains similar despite
its lower E2 PSNR. Neither underwater nor clear/dewatered visual improvement
is established. Clear has no observed ground truth; tone mapping makes
clear.png darker than raw-J and is not evidence of successful restoration.

### Observed runtime

| Scene | E0 seconds | E1 seconds | E2 seconds | E3 seconds |
|---|---:|---:|---:|---:|
| JapaneseGradens-RedSea | 858.67 | 975.68 | 1079.97 | 1065.69 |
| Panama | 1238.09 | 1477.32 | 1519.25 | 1583.88 |

Source: each run's `training_summary.json:training_wall_seconds` for
11999 continuation updates. E2 is +25.77%/+22.71% over E0; E3 is
+24.11%/+27.93%. These are observed run durations under concurrent jobs,
including their actual topology and callback workload, not an isolated
kernel benchmark. They cannot resolve small E2/E3 overhead differences.

### Four-scene extension

E1 was frozen before Curasao/IUI3 training. Their E0/E1 source, model,
optimizer, scheduler, projector, RNG manifest and camera sequence have
been checked for exact pairing. All 12 registered runs and final audits
completed successfully. The following is the fixed TABA configuration's
four-scene validation; GCRC remains off in E1.

| Scene | Arm | PSNR | SSIM | LPIPS | MSE |
|---|---|---:|---:|---:|---:|
| Curasao | E0 | 32.492798 | 0.959038 | 0.125373 | 0.000721747 |
| Curasao | E1 TABA | 32.465895 | 0.958955 | 0.125403 | 0.000723295 |
| IUI3-RedSea | E0 | 30.738779 | 0.908854 | 0.199830 | 0.001572812 |
| IUI3-RedSea | E1 TABA | 30.676849 | 0.908857 | 0.199582 | 0.001662874 |
| JapaneseGradens-RedSea | E0 | 24.708965 | 0.895309 | 0.134818 | 0.005427655 |
| JapaneseGradens-RedSea | E1 TABA | 24.905279 | 0.895460 | 0.134212 | 0.005105918 |
| Panama | E0 | 31.562361 | 0.947441 | 0.103408 | 0.000702837 |
| Panama | E1 TABA | 31.533820 | 0.947416 | 0.103150 | 0.000707936 |
| Equal-scene mean | E0 | 29.875726 | 0.927660 | 0.140857 | 0.002106263 |
| Equal-scene mean | E1 TABA | 29.895461 | 0.927672 | 0.140587 | 0.002050006 |

E1 versus paired E0: PSNR +0.019735 dB, SSIM +0.000011509,
LPIPS -0.000270613 and MSE -0.000056257. Curasao loses 0.026903 dB
and all four metrics worsen slightly. IUI3 loses 0.061930 dB and MSE
worsens, despite marginally better SSIM/LPIPS. IUI3 MTN_5894 loses
0.345888 dB; the other three views gain only 0.022438 to 0.044678 dB.
The Japanese screening gain does not transfer as PSNR gains on the two
extension scenes. GCRC's Curasao/IUI3 performance was not measured and
must not be inferred from these TABA results.

| Four-scene reference | PSNR | SSIM | LPIPS | MSE |
|---|---:|---:|---:|---:|
| Current paired E0 | 29.875726 | 0.927660 | 0.140857 | 0.002106263 |
| Current selected TABA E1 | 29.895461 | 0.927672 | 0.140587 | 0.002050006 |
| Historical mainline BND ON | 29.899295 | 0.927596 | 0.140541 | 0.002031043 |
| Historical TABA-DCU | 29.924975 | 0.927634 | 0.140667 | 0.002002410 |
| Historical GRAC | 29.883284 | 0.927647 | 0.140731 | 0.002083455 |
| Historical BND OFF | 30.031764 | 0.927403 | 0.137838 | 0.002015746 |

Current E1 is 0.136303 dB below historical BND OFF, with LPIPS worse by
0.002748668, MSE worse by 0.000034260 and SSIM higher by 0.000268963.
It is 0.029514 dB below historical TABA, 0.003834 below historical mainline
and 0.012177 above historical GRAC. These are cross-run reference values,
not new paired ablations. There is no four-scene GCRC-versus-BND-OFF gap
measurement. The small selected-TABA gain does not close the BND OFF gap.

Extension training times (11999 updates): Curasao E0/E1 1239.85/1413.57
seconds; IUI3 E0/E1 787.67/936.95 seconds. The same timing caveats apply.

### Final visual scope

In addition to the screen checks above, inspected the six Curasao/IUI3
E0/E1 sheets and all three four-scene overviews. Native final E0/E1 checks
also cover IUI3 MTN_5894 underwater and raw-J, and Curasao MTN_1288 raw-J.
Total direct visual inspection: 15 distinct contact sheets and 10 native
files, covering all 13 evaluation views through the sheets.

Curasao foreground color and detail remain similar, but E1 MTN_1288 has
a conspicuous blue vertical stripe at the right edge of the clear/raw-J
image; both arms retain black uncovered water regions and stretched distant
geometry. IUI3 retains left/bottom boundary smears and black missing regions
in clear/raw-J. TABA changes these shapes without consistently removing
them. There is no consistent underwater or clear/dewatered visual gain.
These observations cannot establish clear-image accuracy without clear GT.

### Final verification and recommendation

| Check | Result and evidence relative to `<out>` |
|---|---|
| Pytest | 82 passed, `engineering/test_results.json` |
| Python syntax | All 11 GCRC-related implementation/test/report files compile |
| Whitespace | `git diff --check` passes |
| Historical default path | Exact forward/loss/optimizer equality on both screen sources |
| Old checkpoint / cold / diagnostic modes | Pass, `engineering/verified/<scene>/audit.json` |
| State roundtrip, invalid state rejection | Pass on both screen sources and unit tests |
| Split/clone/prune and Adam alignment | Tests pass; all 36 saved training checkpoints pass final audit |
| CICA/OCMC and detached gates | Engineering audits and all 12 run gradient audits pass |
| CUDA | No regression against reference; historical RAOC strict forward/backward tolerance failures remain |
| Source / camera / RNG pairing | All four scenes pass |
| Full asset integrity | `artifact_integrity_audit.json`: passed |
| GCRC history / optimizer alignment | `gcrc_artifact_audit.json`: passed |
| Retained assets | 36 training checkpoints, 456 native PNGs, 21 contact sheets, 13 distinct GT views |
| Bounded outputs | All logged/evaluated P(J>1)=P(intrinsic>1)=0; corrected RGB strictly (0,1), exact fallback permits endpoints |
| Training source snapshot | All 8 source hashes match `code_snapshot/manifest.json` after training |
| Runtime environment | All 12 manifests use `water_splatting`, physical GPUs 6-9 only |
| Completion | Coordinator records `complete_runs=12, audit_passed=true`; all training workers exited |

The final verification manifest is `<out>/final_verification.json`. GPU
6/7/8/9 were idle with 1 MiB reported per GPU after the run. There were no
failure.json records, asset deletions, installs, commits or pushes. The
worktree remains dirty and includes prior experiments; its exact final
status is recorded in that manifest. Audit roundtrip checkpoints and legacy
TABA companion files are additional retained assets, outside the 36-run-
checkpoint count.

Retain GCRC implementation, all four arms and negative results as a
research ablation. Do not promote it to the mainline: it does not improve
mean PSNR over paired E0, does not beat TABA, has no demonstrated advantage
over historical GRAC, and adds about 23-26% observed training time for E2.
Japanese improves in some metrics, Panama degrades, and no consistent visual
benefit is established. Keep `gcrc_enabled=False`, `gcrc_mode="disabled"`
and GRAC off. A future experiment would need to address the weak six-factor
product, the raw-axis luminance change and the confounding of view history
with parameter evolution; no such change or retuning was applied here.

## Changed files

- `water_splatting/gcrc.py`: history, factors, correction, diagnostics.
- `water_splatting/water_splatting.py`: optional config/state/mode/prepass integration and lifecycle.
- `scripts/experiments/run_direct_mdrr_cica_scene.py`: config, metadata, statistics and gradient audit.
- `scripts/experiments/run_gcrc_bnd_on.py`: fixed E0-E3 worker.
- `scripts/experiments/run_gcrc_registered_experiment.py`: registered queue and fixed extension.
- `scripts/experiments/summarize_gcrc_bnd_on.py`: metrics/statistics/contact sheets.
- `scripts/experiments/summarize_taba_bnd_on.py`: optional exact-fallback bound eligibility; prior defaults unchanged.
- `scripts/diagnostics/audit_gcrc_training.py`: real-source engineering audit.
- `scripts/diagnostics/audit_gcrc_artifacts.py`: asset/history/optimizer audit.
- `scripts/diagnostics/audit_taba_artifacts.py`: optional exact-fallback bound check; prior defaults unchanged.
- `tests/test_gcrc.py`: numerical, gradient, fallback, state and topology tests.
- This research record.

Defaults remain `gcrc_enabled=False`, `gcrc_mode="disabled"`, GRAC off.
