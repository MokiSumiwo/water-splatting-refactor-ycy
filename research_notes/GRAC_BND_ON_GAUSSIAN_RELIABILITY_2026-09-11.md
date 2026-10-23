# GRAC BND ON: Gaussian Reliability-Aware Chromatic Appearance Control

Repository: `/mnt/new/home_old/ycy/water-splatting-refactor`.
Branch: `research/bvam-bnd-on`. Implementation started on September 10;
formal training is registered on September 11, 2026. Existing uncommitted
BRASH/BVAM/TABA work is retained. No commit, push, install or asset deletion.
Output root: `outputs/grac_bnd_on_20260911` (abbreviated `<out>`).

## Registered mechanism

This experiment replaces uniform transmission-only chroma allocation with
per-Gaussian history. It introduces no trainable parameter, expert, loss,
or modification of medium/geometry physics, CICA targets or OCMC projector.

```text
delta = s_full - s_base
delta_luma = mean_RGB(delta)
d = delta - delta_luma
a = norm(d)
u = d / (a + 1e-8)
p_new = normalize(0.95 * p_old + 0.05 * detach(u))
m_new = 0.95 * m_old + 0.05 * detach(a)
m2_new = 0.95 * m2_old + 0.05 * detach(a*a)
variance = max(m2 - m*m, 0)
direction = clamp((1 + dot(detach(u), p)) / 2, 0, 1)
magnitude = 1 / (1 + sqrt(variance) / (abs(m) + 1e-5))
visibility = 1 - exp(-visibility_count / 8)
R = clamp(0.50*direction + 0.30*magnitude + 0.20*visibility, 0, 1)
g = 0.35 + 0.65*R
s_grac = s_full + (detach(g)-1)*d
rgb_grac = existing_safe_sigmoid(s_grac)
```

The current gate uses history from previous completed iterations. The
current ungated residual is recorded once, then history updates in the
existing `after_train` callback after backward/optimizer work and before
refinement. A repeated callback at the same step cannot double-count.
Evaluation, controller refreshes and standalone gradient-audit forwards
do not update history. The pending observation is cleared at `step_cb`.

Visibility is exactly `(radii > 0)`, the existing refinement mask. Counts
are training observations, including repeats of the same camera; they are
not numbers of distinct independent viewpoints. EMA mixes effects of
view changes and optimization changes, so R is a pseudo reliability
heuristic, not measured correctness or a calibrated confidence.

Prototypes initialize to zero with initialized=false. Only finite visible
residuals with norm > 1e-5 update direction. Finite visible magnitudes,
including zero, update magnitude moments. The first valid magnitude
initializes m=a and m2=a*a to avoid zero-initialization bias; subsequent
observations follow the registered EMA. An additional magnitude_count
tracks whether two valid samples exist. With insufficient magnitude
history or near-zero historical magnitude, stability=1. Without a valid
prototype or with a near-zero current residual, direction=1.

Before step 3000, with fewer than four visible observations, or with invalid
statistics, GRAC uses g=1. Missing source history starts from zero rather
than inferring history from the source checkpoint. An ordinary invalid
sample preserves the original logits/RGB. The existing safe sigmoid clips
floating-point endpoints in active modes; baseline E0 keeps its original
rounded endpoints to preserve exact disabled equivalence.

All history, R and gates are detached. Full/base logits are N x 3 from
features_dc N x 3 and degree-3 features_rest N x 15 x 3. GRAC alone preserves
mean logits up to roundoff. It retains appearance gradients to DC/rest;
the existing photometric renderer still has its ordinary geometry/medium
gradients. CICA remains restricted to features_dc, and OCMC is detached.

## State and topology

`GaussianReliabilityState` is a child module containing persistent buffers,
not optimizer parameters: direction_prototype (N x 3); magnitude_ema,
magnitude_sq_ema (N x 1 floats); visibility_count, magnitude_count (N x 1
int64); reliability_initialized_mask (N x 1 bool); last_update_step (scalar).
Floating buffers use the appearance device/dtype; counters/masks retain
their integer/bool dtypes when moved. Memory is O(N), about 37 bytes per
Gaussian for float32 persistent state, plus one pending residual/mask.

Old checkpoints auto-initialize these buffers when GRAC history is needed.
Disabled/TABA-only models add no history keys and accept old or GRAC-bearing
checkpoints. A GRAC checkpoint's complete buffer set is restored strictly;
partial, malformed or nonfinite history is rejected as corruption. Frozen
configs are also stored in checkpoint metadata.

Refinement appends `[old, split_parents.repeat(n_samples), clone_parents]`,
the exact Gaussian parameter ordering. Every history field follows this
order. `cull_gaussians` applies the same removal mask to all history, while
existing optimizer helpers update Adam state. Unit tests exercise the real
refinement/cull methods with identity-labelled history and Adam moments.
Shape mismatches fail explicitly; no per-frame history list accumulates.

## Registered matrix and combination

All arms: BND ON, M1+OCMC+CICA, Auxiliary/RAOC/BRASH/BVAM/BARR/BHTC/BGTR OFF,
seed 42, continuation from the registered per-scene source at step 3000 to
14999. CICA's original 10000 start and 10000/12000/14000 refresh are retained.
Only conda `water_splatting` and physical GPUs 6,7,8,9 are used; at most one
training process per physical GPU. No compact-output mode.

| Arm | grac_enabled | grac_mode | TABA |
|---|---|---|---|
| E0 | false | disabled | disabled |
| E1 | true | grac | disabled |
| E2 | true | taba | prior DCU directional E2 |
| E3 | true | grac_taba | GRAC plus DCU directional |

All GRAC hyperparameters are the requested defaults above. TABA uses
mid=.50, temperature=.15, floor=.20, uncertainty_lambda=0, axis_eps=1e-4.
Transmission and water axis come from the existing detached, ungated BND
rasterizer prepass, sampled at rounded Gaussian screen centers. No new
transmission estimator or threshold search is introduced.

E3 combines directionally, using the original d and requested raw water
axis w. Let parallel=dot(d,w)*w and orthogonal=d-parallel. For active GRAC:

```text
g_parallel = clamp(g_taba * g_grac, 0.35, 1)
s_combined = s_full + (g_grac-1)*d + (g_parallel-g_grac)*parallel
```

Thus g_grac scales the orthogonal component and g_parallel scales the
water-aligned component. A weak/invalid TABA axis uses parallel=d, matching
TABA's full-chroma fallback. Cold/invalid GRAC preserves TABA exactly;
the .35 final floor applies only when GRAC is active. This is an explicit
exception to a universal final floor to avoid altering TABA during GRAC
cold start. The raw RGB axis is not zero-mean, so E2/E3 inherit TABA's
possible mean-logit changes; GRAC alone does not introduce this effect.

The eight JapaneseGradens-RedSea/Panama runs all use strictly paired source
model, optimizer, scheduler, projector, RNG and camera sequence. After all
eight complete, select the stable E1/E2/E3 with highest two-scene mean final
PSNR, freeze it, then run Curasao/IUI3-RedSea E0 plus the selected arm.
The original two-scene E0/selected runs provide the other half of the
four-scene validation. Selection is exploratory on these test scenes.

## Engineering verification

56 pytest tests pass, including EMA, cold start, invalid state, numerical
bounds, detached gradients, state roundtrip, and real split/clone/prune plus
Adam alignment. The two real-source audits pass disabled forward/loss and
historical hash equivalence, strict old-state loading, zero initialization,
state roundtrip, frozen evaluation, warmed gates, once-per-iteration count,
appearance gradients, raw-J bounds and CICA/OCMC gradient routing.

Evidence is in `<out>/engineering/test_results.json` and
`<out>/engineering/verified/<scene>/audit.json`. Initial audit attempts used
the intentionally detached diagnostic RGB for backward; this audit error
was corrected to inspect differentiable logits, and failed logs are kept.
CUDA strict equivalence repeats the historical RAOC tolerance failures,
with errors exactly equal to the committed reference. This is a verified
nonregression, not a strict equivalence pass. RAOC remains off.

Training source files are frozen in `<out>/code_snapshot/manifest.json`.
Small single-seed differences remain limited by existing CUDA atomic
reduction nondeterminism. Compare against this experiment's paired E0,
not previous experiments' final metrics.

The audit-only script was strengthened during training: an earlier
`forward_history_unchanged` check captured the update-step counter too late
to prove its claim. It now hashes every history buffer before and after
the training forward, and checks backward separately. Both real scenes
pass this stronger audit in `<out>/engineering/history_verified/<scene>/audit.json`.
No frozen training source changed. The first rerun rejected the broad
CUDA visibility mask before model creation; reruns exposed only physical
GPU 6, and the rejected invocation log is retained.

## Evaluation and artifacts

Use project-local metrics.py on native exactly paired [0,1] float RGB;
PSNR/SSIM/VGG-LPIPS and paired MSE are computed before PNG quantization.
No resize, crop or extra color conversion. Contact-sheet thumbnails alone
are resized. Native clear is the unchanged J/(1+J) display; raw J is saved
separately. There is no clear-image ground truth.

GRAC statistics summarize currently visible Gaussians. Scene quantiles are
means of per-view quantiles, not pooled population quantiles. E0/E2 do not
track GRAC history: their neutral diagnostic placeholders are marked by
grac_history_available=0 and are not measured reliability. Image RGB bounds
use global view extrema. Object/medium residuals are mean |rgb_object-GT|
and mean |rgb_medium-(GT-rgb_object)|, without separate component GT.
The inherited `g_chroma_*` columns and first "gate mean" table in report.md
describe the TABA gate, so E1 correctly has g_chroma=1 there. GRAC's applied
gate is in `grac_gate_*`; E3's combined water-parallel gate is in
`grac_final_parallel_gate_*`. These are different quantities.

The reused runner stores legacy `taba_result.json`, `taba_training.jsonl`
and snapshot names alongside explicit GRAC filenames. The first step-5000
JSONL row is a source-state gradient audit, followed by formal records
starting at step 3500. It is not the trained step-5000 snapshot. Snapshot
CSV/metrics/PNG use the actual saved checkpoints. Evaluation never updates
GRAC history, so test-camera order cannot change subsequent reliability.

| Asset | Relative to `<out>` |
|---|---|
| Checkpoints | `<scene>/<arm>/checkpoints/step-{000005000,000010000,000014999}.ckpt` |
| Native renders | `renders/<arm>/<scene>/step_{05000,10000,14999}/<view>/A2/{underwater,clear,clear_raw_display_clamp01,gt}.png` |
| Metrics | `metrics_final.csv`, `metrics_all_steps.csv`, `<scene>/<arm>/{evaluation_metrics,per_view_metrics}.csv` |
| GRAC history diagnostics | `<scene>/<arm>/grac_training.jsonl`, `<scene>/<arm>/grac_statistics_step_{05000,10000,14999}.csv` |
| All evaluation statistics | `diagnostics_all_steps.csv`, `report.md` |
| Gradient audits | `<scene>/<arm>/gradient_routing_audit.json` |
| Comparison images | `contact_sheets/` |
| Pairing and summaries | `summary.json`, `two_scene_summary.json`, `four_scene_selection.json` |
| Asset/history audit | `artifact_integrity_audit.json`, `grac_artifact_audit.json` |

## Results

### Two-scene screen

All eight runs completed 11999 updates from the paired step-3000 source
through step 14999, retaining all three requested checkpoints and renders.
The frozen selection is E1 (GRAC only), the highest mean final PSNR among
E1/E2/E3. E0 has higher mean PSNR than all candidates. Selection therefore
does not establish a gain over baseline. See `<out>/four_scene_selection.json`
and the complete `<out>/two_scene_report.md`.

| Scene | Arm | PSNR | SSIM | LPIPS | MSE |
|---|---|---:|---:|---:|---:|
| JapaneseGradens-RedSea | E0 | 24.897843 | 0.896009 | 0.134499 | 0.005091522 |
| JapaneseGradens-RedSea | E1 | 24.864281 | 0.895694 | 0.134319 | 0.005219884 |
| JapaneseGradens-RedSea | E2 | 24.781452 | 0.895549 | 0.134572 | 0.005176200 |
| JapaneseGradens-RedSea | E3 | 24.735481 | 0.894713 | 0.134477 | 0.005220188 |
| Panama | E0 | 31.518516 | 0.947330 | 0.103145 | 0.000710419 |
| Panama | E1 | 31.529104 | 0.947272 | 0.103304 | 0.000708536 |
| Panama | E2 | 31.568525 | 0.947316 | 0.102505 | 0.000702348 |
| Panama | E3 | 31.527654 | 0.947208 | 0.103602 | 0.000708949 |
| Two-scene mean | E0 | 28.208179 | 0.921670 | 0.118822 | 0.002900970 |
| Two-scene mean | E1 | 28.196692 | 0.921483 | 0.118811 | 0.002964210 |
| Two-scene mean | E2 | 28.174988 | 0.921433 | 0.118538 | 0.002939274 |
| Two-scene mean | E3 | 28.131567 | 0.920960 | 0.119039 | 0.002964568 |

GRAC-only changes JapaneseGradens PSNR by -0.033562 dB and Panama by
+0.010588 dB. JapaneseGradens SSIM/MSE worsen while LPIPS slightly improves;
Panama MSE slightly improves while SSIM/LPIPS worsen. The mean PSNR change
is -0.011487 dB, with no consistent metric improvement. E2 improves Panama
by +0.050009 dB but loses -0.116391 dB on JapaneseGradens. E3 is worse than
E1 by -0.128800 dB on JapaneseGradens and -0.001450 dB on Panama, averaging
-0.065125 dB; its mean SSIM, LPIPS and MSE are also worse than E1.

E1's JapaneseGradens view-level PSNR changes are MTN_1090 -0.148272,
MTN_1098 -0.002739, MTN_1106 +0.050323 dB. MTN_1090 remains the difficult
view (19.140055 dB with E1). Panama changes are MTN_1539 +0.053356,
MTN_1529 +0.004768, MTN_1547 -0.026360 dB. These mixed signs also argue
against a uniform improvement.

### Screen visual inspection

Inspected all six four-arm contact sheets (underwater, clear and raw J for
both scenes), plus native E0/E1 underwater and raw-J images for JapaneseGradens
MTN_1090 and Panama MTN_1539, and JapaneseGradens MTN_1090 underwater GT.
JapaneseGradens retains the stretched left/bottom boundary artifacts in
MTN_1090; GRAC changes their shape but does not remove them. Clear/dewatered
views retain black uncovered regions and smeared geometry at the top.
Interior colors and structures remain very similar across arms.
Panama retains nearly identical coral texture, chart appearance and depth
coloration, with small local differences and persistent peripheral artifacts.
No consistent underwater or clear/dewatered visual improvement is evident.
There is no clear-image GT, so this is a comparative visual observation,
not a claim about recovered true object colors.

### Fixed four-scene extension

All 12 registered runs finished successfully. The coordinator exited 0
with complete_runs=12 and audit_passed=true. Curasao/IUI3-RedSea use the
same frozen E1 defaults as the screen. JapaneseGradens/Panama reuse their
completed paired E0/E1 runs. Means weight the four scenes equally.

| Scene | Arm | PSNR | SSIM | LPIPS | MSE |
|---|---|---:|---:|---:|---:|
| Curasao | E0 | 32.463268 | 0.958981 | 0.125736 | 0.000726580 |
| Curasao | E1 | 32.469610 | 0.959028 | 0.125683 | 0.000729553 |
| IUI3-RedSea | E0 | 30.748594 | 0.908817 | 0.199332 | 0.001622210 |
| IUI3-RedSea | E1 | 30.670141 | 0.908595 | 0.199617 | 0.001675846 |
| JapaneseGradens-RedSea | E0 | 24.897843 | 0.896009 | 0.134499 | 0.005091522 |
| JapaneseGradens-RedSea | E1 | 24.864281 | 0.895694 | 0.134319 | 0.005219884 |
| Panama | E0 | 31.518516 | 0.947330 | 0.103145 | 0.000710419 |
| Panama | E1 | 31.529104 | 0.947272 | 0.103304 | 0.000708536 |
| Four-scene mean | E0 | 29.907055 | 0.927784 | 0.140678 | 0.002037683 |
| Four-scene mean | E1 | 29.883284 | 0.927647 | 0.140731 | 0.002083455 |

Four-scene E1 minus paired E0: PSNR -0.023771 dB, SSIM -0.000137227,
LPIPS +0.000052743, MSE +0.000045772. All four mean metrics worsen.
Curasao has only a +0.006342 dB PSNR change; SSIM/LPIPS slightly improve
but MSE worsens. Its view changes are MTN_1288 -0.039145, MTN_1296 +0.054363,
MTN_1304 +0.003807 dB. Mean PSNR and mean MSE need not move in opposite
directions because the PSNR logarithm is applied separately per view.

IUI3 loses -0.078453 dB and worsens all four metrics. All four view-level
PSNR changes are negative: MTN_5903 -0.008804, MTN_5894 -0.169291,
MTN_5911 -0.074665, MTN_5928 -0.061050 dB. Thus GRAC does not preserve
IUI3 performance; Curasao's mixed near-zero change is not convincing
evidence of a retained gain. The two extension scenes average -0.036055 dB.

### Final reliability and bounds

The complete mean/p10/p50/p90 tables for visibility count, direction,
magnitude stability, visibility confidence, R, GRAC gate and final parallel
gate are in `<out>/report.md` and `<out>/diagnostics_all_steps.csv`, together
with all requested residual, object/medium, clear-range, J and tau metrics.
The following values use final-step visible Gaussians and average views.

| Scene | Arm | Count mean | Direction | Magnitude | Visibility | R | GRAC gate |
|---|---|---:|---:|---:|---:|---:|---:|
| JapaneseGradens-RedSea | E1 | 9504.737 | 0.888318 | 0.789173 | 0.999980 | 0.880907 | 0.922592 |
| JapaneseGradens-RedSea | E3 | 9499.803 | 0.895862 | 0.795550 | 0.999979 | 0.886592 | 0.926287 |
| Panama | E1 | 9183.258 | 0.877040 | 0.783985 | 1.000000 | 0.873716 | 0.917915 |
| Panama | E3 | 9189.540 | 0.876786 | 0.783366 | 1.000000 | 0.873403 | 0.917712 |
| Curasao | E1 | 10024.976 | 0.952861 | 0.825072 | 1.000000 | 0.923952 | 0.950569 |
| IUI3-RedSea | E1 | 8423.250 | 0.959760 | 0.845481 | 0.999984 | 0.933521 | 0.956791 |

| Scene | E1 gate p10 | p50 | p90 | Active fraction | Fallback fraction |
|---|---:|---:|---:|---:|---:|
| JapaneseGradens-RedSea | 0.818173 | 0.945505 | 0.986233 | 0.999981 | 0.00001928 |
| Panama | 0.811318 | 0.943594 | 0.982883 | 1.000000 | 0 |
| Curasao | 0.901954 | 0.964136 | 0.989686 | 1.000000 | 0 |
| IUI3-RedSea | 0.907069 | 0.970822 | 0.992529 | 0.999985 | 0.00001495 |

History is active, so the negative outcome is not a disabled/diagnostic-only
experiment. Visibility confidence saturates rapidly: count p90 is 11999
in every E1 scene, with histories inherited by split/clone children. The
visibility term contributes almost a constant 0.2 to R by the end;
direction and magnitude dominate the remaining variation. E1 mean gates
retain about 92-96% of the chromatic residual. High reliability in IUI3
still accompanies worse image metrics, so this proxy is not a validated
measure of correct object color. No causal failure mechanism is established
by this single-seed experiment.

Across all saved active-arm evaluations, intrinsic RGB lies in
[4.768109462816028e-9, 0.9999998807907104], strictly within (0,1).
All formal logged and evaluated P(J>1) and P(intrinsic_rgb>1) equal zero;
raw J is checked before display clamping. Uncovered pixels may have J=0.
The existing clear display J/(1+J) is in [0,0.499975] in final E1 views.
E0 retains baseline floating-point endpoints for exact disabled equivalence.
E1 mean absolute luminance-logit change is about 0.9e-8 to 1.3e-8;
E2/E3 can inherit TABA's raw-axis mean-logit change as described above.

### Extension visual inspection

Inspected all Curasao and IUI3 E0/E1 contact sheets for underwater, clear
and raw J, plus the three four-scene overviews. Also inspected native
IUI3 MTN_5894 E0/E1 underwater images and underwater GT. Curasao retains
nearly identical subject detail and chart colors. Its clear/dewatered
top boundary remains uncovered, and E1 has a visible vertical streak at
the upper-right edge of MTN_1288. IUI3 retains large left/bottom artifacts
in MTN_5894 and dark or blue regions above the reconstructed surface in
clear/dewatered images. Interior differences are small; no consistent
improvement in underwater or clear/dewatered visual quality is evident.
The four-scene overview chooses the first sorted view in each scene; all
13 test views are available in the per-scene sheets and native renders.

### Reference comparison and decision

| Four-scene result | PSNR | SSIM | LPIPS | MSE |
|---|---:|---:|---:|---:|
| User-supplied BND OFF reference | 30.031764 | 0.927403 | 0.137838 | 0.002015746 |
| Historical BND ON mainline | 29.899295 | 0.927596 | 0.140541 | 0.002031043 |
| Historical TABA-DCU | 29.924975 | 0.927634 | 0.140667 | 0.002002410 |
| This paired E0 | 29.907055 | 0.927784 | 0.140678 | 0.002037683 |
| This GRAC E1 | 29.883284 | 0.927647 | 0.140731 | 0.002083455 |

GRAC is 0.148480 dB below the supplied BND OFF reference, with LPIPS
0.002892779 and MSE 0.000067709 worse, although SSIM is 0.000244113 higher.
The same-run E0 gap is 0.124709 dB. GRAC therefore does not close this gap.
It is also 0.041691 dB below historical TABA-DCU; historical comparisons
are contextual, not paired causal estimates. CUDA atomic nondeterminism
and one seed limit interpretation of these small differences.

Retain GRAC's implementation, tests and complete positive/negative assets
as a reproducible research ablation. Do not promote it to the default
training recipe: this version does not improve the paired four-scene mean,
does not improve JapaneseGradens, does not retain IUI3 performance, and
offers no consistent perceptual gain. GRAC+TABA is worse than GRAC alone
on the registered screen. Keep grac_enabled=False and grac_mode="disabled".
No further threshold search was performed.

### Runtime and final audit

| Scene | E0 training seconds | E1 training seconds | Observed increase |
|---|---:|---:|---:|
| JapaneseGradens-RedSea | 847.41 | 894.14 | 5.51% |
| Panama | 1230.80 | 1338.86 | 8.78% |
| Curasao | 1222.98 | 1290.43 | 5.51% |
| IUI3-RedSea | 782.74 | 856.97 | 9.48% |

These are runner-reported elapsed training times, not isolated kernel
benchmarks. Concurrent workers, different GPUs and evolving Gaussian
counts can affect the ratios. All runs completed without the specified
engineering stopping conditions. Persistent history scales with current
Gaussian count; all saved history and Adam moment shapes align.

- 56 pytest tests passed before training; no training implementation changed afterward.
- Relevant Python compilation and git diff --check passed.
- Both original and strengthened real-scene loading/history/gradient audits passed.
- All four scenes passed source/model/optimizer/scheduler/projector/RNG/camera pairing checks.
- All 12 run-level gradient audits passed; CICA escape gradients are zero and OCMC projectors are unchanged.
- CUDA audit shows no regression against the committed reference, but strict forward/backward tolerance checks still fail in historical RAOC diagnostics. This is not reported as a strict CUDA pass; RAOC is off.
- Generic artifact audit and GRAC state/optimizer audit passed: 12 runs, 36 training checkpoints with matching hashes, 456 native RGB PNGs, 21 contact sheets, 13 unique test GT views with exact original pixels.
- Training code hashes match all seven frozen snapshot entries. Full audit roundtrip checkpoints are retained separately from the 36 training checkpoints.

Final audit evidence: `<out>/artifact_integrity_audit.json`,
`<out>/grac_artifact_audit.json`, `<out>/engineering/test_results.json`,
`<out>/engineering/history_verified/<scene>/audit.json`, and
`<out>/engineering/cuda_nonregression.json`. Full training log:
`<out>/logs/registered_queue.log`. No checkpoint or render was deleted.
The checkpoints contain a training RNG manifest rather than the full RNG
state, so exact continuation of an interrupted trajectory is not claimed;
these registered runs completed uninterrupted from their paired source.

Final branch is `research/bvam-bnd-on`; the worktree remains dirty with
the pre-existing research work and this GRAC implementation. The full
status, final compilation checks, GPU status and evidence paths are saved
in `<out>/final_verification.json`. All GRAC workers and the coordinator
have exited; physical GPUs 6-9 are idle. No commit or push was performed.

## Changed files

- `water_splatting/grac.py`: reliability helper, persistent state and statistics.
- `water_splatting/water_splatting.py`: config, integration, callbacks and state topology.
- `scripts/experiments/run_direct_mdrr_cica_scene.py`: GRAC config/metadata/statistics and compatibility audit.
- `scripts/experiments/run_grac_bnd_on.py`: fixed E0-E3 runner.
- `scripts/experiments/run_grac_registered_experiment.py`: screen and fixed extension coordinator.
- `scripts/experiments/summarize_grac_bnd_on.py`: metrics, statistics and comparisons.
- `scripts/diagnostics/audit_grac_training.py`: real-source engineering audit.
- `scripts/diagnostics/audit_grac_artifacts.py`: assets and checkpoint/history/optimizer alignment.
- `tests/test_grac.py`: numerical and topology tests.
- This research record.

Defaults remain `grac_enabled=False`, `grac_mode="disabled"`.

## Revalidation of repeated request

The attachment received again on September 11, 2026, under identifier
`746d0e5f-8521-47f1-88d3-92eab6ea230b`, is byte-identical to the original
GRAC request `48c0ced2-b94c-48f1-aa64-9ce9d06412e5`. Both SHA256 digests are
`30ec27d01c7537072f65d6ced4107b0d56be26e7c2b0cddba5af71069474ae66`.
The requested 12 completed runs are reused. This revalidation did not launch
another training experiment, recompute the full metric set, alter training
code or change the registered configurations or reported metrics.

The subsequent GCRC work extended the shared model and direct scene runner.
Five of the seven current files still match the original GRAC training
snapshot, including `grac.py`, `taba.py`, the appearance helper and both
GRAC/TABA runners. The shared model and direct scene runner differ because
of the GCRC integration; all seven original snapshot files remain intact.
The prior final-verification record describes the code at original completion,
not a claim that these shared files never changed afterward.

Current-code verification for this repeated request:

- All 82 current tests pass in 6.93 seconds, with two dependency deprecation warnings.
- Both real-source audits pass on physical GPUs 6 and 8, covering disabled
  exact forward/loss/optimizer behavior, old checkpoint loading, E1/E2/E3,
  state roundtrip, frozen forward/backward/evaluation history, once-only
  updates, active gates, appearance gradients, strict intrinsic RGB bounds,
  raw-J bounds and CICA/OCMC routing. New evidence is stored under
  `<out>/engineering/revalidation_746d0e5f/<scene>/audit.json`; roundtrip
  checkpoints are retained beside these new audit files.
- The complete asset audit was rerun read-only against all 12 historical
  runs: 36 checkpoint hashes and state/Adam alignments, 456 native RGB PNGs,
  21 contact sheets, 13 unique original GT views, all four scene pairings,
  statistics and zero-overflow records pass. Original audit files were
  preserved; this result is recorded in `<out>/request_revalidation_746d0e5f.json`.
- Both four-scene underwater and raw-J contact sheets were inspected again.
  Persistent boundary smears, uncovered black regions and the Curasao
  vertical streak remain; no consistent visual improvement is established.
- Relevant Python compilation and `git diff --check` pass. The earlier
  CUDA nonregression audit remains the CUDA-equivalence evidence; it was
  not rerun here and still does not constitute a strict RAOC tolerance pass.

Numerical clarification: GRAC-enabled modes apply the unchanged safe_sigmoid
even when a fallback gate is one. Finite logits then remain unchanged, but
rounded baseline RGB endpoints may be moved inside (0,1). Thus GRAC fallback
does not promise bitwise equality to the baseline at saturated endpoints.
The fully disabled path remains bitwise equivalent. Raw J is bounded in
[0,1] and may equal zero on uncovered pixels. These limits were retained
without changing the implementation or historical results.

The decision remains to retain GRAC and all positive/negative assets for
research while leaving it disabled by default. Current worktree status,
source hashes, environment, audit evidence and GPU state are captured in
the new request-revalidation manifest. No commit, push, install or asset
deletion was performed.
