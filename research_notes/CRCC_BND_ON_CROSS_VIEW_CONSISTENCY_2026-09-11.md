# CRCC: Clear-Radiance Cross-View Consistency, BND ON

Repository: `/mnt/new/home_old/ycy/water-splatting-refactor`.
Branch: `research/bvam-bnd-on`. Registered September 11, 2026.
Output root: `outputs/crcc_bnd_on_20260911` (`<out>` below).
Final status: all 14 runs trained to step14999 and evaluated at
5000/10000/14999. All 42 training checkpoints and 540 native PNGs retained.
CRCC is retained as an experiment, with its default disabled. Small paired
underwater metric gains did not produce consistent clear-image improvement.
All existing dirty research work and historical experiments are retained.
No install, commit, push, deletion or compact mode. All Python/test/training/
evaluation uses conda `water_splatting`; only physical GPUs 6,7,8,9, at most
one training worker per GPU.

## Registered mathematics

```text
J_i(v) = existing_safe_sigmoid(current_appearance_logits_i(v))
c_i(v) = J_i(v) - mean_RGB(J_i(v))
target = detach(previous_completed_iteration_chroma_ema_i)
r_i = c_i(v) - target
rho_i = mean_RGB(sqrt(r_i^2 + 1e-6))
support_i = 1 - exp(-prior_observation_count_i / 4)
medium_weight_i = .25 + .75 * clamp(1 - mean_RGB(T_i), 0, 1)
active_i = visible & finite_current & valid_T & initialized & prior_count>=2
w_i = clamp(active_i * support_i * medium_weight_i * 1, 0, 1)
L_CRCC = sum(w_i * rho_i) / (sum(w_i) + 1e-8)
lambda_CRCC = .02 * clamp((step-3000)/2000, 0, 1)
L_total = L_existing + lambda_CRCC * L_CRCC
```

The attachment repeats the medium floor in its final-weight formula. We
apply the floor once, matching its medium-weight definition. Its invalid-T
clauses also differ; the final fallback requirement takes precedence:
invalid transmission retains diagnostic medium_weight=.25 but contributes
zero loss weight and does not update EMA. No alpha/contribution aggregate
is exposed by this renderer, so w_alpha=1. These choices are fixed before
training. There is no additional gate, expert or learnable parameter.

Only RGB-space chroma is penalized. There is no luma loss, although changing
appearance parameters can indirectly change brightness. Constant zero
chroma is a valid achromatic observation, not an empty residual. Two prior
valid observations are required by the explicit count>=2 rule: the third
forward after a cold start can first have nonzero weight. Repeated cameras
count; this is not a distinct-camera counter. EMA mixes view changes and
parameter evolution and cannot certify true intrinsic colors.

## Actual tensors and gradient boundaries

The bounded Gaussian appearance path in `fields/gaussian_appearance.py`
evaluates features_dc `(N,3)` and features_rest `(N,15,3)` at detached
camera-to-Gaussian directions. Appearance logits and RGB are `(N,3)`.
CRCC reuses these logits, not final underwater RGB. E3 uses the existing
directional TABA-adjusted logits so its target matches rendered intrinsic
appearance. `safe_sigmoid` is unchanged. Active CRCC applies it to renderer
RGB as well, giving strict `(0,1)` Gaussian intrinsic values even when
the cold consistency loss is zero. This can change extreme baseline
sigmoid endpoints by epsilon. The disabled path keeps exact original bits.

The existing final underwater rasterizer supplies rgb_object, rgb_medium,
rgb_clear `(H,W,3)`, normalized expected depth `(H,W,1)` and accumulation.
The existing model computes tau_D=medium_attn*depth and
T=exp(-clamp(tau_D,0)). CRCC passes these detached renderer-derived tensors
to `sample_taba_transmission`, sampling at rounded projected `xys (N,2)`.
Samples require finite positive depth/accumulation, valid image position and
radii>0. Visibility is exactly refinement's `radii > 0`.
CRCC-only adds no extra renderer prepass; E3 retains TABA's original detached
prepass for its gate, then uses final-render T for consistency weights.

CRCC's target, masks, count, support, T/tau_D/depth, weights and coefficient
are detached. Its only parameter gradients are features_dc/features_rest.
The training runner removes CRCC loss before existing photometric routing
and before CICA gradient-norm calibration, then adds only its appearance
gradients. Existing main/medium/geometry/OCMC/CICA formulas and weights are
unchanged. CICA remains DC-only; OCMC projector stays detached and unchanged.
No CRCC gradients reach medium, geometry, projectors, gates or GRAC/GCRC.

Raw image J can be zero at uncovered pixels, despite strict Gaussian
intrinsic RGB bounds. A chroma-only appearance loss cannot create missing
geometry or guarantee removal of black holes, haze or stripe artifacts.
There is no real clear-image ground truth in this experiment.

## EMA and topology

Persistent buffers in optional `ClearRadianceHistory` are
crcc_chroma_ema `(N,3)` float, crcc_initialized `(N,1)` bool,
crcc_observation_count `(N,1)` int64 and scalar last_update_step int64.
This costs approximately 21 bytes/Gaussian in float32, plus one pending
observation, without an accumulating frame list or optimizer parameters.

First accepted visible finite sample sets EMA=c and count=1. Later samples
set EMA=.95*old+.05*c and increment count. Forward reads previous history
and queues a detached sample. Only actual `after_train` consumes it, once,
after optimizer work and before refinement. Eval, backward and controller
forwards do not mutate history; step_cb clears stale pending observations.

Old checkpoints initialize zero history; disabled mode accepts old or new
checkpoints without activating optional history. Complete state roundtrips
exactly. Partial/corrupt/nonfinite/misaligned state is rejected. Floats move
with model dtype/device; bool/int64 remain discrete. Runtime helper shape
mismatch yields zero weight and a diagnostic; actual model/topology state
misalignment raises as required by the engineering stopping conditions.

Split/clone buffers follow exact parameter order
`[old, split_parents.repeat(n_samples), clone_parents]`; prune uses the same
removal mask. The existing Adam topology helpers remain responsible for
optimizer state. Tests exercise real model refinement/cull with labelled
lineage; all 42 saved training checkpoints passed state and Adam alignment
audits. Inherited observations retain their counts.

## Fixed protocol and selection

M1+OCMC+CICA; BND ON; Auxiliary/RAOC/BRASH/BVAM/BARR/BHTC/BGTR/GRAC/GCRC OFF.
Seed42, same verified step3000 checkpoint and camera sequence per scene.
Each continuation runs 11999 updates to14999 and preserves5000/10000/14999.

| Arm | crcc_enabled | crcc_mode | TABA |
|---|---|---|---|
| E0 | false | disabled | off |
| E1 | false | taba | existing DCU E2 directional |
| E2 | true | crcc | off |
| E3 | true | crcc_taba | existing DCU E2 directional |

All CRCC defaults are exactly the registered values above; beta=.95,
Charbonnier eps=.001, support scale4, minimum observations2, floor=.25,
start3000, ramp2000, lambda_max=.02. TABA retains mid=.5, temperature=.15,
floor=.2, uncertainty lambda0 and water_axis_eps1e-4. No hyperparameter search.

Screen all four arms in JapaneseGradens-RedSea and Panama. After all eight
runs complete, rank stable E1/E2/E3 using equal mean rank of two-scene mean
PSNR/SSIM (higher) and LPIPS/MSE (lower); ties use PSNR. Inspect all underwater,
clear and raw-clear display sheets; veto serious visual regressions before
extension, recording the decision and sources. Extend the single selected
configuration plus paired E0 in Curasao and IUI3-RedSea. This is exploratory
selection using the two screening scenes, not independent validation there.

Expected final assets:12 runs,36 training checkpoints,456 native PNGs and
21 contact sheets, plus engineering roundtrip checkpoints and complete
logs/statistics. Snapshot training/evaluation code before launching. No
training code changes during experiments. Recheck source/code hashes.

Evaluation uses project-local `metrics.py`, original native float RGB,
exact view/GT pairing, no resize/crop/color conversion. Contact-sheet
resizing is for visualization only. Underwater prediction is evaluated in
[0,1]. clear.png retains existing J/(1+J) display; the extra raw display
uses clamp01. Metrics are not recomputed from tone-mapped clear PNGs.

## Engineering evidence at registration

Both real-scene audits in `<out>/engineering/verified_v3/<scene>/audit.json`
pass: old source loading, historical disabled forward hash, exact disabled
forward/loss, identical optimizer updates with identical gradients, state
roundtrip, cold fallback, bounded outputs, frozen eval/backward history,
once-per-step callbacks, nonzero warm CRCC gradients to DC/rest and zero
escape. CICA/OCMC routing audits pass for all E1/E2/E3 modes.

Independent repeated CUDA backwards are not bit-identical: this occurs for
both baseline repeats and explicit-disabled repeats. Differences are
reported per parameter with the same fixed atol5e-5/rtol1e-4 check. This
must not be described as bit-exact full CUDA optimizer trajectories.
The initial strict hash failure is preserved in engineering/verified.
The intermediate diagnostic script's empty-gradient reduction failure is
corrected in v3. No formal training assets existed during these changes.

`engineering/cuda_nonregression.json`: no_regression=true, but historical
RAOC strict forward=false and backward=false. Current and committed
reference discrepancies are identical. This historical failure is not
reported as a strict pass and does not block formal experiments.

Pytest, py_compile and git diff --check evidence is generated by
`scripts/diagnostics/verify_crcc_engineering.py` before launch. Final artifact
audits and results are appended below after all runs and visual review.

## Artifact paths

- Source manifests: `<out>/{source_asset_audit,four_scene_source_audit}.json`.
- Frozen files: `<out>/code_snapshot/manifest.json` and source copies.
- Checkpoints: `<out>/<scene>/<arm>/checkpoints/step-{step:09d}.ckpt`,
  step in (5000,10000,14999), e.g. step-000014999.ckpt.
- Native renders: `<out>/renders/<arm>/<scene>/step_<step>/<view>/A2/`.
- RGB metrics: `<out>/<scene>/<arm>/{evaluation_metrics,per_view_metrics}.csv`.
- CRCC: `<out>/<scene>/<arm>/crcc_training.jsonl` and
  `crcc_statistics_step_{05000,10000,14999}.csv`.
- Gradients: `<out>/<scene>/<arm>/gradient_routing_audit.json` and
  `training_summary_A2.csv` with per-group CRCC norms.
- Sheets: `<out>/contact_sheets/`; aggregate CSVs and report at `<out>/`.
- Historical references: BND OFF30.031764/.927403/.137838/.002015746;
  BND ON29.899295/.927596/.140541/.002031043;
  TABA29.895461/.927672/.140587/.002050006;
  GRAC29.883284/.927647/.140731/.002083455 (PSNR/SSIM/LPIPS/MSE).
  New paired E0 is the primary comparator because single-seed CUDA
  nondeterminism has produced approximately .19dB Japanese baseline variation.

## Screen completed and extension decision

All eight runs reached14999 and evaluated5000/10000/14999. All pairing and
integrity checks passed. Tests:111 passed, two dependency warnings;
py_compile and git diff --check passed. Training/evaluation code hashes
remain identical to the registered snapshot.

| Two-scene mean | PSNR | SSIM | LPIPS | MSE | Mean rank |
|---|---:|---:|---:|---:|---:|
| E0 | 28.102653 | .920959 | .118718 | .003031300 | reference |
| E1 | 28.173130 | .921635 | .118363 | .002915792 | 1.25 |
| E2 | 28.180829 | .921634 | .118609 | .002961897 | 2.00 |
| E3 | 28.137180 | .921231 | .118502 | .002969814 | 2.75 |

E1 wins the registered rule; E2's PSNR advantage over E1 is only .007700dB
while the other three metrics favor E1 (SSIM differs by only .00000113).
All six full screening sheets were visually inspected before extension.
There is no gross new visual failure justifying a candidate veto. Underwater
differences are subtle. Japanese black upper regions, blue haze and edge
stretching persist; Panama cyan upper background and streaks persist. E2
reduces global Japanese near-black pixel fraction without eliminating the
visible failure shapes. No true clear ground truth or significance claim.

The selected unified four-scene configuration is therefore E1. Because
E1 disables CRCC, two supplementary fixed E2 runs (Curasao and IUI3) are
registered BEFORE extension training, in
`<out>/crcc_only_four_scene_extension`. They use the same source, seed42,
camera sequence and full snapshot/evaluation protocol. This enables direct
CRCC four-scene reporting without changing the selected E1 or selecting
parameters per scene. Total deliverables become14 runs,42 training
checkpoints and540 native PNGs, plus the engineering/visualization assets.
The registered12-run audit remains separate from the supplementary audit.

## Final metrics

All values below are step14999, using local `metrics.py` on exactly paired
native-resolution float RGB. Scene means weight views equally; aggregate
means weight scenes equally. PSNR is averaged per view, not calculated from
the aggregate MSE. There are 3 test views each for JapaneseGradens, Panama
and Curasao, and 4 for IUI3. These small test sets and the single seed limit
generalization. E3 was screened only in the first two scenes, as registered.

| Scene | Arm | PSNR | SSIM | LPIPS | MSE |
|---|---|---:|---:|---:|---:|
| JapaneseGradens-RedSea | E0 | 24.738506 | .895054 | .133769 | .005342901 |
| JapaneseGradens-RedSea | E1 | 24.810076 | .895860 | .133529 | .005124354 |
| JapaneseGradens-RedSea | E2 | 24.817813 | .895883 | .133875 | .005217838 |
| JapaneseGradens-RedSea | E3 | 24.753318 | .895088 | .133968 | .005229472 |
| Panama | E0 | 31.466801 | .946864 | .103668 | .000719700 |
| Panama | E1 | 31.536183 | .947410 | .103196 | .000707230 |
| Panama | E2 | 31.543845 | .947385 | .103342 | .000705957 |
| Panama | E3 | 31.521041 | .947375 | .103035 | .000710157 |
| Curasao | E0 | 32.596074 | .959136 | .125974 | .000689258 |
| Curasao | E1 | 32.450913 | .959160 | .125705 | .000730332 |
| Curasao | E2 | 32.459662 | .959122 | .125832 | .000732580 |
| IUI3-RedSea | E0 | 30.613711 | .908443 | .199635 | .001722094 |
| IUI3-RedSea | E1 | 30.693582 | .909216 | .199468 | .001670037 |
| IUI3-RedSea | E2 | 30.712737 | .909106 | .199481 | .001624346 |
| Two-scene mean | E0 | 28.102653 | .920959 | .118718 | .003031300 |
| Two-scene mean | E1 | 28.173130 | .921635 | .118363 | .002915792 |
| Two-scene mean | E2 | 28.180829 | .921634 | .118609 | .002961897 |
| Two-scene mean | E3 | 28.137180 | .921231 | .118502 | .002969814 |
| Four-scene mean | E0 | 29.853773 | .927374 | .140762 | .002118488 |
| Four-scene mean | E1 | 29.872689 | .927912 | .140475 | .002057988 |
| Four-scene mean | E2 | 29.883514 | .927874 | .140633 | .002070180 |

The complete precision table is
[metrics_final_including_CRCC_four_scene.csv](../outputs/crcc_bnd_on_20260911/metrics_final_including_CRCC_four_scene.csv).
All intermediate metrics remain in each run and both aggregate reports.

## Findings and decision

- JapaneseGradens: E2 improves paired PSNR by .079308dB, SSIM by .000829,
  and MSE by .000125063, but LPIPS increases .000106. This is a small mixed
  underwater result. Large upper black areas, blue haze and edge stretching
  remain in clear rendering; E3 does not improve the outcome.
- Panama: E2 improves all four paired underwater metrics (PSNR+.077045dB).
  Visual differences are subtle. Cyan upper background and stretched
  streaks remain across E0/E1/E2/E3; no convincing clear artifact removal.
- Curasao: E2 loses .136412dB PSNR, slightly lowers SSIM and raises MSE;
  LPIPS improves slightly. Clear rendering still has large top/right black
  regions and a cyan boundary. MTN_1288 shows a conspicuous broad gray
  streak at the upper right in E2. The baseline green upper-left streak is
  less apparent, but this is not a reliable overall improvement.
- IUI3: E2 improves all four underwater metrics (PSNR+.099025dB), while
  clear upper-background black coverage worsens. Thin radiating streaks
  around the top/left horizon are conspicuous, especially MTN_5903 and
  MTN_5911. Foreground appearance is broadly similar. This adverse clear
  result must not be hidden by the underwater metric gains.
- TABA comparison: E2's four-scene PSNR is .010826dB above newly paired E1,
  but SSIM, LPIPS and MSE favor E1. CRCC is not comprehensively better than
  TABA. The registered multi-metric screen correctly selected E1; the
  supplementary E2 results do not change that selection retrospectively.
- Combination: E3 minus E2 two-scene PSNR is -.043650dB, with worse SSIM
  and MSE, and slightly better LPIPS. CRCC+TABA is not better overall than
  CRCC-only and was not extended to the other scenes.
- Underwater visuals show no consistent readily visible gain. Clear
  visuals do not show stable improvement in black holes, cyan/blue residue
  or stripes. This experiment does not meet its primary visual objective.
- E2 minus historical BND OFF: PSNR-.148250dB, SSIM+.000471,
  LPIPS+.002795 and MSE+.000054434. Capacity loss is not recovered overall.
  E2 also trails historical BND ON by .015781dB and historical TABA by
  .011947dB. These historical runs are secondary references, not paired
  causal estimates; current E0 remains the primary comparator.
- Retain CRCC implementation, all configurations and negative results for
  reproducibility. Keep `crcc_enabled=False`, `crcc_mode="disabled"` as
  defaults. The evidence does not justify promoting CRCC to the mainline
  training configuration. No per-scene tuning or default change was made.

Small metric differences are descriptive, not statistically established
improvements. Same source and camera ordering do not eliminate CUDA atomic
nondeterminism. Active CRCC also uses strict safe sigmoid at extreme
endpoints, whereas E0 preserves the original baseline implementation; this
is a disclosed difference beyond the auxiliary loss itself.

## Visual evidence

The recorded inspection covers every final test view via all-arm sheets
in the screen scenes and E0/E1/E2 sheets in both extension scenes, plus
four-scene overviews. Native Japanese MTN_1090, Curasao MTN_1288 and IUI3
MTN_5903 clear images were also inspected. The complete source list and
qualitative findings are in
[visual_review.json](../outputs/crcc_bnd_on_20260911/visual_review.json).
This is agent visual inspection, not a blinded human rating study. There
is no real clear-image ground truth.

| Scene | E0 near-black | E2 near-black | E0 cyan-like | E2 cyan-like |
|---|---:|---:|---:|---:|
| JapaneseGradens | .045373 | .025348 | .044085 | .043613 |
| Panama | .000024 | .000065 | .163916 | .164709 |
| Curasao | .093697 | .078648 | .013113 | .017557 |
| IUI3 | .082957 | .137319 | .142789 | .095037 |

These are mean per-view fractions of native 8-bit raw-clear display pixels:
near-black is max(RGB)<=3/255; cyan-like is (G+B)/2-R>.1 and G,B>.1.
They mix real colors, shadows and artifacts. They are not hole segmentation,
stripe measurements, physical accuracy or clear reconstruction metrics.
In particular, IUI3's smaller cyan fraction accompanies more black pixels.
Definitions, dark-pixel fractions and all per-view values are preserved in
`<out>/visual_proxies.json` and `visual_proxies_per_view.csv`.

## Training scales

The following loss ratios compare weighted CRCC with the existing logged
`L_RGB` (the runner's base loss) on training views. They do not compare to
held-out test MSE. Charbonnier includes its nonzero epsilon floor.

| Scene | Arm | Median CRCC/base | Max CRCC/base | Max DC grad L2 | Max rest grad L2 | Training seconds |
|---|---|---:|---:|---:|---:|---:|
| JapaneseGradens | E2 | .000472 | .000927 | 9.231e-7 | 3.575e-6 | 985.9 |
| JapaneseGradens | E3 | .000464 | .000936 | 9.227e-7 | 3.571e-6 | 1144.6 |
| Panama | E2 | .000763 | .001787 | 1.297e-6 | 5.022e-6 | 1430.7 |
| Panama | E3 | .000754 | .001773 | 1.298e-6 | 5.003e-6 | 1669.9 |
| Curasao | E2 | .000360 | .001403 | 1.191e-6 | 4.612e-6 | 1394.4 |
| IUI3 | E2 | .000411 | .000677 | 1.456e-6 | 5.641e-6 | 925.6 |

The regularizer is numerically weak at this registered coefficient:
median weighted loss is about .036%-.076% of the existing base loss, and
maximum logged ratios stay below .179%. This is an observation, not proof
that increasing lambda would fix the visual defects. CRCC appearance
gradients are nonzero; medium/geometry CRCC gradients and CICA escape remain
zero in all logged training audits. Counts become large and support mostly
saturates because repeated views and inherited lineage count as observations.

Largest logged invalid-transmission sample fraction is .076237 (IUI3 E2),
which includes out-of-image or otherwise invalid center samples; it is not
a NaN frequency. Those samples have zero effective loss weight. No NaN/Inf
or state mismatch was reported. Full timings, populations and gradient
scales for all14 runs are in `<out>/comparisons.json`.

Final held-out-view diagnostics below are scene means, except intrinsic
bounds which are the extrema across views. Counts are means over visible
Gaussians. Quantiles in the full CSV are averages of per-view quantiles,
not pooled-population quantiles. Lambda is .02 for every active arm here.

| Scene | Arm | Raw loss | Weighted loss | Visible | Effective | Initialized ratio | Mean count | Mean support | Mean medium weight |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| JapaneseGradens | E2 | .003946 | .000078923 | 512804.3 | 504568.7 | .999867 | 9328.3 | .999853 | .357849 |
| JapaneseGradens | E3 | .003898 | .000077964 | 512154.7 | 503965.7 | .999866 | 9334.8 | .999855 | .356015 |
| Panama | E2 | .005305 | .000106109 | 576746.3 | 563181.7 | .999961 | 9074.4 | .999960 | .443854 |
| Panama | E3 | .005286 | .000105722 | 577163.0 | 563513.0 | .999958 | 9082.5 | .999957 | .446238 |
| Curasao | E2 | .003157 | .000063145 | 671549.0 | 656830.7 | .999978 | 9888.6 | .999976 | .588674 |
| IUI3 | E2 | .002598 | .000051954 | 394612.5 | 384703.5 | .999905 | 8184.0 | .999895 | .460902 |

| Scene | Arm | Chroma magnitude | EMA residual magnitude | Object residual | Medium residual | Intrinsic min | Intrinsic max | Raw J p99 | tau_D p90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| JapaneseGradens | E2 | .042202 | .007127 | .113124 | .034727 | .000862107 | .999999881 | .848932 | .896999 |
| JapaneseGradens | E3 | .041858 | .007047 | .113258 | .034818 | .001893824 | .999999881 | .845551 | .884043 |
| Panama | E2 | .054873 | .009960 | .059384 | .013775 | .004877942 | .999999881 | .813695 | .860833 |
| Panama | E3 | .055966 | .009918 | .059199 | .013761 | .014288567 | .999999881 | .818257 | .876551 |
| Curasao | E2 | .056921 | .005507 | .091771 | .016306 | 1.424e-8 | .999999881 | .742215 | 1.842217 |
| IUI3 | E2 | .051362 | .004493 | .240505 | .018617 | 4.366e-9 | .999999881 | .869967 | 2.016517 |

Chroma and EMA residual magnitudes are mean RGB-vector L2 norms. Object
residual is mean|rgb_object-GT|; medium residual is
mean|rgb_medium-(GT-rgb_object)|. Neither has separately observed object or
medium ground truth. E0/E1 have zero CRCC loss and inactive history; their
full diagnostics and all count/support/medium-weight p10/p50/p90 values are
in `diagnostics_all_steps_including_CRCC_four_scene.csv`.

## Final audit record

- `pytest tests -q`: 111 passed, 2 dependency warnings, 7.56s.
- `py_compile` and `git diff --check`: passed. The final reporting scripts
  are compiled separately by the final verification script.
- Both real-scene engineering audits passed in `engineering/verified_v3`:
  legacy checkpoint load, exact disabled forward/loss, same-gradient exact
  optimizer update, EMA callback timing, state roundtrip, bounds and routing.
- All 42 training checkpoints passed Gaussian/state/Adam shape checks;
  labelled split/clone/prune lineage was verified by the topology tests.
  Native GT resolution and original pixels, checkpoint hashes,
  source model/optimizer/scheduler state, camera ordering, RNG manifest and
  CICA camera bank pairings passed the main and supplementary audits.
- CRCC gradients update DC/rest only; CICA remains DC-only, OCMC projector
  hashes are unchanged. Optional GRAC/GCRC states are absent in CRCC runs.
- Frozen training/evaluation code: all 16 snapshot hashes unchanged.
- Historical RAOC strict CUDA forward/backward tolerance checks still fail;
  current discrepancies equal the committed reference, so no new regression.
  Independent CUDA optimizer hashes also differ for baseline repeats; only
  the same-gradient disabled optimizer comparison is bit-exact. Neither
  limitation is described as a strict pass.
- All active CRCC Gaussian intrinsic values remain strictly inside (0,1).
  Recorded P(J>1) and P(intrinsic_rgb>1) are zero for every arm. Raw image J
  includes zero-valued uncovered pixels; E0 can also hit intrinsic endpoints.
  Therefore a claim that every pixel of every arm is strictly in (0,1)
  would be false. `clear_rgb_*` describes the existing J/(1+J) display;
  `J_p99` describes the raw compositor output, not that display.
- Supplementary coordinator status was interrupted, but both workers
  completed without restart or retraining. Its stale `queue_status.json`
  is retained, with separate `completion_recovered.json` proving final
  step/update counts, completed asset audit and no remaining workers.

Main artifact audit:36 training checkpoints,456 native PNGs,21 registered
contact-sheet entries. Supplement adds6 unique checkpoints and84 unique
PNGs. The linked E0/E2 four-scene report references existing assets without
duplicating checkpoint or PNG contents. Total:14 unique runs,42 training
checkpoints,540 native PNGs,45 actual JPEG contact-sheet files including
preserved previews/additional comparisons. Engineering roundtrip checkpoints
are separate from the42 training checkpoints.

Authoritative completion evidence is
[final_verification.json](../outputs/crcc_bnd_on_20260911/final_verification.json),
alongside `crcc_artifact_audit.json`, `artifact_integrity_audit.json`,
`supplementary_verification.json` and `engineering/test_results.json`.
Sampled GPU memory maxima and final process state are recorded there;
sampling is not a continuous allocator high-water measurement.
Observed device-memory maxima were GPU6=8998MiB, GPU7=9070MiB,
GPU8=9800MiB and GPU9=10456MiB. No unbounded growth was observed. All
experiment workers and coordinators exited; GPUs6/7/8/9 had no compute
processes and0% utilization at the final check. Final validation checked
1256 full-view/logged records plus2283 transmission-bucket records. The
finalizer was corrected to distinguish pixel-only bucket rows and inactive
CRCC gradient columns; these were reporting schema issues, not training
failures. Frozen training code and artifacts were unchanged.

## Final deliverable index

`<out>` is `outputs/crcc_bnd_on_20260911`. `<extra>` is
`<out>/crcc_only_four_scene_extension`, used only for Curasao/E2 and
IUI3-RedSea/E2. All other runs use `<out>` as their root.

| Deliverable | Location relative to the repository |
|---|---|
| Full precision final metrics | `<out>/metrics_final_including_CRCC_four_scene.csv` |
| Intermediate/final metrics | `<run-root>/<scene>/<arm>/evaluation_metrics.csv` |
| Per-view test metrics | `<run-root>/<scene>/<arm>/per_view_metrics.csv` |
| Checkpoints | `<run-root>/<scene>/<arm>/checkpoints/step-{step:09d}.ckpt` |
| Native underwater | `<run-root>/renders/<arm>/<scene>/step_<step>/<view>/A2/underwater.png` |
| Native clear display | Same directory, `clear.png` |
| Native raw-clear display | Same directory, `clear_raw_display_clamp01.png` |
| CRCC training loss/statistics | `<run-root>/<scene>/<arm>/crcc_training.jsonl` |
| All training-view statistics | `<run-root>/<scene>/<arm>/crcc_statistics_step_{step:05d}.csv` |
| Aggregated test statistics | `<out>/diagnostics_all_steps_including_CRCC_four_scene.csv` |
| Gradient audit | `<run-root>/<scene>/<arm>/gradient_routing_audit.json` |
| Training gradient norms | `<run-root>/<scene>/<arm>/training_summary_A2.csv` |
| Source/config/topology | Per-run `source_config.yml`, `experiment_manifest.json`, `refinement_events.csv` |
| Screen and selected E1 sheets | `<out>/contact_sheets/` |
| Supplementary E2 and E0/E1/E2 sheets | `<out>/crcc_only_four_scene_report/contact_sheets/` |
| Paired comparisons/timings | `<out>/comparisons.json` |
| Visual inspection | `<out>/visual_review.json` |
| Human-readable metrics/proxies appendix | `<out>/results_appendix.md` |
| Final verification and worktree snapshot | `<out>/final_verification.json` |

Representative views:
[underwater four-scene E0/E2](../outputs/crcc_bnd_on_20260911/crcc_only_four_scene_report/contact_sheets/four_scene_E0_vs_E2_underwater.jpg),
[raw clear four-scene E0/E2](../outputs/crcc_bnd_on_20260911/crcc_only_four_scene_report/contact_sheets/four_scene_E0_vs_E2_clear_raw_display_clamp01.jpg),
[IUI3 all-view E0/E1/E2 clear](../outputs/crcc_bnd_on_20260911/crcc_only_four_scene_report/contact_sheets/IUI3-RedSea_E0_vs_E1_vs_E2_clear_raw_display_clamp01.jpg).

## Files changed for CRCC

New implementation/tests:
`water_splatting/crcc.py`, `tests/test_crcc.py`.

Integration into existing files:
`water_splatting/water_splatting.py`, `water_splatting/taba.py`,
`scripts/experiments/run_direct_mdrr_cica_scene.py`,
`scripts/experiments/summarize_taba_bnd_on.py` (optional selector/statistics
arguments preserving existing defaults).

New experiment/report drivers in `scripts/experiments/`:
`run_crcc_bnd_on.py`, `run_crcc_registered_experiment.py`,
`run_crcc_supplementary_extension.py`, `summarize_crcc_bnd_on.py`.

New diagnostics in `scripts/diagnostics/`:
`audit_crcc_training.py`, `audit_crcc_artifacts.py`,
`audit_crcc_supplementary_extension.py`, `verify_crcc_engineering.py`,
`monitor_crcc_progress.py`, `summarize_crcc_visual_proxies.py`,
`compare_crcc_results.py`, `finalize_crcc_verification.py`.

This research note and `<out>` contain the new documentation and assets.
The branch remains `research/bvam-bnd-on` with uncommitted changes. Existing
dirty files, including historical appearance/refactor code and older module
notes/scripts/tests, remain in place and are not attributed wholesale to
CRCC. No commit, push, dependency install, new environment, deletion or
compact operation was performed. Only conda `water_splatting` and physical
GPUs6/7/8/9 were used for this experiment.
