# FSB Curriculum: Completed Four-Scene Experiment

The prespecified curriculum did not improve strict FSB overall. Retain the
original FSB candidate; do not promote this schedule or launch a schedule
sweep from these test results. This is a completed negative result for one
schedule and seed, not evidence that every possible curriculum must fail.

## Final Results

All four scenes completed step 14999 with exit code zero. Each continuation
performed 11,999 updates from its step-3000 source. Independent checkpoint
reload and evaluation cover all 13 existing test views (3, 4, 3, 3).

| Scene | BND ON+CICA PSNR | FSB PSNR | Curriculum PSNR | Delta vs FSB | Delta SSIM vs FSB | Delta LPIPS vs FSB |
|---|---:|---:|---:|---:|---:|---:|
| Curasao | 32.493907 | 32.496067 | 32.566875 | +0.070808 | +0.000073 | +0.000472 |
| IUI3-RedSea | 30.732962 | 30.713837 | 30.682673 | -0.031163 | +0.000314 | +0.000646 |
| JapaneseGradens-RedSea | 24.796497 | 24.835192 | 24.782654 | -0.052538 | -0.000774 | +0.001832 |
| Panama | 31.501145 | 31.562564 | 31.501119 | -0.061445 | +0.000315 | -0.001082 |
| Macro | 29.881128 | 29.901915 | 29.883330 | -0.018585 | -0.000018 | +0.000467 |

| Method | Macro PSNR | Macro SSIM | Macro LPIPS | Macro MSE |
|---|---:|---:|---:|---:|
| BND ON+CICA | 29.881128 | 0.927658066 | 0.120722171 | 0.002056750 |
| Strict FSB | 29.901915 | 0.927423337 | 0.120233491 | 0.002050886 |
| Curriculum | 29.883330 | 0.927405274 | 0.120700672 | 0.002063692 |

Scene metrics average their own views; macro metrics give each scene equal
weight. Mean view PSNR is not PSNR computed from mean MSE. Curriculum improves
PSNR in 4/13 individual views and only one scene average. Against BND ON+CICA,
its macro PSNR is +0.002203 dB and SSIM is -0.000252792. This does not establish
a meaningful improvement, close the BND OFF gap, or support a SOTA claim.

## Intervention And Checks

The [registered protocol](FSB_CURRICULUM_PROTOCOL_2026-09-15.md) fixes:

```text
u = clamp((absolute_step - 3000) / 5000, 0, 1)
alpha = u*u*(3 - 2*u)
Gaussian_RGB = (1-alpha)*BND_RGB + alpha*FSB_RGB
```

Alpha is 0 at 3000, 0.352 at 5000, and 1 at 8000 and thereafter. Mixing is
performed on bounded per-Gaussian colors before rasterization. The final
color formula is exactly FSB; the experiment changes the optimization path.
SH-degree scales remain (1,1,1), OCMC+CICA stay ON, auxiliary appearance stays
OFF, and CICA starts at 10000. Seed is 42. There is no new loss, rasterizer
change, optimizer reset, or learning-rate change. Normal joint optimization
and refinement can still change geometry, opacity, and medium parameters.

The zero-step audit passed on all four scenes. Three fixed training cameras
per scene verified exact BND and FSB endpoint equality. Appearance gradients
were finite on the first training camera at alpha 0, 0.352, 0.5, and 1. Model
and optimizer hashes remained unchanged. For fixed model parameters, the
schedule-only per-Gaussian RGB change is bounded by 0.0003 per step; this is
not a bound on RGB changes caused by actual training updates.

The MSE-only, read-only Adam prediction at alpha 0 versus 1 changed non-DC SH
update norms by at most 0.0364%, and DC update norms by at most 0.9220%.
The smaller bound applies only to non-DC SH. This limited diagnostic found no
large update-norm jump on sampled views; it does not prove full-objective or
long-term optimizer stability.

The separate short runs passed the registered gross-instability gate:

| Scene | Training-camera mean MSE change vs FSB at 5000 |
|---|---:|
| Curasao | +0.3572% |
| IUI3-RedSea | -1.2693% |
| JapaneseGradens-RedSea | +0.0059% |
| Panama | -1.0161% |

This gate only excluded increases above 25% on three predetermined training
cameras. It did not establish novel-view performance or select the schedule.
The formal runs restarted from the original sources, independently of the
short runs. Checkpoints remain at short-run 5000 and formal 8000 and 14999.
Only the fixed final checkpoint determines the reported comparison.

Paired source checkpoint, model, optimizer, scheduler, OCMC projector, RNG,
and camera-sequence audits all pass. Independent reload reproduces stored
final PSNR/SSIM/LPIPS/MSE within 1e-4 dB/1e-5/1e-5/1e-7. Every native GT image
hash matches across methods. Final curriculum rendering equals constant FSB
exactly when using the same final checkpoint. All evaluated Gaussian colors
are finite and within [0,1]. Seven archived source/protocol snapshots match
their recorded hashes and the current implementation at finalization.

Schedule configuration is saved under checkpoint `metadata.headroom_config`
and validated against the manifest on load. The older top-level
`checkpoint_metadata_has_degree_weights` flag in the reload JSON does not
inspect this nested record; its false value does not mean the new curriculum
checkpoints lack their degree scales or schedule.

## Visual Findings

Native GT, underwater, and clear PNGs are retained. Contact sheets cover every
test view and include absolute curriculum-FSB differences amplified by 4.
Fixed center crops use the same center 40% rectangle for every method.
There is no per-image exposure adjustment or normalization.

The clear panels and black-area statistic use the existing renderer output
`rgb_clear = J/(J+1)`, not untonemapped intrinsic radiance J. This fixed display
mapping is identical across all methods. The observed curriculum clear
channel range is approximately [0, 0.499975], while per-Gaussian colors remain
within [0,1]. The tone-mapped clear range must not be reported as a raw
intrinsic-radiance bound.

At contact-sheet scale, underwater differences are small and localized to
surface details. Clear changes are more visible at distant surfaces and
background boundaries. Curasao's distant boundary changes, but the existing
upper streaks persist. IUI3 shows a larger near-black region with a changed
blue-to-black boundary. JapaneseGradens and Panama retain distant streaks
and background artifacts. Visual inspection does not show consistent
four-scene removal of black areas or streaks.

| Scene | FSB clear black fraction | Curriculum clear black fraction |
|---|---:|---:|
| Curasao | 11.4204% | 10.7014% |
| IUI3-RedSea | 14.9268% | 18.7397% |
| JapaneseGradens-RedSea | 4.7804% | 5.1380% |
| Panama | 0.0067% | 0.0096% |
| Macro | 7.7836% | 8.6472% |

Black means channel-mean displayed RGB below 0.02. Macro pixel coverage
`accumulation > 0.01` changes from 95.1356% to 94.8256%. These quantities
include open-water/background regions and are not object-hole masks. There
is no clear GT, so neither black-area reduction nor increased coverage would
prove more accurate dewatering. This run provides no clear-quality reason
to override its failed underwater metric gates.

Measured warm-render macro latency is 22.268 ms for FSB and 22.247 ms for
curriculum. Mean recorded peak allocated CUDA memory is 2129.3 and 2127.2 MB.
These sequential evaluation measurements include resident model and cached
tensors; they are not a controlled speed or memory benchmark.

## Decision And Next Priority

Three registered metric gates fail: macro PSNR does not reach FSB, macro SSIM
does not reach BND ON+CICA, and macro LPIPS exceeds FSB. The per-scene PSNR
loss limit of 0.10 dB passes, but this alone is insufficient for promotion.
Retain original FSB and record curriculum as not promoted.

The combination of endpoint equality, modest sampled update-norm changes,
and unsuccessful final results gives no positive evidence that a gradual
BND-to-FSB switch fixes the current performance limit. It does not isolate
the cause of the BND ON/OFF gap: optimization, capacity, and the coupled
geometry/medium decomposition remain possible contributors.

The next useful priority is paired repeat-seed validation of original FSB
against BND ON+CICA before adding another FSB variant. Its current macro
PSNR advantage is only 0.020787 dB and its macro SSIM is slightly lower.
Continuation repeats from a shared checkpoint measure continuation
variability; full-training repeats are required to measure end-to-end
variability. Any future schedule selection needs a genuine validation split
and fresh paired source models excluding those validation views from fitting.
The existing repeatedly inspected test set cannot become a sealed benchmark
by relabeling it. No additional schedule or seed runs were launched here.

## Artifacts

- [Full metrics, runtime, per-scene sheets and crop links](../outputs/fsb_curriculum_four_scene_20260915/comparison/summary/REPORT.md)
- [All 13 underwater views](../outputs/fsb_curriculum_four_scene_20260915/comparison/summary/underwater_all_views.jpg)
- [All 13 clear views](../outputs/fsb_curriculum_four_scene_20260915/comparison/summary/clear_all_views.jpg)
- [Per-view metric deltas](../outputs/fsb_curriculum_four_scene_20260915/comparison/summary/per_view_deltas.csv)
- [Recorded promotion decision](../outputs/fsb_curriculum_four_scene_20260915/comparison/summary/decision.json)
- [Source provenance](../outputs/fsb_curriculum_four_scene_20260915/provenance/manifest.json)
- [Step-8000 schedule metadata audit](../outputs/fsb_curriculum_four_scene_20260915/provenance/checkpoint_schedule_audit.json)

Zero-step artifacts: `outputs/fsb_curriculum_zero_20260915_v2`.
Short-run artifacts: `outputs/fsb_curriculum_short_20260915` and
`outputs/fsb_curriculum_short_audit_20260915`. The initial zero-step attempt
used a no-gradient inference wrapper for its gradient check; its failed
directory is retained and is not counted as evidence against the mechanism.
The successful audit uses the differentiable model path.

Only conda `water_splatting` and physical GPUs 6-9 were used. Historical
assets were preserved; there was no dependency installation or commit/push.
All experiment and evaluation processes have finished.
