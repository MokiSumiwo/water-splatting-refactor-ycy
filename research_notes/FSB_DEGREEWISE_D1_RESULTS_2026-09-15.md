# Degree-wise FSB D1: Completed Four-Scene Experiment

This completes the pending D1 experiment described in
[the diagnostic note](FSB_DEGREEWISE_DIAGNOSTIC_2026-09-15.md).
The completed run is `outputs/fsb_degree_d1_four_scene_strict_20260915_v2`.
The interrupted v1 remains historical and is not used for conclusions.

## Results

All scenes finished step 14999 with exit code zero. All 13 test views were
independently re-rendered from each method's checkpoint. Native GT PNG hashes
match across all three methods at every camera. Reloaded PSNR, SSIM, LPIPS,
and MSE reproduce their original evaluation within the audit tolerances
(1e-4 dB, 1e-5, 1e-5, 1e-7 respectively).

| Scene | BND ON+CICA PSNR | FSB PSNR | D1 PSNR | D1-FSB PSNR | D1-FSB SSIM | D1-FSB LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| Curasao | 32.493907 | 32.496067 | 32.429468 | -0.066599 | +0.000008 | +0.001009 |
| IUI3-RedSea | 30.732962 | 30.713837 | 30.666403 | -0.047434 | +0.000115 | +0.000753 |
| JapaneseGradens-RedSea | 24.796497 | 24.835192 | 24.814587 | -0.020605 | +0.000417 | +0.000454 |
| Panama | 31.501145 | 31.562564 | 31.469561 | -0.093003 | +0.000203 | -0.000832 |
| Macro | 29.881128 | 29.901915 | 29.845005 | -0.056910 | +0.000186 | +0.000346 |

| Macro metric | BND ON+CICA | FSB | D1 |
|---|---:|---:|---:|
| SSIM | 0.927658066 | 0.927423337 | 0.927609246 |
| LPIPS | 0.120722171 | 0.120233491 | 0.120579619 |
| MSE | 0.002056750 | 0.002050886 | 0.002066353 |

Each scene receives equal weight, with metrics averaged over its test views.
The mean of view PSNRs is not PSNR computed from mean MSE. D1 improves PSNR
on 4/13 individual views, but loses to FSB in every scene average. Relative
to BND ON+CICA its macro PSNR is -0.036123 dB and SSIM is -0.000048820.
These are single-seed observations, not a statistical significance claim.

## What Was Actually Tested

D1 computes `headroom(c0, r1 + 0.75*r2 + 0.50*r3)`. This is fixed SH
coefficient scaling before the nonlinear map. It is NOT a degree-specific
convex RGB mixture of BND and FSB. The earlier description as a new
degree-wise bounded capacity mechanism was too strong.

Since SH is linear and all three scales are positive, trainable coefficients
can absorb the inverse scales. The set of representable colors is unchanged;
initialization, gradients, and optimization trajectories change. A CUDA
forward/VJP audit on up to 4096 actual D1 Gaussians per scene verified:

- Degree-sum output equals FSB on scaled coefficients: max error 5.96e-8.
- VJP agreement: max error 2.24e-8.
- Inverse coefficient scaling recovers unscaled FSB: max error 5.96e-8.
- Zero residual preserves DC exactly; sampled RGB remains within [0,1].

Measured final raw coefficient RMS in degree 3 grows to 1.098-1.156 times
FSB, while the effective degree-3 RMS is 0.549-0.578 times FSB. Degree-2
effective RMS is 0.814-0.852 times FSB. These are population statistics;
Gaussian topology differs after normal joint training and is not matched
point-by-point. The measurements support a changed frequency distribution,
not a new identifiable appearance model or proof of causation for every
image difference. Three deterministic training cameras additionally record
per-degree SH increments and cumulative headroom utilization.

## Visual Findings

The exports retain native underwater, clear, and GT images, plus absolute
D1-FSB differences x4. All contact sheets use the same display mapping;
center crops use the same fixed center 40% rectangle. No exposure adjustment,
per-image normalization, or test-based crop selection is applied.

Underwater differences are relatively small at contact-sheet scale. Clear
differences are more visible at distant surfaces and background boundaries,
including IUI3 blue regions and the upper Panama/Curasao regions. Existing
background streaks remain visible. Visual inspection does not show a
consistent four-scene removal of black areas or streaks.

Macro clear black fraction (mean RGB below 0.02) changes from 7.7836% for FSB
to 6.2569% for D1, and coverage A>0.01 from 95.1356% to 96.4653%. These
descriptive changes coexist with worse underwater PSNR; no clear GT exists,
so they do not establish better dewatering accuracy or repaired geometry.

Warm render mean latency was about 22.20 ms for FSB and 23.86 ms for D1.
The evaluator records process peak allocated CUDA memory and per-view
timings, not a controlled performance benchmark. All individual results
and measurement definitions are in the generated report.

## Protocol And Decision

- Environment: existing conda `water_splatting`, physical GPUs 6-9 only;
  no installation, commit, push, or historical asset replacement this run.
- Paired start model, optimizer, scheduler, OCMC projector, continuation RNG,
  camera sequence and source checkpoint hashes all match across methods.
- OCMC+CICA ON, auxiliary appearance OFF, seed 42, normal refinement.
  Geometry/opacity/medium are jointly trained under the existing objective;
  they are not frozen. Only the appearance mapping is intervened on.
- Older checkpoints lack fixed degree weights in their metadata. The
  independent evaluator explicitly loads settings from the recorded manifest
  and saves checkpoint/manifest hashes. Use that manifest for future reloads.
- D1's 5k smoke checked engineering correctness, not a validation metric.
  The milder D1 configuration was chosen before observing 15k test results.
  D2 remains smoke-only; this experiment does not establish that D2 failed.
- D1 does not meet the joint PSNR/SSIM objective. Do not promote D1 or start
  D2 15k solely because smoke passed. Retain strict FSB as the candidate.

The next priority is FSB curriculum, beginning with a fixed schedule and
training-view-only zero-step checks of RGB continuity, gradient magnitude,
and optimizer-state effects during the transition. A genuine validation
protocol is needed before choosing a schedule or checkpoint; using already
fitted training views cannot certify novel-view generalization. Avoid
another full sweep driven by the repeatedly observed test set.

## Artifacts

- [Full metrics, audits, per-scene links and runtime table](../outputs/fsb_degree_d1_four_scene_strict_20260915_v2/comparison/summary/REPORT.md)
- [All 13 underwater views](../outputs/fsb_degree_d1_four_scene_strict_20260915_v2/comparison/summary/underwater_all_views.jpg)
- [All 13 clear views](../outputs/fsb_degree_d1_four_scene_strict_20260915_v2/comparison/summary/clear_all_views.jpg)
- [Per-view metric differences](../outputs/fsb_degree_d1_four_scene_strict_20260915_v2/comparison/summary/per_view_deltas.csv)
- [SH degree diagnostics](../outputs/fsb_degree_d1_four_scene_strict_20260915_v2/comparison/summary/degree_diagnostics.csv)

Reproduction tools: `scripts/experiments/run_fsb_bnd_on_four_scene_strict.py`
with `--max-step 14999 --degreewise 1,0.75,0.5` and a fresh output directory;
`scripts/diagnostics/evaluate_fsb_degreewise_scene.py` for independent scene
evaluation; `scripts/diagnostics/summarize_fsb_degreewise.py` for aggregation.
