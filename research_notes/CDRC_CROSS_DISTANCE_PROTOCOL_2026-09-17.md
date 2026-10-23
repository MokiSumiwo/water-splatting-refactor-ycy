# CDRC Cross-Distance Correspondence Diagnostic Protocol

This is a frozen, read-only diagnostic for the first stage of Cross-Distance
Radiance Calibration (CDRC). It does not train, change a checkpoint, write a
model state, or claim clear-image restoration accuracy.

## Question

Do the original underwater training images contain reliable correspondences
for the same surface location at materially different optical distances, and
does a simple shared-radiance transport model predict a held-out training view
better than the current renderer or copying a source pixel?

## Inputs

* FSB+CICA `step-14999` checkpoints from
  `outputs/fsb_cica_fork_repeat_20260916_v1/<scene>/R0/C0a`.
* Existing frozen maps and camera metadata from
  `outputs/fsb_far_field_diagnostic_20260916_v1`.
* Original `images/ColorImage` underwater PNGs and the registered train lists.

The existing maps are used only as model-dependent geometry, transmission, and
medium estimates. SIFT and patch tests use the underwater images only; clear
renders are never used to select a match.

## Correspondence tests

For every train-train image pair, the script records:

1. raw SIFT matches;
2. Lowe ratio and mutual matches;
3. matches whose target depth can be projected into a source with
   `A >= 0.9`, `depth CV <= 0.1`, relative depth error <= 5%, and a two-pixel
   round trip;
4. matches with local grayscale ZNCC >= 0.50;
5. the subset with a depth ratio >= 1.25 (cross-distance support).

The pair loop is deterministic. Pair rows are retained even when no match
survives. Surface support is reported separately from sparse keypoint support;
neither is a semantic object mask.

## Leave-one-view prediction

For every validated directed match, the target image is withheld. The source
observation is used to estimate a shared intrinsic color with the frozen
source transmission and additive medium:

`J_hat = (I_source - M_source) / max(t_source, 0.10)`

and the target is predicted as `t_target * J_hat + M_target`. Values are
clipped only for error reporting. The following controls are reported on the
same target pixels:

* current frozen renderer prediction;
* source underwater pixel copy;
* calibrated shared-radiance prediction.

This is a local predictive diagnostic, not an optimization objective. A gain on
training images does not imply a gain on held-out test GT, and no clear GT is
available.

## Admission rule for training

CDRC may proceed to a small paired training experiment only if at least three
scenes have substantial cross-distance support, the support reaches the
far-field candidate regions rather than only high-texture near regions, and
the calibrated prediction improves over both controls after aggregation by
scene and distance-gap bucket. If correspondence support is sparse or the
calibrated prediction is no better than controls, CDRC is rejected as a
training mechanism; LSPR/MST must not reuse the same weak evidence as if it
were independent supervision.

## Limitations

The model depth, transmission, and medium are not ground truth. A shared
intercept is only a diagnostic approximation to view-dependent medium. SIFT
can still match repeated texture, and model reprojection can accept common
geometry errors. Results therefore establish evidence for or against
actionability, not physical parameter correctness.

## Pre-run corrections and fixed decision thresholds (2026-09-17)

Before executing either smoke or full statistics, code review found the following
necessary corrections. These supersede the approximate formula and match counters above.

* The verified split is **75 train / 13 eval**: Curasao 18/3, IUI3 25/4,
  JapaneseGradens 17/3, Panama 15/3. Lists are parsed with `splitlines()`, and
  their sets must exactly match the frozen audit metadata.
* Frozen `t = sum(w_i t_i) / A` is a normalized contribution transmission.
  For one shared surface radiance, the coefficient is `S = sum(w_i t_i) = A*t`.
  The primary bounded prediction is `M_target + S_target * clip(J_hat,0,1)` with
  `J_hat=(I_source-M_source)/max(S_source,0.10)`. The old t-only approximation,
  unbounded S-based inverse, and multi-source bounded least squares are controls.
  This is still a **single shared-radiance approximation**, not an exact inversion
  of a multilayer, direction-dependent Gaussian scene. Color floor activation,
  J outside [0,1], and output clipping are reported.
* Both directions must pass Lowe ratio and nearest-neighbor mutuality. Distinct
  SIFT orientations at an identical location are deduplicated. Both directions
  of model depth and feature reprojection must pass. Transport is sampled at
  measured SIFT feature locations, not at a different model-projected location.
* The primary subset predicts **far targets from closer training views**, using
  the existing scene depth boundaries. All-cross and other depth bands are
  secondary. Each target keypoint is counted once per scope. Nearest source is
  selected by depth, never by RGB error; all valid closer sources supply the
  secondary least-squares control. Depth means camera-z in renderer coordinates,
  not calibrated metric optical path length. Actual transmission advantage is
  evaluated independently.
* Sparse feature coverage is supported/candidate keypoints. Spatial coverage is
  supported/candidate stride-4 cells, with identical denominators and masks.
  Sparse feature coverage is not dense surface coverage or semantic object recall.
* Native underwater images are sampled bilinearly. Frozen renderer/transport
  maps are stride 4; their interpolation is a limitation. Report target/source
  image-vs-grid interpolation error, baseline grid-target error, and a same-grid
  source/target transport control. Checkpoint and all inputs are hashed before
  and after, and image size, sampled GT, `S=A*t`, and `pred=clip(obj+M)` are checked.
* Eval-target/train-source matches are an additional diagnostic over **all 13
  test images**. Eval RGB never supplies source radiance or affects thresholds.
  Feature/ZNCC selection does inspect target RGB, so selected-pixel errors are
  neither whole-image PSNR nor an unbiased evaluation of all pixels. The frozen
  model has already seen training targets; training leave-one-source fits are
  not leave-one-view-out retraining.

Scene admission (fixed before first result): at least 100 unique training target
keypoints with a closer source, at least 50 far targets across at least 5 views,
at least 2% of far candidate keypoints supported; median source transmission
advantage >=0.03. Primary bounded CDRC must beat renderer and source copy in both
pooled RMSE and mean per-target-view RMSE, and in each distance-ratio bucket
[1.25,1.5) / >=1.5 containing >=20 points. At least 3/4 scenes must pass for any
training. These are practical screening thresholds, not evidence of novelty or
statistical significance. View-cluster bootstrap intervals are descriptive,
especially with only 3-4 evaluation views. No threshold relaxation after failure.

CPU-only image/matrix analysis runs inside conda `water_splatting`. Each scene's
visibility is restricted to its assigned physical GPU 6/7/8/9, but no CUDA context,
checkpoint load, renderer invocation, backward pass, or optimizer is needed.
Outputs use new directories and refuse to replace an existing scene directory.

### Native-sampling amendment after smoke; before full run

The six-train/one-eval Curasao smoke exposed severe stride-4 interpolation error
at textured keypoints. It can reverse the renderer-vs-CDRC ranking. Both smoke
outputs are retained, but neither is a formal scene result. Formal analysis
therefore requires `--native`: load the same frozen checkpoint under `no_grad`,
render native images and a unit-Gaussian-color coefficient call, and sample all
geometry, transmission, medium, and renderer colors at the exact same feature
positions. Store only feature arrays, with native replay checks against all
existing stride-4 maps and model/ checkpoint hash equality before and after.
The actual invocation uses GPUs 6/7/8/9. This supersedes the CPU-only description;
there are still no backward passes or model updates. Spatial support backgrounds
may remain stride 4. The stride-4 transport and renderer controls use entirely
stride-4 source/target coefficients and observations, and remain sensitivity
checks rather than primary metrics. No admission threshold is changed.
