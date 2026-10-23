# FSB CICA Shared-Prefix Repeat Protocol

Registered before any new endpoint evaluation. No test-based tuning or scene selection.

## Question And Design

Does the exact FSB CICA derivative improve continuation results beyond identical-run
variation, when pre-CICA training cannot drift between C0 and C1?

- Four scenes: Curasao, IUI3-RedSea, JapaneseGradens-RedSea, Panama, on physical
  GPUs 6, 7, 8, 9 respectively. Conda water_splatting only.
- For each scene train one R0 (historical RGB backward) and one R1 (backward-only
  alpha-cap alignment) prefix from the registered step-3000 source to step 9999.
- From each serialized prefix run C0a, C1a, C1b, C0b, in that fixed order.
  C0 is historical CICA derivative; C1 is the actual FSB derivative.
  a/b repeat the same seed/state, measuring continuation execution variability;
  these are not independent random-seed or independent-prefix replicates.
- 8 prefix runs and 32 continuations total. Continuations start at 10000 and
  end at 14999 (5000 updates). No CICA-free training interval is duplicated
  within a renderer/scene block.
- FSB alpha=1, degree scales=1/1/1, SH3, M1+OCMC+CICA. Auxiliary appearance,
  BRASH, BVAM, MDRR, GRAC/GCRC/CRCC disabled. Normal topology callbacks retained.
- Preserve the original 10% training-DC-gradient lambda calibration at 10000;
  refresh targets at 10000/12000/14000. Effects include calibration changes.
- Renderer main effects remain conditional on different learned prefixes;
  this experiment primarily isolates CICA within each fixed renderer.

## Restoration Gate

Save actual Python, NumPy, Torch CPU and CUDA RNG states, model state, optimizer
moments, schedulers, scalers, OCMC bundle and refinement accumulated statistics.
Save appearance LR application state to avoid double adjustment.

Model loading replaces Gaussian Parameter objects, so recreate optimizers
against the loaded parameters before restoring moments and schedulers.
Assert parameter identity and state hashes before the first update.
Require source scene, renderer and FSB parameterization to match and prohibit
forking a checkpoint in which CICA was already active.

A short 3199-to-3201 smoke crosses a densification callback and must complete
with matching restored hashes and live optimizer bindings. Same initial state
does not guarantee bitwise subsequent CUDA reductions; quantify this via a/b.

## Outcomes

- Evaluate every original test view at 14999 (3/4/3/3). Preserve float-derived
  PSNR, SSIM, LPIPS, per-view rows, bound diagnostics and raw-display summaries.
- Report both repetitions, within-condition |a-b| and C1-minus-C0 repeat means,
  per scene and four-scene equal-weight means for each renderer. Do not infer
  statistical significance or population variance from two repetitions.
- A positive cross-scene candidate requires positive mean PSNR change under
  both renderers, at least 3/4 positive scenes under each renderer, no scene
  loss exceeding 0.05 dB, and each renderer's macro gain larger than its
  mean same-condition absolute repeat difference. This is an operational
  screening rule, not a confidence interval; failed gates mean no promotion.
- Report SSIM/LPIPS conflicts regardless of the PSNR rule. Clear images have
  no GT; darkness, hue, and boundedness cannot prove dewatering accuracy.
- All-view comparisons use fixed display scaling, no exposure/color changes.

## Storage And Provenance

Write new assets under /mnt/d/home/ycy/water-splatting-experiments/.
Retain all 8 prefix checkpoints and 32 endpoints. Archive worker/model/CUDA
sources and both binaries with SHA256 before launching. Reject mid-run changes
to the archived source set. Do not modify or delete historical assets.

No dependency installation, GPU 0-5 use, commit or push.
