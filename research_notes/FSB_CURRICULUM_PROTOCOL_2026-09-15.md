# FSB curriculum: prespecified experiment

One candidate only, registered before training/evaluation. BND ON+CICA and
strict FSB are existing controls. Fixed seed 42 and identical scene-specific
step-3000 source model/optimizer/scheduler/projector/camera sequence.

`u=clamp((step-3000)/5000,0,1); alpha=u*u*(3-2*u)`.
`RGB=(1-alpha)*BND_RGB + alpha*FSB_RGB`.

Alpha is 0 at step 3000, 0.352 at 5000, 1 at 8000 and thereafter. It depends
only on absolute step, equally in train and eval. Final rendering is exactly
the existing FSB formula. SH-degree scales remain (1,1,1). CICA starts at
10000 as before. No optimizer reset, LR change, additional loss or renderer
modification. Geometry/opacity/medium remain jointly optimized as before.

Stage 1: frozen train-camera-only zero-step endpoint, continuity, gradient,
Adam-momentum alignment and state-immutability audit. Require finite output
and appearance gradients, bounded Gaussian RGB, BND endpoint equality at
3000, FSB endpoint equality at 8000, continuous deterministic schedule, and
unchanged model/optimizer hashes. No optimizer step in this audit.

Stage 2: four-scene short continuation to step 5000; checkpoint/reload/route
checks, finite losses, source pairing, alpha log verification. On fixed
training-camera indices [0, middle, last], compare against strict FSB@5000.
If any scene has >25% mean training MSE increase, stop before formal training.
This is only a gross-instability gate, not a validation/generalization result.
No schedule tuning from either training or test-view metrics.

Stage 3: if engineering and short stability checks pass, one prespecified
four-scene continuation from the original source to 14999. Keep checkpoints
at 8000 and 14999, plus the separate short-run 5000 checkpoints, because the
workspace has limited remaining disk capacity. Do not delete old assets.
Compare the fixed final checkpoint on all 13 test views; no best-checkpoint
selection. Archive model/optimizer/camera/gradient/config audits, hashes,
per-view and macro PSNR/SSIM/LPIPS/MSE, underwater/clear/difference images.

Promotion requires macro PSNR >= strict FSB, macro SSIM >= BND ON+CICA,
macro LPIPS no worse than strict FSB, no scene PSNR drop >0.10 dB vs strict
FSB, and no clear visible regression. If this gate fails retain FSB; do not
launch a schedule sweep. Single-seed results remain exploratory, not SOTA
or evidence of significance. Clear has no GT and its proxy changes cannot
prove reconstruction accuracy. Repeated test-set research is not a sealed
benchmark; genuine validation is required before future schedule tuning.

Use only existing conda water_splatting and physical GPUs 6-9. No dependency
installation, commit/push, messages to others, or overwrite of prior outputs.
