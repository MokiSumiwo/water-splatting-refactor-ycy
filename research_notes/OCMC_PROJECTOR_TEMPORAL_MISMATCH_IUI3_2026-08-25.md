# OCMC Projector Temporal Mismatch Audit

## CODE FACT
OCMC remains the detached 9-D camera-conditioned medium projector implemented in `water_splatting/fields/medium_field.py` and installed through `water_splatting/water_splatting.py`.
This task does not alter the projector equation or the medium MLP.

## CONFIG FACT
Diagnostic checkpoints: `(8000, 13000, 14999)`.
Projector population: `GENERAL` with the registered GENERAL train-ray bank.
Identity means projector disabled at forward time. Stale means the checkpoint-saved bundle. Fresh means the projector recomputed from the current checkpoint state with the same estimator.

## EXPERIMENTAL FACT
Outputs: `/mnt/new/home_old/ycy/water-splatting-refactor/outputs/ocmc_projector_temporal_mismatch_iui3_20260825`.
Checkpoint mapping rows: `3`.
Per-camera final rows: `50`.
Per-eval-view final rows: `12`.

## QUANTITATIVE RESULT
Identity utility mean: `0.000147`.
Stale utility mean: `0.000058`.
Fresh utility mean: `0.000061`.
Fresh-vs-stale utility gap: `0.000003`.
Final eval fresh-vs-stale delta: `{'PSNR': -0.13374805450439453, 'SSIM': -6.183981895446777e-05, 'LPIPS': 0.00011550262570381165, 'MSE': 1.0632040357450023e-05}`.

## INFERENCE
Candidate classification: `PROJECTOR_FAILURE_CAUSE_NOT_RESOLVED`.
Secondary classification: `OCMC_PROJECTION_PRINCIPLE_NOT_SUPPORTED`.
No training, optimizer step, or projector redesign was performed.
