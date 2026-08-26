# HELD-OUT SINGLE-MODE CAMERA UTILITY VALIDATION

## CODE FACT
The selected mode is frozen from the archived 14999 GENERAL mode-wise audit.
The held-out evaluation uses the current formal C1 checkpoint and a disjoint deterministic ray bank.

## CONFIG FACT
Selection source: `outputs/modewise_camera_utility_observability_iui3_20260826`.
Source checkpoint: `outputs/m1_ocmc_causal_iui3_20260825/checkpoints/C1/step-000014999.ckpt`.
Output dir: `outputs/heldout_single_mode_camera_utility_iui3_20260826`.

## EXPERIMENTAL FACT
CONDA_ENV: `water_splatting`.
CUDA_VISIBLE_DEVICES: `6`.
GPU: `NVIDIA GeForce RTX 3080`.
Selected mode: `mode_01`.
Frozen selection label: `SINGLE_MODE_CONTEXT_UTILITY_TENTATIVE`.

## QUANTITATIVE RESULT
GENERAL C_utility mean: `1.0635209946485702e-05`.
GENERAL C_utility median: `0.0`.
GENERAL camera-positive fraction: `0.88`.
GENERAL relative utility contribution: `0.10192230427231566`.
GENERAL utility-per-energy: `2.6762403760932832e-05`.
M_SAFE C_utility mean: `4.468242593702598e-07`.
Eval PSNR full: `30.793774604797363`.
Eval PSNR minus: `30.88234043121338`.

## INFERENCE
The task is read-only. Phase B is only justified if the frozen single-mode utility remains positive and not sample-specific on the held-out bank.
