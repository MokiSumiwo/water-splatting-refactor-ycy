# SELECTED-MODE UTILITY / RECONSTRUCTION ALIGNMENT AUDIT

## CODE FACT
This task reused the frozen selected mode_01, the previous held-out GENERAL bank, and the previous swap bank.
No training, no optimizer step, and no mode reselection were performed.

## CONFIG FACT
Selection source: `/mnt/new/home_old/ycy/water-splatting-refactor/outputs/m1_ocmc_causal_iui3_20260825/checkpoints/C1/step-000014999.ckpt`.
Previous held-out bank: `outputs/heldout_single_mode_camera_utility_iui3_20260826`.
Output dir: `outputs/selected_mode_utility_reconstruction_alignment_iui3_20260826`.

## EXPERIMENTAL FACT
CONDA_ENV: `water_splatting`.
CUDA_VISIBLE_DEVICES: `6`.
GPU: `NVIDIA GeForce RTX 3080`.
Frozen mode match: `True`.
Bank reuse strict overlap: `True`.

## QUANTITATIVE RESULT
GENERAL mean C_utility: `1.063521094890163e-05`.
GENERAL median C_utility: `0.0`.
GENERAL mean C_rgb: `8.682653807703944e-06`.
GENERAL median C_rgb: `0.0`.
GENERAL Q1/Q2: `0.4260546875` / `0.0512890625`.
GENERAL Spearman(A, C_rgb): `0.11358866946816908`.
GENERAL eval delta PSNR: `0.08856582641601562`.
GENERAL eval delta LPIPS: `0.00036728382110595703`.

## INFERENCE
Primary classification: `SPARSE_CONTEXT_DEPENDENT_SUPPORT`.
Global-gating classification: `GLOBAL_MODE_UTILITY_GATING_NOT_SUPPORTED`.
Reason: Alignment exists, but it concentrates in lower-depth/lower-tau/high-activation contexts and eval remains mixed.
Next task: `Design a ray/context-adaptive capacity-allocation preflight.`.
