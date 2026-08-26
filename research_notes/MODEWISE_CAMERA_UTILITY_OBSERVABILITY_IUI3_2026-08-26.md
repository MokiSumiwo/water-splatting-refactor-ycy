# MODE-WISE CAMERA UTILITY VS OBSERVABILITY AUDIT

## CODE FACT
The audit reuses the existing OCMC C1 checkpoints and the established 9-D raw-medium basis.
GENERAL rays provide the observability basis; the same basis is then applied to M_SAFE utility counterfactuals.

## CONFIG FACT
Checkpoints: `(5000, 10000, 14999)`.
Source output dir: `outputs/m1_ocmc_causal_iui3_20260825`.
Output dir: `outputs/modewise_camera_utility_observability_iui3_20260826`.

## EXPERIMENTAL FACT
CONDA_ENV: `water_splatting`.
CUDA_VISIBLE_DEVICES: `6`.
GPU: `NVIDIA GeForce RTX 3080`.

## QUANTITATIVE RESULT
Classification: `MIXED_MODE_UTILITY_STRUCTURE`.
M_SAFE median Spearman(sigma, utility): `0.45`.
GENERAL median Spearman(sigma, utility): `0.21666666666666667`.
M_SAFE signal strength: `1.5399078661343996e-06`.
GENERAL signal strength: `4.724820702161604e-06`.

## INFERENCE
This diagnostic only tests whether camera utility tracks observability in a stable mode-wise way.
No training, optimizer step, or projector redesign was performed.
