# M1 RAOC Causal Four-Scene Experiment

Date: 2026-08-27
Branch: `research/m1-bounded-intrinsic`
Experiment: `M1-RAOC-CAUSAL-FOUR-SCENE`

## Registered protocol

The formal comparison is C0=OCMC versus C1=RAOC under the registered bounded-SH3, BND-from-scratch, seed-42 configuration. The only intended treatment difference is medium-capacity allocation. The formal run uses 15,000 training steps, refreshes at steps `0`, `5000`, and `10000`, and evaluates checkpoints at steps `3000`, `5000`, `8000`, `10000`, `13000`, and `14999`.

The four scenes and physical GPUs are fixed as follows:

| Scene | Physical GPU |
| --- | ---: |
| Curasao | 6 |
| IUI3-RedSea | 7 |
| JapaneseGradens-RedSea | 8 |
| Panama | 9 |

Each worker exposes exactly one assigned GPU as logical `cuda:0`. The calibration bank is train-only, capped at 25 cameras and 1024 rays per camera. The camera sequence, seed-42 start state, and step-0 basis/state are audited for equality between branches. IUI3 reuses the previously locked `M_SAFE` population diagnostically; all other scenes use `GENERAL` as the formal population.

RAOC uses the registered evidence quantity `e_i,p = |a_i,p| * s_i,p`, train-only per-mode median scales, and the registered composition `g_keep = 1 - (1 - g_obs) * (1 - g_local)`. Capacity diagnostics use the same-state `C1_same_state_OCMC` counterfactual.

## Validation smoke

A complete Curasao `--max-steps 3` run completed before the formal run. It trained both branches, saved and reloaded checkpoints at steps 0 and 2, evaluated train/eval views, ran decomposition and mechanism diagnostics, and produced all required CSV/JSON artifacts. The smoke passed:

- `START_STATE_EQUIVALENCE=true`
- `STEP0_BASIS_EQUIVALENCE=true`
- camera sequence match with 3 steps
- decomposition safety for all 8 branch/split/checkpoint rows
- non-degenerate RAOC allocation and high-evidence rescue greater than low-evidence rescue
- context utility pair, aggregate, and causal-delta rows

The smoke is validation-only and is not a formal result. Its temporary output is outside the repository formal output directory.

During smoke validation, two existing diagnostic helpers exposed an out-of-range deterministic quantile subsample index; both now clamp the integer index to the valid flattened range. The RAOC diagnostic also now places the C1 global gate on the active device before constructing the same-state OCMC counterfactual, and casts the weak-mode vector to the control-array dtype for stratified energy reporting.

## Formal execution and recovery audit

The first formal launch is preserved under `outputs/m1_raoc_causal_four_scene_20260827_attempt1_oom/` and is excluded from all formal statistics. IUI3 C1 failed with a requested 1.04 GiB allocation while only 411.69 MiB was free, and JapaneseGradens C1 failed with a requested 816 MiB allocation while only 793.69 MiB was free. The allocation came from the analytic medium-Jacobian action workspace. Its default chunk was reduced from `16384` to `4096` and a positive-chunk guard was added. This changes only workspace partitioning, not the Jacobian, RAOC equations, samples, refreshes, or training protocol. The complete Curasao smoke passed again before relaunch.

The retained formal run is `outputs/m1_raoc_causal_four_scene_20260827/`. All four workers used Python 3.8.20, Torch 2.1.2+cu118, one logical `cuda:0`, and exactly their assigned physical RTX 3080 GPU. Physical GPUs 0-5 were not exposed. Each C0 and C1 arm produced checkpoints at `3000`, `5000`, `8000`, `10000`, `13000`, and `14999`.

IUI3 completed training, evaluation, and decomposition before its mechanism diagnostic raised `IndexError: index 1024 is out of bounds for dimension 0 with size 1024`. `_analysis_general` had passed unsorted sampled image indices as the sorted gradient-union lookup table; image and ground-truth indexing was valid, but `grad_union` row lookup was not. The runner now constructs a sorted unique gradient union while retaining the original pixel indices for image rows. A `--postprocess-only` path reconstructed the locked calibration samples from `calibration_bank.json`, reused all existing checkpoints/evaluation artifacts, and reran only mechanism diagnostics and summary on GPU 7. No IUI3 training or evaluation was repeated. JapaneseGradens diagnostics were also deterministically replayed from its completed checkpoints and calibration bank on GPU 8; this replay was read-only and did not repeat training or evaluation.

## Formal results

| Scene | Classification | Utility delta C1-C0 | PSNR delta (dB) | SSIM delta | LPIPS delta | High/low rescue |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Curasao | `RAOC_UTILITY_RECOVERY_NOT_SUPPORTED` | 0.000307816 | -0.312269 | -0.000543 | +0.002475 | 0.964872 / 0.065258 |
| IUI3-RedSea | `RAOC_CAPACITY_REALLOCATION_SUPPORTED` | 0.000407326 | +0.139137 | +0.000070 | -0.000800 | 0.970112 / 0.001012 |
| JapaneseGradens-RedSea | `RAOC_CAPACITY_REALLOCATION_SUPPORTED` | 0.000017892 | +0.170146 | +0.002879 | -0.000347 | 0.887315 / 0.050909 |
| Panama | `RAOC_UTILITY_RECOVERY_NOT_SUPPORTED` | 0.000072903 | -0.206036 | -0.001672 | +0.001280 | 0.942053 / 0.213042 |

RAOC recovered positive final correct-context utility in all four scenes and high-evidence rescue exceeded low-evidence rescue in all four, with no over-rescue scene. The final C1 weak-capacity kept/full ratios were `0.910902` (Curasao GENERAL), `0.811999` (IUI3 GENERAL), `0.342137` (IUI3 locked M_SAFE), `0.726517` (JapaneseGradens GENERAL), and `0.875613` (Panama GENERAL). OCMC suppression therefore remained measurable, especially on IUI3 M_SAFE, but RAOC restored substantial capacity in every scene.

The aggregate classification is `RAOC_MULTI_SCENE_TENTATIVE`; RGB is `RGB_MULTI_SCENE_MIXED`. Mean scene-level PSNR delta was `-0.052255 dB` and median was `-0.033449 dB`, with two positive scenes. Mean SSIM delta was `+0.000183`, mean LPIPS delta was `+0.000652`, and mean MSE delta was `-0.00007784`.

The preregistered strong-support criteria failed in exactly two places: only two scenes had a supported/RGB-mixed mechanism classification, below the required three, and mean scene-level PSNR delta was negative. All other registered criteria passed: all four scenes completed, causal validity held, utility was positive in at least three scenes, selective rescue held in at least three scenes, there was at most one over-rescue scene, decomposition was safe in every scene, and at least two scenes improved PSNR. The criteria were not changed after observing results.

The final protocol audit reports `protocol_complete_all_scenes=true`. Every scene passed start-state and step-0 basis equivalence, 15K camera-sequence matching, all 24 branch/step/split decomposition rows, finite weak-capacity ratios, complete `C1_same_state_OCMC` counterfactual coverage, and context `pair`, `aggregate`, and `causal_delta` coverage. `M_SAFE` appears only for IUI3; the other three scenes contain GENERAL diagnostics only.
