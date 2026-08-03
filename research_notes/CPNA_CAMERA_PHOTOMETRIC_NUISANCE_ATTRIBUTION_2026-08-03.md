# CPNA Camera Photometric Nuisance Attribution

Date: 2026-08-03

## Objective

After BLMF was stopped, this audit tests whether M1's remaining error is better explained by camera pose issues or by global exposure / white-balance nuisance factors.

CPNA is a frozen-checkpoint audit only. It does not train a new model and does not implement CPNF. The goal is to decide whether a later camera optimizer or photometric nuisance module has mechanism support.

M1 checkpoints remain:

- `medium_context_mode=dir_xy_camera`
- `b_inf_mode=tied`
- `B_inf = medium_rgb = A`
- No Gaussian refinement, pseudo-depth, GIVAR, BLMF, or photometric training module is enabled.

## Code Changes

- Added `scripts/diagnostics/audit_camera_data_consistency.py`
  - Audits train/eval camera-image mappings, image sizes, intrinsics, camera centers, view directions, distortion params, duplicate filenames, split overlap, and camera optimizer state.
- Added `scripts/diagnostics/diagnose_cpna_pose_oracle.py`
  - Runs frozen small-SE(3) pose oracle for eval views.
  - Tests P0 identity, P1 rotation-only, P2 translation-only, and P3 rotation+translation.
  - Keeps model parameters frozen and records best-so-far pose within the local optimization bound.
- Added `scripts/diagnostics/diagnose_cpna_photometric_oracle.py`
  - Fits per-image global exposure, white balance, and exposure+white-balance oracles on frozen predictions.
  - Uses approximate linear RGB, no additive bias, no gamma parameter, and no spatial map.
  - Reports whole-image metrics, object/water region L1 deltas, fitted gains, and train-view leave-one-out pose-KNN predictability.
- Added default-off deterministic replay logging in `water_splatting/water_splatting.py`
  - `deterministic_audit_log_path: Optional[str] = None`
  - `deterministic_audit_log_every: int = 100`
  - Writes JSONL loss, gradient, refinement, mask-hash, cull, and opacity reset events for future strict R0/R1/R2 replay.
- Updated `scripts/experiments/backscatter_consistent_binf_iui3_redsea.sh` to pass deterministic audit flags.
- Corrected `research_notes/MFRS_MEDIUM_FREQUENCY_RESPONSIBILITY_2026-08-01.md` wording: A0 vs A1 is an observed from-scratch trajectory discrepancy, not a strict noise lower bound.

## Phase -1R Status

The requested strict deterministic replay still requires a shared JapaneseGradens `step-000003000.ckpt`. No such checkpoint exists in the current `outputs/` tree.

This round adds the missing replay instrumentation so the next strict run can compare:

- `camera_index`
- training loss
- Gaussian count
- active SH degree
- xys gradient mean / p95
- high-gradient count
- split / duplicate / cull counts
- opacity reset events
- SHA1 hashes for high-grad, split, duplicate, and cull masks

The replay should be run later as:

1. Train M1 to 3000 with `steps_per_save=3000` and `save_only_latest_checkpoint=False`.
2. Resume R0/R1/R2 from the same checkpoint to 5000 with `DETERMINISTIC_AUDIT_LOG_PATH` set.
3. Compare JSONL event streams for first divergence.

## Phase 0A Data and Camera Audit

Commands:

```bash
CUDA_VISIBLE_DEVICES=6 /opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/audit_camera_data_consistency.py \
  --load-config outputs/cross_scene_japanesegradens_redsea_m1_seed42_15000/water-splatting/cross_scene_japanesegradens_redsea_m1_seed42_15000_20260730_cross_scene/config.yml \
  --output-json outputs/cpna_20260803/JapaneseGradens_camera_data_consistency.json
```

Same command pattern was run for IUI3, Curasao, and Panama.

Results:

| Scene | Train / Eval Views | Duplicate Files | Shared Train/Eval Files | Size/Intrinsics Match | Distortion Nonzero | camera_opt State |
| --- | ---: | --- | --- | --- | ---: | --- |
| JapaneseGradens | 17 / 3 | No | No | Yes | 0 | Not active |
| IUI3 | 25 / 4 | No | No | Yes | 0 | Not active |
| Curasao | 18 / 3 | No | No | Yes | 0 | Not active |
| Panama | 15 / 3 | No | No | Yes | 0 | Not active |

Important finding:

- `camera_opt` is present in `TrainerConfig`, but absent from `model.get_param_groups()` and absent from checkpoint `optimizers` / `schedulers`.
- Therefore current M1 checkpoints did not perform camera optimization.
- This config residue should be removed or implemented explicitly before any claim about camera optimization behavior.

The audit did not find obvious image/camera mismatches: image sizes match camera sizes, distortion params are zero for undistorted images, train/eval splits do not share filenames, and metadata camera indices are unique. Natural-sort checks are false in several train splits because explicit `train_list.txt` / `test_list.txt` ordering is used.

## Phase 0B Pose Oracle

Command pattern:

```bash
CUDA_VISIBLE_DEVICES=6 /opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/diagnose_cpna_pose_oracle.py \
  --load-config outputs/cross_scene_japanesegradens_redsea_m1_seed42_15000/water-splatting/cross_scene_japanesegradens_redsea_m1_seed42_15000_20260730_cross_scene/config.yml \
  --scene-name JapaneseGradens \
  --split eval \
  --max-images 0 \
  --steps 10 \
  --output-dir outputs/cpna_20260803
```

Results:

| Scene | Best Pose Mode | dPSNR | dSSIM | dLPIPS | Mean Rotation | Mean Translation | Gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| JapaneseGradens | P1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 deg | 0.0000 | Fail |
| IUI3 | P1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 deg | 0.0000 | Fail |
| Curasao | P2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 deg | 0.0000 | Fail |
| Panama | P1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 deg | 0.0000 | Fail |

The local pose search did not find any improving small pose update. Failed updates were worse than identity and the best-so-far state remained the original pose. This does not support a camera-pose correction module as the next route.

## Phase 0C Exposure / White-Balance Oracle

Command pattern:

```bash
CUDA_VISIBLE_DEVICES=6 /opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/diagnose_cpna_photometric_oracle.py \
  --load-config outputs/cross_scene_japanesegradens_redsea_m1_seed42_15000/water-splatting/cross_scene_japanesegradens_redsea_m1_seed42_15000_20260730_cross_scene/config.yml \
  --scene-name JapaneseGradens \
  --max-eval-images 0 \
  --max-train-images 0 \
  --fit-steps 100 \
  --output-dir outputs/cpna_20260803
```

Results for CEW oracle:

| Scene | CEW dPSNR | CEW dSSIM | CEW dLPIPS | Object L1 Delta | Water L1 Delta | Exposure p95 | WB Min p95 | WB Max p95 | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| JapaneseGradens | +0.2063 | +0.00061 | -0.00030 | -0.00143 | -0.00175 | 1.0696 | 0.9960 | 1.0202 | Fail |
| IUI3 | +0.5430 | +0.00035 | -0.00050 | -0.00162 | -0.00124 | 1.0274 | 0.9964 | 1.0174 | Fail |
| Curasao | +2.6750 | +0.00499 | -0.00280 | -0.00453 | -0.01277 | 1.1085 | 0.9953 | 1.0339 | Fail |
| Panama | +0.3855 | +0.00020 | -0.00135 | -0.00163 | +0.00062 | 1.0161 | 0.9845 | 1.0535 | Fail |

Interpretation:

- Exposure / WB oracle improves PSNR in all scenes.
- Curasao has a large oracle gain and passes the direct oracle metric threshold.
- JapaneseGradens does not meet the LPIPS improvement threshold (`-0.00030` vs required `<= -0.002`).
- IUI3 and Panama also do not meet LPIPS threshold.
- Panama worsens water-region L1, so the improvement is not uniformly object/water safe.

## Phase 0D Train-View Pose-KNN Predictability

LOO predictability checks whether train-view fitted CEW parameters can be predicted from neighboring camera poses without GT.

| Scene | LOO dPSNR | LOO dLPIPS | PSNR Gain Recovery | LPIPS Gain Recovery | Predictability |
| --- | ---: | ---: | ---: | ---: | --- |
| JapaneseGradens | -0.1585 | +0.00010 | -191.6% | 123.7% | Fail |
| IUI3 | +0.0407 | -0.00027 | 24.1% | 71.0% | Fail |
| Curasao | -0.3083 | +0.00039 | -111.6% | -5.0% | Fail |
| Panama | -0.0334 | -0.00010 | -21.5% | -162.9% | Fail |

The oracle parameters are not reliably predictable from pose-neighbor interpolation. Even where an image-specific oracle is strong, it does not translate into a legal novel-view module.

## Attribution Matrix

| Route | Evidence |
| --- | --- |
| OP: Pose Oracle | No scene shows positive small-pose improvement; identity remains best |
| OC: Exposure/WB Oracle | Direct oracle improves PSNR, but JapaneseGradens LPIPS gain is too small and pose-KNN predictability fails |
| OPC: Pose + Exposure/WB | Pose contributes no positive gain; combined route would be dominated by OC and inherits its predictability failure |

## Decision

Do not implement CPNF now.

The current evidence says:

- Camera optimizer is not active despite config residue.
- There is no frozen pose-oracle evidence for small camera correction.
- Per-image photometric oracle can improve PSNR, especially on Curasao, but its LPIPS benefit is weak on JapaneseGradens and not pose-predictable.
- A legal novel-view exposure/WB module requires pose-KNN or another GT-free predictor; current LOO checks fail.

The next useful step is not another training module. It is a strict deterministic replay audit from a shared step-3000 checkpoint, plus cleanup of the misleading `camera_opt` config entry or explicit implementation of a real camera optimizer if desired.
