# MCGR Correspondence-Gated Gaussian Refinement

Date: 2026-08-01

## Objective

Implement and test MCGR, a training-only M1 refinement module that uses pseudo depth only to build cross-view correspondence. MCGR does not add a pseudo-depth loss, does not supervise clear-water appearance, does not change the underwater renderer, and does not change inference outputs.

M1 remains:

- `medium_context_mode=dir_xy_camera`
- `b_inf_mode=tied`
- `B_inf = medium_rgb = A`
- Old J/TBAP/TMICA/TACMD/capacity/cleanup/pruning directions remain disabled in MCGR experiment scripts.

## Code Changes

- Added `water_splatting/geometry/mcgr.py`
  - Loads `view_XXXX_mcgr.pt` correspondence payloads.
  - Builds detached Sobel + 5x5 high-pass underwater RGB residual maps.
  - Maintains a CPU float16 per-view residual bank through the training loop.
  - Warps neighbor residual-bank entries through precomputed correspondences.
  - Samples persistent residual and correspondence confidence at Gaussian projection centers.
  - Selects conservative extra split/duplicate candidates by OR-ing MCGR candidates into the existing `high_grads` mask.
- Integrated MCGR into `water_splatting/water_splatting.py`
  - Added default-off config flags.
  - Added per-Gaussian MCGR buffers with split/duplicate/cull synchronization.
  - Added optional gradient-coherence buffers and candidate gating, default off.
  - Added JSONL diagnostics for evidence and refinement events.
  - Added no loss terms; `loss_dict` remains unchanged except existing M1/MV-GAR paths.
- Added `scripts/preprocess/build_mcgr_correspondences.py`
  - Uses Nerfstudio train-view order, so payload `view_XXXX_mcgr.pt` matches `camera_index`.
  - Reads MV-GAR aligned pseudo-depth payloads.
  - Builds a COLMAP shared-track camera graph.
  - Precomputes bilinear correspondences, log-depth consistency, front/occlusion gates, forward-backward cycle consistency, and RGB-gradient structure consistency.
- Updated `scripts/experiments/backscatter_consistent_binf_iui3_redsea.sh`
  - Passes MCGR flags and logs them in the run manifest.
- Added MCGR wrappers:
  - `scripts/experiments/mcgr_5k_common.sh`
  - `scripts/experiments/mcgr_c0_m1_japanesegradens_5k.sh`
  - `scripts/experiments/mcgr_c1_diag_japanesegradens_5k.sh`
  - `scripts/experiments/mcgr_c2_persistent_japanesegradens_5k.sh`
  - `scripts/experiments/mcgr_c3_persistent_coherence_japanesegradens_5k.sh`
  - `scripts/experiments/mcgr_c0_m1_iui3_5k.sh`
  - `scripts/experiments/mcgr_c2_persistent_iui3_5k.sh`
  - `scripts/experiments/mcgr_c3_persistent_coherence_iui3_5k.sh`

## New Config Flags

```python
mcgr_enabled: bool = False
mcgr_diagnostic_only: bool = True
mcgr_correspondence_dir: Optional[str] = None
mcgr_log_path: Optional[str] = None
mcgr_start_step: int = 1000
mcgr_stop_step: int = 10000
mcgr_residual_bank_downscale: int = 4
mcgr_residual_ema_decay: float = 0.80
mcgr_residual_max_age_epochs: float = 2.0
mcgr_highpass_weight: float = 0.35
mcgr_residual_match_tau: float = 0.30
mcgr_min_valid_neighbors: int = 2
mcgr_min_view_count: int = 3
mcgr_min_mean_confidence: float = 0.30
mcgr_persistent_quantile: float = 0.85
mcgr_accumulation_mid: float = 0.40
mcgr_accumulation_temp: float = 0.08
mcgr_depth_std_kappa: float = 0.25
mcgr_gradient_coherence_enabled: bool = False
mcgr_gradient_coherence_threshold: float = 0.35
mcgr_max_extra_ratio_to_base: float = 0.20
mcgr_max_extra_fraction_per_refine: float = 0.001
```

## Phase 0 Correspondence Audit

Commands:

```bash
/opt/anaconda3/envs/water_splatting/bin/python scripts/preprocess/build_mcgr_correspondences.py \
  --data undistorted_data/undistorted_JapaneseGradens-RedSea \
  --depth-dir undistorted_data/undistorted_JapaneseGradens-RedSea/depthAnything_u16 \
  --pseudo-depth-dir outputs/mvgar_pseudo_depth/japanesegradens_train \
  --output-dir outputs/mcgr_correspondences/japanesegradens_train

/opt/anaconda3/envs/water_splatting/bin/python scripts/preprocess/build_mcgr_correspondences.py \
  --data undistorted_data/undistorted_IUI3-RedSea \
  --depth-dir undistorted_data/undistorted_IUI3-RedSea/depthAnything_u16 \
  --pseudo-depth-dir outputs/mvgar_pseudo_depth/iui3_train \
  --output-dir outputs/mcgr_correspondences/iui3_train
```

Results:

| Scene | Train Views | Two-Neighbor Coverage | Three-Neighbor Coverage | Open-Water Support | Structure Support | Log-Depth p50 | Log-Depth p90 | Cycle p50 | Cycle p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| JapaneseGradens | 17 | 0.91% | 0.05% | 65.39% | 34.61% | 0.0049 | 0.0125 | 1.50 px | 2.70 px |
| IUI3 | 25 | 0.31% | 0.02% | 78.97% | 17.03% | 0.0067 | 0.0156 | 1.43 px | 2.58 px |

The log-depth and cycle-error statistics are acceptable where correspondences exist, but dense support is far below the planned gate. IUI3 two-neighbor coverage is below the 3% hard stop, and most supported pixels are low-gradient/open-water by the current structure proxy.

Decision: do not run formal C0-C3 5k screening for MCGR V0/V1. The correspondence bank is too sparse and too water-dominated to justify adding densification candidates.

## Smoke Tests

Static checks:

```bash
/opt/anaconda3/envs/water_splatting/bin/python -m py_compile \
  water_splatting/geometry/mcgr.py \
  water_splatting/geometry/__init__.py \
  water_splatting/water_splatting.py \
  scripts/preprocess/build_mcgr_correspondences.py

bash -n scripts/experiments/backscatter_consistent_binf_iui3_redsea.sh \
  scripts/experiments/mcgr_5k_common.sh \
  scripts/experiments/mcgr_c0_m1_japanesegradens_5k.sh \
  scripts/experiments/mcgr_c1_diag_japanesegradens_5k.sh \
  scripts/experiments/mcgr_c2_persistent_japanesegradens_5k.sh \
  scripts/experiments/mcgr_c3_persistent_coherence_japanesegradens_5k.sh \
  scripts/experiments/mcgr_c0_m1_iui3_5k.sh \
  scripts/experiments/mcgr_c2_persistent_iui3_5k.sh \
  scripts/experiments/mcgr_c3_persistent_coherence_iui3_5k.sh

git diff --check
```

300-step diagnostic smoke:

```bash
STAMP=20260801_mcgr_smoke_diag GPU=6 MAX_NUM_ITERATIONS=300 MODEL_NUM_STEPS=300 \
STEPS_PER_SAVE=300 RUN_EVAL=0 RUN_CLOSURE_DIAG=0 BUILD_MVGAR_DEPTH=0 BUILD_MCGR_CORRESPONDENCE=0 \
scripts/experiments/mcgr_c1_diag_japanesegradens_5k.sh
```

Result:

- Training completed without CUDA/autograd errors.
- `logs/mcgr_c1_diag_japanesegradens_seed42_300_20260801_mcgr_smoke_diag/mcgr.jsonl` was written.
- Step 0 evidence loaded payload successfully, with `mcgr_evidence_gaussian_fraction=5.38%`; persistent residual was zero because no neighbor residual bank entries existed yet.

800-step refinement smoke:

```bash
STAMP=20260801_mcgr_smoke_refine GPU=7 MAX_NUM_ITERATIONS=800 MODEL_NUM_STEPS=800 \
STEPS_PER_SAVE=800 RUN_EVAL=0 RUN_CLOSURE_DIAG=0 BUILD_MVGAR_DEPTH=0 BUILD_MCGR_CORRESPONDENCE=0 \
MCGR_START_STEP=0 scripts/experiments/mcgr_c2_persistent_japanesegradens_5k.sh
```

Result:

- Training completed without CUDA/autograd errors.
- Densification at step 700 completed, including split/duplicate/cull synchronization.
- Base M1 high-gradient candidates: 17,024.
- MCGR supported Gaussians: 3,177.
- MCGR min-view-qualified Gaussians: 174.
- MCGR confidence-qualified Gaussians: 1.
- MCGR extra candidates: 0.
- MCGR split/duplicate count: 0/0.
- Gaussian count after cull: 24,521.

## 5k Gate Decision

Formal 5k C0-C3 screening was not launched because Phase 0 failed before training:

- IUI3 two-neighbor coverage was 0.31%, below the 3% hard stop.
- Three-neighbor coverage was effectively zero in both scenes.
- Supported pixels were dominated by low-gradient/open-water regions.
- 800-step smoke produced no MCGR extra candidates under the conservative V0 gate.

This is a negative result for MCGR V0/V1 as designed. The current pseudo-depth correspondence path is not reliable enough to drive cross-scene Gaussian refinement.

## Interpretation

MCGR confirms the main weakness of the pseudo-depth route: sparse depths can align well at selected points, but dense, repeated, object-safe correspondence is not available at sufficient coverage in the current underwater data. This weakens the hypothesis that M1's JapaneseGradens deficit can be safely solved by pseudo-depth-guided Gaussian densification.

The evidence does not confirm that M1 loss is caused by Gaussian detail/refinement shortage. Combined with MPDR and MV-GAR, the Gaussian refinement direction now has three negative or unsafe signals:

- Single-view detail evidence is too ambiguous.
- Direct pseudo-depth surface anchoring improves JapaneseGradens PSNR but hurts LPIPS/IUI3 safety.
- Multi-view correspondence-gated evidence is too sparse to select useful conservative candidates.

## Next Step

Stop MCGR before 5k/15k. The next analysis should shift away from pseudo-depth Gaussian densification and toward cross-view appearance consistency, SH representation/curriculum, camera/datamanager consistency, or medium/Gaussian frequency decomposition.
