# BND-AWARE-REFINE

`Bounded-Aware Proxy-Guided Budget-Matched Refinement Allocation Causal Test`

## Repo And Execution

CODE FACT:

- Repository: `/mnt/new/home_old/ycy/water-splatting-refactor`
- Branch: `research/m1-bounded-intrinsic`
- Start HEAD: `288731e133800a7f5f7ad7c861909774271a35e8`
- The interrupted non-GPU-restricted run was stopped and its incomplete experiment directories were removed.
- The formal rerun used `CUDA_VISIBLE_DEVICES=6`, which maps the process to physical GPU 6.
- Future training GPU policy for this research line is physical GPUs `6,7,8,9`; no training run in this experiment used GPUs 0-5 after the restart.

The historical untracked files
`scripts/diagnostics/render_gmvc_curasao_contact_sheet.py` and
`scripts/diagnostics/summarize_gmvc_four_scene_visual_audit.py` were left
untouched and uncommitted.

## Research Question

This controlled intervention tests whether the locked BND-HARDNESS observability
signal has future intervention value when used only to allocate the existing
refinement budget. The experiment does not change the renderer, RGB loss,
medium, SH parameterization, optimizer, scheduler, pruning, opacity policy, or
Gaussian grow budget.

## Start State

CODE FACT:

- Scene: Panama.
- Source: formal BND-K1 checkpoint at nominal step 3000.
- Actual checkpoint step: 3000.
- Source Gaussian count: 566332.
- Optimizer and scheduler state were loaded from the source checkpoint.
- `intrinsic_color_parameterization = bounded_sh3`.
- SH degree: 3.
- `rasterize_mode = classic`.
- `medium_context_mode = dir_xy_camera`.
- `b_inf_mode = tied`.
- `infinite_water_enabled = False`.
- `coarse_depth_supervision_enabled = False`.
- Train views: 15.
- Held-out evaluation views: `MTN_1529`, `MTN_1539`, `MTN_1547`.

## Refinement Source Audit

CODE FACT:

- Refinement score: `(xys_grad_norm / vis_counts) * 0.5 * max(last_size)`.
- Densification is based on the existing absolute gradient threshold
  `densify_grad_thresh = 0.0008`.
- Split candidates use the existing large-scale/screen-size type rule.
- Duplicate candidates use the existing small-scale type rule.
- R0 applies the original threshold masks.
- Pruning uses the original opacity and size/screen gates.
- Opacity reset behavior is unchanged.
- `stop_split_at = 10000`.
- Guided branches change only parent selection inside a fixed candidate-pool and
  quota procedure; they do not change the refinement threshold, pruning, or
  optimizer update.

## Guidance Definition

CONFIG FACT:

- Locked hardness:
  `S_HARD = 0.5 * percentile_rank(S_RES_PERSIST) + 0.5 * percentile_rank(S_BOUND)`.
- `S_RES_PERSIST` is formed from K1@1k and K1@3k training-view residual ranks.
- `S_BOUND` is the K1@3k bounded full-SH clear-response rank.
- Guidance is regenerated from training views only and frozen at K1@3k.
- Gaussian mapping is projected-center bilinear sampling from per-view maps,
  averaged over valid projected views.
- A projected Gaussian is valid when its projected center and radius are finite,
  positive/in-frame, and the renderer produced a valid screen projection.
- Brightness control is the percentile rank of ground-truth mean RGB on the same
  K1@3k training-view support domain.
- Hardness/brightness sidecars are detached non-parameter state and are
  propagated through split, duplicate, and prune mutations.

Measured guidance audit:

| Quantity | Value |
| --- | ---: |
| Source Gaussian count | 566332 |
| No-valid-guidance-view fraction | 0.0009764591 |
| Hardness mean | 0.5713971 |
| Hardness p50 | 0.5859281 |
| Hardness p90 | 0.8561505 |
| Brightness mean | 0.5277609 |
| Brightness p50 | 0.5159105 |
| Brightness p90 | 0.9122593 |
| Hardness/brightness Spearman | 0.8879211 |

## Leakage Audit

EXPERIMENTAL FACT:

- Held-out evaluation views were not used for guidance: `False`.
- M1 labels were not used for guidance: `False`.
- Oracle labels were not used for guidance: `False`.
- Future K1 checkpoints after 3k were not used for guidance: `False`.
- Offline labels (`PERSISTENT_BND_HARD`, `BND_HARD_CORE`, `M1_HIGH_J`) were
  evaluation-only and were not training signals.

## Branches

Three continuations were run from the same K1@3k state:

- `R0`: baseline refinement.
- `RH`: locked-hardness guided refinement.
- `RB`: brightness-control guided refinement.

Initial parameter equivalence, optimizer-state equivalence, and forward
equivalence were all `PASS` with zero reported maximum absolute difference.
The paired camera sequence was identical across branches:
11999 continuation steps, from absolute step 3001 through 14999.

## Reference Budget And Budget Match

EXPERIMENTAL FACT:

- Nonzero grow events: 42.
- R0 cumulative split quota: 5886431.
- R0 cumulative duplicate quota: 3280440.
- R0 cumulative grow quota: 9166871.
- R0/RH/RB event-wise split and duplicate quotas matched exactly.
- Guided quota shortfall: 0.
- The guided branches selected from a fixed top-2K candidate pool.
- Mean selected-minus-candidate guidance lift:

| Branch | Kind | Guidance lift | Base-score lift |
| --- | --- | ---: | ---: |
| RH | split | 0.1019414 | 0.0003304 |
| RH | duplicate | 0.0752004 | 0.0004323 |
| RB | split | 0.1277109 | 0.0003232 |
| RB | duplicate | 0.1144283 | 0.0003891 |

The guided selection was therefore active and measurable. However, pruning
remained branch-dependent because the selected Gaussian identities differed.

## Global RGB

All values below are evaluation outputs from the same evaluation code and
paired evaluation views. The nominal 15k snapshot is represented by actual
step 14999 in this runner.

| Actual step | Branch | PSNR | SSIM | LPIPS | MSE |
| ---: | --- | ---: | ---: | ---: | ---: |
| 3000 | R0 | 20.869651 | 0.597971 | 0.440031 | 0.00859016 |
| 3000 | RH | 20.869651 | 0.597971 | 0.440031 | 0.00859016 |
| 3000 | RB | 20.869651 | 0.597971 | 0.440031 | 0.00859016 |
| 4000 | R0 | 26.363916 | 0.822787 | 0.261421 | 0.00232919 |
| 4000 | RH | 26.382213 | 0.822972 | 0.260958 | 0.00232179 |
| 4000 | RB | 26.375023 | 0.822689 | 0.261693 | 0.00232294 |
| 5000 | R0 | 26.478004 | 0.823420 | 0.252343 | 0.00227645 |
| 5000 | RH | 26.436104 | 0.823419 | 0.254426 | 0.00229892 |
| 5000 | RB | 26.410904 | 0.823019 | 0.252909 | 0.00231504 |
| 8000 | R0 | 30.966071 | 0.942205 | 0.087218 | 0.00080611 |
| 8000 | RH | 30.943176 | 0.941944 | 0.088559 | 0.00081286 |
| 8000 | RB | 30.900490 | 0.941434 | 0.088280 | 0.00082243 |
| 10000 | R0 | 31.102845 | 0.944515 | 0.083243 | 0.00078519 |
| 10000 | RH | 31.129498 | 0.944422 | 0.084036 | 0.00077976 |
| 10000 | RB | 31.036413 | 0.944007 | 0.084125 | 0.00079895 |
| 13000 | R0 | 31.475409 | 0.948480 | 0.075712 | 0.00072297 |
| 13000 | RH | 31.481044 | 0.948384 | 0.076322 | 0.00072162 |
| 13000 | RB | 31.429218 | 0.948103 | 0.076487 | 0.00073209 |
| 14999 | R0 | 31.461165 | 0.947837 | 0.076619 | 0.00072124 |
| 14999 | RH | 31.482185 | 0.947769 | 0.076923 | 0.00071714 |
| 14999 | RB | 31.410980 | 0.947487 | 0.076984 | 0.00073024 |

At actual step 14999:

- RH-R0: `dPSNR = +0.0210196`, `dSSIM = -0.0000688`,
  `dLPIPS = +0.0003040`.
- RB-R0: `dPSNR = -0.0501855`, `dSSIM = -0.0003501`,
  `dLPIPS = +0.0003651`.
- RH-RB: `dPSNR = +0.0712051`.

## Persistent Bounded-Hard And BND Hard Core

Final aggregate MSE:

| Label | R0 | RH | RB | RH relative improvement vs R0 |
| --- | ---: | ---: | ---: | ---: |
| `PERSISTENT_BND_HARD` | 0.00634047 | 0.00629103 | 0.00642883 | 0.77973% |
| `BND_HARD_CORE` | 0.01054546 | 0.01045692 | 0.01083570 | 0.83967% |
| `M1_HIGH_J` | 0.00473708 | 0.00470993 | 0.00486007 | 0.57377% |

The per-view final RGB and region metrics are stored in
`outputs/bnd_aware_refine_panama_20260812/per_view_rgb_metrics.csv` and
`per_view_metrics.csv`. They include `MTN_1529`, `MTN_1539`, and `MTN_1547`.

## Gaussian Population And Causal Validity

Final Gaussian counts at actual step 14999:

| Branch | Count | Relative to R0 |
| --- | ---: | ---: |
| R0 | 1093417 | 0 |
| RH | 1070385 | -2.1064% |
| RB | 1078696 | -1.3463% |

CODE/EXPERIMENTAL FACT:

- Grow quotas were exactly matched.
- Pruning counts were not exactly matched after branch-specific selection.
- The pre-defined approximate final-population gate therefore failed:
  `FINAL_POP_APPROX_MATCHED = False`.
- `BND_AWARE_REFINE_CAUSAL_VALID = False` because the causal validity gate
  requires both exact grow-budget matching and approximate final population
  matching.

This is an identifiability/control limitation of the completed run, not a
training crash. The outputs remain valid for external inspection, but the
formal causal intervention is not accepted by the pre-defined gate.

## Decomposition Safety

At actual step 14999:

| Branch | J p99 | P(J>1) | tau p90 | tau p99 | P(T<0.1) | P(c>0.99) | P(|s|>5) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R0 | 0.836442 | 0 | 1.003144 | 1.910212 | 0.00033967 | 0.0161350 | 0.0157906 |
| RH | 0.828439 | 0 | 0.990998 | 1.888144 | 0.00014053 | 0.0185403 | 0.0181664 |
| RB | 0.840589 | 0 | 0.985242 | 1.919432 | 0.00010364 | 0.0188794 | 0.0185112 |

Safety flags:

- `PERCEPTUAL_SAFE_RH = True`.
- `TAU_SAFE_RH = True`.
- `BOUNDARY_SAFE_RH = True`.

These are numerical decomposition checks only and are not visual-quality
judgments.

## Formal Classification

QUANTITATIVE RESULT:

```text
INCONCLUSIVE
```

The automatic classifier did not accept a causal refinement-allocation claim.
The result is not `PROXY_GUIDED_REFINEMENT_STRONG` or
`PROXY_GUIDED_REFINEMENT_PARTIAL` because the final population gate failed.
The RH branch did show a positive but small final aggregate RGB delta and lower
final persistent-hard/core MSE than R0, while the RB branch was lower in global
RGB metrics and higher in those aggregate region MSEs. Those observations are
reported as experimental facts, not as an accepted causal conclusion.

## Main Scientific Conclusion

1. The K1@3k hardness signal was deployable as an intervention sidecar and
   selected higher-hardness parents under the intended budget procedure.
2. Grow budget equality, initial state equivalence, and camera equality passed.
3. The final population equality condition did not pass because pruning
   diverged after branch-specific parent selection.
4. Therefore this run does not establish that proxy-guided refinement itself
   caused the small RH/R0 differences.
5. The current evidence is insufficient to claim that proxy-guided allocation
   recovers bounded representation capacity or that spatial refinement capacity
   is the main Panama bottleneck.
6. No new training intervention is automatically started from this result.

## Next Single-Factor Experiment

RECOMMENDATION:

Run one control-only refinement intervention that preserves the same selected
parent identities and grow quotas across branches while also enforcing
branch-matched pruning decisions, or alternatively uses a fixed final Gaussian
population mask. This is the minimum follow-up needed to separate allocation
effects from population/pruning effects. Do not combine it with a new loss,
medium change, SH change, or guidance sweep.

## Visual Assets

All generated visual assets are under:

`renders/bnd_aware_refine_panama_20260812/`

The complete index is:

`renders/bnd_aware_refine_panama_20260812/VISUAL_COMPARE_INDEX.md`

Key outputs:

- `contact_sheet_final_rgb.png`
- `contact_sheet_final_residual.png`
- `plot_psnr_trajectory.png`
- `plot_ssim_trajectory.png`
- `plot_lpips_trajectory.png`
- `plot_persistent_hard_mse_trajectory.png`
- `plot_bnd_hard_core_mse_trajectory.png`
- `plot_per_view_final_psnr.png`
- `plot_per_view_persistent_hard_mse_final.png`
- `plot_per_view_bnd_hard_core_mse_final.png`
- `plot_selection_guidance_lift.png`
- `plot_selection_base_score_lift.png`
- `plot_brightness_control_final_psnr.png`
- `plot_gaussian_count_trajectory.png`
- `plot_decomposition_tau_p90.png`
- `plot_grow_budget_match.png`
- `final_classification_summary_sheet.png`

Structured outputs are under:

`outputs/bnd_aware_refine_panama_20260812/`

The main machine-readable files are:

- `bnd_aware_refine_final_summary.json`
- `causal_validity.json`
- `global_rgb_metrics.csv`
- `per_view_rgb_metrics.csv`
- `per_view_metrics.csv`
- `decomposition_safety.csv`
- `gaussian_count_trajectory.csv`
- `refinement_budget_match.csv`
- `selection_priority_statistics.csv`
- `selection_guidance_lift_summary.csv`
- `snapshot_manifest.csv`

Visual assets are ready for external/manual analysis.
No subjective clear-image correctness judgment was made.
