# RAOC CUDA Training Equivalence Validation

Date: 2026-08-29

## Scope

This engineering validation freezes the fused implementation at commit `43b72f4` and compares `camera_medium_raoc_backend='reference'` with `camera_medium_raoc_backend='cuda_fused'`. No RAOC equation, medium architecture, loss, optimizer, scheduler, observability state, or refinement policy was changed. The canonical archived C1 RAOC checkpoint at step 3000 was used, with optimizer and scheduler state restored and the archived camera sequence replayed. No new 15K experiment was run.

The strict `1e-6` criterion is insufficient as the sole training criterion: FMA, reduction order, register accumulation, fused kernels, and temporary materialization can change ordinary FP32 rounding while preserving the scientific equations. The validation therefore uses the registered output, gradient, trajectory, held-out metric, topology, and safety tolerances below.

## Registered tolerances

Operator: pred-image max `<= 5e-4`, mean `<= 1e-5`; Delta_z_raoc_std max `<= 5e-4`; g_keep max `<= 1e-4`; medium-gradient relative L2 `<= 1e-3`; cosine `>= 0.99999`; all values finite. Frozen evaluation: mean PSNR `<= 0.01 dB`, SSIM `<= 1e-4`, LPIPS `<= 1e-4`, MSE `<= 2e-6`, and no per-view PSNR difference over `0.02 dB`. Fixed topology: 500 finite steps, mean relative loss `<= 1e-3`, final PSNR `<= 0.01 dB`, SSIM/LPIPS `<= 1e-4`, medium parameter relative L2 `<= 5e-3`. Normal topology: 500 finite steps, count relative difference `<= 2%`, final PSNR `<= 0.02 dB`, SSIM/LPIPS `<= 2e-4`, MSE `<= 2e-4`, and no pathological topology behavior.

## Intermediate localization

The direct localization is `MULTIPLE_SMALL_FP_EFFECTS`. Raw medium outputs and detached calibration state are common; the largest direct difference is in sensitivity/reduction and its downstream gate/reconstruction tail. No scale, orientation, activation derivative, modal equation, or detached-backward semantic mismatch was identified.

- Q50: pred max `0.000488281`, Delta_z_raoc_std max `0.0020752`, g_keep max `8.45492e-05`, gradient relative L2 `7.18681e-05`, cosine `1`.
- Q80: pred max `0.000488281`, Delta_z_raoc_std max `0.00207472`, g_keep max `7.21216e-06`, gradient relative L2 `0`, cosine `1`.

Q50 and Q80 both reproduce finite gradients and near-identical rendered output. However, both fail the pre-registered direct `Delta_z_raoc_std` max gate (`Q50 0.0020752`, `Q80 0.00207472` versus `5e-4`). This gate is retained and was not relaxed.

## Frozen evaluation

The four-scene frozen checkpoint evaluation passed its registered backend metric and safety limits for Curasao, IUI3-RedSea, JapaneseGradens-RedSea, and Panama. Reference and fused were evaluated from the same frozen state without optimizer updates; per-view PSNR deltas and B_inf, beta_B, beta_D, tau, transmission, P(J>1), and J_p99 remained within the recorded floating-point tolerance.

## Fixed topology

The controlled 500-step tests restored the same Gaussian, medium, optimizer, scheduler, RNG, RAOC state, and camera sequence. Post-step bookkeeping was preserved and only topology mutation was suppressed.

- `IUI3-RedSea`: equivalent=`False`, mean relative loss delta `0.00181298`, final PSNR delta `0.0864182 dB`, medium parameter relative L2 `0.0101937`.
- `Panama`: equivalent=`True`, mean relative loss delta `8.10741e-05`, final PSNR delta `0.000476837 dB`, medium parameter relative L2 `0.00107934`.

IUI3 did not pass the fixed-topology acceptance gate, while Panama passed. This prevents a clean fixed-topology equivalence claim across the executed fixed scenes.

## Normal topology

All four 500-step runs were finite and completed without OOM. Camera sequence hashes matched with zero mismatch. Counts stayed close and below the 2% registered count limit, but each scene showed a first count divergence during normal refinement and the final metric drift was not clean:

- `Curasao`: first count divergence `3200`, final count relative delta `0.00105355`, mean/p95 loss relative delta `0.00557557/0.0282665`, PSNR delta `0.11286 dB`, SSIM delta `-0.000102421`, LPIPS delta `0.00350523`, MSE delta `-2.25136e-05`.
- `IUI3-RedSea`: first count divergence `3200`, final count relative delta `0.00160139`, mean/p95 loss relative delta `0.00998956/0.0384924`, PSNR delta `-0.0197082 dB`, SSIM delta `0.000435308`, LPIPS delta `0.000935093`, MSE delta `-1.55562e-05`.
- `JapaneseGradens-RedSea`: first count divergence `3200`, final count relative delta `0.00104336`, mean/p95 loss relative delta `0.0086002/0.0361415`, PSNR delta `-0.0630544 dB`, SSIM delta `-0.000226935`, LPIPS delta `0.000445381`, MSE delta `7.74944e-05`.
- `Panama`: first count divergence `3300`, final count relative delta `0.00100888`, mean/p95 loss relative delta `0.00749843/0.028697`, PSNR delta `0.0392475 dB`, SSIM delta `-0.000137905`, LPIPS delta `0.00078543`, MSE delta `-2.51058e-05`.

The first divergence was step 3200 for Curasao, IUI3-RedSea, and JapaneseGradens-RedSea, and step 3300 for Panama. The recorded trigger statistics show small threshold-sensitive split/duplicate/prune differences rather than a runaway explosion or collapse. Nevertheless, the resulting topology changes amplify into metric drift beyond the registered normal-topology limits in multiple scenes, so this cannot be treated as an isolated harmless count difference.

The per-backend first-event split/duplicate/prune counts and threshold margins are in `topology_divergence_analysis.json`. The minimum absolute trigger margins are:

- `Curasao` threshold minimum absolute margins at first divergence: reference `1.80444e-09`, cuda_fused `1.74623e-10`.
- `IUI3-RedSea` threshold minimum absolute margins at first divergence: reference `3.25963e-09`, cuda_fused `9.8953e-10`.
- `JapaneseGradens-RedSea` threshold minimum absolute margins at first divergence: reference `6.98492e-09`, cuda_fused `5.23869e-09`.
- `Panama` threshold minimum absolute margins at first divergence: reference `8.14907e-10`, cuda_fused `1.16415e-09`.

## Memory A/B

The memory cause classification is `MIXED_RAOC_TEMPORARIES_AND_ALLOCATOR_RESERVATION`. The synchronized trajectories separate `allocated_after`, peak-after-backward, Gaussian count, visible count, and refinement events:

- `reference`: allocated median/p95/max `1.15108e+09/1.40485e+09/1.48754e+09`; peak-after-backward median/p95/max `2.11932e+09/4.42853e+09/5.58765e+09`; reserved median/max `4.13768e+09/1.17336e+10`.
- `cuda_fused`: allocated median/p95/max `1.15043e+09/1.40346e+09/1.48586e+09`; peak-after-backward median/p95/max `1.34691e+09/1.69102e+09/1.70248e+09`; reserved median/max `1.59908e+09/2.37607e+09`.

Reference-only peak-after-backward excess with similar allocated-after medians indicates RAOC temporaries dominate peak memory. The larger reserved footprint is attributable to allocator reservation, while refinement creates synchronized topology-related spikes in both arms.

## Synchronized performance

The representative IUI3 benchmark used CUDA events, 20 warmups, and 50 timed iterations per backend and measurement. The synchronized complete forward/backward medians were `272.725 ms` reference and `16.1372 ms` fused, for `16.9004x` reference-over-fused. Forward, backward, complete, and training-step timings are in `performance_reference_vs_fused.json`. The previous approximately `32.14x` report is classified `OVER_ESTIMATED` under this timing definition.

## Decision

Primary classification: `RAOC_CUDA_TRAINING_EQUIVALENCE_NOT_SUPPORTED`

Formal backend decision: `CUDA_FUSED_FORMAL_BACKEND_NOT_APPROVED`

The fused backend is not approved for formal science. Keep `camera_medium_raoc_backend='reference'`; do not resume a new 15K causal experiment or mix reference and fused backends between causal arms. The next scientific task should only be selected after reviewing this validation artifact and resolving the failed operator/training gates.

## Repository hygiene

Only the dedicated validation script and this research note are intended for commit. Historical GMVC scripts and unrelated Q50/Q80 experiment scripts were left untouched and unstaged. Outputs, checkpoints, logs, and compiled binaries remain untracked or ignored.
