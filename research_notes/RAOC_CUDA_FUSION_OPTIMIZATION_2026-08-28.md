# RAOC CUDA Fusion Optimization

## Scope

This was a pure engineering optimization of the existing WaterSplatting
RAOC execution path.  No RAOC equation, loss, optimizer, densification rule,
renderer quality setting, resolution, or mode count was changed.  No new
formal 15K causal experiment was launched.

Starting state was branch `research/m1-bounded-intrinsic`, HEAD
`4cd40a6 Complete four-scene RAOC causal experiment`.  The four allowed
physical GPUs were free before profiling; no existing formal job was killed or
modified.  All profiling and smoke processes exposed exactly one of GPUs
6/7/8/9.

## Reference Path And Bottleneck

The reference path predicts the full and neutral medium, builds the analytic
compositor directional actions in PyTorch chunks, projects the standardized
residual into nine modes, computes sensitivity/evidence/gates, reconstructs
the residual, and then runs the existing differentiable renderer.  The
analytic action avoids a global Jacobian but materializes an equivalent
chunked `[N, 9, 3]` action tensor before the norm and gate calculations.

The measured local control path was the dominant RAOC-specific cost.  At
step 14999, its reference time was 3537.5 ms / 1039.4 ms / 2422.8 ms /
3634.6 ms for Curasao / IUI3 / JapaneseGradens / Panama, respectively.  The
separately timed full forward plus backward measurements were 3624.4 ms /
1085.2 ms / 2482.1 ms / 3734.5 ms.  These are separate CUDA-event
measurements, so the control measurement is diagnostic and is not added to the
full-step timing.

## Optimization Design

The new `cuda_fused` backend is a standalone CUDA operator.  One kernel
handles all nine modes per ray and keeps fixed-size arrays in registers.  It
computes the compositor local directional sensitivity directly from the
sorted Gaussian/tile state without materializing a global `[N, 3, 9]` tensor,
then fuses:

- compositor local sensitivity and the nine `J_p v_i` actions;
- modal coefficient projection;
- evidence;
- local gate and keep gate;
- modal reconstruction.

The existing `reference` backend remains available and remains the default.
The backend receives `q` as a detached nine-value input, so Q50, Q80, and
future externally supplied calibration states do not require CUDA code changes.

The custom autograd wrapper saves only detached `basis` and `keep_gate` for
the residual VJP.  Gate/evidence/sensitivity inputs do not participate in
autograd.  The current VJP uses the existing PyTorch modal helper to preserve
the reference GEMM accumulation order.  No full Jv tensor is saved for
backward, and no second-order gate graph is constructed.

## Build

Production build completed with CUDA 11.8 and the environment's PyTorch
2.1.2+cu118:

```text
RAOC_PRECISE_MATH=0 CUDA_HOME=/usr/local/cuda-11.8 \
PATH=/usr/local/cuda-11.8/bin:$PATH \
/opt/anaconda3/envs/water_splatting/bin/python setup.py build_ext --inplace --force
```

The production extension uses `--use_fast_math`.  A reproducible audit build
is available with `RAOC_PRECISE_MATH=1`; it adds precise division/square-root,
FTZ-off, and FMA-off flags.  The compiled extension exports
`raoc_fused_forward` and `raoc_fused_backward`, and kernel launch checks are
enabled.

## Equivalence And Regression Results

The modal-only contract audit passed at approximately `1e-7` and confirmed
that arbitrary Q50/Q80 q inputs change the gate, all nine modes are evaluated,
diagnostics are detached, the medium path has a nonzero direct gradient, and
the direct Gaussian gradient is zero as expected.

The real compositor audit did not pass the required strict FP32 target.  The
largest step-3000 fast-math errors across the four scenes were approximately:

| Quantity | Maximum absolute error |
| --- | ---: |
| sensitivity | `3.66e-5` |
| evidence | `1.70e-5` |
| local gate | `2.36e-3` |
| keep gate | `1.25e-3` |
| reconstructed standardized residual | `2.20e-3` |
| complete model `pred_image` | `2.44e-4` |

The IUI3 model-level medium MLP gradient relative L2 error was
`7.19e-5`.  The fused and reference outputs and gradients were finite.  The
remaining differences are consistent with per-ray serial compositor
accumulation and exponent/rounding order versus vectorized PyTorch
`cumprod`/`cummax`/`sum` operations.  Changing exp variants, accumulator
precision, FMA, and precise build flags did not reach the strict gate target.

Legacy pre-RAOC checkpoint loading passed.  Archived calibrated RAOC state
loading passed.  RAOC-disabled output was unchanged, and repeated OCMC output
was unchanged.  The 20-step matched smoke passed on all four scenes: both
backends were finite, Gaussian counts matched at every step, and maximum loss
differences were `9.86e-6` / `1.84e-5` / `3.25e-6` / `6.01e-6` for Curasao /
IUI3 / JapaneseGradens / Panama.

## Runtime Results

The following values are from archived C1 checkpoints and CUDA-event timing,
with three warm-up calls and five timed control calls.  At step 14999:

| Scene | Reference RAOC control ms | Fused RAOC control ms | Reference full fwd+bwd ms | Fused full fwd+bwd ms | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Curasao | 3537.5 | 23.1 | 3624.4 | 111.4 | `32.53x` |
| IUI3-RedSea | 1039.4 | 4.3 | 1085.2 | 48.9 | `22.20x` |
| JapaneseGradens-RedSea | 2422.8 | 13.4 | 2482.1 | 69.5 | `35.73x` |
| Panama | 3634.6 | 23.2 | 3734.5 | 115.1 | `32.44x` |

The mean of the four step-14999 speedups is `30.72x`; the mean over all
four scenes and four topology profile rows is `32.14x`.  The profiles show no
scene becoming slower in the measured full forward/backward path.

For context, a read-only OCMC baseline profile was also collected at the
same archived C1 step-14999 checkpoints.  OCMC does not execute the per-ray
RAOC control path, so its control-stage timing is not applicable.  Its full
forward plus backward times were 431.0 ms / 290.2 ms / 290.2 ms / 453.5 ms
for Curasao / IUI3 / JapaneseGradens / Panama, respectively.  These values
are a separate engineering baseline and are not used to claim a causal
quality result.

## Memory Diagnosis

The reference peak allocated memory at step 14999 was 4152.2 / 2516.6 /
3696.6 / 4188.7 MB for Curasao / IUI3 / JapaneseGradens / Panama.  The fused
values were 3488.0 / 2261.4 / 2171.6 / 3453.1 MB.  The corresponding peak
reserved values were reference 5300 / 3030 / 5368 / 5328 MB and fused 4766 /
3060 / 3028 / 4792 MB.

Across all 16 reference topology observations, Gaussian count versus peak
allocated Pearson correlation was `0.768`, while intersection count versus
peak allocated correlation was `0.940`.  Per scene, Gaussian correlations
were `0.989-0.999` and intersection correlations `0.974-0.997`.  The evidence
supports a mixed diagnosis: Gaussian topology establishes persistent base
growth; visible/tile-intersection workload drives view-dependent renderer
workspace; reference RAOC temporary `[N,9,3]`-equivalent tensors create an
avoidable peak; and reserved memory includes PyTorch caching allocator
reservation.  `empty_cache()` was not used inside timed model iterations.

The fused kernel removes the large Jv action tensor and several separate
pointwise intermediates from the control path.  The current integration still
returns diagnostic `[N,9]` tensors for evidence, local gate, keep gate, and
sensitivity, because existing diagnostics consume them.  The deeper renderer
pass remains outside the standalone operator boundary.

## Limitations And Recommendation

The fused path is materially faster and generally uses less peak allocated
memory, but it is not numerically equivalent to the frozen reference at the
required strict threshold.  Therefore the final classification is:

`RAOC_CUDA_OPTIMIZATION_NOT_READY`

Recommended backend for future formal Q50/Q80 15K training is still
`camera_medium_raoc_backend = "reference"`.  Keep `cuda_fused` available for
continued engineering analysis only.  The single largest remaining runtime
issue is exact compositor floating-point equivalence, followed by the
standalone renderer/operator boundary.  The next engineering task should be a
reference-compatible compositor reduction design, ideally sharing the exact
renderer traversal and reduction order, followed by repeating the complete
equivalence audit before considering production activation.

The OCMC baseline is recorded in
`outputs/raoc_cuda_fusion_optimization_20260828/ocmc_baseline.json` and the
reference/fused/OCMC timing joins are in
`outputs/raoc_cuda_fusion_optimization_20260828/four_scene_runtime.json`.

All machine-readable results are under
`outputs/raoc_cuda_fusion_optimization_20260828/` and are intentionally not
tracked.
