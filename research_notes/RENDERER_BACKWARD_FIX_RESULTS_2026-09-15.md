# Renderer Backward Alpha-Cap Fix: Frozen Four-Scene Diagnostic

The previously observed native VJP error was isolated to an alpha-cap mismatch
and tested with a separately named JIT extension. `forward.cu` caps alpha at
0.999; both backward kernels had used 0.99. The working-tree source now uses
0.999 in both backward locations. The historical binary
`water_splatting/csrc.so` was not overwritten; all results below use the
separate `water_splatting_cuda_fsb_fix_20260915` extension.

No model or optimizer parameter was updated. The tested checkpoints are the
step-14999 BND ON+CICA and strict FSB models, six CICA training-bank views per
method and scene (48 model-view cases total). Test views and metrics were not
used.

## Synthetic Verification

For a one-pixel, one-Gaussian zero-medium scene, the fixed backward color VJP
matches the forward derivative:

| Opacity | Forward derivative | Fixed backward derivative | Absolute error |
|---:|---:|---:|---:|
| 0.500 | 0.500000000 | 0.500000000 | 0 |
| 0.990 | 0.990000010 | 0.989999950 | 5.96e-8 |
| 0.995 | 0.995000005 | 0.994999945 | 5.96e-8 |
| 0.999 | 0.999000013 | 0.999000013 | 0 |

The old backward values at opacity 0.995 and 0.999 were 0.495 and 0.098999,
respectively. The synthetic fix passes within 2e-6.

## Real Frozen Views

The fixed backward gradient is compared with the unchanged forward-based CICA
color reference. Forward images are rendered by both extensions and should
remain equal up to floating-point order.

| Scene | Method | Old/reference mean relative L2 | Fixed/reference mean relative L2 | Fixed max | Max forward absolute difference |
|---|---|---:|---:|---:|---:|
| Curasao | BND+CICA | 0.8709% | 0.0004% | 0.0008% | 0.001108 |
| Curasao | FSB | 0.9598% | 0.0002% | 0.0004% | 0.000450 |
| IUI3 | BND+CICA | 0.7856% | 0.0035% | 0.0104% | 0.000235 |
| IUI3 | FSB | 0.8712% | 0.0033% | 0.0058% | 0.000697 |
| JapaneseGradens | BND+CICA | 0.7231% | 0.0013% | 0.0033% | 0.000795 |
| JapaneseGradens | FSB | 0.7719% | 0.0009% | 0.0015% | 0.000787 |
| Panama | BND+CICA | 1.4983% | 0.0031% | 0.0096% | 0.000785 |
| Panama | FSB | 1.4143% | 0.0037% | 0.0062% | 0.000627 |

The fixed VJP is therefore within approximately 1e-2 relative L2 of the
forward reference on every tested case, while the old implementation was
approximately 0.7%-2.6% from that reference. The remaining small residual is
floating-point/reduction error and other renderer details; alpha-cap repair
does not prove every backward path is mathematically exact.

The native renderer's forward output remains unchanged to sub-1e-3 absolute
floating-point differences. No coverage, depth, clear image, or model hash was
used as a changed result; this is a gradient correctness diagnostic only.

## Decision

The alpha-cap fix is technically valid and should be retained as a correctness
patch for future controlled training. It is not a new FSB mechanism and no
PSNR or dewatering improvement was measured. The source change is present in
the dirty worktree, but the production shared object remains the historical
build until an explicitly controlled rebuild is authorized.

Next, run the CICA compatibility experiment with two independent switches:

1. historical renderer backward + historical CICA derivative;
2. fixed renderer backward + historical CICA derivative;
3. historical renderer backward + exact FSB CICA derivative;
4. fixed renderer backward + exact FSB CICA derivative.

Start with a short engineering run and then use a prespecified four-scene
training/evaluation protocol. Keep the old binary as a control and do not
attribute a combined gain to either correction. Do not tune switches using
the repeatedly inspected test set. Clear output remains diagnostic because
there is no independent clear ground truth.

## Artifacts

- [Synthetic and real-view JSON](../outputs/renderer_backward_fix_diagnostic_20260915_real_v3/synthetic.json)
- [IUI3 JSON](../outputs/renderer_backward_fix_diagnostic_20260915_real_v3_7/synthetic.json)
- [JapaneseGradens JSON](../outputs/renderer_backward_fix_diagnostic_20260915_real_v3_8/synthetic.json)
- [Panama JSON](../outputs/renderer_backward_fix_diagnostic_20260915_real_v3_9/synthetic.json)
- [Fix protocol](RENDERER_BACKWARD_FIX_PROTOCOL_2026-09-15.md)
- [Earlier CICA/FSB derivative results](FSB_CICA_DERIVATIVE_RESULTS_2026-09-15.md)

The temporary JIT build is under `.torch_extensions/water_splatting_cuda_fsb_fix_20260915`.
All diagnostic processes have finished. No dependency was installed and no
commit or push was performed.
