# Renderer Forward/Backward Alpha-Cap Fix Protocol

The preceding frozen diagnostic found that the native renderer forward pass
uses `min(0.999, opacity * exp(-sigma))`, while both backward kernels use
`min(0.99, opacity * exp(-sigma))`. This protocol tests only that consistency
fix. The patched CUDA source is built as a separately named JIT extension and
never replaces the historical `water_splatting/csrc.so`.

Stage 1 is a synthetic one-pixel, one-Gaussian audit at opacity 0.5, 0.99,
0.995 and 0.999, requiring the color VJP to match the native forward color
derivative within 2e-6. Stage 2 reuses the fixed extension on the step-14999
BND and FSB checkpoint bank views from the preceding derivative diagnostic.
It compares native and fixed backward against the unchanged native forward
adjoint, separately for GT-pred and a deterministic signed cotangent. It
reports whether the patch reduces the discrepancy without changing forward
images, coverage, depth, clear output, model hashes, or checkpoint bytes.

No model or optimizer update is permitted. A successful synthetic result only
establishes local alpha-cap consistency; it does not establish a training or
novel-view gain. Any later training must use independent paired switches and
must preserve the old binary as a control.
