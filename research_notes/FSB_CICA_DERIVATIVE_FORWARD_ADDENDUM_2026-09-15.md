# Forward-Reference Addendum To The Frozen CICA Audit

The initial Curasao BND negative control is retained under
`outputs/fsb_cica_derivative_diagnostic_20260915`. Its local color derivative
checks passed exactly and factorization reproduced the CICA kernel within
8e-6 relative L2, but the registered native backward comparison failed:
residual VJP discrepancies were 0.60%-1.18%; fixed signed cotangents gave
7.65%-10.29%. This cannot be attributed to FSB because this model is BND.

Source inspection found alpha capped at 0.999 in RGB forward and 0.99 in RGB
backward. Preserve the original failure and threshold. Do not change either
kernel. A separate run, `outputs/fsb_cica_derivative_diagnostic_20260915_v2`,
will retain the original backward-pass flag and add an independent forward
reference, rather than treating native backward as exact ground truth.

At fixed support and medium, direct RGB is linear in Gaussian colors. Render
signed tangent colors SH_C0*d*direction through the unmodified native forward
to obtain the exact forward JVP. Check its dot product with the image
cotangent against the factorized CICA VJP dotted with the parameter direction.
Use the normalized corrected-minus-native VJP as the direction to probe the
observed discrepancy; no parameter update is performed. Keep the same 1e-3
relative-error tolerance. Also record a one-pixel, one-Gaussian native test at
opacities 0.5, 0.99, 0.995, 0.999 to isolate the cap discrepancy.

Complete all four scenes and the target/gradient analysis if the local color,
factorization, forward-adjoint and preservation checks pass. Continue to
report native-backward failures explicitly. The new forward check is an
adaptive diagnostic prompted by the negative control, not a retrospectively
changed promotion criterion. No training or kernel repair is authorized by
this diagnostic itself.
