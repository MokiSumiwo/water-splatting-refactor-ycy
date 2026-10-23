# BND-OFF Teacher Quality Audit (2026-09-14)

This read-only audit compares the current BND-ON frozen checkpoint with the
historical BND-OFF checkpoint on all 13 existing test views. The historical
checkpoint is not a matched retraining comparator; it is used to decide
whether a future underwater-only teacher branch is technically justified.

## Results

| Scene | BND ON PSNR | BND OFF PSNR | Teacher-better view fraction | Oracle global blend PSNR | Gain vs ON |
| --- | ---: | ---: | ---: | ---: | ---: |
| JapaneseGradens-RedSea | 24.7371 | 24.6281 | 1/3 | 25.0361 | +0.2990 |
| Panama | 31.4668 | 32.1787 | 3/3 | 32.6234 | +1.1566 |
| Curasao | 32.5961 | 32.2153 | 2/3 | 32.9064 | +0.3103 |
| IUI3-RedSea | 30.6132 | 31.1124 | 4/4 | 31.2945 | +0.6814 |
| Macro mean | 29.8533 | 30.0336 | 75.0% | 30.4651 | +0.6118 |

The BND-OFF teacher is useful on average but is not uniformly better. It is
worse on two of three JapaneseGradens views and one of three Curasao views.
The oracle blend uses GT to choose a global coefficient and is therefore an
upper-bound diagnostic only; it is not a valid test-time method. Pixel-wise
oracle blending reaches 31.7569 dB (+1.9036 dB), confirming that the two
representations contain complementary errors, but also that a learned trust
signal is required.

Artifacts:

- `outputs/bnd_transport_recovery_20260912_formal/teacher_quality_summary.json`
- `outputs/bnd_transport_recovery_20260912_formal/teacher_quality_per_view.csv`
- `outputs/bnd_transport_recovery_20260912_formal/<scene>/teacher_quality_contact_sheet.jpg`

## Decision

Do not train direct full-strength BND-OFF distillation. The next experiment
should be a small underwater-only residual branch with a bounded correction,
teacher confidence estimated without GT, and a held-out-view trust-region.
Clear rendering and Gaussian geometry remain frozen.
