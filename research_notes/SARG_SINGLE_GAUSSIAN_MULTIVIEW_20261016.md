# Single-Gaussian multi-view candidate screening

This screen uses the completed v6 checkpoints but a new view split: every
third ordinary training view is reserved, half for calibration and half for
final audit; the other views are fitting views. The previous stride-four split
is not reused for selection. Code:

`research_runs/sarg_single_gaussian_multiview_20261016/code/screen.py`

For each fitting view, native additive clear RGB is attributed to individual
Gaussians inside a descriptive far residual region (accumulation > 0.8,
upper-depth 30%, B > R + .04 or G > R + .04). Per-view contribution is
normalized by that view's regional mass. Candidates require:

- not protected by the frozen surface-anchor mask;
- normalized mean contribution >= .003;
- contribution above .002 in at least one quarter of fitting views (minimum 3);
- at least three fitting views with mean absolute clear deletion >= 1e-5;
- all fitting/calibration local support tiles pass the 1% L1/MSE gate;
- all independent audit views pass the same local and inherited ROI gates.

The script records score, cross-view standard deviation, view counts, support
tiles, local failures, ROI metrics and model hash. It does not update model
parameters.

## Results

| Scene | Candidates after score/anchor filter | Admitted without audit | Audit-passing candidates |
| --- | ---: | ---: | ---: |
| Panama | 0 | 0 | 0 |
| IUI3-RedSea | 6 | 0 | 0 |

The six IUI3 candidates have normalized scores from .00318 to .00738 and
9--16 active fitting views, but fail the strict local/ROI screen. Panama's
anchor protection and new multi-view threshold remove all candidates before
intervention testing. Both model hash checks remain unchanged.

## Decision

Do not start 15k training. The new split provides no evidence for a safe
single-Gaussian deletion mechanism. Lowering the contribution threshold or
relaxing local gates would manufacture candidates without evidence that object
geometry is preserved.

The next useful experiment is diagnostic rather than training: for the six
IUI3 candidates and the highest-scoring protected Panama Gaussians, compare
their contribution support against anchor reprojection, depth-layer overlap,
and medium replacement across views. This will determine whether the blocker
is conservative anchor protection, an over-broad residual mask, or genuine
object/medium entanglement. A future candidate should use a fresh checkpoint
or a second independent view split before any training run.
