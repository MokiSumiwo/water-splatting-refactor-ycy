# Local contribution grouping frozen validation

## Protocol

Code: `research_runs/sarg_local_contribution_frozen_20261016/code/validate.py`.
Both completed v6 checkpoints were frozen. Same fitting/calibration/audit split
as strict v7: Panama 11/2/2, IUI3 18/4/3 ordinary training cameras. No test
camera participates. Model hashes remained unchanged in both scenes.

Three arms: strict v7 reproduction; local grouping with the same strict ROI
gates; local grouping plus local photometric vetoes. All use 32-member groups,
72-group maximum, 10% per-group and 20% cumulative per-fitting-view additive
residual contribution caps, anchor protection and inherited ray gates.

Local grouping replaces spatial voxel bins with dominant residual-contribution
fitting camera, projected centre in a 10x10 normalized image grid, and log-depth
bins of ratio 1.25. Dominant camera comes from native per-Gaussian regional RGB
contribution. This is a projection/contribution proxy: it is not a full
pixel-footprint clustering method and does not certify similar support in all
other cameras. The bins can combine spatially separated Gaussians.

Local veto: native clear RGB mean absolute change > 1e-5, dilated with a 5x5
max filter (two-pixel halo), partitioned into fixed 32x32 tiles. Check tiles
with at least 16 affected pixels. Evaluate mean underwater L1/MSE change only
on this support. Use inherited relative 1% tolerances and floors/caps separately
for each tile; every tile must pass. Apply this at individual and cumulative
fitting probes and cumulative calibration probes. Record final audit tiles
without using them for selection. Pixel maximum and p99 increases are also
recorded as diagnostics, not calibrated rejection thresholds. This screen does
not provide semantic real-surface labels or independently protect such pixels.

## Results

| Scene / arm | Tested RGB coverage | Accepted RGB coverage | Groups | Audit signed regional RGB removed | Audit failed tiles / evaluated tiles |
| --- | --- | --- | --- | --- | --- |
| Panama strict v7 | 7.412% | 0.501% | 4 | 0.171%, 0.191% | 1/12, 2/16 |
| Panama local | 13.035% | 1.224% | 13 | 0.175%, 0.095% | 18/46, 23/70 |
| Panama local + veto | 13.035% | 0% | 0 | 0%, 0% | no intervention |
| IUI3 strict v7 | 9.231% | 1.978% | 17 | 0.572%, 0.360%, 0.107% | 14/66, 21/67, 13/48 |
| IUI3 local | 15.625% | 3.979% | 16 | 1.880%, 1.164%, 0.387% | 16/86, 37/86, 16/72 |
| IUI3 local + veto | 15.625% | 0% | 0 | 0%, 0%, 0% | no intervention |

Every nonempty audit ROI passed the old mean photometric check. Nevertheless,
even the strict-v7 control has failing local tiles. Individual local-gated
probes fail the local screen in 409/480 Panama and 460/503 IUI3 renders. These
counts include all probes, not just the originally accepted groups.
Full records, tile coordinates, tolerances, and generated sarg variants are
under each scene's arm directories; aggregate results are scene `results.json`.

## Interpretation and decision

Do not start two fresh 15k runs. Local grouping increases candidate coverage;
IUI3 causal removal increases, but local photometric regressions remain.
Panama does not gain consistent audit removal. Adding all-tile rejection leaves
no accepted intervention; this is not a successful safe-removal result.

The result supports the concern about bounding-box error dilution. It does not
prove all groups are semantic object/medium mixtures: local thresholds may be
conservative, and the change-derived support is itself intervention dependent.
These reused original training views are not an unseen generalization test.
Only final v6 checkpoints and one split were evaluated.

Next isolate the source of local failures: save full/drop/error/support maps
for high-contribution rejected groups, and compare spatially local singletons
or small subgroups against their parent under identical local gates. Use those
results to distinguish inseparable object contribution from grouping artifacts
before changing thresholds or implementing training suppression. Actual
per-pixel contribution footprints across multiple fitting views would improve
on the present projected-centre proxy. Do not relax safety merely to obtain a
nonzero accepted set.
