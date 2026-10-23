# Rejected-group split diagnostics

Frozen v6 checkpoints were used with the same calibration and audit camera
split as the local-contribution experiment. No parameters changed. For the
eight highest residual-contribution groups with local failures in each scene,
the script saved full/drop/impact images and local tile diagnostics for the
whole group, every chunk of 2/4/8, and every singleton member:

`research_runs/sarg_group_failure_diagnostics_20261016/code/diagnose.py`

Each variant was tested on the two calibration and remaining reserved views.
The local gate is the same 32x32 support-tile gate used in the previous run;
the inherited ROI photometric gate is also recorded. Outputs are under each
scene's `group_XX/` directory and aggregate records are in
`Panama_summary.json` and `IUI3-RedSea_summary.json`.

## Findings

Most rejected groups remain locally unsafe after splitting, but the exhaustive
singleton scan finds a sparse safe tail. Panama has 31 passing singletons out
of 217 tested; IUI3 has 5 out of 105. Panama's largest passing singleton
impact over the four views is about `2.2e-4` mean absolute RGB; IUI3 passing
singleton impacts are only about `1e-7` to `1e-6`, effectively negligible. The
high-contribution IUI3 groups and most Panama groups retain failures even at
singleton size. Both within-group mixing and view-dependent individual
Gaussian entanglement are present.

## Decision

Do not train with the current rejected-group candidates. Splitting groups does
not expose a useful safe budget: IUI3 has no useful safe tail, while
high-contribution members remain locally damaging. The likely limitation is
view-dependent object/medium entanglement plus an overly broad residual target,
rather than group cardinality alone.

The saved maps should be used to classify failures before changing thresholds:
compare impact support with anchor reprojection support, depth-layer overlap,
and medium replacement. A subsequent candidate should rank individual
Gaussians by multi-view residual contribution, require a minimum useful removal
effect, and retain strict local gates, then retest on a fresh frozen
checkpoint/split. Do not relax local gates to obtain a nonzero acceptance set,
and do not start the two 15k runs yet.
