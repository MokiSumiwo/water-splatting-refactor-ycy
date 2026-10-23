# SARG-v7 strict residual-budget candidate

## Decision

Defer two-scene 15k training. The frozen candidate passes the reserved-view
photometric screen, but causal residual removal remains small. This is a
candidate admission implementation, not a completed training integration.

## Reproducible implementation

`research_runs/sarg_v7_strict_candidate_20261016/code/validate.py` loads the
completed v6 15k checkpoint without updating its parameters. The script writes
the exact generated `sarg_variant.py` for each scene and complete individual
and cumulative deletion checks under `strict_residual_budget/refresh/`.

Run with the water_splatting environment, one CUDA device per scene:

```bash
CUDA_VISIBLE_DEVICES=0 CONDA_DEFAULT_ENV=water_splatting /opt/anaconda3/envs/water_splatting/bin/python research_runs/sarg_v7_strict_candidate_20261016/code/validate.py --scene Panama
```

Use IUI3-RedSea as the other scene argument. No test views enter ranking,
selection, calibration or final audit.

## Candidate mechanism

- Fitting views: ordinary training views excluding every fourth view.
- Residual region: accumulation > 0.8, positive finite depth, upper depth
  30% within supported pixels, and B > R + 0.04 or G > R + 0.04.
- Native compositing attributes clear RGB to each Gaussian within that region;
  attribution is checked against the native regional sum (relative error < 3e-5).
- Normalize mass by each view's total regional mass and average across fitting
  views for ranking. Retain v6 geometric eligibility and anchor protection.
- Spatial groups contain at most 32 members; retain at most 72 groups.
- A candidate group contributes at most 10% of regional mass in any fitting
  view; the entire tested set contributes at most 20% in any fitting view.
- Greedy allocation uses mean per-view mass / (0.02 + already covered mass).
  These are additive candidate budgets, not guarantees about causal deletion.
- All active individual and cumulative fitting probes must pass the inherited
  photometric and replacement gates. Retain same-ray far responsibility gates.
- Split every-fourth reserved views alternately into calibration and final
  audit. Each cumulative union must pass photometric checks in every calibration
  view where its projected ROI exists. No ROI means no projected intervention.
- Final audit views do not select or reject groups during this run.

## Frozen results

| Scene | Fit/calibration/audit | Tested regional RGB | Accepted regional RGB | Accepted groups | Final audit |
| --- | --- | --- | --- | --- | --- |
| Panama | 11/2/2 | 7.412% | 0.501% | 4 | 2/2 pass |
| IUI3-RedSea | 18/4/3 | 9.231% | 1.978% | 17 | 3/3 pass |

Actual cumulative deletion, measured against each audit view's frozen full-model
residual region, removes signed RGB mass fractions of 0.171% and 0.191% in
Panama; 0.572%, 0.360% and 0.107% in IUI3. These are causal clear-image changes,
not changes in semantic water volume or blue-pixel fractions. All final audit
ROIs pass both inherited photometric and medium replacement checks. Both model
hash comparisons report unchanged parameters. The full per-view results are in
the scene `results.json` files.

Strict fitting rejection substantially reduces the accepted coverage compared
with the preceding balanced-budget experiment (Panama 2.641%, IUI3 5.207%).
The increased candidate coverage therefore does not translate directly into
safe, useful removal. The 10%/20% caps are fixed exploratory settings, not tuned
or established optimal budgets.

## Limits and next training design

These reserved cameras are still original training views. Their outputs were
also inspected in prior experiments, so the final audit is computationally
excluded from this selection but is not a fresh, unseen benchmark. Only one
checkpoint and one split were tested; no water-free ground truth is available.
Blue-green surfaces may be real objects. Strict calibration currently vetoes
photometric regressions; replacement and segment responsibility are still
fitting-view gates and must not be described as certified in every camera.

Before a full training integration, test support-local grouping and cumulative
ROI budgets: large union bounding boxes can dilute localized errors. Require
consistent signed removal within the target support, retain per-view vetoes,
and record accepted contribution across refreshes so training cannot silently
move residual mass to untested groups. A training implementation must recompute
attribution after topology changes and include its preprocessing in refresh cost;
the frozen normalized arrays must never be reused during training.

Do not increase suppression strength or remove anchor protection based on this
screen. Establish useful causal removal on multiple frozen checkpoints/splits
before authorizing the expense of two fresh 15k runs.
