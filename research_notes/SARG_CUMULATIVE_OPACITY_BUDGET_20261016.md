# Expanded-pool cumulative attenuation

Artifacts: `research_runs/sarg_cumulative_opacity_budget_20261016/`, including
`code/run.py` and scene `results.json` with all acceptance and final tile checks.
Frozen v6 checkpoints, no test cameras or parameter updates. Existing stride-3
fit/calibration/audit split is reused; this is exploratory, not fresh generalization.

## Fixed protocol

Top 128 unprotected candidates per scene ranked by existing fitting-view regional
contribution, visible contribution in at least three fitting views. This broadens
the previous score threshold exploratorily. Single rate 10%; sum of rates <=3.2
(at most 32 members); per-fitting-view attenuated additive residual contribution
<=0.001. The additive budget is a ranking proxy, not actual causal removal.

Greedy candidate order is fixed by score. Every addition rerenders cumulative
attenuation in all fitting/calibration cameras, checking 32px tiles on the union
of full-deletion member supports and separately on the newly added member's
support. Targets/supports use the original frozen model; no attenuation-dependent
support shrinkage. Audit cameras do not select members.

Final check repeats cumulative attenuation on every camera, on BOTH the union
support and EVERY selected member's support (not only the most recently added).
The final check can invalidate the selected set; no post-audit repair/tuning.
Useful fitting effect requires target signed mean >=1e-5 in >=3 fitting views;
useful audit effect is reported as all audit views >=1e-5. These are exploratory
operational thresholds, not proven perceptual improvement or geometry safety.

## Results

| Scene | Pool | Tentatively selected | Max view additive budget | Mean audit regional RGB reduction | Final all-support safety |
| --- | --- | --- | --- | --- | --- |
| Panama | 128 | 4 | 0.0009881 | 0.01105% | pass |
| IUI3 | 128 | 5 | 0.00099999 | 0.00639% | fail |

Panama IDs: 1038610, 926803, 202615, 266619.
IUI3 tentative IDs: 640168, 236948, 151993, 498529, 263738.
Both have positive useful audit signed mean changes under the specified floor.
But IUI3 fails the final per-member-support check despite passing union checks.
The exact camera/member failures are recorded in final member_checks. This is
another example of union support diluting localized error. Both hashes match.

## Decision

Do not launch a paired short-training experiment yet. Panama alone passes this
screen, but IUI3's cumulative candidate is invalid. Actual benefits also remain
very small and no geometric preservation is established.

Next improve greedy acceptance to recheck EVERY retained member support after
each addition, rather than just the new support plus union. This must use only
fitting/calibration cameras; do not repair using audit failures. Then evaluate
on another frozen checkpoint or prospectively fixed split. A later short pilot
must preserve a cumulative opacity cap relative to the original model, compare
against an unmodified continuation, and measure rebound, transfer and original
pruning events. This run does not mutate opacities or exercise training/pruning.
