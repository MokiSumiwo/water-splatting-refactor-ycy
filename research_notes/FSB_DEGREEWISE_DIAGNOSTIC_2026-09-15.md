# Degree-wise FSB diagnostic (2026-09-15)

Implemented `compute_bounded_degreewise_headroom_gaussian_colors`, which keeps
the degree-0 SH base unchanged and scales the degree-1/2/3 logit increments
before applying the existing asymmetric headroom map. No geometry, medium,
loss, or underwater renderer code is changed.

Two four-scene 5k smoke runs completed successfully:

- D1: `(alpha_1, alpha_2, alpha_3) = (1.0, 0.75, 0.50)`
- D2: `(1.0, 0.50, 0.25)`

For every scene and both schedules, `START_STATE_EQUIVALENCE=true`, forward
outputs were finite, and checkpoint reload hashes matched. The D1 15k run was
started but stopped before completion after process inspection; no D1 formal
metrics are reported or used for promotion. The smoke outputs are retained in
`outputs/fsb_degree_d1_four_scene_20260915_smoke` and
`outputs/fsb_degree_d2_four_scene_20260915_smoke`.

The next valid action is to rerun D1 15k in a fresh output directory with a
single monitored launcher, then evaluate all four scenes and generate the
underwater/clear/difference sheet. Until that finishes, strict FSB remains the
best promoted candidate.
