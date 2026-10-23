# Far-field direct training, 2026-09-17

User explicitly requests implementing/training CDRC, LSPR, MST directly, overriding
the earlier CDRC diagnostic admission rule. No scientific diagnostic is required
before training. Numerical correctness, paired controls and evaluation still apply.

## Common training protocol

Four scenes on physical GPU 6/7/8/9, conda water_splatting. No dependency installs,
no commit/push, no overwriting old assets. Frozen evidence: prior R0/C0a FSB+CICA
step 14999, with exact model/optimizer/scheduler/RNG/runtime restore and unchanged
OCMC projector. Each candidate starts from that same checkpoint, not a previous
candidate. Continue 2000 updates (15000..16999); deterministic shared camera order.
Baseline A and B repeat the same continuation. Normal refinement callbacks remain.
CICA coefficient is inherited; its target bank refreshes at start/every 500 updates
and after population change, identically in all arms. Renderer R0, legacy CICA,
FSB bounded headroom are kept common. Results concern continuation from this state,
not training from scratch or a SOTA benchmark.

All original 13 evaluation views: full image PSNR/SSIM/LPIPS and unclamped clear
range. Also fixed near/mid/far masks, clear deviations/edges/black fraction as
visual diagnostics, not clear accuracy. Compare against paired continued baseline,
and separately display initial checkpoint. Equal-weight scene averages. Preserve
endpoint checkpoints and native underwater/clear images. No testing-derived source
matching, stopping, hyperparameters or checkpoint selection.

## CDRC implementation

Use existing independently verified train/train cross-distance correspondences;
do not relax matching thresholds or use test RGB as supervision. On every fourth
update, cycle deterministically through matched view pairs. Keep FSB color bounded.
From differentiable native rendering compute `S=sum(w*t)`, `D=sum(w*t*c)`,
`J_eff=D/max(S,0.05)` and additive `M`. Minimize reciprocal forward Huber error:

`0.5 * [rho(S_far*J_near+M_far-I_far) + rho(S_near*J_far+M_near-I_near)]`.

`rho` is Huber delta .05 divided by .05. The loss weight is .05; add its gradients
only to appearance and medium groups, clipped in joint norm to at most .1 times
the corresponding existing update gradient norm. Geometry retains base gradients.
Current A>=.9, CV<=.1 checks prevent training on correspondences made stale by
normal refinement. No `(GT-M)/t` pseudo-radiance, no unbounded color injection.

`J_eff` is transmission-weighted effective color, not exact clear radiance of a
single physical surface. This shared-color approximation can fail for mixed layers,
angular appearance or lighting. The legacy rasterizer ignores the clear buffer's
upstream derivative; hence this implementation explicitly uses differentiable
object and unit-color RGB, not a loss on that unsupported clear buffer.

## LSPR and MST

The currently available request/history and repository do not contain their full
mechanism definitions. Clarification was requested while CDRC implementation and
training proceed. These names will not be silently assigned unrelated mechanisms.
Their exact implementation and any protocol extension must be recorded before
those runs, without changing the completed CDRC or baseline assets.

### Execution correction, before successful CDRC continuation

The first BASE_A run completed. CDRC then exhausted GPU memory before 100 updates:
`unittest.mock` call histories retained full render graphs. Replace interception
with a `try/finally` method wrapper and collect unused graphs/cache every 25 updates.
No objective, gradient weight, data order or initial state changed. Failed logs stay
in `far_field_direct_training_20260917_v1`; successful reruns and BASE_B use a new
`v2` directory, linking the completed BASE_A results for comparison. A 120-update
Panama run crosses refinement and confirms the graph retention issue is resolved.
