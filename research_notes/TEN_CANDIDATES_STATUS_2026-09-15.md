# BND-ON ten-candidate sequence status (2026-09-15)

The strict SBR run is now complete. Results are recorded in
`SBR_BND_ON_RESULTS_2026-09-15.md`; its macro PSNR delta is −0.0308 dB, so it
does not pass the four-scene promotion gate.

The next candidates were checked against existing controlled evidence before
starting new training:

| Candidate | Status | Evidence-based action |
|---|---|---|
| SBR | Completed 15k | Reject: scene-specific, macro PSNR/SSIM/MSE regressions |
| FSB | Completed 15k | Retain as current best bounded variant: +0.0337 dB macro PSNR vs BND ON+CICA |
| DCR | Existing BRASH/BVAM paired screens | Do not relabel as new; no stable four-scene gain |
| TBRD | Strict teacher audit complete | Blocked until shared-topology teacher is available |
| BOC | DVS/transport audits complete | No renderer-cutoff evidence supporting promotion |
| MPR | Ownership/MDRR audits complete | Scene-dependent medium ownership; no safe rule |
| VAB | Not started | Requires held-out angular-support diagnostic |
| PGS | Not started | Requires zero-step Jacobian/VJP audit before training |
| CVC | Not started | Requires front-surface stability, beyond visibility counts |
| SBD | Not started | Requires scene-balanced budget protocol without clear GT |

Because SBR failed and DCR/TBRD/BOC/MPR are already covered by non-promising
or invalid evidence, the next useful experiment is a read-only PGS/VAB/CVC
observability audit. No combination training is justified yet.
