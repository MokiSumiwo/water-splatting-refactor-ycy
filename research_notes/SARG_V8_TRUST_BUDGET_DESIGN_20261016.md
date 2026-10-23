# SARG-v8 native-evidence trust-target training

## Objective and hypothesis

Train fresh representations rather than imposing deletion on a completed v6
model. Prior diagnostics show both object/medium entanglement and narrow safe
attenuation intervals. V8 tests whether early, finite opacity target attraction
lets native geometry, colour and medium learning redistribute responsibility
without a monotonically decreasing opacity penalty. This is not a learned
two-colour physical decomposition or a claim of semantic water identification.

## Protocol

Run: `research_runs/sarg_v8_trust_budget_panama_iui3_15k_20261016/`.
Fresh COLMAP initialization for Panama and IUI3-RedSea, seed42, 15000 updates,
corrected_v1+BND+M1+OCMC+CICA, original camera sequence/optimizers/topology.
No historical checkpoint is loaded. 2-step fresh smoke runs passed both scenes;
Panama smoke also exercised nonzero target pressure. Formal training uses a
separate fresh directory and first refresh at1200, then every1000 and endpoint.
Protocol source hashes include v8.py. Final native evaluation and postanalysis
are queued in finalize.py; training completion alone is not an efficacy result.

## Mechanism

1. At refresh, attribute native clear RGB to each Gaussian over fitting-view
   supported far blue/green pixels. Validate relative attribution conservation
   <3e-5. This descriptive support is not a water label. Normalize independently
   by each view's regional RGB sum. Persist detached evidence EMA (.7 old/.3 new)
   through row-aligned topology state; new children have zero evidence.
2. Preserve anchor ellipsoid protection and >=3 fitting-camera visibility.
   Candidate mean depth must exceed the 30th percentile of sufficiently visible
   Gaussians. Rank by equal current contribution and EMA score. Top128 enter
   spatial microgroup construction, four members maximum, cell size .009 anchor
   extent. Attempt top32 groups. Recompute after each topology-dependent refresh.
3. A new candidate target is 90% of first-admitted opacity in the reference
   epoch; repeated refreshes cannot repeatedly multiply the reference by .9.
   If native learning has already lowered alpha below the reference target, use
   current alpha for the new target. This is a finite soft target, not a hard
   clipping constraint on the native optimizer.
4. Per fitting-camera additive attenuated contribution budget <=.001; at most32
   accepted members. Trial full native cumulative rerenders in up to7 fitting
   probes and ALL every-fourth training calibration cameras. Test each retained
   microgroup's full-deletion support separately after EVERY addition, avoiding
   union-region dilution. Support is fixed before attenuation, >1e-5 clear change
   with 5x5 dilation. 32px tile L1/MSE gates use inherited1% floors/caps. All must
   pass. Cumulative target-region signed mean clear decrease >=1e-5 in >=3 fitting
   probes is required. Not all fitting cameras receive counterfactual gates;
   all contribute ranking and the budget. Test cameras never enter selection.
5. Only currently revalidated candidates receive symmetric squared attraction
   to target, normalized by max(.1 reference alpha,.01), ramp800. Calibrate
   gradient norm to10% of native candidate opacity gradient (weight clamps
   [2e-5,.005]). This loss routes only to opacity. Native colour/geometry/medium
   losses remain active; rejected candidates lose pressure immediately.
6. Preserve original densification veto for currently admitted candidates and
   original pruning. Native opacity reset clears evidence/reference/targets,
   beginning a new explicitly observable reference epoch. Targets are not a
   lifetime opacity lower bound. Native pruning is not a geometry guarantee.

## Deliberate differences from v5/v6

V8 does NOT use binary full-deletion admission, same-ray medium-replacement
thresholds or neutral-colour penalties. Native appearance and medium learning
can respond to finite opacity pressure but there is no extra appearance branch.
This choice follows the negative fixed-geometry neutral-colour control and must
not be described as verified physical disentanglement. Legacy parameter fields
remain for source compatibility; v8.py specifies the active gates. Legacy
individual/joint segmented files are empty compatibility artifacts, not zero
candidate findings; use v8_tests.json and refresh summary for actual admission.

## Evaluation and limits

Compare native PSNR/SSIM/LPIPS and fixed-baseline far residual colours to baseline,
v5 and v6; inspect images, anchor support, endpoint removal and training role
history. Fixed-mask colour statistics do not establish true water removal. Only
two scenes and one seed. Counterfactual safe-at-refresh checks do not guarantee
safety after subsequent optimizer/topology changes. EMA is evidence persistence,
not independent multi-view semantic certainty. Any negative result must be
reported rather than compensated by threshold tuning during the formal run.
