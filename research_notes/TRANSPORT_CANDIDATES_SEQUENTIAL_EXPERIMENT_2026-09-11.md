# Sequential Optical Transport Candidate Experiments

User authorized sequential implementation and experiments on 2026-09-11.
Objective: find reproducible improvements and assess comparison with published
methods under compatible protocols. No SOTA claim is justified by local
baseline improvement alone. Existing results and failed attempts are retained.

Environment remains conda water_splatting, physical GPUs 6-9, no dependency
installation and no commit/push. Core changes must be opt-in and separately
validated. Separate experimental CUDA extensions may be compiled with the
already installed CUDA 11.8 toolkit without replacing water_splatting/csrc.so.

## Sequence

| ID | Candidate | Status | Real-data prerequisite |
| --- | --- | --- | --- |
| 1 | Gaussian internal optical path integration | Completed numerical/frozen controls; thickness screen failed 0/4 | Existing covariance, cameras, RGB |
| 2 | Shared world-coordinate segment transport | Completed 12 x 5k medium fits; promotion failed | Existing RGB; identifiability must be measured |
| 3 | Joint refractive projection and water path | Completed synthetic calibration; real data unavailable | Housing geometry or calibration |
| 4 | Forward scattering versus geometric footprint | Completed 8 x 5k PSF fits; promotion failed | Existing RGB; distinguish camera blur |
| 5 | Incident/outgoing illumination transport | Completed synthetic known-light fits; real data unavailable | Constrained or known illumination |
| 6 | Camera radiometric response | Completed synthetic and 8 x 5k real fits; promotion failed | RAW, response or exposure metadata preferred |
| 7 | Sensor spectral integration | Completed synthetic fits; real response unavailable | Sensor response and constrained spectral basis |
| 8 | Polarimetric joint reconstruction | Completed known-signature synthetic separation; real data unavailable | Registered polarization measurements |
| 9 | Clear solution ambiguity range | Completed synthetic and 8 real searches; all 75 full train / 13 test views checked | Existing RGB plus synthetic reference |
| 10 | Active acquisition for decomposition | Completed 100-repeat synthetic comparison; real capture unavailable | Additional observations/capture access |

Missing observations cannot be replaced by arbitrary per-view fitting.
Candidates with missing real data can receive numerical/synthetic validation,
but that does not constitute four-scene or SOTA validation.

## Candidate 1 Registration

First use the existing CRCC E0/E2 final checkpoints on every test view, with
unchanged geometry, colors, alpha compositing and original medium scattering.
Capture the exact projection covariance and final raster arguments.

Under the original clipped perspective Jacobian and 0.3-pixel blur, compute
the Gaussian conditional z-depth mean and variance at each pixel. Compare:

1. Existing center-depth attenuation.
2. Conditional-mean point-depth attenuation.
3. Positive-truncated normal attenuation at conditional mean and variance.
4. Positive-truncated normal at center mean with conditional variance.

Use E[exp(-beta*z) | z >= 0]. The fourth arm isolates thickness from mean
shift. Preserve all original eligibility and early-stop semantics and verify
base direct RGB/clear/accumulation against the native CUDA renderer.

This first experiment isolates direct attenuation. Fixed center sorting and
alpha weights do not model continuous-volume interleaving of thick Gaussians;
do not call it an exact overlapping-volume renderer. Numerical integration
validates the per-contribution expectation only. A subsequent training model
must address ordering before a full physical-renderer claim.

Predeclared screening: thickness-only mean absolute direct-image change must
reach 1e-4 and change more than 1% of image pixels by at least 1/255 in at
least two scenes to motivate a full thickness implementation. These are
practical effect thresholds, not significance or accuracy claims. A large
conditional-mean effect alone does not support thickness integration.

If this screen passes, validate gradients and perform matched short training
before 15k continuation. If it fails, retain the negative result and advance
the sequence. A frozen correction's underwater PSNR is descriptive; learning
benefit and clear accuracy require their own experiments.

## Candidate 1 Result

All 26 E0/E2 final test-view comparisons completed in
`outputs/transport_candidates_20260911/01_optical_path/verified_v2`.
Thickness-only effects fail the registered gate in all four scenes. Every
view has zero pixels with thickness-only direct change above 1/255. The
implementation is retained but this is not a reason to change the training
renderer. Conditional-mean shifts are a distinct projection approximation,
not evidence that finite Gaussian thickness matters in these checkpoints.

Ten numerical/gradient tests passed. Actual CUDA two-Gaussian integration
matched the double reference within 2e-6 and zero-medium arms were exact.
The initial CUDA build used FMA and failed native equivalence. A separate
precision probe found that the installed renderer uses non-FMA behavior;
matching `--fmad=false --prec-div=true --prec-sqrt=true --ftz=false` restored
direct/clear equality. The failed attempt and precision probe are retained.

## Candidate 2 Registration

Use CRCC E0 final snapshots and freeze all Gaussian geometry, opacity and
view-dependent colors. Train replacement medium representations with fixed
native per-ray contribution lists, including every accepted contributor and
the original threshold/early termination. No pseudo-depth, color penalty,
densification or image-space exposure adjustment is introduced.

Compare constant medium (9 parameters), shared 8-cubed world grid (4617),
and camera-conditioned 59-wide two-hidden-layer MLP (4679). All begin at
the same homogeneous medium, derived from training cameras only. The two
spatially varying arms differ by less than 2% in parameter count.

Use 4096 deterministically selected training pixels per camera, batch 256,
Adam lr 0.003 for 5000 updates, seed 42 and the same batch sequence for all
arms. Optimize RGB MSE with frozen Gaussian weights/colors. No heldout GT
enters training, initialization or stopping. This is a replacement-medium
screen, not a full joint Gaussian continuation or SOTA comparison.

Integrate positive direct/backscatter extinction and bounded asymptotic
radiance over 64 uniform z segments using physical ds = ||ray|| dz.
Interpolate cumulative optical depth/scatter at each contributor's depth.
The last medium sample extends to infinity. Compare 64 versus 128 segments
on fixed rays to quantify quadrature error before interpreting the result.

Evaluate all native test images with local PSNR/SSIM/LPIPS/MSE, preserve
baseline outputs, and record train curves/runtime/memory/checkpoints.
Promote only if the grid improves mean PSNR over the matched MLP by at
least 0.05 dB, loses no more than 0.1 dB on any scene, and does not worsen
mean LPIPS. Absolute comparison with the frozen original medium is also
required before suggesting further joint training. Do not select checkpoints
or hyperparameters from heldout results.

## Results Through Candidate 6

Candidate 2 used 12 replacement-medium fits of 5000 steps and all 13 full
test views. World-grid minus matched ray-MLP PSNR was -0.3143, -1.3447,
+0.7320, -2.1671 dB (Japanese, Panama, Curasao, IUI3). Grid minus original
was -0.3086, -1.2481, -2.8567, -2.1483 dB. Promotion failed. Numerical
64-versus-128 segment MAE on sampled rays is approximately 1e-4 to 2.4e-4;
the implementation uses quadrature, not exact spatial volume integration.
This result concerns frozen-Gaussian replacement, not all possible jointly
trained world-medium models. Six operator tests passed.

Candidate 3 fitted air gap and water index from known synthetic calibration
points with fixed glass thickness: heldout coordinate RMSE 3.8e-5 m versus
0.1562 m for the pinhole control. Known housing assumptions and scene metric
scale are essential. This is not a real image renderer or four-scene result.

Candidate 4 trained two two-parameter PSF models per scene for 5000 steps.
Both use the same normalized kernel bank on three native direct depth
layers; one is distance-independent, the other distance-dependent. The
distance-dependent advantage is below 0.001 dB in every scene and promotion
failed. Learned solutions are nearly identity. This convolves visible
direct layers only and does not model hidden radiance or multiple scattering.
Three operator tests passed.

Candidate 5 uses known synthetic source illumination and incident/outgoing
distances. Single-path test MSE is 8.6687e-4; matched dual-path MSE is
1.05e-14. This confirms a deliberately model-correct example only. Real
incident light paths are not measured in the available data.

Candidate 6 first verified the known-sRGB synthetic case. Its real experiment
compares 9 global calibration parameters in each arm with frozen Gaussian
colors and exact native contributions. Each of 8 models trained for 5000
steps. Linear-sRGB minus encoded PSNR is -0.2357, -0.4869, -0.6928,
-2.3455 dB; LPIPS also worsened in all scenes. Promotion failed. This is a
frozen-appearance conversion under an assumed response; it does not reject
calibrated linear image formation with compatible jointly learned colors.
The source PNGs have no EXIF/RAW response evidence. Original outputs remain
linked separately; lower black fractions do not establish clear accuracy.
The runner did not instrument per-arm training wall time or peak memory;
view workflow timings are retained in its logs, so no rendering speed claim
is made from this experiment.

## Candidate 9 Registration

Search two directions of mean clear RGB displacement, solely as ambiguity
probes. Freeze native Gaussian support, geometry, opacity and source model.
Use an augmented 12-parameter family: a global color logit shift bounded
by +/-0.5, RGB direct/backscatter log scales bounded by +/-1, and ambient
RGB logit shifts bounded by +/-1. Colors must be in [0,1]; reject other
checkpoints. Zero parameters reproduce native outputs within 2e-6.

Use the same 4096 training pixels per camera, 32 per camera per update,
Adam 0.003, 1500 steps in each direction, seed 42, raw parameter bounds
[-3,3]. Let epsilon=1/255. Optimize the registered objective
`4*relu(max_view_sample_MSE/epsilon^2 - 0.7)^2 - sign*mean_clear_change/0.05`.
The reference is the original underwater prediction; GT is not an objective.

After search, test parameter scales 1, 0.5, 0.25, 0.125, 0 on ALL pixels
of EVERY training view. Select the largest predefined scale with per-image
RGB RMSE <=1/255 in every training image. Test views play no role in this
selection. Include scale zero as the identity; finding no nonzero feasible
change is an admissible negative result. The per-image constraint allows
larger individual pixel errors, which are reported explicitly.

Evaluate both selected alternatives on all 13 full test views, retaining raw
float arrays, original render links, clear/underwater differences, color
extrema, black fractions, runtime and peak allocated memory. Support cannot
change, and zero-support pixels cannot gain clear color. This finite family
does not establish a global bound, posterior interval, calibrated uncertainty,
or improved reconstruction. It also is not exactly the original MLP/SH
parameter space: the global post-transform family is an explicit augmentation.

## Benchmark Boundary

README.md explicitly states that paper splits differ from Nerfstudio default
splits and follow SeaThru-NeRF. Current screening uses the existing local
CRCC E0 checkpoints and local 3/3/3/4 test views. Published SOTA comparisons
require matching split, preprocessing, camera calibration, image resolution,
training budget and metric implementations. No such reproduced benchmark or
specific user-selected target method exists in this experiment. Local gains
or synthetic fits are not evidence of SOTA or novelty over published methods.

The available SeaFree-GS reference README describes dual-color degradation,
line-of-sight water prediction and pseudo-depth/content losses. That local
reference inspection is not a comprehensive literature or novelty audit.
The current runs use explicit test lists; the README warning about default
splits alone does not prove that these lists differ from the paper lists.
Published-protocol equivalence remains unverified, rather than disproven.

## Results For Candidates 7-10

Candidate 7 used known six-band spectra and a known 3x6 sensor response.
Training distances were below 4 and test distances extended to 6. RGB
single-exponential test MSE was 2.5858e-4; the six-band integral was
3.05e-15. The six-band arm has six extinction parameters versus three in
the RGB arm and receives known spectra. This confirms the numerical model
on a constructed example, not matched-capacity performance or recovery of
unknown spectra from ordinary RGB. Sensor response and spectral constraints
are absent in the current four-scene assets.

Candidate 8 used four ideal polarizer angles, known distinct object/medium
Stokes signatures and Gaussian noise sigma=0.001. The decomposition design
has rank two instead of the rank-one ordinary intensity observation, and
separation RMSE is 0.003242. Unknown polarization signatures remain ambiguous.
No polarized training or true joint reconstruction was possible from the
available ordinary RGB images. Eight shared physical-operator tests passed.

Candidate 9 first demonstrated a finite homogeneous scalar feasible family:
one observation allowed clear values approximately [0,1], while four known
distances narrowed the sampled range to [0.71634,0.72388] around truth 0.72.
This synthetic example is not a Gaussian uncertainty calibration.

The real restricted family then completed 8 searches of 1500 steps. All
eight unscaled candidates passed the registered constraint on all pixels
of all 75 training views. Worst full training-image RGB RMSE was 0.002763,
below 1/255. The largest individual training-pixel channel error was
0.017185, illustrating that this is not a per-pixel error guarantee.
All 13 test views were evaluated only after training-set selection.

| Scene | Lower clear mean displacement | Higher clear mean displacement | Mean test underwater RMSE, lower / higher |
| --- | ---: | ---: | ---: |
| JapaneseGradens-RedSea | -0.004742 | +0.004320 | 0.002462 / 0.002278 |
| Panama | -0.005615 | +0.005877 | 0.002581 / 0.002632 |
| Curasao | -0.005755 | +0.006078 | 0.002283 / 0.002231 |
| IUI3-RedSea | -0.006239 | +0.005511 | 0.002038 / 0.002057 |

These are modest nonunique decompositions in the augmented family. They do
not establish a large unresolved uncertainty interval, removal of artifacts,
or clear accuracy. Geometry and coverage are unchanged. Float32 alternative
renders, difference PNGs, extrema, black fractions, runtime and memory are
retained. Five radiometric/family tests passed, including zero-support and
bounded-endpoint identity checks.

Candidate 10 compared six observations chosen uniformly versus greedy
D-optimal selection from 32 possible distances, with 100 paired noise trials
at sigma=0.01. Clear RMSE was 0.012580 versus 0.010483. Extinction RMSE was
0.037069 versus 0.041277, and background RMSE was 0.025084 versus 0.027216:
the clear improvement does not extend to every recovered parameter. The
design uses the true nominal synthetic parameters and a correct homogeneous
model; robustness to estimated parameters and model mismatch is untested.
No real active capture, new camera observations or hardware experiment ran.

## Final Verification And Decision

- 28 real-data module fits completed 5000 updates each: 12 medium, 8 PSF,
  and 8 radiometric. These are frozen-Gaussian screens, not 28 full 3DGS runs.
- 8 ambiguity searches completed 1500 updates each with complete training
  image certification. They are uncertainty probes, not quality training.
- Candidate 1 evaluated 26 E0/E2 test-view cases; candidates 2, 4, 6 and 9
  each covered every one of the 13 E0 test views across four scenes.
- Synthetic/operator experiments completed for 3, 5, 6, 7, 8, 9 and 10.
  Candidates 3, 5, 7, 8 and 10 still lack real measurement validation.
- 32 focused unit tests passed across the five new test modules. All new
  Python renderer modules and experiment/audit scripts passed compilation.
- Final artifact verification passed: 16 registered file/checkpoint hashes,
  including all 8 source snapshots, matched; PNGs and float32 alternatives
  were readable and finite; native equivalence remained below 2e-6.
- All source model parameters, installed csrc.so and original render assets
  were preserved. Existing dirty worktree changes were not reverted. Failed
  attempts remain alongside the authoritative verified outputs.

Mean scene PSNR differences versus matched controls were -0.77350 dB for
world-medium transport, +0.000117 dB for distance-dependent PSF, and
-0.94024 dB for assumed linear-sRGB conversion. All three promotion gates
failed. Do not advance these implementations to 15k joint training on the
basis of this screen, combine them to obscure negative results, or claim
SOTA. Their restricted design means these results do not rule out every
possible physically calibrated or jointly trained version of the ideas.

Visual review inspected the complete Panama medium sheet, Curasao PSF sheet,
IUI3 radiometric sheet, and Japanese ambiguity sheet. Medium changes are
visible, PSF differences are negligible, and clear artifacts persist in the
radiometric/ambiguity variants. These are qualitative checks, not clear GT.

Artifacts:

- [All-scene visual index](../outputs/transport_candidates_20260911/review/report.md)
- [Final verification](../outputs/transport_candidates_20260911/review/verification.json)
- [Paired score plot](../outputs/transport_candidates_20260911/review/paired_psnr.png)
- [Asset inventory](../outputs/transport_candidates_20260911/asset_inventory.json)

Reproduce with the existing conda water_splatting environment and physical
GPUs 6-9. Each runner rejects an existing output directory; supply a new
`--output-root` for a repeat, preserving this experimental record. No
dependencies were installed and no commit or push was performed.
