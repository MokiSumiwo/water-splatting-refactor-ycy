# BND Object Multi-View Consistency

## CODE FACT

CODE FACT: `clear_object_fullsh_raw` is returned from CUDA `out_clr`, but its backward gradient is not routed in the current rasterizer binding.
CODE FACT: `direct_object_signal` equals `rgb_object` and receives gradients through the rasterizer `out_img` gradient path.
CODE FACT: The core model defaults remain unchanged; OMVC is injected only in `scripts/diagnostics/run_bnd_omvc_panama.py` for O1.

## CONFIG FACT

CONFIG FACT: Panama matched continuation starts from BND-K1@3000.
CONFIG FACT: C0 is standard BND-K1 continuation. O1 differs only by OMVC loss.
CONFIG FACT: OMVC active interval is absolute steps 4000-10000, then lambda is zero.
CONFIG FACT: OMVC target is `direct_object_signal` because `clear_object_fullsh_raw` is not a safe gradient target.
CONFIG FACT: `B_H = 0.007435990124940873`, `B_V = 0.007435990124940873`.
CONFIG FACT: `lambda_omvc = 0.1` selected by the preregistered gradient rule.

## EXPERIMENTAL FACT

EXPERIMENTAL FACT: C0 and O1 use the same start checkpoint, explicit central-camera sequence, and matched initial RNG state.
EXPERIMENTAL FACT: Offline hard-region labels are used only for evaluation, not training.

## QUANTITATIVE RESULT

QUANTITATIVE RESULT: Final dPSNR O1-C0 = `-0.12375640869140625`.
QUANTITATIVE RESULT: Final dSSIM O1-C0 = `-0.0006786386171976355`.
QUANTITATIVE RESULT: Final dLPIPS O1-C0 = `0.0025171091159184728`.
QUANTITATIVE RESULT: Target object consistency relative improvement = `-0.02433009815245554`.
QUANTITATIVE RESULT: Clear-J diagnostic consistency relative improvement = `-0.03415807766879837`.
QUANTITATIVE RESULT: O1 `P(J>1) = 0.0`.
QUANTITATIVE RESULT: Classification = `OMVC_NOT_SUPPORTED`.

## INFERENCE

INFERENCE: This experiment does not claim true geometry, true colors, or full OceanSplat behavior. It tests whether a bounded-object branch consistency intervention improves cross-view object consistency and RGB metrics under BND safety gates.
INFERENCE: O1 did not improve the registered object-consistency mechanism. The target object consistency error increased by `2.433%` relative to C0, and the clear-J diagnostic consistency error increased by `3.416%`.
INFERENCE: O1 also reduced underwater RGB quality: final PSNR changed by `-0.123756 dB`, SSIM by `-0.000679`, and LPIPS by `+0.002517`.
INFERENCE: Hard-region diagnostics moved in the wrong direction. `PERSISTENT_BND_HARD`, `BND_HARD_CORE`, and `M1_HIGH_J` all had higher RGB MSE and higher object-consistency error under O1.
INFERENCE: BND decomposition safety was preserved. O1 had `P(J>1)=0`, `P(c>0.99)=0.016208`, and `P(|s_full|>5)=0.015885`.
INFERENCE: The formal classification is `OMVC_NOT_SUPPORTED`.

## HYPOTHESIS

HYPOTHESIS: If the target-object consistency metric improves without RGB or decomposition harm, the OceanSplat-derived line remains eligible for one further single-factor mechanism test.
HYPOTHESIS: Because the registered OMVC mechanism itself did not improve, continuing OceanSplat-derived depth/alpha mechanisms immediately is not the best next single-factor move.

## Next Single-Factor Recommendation

INFERENCE: The next single-factor experiment should be `SeaFree CB-FG`, a foreground-aware photometric responsibility test, explicitly compared against prior `LOSSRESP` and `UNORM` evidence. Do not combine it with BG supervision, synthetic epipolar depth, depth residual loss, or depth-aware alpha.

## Roadmap

Candidate A: OceanSplat-derived OMVC is the current tested line.
Candidate B: SeaFree CB-FG remains a future mechanism and must be compared against LOSSRESP and UNORM.
Candidate C: SeaFree CB-BG is not supported for Panama with the current locked mask; future priority scenes are Curasao and IUI3 after reusing locked PW audit.
Candidate D: OceanSplat synthetic epipolar depth, depth residual, and depth-aware alpha can only be considered one factor at a time after OMVC.
