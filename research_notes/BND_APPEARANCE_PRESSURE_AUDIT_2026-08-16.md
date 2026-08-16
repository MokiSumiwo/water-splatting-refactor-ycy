# BND Appearance Pressure Audit - 2026-08-16

## Scope
CONFIG FACT: This audit is read-only, zero-training, and performs no optimizer step, checkpoint write, CDEPTH/AA training, CB-FG training, OMVC work, or CUDA backward modification.

HYPOTHESIS: Some BND-specific RGB regression pixels may combine positive observed-radiance underfit with high bounded intrinsic appearance pressure.

## Repo
EXPERIMENTAL FACT: The audit ran on branch `research/m1-bounded-intrinsic` at HEAD `c21409b5caaa7662823d9cbf9e29c4a01d68dfa3`.

## Environment
EXPERIMENTAL FACT: `CONDA_ENV=water_splatting`, `PYTHON_PATH=/opt/anaconda3/envs/water_splatting/bin/python`, `TORCH_VERSION=2.1.2+cu118`.

EXPERIMENTAL FACT: `CUDA_VISIBLE_DEVICES=6`; PyTorch logical `cuda:0` mapped to physical GPU `6`, `NVIDIA GeForce RTX 3080`.

## Recovered Historical References
QUANTITATIVE RESULT: LOSSRESP recovered `highj_error_share_mse=0.33781035921730446`, `highj_total_grad_share=0.020002139310155452`, `SEAFREE_SPECIFIC_HYPOTHESIS=NOT_SUPPORTED`.

QUANTITATIVE RESULT: UNORM recovered `STAGE_PSNR_GAIN=-0.04113515218099195`, `HIGH_J_MSE_GAP_RECOVERY=0.511933552865014`, `RGB_SAFETY=False`.

QUANTITATIVE RESULT: AA recovered `AA_PSNR_GAIN=0.3043301900227853`, `HIGH_J_MSE_GAP_RECOVERY=0.271191118170705`, `MECHANISM_SUPPORT=NOT_SUPPORTED`, `RGB_SAFETY=False`.

QUANTITATIVE RESULT: CDEPTH recovered `CDEPTH_PSNR_GAIN=0.25494639078775805`, `HIGH_J_MSE_GAP_RECOVERY=0.3839534834123003`, `Hypothesis=PERFORMANCE_EFFECT_WITHOUT_GEOMETRY_EVIDENCE`, `RGB_SAFETY=False`.

## Signed BND Residual Direction
CODE FACT: `R_PLUS=mean_rgb(max(I_GT-I_BND,0))`; `R_MINUS=mean_rgb(max(I_BND-I_GT,0))`.

QUANTITATIVE RESULT: On train pixels with `delta_e_BND>0`, `R_PLUS mean=0.0026982298586517572`, `median=0.00018701453518588096`; `R_MINUS mean=0.004135119263082743`, `median=0.0029114484786987305`.

QUANTITATIVE RESULT: Train Spearman with `positive_delta_e_BND`: `R_PLUS=-0.07677031583350796`, `R_MINUS=0.2708860631482067`.

INFERENCE: The formal train population does not support positive-radiance underfit as the primary signed direction of BND-specific regression.

## BAP Definition
CODE FACT: `BAP(p)=R_PLUS(p)*J_MAX(p)`.

CODE FACT: `J_MAX=max_rgb(outputs["clear_object_fullsh_raw"])` is a bounded intrinsic / dewatered proxy diagnostic, not true color.

CONFIG FACT: The only candidate mechanism score is BAP. Controls are `R_PLUS`, `J_MAX`, `FAW`, and raw darkness.

## BAP vs BND Regression
QUANTITATIVE RESULT: Train Spearman with `positive_delta_e_BND`: `BAP=-0.08627321436011615`, `R_PLUS=-0.07677031583350796`, `J_MAX=0.15833438475636777`, `FAW=-0.20710458796368403`, `darkness=-0.21145816107539842`.

QUANTITATIVE RESULT: BAP top-10/top-20/top-30 positive-regression enrichment was `1.181919294208339`, `1.0580014678983176`, `0.9935690605793923`.

QUANTITATIVE RESULT: BAP top-10/top-20/top-30 positive-excess concentration was `7.501175740642932`, `3.823292912323302`, `2.5740656327828777`.

## Control-Signal Comparison
QUANTITATIVE RESULT: Mean top-quantile positive-regression enrichment: BAP `1.0778299408953496`, R_PLUS `1.14386904745864`, J_MAX `1.0502130285317202`, FAW `0.9044926708839065`, darkness `0.9048415150042942`.

INFERENCE: BAP is better aligned than FAW/darkness on the train top-quantile enrichment diagnostic, but it does not add information beyond R_PLUS-only and J_MAX-only controls.

## Hard-Region Alignment
QUANTITATIVE RESULT: Train `M1_HIGH_J` coverage `0.051671340454998764`, BAP enrichment `8.546723533480305`, BAP top-10 overlap `0.4731692648744015`, mean `delta_e_BND=0.0007148287841118872`.

QUANTITATIVE RESULT: Train `PERSISTENT_BND_HARD` coverage `0.06497456771365778`, BAP enrichment `8.449364540922664`, BAP top-10 overlap `0.602069445012474`, mean `delta_e_BND=0.0006463516619987786`.

QUANTITATIVE RESULT: Train `BND_HARD_CORE` coverage `0.023231135245991328`, BAP enrichment `17.26672068859654`, BAP top-10 overlap `0.7072475548115172`, mean `delta_e_BND=0.0016121367225423455`.

INFERENCE: BAP overlaps the registered hard regions, but this hard-region alignment is insufficient because the signed residual direction and component-ablation criteria fail.

## Sigmoid Capacity Probe
EXPERIMENTAL FACT: Probe status `PARTIAL_CHAIN_RULE_FROM_EXPOSED_LOGITS`; non-detached `gaussian_view_logits` was available on the normal underwater RGB forward path, and `dL/dc` was reconstructed by chain rule from `dL/ds_full` and `c(1-c)`.

CONFIG FACT: Probe view `MTN_1538`; BAP-high was top 20% valid pixels by BAP; control was equal-count lowest-BAP valid pixels excluding BAP top-20.

QUANTITATIVE RESULT: BAP_TOP20 `median c=0.33976560831069946`, `median c(1-c)=0.21295015513896942`, `P(c>0.9)=0.02328734154785235`, `P(c>0.99)=0.01474328966890675`, `||dL/ds||/||dL/dc||=0.20269476583445745`.

QUANTITATIVE RESULT: Control `median c=0.29430532455444336`, `median c(1-c)=0.20099641382694244`, `P(c>0.9)=0.012323634855807161`, `P(c>0.99)=0.00635409901033697`, `||dL/ds||/||dL/dc||=0.1757802533316696`.

INFERENCE: BAP-high has a larger saturated tail, but median sigmoid derivative and the chain-rule gradient ratio are not reduced relative to control. This is not strong evidence for reduced bounded-intrinsic optimization leverage.

## Gradient Responsibility
QUANTITATIVE RESULT: BAP_TOP20 aggregate grad L2: object appearance `0.00020619172402023072`, object geometry `0.08404450016911157`, medium `0.10792574187393351`.

QUANTITATIVE RESULT: Control aggregate grad L2: object appearance `0.0002851629451704866`, object geometry `0.06868346397872131`, medium `0.12132490691137801`.

EXPERIMENTAL FACT: `parameter_delta_max_abs=0.0`.

INFERENCE: The no-step masked-loss probe does not show a clean BAP-high shift toward bounded object appearance responsibility.

## AA / CDEPTH Recoverability Alignment
QUANTITATIVE RESULT: Train/eval Spearman(BAP, positive recovery_AA) was `0.00411499555592446` / `0.08612168728877904`.

QUANTITATIVE RESULT: Train/eval Spearman(BAP, positive recovery_CDEPTH) was `-0.009703103350245714` / `0.022042815711290367`.

QUANTITATIVE RESULT: Eval Spearman controls for AA: `R_PLUS=0.09495952706893401`, `J_MAX=0.11444414453057425`, `FAW=-0.11406818857019031`, `darkness=-0.11162298453362854`.

QUANTITATIVE RESULT: Eval Spearman controls for CDEPTH: `R_PLUS=0.032336982052305584`, `J_MAX=0.015079195046920754`, `FAW=-0.03690386250247908`, `darkness=-0.03656777125426241`.

INFERENCE: BAP has weak retrospective recoverability alignment and is not consistently better than the R_PLUS/J_MAX component controls.

## Classification
INFERENCE: `BAP_NOT_SUPPORTED`.

INFERENCE: Primary causes are negative train signed association for R_PLUS, BAP Spearman below zero on train, and no added value beyond R_PLUS/J_MAX.

RECOMMENDATION: `CLOSE PANAMA LOSS-RESPONSIBILITY LINE`.

## Outputs
EXPERIMENTAL FACT: Tables and manifests were written under `outputs/bnd_appearance_pressure_audit_20260816/` and are intentionally not committed.
