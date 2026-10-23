# BND ON 后续十个候选的执行筛选记录（2026-09-14）

本轮按“先诊断、后小规模训练、最后完整训练”的顺序审查 SBR、FSB、DCR、TBRD、BOC、MPR、VAB、PGS、CVC、SBD。根据已有代码和实验记录，先做重复性审计，避免把已完成或证据不足的路线重复包装为新方法。

| 顺序 | 候选 | 当前执行状态 | 已有证据/判断 |
|---:|---|---|---|
| 1 | SBR selective bounded relaxation | 未启动新训练 | 需要新的 per-Gaussian 稳定性/饱和率诊断；现有 `BND-PW` 与 `BND-SHSTRUCT` 部分资产缺失，无法直接完成全场景冻结审计。 |
| 2 | FSB frequency-split bounded appearance | 已有实现性证据 | `bounded_headroom_sh3` 已实现并完成数学、零残差、Jacobian 和初始化等价审计；历史 BND-HR 未证明稳定跨场景收益，不能重复宣称新方法。 |
| 3 | DCR dual-capacity radiance | 已有近似尝试 | BRASH/BVAM 已测试 bounded base + residual/专家通道；BRASH learned/reliability 及 BVAM pseudo-target 均未形成稳定四场景收益。 |
| 4 | TBRD teacher residual distillation | 已完成可行性审计 | BND-OFF teacher 与 BND-ON topology 不一致，不能直接蒸馏；严格 teacher 记录已明确停止 UGTR。 |
| 5 | BOC visibility compositing | 已有 transport/ownership 负结果 | DVS 冻结对照显示介质 cutoff 不是 clear 黑洞来源；已有 ownership/compositing 系列未形成安全四场景方案。 |
| 6 | MPR medium-preservation reservation | 已有 medium ownership 系列 | M2、GMVC、MDRR、transmission-gated gain 均显示 medium 归属信号具有场景依赖，尚无可靠动态预算准则。 |
| 7 | VAB view-adaptive bound | 未启动 | 需要 per-view angular support 与 held-out 新视角风险审计；当前没有足够的稳定边界证据支持直接训练。 |
| 8 | PGS photometric gradient shielding | 已有梯度路由审计基础 | OCMC/CICA/BRASH 等路由审计证明模块可隔离梯度，但未证明 Jacobian 投影能提升 BND ON 质量；应先做零步 VJP 诊断。 |
| 9 | CVC visibility-calibrated appearance | 已有跨视角可靠性近似 | GRAC/GCRC/BRASH reliability 结果表明可见性计数不等于颜色可靠性；需要新的前表面稳定性定义，暂不直接训练。 |
| 10 | SBD scene-balanced dynamic bounds | 未启动 | 需要场景级 headroom 预算和跨场景公平性协议；无真实 clear GT，不能用 clear 亮度选择预算。 |

## 当前可执行结论

1. 现有证据不支持立即并行启动十个 15k 训练。多项候选已经由 BRASH、BVAM、BND-HR、GMVC、MDRR、DVS、teacher audit 等实验覆盖，重复训练不能增加有效信息。
2. 第一优先级应是补齐冻结的 per-Gaussian saturation/support/VJP 诊断，并确保所需历史 checkpoint 和配置资产存在。当前 `audit_bnd_aa_eligibility.py` 与 `diagnose_bnd_sh_residual_structure.py` 因缺失历史配置/CSV 无法直接运行，错误已保留在本轮日志中。
3. 只有在诊断确认“可靠 Gaussian 的 BND 饱和与 held-out RGB 误差显著重叠”后，才值得实现 SBR/FSB/CVC 的小规模四场景筛选；否则继续训练会重复已有失败机制。
4. TBRD 必须使用共享 BND-ON topology 的 adapter teacher；当前独立 legacy teacher 不满足条件。BOC/MPR/VAB/PGS/SBD 同样需要先定义不会改变 underwater 调用语义的冻结指标。
5. 本文不声称十个候选已经完成训练，也不声称获得 SOTA。它记录的是按证据边界完成的执行筛选。

## 诊断补充

统一读取四个 BND ON CICA ON 最终 checkpoint 的冻结诊断已完成，输出在 `outputs/bnd_capacity_diagnostic_20260914/`。颜色饱和率（任一 RGB 通道 `<0.01` 或 `>0.99`）为：JapaneseGradens 1.72%、Panama 1.60%、IUI3 1.79%、Curasao 3.68%；接近边界率（`<0.05` 或 `>0.95`）为 2.11%–4.89%。高误差像素区域均具有较高 accumulation/transmission，但当前 renderer 输出没有 pixel-Gaussian contribution matrix，因此尚不能证明这些区域由饱和 Gaussian 造成。

基于该证据，先启动 FSB 四场景正式配对训练（`bounded_headroom_sh3`，CICA ON，step 3000→14999）；SBR、DCR、VAB、SBD 等允许 BND 放宽的候选等待 FSB 结果和 per-Gaussian 贡献映射后再决定。
