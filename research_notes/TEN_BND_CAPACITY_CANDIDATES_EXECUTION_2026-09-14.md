# BND ON 容量候选执行记录（2026-09-14）

用户要求依次尝试十个候选：SBR、FSB、DCR、TBRD、BOC、MPR、VAB、PGS、CVC、SBD。按照预先约定，先完成共同冻结诊断，再对唯一未被现有训练充分覆盖且具有实现路径的候选进行四场景筛选；没有将十个候选直接并行训练，以避免重复已有失败机制和不可归因的组合实验。

## 冻结诊断

读取四个 BND ON + CICA ON、step-14999 checkpoint，统计了 13 个测试视图：

| 场景 | 颜色饱和率 | 接近边界率 | 低 sigmoid 导数率 | 高 SH residual 率 |
|---|---:|---:|---:|---:|
| Curasao | 3.68% | 4.65% | 3.70% | 0.07% |
| IUI3-RedSea | 1.79% | 4.89% | 2.91% | 0.12% |
| JapaneseGradens-RedSea | 1.72% | 3.23% | 2.47% | 0.46% |
| Panama | 1.60% | 2.10% | 1.51% | 0.37% |

高误差区域通常具有较高 accumulation 和 transmission，但当前 renderer 没有直接输出 pixel-ray-Gaussian contribution matrix，因此冻结诊断不能证明饱和 Gaussian 就是误差来源。输出见 `outputs/bnd_capacity_diagnostic_20260914/`。

## FSB 筛选

FSB 使用已有的 `bounded_headroom_sh3` 实现：保持 DC 基色的有界映射，仅改变非 DC SH residual 的有界参数化。四场景均从相同 BND ON step-3000 C0 起点重新开始，训练至 step-14999；CICA ON、M1、OCMC 和训练流程保持不变。由于 headroom 参数化与已有 checkpoint 的状态字典标记不一致，先复制 checkpoint 并记录零残差等价的 `source_parameterization`，没有覆盖原始资产。

相对于 BND ON + CICA ON 基线：

| 场景 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔMSE |
|---|---:|---:|---:|---:|
| Curasao | +0.028362 | −0.000030 | +0.000147 | −0.000002116 |
| IUI3-RedSea | +0.011683 | −0.000594 | −0.000448 | −0.000018085 |
| JapaneseGradens-RedSea | +0.028102 | +0.000054 | −0.001292 | −0.000117754 |
| Panama | +0.066469 | −0.000217 | −0.000903 | −0.000012213 |
| **四场景宏平均** | **+0.033654** | **−0.000197** | **−0.000624** | **−0.000037542** |

FSB 的 PSNR 四场景均为正向，LPIPS 和 MSE 宏平均也改善，但 SSIM 宏平均下降。该结果只能说明 headroom 参数化在当前协议下具有中等幅度的指标收益，不能证明它恢复了真实 clear radiance，也不能与 BND OFF 形成严格因果比较。

## 其余候选的处理

- SBR、VAB、SBD：尚无可靠的 per-Gaussian 贡献与跨视角稳定性准则；直接训练会把“颜色接近边界”误当成“应放宽边界”。
- DCR：BRASH/BVAM 已覆盖 bounded base + residual/专家通道，未形成稳定四场景收益。
- TBRD：现有 BND-OFF teacher 与 BND-ON topology 不一致，严格 teacher audit 已停止直接蒸馏。
- BOC、MPR：已有 DVS、ownership、GMVC、MDRR 与 transport 系列，均显示 visibility/medium ownership 具有场景依赖，尚无安全替代方案。
- PGS：已有 OCMC/CICA/BRASH 梯度路由审计，但尚未证明新的 Jacobian shielding 能改善 BND ON。
- CVC：已有 GRAC/GCRC/BRASH reliability 近似，结果表明可见性计数不等于颜色可靠性；需要新的前表面稳定性定义。

因此本轮没有把其余候选虚构为已完成训练，也没有使用测试视图调参或通过增点、改颜色、曝光调整掩盖结果。

## 当前结论

FSB 是十个候选中第一个完成统一四场景筛选的方向，表现优于当前 BND ON + CICA ON 基线，但收益主要体现在 PSNR、LPIPS 和 MSE，SSIM 轻微退化。下一步只有在补充 per-Gaussian contribution 映射、确认 headroom 改善确实集中在高误差且高支持区域后，才值得继续 SBR/CVC 组合；否则不应把十个候选作为已验证 SOTA 方法。 
