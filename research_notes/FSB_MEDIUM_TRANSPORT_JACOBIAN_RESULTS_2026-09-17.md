# FSB 介质传输 Jacobian 四场景结果

完成 75 张训练图、全部 13 张测试图，物理 GPU 6/7/8/9、conda water_splatting。模型冻结，只用训练水下 RGB 拟合每场景 9 个全局介质输出校正系数：逐通道 beta_D、beta_B 对数缩放和 B_inf logit 偏移。

## 结论

**不进入远景专用 transport-only 训练。** 没有场景同时满足 far_fit 的测试全图 PSNR 与稳定远景 RMSE 改善，预登记要求至少 3/4。这不等于介质完全不可改进：全图拟合 all_fit 的四场景平均 PSNR 增益约 +0.08155 dB，且 raw/display clear 完全不变，应保留为校准对照。

| 场景 | 全图拟合 ΔPSNR | 全图拟合远景 RMSE 变化 | 远景拟合 ΔPSNR | 远景拟合远景 RMSE 变化 |
|---|---:|---:|---:|---:|
| Curasao | +0.28911 dB | +4.56% | +0.29021 dB | +5.62% |
| IUI3-RedSea | +0.00169 dB | -0.14% | -0.00193 dB | -0.10% |
| JapaneseGradens-RedSea | +0.00974 dB | -0.52% | -0.00558 dB | -0.37% |
| Panama | +0.02565 dB | -5.26% | -0.05272 dB | -3.41% |

RMSE 变化为负才改善。场景 PSNR 为原生逐视图 PSNR 平均；远景 RMSE 由冻结候选 mask 下的同格点 SSE 合计计算，不能互相替代。

| 方法 | 四场景等权 PSNR | ΔPSNR | ΔSSIM | ΔLPIPS |
|---|---:|---:|---:|---:|
| baseline | 29.84226 | +0.00000 | +0.000000 | +0.000000 |
| all_fit | 29.92381 | +0.08155 | -0.000228 | +0.000041 |
| far_fit | 29.89975 | +0.05749 | +0.000081 | +0.000031 |

## 机制发现

1. **少量介质校正确实可以改变带水体图像而精确保留当前 clear。** 每次扰动和非线性校正均验证了 clear、accumulation、depth 不变；不是仅因为“参数未训练”就假定 clear 不变。这个结论只针对当前生产 forward 和本轮正介质扰动范围。
2. **全图改善与远景改善不是同一个目标。** Curasao far_fit 全图 PSNR +0.29021 dB，但稳定远景 RMSE +5.62%；按 stride-4 SSE 分账，稳定表面集合的误差略增，收益来自其补集。补集含低覆盖、深度分散等区域，不能直接命名为水域或真实物体。
3. **平均 PSNR 的正增益也不保证平均 MSE 下降。** Curasao all_fit 平均 PSNR +0.28911 dB，但平均 MSE 从 0.000715058 增至 0.000717687；首个测试视图 -0.27762 dB，另两个 +0.26924/+0.87571 dB。该现象来自逐视图对数指标的聚合，不能隐去逐视图退化。
4. **Panama 是全图校准的局部正信号。** all_fit 的全图 PSNR +0.02565 dB、远景 RMSE -5.26%、LPIPS -0.000151，但近景 RMSE +0.26%，仍有两个测试视图轻微 PSNR 下降。far_fit 虽远景 RMSE -3.41%，全图却 -0.05272 dB。
5. **IUI3/JapaneseGradens 的可解释残差较少。** all_fit 训练线性 MSE 分别仅改善 1.59%/0.19%；JapaneseGradens 留一训练视图线性 MSE 反而 +1.43%。全图测试 PSNR 增益只有 +0.00169/+0.00974 dB，不能据此认为解决了主要瓶颈。
6. **本轮不存在此前误报的系数饱和。** 修正后两组四场景系数裁剪均为 0。远景正规方程的条件数较大、跨视图稳定性较弱；Curasao far_fit 留一训练视图线性 MSE +5.24%，Panama 为 -5.41%。方向可辨识性和实际泛化均因场景而异。
7. **没有改善去水体远景。** Clear 数值保持完全相同；带水体残差投影不能单独证明介质参数正确，也无法区分原模型中的几何/外观混淆。上一轮 BND 区间可达诊断只是每像素乐观下界，不能推出存在可训练的有界颜色解。

## 下一步建议

- 暂停只优化远景 mask 的全局介质训练，也不放宽本轮准入规则。
- 将 all_fit 作为“保留 clear 的终点介质校准”对照，在已有 R0/C0b 同配置重复 checkpoint 上按相同规则复核；不按测试图筛选 all_fit/far_fit 或调整参数界限。该复核尚未执行，不能称为跨重复稳定收益。
- 如果校准收益可复现，再用训练内部留视图选择有约束的全图更新，要求非线性训练全图误差不升、近/远区域均无明显回退。小规模验证优先于修改整个介质网络。
- IUI3/JapaneseGradens 远景 clear 模糊的下一独立诊断应仍放在高深度分散/多层混合与方向外观上；当前 9 维介质 family 的解释能力不足，不能靠叠加相同 reweighting 解决。

## 实现与核验

v1 归一化解漏还原列尺度，并存在 clamp、区域和指标口径错误，已标记 INVALID_RESULT。此前关于“饱和”和大幅降质的判断撤回，旧目录保留以便审计；本报告只引用 v2。

修正后已验证：正规方程求解与独立 ridge 求解一致；原基线输出地图最大差异 0；全部测试 PNG 与原 PSNR/SSIM/LPIPS/MSE 一致；clear/support 改变 0；模型及输入哈希不变；0 次 backward/optimizer update。有限差分步长减半相对差异最大约 1.92%，出现在 JapaneseGradens；该场景存在输出裁剪，本轮未将两者做逐像素因果归因。最终指标使用真实非线性 renderer，不依赖线性预测替代。

没有安装依赖、改生产 renderer、写新 checkpoint 或 commit/push；新结果与历史资产分开。

## 结果入口

- [完整报告](../outputs/fsb_medium_transport_jacobian_20260917_v2/summary/REPORT.md)
- [全部 13 张测试图拼接](../outputs/fsb_medium_transport_jacobian_20260917_v2/summary/index.html)
- [四场景总览](../outputs/fsb_medium_transport_jacobian_20260917_v2/summary/four_scene_overview.png)
- [全图及分区变化图](../outputs/fsb_medium_transport_jacobian_20260917_v2/summary/paired_metrics.png)
- [四场景指标 CSV](../outputs/fsb_medium_transport_jacobian_20260917_v2/summary/scene_metrics.csv)
- [逐视图增减](../outputs/fsb_medium_transport_jacobian_20260917_v2/final_verification/per_view_changes.csv)
- [区域误差分账](../outputs/fsb_medium_transport_jacobian_20260917_v2/final_verification/spatial_gain_accounting.csv)
- [准入判定](../outputs/fsb_medium_transport_jacobian_20260917_v2/summary/decision.json)
- [协议](FSB_MEDIUM_TRANSPORT_JACOBIAN_PROTOCOL_2026-09-17.md)
