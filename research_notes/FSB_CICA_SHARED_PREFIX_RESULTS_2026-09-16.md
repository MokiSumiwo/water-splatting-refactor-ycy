# FSB CICA 共享前缀与重复对照实验结果

## 结论

8 个共享前缀、32 次续训、全部原测试视图评估均已完成。
精确 FSB CICA 导数的四场景平均 PSNR 变化：旧 renderer 下 **+0.00174 dB**，
修正 renderer 下 **+0.00289 dB**，均小于对应同配置重复差值。
两种 renderer 均未通过实验前登记的收益筛选规则，因此不将该修正作为精度创新推广。

这收窄了上一轮解释：消除 CICA 启用前的轨迹差异后，导数修正的终点效应非常小。
导数一致性有实现上的意义，但当前没有证据表明其能解决 BND ON 的主要指标差距。

## 协议与验证

四场景分别使用物理 GPU 6/7/8/9，conda water_splatting；每场景各训练 R0/R1 的
step-9999 共享前缀，然后从同一个前缀分叉 C0a/C1a/C1b/C0b 到 step 14999。
R0 为旧 backward，R1 为仅替换 backward 的 alpha cap 修正。
C0 为旧 sigmoid 近似导数，C1 为精确 FSB DC 导数。
a/b 是同一状态的执行重复，不是不同 seed 或独立前缀重复。

修正恢复代码，保存并恢复真实 RNG、refinement 累计量、模型、优化器、调度器等状态；
加载替换 Gaussian Parameter 后重新创建优化器，并检查其绑定到当前参数。
短程测试成功跨过 3200 步 split/prune。
正式实验中每个场景/renderer 的四路恢复状态哈希均一致，32 路恢复审计全部通过。
普通 refinement 逻辑仍启用，未靠冻结几何缩小运行波动。

保留原 CICA 的训练梯度 10% lambda 标定策略，因此 C0/C1 对比包括其自动标定响应。
每组 renderer/CICA/重复组合覆盖全部 13 张原测试视图（四场景分别 3/4/3/3），
共完成 104 次终点视图评估，没有基于测试指标筛选终点或重复。
指标先在场景内平均，再做四场景等权平均。

## 结果

| Renderer | 指标 | C0 两次均值 | C1 两次均值 | C1-C0 | 同配置绝对重复差值均值 |
|---|---|---:|---:|---:|---:|
| R0 | PSNR ↑ | 29.840955 | 29.842693 | +0.001738 | 0.004778 |
| R0 | SSIM ↑ | 0.927700 | 0.927706 | +0.000005 | 0.000033 |
| R0 | LPIPS ↓ | 0.120713 | 0.120799 | +0.000086 | 0.000093 |
| R1 | PSNR ↑ | 29.908220 | 29.911113 | +0.002893 | 0.005143 |
| R1 | SSIM ↑ | 0.927424 | 0.927433 | +0.000009 | 0.000034 |
| R1 | LPIPS ↓ | 0.120588 | 0.120524 | -0.000064 | 0.000098 |

重复差值定义为场景内 `(abs(C0a-C0b)+abs(C1a-C1b))/2`，再跨场景平均，
不是置信区间，也不是可推广到所有随机种子的噪声方差估计。

| 场景 | R0 下 C1-C0 PSNR | R0 重复差值 | R1 下 C1-C0 PSNR | R1 重复差值 |
|---|---:|---:|---:|---:|
| Curasao | -0.002535 | 0.002465 | +0.004142 | 0.001388 |
| IUI3 | -0.000730 | 0.004446 | +0.006848 | 0.007242 |
| JapaneseGradens | +0.008389 | 0.001025 | +0.002305 | 0.004973 |
| Panama | +0.001828 | 0.011179 | -0.001725 | 0.006969 |

R0 只有 2/4 场景为正；R1 为 3/4。两种 renderer 的平均收益均未超过重复差值。
R0/R1 本身由不同的训练前缀得到，不能将其绝对均值之差当作本轮严格隔离的 renderer 效应。

## 视觉与资产

所有记录的 `P_J_gt_1` 均为 0。无 clear GT，不能宣称去水恢复准确率提高。
生成了四场景、两种 renderer、两类输出的 16 张拼接图，另有 16 张固定放大 10 倍的
绝对差异拼接图，共 32 张，覆盖所有测试视图及 a/b 重复。
差异图并列展示导数修正差异与同配置重复差异；不挑最好的一次、不调曝光、不调整单图颜色。

40 个 checkpoint 均保留，SHA256 校验通过。训练源码和两个 CUDA 二进制在启动前归档；
启动后发现归档源集合变动会中止新 worker，正式任务均正常完成。
这些新产物存储在 `/mnt/d`，未被同日 outputs/renders 清理删除。

## 完成情况独立复核

清理后再次从原始逐视图 CSV 重算所有场景指标、修正差值和重复差值，结果与正式汇总一致。
40 个任务退出码均为 0；32 路续训各完成 5000 次更新；8 个场景/renderer 分组内的
模型、优化器、调度器、RNG 和 refinement 恢复状态哈希均一致，优化器参数绑定检查通过。
重新计算并通过 40 个保留 checkpoint、509 项源码/评估资产、12 项归档源码/二进制的哈希校验。
本次复核仅使用 CPU，没有追加训练、选择重复或重新调整终点。

[收益与重复差值比较图](../outputs/fsb_cica_fork_repeat_20260916_v1/completion_verification_20260916_v2/psnr_gain_vs_repeat_gap.png)
中蓝柱是 C1-C0，灰柱是观察到的同配置绝对重复差值，不能解释为置信区间。
这支持“不足以证明稳定收益”的判断，不等于证明导数修正没有任何作用。

## 后续判断

停止把 CICA 导数修正视为追赶 BND OFF 指标的主优化路线。
保留其实现正确性开关，不再仅为此扩大训练规模或堆叠 gate。
若继续寻求有意义的收益，应先提出能解释明显剩余误差的独立机制，
并继续使用共享前缀、重复对照和保留 checkpoint 的实验方式。

## 文件

- [实验前协议](FSB_CICA_SHARED_PREFIX_PROTOCOL_2026-09-16.md)
- [完整结果报告](../outputs/fsb_cica_fork_repeat_20260916_v1/summary_20260916/REPORT.md)
- [全部测试视图图库](../outputs/fsb_cica_fork_repeat_20260916_v1/summary_20260916/index.html)
- [逐次指标](../outputs/fsb_cica_fork_repeat_20260916_v1/summary_20260916/metrics.csv)
- [差值与重复对照](../outputs/fsb_cica_fork_repeat_20260916_v1/summary_20260916/contrasts.csv)
- [共享状态审计](../outputs/fsb_cica_fork_repeat_20260916_v1/summary_20260916/pairing_audit.json)
- [保留的 40 个 checkpoint 清单](../outputs/fsb_cica_fork_repeat_20260916_v1/summary_20260916/retained_checkpoints.csv)
- [独立完成复核](../outputs/fsb_cica_fork_repeat_20260916_v1/completion_verification_20260916_v2/verification.json)
- [收益与重复差值比较图（PDF）](../outputs/fsb_cica_fork_repeat_20260916_v1/completion_verification_20260916_v2/psnr_gain_vs_repeat_gap.pdf)
