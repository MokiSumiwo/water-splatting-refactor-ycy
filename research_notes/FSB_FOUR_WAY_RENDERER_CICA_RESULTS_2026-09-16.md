# FSB 四路 Renderer/CICA 配对实验结果

## 结论

四场景、四路共 16 次训练及全部 13 个测试视图的评估已完成。
本次接续发现 `four_way_fsb_renderer_cica_strict_20260916_v4` 已有完整结果，
因此复用这些结果，补做证据核查和可视化，没有重复训练。

**当前没有证据表明两项导数修正能稳定提升 FSB 的带水新视角 PSNR。**
双修正相对原版平均 PSNR 为 **-0.01561 dB**，SSIM 为 **+0.000331**，
LPIPS 为 **-0.000607**。修正改善了导数一致性，但不能据此宣称达到 SOTA，
也未证明缩小 BND ON/OFF 差距。

## 实验设置

| 分支 | Renderer backward | CICA DC 导数 |
|---|---|---|
| R0C0 | 历史版本 | 历史 sigmoid 近似 |
| R1C0 | alpha cap 从 0.99 对齐至 0.999 | 历史 sigmoid 近似 |
| R0C1 | 历史版本 | FSB 实际解析导数 |
| R1C1 | alpha cap 对齐 | FSB 实际解析导数 |

- 四路均为 BND ON、FSB alpha=1、SH3、M1+OCMC+CICA；辅助 appearance 关闭。
- Curasao、IUI3、JapaneseGradens、Panama 分别使用物理 GPU 6、7、8、9。
  同一场景的四路依次运行，四个场景并行；均使用 conda `water_splatting`。
- 同一场景共享 step-3000 源 checkpoint、模型/优化器/调度器状态、seed 42、
  相机序列和 CICA 六视图训练 bank。续训 11999 次更新，终点固定为 14999。
- CICA 从 10000 启用，10000/12000/14000 刷新。保留原有训练梯度 10% 标定规则，
  各路的实际 lambda 不完全相同，因此比较包含自动标定的响应。
- 四路 forward 都指向原 `water_splatting.csrc`；R1 仅替换两个 backward 入口。
  本轮实际训练走 RGB 路径。
- 全部原测试视图为 3/4/3/3 张；四场景汇总采用场景等权平均。
  指标来自训练后原始 float tensor 评估，没有用 PNG 重新计算替代。

## 指标

| 四场景平均 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 相对 R0C0 的 PSNR |
|---|---:|---:|---:|---:|
| R0C0 | 29.92068 | 0.927656 | 0.120903 | 0 |
| R1C0 | 29.87930 | 0.927653 | 0.120285 | -0.04139 |
| R0C1 | 29.86167 | 0.927616 | 0.120320 | -0.05901 |
| R1C1 | 29.90507 | 0.927987 | 0.120296 | -0.01561 |

| 场景 | R0C0 PSNR | R1C0 PSNR | R0C1 PSNR | R1C1 PSNR | 双修正变化 |
|---|---:|---:|---:|---:|---:|
| Curasao | 32.51175 | 32.49305 | 32.42306 | 32.56958 | +0.05783 |
| IUI3 | 30.76487 | 30.75879 | 30.70742 | 30.72797 | -0.03690 |
| JapaneseGradens | 24.85131 | 24.74375 | 24.75281 | 24.83457 | -0.01674 |
| Panama | 31.55480 | 31.52159 | 31.56338 | 31.48816 | -0.06664 |

四场景 PSNR 的描述性 factorial 对比：renderer 主效应 +0.00101 dB，
CICA 主效应 -0.01662 dB，交互项 +0.08479 dB。
交互项只说明组合结果不是两个单独变化的简单相加；不能据此证明互补机制，
因为只有一次训练且存在下述轨迹波动。

## 严格配对的证据边界

起始模型、优化器、调度器、RNG、OCMC projector 和初始 forward 的哈希在每个场景内完全一致。
相机序列、训练 bank、配置和测试 GT 一致；16 路起始状态及梯度路由审计均通过。
本次重新检查了源 checkpoint 的 SHA256。

**但同一 renderer 的 C0/C1 两路，在 CICA 尚未启用前已经出现轨迹差异。**
3001 步的训练 PSNR 和 Gaussian 数量相同，3500 步后已有差异。
因此可以确认“实验起始条件严格配对”，不能声称“完整训练轨迹逐位可复现”。
CUDA 并行归约及 densification 对微小误差的放大是可能原因，本轮没有独立隔离原因。
这些小幅终点变化混合了方法效应和运行波动，不能报告为显著或稳定收益。

此外，v4 没有归档不可变的训练脚本快照或二进制哈希，当前源码和补充审计无法追溯证明
当时执行的每一行实现。run manifest 保存了使用的开关、后端路由、源状态与运行参数。

上一轮启用了 `compact-output`，在成功评估后删除了 16 个临时终点 checkpoint。
原始指标、逐视图结果、日志、PNG 和 checkpoint 哈希仍在，但无法独立重新渲染这些终点。
本次未删除任何资产。后续若继续训练，应保留终点和必要的恢复状态，可使用空闲的 `/mnt/d`。

## Backward-only 补充审计

补充审计在 GPU 6–9 执行，未更新任何参数。使用仍保留的历史 BND+CICA、FSB checkpoint，
每场景 2 个模型各 6 个训练视图，共 48 例；它不是 v4 训练前的预注册门槛，
也不应冒充 v4 已删除终点的复验。

| 场景 | 旧颜色 VJP 平均相对 L2 | 修正后平均相对 L2 | forward 最大差异 |
|---|---:|---:|---:|
| Curasao | 0.91535% | 0.000345% | 0 |
| IUI3 | 0.82842% | 0.003351% | 0 |
| JapaneseGradens | 0.74748% | 0.001097% | 0 |
| Panama | 1.45627% | 0.003437% | 0 |

独立 forward 基向量测试覆盖单 Gaussian 高 alpha、多 Gaussian、early termination、
介质衰减与贡献阈值、alpha cap 五类情况；修正后颜色梯度最大绝对误差 1.20e-7。
真实视图参考为沿 forward 合成规则累计的 CICA 线性颜色 Jacobian。
生产二进制未修改，各进程加载相同的修正二进制。
这验证的是 RGB 颜色 VJP，**并未证明 opacity、geometry、medium 或 ND 的全部导数正确**。

对前一份 `RENDERER_BACKWARD_FIX_RESULTS_2026-09-15.md` 作明确更正：
其中整个 JIT 后端切换的 forward 最大差异达到 0.001108，
超过当时 2e-6 门槛，对应 JSON 的 `real_scene_all_pass` 为 false。
不能把此前结果表述为 forward 等价通过。当前通过的是仅替换 backward 的独立审计，
原失败记录保留，没有修改门槛或覆盖旧数据。

## 去水可视化与资源

拼接图已覆盖全部 13 个测试视图和全部四路，包含 underwater GT、underwater render、
clear render 及统一放大 10 倍的 underwater 绝对差异。
概览图固定选择各场景按名称排序的第一张测试视图，各场景图包含所有测试视图。
没有调曝光、归一化单张图或改变颜色，拼接保持原始宽高比。

目视概览中，各路整体去水外观相近，黑区和条纹仍然可见，没有一致的修复效果。
所有已记录的 `P_J_gt_1` 均为 0，说明本轮没有通过突破 BND 颜色上界获得表现。

| clear 黑区比例，显示 PNG max RGB < 0.02 | R0C0 | R1C0 | R0C1 | R1C1 |
|---|---:|---:|---:|---:|
| Curasao | 9.869% | 9.743% | 9.131% | 10.757% |
| IUI3 | 7.177% | 12.242% | 8.208% | 8.659% |
| JapaneseGradens | 4.201% | 4.442% | 5.525% | 3.558% |
| Panama | 0.000% | 0.005% | 0.001% | 0.017% |

上述统计包含真实暗区、背景和伪影，不能等同于“黑洞率”。
它基于 clamp 后的 8-bit PNG，不是原始 float 数据；没有 clear GT，
不能将黑色变少、变亮或颜色变化解释为恢复准确率提高。

四路逐视图平均渲染耗时约 23.2–23.4 ms，记录的渲染峰值分配显存约 3.7 GiB。
这些是单次原始运行记录，包含模型常驻内存，未开展独立性能基准。
各路四场景训练累计约 71 GPU 分钟；详细逐场景记录见 `resources.csv`。

## 后续决定

保留 FSB 原版作为当前性能参照，两项修正保留为可控研究开关，不作为新精度收益宣传。
没有理由根据本轮测试结果选择按场景混用的开关。

若下一轮继续追究 CICA 的因果作用，优先从每种 renderer 的同一个 pre-CICA 状态分叉，
保存实际 RNG、模型、优化器和 refinement 状态，再比较 C0/C1。
先增加同配置重复对照以量化运行波动，并在存储充足的目录保留全部终点。
这一步应先于添加新的 FSB gate、curriculum 或依据本测试集调参。

## 文件

- [完整四场景指标与说明](../outputs/four_way_fsb_renderer_cica_strict_20260916_v4/summary_20260916/REPORT.md)
- [全部测试视图拼接图库](../outputs/four_way_fsb_renderer_cica_strict_20260916_v4/summary_20260916/index.html)
- [带水概览图](../outputs/four_way_fsb_renderer_cica_strict_20260916_v4/summary_20260916/visualizations/four_scene_underwater.png)
- [去水概览图](../outputs/four_way_fsb_renderer_cica_strict_20260916_v4/summary_20260916/visualizations/four_scene_clear.png)
- [配对条件核查](../outputs/four_way_fsb_renderer_cica_strict_20260916_v4/summary_20260916/pairing_audit.json)
- [CICA 启用前轨迹差异](../outputs/four_way_fsb_renderer_cica_strict_20260916_v4/summary_20260916/pre_cica_trajectory_drift.csv)
- [逐视图指标](../outputs/four_way_fsb_renderer_cica_strict_20260916_v4/summary_20260916/per_view_metrics.csv)
- [资源记录](../outputs/four_way_fsb_renderer_cica_strict_20260916_v4/summary_20260916/resources.csv)
- [补充审计目录](../outputs/four_way_fsb_contract_20260916/)

未安装依赖，未 commit/push，未使用 GPU 0–5。
