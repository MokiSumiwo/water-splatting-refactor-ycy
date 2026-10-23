# FSB renderer × CICA 四路严格同起点配对结果

2026-09-16，四场景 × 四配置的 16 次训练和终点评估全部完成。使用物理 GPU 6、7、8、9，分别对应 Curasao、IUI3-RedSea、JapaneseGradens-RedSea、Panama。全部 13 个测试视图已评估，共 52 个 model-view 配对；没有剩余训练任务。

## 宏平均结果

每个场景先平均全部测试视图，再对四个场景等权平均。PSNR/SSIM 越大越好，LPIPS/MSE 越小越好。

| 配置 | renderer backward | CICA DC 导数 | PSNR | ΔPSNR | SSIM | LPIPS | MSE |
|---|---|---|---:|---:|---:|---:|---:|
| R0C0 | 历史 | 历史 | 29.920682 | — | 0.927656 | 0.120903 | 0.002015663 |
| R1C0 | 修正 | 历史 | 29.879296 | −0.041386 | 0.927653 | 0.120285 | 0.002075444 |
| R0C1 | 历史 | FSB 精确 | 29.861668 | −0.059014 | 0.927616 | 0.120320 | 0.002125896 |
| R1C1 | 修正 | FSB 精确 | 29.905070 | −0.015612 | 0.927987 | 0.120296 | 0.002025813 |

| 场景 | R0C0 PSNR | R1C0 PSNR | R0C1 PSNR | R1C1 PSNR | R1C1−R0C0 |
|---|---:|---:|---:|---:|---:|
| Curasao | 32.511753 | 32.493055 | 32.423055 | 32.569581 | +0.057828 |
| IUI3-RedSea | 30.764867 | 30.758795 | 30.707425 | 30.727966 | −0.036901 |
| JapaneseGradens-RedSea | 24.851311 | 24.743747 | 24.752813 | 24.834572 | −0.016740 |
| Panama | 31.554796 | 31.521587 | 31.563378 | 31.488160 | −0.066636 |

联合修正相对仅 renderer 修正：宏平均 PSNR +0.025774 dB；相对仅 CICA 修正：+0.043402 dB。相对原版仍然 −0.015612 dB，不能以选取较弱的对照宣称整体提升。

## 配对与证据边界

同场景四路均使用同一个 step-3000 checkpoint，seed=42，训练 3001–14999 共 11999 次更新。FSB α=1、M1+OCMC+CICA ON、Auxiliary OFF，CICA 从 step=10000 启用。初始模型、optimizer、scheduler、scaler、RNG、camera sequence、CICA bank、OCMC projector、locked config 与首次 forward 哈希均完全配对。

原 production forward、SH 和 projection 继续使用历史 `csrc.so`；仅两个 rasterizer backward 函数使用独立修正版。生产二进制未覆盖。合成 integration 复核通过，普通 BND 的两种 CICA 模式逐 bit 相同；FSB evidence 与自动微分参考误差小于 4e-6。此前的四场景冻结 VJP 诊断仍是历史 checkpoint 的证据，并非本轮终点测量。

严格同起点不等于确定性轨迹：在 CICA 尚未启用时，同 renderer 的 C0/C1 已出现 loss 与 Gaussian 数量差异。只有一个 seed，没有同配置重复，不能把微小差分视作精确因果效应或统计显著性。CICA λ 根据相同 0.10 梯度校准规则分别生成，不是固定相同数值。

新终点 checkpoint 在 `/dev/shm` 保存、独立加载评估后按磁盘限制清理；其哈希、指标、日志及渲染保留。历史资产未删除，未安装依赖，未 commit/push。

## 去水图发现与结论

所有场景配置均记录到 `P_J_gt_1=0`，有界性保留。拼接图中物体区域整体外观接近，但 Curasao 的上方黑背景、IUI3 的蓝黑背景、JapaneseGradens 的顶部黑区和条纹仍存在；Panama 的上方模糊结构也未消失。clear 背景差异明显于带水图，没有跨场景一致的视觉修复。无真实 clear GT，不能把黑像素减少当作准确度提升。

本轮不晋级为新的 FSB 优化版本或 SOTA。两项改动有导数一致性依据，但未表现为稳定带水 PSNR 收益。若继续归因，下一步应从同一 step-9999 状态分别分叉 C0/C1，并加入同配置重复测量噪声，再讨论新的结构或参数变体；本轮未新增此实验。

## 交付资产

- [完整四场景报告：指标、差分、耗时、显存与局限](../outputs/four_way_fsb_renderer_cica_strict_20260916_v4/comparison/REPORT.md)
- [全部视图可视化浏览页](../outputs/four_way_fsb_renderer_cica_strict_20260916_v4/comparison/index.html)
- [带水图：GT + 四配置](../outputs/four_way_fsb_renderer_cica_strict_20260916_v4/comparison/visualizations/underwater_all_views.jpg)
- [去水图：四配置](../outputs/four_way_fsb_renderer_cica_strict_20260916_v4/comparison/visualizations/clear_all_views.jpg)
- [差异图：统一放大 8 倍](../outputs/four_way_fsb_renderer_cica_strict_20260916_v4/comparison/visualizations/difference_all_views.jpg)
- [逐场景完整指标 CSV](../outputs/four_way_fsb_renderer_cica_strict_20260916_v4/comparison/scene_metrics.csv)
- [严格配对审计](../outputs/four_way_fsb_renderer_cica_strict_20260916_v4/comparison/audit.json)
- [CICA 启用前轨迹差异](../outputs/four_way_fsb_renderer_cica_strict_20260916_v4/comparison/pre_cica_trajectory_drift.csv)
- [零更新 integration 审计](../outputs/four_way_fsb_renderer_cica_strict_20260916_v4/integration_audit_20260916.json)
- [完成状态](../outputs/four_way_fsb_renderer_cica_strict_20260916_v4/completion_status.json)

PNG 差异含 8-bit 量化；数值指标来自训练后浮点 render 评估。可视化包含每个场景的全部测试视图、固定中心裁剪与原始 clear 显示图，没有按指标挑图或额外改色。
