# outputs / renders 存储清理记录

按用户明确要求删除已淘汰或不重要的实验资产，未更改训练代码、数据集或历史指标。

## 清理结果

- 删除 490 个文件，其中 checkpoint 40 个。
- 删除文件总字节数 22,715,547,566，约 **21.16 GiB**。
- `outputs` 普通文件约从 47.87 GiB 降至 26.75 GiB；`du -sh` 显示约 27G。
- `renders` 普通文件约从 1.00 GiB 降至 0.96 GiB；`du -sh` 显示约 993M。
- 本次开始清理时 `/mnt/new` 已有约 352G 空闲，结束时约 373G；此前的空间变化不计入本次释放量。

| 删除类别 | 文件数 | 约 GiB |
|---|---:|---:|
| 已淘汰 CRCC 的权重和逐视图渲染 | 420 | 14.864 |
| 已有新严格消融替代的旧 CICA 对照权重 | 4 | 2.450 |
| 已淘汰 SBR 权重 | 4 | 2.440 |
| 旧 OCMC 的中间 checkpoint | 8 | 1.335 |
| 已由 verified_v2 替代的早期 DVS 渲染 | 36 | 0.033 |
| BVAM、旧 seafree、combined、single-factor 的过时渲染 | 18 | 0.033 |

## 保留范围

保留全部数值结果、配置、日志、研究报告和 CRCC 对比拼接图。
保留最新严格 BND ON CICA 开关消融、BND OFF 基线、严格 teacher 参照、最佳 FSB、
主要 M1/OCMC 对照、best-single 复现实验，以及现阶段需要的源 checkpoint。
旧 OCMC 实验仍保留 step 3000 和 14999 端点。

保留 58 个本地 checkpoint，并核对 inode、大小、修改时间未变。
最新四路结果及其 13 个测试视图拼接图仍保留。
最新 shared-prefix 重复实验位于 `/mnt/d/home/ycy/water-splatting-experiments/fsb_cica_fork_repeat_20260916_v1`，
全部 8 个前缀、32 个终点 checkpoint 以及完整渲染结果均保留。

扫描和删除均不跟随软链接，因此 `outputs/fsb_cica_fork_repeat_20260916_v1`
没有引发对 `/mnt/d` 的删除。

每个发生删除的一级实验目录新增 `ASSET_CLEANUP_2026-09-16.json`。
旧 `checkpoint_retained: true` 字段仅描述当时评估时的状态；当前可用性以清理记录为准。
被删除的权重无法直接恢复；仍可依据保留的源码、配置和源 checkpoint 重跑相应实验。

## 清单

- [完整删除计划](asset_cleanup_20260916/deletion_plan.json)
- [已实际删除文件流水](asset_cleanup_20260916/deleted_files.jsonl)
- [删除后验证结果](asset_cleanup_20260916/result.json)
- [清理脚本](../scripts/maintenance/prune_retired_experiment_assets.py)

未安装依赖，未 commit/push。
