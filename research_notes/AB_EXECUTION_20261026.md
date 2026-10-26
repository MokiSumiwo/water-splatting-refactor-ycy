# A(0.1)+B 四场景组合实验

2026-10-26。用户明确要求 A+B 接线并完成四场景各 15,000 步训练及带水宏平均。本轮采用历史较优的 A(0.1)，与 B 的中心到支持面 Huber 约束组合。B 的历史 Curasao 单场景保护条件未通过，残留证据门槛也未建立；此次按用户要求执行探索性组合，不将单项晋级条件改写为通过，也不修改原来的判定阈值。

## 固定训练协议

- Curasao / IUI3-RedSea / JapaneseGradens-RedSea / Panama，seed 42，各 15,000 次正式更新，共 60,000。均从原始 COLMAP 与随机介质网络初始化；不使用 smoke 或历史训练 checkpoint 作为正式初值。
- 基础为 M1 + OCMC + CICA + BND ON，corrected_v1 与原 A/B 的投影修复；其他创新模块关闭。CICA 在 10k 启用，沿用原有刷新和校准。
- A eta 上限 0.1，501–2000 渐入，2001–8000 保持，8001–9999 衰减；10k 后关闭。
- B 沿用冻结的 COLMAP 支持面、过滤条件与时间表，每四步一次。初次有效支持校准 lambda，逐高斯上限为对应原生中心梯度范数的 5% × schedule。
- 组合定义为 `g_native + delta_A + delta_B`；B 的校准和 cap 使用 A 修改前的原生梯度，避免把 A 改动再次计入 B 的预算。只改变 means 梯度。A 的额外 VJP 沿用已验证的增密统计隔离上下文；B 的 Huber 梯度不经过 renderer。
- Adam 的真实 step 被透明包装以记录位移和矩状态，原 optimizer/scheduler 调用与计数保留。B 原有 post hook 未触发时由包装补记；已触发则 pending=None，不会重复记录。
- 正式目录 `research_runs/AB_20261026/formal/<scene>/AB`。基础 worker 内部 `arm=B` 表示原生 M1+OCMC+CICA 路径，不是单独的表面支持 B；`AB_config.json`、experiment 字段和目录标识实际组合。

## 工程验证与偏差记录

四场景各两张有 B 支持的训练相机，在既有 B0@10k 状态上执行零更新 off/A010/B/AB 审计。四种分支的非 means 梯度、CPU/CUDA RNG 与模型可见增密统计均精确保持；关闭分支精确还原原梯度，A/B 非零梯度和 B cap 均通过。组合梯度与独立增量和的相对 L2 最大约 1.8e-9；模型哈希未改变。

最终入口每场景 20 步 smoke 均通过终点模型重载哈希及有限 forward。初始模型、RNG、完整 15k 相机序列与历史 B0 完全一致。Panama 的 smoke 触发非零 B 支持，其他场景初期覆盖为零；因此采用成熟 checkpoint 的零更新审计补充非零接线验证。

本轮 smoke 实际 **180 次**更新：初版四场景 80 次、Panama 记录器复核 20 次、最终入口四场景 80 次。超过原方案每场景最多 20 次的工程 smoke 预算，保留为实施偏差，不混入 60,000 次正式更新。初版记录器 step 标签由 0 起，与实际训练步差 1；初版与中间版本的 scheduler 包装警告及输出均保留。最终入口已修正 step 标签和计数兼容性，未出现该警告。全部 smoke 为一次性状态。

审计与 smoke：`research_runs/AB_validation_20261026/`。正式启动前封存源代码快照和 SHA256，见 `research_runs/AB_20261026/code/`、`run_manifest.json`。训练中不改动训练代码或辅助参数。0/10k/15k checkpoint 均保留。

## 评价口径

固定最终 15k，不依据中间或终点测试选择 checkpoint。使用项目 metrics.py 对原分辨率 float 图像计算 PSNR/SSIM/LPIPS-VGG/MSE；先在场景内对测试视角取平均，再对四场景等权平均（3/4/3/3 张测试图，不能直接将 13 张全平均替代宏平均）。原 B0、A010、B 和本轮 AB 的最终 float 会通过同一 worker 调用路径复算，并检查 GT、视角和形状一致。

原始全图/固定训练区域、真实相机序列视频、主导与后层贡献诊断沿用已有 exporter。它们是诊断材料，不将蓝绿颜色或来源高斯直接当水体标签，不由带水指标推断去残留成功。

复跑单场景：

```bash
CUDA_VISIBLE_DEVICES=6 CONDA_DEFAULT_ENV=water_splatting \
/opt/anaconda3/envs/water_splatting/bin/python -m scripts.experiments.run_AB_worker \
  --scene Curasao --gpu 6 --arm AB \
  --support research_runs/surface_support_B_20261023/support/Curasao \
  --output research_runs/AB_replay/Curasao/AB
```

输出路径须不存在，数据、source_config、冻结 A 复核区域和 B 支持资产须存在。模型可从 15k checkpoint 原生重载；从零训练可按此入口复跑。此入口未提供活跃 B 阶段任意中断点恢复参数，不宣称已经验证中途逐步恢复。
