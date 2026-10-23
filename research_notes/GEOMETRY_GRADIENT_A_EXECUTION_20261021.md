# 几何梯度分流 A：执行记录

用户已授权按 BLUE_GREEN_RESIDUAL_EXPERIMENT_PLAN_20261021.md 实施。首阶段八组（四场景B0/A）、每组从零15,000更新。目录：`research_runs/geometry_gradient_A_20261021`。

## 已完成接线与必要修复

- 保持原生forward；从同一完整RGB loss的图像伴随计算direct/medium中心VJP，额外VJP后恢复屏幕梯度及绝对梯度缓冲，避免改变增密统计。
- 发现投影forward对协方差使用视锥截断，但原projection backward未应用相同截断与链式导数。制作实验目录内独立CUDA扩展，仅替换projection backward。两组共同使用，历史二进制保持不动。
- 原真实视图全体中心有限差分部分失败；缩小到单高斯和三个更小步长仍失败。修复后四场景各两训练图通过有限差分及分支和检查；分支和归一化L2误差低于1e-3。原失败与修复后记录均保留，不能把修复收益归给A。
- 每场景20步真实smoke，共80步，提前触发A与CICA；附加中心梯度非零，终态模型reload hash一致。未以smoke权重初始化正式训练。
- 支持由COLMAP >=3真实训练相机、重投影误差<=1像素的锚点提供。动态高斯中心距离和三倍最大scale均须小于锚点局部4NN间距。这是保守稀疏结构支持，不等于完整footprint表面真值。
- 固定三张训练相机作为辅助验证/原分辨率review，支持构造不使用其观测；这些相机仍参与正常RGB训练。该排除在A正式训练启动前确定。smoke最初使用全训练相机支持，不重跑以免超过20步预算；正式支持构造会记录排除名单。这是smoke与正式支持资产的明确差异。

## 评估边界

每场景三训练视图GT/underwater/clear和固定区域已冻结。已人工阅读四场景contact sheets：IUI3上方开放水外观、Japanese上方蓝灰拉丝、Curasao上方颜色片层、Panama远景蓝灰覆盖可见。但单凭这些图不足以给所有区域贴可信开放水域或表面前浮层标签，故残留区域保留U（未知混合），保护区选择可见前景珊瑚/海床。

不会将蓝绿色像素数量减少、黑区扩大或模型自己的深度当20%去水体成功证据。若最终独立几何证据仍不足，残留门槛记不可判定；保留配对训练、视觉和带水指标结果，不编造成功率。

指标统一使用metrics.py在原生float tensor上计算，固定step15000，13测试视图。测试不用于调参。保存0/10000/15000 checkpoint，终点视频以真实训练相机顺序展示，不插值姿态。

## 自动终点交付接线（2026-10-21 12:31 UTC）

正式 campaign 继续运行，四个场景分别使用 GPU 6–9，B0 后自动进入 A。独立后处理进程 `scripts/experiments/postprocess_geometry_gradient_A_campaign.py` 已启动，使用空闲 GPU 0–3，等待每个分支 `final_summary.json` 后导出，不新增训练更新。全部八分支完成导出后执行 `summarize_geometry_gradient_A.py`，写入 `endpoint_report.json/md`、逐视图指标、保护区诊断与原分辨率 B0/A clear 拼图。失败写入 `postprocess_status.json` 并保留日志。

`export_geometry_gradient_A.py --verify-smoke --scene IUI3-RedSea --arm A` 已在现有 20 步 smoke checkpoint 上完成：25 个训练相机、3 个固定相机浮点资产、4 个测试视图浮点指标、真实相机视频、固定区域贡献来源与遮蔽诊断。未新增训练。此验证说明导出链路可运行，不证明最终模型科学有效。

贡献诊断按固定区域原生 clear 合成权重 alpha*T 乘 full-SH 非负通道均值排序，覆盖至少 80% 的该诊断质量；分别保留主导组和遮蔽主导组后的后层图。导出临时改变的 opacity 在 finally 中恢复，checkpoint 不修改。它不是物理水体标签或高斯删除名单；所有未确证的区域仍为 U。

初始封存代码与正式 A 的结构支持修订分别保存在 `code/geometry_gradient.py`、`code/geometry_gradient_formal_A.py`，时间关系及哈希见 `code/support_revision_provenance.json`。后处理代码也单独封存在 `code/postprocess/`。目前训练诊断尚未完整记录逐步实际中心位移、Adam 矩及训练峰值显存，最终报告不得把梯度范数当作这些量；需如实列为交付限制。
