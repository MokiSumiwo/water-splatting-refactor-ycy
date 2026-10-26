# A(0.1)+C2 组合前验证记录

日期：2026-10-26。本文记录组合训练前的诊断、工程修正和修正后的 C2 四场景重跑。

## 决策

原方案允许在单模块证据成立后选择一个组合（A+B 或 A+C）。本轮选择检查
A(0.1)+C2，但不启动其四场景 15,000 步正式训练：修正后的 C2 满足带水质量
保护门槛，却没有满足残留改善门槛。固定 O 区中只有 Curasao 和
JapaneseGradens-RedSea 达到至少 20% 的占据下降；IUI3-RedSea 占据上升，
Panama 仅小幅下降。因此当前证据是 2/4，而不是方案要求的至少 3/4。

组合短程 smoke 和零更新梯度审计已经完成，用于验证接线和保存恢复；它们不算
正式组合训练，也不把组合结果写成科学效果。

## 历史 C2 的工程更正

历史 C2 的占据项通过 renderer 反向传播时，会改写 CUDA backward 使用的
`xys_grad_abs`。该缓冲区参与后续增密统计，且旧 C2 没有恢复它；因此历史 C2
不是“只改变 medium/opacity 梯度”的干净消融。历史 checkpoint 和指标仍是有效的
实际运行记录，但不能继续用来作该机制的单独归因。A 与 C2 直接串接时还会触发
A 的 scratch 恢复与 autograd saved tensor 版本冲突。

本轮新增隔离入口：在辅助 VJP 之前，将模型可见的增密统计重新绑定到原生统计
副本；CUDA 继续写入计算图持有的旧 scratch，后续增密读取保留的原生统计。
A 的梯度恢复也不再修改仍被图保存的 tensor。原始单模块实现未改写，
旧结果仍可复现。

## 修正 C2 带水指标

四个场景均从原始 COLMAP/随机网络初始化开始，seed 42、15,000 步，沿用原 C2
掩码、调度、5% 分组梯度上限和数据。指标先在场景内测试视角平均，再对四场景
等权平均。

| 分支 | PSNR | SSIM | LPIPS | MSE |
|---|---:|---:|---:|---:|
| B0 | 29.898367 | 0.928811 | 0.139581 | 0.002221705 |
| A(0.1) | 29.999254 | 0.929202 | 0.139924 | 0.002038364 |
| 历史 C2（未隔离） | 30.036043 | 0.929403 | 0.139923 | 0.002091762 |
| **修正 C2** | **29.922419** | **0.928844** | **0.140009** | **0.002178466** |

修正 C2 相对配对 B0 的变化为 PSNR `+0.024052 dB`、SSIM `+0.000033`、
LPIPS `+0.000428`、MSE `-0.00004324`。各场景 PSNR 变化为：

| 场景 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔMSE |
|---|---:|---:|---:|---:|
| Curasao | -0.040251 | +0.000129 | +0.000233 | +0.00001234 |
| IUI3-RedSea | -0.042686 | -0.000050 | +0.001101 | +0.00003683 |
| JapaneseGradens-RedSea | -0.006944 | -0.001015 | +0.001150 | -0.00019415 |
| Panama | +0.186090 | +0.001067 | -0.000773 | -0.00002797 |

四项带水质量保护条件均通过：宏平均 PSNR 下降不超过 0.05 dB、单场景下降
不超过 0.10 dB、宏平均 SSIM 下降不超过 0.001、LPIPS 增加不超过 0.002。
这只是 seed 42 的质量保护结果，不构成稳定提升证据。

## 固定区域诊断

下表为 15,000 步固定训练相机区域的投影占据均值。O 掩码参与了 C 训练，因此
这些数字是 in-sample 诊断，不是独立验证集；U 仍是未知区，P 是结构保护区。
先前只读检查了 16 个 15k 状态（B0/A010/C1/历史 C2 × 四场景）以及 Panama
三个 10k 状态。历史 C2 的 Panama O 占据在 10k 已高于 B0，不能将其恶化
仅归因于 10k 后辅助项关闭。原 A010 的 O 占据在前三场景下降，Panama 上升约
7.05%；全图观察仍显示各场景有残留，不将这些 O 数字解释为全图清除。

| 场景 | B0 O 占据 | 修正 C2 O 占据 | 相对变化 |
|---|---:|---:|---:|
| Curasao | 0.559587 | 0.014494 | -97.410% |
| IUI3-RedSea | 0.081562 | 0.326650 | +300.494% |
| JapaneseGradens-RedSea | 0.071775 | 0.024276 | -66.177% |
| Panama | 0.842078 | 0.803705 | -4.557% |

因此达到至少 20% 占据下降的场景只有 Curasao 和 JapaneseGradens-RedSea，
数量为 2/4，这是数值代理满足的数量，并不代表两个场景已完整通过独立视觉验证。
原曝光/共同 clamp 的三个固定相机全图显示：Curasao 上方大片残留减少，但
MTN_1289 仍有明显亮带；IUI3 的上方蓝色片层扩展；Japanese 的小 O 区改善，
全图上方蓝灰色片层仍在；Panama 的蓝灰色背景未明显消失。前景未见大范围新增
孔洞，但这些观察不足以证明三维几何安全。P 区带水 MSE 相对 B0 的变化为
`+2.09%、-3.70%、-0.10%、+21.00%`；clear 边缘代理变化为
`-0.08%、+0.65%、+1.01%、-2.34%`。Panama 的局部误差增加值得保留，
不能被全图 PSNR 的提高掩盖。

固定区域图和主导来源图保存在
`research_runs/AC_validation_20261026/review/`；它们只用于复核，不把来源高斯
当作删除名单。

## 工程审计与复现资产

- 零更新 `off/A010/C2/A010_C2` 梯度路由审计：四场景通过，组合相对梯度误差
  最大约 `2.71e-7`，阈值 `1e-5`。
- 四场景组合 smoke：每场 20 次更新，模型重载哈希和 forward 有限性通过；初始
  模型、RNG、相机序列与 B0 一致，新增梯度 cap 通过。
- 修正 C2 正式运行：四场景均 exit code 0，均完成 15,000 步；终点 checkpoint
  可重载，诊断脚本确认模型哈希未被修改。
- 结果 JSON：`research_runs/AC_validation_20261026/corrected_C2_metrics.json`、
  `corrected_C2_regions_summary.json`、`corrected_C2_integrity.json`。
- 正式 checkpoint：`research_runs/AC_validation_20261026/formal_C2/`。
- 终点复算、原分辨率 float 测试图、训练相机序列视频：
  `research_runs/AC_validation_20261026/endpoint_C2/`。重载后 PSNR/SSIM/MSE
  与训练导出逐视角完全一致，LPIPS 最大差异 `3.13e-7`。
- 训练用时依场景顺序为 2018/1359/1630/2214 秒；13 张最终测试图的单次渲染
  计时折算各场景约 37.85/59.39/57.20/36.65 FPS。这是该环境的短样本计时，
  不是严格的稳定吞吐基准。模型并非训练零开销，本轮未完整采样训练显存峰值。
- 首次保护区渲染已生成全部 48 组浮点 crop 和 manifest，随后在最后一行 JSON
  打印因 `flush` 参数位置错误退出；已修复打印，保留失败日志和生成文件。
  该错误发生在渲染/写盘完成之后，不影响正式训练或终点复算。

代码和小型证据纳入 Git；checkpoint、视频、完整浮点图留在本机的
`research_runs/`，不声称这些大文件已上传 GitHub。小型摘要见
[AC_validation_20261026_evidence](AC_validation_20261026_evidence)。

复跑单场景（输出路径必须不存在；原数据、源配置、冻结 mask 需已在本机）：

```bash
CONDA_DEFAULT_ENV=water_splatting CUDA_VISIBLE_DEVICES=1 \
  /opt/anaconda3/envs/water_splatting/bin/python -m scripts.experiments.run_AC_pair_worker \
  --scene Curasao --arm C2 --gpu 1 \
  --mask-dir research_runs/C_four_scene_20261025/assets/Curasao/masks \
  --output research_runs/AC_validation_replay/formal_C2/Curasao/C2 \
  --decision research_notes/AC_validation_20261026_evidence/corrected_C2_decision.json
```

IUI3 的 mask 路径使用 `C_open_water_20261024/assets/IUI3-RedSea/masks`，其余
场景使用 `C_four_scene_20261025/assets/<scene>/masks`。配置、训练代码和
checkpoint 哈希见证据 JSON。旧 A010 复现入口保持原样。

下一步建议先解释 IUI3 的片层增加以及 Panama 的低覆盖局限，并补齐辅助目标外的
固定验证相机设计。IUI3 已有 25/25 训练图 O mask，不能把失败简单解释为缺少
训练相机覆盖；也不应仅因结果不佳就扩大掩码或增强占据压力。任何新设计都应
披露已观察本轮测试结果的探索性依赖。在残留门槛达到前，不启动 A+C2，也不
进行 seed 43/44 的最终复验。
