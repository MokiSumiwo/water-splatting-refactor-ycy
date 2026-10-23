# SARG-v8：15k 端点评估（2026-10-16）

## 完成与完整性

Panama、IUI3-RedSea 均从零完成 15000 次 optimizer 更新，最终 step=14999，无 failure.json。最终 checkpoint 分别为 889905574、612404710 字节；实际 SHA256 与 completion.json 一致，均已成功加载并严格恢复模型，原生测试 PSNR 与训练终点评估一致（误差 <1e-4）。检查包含 model、optimizers、schedulers、scalers、rng_state、training_runtime_state、OCMC、CICA、SARG 状态。另有 final_roles.pt/json。未进行实际续训验证。

实验目录：`research_runs/sarg_v8_trust_budget_panama_iui3_15k_20261016/`。
最终模型：`formal/<scene>/checkpoints/step-000014999.ckpt`（latest.ckpt 为同一文件硬链接）。
完整性审计：`endpoint_review/checkpoint_integrity.json`。
后处理已全部完成，模型参数保持不变：`postanalysis_completion.json`。

## 定量结果

|场景|方法|水下重建 PSNR↑|SSIM↑|LPIPS↓|固定远景蓝色占比↓|自身远景蓝绿占比↓|
|---|---|---:|---:|---:|---:|---:|
|Panama|baseline|31.376|0.9476|0.1045|72.71%|78.42%|
|Panama|v6|31.559|0.9481|0.1046|72.00%|81.23%|
|Panama|v8|31.341|0.9468|0.1048|72.30%|81.43%|
|IUI3|baseline|30.796|0.9085|0.2007|45.58%|60.52%|
|IUI3|v6|30.816|0.9112|0.1988|51.99%|71.55%|
|IUI3|v8|30.837|0.9101|0.1990|35.00%|64.10%|

测试视图数 Panama=3、IUI3=4；六方法拼图每场景展示3个代表测试视图。PSNR/SSIM/LPIPS 衡量原始水下图像重建，不能代表清晰场景恢复。固定远景来自 baseline 支撑充分且深度最高30%的像素，蓝色定义 B>R+.05 且 B>G+.02；自身远景蓝绿占比使用各模型自身区域，不能与固定区域混用。颜色阈值统计不衡量残留面积、亮度或几何真实性，不能单独判定成功。

## 重点：去水体图像目视结论

已检查六方法全图拼图，以及 baseline/v6/v8 的上方42%区域局部对比（Panama MTN_1529、IUI3 MTN_5911）。保持原渲染输出，无独立自动调色。

- Panama：前景珊瑚整体结构仍在，但远景仍有大面积蓝灰色片状、拉丝状残留及黑色间隙。v8 未呈现稳定、明确的可视化改善；局部残留重新分布，不能解释为去除成功。固定远景蓝色占比仅比 baseline 下降0.41个百分点，自身远景指标反而更高。
- IUI3：珊瑚与海床主体仍在，v8 比 v6 某些位置的蓝雾变暗，但背景仍明显存在蓝色雾团与细长结构。MTN_5911 中蓝色残留仍横跨上方背景，目视无法支持相对 baseline 的一致改善。固定远景蓝色占比下降10.58个百分点，但自身远景蓝绿占比仍比 baseline 高3.89个百分点，必须结合图像看待。
- 拼图未发现广泛主体塌陷，但二维外观和稀疏锚点不构成几何无损证明；没有清晰真值或独立几何评估。

因此：v8 训练和评估有效完成，但尚未实现两个场景上稳定、明显的远景水体残留去除，不能作为成功机制定稿。

## 机制诊断与限制

末次刷新 Panama 32组全部拒绝，0个有效水体成员；IUI3 接受12组14成员，总人口753943，预算最大0.0009991。Panama总人口1101253。历史曾入选且被原生剪枝删除分别16/38个；这仅表示身份统计，不能证明删掉的是水体。

最终候选全删除使固定远景 clear 的平均绝对变化：Panama=0，IUI3=0.000166（约0.0166%的[0,1]范围）。这是反事实完整删除效应，不是训练透明度衰减的实际收益。说明最终可操作覆盖仍很小；两模型从零训练后的整体外观差异不能全部归因于这些候选。

损失校准还有待修正：名义目标为原生 opacity 梯度10%，两场景的无截断系数约3.1–3.4e-7，却被最小值2e-5抬高，不能宣称实际达到了10%校准。当前机制没有显式可学习的场景/介质双外观分解，软目标也不保证硬透明度上界。

下一步优先做：按残留区域量化“全部候选→排名截断→小组准入→实际衰减”的贡献覆盖损失，区分预算不足、保护门控和表达混合；修复梯度下限，再考虑训练对照。不要仅扩大删除强度或因颜色阈值下降就认定有效。

## 产物

- `endpoint_review/Panama_clear_comparison.jpg`
- `endpoint_review/IUI3-RedSea_clear_comparison.jpg`
- `endpoint_review/Panama_far_crop.png`
- `endpoint_review/IUI3-RedSea_far_crop.png`
- `endpoint_review/summary.json`
- `formal/<scene>/postanalysis/views/<view>/clear.png`、`underwater.png`、`gt.png`、删除影响图与 native.npz。
- 全部测试视图的原生渲染另存 `formal/<scene>/evaluation/SARG_15000/`。
