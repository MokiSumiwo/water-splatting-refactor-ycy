# 水下 3D Gaussian Splatting：创新机制尝试与实验发现总表

整理日期：2026-09-12。仓库：`/mnt/new/home_old/ycy/water-splatting-refactor`。

整理时当前分支为 `research/bvam-bnd-on`，HEAD 为 `3a0df36`，含既有未提交实验文件；本文同时读取了历史归档和相邻工作树，因此不把当前分支的默认配置当作所有历史实验的配置。

本文汇总截至该日期能从实验记录核验的机制尝试，覆盖 2026-07-23 至 2026-09-11，包括早期 M1–M4、背景归属与容量控制、GMVC、BND、可辨识性与可靠性、TABA/CRCC/DVS，以及最后十个光学传输候选。**这里的“尝试”不等于已证明的新颖贡献，也不等于成功方法。** 同一机制的权重、阶段和消融实验合并列出，不将每个 run 计作一个独立创新点。

本次只整理文档，没有重新训练、评估或修改模型，也没有安装依赖、覆盖历史资产、commit 或 push。数字来自已保存的报告，并在发现记录冲突时标明限制；本次没有逐个重算所有历史 checkpoint 的指标。

## 1. 如何阅读结论

### 1.1 总体判断

- **有配对重建收益、值得作为参考基线的模块：M1、OCMC、CICA。** M1 有明显场景依赖；OCMC 收益小且不代表真实介质恢复；CICA 的证据相对完整，但仍不能代替严格公开基准与多 seed 验证。
- **BND 的主要价值是限制 intrinsic radiance 的范围。** 它不是四场景均提升 underwater RGB 的模块。围绕其容量、优化、几何和损失的补救多数没有同时保住重建、感知质量和机制指标。
- **曾有局部正例、但没有稳定跨场景成立的路线：** 早期 P3/容量控制、GMVC、光学深度缩放、CDEPTH、TABA、TABA-DCU、CRCC 等。应保留其局部证据，同时保留失败场景。
- **DVS 的“介质阈值剔除已有 clear 贡献”假设已被当前实现和冻结对照否定。** 不是“效果很小但值得调参”，而是该前提在所审计路径上不成立。
- **最新十个候选没有产生已通过真实四场景晋级门槛的新方法。** 有真实冻结拟合失败、有合成正例，也有关于分解歧义的诊断；不能合并宣称完成十个新方法的联合训练。
- **目前没有足够证据宣称 SOTA，也没有真实 clear GT 支持“去水更准确”。** 更亮、蓝绿更少、黑区更少、`J>1` 更少，分别只是外观或约束指标。

### 1.2 指标与证据边界

| 项目 | 本文约定 |
| --- | --- |
| 场景缩写 | C = Curasao；I = IUI3-RedSea；J = JapaneseGradens-RedSea；P = Panama。`JapaneseGradens` 沿用资产中的拼写。 |
| 差值 | `Δ = 候选 - 同一实验的对照`。PSNR 单位 dB；PSNR/SSIM 越高越好，LPIPS/MSE 越低越好。 |
| 均值 | 未另说明时是该实验所列场景的等权均值，不是把所有图像混在一起重新计算的 PSNR。 |
| 不可横比 | 不同日期的 baseline、BND/Aux 开关、续训起点、数据划分、分辨率、LPIPS 实现、seed 和优化状态可能不同。只在明确配对的实验内解释差值。 |
| 单次小增益 | 约 0.01–0.05 dB 的单 seed 增益不等于统计显著。近期 Japanese 基线还存在约 0.19 dB 的跨 run 波动，不能把不同实验 E0 混作同一个对照。 |
| clear 的定义 | 历史 `J_proxy_raw`、DC proxy、原生 full-SH raw clear、clamp01 和 `J/(1+J)` 显示结果不同。代理量改善不自动说明真实 clear 改善。 |
| 状态 | “训练完成”表示存在相应训练结果；“诊断”表示未完成该候选正式训练；“合成”表示结论依赖已知模拟条件；“未核验”表示只见用户历史说明。 |
| 来源 | `[Sxx]` 为当前工作树或相邻工作树报告链接；`Axx` 为第 10 节列出的固定 Git 归档来源。 |

### 1.3 可保留的历史主线成绩

9 月 7 日报告记录的固定主线为 `M1 + OCMC + CICA`、`BND OFF`、`Auxiliary OFF`，四场景均值为 PSNR **30.031764**、SSIM **0.927403**、LPIPS **0.137838**、MSE **0.002015746**。这是该报告协议下的历史参考值，不是当前所有分支的统一排行榜，也不能与归档原始 baseline 29.668487 直接相减作为新配对结论。后续 BND ON 研究是在另一个取舍下探索。[S01][S02]

## 2. 早期表示、背景归属与容量控制

以下以 IUI3 为主；跨场景和后续复验单独注明。早期记录中的“当前最佳”只表示当时阶段的选择。

| 编号 / 机制 | 实验结果 | 对应发现与最终判断 | 来源 |
| --- | --- | --- | --- |
| E01 M1：相机与像素上下文介质场 | 早期 I：baseline 29.8790 → M1 31.1314，约 **+1.2524**。后续三场景 M1：C +0.0896、J -0.1533、P +0.3543。8 月固定模型上下文对照，I 完整 camera context 比 neutral +0.186090。 | 22D 输入为方向编码 16D、图像位置/半径 3D、归一化相机中心 3D，不是 learned camera latent。表达能力有效，但会引入视角记忆和介质/外观歧义；不能称全场景必胜。 | A01、A02、A21；[S03] |
| E02 M1 因子拆分与多 seed | J 的 seeds 42/123/3407 均出现 PSNR/SSIM 退化。seed42：context+implicit 比原始 -0.3046；dir-only+tied +0.0355；完整 M1 -0.1533。 | J 的主要风险更接近 camera/XY context，而不是单独 `B_inf=tied`；LPIPS 并非所有 seed 都变差。 | A22 |
| E03 M2：无限水体 ownership | `alpha_depth` 在 I 得到 31.0696，比 M1 -0.0618；水区 Gaussian 泄漏部分下降。 | 显式水/物归属可改变分解，但仅靠 RGB mix、背景解释和容量惩罚不能同时保证物体保留。最终未取代 M1。 | A01、A03 |
| E04 M2 E1–E4：去 near-zero、accumulation 权重、延迟与 tail approximation | `accum=0.002` 曾成为第一阶段候选，PSNR/SSIM 优于该轮 M1，但 LPIPS 约 +0.0015，超 +0.001 门槛；tail+delay 更强清理伴随质量损失。 | 清理收益主要依赖容量压力；强清理不等于正确分解。后续公共 mask 与多 seed 检验降低了早期正面判断。 | A04、A05 |
| E05 M2：深度软化与 hit-aware support | D2 单 seed PSNR 31.2212，但多 seed 不稳定；H2/H3 用 `1-q_hit` 放松容量压力后，far leakage 反弹，未通过门槛。 | `q_hit` 能显示表面，却与已有 accumulation 耦合，会保护已有错误占据；不能直接把“已命中”当成“应保留”。 | A05、A06 |
| E06 M2：保守 capacity floor | C2 相对 C0 PSNR +0.1150、边界保留改善，但 common far clear 增约 6.6%，超过 5% 门槛；其他配置损害 RGB 或物体保留。 | 高精度 object evidence 不等于足够覆盖；动态 self-protection 未形成稳定方案。 | A06 |
| E07 M2：单调/budget capacity 与 closed-tail | closed-tail CT0/CT1/CT2 的 I PSNR 为 31.0320/30.8521/31.0948。CT0 水区 accumulation 0.377473，CT1 质量下降，CT2 LPIPS 0.1790、边界保留 0.9407。 | 数值闭合等价通过，但训练收益未通过；三 seed capacity 与最小 closed-tail 矩阵均未形成稳定替代。 | A07 |
| E08 M3：贡献驱动 Gaussian cleanup | I dry-run PSNR 31.1794，但清理候选数为 **0**，没有实际结构干预。 | 这是流程可运行证据；不能把 PSNR 差异归因于 cleanup，也不能称清理机制有效。 | A01、A08 |
| E09 M4：受约束外观、延迟 SH、DC 软限制、通道平衡 | I：M4 31.0922、M4b 31.0142、M4c 31.1179，均未超过早期 M1 31.1314。 | 该阶段过曝动机部分受到显示方式影响；叠加外观约束未提供稳定收益。后来的 full-SH 原始值审计则确实发现 `J>1`，二者不能混为一谈。 | A01、A09、A34 |
| E10 DualColor：水下 full-SH 与独立 clear 分支 | 固定几何/介质，underwater 保持；最佳 dual clear 的 J-blue 0.2052，差于 M1 0.1691；Water J 0.001024，差于 0.000928。DC-only 边界保留降至 0.8744。 | 增加 clear 分支没有增加真实观测约束；水下不变不能证明 clear 正确。未进入 joint stage2。 | A10 |
| E11 backscatter-consistent `B_inf`：tied / bounded residual / independent | tied bg=0.005 降低 far accumulation 0.407096→0.368988、far clear 0.083962→0.068289，但 RGB/LPIPS 未过关；小 residual A3 PSNR 31.2954，但 far accumulation 和 Water J 更差。 | 闭合成立与 ownership 正确是两件事。FG transmission-aware loss 能恢复 RGB，但可把解释转回 Gaussian；大 residual 和 independent 分支不保留。 | A11 |
| E12 精确背景监督：raw `B_inf`、renderer-consistent medium、BG-J | N1/J 系列没有严格全指标通过者；`bg_medium`、更大 BG-J 权重可降低某些尾项，却增加 far accumulation 或 J-blue。 | 背景颜色拟合改善可能掩盖 Gaussian 过度解释。精确但保守的 mask 本身无法消除远水区表示污染。 | A12、A13 |
| E13 背景归属驱动 densification gate | 高精度水 mask 下背景 split fraction 约 0.15%；较宽历史 mask 为 7.46%，仍低于原定 10% 门槛。F1/F2 正式训练未启动。 | 没有证实背景增点是该 mask 下的主要驱动；不能因为水区有 Gaussian 就推断其来自当前 split 压力。诊断停止。 | A12 |
| E14 可微 clear proxy 与 chroma sweep | 旧原生 clear backward 梯度为零；零介质独立 RGB 调用打通 proxy 梯度。确定性续训 C4/C4.5/C5 比 R0 的 PSNR +0.0115/-0.0046/+0.0040，J-blue -5.9%/-10.4%/-13.0%，Water Accum 几乎不变。 | 可以约束色偏，但不能替代占据修复；早期未匹配续训得到的更大改善不可直接用于因果归因。 | A13、A14 |
| E15 opacity gradient surgery：只放大已有 opacity-down 梯度 | G1/G2 比匹配 C5 的 Water Accum -8.1%/-8.7%，PSNR -0.0078/+0.0026；物体保留进一步降至约 0.963–0.964。扩大到 51 个候选仍未通过。 | 有容量作用，但 Far Clear/J-blue 没有同步改善；28/51 个静态候选很快被正常 cull 清空，候选集合需要生命周期管理。停止增强倍率。 | A14 |
| E16 medium explainability 单项 | I 确定性 R0 30.8838 → A1 31.0934，约 +0.2096；Water Accum 0.7952→0.4120，Water J 0.02636→0.000930。 | 可解释性信号有局部作用，但 J-blue 略差、Object J retention 0.9667，仍有物体安全问题。 | A15 |
| E17 training-only responsibility release | A2 的 PSNR 31.1350，但 Far Accum 0.9996、Water Accum 0.9997，J-blue 0.3530；route+capacity/proxy 也未恢复。 | 不是单纯 loss 权重不够，而是 support inversion：训练路由让水区支持退化，Gaussian 接管解释。拒绝该实现。 | A15 |
| E18 no-route budgeted capacity + clear chroma | C2-B02 比确定性 R0 PSNR +0.2301；Far Accum -70.4%、Far Clear -26.8%、Water Accum -99.1%。Object Acc Ret 仅 0.9313。 | 是真实局部清理信号，但较强容量压力损伤物体；该阶段旧 gate 的“通过”不能替代后来更严格的 Object Acc Ret 要求。 | A15 |
| E19 object-safe proxy geometry：P1–P20 | P3 关闭 proxy geometry、opacity gradient=0.5：I PSNR 31.2235，Object Acc Ret 0.9751；P19/P20 仅对 chroma 排除物体/边界后，Object Acc Ret 降至 0.9275/0.9231。 | P3 是该系列局部折中；全局缩放颜色梯度、排除区域、增强 opacity 均未支配它。P 系列终止于 P20。 | A17 |
| E20 P3 跨场景验证 | 相对 M1：C -0.0067、J -0.0921、P -0.0016，三场景均值 -0.0335；P 的水区 accumulation 0.1361→0.9990，Water J 0.0608→0.7515。 | I 的成功不能泛化；P3 同时改 explainability、capacity、proxy chroma，不能把结果归因于某一个独立因素。 | A21 |
| E21 core–halo capacity | E4 有局部 PSNR/清理收益，但 Object Acc Ret 0.9110；E5/E6/E7 未同时保住物体、边界和 far clear。E7 比当前代码 C2 增加 far residual 连通块。 | Halo support 与物体/边界混叠；继续调小权重也未解决归属问题。停止该系列。 | A16 |
| E22 conflict-aware capacity：Q/R 系列 | 修正分支隔离后，R1–R5 均未超过 P3；如 R2 PSNR 30.7300、Object Acc Ret 0.9473。 | 旧 Q2/Q3 全局崩坏来自 hook 误改主梯度，是实现问题；修正后依然无合格方案，但不能用旧崩坏直接否定冲突门控原理。 | A18 |
| E23 shrink-only / P90 clipping / persistence | S1 比 P3 -0.1469，Far Clear 0.0617→0.0749；S2 更差，边界保留 0.9269；S3 persistence 未运行。 | 原容量梯度的 scale 分量已基本 shrink-only，新增符号门控没有移除主要坏方向；持久性也不能区分物体和伪影。 | A19 |
| E24 M2-Core Release T1–T5 | T1/T2/T3 比 P3 PSNR -0.2538/-0.0461/-0.2183；T1/T3 物体占据受损，T2 多项远水指标更差。T4/T5 未运行。 | 提前密集压力足够强，但 support 不准；top-connected 支持在 2/4 视图为零。不是再加 object-radiance loss 就能补救。 | A20 |

## 3. 早期颜色、介质光谱与细节分配

| 编号 / 机制 | 实验结果 | 对应发现与最终判断 | 来源 |
| --- | --- | --- | --- |
| E25 TACMD：tail anchor、BS band/convergence、counterfactual projection | I 的 T1 降 J-blue 24.4%且 RGB 安全，但 Water Accum +45.3%；J 的 T4 同时改善部分远水指标，却 PSNR -0.2167。 | 颜色归属压力能转移残差，但未解决 occupancy；tail evidence 可作诊断，BS/counterfactual active loss 未保留。 | A23 |
| E26 TMICA：沿 detached 水色轴约束远物体 clear | R2/R3 在两场景降低 J-blue 约 70–88%；I PSNR 损失约 0.15–0.21，Object Acc Ret 约 0.94/0.93；J R3 -0.0730 且 Far Clear +12.1%。 | 水色轴确有干预能力，但色偏代理改善伴随物体/重建损伤；不能作为最终方法。 | A24 |
| E27 viewwise medium inversion | 四场景简化反演中，仅 BS mean 将 far-near 色差 gap 0.1772→0.1025；统一全部 medium 则 0.1772→0.4383，clip rate 18.8%。平滑几乎无效。 | 泛化的“介质太自由”解释过强；BS 有线索，但单深度闭式反演不等于原生多层 renderer。 | A25 |
| E28 view-consistent BS spectrum | 原生冻结重渲染：BS mean / spectrum mean / spectrum shrink0.5 的四场景均值 ΔPSNR -1.0741/-0.3451/-0.0965；均 0/4 完整安全门槛。 | 离线反演有效的频谱替换，在原生 RGB 中并不安全，且原生 clear 不随 BS 替换变化。未进入正式训练。 | A26 |
| E29 TBAP：通道 transmission 梯度预条件 | 诊断中 far R/G/B 梯度放大约 1.363/1.606/1.840，强化了 G/B 而不是预期红通道。P0–P3 正式训练未运行。 | 直接信任已学习、可能违反谱序的 attenuation，会强化错误方向；拒绝当前预条件来源。 | A27 |
| E30 Scalar TBAP | 两场景各四训练视图，三种 scalar 模式全部未过梯度门槛；最远 DC 增益约 1.14–1.16 倍，部分颜色梯度比例漂移超 5%。 | 像素级同通道标量在 Gaussian 聚合后，也不保证每个 Gaussian 的 RGB 梯度方向不变。停止正式训练。 | A28 |
| E31 bounded / hierarchical attenuation spectrum | 离线 `rho=0.2/0.3` 近似 RGB 安全，但谱序违规仍约 90%+；两场景 view-level spectrum violation=100%，residual-induced violation=0。 | 主要违规已存在于视图级基谱，不是单纯 pixel residual 过大；该轮只有诊断与 pilot 方案，不能写成完成层级介质训练。 | A28、A29 |
| E32 MPDR：object-safe detail residual 引导 refinement | 两场景 5k：G1 J +0.1850、I -0.0748；J LPIPS 仅 -0.000265，未达 -0.0015 门槛。G2 增点/时间开销过大。 | 更多细节点不必然带来可接受的感知质量或跨场景安全性；未进入 G4/15k。 | A30 |
| E33 MPDR 配套 SH curriculum | 5k：J +0.1925 但 LPIPS +0.001422；I +0.3112 且 LPIPS 改善，但抽查视图出现黄绿伪影。 | 改变 SH 学习阶段可以提高 PSNR，却不能仅靠均值证明稳定外观。未晋级。 | A30 |

## 4. GMVC 与直接光学深度实验

GMVC 是跨视图介质标定系列，**不等同于 MV-GAR 或 MCGR**。下面将它的主要已执行版本分开；大量 lambda/ramp/object-weight/hold-step 消融归入相应版本，详细矩阵见归档长报告。

| 编号 / 机制 | 实验结果 | 对应发现与最终判断 | 来源 |
| --- | --- | --- | --- |
| E34 GMVC：offline detached consensus / active DC proxy | 修复旧 C4 无效梯度后，200/500-step active proxy 产生约 5%/8% DC 辅助梯度；J 500-step PSNR 仅约 +0.009/+0.011，`E_J` 反而更高。 | 梯度打通仍不等于新观测。旧 M1 consensus 更像自蒸馏，不能消除原分解歧义；未扩展该版本。 | A31 §18 |
| E35 GMVC 低维 oracle / Pareto | 固定 track 上四场景 transfer -5.6% 至 -33.0%、J-var -13.4% 至 -49.1%；但部分 normalized closure 更差、residual saturation 可达 40.7%。 | 公共介质中心+有限视角变化有可用信号；受限 oracle 拟合不能当成完整重建效果。分母审计与正则化后，J 仍无单一全门槛方案。 | A31 §19–20 |
| E36 GMVC online V1：当前视图 closure | Tuned T2：P +0.0448、J +0.0122；所有在线版本均未满足联合解耦门槛，object-J variance 是主要失败项。 | 提高 closure 梯度也没解决问题；固定 partner 的局部校正没有复现 oracle 的跨视图收益。 | A31 §21 |
| E37 GMVC V2：symmetric closure / profiled radiance | C profile1.0/0.5 PSNR -0.277/-0.163；P profile1.0 transfer -2.4%、J-var -6.4%，不足 -5%/-10% 门槛。显式 bounded medium projection 两场景 RGB 严重下降。 | 对称约束有弱信号，但未同时通过 RGB 与解耦门槛；bounded projection 组合停止。 | A31 §22 |
| E38 GMVC V3：交替 medium / DC 校准 | C 最佳安全版本 transfer -1.36%、J-var -7.77%；P 安全版本 transfer -0.36%、J-var +0.07%。 | object 阶段 DC-only 路由正确，C RGB 有恢复；机制增益不足，且 P 无稳定解耦收益。后续围绕固定 bank、profile 权重和 medium hold 做配对控制。 | A31 §23–38 |
| E39 GMVC 固定 P30-MHOLD | 四场景至 15k，比 phase-matched A0：C +0.0356、J -0.0050、I -0.0026、P -0.0142，均值约 +0.0034。8/8 eval banks 的 transfer 改善。 | medium hold 审计通过，C 有局部全面机制收益；其余场景 J-var/DC-var 更差，I/P 存在较大视图间得失抵消。只支持局部标定和 transfer 改善。 | A31 §41 |
| E40 GMVC OAW：observability-aware profile weighting | C 相对 A0 -0.0376，相对 P30-MHOLD -0.0732；J RGB 小增益，但 J-var/DC-var 相对 P30 继续变差。 | 可观测程度只能预测部分 transfer 获益，不能保证 target 正确；对应质量、非朗伯性、局部几何与残差可靠性仍关键。未跨场景推广。 | A31 §39–40 |
| E41 D010：直接光学深度缩放 | C 10k→13k：gamma=0.1 比 gamma=1 PSNR +0.212739，tau p90 -31.51%、`P(J>1)` -42.52%；续训至 15k +0.154316，从头训练 +0.161348。后续固定 1,140,794 个 Gaussian 的 NOREFINE 对照：PSNR -0.078972，但 tau p90 -37.656%、`P(J>1)` -49.658%，通过该轮安全门槛。 | raw beta 未完全抵消有效光学深度下降；固定点数仍有 lower-tau/lower-J 效应，说明不完全依赖点数变化，但 RGB 增益并非固定点数下也成立。仅 C 单场景，不证明物理标定或 clear GT 恢复。 | A34、A35 |
| E42 optical-depth 背景独立监督 BG010 | C +0.093211，但背景残差仅 -3.54%，未达 -20%；tau p90 -1.01%，`P(J>1)` 未下降。 | 背景直接监督没有自然解除 tau/J 饱和；不能把 RGB 小涨当作该机制成立。 | A34 |
| E43 D010 + BG010 | C 相对 D100 +0.181189；保留 A 类 optical-depth 效应，但背景残差仅 -2.79%，且未优于单 D010 的整体表现。 | 组合没有给出独立的背景监督贡献。该轮 soft-bound C/SH0 未运行，不计作完成实验。 | A34 |
| E43a D010 后续 SH3 soft intrinsic bound | C 从 D010@13k 续训至 15k，以初始梯度 1%/5%/10% 标定软惩罚。相对 D010：PSNR -0.083361/-0.117295/-0.148105；10% 组 `P(J>1)` -30.093%、J p99 -26.303%。 | 软边界确实减少 Gaussian color overflow 和 image J 饱和；弱组收益不足，强组 RGB 超过 -0.1 dB 门槛。三组均未过完整 gate；这与后来从头训练 sigmoid BND 是不同实验。 | A35 §B1 |
| E43b BGI：有限介质积分监督 | C 比 D010 +0.051652，SSIM +0.000150、LPIPS -0.000166；背景 finite-medium L1 -3.794%，但 tau p90 +1.868%、`P(J>1)` +3.581%。 | RGB 小涨没有解除 direct/intrinsic 补偿；旧 BG 梯度已超过 5%，失败不能只归因于权重太小，target/mask 仍是问题。 | A35 §B3 |
| E43c SeaFree 式 foreground-aware weighting / coarse depth 准入 | FAW 梯度已标定至 5%，但前景 mask 平均覆盖 97.9033%，超过 95% 异常门槛，未训练；当轮 depth trigger 为 false，DEPTH-G05 与 COMBO-1 未运行。 | FAW 实现和梯度诊断完成，但支持范围没有区分前景；不能写成该轮做过粗深度训练。后来的 P 场景 CDEPTH 见 E50。 | A35 §B2/B4 |

## 5. BND 及其补救路线

| 编号 / 机制 | 实验结果 | 对应发现与最终判断 | 来源 |
| --- | --- | --- | --- |
| E44 BND：bounded SH3 intrinsic color | 匹配四场景 BND 相对 legacy：C +0.404679、I -0.325708、J +0.091778、P -0.812031；均值 **-0.160320**。 | 有界性抑制非法高 radiance，但把原来用于补偿 RGB 的自由度拿掉；不是四场景质量提升。保留为明确取舍的实验基座。 | [S04][S05][S06] |
| E45 AOPT：外观优化器等效/学习率恢复 | P LR×2 比 K1 +0.031265，只恢复约 3.857% RGB gap；×4 更差。 | SH 利用率恢复并不自动恢复图像；不能把 BND 退化简单归因于学习率缩放。 | [S07] |
| E46 STAGE：阶段性固定介质 | P 比匹配 restart -0.068324，三个测试视图均退化。 | 保持 bounded safety，但没有找到更好的 object/medium 优化解。 | [S08] |
| E47 HR：非对称 bounded headroom | P +0.069805，MSE gap 恢复 8.08%，未达 +0.1 dB / 20% 门槛。 | 仍有少量可优化空间，但简单 headroom 不是主要瓶颈。 | [S09] |
| E48 AA：抗锯齿 rasterization | P 三视图 ΔPSNR +0.372797/+0.087440/+0.452753，均值 +0.304330；LPIPS 三视图都变差，Gaussian 约 1.178M→1.970M。 | RGB 增益伴随明显容量增长、饱和比例和感知代价；不能当成干净的 BND 修复。 | [S10] |
| E49 UNORM：移除 inverse-pred 强度归一化 | P -0.041135；high-J 区域 gap 恢复 51.19%，但整体/感知/low-J 与 tau 安全项失败。 | 高亮难区存在 loss 权重问题，却不能靠单一全局 photometric 改写解决。 | [S11] |
| E50 CDEPTH：SeaFree 式粗深度 Pearson 约束 | P +0.254946，SSIM -0.002492、LPIPS +0.005410；high-J 区域伪几何证据没有相应改善。 | 有性能变化，但没有证实源自几何准确性；后续需区分优化路径与 densification 历史。 | [S12] |
| E51 CDEPTH early-off | 3k 关停 depth loss 后 P +0.219886；保留约 68.9% PSNR 与 89.3% high-J 收益，感知 trade-off 未修复。 | 早期路径影响真实存在，但没有满足完整收益门槛；本补救系列关闭。 | [S13] |
| E52 BND-aware refinement allocation | P 从同一 3k 状态续训：RH-R0 +0.021020，SSIM -0.000069、LPIPS +0.000304；hard-core MSE 改善不足 1%。 | grow quota 一致，但 pruning 与最终人口不同；guidance 与亮度 rho=0.8879，无法干净归因于困难区分配。结论为 inconclusive。 | [S14] |
| E53 OMVC：object multi-view consistency | P -0.123756，SSIM -0.000679、LPIPS +0.002517；target consistency error +2.433%、clear error +3.416%。 | direct-object 代理约束未转化为一致性收益；原生 clear backward 不可直接使用也是实现约束。拒绝该版本。 | [S15] |

## 6. 可观测性、可辨识性与训练控制

| 编号 / 机制 | 实验结果 | 对应发现与最终判断 | 来源 |
| --- | --- | --- | --- |
| E54 MIC：介质方差正则 | I -0.229170；最终 raw betaD variance 为对照 4.106 倍，即使 Jacobian conditioning 改善。 | 改善局部条件数不等于改善重建或稳定介质；方差惩罚可能被参数补偿绕过。关闭该路线。 | [S16][S17] |
| E55 OCMC：相机残差可观测子空间投影 | I 初始因果实验 +0.130078，SSIM -0.000974、LPIPS -0.000144；后续匹配四场景均值 +0.084211，SSIM -0.000355、LPIPS -0.000160、MSE +0.00000352。 | 对 full-neutral 的 9D 介质残差施加 detached projector，抑制弱方向，但也削弱正确上下文效用。部分可操作，保留为基线组件，不能声称介质已可辨识。 | [S18][S06] |
| E56 RAOC：ray-adaptive observability rescue | 四场景 C -0.312269、I +0.139137、J +0.170146、P -0.206036；均值 -0.052255，LPIPS +0.000652。 | 4/4 场景 contextual utility 改善，只有 2/4 RGB 改善。效用代理不能替代重建指标；不设默认开启。 | [S19] |
| E57 Auxiliary：SH-opacity tangent 正交化 | 四场景均值 +0.036275；C -0.045276、I +0.090387、J +0.027695、P +0.072294。 | lineage 机制 gate 未通过，四场景都有 SH shrink，选择性降低歧义的定位证据不足；更像通用优化/收缩效应。后续主线 Aux OFF。 | [S20][S21] |
| E58 CICA：跨视图 intrinsic chroma anchor | bounded+Aux 对照均值 +0.041539；修正后的 legacy+Aux OFF 四场景均值 **+0.115898**，SSIM +0.000303、LPIPS -0.000281、MSE -0.000109199，四场景 PSNR 均改善。 | 用真实贡献/DC Jacobian、至少三视图、robust log-chroma target，训练限定 DC 辅助梯度。是较完整的配对正例，但无 clear GT，不能直接宣称真实颜色恢复。 | [S22][S01] |
| E59 MDRR：direct/finite/tail degradation responsibility routing | 四场景均值 **-6.916197**；C -3.651619、I -1.145608、J -18.071080、P -4.796481。 | 诊断阶段的局部责任线索没有转化为稳定训练；完整路由严重破坏优化，必须保留负结果。 | [S22] |
| E60 CICA + MDRR | 比 A0 均值 -3.352766；比 CICA-only -3.394305。 | 两项作用不是简单叠加，MDRR 干扰有效的 CICA 路径。拒绝组合。 | [S22] |
| E61 OCMC/CICA schedule 与梯度比例调度 | 修正单因素配对：OCMC-only -0.052833、CICA-only -0.132396、组合 -0.021718；J 均明显下降。CICA 5% 变体也未形成更好选择。 | 原 `ocmc_ramp` 曾误关 CICA，不能当单因素证据；以修正 follow-up 为准，不更改固定主线。 | [S02][S23] |
| E62 自适应 OCMC spectrum | 两场景：J +0.045783、P -0.125144，均值 -0.039680。 | 自适应截断不能保证压力场景安全；没有通过筛选。 | [S24] |
| E63 confidence-aware CICA | E1：J -0.019183、P +0.048264，均值 +0.014540，但 SSIM/MSE 更差；E2 support=5：J +0.039492、P -0.040648，均值 -0.000578。 | 更严格支持/置信度门控没有形成一致收益，不推广。 | [S25] |
| E64 geometry-transmission utility refinement | 两场景 J -0.531364、P -0.831520，均值 -0.681442、LPIPS +0.036517。 | 将该效用代理直接用于点分配有明显负收益，不能证明它识别了可修复几何。 | [S26] |
| E65 utility-coupled pruning | 两场景均发生 population collapse，最低数量 J 71,871、P 39,664；终止后最终指标 **N/A**。 | 是稳定性失败；不是“没有提升”，也不能把缺失指标写成 0。 | [S27] |
| E66 hard-view curriculum | J -0.121799、P -0.176186，均值 -0.148993。 | 全视图覆盖和高采样熵仍保持，因此不是简单采样塌缩；hardness 代理本身未帮助优化。 | [S28] |

## 7. Gaussian 外观与可靠性：9 月 9–11 日

以下主要在 BND ON 下运行。筛选报告中的 E0/E1/E2 只在各自实验内有效；特别注意“胜者扩展”到底扩展的是哪一个机制。

| 编号 / 机制 | 实验结果 | 对应发现与最终判断 | 来源 |
| --- | --- | --- | --- |
| E67 BRASH：分离 bounded base/view head 与可靠性门控 | 两场景 8 runs：fixed gate 均值 +0.06177，但 MSE 更差；learned gate +0.00692、LPIPS 更差；reliability prior +0.01468，J -0.03233。 | reliability 高置信比例为 0、p90 约 0.042/0.008；未形成可信的条件容量分配。未晋级。compact checkpoints/renders 后来已删除，仅保留记录/汇总。 | [S29] |
| E68 BVAM：bounded 双专家 mixture | E3 两场景均值 -0.018913；E4 pseudo 两场景 +0.009859，低于 +0.03 门槛；E4 J +0.030036、P -0.010319。 | gate 接近 0.5/0.5，E3 collapse=100%，E4 也约 61%/72%；新增专家未形成可靠分工。未扩展四场景。 | [S30] |
| E69 BARR：bounded appearance responsibility routing | 只做两场景 VJP 诊断。分组权重约 0.27/0.40/0.32、熵约 1.08；medium winner 88.2%/93.3%。 | 归一化后近均匀的权重与 winner 统计不能证明物理归属，且受参数数量影响；未通过训练准入。 | [S31] |
| E70 BHTC：bounded headroom temperature control | 只做诊断。J/P 饱和比例 0.28%/1.58%，饱和/残差相关约 0.015/0.365，headroom 约 0.19–0.20。 | P 有局部信号，J 不支持共同 headroom 瓶颈；未运行 temperature 正式训练。 | [S32] |
| E71 BGTR：geometry-transmission refinement | 只做诊断。候选 J 270,887/810,368，P 246,654/1,096,266；几何代理/残差相关约 0.0295/-0.0102。 | 候选多不代表证据强；coarse center proxy 与缺失历史 refinement 统计不足以支持干预。未训练。 | [S33] |
| E72 TABA：transmission-aware chromatic bounded appearance | 四场景 C +0.097521、I +0.084582、J -0.039798、P -0.025919；均值 +0.029097，SSIM +0.000315、LPIPS +0.000213、MSE +0.000026916。 | 主要收缩 chroma 而非 luma logits，black regions 仍存在；PSNR 均值小涨伴随其他代价，默认 OFF。 | [S34] |
| E73 TABA-DCU：水色轴方向与 uncertainty gate | directional-only 最好；完整 uncertainty 变体两场景均值 -0.054945/-0.030670。directional-only 四场景均值 +0.025680，SSIM +0.000038、LPIPS +0.000126、MSE -0.000028633。 | 方向控制有小信号，但更复杂 uncertainty 不能带来收益，也未证实缓解黑洞/条纹；不晋级默认。 | [S35] |
| E74 GRAC：Gaussian 历史梯度/可见性可靠性 | 四场景均值 -0.023771，四项均值指标均变差；I -0.078453。与 TABA 组合继续变差。 | 计数饱和、gate 约 0.92–0.96，只表示频繁被优化/看见，不等于颜色可靠。默认 OFF。 | [S36] |
| E75 GCRC：跨视图污染风险六因素乘积 | 仅两场景 E2：J +0.067907、P -0.075680，均值 -0.003886；LPIPS -0.000888。修正系数 `alpha=0.5*C` 均值约 0.00042/0.00124，此处不是 Gaussian opacity 或 pixel accumulation。 | 多因素乘积使控制很弱，且跨场景 trade-off 明显；筛选胜者是 TABA E1，后续四场景扩展也是 TABA，不能记作 GCRC 四场景成功。 | [S37] |
| E76 CRCC：跨视图 residual-chroma 一致性 | CRCC-only 四场景 C -0.136412、I +0.099025、J +0.079308、P +0.077045，均值 +0.029741；TABA 是首轮多指标筛选胜者。 | clear 黑洞/条纹未稳定改善；I black fraction 0.082957→0.137319。CRCC+TABA 两场景比 CRCC-only -0.043650。默认 OFF。 | [S38] |
| E77 DVS：介质衰减与 clear 可见性解耦 | 四场景 E0/E2 共 8 checkpoint、13 个独立测试相机、26 次冻结对照：原始 clear 与零介质 clear **逐位相同**，accumulation/depth 等相同，黑区恢复为 0，原 underwater 完整保留。 | 实际 cutoff 先执行 `min(0, min(beta))`；beta 非负时就是 `alpha < 1/255`，不依赖正介质衰减。原假设在该实现上被否定，不进入训练，也不宣称提高 underwater PSNR。 | [S39] |

### 7.1 DVS 留下的独立工程发现

1. 原生 clear 是 `sum(alpha * T * color)`，没有 clear residual background；未覆盖处为黑色并不等于被介质阈值剔除。
2. `next_T <= 1e-4` 的 early termination 发生在累计之前，会省略触发终止的 Gaussian；两组都保留此语义，DVS 没有检验“修改共同 alpha/终止阈值”的另一假设。
3. RGB CUDA、ND CUDA、Torch reference、全图无交集分支的输出和背景语义并不完全相同，不能任意互换。
4. 当前 wrapper 不向 CUDA 传递 clear/depth/moment 输出梯度，独立 forward 对照不自动提供训练所需 backward。
5. 零介质 clear proxy 早在 2026-07-27 归档已有实现，因此独立调用本身不能作为新的方法贡献。[S39]、A13

## 8. 最新十个光学传输候选

这十项来自同一顺序实验记录：[S40]。真实数据实验使用既有 CRCC E0 冻结快照，除第 1 项同时检查 E0/E2 外，不能与新的完整 joint-training 方法混称。累计完成 28 个 5k 的真实模块拟合（12+8+8）、8 个 1500-step 歧义搜索，以及数值/合成验证；**没有完成十项各自的四场景 15k 联合训练**。

| 编号 / 原候选 | 已完成实验及结果 | 对应发现与边界 | 来源 |
| --- | --- | --- | --- |
| E78 / 1 Gaussian 内部光程积分 | 26 冻结视图。thickness-only direct MAE 约 1.5e-7–6.4e-7；所有视图变化 >1/255 的像素为 0，四场景 0/4 通过预设 effect gate。 | 当前 checkpoint 的厚度效应不足；conditional-mean depth 位移是另一个效应，不能替厚度假设背书。独立 CUDA 精度修正后等价验证通过，未改训练 renderer。 | [S40][S41] |
| E79 / 2 世界坐标共享介质网格与分段传输 | 4 场景×3 表示×5k，固定 Gaussian。grid 比参数量匹配 ray-MLP 均值 -0.773504；比原 medium -1.640425。 | 当前 frozen replacement 失败；64/128 段仍有约 1e-4–2.4e-4 数值误差。这不排除将来不同 joint training，但不能据此晋级。 | [S40][S41] |
| E80 / 3 折射投影与水中路径联合标定 | 已知平面舷窗合成校准，120 点；heldout 坐标 RMSE 约 3.8e-5 m，比 pinhole 0.1562 m 小。 | 已知玻璃厚度、场景尺度等条件下模型可解；真实 housing/标定缺失，未进行真实四场景图像重建。 | [S40][S41] |
| E81 / 4 深度相关前向散射 PSF | 4 场景×2 模型×5k，depth-dependent 比普通 PSF 均值约 +0.000117，各场景优势 <0.001 dB。 | 学到的解接近恒等；只卷积可见 direct depth layers，不含被遮挡辐射/多重散射。未支持该解释，未晋级。 | [S40][S41] |
| E82 / 5 入射与出射双路径照明传输 | 已知光源/距离的合成例，single-path test MSE 8.6687e-4，dual-path 1.05e-14。 | 证明匹配构造的双路径模型能拟合，不证明普通 RGB 能恢复未知光照；真实入射路径缺测，未实景训练。 | [S40][S41] |
| E83 / 6 相机响应与线性 radiometric domain | 8×5k 冻结外观拟合，linear-sRGB 比 encoded 均值 -0.940240，I -2.345531，四场景 LPIPS 均更差。 | 假设响应下直接转换已有 encoded colors 不成立；不否定具备 RAW/标定且联合学习颜色的线性模型。该 runner 未记录每组训练总耗时/峰值显存，不作速度宣称。 | [S40][S41] |
| E84 / 7 sensor spectral integration | 已知六波段光谱与 3×6 sensor response，外推 test MSE：RGB 2.5858e-4，six-band 3.05e-15。 | six-band 参数更多且享有已知谱信息，是数值/表达能力正例；当前没有真实 sensor response 和足够光谱约束。 | [S40][S41] |
| E85 / 8 偏振联合分解 | 四个理想偏振角、已知不同 Stokes signatures；设计矩阵 rank2 而普通强度 rank1，带噪 separation RMSE 0.003242。 | 已知偏振 signature 能增加观测秩；未知 signature 仍有歧义。当前没有配准偏振数据，不是已完成真实联合重建。 | [S40][S41] |
| E86 / 9 clear 可行解族与歧义范围 | 8×1500-step 受限搜索。全部 75 张训练全图均满足相对原 underwater 预测的每图 RMSE<=1/255，最差约 0.002763；13 测试视图 clear 均值可移动约 ±0.004–0.006，有限族宽度约 0.009–0.012；覆盖没有变化。 | 在近似相同 underwater 下确有不同 clear 解；这是新增 12D 全局后变换族，非原 MLP/SH 的完整可行域。最大单训练像素误差 0.017185，不能误写为逐像素<=1/255；不是 posterior、空洞修复或质量提升。 | [S40][S41] |
| E87 / 10 面向分解的主动采集 | 合成 32 个候选距离中选 6 个，100 次噪声重复；clear RMSE 0.01258→0.010483，但 beta/background 误差变差。 | 观测设计可改善某个目标，不保证所有参数同时更准；设计用了合成真值 nominal，无真实新增采集。 | [S40][S41] |

这些合成正例没有完成全面文献新颖性审查。它们说明特定附加观测或物理模型可能有价值，不能作为已超越常规去水重渲染或公开方法的证据。

## 9. 仅诊断、可行性审计与工程验证

这些工作也是已尝试路线的一部分，但不额外包装成成功创新点。这里列出它们推翻或限定了什么，而不制造不存在的训练 PSNR。

| 编号 / 诊断方向 | 结果与发现 | 对后续机制的含义 | 来源 |
| --- | --- | --- | --- |
| D01 renderer color-bias / surface-hybrid 可行性 | 原生多层与简化表面近似审计未支持直接换 surface/hybrid renderer；medium-aligned clear 能量仍存在。 | 不把简化闭式反演的改善当成 renderer 替换依据；引出了 TBAP，而 TBAP 随后失败。 | A32 |
| D02 SeaFree 实现对照与合法 bounded 解 | SeaFree 相关有界合法解在 high-J gap 上恢复约 61.84%。 | 反驳“bounded 表示必然不能拟合”的强断言，但复合实现差异没有唯一归因。不是完整复现对手或 SOTA 对照。 | A35、A36；[S42][S43] |
| D03 BND RGB/SH/loss/recomposition 审计 | high-J 难区与残差/权重失配有信号，SH 利用率与 object-medium compensation 也变化。 | 分别催生 AOPT、HR、UNORM、STAGE；这些实际干预的后续失败优先于诊断中的候选建议。 | [S44][S45][S46][S47] |
| D04 CDEPTH 路径分解与 early history | 已做 high-J structural signature、fixed-population rollout、direct-object path、densification-trigger 和 mitigation 审计。 | CDEPTH 的 RGB 效应不能直接解释成伪深度带来更准几何；早期优化/拓扑历史仍是混杂因素。最终以 early-off 未解决 trade-off 为准。 | [S48][S49][S50][S51][S52][S53][S13] |
| D05 MEDCTX reduction | 未发现额外 context 使用与 BND hard region 有足够对应。 | 没有进入新的 context-reduction 训练。 | [S54] |
| D06 CBFG 与 BAP appearance pressure | inverse-intensity FAW 与 positive BND error rho=-0.207567，high-J enrichment=0.464；BAP train rho=-0.086273，没有超出 residual/Jmax 的新信息。 | 当前代理不对齐主要坏区；不支持继续设计基于它们的训练权重。 | [S55][S56] |
| D07 depth-opacity | `RZrel>1` 仅约 0.1649%；与误差相关 rho 约 0.0274，最高约 0.0481。 | 当前数据不支持把深度/alpha 模型失配作为主要解释；depth-aware alpha 未训练。 | [S57] |
| D08 pure-water background anchor / CB-BG readiness | P 支持覆盖约 0.1454%；C/I 约 4.173%/32.97%，但后两者晚期物体污染约 27.3%/33.4%，B_inf 已较接近背景。 | 高覆盖也可能混入物体，独立锚定证据弱；没有新增 CB-BG 正式训练。 | [S58][S59][S60] |
| D09 responsibility drift | I 的 Δaccumulation/误差相关 rho=-0.0283，缺少 lineage 证据。 | 未证明有害 ownership drift 是主因，不能直接据此 prune。 | [S61] |
| D10 medium/camera identifiability preflight | 局部 Jacobian 条件性与相机残差效用可测，但局部可辨识性和数据拟合并不相同。 | 为 MIC/OCMC 提供诊断依据；最终效用必须看各自训练结果，不能用 preflight 替代。 | [S62][S63][S17][S03] |
| D11 OCMC projector stale / mode utility / LCS | fresh projector 效用只约 +0.000003，而 PSNR -0.133748；单模式效用不稳定映射到 RGB；LCS 广义 train AUROC 0.669，但 eval rho=-0.131。 | 简单刷新 projector 或用局部效用预测器不解决泛化；RAOC 的后续跨场景失败与此一致。 | [S64][S65][S66][S67][S68][S69] |
| D12 Candidate C residual / resplit replication | 新划分 22 个 heldout views 中，support-error 关系 3/4 场景复现，但 neighbor leave-one-out 0/4 通过。 | 有相关性，尚无可靠可操作的跨视图补救机制；不能据此直接加一致性正则。 | [S70][S71][S72][S73] |
| D13 low-support Gaussian proxy 与抑制 | proxy 相关 median rho 约 0.777；将低支持组 alpha 置零后四场景 PSNR 均降：C -0.006661、I -0.973782、J -0.660028、P -0.004952。 | “低支持”不等于“应删除”。LPIPS 表格却是四场景改善，不能写成所有指标均差；见第 11 节记录冲突。 | [S74][S75] |
| D14 observation-aware optimization | 扰动/控制 preflight 0/4 场景通过。 | 当前支持度不能可靠选择有益优化方向；未升级训练模块。 | [S76] |
| D15 Gaussian view consistency / SH-medium entanglement | 前者仅 2/4 tentative，后者 0/4 支持。 | 有限视图一致性证据不能推出共同 SH-medium 病因。 | [S77][S78] |
| D16 Gaussian subspace identifiability | SH-opacity tangent overlap 约 0.998–0.9996，四场景诊断通过。 | 高子空间重叠确实存在；但 Auxiliary 训练后的收益未满足对应机制归因，不能把高 overlap 当成最终方法成功。 | [S79][S80][S20] |
| D17 MDRR direct / full-medium preflight | direct-only 和 full-medium 责任分析提供局部候选信号。 | 后续完整 MDRR 训练严重退化；最终记录必须以训练失败为准。 | [S81][S82][S22] |
| D18 Gaussian lineage、状态生命周期与等价重构 | split/clone/prune sidecar、可见性 bitset 与低支持状态 smoke 已验证；早期 resume 也暴露相机序列/优化状态差异。 | 是保障配对实验可信的工程基础，没有独立质量收益；不能列为重建创新。 | A14、A33；[S83][S84] |
| D19 RAOC CUDA / hybrid acceleration | fusion/operator 有等价与加速记录；后续 Q50/Q80 modal residual 最大误差约 0.002075，超过 5e-4 门槛。 | 工程层面局部成功不等于正式训练等价；Q50/Q80 未进行被认可的 15k 正式训练。 | [S85][S86][S87] |
| D20 optical-path CUDA 精度对齐 | 初版 FMA 行为未匹配原生输出；独立 extension 使用 `--fmad=false --prec-div=true --prec-sqrt=true --ftz=false` 后 direct/clear 等价恢复。 | 数值实现通过后，thickness screen 仍失败，必须保留两种不同结论。最后十项汇总记录 32 个参数化测试用例通过。 | [S40][S41] |

### 9.1 用户提供、当前本地证据未独立核验的路线

| 机制 | 用户提供的历史结果 | 本文能够下的结论 |
| --- | --- | --- |
| MV-GAR | J 约 +0.134 dB 但 LPIPS 更差；I 约 -0.1845 dB；更小 surface weight 仍未过安全门槛；两视图支持约 0.31%，三视图近零。 | 用户给出的路径位于另一台机器的 `water-splatting-refactor-ycy`；本次本地记录未找到对应可核验报告。保留为历史说明，不与本地已核验结果混算。 |
| MCGR | 多视图支持稀疏，重复、可靠、object-safe correspondence 不足。 | 当前缺少本地可复查的精确指标/配置，不能补造 PSNR。也不能用 GMVC 结果替代 MCGR。 |
| MVSR | 机制与 MV-GAR/MCGR 高度重叠，涉及伪深度、coverage、surface concentration；提出诊断与准入要求。 | 当前没有找到独立完成训练记录，应记为被质疑/未晋级的提案，不能称已验证失败或成功的训练方法。 |

## 10. 来源与可追溯性

### 10.1 本次核验范围

1. 当前 `research_notes/` 的实验、诊断、状态和总结文档，以及 `outputs/task1..task4_*20260908` 的报告。
2. 相邻 BARR、BHTC、BGTR 工作树的对应研究记录。
3. 归档分支 `archive/gmvc-dewatering-full-20260808` 中 7 月至 8 月初的研究记录，使用固定提交 **`dd1a63e56ff00b1c6d3309cc02d1505c6a6393c1`** 读取，没有切换分支。归档与 clean M1/BND 分支的关系见 [S88]。
4. 最新实验的综合结果入口：[十候选综合报告][S41]；主线取舍：[9 月 7 日状态][S01]；历史配对指标：[9 月 5 日汇总][S06]。

这是一份“现有可核验记录的全范围汇总”，不声称枚举所有机器、未保存 notebook、已丢失日志或未提供分支中的实验。原始研究文档中的完整 run 矩阵、逐视图结果、图像和资产路径仍是细查入口。

### 10.2 归档来源索引

下表文件均位于上述固定提交的 `research_notes/` 中。部分源文档引用的 checkpoint/render 现已不在当前工作树；**Git 文档可读取不代表其大型实验资产仍存在。** 不为不存在的本地文件制造链接。

读取示例：

```bash
git show dd1a63e56ff00b1c6d3309cc02d1505c6a6393c1:research_notes/M1_CONTEXT_MEDIUM_2026-07-23.md
```

| 引用 | 归档文件 |
| --- | --- |
| A01 | `ALL_EXPERIMENT_RESULTS_MEETING_2026-07-23.md`；`EXPERIMENT_RESULTS_2026-07-23.md` |
| A02 | `M1_CONTEXT_MEDIUM_2026-07-23.md` |
| A03 | `M2_INFINITE_WATER_OWNERSHIP_2026-07-23.md` |
| A04 | `M2_REFINEMENT_E1_E4_2026-07-24.md` |
| A05 | `M2_PHASE2_HITAWARE_DEPTH_2026-07-24.md` |
| A06 | `M2_PHASE25_OBJECT_RETENTION_CAPACITY_2026-07-24.md` |
| A07 | `M2_PHASE3_SIMPLIFY_CLOSED_TAIL_2026-07-24.md` |
| A08 | `M3_CONTRIBUTION_CLEANUP_2026-07-23.md` |
| A09 | `M4_CONSTRAINED_APPEARANCE_2026-07-23.md` |
| A10 | `M1_DUAL_COLOR_APPEARANCE_2026-07-25.md` |
| A11 | `BACKSCATTER_CONSISTENT_BINF_2026-07-25.md` |
| A12 | `BACKGROUND_ATTRIBUTION_DENSIFICATION_2026-07-26.md` |
| A13 | `NEXT_EXPERIMENT_PLAN_GRADIENT_ATTRIBUTION_2026-07-27.md`，含实际执行结果，不只是计划 |
| A14 | `OPACITY_GRADIENT_SURGERY_PLAN_2026-07-27.md`，含实际执行结果与确定性续训修正 |
| A15 | `MEDIUM_EXPLAINABLE_ATTRIBUTION_2026-07-28.md` |
| A16 | `CORE_HALO_CAPACITY_2026-07-28.md` |
| A17 | `OBJECT_SAFE_PROXY_GEOMETRY_2026-07-29.md` |
| A18 | `CONFLICT_AWARE_CAPACITY_CONTROL_2026-07-29.md` |
| A19 | `SHRINK_ONLY_CAPACITY_CONTROL_2026-07-29.md` |
| A20 | `M2_CORE_RELEASE_T_SERIES_2026-07-30.md` |
| A21 | `M1_P3_CROSS_SCENE_GENERALIZATION_2026-07-30.md` |
| A22 | `JAPANESEGRADENS_M1_SEED_AND_FACTOR_DIAGNOSTICS_2026-07-30.md` |
| A23 | `TACMD_2026-07-31.md` |
| A24 | `TMICA_2026-07-31.md` |
| A25 | `VIEWWISE_MEDIUM_PARAMETER_INVERSION_2026-07-30.md` |
| A26 | `VIEWWISE_BS_SPECTRUM_SHRINK_2026-07-30.md` |
| A27 | `TBAP_2026-07-31.md` |
| A28 | `SCALAR_TBAP_2026-07-31.md` |
| A29 | `HIERARCHICAL_ATTENUATION_SPECTRUM_2026-07-31.md` |
| A30 | `MPDR_GAUSSIAN_DETAIL_REFINEMENT_2026-08-01.md` |
| A31 | `GMVC_MULTI_VIEW_MEDIUM_CALIBRATION_EXPERIMENT_REPORT_2026-08-03.md`，含后续 §18–42 |
| A32 | `RENDERER_COLOR_BIAS_AUDIT_2026-07-31.md` |
| A33 | `STEP1_EQUIVALENT_REFACTOR_2026-07-23.md` |
| A34 | `DEWATERING_OPTICAL_DEPTH_EXPERIMENT_REPORT_2026-08-07.md` |
| A35 | `DEWATERING_SEAFREE_FACTOR_AUDIT_2026-08-08.md` |
| A36 | `SEAFREE_WATERSPLATTING_IMPLEMENTATION_AUDIT_2026-08-08.md` |

### 10.3 当前文档链接

表格中的 `[Sxx]` 可直接打开对应报告。相邻工作树的三个链接依赖这些目录仍保留；本次整理时均存在。输出目录通常被 Git 忽略，因此这些链接的可用性不等于资产已纳入版本控制。

[S01]: PROJECT_STATE_AND_BND_TRADEOFF_2026-09-07.md
[S02]: M1_OCMC_CICA_BND_OFF_OPTIMIZATION_2026-09-07.md
[S03]: M1_CAMERA_CONTEXT_IDENTIFIABILITY_IUI3_2026-08-25.md
[S04]: BOUNDED_SH3_SCRATCH_EXPERIMENT_2026-08-08.md
[S05]: BOUNDED_SH3_CROSS_SCENE_VALIDATION_2026-08-08.md
[S06]: IMPROVEMENT_METRIC_SUMMARY_2026-09-05.md
[S07]: BND_APPEARANCE_OPTIMIZER_EQUIVALENCE_2026-08-09.md
[S08]: BND_STAGED_OBJECT_MEDIUM_OPTIMIZATION_2026-08-10.md
[S09]: BND_HEADROOM_SH3_APPEARANCE_2026-08-10.md
[S10]: BND_ANTIALIASED_RASTERIZATION_2026-08-10.md
[S11]: BND_UNORM_PHOTOMETRIC_LOSS_2026-08-10.md
[S12]: BND_SEAFREE_COARSE_DEPTH_PANAMA_2026-08-11.md
[S13]: BND_CDEPTH_EARLY_OFF_FINAL_EXPERIMENT_2026-08-11.md
[S14]: BND_AWARE_REFINEMENT_CAUSAL_TEST_2026-08-12.md
[S15]: BND_OBJECT_MULTI_VIEW_CONSISTENCY_2026-08-12.md
[S16]: BND_MIC_CAUSAL_IUI3_2026-08-25.md
[S17]: BND_MIC_IMPLEMENTATION_PREFLIGHT_2026-08-25.md
[S18]: M1_OCMC_CAUSAL_IUI3_2026-08-25.md
[S19]: M1_RAOC_CAUSAL_FOUR_SCENE_2026-08-27.md
[S20]: IDENTIFIABILITY_GAIN_MECHANISM_AUDIT_2026-09-02.md
[S21]: IDENTIFIABILITY_MODULE_CAUSAL_EXPERIMENT_2026-09-02.md
[S22]: DIRECT_MDRR_CICA_FOUR_SCENE_EXPERIMENT_2026-09-03.md
[S23]: M1_OCMC_CICA_SINGLE_FACTOR_FOLLOWUP_2026-09-08.md
[S24]: ../outputs/task1_adaptive_ocmc_screening_20260908/task1_adaptive_ocmc_report.md
[S25]: ../outputs/task2_confidence_cica_screening_20260908/task2_confidence_cica_report.md
[S26]: ../outputs/task3_geometry_transmission_screening_20260908/task3_geometry_transmission_report.md
[S27]: ../outputs/task3_geometry_transmission_prune_screening_20260908/task3_geometry_transmission_report.md
[S28]: ../outputs/task4_hard_view_curriculum_screening_20260908/task4_hard_view_curriculum_report.md
[S29]: BRASH_BND_ON_CAPACITY_RECOVERY_2026-09-09.md
[S30]: BVAM_BND_ON_BOUNDED_APPEARANCE_2026-09-09.md
[S31]: ../../water-splatting-refactor-barr/research_notes/BARR_BND_ON_RESPONSIBILITY_ROUTING_2026-09-09.md
[S32]: ../../water-splatting-refactor-bhtc/research_notes/BHTC_BND_ON_HEADROOM_CONTROL_2026-09-09.md
[S33]: ../../water-splatting-refactor-bgtr/research_notes/BGTR_BND_ON_GEOMETRY_TRANSMISSION_REFINEMENT_2026-09-09.md
[S34]: TABA_BND_ON_TRANSMISSION_AWARE_APPEARANCE_2026-09-10.md
[S35]: TABA_DCU_BND_ON_DIRECTIONAL_CHROMATIC_2026-09-10.md
[S36]: GRAC_BND_ON_GAUSSIAN_RELIABILITY_2026-09-11.md
[S37]: GCRC_BND_ON_CROSS_VIEW_CONTAMINATION_2026-09-11.md
[S38]: CRCC_BND_ON_CROSS_VIEW_CONSISTENCY_2026-09-11.md
[S39]: DVS_FROZEN_VISIBILITY_DIAGNOSTIC_2026-09-11.md
[S40]: TRANSPORT_CANDIDATES_SEQUENTIAL_EXPERIMENT_2026-09-11.md
[S41]: ../outputs/transport_candidates_20260911/review/report.md
[S42]: SEAFREE_BOUNDED_LEGAL_SOLUTION_ATTRIBUTION_2026-08-10.md
[S43]: SEAFREE_WS_DEGRADATION_COMPOSITING_AUDIT_2026-08-10.md
[S44]: BND_RGB_TRADEOFF_DIAGNOSIS_2026-08-09.md
[S45]: BND_SH_RESIDUAL_STRUCTURE_AUDIT_2026-08-10.md
[S46]: BND_LOSS_RESPONSIBILITY_ALIGNMENT_AUDIT_2026-08-10.md
[S47]: BND_OBJECT_MEDIUM_RECOMPOSITION_DIAGNOSIS_2026-08-10.md
[S48]: BND_CDEPTH_HIGHJ_STRUCTURAL_SIGNATURE_2026-08-11.md
[S49]: BND_CDEPTH_FIXED_POPULATION_ROLLOUT_2026-08-11.md
[S50]: BND_CDEPTH_DIRECT_OBJECT_OPTIMIZATION_PATH_2026-08-11.md
[S51]: BND_CDEPTH_DENSIFICATION_TRIGGER_AUDIT_2026-08-11.md
[S52]: BND_CDEPTH_MITIGATION_PATH_AUDIT_2026-08-11.md
[S53]: BND_HARD_REGION_OBSERVABILITY_2026-08-11.md
[S54]: BND_MEDIUM_CONTEXT_UTILIZATION_AUDIT_2026-08-12.md
[S55]: BND_CBFG_ALIGNMENT_AUDIT_2026-08-15.md
[S56]: BND_APPEARANCE_PRESSURE_AUDIT_2026-08-16.md
[S57]: BND_DEPTH_OPACITY_AUDIT_2026-08-16.md
[S58]: BND_PURE_WATER_BACKGROUND_ANCHOR_READINESS_2026-08-12.md
[S59]: BND_PW_AUDIT_CURASAO_2026-08-17.md
[S60]: BND_PW_AUDIT_IUI3_2026-08-25.md
[S61]: BND_RESPONSIBILITY_DRIFT_AUDIT_IUI3_2026-08-25.md
[S62]: BND_MEDIUM_IDENTIFIABILITY_PREFLIGHT_IUI3_2026-08-25.md
[S63]: CAMERA_MEDIUM_OBSERVABILITY_PREFLIGHT_2026-08-25.md
[S64]: OCMC_PROJECTOR_TEMPORAL_MISMATCH_IUI3_2026-08-25.md
[S65]: MODEWISE_CAMERA_UTILITY_OBSERVABILITY_IUI3_2026-08-26.md
[S66]: HELDOUT_SINGLE_MODE_CAMERA_UTILITY_IUI3_2026-08-26.md
[S67]: SELECTED_MODE_UTILITY_RECONSTRUCTION_ALIGNMENT_IUI3_2026-08-26.md
[S68]: LOCAL_CONTEXTUAL_SUPPORT_PREDICTOR_IUI3_2026-08-27.md
[S69]: RAOC_RAY_ADAPTIVE_OBSERVABILITY_PREFLIGHT_2026-08-27.md
[S70]: OCMC_RESIDUAL_FAILURE_MODE_AUDIT_2026-08-31.md
[S71]: FOCUSED_C_CAMERA_RESIDUAL_REPLICATION_2026-08-31.md
[S72]: DATA_SPLIT_FEASIBILITY_AUDIT_2026-08-31.md
[S73]: OCMC_CANDIDATE_C_RESPLIT_REPLICATION_2026-08-31.md
[S74]: ISOLATE_LOW_SUPPORT_GAUSSIAN_PROXY_2026-08-31.md
[S75]: LOW_SUPPORT_CAUSAL_INTERVENTION_PREFLIGHT_2026-09-01.md
[S76]: OBSERVATION_AWARE_OPTIMIZATION_PREFLIGHT_2026-09-01.md
[S77]: GAUSSIAN_VIEW_CONSISTENCY_AUDIT_2026-09-01.md
[S78]: APPEARANCE_MEDIUM_ENTANGLEMENT_AUDIT_2026-09-01.md
[S79]: GAUSSIAN_PARAMETER_IDENTIFIABILITY_AUDIT_2026-09-01.md
[S80]: GAUSSIAN_IDENTIFIABILITY_MODULE_DESIGN_2026-09-01.md
[S81]: MULTIVIEW_DEGRADATION_RESPONSIBILITY_PREFLIGHT_2026-09-03.md
[S82]: FULL_MEDIUM_DEGRADATION_RESPONSIBILITY_DIAGNOSTIC_2026-09-03.md
[S83]: LOW_SUPPORT_STATE_LIFECYCLE_PREFLIGHT_2026-09-01.md
[S84]: GAUSSIAN_LINEAGE_SIDECAR_SMOKE_VALIDATION_2026-09-01.md
[S85]: RAOC_CUDA_FUSION_OPTIMIZATION_2026-08-28.md
[S86]: RAOC_CUDA_TRAINING_EQUIVALENCE_2026-08-29.md
[S87]: RAOC_HYBRID_CUDA_ACCELERATION_2026-08-31.md
[S88]: M1_BND_CLEAN_BRANCH_AUDIT_2026-08-08.md
[S89]: PROJECT_HANDOFF_PROMPT_FOR_CHATGPT.md

## 11. 已发现的记录冲突与解读修正

| 问题 | 本文处理 |
| --- | --- |
| DVS 最初只读到了含 attenuation 的条件式 | 采用完整控制流和实际冻结对照：`min(0,min(beta))` 使非负 attenuation 不参与剔除；覆盖旧推断。 |
| 低支持抑制报告正文说 LPIPS 变差，但表格差值均为负 | 采用表格符号：PSNR 四场景下降，LPIPS 四场景改善；不归纳成“全部指标都变差”。[S75] |
| BVAM E3 Japanese 差值存在算术不一致 | 原列 24.566895 - 24.577732 = -0.010837，非正文 -0.011070。本表不重复该错误的单场景差值；E3 总均值仍标为源报告汇总值。[S30] |
| GCRC 四场景 report 标题可能让人误读 | 四场景 winner extension 是 TABA E1；GCRC E2 只有两场景候选训练证据。[S37] |
| CRCC 首轮 winner 与 CRCC-only 四场景不同 | 首轮 winner 是 TABA；CRCC-only 后来有单独扩展，其 C/I 结果不能与 winner extension 混用。[S38] |
| OCMC ramp 的早期对照并非单因素 | 使用后续修正的单因素实验，标注旧 run 的 CICA OFF 混杂。[S23] |
| 旧 opacity/conflict 结果受状态同步与 hook 影响 | 采用修正隔离/匹配 resume 的版本；旧实现崩坏不当作机制原理失败。A14、A18 |
| 早期 per-model q90 water mask 与固定 common mask 不同 | 不横向混合其清理百分比；M2 后续固定 mask 和多 seed 结论优先。A05–A07 |
| 文档中“current best”“pass”是阶段性表述 | 保留其日期、对照与 gate；不覆盖后来跨场景失败，也不把 RGB safety pass 写成质量提升。 |
| 旧交接文档仍推荐 bounded+Auxiliary 的早期主线 | [S89] 描述的是 9 月初实验状态；9 月 7 日主线取舍已更新为 BND OFF、Aux OFF。本文分别记录两种对照，不用旧交接说明覆盖较新的配对结果。[S01] |
| 归档清理造成资产缺失 | 只陈述报告仍有的结果；BRASH 等已删除的 checkpoints/renders 不写成可直接复现。 |

## 12. 这些实验共同说明了什么

1. **表达能力和分解可信度不是同一个目标。** M1 能显著改善部分场景的 RGB；无约束 clear、过大的 medium 自由度和高阶 SH 也能互相补偿。BND 限制补偿后，RGB 下降不自动证明其物理假设错误。
2. **代理指标必须经过实际干预验证。** utility、observability、低支持、色偏、high-J、局部子空间重叠等都有可测信号，但 RAOC、MIC、pruning、Auxiliary 等显示：相关性或局部数学性质不能替代训练因果收益。
3. **清掉颜色与修复表面是不同问题。** chroma 约束可以明显减少 J-blue，却留下 occupancy、黑洞和条纹，甚至损伤真实物体；更亮或更少黑色不足以判断去水准确性。
4. **跨场景与逐视图结果不可省略。** P3、GMVC、TABA、CRCC 等均出现均值掩盖局部退化。小增益还需要同协议多 seed；不能靠换 baseline、选择某个好视图或后处理掩盖负结果。
5. **真实附加观测比再加自由参数更有说服力。** 折射标定、已知照明、光谱响应、偏振与主动采集在受控合成条件下有理由成立，但当前数据没有这些条件，尚不能形成已验证的新方法。

### 12.1 2026-09-14 BND-preserving transport recovery 补充

新增的 A1/A2/A3 固定 Gaussian 传输实验已完成。四场景均值为 A0 `29.853773`
dB、A1 `29.396507` dB、A2 `29.394938` dB、A3 `29.397197` dB；A3 相对 A1
仅 `+0.000690` dB，且 Curasao 平均下降约 `1.93` dB，未通过安全门槛。A3
的厚度分布系数在各场景仍接近零，说明优化没有支持显著的分布偏离。所有
clear、accumulation、depth、transmittance 均逐位保持，故该结果不能归因于
去水图变化。该方向已否定，不进入后续颜色或教师蒸馏阶段。完整记录见
`research_notes/BND_PRESERVING_TRANSPORT_RECOVERY_2026-09-12.md` 和
`outputs/bnd_transport_recovery_20260912_formal/`。
6. **当前最稳妥的研究资产是：配对基线、明确的失败边界和可追溯诊断。** 后续创新应明确新增了什么独立观测或约束、解决哪个已证实的问题，并分别验证 underwater 重建与 clear 分解，不能以“争取 SOTA”为理由省略这些区分。
