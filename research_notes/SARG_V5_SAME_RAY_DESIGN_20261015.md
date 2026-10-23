# SARG-v5：同光线对象—介质分段责任

目标：纠正 v4 对图像像素深度分箱的限制，在同一条像素光线上拆分对象贡献与介质响应，再检验候选删除的责任替代关系。Panama、IUI3-RedSea 从 COLMAP 初始化各训练 15000 更新；seed42，corrected_v1+BND+M1+OCMC+CICA，优化器与拓扑时序继承 v4。正式训练不加载历史权重。

## 冻结探针

独立 CUDA 扩展重用原生投影与排序 tile bins，逐像素重放 Gaussian alpha 截断、前向透射率和提前终止规则。区间边界取完整模型同光线原生 last contributing depth 的 1/4、1/2、3/4；最后区间延伸到无穷，包含真实介质尾项。删除前后固定这些边界，介质参数冻结。

对象：O_s = sum_{z_i in s} T_i alpha_i exp(-beta_att z_i) J_i。
介质：将每个原生 [z_prev,z_i] 积分与区间求交后累积 T_i B [exp(-beta_bs lo)-exp(-beta_bs hi)]；尾项同样拆分，最终积分到无穷。clear 的 alpha 加权贡献也独立累计，供守恒复核。

每次探针检查 sum_s O_s、sum_s M_s、sum_s clear_s 与原生输出一致。继承项目既有浮点重放容差：每通道平均绝对误差<=1e-6，误差>1e-5 的元素比例<=2e-4；保留最大误差。极少像素在 alpha 截断边界受 CUDA contraction 影响，不用任意严格的全局最大误差判定。

## 接受与训练

每段仅在 positive removed object>1e-6 的实际贡献像素计算责任，至少32像素；r_s=clamp(sum positive medium replacement/sum positive removed object,0,1)。缺失最远段明确为 invalid，责任取0，不能回退到其他段。跨视角有效性、完整图像重建安全、联合删除与表面锚点保护继承 v4；far median 门槛0.08。依旧是保守起点，不能保证所有视图均安全。

分段 residual=1-r_s，不等于重建误差，也不证明真实水体身份。远段双权重、对象变化量加权整体分数保存为审计；训练置信度继续采用跨视角远段责任，与 v4 保持其余正则策略相同。只改变 detached 角色验证，训练前向/反向及损失权重上限不变。

## 验证与实验目录

代码：`research_runs/sarg_v5_ray_segments_panama_iui3_15k_20261015/code/`。两层解析光线测试检查对象/介质积分守恒、对象深度区间归属及删除响应。两个场景分别做2更新 smoke；使用 v4 最终模型进行只读 preflight，历史模型只用于验证，不用于正式初始化。失败的早期最大误差审计保留在 preflight；按既有项目浮点容差重跑在 preflight2。

正式结果写入 `formal/{Panama,IUI3-RedSea}`，不覆盖 v4。完整训练后再比较固定 baseline 远景颜色、近远介质、原生指标与删除影响，单独报告低信号和缺失远段。没有无水GT，不以蓝绿色阈值作为训练标签。

验证实际结果：两个2步 smoke 通过；解析光线测试通过。冻结审计 Panama/IUI3 接受14/8组且模型哈希不变。最大单视图平均绝对误差分别3.62e-8/2.50e-8，误差>1e-5比例最大2.71e-5/7.37e-6，均低于预注册容差；局部最大误差约9.88e-4/8.01e-4，保存在 validation_summary.json，不能声称逐元素完全相等。

正式两场景均已通过1200步首次刷新，后续刷新持续写出真实同光线分段结果。缺失远段明确记录为 far_valid=false，门槛统计赋0；它不是训练错误，也不是有效替代证据。
