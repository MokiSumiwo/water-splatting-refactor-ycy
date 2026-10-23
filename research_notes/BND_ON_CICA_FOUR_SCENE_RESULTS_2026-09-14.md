# BND ON CICA 四场景严格归因结果（2026-09-14）

四个场景均完成 BND ON、M1、OCMC 条件下的 CICA OFF/ON 配对实验。JapaneseGradens 和 Panama 使用既有严格配对；Curasao 和 IUI3-RedSea 从相同 BND ON step-3000 C0 checkpoint 先完成到 step-5000，再续训到 step-14999。最终评估统一筛选 `absolute_step=14999`、`split=eval`。

| 场景 | CICA OFF PSNR | CICA ON PSNR | ΔPSNR | ΔSSIM | ΔLPIPS | ΔMSE |
|---|---:|---:|---:|---:|---:|---:|
| Curasao | 32.500129 | 32.493907 | -0.006222 | +0.000117 | -0.000233 | +0.000006883 |
| IUI3-RedSea | 30.638429 | 30.732962 | +0.094533 | +0.000327 | -0.000443 | -0.000097920 |
| JapaneseGradens-RedSea | 24.735882 | 24.796497 | +0.060614 | +0.000041 | +0.000221 | -0.000169653 |
| Panama | 31.426553 | 31.501145 | +0.074592 | +0.000110 | +0.000244 | -0.000012306 |
| **四场景宏平均** | **29.825248** | **29.881128** | **+0.058853** | **+0.000163** | **-0.000083** | **-0.000070531** |

CICA 在四场景中 3 个场景提升 PSNR，Curasao 轻微下降。宏平均 PSNR 提升 `+0.058853 dB`，SSIM 和 MSE 同方向改善，LPIPS 宏平均略降 `−0.000083`。因此 CICA 在 BND ON 下有小幅、跨场景但不完全稳定的收益，不能视为稳定的大幅提升。

四个补充分支的最终 step、checkpoint、OCMC projector 和梯度审计均通过：`final_step=14999`、`checkpoint_retained=true`、`ocmc_projector_unchanged=true`、`GRADIENT_ROUTING_AUDIT=true`、`cica_gradient_escape_l2=0`。

完整数据见 [four_scene_summary.json](../outputs/bnd_on_cica_ablation_extra3_20260914/four_scene_summary.json)。前两场景及可视化见 [BND_ON_CICA_ABLATION_RESULTS_2026-09-14.md](BND_ON_CICA_ABLATION_RESULTS_2026-09-14.md)。

注意：BND OFF 的四场景 CICA 收益为 `+0.115898 dB` PSNR；BND ON 四场景宏平均收益为 `+0.058853 dB`，约为其 51%。BND ON 与 BND OFF 的绝对指标不能仅凭本表直接比较为因果差值，必须保持相同 source checkpoint、配置和数据划分才可进行严格差距分析。
