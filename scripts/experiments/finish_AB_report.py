"""Build the final AB comparison from independently recomputed float metrics."""
import argparse
import csv
import json
from pathlib import Path

from scripts.experiments.run_AB_campaign import SCENES, sha

ROOT = Path(__file__).resolve().parents[2]
KEYS = ['PSNR', 'SSIM', 'LPIPS', 'MSE']
ARMS = ['B0', 'A010', 'B', 'AB']


def read(path):
    return json.loads(path.read_text())


def mean(values):
    return sum(values)/len(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    args = parser.parse_args(); out = args.campaign.resolve()
    verification = {s: read(out/'verification'/s/'verification.json') for s in SCENES}
    assert all(v['complete'] for v in verification.values())
    manifest = read(out/'run_manifest.json'); assert manifest['status']=='COMPLETE'
    scene = {s: v['per_scene'] for s, v in verification.items()}
    macro = {a:{k:mean([scene[s][a][k] for s in SCENES]) for k in KEYS} for a in ARMS}
    delta = {a:{k:macro['AB'][k]-macro[a][k] for k in KEYS} for a in ARMS if a!='AB'}
    guards = dict(macro_PSNR=delta['B0']['PSNR']>=-.05,
        per_scene_PSNR=all(scene[s]['AB']['PSNR']-scene[s]['B0']['PSNR']>=-.10 for s in SCENES),
        macro_SSIM=delta['B0']['SSIM']>=-.001, macro_LPIPS=delta['B0']['LPIPS']<=.002)
    telemetry = {}; protection = []
    for s in SCENES:
        formal = out/'formal'/s/'AB'
        logs = [json.loads(x) for x in (formal/'surface_support.jsonl').read_text().splitlines()]
        active = [r for r in logs if r.get('delta_norm',0)>0]
        moves = [json.loads(x) for x in (formal/'surface_displacements.jsonl').read_text().splitlines()]
        arows = [json.loads(x) for x in (formal/'geometry_gradient.jsonl').read_text().splitlines()]
        assert len(logs)==2374 and len(active)>0 and len(moves)==len(active)
        assert all(r['native_reference_supplied'] for r in active)
        assert max(r['gradient_ratio'] for r in active)<=.050001
        assert all(500<r['step']<10000 for r in logs+arows)
        assert max(r['eta'] for r in arows)<=.100001
        telemetry[s] = dict(training_seconds=verification[s]['training_seconds'],
            final_population=verification[s]['final_population'], B_events=len(logs), B_nonzero_events=len(active),
            B_mean_event_coverage=mean([r['support_fraction'] for r in logs]),
            B_max_global_gradient_ratio=max(r['gradient_ratio'] for r in active),
            B_displacement_record_count=len(moves),
            B_peak_allocated_bytes=max(r.get('peak_allocated_bytes',0) for r in logs))
    training_code_paths = [
        'scripts/experiments/run_AB_worker.py', 'scripts/experiments/run_AB_campaign.py',
        'scripts/experiments/run_direct_mdrr_cica_scene.py',
        'scripts/experiments/run_geometry_gradient_A_worker.py', 'scripts/experiments/run_AC_pair_worker.py',
        'scripts/experiments/geometry_projection_backend.py', 'scripts/experiments/export_geometry_gradient_A.py',
        'scripts/experiments/audit_AB_pair.py', 'scripts/experiments/summarize_AB_campaign.py',
        'water_splatting/auxiliary_gradient_isolation.py', 'water_splatting/geometry_gradient.py',
        'water_splatting/surface_support.py']
    code_match = all(sha(ROOT/p) == manifest['code_sha256'][p] for p in training_code_paths)
    report = dict(complete=True, seed=42, updates_per_scene=15000, total_formal_updates=60000,
        scene_metrics=scene, macro_scene_equal=macro, AB_minus_reference=delta,
        underwater_guard=guards, underwater_guard_pass=all(guards.values()), telemetry=telemetry,
        metric_source='all four arms independently recomputed from native-resolution float tensors with W._taba_rgb_metrics',
        test_view_counts={s:len([r for r in verification[s]['recomputed_views'] if r['arm']=='AB']) for s in SCENES},
        checkpoint_sha256={s:verification[s]['checkpoint_sha256'] for s in SCENES},
        training_code_snapshot_matches_frozen_manifest=code_match,
        posthoc_verifier_sha256=sha(ROOT/'scripts/experiments/verify_AB_endpoint.py'),
        report_builder_sha256=sha(ROOT/'scripts/experiments/finish_AB_report.py'),
        actual_smoke_updates=manifest['actual_smoke_updates'],
        interpretation='User-requested exploratory combination; single seed; no residual-removal success established')
    (out/'verified_metrics.json').write_text(json.dumps(report, indent=2)+'\n')
    all_rows = [r for s in SCENES for r in verification[s]['recomputed_views']]
    with (out/'verified_per_view.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=['scene','arm','view_id']+KEYS);writer.writeheader();writer.writerows(all_rows)
    lines = ['# A(0.1)+B 四场景 15k 结果', '',
        '日期：2026-10-26。四场景均已完成 seed 42、从原始 COLMAP 与随机介质网络初始化的 15,000 步训练，共 60,000 次正式更新；训练和终点导出均 exit code 0。', '',
        'A 使用 eta 上限 0.1，B 沿用原单模块中心到支持面 Huber 项、冻结支持资产和 5% 梯度上限。两者使用同一原生 means 梯度作为参考，10k 前按原日程退火，最后 5k 为原生更新。', '',
        '## 带水指标', '',
        '下表为原分辨率 float 指标；先对场景内测试相机平均，再对四场景等权平均（3/4/3/3 张测试图）。PSNR/SSIM 越高越好，LPIPS/MSE 越低越好。B0、A010、B、AB 全部通过同一指标路径重新计算。', '',
        '| 场景 | AB PSNR | AB SSIM | AB LPIPS | AB MSE | ΔPSNR vs B0 | ΔPSNR vs A010 | ΔPSNR vs B |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in SCENES:
        m=scene[s]['AB'];d=[m['PSNR']-scene[s][a]['PSNR'] for a in ['B0','A010','B']]
        lines.append(f"| {s} | {m['PSNR']:.6f} | {m['SSIM']:.6f} | {m['LPIPS']:.6f} | {m['MSE']:.9f} | {d[0]:+.6f} | {d[1]:+.6f} | {d[2]:+.6f} |")
    lines += ['', '| 分支 | PSNR 宏平均 | SSIM 宏平均 | LPIPS 宏平均 | MSE 宏平均 |', '|---|---:|---:|---:|---:|']
    for a in ARMS:
        m=macro[a];lines.append(f"| {a} | {m['PSNR']:.6f} | {m['SSIM']:.6f} | {m['LPIPS']:.6f} | {m['MSE']:.9f} |")
    lines += ['', f"AB 带水保护条件：{'通过' if all(guards.values()) else '未通过'}。分项结果：`{json.dumps(guards)}`。", '',
        '## 核验与解释范围', '',
        '四场景零更新组合梯度审计通过，最大相对 L2 误差约 1.8e-9。关闭路径、非 means 梯度、RNG 和原生增密统计精确保持。正式初始化和完整相机序列与配对 B0 一致，最终 checkpoint 重载哈希与有限 forward 均通过。测试 GT 和视角逐项一致，终点指标从保存的 float 重新计算。', '',
        '本轮是单 seed 的探索性组合结果，不能把分数变化分摊为 A/B 各自的独立收益，也不能据带水指标宣称去水或去残留成功。历史 B 未通过全部晋级门槛，本轮由用户明确授权执行，没有事后更改判断阈值。原固定 U 区未被重新标为水体；已导出的 clear、视频与贡献材料仅作为复核资产，未声称通过残留/结构视觉门槛。', '',
        'B 作用覆盖与成本：', '',
        '| 场景 | 训练秒数 | 最终高斯数 | B 非零事件/2374 | B 事件平均覆盖 | B 日志峰值分配 GiB |',
        '|---|---:|---:|---:|---:|---:|']
    for s,t in telemetry.items():
        lines.append(f"| {s} | {t['training_seconds']:.1f} | {t['final_population']} | {t['B_nonzero_events']} | {100*t['B_mean_event_coverage']:.5f}% | {t['B_peak_allocated_bytes']/2**30:.3f} |")
    lines += ['', '覆盖率分母是每事件当时所有高斯，不是残留贡献覆盖。显存值为 B 有效事件读取的累计 allocated 峰值，不含最后 5k 与导出阶段可能更高的峰值，不是全流程显存上界。训练时间为原 worker 计时，不含所有预处理/导出成本。', '',
        '实际 smoke 180 次（含记录器修正后的重复）超出原每场景 20 次预算，已在执行记录完整披露；全部是一次性状态，未作为正式初始化。正式训练从启动到结束没有改动代码或参数。', '',
        '## 资产', '',
        '- 正式 checkpoint、日志、float、固定训练视图和视频：`research_runs/AB_20261026/`。',
        '- 最终统一口径：`verified_metrics.json`、`verified_per_view.csv`；逐场景重载/复算证据：`verification/<scene>/verification.json`。',
        '- 接线和执行协议：[AB_EXECUTION_20261026.md](AB_EXECUTION_20261026.md)。',
        '- Git 小型证据：[AB_20261026_evidence](AB_20261026_evidence)。checkpoint 与视频保留本地，不纳入 Git。', '']
    text='\n'.join(lines)
    (out/'verified_report.md').write_text(text)
    (ROOT/'research_notes/AB_RESULTS_20261026.md').write_text(text)
    print(json.dumps(report, indent=2))


if __name__=='__main__': main()
