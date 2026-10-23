#!/usr/bin/env python3
"""Persist final CRCC evidence after completed training and human-readable visual review."""

import csv
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from scripts.experiments.run_crcc_bnd_on import DEFAULT_OUT
from scripts.experiments.run_taba_bnd_on import check_environment, gpu_processes, read_json, write_json, sha256
from scripts.diagnostics.summarize_crcc_visual_proxies import collect as visual_proxies


if __name__=='__main__':
    check_environment()
    root=DEFAULT_OUT
    summary=read_json(root/'summary.json')
    extra=read_json(root/'crcc_only_four_scene_report/summary.json')
    extra_scenes=('Curasao','IUI3-RedSea')
    extra_runs=[r for r in extra['states'] if r['variant']=='E2' and r['scene'] in extra_scenes]
    runs=summary['states']+extra_runs
    diagnostics=summary['diagnostics']+[r for r in extra['diagnostics']
        if r['variant']=='E2' and r['scene'] in extra_scenes]
    means=summary['means']+[r for r in extra['means']
        if r['variant']=='E2' and r['scene']=='four_scene_mean']
    assert len(runs)==14 and len(diagnostics)==42
    assert all(r['complete'] and r['gradient_audit_passed'] and r['integrity_passed'] for r in runs)
    assets=read_json(root/'crcc_artifact_audit.json')
    tests=read_json(root/'engineering/test_results.json')
    cuda=read_json(root/'engineering/cuda_nonregression.json')
    review=read_json(root/'visual_review.json')
    supplementary=read_json(root/'supplementary_verification.json')
    assert supplementary['passed'] and supplementary['state_and_adam_alignment_passed']
    assert assets['passed'] and assets['crcc_state_and_optimizer_alignment_passed']
    assert assets['checkpoint_count']==36 and assets['render_count']==456 and assets['contact_sheet_count']==21
    assert summary['four_scene_complete'] and summary['visual_selection_confirmed']
    assert review['complete']
    assert tests['pytest_passed'] and tests['py_compile_passed'] and cuda['no_regression']
    verified=[]
    for row in read_json(root/'code_snapshot/manifest.json')['files']:
        assert sha256(ROOT/row['path'])==row['sha256']
        verified.append(row['path'])
    gpus=gpu_processes()
    assert not any(gpus.values()),gpus
    worker_names={'run_crcc_bnd_on.py','run_crcc_registered_experiment.py',
                  'run_crcc_supplementary_extension.py','run_direct_mdrr_cica_scene.py'}
    workers=[]
    for line in subprocess.check_output(['ps','-eo','pid=,args='],text=True).splitlines():
        fields=line.split()
        if len(fields)>2 and Path(fields[1]).name.startswith('python') and Path(fields[2]).name in worker_names:
            workers.append(line.strip())
    assert not workers,workers
    diff=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True)
    assert diff.returncode==0,diff.stdout+diff.stderr
    observations=[json.loads(line) for line in (root/'runtime_monitor.jsonl').read_text().splitlines()]
    peak={str(gpu):max(r['memory_mib'] for obs in observations for r in obs['gpus'] if r['gpu']==gpu) for gpu in (6,7,8,9)}
    probes={scene:read_json(root/'engineering/verified_v3'/scene/'audit.json') for scene in ('JapaneseGradens-RedSea','Panama')}
    assert all(row['passed'] for row in probes.values())
    checkpoints={Path(p).resolve() for run in runs for p in run['checkpoint_paths']}
    native_pngs={p.resolve() for source in (root,root/'crcc_only_four_scene_extension')
                 for p in (source/'renders').rglob('*.png')}
    assert len(checkpoints)==42 and all(p.is_file() for p in checkpoints)
    assert len(native_pngs)==540
    samples=[]
    bucket_samples=[]
    completions=[]
    training_gradients=[]
    for run in runs:
        directory=Path(run['path'])
        completion=read_json(directory/'crcc_result.json')
        assert completion['completed_updates']==11999 and completion['final_step']==14999
        assert completion['snapshot_steps']==[5000,10000,14999]
        completions.append(dict(scene=run['scene'],variant=run['variant'],path=str(directory),
                                final_step=completion['final_step'],completed_updates=completion['completed_updates']))
        values=[json.loads(line) for line in (directory/'crcc_training.jsonl').read_text().splitlines()]
        for step in (5000,10000,14999):
            with (directory/f'crcc_statistics_step_{step:05d}.csv').open() as handle:
                for row in csv.DictReader(handle):
                    numeric={k:float(v) for k,v in row.items() if k not in ('view_id','bucket') and v}
                    (values if row['bucket']=='all' else bucket_samples).append(numeric)
        with (directory/'per_view_metrics.csv').open() as handle:
            values.extend({k:float(v) for k,v in row.items() if k not in ('arm','split','view_id') and v}
                          for row in csv.DictReader(handle))
        samples.extend((run['variant'],row) for row in values)
        with (directory/'training_summary_A2.csv').open() as handle:
            training_gradients.extend(dict(scene=run['scene'],variant=run['variant'],**row)
                                      for row in csv.DictReader(handle))
    zero_overflow=all(row['P_J_gt_1']==row['P_intrinsic_rgb_gt_1']==0 for _,row in samples)
    assert all(row['P_J_gt_1']==0 for row in bucket_samples)
    strict_intrinsic=all(0<row['intrinsic_rgb_min']<=row['intrinsic_rgb_max']<1
                         for variant,row in samples if variant in ('E2','E3'))
    no_mismatch=all(row['crcc_state_mismatch']==0 for _,row in samples)
    finite_samples=all(math.isfinite(v) for _,row in samples for v in row.values()
                       if isinstance(v,(int,float)))
    finite_samples=finite_samples and all(math.isfinite(v) for row in bucket_samples for v in row.values())
    escape_groups=('means','scales','quats','opacities','medium_mlp','direction_encoding')
    no_gradient_escape=all(float(row['crcc_'+group+'_grad_l2'])==0
                           for row in training_gradients if row['variant'] in ('E2','E3') for group in escape_groups)
    assert all(row['crcc_lambda']==row['crcc_loss_weighted']==0
               for variant,row in samples if variant in ('E0','E1'))
    assert all(any(float(row['crcc_features_dc_grad_l2'])>0 and float(row['crcc_features_rest_grad_l2'])>0
                   for row in training_gradients if row['scene']==run['scene'] and row['variant']==run['variant'])
               for run in runs if run['variant'] in ('E2','E3'))
    assert zero_overflow and strict_intrinsic and no_mismatch and finite_samples and no_gradient_escape
    standalone_scripts=[ROOT/'scripts/diagnostics'/name for name in (
        'compare_crcc_results.py','summarize_crcc_visual_proxies.py','finalize_crcc_verification.py',
        'audit_crcc_supplementary_extension.py','monitor_crcc_progress.py','verify_crcc_engineering.py')]
    standalone_scripts.append(ROOT/'scripts/experiments/run_crcc_supplementary_extension.py')
    subprocess.run([sys.executable,'-m','py_compile',*[str(p) for p in standalone_scripts]],check=True)
    queue=root/'crcc_only_four_scene_extension/queue_status.json'
    queue_hash=sha256(queue)
    write_json(root/'crcc_only_four_scene_extension/completion_recovered.json',dict(
        complete=True,verified_at_utc=datetime.now(timezone.utc).isoformat(),
        reason='Coordinator session was interrupted. Workers completed without restarting or retraining; the old queue status is retained.',
        preserved_queue_status=str(queue),preserved_queue_status_sha256=queue_hash,
        completed_runs=[r for r in completions if r['variant']=='E2' and r['scene'] in extra_scenes],
        supplementary_artifact_audit=str(root/'supplementary_verification.json'),
        gpu_processes_after=gpus,experiment_workers_after=workers))
    assert sha256(queue)==queue_hash
    with (root/'diagnostics_all_steps_including_CRCC_four_scene.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)
    result=dict(complete=True,verified_at_utc=datetime.now(timezone.utc).isoformat(),
        run_count=len(runs),main_registered_run_count=len(summary['final_metrics']),
        total_unique_runs=14,total_unique_training_checkpoints=42,total_unique_native_pngs=540,
        supplementary_CRCC_only_four_scene_audit_passed=True,
        main_registered_contact_sheet_count=21,
        all_unique_contact_sheet_files=len(list((root/'contact_sheets').glob('*.jpg')))+len(list((root/'crcc_only_four_scene_report/contact_sheets').glob('*.jpg'))),
        checkpoint_count=assets['checkpoint_count'],native_png_count=assets['render_count'],
        contact_sheet_count=assets['contact_sheet_count'],pytest_passed=tests['pytest_passed'],
        py_compile_passed=tests['py_compile_passed'],diff_check_passed=True,
        real_training_audits_passed=all(row['passed'] for row in probes.values()),
        cuda_no_regression=cuda['no_regression'],
        historical_RAOC_strict_forward_pass=cuda['strict_forward_pass'],
        historical_RAOC_strict_backward_pass=cuda['strict_backward_pass'],
        independent_CUDA_backward_optimizer_hash_exact={k:v['independent_CUDA_backward_optimizer_hash_exact'] for k,v in probes.items()},
        identical_gradient_disabled_optimizer_exact=all(v['checks']['disabled_optimizer_same_gradient_update_exact'] for v in probes.values()),
        all_pairings_passed=all(row['passed'] for row in summary['pairings']+extra['pairings']),
        all_native_gt_pixels_exact=assets['gt_original_pixels_exact'] and supplementary['gt_original_pixels_exact'],
        state_and_adam_alignment_passed=assets['crcc_state_and_optimizer_alignment_passed'] and supplementary['state_and_adam_alignment_passed'],
        zero_J_and_intrinsic_overflow=zero_overflow,
        active_CRCC_strict_intrinsic_bounds=strict_intrinsic,
        range_check_scope='All 14 runs: logged training views, every training-view snapshot and held-out test view at 5000/10000/14999. Transmission buckets checked separately for raw J overflow.',
        range_diagnostic_rows=len(samples),transmission_bucket_rows=len(bucket_samples),all_statistics_finite=finite_samples,
        no_CRCC_state_mismatch=no_mismatch,no_CRCC_training_gradient_escape=no_gradient_escape,
        final_reporting_scripts_py_compile_passed=True,
        final_diagnostics=[row for row in diagnostics if row['step']==14999],
        code_snapshot_unchanged=verified,visual_review_complete=True,
        observed_gpu_peak_mib=peak,gpu_processes_after=gpus,all_experiment_gpu_workers_exited=True,
        experiment_workers_after=workers,
        selected_variant=summary['selected_variant'],means=means,
        git_branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        monitoring_note='Sampled nvidia-smi process/device memory, not a continuous allocator high-water mark.')
    write_json(root/'final_verification.json',result)
    proxies=visual_proxies(root)
    extra_rows=[r for r in extra['final_metrics']+extra['means'] if r['variant']=='E2'
                and r['scene'] in ('Curasao','IUI3-RedSea','four_scene_mean')]
    lines=['# CRCC result appendix','','| Scene | Arm | PSNR | SSIM | LPIPS | MSE |',
           '|---|---|---:|---:|---:|---:|']
    for row in summary['final_metrics']+summary['means']+extra_rows:
        lines.append(f"| {row['scene']} | {row['variant']} | {row['PSNR']:.6f} | {row['SSIM']:.6f} | {row['LPIPS']:.6f} | {row['MSE']:.9f} |")
    lines+=['','## Descriptive visual proxies','','| Scene | Arm | Near black | Dark | Cyan-like |',
            '|---|---|---:|---:|---:|']
    for row in proxies:
        lines.append(f"| {row['scene']} | {row['variant']} | {row['near_black_fraction']:.6f} | {row['dark_fraction']:.6f} | {row['cyan_like_fraction']:.6f} |")
    lines+=['','These are global native 8-bit display fractions, not accuracy or artifact segmentation.',
            'See visual_proxies.json for definitions and limitations.',
            '', 'Full evidence: final_verification.json, crcc_artifact_audit.json and visual_review.json.']
    (root/'results_appendix.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result),flush=True)
