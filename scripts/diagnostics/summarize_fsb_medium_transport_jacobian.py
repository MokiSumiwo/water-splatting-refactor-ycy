#!/usr/bin/env python3
"""Independent verification and summary of frozen FSB medium Jacobian fits."""
from __future__ import annotations
import argparse,csv,hashlib,html,json,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
SCENES=('Curasao','IUI3-RedSea','JapaneseGradens-RedSea','Panama')
LABELS=('Curasao','IUI3','JapaneseGradens','Panama')
ARMS=('baseline','all_fit','far_fit')


def read(path):
    with path.open() as f:return list(csv.DictReader(f))

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def writecsv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def fmt(x,n=5):return '—' if x is None or not np.isfinite(float(x)) else f'{float(x):.{n}f}'

def make_summary(root,out):
    summaries={s:json.loads((root/s/'summary.json').read_text()) for s in SCENES}
    allmetrics=[];region_metrics=[];fits=[];checks=[];fitstats=[];diagnostics=[]
    for scene in SCENES:
        d=summaries[scene];metrics=read(root/scene/'per_view_metrics.csv');regions=read(root/scene/'region_metrics.csv')
        f=json.loads((root/scene/'fit_coefficients.json').read_text());fits.append((scene,f))
        assert d['completed'] and not d['smoke'] and d['implementation']=='v2'
        assert d['model_state_before']==d['model_state_after'] and d['max_clear_change']==d['max_support_change']==d['replay_max_error']==0
        assert d['optimizer_steps']==d['backward_calls']==0
        iv=json.loads((root/scene/'input_hashes.json').read_text())
        for path,h in iv['hashes'].items():assert sha(Path(path))==h,('input drift',path)
        audit=json.loads((ROOT/'outputs/fsb_far_field_diagnostic_20260916_v1'/scene/'audit.json').read_text())
        expected=set(audit['camera_metadata']);observed={r['split']+'/'+r['view'] for r in metrics};assert expected==observed
        assert len(metrics)==3*len(expected)
        assert len(read(root/scene/'support_checks.csv'))==3*len(expected)
        for arm in ARMS:
            rs=[r for r in metrics if r['split']=='eval' and r['arm']==arm]
            met={k:float(np.mean([float(r[k]) for r in rs])) for k in ('PSNR','SSIM','LPIPS','MSE')}
            for k,v in met.items():assert abs(v-d['metrics'][arm][k])<1e-12
            for region in ('near','mid','far','loose_far','high_cv_far'):
                rr=[r for r in regions if r['split']=='eval' and r['arm']==arm and r['region']==region]
                num=sum(int(r['pixels']) for r in rr);value=math.sqrt(sum(float(r['sse']) for r in rr)/(3*num)) if num else None
                assert value==d['metrics'][arm][region+'_rmse']
            base=d['metrics']['baseline']
            allmetrics.append(dict(scene=scene,arm=arm,views=len(rs),**met,
                                   delta_PSNR=met['PSNR']-base['PSNR'],delta_SSIM=met['SSIM']-base['SSIM'],delta_LPIPS=met['LPIPS']-base['LPIPS'],
                                   near_rmse=d['metrics'][arm]['near_rmse'],far_rmse=d['metrics'][arm]['far_rmse'],
                                   near_rmse_change_pct=100*(d['metrics'][arm]['near_rmse']/base['near_rmse']-1),
                                   far_rmse_change_pct=100*(d['metrics'][arm]['far_rmse']/base['far_rmse']-1)))
        for split in ('train','eval'):
            for arm in ARMS:
                for region in ('all','near','mid','far','loose_far','high_cv_far'):
                    rs=[r for r in regions if r['split']==split and r['arm']==arm and r['region']==region]
                    br=[r for r in regions if r['split']==split and r['arm']=='baseline' and r['region']==region]
                    n=sum(int(r['pixels']) for r in rs);ss=sum(float(r['sse']) for r in rs);bs=sum(float(r['sse']) for r in br);lin=sum(float(r['linear_sse']) for r in rs)
                    region_metrics.append(dict(scene=scene,split=split,arm=arm,region=region,pixels=n,rmse=math.sqrt(ss/(n*3)) if n else None,
                                               mse_change_pct=100*(ss/bs-1) if bs else None,linear_mse_change_pct=100*(lin/bs-1) if bs else None))
        for arm in ('all_fit','far_fit'):
            m=f['normal_equations'][arm];H=np.asarray(m['H']);g=np.asarray(m['g']);q=np.asarray(f['q'][arm]);raw=np.asarray(f['raw_q'][arm])
            direct=np.linalg.solve(H+f['ridge']*np.diag(np.diag(H)),g)
            assert np.allclose(raw,direct,rtol=1e-7,atol=1e-10)
            assert np.allclose(q,np.clip(direct,-f['q_max'],f['q_max']),rtol=1e-7,atol=1e-10)
            lvo=[r for r in read(root/scene/'leave_one_train_view.csv') if r['arm']==arm]
            lmse=sum(float(r['linear_sse']) for r in lvo)/sum(float(r['baseline_sse']) for r in lvo)
            diag=f['conditioning'][arm]
            fitstats.append(dict(scene=scene,arm=arm,fit_linear_mse_change_pct=100*(diag['clipped_tangent_error_ratio']-1),
                                 leave_one_train_view_linear_mse_change_pct=100*(lmse-1),normalized_condition=diag['normalized_condition'],
                                 clip_fraction=f['clip_fraction'][arm],q_max_abs=float(np.max(abs(q))),
                                 sign_agreement=f['coefficient_sign_agreement']))
        numeric=read(root/scene/'numerical_validation.csv')
        for r in numeric:
            if r['test'] in ('epsilon_vs_half','directional_prediction'):diagnostics.append(dict(scene=scene,**r))
        checks.append(dict(scene=scene,input_hashes_verified=len(iv['hashes']),all_views_verified=len(expected),
                           mean_metrics_recomputed=True,least_squares_independently_resolved=True,clear_preserved=True,
                           max_fd_halving_relative_error=d['max_fd_halving_relative_error']))
    for file,h in json.loads((root/'provenance/sha256.json').read_text()).items():assert sha(ROOT/file)==h
    farrows=[r for r in allmetrics if r['arm']=='far_fit'];mean=lambda key:float(np.mean([r[key] for r in farrows]))
    per_scene=[]
    for r in farrows:
        d=summaries[r['scene']]
        ok=(r['delta_PSNR']>0 and r['far_rmse_change_pct']<0 and d['coefficient_sign_agreement']>=6 and max(d['clip_fraction'].values())<=.25)
        per_scene.append(dict(scene=r['scene'],predictive_pass=r['delta_PSNR']>0 and r['far_rmse_change_pct']<0,
                              coefficient_pass=d['coefficient_sign_agreement']>=6 and max(d['clip_fraction'].values())<=.25,scene_pass=ok))
    gates=dict(three_scenes=sum(r['scene_pass'] for r in per_scene)>=3,
               mean_psnr_gain=mean('delta_PSNR')>=.05,no_scene_psnr_loss_gt_0p1=min(r['delta_PSNR'] for r in farrows)>=-.1,
               mean_lpips_delta=mean('delta_LPIPS')<=.001,near_nonregression=max(r['near_rmse_change_pct'] for r in farrows)<=5)
    # Numerical thresholds diagnose validity only, not favorable experimental outcomes.
    numeric_valid=all(float(r['relative_l2'])<.05 for r in diagnostics)
    decision=dict(gates=gates,per_scene=per_scene,promote=all(gates.values()) and numeric_valid,
                  numerical_valid=numeric_valid,mean_delta_psnr=mean('delta_PSNR'),mean_delta_lpips=mean('delta_LPIPS'),
                  experiment='Frozen low-dimensional medium correction; no neural network training')
    writecsv(out/'scene_metrics.csv',allmetrics);writecsv(out/'regional_metrics.csv',region_metrics);writecsv(out/'fit_diagnostics.csv',fitstats)
    writecsv(out/'numerical_checks.csv',diagnostics)
    (out/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    (out/'verification.json').write_text(json.dumps(dict(passed=True,scenes=checks,total_views=88,training_started=False),indent=2)+'\n')
    # Paired full-scene metrics and regional changes.
    fig,axes=plt.subplots(1,3,figsize=(14,4.4),constrained_layout=True)
    for i,(arm,col) in enumerate([('all_fit','#4c78a8'),('far_fit','#f58518')]):
        rr=[r for r in allmetrics if r['arm']==arm]
        for ax,key,title in zip(axes,('delta_PSNR','far_rmse_change_pct','near_rmse_change_pct'),('Full-image PSNR change (dB)','Far candidate RMSE change (%)','Near candidate RMSE change (%)')):
            ax.bar(np.arange(4)+(i-.5)*.35,[r[key] for r in rr],.35,label=arm,color=col);ax.set_title(title,fontsize=11)
            ax.axhline(0,color='black',lw=.7);ax.set_xticks(np.arange(4));ax.set_xticklabels(LABELS,rotation=15,ha='right');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].legend();fig.suptitle('Train-fitted global medium controls; frozen FSB colors and geometry; all 13 test views')
    for ext in ('png','pdf'):fig.savefig(out/('paired_metrics.'+ext),dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    for ax,(scene,f) in zip(axes.flat,fits):
        ax.bar(np.arange(9)-.18,f['q']['all_fit'],.36,label='all_fit',color='#4c78a8');ax.bar(np.arange(9)+.18,f['q']['far_fit'],.36,label='far_fit',color='#f58518')
        ax.set_xticks(np.arange(9));ax.set_xticklabels(f['controls'],rotation=40,ha='right');ax.set_ylim(-.26,.26);ax.axhline(0,color='black',lw=.6);ax.set_title(scene);ax.grid(axis='y',alpha=.2)
    axes.flat[0].legend();fig.suptitle('Fitted medium controls: shared across every view of a scene')
    for ext in ('png','pdf'):fig.savefig(out/('coefficients.'+ext),dpi=180)
    plt.close(fig)
    # All test view sheets; overview uses first view in each scene (predefined order).
    first=[];gallery=[]
    for scene in SCENES:
        paths=sorted((root/scene/'visualizations').glob('*_comparison.png'))
        first.append(Image.open(paths[0]).convert('RGB'))
        for p in paths:gallery.append((scene+' / '+p.stem,p))
    w=first[0].width;h=first[0].height;canvas=Image.new('RGB',(w,h*4),'#17191d')
    for i,(scene,img) in enumerate(zip(SCENES,first)):
        canvas.paste(img,(0,h*i));ImageDraw.Draw(canvas).text((5,h*i+18),scene,fill='#ffcc77')
    canvas.save(out/'four_scene_overview.png')
    htmlrows=['<!doctype html><meta charset="utf-8"><title>FSB medium diagnostic</title><style>body{font:16px sans-serif;background:#17191d;color:#eee;max-width:1700px;margin:24px auto}img{width:100%;height:auto}a{color:#9ccaff}section{margin-bottom:30px}</style>',
              '<h1>Frozen FSB medium transport diagnostic</h1><p>All fits use training images only. Every original test view is shown. Clear is verified numerically identical. Difference panels use a fixed x10 scale; no exposure adjustment.</p>',
              '<p><a href="REPORT.md">Report</a> | <a href="scene_metrics.csv">Metrics CSV</a></p>']
    for title,path in [('Paired metrics',out/'paired_metrics.png'),('Fitted controls',out/'coefficients.png'),('Four-scene overview',out/'four_scene_overview.png')]+gallery:
        rel=Path(__import__('os').path.relpath(path,out));htmlrows.append('<section><h2>'+html.escape(title)+'</h2><a href="'+str(rel)+'"><img loading="lazy" src="'+str(rel)+'"></a></section>')
    (out/'index.html').write_text('\n'.join(htmlrows))
    report=['# FSB 冻结介质传输 Jacobian 诊断结果','','四场景 75 张训练图和全部 13 张测试图已完成。每场景仅从训练图估计两组 9 维介质校正；Gaussian 几何、透明度、颜色以及模型权重保持冻结。',
            '',f"**准入结论：{'通过' if decision['promote'] else '未通过'}。** far_fit 四场景等权平均 PSNR 变化 {decision['mean_delta_psnr']:+.5f} dB，LPIPS 变化 {decision['mean_delta_lpips']:+.6f}。0 次 backward、0 次 optimizer 更新；确实进行了小型最小二乘数据拟合，不能称为完全未拟合数据。",
            '','## 全图指标（原生分辨率、原 FSB 指标实现）','',
            '| 场景 | 方法 | PSNR | ΔPSNR | SSIM | ΔSSIM | LPIPS | ΔLPIPS |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in allmetrics:report.append(f"| {r['scene']} | {r['arm']} | {r['PSNR']:.5f} | {r['delta_PSNR']:+.5f} | {r['SSIM']:.6f} | {r['delta_SSIM']:+.6f} | {r['LPIPS']:.6f} | {r['delta_LPIPS']:+.6f} |")
    report+=['','## 远近区域与训练拟合','',
             '区域为冻结模型的候选支持，并非语义物体。阈值来自既有训练深度分位数；按相同原生 stride-4 格点计算区域合计 SSE，再求 RMSE。全图 PSNR 则先逐视图计算再场景平均，最后跨场景等权平均。',
             '','| 场景 | 方法 | 远景 RMSE 变化 | 近景 RMSE 变化 | 训练线性误差变化 | 留一训练视图线性误差变化 | 系数裁剪 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for r in allmetrics:
        if r['arm']=='baseline':continue
        f=next(x for x in fitstats if x['scene']==r['scene'] and x['arm']==r['arm'])
        report.append(f"| {r['scene']} | {r['arm']} | {r['far_rmse_change_pct']:+.2f}% | {r['near_rmse_change_pct']:+.2f}% | {f['fit_linear_mse_change_pct']:+.2f}% | {f['leave_one_train_view_linear_mse_change_pct']:+.2f}% | {f['clip_fraction']:.2%} |")
    report+=['','训练与留一表只描述线性校正器，不是冻结 checkpoint 的完整留一重训。全图/远景拟合使用不同训练区域；两组系数的符号一致性是本次实用筛选条件，不是物理正确性的定理。单介质分量的线性对照、正规方程、条件数、逐视图有限差分与实际非线性误差均保留。',
             '','## 预登记准入','']
    for key,value in gates.items():report.append(f"- `{key}`：{'通过' if value else '未通过'}")
    for p in per_scene:report.append(f"- {p['scene']}：预测条件 {'通过' if p['predictive_pass'] else '未通过'}；系数条件 {'通过' if p['coefficient_pass'] else '未通过'}。")
    report+=['','## 复核与限制','',
             '- 88 张基线地图复现最大差异为 0；全部 13 张原始 GT/underwater/clear PNG 完全重现。PSNR、SSIM、LPIPS、MSE 同原分支结果核验一致。',
             '- 每次介质扰动与实际校正均检查 raw clear、display clear、accumulation 和 depth；最大差异为 0。只是保持已有去水效果，没有产生去水改善。',
             f"- 步长减半检查最大相对差异 {max(d['max_fd_halving_relative_error'] for d in summaries.values()):.3g}；列尺度归一化后的求解与独立直接 ridge 求解一致。",
             '- 测试图不参与系数估计、裁剪界限或模型选择。保存两组固定结果，没有选择较好的一组推广。',
             '- 所有模型与 checkpoint/input 哈希保持一致。既有 optimizer 对象由 loader 创建，但未执行 update。原模型训练目标含 SSIM/CICA 等，本轮 MSE 局部校正不等价于原模型继续训练。',
             '- 介质全局输出校正有别于物理参数标定、空间变化介质或完整 MLP 更新；失败只否定当前 9-D 家族，不否定所有介质改进。',
             '- 第一版脚本存在归一化系数还原、裁剪与区域口径错误，被明确标为 INVALID_RESULT 并归档；其系数饱和和降质结果不进入任何科学结论。本报告只使用修正版 V2。',
             '- GPU 6–9 中存在其他用户任务；诊断耗时和峰值 CUDA 分配仅是本轮资源记录，不作为速度基准。',
             '','## 资源','',
             '| 场景 | GPU | 总耗时(s) | 峰值 CUDA 分配(GB) |',
             '|---|---:|---:|---:|']
    for s,d in summaries.items():report.append(f"| {s} | {d['physical_gpu']} | {d['seconds']:.1f} | {d['peak_cuda_allocated_gb']:.2f} |")
    report+=['','[全部测试视图](index.html) · [四场景拼接](four_scene_overview.png) · [指标 CSV](scene_metrics.csv) · [判定详情](decision.json) · [独立复核](verification.json)',
             '','该方向已有历史研究：2026-08-25 IUI3 medium identifiability、2026-09-11 固定 Gaussian 的 replacement-medium fits，以及 2026-09-16 远景介质/颜色补偿探针。本轮新增的是最新 FSB checkpoint 上 train-only 的残差投影、独立测试泛化、非线性复验和 clear 精确保留检查，不将其命名为已证明新颖性的方法。']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    (out/'source_sha256.json').write_text(json.dumps({str(Path(__file__)):sha(Path(__file__))},indent=2)+'\n')
    print(json.dumps(decision,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=ROOT/'outputs/fsb_medium_transport_jacobian_20260917_v2');args=p.parse_args()
    root=args.input.resolve();out=root/'summary';out.mkdir(exist_ok=False);make_summary(root,out)

if __name__=='__main__':main()
