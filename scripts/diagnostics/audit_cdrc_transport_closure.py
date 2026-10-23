#!/usr/bin/env python3
"""Post-result explanatory closure check, no thresholds or model changes."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
SCENES=('Curasao','IUI3-RedSea','JapaneseGradens-RedSea','Panama')


def read(p):
    with p.open() as f:return list(csv.DictReader(f))


def vec(r,label):return np.asarray([float(r[label+'_'+c]) for c in 'rgb'])


def csvwrite(p,rs):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,default=ROOT/'outputs/cdrc_cross_distance_diagnostic_20260917_v1');args=p.parse_args()
    root=args.input.resolve();out=root/'closure_supplement';out.mkdir(exist_ok=False)
    perpoint=[];summary=[]
    for scene in SCENES:
        predictions=read(root/scene/'prediction_rows.csv')
        correspondences=read(root/scene/'correspondences.csv')
        keyed={(r['target'],r['keypoint'],r['source']):r for r in correspondences}
        with np.load(root/scene/'native_keypoint_samples.npz') as samples:
            for r in predictions:
                if r['scope']!='closer_source':continue
                obs=keyed[(r['target'],r['keypoint'],r['source'])]
                ps=samples[r['source'].replace('/','__')+'__pred'][int(obs['source_keypoint'])].astype(np.float64)
                ss=vec(obs,'S_source');st=vec(obs,'S_target');ms=vec(obs,'M_source');mt=vec(obs,'M_target')
                source=vec(obs,'source_rgb');target=vec(obs,'target');pt=vec(obs,'renderer')
                k=st/np.maximum(ss,.1)
                target_residual=pt-target
                transported_residual=k*(source-ps)
                closure=k*(ps-ms)+mt-pt
                prediction=k*(source-ms)+mt
                residual=prediction-target
                identity=float(np.max(np.abs(residual-target_residual-transported_residual-closure)))
                assert identity<1e-12
                assert abs(float(np.mean(residual**2))-float(r['cdrc_mass_unclipped_mse']))<1e-12
                Jsm=(ps-ms)/np.maximum(ss,.1);Jtm=(pt-mt)/np.maximum(st,.1)
                rec=dict(scene=scene,split=r['split'],target=r['target'],keypoint=int(r['keypoint']),region=r['target_region'],source=r['source'],
                         source_count=int(r['source_count']),k_mean=float(k.mean()),k_gt1_fraction=float((k>1).mean()),
                         source_floor_fraction=float((ss<.1).mean()),target_floor_fraction=float((st<.1).mean()),
                         target_model_mse=float(np.mean(target_residual**2)),source_model_mse=float(np.mean((ps-source)**2)),
                         transported_source_residual_mse=float(np.mean(transported_residual**2)),model_closure_mse=float(np.mean(closure**2)),
                         model_shared_J_difference_mse=float(np.mean((Jsm-Jtm)**2)),raw_transfer_mse=float(np.mean(residual**2)),
                         target_source_cross_term=float(2*np.mean(target_residual*transported_residual)),
                         target_closure_cross_term=float(2*np.mean(target_residual*closure)),
                         source_closure_cross_term=float(2*np.mean(transported_residual*closure)),identity_max_error=identity)
                e=rec['target_model_mse']+rec['transported_source_residual_mse']+rec['model_closure_mse']+rec['target_source_cross_term']+rec['target_closure_cross_term']+rec['source_closure_cross_term']
                assert abs(e-rec['raw_transfer_mse'])<1e-12
                perpoint.append(rec)
        for split in ('train','eval'):
            for region in ('all','far'):
                rr=[r for r in perpoint if r['scene']==scene and r['split']==split and (region=='all' or r['region']==region)]
                row=dict(scene=scene,split=split,region=region,n=len(rr),target_views=len({r['target'] for r in rr}))
                for key in ('target_model','source_model','transported_source_residual','model_closure','model_shared_J_difference','raw_transfer'):
                    row[key+'_rmse']=float(np.sqrt(np.mean([r[key+'_mse'] for r in rr]))) if rr else None
                for key in ('k_mean','k_gt1_fraction','source_floor_fraction','target_floor_fraction','target_source_cross_term','target_closure_cross_term','source_closure_cross_term'):
                    row[key]=float(np.mean([r[key] for r in rr])) if rr else None
                row['multi_source_fraction']=float(np.mean([r['source_count']>=2 for r in rr])) if rr else None
                summary.append(row)
    csvwrite(out/'point_closure.csv',perpoint);csvwrite(out/'scene_closure.csv',summary)
    report=['# CDRC 冻结传输闭合补充检查','','这是看到正式诊断结果后进行的解释性补充，不修改任何准入阈值，不用于选择新模型。',
            '',r'令 K=S_t/max(S_s,0.1)，原渲染为 P_s、P_t，真实水下观测为 I_s、I_t。对无界传输的误差有精确恒等式：',
            '',r'`CDRC_raw - I_t = (P_t-I_t) + K*(I_s-P_s) + [K*(P_s-M_s)+M_t-P_t]`',
            '',r'三项分别为目标原渲染残差、迁移后的来源观测残差、原模型传输闭合残差。最后一项直接检查冻结模型自身是否符合当前跨视角单共享颜色近似。',
            '',r'三项相关且存在交叉项；不能把各自能量占比当作因果贡献百分比。主实验使用的有界裁剪不满足这一线性分解，因此这里仅分析已保留的无界控制，不改变主结果。',
            '','| 场景 | 划分 | 远景点数 | 目标原渲染 RMSE | 迁移后来源残差 RMS | 模型闭合 RMS | 无界传输 RMSE |',
            '|---|---|---:|---:|---:|---:|---:|']
    for r in summary:
        if r['region']!='far':continue
        report.append('| '+r['scene']+' | '+r['split']+' | '+str(r['n'])+' | '+' | '.join('—' if r[k] is None else f'{r[k]:.5f}' for k in ('target_model_rmse','transported_source_residual_rmse','model_closure_rmse','raw_transfer_rmse'))+' |')
    report+=['','闭合残差可以来自方向外观、多层贡献改变、几何/对应误差、介质分解不一致或截断；这一步不能区分这些原因，也不能证明介质估错。由相同 renderer 估计的“更近”位置未必具有可直接迁移的相同颜色。',
             '','每点线性与能量恒等式验证通过；source prediction 来自正式实验已保存的原生特征采样，没有额外渲染或优化。所有逐点项、交叉项、floor 激活和多来源比例均保留。']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    (out/'verification.json').write_text(json.dumps(dict(passed=True,n=len(perpoint),max_identity_error=max(r['identity_max_error'] for r in perpoint),training_started=False,post_hoc_explanatory=True),indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),constrained_layout=True)
    for ax,split in zip(axes,('train','eval')):
        rr=[r for r in summary if r['split']==split and r['region']=='far']
        for i,(key,label) in enumerate((('target_model_rmse','Target model residual'),('transported_source_residual_rmse','Transported source residual'),('model_closure_rmse','Model transport closure'))):
            ax.bar(np.arange(4)+(i-1)*.24,[r[key] if r[key] is not None else np.nan for r in rr],.24,label=label)
        ax.set_xticks(np.arange(4));ax.set_xticklabels(['Curasao','IUI3','JapaneseGradens','Panama']);ax.set_title(split+' far / closer sources');ax.set_ylabel('RGB RMS; correlated terms are NOT additive');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ymax=max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:ax.set_ylim(0,ymax)
    axes[0].legend(fontsize=8);fig.suptitle('Frozen transport closure: explanatory diagnostic, not causal attribution')
    for ext in ('png','pdf'):fig.savefig(out/('closure_comparison.'+ext),dpi=180)
    plt.close(fig)
    print((out/'REPORT.md').read_text())

if __name__=='__main__':main()
