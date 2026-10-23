#!/usr/bin/env python3
"""Post-hoc bounded radiance envelope, frozen transport; no optimization."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
SCENES=('Curasao','IUI3-RedSea','JapaneseGradens-RedSea','Panama')
MAP=ROOT/'outputs/fsb_far_field_diagnostic_20260916_v1'

def read(p):
    with p.open() as f:return list(csv.DictReader(f))

def vec(r,k):return np.array([float(r[k+'_'+c]) for c in 'rgb'])

def writecsv(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def stats(gt,pred,M,S):
    lo=np.clip(M,0,1);hi=np.clip(M+S,0,1)
    feasible=np.minimum(np.maximum(gt,lo),hi)
    error=(pred-gt)**2;floor=(feasible-gt)**2
    assert np.max(np.maximum(lo-pred,pred-hi),initial=0)<2e-6
    assert np.sum(floor)<=np.sum(error)+1e-7
    return dict(pixels=len(gt),below_channels=int((gt<lo-1e-6).sum()),above_channels=int((gt>hi+1e-6).sum()),
                baseline_sse=float(error.sum()),unavoidable_sse=float(floor.sum()),
                upper_shortfall_sum=float(np.maximum(gt-hi,0).sum()),max_bound_violation=float(np.max(np.maximum(lo-pred,pred-hi),initial=0)))

def aggregate(rs):
    n=sum(r['pixels'] for r in rs);e=sum(r['baseline_sse'] for r in rs);f=sum(r['unavoidable_sse'] for r in rs)
    return dict(pixels=n,below_fraction=sum(r['below_channels'] for r in rs)/max(3*n,1),above_fraction=sum(r['above_channels'] for r in rs)/max(3*n,1),
                baseline_rmse=float(np.sqrt(e/(3*n))) if n else None,oracle_bound_rmse=float(np.sqrt(f/(3*n))) if n else None,
                unavoidable_error_energy_fraction=f/e if e>0 else None)

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=ROOT/'outputs/cdrc_cross_distance_diagnostic_20260917_v1');args=p.parse_args()
    root=args.input.resolve();out=root/'bounded_feasibility_supplement';out.mkdir(exist_ok=False)
    sparse=[];dense=[];grouped=[]
    for scene in SCENES:
        predictions=read(root/scene/'prediction_rows.csv')
        obs={(r['target'],r['keypoint'],r['source']):r for r in read(root/scene/'correspondences.csv')}
        for split in ('train','eval'):
            pp=[r for r in predictions if r['split']==split and r['scope']=='closer_source' and r['target_region']=='far']
            oo=[obs[(r['target'],r['keypoint'],r['source'])] for r in pp]
            arr=lambda k:np.stack([vec(r,k) for r in oo]) if oo else np.empty((0,3))
            st=stats(arr('target'),arr('renderer'),arr('M_target'),arr('S_target'))
            J=(arr('source_rgb')-arr('M_source'))/np.maximum(arr('S_source'),.1)
            sparse.append(dict(scene=scene,split=split,**aggregate([st]),
                               J_below0_fraction=float((J<0).mean()) if J.size else None,
                               J_above1_fraction=float((J>1).mean()) if J.size else None,
                               J_min=float(J.min()) if J.size else None,J_max=float(J.max()) if J.size else None,
                               unbounded_transfer_rmse=float(np.sqrt(np.mean([float(r['cdrc_mass_mse']) for r in pp]))) if pp else None,
                               bounded_transfer_rmse=float(np.sqrt(np.mean([float(r['cdrc_bounded_mse']) for r in pp]))) if pp else None))
        a=json.loads((MAP/scene/'audit.json').read_text());bounds=a['depth_boundaries']
        for name in sorted(a['camera_metadata']):
            split,view=name.split('/')
            with np.load(MAP/scene/'maps'/(split+'_'+view+'.npz')) as m:
                for maskname in ('all_image','surface_candidate','A_ge_0p5'):
                    base=np.ones(m['depth'].shape,bool) if maskname=='all_image' else ((m['A']>=(.9 if maskname=='surface_candidate' else .5))&np.isfinite(m['depth'])&(m['depth']>0))
                    if maskname=='surface_candidate':base&=m['cv']<=.1
                    for region in (('all',) if maskname=='all_image' else ('all','near','mid','far')):
                        mask=base.copy()
                        if region!='all':
                            lo,hi={'near':(0,bounds[0]),'mid':(bounds[0],bounds[1]),'far':(bounds[1],np.inf)}[region]
                            mask&=(m['depth']>=lo)&(m['depth']<hi)
                        dense.append(dict(scene=scene,split=split,view=view,mask=maskname,region=region,**stats(m['gt'][mask],m['pred'][mask],m['medium'][mask],m['S'][mask])))
        for split in ('train','eval'):
            for mask in ('all_image','surface_candidate','A_ge_0p5'):
                for region in (('all',) if mask=='all_image' else ('all','near','mid','far')):
                    rr=[r for r in dense if r['scene']==scene and r['split']==split and r['mask']==mask and r['region']==region]
                    grouped.append(dict(scene=scene,split=split,mask=mask,region=region,views=len(rr),**aggregate(rr)))
    writecsv(out/'sparse_far_feasibility.csv',sparse);writecsv(out/'dense_per_view.csv',dense);writecsv(out/'dense_summary.csv',grouped)
    fmt=lambda x:'—' if x is None else f'{x:.5f}'
    pct=lambda x:'—' if x is None else f'{100*x:.2f}%'
    report=['# 固定传输下 BND 可达范围补充诊断','','这是 CDRC 正式筛选结果后的解释性检查，不作为预注册结果或新方法收益。',
            '',r'对固定介质、Gaussian 位置/透明度/排序/支持，且每个参与合成的颜色均满足 0<=c_i<=1，有 `M <= I_pred <= M+S`，其中 `S=sum(w_i*t_i)`；再按正式图像输出裁剪到 [0,1]。',
            '',r'把真实 underwater GT 投影到这个区间，得到任何有界颜色调整都无法突破的**乐观误差下界**。该 oracle 每个像素/通道独立选择颜色，松弛了共享 Gaussian、FSB 参数化和跨视角一致性，所以不是可训练模型或恢复后的图像。',
            '',r'稀疏表使用原生特征点；稠密表使用固定 stride-4 原生采样格，GT 与输出位置一致，不在特征处插值。全部已有地图覆盖 75 train/13 eval。',
            '','## 稀疏远景支持：无界收益与 BND 冲突','',
            '| 场景 | 划分 | N | 原渲染 RMSE | 无界 CDRC | 有界 CDRC | 乐观 BND 下界 | 固定传输不可达误差能量 / 原误差 | 来源 J>1 |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in sparse:report.append('| '+r['scene']+' | '+r['split']+' | '+str(r['pixels'])+' | '+' | '.join(fmt(r[k]) for k in ('baseline_rmse','unbounded_transfer_rmse','bounded_transfer_rmse','oracle_bound_rmse'))+' | '+pct(r['unavoidable_error_energy_fraction'])+' | '+pct(r['J_above1_fraction'])+' |')
    report+=['','本组被支持的远景点中，反演越界均为 **J>1**，不是负颜色。Curasao/IUI3 的无界对照确有局部收益，不能概括为“跨距离信息完全无用”；但这些收益不能直接搬入保持 BND 的方法。Panama 样本极少，JapaneseGradens 没有远景样本。',
             '','## 稠密固定格验证：避免特征点筛选偏差','',
             '| 场景 | 划分 | 区域 | 格点数 | 原 RMSE | BND 下界 RMSE | 不可达误差能量比例 | GT 高于 M+S 通道比例 |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for r in grouped:
        if r['split']!='eval' or not ((r['mask']=='surface_candidate' and r['region']=='far') or r['mask']=='all_image'):continue
        report.append('| '+r['scene']+' | '+r['split']+' | '+('全图' if r['mask']=='all_image' else '稳定远景候选')+' | '+str(r['pixels'])+' | '+fmt(r['baseline_rmse'])+' | '+fmt(r['oracle_bound_rmse'])+' | '+pct(r['unavoidable_error_energy_fraction'])+' | '+pct(r['above_fraction'])+' |')
    report+=['','误差能量比例只是在“传输和几何均固定、颜色仍有界”条件下的代数限制，**不是介质错误的因果占比**。M 与 S 同时受介质、几何、opacity 和离散支持影响；未证明只改介质就能达到下界，也没有证明需要放松 BND。',
             '','后续更具体的问题是：在保留有界颜色的前提下，允许小范围、低维的传输参数变化，能否改善这些不可达像素，同时保持其它视图/近景拟合和 clear 外观。先验证冻结颜色、几何的传输灵敏度，再决定是否训练；不能把这里的 oracle 写回图像或当作伪标签。']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    (out/'verification.json').write_text(json.dumps(dict(passed=True,post_hoc_explanatory=True,views=88,model_updates=0,max_predicted_envelope_violation=max(r['max_bound_violation'] for r in dense)),indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for ax,mask,region,title in [(axes[0],'all_image','all','All image'),(axes[1],'surface_candidate','far','Stable far-field candidates')]:
        rr=[r for r in grouped if r['split']=='eval' and r['mask']==mask and r['region']==region]
        for i,(key,label) in enumerate((('baseline_rmse','Frozen renderer'),('oracle_bound_rmse','Optimistic bounded-color error floor'))):
            ax.bar(np.arange(4)+(i-.5)*.35,[r[key] for r in rr],.35,label=label)
        ax.set_xticks(np.arange(4));ax.set_xticklabels(['Curasao','IUI3','JapaneseGradens','Panama']);ax.set_title(title);ax.set_ylabel('Underwater RGB RMSE; native stride-4 samples');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].legend(fontsize=8);fig.suptitle('Fixed medium + geometry: remaining error cannot all be fixed by bounded color')
    for ext in ('png','pdf'):fig.savefig(out/('bounded_feasibility.'+ext),dpi=180)
    plt.close(fig);print((out/'REPORT.md').read_text())

if __name__=='__main__':main()
