"""Aggregate exact cut probes; plot comparable r grids and paired differences."""
import argparse
import csv
import gzip
from fractions import Fraction
import json
import math
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RAW_METRICS=['total','reason','answer','first_step','first8','firsttoken']
METRICS=RAW_METRICS+['first_step_body','total_body','first8_body']
GRID=[Fraction(i,12) for i in range(1,12)]

def read(path):
    result=defaultdict(dict)
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as f:
        for r in csv.DictReader(f):
            key=(int(r['row']),int(r['m']),int(r['j']))
            group=result[r['dataset']]
            if key not in group:
                group[key]=dict(weight=float(r['weight']),sums=np.zeros((7,len(METRICS))),counts=np.zeros(len(METRICS)),seen=set())
            t=group[key]; k=int(r['k']); assert k not in t['seen']; t['seen'].add(k)
            sums=[float(r[m+'_sum']) for m in RAW_METRICS]
            counts=[int(r[m+'_n']) for m in RAW_METRICS]
            sums += [sums[3]-sums[5],sums[0]-sums[5],sums[4]-sums[5]]
            counts += [counts[3]-1,counts[0]-1,counts[4]-1]
            t['sums'][k]=sums
            counts=np.array(counts); assert counts.min()>0
            if k: assert np.array_equal(t['counts'],counts)
            t['counts']=counts
    for group in result.values():
        assert all(t['seen']==set(range(7)) for t in group.values())
    return result

def choose(group,r):
    return [(key,t) for key,t in group.items() if key[2]==min(key[1]-1,max(1,(key[1]*r.numerator+r.denominator-1)//r.denominator))]

def aggregate(items):
    counts=sum(t['weight']*t['counts'] for _,t in items)
    sums=sum(t['weight']*t['sums'] for _,t in items)
    macro=sum(t['weight']*t['sums']/t['counts'] for _,t in items)/sum(t['weight'] for _,t in items)
    return sums/counts,macro

def matrix(group,grid=GRID):
    return np.stack([aggregate(choose(group,r))[0] for r in grid],axis=1)

def heat(ax,values,labels,title,*,delta=False,vmin=None,vmax=None):
    if delta:
        bound=max(abs(values.min()),abs(values.max()),.001)
        im=ax.imshow(values,cmap='RdBu_r',vmin=-bound,vmax=bound,aspect='auto')
    else:
        im=ax.imshow(values,cmap='viridis_r',aspect='auto',vmin=vmin,vmax=vmax)
    ax.set_xticks(range(len(labels)),labels,fontsize=9); ax.set_yticks(range(7),['0 (control)']+list(map(str,range(1,7))),fontsize=9)
    ax.set_xlabel('Explicit cut ratio r (ceil + clamp)'); ax.set_ylabel('Latent cut k'); ax.set_title(title,fontsize=12,pad=10)
    for k in range(7):
        for j in range(len(labels)):
            norm=im.norm(values[k,j]); color='white' if ((norm<.25 or norm>.75) if delta else norm>.55) else '#111111'
            ax.text(j,k,f'{values[k,j]:.2f}',ha='center',va='center',color=color,fontsize=8)
    plt.colorbar(im,ax=ax,fraction=.025,pad=.02,label='nats / token')
    return im

def alignment(group,metric,bootstrap=300):
    """Target and k fixed effects; cluster bootstrap by row within m strata."""
    mid=METRICS.index(metric); keys=list(group)
    rowkeys=sorted(set((key[0],key[1]) for key in keys)); rowidx={v:i for i,v in enumerate(rowkeys)}
    owners=np.array([rowidx[(key[0],key[1])] for key in keys]); ms=np.array([key[1] for key in keys])
    weights=np.array([group[key]['weight']/(key[1]-1) for key in keys])
    y=np.array([group[key]['sums'][1:,mid]/group[key]['counts'][mid] for key in keys])
    distance=np.array([[abs(k/6-key[2]/key[1]) for k in range(1,7)] for key in keys])
    yc=y-y.mean(axis=1,keepdims=True); dc=distance-distance.mean(axis=1,keepdims=True)
    def beta(w):
        yr=yc-np.average(yc,axis=0,weights=w); dr=dc-np.average(dc,axis=0,weights=w)
        return float((w[:,None]*dr*yr).sum()/(w[:,None]*dr**2).sum())
    strata=defaultdict(list)
    for i,(_,m) in enumerate(rowkeys): strata[m].append(i)
    rng=np.random.default_rng(20261020); bs=[]
    for _ in range(bootstrap):
        mult=np.zeros(len(rowkeys))
        for values in strata.values():
            draws=rng.choice(values,len(values),replace=True); mult+=np.bincount(draws,minlength=len(rowkeys))
        bs.append(beta(weights*mult[owners]))
    return dict(coefficient=beta(weights),ci95=np.quantile(bs,[.025,.975]).tolist(),
                meaning='slope of loss against abs(k/6-j/m), controlling target and k fixed effects; k=1..6',
                bootstrap='300 sample-cluster replicates, stratified by retained chain length',samples=len(rowkeys))

def paired_contrast(group,metric='first_step_body'):
    mid=METRICS.index(metric)
    early=sorted(choose(group,Fraction(1,6)))
    late=sorted(choose(group,Fraction(5,6)))
    assert [(k[0],k[1]) for k,_ in early]==[(k[0],k[1]) for k,_ in late]
    weights=np.array([t['weight'] for _,t in early])
    sums=np.array([[t['sums'][6,mid]-t['sums'][1,mid] for _,t in selected] for selected in [early,late]])
    ns=np.array([[t['counts'][mid] for _,t in selected] for selected in [early,late]])
    def measure(w):
        d=(sums*w).sum(1)/(ns*w).sum(1)
        return float(d[1]-d[0])
    strata=defaultdict(list)
    for i,(key,_) in enumerate(early): strata[key[1]].append(i)
    rng=np.random.default_rng(20261021); bs=[]
    for _ in range(300):
        mult=np.zeros(len(early))
        for values in strata.values():
            mult+=np.bincount(rng.choice(values,len(values),replace=True),minlength=len(early))
        bs.append(measure(weights*mult))
    return dict(value=measure(weights),ci95=np.quantile(bs,[.025,.975]).tolist(),
                definition='[L(6,5/6)-L(1,5/6)] - [L(6,1/6)-L(1,1/6)]',
                meaning='negative means late targets favor k6 over k1 more than early targets do',
                metric=metric,bootstrap='300 sample-cluster replicates within m strata')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('directory'); args=ap.parse_args(); dest=Path(args.directory)
    path=dest/'per_target.csv'
    groups=read(path if path.exists() else dest/'per_target.csv.gz')
    manifest=json.loads((dest/'manifest.json').read_text()); summary={}
    model_label=manifest.get('model_label','Base CODI')
    plt.rcParams.update({'font.family':'DejaVu Sans','figure.dpi':160,'savefig.dpi':190})
    labels=[str(r) for r in GRID]
    for dataset,group in groups.items():
        mat=matrix(group); np.save(dest/f'{dataset}-matrix.npy',mat)
        rows=[]
        for ci,r in enumerate(GRID):
            items=choose(group,r); assert len(items)==manifest['datasets'][dataset]
            micro,macro=aggregate(items)
            for k in range(7):
                row=dict(dataset=dataset,k=k,r=str(r),samples=len(items))
                for i,metric in enumerate(METRICS): row[metric+'_token_mean']=micro[k,i]; row[metric+'_sample_mean']=macro[k,i]
                rows.append(row)
        with (dest/f'{dataset}-grid.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        # All distinct global r intervals induced by any retained chain length.
        lengths=sorted({key[1] for key in group})
        length_rows=[]
        for m in lengths:
            for j in range(1,m):
                items=[(key,t) for key,t in group.items() if key[1]==m and key[2]==j]
                values,_=aggregate(items)
                for k in range(7):
                    record=dict(m=m,j=j,r=str(Fraction(j,m)),k=k,samples=len(items))
                    record.update({metric:values[k,i] for i,metric in enumerate(METRICS)})
                    length_rows.append(record)
        with (dest/f'{dataset}-by-length.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(length_rows[0]));writer.writeheader();writer.writerows(length_rows)
        points=sorted({Fraction(0),Fraction(1)}|{Fraction(j,m) for m in lengths for j in range(1,m-1)})
        exhaustive=[]
        for lo,hi in zip(points,points[1:]):
            representative=(lo+hi)/2; micro,macro=aggregate(choose(group,representative))
            for k in range(7):
                row=dict(k=k,r_interval=f'({lo},{hi}'+(')' if hi==1 else ']'),r_representative=str(representative))
                row.update({metric:micro[k,i] for i,metric in enumerate(METRICS)}); exhaustive.append(row)
        with (dest/f'{dataset}-all-r-intervals.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(exhaustive[0]));writer.writeheader();writer.writerows(exhaustive)
        title='Training-cache probe (stratified, population-weighted)' if dataset.startswith('train') else 'Held-out validation (equation-format probe)'
        fig,axes=plt.subplots(2,2,figsize=(16,9),layout='constrained')
        heat(axes[0,0],mat[:,:,0],labels,'Full suffix + answer CE')
        heat(axes[0,1],mat[:,:,3],labels,'First reasoning step CE')
        heat(axes[1,0],mat[:,:,0]-mat[6:7,:,0],labels,'Full CE minus k=6 on SAME targets',delta=True)
        heat(axes[1,1],mat[:,:,3]-mat[6:7,:,3],labels,'First-step CE minus k=6 on SAME targets',delta=True)
        fig.suptitle(title+f"\n{model_label} epoch40 | {manifest['datasets'][dataset]:,} samples | lower is better",fontsize=17)
        fig.savefig(dest/f'{dataset}-heatmap.png'); fig.savefig(dest/f'{dataset}-heatmap.pdf'); plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(16,4.8),layout='constrained')
        heat(axes[0],mat[:,:,0],labels,'Full suffix + answer CE')
        heat(axes[1],mat[:,:,6],labels,'First-step content CE (exclude initial << token)')
        fig.suptitle(title+f" | n={manifest['datasets'][dataset]:,} | {model_label} epoch40",fontsize=16)
        fig.savefig(dest/f'{dataset}-overview.png'); plt.close(fig)
        if 6 in lengths:
            fixed=np.stack([aggregate([(key,t) for key,t in group.items() if key[1]==6 and key[2]==j])[0] for j in range(1,6)],axis=1)
            fixed_labels=[str(Fraction(j,6)) for j in range(1,6)]
            fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
            heat(axes[0],fixed[:,:,0],fixed_labels,'Full suffix + answer CE')
            heat(axes[1],fixed[:,:,6],fixed_labels,'First-step content CE (exclude <<)')
            n=sum(key[1]==6 and key[2]==1 for key in group)
            fig.suptitle(f'{model_label} | {dataset} | m=6 retained explicit steps | n={n}',fontsize=15)
            fig.savefig(dest/f'{dataset}-m6.png'); plt.close(fig)
        # Fixed m removes ratio rounding / chain-length composition ambiguity.
        shown=[m for m in [3,4,5,6,7,8] if m in lengths]
        fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
        for ax,m in zip(axes.flat,shown):
            per_m=[]
            for j in range(1,m): per_m.append(aggregate([(key,t) for key,t in group.items() if key[1]==m and key[2]==j])[0][:,6])
            vals=np.stack(per_m,axis=1)
            n=len({key[0] for key in group if key[1]==m})
            heat(ax,vals,[str(Fraction(j,m)) for j in range(1,m)],f'm={m} retained steps; n={n}')
        for ax in list(axes.flat)[len(shown):]: ax.set_visible(False)
        fig.suptitle(title+'\nFirst-step content CE (exclude <<) at EVERY distinct cut, grouped by chain length',fontsize=16)
        fig.savefig(dest/f'{dataset}-by-length.png'); plt.close(fig)
        best={metric:[int(mat[1:,j,i].argmin())+1 for j in range(len(GRID))] for i,metric in enumerate(METRICS)}
        summary[dataset]=dict(samples=manifest['datasets'][dataset],distinct_targets=len(group),r_intervals=len(points)-1,
            display_ratios=list(map(str,GRID)),best_k=best,
            matrices={metric:mat[:,:,i].tolist() for i,metric in enumerate(METRICS)},
            paired_contrast=paired_contrast(group),
            alignment={metric:alignment(group,metric) for metric in ['total','first_step','first_step_body','first8_body']})
    (dest/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({key:{k:v for k,v in value.items() if k!='matrices'} for key,value in summary.items()},indent=2))
if __name__=='__main__': main()
