"""Reaggregate exhaustive per-target scores by absolute explicit cut j."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from analyze import read, aggregate, heat, METRICS, plt


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base',default='experiments/base-codi-cut-probe-20261020')
    ap.add_argument('--fixed',default='experiments/fixed-codi-cut-probe-20261021')
    ap.add_argument('--output',default='experiments/cut-probe-absolute-j')
    args=ap.parse_args(); dest=Path(args.output); dest.mkdir(parents=True,exist_ok=True)
    paths=[Path(args.base),Path(args.fixed)]
    groups=[read(p/'per_target.csv.gz') for p in paths]
    labels=['Base CODI','Fixed-setting CODI']
    provenance=[]
    for p in paths:
        manifest=json.loads((p/'manifest.json').read_text())
        provenance.append(dict(directory=str(p.resolve()),checkpoint=manifest['checkpoint'],
            checkpoint_sha256=manifest['checkpoint_sha256'],
            raw_scores_sha256=hashlib.sha256((p/'per_target.csv.gz').read_bytes()).hexdigest()))
    records=[]; summary={}
    for ds in groups[0]:
        ga,gb=[g[ds] for g in groups]
        assert ga.keys()==gb.keys()
        for key in ga:
            assert ga[key]['weight']==gb[key]['weight']
            np.testing.assert_array_equal(ga[key]['counts'],gb[key]['counts'])
        lengths=sorted({key[1] for key in ga})
        scopes=[('all-lengths',lambda key:True,max(lengths)-1),
                ('common-m-ge6',lambda key:key[1]>=6,5)]
        scopes.extend((f'm{m}',lambda key,m=m:key[1]==m,m-1) for m in lengths)
        summary[ds]={}
        for scope,predicate,maxj in scopes:
            js=list(range(1,maxj+1)); mats=[]; ns=[]
            for model,group in zip(labels,[ga,gb]):
                cols=[]; counts=[]
                for j in js:
                    items=[(key,t) for key,t in group.items() if predicate(key) and key[2]==j]
                    assert items
                    micro,macro=aggregate(items); cols.append(micro);counts.append(len(items))
                    for k in range(7):
                        rec=dict(dataset=ds,scope=scope,model=model,k=k,j=j,samples=len(items))
                        rec.update({metric:float(micro[k,i]) for i,metric in enumerate(METRICS)})
                        records.append(rec)
                mats.append(np.stack(cols,axis=1));ns.append(counts)
            assert ns[0]==ns[1]
            if scope!='all-lengths':assert len(set(ns[0]))==1
            summary[ds][scope]=dict(j=js,samples=ns[0],best_k={model:{metric:(mat[1:,:,mi].argmin(axis=0)+1).tolist() for mi,metric in enumerate(METRICS)} for model,mat in zip(labels,mats)})
            fig,axes=plt.subplots(2,2,figsize=(max(14,1.2*maxj+3),9),layout='constrained')
            for row,(model,mat) in enumerate(zip(labels,mats)):
                for col,(mi,title) in enumerate([(0,'Full suffix + answer CE'),(6,'First-step content CE (exclude <<)')]):
                    heat(axes[row,col],mat[:,:,mi],[f'{j}\nn={n}' for j,n in zip(js,ns[0])],model+'\n'+title)
                    axes[row,col].set_xlabel('Explicit prefix length j (predict from step j+1)')
                    if row==1 and scope.startswith('m') and scope!='common-m-ge6':
                        from matplotlib.patches import Rectangle
                        m=int(scope[1:])
                        for k,q in [(3,2),(4,2),(5,4)]:
                            j=min(m-1,max(1,(m*q+5)//6))
                            axes[row,col].add_patch(Rectangle((j-1-.48,k-.48),.96,.96,fill=False,edgecolor='#ff4d4d',linewidth=2))
            title='Training-cache probe (population-weighted)' if ds.startswith('train') else 'Validation equation-format probe'
            subtitle='All eligible lengths; different samples across j' if scope=='all-lengths' else ('Same samples in every column; m >= 6; j=1..5' if scope=='common-m-ge6' else f'Same samples in every column; m={scope[1:]}; red boxes = trained cuts')
            fig.suptitle(title+'\n'+subtitle+'\nExact j | independent color scales | lower is better',fontsize=14)
            fig.savefig(dest/f'{ds}-{scope}.png',dpi=180)
            if scope in ['all-lengths','common-m-ge6','m6']:fig.savefig(dest/f'{ds}-{scope}.pdf')
            plt.close(fig)
    with (dest/'absolute-j-results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    (dest/'summary.json').write_text(json.dumps(summary,indent=2))
    (dest/'manifest.json').write_text(json.dumps(dict(sources=provenance,reused_existing_exhaustive_forward_scores=True,
        j_definition='number of retained explicit equation steps skipped; predict steps j+1..m plus answer; no gold explicit prefix supplied',
        j_range='1..m-1 proper suffixes; j=0 full-chain and j=m answer-only are outside this probe',
        aggregation='weighted token CE: sum(weight * NLL) / sum(weight * tokens); no 0.3 multiplier',
        scopes='all-lengths uses all m>j; common-m-ge6 keeps same cohort for j1..5; mN fixes chain length',
        paired_target_validation='identical sample IDs, m, j, target token counts, weights; all k0..6 present'),indent=2))
    print(json.dumps({ds:{scope:v for scope,v in scopes.items() if scope in ['all-lengths','common-m-ge6','m6']} for ds,scopes in summary.items()},indent=2))

if __name__=='__main__':main()
