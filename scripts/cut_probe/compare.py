"""Paired, identical-target comparison of two frozen model probes."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from analyze import read, aggregate, matrix, heat, GRID, METRICS


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('base'); ap.add_argument('fixed'); args=ap.parse_args()
    base=Path(args.base); fixed=Path(args.fixed)
    a=read(base/'per_target.csv.gz'); b=read(fixed/'per_target.csv.gz')
    results={}
    for ds in a:
        assert a[ds].keys()==b[ds].keys()
        for key in a[ds]:
            assert a[ds][key]['weight']==b[ds][key]['weight']
            np.testing.assert_array_equal(a[ds][key]['counts'],b[ds][key]['counts'])
        av,_=aggregate(list(a[ds].items())); bv,_=aggregate(list(b[ds].items()))
        results[ds]={'paired_targets':len(a[ds]),'pooled_all_targets':{m:{'base':av[:,i].tolist(),'fixed':bv[:,i].tolist()} for i,m in enumerate(METRICS)}}
        # Row-normalized by column minimum exposes preferred k independently of suffix difficulty.
        mats=[matrix(g[ds]) for g in [a,b]]
        fig,axes=plt.subplots(2,2,figsize=(16,9),layout='constrained')
        for col,mi in enumerate([0,6]):
            vmax=max((v[1:,:,mi]-v[1:,:,mi].min(axis=0)).max() for v in mats)
            for row,(label,mat) in enumerate(zip(['Base CODI','Fixed-setting CODI'],mats)):
                vals=mat[:,:,mi]-mat[1:,:,mi].min(axis=0)
                im=heat(axes[row,col],vals,list(map(str,GRID)),label+' | '+['full CE','first-step content CE'][col]+' above best k=1..6',vmin=0,vmax=vmax)
        fig.suptitle(ds+' | SAME samples and suffixes | column minimum subtracted\nLower means closer to best latent cut for this suffix; k=0 is a control',fontsize=15)
        fig.savefig(fixed/f'{ds}-comparison-relative.png');plt.close(fig)
        if ds.startswith('train'):
            fixedm=[np.stack([aggregate([(key,t) for key,t in g[ds].items() if key[1]==6 and key[2]==j])[0] for j in range(1,6)],axis=1) for g in [a,b]]
            fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
            for col,mi in enumerate([0,6]):
                for row,(label,mat) in enumerate(zip(['Base CODI','Fixed-setting CODI'],fixedm)):
                    heat(axes[row,col],mat[:,:,mi],['1/6','1/3','1/2','2/3','5/6'],label+' | '+['full suffix + answer CE','first-step content CE'][col])
                    if row==1:
                        for k,j in [(3,1),(4,1),(5,3)]:
                            from matplotlib.patches import Rectangle
                            axes[row,col].add_patch(Rectangle((j-.48,k-.48),.96,.96,fill=False,edgecolor='#ff4d4d',linewidth=2))
            fig.suptitle('Matched training samples | m=6 | n=512 | nats/token\nRed boxes: trained (k,r) settings; each panel has its own color scale',fontsize=15)
            fig.savefig(fixed/'m6-base-vs-fixed.png');fig.savefig(fixed/'m6-base-vs-fixed.pdf');plt.close(fig)
            results[ds]['m6']={label:{m:mat[:,:,i].tolist() for i,m in enumerate(METRICS)} for label,mat in zip(['base','fixed'],fixedm)}
    (fixed/'comparison.json').write_text(json.dumps(results,indent=2))
    print('Verified identical target IDs, counts and weights for both datasets.')

if __name__=='__main__': main()
