"""Plot k-r and exact k-j views of a frozen-model probe using shared aggregation."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from analyze import read,aggregate,matrix,heat,GRID,METRICS,plt
ap=argparse.ArgumentParser();ap.add_argument('directory');ap.add_argument('--reference',required=True)
a=ap.parse_args();d=Path(a.directory)
groups=read(d/'per_target.csv.gz');reference=read(Path(a.reference)/'per_target.csv.gz')
meta=json.loads((d/'manifest.json').read_text()); records=[]
for ds,g in groups.items():
 assert g.keys()==reference[ds].keys()
 for key,t in g.items():
  assert t['weight']==reference[ds][key]['weight']
  np.testing.assert_array_equal(t['counts'],reference[ds][key]['counts'])
 for scope in ['all-lengths','common-m-ge6','m6']:
  sub={key:t for key,t in g.items() if scope=='all-lengths' or (key[1]>=6 if scope=='common-m-ge6' else key[1]==6)}
  if not sub:continue
  js=list(range(1,max(key[1] for key in sub))) if scope=='all-lengths' else list(range(1,6))
  cols=[];ns=[]
  for j in js:
   items=[(key,t) for key,t in sub.items() if key[2]==j];cols.append(aggregate(items)[0]);ns.append(len(items))
  jm=np.stack(cols,axis=1);rm=matrix(sub)
  fig,axs=plt.subplots(2,2,figsize=(17,10),layout='constrained')
  for row,(mode,mat,labels) in enumerate([('r',rm,[str(r) for r in GRID]),('j',jm,[f'{j}\nn={n}' for j,n in zip(js,ns)])]):
   for col,mi in enumerate([0,6]):
    heat(axs[row,col],mat[:,:,mi],labels,'Full suffix + answer CE' if mi==0 else 'First-step content CE (exclude <<)')
    if mode=='j':axs[row,col].set_xlabel('Explicit prefix length j (predict from step j+1)')
   for ci,v in enumerate(GRID if mode=='r' else js):
    for k in range(7):
     records.append(dict(dataset=ds,scope=scope,axis=mode,cut=str(v),k=k,**{m:float(mat[k,ci,i]) for i,m in enumerate(METRICS)}))
  note='Different samples across j columns' if scope=='all-lengths' else 'Same samples across every column'
  fig.suptitle(f"{meta['model_label']} epoch40 | {ds} | {scope}\n{note} | population-weighted token CE | lower is better",fontsize=15)
  for ext in ['png','pdf']:fig.savefig(d/f'{ds}-{scope}-kr-kj.{ext}',dpi=170)
  plt.close(fig)
with (d/'kr-kj-results.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
(d/'paired-sample-check.json').write_text(json.dumps(dict(reference=a.reference,identical_sample_ids_weights_token_counts=True,j_definition='skipped explicit steps; target steps j+1..m plus answer',no_training_weight_multiplier=True),indent=2))
print('paired sample validation passed; plots complete')
