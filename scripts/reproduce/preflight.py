"""Fail closed if the code, historical cache or core environment has drifted."""
import argparse, hashlib, importlib.metadata, json, os, sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
ROOT=Path(os.environ.get('CODI_ASSET_ROOT','/home/tiger/codi-ab-c71db6e-20260909'))
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--method',choices=['codi','sim','continuation'],required=True);a=p.parse_args()
 c=json.loads((REPO/'configs/reproduction.json').read_text())
 for rel,expected in c['core_sha256'].items():assert sha(REPO/rel)==expected,('core changed',rel)
 cache=ROOT/'continuation-search-20260930/runs'/('random-cut' if a.method=='continuation' else 'base-codi')/'dataset.pt'
 assert sha(cache)==c['cache_sha256'][a.method],('training cache changed',str(cache))
 expected={'torch':'2.6.0+cu124','transformers':'4.52.4','peft':'0.15.2','accelerate':'1.7.0','datasets':'3.6.0','tokenizers':'0.21.1','numpy':'1.26.4','safetensors':'0.5.3'}
 actual={k:importlib.metadata.version(k) for k in expected}
 assert actual==expected,(actual,expected)
 assert sys.version_info[:2]==(3,11),sys.version
 for rel in ['assets/models/gpt2','continuation-search-20260930/data/validation','gsm8k-aug-id-20261014/data']:
  assert (ROOT/rel).exists(),rel
 print(json.dumps(dict(method=a.method,core_hashes_match=True,cache_sha256=sha(cache),environment=actual)),flush=True)
if __name__=='__main__':main()
