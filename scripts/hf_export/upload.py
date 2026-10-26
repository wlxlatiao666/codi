"""Upload prepared final-checkpoint bundles; verify remote contents at commit SHA."""
import argparse,hashlib,json
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/tmp/codi-hf-upload'));ap.add_argument('--names',nargs='+',default=['codi-base-gpt2','codi-fixed-three-cut-gpt2','codi-all-cuts-gpt2']);a=ap.parse_args()
 names=a.names
 # Check all selected bundles before any remote repository mutation.
 manifests={}
 for name in names:
  p=a.root/name;m=json.loads((p/'manifest.json').read_text())
  assert m['repo_id']=='wlxlatiao/'+name and m['private'] is True
  for f,d in m['files'].items():assert (p/f).stat().st_size==d['bytes'] and sha(p/f)==d['sha256'],(name,f)
  manifests[name]=m
 api=HfApi();assert api.whoami()['name']=='wlxlatiao','Log in as wlxlatiao first'
 completed={}
 for name,m in manifests.items():
  repo=m['repo_id'];folder=a.root/name
  api.create_repo(repo_id=repo,repo_type='model',private=True,exist_ok=True)
  assert api.model_info(repo).private,'Existing target must be private before uploading'
  commit=api.upload_folder(repo_id=repo,folder_path=folder,repo_type='model',ignore_patterns=['.cache/**','.git/**'],commit_message='Archive final epoch40 CODI checkpoint with training state and evaluation provenance')
  info=api.model_info(repo,revision=commit.oid,files_metadata=True)
  files={f.rfilename:f for f in info.siblings}
  expected=dict(m['files']);expected['manifest.json']=dict(bytes=(folder/'manifest.json').stat().st_size,sha256=sha(folder/'manifest.json'))
  for f,d in expected.items():
   remote=files[f];assert remote.size==d['bytes'],(repo,f,'size')
   if remote.lfs is not None:
    actual=remote.lfs.sha256 if hasattr(remote.lfs,'sha256') else remote.lfs['sha256']
   else:
    q=hf_hub_download(repo_id=repo,filename=f,revision=commit.oid);actual=sha(q)
   assert actual==d['sha256'],(repo,f,'hash')
  completed[name]=dict(repo_id=repo,url='https://huggingface.co/'+repo,commit=commit.oid,private=info.private,verified_files=len(expected),sha256_verified=True)
  (a.root/'upload-results.json').write_text(json.dumps(completed,indent=2)+'\n')
  print(json.dumps(completed[name]),flush=True)
if __name__=='__main__':main()
