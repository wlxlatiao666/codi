"""Optional resource routing only; called before torch/CUDA imports.

Existing controllers hold their own experiment locks. A routed evaluator also
holds the destination lock until process exit. Do not add a route to an outer
lock that the calling controller already holds (flock is not recursive).
"""
import json
import os
import sys
from pathlib import Path
import fcntl


def route_evaluation(argv=None,env=None,config=None):
    argv=sys.argv[1:] if argv is None else argv
    env=os.environ if env is None else env
    config=Path(__file__).with_name('eval_routes.json') if config is None else Path(config)
    if not config.exists():return None
    def arg(name):
        for i,s in enumerate(argv):
            if s==name and i+1<len(argv):return argv[i+1]
            if s.startswith(name+'='):return s.split('=',1)[1]
        return None
    cp=arg('--checkpoint')
    if cp is None:return None
    for route in json.loads(config.read_text()):
        if env.get('CUDA_VISIBLE_DEVICES')!=route['from_gpu']:continue
        try:Path(cp).resolve().relative_to(Path(route['checkpoint_root']).resolve())
        except ValueError:continue
        lock=Path(route['lock']);lock.parent.mkdir(parents=True,exist_ok=True)
        handle=lock.open('a')
        fcntl.flock(handle,fcntl.LOCK_EX)
        env['CUDA_VISIBLE_DEVICES']=route['to_gpu']
        print('EVAL_GPU_ROUTE '+json.dumps(dict(checkpoint=cp,gpu=route['to_gpu'],lock=str(lock))),flush=True)
        return handle
    return None
