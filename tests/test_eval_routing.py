import fcntl
import importlib.util
import json
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('eval_routing',Path(__file__).resolve().parents[1]/'scripts/reproduce/eval_routing.py')
routing=importlib.util.module_from_spec(spec);spec.loader.exec_module(routing)


def test_only_intended_evaluator_routed_and_gpu_lock_held(tmp_path):
    root=tmp_path/'aligned/runs';lock=tmp_path/'gpu3.lock';config=tmp_path/'routes.json'
    config.write_text(json.dumps([dict(checkpoint_root=str(root),from_gpu='2',to_gpu='3',lock=str(lock))]))
    for cp,gpu in [(tmp_path/'other/runs/cp','2'),(root/'cp','3')]:
        env={'CUDA_VISIBLE_DEVICES':gpu}
        assert routing.route_evaluation(['--checkpoint',str(cp)],env,config) is None
        assert env['CUDA_VISIBLE_DEVICES']==gpu
    env={'CUDA_VISIBLE_DEVICES':'2'}
    h=routing.route_evaluation(['--checkpoint',str(root/'cp')],env,config)
    assert env['CUDA_VISIBLE_DEVICES']=='3'
    with lock.open('a') as second:
        with pytest.raises(BlockingIOError):fcntl.flock(second,fcntl.LOCK_EX|fcntl.LOCK_NB)
        h.close();fcntl.flock(second,fcntl.LOCK_EX|fcntl.LOCK_NB)
