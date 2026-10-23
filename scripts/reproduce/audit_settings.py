"""Audit recorded launch hyperparameters without importing torch or starting training."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
METHODS=['codi','sim','fixed_original','fixed_control','all_cuts','aligned_step','window_suffix']
METHOD_KEYS={'use_decoder','continuation_weight'}
OPERATIONAL={'output_dir','save_total_limit','print_loss'}
def audit():
    launches={}
    for method in METHODS:
        paths=list((ROOT/'results/provenance'/method).glob('*launch*.json'))
        assert len(paths)==1,(method,paths)
        cmd=json.loads(paths[0].read_text())['command']
        flags=dict(zip(cmd[2::2],cmd[3::2]))
        launches[method]={k.removeprefix('--'):v for k,v in flags.items()}
    shared={k:v for k,v in launches['codi'].items() if k not in METHOD_KEYS|OPERATIONAL}
    for method,flags in launches.items():
        actual={k:v for k,v in flags.items() if k not in METHOD_KEYS|OPERATIONAL}
        assert actual==shared,(method,{k:(shared.get(k),actual.get(k)) for k in shared.keys()|actual.keys() if shared.get(k)!=actual.get(k)})
        assert flags['use_decoder']==str(method=='sim')
        assert float(flags['continuation_weight'])==(0 if method in ['codi','sim'] else .3)
    ref=json.loads((ROOT/'configs/reproduction.json').read_text())['training_reference']
    starts={}
    for method in ['all_cuts','aligned_step','window_suffix']:
        d=json.loads((ROOT/'results/provenance'/method/'start-0.json').read_text())
        for key,expected in [('samples',ref['train_samples']),('max_steps',ref['max_steps']),('main_data_sha256',ref['main_data_sha256']),('model_state_sha256',ref['initial_state_sha256'])]:
            assert d[key]==expected,(method,key)
        starts[method]={k:d[k] for k in ['samples','max_steps','main_data_sha256','model_state_sha256']}
    return dict(passed=True,shared_training_parameters=shared,method_differences={m:{k:v for k,v in d.items() if k in METHOD_KEYS} for m,d in launches.items()},operational_differences={m:{k:v for k,v in d.items() if k in OPERATIONAL} for m,d in launches.items()},new_methods_initialization_and_data=starts,limitations=['SIM-CoT uses its original independent decoder and explain_loss_factor=1, not continuation loss=0.3.','Auxiliary target eligibility and aggregation differ intentionally by method.','fixed_control resumes the shared epoch8 fixed-policy warmup; it is not the same run as fixed_original.','Equal seed and shared initialization do not imply identical dropout RNG trajectories across different forward methods.'])
if __name__=='__main__':
    result=audit();dest=ROOT/'results/settings-audit.json';dest.write_text(json.dumps(result,indent=2)+'\n');print('PASS: shared hyperparameters, main data and new-method initialization match')
