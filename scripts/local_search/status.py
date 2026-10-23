"""Compact read-only status; can run on the CPU devbox sharing the filesystem."""
import json
import time
from runtime import EXP

result = {'experiment': str(EXP), 'queried_at': time.time()}
for name in ['controller-process', 'search-progress', 'smoke-verification', 'search-error', 'smoke-error']:
    path = EXP / (name + '.json')
    if path.exists():
        result[name] = json.loads(path.read_text())
runs = []
for path in (EXP / 'runs').glob('*/progress.json'):
    info = json.loads(path.read_text())
    info['run'] = path.parent.name
    info['seconds_since_update'] = round(time.time() - info['updated'])
    throughput = path.parent / 'throughput.json'
    if throughput.exists():
        info['throughput'] = json.loads(throughput.read_text())
    runs.append(info)
result['runs'] = runs
result['decisions'] = [json.loads(p.read_text())['decision']
                       for p in sorted(EXP.glob('round-*-selection.json'))]
print(json.dumps(result, indent=2))
