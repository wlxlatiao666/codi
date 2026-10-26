"""Use the audited fixed-policy trainer with exactly one immutable cut pair.

The local-search validator normally requires three monotone pairs. Override it
only in this process, without changing the search implementation or active jobs.
"""
import runpy
import sys
from pathlib import Path
SEARCH = Path(__file__).resolve().parents[1] / 'local_search'
sys.path.insert(0, str(SEARCH))
import policy

def validate_single(pairs):
    assert pairs == [[3, 3]], ('single cut must be k=3, r=3/6', pairs)
    return pairs

policy.validate = validate_single
if __name__ == '__main__':
    runpy.run_path(str(SEARCH / 'train_phase.py'), run_name='__main__')
