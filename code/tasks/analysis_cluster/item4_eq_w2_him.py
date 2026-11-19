"""Item 4 geometric W2 (eq_w2) on HIM: lit_metrics.eq_cost_matrix + assignment, gen subsample (same as old) vs the
COMMON ref subsample of each rep. Input eq_inputs.npz written by item4_lj13_local.py. Read-only on existing data."""
import sys, json, math, time, numpy as np, torch
from multiprocessing import Pool
sys.path.insert(0, '/lustre/him/him-mam/mqian/unify_20261118/lj_buffer/code'); import lit_metrics as LM
from scipy.optimize import linear_sum_assignment
W = '/lustre/him/him-mam/mqian/unify_20261118'
_z = np.load(f'{W}/eq_inputs.npz'); z = {k: _z[k] for k in _z.files}; _z.close()
def job(key):
    torch.set_num_threads(1); t0 = time.time()
    rep = int(key.split('_rep')[1].split('_')[0])
    g = torch.from_numpy(z[key]); r = torch.from_numpy(z[f'ref_rep{rep}'])
    M = LM.eq_cost_matrix(g, r, 13, 3); ri, ci = linear_sum_assignment((M ** 2).numpy())
    return key, float(math.sqrt((M[ri, ci] ** 2).mean())), time.time() - t0
keys = sorted(k for k in z if k.endswith('_gen'))
with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as p:
    res = {k: dict(eq_w2=v, sec=s) for k, v, s in p.imap_unordered(job, keys)}
for k in keys: print(k, res[k])
json.dump(res, open(f'{W}/item4_eq_w2_common.json', 'w'), indent=1)
