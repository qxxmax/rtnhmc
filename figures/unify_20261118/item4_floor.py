"""Item 4 reference floor for the 1D metrics: val split vs test split (energy W2, distance W2), n=2000.
(a) old floor20 json (independent 2000-subsamples of both); (b) val 2000-subsample vs the 3 COMMON test subsamples;
(c) val 2000-subsample vs the full test split (10000) = floor for the 'full_ref10000' variant."""
import os
import sys, json, math, numpy as np, torch
sys.path.insert(0, os.environ.get('NHMC_ROOT', '.') + '/particle'); import lit_metrics as LM
D = os.environ.get('NHMC_ROOT', '.') + '/results_sync/revision/unify_20261118/lj13_data'
f20 = json.load(open(os.environ.get('NHMC_ROOT', '.') + '/results_sync/particle_202610/lit/stat2/floor20_lj13.json'))
ref = LM.load_ref(D, 'lj13', 'test'); val = LM.load_ref(D, 'lj13', 'val')
Er = LM.energy('lj13', ref).numpy(); Ev = LM.energy('lj13', val).numpy(); Pr = LM.pair_dists(ref, 13, 3).numpy(); Pv = LM.pair_dists(val, 13, 3).numpy()
common = [torch.randperm(len(ref), generator=torch.Generator().manual_seed(rep))[:2000].numpy() for rep in range(3)]
rs = np.random.default_rng(7); B = []; C = []
for i in range(200):
    vi = rs.choice(len(val), 2000, replace=False)
    B.append([math.sqrt(LM.w2_1d_sq(Er[c], Ev[vi])) for c in common]); C.append((math.sqrt(LM.w2_1d_sq(Er, Ev[vi])), math.sqrt(LM.w2_1d_sq(Pr.ravel(), Pv[vi].ravel()))))
B = np.array(B); C = np.array(C)
st = lambda a: dict(mean=float(np.mean(a)), sd=float(np.std(a, ddof=1)), p05=float(np.percentile(a, 5)), p95=float(np.percentile(a, 95)))
out = dict(floor20_old=dict(energy_w2=st([r['energy_w2'] for r in f20['reps']]), dist_w2=st([r['dist_w2'] for r in f20['reps']]), eq_w2=st([r['eq_w2'] for r in f20['reps']])),
           val2000_vs_common_ref_mean3=st(B.mean(1)), val2000_vs_common_ref_each=[st(B[:, j]) for j in range(3)],
           val2000_vs_fullref_energy=st(C[:, 0]), val2000_vs_fullref_dist=st(C[:, 1]))
print(json.dumps(out, indent=1))
json.dump(out, open(os.environ.get('NHMC_ROOT', '.') + '/results_sync/revision/unify_20261118/item4_lj13_floor.json', 'w'), indent=1)
