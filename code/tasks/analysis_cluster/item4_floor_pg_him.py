"""LJ13 reference floor for pair-distance W2 and geometric W2 (eq_w2) under the common-reference protocol,
same as the energy floor in item4_floor.py: 200 val 2000-subsamples (rs = default_rng(7), rs.choice(10000, 2000,
replace=False)) vs the 3 COMMON test subsamples randperm(10000, Generator(seed=rep))[:2000], rep = 0,1,2;
per draw the mean over the 3 reps; report mean +- sd over the 200 draws. Metric code = lit_metrics (same as the
method values in item4). eq_w2: M_ij = lit_metrics.eq_cost_matrix(val rows, ref cols) computed once on the union
of indices (entry-wise identical), then per (draw, rep) the assignment on the 2000x2000 submatrix of M^2."""
import sys, json, math, time, numpy as np, torch
from multiprocessing import Pool
W = '/lustre/him/him-mam/mqian/unify_20261118'
sys.path.insert(0, f'{W}/lj_buffer/code'); import lit_metrics as LM
from scipy.optimize import linear_sum_assignment
NP = int(sys.argv[1]) if len(sys.argv) > 1 else 4
O = f'{W}/floor_pg_20261118'
ref = LM.load_ref(f'{W}/lj_buffer/refs', 'lj13', 'test'); val = LM.load_ref(f'{W}/lj_buffer/refs', 'lj13', 'val')
Er = LM.energy('lj13', ref).numpy(); Ev = LM.energy('lj13', val).numpy()
Pr = LM.pair_dists(ref, 13, 3).numpy(); Pv = LM.pair_dists(val, 13, 3).numpy()
common = [torch.randperm(len(ref), generator=torch.Generator().manual_seed(rep))[:2000].numpy() for rep in range(3)]
rs = np.random.default_rng(7); VI = [rs.choice(len(val), 2000, replace=False) for _ in range(200)]
E = np.array([[math.sqrt(LM.w2_1d_sq(Er[c], Ev[vi])) for c in common] for vi in VI])
D = np.array([[math.sqrt(LM.w2_1d_sq(Pv[vi].ravel(), Pr[c].ravel())) for c in common] for vi in VI])
print('energy floor (check 0.4603+-0.0949):', E.mean(1).mean(), E.mean(1).std(ddof=1), flush=True)
print('dist floor:', D.mean(1).mean(), D.mean(1).std(ddof=1), flush=True)
UV = np.unique(np.concatenate(VI)); UR = np.unique(np.concatenate(common))
pv = {int(v): i for i, v in enumerate(UV)}; pr = {int(v): i for i, v in enumerate(UR)}
print('union sizes', len(UV), len(UR), flush=True)
xv = val[UV]; xr = ref[UR]
def rows(ch):
    torch.set_num_threads(1); return ch, LM.eq_cost_matrix(xv[ch[0]:ch[1]], xr, 13, 3).numpy()
t0 = time.time(); M = np.empty((len(UV), len(UR)))
chunks = [(a, min(a + 100, len(UV))) for a in range(0, len(UV), 100)]
with Pool(NP) as p:
    for (a, b), m in p.imap_unordered(rows, chunks): M[a:b] = m
print('cost matrix', M.shape, time.time() - t0, 's', flush=True)
M2 = M ** 2
def eq(args):
    i, rep = args
    S = M2[np.ix_([pv[int(v)] for v in VI[i]], [pr[int(v)] for v in common[rep]])]
    ri, ci = linear_sum_assignment(S); return i, rep, float(math.sqrt(S[ri, ci].mean()))
t0 = time.time(); G = np.empty((200, 3))
with Pool(NP) as p:
    for i, rep, v in p.imap_unordered(eq, [(i, r) for i in range(200) for r in range(3)], chunksize=4): G[i, rep] = v
print('assignments', time.time() - t0, 's', flush=True)
# direct cross-check on draw 0, rep 0 with the unmodified lit_metrics path
torch.set_num_threads(NP); Md = LM.eq_cost_matrix(val[VI[0]], ref[common[0]], 13, 3)
ri, ci = linear_sum_assignment((Md ** 2).numpy()); direct = float(math.sqrt((Md[ri, ci] ** 2).mean()))
print('cross-check draw0 rep0 direct', direct, 'via union matrix', G[0, 0], flush=True)
st = lambda a: dict(mean=float(np.mean(a)), sd=float(np.std(a, ddof=1)), p05=float(np.percentile(a, 5)), p95=float(np.percentile(a, 95)))
out = dict(protocol=__doc__, energy_w2=st(E.mean(1)), dist_w2=st(D.mean(1)), eq_w2=st(G.mean(1)),
           dist_w2_each=[st(D[:, j]) for j in range(3)], eq_w2_each=[st(G[:, j]) for j in range(3)],
           crosscheck_draw0_rep0=dict(direct=direct, union=float(G[0, 0])), per_draw=dict(energy=E.tolist(), dist=D.tolist(), eq=G.tolist()))
json.dump(out, open(f'{O}/item4_floor_pg.json', 'w'), indent=1)
print(json.dumps({k: out[k] for k in ('energy_w2', 'dist_w2', 'eq_w2')}, indent=1))
