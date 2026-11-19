"""Ad-hoc check: rt_cmcd_phi4.py --save-final, straight run vs killed (RT_CMCD_KILL_AT) + resumed: the final-state
files (x and carried IMH record W) and the run summaries must be identical."""
import json, os, pathlib, subprocess, sys
import torch
R = str(pathlib.Path(__file__).resolve().parents[3])                     # repo root
CT = R + '/revision/cmcd_torch'
OUT = pathlib.Path(__file__).resolve().parent / 'savefinal_resume'; OUT.mkdir(exist_ok=True)
for f in OUT.glob('*'):
    f.unlink()
ARGS = ['--model', R + '/results_sync/models/cmcd_torch/cmcdz2_k0.27_s0_T80_dt0.1_it10000/model.pt', '--kappa', '0.27',
        '--methods', 'imh', 'rt_partial:40', '--n-chains', '8', '--path-budget', '640', '--n-runs', '2', '--burn-fraction', '0.2',
        '--window-ratio', '50', '--max-paths', '12000', '--device', 'cpu', '--threads', '1', '--save-final', '--seed', '20261105']
def run(out, kill=None):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1')
    if kill is not None:
        env['RT_CMCD_KILL_AT'] = str(kill)
    r = subprocess.run([sys.executable, CT + '/rt_cmcd_phi4.py'] + ARGS + ['--out', str(out)], env=env, capture_output=True, text=True, cwd=CT)
    return r.returncode, r.stdout[-800:] + r.stderr[-800:]
rc, log = run(OUT / 'a.json'); assert rc == 0, log
ok = True
for kill in (30, 100):
    tag = f'k{kill}'
    rc1, l1 = run(OUT / f'{tag}.json', kill)
    rc2, l2 = run(OUT / f'{tag}.json')
    same = True
    for meth in ('imh', 'rt_partial40'):
        for r in (0, 1):
            fa = torch.load(OUT / f'a_final_{meth}_run{r}.pt', weights_only=False)
            fb = torch.load(OUT / f'{tag}_final_{meth}_run{r}.pt', weights_only=False)
            same &= bool(torch.equal(fa['x'], fb['x']))
            same &= (fa['W'] is None and fb['W'] is None) or bool(torch.equal(fa['W'], fb['W']))
            same &= fa['n_transitions'] == fb['n_transitions'] and fa['record_source'] == fb['record_source']
    ja, jb = json.load(open(OUT / 'a.json')), json.load(open(OUT / f'{tag}.json'))
    for m in ja['methods']:
        for ra, rb in zip(ja['methods'][m]['runs'], jb['methods'][m]['runs']):
            same &= all(ra[k]['mean'] == rb[k]['mean'] and ra[k]['tau_int'] == rb[k]['tau_int'] for k in ('absM', 'sign'))
            same &= ra['n_transitions'] == rb['n_transitions'] and ra['acceptance'] == rb['acceptance']
    ok &= same and rc1 == 3 and rc2 == 0
    print(tag, 'kill rc', rc1, 'resume rc', rc2, 'identical', same, flush=True)
print('PASS' if ok else 'FAIL')
