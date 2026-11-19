#!/usr/bin/env python3
"""Bitwise resume test of rt_cmcd_phi4.py (CPU, cmcd27 model, tiny budget, window extension on): straight run vs runs
killed by RT_CMCD_KILL_AT (a) inside the first segment, (b) inside an extension segment, then resumed. All run
summaries (means, taus, acceptance, holding, n_transitions) must be identical. Usage: [tag kill_at]."""
import json
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
PY = sys.executable
CT = str(HERE.parent)                                   # repo copy: revision/cmcd_torch
MODEL = os.environ.get('CMCD27_MODEL', str(pathlib.Path(os.environ.get('NHMC_ROOT', '.')) / 'models/cmcd_torch_L8/'
                                                         'cmcdz2_k0.27_s0_T80_dt0.1_it10000/model.pt'))
OUT = HERE / 'resume_test'
OUT.mkdir(exist_ok=True)
ARGS = ['--model', MODEL, '--kappa', '0.27', '--methods', 'imh', 'rt',
        'rt_partial:40', '--n-chains', '8', '--path-budget', '640', '--n-runs', '2', '--burn-fraction', '0.2',
        '--window-ratio', '50', '--max-paths', '12000', '--device', 'cpu', '--threads', '1']


def run(out, kill=None):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1')
    if kill is not None:
        env['RT_CMCD_KILL_AT'] = str(kill)
    r = subprocess.run([PY, f'{CT}/rt_cmcd_phi4.py'] + ARGS + ['--out', str(out)], env=env, capture_output=True,
                       text=True, cwd=CT)
    return r.returncode, r.stdout[-1500:] + r.stderr[-1500:]


def summary(path):
    d = json.load(open(path))
    out = {}
    for m, e in d['methods'].items():
        out[m] = [({k: r.get(k) for k in ('acceptance', 'n_transitions', 'extensions', 'burn')},
                   {kk: (r[kk]['mean'], r[kk].get('tau_int')) for kk in ('absM', 'M2', 'action', 'sign')},
                   r.get('holding')) for r in e['runs']]
    return out


cases = [('killseg0', 30), ('killext', 100)] if len(sys.argv) < 3 else [(sys.argv[1], int(sys.argv[2]))]
if not (OUT / 'cmcd_straight.json').exists():
    rc, log = run(OUT / 'cmcd_straight.json')
    assert rc == 0, log
A = summary(OUT / 'cmcd_straight.json')
ok = True
for tag, kill in cases:
    f = OUT / f'cmcd_{tag}.json'
    for g in OUT.glob(f'cmcd_{tag}*'):
        g.unlink()
    rc1, log1 = run(f, kill)
    rc2, log2 = run(f)
    B = summary(f)
    same = A == B
    ok &= same and rc1 == 3 and rc2 == 0
    print(tag, 'kill rc', rc1, 'resume rc', rc2, 'identical:', same,
          '| resume lines:', [l for l in log2.splitlines() if 'resumed' in l][:4], flush=True)
print('PASS' if ok else 'FAIL')
