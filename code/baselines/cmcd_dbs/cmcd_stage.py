#!/usr/bin/env python3
"""Windows/torch_env port of improve/cmcd_torch/launch_repro.sh (same python commands and arguments; only
python path, --device cuda and --threads 2 differ). Stage: cmcd27 | cmcd30 | dbs27 | dbs30.
Train (resume if ckpt.pt exists, skip if final_eval.json exists), then the RT/IMH/SNIS chains (skip if done)."""
import os
import subprocess
import sys

D = os.environ.get('CMCD_TORCH_DIR', os.path.dirname(os.path.abspath(__file__)))   # repo copy: this folder
PY = sys.executable
stage = sys.argv[1]
dev = sys.argv[2] if len(sys.argv) > 2 else 'cuda'
# stage = <alg><kappa*100>, e.g. cmcd27, dbs30, cmcd28, dbs20 (kappa .20-.30)
alg = 'cmcd' if stage.startswith('cmcd') else 'dbs'
k = '0.' + stage[len(alg):]
assert alg in ('cmcd', 'dbs') and len(stage) - len(alg) == 2, stage
family = '--family' in sys.argv          # model-quality family (2026-11-04): snis imh rt rtp:40, window rule, burn .2
out = f'runs/{alg}z2_k{k}_s0_T80_dt0.1_it10000'
os.chdir(D)
os.makedirs(out, exist_ok=True)


def run(cmd, log):
    print('RUN', ' '.join(cmd), flush=True)
    with open(log, 'a') as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc:
        raise SystemExit(f'rc={rc} for {cmd[1]}')


if not os.path.exists(f'{out}/final_eval.json'):
    cmd = [PY, '-u', 'cmcd_phi4.py', '--alg', alg, '--kappa', k, '--T', '80', '--dt', '0.1', '--iters', '10000',
           '--batch', '500', '--lr', '5e-4', '--seed', '0', '--n-final', '102400', '--chunk', '4096', '--device', dev,
           '--threads', '2', '--ckpt-min', '10', '--out', out]
    if os.path.exists(f'{out}/ckpt.pt'):
        cmd.append('--resume')
    run(cmd, f'{out}/train.log')
if not os.path.exists(f'{out}/rt_chains.json'):
    meths = ['snis', 'imh', 'rt', 'rt_partial:40'] if family else ['snis', 'imh', 'rt', 'rt_partial:40', 'rt_partial:72']
    extra = ['--burn-fraction', '0.2', '--window-ratio', '50', '--max-paths', str(2 ** 22)] if family else []
    run([PY, '-u', 'rt_cmcd_phi4.py', '--model', f'{out}/model.pt', '--kappa', k, '--methods', *meths, *extra,
         '--n-chains', '256', '--path-budget', '500000', '--n-runs', '3',
         '--init', 'proposal', '--device', dev, '--threads', '2', '--out', f'{out}/rt_chains.json'], f'{out}/rt.log')
print('stage done', stage, flush=True)
