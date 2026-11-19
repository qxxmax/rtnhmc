#!/usr/bin/env python3
"""Checkpoint / resume test for cmcd_phi4.py: an uninterrupted run and a run killed at iteration K (--stop-at, the
checkpoint is written exactly as the periodic one) then resumed (--resume) must give bitwise identical parameters,
Adam state, in-training eval history and final evaluation. CPU by default; pass --device cuda to test the CUDA RNG
path (only when the GPU may be used). Small config: L = 4, T = 10, batch 64, 30 iterations, Z2 on, cmcd and dbs."""
import json
import os
import shutil
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(os.path.dirname(HERE), 'cmcd_phi4.py')
OUT = os.path.join(os.path.dirname(HERE), 'runs', 'test_resume')
dev = sys.argv[sys.argv.index('--device') + 1] if '--device' in sys.argv else 'cpu'


def run(out, extra):
    cmd = [sys.executable, SCRIPT, '--kappa', '0.27', '--L', '4', '--T', '10', '--batch', '64', '--iters', '30',
           '--n-evals', '3', '--eval-samples', '256', '--n-final', '2048', '--chunk', '1024', '--threads', '2',
           '--device', dev, '--out', out] + extra
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr); raise SystemExit('run failed')
    return r.stdout


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    res, ok = {}, True
    for alg in ('cmcd', 'dbs'):
        a, b = os.path.join(OUT, alg + '_full'), os.path.join(OUT, alg + '_resumed')
        run(a, ['--alg', alg])
        run(b, ['--alg', alg, '--stop-at', '13'])
        log = run(b, ['--alg', alg, '--resume'])
        assert 'resumed from' in log
        sa = torch.load(os.path.join(a, 'ckpt.pt'), weights_only=False)
        sb = torch.load(os.path.join(b, 'ckpt.pt'), weights_only=False)
        dm = max(float((sa['model'][k].float() - sb['model'][k].float()).abs().max()) for k in sa['model'])
        do = 0.0
        for k, v in sa['opt']['state'].items():
            for kk in ('exp_avg', 'exp_avg_sq'):
                do = max(do, float((v[kk] - sb['opt']['state'][k][kk]).abs().max()))
        ha = [(h['it'], h['elbo']) for h in sa['hist']]
        hb = [(h['it'], h['elbo']) for h in sb['hist']]
        fa = json.load(open(os.path.join(a, 'final_eval.json')))
        fb = json.load(open(os.path.join(b, 'final_eval.json')))
        same_final = all(fa[k] == fb[k] for k in ('logZ', 'ess_frac', 'abs_mag_rw', 'var_logw'))
        res[alg] = dict(max_param_diff=dm, max_adam_state_diff=do, hist_equal=ha == hb, final_equal=same_final,
                        final_logZ=fa['logZ'])
        ok &= dm == 0.0 and do == 0.0 and ha == hb and same_final
        print(alg, res[alg])
    res['device'] = dev
    res['PASS'] = bool(ok)
    json.dump(res, open(os.path.join(HERE, f'test_resume_{dev}.json'), 'w'), indent=1)
    print('PASS' if ok else 'FAIL')


if __name__ == '__main__':
    main()
