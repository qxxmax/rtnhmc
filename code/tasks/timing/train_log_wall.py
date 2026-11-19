#!/usr/bin/env python3
"""Part 2 (note only): training time reconstructed from existing train_log.csv files.

train_cond_phi4_kappa_explicit.py writes one row every log_interval (200) epochs with samp_per_sec = batch / (time of
that single epoch). Training time = sum over segments (e_{i-1}, e_i] of (e_i - e_{i-1}) * batch / sps_i (first segment
from epoch 0). Excludes the evaluation every 2000 epochs and process start-up. Also reported: the last wall_sec of
eval_log.csv (wall clock since the start of the last process; smaller than the total if the run was resumed).
Pre-buffer (epoch <= buffer start) and buffer phase are summed separately.
"""
import csv
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IQCB = os.environ.get('NHMC_ROOT', '.') + '/results_sync/revision/iqcb_figdata_20261117/phi4_buffer'
MODELS = os.environ.get('NHMC_ROOT', '.') + '/results_sync/models/phi4_buffer'   # packaged native models (L4/L12/L16 k0.27, k0.30)
HIM = os.path.join(ROOT, 'train_logs_him')


def one(d):
    cfg = json.load(open(os.path.join(d, 'config.json')))
    rows = list(csv.DictReader(open(os.path.join(d, 'train_log.csv'))))
    rows = [r for r in rows if r['epoch'].isdigit()]
    if not rows:
        return None
    bs = cfg['batch_size']
    bstart = (cfg.get('buffer') or {}).get('start', 10 ** 9)
    tot = pre = post = 0.0
    e_prev = 0
    for r in rows:
        e, sps = int(r['epoch']), float(r['samp_per_sec'])
        dt = (e - e_prev) * bs / sps
        tot += dt
        if e <= bstart:
            pre += dt
        else:
            post += dt
        e_prev = e
    wall = None
    ev = os.path.join(d, 'eval_log.csv')
    if os.path.isfile(ev):
        er = [r for r in csv.DictReader(open(ev)) if r.get('wall_sec')]
        if er:
            wall = float(er[-1]['wall_sec'])
    resumed = None
    tl = os.path.join(d, 'train.log')
    if os.path.isfile(tl):
        txt = open(tl, errors='replace').read()
        resumed = txt.count('EXACT RESUME')
    return dict(L=cfg['L'], T=cfg['num_steps'], n_leapfrog=cfg['n_leapfrog'], kappa=cfg['kappa_min'],
                batch=bs, epochs_cfg=cfg['epochs'], epochs_logged=e_prev, finished=e_prev >= cfg['epochs'],
                train_h_from_sps=tot / 3600, pre_buffer_h=pre / 3600, buffer_phase_h=post / 3600,
                eval_wall_last_h=None if wall is None else wall / 3600, n_resumes=resumed)


def main():
    out = dict(method=__doc__, iqcb={}, him={})
    for d in sorted(glob.glob(os.path.join(IQCB, '*'))):
        if d.endswith('_oom1') or not os.path.isfile(os.path.join(d, 'train_log.csv')):
            continue
        r = one(d)
        if r:
            r['source'] = d
            out['iqcb'][os.path.basename(d)] = r
    for d in sorted(glob.glob(os.path.join(MODELS, '*'))):
        n = os.path.basename(d)
        if n in out['iqcb'] or not os.path.isfile(os.path.join(d, 'train_log.csv')) or not os.path.isfile(os.path.join(d, 'config.json')):
            continue
        r = one(d)
        if r:
            r['source'] = d
            out['iqcb'][n] = r
    best = {}
    for d in sorted(glob.glob(os.path.join(HIM, '*', 'revision_runs', 'kscan_cpu', '*'))):
        if not os.path.isfile(os.path.join(d, 'train_log.csv')):
            continue
        r = one(d)
        if not r:
            continue
        n = os.path.basename(d)
        r['source'] = 'HIM himout/' + os.path.relpath(d, HIM)
        if n not in best or r['epochs_logged'] > best[n]['epochs_logged']:
            best[n] = r
    out['him'] = dict(sorted(best.items()))
    json.dump(out, open(os.path.join(ROOT, 'train_log_wall.json'), 'w'), indent=1)
    for src in ('iqcb', 'him'):
        for n, r in out[src].items():
            print(f"{src:4s} {n:22s} L{r['L']:<3d} T{r['T']:<3d} ep {r['epochs_logged']:6d} fin {r['finished']!s:5s} "
                  f"sps-sum {r['train_h_from_sps']:6.2f} h (pre {r['pre_buffer_h']:.2f}, buf {r['buffer_phase_h']:.2f}) "
                  f"evalwall {r['eval_wall_last_h'] if r['eval_wall_last_h'] is None else round(r['eval_wall_last_h'], 2)} "
                  f"resumes {r['n_resumes']}")


if __name__ == '__main__':
    main()
