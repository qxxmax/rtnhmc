#!/usr/bin/env python3
"""Collect the HIM timing benchmark (job 94496, out/main + out/repeat) into results.json and table_draft.md.

Training: train_log.csv of the unmodified training script with --log-interval 1 (samp_per_sec = 48 / epoch time).
Epochs 1-10 are warm-up, epochs 11-60 are timed. Phase A: no buffer term (as epochs 1-1000 of a real run); phase B:
buffer term on from epoch 1 (as epochs 1001-20000); with --buffer-every 2 the even epochs also advance the 64 RT
chains by one step, so B is reported per two-epoch cycle: (median even + median odd) / 2.
Sampling: bench_sample.py JSONs. Counts: exact from cond_kernels.CondNHMCResidual2Kernel / CondZ2Wrapper and
phi4_gpu.fwd_partial / rev_partial (see COUNTS below).
"""
import csv
import json
import os
import statistics as stt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, 'raw')          # copy of HIM bench_timing_20261118/out
WARM, NT = 10, 50

CONFIGS = [  # name, L, T, role
    ('nat_L4_T16', 4, 16, 'native'), ('nat_L8_T16', 8, 16, 'native'), ('nat_L12_T16', 12, 16, 'native'),
    ('nat_L16_T16', 16, 16, 'native'),
    ('ks_L8_T4', 8, 4, 'Tscan'), ('ks_L8_T8', 8, 8, 'Tscan'), ('ks_L8_T16', 8, 16, 'Tscan'),
    ('ks_L8_T32', 8, 32, 'Tscan'), ('ks_L8_T64', 8, 64, 'Tscan'),
    ('ks_L16_T8', 16, 8, 'Tscan'), ('ks_L16_T32', 16, 32, 'Tscan'), ('ks_L16_T64', 16, 64, 'Tscan'),
]
NLF = 5


def counts(T, method, d=None):
    """per chain per update, as executed. One stage (forward_step or reverse_step of the resid2 kernel):
    leapfrog with n_lf drifts and n_lf + 1 kicks, each kick = one -grad S (phi4_force) + one learned-force net;
    plus 4 momentum nets (fwd mu, fwd sigma, bwd mu, bwd sigma). Path end: one action S."""
    st = T if method == 'imh' else 2 * T if method == 'rt' else 2 * d
    return dict(stages=st, grad_S=(NLF + 1) * st, learned_force_net=(NLF + 1) * st, momentum_nets=4 * st,
                drifts=NLF * st, action_S=1 if method == 'imh' else 2,
                note='plus 3 observable evaluations per update (x, y, new state), each with one action S')


def train_times(path):
    rows = [r for r in csv.DictReader(open(path)) if r['epoch'].isdigit()]
    dt = {int(r['epoch']): 48.0 / float(r['samp_per_sec']) for r in rows}
    ep = [e for e in sorted(dt) if WARM < e <= WARM + NT]
    if len(ep) < NT:
        return None
    v = [dt[e] for e in ep]
    ev = [dt[e] for e in ep if e % 2 == 0]
    od = [dt[e] for e in ep if e % 2 == 1]
    return dict(n=len(v), median=stt.median(v), mean=stt.mean(v), sd=stt.stdev(v), median_even=stt.median(ev),
                median_odd=stt.median(od), cycle=(stt.median(ev) + stt.median(od)) / 2)


def main():
    res = dict(
        job=dict(cluster='HIMSTER3', job_id=94496, node='him0046', dir='/lustre/him/him-mam/mqian/bench_timing_20261118',
                 test_salloc=94486),
        device=None, method=__doc__, configs={}, series={},
        definitions=dict(
            update='one transition of 4 chains held in one batched tensor (longchain.py step loop); s/update is the '
                   'median over 200 timed updates after 20 warm-up updates',
            forward_path='PhiStepper.start = base draw + T forward stages + action; s per path at batch 1024 '
                         '(SNIS / pre-generated path-IMH proposals) and at batch 4',
            prt_d_T4='pRT with return depth d = T/4 (s = 3T/4)', prt_d4='pRT with fixed return depth d = 4 (s = T - 4); '
                     'at T = 16 both are s = 12 (paper); at T = 4, d = 4 is the full RT',
            train_epoch='s per epoch, batch 48, median of epochs 11-60; A = before the buffer term (epochs 1-1000 of '
                        'a run), B = buffer phase per two-epoch cycle (epochs 1001-20000)'))
    for tag in ('main', 'repeat'):
        envf = os.path.join(RAW, tag, 'env_start.txt')
        if os.path.isfile(envf) and res['device'] is None:
            txt = open(envf).read().splitlines()
            g = lambda k: next((l.split(':', 1)[1].strip() for l in txt if l.startswith(k)), None)
            res['device'] = dict(host_line=txt[0], cpu=g('Model name'), sockets=g('Socket(s)'),
                                 cores_per_socket=g('Core(s) per socket'), threads_per_core=g('Thread(s) per core'),
                                 torch=next((l.split()[1] for l in txt if l.startswith('torch ')), None),
                                 threads=4, pinning='taskset to 4 cores of an 80-core allocation; sequential runs',
                                 load_start=next((l for l in txt if 'load average' in l), None))
    for name, L, T, role in CONFIGS:
        c = dict(L=L, T=T, V=L * L, role=role, n_leapfrog=NLF, kappa=0.27)
        for tag in ('main', 'repeat'):
            o = os.path.join(RAW, tag)
            tr = {ph: train_times(os.path.join(o, f'train_{name}_{ph}', 'train_log.csv'))
                  for ph in 'AB' if os.path.isfile(os.path.join(o, f'train_{name}_{ph}', 'train_log.csv'))}
            sj = os.path.join(o, f'sample_{name}.json')
            if not tr and not os.path.isfile(sj):
                continue
            e = dict(train=tr)
            if tr.get('A') and tr.get('B'):
                e['train_20000_epochs_h_est'] = (1000 * tr['A']['median'] + 19000 * tr['B']['cycle']) / 3600
            if os.path.isfile(sj):
                z = json.load(open(sj))
                m = {}
                for k, v in z['methods'].items():
                    meth = 'rtp' if k.startswith('rtp') else k
                    d = v['depth'] if meth == 'rtp' else None
                    lab = k
                    if meth == 'rtp':
                        lab = 'prt_d4' if d == 4 else ('prt_dT4' if d == T // 4 else k)
                        if d == 4 and d == T // 4:
                            lab = 'prt_d4=dT4'
                    m[lab] = dict(key=k, s=v['s'], depth=d, sec_per_update=v['sec_per_update'],
                                  counts=counts(T, meth, d), accepted_in_timed=v['accepted'])
                e['sampling'] = m
                e['forward_path'] = dict(batch1024_s_per_path=z['fwd_batch_large']['sec_per_path_median'],
                                         batch4_s_per_path=z['fwd_batch_small']['sec_per_path_median'],
                                         batch1024_calls=z['fwd_batch_large']['sec_per_call'],
                                         counts=counts(T, 'imh'))
                e['affinity'] = z.get('affinity')
            c[tag] = e
        res['configs'][name] = c
    # plot-ready series (main run): s/update vs T at fixed L, vs V at fixed T = 16
    def su(cn, lab):
        s = res['configs'][cn].get('main', {}).get('sampling', {})
        for k, v in s.items():
            if k == lab or (lab in ('prt_d4', 'prt_dT4') and k == 'prt_d4=dT4'):
                return v['sec_per_update']['median']
        if lab == 'prt_d4' and res['configs'][cn]['T'] == 4:      # d = 4 at T = 4 is the full RT
            return s.get('rt', {}).get('sec_per_update', {}).get('median')
        return None
    meths = ('imh', 'rt', 'prt_dT4', 'prt_d4')
    ser = {}
    for Lf, names in ((8, ['ks_L8_T4', 'ks_L8_T8', 'ks_L8_T16', 'ks_L8_T32', 'ks_L8_T64']),
                      (16, ['ks_L16_T8', 'nat_L16_T16', 'ks_L16_T32', 'ks_L16_T64'])):
        ser[f'vs_T_L{Lf}'] = {m: dict(T=[res['configs'][n]['T'] for n in names], sec=[su(n, m) for n in names])
                              for m in meths}
        ser[f'vs_T_L{Lf}']['forward_path_batch1024'] = dict(
            T=[res['configs'][n]['T'] for n in names],
            sec=[res['configs'][n].get('main', {}).get('forward_path', {}).get('batch1024_s_per_path') for n in names])
    names = ['nat_L4_T16', 'nat_L8_T16', 'nat_L12_T16', 'nat_L16_T16']
    ser['vs_V_T16'] = {m: dict(V=[res['configs'][n]['V'] for n in names], sec=[su(n, m) for n in names])
                       for m in ('imh', 'rt', 'prt_dT4')}
    ser['vs_V_T16']['forward_path_batch1024'] = dict(
        V=[res['configs'][n]['V'] for n in names],
        sec=[res['configs'][n].get('main', {}).get('forward_path', {}).get('batch1024_s_per_path') for n in names])
    ser['note'] = ('L16 T16 in vs_T_L16 is the native L16 model (the paper T-scan uses it); ks_L8_T16 is the separately '
                   'trained K-scan model (same architecture as nat_L8_T16). prt_dT4 and prt_d4 coincide at T = 16; at '
                   'T = 4 prt_d4 is the full RT (rt value repeated).')
    res['series'] = ser
    tw = os.path.join(ROOT, 'train_log_wall.json')
    if os.path.isfile(tw):
        res['training_logs_note'] = dict(file='train_log_wall.json', caveat=(
            'devices differ (iqcb: shared L40S GPU; HIM: CPU, 4 or 16 threads); samp_per_sec is logged at epochs that '
            'are multiples of 200, which are even epochs and therefore always include the buffer RT step: the sum over '
            'segments overestimates the buffer phase; eval_log wall_sec is the wall time since the last (re)start'))
    json.dump(res, open(os.path.join(ROOT, 'results.json'), 'w'), indent=1)
    print('wrote results.json')


if __name__ == '__main__':
    main()
