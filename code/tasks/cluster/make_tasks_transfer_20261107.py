#!/usr/bin/env python3
"""Full transfer matrix (user 2026-11-07): single-kappa model of source size s in {4, 8, 12, 16} run at target size
t in {4, 8, 12, 16, 24, 32}, t != s, same kappa, kappa .20-.30; RT / pRT and raw / SNIS on CPU, IMH on GPU.
s = 16 -> t = 24 / 32 is in tasks_transfer16_*. Output names T{s}_L{t}_k0{k}_{method}. Caps: pRT / IMH 8e5,
RT 8e5 (t <= 16), 2e5 (t = 24), 1e5 (t = 32). Priority below the main L4-L16 jobs (nice 850 CPU, 1045 GPU)."""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sp = importlib.util.spec_from_file_location('sg', os.path.join(HERE, 'make_tasks_single_20261107.py'))
SG = importlib.util.module_from_spec(sp)
sp.loader.exec_module(SG)
S, PY, K, mdir, chain, fix, write, DONE = SG.S, SG.PY, SG.K, SG.mdir, SG.chain, SG.fix_dir, SG.write, SG.DONE
SRC, TGT = (4, 8, 12, 16), (4, 8, 12, 16, 24, 32)


def tline(s, t, k, m, dev, nmax):
    l = chain(s, k, m, 'E2', dev, nmax=nmax, out=f'T{s}_L{t}_k0{k}_{m}')
    return fix(l, s, k).replace(f'--kappa 0.{k} ', f'--L {t} --kappa 0.{k} ').replace(f' || [ -f $O/buf_L{s}_k0{k}_{m}.json ]', '')


def rawsnis(s, t, k, i):
    out = f'T{s}_L{t}_k0{k}'
    l = (f'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= NHMC_CODE={S}/revision_code/nhmc_code; '
         f'cd {S}/revision_code/revision/longchain; O={S}/revision_runs/raw_snis; mkdir -p $O; D={S}/revision_runs/phi4_buffer/{mdir(s, k)}; '
         f'[ -f $O/{out}.json ] || {{ for i in $(seq 2880); do [ -f $D/kernel_final.pt ] && break; sleep 60; done; [ -f $D/kernel_final.pt ] || exit 3; '
         f'{PY} raw_snis_obs.py --run-dir $D --ckpt kernel_final.pt --L {t} --kappa 0.{k} --n 65536 --batch 2048 '
         f'--seed {20261400 + i} --out $O/{out}.json > $O/{out}.log 2>&1; }}')
    return fix(l, s, k)


if __name__ == '__main__':
    pairs = [(s, t) for s in SRC for t in TGT if t != s and not (s == 16 and t in (24, 32))]
    cells = [(s, t, k) for s, t in pairs for k in K]
    cells.sort(key=lambda c: ((c[0], c[2]) not in DONE, c[1] in (24, 32), c[0], c[1], c[2]))   # ready models, small targets first
    rtcap = lambda t: 800000 if t <= 16 else (200000 if t == 24 else 100000)
    cpu = []
    for i, (s, t, k) in enumerate(cells):
        cpu += [tline(s, t, k, 'rtp12', 'cpu', 800000), tline(s, t, k, 'rt', 'cpu', rtcap(t)), rawsnis(s, t, k, i)]
    write('tasks_transfer_cpu_20261107.txt', 'transfer matrix s -> t (s in 4/8/12/16, t in 4/8/12/16/24/32), all kappa: pRT, RT, raw / SNIS', cpu)
    write('tasks_transfer_imhG_20261107.txt', 'transfer matrix s -> t, IMH (GPU pregen, cap 8e5)',
          [tline(s, t, k, 'imhG', 'gpu', 800000) for s, t, k in cells])
    print(len(pairs), 'pairs')
