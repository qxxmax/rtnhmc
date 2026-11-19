#!/usr/bin/env python3
"""Dynamic-scaling extras in the protocol of Albergo, Kanwar, Shanahan 2019 (arXiv:1904.12072), user 2026-11-08.
(a) denser kappa near criticality for the line-of-constant-U4 analysis: new single-kappa models (same recipe) at
    kappa .265 / .275 for L = 4 / 8 / 12 / 16, then IMH (GPU), RT / pRT, raw / SNIS and chain free energy (CPU);
    Wolff / HMC references exist on the TI grid (step .005, .001 on .25-.29 for L12 / L16).
(b) pRT truncation scan at kappa .27 (fixed-acceptance comparison): rtp:s, s in {2, 4, 6, 8, 10, 14} (s = 12 and RT = s 0
    exist), L = 4 / 8 / 12 / 16, same stop rule, cap 2e5; g at fixed acceptance is read off by interpolation in s.
Priority: after the main L4-L16 set, before L6 / L10 and the transfer matrix (GPU nice 1020, CPU nice 820)."""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sp = importlib.util.spec_from_file_location('sg', os.path.join(HERE, 'make_tasks_single_20261107.py'))
SG = importlib.util.module_from_spec(sp)
sp.loader.exec_module(SG)
S, PY, mdir, train, chain, fix, write, logz = SG.S, SG.PY, SG.mdir, SG.train, SG.chain, SG.fix_dir, SG.write, SG.logz
LS, KN = (4, 12, 16, 8), (265, 275)


def rawsnis(L, k, i):
    out = f'E2_L{L}_k0{k}'
    return (f'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= NHMC_CODE={S}/revision_code/nhmc_code; '
            f'cd {S}/revision_code/revision/longchain; O={S}/revision_runs/raw_snis; mkdir -p $O; D={S}/revision_runs/phi4_buffer/{mdir(L, k)}; '
            f'[ -f $O/{out}.json ] || {{ for i in $(seq 2880); do [ -f $D/kernel_final.pt ] && break; sleep 60; done; [ -f $D/kernel_final.pt ] || exit 3; '
            f'{PY} raw_snis_obs.py --run-dir $D --ckpt kernel_final.pt --L {L} --kappa 0.{k} --n 65536 --batch 4096 '
            f'--seed {20261500 + i} --out $O/{out}.json > $O/{out}.log 2>&1; }}')


def rtp(L, k, s):
    out = f'E2_L{L}_k0{k}_rtp{s}'
    D = f'{S}/revision_runs/phi4_buffer/{mdir(L, k)}'
    l = (f'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= NHMC_CODE={S}/revision_code/nhmc_code; '
         f'cd {S}/revision_code/revision/longchain; O={S}/revision_runs/longchain_min; D={D}; '
         f'[ -f $O/{out}.json ] || {{ for i in $(seq 2880); do [ -f $D/kernel_final.pt ] && break; sleep 60; done; [ -f $D/kernel_final.pt ] || exit 3; '
         f'{PY} longchain.py --kind phi --run-dir $D --ckpt kernel_final.pt --kappa 0.{k} --method rtp:{s} --n-chains 4 '
         f'--n-start 10000 --n-max 200000 --device cpu --threads 1 --seed 20261109 --out $O/{out}.json > $O/{out}.log 2>&1; }}')
    return fix(l, L, k)


if __name__ == '__main__':
    cells = [(L, k) for L in LS for k in KN]
    g = [train(L, k) for L, k in cells] + [fix(chain(L, k, 'imhG', 'E2', 'gpu'), L, k) for L, k in cells if L != 4]
    write('tasks_zscan_gpu_20261108.txt', 'new single-kappa models kappa .265 / .275 at L = 4 / 8 / 12 / 16, then IMH (L >= 8)', g)
    c = [rtp(L, 27, s) for L in LS for s in (14, 10, 8, 6, 4, 2)]
    for i, (L, k) in enumerate(cells):
        c += [fix(chain(L, k, m, 'E2', 'cpu'), L, k) for m in (('imh', 'rt', 'rtp12') if L == 4 else ('rt', 'rtp12'))]
        c += [rawsnis(L, k, i)] + [fix(logz(L, k, m, 500 + 3 * i + j), L, k) for j, m in enumerate(('rt', 'rtp12', 'imh'))]
    write('tasks_zscan_cpu_20261108.txt', 'pRT s-scan at kappa .27 (L = 4 / 12 / 16 / 8), then chains / raw-SNIS / logZ for kappa .265 / .275', c)
