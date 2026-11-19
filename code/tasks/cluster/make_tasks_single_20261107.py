#!/usr/bin/env python3
"""Main setup of the paper (user 2026-11-07): one single-kappa buffer model per (L, kappa), L in {4, 6, 8, 10, 12, 16},
kappa .20-.30; same recipe everywhere (nhmc_resid2, Z2 wrapper, odd nets, p_flip 0.5, RT replay buffer, 20000 epochs, seed 0;
models at revision_runs/phi4_buffer, kappa .27 dirs named L*_k0.27_buf_s0). IMH / RT / pRT chains (longchain.py, kernel_final.pt)
and chain free energies (logz_chains.py) on every cell. L = 24 / 32: no training; transfer test of the L = 16 models.
Trainings resume from last.pt, chain lines skip existing outputs and wait for kernel_final.pt."""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sp = importlib.util.spec_from_file_location('mx', os.path.join(HERE, 'make_tasks_matrix_20261107.py'))
MX = importlib.util.module_from_spec(sp)
sp.loader.exec_module(MX)
S, PY, K, mdir, train, chain, write = MX.S, MX.PY, MX.K, MX.mdir, MX.train, MX.chain, MX.write
MODELS_LIB = f'{S}/revision_code/results_sync/models/phi4_buffer'
# trained already (cluster phi4_buffer or the lib copy); everything else is trained / resumed here
DONE = ({(4, k) for k in K} | {(6, 27), (6, 30)} | {(12, k) for k in (21, 24, 27, 30)} | {(16, k) for k in (20, 21, 24, 27, 29, 30)})
LS = (4, 6, 8, 10, 12, 16)


def fix_dir(line, L, k):
    """chains: model dir from the cluster training dir, else the lib copy (kappa .27 / .30 models uploaded there)."""
    a = f'D={S}/revision_runs/phi4_buffer/{mdir(L, k)};'
    b = a + f' [ -f $D/kernel_final.pt ] || [ ! -f {MODELS_LIB}/{mdir(L, k)}/kernel_final.pt ] || D={MODELS_LIB}/{mdir(L, k)};'
    return line.replace(a, b)


def logz(L, k, m, i):
    out = f'E2_L{L}_k0{k}_{m}'
    meth = {'imh': 'imh', 'rt': 'rt', 'rtp12': 'rtp:12'}[m]
    return (f'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= NHMC_CODE={S}/revision_code/nhmc_code; '
            f'cd {S}/revision_code/revision/longchain; O={S}/revision_runs/logz_chains; mkdir -p $O; D={S}/revision_runs/phi4_buffer/{mdir(L, k)}; '
            f'[ -f $O/{out}.json ] || {{ for i in $(seq 2880); do [ -f $D/kernel_final.pt ] && break; sleep 60; done; '
            f'[ -f $D/kernel_final.pt ] || exit 3; {PY} logz_chains.py --run-dir $D --ckpt kernel_final.pt --L {L} --kappa 0.{k} '
            f'--method {meth} --n 50000 --n-fwd 65536 --seed {20263700 + i} --out $O/{out}.json > $O/{out}.log 2>&1; }}')


if __name__ == '__main__':
    todo = [(L, k) for L in LS for k in K if (L, k) not in DONE]
    t16 = [(16, k) for k in (22, 23, 25, 26, 28)] + [(12, 20)]
    t12 = [(12, k) for k in (22, 23, 25, 26, 28, 29)]
    small = [c for c in todo if c not in t16 + t12]
    write('tasks_single_train_A_20261107.txt', 'resume L16 .22/.23/.25/.26/.28 + L12 .20 (one GPU)', [train(*c) for c in t16])
    write('tasks_single_train_B_20261107.txt', 'resume L12 .22/.23/.25/.26/.28/.29 (one GPU)', [train(*c) for c in t12])
    write('tasks_single_train_C_20261107.txt', 'new L10 / L8 / L6 single-kappa models (one GPU)', [train(*c) for c in sorted(small, key=lambda c: (-c[0], c[1]))])
    ready = lambda c: c in DONE
    gi = sorted([(L, k) for L in (8, 10, 12, 16) for k in K], key=lambda c: (not ready(c), c))
    write('tasks_single_imhG_20261107.txt', 'IMH (GPU pregen) L = 8 / 10 / 12 / 16, all kappa (ready models first)',
          [fix_dir(chain(L, k, 'imhG', 'E2', 'gpu'), L, k) for L, k in gi])
    cc = [(L, k, m) for L in LS for k in K for m in (('imh', 'rt', 'rtp12') if L in (4, 6) else ('rt', 'rtp12'))]
    cc.sort(key=lambda c: (not ready(c[:2]), c[0], c[1]))
    write('tasks_single_cpu_20261107.txt', 'RT / pRT all cells, IMH on CPU for L = 4 / 6 (ready models first)',
          [fix_dir(chain(L, k, m, 'E2', 'cpu'), L, k) for L, k, m in cc])
    lz = sorted([(L, k, m) for L in LS for k in K for m in ('rt', 'rtp12', 'imh')], key=lambda c: (not ready(c[:2]), c[0], c[1]))
    write('tasks_single_logz_20261107.txt', 'chain free energy (SNIS / X-rev / X + SNIS) all cells, n = 5e4 per chain',
          [fix_dir(logz(L, k, m, i), L, k) for i, (L, k, m) in enumerate(lz)])
    tr = []
    for L in (24, 32):
        for k in (24, 27, 30):
            for m in ('rtp12', 'rt'):
                l = chain(16, k, m, 'E2', 'cpu', nmax=1000, nstart=1000, out=f'E2tr16_L{L}_k0{k}_{m}')
                tr.append(fix_dir(l, 16, k).replace(f'--kappa 0.{k} ', f'--L {L} --kappa 0.{k} ').replace(f' || [ -f $O/buf_L16_k0{k}_{m}.json ]', ''))
    write('tasks_single_transfer_20261107.txt', 'L = 24 / 32 transfer test of the L = 16 single-kappa models (kappa .24 / .27 / .30), 1000 transitions', tr)
