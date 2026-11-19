#!/usr/bin/env python3
"""One setup for the whole paper (user 2026-11-07): one kappa-conditional model per L in {4, 6, 8, 10, 12, 16}, same recipe as
paper_L8 (nhmc_learned, Z2 wrapper, kappa in [0.20, 0.30] explicit, 16 stages x 5 leapfrog, batch 48, 30000 epochs, embed 128,
channels 32 32, seed 0); L = 8 retrained with the current code (paper_L8 kept as a reproducibility check). Every chain uses the
model of its own L (best.pt). L = 24 / 32: no training, transfer test of the L = 16 model only.
Files: cond_train (GPU x2), cond_imhG (GPU), cond_cpu (CPU: rt / rtp:12 all, imh L4 / L6), cond_logz (CPU), cond_transfer (CPU)."""
S = '/media/scratch/mqian'
PY = f'{S}/envs/sps/bin/python'
K = range(20, 31)
LS = (4, 6, 8, 10, 12, 16)
MD = f'{S}/revision_runs/phi4_cond'
LCH = f'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NHMC_CODE={S}/revision_code/nhmc_code; cd {S}/revision_code/revision/longchain'
METH = {'imhG': 'imh --imh-pregen --pregen-batch 8192', 'imh': 'imh', 'rt': 'rt', 'rtp12': 'rtp:12'}
WAIT = 'for i in $(seq 2880); do [ -f $D/kernel_final.pt ] && break; sleep 60; done; [ -f $D/kernel_final.pt ] || exit 3'


def train(L):
    return (f'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; cd {S}/revision_code/nhmc_code; D={MD}/L{L}_cond_s0; mkdir -p $D; '
            f'[ -f $D/kernel_final.pt ] || {PY} -u train_cond_phi4_kappa_explicit.py --kernel nhmc_learned --use-z2 --L {L} '
            f'--kappa-min 0.20 --kappa-max 0.30 --num-steps 16 --n-leapfrog 5 --batch-size 48 --epochs 30000 --embed-dim 128 '
            f'--channels 32 32 --seed 0 --device cuda --save-dir $D >> $D/train.log 2>&1; [ -f $D/kernel_final.pt ] || exit 3; '
            f'[ -f $D/eval_N262144.json ] || {PY} -u eval_generic.py --run-dir $D --n 262144 --batch 16384 --ckpt best.pt '
            f'--kappas {" ".join(f"0.{k}" for k in K)} --device cuda --out $D/eval_N262144.json > $D/eval.log 2>&1')


def chain(L, k, m, dev, nmax=None, nstart=10000, Lrun=None, out=None, src=None):
    src = src or L
    d = '--device cuda --mem-fraction 0.10' if dev == 'gpu' else '--device cpu --threads 1'
    pre = '' if dev == 'gpu' else 'export CUDA_VISIBLE_DEVICES=; '
    nmax = nmax or (3200000 if m.startswith('imh') else 800000)
    out = out or f'C_L{L}_k0{k}_{m}'
    return (f'{LCH}; {pre}O={S}/revision_runs/longchain_min; D={MD}/L{src}_cond_s0; [ -f $O/{out}.json ] || {{ {WAIT}; '
            f'{PY} longchain.py --kind phi --run-dir $D --ckpt best.pt --L {L} --kappa 0.{k} --method {METH[m]} --n-chains 4 '
            f'--n-start {nstart} --n-max {nmax} {d} --seed 20261110 --out $O/{out}.json > $O/{out}.log 2>&1; }}')


def logz(L, k, m, i):
    out = f'C_L{L}_k0{k}_{m}'
    meth = {'imh': 'imh', 'rt': 'rt', 'rtp12': 'rtp:12'}[m]
    return (f'{LCH}; export CUDA_VISIBLE_DEVICES=; O={S}/revision_runs/logz_chains; mkdir -p $O; D={MD}/L{L}_cond_s0; '
            f'[ -f $O/{out}.json ] || {{ {WAIT}; {PY} logz_chains.py --run-dir $D --ckpt best.pt --L {L} --kappa 0.{k} --method {meth} '
            f'--n 50000 --n-fwd 65536 --seed {20263500 + i} --out $O/{out}.json > $O/{out}.log 2>&1; }}')


def write(name, head, lines):
    open(name, 'w').write('\n'.join([f'# {head}'] + lines) + '\n')
    print(name, len(lines))


if __name__ == '__main__':
    write('tasks_cond_train_A_20261107.txt', 'kappa-conditional models L = 4 / 6 / 8 / 10 (one GPU)', [train(L) for L in (10, 8, 6, 4)])
    write('tasks_cond_train_B_20261107.txt', 'kappa-conditional models L = 16 / 12 (one GPU)', [train(L) for L in (16, 12)])
    order = (4, 6, 8, 10, 12, 16)                                    # small L first (their models are ready first)
    write('tasks_cond_imhG_20261107.txt', 'IMH (GPU pregen) L = 8 / 10 / 12 / 16, all kappa',
          [chain(L, k, 'imhG', 'gpu') for L in (8, 10, 12, 16) for k in K])
    write('tasks_cond_cpu_20261107.txt', 'rt / rtp:12 all L and kappa, IMH on CPU for L = 4 / 6',
          [chain(L, k, m, 'cpu') for L in order for k in K for m in (('imh', 'rt', 'rtp12') if L in (4, 6) else ('rt', 'rtp12'))])
    write('tasks_cond_logz_20261107.txt', 'chain free energy (SNIS / X-rev / X + SNIS), all L, kappa, method; n = 5e4 per chain',
          [logz(L, k, m, i) for i, (L, k, m) in enumerate((L, k, m) for L in order for k in K for m in ('rt', 'rtp12', 'imh'))])
    write('tasks_cond_transfer_20261107.txt', 'L = 24 / 32 transfer test of the L = 16 model: rt / rtp:12 at kappa .24 / .27 / .30, 1000 transitions',
          [chain(L, k, m, 'cpu', nmax=1000, nstart=1000, out=f'Ctr16_L{L}_k0{k}_{m}', src=16) for L in (24, 32) for k in (24, 27, 30)
           for m in ('rtp12', 'rt')])
