#!/usr/bin/env python3
"""Full size matrix (2026-11-07): L in {4, 6, 8, 10, 12, 16, 24, 32} x kappa .20-.30 x {IMH, RT, pRT} x {paper_L8 (kappa-conditional,
trained at L = 8), per-kappa buffer model}. This script writes the stage-A task files:
  refs  (CPU): Wolff + HMC 1e6 on the TI grid for L = 6 / 10 (kappa 0-.30 step .005) and L = 24 / 32 (+ .001 on .251-.289);
               the eleven kappa .20-.30 points first.
  smoke (CPU): L = 24 / 32, paper_L8, rt and rtp:12 at kappa .24 / .27 / .30, 1000 transitions (acceptance, time per step).
  gpuA  (GPU): per-kappa models L = 6 (.20-.29 without .27) and L = 10 (.20-.30), then IMH (pregen) for L = 10 (both models) and L = 6 E2.
  cpuA  (CPU): rt / rtp:12 for L = 6 / 10, both models; L = 6 paper_L8 IMH on CPU.
L = 24 / 32 production chains and per-kappa training are decided from the smoke timings (stage B)."""
S = '/media/scratch/mqian'
PY = f'{S}/envs/sps/bin/python'
K = range(20, 31)
LCH = f'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NHMC_CODE={S}/revision_code/nhmc_code; cd {S}/revision_code/revision/longchain; O={S}/revision_runs/longchain_min'
P8 = f'--kind phi --run-dir {S}/phi4_bench/nhmc_runs/paper_L8 --ckpt best.pt'
METH = {'imhG': 'imh --imh-pregen --pregen-batch 8192', 'imh': 'imh', 'rt': 'rt', 'rtp12': 'rtp:12'}


def mdir(L, k):
    return f'L{L}_k0.27_buf_s0' if k == 27 else f'L{L}_k0{k}_buf_s0'


def chain(L, k, m, model, dev, nmax=None, nstart=10000, out=None):
    d = '--device cuda --mem-fraction 0.10' if dev == 'gpu' else '--device cpu --threads 1'
    nmax = nmax or (3200000 if m.startswith('imh') else 800000)
    pre = '' if dev == 'gpu' else 'export CUDA_VISIBLE_DEVICES=; '
    if model == 'E1':
        out = out or (f'p8_k0{k}_{m}' if L == 8 else f'E1_L{L}_k0{k}_{m}')
        return (f'{LCH}; {pre}[ -f $O/{out}.json ] || {PY} longchain.py {P8} --L {L} --kappa 0.{k} --method {METH[m]} --n-chains 4 '
                f'--n-start {nstart} --n-max {nmax} {d} --seed 20261109 --out $O/{out}.json > $O/{out}.log 2>&1')
    out = out or f'E2_L{L}_k0{k}_{m}'
    n = mdir(L, k)
    return (f'{LCH}; {pre}D={S}/revision_runs/phi4_buffer/{n}; [ -f $O/{out}.json ] || [ -f $O/buf_L{L}_k0{k}_{m}.json ] || '
            f'{{ for i in $(seq 4320); do [ -f $D/kernel_final.pt ] && break; sleep 60; done; [ -f $D/kernel_final.pt ] || exit 3; '
            f'{PY} longchain.py --kind phi --run-dir $D --ckpt kernel_final.pt --kappa 0.{k} --method {METH[m]} --n-chains 4 '
            f'--n-start {nstart} --n-max {nmax} {d} --seed 20261109 --out $O/{out}.json > $O/{out}.log 2>&1; }}')


def train(L, k):
    D = f'{S}/revision_runs/phi4_buffer/{mdir(L, k)}'
    return (f'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; cd {S}/revision_code/nhmc_code; D={D}; mkdir -p $D; [ -f $D/kernel_final.pt ] || '
            f'{PY} -u train_cond_phi4_kappa_explicit.py --kernel nhmc_resid2 --use-z2 --odd-nets --fixed-pflip 0.5 --L {L} --n-leapfrog 5 '
            f'--batch-size 48 --buffer --buffer-start 1000 --buffer-chains 64 --buffer-every 2 --buffer-size 20000 --buffer-weight 1.0 '
            f'--device cuda --kappa-min 0.{k} --kappa-max 0.{k} --num-steps 16 --epochs 20000 --seed 0 --save-dir $D >> $D/train.log 2>&1; '
            f'[ -f $D/kernel_final.pt ] || exit 3; [ -f $D/eval_N262144.json ] || {PY} -u eval_generic.py --run-dir $D --n 262144 '
            f'--batch 16384 --kappas 0.{k} --device cuda --out $D/eval_N262144.json > $D/eval.log 2>&1')


def ref(L, km, alg):
    t = f'L{L}_k0{km:03d}_{alg}'
    return (f'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=; cd {S}/revision_code/revision/baselines/spin_wolff_hmc/spin; '
            f'O={S}/revision_runs/ti_wh; mkdir -p $O; ls $O/*_{t}.json >/dev/null 2>&1 || {PY} run_spin.py --theory phi4 --L {L} '
            f'--params kappa={km / 1000:.3f} lam=0.022 --algo {alg} --n-samples 1000000 --burn 10000 --seed 20261107 --tag {t} '
            f'--out-dir $O > $O/{t}.log 2>&1')


def write(name, head, lines):
    open(name, 'w').write('\n'.join([f'# {head}'] + lines) + '\n')
    print(name, len(lines))


if __name__ == '__main__':
    refs = []
    for L in (6, 10, 24, 32):
        grid = list(range(0, 301, 5)) + ([k for k in range(251, 290) if k % 5] if L >= 24 else [])
        grid.sort(key=lambda km: (not (200 <= km <= 300 and km % 10 == 0), km))
        refs += [(L, km, alg) for km in grid for alg in ('wolff', 'hmc')]
    refs.sort(key=lambda r: (not (200 <= r[1] <= 300 and r[1] % 10 == 0), r[0], r[1]))
    write('tasks_matrix_refs_20261107.txt', 'Wolff / HMC 1e6 on the TI grid, L = 6 / 10 / 24 / 32 (main kappa points first)',
          [ref(*r) for r in refs])
    write('tasks_matrix_smoke_20261107.txt', 'L = 24 / 32 smoke: paper_L8 rt / rtp:12, 1000 transitions, acceptance and time per step',
          [chain(L, k, m, 'E1', 'cpu', nmax=1000, nstart=1000, out=f'smoke_L{L}_k0{k}_{m}') for L in (24, 32) for k in (24, 27, 30)
           for m in ('rt', 'rtp12')])
    g = [train(L, k) for L in (10, 6) for k in K if not (L == 6 and k in (27, 30))]
    g += [chain(10, k, 'imhG', 'E1', 'gpu') for k in K if k not in ()] + [chain(L, k, 'imhG', 'E2', 'gpu') for L in (10, 6) for k in K]
    write('tasks_matrix_gpuA_20261107.txt', 'per-kappa models L = 10 / 6, then IMH (GPU pregen) for L = 10 (both models) and L = 6 E2', g)
    c = [chain(6, k, 'imh', 'E1', 'cpu') for k in K]
    c += [chain(L, k, m, 'E1', 'cpu') for L in (6, 10) for k in K for m in ('rt', 'rtp12')]
    c += [chain(L, k, m, 'E2', 'cpu') for L in (6, 10) for k in K for m in ('rt', 'rtp12')]
    write('tasks_matrix_cpuA_20261107.txt', 'rt / rtp:12 for L = 6 / 10 (paper_L8 then per-kappa), L = 6 paper_L8 IMH on CPU', c)
