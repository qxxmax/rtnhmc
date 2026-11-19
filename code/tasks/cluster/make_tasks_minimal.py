#!/usr/bin/env python3
"""Writes the minimal-matrix single-chain task files (2026-11-05):
  tasks_minimal_KL.txt  kappa axis rest (paper_L8 kappa .28 .26 .24 .22 .21) + L axis (E1: paper_L8 at L = 4, 6, 10, 12, 16,
                        kappa .27 / .30), 45 one-core tasks;
  tasks_minimal_DL.txt  DW (7 models) and LJ13, x {imh, rt, partial rt}, 24 one-core tasks;
  tasks_minimal_GPU.txt one GPU: CMCD/DBS training + all IMH long chains (pre-generated proposals), 43 tasks;
  tasks_minimal_S.txt   samplers: CMCD+Z2 and DBS+Z2 at the other kappa (.20-.26, .29: train, then 3 chains; 16 four-core
                        tasks) and at kappa .28 (models in results_sync/models/cmcd_torch; 6 one-core tasks).
With tasks_minimal_AB.txt and the local runs this gives NHMC (paper_L8), CMCD and DBS at all 11 kappa x {imh, rt, partial rt}.
All chains: revision/longchain/longchain.py, 4 independent chains per task, proposal starts, adaptive length 1e4 -> 2e5,
checkpoint every 20 min (resubmission resumes). HMC+Z2 json files are reference numbers only."""
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
S = '/media/scratch/mqian'
RC = f'{S}/revision_code'
RB = f'{RC}/results_sync/revision'
PYB = f'{S}/envs/sps/bin/python'
LC = f'{PYB} longchain.py'
O = f'{S}/revision_runs/longchain_min'
H1 = (f'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= NHMC_CODE={RC}/nhmc_code; '
      f'cd {RC}/revision/longchain; O={O}; mkdir -p $O; ')
P8 = f'--kind phi --run-dir {S}/phi4_bench/nhmc_runs/paper_L8 --ckpt best.pt'
TAIL = '--n-chains 4 --n-start 10000 --n-max 200000 --device cpu --threads 1'
NHMC_M = (('imh', 'imh'), ('rt', 'rt'), ('rtp:12', 'rtp12'))
UDB_M = (('imh', 'imh'), ('rt', 'rt'), ('rtp:40', 'rtp40'))


def chain(name, args):
    return f'[ -f $O/{name}.json ] || {{ {LC} {args} {TAIL} --out $O/{name}.json > $O/{name}.log 2>&1; }}'


def kl():
    L = []
    for k in ('28', '26', '24', '22', '21'):
        for m, t in NHMC_M:
            L.append(H1 + chain(f'p8_k0{k}_{t}', f'{P8} --kappa 0.{k} --method {m} --ref {RB}/phi4/hmc_k0{k}.json'))
    refs = {4: 'phi4_E1/hmc_L4_k0{k}.json', 6: 'phi4_E1/hmc_L6_k0{k}.json', 10: 'phi4_E1/hmc_L10_k0{k}.json',
            12: 'phi4/refL/hmc_L12_k0{k}.json', 16: 'phi4/refL/hmc_L16_k0{k}.json'}
    for k in ('27', '30'):
        for Lx in (4, 6, 10, 12, 16):
            for m, t in NHMC_M:
                L.append(H1 + chain(f'E1_L{Lx}_k0{k}_{t}',
                                    f'{P8} --L {Lx} --kappa 0.{k} --method {m} --ref {RB}/{refs[Lx].format(k=k)}'))
    hdr = """# Minimal matrix: kappa axis (rest) + L axis, long single chains, CPU, 1 core per task, 45 tasks (make_tasks_minimal.py).
# K (15): paper_L8 kappa .28 .26 .24 .22 .21 x {imh, rt, rtp:12}; with tasks_minimal_AB.txt and the local .27 / .30 runs: all 11 kappa.
# L (30): E1, paper_L8 applied at L = 4, 6, 10, 12, 16 (L = 8 = the kappa axis), kappa .27 / .30 x {imh, rt, rtp:12}.
#   --L test (local): L = 4 kappa .27 RT, 4 chains x 4000: acceptance .57-.59, |M| z -1.4 .. +0.6 vs hmc_L4_k027.
# Time: kappa tasks 0.1-2 h; L <= 10 < 2 h; L = 12 / 16 RT ~1-4 h; IMH at L = 16 kappa .27 may run to the 2e5 cap (~20 h).
# Order: kappa first, then L small -> large. Outputs $O = /media/scratch/mqian/revision_runs/longchain_min/.
"""
    (HERE / 'tasks_minimal_KL.txt').write_text(hdr + '\n'.join(L) + '\n', newline='\n')
    return len(L)


def samplers():
    L = []
    for k in ('29', '26', '25', '24', '23', '22', '21', '20'):
        for alg, tag in (('cmcd', 'cmcd'), ('dbs', 'dbs')):
            run = f'{alg}z2_k0.{k}_s0_T80_dt0.1_it10000'
            D = f'{S}/revision_runs/pack2/cmcd_family/{run}'          # same directory as pack 2 A (training shared)
            train = (f'cd {RC}/revision/cmcd_torch; D={D}; mkdir -p $D; [ -f $D/final_eval.json ] || {{ RES=; '
                     f'[ -f $D/ckpt.pt ] && RES=--resume; {PYB} -u cmcd_phi4.py --alg {alg} --kappa 0.{k} --T 80 --dt 0.1 '
                     f'--iters 10000 --batch 500 --lr 5e-4 --seed 0 --n-final 102400 --chunk 4096 --device cpu --threads 4 '
                     f'--ckpt-min 10 $RES --out $D >> $D/train.log 2>&1; }}; ')
            chains = ' & '.join(
                chain(f'{tag}_k0{k}_{t}', f'--kind cmcd --model $D/model.pt --kappa 0.{k} --method {m} '
                                          f'--ref {RB}/phi4/hmc_k0{k}.json') for m, t in UDB_M)
            L.append(f'export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= NHMC_CODE={RC}/nhmc_code; '
                     + train + f'[ -f $D/final_eval.json ] || exit 3; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; '
                     f'cd {RC}/revision/longchain; O={O}; mkdir -p $O; ' + chains + ' & wait')
    for alg, tag in (('cmcd', 'cmcd'), ('dbs', 'dbs')):
        model = f'{RC}/results_sync/models/cmcd_torch/{alg}z2_k0.28_s0_T80_dt0.1_it10000/model.pt'
        for m, t in UDB_M:
            L.append(H1 + chain(f'{tag}_k028_{t}', f'--kind cmcd --model {model} --kappa 0.28 --method {m} '
                                                   f'--ref {RB}/phi4/hmc_k028.json'))
    hdr = """# Minimal matrix: samplers (CMCD+Z2, DBS+Z2 torch port, L = 8), long single chains, 22 tasks (make_tasks_minimal.py).
# 16 four-core tasks: kappa .29 .26 .25 .24 .23 .22 .21 .20 x {cmcd, dbs}: train (official recipe, cmcd_phi4.py, 10k iters,
#   ~1-1.5 h at 4 threads, resumes from ckpt.pt; same directory as pack 2 A, so a model trained there is reused), final
#   SNIS eval N = 102400, then 3 chains {imh, rt, rtp:40} in parallel (1 core each). rc 3 = training not finished (resubmit).
# 6 one-core tasks: kappa .28 x {cmcd, dbs} x {imh, rt, rtp:40} (models results_sync/models/cmcd_torch, SHA256SUMS).
# kappa .27 / .30 exist (local long chains; dbs27 in tasks_minimal_AB.txt).
# Outputs: chains /media/scratch/mqian/revision_runs/longchain_min/{cmcd,dbs}_k0XX_{imh,rt,rtp40}.json; models pack2/cmcd_family/.
"""
    (HERE / 'tasks_minimal_S.txt').write_text(hdr + '\n'.join(L) + '\n', newline='\n')
    return len(L)


def dw_lj():
    L = []
    HD = (f'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= NHMC_CODE={RC}/nhmc_code; '
          f'cd {RC}/revision/longchain; O={O}; mkdir -p $O; ')
    dw = [('dw4_A', f'{S}/protein/NHMC_DW_PHI4_RUNNABLE_20260921/dw_v2/results/dw4_anneal_seed0', 4),
          ('dw8_A', f'{S}/protein/NHMC_DW_PHI4_RUNNABLE_20260921/dw_v2/results/dw8_long_score_seed0', 5),
          ('dw4_loss_lv', f'{S}/protein/paper_repeat/dw_v2/results/dw4_loss_lv_seed0', 4),
          ('dw4_loss_lvess', f'{S}/protein/paper_repeat/dw_v2/results/dw4_loss_lvess_seed0', 4),
          ('dw4_untrained', f'{S}/protein/paper_repeat/dw_v2/results/dw4_untrained_seed0', 4),
          ('dw4_B_learned_s1', f'{S}/revision_runs/dw/B_dw4_learned_s1', 4),
          ('dw4_B_fixed_s0', f'{S}/revision_runs/dw/B_dw4_fixed_s0', 4)]
    for tag, rd, K in dw:
        for m, t in (('imh', 'imh'), ('rt', 'rt'), (f'rtp:{K}', f'rtp{K}')):
            L.append(HD + chain(f'{tag}_{t}', f'--kind dw --run-dir {rd} --ckpt kernel_best.pt --method {m}'))
    HL = (f'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= LJ_CODE={S}/lj_buffer/code_rtlocal; '
          f'cd {RC}/revision/longchain; O={O}; mkdir -p $O; ')
    for m, t in (('imh', 'imh'), ('rt', 'rt'), ('rtp:8', 'rtp8')):
        name = f'lj13_{t}'
        L.append(HL + f'[ -f $O/{name}.json ] || {{ {LC} --kind lj --run-dir {S}/lj_buffer/runs/lj13sc_a05T56_s0 '
                 f'--ckpt kernel_best.pt --method {m} --ref {S}/lj_buffer/refs --n-chains 4 --n-start 5000 '
                 f'--n-max 80000 --device cpu --threads 1 --out $O/{name}.json > $O/{name}.log 2>&1; }}')
    hdr = """# Minimal matrix: non-lattice targets, long single chains, CPU, 1 core per task, 24 tasks (make_tasks_minimal.py).
# DW (ManyWell, analytic exact moments and mode probabilities; --kind dw = dw_compare transitions, identity test
#   revision/longchain/test_kinds_dw_lj.py PASS; flip probability 0.05 = dw_compare default):
#   dw4_A, dw8_A (comparison A models), dw4 loss_lv / loss_lvess / untrained (sweep), dw4 B learned s1 / fixed s0
#   x {imh, rt, rtp:K} (K = 4 for DW4, 5 for DW8 as in comparison A). Per chain also mode_l1 and mode_change_rate.
#   Expected minutes per task (tiny model), cap 2e5.
# LJ13 (--kind lj = lj13_compare transitions with LJ_CODE=lj_buffer/code_rtlocal; reference = test split means,
#   reference states never used as starts): {imh, rt, rtp:8}, 4 chains, start 5e3, cap 8e4 (~0.5 core-s per path:
#   5e3 transitions ~3-6 h, cap ~45-90 h; checkpoint every 20 min, resubmission resumes).
# Outputs /media/scratch/mqian/revision_runs/longchain_min/{tag}_{method}.json.
"""
    (HERE / 'tasks_minimal_DL.txt').write_text(hdr + '\n'.join(L) + '\n', newline='\n')
    return len(L)


def gpu():
    """one L40S: CMCD/DBS training (16) and every IMH long chain from pre-generated proposals (43); cap 8e5."""
    L = []
    HG = (f'export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NHMC_CODE={RC}/nhmc_code; '
          f'cd {RC}/revision/longchain; O={O}; mkdir -p $O; ')
    G = '--imh-pregen --pregen-batch 8192 --n-chains 4 --n-start 10000 --n-max 800000 --device cuda --mem-fraction 0.10'

    def g(name, args):
        return f'[ -f $O/{name}.json ] || {{ {LC} {args} {G} --out $O/{name}.json > $O/{name}.log 2>&1; }}'
    for k in ('29', '26', '25', '24', '23', '22', '21', '20'):
        for alg in ('cmcd', 'dbs'):
            run = f'{alg}z2_k0.{k}_s0_T80_dt0.1_it10000'
            D = f'{S}/revision_runs/pack2/cmcd_family/{run}'
            L.append(f'export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NHMC_CODE={RC}/nhmc_code; cd {RC}/revision/cmcd_torch; '
                     f'D={D}; mkdir -p $D; [ -f $D/final_eval.json ] || {{ RES=; [ -f $D/ckpt.pt ] && RES=--resume; '
                     f'{PYB} -u cmcd_phi4.py --alg {alg} --kappa 0.{k} --T 80 --dt 0.1 --iters 10000 --batch 500 '
                     f'--lr 5e-4 --seed 0 --n-final 102400 --chunk 4096 --device cuda --ckpt-min 10 $RES --out $D '
                     f'>> $D/train.log 2>&1; }}; [ -f $D/final_eval.json ] || exit 3; cd {RC}/revision/longchain; '
                     f'O={O}; mkdir -p $O; ' + g(f'{alg}_k0{k}_imhG', f'--kind cmcd --model $D/model.pt --kappa 0.{k} '
                                                                     f'--method imh --ref {RB}/phi4/hmc_k0{k}.json'))
    for alg in ('cmcd', 'dbs'):
        for k in ('27', '28', '30'):
            model = f'{RC}/results_sync/models/cmcd_torch/{alg}z2_k0.{k}_s0_T80_dt0.1_it10000/model.pt'
            L.append(HG + g(f'{alg}_k0{k}_imhG', f'--kind cmcd --model {model} --kappa 0.{k} --method imh '
                                                  f'--ref {RB}/phi4/hmc_k0{k}.json'))
    for k in ('20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30'):
        L.append(HG + g(f'p8_k0{k}_imhG', f'{P8} --kappa 0.{k} --method imh --ref {RB}/phi4/hmc_k0{k}.json'))
    refs = {4: 'phi4_E1/hmc_L4_k0{k}.json', 6: 'phi4_E1/hmc_L6_k0{k}.json', 10: 'phi4_E1/hmc_L10_k0{k}.json',
            12: 'phi4/refL/hmc_L12_k0{k}.json', 16: 'phi4/refL/hmc_L16_k0{k}.json'}
    for k in ('27', '30'):
        for Lx in (4, 6, 10, 12, 16):
            L.append(HG + g(f'E1_L{Lx}_k0{k}_imhG', f'{P8} --L {Lx} --kappa 0.{k} --method imh '
                                                     f'--ref {RB}/{refs[Lx].format(k=k)}'))
    hdr = """# Minimal matrix, GPU part (one L40S is enough), 43 tasks (make_tasks_minimal.py). Run 4-6 at a time on one card.
# 1) 16 tasks: CMCD+Z2 / DBS+Z2 training for kappa .29 .26 .25 .24 .23 .22 .21 .20 on cuda (same directories as
#    tasks_minimal_S.txt / pack 2 A: whichever finishes first, the other skips the training), then the IMH long chain.
# 2) 27 tasks: IMH long chains from pre-generated proposals (longchain.py --imh-pregen; proposals do not depend on the
#    state, so they are generated in batches of 8192 on the GPU; local timing 17 s - 3 min per cell on a shared laptop
#    GPU): CMCD/DBS kappa .27 .28 .30, paper_L8 all 11 kappa, E1 L = 4, 6, 10, 12, 16 x kappa .27 / .30.
#    Cap 8e5 transitions per chain (4x the CPU cap): for kappa >= .28 the IMH tau grows with the run length (heavy weight
#    tail, see spectral/README.md), so the longer cap gives a better lower bound. Outputs *_imhG.json.
# If this file runs, the IMH lines of tasks_minimal_KL.txt / _S.txt may be skipped (same chain law, shorter cap).
# RT / partial RT single chains stay on the CPU (4 chains per step: latency bound, no GPU gain), as do DW and LJ13.
"""
    (HERE / 'tasks_minimal_GPU.txt').write_text(hdr + '\n'.join(L) + '\n', newline='\n')
    return len(L)


if __name__ == '__main__':
    print('KL', kl(), 'S', samplers(), 'DL', dw_lj(), 'GPU', gpu())
