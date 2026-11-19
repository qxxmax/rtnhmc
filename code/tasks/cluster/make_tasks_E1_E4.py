#!/usr/bin/env python3
"""Writes the E1 (phi4 size-transfer model-error sweep) task files and the no-HMC re-run pack (phi4 L = 8,
paper_L8, comparison A over kappa .20-.30, long RT check, one-sector relaxation).
One shell command per line, for job_cpu_pack.sh (TASKS=<file> P=<parallel>). Every line skips work whose
output json already exists, so a pack can be resubmitted after a time-out.

Rule (2026-11-04): NHMC chains (path-IMH, RT, partial RT) use no HMC at all. Every chain starts from the sampler
itself (--init proposal: forward-path endpoints with their own forward path as record; proposal_plus for the
one-sector relaxation). HMC + Z2 (phi4_hmc_z2.py) appears only as an external reference number for z-scores.
New chain outputs go to new directories, so no old HMC-started file can satisfy a skip guard.
E4 (LJ13 PT reference) is dropped: the LJ13 PT reference exists (lj_buffer/runs/lj13_reference)."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PY = '/media/scratch/mqian/envs/sps/bin/python'
CODE = '/media/scratch/mqian/revision_code/revision'
RR = '/media/scratch/mqian/revision_runs'
REF8 = f'{RR}/phi4'                  # L = 8 HMC+Z2 references hmc_k0xx.json (existing .27-.30, new .20-.26)
REFL = f'{RR}/phi4/refL'             # L = 12 / 16 HMC+Z2 references of cluster job 26830
R = f'{RR}/phi4_E1'                  # E1 HMC+Z2 references (L = 4, 6, 10 written here; L = 12, 16 copied here)
Q = f'{RR}/phi4_E1_noHMC'            # every proposal-start comparison output (E1 sizes and L = 8 comparison A)
N = f'{RR}/phi4_noHMC'               # L = 8 long RT check and one-sector relaxation (kept out of the E1 pool)
RUN = '/media/scratch/mqian/phi4_bench/nhmc_runs/paper_L8'
ENV = (f'export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NHMC_CODE=/media/scratch/mqian/phi4_bench/nhmc_code; '
       f'cd {CODE}/phi4; R={R}; Q={Q}; N={N}; mkdir -p $R $Q $N')
SEED = 20761029                      # E1 compare: run r uses seed + 1000 r (+ 31 len(method) inside phi4_compare);
                                     # new base, so no run repeats a seed of the discarded HMC-bank E1 runs
SEED_REF = 20261029                  # HMC+Z2 references (unchanged)
SEED8 = 20461029                     # no-HMC L = 8 re-run: new seed base
INIT = '--init proposal --burn-fraction 0.2'
KAPPAS = ('0.27', '0.30')
KAPPAS8 = ('0.20', '0.21', '0.22', '0.23', '0.24', '0.25', '0.26', '0.27', '0.28', '0.29', '0.30')
OLD_REF8 = ('0.27', '0.28', '0.29', '0.30')       # hmc_k0xx.json that already exist on the cluster
# L: (n_chains, path budget per run for the chain methods). Budgets give T_IMH = budget / n_chains transitions
# (RT T/2, partial RT s=12 2T); fewer chains at large L buy longer chains (CPU cost per path is ~flat in n_chains).
BUDGET = {4: (256, 5_120_000), 6: (256, 5_120_000), 10: (128, 3_200_000), 12: (64, 1_920_000), 16: (64, 2_560_000)}
# L = 8 comparison A: 256 chains; kappa >= .26 5.12M paths (T_IMH 20000), kappa <= .25 2.56M (T_IMH 10000).
B8_HARD, B8_EASY = 5_120_000, 2_560_000
LONG_RT = 4 * B8_HARD                # kappa .30 long RT check: 4x longer chains (RT 40000 transitions)
RELAX = 64_000                       # one-sector relaxation: 64k paths per run, 256 chains
SNIS_PATHS = 500_000                 # E1 SNIS (weight ESS axis only)
METHODS = ('imh', 'rt', 'rt_partial:12')
WAIT = 'for i in $(seq 480); do [ -f {f} ] && break; sleep 60; done; sleep 10; '   # wait <= 8 h for a reference


def kt(k):
    return 'k' + k.replace('0.', '0')


def line(cmd, out, pre=''):
    """pre (wait for / copy a reference) runs only if the output does not exist yet."""
    return f'{ENV}; [ -f {out}.json ] || {{ {pre}{PY} {cmd} --out {out}.json > {out}.log 2>&1; }}'


def hmc_ref(L, k, out):
    return line(f'phi4_hmc_z2.py --L {L} --kappa {k} --lam 0.022 --force-budget 192000000 --n-runs 3 '
                f'--seed {SEED_REF} --device cpu --threads 4', out)


def refs():
    """E1 references for L = 4, 6, 10. L = 12 / 16 come from job 26830 (refL), L = 8 from REF8."""
    return [hmc_ref(L, k, f'$R/hmc_L{L}_{kt(k)}') for L in (10, 6, 4) for k in KAPPAS]


def compare(Ls, methods=METHODS + ('snis',), seed=SEED):
    out = []
    for L in Ls:                     # longest tasks first (xargs runs lines in order)
        nc, B = BUDGET[L]
        chunk = max(1024, 262144 // (L * L))
        for k in KAPPAS:
            ref = f'$R/hmc_L{L}_{kt(k)}.json'
            pre = (WAIT.format(f=f'{REFL}/hmc_L{L}_{kt(k)}.json') + f'cp -n {REFL}/hmc_L{L}_{kt(k)}.json {ref}; '
                   if L in (12, 16) else '')
            for m in methods:
                tag = m.replace(':', '')
                for r in range(3):
                    o = f'$Q/E1_L{L}_{kt(k)}_{tag}_r{r}'
                    b = SNIS_PATHS if m == 'snis' else B
                    init = '' if m == 'snis' else INIT + ' '
                    out.append(line(f'phi4_compare.py --run-dir {RUN} --device cpu --threads 4 --L {L} --kappa {k} '
                                    f'--methods {m} {init}--n-chains {nc} --path-budget {b} --snis-chunk {chunk} '
                                    f'--n-runs 1 --seed {seed + 1000 * r} --ref {ref}', o, pre))
    return out


def l8_task(k, m, r, budget, out, init=INIT, seed=SEED8):
    ref = f'{REF8}/hmc_{kt(k)}.json'
    pre = '' if k in OLD_REF8 else WAIT.format(f=ref)
    return line(f'phi4_compare.py --run-dir {RUN} --device cpu --threads 4 --L 8 --kappa {k} --methods {m} '
                f'{init} --n-chains 256 --path-budget {budget} --n-runs 1 --seed {seed + 1000 * r} --ref {ref}',
                out, pre)


def rerun_noHMC():
    out = []
    # 1. HMC+Z2 reference numbers for the kappas without one (reference only; minutes each)
    for k in KAPPAS8:
        if k not in OLD_REF8:
            out.append(hmc_ref(8, k, f'{REF8}/hmc_{kt(k)}'))
    # 2. kappa .30 long RT check (longest tasks, start first); seeds disjoint from comparison A
    for r in range(4):
        out.append(l8_task('0.30', 'rt', r, LONG_RT, f'$N/rtlong_L8_k030_rt_r{r}', seed=SEED8 + 100_000))
    # 3. comparison A, hard (high) kappa first
    for k in reversed(KAPPAS8):
        B = B8_HARD if float(k) >= 0.26 else B8_EASY
        for r in range(4):
            for m in METHODS:
                out.append(l8_task(k, m, r, B, f'$Q/E1_L8_{kt(k)}_{m.replace(":", "")}_r{r}'))
    # 4. one-sector relaxation (proposal_plus), short
    for k in ('0.30', '0.27'):
        for m in METHODS:
            for r in range(4):
                out.append(l8_task(k, m, r, RELAX, f'$N/relax_L8_{kt(k)}_{m.replace(":", "")}_r{r}',
                                   init='--init proposal_plus --burn-fraction 0.2', seed=SEED8 + 200_000))
    return out


# ----------------------------------------------------------------------------- DW and LJ13 (no exact / reference starts)
DWC = f'{CODE}/dw'
DWR = '/media/scratch/mqian/protein/NHMC_DW_PHI4_RUNNABLE_20260921/dw_v2/results'
DWO = f'{RR}/dw_noHMC'               # comparison A, comparison B (RT), worst-mode relaxation
DWS = f'{RR}/dw_sweep_noHMC'         # 81-model sweep
DW_ENV = f'export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; cd {DWC}; mkdir -p {DWO} {DWS}'
DW_INIT = '--init proposal --burn_fraction 0.2'
# Gate: T_post = 0.8 T >= 250 tau (Sokal window ~ 5 tau < T_post / 50).
DW_A_BUDGET = 3_200_000              # 6.25x the old 512k: T_IMH 12500, T_RT 6250 (DW8 RT tau(odd_sq) 16.8 -> 5000 post)
DW_B_BUDGET = 4_608_000              # comparison B RT: T_RT 9000, post 7200 (old B RT tau(odd_sq) 16-26 -> <= 28.8)
DW_SWEEP_BUDGET = 3_072_000          # 20x the old 153.6k: T_IMH 12000, T_RT 6000 (post 9600 / 4800: tau <= 38 / 19)
DW_SWEEP_HARD = 2 * DW_SWEEP_BUDGET  # old tau_IMH > 23: T_IMH 24000, post 19200 (tau <= 77); frozen ones get flagged
DW_SWEEP_HARD_MODELS = ('dw2_loss_lv_seed0', 'dw2_loss_lvess_seed0', 'dw2_untrained_seed0', 'dw4_loss_lv_seed0',
                        'dw4_loss_lvess_seed0', 'dw4_untrained_seed0', 'dw8_loss_lv_seed0', 'dw8_loss_lvess_seed0',
                        'dw8_untrained_seed0', 'dw8_v2_lv_seed0')   # old sweep tau_IMH 23.7-540; all others <= 22.6
SEED_DW = 20461029                   # new seed base (the old runs used 20261029; same chain seed would coalesce)


def dw_line(run_dir, methods, budget, out, extra='', init=DW_INIT, n_runs=3, seed=SEED_DW):
    return (f'{DW_ENV}; [ -f {out}.json ] || {PY} dw_compare.py --run_dir {run_dir} --methods {methods} '
            f'{init} {extra}--n_chains 256 --path_budget {budget} --n_runs {n_runs} --seed {seed} --threads 4 '
            f'--out {out}.json > {out}.log 2>&1')


def dw_tasks():
    out = []
    models = [m.strip() for m in open(os.path.join(HERE, 'dw_sweep_models.txt')) if m.strip()]
    hard = [m for m in models if os.path.basename(m) in DW_SWEEP_HARD_MODELS]
    assert len(hard) == len(DW_SWEEP_HARD_MODELS), hard
    easy = [m for m in models if m not in hard]
    for m in hard + easy:            # hard (long) models first
        n = os.path.basename(m)
        B = DW_SWEEP_HARD if m in hard else DW_SWEEP_BUDGET
        out.append(dw_line(m, 'path_imh round_trip', B, f'{DWS}/{n}'))
        out.append(dw_line(m, 'snis', 153_600, f'{DWS}/{n}_snis', init=''))   # SNIS: no start state
    a = []
    for name, K in (('dw8_long_score_seed0', 5), ('dw4_anneal_seed0', 4)):
        for meth in ('path_imh', 'round_trip', f'rt_partial:{K}'):
            a.append(dw_line(f'{DWR}/{name}', meth, DW_A_BUDGET, f'{DWO}/{name.split("_")[0]}_A_{meth.replace(":", "")}'))
        a.append(dw_line(f'{DWR}/{name}', f'path_imh round_trip rt_partial:{K}', 256_000,
                         f'{DWO}/{name.split("_")[0]}_relax_worst', init='--init proposal_worst --burn_fraction 0.2',
                         seed=SEED_DW + 200_000))   # not the comparison-A chain seeds
    for b in ('fixed_s0', 'fixed_s1', 'fixed_s2', 'learned_s1', 'learned_s2'):   # ~5 h per run: one task per run
        for r in range(3):
            a.insert(0, dw_line(f'{RR}/dw/B_dw4_{b}', 'round_trip', DW_B_BUDGET, f'{DWO}/B_dw4_{b}.A_r{r}',
                                n_runs=1, seed=SEED_DW + 1000 * r))
    return a + out                   # comparison A / B first, then the sweep


LJC = f'{CODE}/lj13'
LJB = '/media/scratch/mqian/lj_buffer'
LJO = f'{RR}/lj13_noHMC'
LJ_BUDGET = 400_000                  # 32 chains: T_IMH 12500, T_RT 6250, T_partial(K=8) 43750 (post: tau <= 40 / 20)
LJ_CHAINS, LJ_THREADS = 32, 5       # the gate needs long chains, not many; 9 tasks x 5 threads = 45 cores
SEED_LJ = 20461029


def lj13_tasks():
    out = []
    for meth in ('imh', 'rt', 'rt_partial:8'):
        tag = meth.replace(':', '')
        for r in range(3):
            d = f'{LJO}/A_{tag}_r{r}'
            out.append(f'export OMP_NUM_THREADS={LJ_THREADS} MKL_NUM_THREADS={LJ_THREADS} LJ_CODE={LJB}/code_rtlocal; cd {LJC}; mkdir -p {d}; '
                       f'[ -f {d}/res.json ] || {PY} lj13_compare.py --run-dir {LJB}/runs/lj13sc_a05T56_s0 '
                       f'--refs {LJB}/refs --methods {meth} --init proposal --burn-fraction 0.2 --n-chains {LJ_CHAINS} '
                       f'--path-budget {LJ_BUDGET} --n-runs 1 --seed {SEED_LJ + 1000 * r} --threads {LJ_THREADS} --device cpu '
                       f'--out {d}/res.json > {d}.log 2>&1; cd {LJB}/code; for f in {d}/res/*_chain.npz; do '
                       f'[ -f ${{f%.npz}}_lit2k.json ] || {PY} lit_metrics.py --system lj13 --refs ../refs --samples $f '
                       f'--n 2000 --reps 3 --threads 1 --out ${{f%.npz}}_lit2k.json > ${{f%.npz}}_lit2k.log 2>&1; done')
    return out


def write(name, lines, head, threads=4):
    with open(os.path.join(HERE, name), 'w', newline='\n') as f:
        f.write(f'# {head}\n# generated by make_tasks_E1_E4.py; {len(lines)} tasks, {threads} threads each\n')
        f.write('\n'.join(lines) + '\n')
    print(name, len(lines))


if __name__ == '__main__':
    write('tasks_E1_refs.txt', refs(), 'E1 HMC+Z2 references L = 4, 6, 10 (reference numbers only; run first)')
    write('tasks_E1_L16_L12.txt', compare([16, 12]),
          'E1 compare, L = 16 and 12, proposal starts (needs job 26830 refL references)')
    write('tasks_E1_L10_L6_L4.txt', compare([10, 6, 4]),
          'E1 compare, L = 10, 6, 4, proposal starts (needs tasks_E1_refs.txt done)')
    write('tasks_rerun_noHMC.txt', rerun_noHMC(),
          'no-HMC re-run, phi4 L = 8 paper_L8: refs, kappa .30 long RT, comparison A kappa .30 -> .20, relaxation')
    write('tasks_rerun_noHMC_dw.txt', dw_tasks(),
          'no-exact-start re-run, DW: comparison A (DW8, DW4), worst-mode relaxation, comparison B RT, 81-model sweep')
    write('tasks_rerun_noHMC_lj13.txt', lj13_tasks(),
          'no-reference-start re-run, LJ13 comparison A (IMH, RT, partial RT K = 8; 3 runs; + lit_metrics W2)', threads=LJ_THREADS)
