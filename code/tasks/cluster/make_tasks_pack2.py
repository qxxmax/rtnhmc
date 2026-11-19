#!/usr/bin/env python3
"""Cluster CPU pack 2 (2026-11-04): tasks moved from the local GPU (README_cluster_pack2.md).

  A  CMCD / DBS (+Z2) family at L = 8: train (official recipe, 10000 it) + final eval + chains
     (snis, imh, rt, rt_partial:40; 3 runs, proposal starts, burn .2, window rule 50, cap 2^22 paths per run)
  B  alpha mechanism test: CMCD27, CMCD30, DBS30 (rt_cmcd_phi4 --save-final + alpha_cmcd.py) and
     DW4 / DW8 v4 comparison-A kernels (alpha_dw.py chains + alpha); 512 chains, M = 512, R = 64
Packs C (l8opt), D (untrained paper_L8) and E (MW32) of the first plan run on the local CPU instead (coordinator
2026-11-04) and are not generated here.
Every line skips work whose output (or .done marker) exists; 4 threads per task.
Usage: python make_tasks_pack2.py [--skip-family cmcd0.29,dbs0.20,...]   (cells trained elsewhere)
"""
import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SCR = '/media/scratch/mqian'
CODE = f'{SCR}/revision_code'                 # rsync of revision/, results_sync/models/, results_sync/logz_ref.json
PY = f'{SCR}/envs/sps/bin/python'
RUNS = f'{SCR}/revision_runs'
P2 = f'{RUNS}/pack2'
MODELS = f'{CODE}/results_sync/models/cmcd_torch'     # model.pt + final_eval.json + SHA256SUMS
REFDIR = f'{RUNS}/phi4'                       # HMC+Z2 references (reference numbers only)
ENV = f'export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NHMC_CODE={SCR}/phi4_bench/nhmc_code NHMC_BENCH={CODE}'
DW_RES = f'{SCR}/protein/NHMC_DW_PHI4_RUNNABLE_20260921/dw_v2/results'
# .27 / .28 / .30 are trained locally (gpu_E: cmcd27/30 + dbs30 done, dbs27 / cmcd28 / dbs28 training)
FAMILY_KAPPAS = ['0.20', '0.21', '0.22', '0.23', '0.24', '0.25', '0.26', '0.29']
DEFAULT_SKIP = ''


def run_name(alg, k):
    return f'{alg}z2_k{k}_s0_T80_dt0.1_it10000'


def pack_a(skip):
    lines = ['# Pack A: CMCD / DBS +Z2 family, L = 8 (cmcd_phi4.py official recipe; rt_cmcd_phi4.py chains).',
             '# Per task: train (CMCD ~1 h, DBS ~1.5 h at 4 threads; resumes from ckpt.pt), final eval N = 102400,',
             '# then chains: snis imh rt rt_partial:40, 256 chains, 500k paths x 3 runs, proposal starts, burn .2,',
             '# window rule (Sokal window < n_post / 50 by doubling, cap 2^22 paths per run). The chains wait up to 1 h',
             '# for the reference and stop (rc 3, retried at the next submission) if it is still missing:',
             f'# for the HMC+Z2 reference {RUNS}/phi4/hmc_k0XX.json (kappa .20-.26 come from tasks_rerun_noHMC.txt).',
             f'# Skipped (local GPU): {",".join(sorted(skip)) or "-"}']
    for k in sorted(FAMILY_KAPPAS, reverse=True):          # hard (high kappa) cells first
        for alg in ('dbs', 'cmcd'):                        # DBS first (longer training)
            if f'{alg}{k}' in skip:
                continue
            o = f'{P2}/cmcd_family/{run_name(alg, k)}'
            ref = f'{REFDIR}/hmc_k{int(round(float(k) * 100)):03d}.json'
            lines.append(
                f'{ENV}; cd {CODE}/revision/cmcd_torch; O={o}; mkdir -p $O; '
                f'[ -f $O/final_eval.json ] || {{ RES=; [ -f $O/ckpt.pt ] && RES=--resume; {PY} -u cmcd_phi4.py '
                f'--alg {alg} --kappa {k} --T 80 --dt 0.1 --iters 10000 --batch 500 --lr 5e-4 --seed 0 --n-final 102400 '
                f'--chunk 4096 --device cpu --threads 4 --ckpt-min 10 $RES --out $O >> $O/train.log 2>&1; }}; '
                f'[ -f $O/final_eval.json ] && [ ! -f $O/rt_chains.done ] && {{ REF={ref}; '
                f'for i in $(seq 12); do [ -f $REF ] && break; sleep 300; done; [ -f $REF ] || {{ echo "missing $REF"; exit 3; }}; '
                f'{PY} -u rt_cmcd_phi4.py '
                f'--model $O/model.pt --kappa {k} --methods snis imh rt rt_partial:40 --burn-fraction 0.2 '
                f'--window-ratio 50 --max-paths 4194304 --n-chains 256 --path-budget 500000 --n-runs 3 --init proposal '
                f'--device cpu --threads 4 --ref $REF --out $O/rt_chains.json > $O/rt.log 2>&1 && touch $O/rt_chains.done; }} || echo "chains skipped (done) or failed: $O"')
    return lines


def pack_b():
    lines = ['# Pack B: alpha mechanism test (alpha_test_NOTE.md). Chains: 512 chains x 1 run, 768000 paths, proposal',
             '# starts, burn .2, window rule 50 (cap 2^24 paths CMCD/DBS, 2^23 DW); final states saved; then the alpha',
             '# (chain seed base 20261105: disjoint from the local gpu_E chain runs of the same models, base 20261103)',
             '# test with M = 512 states, R = 64 proposals per state, plus the weight-aware IMH pool (512 ... 8192).',
             '# dbs27: add the commented line once results_sync/models/dbsz2_k0.27_* is uploaded.']
    cm = [('cmcdz2_k0.27_s0_T80_dt0.1_it10000', '0.27', 'cmcd27'), ('cmcdz2_k0.30_s0_T80_dt0.1_it10000', '0.30', 'cmcd30'),
          ('dbsz2_k0.30_s0_T80_dt0.1_it10000', '0.30', 'dbs30'), ('dbsz2_k0.27_s0_T80_dt0.1_it10000', '0.27', 'dbs27')]
    for name, k, tag in cm:
        o = f'{P2}/alpha'
        ref = f'{REFDIR}/hmc_k{int(round(float(k) * 100)):03d}.json'
        line = (f'{ENV}; cd {CODE}/revision/cmcd_torch; O={o}; mkdir -p $O; M={MODELS}/{name}/model.pt; '
                f'[ -f $O/chains_{tag}.done ] || {{ {PY} -u rt_cmcd_phi4.py --model $M --kappa {k} '
                f'--methods imh rt rt_partial:40 --burn-fraction 0.2 --window-ratio 50 --max-paths 16777216 --n-chains 512 '
                f'--path-budget 768000 --n-runs 1 --init proposal --seed 20261105 --device cpu --threads 4 --ref {ref} --save-final '
                f'--out $O/chains_{tag}.json > $O/chains_{tag}.log 2>&1 && touch $O/chains_{tag}.done; }}; '
                f'[ -f $O/chains_{tag}.done ] && [ ! -f $O/alpha_{tag}.done ] && {{ {PY} -u alpha_cmcd.py --model $M '
                f'--kappa {k} --chains $O/chains_{tag}.json --methods imh rt rt_partial:40 --M 512 --R 64 --chunk 32 '
                f'--imh-pool 512,1024,2048,4096,8192 --pool-reps 10 --threads 4 --out $O/alpha_{tag}.json '
                f'> $O/alpha_{tag}.log 2>&1 && touch $O/alpha_{tag}.done; }} || echo "alpha skipped (done) or failed: {tag}"')
        lines.append(('# ' if tag == 'dbs27' else '') + line)
    for model, meths, tag in (('dw8_long_score_seed0', 'path_imh round_trip rt_partial:5', 'dw8'),
                              ('dw4_anneal_seed0', 'path_imh round_trip rt_partial:4', 'dw4')):
        o = f'{P2}/alpha'
        lines.append(
            f'{ENV}; cd {CODE}/revision/gpu_common; O={o}; mkdir -p $O; RD={DW_RES}/{model}; '
            f'[ -f $O/chains_{tag}.done ] || {{ {PY} -u alpha_dw.py chains --run_dir $RD --checkpoint kernel_best.pt '
            f'--p_flip 0.05 --methods {meths} --n_chains 512 --path_budget 2097152 --n_runs 1 --burn_fraction 0.2 '
            f'--window_ratio 50 --max_paths 8388608 --threads 4 --out $O/chains_{tag}.json > $O/chains_{tag}.log 2>&1 '
            f'&& touch $O/chains_{tag}.done; }}; [ -f $O/chains_{tag}.done ] && [ ! -f $O/alpha_{tag}.done ] && '
            f'{{ {PY} -u alpha_dw.py alpha --run_dir $RD --checkpoint kernel_best.pt --p_flip 0.05 --methods {meths} '
            f'--chains $O/chains_{tag}.json --M 512 --R 64 --chunk 64 --imh_pool 512,1024,2048,4096,8192 --pool_reps 10 '
            f'--threads 4 --out $O/alpha_{tag}.json > $O/alpha_{tag}.log 2>&1 && touch $O/alpha_{tag}.done; }} || echo "alpha skipped (done) or failed: {tag}"')
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-family', default=DEFAULT_SKIP, help='comma list of <alg><kappa> trained elsewhere')
    a = ap.parse_args()
    skip = {s for s in a.skip_family.split(',') if s}
    for tag, lines in (('A', pack_a(skip)), ('B', pack_b())):
        path = os.path.join(HERE, f'tasks_pack2_{tag}.txt')
        n = sum(1 for ln in lines if not ln.startswith('#'))
        with open(path, 'w', newline='\n') as f:
            f.write(f'# generated by make_tasks_pack2.py; {n} tasks, 4 threads each\n')
            f.write('\n'.join(lines) + '\n')
        print(path, n, 'tasks')


if __name__ == '__main__':
    main()
