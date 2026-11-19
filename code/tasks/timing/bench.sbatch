#!/bin/bash
#SBATCH -J force
#SBATCH -p experiment
#SBATCH -A himster3exp
#SBATCH -N 1
#SBATCH -c 80
#SBATCH --mem=48G
#SBATCH -t 16:00:00
#SBATCH --nice=1000
#SBATCH -o /lustre/him/him-mam/mqian/bench_timing_20261118/logs/bench-%j.out
# Appendix-B timing table: 80 cores allocated (cap), all measurements sequential on the first 4 of them,
# 4 threads; the other 76 allocated cores stay idle so no other job runs on them during the benchmark.
export STOP_S=167400
X=/lustre/him/him-mam/mqian/bench_timing_20261118
echo "job $SLURM_JOB_ID $SLURM_JOB_NODELIST $(date -Is) cpus $(python3 -c 'import os;print(sorted(os.sched_getaffinity(0)))')"
srun -n1 -c 80 bash $X/code/run_bench.sh $X/code/configs_main.txt auto main
srun -n1 -c 80 bash $X/code/run_bench.sh $X/code/configs_repeat.txt auto repeat
echo "done $(date -Is)"
