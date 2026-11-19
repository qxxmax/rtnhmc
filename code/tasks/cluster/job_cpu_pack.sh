#!/bin/bash
#SBATCH -J force
#SBATCH -p iqcb
#SBATCH -c 46
#SBATCH --mem=96G
#SBATCH -t 2-00:00:00
#SBATCH -o /media/scratch/mqian/revision_runs/logs/cpupack-%j.out
# CPU-only pack: TASKS = one shell command per line, run P at a time (each line sets its own threads).
# Re-submitting the same pack skips finished tasks ([ -f json ] guards) and resumes partial chain tasks of
# phi4/dw/lj13_compare.py from <out>.ckpt/ (checkpoint every 20 min and at the SIGTERM of the time limit).
set -uo pipefail
export CUDA_VISIBLE_DEVICES= MKL_NUM_THREADS=1 MPLBACKEND=Agg
grep -v "^\s*#" "$TASKS" | grep -v "^\s*$" > /media/scratch/mqian/revision_runs/logs/cpupack-${SLURM_JOB_ID}.tasks
export TF=/media/scratch/mqian/revision_runs/logs/cpupack-${SLURM_JOB_ID}.tasks
seq 1 $(wc -l < $TF) | xargs -P ${P:-13} -I{} bash -c "echo \"[\$(date +%T)] start {}\"; sed -n \"{}p\" \$TF | bash; echo \"[\$(date +%T)] done {} rc=\$?\""
echo "END $(date -Is)"
