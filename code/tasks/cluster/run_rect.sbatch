#!/bin/bash
# Rectangular Nx x 8 free energy (2026-11-08): phase A trains the two 64 x 8 models (parallel on one card), phase B runs
# raw / SNIS and the logZ chains of the frozen models at Nx = 64 ... 1024 (P lines in parallel).
#SBATCH -J force
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 2-00:00:00
#SBATCH --nice=1000
#SBATCH -o /media/scratch/mqian/logs/rect-%j.out
SRC=/media/scratch/mqian/revision_code/results_sync/revision/tasks
for ph in A B; do
  TF=/media/scratch/mqian/logs/rect${ph}-${SLURM_JOB_ID}.tasks
  grep -v "^\s*#" $SRC/tasks_rect_${ph}_20261108.txt | grep -v "^\s*$" > $TF
  N=$(wc -l < $TF); P=$([ $ph = A ] && echo 2 || echo 4); echo "phase $ph tasks $N $(date +%T)"
  seq 1 $N | xargs -P $P -I{} bash -c "sed -n {}p $TF | bash > /media/scratch/mqian/logs/rect${ph}-${SLURM_JOB_ID}-{}.log 2>&1; echo \$(date +%T) $ph line {} rc \$?"
done
