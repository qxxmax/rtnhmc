#!/bin/bash
#SBATCH -J force
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 2-00:00:00
#SBATCH --nice=990
#SBATCH -o /media/scratch/mqian/logs/cTrA-%j.out
TF=/media/scratch/mqian/logs/cTrA-${SLURM_JOB_ID}.tasks
grep -v "^\s*#" /media/scratch/mqian/revision_code/results_sync/revision/tasks/tasks_cond_train_A_20261107.txt | grep -v "^\s*$" > $TF
N=$(wc -l < $TF); echo "tasks $N $(date +%T)"
seq 1 $N | xargs -P 4 -I{} bash -c "sed -n {}p $TF | bash > /media/scratch/mqian/logs/cTrA-${SLURM_JOB_ID}-{}.log 2>&1; echo \$(date +%T) line {} rc \$?"
