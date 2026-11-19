#!/bin/bash
#SBATCH -J force
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -t 2-00:00:00
#SBATCH --nice=980
#SBATCH -o /media/scratch/mqian/logs/sImhR-%j.out
TF=/media/scratch/mqian/logs/sImhR-${SLURM_JOB_ID}.tasks
grep -v "^\s*#" /media/scratch/mqian/revision_code/results_sync/revision/tasks/tasks_single_imhG_ready_20261107.txt | grep -v "^\s*$" > $TF
N=$(wc -l < $TF); echo "tasks $N $(date +%T)"
seq 1 $N | xargs -P 10 -I{} bash -c "sed -n {}p $TF | bash > /media/scratch/mqian/logs/sImhR-${SLURM_JOB_ID}-{}.log 2>&1; echo \$(date +%T) line {} rc \$?"
