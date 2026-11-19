#!/bin/bash
#SBATCH -J force
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 2-00:00:00
#SBATCH --nice=800
#SBATCH -o /media/scratch/mqian/logs/lzc-%j.out
TF=/media/scratch/mqian/logs/lzc-${SLURM_JOB_ID}.tasks
grep -v "^\s*#" /media/scratch/mqian/revision_code/results_sync/revision/tasks/tasks_logz_chains_20261107.txt | grep -v "^\s*$" > $TF
N=$(wc -l < $TF); echo "tasks $N $(date +%T)"
seq 1 $N | xargs -P 16 -I{} bash -c "sed -n {}p $TF | bash; echo \$(date +%T) line {} rc \$?"
