#!/bin/bash
#SBATCH -J force
#SBATCH -c 4
#SBATCH --mem=12G
#SBATCH -t 2-00:00:00
#SBATCH --nice=980
#SBATCH -o /media/scratch/mqian/logs/E2ch_k024-%j.out
TF=/media/scratch/mqian/logs/E2ch_k024-${SLURM_JOB_ID}.tasks
grep -v "^\s*#" /media/scratch/mqian/revision_code/results_sync/revision/tasks/tasks_E2_chains_k024.txt | grep -v "^\s*$" > $TF
N=$(wc -l < $TF); echo "tasks $N $(date +%T)"
seq 1 $N | xargs -P 4 -I{} bash -c "sed -n {}p $TF | bash > /media/scratch/mqian/logs/E2ch_k024-${SLURM_JOB_ID}-{}.log 2>&1; echo \$(date +%T) line {} rc \$?"
