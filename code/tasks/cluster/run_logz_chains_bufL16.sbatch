#!/bin/bash
#SBATCH -J force
#SBATCH -c 6
#SBATCH --mem=24G
#SBATCH -t 2-00:00:00
#SBATCH --nice=800
#SBATCH -o /media/scratch/mqian/logs/lzc16-%j.out
TF=/media/scratch/mqian/logs/lzc16-${SLURM_JOB_ID}.tasks
grep -v "^\s*#" /media/scratch/mqian/revision_code/results_sync/revision/tasks/tasks_logz_chains_bufL16.txt | grep -v "^\s*$" > $TF
N=$(wc -l < $TF); echo "tasks $N $(date +%T)"
seq 1 $N | xargs -P 6 -I{} bash -c "sed -n {}p $TF | bash; echo \$(date +%T) line {} rc \$?"
