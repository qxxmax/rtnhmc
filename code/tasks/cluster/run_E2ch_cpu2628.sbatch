#!/bin/bash
#SBATCH -J force
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 2-00:00:00
#SBATCH --nice=980
#SBATCH -o /media/scratch/mqian/logs/E2ch_cpu2628-%j.out
TF=/media/scratch/mqian/logs/E2ch_cpu2628-${SLURM_JOB_ID}.tasks
grep -v "^\s*#" /media/scratch/mqian/revision_code/results_sync/revision/tasks/tasks_E2_chains_cpu_k026k028.txt | grep -v "^\s*$" > $TF
N=$(wc -l < $TF); echo "tasks $N $(date +%T)"
seq 1 $N | xargs -P 8 -I{} bash -c "sed -n {}p $TF | bash > /media/scratch/mqian/logs/E2ch_cpu2628-${SLURM_JOB_ID}-{}.log 2>&1; echo \$(date +%T) line {} rc \$?"
