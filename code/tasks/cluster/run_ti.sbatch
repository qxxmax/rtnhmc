#!/bin/bash
#SBATCH -J force
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 2-00:00:00
#SBATCH -o /media/scratch/mqian/logs/ti-%j.out
TF=/media/scratch/mqian/logs/ti-${SLURM_JOB_ID}.tasks
grep -v "^\s*#" /media/scratch/mqian/revision_code/results_sync/revision/tasks/tasks_ti_wolff_hmc.txt | grep -v "^\s*$" > $TF
N=$(wc -l < $TF); echo "tasks $N $(date +%T)"
seq 1 $N | xargs -P 8 -I{} bash -c "sed -n {}p $TF | bash; echo \$(date +%T) line {} rc \$?"
