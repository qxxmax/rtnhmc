#!/bin/bash
#SBATCH -J force
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 2-00:00:00
#SBATCH --nice=980
#SBATCH -o /media/scratch/mqian/logs/whsign-%j.out
TF=/media/scratch/mqian/logs/whsign-${SLURM_JOB_ID}.tasks
grep -v "^\s*#" /media/scratch/mqian/revision_code/results_sync/revision/tasks/tasks_wh_sign_20261107.txt | grep -v "^\s*$" > $TF
N=$(wc -l < $TF); echo "tasks $N $(date +%T)"
seq 1 $N | xargs -P 8 -I{} bash -c "sed -n {}p $TF | bash; echo \$(date +%T) line {} rc \$?"
