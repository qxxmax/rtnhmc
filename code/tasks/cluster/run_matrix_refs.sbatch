#!/bin/bash
#SBATCH -J force
#SBATCH -c 8

#SBATCH --mem=24G
#SBATCH -t 2-00:00:00
#SBATCH --nice=800
#SBATCH -o /media/scratch/mqian/logs/mxrefs-%j.out
TF=/media/scratch/mqian/logs/mxrefs-${SLURM_JOB_ID}.tasks
grep -v "^\s*#" /media/scratch/mqian/revision_code/results_sync/revision/tasks/tasks_matrix_refs_20261107.txt | grep -v "^\s*$" > $TF
N=$(wc -l < $TF); echo "tasks $N $(date +%T)"
seq 1 $N | xargs -P 8 -I{} bash -c "sed -n {}p $TF | bash > /media/scratch/mqian/logs/mxrefs-${SLURM_JOB_ID}-{}.log 2>&1; echo \$(date +%T) line {} rc \$?"
