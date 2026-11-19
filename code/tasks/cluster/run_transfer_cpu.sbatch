#!/bin/bash
#SBATCH -J force
#SBATCH -c 16

#SBATCH --mem=48G
#SBATCH -t 2-00:00:00
#SBATCH --nice=850
#SBATCH -o /media/scratch/mqian/logs/trcpu-%j.out
TF=/media/scratch/mqian/logs/trcpu-${SLURM_JOB_ID}.tasks
grep -v "^\s*#" /media/scratch/mqian/revision_code/results_sync/revision/tasks/tasks_transfer_cpu_20261107.txt | grep -v "^\s*$" > $TF
N=$(wc -l < $TF); echo "tasks $N $(date +%T)"
seq 1 $N | xargs -P 16 -I{} bash -c "sed -n {}p $TF | bash > /media/scratch/mqian/logs/trcpu-${SLURM_JOB_ID}-{}.log 2>&1; echo \$(date +%T) line {} rc \$?"
