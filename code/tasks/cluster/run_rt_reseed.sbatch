#!/bin/bash
#SBATCH -J force
#SBATCH -c 6
#SBATCH --mem 12G
#SBATCH -t 1-00:00:00
#SBATCH --nice=1000
#SBATCH -o /media/scratch/mqian/logs/rtreseed-%j.out
cd /media/scratch/mqian
tr '\n' '\0' < /media/scratch/mqian/revision_code/results_sync/revision/tasks/tasks_rt_reseed_20261106.txt | xargs -0 -P 6 -I{} bash -c '{}'
