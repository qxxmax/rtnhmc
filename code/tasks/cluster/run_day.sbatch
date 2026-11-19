#!/bin/bash
# Day runner (2026-11-08): like run_night.sbatch but stops at 19:35 the same day (children get SIGTERM; longchain
# checkpoints), no resubmission; the night runner of the same task file continues. Variables: TFILE, TAG, P.
#SBATCH -J force
#SBATCH -t 10:00:00
#SBATCH --nice=850
cd /media/scratch/mqian
SRC=/media/scratch/mqian/revision_code/results_sync/revision/tasks
TF=/media/scratch/mqian/logs/${TAG}-${SLURM_JOB_ID}.tasks
grep -v "^\s*#" $SRC/$TFILE | grep -v "^\s*$" > $TF
N=$(wc -l < $TF); echo "tasks $N $(date +%T)"
END=$(date -d 'today 19:35' +%s)
timeout -s TERM $((END - $(date +%s))) bash -c "seq 1 $N | xargs -P $P -I{} bash -c 'sed -n {}p $TF | bash > /media/scratch/mqian/logs/${TAG}-${SLURM_JOB_ID}-{}.log 2>&1; echo \$(date +%T) line {} rc \$?'"
echo "stopped $(date +%T)"
