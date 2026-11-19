#!/bin/bash
# Until runner (2026-11-11): like run_day.sbatch but stops at $STOP (e.g. "2026-11-11 08:00") (children get SIGTERM; longchain
# checkpoints), no resubmission; the night runner of the same task file continues. Variables: TFILE, TAG, P, STOP.
#SBATCH -J force
#SBATCH -t 10:00:00
#SBATCH --nice=850
cd /media/scratch/mqian
SRC=/media/scratch/mqian/revision_code/results_sync/revision/tasks
TF=/media/scratch/mqian/logs/${TAG}-${SLURM_JOB_ID}.tasks
grep -v "^\s*#" $SRC/$TFILE | grep -v "^\s*$" > $TF
N=$(wc -l < $TF); echo "tasks $N $(date +%T)"
END=$(date -d "$STOP" +%s)
timeout -s TERM $((END - $(date +%s))) bash -c "seq 1 $N | xargs -P $P -I{} bash -c 'sed -n {}p $TF | bash > /media/scratch/mqian/logs/${TAG}-${SLURM_JOB_ID}-{}.log 2>&1; echo \$(date +%T) line {} rc \$?'"
echo "stopped $(date +%T)"
