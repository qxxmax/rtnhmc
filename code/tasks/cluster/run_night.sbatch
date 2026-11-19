#!/bin/bash
# Night-only runner (user 2026-11-08: no large CPU use during the day). Runs the task file 20:00 -> ~07:40, children get
# SIGTERM at the end (longchain / lj13_compare checkpoint on SIGTERM, task lines skip finished outputs), then resubmits
# itself for the next night unless every line returned 0. Variables (sbatch --export): TFILE (task file name in this
# directory), TAG, P (parallel lines). CPUs = -c of the submission.
#SBATCH -J force
#SBATCH -t 12:00:00
#SBATCH --nice=850
cd /media/scratch/mqian
SRC=/media/scratch/mqian/revision_code/results_sync/revision/tasks
TF=/media/scratch/mqian/logs/${TAG}-${SLURM_JOB_ID}.tasks
grep -v "^\s*#" $SRC/$TFILE | grep -v "^\s*$" > $TF
N=$(wc -l < $TF); echo "tasks $N $(date +%T)"
END=$(date -d 'today 07:40' +%s); [ $END -lt $(date +%s) ] && END=$(date -d 'tomorrow 07:40' +%s)
timeout -s TERM $((END - $(date +%s))) bash -c "seq 1 $N | xargs -P $P -I{} bash -c 'sed -n {}p $TF | bash > /media/scratch/mqian/logs/${TAG}-${SLURM_JOB_ID}-{}.log 2>&1; echo \$(date +%T) line {} rc \$?'"
sleep 90
OK=$(grep -c " rc 0$" /media/scratch/mqian/logs/${TAG}-${SLURM_JOB_ID}.out 2>/dev/null); OK=${OK:-0}   # grep -c prints 0 itself (exit 1)
echo "done $OK / $N $(date +%T)"
if [ "$OK" -lt "$N" ]; then
  sbatch --begin=20:00 -c $SLURM_CPUS_PER_TASK --mem=${SLURM_MEM_PER_NODE}M --export=ALL,TFILE=$TFILE,TAG=$TAG,P=$P \
    -o /media/scratch/mqian/logs/${TAG}-%j.out $SRC/run_night.sbatch
fi
