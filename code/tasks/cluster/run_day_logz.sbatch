#!/bin/bash
#SBATCH -J force
#SBATCH -c 24
#SBATCH --mem=64G
#SBATCH -t 9:50:00
#SBATCH --nice=800
#SBATCH -o /media/scratch/mqian/logs/dLogz-%j.out
# daytime helper for the chain free energy: same task lines as the main logz runner, taken from the end, lock = the
# task log file (created at python launch); models not ready -> skipped; stops by 19:40, unfinished tasks are reset
SRC=/media/scratch/mqian/logs/sLogz-60127.tasks
TF=/media/scratch/mqian/logs/dLogz-${SLURM_JOB_ID}.tasks
tac $SRC | sed -E "s#for i in \\\$\(seq [0-9]+\); do \[ -f \\\$D/kernel_final.pt \] \&\& break; sleep 60; done; ##" > $TF
END=$(date -d "today 19:40" +%s)
N=$(wc -l < $TF); echo "tasks $N $(date +%T) waitloops $(grep -c "seq 2880" $TF)"
run() { L=$(sed -n ${1}p $TF); R=$(( END - $(date +%s) )); [ $R -gt 600 ] || exit 0
  OUT=$(echo "$L" | grep -o "\$O/[A-Za-z0-9_]*\.json" | tail -1 | sed "s#\\\$O#/media/scratch/mqian/revision_runs/logz_chains#")
  timeout -s TERM $R bash -c "$L"; rc=$?
  [ $rc -eq 124 ] && rm -f ${OUT%.json}.log && echo "reset $OUT"
  echo "$(date +%T) line $1 rc $rc"; }
export -f run; export TF END
seq 1 $N | xargs -P 24 -I{} bash -c "run {}"
