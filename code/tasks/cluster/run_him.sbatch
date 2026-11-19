#!/bin/bash
# HIMSTER3 pool runner (2026-11-11). Any number of copies of this job work through one shared pool of task lines: a line is
# claimed by an atomic mkdir in $POOL/claim, so each line runs once whatever the number of jobs; a line that does not
# return 0 releases its claim and is not retried by the same job (task lines skip finished outputs and resume from
# checkpoints, so a later job retries it).
# Before the time limit the children get SIGTERM (longchain / raw_snis checkpoint). Variables (sbatch --export):
# POOL (pool dir, made by him_pool.sh: tasks.txt, claim/), P (parallel lines per job; -c = P x threads per line).
#SBATCH -J force
#SBATCH -p experiment
#SBATCH -A himster3exp
#SBATCH -t 12:00:00
B=/lustre/him/him-mam/mqian
cd $B
N=$(wc -l < $POOL/tasks.txt)
echo "pool $POOL lines $N job $SLURM_JOB_ID $(hostname) $(date +%T)"
worker() {
  while :; do
    k=0
    for i in $(seq 1 $N); do
      [ -e $POOL/done/$i ] || [ -e $POOL/fail/$SLURM_JOB_ID.$i ] && continue
      mkdir $POOL/claim/$i 2>/dev/null || continue
      k=1
      sed -n ${i}p $POOL/tasks.txt | bash > $POOL/logs/$i.log 2>&1
      rc=$?
      echo "$(date +%T) line $i rc $rc"
      if [ $rc -eq 0 ]; then touch $POOL/done/$i; else touch $POOL/fail/$SLURM_JOB_ID.$i; rmdir $POOL/claim/$i; [ $rc -ge 128 ] && exit; fi
      break
    done
    [ $k -eq 0 ] && return
  done
}
mkdir -p $POOL/fail
export -f worker; export POOL N
END=$(( $(date +%s) + 11 * 3600 + 40 * 60 ))
timeout -s TERM $((END - $(date +%s))) bash -c 'for p in $(seq 1 '$P'); do worker & done; wait'
echo "stopped $(date +%T)"
