#!/bin/bash
# HIMSTER3 staging runner v2 (2026-11-12), same usage as him_stage.sh:
#   salloc -J force -p experiment -A himster3exp -N1 -c CPUS --mem=MEM -t 12:00:00 bash him_stage2.sh LINES P [EXTRA_TAR]
# Changes vs him_stage.sh: (1) a pull only advances its marker after the login-side tar has succeeded (a failed or hung
# pull no longer marks files as delivered); (2) heavy resume state (*.ckpt.pt, revision_runs/pregen/) is pulled every
# 60 min, everything else every 15 min, both at the end; (3) each pull has a time limit (a hung Lustre client does not
# block the job past its stop time). Edit this file only by appending (running jobs read it incrementally).
B=/lustre/him/him-mam/mqian
LINES=$1 P=$2 EXTRA=$3
J=$SLURM_JOB_ID W=/tmp/mq_$SLURM_JOB_ID
[ -s "$LINES" ] || { echo "no lines file $LINES"; exit 1; }
OUT=$B/himout/$J; mkdir -p $OUT
LOG=$B/logs/stage-$J; mkdir -p $LOG
echo "job $J $SLURM_JOB_NODELIST lines $(wc -l < $LINES) P $P $(date +%T)"
sed "s|/tmp/mq/|$W/|g" $LINES > $LOG/lines.txt
sbcast -f $B/stage/base.tar $W.base.tar || exit 1
sbcast -f $LOG/lines.txt $W.lines.txt || exit 1
sbcast -f $B/revision_code/results_sync/revision/tasks/him_node_run.sh $W.run.sh || exit 1
sbcast -f $B/revision_code/results_sync/revision/tasks/him_pull_node.sh $W.pull.sh || exit 1
[ -n "$EXTRA" ] && { sbcast -f $EXTRA $W.extra.tar || exit 1; }
srun -n1 bash -c "mkdir -p $W && tar xf $W.base.tar -C $W && rm $W.base.tar &&
  { [ ! -f $W.extra.tar ] || tar xf $W.extra.tar -C $W; } &&
  mkdir -p $W/revision_runs/longchain_min $W/revision_runs/raw_snis && touch $W/.t0 $W/.h0" || exit 1
echo "staged $(date +%T)"
srun -n1 --overlap bash $W.run.sh $W $P ${STOP_S:-41400} &
MAIN=$!
pull() {  # $1 = light | heavy (see him_pull_node.sh), $2 = time limit (s); the marker advances only after a good transfer
  timeout $2 bash -c "set -o pipefail; srun -n1 --overlap bash $W.pull.sh $W $1 | tar xf - -C $OUT" &&
    srun -n1 --overlap bash $W.pull.sh $W commit-$1 && echo "$(date +%T) pull $1 ok" || echo "$(date +%T) pull $1 FAILED (retried next time)"
  cp -p $OUT/line*.log $LOG/ 2>/dev/null; rm -f $OUT/line*.log
}
k=0
while kill -0 $MAIN 2>/dev/null; do
  for s in $(seq 1 90); do kill -0 $MAIN 2>/dev/null || break; sleep 10; done
  k=$((k + 1)); pull light 600; [ $((k % 4)) -eq 0 ] && pull heavy 1200
done
wait $MAIN
pull light 900; pull heavy 1500
srun -n1 --overlap rm -rf $W $W.lines.txt $W.extra.tar $W.run.sh $W.pull.sh
echo "done $(date +%T)"
