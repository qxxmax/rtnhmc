#!/bin/bash
# node side of him_stage.sh: run $1.lines.txt (work dir $1) with $2 in parallel (same pattern as run_night.sbatch), stop
# at $3 s (children get SIGTERM and checkpoint).
W=$1
N=$(wc -l < $W.lines.txt)
timeout -s TERM $3 bash -c "seq 1 $N | xargs -P $2 -I{} bash -c 'sed -n {}p $W.lines.txt | bash > $W/line{}.log 2>&1; r=\$?; echo \$(date +%T) line {} rc \$r'"
