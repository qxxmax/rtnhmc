#!/bin/bash
# Make a HIMSTER3 task pool from isis task files (paths /media/scratch/mqian -> /lustre/him/him-mam/mqian) and submit
# NJOBS copies of run_him.sbatch on it. Usage: him_pool.sh NAME NJOBS P CPUS MEM taskfile [taskfile ...]
# (task files in this directory). Re-running with the same NAME clears stale claims (only when no job of the pool runs).
B=/lustre/him/him-mam/mqian
SRC=$B/revision_code/results_sync/revision/tasks
NAME=$1 NJOBS=$2 P=$3 CPUS=$4 MEM=$5; shift 5
POOL=$B/pools/$NAME
mkdir -p $POOL/claim $POOL/done $POOL/logs $B/logs
for f in "$@"; do grep -v "^\s*#" $SRC/$f | grep -v "^\s*$"; done | sed "s|/media/scratch/mqian|$B|g" > $POOL/tasks.txt
squeue -u $USER -h -o "%k" | grep -qx "$NAME" || rm -rf $POOL/claim/*
echo "$NAME: $(wc -l < $POOL/tasks.txt) lines, $(ls $POOL/done | wc -l) done"
for j in $(seq 1 $NJOBS); do
  sbatch -c $CPUS --mem=$MEM --comment=$NAME --export=ALL,POOL=$POOL,P=$P -o $B/logs/$NAME-%j.out $SRC/run_him.sbatch
done
