#!/bin/bash
# Cluster CPU version of train_snapshots.sh (same recipe and snapshot iterations; --device cpu --threads 4), then the
# IMH and RT single chains of every snapshot (same arguments as tasks_minimal_S.txt; all snapshots of one ladder share
# the model dir name, hence the chain seeds). usage: train_snapshots_cluster.sh ALG KAPPA(e.g. 0.27)
ALG=$1; K=$2; KK=$(printf %03d $(python3 -c "print(round($K*100))"))
PY=/media/scratch/mqian/envs/sps/bin/python
export NHMC_CODE=/media/scratch/mqian/revision_code/nhmc_code OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=
D=/media/scratch/mqian/revision_runs/snapshots/${ALG}z2_k$K
cd /media/scratch/mqian/revision_code/revision/cmcd_torch || exit 1
mkdir -p $D
COMMON="--alg $ALG --kappa $K --T 80 --dt 0.1 --iters 10000 --batch 500 --lr 5e-4 --seed 0 --n-final 102400 --chunk 4096 --device cpu --threads 4"
for N in 250 500 1000 2000 4000; do
  [ -f $D/snap_it$N.pt ] && continue
  $PY -u cmcd_phi4.py $COMMON --resume --stop-at $N --out $D >> $D/train.log 2>&1 || exit 2
  cp $D/ckpt.pt $D/snap_it$N.pt
done
[ -f $D/model.pt ] || $PY -u cmcd_phi4.py $COMMON --resume --out $D >> $D/train.log 2>&1 || exit 2
[ -f $D/snap_it10000.pt ] || cp $D/model.pt $D/snap_it10000.pt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
O=/media/scratch/mqian/revision_runs/longchain_min
for N in 250 500 1000 2000 4000 10000; do
  ( cd /media/scratch/mqian/revision_code/revision/cmcd_torch
    [ -f $D/eval_it$N/final_eval.json ] || $PY -u cmcd_phi4.py $COMMON --threads 1 --load $D/snap_it$N.pt --out $D/eval_it$N >> $D/eval.log 2>&1
    cd /media/scratch/mqian/revision_code/revision/longchain
    for M in imh rt; do
      F=snap_${ALG}_k${KK}_it${N}_$M
      [ -f $O/$F.json ] || $PY longchain.py --kind cmcd --model $D/snap_it$N.pt --kappa $K --method $M \
        --ref /media/scratch/mqian/revision_code/results_sync/revision/phi4/hmc_k${KK}.json --n-chains 4 --n-start 10000 \
        --n-max 200000 --device cpu --threads 1 --out $O/$F.json > $O/$F.log 2>&1 &
    done; wait ) &
done
wait; echo done
