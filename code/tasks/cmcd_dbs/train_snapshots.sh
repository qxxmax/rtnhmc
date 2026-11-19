#!/bin/bash
# Fixed-target model-quality ladder (2026-11-06): train CMCD+Z2 / DBS+Z2 with the standard recipe and keep snapshots
# at intermediate iterations. --stop-at N writes ckpt.pt at iteration N and exits; --resume continues bit-exactly
# (model, Adam state, RNG), so the snapshots lie on one uninterrupted training trajectory. Then the N = 102400 SNIS
# evaluation of every snapshot (cmcd_phi4.py --load snap_itN.pt).
# usage: train_snapshots.sh ALG KAPPA OUTDIR
ALG=$1; K=$2; D=$3
W=D:/cc/tmp_wt_models
PY=C:/Users/mxq/miniconda3/envs/torch_env/python.exe
export NHMC_CODE=$W/nhmc_code
cd /d/cc/tmp_wt_models/revision/cmcd_torch || exit 1
mkdir -p $D
COMMON="--alg $ALG --kappa $K --T 80 --dt 0.1 --iters 10000 --batch 500 --lr 5e-4 --seed 0 --n-final 102400 --chunk 4096 --device cuda"
for N in 250 500 1000 2000 4000; do
  [ -f $D/snap_it$N.pt ] && continue
  $PY -u cmcd_phi4.py $COMMON --resume --stop-at $N --out $D >> $D/train.log 2>&1 || exit 2
  cp $D/ckpt.pt $D/snap_it$N.pt
done
[ -f $D/model.pt ] || $PY -u cmcd_phi4.py $COMMON --resume --out $D >> $D/train.log 2>&1 || exit 2
for N in 250 500 1000 2000 4000; do
  [ -f $D/eval_it$N/final_eval.json ] || $PY -u cmcd_phi4.py $COMMON --load $D/snap_it$N.pt --out $D/eval_it$N >> $D/eval.log 2>&1
done
[ -f $D/eval_it10000/final_eval.json ] || $PY -u cmcd_phi4.py $COMMON --load $D/model.pt --out $D/eval_it10000 >> $D/eval.log 2>&1
echo done
