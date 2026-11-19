#!/bin/bash
# Reproduction + RT runs for the CMCD/DBS torch port. NOT started automatically: run only after approval.
# Usage (WSL, inside tmux):  bash launch_repro.sh <stage> [cpu|cuda]
#   stage cmcd27 : train cmcd+Z2 kappa .27 (10000 it) + final eval N=102400, then RT/IMH/SNIS chains
#   stage cmcd30 : same at kappa .30
#   stage dbs27 / dbs30 : DBS+Z2
# Measured CPU cost (4 threads, float32): cmcd 0.36 s/it (~1.0 h), dbs 0.54 s/it (~1.5 h); final eval ~25 s.
# GPU: activations 0.33 GB (cmcd) / 0.52 GB (dbs) + CUDA context -> expect ~1.0-1.3 GB peak per process.
# Chains (rt_cmcd_phi4.py, 256 chains, 500k paths x 3 runs x 5 methods): CPU ~30-45 min (measured 45 s per 150k paths, 2 threads).
set -euo pipefail
STAGE=${1:?stage}; DEV=${2:-cpu}
D=/mnt/d/cc/nhmc_arxiv_review_20261029/improve/cmcd_torch
PY=${PY:-$HOME/miniconda3/envs/raresafe/bin/python}
cd "$D"
case $STAGE in
  cmcd27) ALG=cmcd; K=0.27 ;;
  cmcd30) ALG=cmcd; K=0.30 ;;
  dbs27)  ALG=dbs;  K=0.27 ;;
  dbs30)  ALG=dbs;  K=0.30 ;;
  *) echo "unknown stage"; exit 1 ;;
esac
OUT=runs/${ALG}z2_k${K}_s0_T80_dt0.1_it10000
mkdir -p "$OUT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
RESUME=""; [ -f "$OUT/ckpt.pt" ] && RESUME="--resume"
if [ ! -f "$OUT/final_eval.json" ]; then
  nice -n 19 $PY -u cmcd_phi4.py --alg $ALG --kappa $K --T 80 --dt 0.1 --iters 10000 --batch 500 --lr 5e-4 \
     --seed 0 --n-final 102400 --chunk 4096 --device $DEV --threads 4 --ckpt-min 10 $RESUME --out "$OUT" \
     2>&1 | tee -a "$OUT/train.log"
fi
nice -n 19 $PY -u rt_cmcd_phi4.py --model "$OUT/model.pt" --kappa $K \
   --methods snis imh rt rt_partial:40 rt_partial:72 --n-chains 256 --path-budget 500000 --n-runs 3 \
   --device $DEV --threads 4 --out "$OUT/rt_chains.json" 2>&1 | tee -a "$OUT/rt.log"
