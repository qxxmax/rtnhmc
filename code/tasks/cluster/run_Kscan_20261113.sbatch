#!/bin/bash
# Protocol-length pilot (2026-11-13): does a longer NHMC path train a better kernel? For cell L$1 kappa $2 train
# (a) num_steps 32, (b) num_steps 64, (c) num_steps 16 with n_leapfrog 10; recipe otherwise identical to every cell
# (seed 0). New dirs phi4_buffer/L<L>_k<k>_{K32,K64,lf10}_s0; nothing existing is touched. Each model: eval N=262144
# (seed 7, as the baselines) + three 65536-path evals with seeds 11/12/13; the baseline (K16) gets the same 3 evals
# into the new dir Kscan_base_20261113/.
#SBATCH -J force
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=48G
#SBATCH -t 2-00:00:00
#SBATCH --nice=1000
L=$1 K=$2 BASE=$3
cd /media/scratch/mqian/revision_code/nhmc_code
PY=/media/scratch/mqian/envs/sps/bin/python
R=/media/scratch/mqian/revision_runs/phi4_buffer
KT=$(echo $K | tr -d .)
ev() {  # $1 run dir, $2 out dir
  [ -f $2/eval_N262144.json ] || $PY -u eval_generic.py --run-dir $1 --n 262144 --batch 16384 --kappas 0.$KT --device cuda --out $2/eval_N262144.json > $2/eval.log 2>&1
  for sd in 11 12 13; do [ -f $2/eval_s$sd.json ] || $PY -u eval_generic.py --run-dir $1 --ckpt kernel_final.pt --n 65536 --batch 16384 --kappas 0.$KT --device cuda --seed $sd --out $2/eval_s$sd.json > $2/eval_s$sd.log 2>&1; done
}
for v in K32 K64 lf10; do
  case $v in K32) NS=32 LF=5;; K64) NS=64 LF=5;; lf10) NS=16 LF=10;; esac
  D=$R/L${L}_k${KT}_${v}_s0; mkdir -p $D
  ( export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
    [ -f $D/kernel_final.pt ] || $PY -u train_cond_phi4_kappa_explicit.py --kernel nhmc_resid2 --use-z2 --odd-nets --fixed-pflip 0.5 --L $L --n-leapfrog $LF --batch-size 48 --buffer --buffer-start 1000 --buffer-chains 64 --buffer-every 2 --buffer-size 20000 --buffer-weight 1.0 --device cuda --kappa-min 0.$KT --kappa-max 0.$KT --num-steps $NS --epochs 20000 --seed 0 --save-dir $D >> $D/train.log 2>&1
    [ -f $D/kernel_final.pt ] && ev $D $D
    echo "$v done $(date +%T)" ) &
done
B=/media/scratch/mqian/revision_runs/Kscan_base_20261113/L${L}_k${KT}; mkdir -p $B
for sd in 11 12 13; do [ -f $B/eval_s$sd.json ] || $PY -u eval_generic.py --run-dir $BASE --ckpt kernel_final.pt --n 65536 --batch 16384 --kappas 0.$KT --device cuda --seed $sd --out $B/eval_s$sd.json > $B/eval_s$sd.log 2>&1; done
wait
