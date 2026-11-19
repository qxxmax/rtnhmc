#!/bin/bash
# Protocol-length scan, wave 1 (2026-11-13; design after Bulgarelli, Cellini, Nada arXiv:2412.00200): single-kappa
# nhmc_resid2 models with num_steps K in {8,16,32,64}, same recipe as every cell (n_leapfrog 5, 20000 epochs,
# buffer), several training seeds. Usage: sbatch run_Kscan2_20261113.sbatch L KAPPA "K8 K32 ..." SEED
# New dirs revision_runs/phi4_buffer/L<L>_k<kk>_K<K>_s<seed>; existing dirs with kernel_final.pt are skipped.
# Each model: eval N=262144 (seed 7) + three 65536-path evals (seeds 11/12/13).
#SBATCH -J force
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=48G
#SBATCH -t 2-00:00:00
L=$1 K=$2 VS=$3 SEED=$4
cd /media/scratch/mqian/revision_code/nhmc_code
PY=/media/scratch/mqian/envs/sps/bin/python
R=/media/scratch/mqian/revision_runs/phi4_buffer
KT=$(echo $K | tr -d .)
for v in $VS; do
  NS=${v#K}
  D=$R/L${L}_k${KT}_${v}_s$SEED; mkdir -p $D
  ( export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
    [ -f $D/kernel_final.pt ] || $PY -u train_cond_phi4_kappa_explicit.py --kernel nhmc_resid2 --use-z2 --odd-nets --fixed-pflip 0.5 --L $L --n-leapfrog 5 --batch-size 48 --buffer --buffer-start 1000 --buffer-chains 64 --buffer-every 2 --buffer-size 20000 --buffer-weight 1.0 --device cuda --kappa-min 0.$KT --kappa-max 0.$KT --num-steps $NS --epochs 20000 --seed $SEED --save-dir $D >> $D/train.log 2>&1
    if [ -f $D/kernel_final.pt ]; then
      [ -f $D/eval_N262144.json ] || $PY -u eval_generic.py --run-dir $D --n 262144 --batch 16384 --kappas 0.$KT --device cuda --out $D/eval_N262144.json > $D/eval.log 2>&1
      for sd in 11 12 13; do [ -f $D/eval_s$sd.json ] || $PY -u eval_generic.py --run-dir $D --ckpt kernel_final.pt --n 65536 --batch 16384 --kappas 0.$KT --device cuda --seed $sd --out $D/eval_s$sd.json > $D/eval_s$sd.log 2>&1; done
    fi
    echo "$v s$SEED done $(date +%T)" ) &
done
wait
