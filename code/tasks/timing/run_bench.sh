#!/bin/bash
# Appendix-B timing benchmark (2026-11-18): all configurations one after another on the same node, same 4 cores
# (taskset), OMP/MKL/torch 4 threads. Config columns: name L T s-list(comma) ckpt. Usage (inside an allocation): bash run_bench.sh CONFIGS [CORES] [TAG]
# Outputs go to $X/out/<TAG>; an existing output is never overwritten (that step is skipped).
B=/lustre/him/him-mam/mqian
X=$B/bench_timing_20261118
CFG=$1; CORES=${2:-auto}; TAG=${3:-main}
PY=$B/envs/sps/bin/python
NC=$B/revision_code/nhmc_code
export LONGCHAIN_DIR=$B/revision_code/revision/longchain NHMC_CODE=$NC
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=
O=$X/out/$TAG; mkdir -p $O
# auto: the first 4 cores of this step's own CPU set (cores outside the allocation are not usable)
[ "$CORES" = auto ] && CORES=$(python3 -c "import os; c=sorted(os.sched_getaffinity(0)); print(','.join(map(str,c[:4])))")
TS="taskset -c $CORES"
{
  echo "host $(hostname) job ${SLURM_JOB_ID:-none} cores $CORES start $(date -Is)"
  lscpu
  $PY -c "import torch; print('torch', torch.__version__); print(torch.__config__.parallel_info())"
  free -g; uptime
  squeue -w $(hostname -s) -h -o "%i %u %C %j" 2>/dev/null
} > $O/env_start.txt 2>&1
TRAIN_COMMON="--kernel nhmc_resid2 --use-z2 --odd-nets --fixed-pflip 0.5 --n-leapfrog 5 --batch-size 48 --buffer \
 --buffer-chains 64 --buffer-every 2 --buffer-size 20000 --buffer-weight 1.0 --device cpu --kappa-min 0.27 --kappa-max 0.27 \
 --epochs 60 --seed 0 --log-interval 1 --eval-interval 1000000 --ckpt-minutes 100000"
while read -r NAME L T S CK; do
  [ -z "$NAME" ] || [ "${NAME:0:1}" = "#" ] && continue
  M=$X/models/$NAME
  echo "$(date +%T) $NAME L$L T$T s$S $CK"
  for PH in A B; do
    D=$O/train_${NAME}_$PH
    [ "$PH" = A ] && BS=1000000 || BS=1
    if [ -e $D/train_log.csv ]; then echo "  skip train $PH (exists)"; else
      mkdir -p $D
      ( cd $NC && $TS $PY -u train_cond_phi4_kappa_explicit.py $TRAIN_COMMON --L $L --num-steps $T --buffer-start $BS \
          --init-from $M/$CK --save-dir $D > $D/train.log 2>&1 )
      echo "  $(date +%T) train $PH rc $?"
    fi
  done
  J=$O/sample_${NAME}.json
  if [ -e $J ]; then echo "  skip sample (exists)"; else
    $TS $PY -u $X/code/bench_sample.py --run-dir $M --ckpt $CK --kappa 0.27 --s ${S//,/ } --threads 4 --out $J > $O/sample_${NAME}.log 2>&1
    echo "  $(date +%T) sample rc $?"
  fi
done < $CFG
{ echo "end $(date -Is)"; uptime; squeue -w $(hostname -s) -h -o "%i %u %C %j" 2>/dev/null; } > $O/env_end.txt 2>&1
