#!/bin/bash
#SBATCH -J force
#SBATCH -c 2
#SBATCH -t 3:00:00
#SBATCH --nice=800
#SBATCH -o /media/scratch/mqian/logs/diag-%j.out
export OMP_NUM_THREADS=1
cd /media/scratch/mqian/revision_code/revision/longchain
O=/media/scratch/mqian/revision_runs/longchain_min
for p in 'E2_*_traces.npz' 'buf_*_traces.npz' 'T16_*_traces.npz' 'E2tr16_*_traces.npz'; do
  /media/scratch/mqian/envs/sps/bin/python chain_diagnostics.py --dir $O --pattern "$p" --out $O/diag_$(echo $p | cut -d_ -f1).json &
done; wait
