#!/bin/bash
#SBATCH -J force -p iqcb -c 16 --gres=gpu:1 --mem=96G -t 2-00:00:00
#SBATCH -o /media/scratch/mqian/revision_runs/logs/gpupack-%j.out
# One GPU card, run together: tasks_lj13_gpu.txt (3 LJ13 lines, started first) + tasks_minimal_GPU_cluster.txt
# (16 CMCD/DBS training + imhG lines, unchanged). 9 at a time (3 LJ13 + 6 training slots), 16 CPU.
# Stops the CPU LJ13 step processes of 58555 first (same output files; rt / rtp:8 continue their step checkpoints).
# Re-submitting skips finished tasks ([ -f json ] guards) and resumes from checkpoints.
set -uo pipefail
export MPLBACKEND=Agg
/media/scratch/mqian/envs/sps/bin/python -c 'import torch; assert torch.cuda.is_available(); print("GPU", torch.cuda.get_device_name(0), flush=True)' || exit 1
pkill -u mqian -f 'longchain.py --kind lj .*--device cpu' && echo "stopped CPU LJ13 processes" || echo "no CPU LJ13 processes"
sleep 20
D=/media/scratch/mqian/revision_code/results_sync/revision/tasks
export TF=/media/scratch/mqian/revision_runs/logs/gpupack-${SLURM_JOB_ID}.tasks
cat $D/tasks_lj13_gpu.txt $D/tasks_minimal_GPU_cluster.txt | grep -v "^\s*#" | grep -v "^\s*$" > $TF
seq 1 $(wc -l < $TF) | xargs -P 9 -I{} bash -c "echo \"[\$(date +%T)] start {}\"; sed -n \"{}p\" \$TF | bash; echo \"[\$(date +%T)] done {} rc=\$?\""
echo "END $(date -Is)"
