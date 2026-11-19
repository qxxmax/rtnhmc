#!/bin/bash
#SBATCH -J force
#SBATCH -p iqcb
#SBATCH -c 12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 1-00:00:00
#SBATCH -o /media/scratch/mqian/revision_runs/logs/gpupack-%j.out
# GPU pack: TASKS = one shell command per line, P at a time on ONE GPU (each line sets --device cuda and a
# memory fraction). Re-submitting skips finished tasks ([ -f json ] guards); longchain/cmcd training resume from checkpoints.
set -uo pipefail
export MPLBACKEND=Agg
/media/scratch/mqian/envs/sps/bin/python -c 'import torch; assert torch.cuda.is_available(); print("GPU", torch.cuda.get_device_name(0), flush=True)' || exit 1
grep -v "^\s*#" "$TASKS" | grep -v "^\s*$" > /media/scratch/mqian/revision_runs/logs/gpupack-${SLURM_JOB_ID}.tasks
export TF=/media/scratch/mqian/revision_runs/logs/gpupack-${SLURM_JOB_ID}.tasks
seq 1 $(wc -l < $TF) | xargs -P ${P:-6} -I{} bash -c "echo \"[\$(date +%T)] start {}\"; sed -n \"{}p\" \$TF | bash; echo \"[\$(date +%T)] done {} rc=\$?\""
echo "END $(date -Is)"
