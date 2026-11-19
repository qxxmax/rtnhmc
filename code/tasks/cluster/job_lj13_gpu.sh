#!/bin/bash
#SBATCH -J force -p iqcb -c 4 --gres=gpu:1 --mem=48G -t 2-00:00:00 -o /media/scratch/mqian/logs/lj13gpu-%j.out
# LJ13 single chains on one GPU (tasks_lj13_gpu.txt). Submitted with --dependency=afterany:58558 (same card, after
# the CMCD/DBS training). First stops the CPU LJ13 step processes of job 58555 (same output files; rt / rtp:8 then
# continue their step checkpoints with --prefetch; imh restarts with --imh-pregen). Resubmit to resume.
python_ok=$(/media/scratch/mqian/envs/sps/bin/python -c 'import torch; print(torch.cuda.is_available())')
[ "$python_ok" = "True" ] || { echo "no CUDA"; exit 1; }
pkill -u mqian -f 'longchain.py --kind lj .*--device cpu' && echo "stopped CPU LJ13 processes" || echo "no CPU LJ13 processes"
sleep 20
TF=/media/scratch/mqian/logs/lj13gpu-${SLURM_JOB_ID}.tasks
grep -v "^\s*#" /media/scratch/mqian/revision_code/results_sync/revision/tasks/tasks_lj13_gpu.txt | grep -v "^\s*$" > $TF
seq 1 $(wc -l < $TF) | xargs -P 3 -I{} bash -c "sed -n {}p $TF | bash; echo line {} rc \$?"
