#!/bin/bash
# HIMSTER3 (CPU) submission for the jackknife items that need fresh forward paths (2026-11-17). NOT RUN; review first.
# Run on the HIM login node from $B=/lustre/him/him-mam/mqian after copying this folder to $B/jk_20261117/:
#   scp -o ControlPath=/tmp/cm-%r@%h -r .../jackknife_20261117/cluster himster3:/lustre/him/him-mam/mqian/jk_20261117
# Job name force, nice 1000 (yield to others), STOP_S for multi-hour jobs (him_stage2.sh default stops lines at 11.5 h).
set -e
B=/lustre/him/him-mam/mqian
C=$B/jk_20261117
cd $C
# EXTRA tar: the two new scripts under jk/, plus the K-scan model dirs (kernel_final.pt + config.json) under revision_runs/kscan_cpu/
rm -rf x && mkdir -p x/jk x/revision_runs/kscan_cpu
cp wtail_var2_jk.py eval_W_dump.py jk_cells_remote.py x/jk/
for d in $B/himout/*/revision_runs/kscan_cpu/*/; do
  n=$(basename $d); [ -f $d/kernel_final.pt ] && [ ! -d x/revision_runs/kscan_cpu/$n ] && cp -r $d x/revision_runs/kscan_cpu/$n
done
ls x/revision_runs/kscan_cpu | wc -l          # expect 35; lines whose model is missing exit 3 (no output)
tar cf extra_jk.tar -C x jk revision_runs
# (1) sigma_s^2 with pivot-group jackknife: 7 models x 4 threads (like wtail2 job 93289)
STOP_S=167400 nohup salloc -J force --nice=1000 -p experiment -A himster3exp -N1 -c 28 --mem=40G -t 1-00:00:00 \
  bash $B/revision_code/results_sync/revision/tasks/him_stage2.sh $C/lines_wtail_jk_him.txt 7 $C/extra_jk.tar > $B/logs/jk_wtail.out 2>&1 &
# (2) per-path W for the K-scan / stage-scan / buffer models: 39 lines x 4 threads, 16 in parallel
STOP_S=167400 nohup salloc -J force --nice=1000 -p experiment -A himster3exp -N1 -c 64 --mem=80G -t 2-00:00:00 \
  bash $B/revision_code/results_sync/revision/tasks/him_stage2.sh $C/lines_evalW_him.txt 16 $C/extra_jk.tar > $B/logs/jk_evalW.out 2>&1 &
# (3) light, login node or a 1-core salloc: per-file jackknife where the traces live on HIM (E2 L6/L10, T16, pilot_bigL)
#   srun -J force -p experiment -A himster3exp -c 1 --mem=8G -t 1:00:00 \
#     $B/envs/sps/bin/python $C/jk_cells_remote.py $C/cells_him.json $B/himout/*/revision_runs/longchain_min $B/himout/*/revision_runs/pilot_bigL
# then copy back: cells_him.json -> jackknife_20261117/, outputs of (1)/(2) appear in $B/himout/<job>/revision_runs/{wtail2_jk_20261117,evalW_jk_20261117}
#
# iqcb (GPU not needed; CPU is fine) for tab:udb_compare long chains (rtp:60) and the NHMC L8 eval rows:
#   cd /media/scratch/mqian && python jk_cells_remote.py cells_udb.json /media/scratch/mqian/revision_runs/longchain \
#       /media/scratch/mqian/figs_20261114/him/93219/revision_runs/longchain_udb_20261114
