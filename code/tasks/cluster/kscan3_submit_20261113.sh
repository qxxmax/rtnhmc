# 2026-11-13: split the over-packed L16 jobs (62037 / 62038 / 62074 held: K32+K64+lf10 on one 48 GB GPU would OOM),
# K64 alone per GPU; every job containing K64 gets one afterany follow-up (training resumes exactly from last.pt).
T=/media/scratch/mqian/revision_code/results_sync/revision/tasks/run_Kscan3_20261113.sbatch
cd /media/scratch/mqian
s() { sbatch --parsable -o logs/Kscan3-%j.out "$@"; }
f() { s --dependency=afterany:$1 $T "${@:2}" >/dev/null; }
a=$(s $T 16 .27 "K32 lf10" 0); b=$(s $T 16 .27 "K64" 0); f $b 16 .27 "K64" 0
c=$(s $T 16 .30 "K32 lf10" 0); d=$(s $T 16 .30 "K64" 0); f $d 16 .30 "K64" 0
e=$(s $T 16 .24 "K32" 0); g=$(s $T 16 .24 "K64" 0); f $g 16 .24 "K64" 0
h=$(s $T 8 .27 "K4" 0); h2=$(s $T 8 .27 "K4" 1); h3=$(s $T 8 .27 "K4" 2)
# follow-ups for the already queued K64 jobs (Kscan2 args)
f 62063 12 .27 "K8 K32 K64" 0; f 62066 12 .27 "K8 K16 K32 K64" 1; f 62070 12 .27 "K8 K16 K32 K64" 2
f 62068 16 .27 "K64" 1; f 62072 16 .27 "K64" 2; f 62065 8 .27 "K8 K16 K32 K64" 1; f 62069 8 .27 "K8 K16 K32 K64" 2
f 62039 8 .27 "K32 K64 lf10" 0; f 62073 8 .24 "K32 K64" 0; f 62075 8 .30 "K32 K64" 0
echo $a $b $c $d $e $g $h $h2 $h3
squeue -u mqian -h -o "%i %T %r" | awk '$1>=62037 && $1<62200' | sort -n | awk '{print $1, $3}' | tr '\n' ' '
