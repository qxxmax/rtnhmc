T=/media/scratch/mqian/revision_code/results_sync/revision/tasks/run_Kscan2_20261113.sbatch
cd /media/scratch/mqian
s() { sbatch --parsable -o logs/Kscan2-%j.out $T "$@"; }
# kappa .27: K8/K32/K64 for L12 s0 and K8 for L16 s0 (L8 / L16 K32,K64 s0 already in 62037/62039/62062)
s 12 .27 "K8 K32 K64" 0
s 16 .27 "K8" 0
# second and third training seeds, kappa .27 (L16 K16 s1/s2 exist from 61668)
for sd in 1 2; do s 8 .27 "K8 K16 K32 K64" $sd; s 12 .27 "K8 K16 K32 K64" $sd; s 16 .27 "K8 K32" $sd; s 16 .27 "K64" $sd; done
# other phases, seed 0
s 8 .24 "K32 K64" 0; s 16 .24 "K32 K64" 0; s 8 .30 "K32 K64" 0
for j in 62037 62038 62039 62062; do scontrol update jobid=$j nice=0; done
squeue -u mqian -h -o "%i %T %y %Q %r" | grep -E "^6(20[3-9]|2[1-9])" 
