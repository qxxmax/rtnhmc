#!/usr/bin/env bash
# Redraw all 26 phi^4 / DW / LJ13 / CMCD-DBS figures of the paper from the record and capture their plotted data.
#   NHMC_ROOT=/path/to/extracted/record  bash redraw_all.sh <outdir> [python]
# Output: <outdir>/<figs dir>/pdf/*.pdf and <outdir>/<figs dir>/figdata/<figure>.json (see capture_v10.py);
# the scripts also rewrite their prov/<figure>.json inside $NHMC_ROOT/results_sync/revision/<figs dir>/.
set -uo pipefail
FAIL=0
OUT=$(realpath -m "$1"); PY=${2:-python}
: "${NHMC_ROOT:?set NHMC_ROOT}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=pdf
R=$NHMC_ROOT/results_sync/revision
CAP=$R/zenodo_20261118/capture_v10.py
run() { local d=$1; shift; echo "== $d $*"; "$PY" "$CAP" "$OUT/$d" "$R/$d/$1" "${@:2}" || { echo "FAILED $d $1"; FAIL=1; }; }
run figs_20261117u3 phi4_figs_u3.py mixing mixing_sign observables rawsnis logz transfer transfer_obs acceptance_all
run figs_20261117u3 fig_phi4_main_u3.py
run figs_20261117u3 make_figs_u3.py
run figs_20261117u3 make_stage_scan_u3.py
run figs_20261117u4 phi4_figs_u4.py transfer_matrix convergence acceptance
run figs_20261117u4 fig_scaling_accept_u4.py
run figs_20261117u4 fig_scaling_u4.py
run figs_20261117u4 make_bar_figs_u4.py
run figs_20261117u4 make_figs_u4.py
run figs_20261117u4 fig5_conditional_flow_u4.py
run figs_20261118u5 fig_scaling_tau_u5.py
run figs_20261118u5 make_bar_prefix_u5.py
run figs_20261118u5 dw_lj_figs_u5.py
run figs_20261117u2 phi4_figs_u2.py sscan
run figs_20261117u2 make_figs_u2.py
run figs_20261117u2 make_fig14_acf_u2.py
echo "redraw finished, failures: $FAIL"
exit $FAIL
