"""tab:phi4_thermodynamics (App. B): SNIS estimates of the kappa-conditional L = 8 model at 11 couplings.

Input : results_sync/local_20261103/v6_data_phi4/phi4_thermodynamics_L8.csv (E), 8 'run' rows per kappa
        (102400 forward paths each) and one 'mean' row; reference columns from AIS (run_manifest.yaml).
Output: LaTeX rows on stdout and tab_phi4_thermodynamics.json in the working directory.
Table value = mean over the 8 groups, error = sd(ddof=1)/sqrt(8); ESS_w/N in percent; reference = absM_ref +- absM_ref_se."""
import csv
import json

import numpy as np

from tab_common import ROOT

F = ROOT + '/results_sync/local_20261103/v6_data_phi4/phi4_thermodynamics_L8.csv'
rows = list(csv.DictReader(open(F)))
out = []
for k in sorted({float(r['kappa']) for r in rows}):
    R = [r for r in rows if r['row_type'] == 'run' and float(r['kappa']) == k]
    assert len(R) == 8, (k, len(R))

    def ms(key, scale=1.0):
        v = np.array([float(r[key]) for r in R]) * scale
        return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))
    a, m2, e = ms('absM'), ms('M2'), ms('ESS_w_frac', 100)
    ref = (float(R[0]['absM_ref']), float(R[0]['absM_ref_se']))
    out.append(dict(kappa=k, absM=a, absM_ref=ref, M2=m2, ess_pct=e))
    print(f'{k:.2f} & ${a[0]:.5f} \\pm {a[1]:.5f}$ & ${ref[0]:.5f} \\pm {ref[1]:.5f}$ & ${m2[0]:.5f} \\pm {m2[1]:.5f}$ & '
          f'${e[0]:.2f} \\pm {e[1]:.2f}$ \\\\')
json.dump(out, open('tab_phi4_thermodynamics.json', 'w'), indent=1)
