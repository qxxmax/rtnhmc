"""tab:udb_refine (App. G): CMCD / DBS full (RT) and truncated (pRT, last quarter) round trips after inference-time step
refinement at L = 8; each cell acceptance / tau_int(|m|) (median over 4 chains of max(Sokal, batch means) / 2).

trained 80 : C4 cluster/HIM/himout/94353/revision_runs/longchain_udb_20261118/ (cells rerun on 2026-11-18) if present,
             else cluster/HIM/himout/93219/revision_runs/longchain_udb_20261114/  ({alg}_k{k}_rt.json, _rtp60.json)
refined xn : C4 cluster/HIM/himout/93377/revision_runs/longchain_udb_refined_20261114/{alg}_k{k}_x{n}h_rt.json and
             _rtp{60 n}.json (task file results_sync/revision/tasks/longchain_refined.py)
As in the paper table, lower bounds are not marked here; criteria_met of every chain is kept in tab_udb_refine.json
(all refined and trained pRT chains at kappa = 0.30, and some at 0.27, do not meet the criterion).  Output: LaTeX rows."""
import json

from tab_common import ROOT, first

H = ROOT + '/cluster/HIM/himout'
NEW = H + '/94353/revision_runs/longchain_udb_20261118'
OLD = H + '/93219/revision_runs/longchain_udb_20261114'
REF = H + '/93377/revision_runs/longchain_udb_refined_20261114'


def fmt(r):
    return '--' if r is None else f"{r['acc']:.2f} / {r['tau']:.1f}"


out = []
for k in ('024', '027', '030'):
    for alg in ('cmcd', 'dbs'):
        for m, d in (('RT', 'rt'), ('pRT', 'rtp')):
            row = [first([f'{NEW}/{alg}_k{k}_{d}{"60" if d == "rtp" else ""}.json', f'{OLD}/{alg}_k{k}_{d}{"60" if d == "rtp" else ""}.json'])]
            for n in (2, 4, 8):
                row.append(first([f'{REF}/{alg}_k{k}_x{n}h_{d}{60 * n if d == "rtp" else ""}.json']))
            out.append(dict(kappa='0.' + k[1:], sampler=f'{alg.upper()} {m}', cells=row))
            kc = '0.' + k[1:] if (alg, m) == ('cmcd', 'RT') else ''
            print(f"{kc} & {alg.upper()} {m} & " + ' & '.join(fmt(r) for r in row) + ' \\\\')
json.dump(out, open('tab_udb_refine.json', 'w'), indent=1)
