"""tab:udb_compare (App. G): NHMC vs CMCD vs DBS at L = 8, 11 couplings (record version of
results_sync/revision/figs/table_udb_compare.py with the record paths).

Var(log w)   : NHMC models/phi4_buffer/L8_k*_buf_s0/eval_N262144.json (var_W, 262144 paths); CMCD / DBS
               models/cmcd_torch_L8/<run>/final_eval.json or cluster/HIM/himout/93321/.../final_eval*.json (var_logw, 102400).
ESS_w/N, k^  : results_sync/revision/unify_20261118/item5_ess.json (median [min, max] of 4 batches of 16384 paths; k-hat
               median over the batches).
chain cells  : acceptance / tau_int(|m|) (one decimal), median over 4 chains of max(Sokal, batch means) / 2.
               NHMC: E2_L8_k*_{imhG|imh, rt, rtp12} via figs_20261117u4/u4_common.find (as the figures);
               CMCD / DBS: cluster/HIM/himout/94353/revision_runs/longchain_udb_20261118/ (RT kappa 0.27 / 0.30 and pRT
               kappa 0.27, rerun 2026-11-18) before cluster/HIM/himout/93219/revision_runs/longchain_udb_20261114/.
'$\\ge$' marks chains with criteria_met = false, '$^\\dagger$' RT chains whose command line has no p_flip.
Output: LaTeX rows on stdout and tab_udb_compare.json."""
import glob
import json
import os
import sys

GE = '$\\ge$'
from tab_common import REV, ROOT, cell, first, load

sys.path.insert(0, REV + '/figs_20261117u4')
import u4_common as U  # noqa: E402

H = ROOT + '/cluster/HIM/himout'
UDB = [H + '/94353/revision_runs/longchain_udb_20261118', H + '/93219/revision_runs/longchain_udb_20261114']
KS = ['020', '021', '022', '023', '024', '025', '026', '027', '028', '029', '030']
ESS = {(r['kappa'], r['alg']): r for r in json.load(open(REV + '/unify_20261118/item5_ess.json'))}


def nh(names):
    for n in names:
        f = U.find(n + '.json')
        if f:
            return cell(f)
    return None


def chain_fmt(r):
    if r is None:
        return '--'
    dag = '$^{\\dagger}$' if r['args'].get('method') == 'rt' and r['args'].get('kind') == 'cmcd' and 'p_flip' not in r['args'] else ''
    return f"{r['acc']:.2f} / {'' if r['ok'] else GE}{r['tau']:.1f}{dag}"


out = []
for k in KS:
    kap = '0.' + k[1:]
    e = None
    for nm in (f'L8_k{kap}_buf_s0', f'L8_k{k}_buf_s0'):
        p = f'{ROOT}/models/phi4_buffer/{nm}/eval_N262144.json'
        if os.path.exists(p):
            e = load(p)
            break
    rows = [dict(sampler='NHMC', var=e['var_W'],
                 imh=nh([f'E2_L8_k{k}_imhG', f'E2_L8_k{k}_imh', f'buf_L8_k{k}_imhG']),
                 rt=nh([f'E2_L8_k{k}_rt', f'buf_L8_k{k}_rt']), prt=nh([f'E2_L8_k{k}_rtp12', f'buf_L8_k{k}_rtp12']))]
    for alg in ('cmcd', 'dbs'):
        run = f'{alg}z2_k{kap}_s0_T80_dt0.1_it10000'
        g = sorted(glob.glob(f'{ROOT}/models/cmcd_torch_L8/{run}/final_eval.json')) + \
            sorted(glob.glob(f'{H}/*/revision_runs/udb_eval_L8_20261114/{run}/final_eval*.json'))
        ev = load(g[0])
        rows.append(dict(sampler=alg.upper(), var=ev.get('var_logw'),
                         imh=first([f'{d}/{alg}_k{k}_imh.json' for d in UDB]),
                         rt=first([f'{d}/{alg}_k{k}_rt.json' for d in UDB]),
                         prt=first([f'{d}/{alg}_k{k}_rtp60.json' for d in UDB])))
    for r in rows:
        s = ESS[(kap, r['sampler'])]
        r.update(kappa=kap, ess=(s['kish_med'], s['kish_min'], s['kish_max']), khat=s['khat_med'])
        print(f"{kap if r['sampler'] == 'NHMC' else ''} & {r['sampler']} & {r['var']:.2f} & {s['kish_med']:.3f} "
              f"[{s['kish_min']:.3f}, {s['kish_max']:.3f}] & {s['khat_med']:.2f} & {chain_fmt(r['imh'])} & "
              f"{chain_fmt(r['rt'])} & {chain_fmt(r['prt'])} \\\\")
        out.append({kk: (vv if kk not in ('imh', 'rt', 'prt') or vv is None else {x: vv[x] for x in ('acc', 'tau', 'ok', 'file')})
                    for kk, vv in r.items()})
json.dump(out, open('tab_udb_compare.json', 'w'), indent=1)
