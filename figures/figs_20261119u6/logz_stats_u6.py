"""Numbers quoted in Sec. VI for the square-lattice log Z figure (same data as fig_logz_u6.py)."""
import json, math, sys
sys.argv = ['x']
import fig_logz_u6 as F   # re-draws the two figures (idempotent)
P = F.P
import ti_logz as T
D = T.load(); A = T.ais_refs()
R = {}
for L in P.LS:
    ti = F.ti_series(D, L)
    r = {}
    for key in ('snis', 'IMH-rev', 'RT-rev', 'pRT-rev', 'IMH-bar', 'RT-bar', 'pRT-bar'):
        d, z = [], []
        for k in P.KAPPAS[L]:
            v = P.chain_logz(f'E2_L{L}_k0{k}').get(key)
            if v is None or P.kmil(k) not in ti:
                continue
            t = ti[P.kmil(k)]
            d.append(v[0] - t[0]); z.append((v[0] - t[0]) / math.hypot(v[1], t[1]))
        r[key] = dict(max_abs=max(map(abs, d)), max_abs_z=max(map(abs, z)), n=len(d), n_z_gt2=sum(abs(q) > 2 for q in z))
    da = [A[(L, round(k / 1000, 3))][0] - ti[k][0] for k in ti if (L, round(k / 1000, 3)) in A]
    r['AIS'] = dict(max_abs=max(map(abs, da)))
    R[L] = r
json.dump(R, open('logz_stats_u6.json', 'w'), indent=1)
for L, r in R.items():
    print(L, {k: (round(v['max_abs'], 4), round(v.get('max_abs_z', 0), 2), v.get('n_z_gt2'), v.get('n')) for k, v in r.items()})
