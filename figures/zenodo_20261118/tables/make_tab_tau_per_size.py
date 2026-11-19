"""tab:tau_per_size (App. D): tau_int(|m|) / tau_int(sign m) of HMC, Wolff and the three path updates, L = 4, 8, 12, 16.

HMC / Wolff, |m|  : results_sync/revision/figs/data/wh_tau_grid.json (1e6 records), tau_absm / 2.
HMC / Wolff, sign : results_sync/revision/figs/data/wh_sign_summary.json (2e5 records), tau_sign / 2; 'no flip' if the
                    record has no sign change.
NHMC              : chain files chosen as in the figure scripts (figs_20261117u4/u4_common.find: E2_* before buf_*,
                    path-IMH imhG > imh), median over the four chains of max(Sokal, batch means) / 2.
Output: LaTeX rows on stdout and tab_tau_per_size.json in the working directory."""
import json
import sys

from tab_common import REV, cell, g2

sys.path.insert(0, REV + '/figs_20261117u4')
import u4_common as U  # noqa: E402

G = json.load(open(REV + '/figs/data/wh_tau_grid.json'))
SG = json.load(open(REV + '/figs/data/wh_sign_summary.json'))
LS = (4, 8, 12, 16)


def ref(alg, L, k):
    a = [v for v in G.values() if v['alg'] == alg and v['L'] == L and round(v['kappa'] * 1000) == k * 10]
    s = [v for v in SG.values() if v['alg'] == alg and v['L'] == L and round(v['kappa'] * 1000) == k * 10]
    ta = a[0]['tau_absm'] / 2
    if not s or not s[0].get('sign_changes'):
        return ta, None
    return ta, s[0]['tau_sign'] / 2


def nhmc(L, k, m):
    c = ([f'E2_L{L}_k0{k}_imhG', f'E2_L{L}_k0{k}_imh', f'buf_L{L}_k0{k}_imhG', f'buf_L{L}_k0{k}_imh'] if m == 'imh'
         else [f'E2_L{L}_k0{k}_{m}', f'buf_L{L}_k0{k}_{m}'])
    for n in c:
        f = U.find(n + '.json')
        if f:
            return cell(f)
    return None


out = []
for k in (24, 27, 30):
    for lab, kind in (('HMC', 'hmc'), ('Wolff', 'wolff'), ('RT', 'rt'), ('pRT', 'rtp12'), ('path-IMH', 'imh')):
        cells = []
        for L in LS:
            if kind in ('hmc', 'wolff'):
                ta, ts = ref(kind, L, k)
                cells.append(dict(L=L, tau=ta, tau_sign=ts))
            else:
                r = nhmc(L, k, kind)
                cells.append(dict(L=L, tau=r['tau'], tau_sign=r['tau_sign'], file=r['file'], criteria_met=r['ok']))
        out.append(dict(kappa=f'0.{k}', update=lab, cells=cells))
        txt = ' & '.join(f"{g2(c['tau'])} / {g2(c['tau_sign']) if c['tau_sign'] is not None else 'no flip'}" for c in cells)
        print(f"{'0.' + str(k) if lab == 'HMC' else ''} & {lab} & {txt} \\\\")
json.dump(out, open('tab_tau_per_size.json', 'w'), indent=1)
