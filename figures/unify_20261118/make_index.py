"""Collect the replacement values of items 1-6 into unify_values.json (for tex / figure edits)."""
import os
import json
H = os.environ.get('NHMC_ROOT', '.') + '/results_sync/revision/unify_20261118/'
i12 = json.load(open(H + 'item12_bar.json')); i3 = json.load(open(H + 'item3_fig14.json')); i4 = json.load(open(H + 'item4_lj13_summary.json'))
i5 = json.load(open(H + 'item5_ess.json')); i6 = json.load(open(H + 'item6_ref_tau.json'))
V = {}
V['1_tab_bar_transfer_tau'] = {k: dict(old=round(v['old_tau_sokal16_half'], 2), new_16chain_median=round(v['unified16_median'], 2),
    seed_median_range=[round(x, 2) for x in v['seed_range']], bar_change_1e4=round(i12['item1_bar_sensitivity'][k]['bar_maxSBM'][0] - i12['item1_bar_sensitivity'][k]['bar_sokal'][0], 4))
    for k, v in i12['item1'].items()}
V['2_fig_bar_prefix_B20'] = {k: [dict(prefix=r['prefix'], bar=round(r['bar20'], 3), err_B20=round(r['err20'], 3), err_B10_old=round(r['err10'], 3)) for r in rows] for k, rows in i12['item2'].items()}
V['3_fig14'] = {k: dict(text=v['text'], n=v['n_transitions'], unified_full=round(v['full']['unified_median'], 2), unified_full_jk=round(v['full']['unified_jk'], 2),
    unified_1e4=round(v['trunc1e4']['unified_median'], 2), unified_1e4_jk=round(v['trunc1e4']['unified_jk'], 2), fig_pooled_full=round(v['full']['fig_pooled_tau'], 2),
    fig_pooled_1e4=round(v['trunc1e4']['fig_pooled_tau'], 2), acc_1e4=round(v['trunc1e4']['acc'], 3)) for k, v in i3.items()}
V['4_lj13'] = dict(per_method={m: {f: [round(x, 4) for x in r] if isinstance(r, list) else round(r, 4) for f, r in v.items()} for m, v in i4['per_method'].items()},
    gap_common=i4['gap_common'], floor_common_ref=i4['floor']['val2000_vs_common_ref_mean3'], floor_fullref=i4['floor']['val2000_vs_fullref_energy'],
    floor20_old=i4['floor']['floor20_old']['energy_w2'], common_ref_energy_mean=i4['common_ref_energy_mean'])
V['5_tab_udb_compare_ess'] = {f"{r['kappa']}_{r['alg']}": dict(old=round(r['old_table_ess'], 3), old_N=r['old_N'],
    kish_16384=f"{r['kish_med']:.3f} [{r['kish_min']:.3f}, {r['kish_max']:.3f}]", psis_ess_16384=f"{r['psis_med']:.3f} [{r['psis_min']:.3f}, {r['psis_max']:.3f}]",
    khat=f"{r['khat_med']:.2f} (max {r['khat_max']:.2f})") for r in i5}
V['6_ref_tau_errors'] = dict(with_traces={k: dict(tau=round(v['tau_sokal'], 3), ms=round(v['ms_err'], 3), jk16=round(v['jk16_err_sokal'], 3)) for k, v in i6['with_traces'].items()},
    jk_over_ms=i6['jk_over_ms_summary'], n_without_traces=len(i6['without_traces']))
json.dump(V, open(H + 'unify_values.json', 'w'), indent=1, ensure_ascii=False)
for k, v in V['5_tab_udb_compare_ess'].items(): print(k, v)
