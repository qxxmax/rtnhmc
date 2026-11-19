"""Item 4 (LJ13 W2 with a common reference subsample), 1D parts (energy W2, pair-distance W2) on the Mac;
writes the 2000-config gen/ref subsamples for the geometric W2 (eq_w2) run on HIM. Read-only on the inputs.
Old protocol (lit_metrics.metrics, no weights): rng = Generator(seed=rep); gen idx = randperm(len(gen))[:2000];
ref idx = randperm(len(ref))[:2000] drawn next from the SAME rng -> depends on len(gen), i.e. differs per method.
New protocol: gen idx unchanged (first draw); ref idx = randperm(10000, Generator(seed=rep))[:2000], identical for all methods."""
import os
import sys, json, math, numpy as np, torch
sys.path.insert(0, os.environ.get('NHMC_ROOT', '.') + '/particle'); import lit_metrics as LM
D = os.environ.get('NHMC_ROOT', '.') + '/results_sync/revision/unify_20261118/lj13_data'
REV = os.environ.get('NHMC_ROOT', '.') + '/results_sync/revision'
OLD = {**{f'A_{m}_r{r}': f'{REV}/lj13_noHMC_93492_20261116/lj13_noHMC/A_{m}_r{r}/res/{m}_run0_chain_lit2k.json' for m in ('imh', 'rt') for r in (0, 1, 2)},
       'A_rt_partial8_r0': f'{REV}/lj13_noHMC_93492_20261116/lj13_noHMC/A_rt_partial8_r0/res/rt_partial8_run0_chain_lit2k.json',
       **{f'A_rt_partial8_r{r}': f'{REV}/him_figdata_20261117_extra/himout/93490/revision_runs/lj13_noHMC/A_rt_partial8_r{r}/res/rt_partial8_run0_chain_lit2k.json' for r in (1, 2)}}
n, d, NE = 13, 3, 2000
ref = LM.load_ref(D, 'lj13', 'test')                       # [10000, 39], mean removed
E_ref = LM.energy('lj13', ref).numpy(); PD_ref = LM.pair_dists(ref, n, d).numpy()
common = [torch.randperm(len(ref), generator=torch.Generator().manual_seed(rep))[:NE] for rep in range(3)]
out = dict(protocol=__doc__, runs={}, common_ref_energy_mean=[float(E_ref[c.numpy()].mean()) for c in common],
           full_ref_energy_mean=float(E_ref.mean()))
eqin = {}
rs = np.random.default_rng(20261118); NB = 200
boot_ref = [rs.choice(len(ref), NE, replace=False) for _ in range(NB)]   # 200 ref subsamples for the ref-noise spread
boot_vals = {}
for key, fo in OLD.items():
    z = np.load(f'{D}/{key}.npz'); gen = LM.remove_mean(torch.from_numpy(np.asarray(z['x'], np.float64)), n, d)
    old = json.load(open(fo)); rows = []
    for rep in range(3):
        rng = torch.Generator().manual_seed(rep)
        gi = torch.randperm(len(gen), generator=rng)[:NE]; ri_old = torch.randperm(len(ref), generator=rng)[:NE]
        g = gen[gi]; eg = LM.energy('lj13', g).numpy(); pg = LM.pair_dists(g, n, d).numpy()
        def m1(ri):
            ri = np.asarray(ri); er = E_ref[ri]; pr = PD_ref[ri]
            dsq = LM.w2_1d_sq(pg.ravel(), pr.ravel())
            return dict(energy_w2=math.sqrt(LM.w2_1d_sq(er, eg)), dist_w2=math.sqrt(dsq), energy_mean_ref=float(er.mean()))
        o = m1(ri_old.numpy()); c = m1(common[rep].numpy())
        full = dict(energy_w2=math.sqrt(LM.w2_1d_sq(E_ref, eg)), dist_w2=math.sqrt(LM.w2_1d_sq(pg.ravel(), PD_ref.ravel())))
        ob = old['reps'][rep]
        rows.append(dict(rep=rep, old_json=dict(energy_w2=ob['energy_w2'], dist_w2=ob['dist_w2'], eq_w2=ob['eq_w2'], energy_mean_ref=ob['energy_mean_ref']),
                         old_reproduced=o, common=c, full_ref10000=full, energy_mean_gen=float(eg.mean())))
        eqin[f'{key}_rep{rep}_gen'] = g.numpy(); 
        if rep == 0:  # ref-subsample noise: same gen subsample (rep 0), 200 ref subsamples
            boot_vals[key] = np.array([math.sqrt(LM.w2_1d_sq(E_ref[b], eg)) for b in boot_ref])
    s = lambda k1, k2: float(np.mean([r[k1][k2] for r in rows]))
    out['runs'][key] = dict(reps=rows, energy_w2_old=s('old_json', 'energy_w2'), energy_w2_old_repro=s('old_reproduced', 'energy_w2'),
        energy_w2_common=s('common', 'energy_w2'), energy_w2_fullref=s('full_ref10000', 'energy_w2'),
        dist_w2_old=s('old_json', 'dist_w2'), dist_w2_common=s('common', 'dist_w2'), dist_w2_fullref=s('full_ref10000', 'dist_w2'),
        eq_w2_old=s('old_json', 'eq_w2'), n_gen=int(len(gen)),
        energy_w2_refboot_rep0=dict(mean=float(boot_vals[key].mean()), sd=float(boot_vals[key].std(ddof=1)),
                                    p05=float(np.percentile(boot_vals[key], 5)), p95=float(np.percentile(boot_vals[key], 95))))
    q = out['runs'][key]
    print(f"{key}: E-W2 old {q['energy_w2_old']:.3f} (repro {q['energy_w2_old_repro']:.3f}) common {q['energy_w2_common']:.3f} full {q['energy_w2_fullref']:.3f} | refboot sd {q['energy_w2_refboot_rep0']['sd']:.3f} | d-W2 old {q['dist_w2_old']:.4f} common {q['dist_w2_common']:.4f} full {q['dist_w2_fullref']:.4f}", flush=True)
# paired differences under common ref subsamples (200 boot ref subsamples): path-IMH minus round trip, per seed pairing
pair = {}
for a in ('rt', 'rt_partial8'):
    dif = np.array([boot_vals[f'A_imh_r{r}'] - boot_vals[f'A_{a}_r{r2}'] for r in range(3) for r2 in range(3)])  # [9, 200]
    pair[f'imh_minus_{a}'] = dict(mean=float(dif.mean()), frac_positive=float((dif > 0).mean()),
                                   per_pair_mean=dif.mean(1).tolist(), sd_over_ref=float(dif.std(1, ddof=1).mean()))
out['paired_refboot'] = pair
for i, c in enumerate(common): eqin[f'ref_rep{i}'] = ref[c].numpy()
np.savez(f'{D}/eq_inputs.npz', **eqin)
json.dump(out, open(os.environ.get('NHMC_ROOT', '.') + '/results_sync/revision/unify_20261118/item4_lj13_1d.json', 'w'), indent=1)
print(json.dumps(pair, indent=1)); print('common ref energy means', out['common_ref_energy_mean'], 'full', out['full_ref_energy_mean'])
