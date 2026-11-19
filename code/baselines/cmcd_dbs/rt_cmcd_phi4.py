#!/usr/bin/env python3
"""Path-space MCMC on a trained (frozen) CMCD / DBS phi^4 sampler (cmcd_phi4.py port of the official UDB code):
SNIS, path independence MH (IMH), round trip (RT) and partial RT, with the observables / statistics of
nhmc-phi4-bench/revision/phi4/phi4_compare.py, so that the numbers are directly comparable with the NHMC runs.

Path measure of the sampler (k = 0..T-1, kernels F_k, B_k as in cmcd_phi4.py):
  Q(tau) = pi_0(x_0) prod_k F_k(x_{k+1} | x_k),   P(tau) = gamma(x_T) prod_k B_k(x_k | x_{k+1}),  gamma = e^{-S}.
P / Z has x_T-marginal pi = gamma / Z and conditional law of (x_{T-1}, ..., x_0) given x_T = the backward chain B.
Define, for a stop stage s (0 <= s <= T-1),
  W_s(tau) = -log gamma(x_T) + sum_{k >= s} [log F_k(x_{k+1} | x_k) - log B_k(x_k | x_{k+1})].
Methods (one transition, state x ~ pi):
  imh          keep the current path; propose a fresh path from Q; accept min(1, w(y) / w(x)),
               log w = -W_0 - log pi_0(x_0). The current path of the exact start is drawn from B given x.
  rt           draw x_{T-1..0} ~ B from x (u = x_0), propose x'_{1..T} ~ F from u, accept min(1, exp(W_0(x) - W_0(y))).
  rt_partial:s same with the pivot u = x_s (pull-back depth T - s, 1 <= s <= T-1): only stages s..T-1 are redrawn.
               Cost 2 (T - s) / T paths per transition.
Exactness: target pi(x) prod_{k>=s} B_k on (x_s..x_T); the proposal keeps x_s and redraws x_{s+1..T} from F, the
ratio is exp(W_s(x) - W_s(y)) (Neal's tempered-transition / path MH construction; u terms cancel).
Optional --z2flip appends the exact global flip phi -> -phi (prob 1/2) after every transition (off by default,
because sector mixing through the learned paths is part of what is measured).

Statistics (revision/chainstats.py conventions): per-transition tau_int = 1 + 2 sum rho with the Sokal window
c = 5 on the chain-averaged ACF (per-chain centred) for |M|, M^2, S/V and sign(M); for sign(M) additionally
tau_global (pooled-mean centred ACF, catches frozen sectors); acceptance; holding-time statistics (max holding);
z-scores vs the HMC+Z2 reference (results_sync/revision/phi4/hmc_k0XX.json): z = (mean - ref) / sqrt(se^2 + se_ref^2),
se = sqrt(Var_ref / n_eff), se_ref = sd_runs / sqrt(n_runs) of the reference.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import cmcd_phi4 as C  # noqa: E402

sys.path.insert(0, C.CLONE)
import chainstats as cs  # noqa: E402

KEYS = ('absM', 'M2', 'action', 'sign')


def tau_global(trace, c=5.0):
    """improve/phi4_obs/collect_and_plot.tau_global: tau_int with the pooled mean subtracted."""
    x = np.asarray(trace, dtype=np.float64)
    x = x - x.mean()
    n = x.shape[0]
    f = np.fft.rfft(x, 2 * n, axis=0)
    ac = np.fft.irfft(f * np.conj(f), axis=0)[:n].mean(1)
    if ac[0] <= 0:
        return 1.0, 0, True
    rho = ac / ac[0]
    tau = 1.0
    for m in range(1, n):
        tau = 1.0 + 2.0 * rho[1:m + 1].sum()
        if m >= c * tau:
            return float(tau), m, True
    return float(tau), n - 1, False


def observables(x, m, with_m4=False):
    M = x.mean(-1).double()
    o = {'absM': M.abs(), 'M2': M ** 2, 'action': (-m.log_gamma(x)).double() / m.dim, 'sign': torch.sign(M)}
    if with_m4:
        o['M4'] = M ** 4
    return o


@torch.no_grad()
def hmc_bank(m, n_chains, n_traj, eps, n_lf, seed, device, dtype, flip=False):
    """HMC start bank (as rt_imh_chains.hmc_bank: chains started with alternating sign, n_traj trajectories)."""
    g = torch.Generator().manual_seed(seed)
    sign = torch.where(torch.arange(n_chains) % 2 == 0, 1.0, -1.0)[:, None]
    x = (sign + 0.3 * torch.randn(n_chains, m.dim, generator=g)).to(device=device, dtype=dtype)
    S = lambda z: -m.log_gamma(z)
    F = lambda z: m.score_gamma(z)
    for _ in range(n_traj):
        p = torch.randn_like(x)
        h0 = S(x) + 0.5 * (p ** 2).sum(-1)
        q, p = x.clone(), p + 0.5 * eps * F(x)
        for i in range(n_lf):
            q = q + eps * p
            if i < n_lf - 1:
                p = p + eps * F(q)
        p = p + 0.5 * eps * F(q)
        h1 = S(q) + 0.5 * (p ** 2).sum(-1)
        a = torch.log(torch.rand_like(h0)) < torch.nan_to_num(h0 - h1, nan=-math.inf)
        x = torch.where(a[:, None], q, x)
        if flip:
            f = torch.rand(n_chains, device=device) < 0.5
            x = torch.where(f[:, None], -x, x)
    return x


@torch.no_grad()
def _save_atomic(path, d):
    tmp = str(path) + '.tmp'
    torch.save(d, tmp)
    os.replace(tmp, path)


def run_chain(m, method, s_stop, x0, n_trans, seed, z2flip=False, bug=None, with_m4=False, ckpt=None, ckpt_minutes=20.0,
              return_W=False):
    """One block of parallel chains. method in {'imh', 'rt', 'rtp'} (rtp = partial RT with stop stage s_stop).
    ckpt: chain-level checkpoint file (periodic save; resume without reseeding, bitwise).
    return_W (repo addition 2026-11-04, default off = unchanged): also return the carried IMH record W (None otherwise)."""
    c = m.coeffs()
    T = m.T
    keys = KEYS + (('M4',) if with_m4 else ())
    W, it0, sec0 = None, 0, 0.0
    if ckpt is not None and os.path.exists(ckpt):
        z = torch.load(ckpt, map_location='cpu', weights_only=False)
        assert z['n_trans'] == n_trans and z['seed'] == seed and z['method'] == method, 'checkpoint mismatch'
        it0, x, tr, acc, sec0 = z['it'], z['x'].to(x0.device), z['tr'], z['acc'], z['sec']
        W = z['W'].to(x0.device) if z['W'] is not None else None
        torch.set_rng_state(z['rng_cpu'])
        if x0.is_cuda and z['rng_cuda'] is not None:
            torch.cuda.set_rng_state(z['rng_cuda'])
        B = x.shape[0]
        print(f'run_chain: resumed {ckpt} at {it0}/{n_trans}', flush=True)
    else:
        torch.manual_seed(seed)
        x = x0.clone()
        B = x.shape[0]
        if method == 'imh':
            u, W = m.reverse_to(x, 0, c, bug=bug)
            W = W + m.prior_logpdf(u, c)                    # W_imh = -log w of the current (stationary) path
        tr = {k: np.empty((n_trans, B)) for k in keys}
        acc = np.zeros((n_trans, B), dtype=bool)
    kill_at = int(os.environ.get('RT_CMCD_KILL_AT', '-1'))
    t0 = time.time()
    t_ck = time.time()
    for it in range(it0, n_trans):
        if ckpt is not None and (it == kill_at or time.time() - t_ck > 60 * ckpt_minutes):
            _save_atomic(ckpt, dict(it=it, n_trans=n_trans, seed=seed, method=method, x=x.cpu(),
                                    W=(W.cpu() if W is not None else None), tr=tr, acc=acc,
                                    sec=sec0 + time.time() - t0, rng_cpu=torch.get_rng_state(),
                                    rng_cuda=(torch.cuda.get_rng_state() if x.is_cuda else None)))
            t_ck = time.time()
            if it == kill_at:
                print(f'RT_CMCD_KILL_AT: checkpoint at {it}, exiting', flush=True)
                raise SystemExit(3)
        if method == 'imh':
            u = m.prior_sample(B, c, x.device, x.dtype)
            y, Wy = m.forward_from(u, 0, c)
            Wy = Wy + m.prior_logpdf(u, c)
            Wx = W
        else:
            s = 0 if method == 'rt' else s_stop
            u, Wx = m.reverse_to(x, s, c, bug=bug)
            y, Wy = m.forward_from(u, s, c)
        la = torch.nan_to_num((Wx - Wy).double(), nan=-math.inf)
        a = torch.log(torch.rand_like(la)) < la
        x = torch.where(a[:, None], y, x)
        if method == 'imh':
            W = torch.where(a, Wy, W)
        if z2flip:
            f = torch.rand(B, device=x.device) < 0.5
            x = torch.where(f[:, None], -x, x)
            if method == 'imh' and not m.z2 and bug != 'imh_flip_keepW':   # bug: negative control of the tests
                # the IMH state is a path and w(-tau) != w(tau) for a non-Z2 model: redraw the stationary record
                # of the flipped chains from B given x (exact Gibbs refresh; one extra path for those chains)
                u2, W2 = m.reverse_to(x, 0, c)
                W = torch.where(f, W2 + m.prior_logpdf(u2, c), W)
        acc[it] = a.cpu().numpy()
        for k, v in observables(x, m, with_m4).items():
            tr[k][it] = v.cpu().numpy()
    if x.is_cuda:
        torch.cuda.synchronize()
    if return_W:
        return tr, acc, sec0 + time.time() - t0, x, (W if method == 'imh' else None)
    return tr, acc, sec0 + time.time() - t0, x


@torch.no_grad()
def run_snis(m, n_paths, seed, chunk=4096):
    torch.manual_seed(seed)
    c = m.coeffs()
    obs = {k: [] for k in KEYS}
    lws, t0 = [], time.time()
    dev = c['eta'].device
    for i in range(0, n_paths, chunk):
        b = min(chunk, n_paths - i)
        u = m.prior_sample(b, c, dev, c['eta'].dtype)
        x, W = m.forward_from(u, 0, c)
        lws.append((-(W + m.prior_logpdf(u, c))).double().cpu())
        for k, v in observables(x, m).items():
            obs[k].append(v.cpu())
    lw = torch.cat(lws).numpy()
    out = dict(n_paths=n_paths, kish_ess=cs.kish_ess(lw), seconds=time.time() - t0,
               n_nonfinite=int((~np.isfinite(lw)).sum()), var_logw=float(np.var(lw[np.isfinite(lw)])))
    out['kish_ess_frac'] = out['kish_ess'] / n_paths
    mx = lw[np.isfinite(lw)].max()
    out['logZ'] = float(mx + np.log(np.mean(np.exp(np.where(np.isfinite(lw), lw, -np.inf) - mx))))
    for k in KEYS:
        out[k] = dict(mean=cs.snis_mean(torch.cat(obs[k]).numpy(), lw))
    return out


def summarise(tr, acc, burn, ref, ref_se, cost):
    out = dict(acceptance=float(acc[burn:].mean()), holding=cs.holding_times(acc[burn:]))
    for k in KEYS:
        mm, v = ref.get(k, (None, None)) if ref else (None, None)
        o = cs.observable_summary(tr[k][burn:], exact_mean=mm, exact_var=v, cost_per_transition=cost)
        if mm is not None:
            se_r = ref_se.get(k, 0.0) if ref_se else 0.0
            o['z_with_ref_se'] = o['error'] / math.sqrt(o['se'] ** 2 + se_r ** 2)
        out[k] = o
    tg, win, okw = tau_global(tr['sign'][burn:])
    if ref and 'sign' in ref:                 # sign z with the pooled-mean tau (frozen sectors)
        xs = np.asarray(tr['sign'][burn:])
        se_g = math.sqrt(ref['sign'][1] * tg / xs.size)
        se_r = ref_se.get('sign', 0.0) if ref_se else 0.0
        out['sign']['se_global'] = se_g
        out['sign']['z_global_with_ref_se'] = out['sign']['error'] / math.sqrt(se_g ** 2 + se_r ** 2)
    out['sign']['tau_global'] = tg
    out['sign']['tau_global_window'] = win
    out['sign']['tau_global_converged'] = okw
    # multi-chain diagnostics (gpu_E/diag.py, coordinator 2026-11-04): split-R-hat, n_post/tau, between-chain
    # variance vs tau-predicted, max holding / n_post, flags
    sys.path.insert(0, C.CLONE)      # code/diag.py
    import diag
    out['diagnostics'] = {}
    flags = set()
    for k in ('absM', 'sign'):
        t = max(out[k]['tau_int'], out[k].get('tau_global', 0.0) if k == 'sign' else 0.0)
        d = diag.chain_diagnostics(np.asarray(tr[k][burn:]), t, acc[burn:] if k == 'absM' else None)
        out['diagnostics'][k] = d
        flags.update(d['flags'])
    out['flags'] = sorted(flags)
    s = tr['sign'][burn:]
    out['sector_change_rate'] = float((s[1:] != s[:-1]).mean())
    sc = (s[1:] != s[:-1]).any(0)
    out['frac_chains_with_sign_change'] = float(sc.mean())
    return out


SEED_RULE = ('sha256("{seed}|{alg}|{kappa:.6f}|{model_seed}|{run}|{method}")[:8] as int; chain stream = that + 1 '
             '(gpu_E seed fix 2026-11-04: no two models / kappas / methods / runs share random streams)')


def chain_seed(base, alg, kappa, model_seed, run, method):
    import hashlib
    h = hashlib.sha256(f'{base}|{alg}|{kappa:.6f}|{model_seed}|{run}|{method}'.encode()).hexdigest()
    return int(h[:8], 16)


def load_model(a, device, dtype):
    if a.load_jax:
        m = C.UDBPhi4(a.alg, a.L, a.kappa, lam=a.lam, T=a.T, z2=a.jax_z2)
        m.load_flax(C.load_msgpack(a.load_jax))
        info = dict(source=a.load_jax)
    else:
        st = torch.load(a.model, map_location='cpu', weights_only=False)
        ar = st['args']
        if abs(ar['kappa'] - a.kappa) > 1e-12 or ar['L'] != a.L:
            print(f"note: model trained at kappa={ar['kappa']} L={ar['L']}, run at kappa={a.kappa} L={a.L}", flush=True)
        m = C.UDBPhi4(ar['alg'], a.L, a.kappa, lam=ar['lam'], T=ar['T'], dt=ar['dt'], z2=ar['z2'], num_hid=ar['num_hid'])
        m.load_state_dict(st['model'])
        info = dict(source=a.model, train_args=ar, train_sec=st.get('train_sec'))
    m = m.to(device=device, dtype=dtype).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default=None, help='model.pt or ckpt.pt from cmcd_phi4.py')
    ap.add_argument('--load-jax', default=None, help='udb_bench params.msgpack instead (non-Z2 main_L8 runs)')
    ap.add_argument('--alg', default='cmcd', help='with --load-jax only')
    ap.add_argument('--jax-z2', action='store_true', help='with --load-jax only: params of a +Z2 run (default non-Z2)')
    ap.add_argument('--kappa', type=float, required=True)
    ap.add_argument('--lam', type=float, default=0.022)
    ap.add_argument('--L', type=int, default=8)
    ap.add_argument('--T', type=int, default=80, help='with --load-jax only')
    ap.add_argument('--methods', nargs='+', default=['snis', 'imh', 'rt', 'rt_partial:40', 'rt_partial:72'])
    ap.add_argument('--n-chains', type=int, default=256)
    ap.add_argument('--path-budget', type=int, default=500_000)
    ap.add_argument('--n-runs', type=int, default=3)
    ap.add_argument('--burn-fraction', type=float, default=0.1)
    ap.add_argument('--window-ratio', type=float, default=0.0, help='extend chains until Sokal window < n_post/ratio')
    ap.add_argument('--max-paths', type=float, default=2 ** 22, help='path cap per run when extending')
    ap.add_argument('--ckpt-minutes', type=float, default=20.0, help='chain-level checkpoint interval')
    ap.add_argument('--init', choices=['bank', 'proposal'], default='proposal')  # user rule 2026-11-04: no HMC starts
    ap.add_argument('--bank-traj', type=int, default=1000)
    ap.add_argument('--bank-eps', type=float, default=0.1)
    ap.add_argument('--bank-lf', type=int, default=10)
    ap.add_argument('--z2flip', action='store_true')
    ap.add_argument('--ref', default='auto', help="HMC+Z2 reference json ('auto': results_sync/revision/phi4/hmc_kXXX.json)")
    ap.add_argument('--dtype', default='float32')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--seed', type=int, default=20261103)
    ap.add_argument('--save-final', action='store_true',
                    help='repo addition: save the final chain states and the carried IMH records per method/run to '
                         '<out stem>_final_<meth>_run<r>.pt (input of alpha_cmcd.py); use it from the first start '
                         'of an --out (runs finished before it was given have no final-state file)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device(a.device)
    dtype = torch.float64 if a.dtype == 'float64' else torch.float32
    m, info = load_model(a, dev, dtype)
    T = m.T
    ref = ref_se = None
    rp = a.ref
    if rp == 'auto':
        rp = os.path.join(C.RESULTS, 'revision', 'phi4', f'hmc_k{int(round(a.kappa * 100)):03d}.json')
    if rp and os.path.exists(rp):
        rj = json.load(open(rp))
        ra = rj.get('args', {})
        if ra.get('L', a.L) != a.L or abs(ra.get('kappa', a.kappa) - a.kappa) > 1e-12 or abs(ra.get('lam', a.lam) - a.lam) > 1e-12:
            raise SystemExit(f'reference {rp} does not match L/kappa/lam')
        ref = {k: (rj['reference'][k]['mean'], rj['reference'][k]['var']) for k in KEYS}
        br = rj.get('between_runs', {})
        ref_se = {k: br[k]['sd_runs'] / math.sqrt(br[k]['n_runs']) for k in KEYS if k in br and 'sd_runs' in br[k]}
    else:
        rp = None
    # MLP evaluations per complete path (x2 for the odd Z2 wrapper): CMCD T + 1 (one net at every point),
    # DBS 2T (forward net at x_0..x_{T-1}, backward net at x_1..x_T). Target-score evaluations: T + 1.
    nep = (2 if m.z2 else 1) * ((T + 1) if m.alg == 'cmcd' else 2 * T)
    model_seed = info.get('train_args', {}).get('seed', 'jax:' + str(info.get('source')))
    result = dict(args=vars(a), model=info, T=T, alg=m.alg, z2=m.z2, reference=rp, ref=ref, ref_se=ref_se,
                  seed_rule=SEED_RULE, model_seed=model_seed,
                  net_evals_per_path=nep, force_evals_per_path=T + 1, methods={})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    prev = json.load(open(a.out)) if os.path.exists(a.out) else None
    if prev is not None:
        keys_ = ('model', 'kappa', 'n_chains', 'path_budget', 'n_runs', 'seed', 'init', 'burn_fraction', 'window_ratio')
        if any(prev['args'].get(k_) != vars(a).get(k_) for k_ in keys_):
            raise SystemExit('resume mismatch with the stored args; use a new --out')
        result['methods'] = prev.get('methods', {})
    stem = os.path.splitext(a.out)[0]
    for meth in a.methods:
        if meth in result['methods'] and 'between_runs' in result['methods'][meth]:
            print(f'{meth}: done in {a.out}, skipped', flush=True)
            continue
        name, s_stop = (meth.split(':')[0], int(meth.split(':')[1])) if ':' in meth else (meth, None)
        if name == 'rt_partial' and not (1 <= s_stop <= T - 1):
            raise SystemExit('rt_partial:s needs 1 <= s <= T-1 (s = 0 is rt)')
        runs = list(result['methods'].get(meth, {}).get('runs_done', []))
        for r in range(len(runs), a.n_runs):
            seed = chain_seed(a.seed, m.alg, a.kappa, model_seed, r, meth)
            if name == 'snis':
                s = run_snis(m, a.path_budget, seed)
            else:
                ppt = 1.0 if name == 'imh' else (2.0 if name == 'rt' else 2.0 * (T - s_stop) / T)
                n_trans = max(int(a.path_budget / (a.n_chains * ppt)), 20)
                if a.init == 'proposal':
                    torch.manual_seed(seed)
                    c = m.coeffs()
                    x0 = m.forward_from(m.prior_sample(a.n_chains, c, dev, dtype), 0, c)[0]
                else:
                    x0 = hmc_bank(m, a.n_chains, a.bank_traj, a.bank_eps, a.bank_lf, seed, dev, dtype)
                meth_code = {'imh': 'imh', 'rt': 'rt', 'rt_partial': 'rtp'}[name]
                ckb = f'{stem}_{meth.replace(":", "")}_r{r}'
                if os.path.exists(ckb + '_done.pt'):
                    z = torch.load(ckb + '_done.pt', map_location='cpu', weights_only=False)
                    tr, acc, sec, xl, seg0 = z['tr'], z['acc'], z['sec'], z['x'].to(dev), z['seg']
                    Wl = z['W'].to(dev) if z.get('W') is not None else None     # repo addition (--save-final)
                    print(f'resumed {ckb}_done.pt: {acc.shape[0]} transitions, segment {seg0}', flush=True)
                else:
                    tr, acc, sec, xl, Wl = run_chain(m, meth_code, s_stop, x0, n_trans, seed + 1, z2flip=a.z2flip,
                                                     ckpt=ckb + '_seg0.ckpt.pt', ckpt_minutes=a.ckpt_minutes,
                                                     return_W=True)
                    seg0 = 0
                    _save_atomic(ckb + '_done.pt', dict(tr=tr, acc=acc, sec=sec, x=xl.cpu(), seg=0,
                                                        W=(Wl.cpu() if Wl is not None else None)))
                    if os.path.exists(ckb + '_seg0.ckpt.pt'):
                        os.remove(ckb + '_seg0.ckpt.pt')
                burn = int(a.burn_fraction * acc.shape[0]) if seg0 > 0 else int(a.burn_fraction * n_trans)

                def win_ratio(trd, b):     # largest Sokal window (per-chain and pooled-mean centred, |M| and sign) / n_post
                    w = [cs.tau_int(trd['absM'][b:])[1], tau_global(trd['absM'][b:])[1],
                         cs.tau_int(trd['sign'][b:])[1], tau_global(trd['sign'][b:])[1]]
                    return max(w) / max(trd['absM'].shape[0] - b, 1)
                seg, ratio = seg0, win_ratio(tr, burn)
                # gpu_E 2026-11-04: extend (doubling, continuing from the final states; IMH redraws its record given x,
                # an exact Gibbs step) until window < n_post / window_ratio or the path cap is reached
                while a.window_ratio > 0 and ratio * a.window_ratio > 1.0 and                         acc.shape[0] * a.n_chains * ppt * 2 <= a.max_paths:
                    seg += 1
                    tr2, acc2, sec2, xl, Wl = run_chain(m, meth_code, s_stop, xl, acc.shape[0], seed + 1 + 7919 * seg,
                                                        z2flip=a.z2flip, ckpt=ckb + f'_seg{seg}.ckpt.pt',
                                                        ckpt_minutes=a.ckpt_minutes, return_W=True)
                    tr = {kk: np.concatenate([tr[kk], tr2[kk]]) for kk in tr}
                    acc = np.concatenate([acc, acc2])
                    sec += sec2
                    _save_atomic(ckb + '_done.pt', dict(tr=tr, acc=acc, sec=sec, x=xl.cpu(), seg=seg,
                                                        W=(Wl.cpu() if Wl is not None else None)))
                    if os.path.exists(ckb + f'_seg{seg}.ckpt.pt'):
                        os.remove(ckb + f'_seg{seg}.ckpt.pt')
                    burn = int(a.burn_fraction * acc.shape[0])
                    ratio = win_ratio(tr, burn)
                n_trans = acc.shape[0]
                cost = dict(paths=ppt, net_evals=ppt * result['net_evals_per_path'], seconds=sec / n_trans)
                s = summarise(tr, acc, burn, ref, ref_se, cost)
                s.update(seconds=sec, n_transitions=n_trans, burn=burn, paths_per_transition=ppt,
                         pullback_depth=(T - (s_stop or 0)) if name != 'imh' else None, extensions=seg,
                         max_window_over_npost=ratio,
                         window_criterion_met=bool(a.window_ratio <= 0 or ratio * a.window_ratio <= 1.0),
                         burnin_check=cs.burnin_check({kk: tr[kk] for kk in ('absM', 'action', 'sign')}, burn,
                                                      early_keys=('absM', 'action', 'sign')))
            s['seed'] = seed
            if a.save_final and name != 'snis':          # repo addition: final states (+ carried IMH record)
                rec = 'carried chain record' if name == 'imh' else None
                if name == 'imh' and Wl is None:          # (only after a resume from a _done.pt without W)
                    cc = m.coeffs()
                    torch.manual_seed(seed + 1 + 7919 * (seg + 1))   # reproducible fallback record
                    ur, Wl = m.reverse_to(xl, 0, cc)
                    Wl = Wl + m.prior_logpdf(ur, cc)
                    rec = 'reverse record drawn given the final x (resume without stored W)'
                fpath = f'{stem}_final_{meth.replace(":", "")}_run{r}.pt'
                _save_atomic(fpath, dict(x=xl.cpu(), W=(Wl.cpu() if Wl is not None else None), method=meth,
                                         kappa=a.kappa, run=r, n_transitions=int(acc.shape[0]), seed=seed,
                                         extensions=seg, record_source=rec))
                s['final_states'] = os.path.basename(fpath)
            runs.append(s)
            result['methods'].setdefault(meth, {})['runs_done'] = runs          # resume point (finished runs)
            tmpj = a.out + '.tmp'
            json.dump(result, open(tmpj, 'w'), indent=1, default=float)
            os.replace(tmpj, a.out)
            if name != 'snis' and os.path.exists(f'{stem}_{meth.replace(":", "")}_r{r}_done.pt'):
                os.remove(f'{stem}_{meth.replace(":", "")}_r{r}_done.pt')
            brief = {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in s.items() if not isinstance(v, dict)}
            brief.update({f'tau_{kk}': round(s[kk]['tau_int'], 2) for kk in KEYS if isinstance(s.get(kk), dict) and 'tau_int' in s[kk]})
            brief.update({f'z_{kk}': round(s[kk]['z_with_ref_se'], 2) for kk in KEYS if isinstance(s.get(kk), dict) and 'z_with_ref_se' in s[kk]})
            if isinstance(s.get('sign'), dict) and 'z_global_with_ref_se' in s['sign']:
                brief['z_sign_global'] = round(s['sign']['z_global_with_ref_se'], 2)
            if 'holding' in s:
                brief['max_hold'] = s['holding']['max_run']
                brief['tau_sign_global'] = round(s['sign']['tau_global'], 2)
            print(f'[k={a.kappa} {meth} run {r}] ' + json.dumps(brief), flush=True)
        agg = {}
        for kk in KEYS:
            means = [rr[kk]['mean'] for rr in runs]
            pooled = float(np.mean([rr[kk].get('var', np.nan) for rr in runs])) if name != 'snis' else None
            agg[kk] = cs.between_runs(means, exact_var=(ref[kk][1] if ref else None), pooled_var=pooled,
                                      cost_per_run=dict(paths=a.path_budget,
                                                        seconds=float(np.mean([rr['seconds'] for rr in runs]))))
            if ref and len(means) > 1:
                agg[kk]['z_between'] = (float(np.mean(means)) - ref[kk][0]) / math.sqrt(
                    np.var(means, ddof=1) / len(means) + (ref_se.get(kk, 0.0) if ref_se else 0.0) ** 2)
        result['methods'][meth] = dict(runs=runs, between_runs=agg)
        tmpj = a.out + '.tmp'
        json.dump(result, open(tmpj, 'w'), indent=1, default=float)
        os.replace(tmpj, a.out)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
