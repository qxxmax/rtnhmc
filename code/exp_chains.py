#!/usr/bin/env python3
"""Long Markov chains of path-IMH, full round trips (RT) and truncated round trips (pRT) with the paper's length rule.

A few (default 4) independent chains of one method, adaptive length (start 1e4 transitions, doubled while any chain
fails: n_post >= 1000 tau_Sokal, n_post >= 20 x max holding, Sokal window < n_post / 50; cap 2e5), proposal start (no HMC),
for a frozen phi^4 sampler: NHMC kernels via paths.py (kernel's own reverse_step) or the CMCD / DBS torch port
(baselines/cmcd_dbs/cmcd_phi4.py, same transition as rt_cmcd_phi4.run_chain). The chains are batched in one tensor and share one
random stream (seeded from the job identity); they are independent because their noise is disjoint.
Methods: imh (record = W of the path that produced the state, carried), rt, rtp:s (partial RT through stage s).
Per chain: Sokal tau (c = 5, single-chain ACF) and window/n, batch-means tau (50 batches), max holding and the
holding-time tail (q99, q99.9, CCDF at 10/50/100/500/1000/5000), running mean of |M| and sign (log-spaced), z vs a
reference (SE from the larger of the Sokal and batch-means tau); burn-in = burn_fraction of the chain.
Checkpoint every --ckpt-minutes (state, record, traces, RNG); restart resumes from it. Traces saved (npz, float32).
--kappa-transfer (cmcd): a model trained at one kappa run against the target at another kappa. For phi models
the kappa passed to the kernel is always --kappa (single-kappa buffer models then get an off-training kappa input).
--L (phi only): run the sampler at a lattice size other than its training size (E1 transfer; the kernel nets are
convolutional); the reference must have the same L. Default: the training L, then seeds are unchanged.
Main estimator: the plain chain average over the chain states. Extra, optional: waste recycling (Rao-Blackwell;
Frenkel 2004, Douc & Robert 2011) records alpha f(Y) + (1 - alpha) f(X_prev), alpha = min(1, exp(W_x - W_y)), per
transition of the same chain (no extra random numbers), reported per chain as '<obs>_rb'.
"""
import argparse
import hashlib
import json
import math
import os
import pathlib
import sys
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))          # library modules of this repository (code/)
sys.path.append(str(HERE / 'baselines' / 'cmcd_dbs'))    # cmcd_phi4.py (CMCD / DBS torch port)
sys.dont_write_bytecode = True
import kernels  # noqa: E402
import paths  # noqa: E402
import phi4  # noqa: E402
from tau import batch_means_tau, chain_criteria, holding_runs, sokal_single  # noqa: E402,F401


def save_json(obj, path):
    path = pathlib.Path(path)
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)



class PhiStepper:
    def __init__(self, run_dir, ckpt, kappa, dev, L=None, L2=None):
        self.k, cfg = kernels.load_model(run_dir, ckpt, dev)
        self.T, self.L, self.lam, self.kappa, self.dev = cfg['num_steps'], cfg['L'], cfg['lam'], kappa, dev
        self.train_L, self.train_L2 = cfg['L'], cfg.get('L2') or cfg['L']     # lattice [L, L2]; L2 = L for square runs
        self.L2 = self.train_L2
        self.id = f'{os.path.basename(os.path.normpath(run_dir))}|{self.k.ckpt_used}'
        if L or L2:                              # transfer to another lattice shape (seed id changes only then)
            self.L = L or self.train_L
            self.L2 = L2 or (self.L if self.train_L2 == self.train_L else self.train_L2)
        if (self.L, self.L2) != (self.train_L, self.train_L2):
            self.id += f'|L{self.L}' if self.L2 == self.L else f'|{self.L}x{self.L2}'

    def start(self, n):
        u, _ = kernels.base_draw(self.k, (n, 1, self.L, self.L2), self.dev, self.kappa)
        return paths.fwd_partial(self.k, u, self.T, 0, self.kappa, self.lam)


    def propose(self, method, s, x, W):
        if method == 'imh':
            u, _ = kernels.base_draw(self.k, x.shape, self.dev, self.kappa)
            y, Wy = paths.fwd_partial(self.k, u, self.T, 0, self.kappa, self.lam)
            return y, W, Wy
        u, Wx = paths.rev_partial(self.k, x, self.T, s, self.kappa, self.lam)
        y, Wy = paths.fwd_partial(self.k, u, self.T, s, self.kappa, self.lam)
        return y, Wx, Wy

    def obs(self, x):
        o = phi4.observables(x, self.lam, self.kappa)
        return o['absM'], o['sign'], o['action']


class CmcdStepper:
    def __init__(self, model, kappa, dev, kappa_transfer=False):
        import cmcd_phi4 as C
        st = torch.load(model, map_location='cpu', weights_only=False)
        ar = st['args']
        self.m = C.UDBPhi4(ar['alg'], ar['L'], kappa, lam=ar['lam'], T=ar['T'], dt=ar['dt'], z2=ar['z2'],
                           num_hid=ar['num_hid'])
        self.m.load_state_dict(st['model'])
        self.m = self.m.to(dev).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.dev, self.kappa, self.T = dev, kappa, self.m.T
        self.L = ar['L']
        assert kappa_transfer or abs(ar['kappa'] - kappa) < 1e-9, \
            f'model trained at kappa {ar["kappa"]}, requested {kappa} (use --kappa-transfer)'
        self.id = f'{ar["alg"]}|{ar["kappa"]}|{ar.get("seed")}|{os.path.basename(os.path.dirname(model))}'
        self.c = self.m.coeffs()

    def start(self, n):
        u = self.m.prior_sample(n, self.c, self.dev, torch.float32)
        x, W = self.m.forward_from(u, 0, self.c)
        return x, W + self.m.prior_logpdf(u, self.c)

    def propose(self, method, s, x, W):
        if method == 'imh':
            u = self.m.prior_sample(x.shape[0], self.c, self.dev, x.dtype)
            y, Wy = self.m.forward_from(u, 0, self.c)
            return y, W, Wy + self.m.prior_logpdf(u, self.c)
        u, Wx = self.m.reverse_to(x, s, self.c)
        y, Wy = self.m.forward_from(u, s, self.c)
        return y, Wx, Wy

    def obs(self, x):
        M = x.mean(-1).double()
        return M.abs(), torch.sign(M), (-self.m.log_gamma(x)).double() / self.m.dim


class DwStepper:
    """ManyWell DW (revision/dw/dw_compare.py): same transitions and same order of random numbers as
    dw_compare.run_chain (path_imh / round_trip / partial_round_trip; non-finite log ratios are rejections).
    rtp:K = partial round trip through the last K stages (dw_compare rt_partial:K). Trace slots: abs_even, sign1
    (sign of the first double-well coordinate; mode indicator), energy; plus the mode index. Exact moments and
    mode probabilities are analytic (dw_compare.exact_moments)."""
    names = ('abs_even', 'sign1', 'energy')
    has_sign = True

    def __init__(self, run_dir, ckpt, dev, p_flip='0.05'):
        import argparse as _ap
        dwd = HERE / 'baselines' / 'dw'
        sys.path.insert(0, str(dwd / 'dw_v2'))
        sys.path.insert(0, str(dwd))
        import dw_compare as D
        self.D = D
        rd = pathlib.Path(run_dir)
        cfg = json.loads((rd / 'config.json').read_text())
        ns = _ap.Namespace(**cfg)
        self.target = D.ManyWellTarget(ns.dim)
        _, k = D.build_kernel(self.target, ns)
        k.load_state_dict(torch.load(rd / ckpt, map_location=dev))
        self.k = k.to(dev).eval()
        if p_flip != 'learned' and hasattr(self.k, 'flip_logits'):
            D.set_fixed_flip_probability(self.k, float(p_flip))
        self.T, self.dev, self.L = int(cfg['num_steps']), dev, None
        self.n_wells = ns.dim // 2
        self.exact, self.p_pos = D.exact_moments(self.n_wells)
        self.mode_p = D.exact_mode_probs(self.n_wells, self.p_pos).numpy()
        self.id = f'dw|{rd.name}|{ckpt}|{p_flip}'

    def reference(self):
        return {k: dict(mean=float(self.exact[k][0]), se=0.0) for k in ('abs_even', 'energy')}

    def start(self, n):
        x, w = self.D.rollout(self.k, self.target, n, self.T, self.dev)
        return x, w

    def step(self, method, s, x, W):
        D, k, tg, T = self.D, self.k, self.target, self.T
        if method == 'imh':
            y, w_y = D.rollout(k, tg, x.shape[0], T, self.dev)
            w_x = W
        elif method == 'rt':
            u, w_x = D.reverse_rollout(k, tg, x, T)
            y, w_y = D.forward_rollout_from(k, tg, u, T)
        else:
            y, w_x, w_y = D.partial_round_trip(k, tg, x, T, s)
        la = (w_x - w_y).double()
        la = torch.where(torch.isfinite(la), la, torch.full_like(la, -math.inf))
        a = torch.log(torch.rand_like(la)) < la
        al = torch.exp(torch.clamp(la, max=0.0))
        fx, fy = self.obs(x), self.obs(y)
        x = torch.where(a[:, None], y, x)
        if method == 'imh':
            W = torch.where(a, w_y, W)
        return x, W, a, al, fx, fy

    def obs(self, x):
        o = self.D.observables(x, self.target)
        return o['abs_even'], o['sign1'], o['energy']

    def mode(self, x):
        return self.D.mode_index(x)


class LjStepper:
    """LJ13 (revision/lj13/lj13_compare.py, kernel code from $LJ_CODE): imh as lj13_compare.run_chain (fresh
    forward paths, carried record), rt / rtp:K through eval_lj_cluster_nhmc.rt_step (K = depth: partial round trip
    through the stage-(T-K) configuration). Trace slots: energy, (no sector observable), radius of gyration.
    --ref: directory with test_split_LJ13-1000.npy (reference means; reference states are never chain starts)."""
    names = ('energy', 'none', 'rg')
    has_sign = False

    def __init__(self, run_dir, ckpt, dev):
        ljd = HERE / 'baselines' / 'lj13'
        sys.path.insert(0, str(ljd))
        import lj13_compare as J
        self.J = J
        rd = pathlib.Path(run_dir)
        cfg = json.load(open(rd / 'config.json'))
        J.ev.CKPT = ckpt
        self.target, self.k, _ = J.ev.build_kernel(cfg, rd, dev)
        self.k.eval()
        self.T, self.n, self.dev, self.L = cfg['num_steps'], cfg['n_particles'], dev, None
        self.id = f'lj|{rd.name}|{ckpt}'

    def reference_from(self, refdir):
        test = torch.tensor(np.load(os.path.join(refdir, 'test_split_LJ13-1000.npy')), dtype=torch.float32,
                            device=self.dev)
        o = self.J.observables(test, self.target, self.n)
        return {'energy': dict(mean=float(o['energy'].mean()), se=float(o['energy'].std() / math.sqrt(test.shape[0]))),
                'rg': dict(mean=float(o['rg'].mean()), se=float(o['rg'].std() / math.sqrt(test.shape[0])))}

    def start(self, n):
        return self.J.fresh_path(self.k, self.target, n, self.T, self.dev)

    def step(self, method, s, x, W):
        J = self.J
        if method == 'imh':
            y, Wy = J.fresh_path(self.k, self.target, x.shape[0], self.T, self.dev)
            la = torch.nan_to_num(W - Wy, nan=-math.inf)
            a = torch.log(torch.rand_like(la)) < la
            al = torch.exp(torch.clamp(la.double(), max=0.0))
            fx, fy = self.obs(x), self.obs(y)
            x = torch.where(a[:, None], y, x)
            W = torch.where(a, Wy, W)
            return x, W, a, al, fx, fy
        fx = self.obs(x)
        xn, a, y, log_a = J.ev.rt_step(self.k, self.target, x, self.T, depth=s if method == 'rtp' else None)
        return xn, W, a, torch.exp(log_a.double()), fx, self.obs(y)

    def obs(self, x):
        o = self.J.observables(x, self.target, self.n)
        return o['energy'], torch.zeros_like(o['energy']), o['rg']



def gen_proposals(st, chain_seed, start, n, batch):
    """proposals start .. start+n-1 of one chain: torch seed (chain_seed, batch index) per batch of `batch` paths,
    so the stream is the same whatever the extension history. Returns W, |M|, sign, S/V as numpy arrays."""
    Ws, Ms, Ss, As = [], [], [], []
    b0, b1 = start // batch, (start + n - 1) // batch
    for b in range(b0, b1 + 1):
        W, M_, S_, A_ = proposal_batch(st, chain_seed, b, batch)
        lo, hi = max(start - b * batch, 0), min(start + n - b * batch, batch)
        Ws.append(W[lo:hi]); Ms.append(M_[lo:hi]); Ss.append(S_[lo:hi]); As.append(A_[lo:hi])
    return np.concatenate(Ws), np.concatenate(Ms), np.concatenate(Ss), np.concatenate(As)


PREGEN_DIR = None     # --pregen-dir: batches written by --pregen-shard runs are read from here instead of recomputed
PREGEN_WAIT = 0.0     # --pregen-wait: seconds to wait for a missing batch (shard runs in parallel) before computing it


def proposal_batch(st, chain_seed, b, batch, save=False):
    """batch b of one chain's proposal stream (W, |M|, sign, S/V); from PREGEN_DIR if a shard run stored it."""
    f = PREGEN_DIR / f'{chain_seed}_{batch}_{b}.npz' if PREGEN_DIR is not None else None
    if f is not None and not save and PREGEN_WAIT > 0:
        t0 = time.time()
        while not f.exists() and time.time() - t0 < PREGEN_WAIT:
            time.sleep(1.0)
    if f is not None and f.exists() and not save:
        z = np.load(f)
        return z['W'], z['M'], z['S'], z['A']
    torch.manual_seed(int(hashlib.sha256(f'{chain_seed}|{b}'.encode()).hexdigest()[:8], 16))
    x, W = st.start(batch)
    M_, S_, A_ = st.obs(x)
    r = W.double().cpu().numpy(), M_.cpu().numpy(), S_.cpu().numpy(), A_.cpu().numpy()
    if save:
        tmp = f.with_name(f.name + '.tmp.npz')
        np.savez(tmp, W=r[0], M=r[1], S=r[2], A=r[3])
        tmp.replace(f)
    return r


def pregen_shard(st, a, seed, shard):
    """--pregen-shard I/N: compute and store the proposal batches g = I, I + N, ... of all chains up to --n-max
    (g = b * n_chains + chain, batch-major so that shards running next to the chain stay just ahead of it). The streams are seeded per (chain, batch), so a later --pregen-dir run reads the
    identical proposals whatever the sharding."""
    i, n = map(int, shard.split('/'))
    nb = (a.n_max + 1 + a.pregen_batch - 1) // a.pregen_batch
    cseeds = [int(hashlib.sha256(f'{seed}|chain{c}'.encode()).hexdigest()[:8], 16) for c in range(a.n_chains)]
    t0, k = time.time(), 0
    for g in range(i, a.n_chains * nb, n):
        b, c = divmod(g, a.n_chains)
        if not (PREGEN_DIR / f'{cseeds[c]}_{a.pregen_batch}_{b}.npz').exists():
            proposal_batch(st, cseeds[c], b, a.pregen_batch, save=True)
            k += 1
    print(f'[pregen shard {shard}] {k} batches of {a.pregen_batch} in {time.time() - t0:.0f}s', flush=True)


def imh_recursion(Wp, Mp, Sp, Ap, logu, state=None):
    """IMH over pre-generated proposals. Wp etc. index the proposals 0..n-1 of this segment; logu[t] the uniform of
    transition t. state = (W_cur, M_cur, S_cur, A_cur) carried from the previous segment, or None: then proposal 0
    is the start and the segment's transitions use proposals 1..n-1 (n-1 transitions), else proposals 0..n-1."""
    j0 = 0
    if state is None:
        Wc, Mc, Sc, Ac = Wp[0], Mp[0], Sp[0], Ap[0]
        if not np.isfinite(Wc):
            raise RuntimeError('nonfinite start record')
        j0 = 1
    else:
        Wc, Mc, Sc, Ac = state
    n = Wp.size - j0
    trM = np.empty(n, np.float32); trS = np.empty(n, np.int8); trA = np.empty(n, np.float32); acc = np.empty(n, bool)
    rb = np.empty((n, 3), np.float32)
    for t in range(n):
        j = j0 + t
        la = Wc - Wp[j]
        if np.isnan(la):
            la = -np.inf
        al = 1.0 if la >= 0 else math.exp(la)
        rb[t] = (al * Mp[j] + (1 - al) * Mc, al * Sp[j] + (1 - al) * Sc, al * Ap[j] + (1 - al) * Ac)
        if logu[t] < la:
            Wc, Mc, Sc, Ac = Wp[j], Mp[j], Sp[j], Ap[j]
            acc[t] = True
        else:
            acc[t] = False
        trM[t], trS[t], trA[t] = Mc, Sc, Ac
    return trM, trS, trA, acc, rb, (Wc, Mc, Sc, Ac)


def run_imh_pregen(st, a, seed):
    st_sign = getattr(st, 'has_sign', True)
    """IMH long chains from pre-generated proposals with the adaptive length rule. Returns traces, history, N."""
    C, NMAX = a.n_chains, a.n_max
    trM = np.zeros((NMAX, C), np.float32); trS = np.zeros((NMAX, C), np.int8)
    trA = np.zeros((NMAX, C), np.float32); acc = np.zeros((NMAX, C), bool)
    trRB = np.zeros((NMAX, C, 3), np.float32)
    cseeds = [int(hashlib.sha256(f'{seed}|chain{c}'.encode()).hexdigest()[:8], 16) for c in range(C)]
    rngs = [np.random.default_rng(cs_) for cs_ in cseeds]
    states = [None] * C
    done, target, history = 0, min(a.n_start, NMAX), []
    t0 = time.time()
    while True:
        for c in range(C):
            # transitions done .. target-1 use proposals done+1 .. target (proposal 0 = start state)
            first = 0 if states[c] is None else done + 1
            n_prop = (target + 1) - first
            Wp, Mp, Sp, Ap = gen_proposals(st, cseeds[c], first, n_prop, a.pregen_batch)
            logu = np.log(rngs[c].random(target - done))
            m_, s_, a_, ac_, rb_, states[c] = imh_recursion(Wp, Mp, Sp, Ap, logu, states[c])
            trM[done:target, c], trS[done:target, c], trA[done:target, c], acc[done:target, c] = m_, s_, a_, ac_
            trRB[done:target, c] = rb_
        done = target
        ok, crit = chain_criteria(trM, trS, acc, target, a.burn_fraction, a.tau_mult, a.hold_mult, a.window_ratio, st_sign)
        history.append(dict(n=target, ok=ok, chains=crit))
        print(f'[pregen IMH] n = {target}: criteria ok = {ok}; ' + '; '.join(
            f'c{c}: tau {d["tau_max"]:.1f} win {d["window_max"]} hold {d["max_hold"]} {d["failing"]}'
            for c, d in enumerate(crit)) + f' [{time.time() - t0:.0f}s]', flush=True)
        if ok or target >= NMAX:
            break
        target = min(2 * target, NMAX)
    return trM, trS, trA, acc, trRB, history, target


def prefetch_round(st, name, s, x, W, K, need):
    """One prefetching round (Brockwell 2006) for kinds with a native step (dw / lj), rt / rtp only.
    A rejected round trip leaves the chain at x, so the next transition again starts at x with fresh noise: the
    K round trips from the current x of each chain are drawn in one batch (replica k = transition it_c + k), and
    the chain uses them in order up to and including the first acceptance; the rest are discarded. The discard
    rule depends only on the earlier replicas, so the used transitions have exactly the law of the step-by-step
    chain. need[c] = transitions still wanted by chain c (0 = inactive). K = 1 reproduces the step loop exactly
    (same call, same random numbers). Returns per-chain lists of (state obs, acc, rb) and the new (x, W)."""
    act = [c for c in range(x.shape[0]) if need[c] > 0]
    xa, Wa = x[act], W[act]
    xr = xa.repeat_interleave(K, 0) if K > 1 else xa
    Wr = Wa.repeat_interleave(K, 0) if K > 1 else Wa
    xn, _, ac, al, fx, fy = st.step(name, s, xr, Wr)
    ac = ac.view(len(act), K).cpu().numpy()
    al = al.view(len(act), K).double()
    rb = torch.stack([al * fy_.view(len(act), K).double() + (1 - al) * fx_.view(len(act), K).double()
                      for fx_, fy_ in zip(fx, fy)], 2).cpu().numpy()                     # [C_act, K, 3]
    so = torch.stack([o_.view(len(act), K).double() for o_ in st.obs(xn)], 2).cpu().numpy()  # obs of the new states
    md = st.mode(xn).view(len(act), K).cpu().numpy() if hasattr(st, 'mode') else np.zeros((len(act), K), np.int16)
    xn = xn.view(len(act), K, *x.shape[1:])
    out = {}
    x, W = x.clone(), W.clone()
    for j, c in enumerate(act):
        hit = np.flatnonzero(ac[j])
        n_use = min(int(hit[0]) + 1 if hit.size else K, need[c])
        out[c] = (so[j, :n_use], ac[j, :n_use], rb[j, :n_use], md[j, :n_use])
        if hit.size and hit[0] < n_use:                 # the accepted replica is used: move the chain
            x[c] = xn[j, hit[0]]
    return out, x, W


def run_prefetch(st, a, name, s, seed, ck_step, ckpf, dev):
    """rt / rtp chains with --prefetch K; same adaptive length rule and outputs as the step loop. Resumes from its
    own checkpoint, or continues a step-mode checkpoint (same chains, new random stream: a valid continuation)."""
    C, NMAX, K = a.n_chains, a.n_max, a.prefetch
    if ckpf.exists():
        z = torch.load(ckpf, map_location='cpu', weights_only=False)
        its, src = list(z['its']), 'prefetch checkpoint'
    elif ck_step.exists():
        z = torch.load(ck_step, map_location='cpu', weights_only=False)
        its, src = [int(z['it'])] * C, f'step checkpoint (transition {z["it"]})'
    else:
        z = None
    if z is not None:
        assert z['trM'].shape == (NMAX, C) and z.get('method') == a.method, 'checkpoint does not match the arguments'
        x, W, target = z['x'].to(dev), z['W'].to(dev), z['target']
        trM, trS, trA, acc, history = z['trM'], z['trS'], z['trA'], z['acc'], z.get('history', [])
        trRB, trX = z['trRB'], z.get('trX', np.zeros((NMAX, C), np.int16))
        if 'rng_cpu' in z and src == 'prefetch checkpoint':
            torch.set_rng_state(z['rng_cpu'])
            if dev.type == 'cuda' and z.get('rng_cuda') is not None:
                torch.cuda.set_rng_state(z['rng_cuda'].cpu())
        else:
            torch.manual_seed(int(hashlib.sha256(f'{seed}|prefetch{K}|from{its[0]}'.encode()).hexdigest()[:8], 16))
        print(f'prefetch K={K}: resumed from {src}, transitions {its}', flush=True)
    else:
        torch.manual_seed(seed)
        x, W = st.start(C)
        assert bool(torch.isfinite(W).all()), 'nonfinite starting paths'
        its, target, history = [0] * C, min(a.n_start, NMAX), []
        trM = np.zeros((NMAX, C), np.float32); trS = np.zeros((NMAX, C), np.int8)
        trA = np.zeros((NMAX, C), np.float32); acc = np.zeros((NMAX, C), bool)
        trRB = np.zeros((NMAX, C, 3), np.float32); trX = np.zeros((NMAX, C), np.int16)

    def save():
        tmp = ckpf.with_suffix('.tmp')
        torch.save(dict(its=its, it=min(its), target=target, method=a.method, prefetch=K, x=x.cpu(), W=W.cpu(),
                        trM=trM, trS=trS, trA=trA, acc=acc, trRB=trRB, trX=trX, history=history,
                        rng_cpu=torch.get_rng_state(),
                        rng_cuda=torch.cuda.get_rng_state() if dev.type == 'cuda' else None), tmp)
        os.replace(tmp, ckpf)
    t0, tck, it_start, rounds = time.time(), time.time(), sum(its), 0
    while True:
        while min(its) < target:
            need = [target - i for i in its]
            out, x, W = prefetch_round(st, name, s, x, W, K, need)
            rounds += 1
            for c, (so_, ac_, rb_, md_) in out.items():
                i0, n_ = its[c], ac_.size
                trM[i0:i0 + n_, c], trS[i0:i0 + n_, c], trA[i0:i0 + n_, c] = so_[:, 0], so_[:, 1], so_[:, 2]
                acc[i0:i0 + n_, c], trRB[i0:i0 + n_, c], trX[i0:i0 + n_, c] = ac_, rb_, md_
                its[c] += n_
            if time.time() - tck > 60 * a.ckpt_minutes:
                save()
                tck = time.time()
                el = time.time() - t0
                print(f'checkpoint at {its} ({el / max(sum(its) - it_start, 1) * 1e3 * C:.1f} ms/transition per chain, '
                      f'{(sum(its) - it_start) / max(rounds, 1) / C:.2f} transitions per round)', flush=True)
        if history and history[-1]['n'] == target:
            ok, crit = history[-1]['ok'], history[-1]['chains']
        else:
            ok, crit = chain_criteria(trM, trS, acc, target, a.burn_fraction, a.tau_mult, a.hold_mult, a.window_ratio,
                                      getattr(st, 'has_sign', True))
            history.append(dict(n=target, ok=ok, chains=crit))
        print(f'n = {target}: criteria ok = {ok}; ' + '; '.join(
            f'c{c}: tau {d["tau_max"]:.1f} win {d["window_max"]} hold {d["max_hold"]} {d["failing"]}'
            for c, d in enumerate(crit)), flush=True)
        if ok or target >= NMAX:
            break
        target = min(2 * target, NMAX)
        save()
    save()
    return trM, trS, trA, acc, trRB, trX, history, target


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--kind', choices=['phi', 'cmcd', 'dw', 'lj'], required=True)
    ap.add_argument('--p-flip', default='0.05', help="dw: fixed flip probability (dw_compare default 0.05) or 'learned'")
    ap.add_argument('--run-dir', default='')
    ap.add_argument('--ckpt', default='best.pt')
    ap.add_argument('--model', default='', help='cmcd: model.pt')
    ap.add_argument('--kappa', type=float, default=0.0)
    ap.add_argument('--method', required=True, help='imh | rt | rtp:s')
    ap.add_argument('--n-chains', type=int, default=4)
    ap.add_argument('--n-start', type=int, default=10000, help='initial transitions per chain')
    ap.add_argument('--n-max', type=int, default=200000, help='cap (transitions per chain)')
    ap.add_argument('--tau-mult', type=float, default=1000.0)
    ap.add_argument('--hold-mult', type=float, default=20.0)
    ap.add_argument('--window-ratio', type=float, default=50.0)
    ap.add_argument('--burn-fraction', type=float, default=0.1)
    ap.add_argument('--ref', default='', help='HMC+Z2 reference json (results_sync/revision/phi4/hmc_kXXX.json)')
    ap.add_argument('--seed', type=int, default=20261104)
    ap.add_argument('--ckpt-minutes', type=float, default=20.0)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--mem-fraction', type=float, default=0.05)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--imh-pregen', action='store_true', help='IMH from pre-generated proposal streams (fast)')
    ap.add_argument('--pregen-batch', type=int, default=8192)
    ap.add_argument('--pregen-dir', default='', help='IMH pregen: proposal batch store (see --pregen-shard)')
    ap.add_argument('--pregen-wait', type=float, default=0.0, help='IMH pregen: wait up to this many s for a missing batch')
    ap.add_argument('--pregen-shard', default='', help="I/N: only compute and store batch shard I of N, then exit")
    ap.add_argument('--prefetch', type=int, default=0,
                    help='rt / rtp, kinds dw / lj: K round trips per chain per round from the current state, used in '
                         'order up to the first acceptance (exact; K = 1 = the step loop); 0 = step loop')
    ap.add_argument('--kappa-transfer', action='store_true',
                    help='cmcd: run a model trained at another kappa against the target at --kappa (exact: the target '
                         'enters the annealing path and the weights; only the learned nets are off-target)')
    ap.add_argument('--L', type=int, default=0, help='phi: lattice size (default: training L of the run config)')
    ap.add_argument('--L2', type=int, default=0, help='phi: second lattice extent (default: L for square runs)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device(a.device)
    if dev.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(a.mem_fraction)
    assert a.kind == 'phi' or not (a.L or a.L2), '--L / --L2 are for --kind phi'
    if a.kind == 'phi':
        st = PhiStepper(a.run_dir, a.ckpt, a.kappa, dev, a.L, a.L2)
    elif a.kind == 'cmcd':
        st = CmcdStepper(a.model, a.kappa, dev, a.kappa_transfer)
    elif a.kind == 'dw':
        st = DwStepper(a.run_dir, a.ckpt, dev, a.p_flip)
    else:
        st = LjStepper(a.run_dir, a.ckpt, dev)
    assert not (a.imh_pregen and a.kind == 'dw'), '--imh-pregen: phi / cmcd / lj only'
    names = getattr(st, 'names', ('absM', 'sign', 'action'))
    st_sign = getattr(st, 'has_sign', True)
    name, s = (a.method.split(':')[0], int(a.method.split(':')[1])) if ':' in a.method else (a.method, 0)
    assert name in ('imh', 'rt', 'rtp') and (name != 'rtp' or 1 <= s <= st.T - 1)
    if name == 'rt':
        s = 0
    seed = int(hashlib.sha256(f'{a.seed}|{st.id}|{a.kappa:.6f}|{a.method}|long'.encode()).hexdigest()[:8], 16)
    ref = None
    if a.kind == 'dw':
        ref = st.reference()
    elif a.kind == 'lj' and a.ref:
        ref = st.reference_from(a.ref)
    elif a.ref:
        rj = json.load(open(a.ref))
        ra = rj.get('args', {})
        assert abs(float(ra.get('kappa', a.kappa)) - a.kappa) < 1e-9 and int(ra.get('L', 8)) == st.L, 'reference kappa/L mismatch'
        br = rj.get('between_runs', {})
        ref = {k: dict(mean=rj['reference'][k]['mean'],
                       se=(br[k]['sd_runs'] / math.sqrt(br[k]['n_runs'])) if k in br and 'sd_runs' in br[k] else 0.0)
               for k in ('absM', 'action')}
    outp = pathlib.Path(a.out)
    ck = outp.with_suffix('.ckpt.pt')
    C, NMAX = a.n_chains, a.n_max
    history = []
    if a.pregen_dir:
        global PREGEN_DIR, PREGEN_WAIT
        PREGEN_DIR = pathlib.Path(a.pregen_dir)
        PREGEN_WAIT = a.pregen_wait
        PREGEN_DIR.mkdir(parents=True, exist_ok=True)
    if a.pregen_shard:
        assert name == 'imh' and a.imh_pregen and PREGEN_DIR is not None, '--pregen-shard needs --imh-pregen --pregen-dir'
        with torch.no_grad():
            pregen_shard(st, a, seed, a.pregen_shard)
        return
    if a.imh_pregen:
        assert name == 'imh', '--imh-pregen is for IMH only'
        with torch.no_grad():
            trM, trS, trA, acc, trRB, history, target = run_imh_pregen(st, a, seed)
    if a.prefetch:
        assert name in ('rt', 'rtp') and hasattr(st, 'step') and not a.imh_pregen, '--prefetch: rt / rtp, kinds dw / lj'
        with torch.no_grad():
            trM, trS, trA, acc, trRB, trX, history, target = run_prefetch(
                st, a, name, s, seed, ck, outp.with_suffix('.pf.ckpt.pt'), dev)
    with torch.no_grad():
        if a.imh_pregen or a.prefetch:
            pass
        elif ck.exists():
            z = torch.load(ck, map_location='cpu', weights_only=False)
            n_old = z['trM'].shape[0]
            if n_old < NMAX and z['trM'].shape[1] == C and z.get('method') == a.method:   # larger --n-max: extend the
                for k_ in ('trM', 'trS', 'trA', 'acc', 'trRB', 'trX'):                 # same chains (exact continuation)
                    if z.get(k_) is not None:
                        z[k_] = np.concatenate([z[k_], np.zeros((NMAX - n_old,) + z[k_].shape[1:], z[k_].dtype)])
                print(f'extending the checkpoint from n_max {n_old} to {NMAX}', flush=True)
            assert z['trM'].shape == (NMAX, C) and z.get('method') == a.method, 'checkpoint does not match the arguments'
            it0, target, x, W = z['it'], z['target'], z['x'].to(dev), z['W'].to(dev)
            trM, trS, trA, acc, history = z['trM'], z['trS'], z['trA'], z['acc'], z.get('history', [])
            trRB = z.get('trRB')
            trX = z.get('trX', np.zeros((NMAX, C), np.int16))
            if trRB is None:                            # checkpoint of the version without waste recycling
                trRB = np.full((NMAX, C, 3), np.nan, np.float32)
            torch.set_rng_state(z['rng_cpu'])
            if dev.type == 'cuda' and z.get('rng_cuda') is not None:
                torch.cuda.set_rng_state(z['rng_cuda'].cpu())
            print(f'resumed at transition {it0} (target {target})', flush=True)
        else:
            torch.manual_seed(seed)
            x, W = st.start(C)
            for _ in range(50):                         # nonfinite start records: redraw those chains
                bad = ~torch.isfinite(W)
                if not bool(bad.any()):
                    break
                x1, W1 = st.start(int(bad.sum()))
                x[bad], W[bad] = x1, W1
            assert bool(torch.isfinite(W).all()), 'could not draw finite starting paths'
            it0, target = 0, min(a.n_start, NMAX)
            trM = np.zeros((NMAX, C), np.float32)
            trS = np.zeros((NMAX, C), np.int8)
            trA = np.zeros((NMAX, C), np.float32)
            acc = np.zeros((NMAX, C), bool)
            trRB = np.zeros((NMAX, C, 3), np.float32)
            trX = np.zeros((NMAX, C), np.int16)

        def save(it_):
            tmp = ck.with_suffix('.tmp')
            torch.save(dict(it=it_, target=target, method=a.method, x=x.cpu(), W=W.cpu(), trM=trM, trS=trS, trA=trA,
                            acc=acc, trRB=trRB, trX=trX, history=history, rng_cpu=torch.get_rng_state(),
                            rng_cuda=torch.cuda.get_rng_state() if dev.type == 'cuda' else None), tmp)
            os.replace(tmp, ck)
        t0, tck, it = time.time(), time.time(), (it0 if not (a.imh_pregen or a.prefetch) else 0)
        while not (a.imh_pregen or a.prefetch):
            while it < target:
                if hasattr(st, 'step'):                            # dw / lj: native transition functions
                    x, W, ac, al, fx, fy = st.step(name, s, x, W)
                    trRB[it] = torch.stack([al * fy_.double() + (1 - al) * fx_.double() for fx_, fy_ in zip(fx, fy)],
                                           1).cpu().numpy()
                    if hasattr(st, 'mode'):
                        trX[it] = st.mode(x).cpu().numpy()
                else:
                    y, Wx, Wy = st.propose(name, s, x, W)
                    la = torch.nan_to_num((Wx - Wy).double(), nan=-math.inf, posinf=math.inf, neginf=-math.inf)
                    ac = torch.log(torch.rand_like(la)) < la
                    al = torch.exp(torch.clamp(la, max=0.0))                  # min(1, exp(la)); no random numbers used
                    fx, fy = st.obs(x), st.obs(y)
                    trRB[it] = torch.stack([al * fy_.double() + (1 - al) * fx_.double() for fx_, fy_ in zip(fx, fy)],
                                           1).cpu().numpy()
                    x = torch.where(ac.view(-1, *([1] * (x.dim() - 1))), y, x)
                    if name == 'imh':
                        W = torch.where(ac, Wy, W)
                M_, S_, A_ = st.obs(x)
                trM[it], trS[it], trA[it], acc[it] = M_.cpu().numpy(), S_.cpu().numpy(), A_.cpu().numpy(), ac.cpu().numpy()
                it += 1
                if time.time() - tck > 60 * a.ckpt_minutes:
                    save(it)
                    tck = time.time()
                    print(f'checkpoint at {it} ({(time.time() - t0) / max(it - it0, 1) * 1e3:.1f} ms/transition)', flush=True)
            if history and history[-1]['n'] == target:      # resumed after the check of this length
                ok, crit = history[-1]['ok'], history[-1]['chains']
            else:
                ok, crit = chain_criteria(trM, trS, acc, target, a.burn_fraction, a.tau_mult, a.hold_mult, a.window_ratio, st_sign)
                history.append(dict(n=target, ok=ok, chains=crit))
            print(f'n = {target}: criteria ok = {ok}; ' + '; '.join(
                f'c{c}: tau {d["tau_max"]:.1f} win {d["window_max"]} hold {d["max_hold"]} {d["failing"]}'
                for c, d in enumerate(crit)), flush=True)
            if ok or target >= NMAX:
                break
            target = min(2 * target, NMAX)
            save(it)
        if not (a.imh_pregen or a.prefetch):
            save(it)
    N = target
    burn = int(a.burn_fraction * N)
    res = dict(args=vars(a), seed=seed, model_id=st.id, n_transitions=N, burn=burn, criteria_met=bool(history[-1]['ok']),
               mode='imh pre-generated proposals' if a.imh_pregen else (f'prefetch K={a.prefetch}' if a.prefetch else 'step'),
               L=st.L, L2=getattr(st, 'L2', None),
               waste_recycling='<obs>_rb: alpha f(Y) + (1 - alpha) f(X_prev) per transition (same chain); optional',
               length_history=history, chains=[],
               rule='start n_start, double while any chain fails n_post >= tau_mult tau_Sokal, n_post >= hold_mult '
                    'max_hold, window < n_post/window_ratio; cap n_max (tau, window, holding of |M| and sign)')
    idx = np.unique(np.round(np.geomspace(1, N, 80)).astype(int) - 1)
    for c in range(C):
        d = {}
        slots = ((names[0], trM[:N, c]), (names[1], trS[:N, c].astype(np.float64)), (names[2], trA[:N, c]),
                 (names[0] + '_rb', trRB[:N, c, 0]), (names[1] + '_rb', trRB[:N, c, 1]), (names[2] + '_rb', trRB[:N, c, 2]))
        for key, tr in slots:
            if not st_sign and key.startswith(names[1]):
                continue
            if not np.all(np.isfinite(tr[burn:])):
                continue
            post = np.asarray(tr[burn:], dtype=np.float64)
            tau, win = sokal_single(post)
            tbm = batch_means_tau(post)
            t_use = max(tau, tbm or 0.0)
            se = math.sqrt(post.var() * t_use / post.size) if post.var() > 0 else 0.0
            e = dict(mean=float(post.mean()), tau_sokal=tau, window=int(win), window_over_n=win / post.size,
                     tau_batch_means=tbm, se=se,
                     running_mean=dict(t=(idx + 1).tolist(),
                                       mean=(np.cumsum(np.asarray(tr, dtype=np.float64))[idx] / (idx + 1)).tolist()))
            rv = ref.get(key.replace('_rb', '')) if ref else None
            if st_sign and key in (names[1], names[1] + '_rb'):
                rv = dict(mean=0.0, se=0.0)
            if rv is not None and se > 0:
                e['ref'] = rv['mean']
                e['z'] = (e['mean'] - rv['mean']) / math.sqrt(se ** 2 + rv['se'] ** 2)
            d[key] = e
        runs = holding_runs(acc[burn:N, c])
        d['acceptance'] = float(acc[burn:N, c].mean())
        d['holding'] = dict(max=int(runs.max()), q99=float(np.quantile(runs, .99)), q999=float(np.quantile(runs, .999)),
                            ccdf={str(L_): float((runs >= L_).mean()) for L_ in (10, 50, 100, 500, 1000, 5000)},
                            max_over_npost=float(runs.max() / (N - burn)),
                            convention='lengths of maximal runs of consecutive rejections; CCDF over runs')
        drivers = [h['chains'][c]['failing'] for h in history if h['chains'][c]['failing']]
        d['length_driver'] = drivers[-1] if drivers else ['none (start length sufficed)']
        if hasattr(st, 'mode_p'):                     # dw: L1 error of the mode frequencies (post burn)
            f = np.bincount(trX[burn:N, c].astype(np.int64), minlength=st.mode_p.size) / (N - burn)
            d['mode_l1'] = float(np.abs(f - st.mode_p).sum())
            md = trX[burn:N, c]
            d['mode_change_rate'] = float((md[1:] != md[:-1]).mean())
        res['chains'].append(d)
        o0, o1 = d[names[0]], d.get(names[1], dict(mean=0.0, tau_sokal=0.0))
        print(f'chain {c}: n {N} acc {d["acceptance"]:.3f} {names[0]} {o0["mean"]:.5f} tau {o0["tau_sokal"]:.1f} '
              f'(BM {o0["tau_batch_means"]}) win/n {o0["window_over_n"]:.4f} z {o0.get("z")} | '
              f'{names[1]} {o1["mean"]:+.4f} tau {o1["tau_sokal"]:.1f} | maxhold {d["holding"]["max"]} '
              f'q99.9 {d["holding"]["q999"]} | driver {d["length_driver"]}', flush=True)
    tr_out = {names[0]: trM[:N], names[2]: trA[:N], 'acc': acc[:N], 'rb': trRB[:N]}
    if st_sign:
        tr_out[names[1]] = trS[:N]
    if a.kind == 'dw':
        tr_out['mode'] = trX[:N]
    np.savez_compressed(str(outp.with_suffix('')) + '_traces.npz', **tr_out)
    save_json(res, outp)
    print('wrote', outp)


if __name__ == '__main__':
    main()
