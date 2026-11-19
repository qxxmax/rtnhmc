"""Free-energy estimators and weight diagnostics.

  SNIS        log Z = log mean_F e^{-W_F}                       (forward paths only)
  X-rev       log Z = -log mean_X e^{+W_R}                      (reverse records of the states of chain X = IMH / RT / pRT)
  X + SNIS    Bennett acceptance ratio in the Meng-Wong bridge form (bar_logz), with the effective size of the
              correlated chain n_R* = chain size / max(tau(W_R), tau(|M|)).
Errors: jackknife over B = 20 blocks (estimates). summarize: Kish ESS fraction, SNIS log Z, Pareto k-hat and reweighted
observables of a set of forward paths. Reference: thermodynamic integration of the Wolff / HMC hop density
(site_quadrature, weights, series, combine, nodes, ti).
"""
import math

import numpy as np
import torch


def lme(x):                      # log mean exp
    m = np.max(x)
    return float(m + np.log(np.mean(np.exp(x - m))))


def tau_of(x):
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    if ac[0] <= 0:
        return 1.0
    cs = np.cumsum(ac[1:] / ac[0])
    for m in range(1, n):
        t = 1 + 2 * cs[m - 1]
        if m >= 5 * t:
            return float(max(t, 1.0))
    return float(max(t, 1.0))


def bar_logz(lwF, lwR, nR_eff, init, weights='bennett'):
    """lwF = log w of forward samples, lwR = log w (= -W_R) of the chain states; returns log Z.
    weights='bennett': optimal bridge alpha = 1 / (n_F q + n_R pi), i.e. the denominator
    s_R w + s_F Z with s_F = n_F / (n_F + n_R*) (Meng & Wong 1996 eq. 4.1 = Bennett 1976). weights='swapped': the
    earlier implementation s_F w + s_R Z (a valid but non-optimal bridge), kept for comparison."""
    nF = len(lwF)
    sF = nF / (nF + nR_eff)
    sR = 1 - sF
    a_w, a_z = (sR, sF) if weights == 'bennett' else (sF, sR)
    lz = init
    for _ in range(200):
        # numerator: mean_F w / (a_w w + a_z Z); denominator: mean_R 1 / (a_w w + a_z Z)   (all in logs)
        dF = np.logaddexp(math.log(a_w) + lwF, math.log(a_z) + lz)
        dR = np.logaddexp(math.log(a_w) + lwR, math.log(a_z) + lz)
        new = lme(lwF - dF) - lme(-dR)
        if abs(new - lz) < 1e-10:
            lz = new
            break
        lz = new
    return float(lz)


EST_VERSION = 2          # 2: Bennett bridge weights (bar); 'bar_swapped' = the version-1 weights


def estimates(z, burn_frac=0.1, B=20):
    WR, WF = z['WR'], z['WF']
    k = int(z['rev_every']) if 'rev_every' in z else 1
    n, C = WR.shape
    b = int(burn_frac * n)
    R = WR[b:]
    m = (R.shape[0] // B) * B
    R = R[:m]
    lwF = -WF
    mF = (len(lwF) // B) * B
    lwF = lwF[:mF]
    tau_w = float(np.median([tau_of(R[:, c]) for c in range(C)]))
    # the chain states mix more slowly than the fresh reverse records: effective chain size from max(tau(W_R), tau(|M|))
    tau_m = float(np.median([tau_of(z['absM'][b * k:(b + m) * k, c]) for c in range(C)])) if 'absM' in z else 1.0
    tau = max(tau_w, tau_m / k, 1.0)

    def est(lwF_, R_):
        lwR = -R_.ravel()
        snis = lme(lwF_)
        rev = -lme(R_.ravel())
        bar = bar_logz(lwF_, lwR, lwR.size / tau, snis)
        bsw = bar_logz(lwF_, lwR, lwR.size / tau, snis, weights='swapped')
        return np.array([snis, rev, bar, bsw])
    full = est(lwF, R)
    Rb = R.reshape(B, -1, C)
    Fb = lwF.reshape(B, -1)
    jk = np.array([est(np.delete(Fb, i, 0).ravel(), np.delete(Rb, i, 0).reshape(-1, C)) for i in range(B)])
    err = np.sqrt((B - 1) / B * ((jk - jk.mean(0)) ** 2).sum(0))
    return dict(snis=(full[0], err[0]), rev=(full[1], err[1]), bar=(full[2], err[2]), bar_swapped=(full[3], err[3]),
                est_version=EST_VERSION, tau_WR=tau_w, tau_M=tau_m, rev_every=k,
                n_post_over_tau=float(m / tau),
                n_chain=int(R.size), n_fwd=int(lwF.size), acc=float(z['acc'][b * k:].mean()))


def pareto_khat(lw):
    """Generalised-Pareto shape of the largest M = min(n/5, 3 sqrt(n)) weights (Zhang & Stephens 2009,
    as in PSIS, Vehtari et al. 2024). k < 0.5: finite variance; 0.5-0.7: unreliable ESS; > 0.7: heavy tail."""
    n = lw.numel(); m = int(min(n / 5, 3 * math.sqrt(n)))
    x = torch.exp(lw.double() - lw.max()).sort().values
    u = x[-m - 1]; y = (x[-m:] - u).clamp_min(1e-300)
    nn_ = y.numel(); prior = 3
    ns = 30 + int(math.sqrt(nn_))
    quart = float(y[int(nn_ / 4 + 0.5) - 1])
    j = torch.arange(1, ns + 1, dtype=torch.float64)
    theta = 1 / float(y[-1]) + (1 - torch.sqrt(ns / (j - 0.5))) / prior / quart
    k = torch.log1p(-theta[:, None] * y[None, :]).mean(1)
    ll = nn_ * (torch.log(-theta / k) - k - 1)
    wt = 1 / torch.exp(ll[None, :] - ll[:, None]).sum(1)
    th = float((theta * wt).sum())
    kh = float(torch.log1p(-th * y).mean())
    return (nn_ * kh + 10 * 0.5) / (nn_ + 10)


def summarize(W, M, V):
    lw = -W.double()
    w = torch.softmax(lw, 0)
    n = lw.numel()
    absm = float((w * M.abs()).sum()); m2 = float((w * M ** 2).sum())
    wr = torch.exp(lw - lw.max())
    logz_se = float(wr.std() / wr.mean() / math.sqrt(n))      # delta method: se(log mean w)
    return dict(ess_frac=float(1 / (w ** 2).sum() / n), logZ=float(torch.logsumexp(lw, 0) - math.log(n)),
                logZ_se=logz_se, max_weight=float(w.max()), pareto_k=pareto_khat(lw),
                abs_mag_rw=absm, m2_rw=m2, chi_rw=V * (m2 - absm ** 2), var_W=float(W.double().var()),
                frac_pos=float((w * (M > 0).double()).sum()), frac_pos_raw=float((M > 0).double().mean()))


def site_quadrature(lam, lo=-8.0, hi=8.0, n=400001):
    grid = np.linspace(lo, hi, n)
    logp = -((1 - 2 * lam) * grid ** 2 + lam * grid ** 4)
    p = np.exp(logp)
    h = grid[1] - grid[0]
    # Simpson
    w = np.ones(n)
    w[1:-1:2] = 4
    w[2:-1:2] = 2
    z1 = h / 3 * np.sum(w * p)
    cdf = np.cumsum(p)
    cdf /= cdf[-1]
    return math.log(z1), grid, cdf



LAM = 0.022
LOGZ1 = site_quadrature(LAM)[0]
FINE_LO, FINE_HI = 250, 290                         # 1e-3 grid where the hop density is steep (L12 / L16)


def weights(n, h):
    """Composite Newton-Cotes weights for n intervals (Simpson; a final 3/8 panel if n is odd)."""
    w = np.zeros(n + 1)
    m = n if n % 2 == 0 else n - 3
    if m < 0:
        raise ValueError(n)
    for j in range(0, m, 2):
        w[j:j + 3] += h / 3 * np.array([1.0, 4.0, 1.0])
    if m != n:
        w[m:m + 4] += 3 * h / 8 * np.array([1.0, 3.0, 3.0, 1.0])
    return w



def series(D, L, alg):
    ks = sorted(k for (l, a, k) in D if l == L and a == alg)
    return {k: D[(L, alg, k)][:2] for k in ks}


def combine(a, b):
    out = {}
    for k in set(a) & set(b):
        (m1, s1), (m2, s2) = a[k], b[k]
        w1, w2 = 1 / s1 ** 2, 1 / s2 ** 2
        out[k] = ((w1 * m1 + w2 * m2) / (w1 + w2), 1 / math.sqrt(w1 + w2))
    return out


def nodes(s, kmax):
    """Integration segments up to kmax (units 1e-3): step 5 everywhere, step 1 inside [FINE_LO, FINE_HI] if complete."""
    fine = all(i in s for i in range(FINE_LO, FINE_HI + 1))
    segs = []
    if fine and kmax > FINE_LO:
        segs.append(list(range(0, FINE_LO + 1, 5)))
        segs.append(list(range(FINE_LO, min(kmax, FINE_HI) + 1)))
        if kmax > FINE_HI:
            segs.append(list(range(FINE_HI, kmax + 1, 5)))
    else:
        segs.append(list(range(0, kmax + 1, 5)))
    return segs


def ti(s, L, kmax, stride):
    """stride 1: nodes as given; stride 2: every other node of each segment (quadrature-error estimate)."""
    V = L * L
    tot, var = 0.0, 0.0
    for seg in nodes(s, kmax):
        seg = seg[::stride]
        if len(seg) == 1:
            continue
        if len(seg) < 3 or any(i not in s for i in seg):
            return None
        hs = (seg[1] - seg[0]) * 1e-3
        w = weights(len(seg) - 1, hs)
        m = np.array([s[i][0] for i in seg])
        e = np.array([s[i][1] for i in seg])
        tot += float(w @ m)
        var += float(((w * e) ** 2).sum())
    return V * LOGZ1 + 2 * V * tot, 2 * V * math.sqrt(var)




if __name__ == '__main__':
    import json
    import sys
    for f in sys.argv[1:]:
        r = estimates(np.load(f))
        print(f, json.dumps({k: (v if not isinstance(v, tuple) else [round(v[0], 5), round(v[1], 5)]) for k, v in r.items()}))
