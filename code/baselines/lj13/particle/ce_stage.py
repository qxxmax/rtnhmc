"""Stage-wise candidate cross-entropy (CE) for the NHMC forward momentum law (LJ / DW code path).

Bridge (the one the leapfrog integrates; requires anneal_base=1):
    log g_k(x) = (1 - b_k) log q0(x) + b_k log gamma(x),     b_k = kernel._beta(k, T),  g_0 = q0.
Per-stage log-weight increment of a path (telescoping, sums to -W up to the constant g_T vs gamma):
    dl_k = log g_k(x_k) - log g_{k-1}(x_{k-1}) + log R_k(-p'_k | x_k) - log F_k(p_k | x_{k-1}).
At a given input x_{k-1} (shared by the K candidates), the F_k-dependent part of the weight is
    g_k(x_k) R_k / F_k, so the stage-local CE target over K candidates p^(i) ~ q (= current F_k, detached) is
        log t_i = log g_k(x_k^(i)) + log R_k(-p'^(i) | x_k^(i)) - log q(p^(i))      (temperature: t ∝ e^{beta log t})
        log s_i = log F_k(p^(i) | x_{k-1}) - log q(p^(i))
        CE_k    = - sum_i t_i log softmax(log s)_i
    With q = F (detached) the gradient is the group-relative score-function estimator
        -sum_i (t_i - 1/K) grad log F_k(p^(i)),
    i.e. no backpropagation through the leapfrog map; only F_k receives gradient.
Optional within-set log-variance (stage-local VarGrad): Var_i[log t_i - log s_i] = Var_i[dl_k^(i)] at fixed x_{k-1}.
The path continues with candidate 0 (on-policy state distribution for every stage).

Exactness: this changes only the training loss; sampling, weights and round-trip MH are untouched.
Cost per bridge: K x T leapfrogs (n_lf + 1 force evaluations each) + K x T reverse-law evaluations.
"""
import torch
from torch.distributions import Normal

from lj13_nhmc import base_log_prob_scaled


def log_bridge(kernel, target, x, t, T):
    if t == 0:
        return base_log_prob_scaled(x, kernel.base_scale)
    b = kernel._beta(t, T)
    return (1.0 - b) * base_log_prob_scaled(x, kernel.base_scale) + b * target.log_prob(x)


def stage_ce_loss(kernel, target, n_bridges, K, T, dev, label_beta=1.0, var_weight=0.0, dtype=torch.float32):
    """Returns (loss, stats). Gaussian base only."""
    if not getattr(kernel, 'anneal_base', True):
        raise ValueError("stage-wise CE needs anneal_base=1 (leapfrog on the geometric bridge)")
    with torch.no_grad():
        x, _, _, _, _ = kernel.sample_base(n_bridges, dev, dtype)
    total = x.new_zeros(())
    lab_ess, lab_max, var_acc = 0.0, 0.0, 0.0
    for t in range(1, T + 1):
        B = x.size(0)
        tn = torch.full((B,), t / T, device=dev, dtype=dtype)
        with torch.no_grad():
            mu_q, ls_q = kernel._momentum_params(x, tn, reverse=False)
            sq = torch.exp(ls_q)
            P = mu_q[None] + sq[None] * torch.randn(K, *x.shape, device=dev, dtype=dtype)   # [K, B, D]
            lq = Normal(mu_q[None], sq[None]).log_prob(P).sum(-1)                            # [K, B]
            xr = x[None].expand(K, *x.shape).reshape(K * B, -1)
            xn, pn = kernel._leapfrog(xr, P.reshape(K * B, -1), kernel.epsilon_at(t, T), kernel._beta(t, T))
            tn1 = torch.full((K * B,), (t + 1) / T, device=dev, dtype=dtype)
            mu_b, ls_b = kernel._momentum_params(xn, tn1, reverse=True)
            lr = Normal(mu_b, torch.exp(ls_b)).log_prob(-pn).sum(-1).view(K, B)
            lg = log_bridge(kernel, target, xn, t, T).view(K, B)
            log_t = lg + lr - lq
            ok = torch.isfinite(log_t).all(0)                                                  # [B]
            t_lab = torch.softmax(label_beta * log_t[:, ok], 0)                                # [K, B']
        mu_f, ls_f = kernel._momentum_params(x, tn, reverse=False)                             # with grad
        lf = Normal(mu_f[None], torch.exp(ls_f)[None]).log_prob(P).sum(-1)                    # [K, B]
        log_s = (lf - lq)[:, ok]
        ce = -(t_lab * torch.log_softmax(log_s, 0)).sum(0).mean()
        loss_t = ce
        if var_weight > 0:
            dl = log_t[:, ok] - log_s                      # = lg + lr - lf (+ const in i)
            loss_t = loss_t + var_weight * dl.var(0).mean()
            var_acc += float(dl.detach().var(0).mean())
        total = total + loss_t
        lab_ess += float((1.0 / (K * t_lab.square().sum(0))).mean())
        lab_max += float(t_lab.max(0).values.mean())
        x = xn.view(K, B, -1)[0].detach()
    stats = dict(ce_label_ess=lab_ess / T, ce_label_max=lab_max / T, ce_stage_var=var_acc / T,
                 ce_force_evals=n_bridges * K * T * (kernel.n_leapfrog + 1))
    return total / T, stats


def path_records(kernel, target, n, T, dev, dtype=torch.float32):
    """Forward paths without gradient. Returns x_T, M = [T, n, D] output records m_k = -p'_k, and W."""
    with torch.no_grad():
        x, lq0, _, _, _ = kernel.sample_base(n, dev, dtype)
        logP = lq0.to(torch.float64)
        M = []
        for t in range(1, T + 1):
            tn = torch.full((n,), t / T, device=dev, dtype=dtype)
            mu_f, ls_f = kernel._momentum_params(x, tn, reverse=False)
            p0 = mu_f + torch.exp(ls_f) * torch.randn_like(mu_f)
            lf = Normal(mu_f, torch.exp(ls_f)).log_prob(p0).sum(1)
            x, pn = kernel._leapfrog(x, p0, kernel.epsilon_at(t, T), kernel._beta(t, T))
            tn1 = torch.full((n,), (t + 1) / T, device=dev, dtype=dtype)
            mu_b, ls_b = kernel._momentum_params(x, tn1, reverse=True)
            lb = Normal(mu_b, torch.exp(ls_b)).log_prob(-pn).sum(1)
            logP = logP + (lf - lb).to(torch.float64)
            M.append(-pn)
        W = logP + (-target.log_prob(x.to(torch.float64)))
    return x, torch.stack(M), W


def inverse_log_q(kernel, x_T, M, T):
    """log Q(z) of the output record z = (x_T, m_1..m_T) under the CURRENT forward model, by the exact
    inverse map (volume preserving): log q0(x_0(z)) + sum_k log F_k(p_k(z) | x_{k-1}(z)). With gradient."""
    cur = x_T
    lf = torch.zeros(x_T.size(0), device=x_T.device, dtype=x_T.dtype)
    for t in range(T, 0, -1):
        x_prev, minus_p0 = kernel._leapfrog(cur, M[t - 1], kernel.epsilon_at(t, T), kernel._beta(t, T))
        tn = torch.full((x_T.size(0),), t / T, device=x_T.device, dtype=x_T.dtype)
        mu_f, ls_f = kernel._momentum_params(x_prev, tn, reverse=False)
        lf = lf + Normal(mu_f, torch.exp(ls_f)).log_prob(-minus_p0).sum(1)
        cur = x_prev
    return base_log_prob_scaled(cur, kernel.base_scale) + lf


def snis_inverse_ce(kernel, target, n, T, dev):
    """Self-normalised inverse-map likelihood (forward KL to P~(z) = exp(-S(x_T)) prod_k r_k(m_k)).
    Valid as written for a FIXED reverse law r_k (momentum_mode='forward_only'); weights use energies only
    (no target gradient), the drift / momentum / step / base parameters get gradient through the inverse map."""
    if getattr(kernel, 'momentum_mode', 'full') != 'forward_only':
        raise ValueError("snis_inverse_ce assumes a fixed reverse law (momentum_mode=forward_only)")
    x_T, M, W = path_records(kernel, target, n, T, dev)
    ok = torch.isfinite(W)
    wbar = torch.softmax(-W[ok], 0).to(x_T.dtype)
    logq = inverse_log_q(kernel, x_T[ok], M[:, ok], T)
    loss = -(wbar.detach() * logq).sum()
    return loss, dict(snis_ess=float(1.0 / (wbar ** 2).sum() / max(int(ok.sum()), 1)))
