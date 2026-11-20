"""Write rtnhmc_demo.ipynb: round-trip updates of a learned NHMC path model on 2D phi^4 (L = 8, kappa = 0.27).

usage: python code/build_notebook.py rtnhmc_demo.ipynb      (from the repository root)
"""
import sys

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))  # noqa: E731

md(r"""
# Round-trip updates of a learned NHMC path model: predicted acceptance and autocorrelation

Companion notebook of M. Qian, *Round-Trip Updates of Learned Nonequilibrium Hamiltonian Paths: Predictable
Acceptance and Scaling in Lattice Field Theory*, [arXiv:2607.15682](https://arxiv.org/abs/2607.15682).

**Target.** Two-dimensional $\phi^4$ theory on a periodic $L\times L$ lattice in the hopping parametrisation,

$$S(\phi)=\sum_x\Big[(1-2\lambda)\phi_x^2+\lambda\phi_x^4-2\kappa\sum_{\mu}\phi_x\phi_{x+\hat\mu}\Big],\qquad \pi(\phi)\propto e^{-S(\phi)},$$

with $\lambda=0.022$. The notebook uses one trained model: $L=8$, $\kappa=0.27$ (near the critical coupling),
$T=16$ path stages with five leapfrog substeps each.

**Paths and work.** A forward path starts from $u_0\sim q_0$ and applies $T$ learned stochastic stages; the reverse
kernel defines the probability of walking the same path backwards. The path work
$$W=\log q_0(u_0)+\sum_t\big[\log q_{F,t}-\log q_{R,t}\big]+S(x_T)$$
is the same for a forward path and for a reverse record, and $\mathbb E_Q[e^{-W}]=Z$.

**Three updates.**
* *path-IMH*: propose a fresh forward path; accept with $\min(1,e^{W_x-W_y})$, where $W_x$ is the work of the path that produced the current state.
* *round trip (RT)*: walk back from the current $x$ along a reverse path to $u_0$, generate forward from the same $u_0$ to $y$, accept with $\min(1,e^{W_x-W_y})$.
* *truncated round trip (pRT)*: the same through the intermediate field $u_s$ ($s=12$ here, so only the last $d=T-s=4$ stages are redrawn), with the partial works $W_s$.

All three leave $\pi$ invariant for any model parameters.

**Two relations tested here.**
1. Acceptance: $\bar\alpha_s=\operatorname{erfc}(\sigma_s/2)$, with $\sigma_s^2=\mathbb E_{u_s}[\operatorname{Var}(W_s\mid u_s)]$ the conditional work variance at a shared intermediate field, measured on forward paths before any chain is run.
2. Autocorrelation of full round trips: $\tau_{\rm int}(|m|)\approx C\,(1/\bar\alpha-1/2)$ with $C=1.18$ (median over 74 native-model points in the paper).

Sections: 1. setup · 2. forward paths and SNIS · 3. conditional work variance and predicted acceptance ·
4. short chains (path-IMH, RT, pRT) · 5. acceptance and $\tau_{\rm int}$ against the predictions and the paper's long chains ·
6. free energy: SNIS, reverse and bridge (BAR) estimates · 7. paper figures from the Zenodo record.
""")

md("## 1. Setup")
code(r"""
import json, math, os, sys, time
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, "code")
import kernels, paths, updates
from phi4 import phi4_action
import tau as TAU
import free_energy as FE
from exp_chains import imh_recursion
from exp_wtail_var import cond_var_pivots, jk

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODE = "auto"                 # "auto", "quick" (laptop CPU, about 10 minutes) or "full" (GPU-sized)
FAST = (MODE == "quick") or (MODE == "auto" and DEV == "cpu")
torch.set_num_threads(min(os.cpu_count() or 1, 8))
torch.backends.cudnn.allow_tf32 = False          # forward and reverse stages must agree to float32 precision
torch.backends.cuda.matmul.allow_tf32 = False

MODEL_DIR, CKPT, KAPPA = "model", "kernel_final.pt", 0.27
if not os.path.exists(os.path.join(MODEL_DIR, CKPT)):
    raise SystemExit("model/kernel_final.pt is missing: see model/README.md")
k, cfg = kernels.load_model(MODEL_DIR, CKPT, torch.device(DEV))
k_cpu, _ = kernels.load_model(MODEL_DIR, CKPT, torch.device("cpu"))   # the sigma^2 estimator runs on the CPU
T, L, LAM = cfg["num_steps"], cfg["L"], cfg["lam"]
V = L * L

# sizes: quick (CPU) / full
N_FWD   = 8192  if FAST else 65536     # forward paths for SNIS and the bridge estimate
N_PIV   = 4096  if FAST else 16384     # continuation paths for sigma^2 (R = 16 per intermediate field)
C       = 16    if FAST else 64        # chains per update
N_IMH   = 1500  if FAST else 8000      # transitions per chain
N_RT    = 600   if FAST else 4000
N_PRT   = 1200  if FAST else 8000
S_PRT   = 12                           # turning stage of the truncated round trip (d = T - s = 4)
C_TAU   = 1.18                         # tau_int ~ C (1/alpha - 1/2), paper value for full round trips
plt.rcParams.update({"figure.dpi": 110, "font.size": 9})
print(f"device {DEV} | fast mode {FAST} | kernel {cfg['kernel']} L={L} T={T} n_lf={cfg['n_leapfrog']} kappa={KAPPA}")
""")

md(r"""
**Stored results used for comparison** (all in `results/`): forward-path evaluation of this model with
262144 paths, the conditional work variance from 16384 continuation paths with a jackknife over 64 pivot groups,
the paper's long chains for this model (four chains each, adaptive length), and the reference $\log Z$ from
thermodynamic integration of Wolff / HMC data and from AIS.
""")
code(r"""
EVAL = json.load(open("results/forward_eval/L8_k0.27_buf_s0_eval_N262144.json"))
EVAL = EVAL[-1] if isinstance(EVAL, list) else EVAL
SIG = json.load(open("results/sigma2_jackknife/L8_k027.json"))["stages"]
LONG = {m: json.load(open(f"results/chains_L8_k027/E2_L8_k027_{f}.json"))
        for m, f in (("path-IMH", "imhG"), ("RT", "rt"), ("pRT", "rtp12"))}

def long_chain_summary(d):
    # paper statistic: per chain max(Sokal, batch means)/2, median over chains; acceptance = mean over chains
    t = np.array([max(c["absM"]["tau_sokal"], c["absM"].get("tau_batch_means") or 0) / 2 for c in d["chains"]])
    a = np.array([c["acceptance"] for c in d["chains"]])
    tm, te, _ = TAU.jk_delete_one(t, np.median)
    am, ae, _ = TAU.jk_delete_one(a, np.mean)
    return dict(acc=am, acc_err=ae, tau=tm, tau_err=te, n=d["n_transitions"], converged=d["criteria_met"])

LONGS = {m: long_chain_summary(d) for m, d in LONG.items()}
ti = [l.split(",") for l in open("results/references/ti_logz.csv")]
hdr = ti[0]; row = [r for r in ti[1:] if r[0] == "8" and abs(float(r[1]) - KAPPA) < 1e-9][0]
LOGZ_TI = float(row[hdr.index("logZ_comb")])
LOGZ_TI_ERR = math.hypot(float(row[hdr.index("se_comb")]), float(row[hdr.index("quad_err")]))
ais = json.load(open("results/references/logz_ref.json"))["results"][f"{KAPPA:.2f}"]
print(f"log Z: TI {LOGZ_TI:.4f} +- {LOGZ_TI_ERR:.4f} | AIS {ais['logZ_mean']:.4f} +- {ais['logZ_sem']:.4f}")
for m, s in LONGS.items():
    print(f"long chains {m:8s}: acceptance {s['acc']:.3f}({s['acc_err']:.3f})  tau_int(|m|) {s['tau']:.2f}({s['tau_err']:.2f})  "
          f"n = {s['n']} per chain, converged {s['converged']}")
""")

md(r"""
## 2. Forward paths and SNIS

$N$ independent forward paths give the importance weights $w=e^{-W}$: Kish effective sample fraction
$\mathrm{ESS}_w/N=(\sum w)^2/(N\sum w^2)$, the SNIS estimate $\log Z=\log\overline{e^{-W}}$ and the Pareto $\hat k$ of the
weight tail (`free_energy.summarize`).
""")
code(r"""
t0 = time.time()
torch.manual_seed(7)
WF, MF = [], []
with torch.no_grad():
    for i in range(N_FWD // 1024):
        x, W = paths.paths(k, 1024, L, KAPPA, LAM, T, DEV)
        WF.append(W.cpu()); MF.append(x.mean(dim=(1, 2, 3)).double().cpu())
WF, MF = torch.cat(WF), torch.cat(MF)
snis = FE.summarize(WF, MF, V)
print(f"{N_FWD} forward paths in {time.time() - t0:.0f} s")
print(f"ESS_w/N {snis['ess_frac']:.3f} (stored, N=262144: {EVAL['ess_frac']:.3f}) | Var(log w) {snis['var_W']:.3f} "
      f"(stored {EVAL['var_W']:.3f}) | Pareto k-hat {snis['pareto_k']:.2f}")
print(f"SNIS log Z {snis['logZ']:.4f} +- {snis['logZ_se']:.4f}  vs TI {LOGZ_TI:.4f}")
fig, ax = plt.subplots(1, 2, figsize=(8, 2.8))
ax[0].hist((WF - WF.mean()).numpy(), 80, density=True); ax[0].set_xlabel("W - <W>"); ax[0].set_ylabel("density")
ax[1].hist(MF.numpy(), 80, density=True, label="raw"); ax[1].hist(MF.numpy(), 80, density=True,
          weights=torch.softmax(-WF.double(), 0).numpy(), histtype="step", label="SNIS"); ax[1].set_xlabel("m"); ax[1].legend()
plt.tight_layout(); plt.show()
""")

md(r"""
## 3. Conditional work variance and the predicted acceptance

For a turning stage $s$ we draw intermediate fields $u_s$ on forward paths and continue each of them $R=16$ times to
$x_T$; the average within-group variance of $W_s=S(x_T)+\sum_{t>s}(\log q_{F,t}-\log q_{R,t})$ estimates
$\sigma_s^2$ (`exp_wtail_var.cond_var_pivots`, the estimator of the paper; jackknife over groups of intermediate
fields). The Gaussian-work prediction of the mean acceptance is $\bar\alpha_s=\operatorname{erfc}(\sigma_s/2)$.
""")
code(r"""
R, BATCH = 16, 1024
SIGMA2, PRED = {}, {}
t0 = time.time()
torch.manual_seed(11)
for s in (0, S_PRT):
    wv = np.concatenate([cond_var_pivots(k_cpu, BATCH // R, R, L, KAPPA, LAM, T, s)[0] for _ in range(N_PIV // BATCH)])
    groups = np.array_split(wv, 16)
    SIGMA2[s] = (float(wv.mean()), jk(groups, np.mean))
    PRED[s] = (math.erfc(math.sqrt(SIGMA2[s][0]) / 2), jk(groups, lambda g: math.erfc(math.sqrt(max(g.mean(), 0)) / 2)))
    st = SIG[str(s)]
    print(f"s = {s:2d}: sigma^2 = {SIGMA2[s][0]:.3f} +- {SIGMA2[s][1]:.3f} (stored {st['cond_var']:.3f} +- {st['cond_var_jk']:.3f}) "
          f"-> erfc(sigma/2) = {PRED[s][0]:.3f} (stored {st['acc_pred']:.3f})")
print(f"{time.time() - t0:.0f} s")
ss = sorted(int(s) for s in SIG)
fig, ax = plt.subplots(1, 2, figsize=(8, 2.8))
ax[0].errorbar([T - s for s in ss], [SIG[str(s)]["cond_var"] for s in ss], [SIG[str(s)]["cond_var_jk"] for s in ss], fmt="o-")
ax[0].set_xlabel("return depth d = T - s"); ax[0].set_ylabel(r"$\sigma_s^2$ (stored)")
z = np.linspace(0, 3, 200); ax[1].plot(z, [math.erfc(math.sqrt(v) / 2) for v in z]); ax[1].set_xlabel(r"$\sigma^2$")
ax[1].set_ylabel(r"erfc$(\sigma/2)$")
for s in (0, S_PRT):
    ax[1].plot(SIGMA2[s][0], PRED[s][0], "o", label=f"s = {s}")
ax[1].legend(); plt.tight_layout(); plt.show()
""")

md(r"""
## 4. Short chains: path-IMH, full and truncated round trips

$C$ independent chains per update, started at endpoints of fresh forward paths (no reference configurations);
burn-in is the first 10 % of each chain. path-IMH proposals do not depend on the state, so they are generated in large
batches first and the accept/reject recursion runs afterwards (`exp_chains.imh_recursion`, the `--imh-pregen` mode of
the paper's chains). Round trips use `updates.truncated_round_trip_step` ($s=0$: full round trip). For RT the record of
the new state is free (the forward path of $y$ if accepted, else the reverse record of $x$), which section 6 uses for
the reverse and bridge estimates of $\log Z$.
""")
code(r"""
RES = {}

# path-IMH from pre-generated proposals
t0 = time.time()
torch.manual_seed(21)
Wp, Mp, Sp, Ap = [], [], [], []
n_prop = C * (N_IMH + 1)
with torch.no_grad():
    for i in range(0, n_prop, 1024):
        u, _ = kernels.base_draw(k, (min(1024, n_prop - i), 1, L, L), DEV, KAPPA)
        x, W = paths.fwd_partial(k, u, T, 0, KAPPA, LAM)
        M = x.mean(dim=(1, 2, 3)).double()
        Wp.append(W.double().cpu().numpy()); Mp.append(M.abs().cpu().numpy()); Sp.append(torch.sign(M).cpu().numpy())
        Ap.append((phi4_action(x.squeeze(1), lam=LAM, kappa=KAPPA).double() / V).cpu().numpy())
Wp, Mp, Sp, Ap = (np.concatenate(a).reshape(C, N_IMH + 1) for a in (Wp, Mp, Sp, Ap))
rng = np.random.default_rng(22)
tr, ac = np.empty((N_IMH, C)), np.empty((N_IMH, C), bool)
for c in range(C):
    m_, s_, a_, acc_, _, _ = imh_recursion(Wp[c], Mp[c], Sp[c], Ap[c], np.log(rng.random(N_IMH)))
    tr[:, c], ac[:, c] = m_, acc_
RES["path-IMH"] = dict(absM=tr, acc=ac, seconds=time.time() - t0)
print(f"path-IMH: {C} x {N_IMH} transitions in {time.time() - t0:.0f} s")

def run_rt(s, n, seed):
    t0 = time.time()
    torch.manual_seed(seed)
    x, _ = updates.start_states(k, C, L, L, T, KAPPA, LAM, DEV)
    tr, acc, WR = np.empty((n, C)), np.empty((n, C), bool), np.empty((n, C))
    for it in range(n):
        x, a, _, Wx, Wy = updates.truncated_round_trip_step(k, x, T, s, KAPPA, LAM)
        tr[it] = x.mean(dim=(1, 2, 3)).abs().double().cpu().numpy()
        acc[it] = a.cpu().numpy()
        WR[it] = torch.where(a, Wy, Wx).double().cpu().numpy()      # full reverse record of the new state (s = 0)
    return dict(absM=tr, acc=acc, WR=WR, seconds=time.time() - t0)

RES["RT"] = run_rt(0, N_RT, 31)
print(f"RT: {C} x {N_RT} transitions in {RES['RT']['seconds']:.0f} s")
RES["pRT"] = run_rt(S_PRT, N_PRT, 32)
print(f"pRT (s = {S_PRT}): {C} x {N_PRT} transitions in {RES['pRT']['seconds']:.0f} s")
""")

md(r"""
## 5. Acceptance and autocorrelation time against the predictions

$\tau_{\rm int}(|m|)$ per chain is $\max(\text{Sokal},\ \text{batch means})/2$ after burn-in, reported as the median over
chains with a delete-one-chain jackknife error (`tau.tau_int_chains`); acceptance errors are delete-one-chain jackknife
errors. The short chains here are 10–100 times shorter than the paper's chains, so their $\tau_{\rm int}$ are rough.
""")
code(r"""
rows = []
for m, r in RES.items():
    b = int(0.1 * r["acc"].shape[0])
    a_c = r["acc"][b:].mean(0)
    am, ae, _ = TAU.jk_delete_one(a_c, np.mean)
    tm, te, _ = TAU.tau_int_chains(r["absM"])
    pred = {"RT": PRED[0][0], "pRT": PRED[S_PRT][0]}.get(m)
    rows.append((m, am, ae, pred, tm, te, C_TAU * (1 / am - 0.5)))
    print(f"{m:8s} acceptance {am:.3f}({ae:.3f})  erfc(sigma/2) {'-' if pred is None else f'{pred:.3f}'}  "
          f"long chains {LONGS[m]['acc']:.3f} | tau_int {tm:.2f}({te:.2f})  long chains {LONGS[m]['tau']:.2f}  "
          f"C(1/alpha - 1/2) {C_TAU * (1 / am - 0.5):.2f}")
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
for m, am, ae, pred, tm, te, tp in rows:
    if pred is not None:
        ax[0].errorbar(pred, am, ae, fmt="o", label=f"{m}, this notebook")
        ax[0].plot(pred, LONGS[m]["acc"], "s", mfc="none", label=f"{m}, paper chains")
    ax[1].errorbar(1 / am - 0.5, tm, te, fmt="o", label=m)
ax[0].plot([0, 1], [0, 1], "k:", lw=0.8); ax[0].set_xlabel(r"erfc$(\sigma_s/2)$"); ax[0].set_ylabel("measured acceptance")
ax[0].legend(fontsize=7)
xx = np.linspace(0, 3, 10); ax[1].plot(xx, C_TAU * xx, "k--", lw=0.8, label=r"$1.18\,(1/\bar\alpha-1/2)$")
ax[1].set_xlabel(r"$1/\bar\alpha-1/2$"); ax[1].set_ylabel(r"$\tau_{\rm int}(|m|)$"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.show()
""")
md(r"""
The relation $\tau_{\rm int}\approx C(1/\bar\alpha-1/2)$ with $C\approx1.2$ is the one for full round trips. Truncated
round trips at this coupling keep most of the amplitude information in the intermediate field, so their $C$ is larger
(in the paper $C$ grows from $3.5$ at $L=4$ to $11$ at $L=16$ at $\kappa=0.27$). path-IMH has an acceptance similar to
RT here but holds longer on states with large weight, which shows up as a longer tail of rejection runs:
""")
code(r"""
for m, r in RES.items():
    b = int(0.1 * r["acc"].shape[0])
    runs = np.concatenate([TAU.holding_runs(r["acc"][b:, c]) for c in range(C)])
    print(f"{m:8s} longest rejection run {runs.max():4d}   99 % quantile {np.quantile(runs, 0.99):.0f}")
""")

md(r"""
## 6. Free energy: SNIS, reverse and bridge estimates

The forward works $W_F$ of section 2 and the reverse records $W_R$ of the RT chain states give three estimates
(`free_energy.estimates`, jackknife over 20 blocks): SNIS $\log\overline{e^{-W_F}}$, the one-way reverse estimate
$-\log\overline{e^{W_R}}$, and the Bennett acceptance ratio (BAR) bridge between the two, with the effective size of the
correlated chain from $\max(\tau(W_R),\tau(|m|))$.
""")
code(r"""
z = dict(WR=RES["RT"]["WR"], WF=WF.double().numpy(), acc=RES["RT"]["acc"], absM=RES["RT"]["absM"])
est = FE.estimates(z)
for key, lab in (("snis", "SNIS"), ("rev", "RT-rev"), ("bar", "RT + SNIS (BAR)")):
    v, e = est[key]
    print(f"{lab:16s} log Z = {v:.4f} +- {e:.4f}   (log Z - log Z_TI)/V = {(v - LOGZ_TI) / V:+.5f}   z = {(v - LOGZ_TI) / math.hypot(e, LOGZ_TI_ERR):+.1f}")
print(f"TI reference {LOGZ_TI:.4f} +- {LOGZ_TI_ERR:.4f}; chain acceptance {est['acc']:.3f}, n_post/tau {est['n_post_over_tau']:.0f}")
""")

md(r"""
## 7. Paper figures

All figures of the paper are redrawn from the Zenodo record (DOI [10.5281/zenodo.24163330](https://doi.org/10.5281/zenodo.24163330))
by the scripts in `figures/` (see the README for the figure-to-script table):

```bash
export NHMC_ROOT=/path/to/extracted/record
bash figures/zenodo_20261118/redraw_all.sh /tmp/redraw python
python figures/figs_20261119u6/fig_logz_u6.py
```
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, sys.argv[1] if len(sys.argv) > 1 else "rtnhmc_demo.ipynb")
